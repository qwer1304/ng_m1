"""
trainer.py -- the training loop. Owns every loop and every call.

The Trainer owns: the two optimizers, the epoch loop, the validation pass,
the call to model.end_epoch(), stop handling, logging, checkpoint/restore and
the FAILURE LOG. It computes no loss (losses.py), builds no network (m1net.py)
and takes no scheduling decision (model.py): the model only says "stop" and the
Trainer executes.

=======================================================================
OPTIMIZERS (follows Ng et al., with their overlap bug fixed)
=======================================================================
  * Adam on model.non_adjacency_parameters(), lr
  * SGD (momentum) on model.adjacency_parameters(), adj_lr
The two sets are disjoint. Learning rates are constant (LR scheduling is
parked; the adjacency LR in particular should stay constant).

=======================================================================
DATA CONTRACT
=======================================================================
Both loaders yield tuples whose FIRST FIVE items are (x, y, t, e, ids); any
further items (the extra fields of ti_dataset) are ignored.
    x (B,C,H,W) float; y,t,e (B,) long; ids (B,) long = row index into the
    ordering the GeodesicTable was built from (needed only if the model's
    injective loss is active: model.needs_geod()).
val_loader: held-out slice of the TRAINING locations. Never L100, never a
held-out location.

=======================================================================
ONE EPOCH
=======================================================================
  1. train pass: one optimizer step per batch; epoch means of every
     UNWEIGHTED term are recorded (rec, kl, sparse, moral, inj, bilip, loss).
  2. validation pass (no grad): val_rec, val_kl (same estimators as in
     training, z sampled), val_acc_macro (see below).
  3. stats = train means + val_* ; status = model.end_epoch(epoch, stats).
  4. log the line, the decisions and the warnings of the model.
  5. checkpoint (epoch boundary only).
  6. stop if status.stop.
A decision made by the model in step 3 takes effect from the next epoch.

Validation accuracy: species predicted from the content latents only, with
the time of day observed: argmax_y log p(mu_c | y, t, e_content-free prior),
uniform label prior, z = mu. Macro accuracy (mean of per-species recall).
It uses the M1 prior as it currently is (frozen and unfrozen nodes alike).

=======================================================================
FAILURE LOG
=======================================================================
Three kinds of events are logged, all to <out_dir>/events.log and stdout:
  SUSPECT  the first epoch at which model.suspect becomes True (a cap was
           hit, so the run continues but must not be trusted).
  FAILURE  the run ended in a failure state; ALSO written as
           <out_dir>/failure.json (see _failure_record). Failure states:
             - "nonfinite_loss"   a loss term or the total is NaN/inf; the
                                  step is skipped (no optimizer step), the
                                  run stops, last.pt is NOT overwritten.
             - "model_stop"       the model asked to stop for a reason other
                                  than a clean finish: KL cap with the head
                                  not learning, or the sink-check cap (no
                                  stable adjacency; nothing frozen).
             - "max_epochs_unfinished"  (schedule enabled only) max_epochs
                                  reached before the graph finished.
  The failure record holds the reason, epoch, caps hit, freeze log, the
  gate diagnostics (flip history, raw adjacency entries nearest 0.5), the
  last stats, current loss weights, the decision/warning logs and
  model.report().
Always written at the end: <out_dir>/report.json (outcome + model.report()).
Outcomes: "finished", "finished_suspect", "max_epochs" (schedule disabled),
"failure:<kind>".

=======================================================================
GRADIENT-SCALE PROBE (probe.py)
=======================================================================
Every cfg.probe.every epochs (BILIP only every cfg.probe.bilip_every), after
end_epoch and before the checkpoint, the Trainer builds probe.n_batches
training batches and calls probe.grad_scale_probe. Output: a table on stdout,
one JSON line per probe in gradscale.jsonl, and one PROBE-FLAG line in
events.log per term whose scaler is outside the band. Diagnostic only: a
probe that raises is logged as a WARNING and training continues. The probe
uses torch.autograd.grad (no .grad is written) and the Trainer draws its
batches inside fork_rng, so neither the optimizers nor the RNG stream of the
run are affected.

=======================================================================
CHECKPOINT (epoch boundaries only)
=======================================================================
last.pt, written atomically every `ckpt_every` epochs and at a normal stop:
  model.state_dict(), model.schedule_state(), both optimizer states, RNG
  states (torch, cuda, numpy, python), history, epoch, the finalized
  RunConfig (as dict), the TrainerConfig (as dict) and `run_args` (the input
  arguments the run was started with, supplied by main).
Trainer.load(path) restores everything and the next fit() continues from
epoch+1. main triggers the restore; the Trainer performs it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from geodesic import GeodesicTable
from losses import kl_mc
from model import M1Model
from probe import grad_scale_probe

TERMS = ("loss", "kl", "rec", "sparse", "moral", "inj", "bilip")


class NonFiniteLoss(RuntimeError):
    """Raised inside step() when a loss term or the total is NaN/inf."""

    def __init__(self, terms: Dict[str, float]):
        super().__init__(f"non-finite loss: {terms}")
        self.terms = terms


@dataclass
class TrainerConfig:
    """
    lr                   Adam lr for everything except the adjacency
    adj_lr, adj_momentum SGD settings for the adjacency (Ng: 1e-3, 0.9)
    grad_clip            max global grad-norm (None = no clipping)
    device               "cpu" / "cuda"
    log_every            print an epoch line every this many epochs (0 = silent)
    ckpt_every           write last.pt every this many epochs (0 = only at stop)
    val_acc              compute the validation accuracy (costs n_species
                         prior passes per validation batch)
    """

    lr: float = 1e-3
    adj_lr: float = 1e-3
    adj_momentum: float = 0.9
    grad_clip: Optional[float] = None
    device: str = "cpu"
    log_every: int = 1
    ckpt_every: int = 1
    val_acc: bool = True


def _jsonable(o):
    """json default: tensors, numpy and anything else become plain python."""
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Trainer:
    def __init__(
        self,
        model: M1Model,
        cfg: TrainerConfig,
        out_dir: str,
        geod: Optional[GeodesicTable] = None,
        run_args: Optional[dict] = None,
    ):
        """
        model    : M1Model
        out_dir  : directory for events.log, log.jsonl, last.pt, report.json,
                   failure.json (created)
        geod     : GeodesicTable; required whenever model.needs_geod() is True
        run_args : the input arguments main was started with (saved in the
                   checkpoint verbatim; must be picklable)
        """
        self.cfg, self.geod, self.run_args = cfg, geod, dict(run_args or {})
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.model = model.to(cfg.device)
        self.opt = torch.optim.Adam(model.non_adjacency_parameters(), lr=cfg.lr)
        self.adj_opt = torch.optim.SGD(
            model.adjacency_parameters(), lr=cfg.adj_lr, momentum=cfg.adj_momentum
        )
        self.history: List[Dict[str, float]] = []
        self.epoch = 0  # last completed epoch
        self._suspect_logged = False
        self._cur = 0  # epoch the event log refers to
        self.last_stats: Dict[str, float] = {}
        self.outcome: Optional[str] = None

    # ------------------------------------------------------------------
    # event log
    # ------------------------------------------------------------------
    def _event(self, kind: str, text: str) -> None:
        """Append one line to events.log and print it."""
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch {self._cur:4d} {kind}: {text}"
        print(line)
        with open(os.path.join(self.out_dir, "events.log"), "a") as f:
            f.write(line + "\n")

    def _write_json(self, name: str, obj) -> None:
        with open(os.path.join(self.out_dir, name), "w") as f:
            json.dump(obj, f, indent=2, default=_jsonable)

    # ------------------------------------------------------------------
    # one optimisation step
    # ------------------------------------------------------------------
    def step(self, batch) -> Dict[str, float]:
        """One optimisation step on one (x, y, t, e, ids) batch."""
        dev = self.cfg.device
        x, y, t, e, ids = batch[:5]
        x, y, t, e = x.to(dev), y.to(dev), t.to(dev), e.to(dev)
        geod_sub = None
        if self.model.needs_geod():
            if self.geod is None:
                raise ValueError("model.needs_geod() is True but no GeodesicTable was given.")
            geod_sub = self.geod.sub(ids.cpu().numpy(), device=dev)
        res = self.model((x, y, t, e), geod_sub=geod_sub, sample=True)
        vals = {k: float(res[k].detach()) for k in TERMS}
        if not all(np.isfinite(v) for v in vals.values()):
            raise NonFiniteLoss(vals)  # before backward: no optimizer step
        self.opt.zero_grad(set_to_none=True)
        self.adj_opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.adj_opt.step()
        return vals

    def _train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        sums, n = {k: 0.0 for k in TERMS}, 0
        for batch in loader:
            r = self.step(batch)
            for k in TERMS:
                sums[k] += r[k]
            n += 1
        return {k: sums[k] / max(n, 1) for k in TERMS}

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader) -> Dict[str, float]:
        """val_rec, val_kl (unweighted, z sampled as in training) and, if
        enabled, val_acc_macro. Independent of the current loss weights."""
        self.model.eval()
        dev, net = self.cfg.device, self.model.net
        nsp, nc = net.cfg.n_species, net.n_content
        rec_s = kl_s = 0.0
        n = 0
        correct = torch.zeros(nsp)
        total = torch.zeros(nsp)
        for batch in loader:
            x, y, t, e = (v.to(dev) for v in batch[:4])
            out = net(x, y, t, e, sample=True)
            rec_s += float(F.mse_loss(out["x_hat"], x))
            kl_s += float(kl_mc(out["log_q"], out["log_p"]))
            n += 1
            if self.cfg.val_acc:
                mu_c = out["mu"][:, :nc]
                scores = []
                for k in range(nsp):
                    yk = torch.full_like(y, k)
                    p = net.prior(out["mu"], yk, t, e)
                    std = torch.exp(0.5 * p["prior_logvar"][:, :nc])
                    lp = torch.distributions.Normal(p["prior_mean"][:, :nc], std)
                    scores.append(lp.log_prob(mu_c).sum(-1))
                pred = torch.stack(scores, 1).argmax(1)
                for k in range(nsp):
                    m = y == k
                    total[k] += int(m.sum())
                    correct[k] += int((pred[m] == k).sum())
        res = {"val_rec": rec_s / max(n, 1), "val_kl": kl_s / max(n, 1)}
        if self.cfg.val_acc:
            present = total > 0
            res["val_acc_macro"] = float((correct[present] / total[present]).mean()) if present.any() else float("nan")
        return res

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader=None,
            eval_fn: Optional[Callable[[M1Model, int], Dict[str, float]]] = None) -> dict:
        """
        Run until the model says stop (or a non-finite loss). Returns the
        result dict {"outcome", "reason", "suspect", "epochs", "history"}.
        eval_fn: optional callback (model, epoch) -> dict of floats, merged
        into the epoch stats (called in eval mode, no grad).
        """
        model, reason, outcome_kind = self.model, None, None
        epoch = self.epoch
        while True:
            epoch += 1
            self._cur = epoch
            t0 = time.time()
            try:
                stats = self._train_epoch(train_loader)
            except NonFiniteLoss as ex:
                reason, outcome_kind = str(ex), "nonfinite_loss"
                self._event("FAILURE", f"nonfinite_loss: {ex.terms}")
                break
            if val_loader is not None:
                stats.update(self.validate(val_loader))
            if eval_fn is not None:
                model.eval()
                with torch.no_grad():
                    stats.update(eval_fn(model, epoch))
            status = model.end_epoch(epoch, stats)
            stats["epoch"] = epoch
            stats["seconds"] = time.time() - t0
            stats["weights"] = dict(status.weights)
            self.history.append(stats)
            self.last_stats = stats
            self.epoch = epoch
            self._log_epoch(stats, status)

            if status.suspect and not self._suspect_logged:
                self._suspect_logged = True
                self._event("SUSPECT", f"run is suspect from now on; caps hit: {model.caps_hit}")
            pc = model.run_cfg.probe
            if pc.enabled and pc.every and epoch % pc.every == 0:
                try:
                    self._probe(train_loader, epoch, pc)
                except Exception as ex:  # diagnostic only: never kill the run
                    self._event("WARNING", f"gradient-scale probe failed: {type(ex).__name__}: {ex}")
            if self.cfg.ckpt_every and epoch % self.cfg.ckpt_every == 0:
                self.save(os.path.join(self.out_dir, "last.pt"))
            if status.stop:
                reason = status.reason
                break

        return self._finish(reason, outcome_kind)

    def _log_epoch(self, s: Dict[str, float], status) -> None:
        c = self.cfg
        with open(os.path.join(self.out_dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(s, default=_jsonable) + "\n")
        for d in status.decisions:
            self._event("decision", d)
        for w in status.warnings:
            self._event("WARNING", w)
        if c.log_every and s["epoch"] % c.log_every == 0:
            extra = ""
            if "val_rec" in s:
                extra += f" vrec={s['val_rec']:.4f} vkl={s['val_kl']:.3f}"
            if "val_acc_macro" in s:
                extra += f" vacc={s['val_acc_macro']:.3f}"
            nun = len(self.model.net.prior._unfrozen())
            print(
                f"epoch {s['epoch']:4d} loss={s['loss']:.4f} kl={s['kl']:.4f} rec={s['rec']:.4f} "
                f"sparse={s['sparse']:.3f} moral={s['moral']:.3f} inj={s['inj']:.4f} "
                f"bilip={s['bilip']:.4f}{extra} unfrozen={nun} ({s['seconds']:.1f}s)"
            )

    # ------------------------------------------------------------------
    # gradient-scale probe
    # ------------------------------------------------------------------
    def _probe(self, loader, epoch: int, pc) -> None:
        dev = self.cfg.device
        d = torch.device(dev)
        fork = ([d.index if d.index is not None else torch.cuda.current_device()]
                if d.type == "cuda" else [])
        lr_by_param = {id(p): g["lr"] for o in (self.opt, self.adj_opt)
                       for g in o.param_groups for p in g["params"]}
        include_bilip = bool(pc.bilip_every) and epoch % pc.bilip_every == 0
        batches = []
        with torch.random.fork_rng(devices=fork):  # keep the run's RNG stream intact
            it = iter(loader)
            for _ in range(pc.n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                x, y, t, e, ids = batch[:5]
                gs = self.geod.sub(ids.cpu().numpy(), device=dev) if self.geod is not None else None
                batches.append((x.to(dev), y.to(dev), t.to(dev), e.to(dev), gs))
        if not batches:
            return
        res = grad_scale_probe(self.model, batches, lr_by_param, include_bilip)
        with open(os.path.join(self.out_dir, "gradscale.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "terms": res}, default=_jsonable) + "\n")
        print(f"probe epoch {epoch} (norm = RSS of lr-scaled grads, unweighted; "
              f"scaler at target weight; band [{pc.band_lo:g}, {pc.band_hi:g}])")
        for name, r in res.items():
            if r["norm"] is None:
                print(f"  {name:7s} skipped: {r['skipped']}")
                continue
            sc = "n/a" if r["scaler_at_target"] is None else f"{r['scaler_at_target']:.3g}"
            ws = "n/a" if r["suggested_weight"] is None else f"{r['suggested_weight']:.3g}"
            note = " (SGD, unvalidated)" if r["adj_only"] else ""
            print(f"  {name:7s} norm={r['norm']:.3g} target_w={r['target']:g} ramp={r['ramp_fraction']:.2f} "
                  f"scaler={sc} w*={ws} flag={r['flag']}{note}")
            if r["flag"] in ("weak", "strong", "no_gradient"):
                self._event("PROBE-FLAG", f"{name}: {r['flag']} (scaler at target {sc}, "
                            f"suggested weight {ws}){note}")

    # ------------------------------------------------------------------
    # end of run: classify, log failure, write report
    # ------------------------------------------------------------------
    def _classify(self, reason: Optional[str], kind: Optional[str]) -> str:
        m = self.model
        if kind == "nonfinite_loss":
            return "failure:nonfinite_loss"
        enabled = m.rules.enabled
        if reason and reason.startswith("finished"):
            return "finished_suspect" if m.suspect else "finished"
        if reason == "max_epochs reached":
            if not enabled:
                return "max_epochs"
            return "failure:max_epochs_unfinished"
        return "failure:model_stop"

    def _failure_record(self, outcome: str, reason: Optional[str]) -> dict:
        m = self.model
        rep = m.report()
        return {
            "outcome": outcome,
            "reason": reason,
            "epoch": self.epoch,
            "suspect": m.suspect,
            "caps_hit": rep["caps_hit"],
            "all_frozen": rep["all_frozen"],
            "freeze_log": rep["freeze_log"],
            "gate_info": rep["gate_info"],  # flip_history, nearest_to_half, unfrozen
            "last_stats": {k: v for k, v in self.last_stats.items() if k != "weights"},
            "weights": self.last_stats.get("weights", {}),
            "decisions": rep["decisions"],
            "warnings": rep["warnings"],
            "report": rep,
        }

    def _finish(self, reason: Optional[str], kind: Optional[str]) -> dict:
        outcome = self._classify(reason, kind)
        self.outcome = outcome
        if outcome.startswith("failure"):
            rec = self._failure_record(outcome, reason)
            self._write_json("failure.json", rec)
            self._event("FAILURE", f"{outcome}: {reason} (details: failure.json)")
        elif outcome == "finished_suspect":
            self._event("SUSPECT", f"finished, but caps were hit: {self.model.caps_hit}")
        else:
            self._event("run", f"{outcome}: {reason}")
        rep = self.model.report()
        rep["outcome"], rep["epochs"] = outcome, self.epoch
        self._write_json("report.json", rep)
        # a non-finite loss must not overwrite the last good checkpoint
        if kind != "nonfinite_loss":
            self.save(os.path.join(self.out_dir, "last.pt"))
        return {"outcome": outcome, "reason": reason, "suspect": self.model.suspect,
                "epochs": self.epoch, "history": self.history}

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Atomic write. Call at epoch boundaries only."""
        ck = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "schedule": self.model.schedule_state(),
            "opt": self.opt.state_dict(),
            "adj_opt": self.adj_opt.state_dict(),
            "history": self.history,
            "suspect_logged": self._suspect_logged,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "run_cfg": dataclasses.asdict(self.model.run_cfg),
            "trainer_cfg": dataclasses.asdict(self.cfg),
            "run_args": self.run_args,
        }
        tmp = path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> dict:
        """Restore everything; the next fit() continues from epoch+1.
        Returns the checkpoint's run_args and configs for main to compare."""
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model.load_schedule_state(ck["schedule"])
        self.opt.load_state_dict(ck["opt"])
        self.adj_opt.load_state_dict(ck["adj_opt"])
        self.history, self.epoch = ck["history"], ck["epoch"]
        self._suspect_logged = ck["suspect_logged"]
        self.last_stats = self.history[-1] if self.history else {}
        r = ck["rng"]
        torch.set_rng_state(r["torch"].cpu())
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
        np.random.set_state(r["numpy"])
        random.setstate(r["python"])
        return {"run_cfg": ck["run_cfg"], "trainer_cfg": ck["trainer_cfg"], "run_args": ck["run_args"]}
