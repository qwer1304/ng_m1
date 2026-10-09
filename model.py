"""
model.py -- ties the network (m1net.M1Net) and the losses (losses.M1Losses),
and takes the GLOBAL scheduling decisions.

The Trainer talks only to M1Model. M1Model:
  * runs the network on a batch and computes the weighted loss (forward),
  * at the end of every epoch, WHEN THE TRAINER CALLS end_epoch(), takes the
    global, cross-loss decisions and tells the Trainer whether to stop,
  * exposes parameter groups, deterministic encoding, the schedule state for
    checkpoints, and a final report.
It adds no trainable parameters of its own and never calls anything on its
own: the Trainer owns every loop and every call.

=======================================================================
WHO DECIDES WHAT
=======================================================================
Each loss owns its computation, ramp, readiness and bookkeeping (losses.py);
the sink-check gate owns the freezing logic and the prior owns the order
mode of the adjacency (m1net.py). The model owns only the rules that involve
more than one of them, listed below, one method each, so each rule can be
read and tuned on its own. All numbers come from config.ScheduleRules;
nothing is hard-coded here.

=======================================================================
THE ORDER BLEND (phase 0, then a ramp to the lower-triangular graph)
=======================================================================
With scheduling enabled the adjacency starts at blend a = 0, the "full" mode
(every latent predicted from all the others, diagonal removed, nothing
learned in the graph): the head and the prior train without any slot order
imprinted on the latents. When the KL ramp is done (KL.ready()),
_rule_tril_start starts a linear ramp of a from 0 to 1 over
schedule.tril_ramp_epochs (the adjacency is (1-a)*full + a*learned
lower-triangular, so the prior changes smoothly and Adam sees no jump).
_rule_tril_ramp advances it, like a loss ramp. When a reaches 1 the mode is
"tril"; KL's settle detector is reset there. SPARSE starts only when KL has
settled after that (or at the cap tril_cap, with a warning). With scheduling
disabled a = 1 from epoch 1.

=======================================================================
end_epoch(epoch, stats): THE FIXED SEQUENCE (epochs are 1-based)
=======================================================================
  1. _update_losses(stats)   every loss advances its ramp and observes stats.
  2. _run_gate(epoch)        the sink-check gate acts, but in enabled mode
                             only once SPARSE has reached gate_open_frac of
                             its target weight (before that the adjacency has
                             no pruning pressure and looks falsely stable).
                             If it froze nodes, the MORAL loss takes its
                             snapshot. A cap hit of the gate becomes a stop
                             request.
  --- only if schedule.enabled (otherwise steps 3 are skipped) ---
  3. the rules, in this order:
       _rule_kl_start             KL starts when REC is ready; cap kl_cap.
       _rule_bilip_start          BILIP starts when REC is ready; cap bilip_cap.
       _rule_tril_start           start the blend ramp "full" -> "tril" when
                                  KL is ready.
       _rule_tril_ramp            advance the blend; at 1, reset KL's settle.
       _rule_sparse_start         SPARSE starts when KL has settled after the
                                  blend reached 1; cap tril_cap.
       _rule_moral_start          MORAL starts at the first sink check.
       _rule_off_when_all_frozen  SPARSE and MORAL switch off when every
                                  content node is frozen.
  4. _run_control(epoch)     decides stop: a stop request, the end of the
                             fine-tune phase, or max_epochs.
  5. returns a Status (stop, reason, suspect, this epoch's decisions and
     warnings, current weights).
stats must contain the epoch means of the unweighted terms
(stats["rec"], stats["kl"], ...), computed by the Trainer; stats["val_kl"]
is used by KL's settle detector when present.
A decision takes effect from the next epoch (weights and the order mode are
read on every forward pass; a ramp started now uses its initial weight next
epoch).

Every decision and warning is appended to a log (epoch, text), kept in the
schedule state, so a finished run shows exactly why each loss started or
stopped when it did.

=======================================================================
DISABLED MODE (schedule.enabled = False)
=======================================================================
Only steps 1 and 2 run, and the gate is on its fixed clock. Every loss uses
its target weight from epoch 1 and the order blend is 1 ("tril") from epoch 1.
The model never says stop except at max_epochs. (The MORAL snapshot is still
taken after a freeze, as in the original model.)

=======================================================================
CHECKPOINT
=======================================================================
schedule_state() / load_schedule_state(s): everything non-learnable that the
model owns or assembles: the losses' state (incl. MORAL's prev_adj/k_min,
REC's plateau history and KL's settle history), the gate state and order
blend, run control (incl. the epochs of the switch), the decision log. The
Trainer calls them; sure_mask / sure_adj are buffers and are saved with the
normal state_dict.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import RunConfig
from losses import M1Losses
from m1net import M1Net, M1NetConfig, ORDER_FULL, ORDER_TRIL


# ======================================================================
# Status / per-epoch log
# ======================================================================
@dataclass
class Status:
    """What end_epoch returns to the Trainer."""

    stop: bool
    reason: Optional[str]
    suspect: bool
    decisions: List[str] = field(default_factory=list)  # this epoch
    warnings: List[str] = field(default_factory=list)  # this epoch
    weights: Dict[str, float] = field(default_factory=dict)  # for the next epoch


class _EpochLog:
    """Collects what happens during one end_epoch call."""

    def __init__(self):
        self.decisions: List[str] = []
        self.warnings: List[str] = []
        self.stops: List[str] = []  # stop requests (reasons)
        self.caps: List[str] = []  # caps hit (names)

    def decide(self, text: str) -> None:
        self.decisions.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def stop(self, text: str) -> None:
        self.stops.append(text)

    def cap(self, name: str) -> None:
        self.caps.append(name)


# ======================================================================
# The model
# ======================================================================
class M1Model(nn.Module):
    """
    Args
        net_cfg  : M1NetConfig (network configuration, includes HeadConfig).
                   Its `gate` field is replaced by run_cfg.gate.
        run_cfg  : RunConfig (loss, gate and schedule sub-configs). It is
                   finalized here (derived fields set, validated).
    """

    def __init__(self, net_cfg: M1NetConfig, run_cfg: RunConfig):
        super().__init__()
        self.run_cfg = run_cfg.finalize()
        net_cfg = dataclasses.replace(net_cfg, gate=self.run_cfg.gate)
        self.net = M1Net(net_cfg)
        self.losses = M1Losses(self.run_cfg.loss, net_cfg.n_content)
        self.rules = self.run_cfg.schedule
        # phase 0 (enabled mode) starts with the full off-diagonal mask (blend
        # 0); a resume overwrites this through load_schedule_state
        self.net.set_order_mode(ORDER_FULL if self.rules.enabled else ORDER_TRIL)
        self._init_run_state()

    def _init_run_state(self) -> None:
        self.frozen_epoch: Optional[int] = None  # epoch at which all nodes froze
        self.tril_epoch: Optional[int] = None  # epoch at which the blend ramp started
        self.tril_pos = 0  # epochs of the blend ramp done so far
        self.tril_done_epoch: Optional[int] = None  # epoch at which the blend reached 1
        self.caps_hit: List[list] = []  # [epoch, name]
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self.decision_log: List[list] = []  # [epoch, text]
        self.warning_log: List[list] = []  # [epoch, text]
        self.gate_info: dict = {}  # diagnostics of a gate cap hit
        self.gate_open_epoch: Optional[int] = None  # first epoch the gate ran

    # ------------------------------------------------------------------
    # forward / small helpers
    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Tuple[torch.Tensor, ...],
        geod_sub: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
            batch    : (x, y, t, e) with x (B,C,H,W) float, y/t/e (B,) long.
            geod_sub : (B, B) geodesic sub-table for this batch; needed iff
                       needs_geod() is True.
            sample   : sample z (training) or use mu (evaluation).
        Returns the dict of loss terms from M1Losses ("kl","rec","sparse",
        "moral","inj","bilip","loss") plus the network output under "out"
        (a dict; not a tensor) for logging/metrics.
        """
        x, y, t, e = batch
        out = self.net(x, y, t, e, sample=sample)
        terms = self.losses(out, x, decode_fn=self.net.head.decode, geod_sub=geod_sub)
        terms["out"] = out
        return terms

    def needs_geod(self) -> bool:
        """True iff the injective loss has a non-zero weight right now (the
        Trainer then has to supply a geodesic sub-table)."""
        return self.losses.terms["inj"].weight() > 0

    def set_rec_baseline(self, value: float) -> None:
        """Trivial REC (data variance of x), measured once by the caller; used
        by the KL cap rule."""
        self.losses.terms["rec"].set_baseline(value)

    @property
    def all_frozen(self) -> bool:
        return len(self.net.prior._unfrozen()) == 0

    # ------------------------------------------------------------------
    # end of epoch: the Trainer calls this once per epoch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def end_epoch(self, epoch: int, stats: Dict[str, float]) -> Status:
        log = _EpochLog()
        self._update_losses(stats)
        self._run_gate(epoch, log)
        if self.rules.enabled:
            self._rule_kl_start(epoch, log)
            self._rule_bilip_start(epoch, log)
            self._rule_tril_start(epoch, log)
            self._rule_tril_ramp(epoch, log)
            self._rule_sparse_start(epoch, log)
            self._rule_moral_start(epoch, log)
            self._rule_off_when_all_frozen(epoch, log)
        status = self._run_control(epoch, log)
        self.decision_log += [[epoch, t] for t in log.decisions]
        self.warning_log += [[epoch, t] for t in log.warnings]
        return status

    # ---------------- step 1 ----------------
    def _update_losses(self, stats: Dict[str, float]) -> None:
        """Every loss advances its ramp and observes the epoch stats."""
        self.losses.end_epoch(stats)

    # ---------------- step 2 ----------------
    def _run_gate(self, epoch: int, log: _EpochLog) -> None:
        """
        The sink-check gate acts (its mode, stable or clock, is in its config).
        If it froze nodes, MORAL snapshots the adjacency (reference for the
        next iteration). A cap hit of the gate is a stop request; its
        diagnostics are kept in self.gate_info.
        """
        if not self._gate_open(epoch, log):
            return
        res = self.net.gate_end_epoch(epoch)
        if res.froze:
            n_frozen = self.net.n_content - len(self.net.prior._unfrozen())
            self.losses.terms["moral"].snapshot(self.net.prior.get_adj().detach(), n_frozen)
            log.decide(f"gate: froze nodes {res.froze} ({n_frozen} of "
                       f"{self.net.n_content} frozen)")
        if res.stop:
            log.cap("sink_check")
            log.stop(res.stop)
            self.gate_info = res.info

    def _gate_open(self, epoch: int, log: _EpochLog) -> bool:
        """
        Enabled mode: the gate stays closed until SPARSE's weight is at
        least gate_open_frac of its target; then it opens for good (the
        gate's own counters start from that epoch). Disabled mode: always
        open (the gate runs on its fixed clock).
        """
        if not self.rules.enabled or self.gate_open_epoch is not None:
            return True
        sp = self.losses.terms["sparse"]
        if sp.started and sp.fraction() >= self.rules.gate_open_frac:
            self.gate_open_epoch = epoch
            log.decide(f"gate: opened (SPARSE at {sp.fraction():.2f} of its target)")
            return True
        return False

    # ---------------- step 3: the rules ----------------
    def _rule_kl_start(self, epoch: int, log: _EpochLog) -> None:
        """
        KL starts its ramp when REC is ready (plateau). Cap: if the plateau
        has not come by `kl_cap` epochs, ask REC whether the head is learning
        at all (REC < rec_frac * baseline): yes -> start KL anyway, with a
        warning; no -> stop the run. Baseline unknown -> cannot judge, start
        KL with a warning saying so.
        """
        T, r = self.losses.terms, self.rules
        if T["kl"].started:
            return
        if T["rec"].ready():
            T["kl"].start()
            log.decide("kl: ramp started (REC plateau)")
        elif epoch >= r.kl_cap:
            log.cap("kl")
            learning = T["rec"].is_learning(r.rec_frac)
            if learning is None:
                T["kl"].start()
                log.warn(f"kl: no REC plateau by epoch {r.kl_cap}; REC baseline unknown, "
                         f"cannot judge; ramp started anyway")
            elif learning:
                T["kl"].start()
                log.warn(f"kl: no REC plateau by epoch {r.kl_cap}; REC is below "
                         f"{r.rec_frac:g} x baseline; ramp started anyway")
            else:
                log.stop(f"kl: no REC plateau by epoch {r.kl_cap} and REC is not below "
                         f"{r.rec_frac:g} x baseline: the head is not learning")

    def _rule_bilip_start(self, epoch: int, log: _EpochLog) -> None:
        """BILIP starts its ramp when REC is ready (plateau). Cap: `bilip_cap`
        epochs, then start anyway, with a warning."""
        T, r = self.losses.terms, self.rules
        if T["bilip"].started:
            return
        if T["rec"].ready():
            T["bilip"].start()
            log.decide("bilip: ramp started (REC plateau)")
        elif epoch >= r.bilip_cap:
            log.cap("bilip")
            T["bilip"].start()
            log.warn(f"bilip: no REC plateau by epoch {r.bilip_cap}; ramp started anyway")

    def _rule_tril_start(self, epoch: int, log: _EpochLog) -> None:
        """
        Start the blend ramp from the full off-diagonal mask to the learned
        lower-triangular one when KL is ready (its ramp reached the target),
        once. The blend stays 0 until the end of the next epoch (as a loss
        ramp uses its initial weight in the epoch after it started). With
        tril_ramp_epochs <= 0 the switch is immediate.
        """
        if self.tril_epoch is not None:
            return
        T, r = self.losses.terms, self.rules
        if T["kl"].ready():
            self.tril_epoch, self.tril_pos = epoch, 0
            log.decide(f"tril: blend ramp started (KL ramp done; "
                       f"{r.tril_ramp_epochs} epochs)")
            if r.tril_ramp_epochs <= 0:
                self._finish_tril(epoch, log)

    def _rule_tril_ramp(self, epoch: int, log: _EpochLog) -> None:
        """
        Advance the blend by one step at the end of every epoch after the
        one in which the ramp started; when it reaches 1, finish.
        """
        if self.tril_epoch is None or self.tril_done_epoch is not None:
            return
        if epoch <= self.tril_epoch:
            return
        R = self.rules.tril_ramp_epochs
        self.tril_pos += 1
        a = min(1.0, self.tril_pos / R)
        self.net.set_order_blend(a)
        if a >= 1.0:
            self._finish_tril(epoch, log)

    def _finish_tril(self, epoch: int, log: _EpochLog) -> None:
        """The blend is 1: mode "tril". KL's settle detector starts from
        here (the history from before it must not count)."""
        self.net.set_order_mode(ORDER_TRIL)
        self.losses.terms["kl"].reset_settle()
        self.tril_done_epoch = epoch
        log.decide("tril: adjacency is lower-triangular now (blend 1)")

    def _rule_sparse_start(self, epoch: int, log: _EpochLog) -> None:
        """
        SPARSE starts its ramp once the blend has reached 1 and KL has
        settled since (KL.settled(), counted from then). Cap: `tril_cap`
        epochs after the blend reached 1, then start anyway, with a warning
        (a reversible decision: the run continues).
        """
        T, r = self.losses.terms, self.rules
        if T["sparse"].started or self.tril_done_epoch is None:
            return
        if T["kl"].settled():
            T["sparse"].start()
            log.decide("sparse: ramp started (KL settled after the switch to tril)")
        elif epoch - self.tril_done_epoch >= r.tril_cap:
            log.cap("tril")
            T["sparse"].start()
            log.warn(f"sparse: KL not settled {r.tril_cap} epochs after the adjacency "
                     f"became lower-triangular (epoch {self.tril_done_epoch}); "
                     f"ramp started anyway")

    def _rule_moral_start(self, epoch: int, log: _EpochLog) -> None:
        """MORAL starts its ramp at the first successful sink check (its
        reference adjacency exists from then on)."""
        T = self.losses.terms
        if not T["moral"].started and self.net.prior.gate.first_check_done:
            T["moral"].start()
            log.decide("moral: ramp started (first sink check)")

    def _rule_off_when_all_frozen(self, epoch: int, log: _EpochLog) -> None:
        """SPARSE and MORAL switch off once every content node is frozen (the
        unresolved block is empty and the graph is fixed)."""
        if not self.all_frozen:
            return
        T = self.losses.terms
        for n in ("sparse", "moral"):
            if not T[n].off:
                T[n].switch_off()
                log.decide(f"{n}: switched off (all content nodes frozen)")

    # ---------------- step 4 ----------------
    def _run_control(self, epoch: int, log: _EpochLog) -> Status:
        """
        Decide stop, in this priority: a stop request (cap rules, gate cap);
        the end of the fine-tune phase (enabled mode only); max_epochs.
        Also keeps the list of caps hit and the epoch at which all nodes froze.
        """
        r = self.rules
        self.caps_hit += [[epoch, c] for c in log.caps]
        if self.all_frozen and self.frozen_epoch is None:
            self.frozen_epoch = epoch
            log.decide("run: all content nodes frozen")
        reason = None
        if log.stops:
            reason = "; ".join(log.stops)
        elif (r.enabled and self.frozen_epoch is not None
              and epoch - self.frozen_epoch >= r.finetune_epochs):
            reason = "finished: all nodes frozen and fine-tune phase done"
        elif epoch >= r.max_epochs:
            reason = "max_epochs reached"
        self.stopped = reason is not None
        self.stop_reason = reason
        return Status(stop=self.stopped, reason=reason, suspect=self.suspect,
                      decisions=list(log.decisions), warnings=list(log.warnings),
                      weights=self.losses.weights())

    @property
    def suspect(self) -> bool:
        """Any cap hit, or (enabled mode) stopped before all nodes froze."""
        return bool(self.caps_hit) or (
            self.rules.enabled and self.stopped and self.frozen_epoch is None)

    # ------------------------------------------------------------------
    # checkpoint state and final report
    # ------------------------------------------------------------------
    def schedule_state(self) -> dict:
        return {
            "losses": self.losses.schedule_state(),
            "net": self.net.schedule_state(),
            "run": {"frozen_epoch": self.frozen_epoch, "tril_epoch": self.tril_epoch,
                    "tril_pos": self.tril_pos, "tril_done_epoch": self.tril_done_epoch,
                    "caps_hit": [list(c) for c in self.caps_hit],
                    "stopped": self.stopped, "stop_reason": self.stop_reason,
                    "gate_info": self.gate_info, "gate_open_epoch": self.gate_open_epoch},
            "decision_log": [list(x) for x in self.decision_log],
            "warning_log": [list(x) for x in self.warning_log],
        }

    def load_schedule_state(self, s: dict) -> None:
        self.losses.load_schedule_state(s["losses"])
        self.net.load_schedule_state(s["net"])
        run = s["run"]
        self.frozen_epoch, self.stopped = run["frozen_epoch"], run["stopped"]
        # .get: checkpoints written before the order mode existed have no such key
        self.tril_epoch = run.get("tril_epoch")
        self.tril_pos = run.get("tril_pos", 0)
        self.tril_done_epoch = run.get("tril_done_epoch")
        if "tril_epoch" in run and self.tril_epoch is not None and self.tril_done_epoch is None \
                and "tril_pos" not in run:  # state from the previous (immediate-switch) version
            self.tril_done_epoch = self.tril_epoch
        self.caps_hit = [list(c) for c in run["caps_hit"]]
        self.stop_reason, self.gate_info = run["stop_reason"], run["gate_info"]
        self.gate_open_epoch = run["gate_open_epoch"]
        self.decision_log = [list(x) for x in s["decision_log"]]
        self.warning_log = [list(x) for x in s["warning_log"]]

    @torch.no_grad()
    def report(self) -> dict:
        """
        Final report (validation accuracy is the Trainer's, not included):
        whether all nodes froze, freeze order and epochs, the epochs at which the
        blend ramp to the lower-triangular order started and finished, caps hit, the final graph with
        degrees (parents = row sums, children = column sums), the suspect
        flag, the decision and warning logs, per-loss info.
        """
        adj = self.net.prior.get_adj().detach().int()
        return {
            "all_frozen": self.all_frozen,
            "freeze_log": [list(x) for x in self.net.prior.gate.freeze_log],  # [epoch, node]
            "order_mode": self.net.order_mode,
            "order_blend": self.net.order_blend,
            "tril_epoch": self.tril_epoch,
            "tril_done_epoch": self.tril_done_epoch,
            "caps_hit": [list(c) for c in self.caps_hit],
            "suspect": self.suspect,
            "stop_reason": self.stop_reason,
            "adjacency": adj.tolist(),
            "parents": adj.sum(1).tolist(),
            "children": adj.sum(0).tolist(),
            "gate_info": self.gate_info,
            "decisions": [list(x) for x in self.decision_log],
            "warnings": [list(x) for x in self.warning_log],
            "losses_info": {n: t.info() for n, t in self.losses.terms.items()},
        }

    # ------------------------------------------------------------------
    # parameters / encoding
    # ------------------------------------------------------------------
    def adjacency_parameters(self) -> List[nn.Parameter]:
        return self.net.adjacency_parameters()

    def non_adjacency_parameters(self) -> List[nn.Parameter]:
        return self.net.non_adjacency_parameters()

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.encode_mu(x)
