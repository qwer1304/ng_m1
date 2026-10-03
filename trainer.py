"""
trainer.py -- the training loop. Nothing else.

The Trainer owns: optimizers, the epoch loop, the periodic structure check,
logging, checkpointing. It does not compute any loss (model.py / losses.py),
build any network (m1net.py) or build any table (geodesic.py).

=======================================================================
OPTIMIZERS (follows Ng et al., with their overlap bug fixed)
=======================================================================
  * Adam on model.non_adjacency_parameters(), lr = cfg.lr
  * SGD (momentum cfg.adj_momentum) on model.adjacency_parameters(),
    lr = cfg.adj_lr
The two parameter sets are disjoint (Ng's code accidentally put the
adjacency in both lists).

=======================================================================
DATA CONTRACT
=======================================================================
The DataLoader must yield tuples whose FIRST FIVE items are (x, y, t, e, ids)
(any further items, e.g. the extra fields of ti_dataset, are ignored):
    x    (B, C, H, W) float   cached IN-VAE latents
    y,t,e (B,) long           species, time-of-day, location labels
    ids  (B,) long            row indices into the SAME ordering that the
                              GeodesicTable was built from
(see data.py-equivalent helper make_loader in main.py).

=======================================================================
LOOP
=======================================================================
for epoch in 1..epochs:
    for each batch: forward -> loss -> backward -> (clip) -> both optimizers
    mean of each loss term over the epoch is recorded in history
    if epoch % check_epoch == 0: stop_flag = model.end_of_check()
    optional eval_fn(model, epoch) callback (return dict merged into history)
    if stop_flag and cfg.stop_when_all_frozen: break
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from geodesic import GeodesicTable
from model import M1Model

TERMS = ("loss", "kl", "rec", "sparse", "moral", "inj", "bilip")


@dataclass
class TrainerConfig:
    """
    epochs               number of epochs
    lr                   Adam lr for everything except the adjacency
    adj_lr, adj_momentum SGD settings for the adjacency (Ng: 1e-3, 0.9)
    check_epoch          run sink freezing every this many epochs (0 = never)
    stop_when_all_frozen stop early when every content node is frozen
    grad_clip            max global grad-norm (None = no clipping)
    device               "cpu" / "cuda"
    log_every            print a line every this many epochs (0 = silent)
    """

    epochs: int = 100
    lr: float = 1e-3
    adj_lr: float = 1e-3
    adj_momentum: float = 0.9
    check_epoch: int = 5
    stop_when_all_frozen: bool = True
    grad_clip: Optional[float] = None
    device: str = "cpu"
    log_every: int = 1


class Trainer:
    def __init__(
        self,
        model: M1Model,
        cfg: TrainerConfig,
        geod: Optional[GeodesicTable] = None,
        eval_fn: Optional[Callable[[M1Model, int], Dict[str, float]]] = None,
    ):
        """
        model  : M1Model
        geod   : GeodesicTable, required iff model's lambda_inj > 0
        eval_fn: optional callback (model, epoch) -> dict of floats, called
                 after each epoch in eval mode without gradients.
        """
        self.model = model.to(cfg.device)
        self.cfg, self.geod, self.eval_fn = cfg, geod, eval_fn
        self.opt = torch.optim.Adam(model.non_adjacency_parameters(), lr=cfg.lr)
        self.adj_opt = torch.optim.SGD(
            model.adjacency_parameters(), lr=cfg.adj_lr, momentum=cfg.adj_momentum
        )
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def step(self, batch) -> Dict[str, float]:
        """One optimisation step on one (x, y, t, e, ids) batch."""
        dev = self.cfg.device
        x, y, t, e, ids = batch[:5]  # extra fields of the TI dataset ride along
        x, y, t, e = x.to(dev), y.to(dev), t.to(dev), e.to(dev)
        geod_sub = None
        if self.model.losses.cfg.lambda_inj and self.geod is not None:
            geod_sub = self.geod.sub(ids.cpu().numpy(), device=dev)
        res = self.model((x, y, t, e), geod_sub=geod_sub, sample=True)
        self.opt.zero_grad(set_to_none=True)
        self.adj_opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.adj_opt.step()
        return {k: float(res[k].detach()) for k in TERMS}

    # ------------------------------------------------------------------
    def fit(self, loader) -> List[Dict[str, float]]:
        """Run the loop; returns (and stores) per-epoch history."""
        c = self.cfg
        for epoch in range(1, c.epochs + 1):
            self.model.train()
            t0, sums, n = time.time(), {k: 0.0 for k in TERMS}, 0
            for batch in loader:
                r = self.step(batch)
                for k in TERMS:
                    sums[k] += r[k]
                n += 1
            rec = {k: sums[k] / max(n, 1) for k in TERMS}
            rec["epoch"] = epoch
            stop = False
            if c.check_epoch and epoch % c.check_epoch == 0:
                stop = self.model.end_of_check()
                rec["n_unfrozen"] = len(self.model.net.prior._unfrozen())
            if self.eval_fn is not None:
                self.model.eval()
                with torch.no_grad():
                    rec.update(self.eval_fn(self.model, epoch))
            self.history.append(rec)
            if c.log_every and epoch % c.log_every == 0:
                extra = f" unfrozen={rec['n_unfrozen']}" if "n_unfrozen" in rec else ""
                print(
                    f"epoch {epoch:4d} loss={rec['loss']:.4f} kl={rec['kl']:.4f} "
                    f"rec={rec['rec']:.4f} sparse={rec['sparse']:.3f} "
                    f"moral={rec['moral']:.3f} inj={rec['inj']:.4f} "
                    f"bilip={rec['bilip']:.4f}{extra} ({time.time() - t0:.1f}s)"
                )
            if stop and c.stop_when_all_frozen:
                print("all content nodes frozen; stopping.")
                break
        return self.history

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights, moral-loss state and history."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "prev_adj": self.model.prev_adj,
                "k_min": self.model.k_min,
                "history": self.history,
            },
            path,
        )

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ck["model"])
        self.model.prev_adj, self.model.k_min = ck["prev_adj"], ck["k_min"]
        self.history = ck["history"]
