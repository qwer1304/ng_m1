"""
losses.py -- ALL loss computations of the M1 pipeline, in one place.

Nothing else in the code base computes a loss. The network (m1net.py) only
produces tensors; the model (model.py) calls M1Losses on them; the Trainer
only calls backward() on the total.

=======================================================================
WHO DECIDES WHAT
=======================================================================
Each loss class is a self-contained object. It owns, behind methods:
  * how its value is computed (forward),
  * its own weight mechanics: target, ramp, normalization (weight()),
  * its own state (started, ramp position, off, plus whatever else it needs),
  * its own end-of-epoch bookkeeping (end_epoch(stats)),
  * its own answer to "am I ready?" (ready()).
The MODEL makes only the global, cross-loss decisions, from the user cfg,
and executes them through the methods below. Examples: "do not start KL
until REC is ready", "start SPARSE when KL is ready", "switch SPARSE and
MORAL off when all content nodes are frozen", "if REC has not plateaued by
epoch 30, ...". The model never computes a loss and never looks inside one.

Methods the model calls on every loss (see ScheduledLoss):
    end_epoch(stats)  once per epoch, FIRST: advances the ramp and lets the
                      loss observe the epoch statistics it cares about.
    ready()           has this loss reached the state other losses may wait
                      for?
    fraction()        current weight / target (0..1), e.g. "SPARSE is at
                      half of its weight".
    start()           start its ramp (the model decides when).
    switch_off()      weight becomes 0 for good (the model decides when).
    weight()          current weight (read on every forward pass).
    schedule_state() / load_schedule_state(s)   its non-learnable state.
Per epoch the model calls end_epoch() on all losses first and only then
start() / switch_off(), so a ramp started at the end of epoch e uses its
initial weight in epoch e+1 and gains one step at the end of each following
epoch.

stats passed to end_epoch: dict with the epoch means of the unweighted terms
(stats["rec"], stats["kl"], ...), computed by the Trainer.

=======================================================================
THE TOTAL LOSS (per minibatch, all terms are batch means / scalars)
=======================================================================
  L =  w_kl * KL + w_rec * REC + w_sparse * SPARSE + w_moral * MORAL
     + w_inj * INJ + w_bilip * BILIP
with w_x = term.weight(). A term whose weight is 0 at this moment is NOT
computed (reported as 0.0), so the costly BLAE terms cost nothing while
inactive. Each class docstring below states the decisions made for that
loss.

=======================================================================
WEIGHT MECHANICS (ScheduledLoss)
=======================================================================
  weight() =
    schedule disabled          -> target (always)
    switched off               -> 0
    ramp not started           -> target * init_frac   (0 if init_frac None)
    ramp started               -> target * (f0 + (1 - f0) * min(1, pos / ramp_epochs))
                                  with f0 = init_frac or 0; ramp_epochs <= 0
                                  means target right at start.
  Constant losses (REC, INJ) are created already started.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacfwd, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel

from config import LossConfig


# ----------------------------------------------------------------------
# Individual loss functions (pure)
# ----------------------------------------------------------------------
def kl_mc(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Single-sample MC KL: mean over batch of (log_q - log_p). Inputs (B,)."""
    return (log_q - log_p).mean()


def sparsity_loss(current_adj: torch.Tensor) -> torch.Tensor:
    """sum((I+A)^T (I+A)) over the unresolved block A of shape (m, m).
    Returns 0 for m == 0. (Unnormalized; SparseLoss divides by a constant.)"""
    m = current_adj.shape[0]
    if m == 0:
        return current_adj.sum() * 0.0
    ipa = torch.eye(m, device=current_adj.device) + current_adj
    return (ipa.T @ ipa).sum()


def moral_loss(adj: torch.Tensor, prev_adj: torch.Tensor, k_min: int) -> torch.Tensor:
    """
    Moral-graph consistency (see MoralLoss).

    adj      (n, n) current adjacency (differentiable).
    prev_adj (n, n) previous binary adjacency (constant).
    k_min    first block size included; blocks k_min..n are summed. If
             k_min > n the sum is empty and 0 is returned.
    """
    n = adj.shape[0]
    total = adj.sum() * 0.0
    eye = torch.eye(n, device=adj.device)
    for k in range(max(int(k_min), 1), n + 1):
        a = eye[:k, :k] + adj[:k, :k]
        p = eye[:k, :k] + prev_adj[:k, :k]
        total = total + ((a.T @ a - p.T @ p) ** 2).sum()
    return total


def injective_loss(geod: torch.Tensor, z: torch.Tensor, thresh: float = 0.3) -> torch.Tensor:
    """
    BLAE InjectiveLoss. Keeps ||z_i - z_j|| between `thresh` and 1 times the
    geodesic distance of the corresponding inputs, so distinct inputs cannot
    collapse to one latent.

    geod   (b, b) geodesic distances of the minibatch (GeodesicTable.sub).
    z      (b, D) latents; PASS mu, not a sampled z.
    thresh lower ratio bound (BLAE default 0.3).
    Pairs used: geod > 5% of the batch maximum and latent distance > 0
    (as in BLAE). Returns 0 if no pair qualifies.
    """
    p = torch.cdist(z, z, p=2)
    upper = torch.triu(torch.ones_like(geod), diagonal=1).bool()
    mask = (geod > 0.05 * geod.max()) & (p > 0) & upper
    if not mask.any():
        return z.sum() * 0.0
    lip = p[mask] / geod[mask]
    return torch.mean(F.relu(math.log(thresh) - torch.log(lip)) + 5.0 * F.relu(lip - 1.0))


def bilipschitz_decoder_loss(
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    L: float = 2.0,
    subset_frac: float = 0.3,
) -> torch.Tensor:
    """
    BLAE decoder-Jacobian bi-Lipschitz regulariser. With J = d decode/d z and
    G = J^T J (D x D pulled-back metric): penalise the off-diagonal entries of
    G (latent directions should stay orthogonal in feature space) and diagonal
    entries above L (upper Lipschitz bound). No lower-bound term (BLAE uses 0).

    decode_fn  (b, D) -> (b, ...) e.g. HEAD.decode.
    z          (b, D) latents; PASS mu.
    L          upper bound on diag(G).
    subset_frac fraction of the batch (random) on which the Jacobian is
               computed.
    The fused attention kernels in AutoencoderKL do not support forward-mode
    AD, so the plain "math" attention backend is forced inside this function.
    """
    b = z.shape[0]
    k = max(1, round(b * subset_frac))
    zs = z[torch.randperm(b, device=z.device)[:k]]
    single = lambda v: decode_fn(v.unsqueeze(0)).squeeze(0)
    with sdpa_kernel(SDPBackend.MATH):
        jac = vmap(jacfwd(single))(zs)  # (k, *out_shape, D)
    jac = jac.reshape(k, -1, zs.shape[1])
    G = torch.bmm(jac.transpose(1, 2), jac)  # (k, D, D)
    D = zs.shape[1]
    diag = torch.diagonal(G, dim1=1, dim2=2)
    off = G[:, ~torch.eye(D, dtype=torch.bool, device=z.device)]
    return torch.mean(off.pow(2).sum(1)) + torch.mean(F.relu(diag - L).pow(2).sum(1))


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class ScheduledLoss(nn.Module):
    """
    Base class of every loss: weight mechanics, state, and the hooks the
    model calls (see the module docstring, "WHO DECIDES WHAT").

    Args
        name            key used in every dict ("rec", "kl", ...).
        target          final weight.
        ramp_epochs     epochs from the initial weight to the target
                        (<= 0: target right at start).
        init_frac       weight before the ramp starts, as a fraction of
                        target. None = inactive (weight 0) until started.
        active_at_start True = created already started (constant terms).
        enabled         False = scheduling disabled globally: weight() is
                        always `target`, and every hook below is a no-op.

    Hooks (a subclass overrides what it needs):
        end_epoch(stats)  advance() then observe(stats). The model calls it
                          on every loss, first, once per epoch.
        observe(stats)    loss-specific bookkeeping from the epoch stats.
        ready()           default: started and the ramp has reached the
                          target (ramp_done).
        advance()/start()/switch_off()/weight()/ramp_done    mechanics.
        schedule_state()/load_schedule_state(s)    non-learnable state.
    """

    def __init__(self, name: str, target: float, ramp_epochs: int = 0,
                 init_frac: Optional[float] = None, active_at_start: bool = False,
                 enabled: bool = True):
        super().__init__()
        self.name = name
        self.target = float(target)
        self.ramp_epochs = int(ramp_epochs)
        self.init_frac = init_frac
        self.enabled = bool(enabled)
        # non-learnable state
        self.started = bool(active_at_start)
        self.pos = 0
        self.off = False

    # ---------------- hooks ----------------
    def end_epoch(self, stats: Dict[str, float]) -> None:
        self.advance()
        if self.enabled:
            self.observe(stats)

    def observe(self, stats: Dict[str, float]) -> None:
        pass

    def ready(self) -> bool:
        return self.ramp_done

    def fraction(self) -> float:
        """Current weight as a fraction of the target (1.0 if the target is 0)."""
        return 1.0 if self.target == 0 else self.weight() / self.target

    def info(self) -> dict:
        """Loss-specific numbers for logging (default: none)."""
        return {}

    # ---------------- mechanics ----------------
    def advance(self) -> None:
        if self.enabled and self.started and not self.off:
            self.pos = min(self.pos + 1, max(self.ramp_epochs, 0))

    def start(self) -> None:
        if self.enabled and not self.started:
            self.started, self.pos = True, 0

    def switch_off(self) -> None:
        if self.enabled:
            self.off = True

    @property
    def ramp_done(self) -> bool:
        return self.started and self.pos >= max(self.ramp_epochs, 0)

    def weight(self) -> float:
        if not self.enabled:
            return self.target
        if self.off:
            return 0.0
        f0 = 0.0 if self.init_frac is None else float(self.init_frac)
        if not self.started:
            return self.target * f0
        if self.ramp_epochs <= 0:
            return self.target
        return self.target * (f0 + (1.0 - f0) * min(1.0, self.pos / self.ramp_epochs))

    # ---------------- state ----------------
    def schedule_state(self) -> dict:
        return {"started": self.started, "pos": self.pos, "off": self.off}

    def load_schedule_state(self, s: dict) -> None:
        self.started, self.pos, self.off = s["started"], s["pos"], s["off"]


# ----------------------------------------------------------------------
# The six terms
# ----------------------------------------------------------------------
class RecLoss(ScheduledLoss):
    """
    REC: F.mse_loss(x_hat, x), i.e. a Gaussian likelihood with fixed unit
    variance (Ng et al.). x is the CACHED IN-VAE latent, not the image.

    Decisions for this loss
      * constant weight from epoch 1; never ramped, never switched off.
      * REC is the loss the others wait for. It owns the PLATEAU detector:
        ready() is True once the epoch-mean REC has changed by less than
        `plateau_tol` (relative) between `plateau_window` epochs ago and now.
        The plateau latches (it does not un-fire).
      * It also answers is_learning(frac): is the last epoch-mean REC below
        frac * baseline, with baseline the trivial reconstruction error (the
        data variance, given by set_baseline)? The model uses this when a
        wait for the plateau reaches its cap.
    """

    def __init__(self, target: float, plateau_window: int = 5, plateau_tol: float = 0.01,
                 enabled: bool = True):
        super().__init__("rec", target, active_at_start=True, enabled=enabled)
        self.plateau_window, self.plateau_tol = int(plateau_window), float(plateau_tol)
        self.history: List[float] = []
        self.plateau = False
        self.last: Optional[float] = None
        self.baseline: Optional[float] = None

    def forward(self, out, x):
        return F.mse_loss(out["x_hat"], x)

    def observe(self, stats):
        rec = float(stats["rec"])
        self.last = rec
        self.history = (self.history + [rec])[-(self.plateau_window + 1):]
        if not self.plateau and len(self.history) == self.plateau_window + 1:
            old, new = self.history[0], self.history[-1]
            if abs(new - old) < self.plateau_tol * max(abs(old), 1e-12):
                self.plateau = True

    def ready(self) -> bool:
        return self.plateau

    def set_baseline(self, value: float) -> None:
        """Trivial REC (data variance of x), measured once by the caller."""
        self.baseline = float(value)

    def is_learning(self, frac: float = 0.5) -> Optional[bool]:
        """last epoch-mean REC < frac * baseline. None if either is unknown."""
        if self.last is None or self.baseline is None:
            return None
        return self.last < frac * self.baseline

    def info(self) -> dict:
        return {"last": self.last, "baseline": self.baseline, "plateau": self.plateau}

    def schedule_state(self) -> dict:
        s = super().schedule_state()
        s.update({"history": list(self.history), "plateau": self.plateau,
                  "last": self.last, "baseline": self.baseline})
        return s

    def load_schedule_state(self, s: dict) -> None:
        super().load_schedule_state(s)
        self.history, self.plateau = list(s["history"]), s["plateau"]
        self.last, self.baseline = s["last"], s["baseline"]


class KLLoss(ScheduledLoss):
    """
    KL: single-sample Monte-Carlo estimate mean_b[log q(z|x) - log p(z|y,t,e)],
    both summed over the D latents (no analytic KL exists because the content
    prior mean depends on z itself).

    Decisions for this loss
      * stage A: weight target * init_frac (kl_init_frac, e.g. 1e-3) from
        epoch 1, so the head learns to reconstruct before the prior pulls.
      * the ramp (linear, kl_ramp_epochs, to target) is started by the model
        once REC.ready() (the plateau). The model owns the cap on that wait.
      * ready() (default) is True once the ramp has reached the target; the
        model starts SPARSE then.
    """

    def __init__(self, target: float, init_frac: float, ramp_epochs: int,
                 enabled: bool = True):
        super().__init__("kl", target, ramp_epochs=ramp_epochs, init_frac=init_frac,
                         enabled=enabled)

    def forward(self, out):
        return kl_mc(out["log_q"], out["log_p"])


class SparseLoss(ScheduledLoss):
    """
    SPARSE: sum of all entries of (I+A)^T (I+A), A = the still-unresolved
    block of the content adjacency (out["current_adj"]); equals the L1 norm of
    the paper. DIVIDED BY the fixed constant n_content*(n_content-1)/2 (the
    number of candidate edges; 1 if n_content < 2), so the weight does not
    have to be retuned when n_content changes. The identity part is NOT
    subtracted, so the value is never 0 while the block is non-empty.

    Decisions for this loss
      * inactive (weight 0) until the model starts it, when KL.ready().
      * linear ramp (sparse_ramp_epochs) from 0 to target.
      * the model switches it off when all content nodes are frozen (the
        block is then empty and the graph is fixed).
    """

    def __init__(self, target: float, n_content: int, ramp_epochs: int,
                 enabled: bool = True):
        super().__init__("sparse", target, ramp_epochs=ramp_epochs, enabled=enabled)
        self.norm = float(max(1, n_content * (n_content - 1) // 2))

    def forward(self, out):
        return sparsity_loss(out["current_adj"]) / self.norm


class MoralLoss(ScheduledLoss):
    """
    MORAL (Ng et al., paper Sec. 4.3; NOT in their released code): sum over
    k = k_min..n_content of
        || (I + A[:k,:k])^T (I + A[:k,:k]) - (I + Aprev[:k,:k])^T (I + Aprev[:k,:k]) ||_F^2
    with A the current adjacency (out["adj"], straight-through), Aprev the
    binary adjacency snapshotted after the previous sink check, and [:k,:k]
    the leading k x k block (index order = causal order). Frozen entries are
    constants and cancel in the difference, so it measures the drift of the
    unresolved block (weighting low-index unfrozen nodes more than the paper
    does; a reweighting is deferred).

    State held HERE: prev_adj (a plain attribute, not a buffer) and k_min.
    snapshot(adj, n_frozen) sets them; the model calls it after a sink check
    that froze nodes. Before the first snapshot MORAL = 0. After a check with
    f frozen nodes, k_min = n_content - f + 1 (paper: k = n+2-t..n with t-1
    sinks removed). Both are saved by schedule_state().

    Decisions for this loss
      * target 0 for the first runs (it is then never computed).
      * inactive until the model starts it, at the first sink check; linear
        ramp (moral_ramp_epochs, a placeholder) from 0 to target.
      * the model switches it off when all content nodes are frozen.
    """

    def __init__(self, target: float, n_content: int, ramp_epochs: int,
                 enabled: bool = True):
        super().__init__("moral", target, ramp_epochs=ramp_epochs, enabled=enabled)
        self.n_content = int(n_content)
        self.prev_adj: Optional[torch.Tensor] = None
        self.k_min: int = self.n_content + 1  # empty sum until the first snapshot

    @torch.no_grad()
    def snapshot(self, adj: torch.Tensor, n_frozen: int) -> None:
        self.prev_adj = adj.detach().cpu().clone()
        self.k_min = self.n_content - int(n_frozen) + 1

    def forward(self, out):
        adj = out["adj"]
        if self.prev_adj is None:
            return adj.sum() * 0.0
        return moral_loss(adj, self.prev_adj.to(adj.device), self.k_min)

    def schedule_state(self) -> dict:
        s = super().schedule_state()
        s.update({"prev_adj": self.prev_adj, "k_min": self.k_min})
        return s

    def load_schedule_state(self, s: dict) -> None:
        super().load_schedule_state(s)
        self.prev_adj, self.k_min = s["prev_adj"], s["k_min"]


class InjLoss(ScheduledLoss):
    """
    INJ: BLAE injective loss (Zhan et al. 2026, InjectiveLoss), computed on
    mu. Needs a (b,b) geodesic sub-table for the minibatch (geod_sub).

    Decisions for this loss
      * a small constant weight from epoch 1 (0.1, a guess); no ramp, never
        switched off. It keeps distinct inputs apart while the head learns.
    """

    def __init__(self, target: float, thresh: float = 0.3, enabled: bool = True):
        super().__init__("inj", target, active_at_start=True, enabled=enabled)
        self.thresh = thresh

    def forward(self, out, geod_sub):
        return injective_loss(geod_sub, out["mu"], self.thresh)


class BilipLoss(ScheduledLoss):
    """
    BILIP: BLAE decoder-Jacobian bi-Lipschitz term, computed on mu. By far the
    most expensive loss (about D decoder passes per sample on a random
    fraction `subset_frac` of the batch; the plain "math" attention backend is
    forced because the fused kernels have no forward-mode rule).

    Decisions for this loss
      * inactive until the model starts it, when REC.ready() (the plateau);
        the model owns the cap on that wait (60 epochs, then start anyway).
      * linear ramp (bilip_ramp_epochs) from 0 to target (1e-4, a
        placeholder).
      * its first computed value is kept (info()["first_value"]) so the
        target can be set against the real magnitude. It is the first value
        computed with weight > 0, i.e. one epoch after the ramp starts.
    """

    def __init__(self, target: float, ramp_epochs: int, L: float = 2.0,
                 subset_frac: float = 0.3, enabled: bool = True):
        super().__init__("bilip", target, ramp_epochs=ramp_epochs, enabled=enabled)
        self.L, self.subset_frac = L, subset_frac
        self.first_value: Optional[float] = None

    def forward(self, out, decode_fn):
        v = bilipschitz_decoder_loss(decode_fn, out["mu"], self.L, self.subset_frac)
        if self.first_value is None:
            self.first_value = float(v.detach())
        return v

    def info(self) -> dict:
        return {"first_value": self.first_value}

    def schedule_state(self) -> dict:
        s = super().schedule_state()
        s["first_value"] = self.first_value
        return s

    def load_schedule_state(self, s: dict) -> None:
        super().load_schedule_state(s)
        self.first_value = s["first_value"]


# ----------------------------------------------------------------------
# Aggregator (LossConfig lives in config.py)
# ----------------------------------------------------------------------
class M1Losses(nn.Module):
    """
    Holds the six terms (self.terms, in the order rec, kl, sparse, moral,
    inj, bilip) and computes the weighted total from the network output.
    Has no learnable parameters.
    """

    ORDER = ("rec", "kl", "sparse", "moral", "inj", "bilip")

    def __init__(self, cfg: LossConfig, n_content: int):
        super().__init__()
        self.cfg = cfg
        en = cfg.schedule
        self.terms = nn.ModuleDict({
            "rec": RecLoss(cfg.lambda_rec, cfg.plateau_window, cfg.plateau_tol, enabled=en),
            "kl": KLLoss(cfg.lambda_kl, cfg.kl_init_frac, cfg.kl_ramp_epochs, enabled=en),
            "sparse": SparseLoss(cfg.lambda_sparse, n_content, cfg.sparse_ramp_epochs, enabled=en),
            "moral": MoralLoss(cfg.lambda_moral, n_content, cfg.moral_ramp_epochs, enabled=en),
            "inj": InjLoss(cfg.lambda_inj, cfg.inj_thresh, enabled=en),
            "bilip": BilipLoss(cfg.lambda_bilip, cfg.bilip_ramp_epochs, cfg.bilip_L,
                               cfg.bilip_subset_frac, enabled=en),
        })

    def weights(self) -> Dict[str, float]:
        """Current weight of every term."""
        return {n: self.terms[n].weight() for n in self.ORDER}

    def end_epoch(self, stats: Dict[str, float]) -> None:
        """Call end_epoch(stats) on every term (the model calls this FIRST
        in its end-of-epoch pass, then makes its start / off decisions)."""
        for n in self.ORDER:
            self.terms[n].end_epoch(stats)

    def schedule_state(self) -> Dict[str, dict]:
        return {n: self.terms[n].schedule_state() for n in self.ORDER}

    def load_schedule_state(self, s: Dict[str, dict]) -> None:
        for n in self.ORDER:
            self.terms[n].load_schedule_state(s[n])

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        x: torch.Tensor,
        decode_fn: Optional[Callable] = None,
        geod_sub: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
            out       dict returned by M1Net.forward.
            x         (B, C, H, W) the network input (reconstruction target).
            decode_fn HEAD.decode; required iff the bilip weight is > 0 now.
            geod_sub  (B, B) geodesic sub-table for this batch's dataset ids;
                      required iff the inj weight is > 0 now.
        Returns dict of scalar tensors: "kl", "rec", "sparse", "moral", "inj",
            "bilip" (each unweighted; 0 if its weight is 0 now) and "loss"
            (weighted total, the tensor to call backward() on).
        """
        T, w = self.terms, self.weights()
        zero = out["mu"].sum() * 0.0
        t: Dict[str, torch.Tensor] = {}
        t["kl"] = T["kl"](out) if w["kl"] else zero
        t["rec"] = T["rec"](out, x) if w["rec"] else zero
        t["sparse"] = T["sparse"](out) if w["sparse"] else zero
        t["moral"] = T["moral"](out) if w["moral"] else zero
        if w["inj"]:
            if geod_sub is None:
                raise ValueError("inj weight > 0 requires geod_sub.")
            t["inj"] = T["inj"](out, geod_sub)
        else:
            t["inj"] = zero
        if w["bilip"]:
            if decode_fn is None:
                raise ValueError("bilip weight > 0 requires decode_fn.")
            t["bilip"] = T["bilip"](out, decode_fn)
        else:
            t["bilip"] = zero
        t["loss"] = sum(w[n] * t[n] for n in self.ORDER)
        return t
