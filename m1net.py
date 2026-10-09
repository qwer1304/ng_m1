"""
m1net.py -- the complete trainable network of the M1 pipeline (everything a
Trainer needs to train, and nothing the Trainer must own).

Requires head.py (class HEAD, HeadConfig) in the same directory.

=======================================================================
WHAT IS IN THIS FILE, AND WHERE EACH PIECE COMES FROM
=======================================================================
  (The frozen IN-VAE feature extractor lives in invae.py; its output is the
   cached tensor x_tilde that M1Net consumes.)
  HEAD             (head.py) trainable diffusers AutoencoderKL + two linear
                   layers. Gives q(z|x), z, x_hat.
  ADJMatrix        Ng et al. reference code (networks.py), ported.
  NodeWiseMLP      New. Content mechanism f_i (shared or per-node weights).
  GateConfig       (config.py) numbers of the sink-check gate.
  SinkGate         New. Pure-python state machine of the sink-check gate
                   (when to freeze). No torch; the tensor work is in M1Prior.
  M1Prior          Ng et al. prior (ANM), extended per the note: domain
                   conditioning split by block (content on c=(y,t), style
                   on r=(t,e)). Includes Ng's sink-freezing heuristic and
                   OWNS the gate that decides when to apply it, and the
                   ORDER BLEND of the adjacency ("full" -> "tril", below).
  M1Net            Ties HEAD + M1Prior together. This is the module a
                   Trainer optimizes.
  (All loss computations, including the BLAE ones, live in losses.py.)

WHAT IS *NOT* IN THIS FILE
    * any loss (see losses.py); the network only exposes what they
      need (out["x_hat"], out["log_q"], out["log_p"], out["adj"],
      out["current_adj"], out["mu"], and head.decode)
    * optimizers, schedules, the GeoD table, the training loop. The model
      (model.py) calls M1Net.gate_end_epoch(epoch) when the Trainer asks it
      to run the end-of-epoch scheduling, and M1Net.set_order_blend(...) when
      its rules say so.

=======================================================================
PIPELINE
=======================================================================
  (once)  image --invae.FrozenInVAE--> x_tilde (B,C,H,W), cached to disk
  (train) x_tilde, y, t, e --M1Net.forward--> dict (see forward docstring)

Latent layout (fixed everywhere): the D = n_content + n_style M1 latents are
ordered   [ content latents (first n_content) | style latents (last n_style) ].

=======================================================================
THE PRIOR (what M1Net computes on top of HEAD)
=======================================================================
Observed integer labels per image: species y, time-of-day t, location e.
  content cell  c = y * n_times + t            (index into content_emb)
  style cell    r = t * n_locations + e        (index into style_emb)

Content block, ANM (Gaussian additive noise; scale depends on c only):
  z_c,i ~ N( f_i^{(c)}( A[i,:] * z_c ),  exp(content_logvar_i^{(c)}) )
  A[i,j]=1 means j is a parent of i. What A is depends on the ORDER BLEND a
  (a float in [0,1], owned by M1Prior, set by the model):
    a = 1  "tril" : strictly lower-triangular 0/1 matrix learned by
             ADJMatrix (causal order = index order, not searched, as in Ng).
    a = 0  "full" : the constant matrix of ones with a zero diagonal. Every
             latent is predicted from ALL the others. Nothing is learned in
             A and no slot order is imprinted on the latents. The diagonal
             is always removed: a latent that sees itself makes the prior
             collapse. In this mode the KL term is a pseudo-likelihood (a
             product of full conditionals), not a normalized joint; it is a
             legitimate training signal but not a true KL.
    0 < a < 1     : A = (1-a)*full + a*tril, a smooth transition (no jump in
             the prior). Reported as order_mode "blend".
  The blend starts at 1 and the model sets 0 at construction when scheduling
  is enabled, then ramps it to 1 (set_order_blend). While a < 1 the
  sink-freezing machinery is inert (nothing is frozen, the gate does
  nothing); at a = 0 the raw adjacency parameter gets no gradient.
Style block (isolated nodes, no parents, independent of species):
  z_s,j ~ N( mu_j^{(r)}, exp(style_logvar_j^{(r)}) )

Consequences of the note that are implemented here:
  * f depends on the content cell c (Ng's code had no domain input in the
    mean path; that is item (i) of the note's summary).
  * Style prior never sees y (property P3).
  * Sharing of f across nodes is a switch (cfg.mechanism_sharing), item (ii).
  * Adjacency vs. non-adjacency parameters are separated cleanly, item (iv).
  * None of Ng's dead code (gumbel, y_rep, concatenated x/d) is carried.

KL term: because f depends on z itself, an analytic Gaussian KL does not
exist; like Ng, the Trainer should use the single-sample estimate
    (out["log_q"] - out["log_p"]).mean()
Both are returned per image, summed over the D latents.

=======================================================================
THE SINK-CHECK GATE (M1Prior.gate_end_epoch)
=======================================================================
The gate decides WHEN the sink-freezing heuristic is applied. It is called
once per epoch, at the end of the epoch, by the model (which the Trainer
asks to do the end-of-epoch scheduling). Two modes (GateConfig.mode):

  "stable"  (scheduling on). After each epoch, compare the binarized
            unresolved block with the previous epoch's. stable_run counts the
            consecutive epochs with no change (reset on any flip, and when
            the set of unfrozen nodes changes). When stable_run >= W(m),
            freeze ALL zero columns at once and reset the counters. If
            epochs_since_reset reaches cap(m) first, nothing is frozen:
            the gate reports the flip history, the raw entries nearest 0.5
            and the nodes frozen so far, and asks the run to stop.
            m = number of unfrozen nodes; W(m) and cap(m) come from
            GateConfig.table.
  "clock"   (scheduling off). Freeze all zero columns every `check_epoch`
            epochs, as in Ng's loop. No stability rule, no cap.

The gate returns a GateResult: events ("first_sink_check", "all_frozen",
"cap_hit:sink_check"), a stop reason if any, the global ids of the nodes
frozen in this call, and an info dict (cap case). No margin (hysteresis) in
the binarization and no intervention at the cap yet: deferred. The gate does
nothing while the order blend is below 1.

Gate state is plain python (SinkGate.state_dict) and goes into the
checkpoint through M1Prior.schedule_state(), together with the order blend;
sure_mask / sure_adj are buffers and are saved with the model.

=======================================================================
DEVIATIONS FROM Ng's SINK-FREEZING CODE (deliberate; read this)
=======================================================================
Ng's find_sinknodes_and_fix indexes the global sure_mask with indices taken
from the *compacted* remaining sub-matrix. That is only correct in the first
round; after some nodes are frozen the indices shift. Here compact indices
are mapped back to global node ids. Also, Ng's final "if sum(current_adj)==0:
should_stop=False" quirk is dropped: find_sinks_and_fix returns True exactly
when every content node is frozen. The heuristic itself (a node whose column
is all zero in the hard adjacency is called a sink of the remaining graph)
is unchanged, and is still a mask heuristic, not the paper's derivative test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GateConfig
from head import HEAD, HeadConfig

# events emitted by the gate (the model routes them; same strings everywhere)
EV_FIRST_SINK_CHECK = "first_sink_check"
EV_ALL_FROZEN = "all_frozen"
EV_CAP_SINK = "cap_hit:sink_check"

# the orders of the adjacency (see module docstring); "blend" is 0 < a < 1
ORDER_FULL = "full"
ORDER_TRIL = "tril"
ORDER_BLEND = "blend"


# ======================================================================
# ADJMatrix  (ported from Ng networks.py)
# ======================================================================
class ADJMatrix(nn.Module):
    """
    Learnable adjacency over n nodes.

    Parameter: adj_matrix (n, n), initialised to ones, so at the start every
    lower-triangular edge is ON (value 1 > 0.5); sparsity then prunes.

    forward(soft=False) returns a strictly lower-triangular (n, n) matrix:
      soft=True  : the raw parameter values.
      soft=False : hard 0/1 (parameter > 0.5) with a straight-through
                   gradient (forward = hard, backward = identity).
    Entry [i, j] (j < i) = 1 means node j is a PARENT of node i.
    """

    def __init__(self, n: int):
        super().__init__()
        self.adj_matrix = nn.Parameter(torch.ones(n, n))

    def forward(self, soft: bool = False) -> torch.Tensor:
        a = self.adj_matrix
        if not soft:
            a = (a > 0.5).float() - a.detach() + a
        return torch.tril(a, diagonal=-1)


# ======================================================================
# NodeWiseMLP  (content mechanism f_i)
# ======================================================================
class NodeWiseMLP(nn.Module):
    """
    Applies a one-hidden-layer MLP to each of N nodes, producing one scalar
    per node.  input (B, N, in_dim) -> output (B, N).

    share=True  : ONE set of weights used for all N nodes (this is what Ng's
                  mean_net does; node identity then enters only through the
                  adjacency mask inside the input).
    share=False : a SEPARATE set of weights per node (a genuine f_i each).
    """

    def __init__(self, n_nodes: int, in_dim: int, hidden: int, share: bool):
        super().__init__()
        g = 1 if share else n_nodes
        self.n_nodes = n_nodes
        self.w1 = nn.Parameter(torch.empty(g, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(g, 1, hidden))
        self.w2 = nn.Parameter(torch.empty(g, hidden, 1))
        self.b2 = nn.Parameter(torch.zeros(g, 1, 1))
        for w, fan_in in ((self.w1, in_dim), (self.w2, hidden)):
            nn.init.uniform_(w, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        n = self.n_nodes
        w1, w2 = self.w1.expand(n, -1, -1), self.w2.expand(n, -1, -1)
        b1, b2 = self.b1.expand(n, -1, -1), self.b2.expand(n, -1, -1)
        h = F.leaky_relu(
            torch.einsum("bni,nih->bnh", inp, w1) + b1.squeeze(1)[None], 0.2
        )
        return (torch.einsum("bnh,nho->bno", h, w2) + b2.squeeze(1)[None]).squeeze(-1)


# ======================================================================
# Sink-check gate: pure-python state machine (GateConfig lives in config.py)
# ======================================================================
class SinkGate:
    """
    Pure-python state machine of the gate (no torch). M1Prior feeds it the
    binarized unresolved block and applies what it decides.

    step(epoch, bits, keep_ids) -> "wait" | "freeze" | "cap"
        bits      flat tuple/list of 0/1 (the binarized unresolved block,
                  row-major) of the nodes in keep_ids.
        keep_ids  global ids of the unfrozen nodes, in order.
    record_freeze(epoch, nodes) -> True iff this was the first successful
        check; logs the freeze and resets the counters.

    State (state_dict; JSON-friendly): stable_run, epochs_since_reset, prev,
    prev_keep, flip_history (since the last reset; None = no reference
    epoch), freeze_log ([epoch, node] pairs), first_check_done.
    """

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.stable_run = 0
        self.epochs_since_reset = 0
        self.prev: Optional[List[int]] = None
        self.prev_keep: Optional[List[int]] = None
        self.flip_history: List[Optional[int]] = []
        self.freeze_log: List[List[int]] = []
        self.first_check_done = False

    def _reset(self) -> None:
        self.stable_run, self.epochs_since_reset = 0, 0
        self.prev, self.prev_keep, self.flip_history = None, None, []

    def step(self, epoch: int, bits, keep_ids: List[int]) -> str:
        if self.cfg.mode == "clock":
            ce = self.cfg.check_epoch
            return "freeze" if (ce > 0 and epoch % ce == 0) else "wait"
        bits, keep_ids = [int(b) for b in bits], [int(k) for k in keep_ids]
        if self.prev is None or self.prev_keep != keep_ids:
            flips = None
            self.stable_run = 0
        else:
            flips = sum(1 for a, b in zip(bits, self.prev) if a != b)
            self.stable_run = self.stable_run + 1 if flips == 0 else 0
        self.flip_history.append(flips)
        self.epochs_since_reset += 1
        self.prev, self.prev_keep = bits, keep_ids
        w, cap = self.cfg.params(len(keep_ids))
        if self.stable_run >= w:
            return "freeze"
        if self.epochs_since_reset >= cap:
            return "cap"
        return "wait"

    def record_freeze(self, epoch: int, nodes: List[int]) -> bool:
        for n in nodes:
            self.freeze_log.append([int(epoch), int(n)])
        first = (not self.first_check_done) and len(nodes) > 0
        if first:
            self.first_check_done = True
        self._reset()
        return first

    def state_dict(self) -> dict:
        return {
            "stable_run": self.stable_run,
            "epochs_since_reset": self.epochs_since_reset,
            "prev": None if self.prev is None else list(self.prev),
            "prev_keep": None if self.prev_keep is None else list(self.prev_keep),
            "flip_history": list(self.flip_history),
            "freeze_log": [list(x) for x in self.freeze_log],
            "first_check_done": self.first_check_done,
        }

    def load_state_dict(self, s: dict) -> None:
        self.stable_run = s["stable_run"]
        self.epochs_since_reset = s["epochs_since_reset"]
        self.prev = None if s["prev"] is None else list(s["prev"])
        self.prev_keep = None if s["prev_keep"] is None else list(s["prev_keep"])
        self.flip_history = list(s["flip_history"])
        self.freeze_log = [list(x) for x in s["freeze_log"]]
        self.first_check_done = s["first_check_done"]


@dataclass
class GateResult:
    """What the gate returns at the end of an epoch."""

    events: List[str] = field(default_factory=list)
    stop: Optional[str] = None  # reason, if the run must stop (cap hit)
    froze: List[int] = field(default_factory=list)  # global ids frozen now
    info: dict = field(default_factory=dict)  # cap case: diagnostics


# ======================================================================
# Config
# ======================================================================
@dataclass
class M1NetConfig:
    """
    head : HeadConfig
        Configuration of HEAD. head.n_latents MUST equal n_content+n_style.
    n_content, n_style : int
        Number of content / style latents. Bounded above by the
        environment-count requirements of the note (sec:reqs).
    n_species, n_times, n_locations : int
        Cardinalities of the integer labels y, t, e. Labels passed to
        forward() must lie in [0, n_*).
    emb_dim : int
        Width of the learned content-cell and style-cell embeddings.
    hidden : int
        Hidden width of every small MLP in the prior.
    mechanism_sharing : "shared" | "per_node"
        "shared"   : one f for all content nodes (Ng's choice).
        "per_node" : a distinct f_i per content node.
    prior_logvar_min, prior_logvar_max : float | None
        Clamp on prior log-variances (content and style), for stability.
    gate : GateConfig
        Sink-check gate numbers (defaults overridable).
    """

    head: HeadConfig
    n_content: int
    n_style: int
    n_species: int
    n_times: int
    n_locations: int
    emb_dim: int = 16
    hidden: int = 32
    mechanism_sharing: str = "shared"
    prior_logvar_min: Optional[float] = -10.0
    prior_logvar_max: Optional[float] = 10.0
    gate: GateConfig = field(default_factory=GateConfig)

    def validate(self) -> None:
        if self.n_content < 1 or self.n_style < 0:
            raise ValueError("need n_content >= 1 and n_style >= 0.")
        if self.head.n_latents != self.n_content + self.n_style:
            raise ValueError(
                f"head.n_latents={self.head.n_latents} != "
                f"n_content+n_style={self.n_content + self.n_style}."
            )
        if self.mechanism_sharing not in ("shared", "per_node"):
            raise ValueError("mechanism_sharing must be 'shared' or 'per_node'.")
        for name in ("n_species", "n_times", "n_locations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1.")
        self.gate.validate()


# ======================================================================
# M1Prior
# ======================================================================
class M1Prior(nn.Module):
    """
    The M1 prior p(z | y, t, e) over the D latents, plus the content graph.

    Holds: content_emb, style_emb, content mechanism f, content log-variance
    net, style (mean, logvar) net, ADJMatrix, the sink-freezing buffers
    sure_mask / sure_adj (both (n_content, n_content), not trainable), the
    sink-check gate (self.gate, a SinkGate) and the order blend
    (self.order_blend in [0,1], plain python state; the model sets it, see
    the module docstring). order_mode is derived from it.
    """

    def __init__(self, cfg: M1NetConfig):
        super().__init__()
        self.cfg = cfg
        nc, ns, E, Hd = cfg.n_content, cfg.n_style, cfg.emb_dim, cfg.hidden
        self.n_content, self.n_style = nc, ns

        self.content_emb = nn.Embedding(cfg.n_species * cfg.n_times, E)
        self.style_emb = nn.Embedding(cfg.n_times * cfg.n_locations, E)

        # f_i^{(c)}: input per node = [A[i,:]*z_c (nc values), cell emb (E)]
        self.mechanism = NodeWiseMLP(
            nc, nc + E, Hd, share=(cfg.mechanism_sharing == "shared")
        )
        # content noise scale: depends on c only (ANM)
        self.content_logvar_net = nn.Sequential(
            nn.Linear(E, Hd), nn.LeakyReLU(0.2), nn.Linear(Hd, nc)
        )
        # style: (mu, logvar) per style latent, depends on r=(t,e) only
        self.style_net = (
            nn.Sequential(nn.Linear(E, Hd), nn.LeakyReLU(0.2), nn.Linear(Hd, 2 * ns))
            if ns > 0
            else None
        )

        self.adj_mat = ADJMatrix(nc)
        self.register_buffer("sure_mask", torch.zeros(nc, nc))
        self.register_buffer("sure_adj", torch.zeros(nc, nc))
        self.gate = SinkGate(cfg.gate)  # plain python state, not a buffer
        self.order_blend: float = 1.0  # plain python state, not a buffer

    # ---------------- order blend ----------------
    def set_order_blend(self, a: float) -> None:
        """Set the blend between the full off-diagonal mask (a=0) and the
        learned lower-triangular one (a=1). The model decides when and how
        fast; this only executes."""
        a = float(a)
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"order blend must be in [0, 1], got {a}")
        self.order_blend = a

    def set_order_mode(self, mode: str) -> None:
        """Convenience: "full" -> blend 0, "tril" -> blend 1."""
        if mode not in (ORDER_FULL, ORDER_TRIL):
            raise ValueError(f"order mode must be {ORDER_FULL!r} or {ORDER_TRIL!r}, got {mode!r}")
        self.set_order_blend(0.0 if mode == ORDER_FULL else 1.0)

    @property
    def order_mode(self) -> str:
        """"full" (a=0), "tril" (a=1) or "blend"."""
        if self.order_blend <= 0.0:
            return ORDER_FULL
        if self.order_blend >= 1.0:
            return ORDER_TRIL
        return ORDER_BLEND

    # ---------------- graph / sink freezing ----------------
    def _unfrozen(self) -> torch.Tensor:
        """Global ids of content nodes not yet frozen (diag of sure_mask==0)."""
        return torch.nonzero(self.sure_mask.diagonal() == 0).flatten()

    def get_adj(self, soft: bool = False, current: bool = False) -> torch.Tensor:
        """
        current=False : full (nc, nc) adjacency. Blend a=1 ("tril"): frozen
                        entries come from sure_adj, the rest from the
                        learnable matrix. a=0 ("full"): the constant
                        ones-minus-identity matrix (no gradient; nothing is
                        ever frozen then). 0<a<1: (1-a)*full + a*learnable
                        (nothing is frozen then either).
        current=True  : only the still-unresolved square sub-block (rows and
                        columns of unfrozen nodes), always taken from the
                        learnable matrix. Shape (m, m), m may be 0. This is
                        what the sparsity penalty is computed on (inactive
                        while the mode is "full").
        """
        adj = self.adj_mat(soft)
        if current:
            keep = self._unfrozen().to(adj.device)
            return adj[keep][:, keep]
        a = self.order_blend
        if a < 1.0:
            n = adj.shape[0]
            full = torch.ones(n, n, device=adj.device) - torch.eye(n, device=adj.device)
            return full if a <= 0.0 else (1.0 - a) * full + a * adj
        return self.sure_mask * self.sure_adj + (1 - self.sure_mask) * adj

    @torch.no_grad()
    def _freeze_zero_columns(self) -> List[int]:
        """
        Ng's sink-freezing heuristic (see module docstring for the fixed
        indexing). Among unfrozen nodes, any node whose column in the hard
        adjacency restricted to unfrozen nodes sums to 0 (it has no
        remaining children) is declared a sink of the remaining graph and
        frozen: its row (parents) and column (children) are copied into
        sure_adj and masked so they stop being learned. All such nodes are
        frozen at once. Returns their global ids.
        """
        keep = self._unfrozen()
        if len(keep) == 0:
            return []
        hard = self.adj_mat(soft=False).detach()
        cur = hard[keep][:, keep]
        sinks: List[int] = [
            int(keep[k]) for k in range(len(keep)) if cur[:, k].sum() == 0
        ]
        for g in sinks:
            self.sure_mask[:, g] = 1
            self.sure_mask[g, :] = 1
            self.sure_adj[:, g] = hard[:, g]
            self.sure_adj[g, :] = hard[g, :]
        return sinks

    @torch.no_grad()
    def find_sinks_and_fix(self) -> bool:
        """
        Apply the heuristic once, unconditionally (kept for direct use).
        Returns True iff every content node is now frozen. Does nothing
        (returns False) while the order blend is below 1.
        """
        if self.order_mode != ORDER_TRIL:
            return False
        self._freeze_zero_columns()
        return len(self._unfrozen()) == 0

    @torch.no_grad()
    def gate_end_epoch(self, epoch: int) -> GateResult:
        """
        The sink-check gate; call once at the END of every epoch (the model
        does it when the Trainer asks for the end-of-epoch scheduling).
        See the module docstring for the two modes. Epochs are 1-based.
        Does nothing while the order blend is below 1.
        """
        res = GateResult()
        if self.order_mode != ORDER_TRIL:
            return res
        keep = self._unfrozen()
        if len(keep) == 0:
            return res
        keep_ids = [int(k) for k in keep]
        hard = self.adj_mat(soft=False).detach()
        bits = (hard[keep][:, keep] > 0.5).flatten().int().tolist()
        action = self.gate.step(epoch, bits, keep_ids)
        if action == "freeze":
            sinks = self._freeze_zero_columns()
            res.froze = sinks
            if self.gate.record_freeze(epoch, sinks):
                res.events.append(EV_FIRST_SINK_CHECK)
            if len(self._unfrozen()) == 0:
                res.events.append(EV_ALL_FROZEN)
        elif action == "cap":
            m = len(keep_ids)
            raw = self.adj_mat(soft=True).detach()[keep][:, keep]
            tri = torch.tril_indices(m, m, offset=-1)
            vals = raw[tri[0], tri[1]]
            order = (vals - 0.5).abs().argsort()[:5].tolist()
            near = [(keep_ids[int(tri[0][o])], keep_ids[int(tri[1][o])], float(vals[o]))
                    for o in order]
            w, cap = self.cfg.gate.params(m)
            res.events.append(EV_CAP_SINK)
            res.stop = (f"sink check: no stable adjacency after "
                        f"{self.gate.epochs_since_reset} epochs with {m} unfrozen "
                        f"nodes (cap {cap}, W {w}); nothing frozen.")
            res.info = {
                "flip_history": list(self.gate.flip_history),
                "nearest_to_half": near,  # (row, col, raw value), global ids
                "frozen_so_far": [list(x) for x in self.gate.freeze_log],
                "unfrozen": keep_ids,
            }
        return res

    def schedule_state(self) -> dict:
        """Non-learnable state (plain python): the gate and the order blend.
        sure_mask / sure_adj are buffers and are saved with the model's
        state_dict."""
        return {"gate": self.gate.state_dict(), "order_blend": self.order_blend}

    def load_schedule_state(self, s: dict) -> None:
        if "gate" in s:
            self.gate.load_state_dict(s["gate"])
            if "order_blend" in s:
                self.set_order_blend(s["order_blend"])
            else:  # state written with the two-valued order mode
                self.set_order_mode(s.get("order_mode", ORDER_TRIL))
        else:  # checkpoint written before the order mode existed: gate dict only
            self.gate.load_state_dict(s)
            self.set_order_blend(1.0)

    # ---------------- prior parameters ----------------
    def _clamp(self, lv: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c.prior_logvar_min is None and c.prior_logvar_max is None:
            return lv
        return lv.clamp(min=c.prior_logvar_min, max=c.prior_logvar_max)

    def forward(self, z, y, t, e):
        """
        Args
            z : (B, D) latents (sampled z, as in Ng; content mean uses z_c).
            y, t, e : (B,) long tensors, labels in [0, n_*).
        Returns dict
            prior_mean   (B, D)  content: f_i(parents), style: mu_j^{(r)}
            prior_logvar (B, D)  log-variances (clamped)
            adj          (nc, nc) adjacency used now: hard lower-triangular
                         (straight-through grad) at blend 1, constant
                         ones-minus-identity at blend 0, the mix between
            current_adj  (m, m)   unresolved sub-block of the learnable
                         matrix, for the sparsity loss
        """
        cfg, nc = self.cfg, self.n_content
        B = z.shape[0]
        z_c = z[:, :nc]
        c_idx = y * cfg.n_times + t
        ce = self.content_emb(c_idx)  # (B, E)

        adj = self.get_adj()  # (nc, nc)
        masked = adj.unsqueeze(0) * z_c.unsqueeze(1)  # (B, nc, nc): row i = A[i,:]*z
        inp = torch.cat([masked, ce.unsqueeze(1).expand(B, nc, -1)], dim=-1)
        mean_c = self.mechanism(inp)  # (B, nc)
        logvar_c = self.content_logvar_net(ce)  # (B, nc)

        if self.n_style > 0:
            r_idx = t * cfg.n_locations + e
            mu_s, logvar_s = self.style_net(self.style_emb(r_idx)).chunk(2, dim=1)
            prior_mean = torch.cat([mean_c, mu_s], dim=1)
            prior_logvar = torch.cat([logvar_c, logvar_s], dim=1)
        else:
            prior_mean, prior_logvar = mean_c, logvar_c

        return {
            "prior_mean": prior_mean,
            "prior_logvar": self._clamp(prior_logvar),
            "adj": adj,
            "current_adj": self.get_adj(current=True),
        }


# ======================================================================
# M1Net
# ======================================================================
class M1Net(nn.Module):
    """
    HEAD + M1Prior. The module the Trainer trains.

    Attributes: head (HEAD), prior (M1Prior), n_content, n_style.
    """

    def __init__(self, cfg: M1NetConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.n_content, self.n_style = cfg.n_content, cfg.n_style
        self.head = HEAD(cfg.head)
        self.prior = M1Prior(cfg)

    def forward(self, x, y, t, e, sample: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args
            x : (B, C, H, W) cached IN-VAE latent (x_tilde).
            y, t, e : (B,) long labels (species, time of day, location).
            sample : True -> z sampled (training); False -> z = mu.
        Returns dict, all per image unless noted:
            mu, logvar   (B, D)   posterior q(z|x)   [mu: BLAE losses]
            z            (B, D)   latent used by prior and decoder
            z_c, z_s     views of z: content (B, nc), style (B, ns)
            x_hat        (B, C, H, W) reconstruction of x
            prior_mean, prior_logvar (B, D)
            log_q        (B,)  log q(z|x), summed over D
            log_p        (B,)  log p(z|y,t,e), summed over D
            adj          (nc, nc)  adjacency used now (see M1Prior.forward)
            current_adj  (m, m)    unresolved block (sparsity penalty)
        """
        h = self.head(x, sample=sample)
        p = self.prior(h["z"], y, t, e)
        std_q = torch.exp(0.5 * h["logvar"])
        std_p = torch.exp(0.5 * p["prior_logvar"])
        log_q = torch.distributions.Normal(h["mu"], std_q).log_prob(h["z"]).sum(-1)
        log_p = torch.distributions.Normal(p["prior_mean"], std_p).log_prob(h["z"]).sum(-1)
        nc = self.n_content
        return {
            **h,
            **p,
            "z_c": h["z"][:, :nc],
            "z_s": h["z"][:, nc:],
            "log_q": log_q,
            "log_p": log_p,
        }

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latents (B, D) for evaluation / MCC / prediction."""
        return self.head.encode(x)[0]

    def find_sinks_and_fix(self) -> bool:
        """Delegates to M1Prior.find_sinks_and_fix (unconditional, one shot)."""
        return self.prior.find_sinks_and_fix()

    def gate_end_epoch(self, epoch: int) -> GateResult:
        """Delegates to M1Prior.gate_end_epoch (the model calls this)."""
        return self.prior.gate_end_epoch(epoch)

    def set_order_blend(self, a: float) -> None:
        """Delegates to M1Prior.set_order_blend (the model calls this)."""
        self.prior.set_order_blend(a)

    def set_order_mode(self, mode: str) -> None:
        """Delegates to M1Prior.set_order_mode."""
        self.prior.set_order_mode(mode)

    @property
    def order_blend(self) -> float:
        return self.prior.order_blend

    @property
    def order_mode(self) -> str:
        return self.prior.order_mode

    def schedule_state(self) -> dict:
        return self.prior.schedule_state()

    def load_schedule_state(self, s: dict) -> None:
        self.prior.load_schedule_state(s)

    # --- optimizer helpers: fixes Ng's bug where the adjacency was in both
    #     optimizer parameter lists and therefore stepped twice ---
    def adjacency_parameters(self) -> List[nn.Parameter]:
        """Exactly one tensor: the learnable adjacency."""
        return [self.prior.adj_mat.adj_matrix]

    def non_adjacency_parameters(self) -> List[nn.Parameter]:
        """Every other trainable parameter (disjoint from the above)."""
        adj = self.prior.adj_mat.adj_matrix
        return [p for p in self.parameters() if p is not adj]


# ======================================================================
# Usage example / smoke test (TerraInc numbers appear ONLY here)
# ======================================================================
if __name__ == "__main__":
    head_cfg = HeadConfig(
        in_channels=32, in_height=16, in_width=16, n_latents=10,
        ae_down_block_types=("DownEncoderBlock2D",) * 3,
        ae_up_block_types=("UpDecoderBlock2D",) * 3,
        ae_block_out_channels=(32, 64, 128),
    )
    cfg = M1NetConfig(head=head_cfg, n_content=8, n_style=2,
                      n_species=10, n_times=2, n_locations=3)
    net = M1Net(cfg)
    B = 16
    x = torch.randn(B, 32, 16, 16)
    y, t, e = (torch.randint(0, n, (B,)) for n in (10, 2, 3))
    for a in (0.0, 0.5, 1.0):
        net.set_order_blend(a)
        out = net(x, y, t, e)
        print(f"--- order blend {a} ({net.order_mode}): adj row sums {out['adj'].sum(1).tolist()}")
        for k, v in out.items():
            print(f"{k:13s}{tuple(v.shape)}")
    out["x_hat"].sum().backward()
    print("gate, epoch 1:", net.gate_end_epoch(1))
