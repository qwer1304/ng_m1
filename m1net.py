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
  M1Prior          Ng et al. prior (ANM), extended per the note: domain
                   conditioning split by block (content on c=(y,t), style
                   on r=(t,e)). Includes Ng's sink-freezing heuristic.
  M1Net            Ties HEAD + M1Prior together. This is the module a
                   Trainer optimizes.
  (All loss computations, including the BLAE ones, live in losses.py.)

WHAT IS *NOT* IN THIS FILE
    * any loss (see losses.py: reconstruction, KL, sparsity, moral,
      BLAE injective, BLAE bi-Lipschitz); the network only exposes what they
      need (out["x_hat"], out["log_q"], out["log_p"], out["adj"],
      out["current_adj"], out["mu"], and head.decode)
    * optimizers, schedules, the GeoD table, the training loop, calling
      M1Net.find_sinks_and_fix() every `check_epoch` epochs (Trainer's job)

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
  A = lower-triangular 0/1 adjacency over content latents; A[i,j]=1 means
  j is a parent of i. Causal order = index order (not searched), as in Ng.
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
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from head import HEAD, HeadConfig


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


# ======================================================================
# M1Prior
# ======================================================================
class M1Prior(nn.Module):
    """
    The M1 prior p(z | y, t, e) over the D latents, plus the content graph.

    Holds: content_emb, style_emb, content mechanism f, content log-variance
    net, style (mean, logvar) net, ADJMatrix, and the sink-freezing buffers
    sure_mask / sure_adj (both (n_content, n_content), not trainable).
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

    # ---------------- graph / sink freezing ----------------
    def _unfrozen(self) -> torch.Tensor:
        """Global ids of content nodes not yet frozen (diag of sure_mask==0)."""
        return torch.nonzero(self.sure_mask.diagonal() == 0).flatten()

    def get_adj(self, soft: bool = False, current: bool = False) -> torch.Tensor:
        """
        current=False : full (nc, nc) adjacency; frozen entries come from
                        sure_adj, the rest from the learnable matrix.
        current=True  : only the still-unresolved square sub-block (rows and
                        columns of unfrozen nodes). Shape (m, m), m may be 0.
                        This is what the sparsity penalty is computed on.
        """
        adj = self.adj_mat(soft)
        if current:
            keep = self._unfrozen().to(adj.device)
            return adj[keep][:, keep]
        return self.sure_mask * self.sure_adj + (1 - self.sure_mask) * adj

    @torch.no_grad()
    def find_sinks_and_fix(self) -> bool:
        """
        Ng's sink-freezing heuristic (see module docstring for the fixed
        indexing). Among unfrozen nodes, any node whose column in the hard
        adjacency restricted to unfrozen nodes sums to 0 (it has no
        remaining children) is declared a sink of the remaining graph and
        frozen: its row (parents) and column (children) are copied into
        sure_adj and masked so they stop being learned.

        Call from the Trainer every `check_epoch` epochs.
        Returns True iff every content node is now frozen (stop signal).
        """
        keep = self._unfrozen()
        if len(keep) == 0:
            return True
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
        return len(self._unfrozen()) == 0

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
            adj          (nc, nc) full hard adjacency (straight-through grad)
            current_adj  (m, m)   unresolved sub-block, for the sparsity loss
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
            adj          (nc, nc)  hard adjacency (straight-through)
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
        """Delegates to M1Prior.find_sinks_and_fix (Trainer calls this)."""
        return self.prior.find_sinks_and_fix()

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
    out = net(x, y, t, e)
    for k, v in out.items():
        print(f"{k:13s}{tuple(v.shape)}")
    out["x_hat"].sum().backward()
    print("frozen all?", net.find_sinks_and_fix())
