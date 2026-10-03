"""
losses.py -- ALL loss computations of the M1 pipeline, in one place.

Nothing else in the code base computes a loss. The network (m1net.py) only
produces tensors; the model (model.py) calls M1Losses on them; the Trainer
only calls backward() on the total.

=======================================================================
THE TOTAL LOSS (per minibatch, all terms are batch means / scalars)
=======================================================================
  L =  lambda_kl     * KL
     + lambda_rec    * REC
     + lambda_sparse * SPARSE
     + lambda_moral  * MORAL
     + lambda_inj    * INJ
     + lambda_bilip  * BILIP

  KL      Ng et al. (trainer.py): single-sample Monte-Carlo estimate
          mean_b [ log q(z_b|x_b) - log p(z_b|y_b,t_b,e_b) ], both summed over
          the D latents. (No analytic Gaussian KL exists because the content
          prior mean depends on z itself.)
  REC     Ng et al.: F.mse_loss(x_hat, x). Corresponds to a Gaussian
          likelihood with fixed unit variance. x is the CACHED IN-VAE latent,
          not the image.
  SPARSE  Ng et al.: sum of all entries of (I+A)^T (I+A), A = the still
          unresolved block of the content adjacency (out["current_adj"]).
          Entries are non-negative so this equals the L1 norm of the paper.
  MORAL   Ng et al. (paper, Sec. 4.3): sum over k = k_min..n_content of
              || (I + A[:k,:k])^T (I + A[:k,:k])
                 - (I + Aprev[:k,:k])^T (I + Aprev[:k,:k]) ||_F^2
          where A = current adjacency (out["adj"], straight-through),
          Aprev = the binary adjacency snapshotted at the previous sink-
          freezing check, and [:k,:k] the leading k x k block (index order =
          causal order, as in Ng's fixed lower-triangular design). It keeps
          the moral graph of the larger blocks consistent with the previous
          iteration. NOT in Ng's released code (they set lambda_moral = 0 in
          their experiments). k_min / Aprev are supplied by the caller
          (model.py maintains them). If either is missing, MORAL = 0.
  INJ     BLAE (Zhan et al. 2026, utils/regularizations.py: InjectiveLoss).
          Needs a (b,b) geodesic sub-table for the minibatch. Computed on mu.
  BILIP   BLAE (trainers/trainers.py, reg_grad='bi-lipschitz'). Decoder
          Jacobian based; needs the decoder callable. Computed on mu. This is
          by far the most expensive term (D decoder passes per sample on a
          fraction of the batch); leave lambda_bilip = 0 to skip it entirely.

A term whose weight is 0 is NOT computed (and reported as 0.0), so the costly
BLAE terms cost nothing when disabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacfwd, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel


# ----------------------------------------------------------------------
# Individual loss functions (pure)
# ----------------------------------------------------------------------
def kl_mc(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Single-sample MC KL: mean over batch of (log_q - log_p). Inputs (B,)."""
    return (log_q - log_p).mean()


def sparsity_loss(current_adj: torch.Tensor) -> torch.Tensor:
    """sum((I+A)^T (I+A)) over the unresolved block A of shape (m, m).
    Returns 0 for m == 0."""
    m = current_adj.shape[0]
    if m == 0:
        return current_adj.sum() * 0.0
    ipa = torch.eye(m, device=current_adj.device) + current_adj
    return (ipa.T @ ipa).sum()


def moral_loss(adj: torch.Tensor, prev_adj: torch.Tensor, k_min: int) -> torch.Tensor:
    """
    Moral-graph consistency (see module docstring).

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
               computed (BLAE uses 0.3).
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
# Config + aggregator
# ----------------------------------------------------------------------
@dataclass
class LossConfig:
    """
    Weights (0 disables a term and skips its computation):
      lambda_kl, lambda_rec, lambda_sparse, lambda_moral, lambda_inj,
      lambda_bilip
    BLAE hyper-parameters:
      inj_thresh          lower ratio bound of the injective loss (0.3)
      bilip_L             upper bound on diag of the pulled-back metric (2.0)
      bilip_subset_frac   fraction of batch for the Jacobian (0.3)
    Reference values in Ng's toy code: lambda_kl=1, lambda_rec=10,
    lambda_sparse=0.01 (their chain/fork cases). The BLAE and moral weights
    have no published anchor for this setting.
    """

    lambda_kl: float = 1.0
    lambda_rec: float = 10.0
    lambda_sparse: float = 0.01
    lambda_moral: float = 0.0
    lambda_inj: float = 0.0
    lambda_bilip: float = 0.0
    inj_thresh: float = 0.3
    bilip_L: float = 2.0
    bilip_subset_frac: float = 0.3


class M1Losses(nn.Module):
    """
    Computes every loss term and the weighted total from the network output.
    Has no parameters.
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        x: torch.Tensor,
        decode_fn: Optional[Callable] = None,
        geod_sub: Optional[torch.Tensor] = None,
        prev_adj: Optional[torch.Tensor] = None,
        k_min: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
            out       dict returned by M1Net.forward.
            x         (B, C, H, W) the network input (reconstruction target).
            decode_fn HEAD.decode; required iff lambda_bilip > 0.
            geod_sub  (B, B) geodesic sub-table for this batch's dataset ids;
                      required iff lambda_inj > 0.
            prev_adj, k_min  moral-loss reference (see moral_loss); optional.
        Returns dict of scalar tensors: "kl", "rec", "sparse", "moral", "inj",
            "bilip" (each unweighted; 0 if disabled) and "loss" (weighted
            total, the tensor to call backward() on).
        """
        c = self.cfg
        zero = out["mu"].sum() * 0.0
        t: Dict[str, torch.Tensor] = {}
        t["kl"] = kl_mc(out["log_q"], out["log_p"]) if c.lambda_kl else zero
        t["rec"] = F.mse_loss(out["x_hat"], x) if c.lambda_rec else zero
        t["sparse"] = sparsity_loss(out["current_adj"]) if c.lambda_sparse else zero
        if c.lambda_moral and prev_adj is not None and k_min is not None:
            t["moral"] = moral_loss(out["adj"], prev_adj, k_min)
        else:
            t["moral"] = zero
        if c.lambda_inj:
            if geod_sub is None:
                raise ValueError("lambda_inj > 0 requires geod_sub.")
            t["inj"] = injective_loss(geod_sub, out["mu"], c.inj_thresh)
        else:
            t["inj"] = zero
        if c.lambda_bilip:
            if decode_fn is None:
                raise ValueError("lambda_bilip > 0 requires decode_fn.")
            t["bilip"] = bilipschitz_decoder_loss(
                decode_fn, out["mu"], c.bilip_L, c.bilip_subset_frac
            )
        else:
            t["bilip"] = zero
        t["loss"] = (
            c.lambda_kl * t["kl"]
            + c.lambda_rec * t["rec"]
            + c.lambda_sparse * t["sparse"]
            + c.lambda_moral * t["moral"]
            + c.lambda_inj * t["inj"]
            + c.lambda_bilip * t["bilip"]
        )
        return t
