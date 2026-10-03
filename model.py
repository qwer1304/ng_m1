"""
model.py -- ties the network (m1net.M1Net) and the losses (losses.M1Losses).

The Trainer talks only to M1Model. M1Model:
  * runs the network on a batch,
  * calls M1Losses on the result,
  * keeps the moral-loss reference state (previous adjacency and k_min),
  * exposes what the Trainer needs: parameter groups, the periodic
    sink-freezing check, deterministic encoding.

It adds no trainable parameters of its own.

=======================================================================
MORAL-LOSS STATE
=======================================================================
The moral loss compares the current adjacency to the binary adjacency of the
previous iteration (Ng et al.). Here an "iteration" is the period between two
calls of end_of_check(). State kept:
    prev_adj : (nc, nc) binary adjacency snapshot, None before the first check
    k_min    : first leading-block size included in the sum. After a check
               with f content nodes frozen, k_min = nc - f + 1, so the sum
               covers the block sizes k_min..nc that reach into the already
               resolved part (paper: k = n+2-t..n with t-1 = f sinks removed).
               Before the first check k_min = nc + 1 (empty sum, loss 0).
This is my reading of the paper's indexing under Ng's fixed causal order
(index order); it is not in Ng's released code.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from losses import LossConfig, M1Losses
from m1net import M1Net, M1NetConfig


class M1Model(nn.Module):
    """
    Args
        net_cfg  : M1NetConfig (network configuration, includes HeadConfig)
        loss_cfg : LossConfig  (weights and BLAE hyper-parameters)
    """

    def __init__(self, net_cfg: M1NetConfig, loss_cfg: LossConfig):
        super().__init__()
        self.net = M1Net(net_cfg)
        self.losses = M1Losses(loss_cfg)
        self.prev_adj: Optional[torch.Tensor] = None
        self.k_min: int = net_cfg.n_content + 1

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
                       lambda_inj > 0.
            sample   : sample z (training) or use mu (evaluation).
        Returns the dict of loss terms from M1Losses ("kl","rec","sparse",
        "moral","inj","bilip","loss") plus the network output under "out"
        (a dict; not a tensor) for logging/metrics.
        """
        x, y, t, e = batch
        out = self.net(x, y, t, e, sample=sample)
        prev = None if self.prev_adj is None else self.prev_adj.to(x.device)
        terms = self.losses(
            out,
            x,
            decode_fn=self.net.head.decode,
            geod_sub=geod_sub,
            prev_adj=prev,
            k_min=self.k_min,
        )
        terms["out"] = out
        return terms

    # ------------------------------------------------------------------
    @torch.no_grad()
    def end_of_check(self) -> bool:
        """
        Periodic structure check; the Trainer calls it every `check_epoch`
        epochs. (1) runs sink freezing, (2) snapshots the moral-loss
        reference (prev_adj, k_min). Returns True iff all content nodes are
        frozen (training may stop).
        """
        done = self.net.find_sinks_and_fix()
        nc = self.net.n_content
        self.prev_adj = self.net.prior.get_adj().detach().clone()
        n_frozen = nc - len(self.net.prior._unfrozen())
        self.k_min = nc - n_frozen + 1
        return done

    # ------------------------------------------------------------------
    def adjacency_parameters(self) -> List[nn.Parameter]:
        return self.net.adjacency_parameters()

    def non_adjacency_parameters(self) -> List[nn.Parameter]:
        return self.net.non_adjacency_parameters()

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.encode_mu(x)
