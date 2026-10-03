"""
head.py -- the trainable HEAD of the M1 pipeline.

=======================================================================
WHERE THIS FITS
=======================================================================
Pipeline (see sec:pseudoinj, "The resulting architecture"):

    image --(frozen IN-VAE encoder, run once, cached)--> x_tilde
    x_tilde --(HEAD.encode)--> q(z|x) = N(mu, exp(logvar))     <- this file
    z ~ q(z|x)  (reparameterization)                            <- this file
    z --(HEAD.decode)--> x_tilde_hat                            <- this file
    M1 prior (mean_net, log_var_net, adj_mat, sink-freezing)    <- NOT here
    BLAE losses (on mu only)                                    <- NOT here

HEAD only produces the recognition distribution q(z|x), a sample z, and a
reconstruction of its own input. It contains NO loss, NO prior, NO KL, NO
BLAE terms, NO optimizer. The caller (Trainer) owns all of that.

Notation used in this file
--------------------------
    B   batch size
    C, H, W   channels / height / width of the HEAD input tensor x_tilde
              (for the chosen IN-VAE candidate: 32, 16, 16)
    D   n_latents = number of M1 latents (content + style), e.g. 10
    (c, h, w)  shape of the AutoencoderKL internal bottleneck, DERIVED at
               construction time by a dry run (see HEAD.bottleneck_shape)
    P   c*h*w = number of scalars in the flattened bottleneck

=======================================================================
DATA FLOW (shapes)
=======================================================================
ENCODE
    x_tilde            (B, C, H, W)
      -> AutoencoderKL.encode -> internal Gaussian over (B, c, h, w)
      -> take its mean (default) or a sample   (B, c, h, w)
      -> flatten                                (B, P)
      -> pre_linear   Linear(P, 2*D)            (B, 2*D)
      -> split                                  mu (B, D), logvar (B, D)
REPARAMETERIZE
    z = mu + exp(0.5*logvar) * eps, eps ~ N(0, I)        (B, D)
DECODE
    z                  (B, D)
      -> post_linear  Linear(D, P)              (B, P)
      -> reshape                                (B, c, h, w)
      -> AutoencoderKL.decode                   (B, C, H, W)   = x_tilde_hat

=======================================================================
DESIGN DECISIONS BAKED IN (all deliberate, all documented)
=======================================================================
1. AutoencoderKL is used UNMODIFIED (no edits to quant_conv /
   post_quant_conv). The only hand-written trainable parts are the two
   linear layers pre_linear and post_linear. pre_linear is the layer that
   actually reduces dimension (P -> 2*D).

2. Where the stochasticity lives. AutoencoderKL internally has its own
   Gaussian latent (its "latent_dist"). M1 needs exactly ONE posterior,
   q(z|x), the one over the D M1 latents. If we also sampled the internal
   Gaussian we would inject a second, un-modelled noise source between x
   and z. Therefore by default we take the internal distribution's MEAN
   (deterministic) and ignore its variance; all noise comes from the
   reparameterization of the M1 posterior. The internal log-variance
   branch of AutoencoderKL's encoder is therefore computed but unused
   (its parameters get no gradient in this default mode). Set
   HeadConfig.internal_latent = "sample" to override.

3. AutoencoderKL's own KL loss is never computed here. Nothing in this
   file returns it. The KL that matters is between q(z|x) (returned here)
   and M1's prior, computed by the caller.

4. Output order of the 2*D scalars from pre_linear is FIXED:
   first D are mu, last D are logvar. This is what split() relies on.

5. logvar can optionally be clamped (HeadConfig.logvar_min/max) to avoid
   exp() overflow / underflow early in training. Clamping is applied to
   the returned logvar, so the caller's KL sees the clamped value.
   Set both to None to disable.

6. No weight initialisation is applied beyond PyTorch / diffusers
   defaults. If you want a custom init, apply it from the caller after
   construction (model.apply(fn)).

7. Everything is determined by HeadConfig. Nothing is hard-coded to the
   TerraInc numbers (32x16x16, D=10); those appear only in the usage
   example at the bottom of the file.

=======================================================================
IMPORTANT NOTE ON THE SPATIAL SIZE OF THE BOTTLENECK
=======================================================================
In diffusers' AutoencoderKL, N entries in `block_out_channels` /
`down_block_types` give only (N - 1) spatial 2x downsamplings, because
the LAST encoder stage does not downsample. Examples for a 16x16 input:
    N = 2 stages -> 16 -> 8          bottleneck spatial 8x8
    N = 3 stages -> 16 -> 8 -> 4     bottleneck spatial 4x4
This class does not assume any of this: it measures the bottleneck shape
with a dry run at construction and exposes it as `bottleneck_shape`.
Always read that attribute (or the printed summary) instead of computing
it by hand.

=======================================================================
DEPENDENCIES
=======================================================================
    torch, diffusers (AutoencoderKL, tested with 0.40.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class HeadConfig:
    """
    Every knob of HEAD. HEAD reads nothing from anywhere else.

    ---- Input geometry (describes the tensor the caller will feed) ----
    in_channels : int
        C. Channels of the HEAD input x_tilde. Must equal the channel count
        of the frozen IN-VAE latent (32 for the chosen candidate).
        Also used as AutoencoderKL's in_channels AND out_channels, since
        the head reconstructs its own input.
    in_height, in_width : int
        H, W. Spatial size of x_tilde (16, 16 for the chosen candidate).
        Must be divisible by 2**(number of downsamplings), where the
        number of downsamplings = len(ae_block_out_channels) - 1.
        Checked at construction.

    ---- M1 interface ----
    n_latents : int
        D. Number of M1 latents (content + style). The head outputs a
        posterior over exactly D scalars per image.

    ---- AutoencoderKL sizing (passed through to diffusers unchanged) ----
    ae_down_block_types : tuple[str, ...]
        One entry per encoder stage, e.g. ("DownEncoderBlock2D",)*3.
        Length MUST equal len(ae_block_out_channels).
    ae_up_block_types : tuple[str, ...]
        One entry per decoder stage, e.g. ("UpDecoderBlock2D",)*3.
        Length MUST equal len(ae_block_out_channels).
    ae_block_out_channels : tuple[int, ...]
        Channel width of each stage. Each must be divisible by
        ae_norm_num_groups.
    ae_layers_per_block : int
        Residual blocks per stage.
    ae_latent_channels : int
        Channels of AutoencoderKL's own bottleneck (the "c" in (c, h, w)).
        NOT the M1 latent count; the M1 reduction happens in pre_linear.
    ae_norm_num_groups : int
        GroupNorm groups inside AutoencoderKL. Every entry of
        ae_block_out_channels must be divisible by it.
    ae_act_fn : str
        Activation name understood by diffusers (default "silu").

    ---- Behaviour ----
    internal_latent : "mean" | "sample"
        Which vector of AutoencoderKL's internal Gaussian is fed to
        pre_linear. "mean" (default, deterministic) is recommended; see
        design decision 2 in the module docstring.
    logvar_min, logvar_max : float | None
        If not None, the M1 logvar is clamped to [logvar_min, logvar_max].
        Either bound may be None independently.
    pre_linear_bias, post_linear_bias : bool
        Whether the two linear layers have a bias term.
    """

    # input geometry
    in_channels: int
    in_height: int
    in_width: int
    # M1 interface
    n_latents: int
    # AutoencoderKL sizing
    ae_down_block_types: Tuple[str, ...]
    ae_up_block_types: Tuple[str, ...]
    ae_block_out_channels: Tuple[int, ...]
    ae_layers_per_block: int = 1
    ae_latent_channels: int = 4
    ae_norm_num_groups: int = 32
    ae_act_fn: str = "silu"
    # behaviour
    internal_latent: str = "mean"
    logvar_min: Optional[float] = -10.0
    logvar_max: Optional[float] = 10.0
    pre_linear_bias: bool = True
    post_linear_bias: bool = True

    def validate(self) -> None:
        """Raise ValueError on any inconsistent setting. Called by HEAD."""
        n_stages = len(self.ae_block_out_channels)
        if len(self.ae_down_block_types) != n_stages:
            raise ValueError(
                f"ae_down_block_types has {len(self.ae_down_block_types)} "
                f"entries but ae_block_out_channels has {n_stages}."
            )
        if len(self.ae_up_block_types) != n_stages:
            raise ValueError(
                f"ae_up_block_types has {len(self.ae_up_block_types)} "
                f"entries but ae_block_out_channels has {n_stages}."
            )
        for ch in self.ae_block_out_channels:
            if ch % self.ae_norm_num_groups != 0:
                raise ValueError(
                    f"block_out_channels entry {ch} is not divisible by "
                    f"ae_norm_num_groups={self.ae_norm_num_groups}."
                )
        n_down = n_stages - 1  # last encoder stage does not downsample
        factor = 2 ** n_down
        if self.in_height % factor or self.in_width % factor:
            raise ValueError(
                f"Input {self.in_height}x{self.in_width} is not divisible by "
                f"{factor} (= 2**{n_down} downsamplings)."
            )
        if self.internal_latent not in ("mean", "sample"):
            raise ValueError("internal_latent must be 'mean' or 'sample'.")
        if self.n_latents < 1:
            raise ValueError("n_latents must be >= 1.")
        if (
            self.logvar_min is not None
            and self.logvar_max is not None
            and self.logvar_min >= self.logvar_max
        ):
            raise ValueError("logvar_min must be < logvar_max.")


# ----------------------------------------------------------------------
# The module
# ----------------------------------------------------------------------
class HEAD(nn.Module):
    """
    Trainable VAE head: x_tilde -> q(z|x) -> z -> x_tilde_hat.

    Trainable parameters (all of them, nothing is frozen inside HEAD):
        self.ae          diffusers.AutoencoderKL, fresh random init
        self.pre_linear  nn.Linear(P, 2*D)
        self.post_linear nn.Linear(D, P)

    Public attributes (read-only by convention)
        cfg              the HeadConfig used
        bottleneck_shape (c, h, w) of AutoencoderKL's internal latent,
                         measured by a dry run at construction
        bottleneck_numel P = c*h*w
        n_latents        D

    Public methods
        encode(x)                 -> (mu, logvar)
        reparameterize(mu, logvar)-> z
        decode(z)                 -> x_hat
        forward(x, sample=True)   -> dict (see forward docstring)
        summary()                 -> str, human-readable description
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.n_latents = cfg.n_latents

        # ---- 1. stock AutoencoderKL, configured but not modified ----
        # in_channels == out_channels == cfg.in_channels because the head
        # reconstructs its own input. sample_size is metadata only for
        # diffusers; it does not change the computation.
        self.ae = AutoencoderKL(
            in_channels=cfg.in_channels,
            out_channels=cfg.in_channels,
            down_block_types=tuple(cfg.ae_down_block_types),
            up_block_types=tuple(cfg.ae_up_block_types),
            block_out_channels=tuple(cfg.ae_block_out_channels),
            layers_per_block=cfg.ae_layers_per_block,
            act_fn=cfg.ae_act_fn,
            latent_channels=cfg.ae_latent_channels,
            norm_num_groups=cfg.ae_norm_num_groups,
            sample_size=max(cfg.in_height, cfg.in_width),
        )

        # ---- 2. measure the bottleneck shape (c, h, w) by a dry run ----
        # Done in eval + no_grad on a single zero image so that it has no
        # side effects (no gradients, no dropout, no RNG use except that
        # "mean" mode does not sample).
        was_training = self.ae.training
        self.ae.eval()
        with torch.no_grad():
            probe = torch.zeros(1, cfg.in_channels, cfg.in_height, cfg.in_width)
            probe_mean = self.ae.encode(probe).latent_dist.mean
        self.ae.train(was_training)
        self.bottleneck_shape: Tuple[int, int, int] = tuple(probe_mean.shape[1:])
        self.bottleneck_numel: int = int(probe_mean[0].numel())

        # ---- 3. the two hand-written trainable linear layers ----
        # pre_linear: THE dimensionality-reducing step, P -> 2*D.
        self.pre_linear = nn.Linear(
            self.bottleneck_numel, 2 * cfg.n_latents, bias=cfg.pre_linear_bias
        )
        # post_linear: D -> P, expands z back to the flat bottleneck.
        self.post_linear = nn.Linear(
            cfg.n_latents, self.bottleneck_numel, bias=cfg.post_linear_bias
        )

        # ---- 4. end-to-end shape check (fail at construction, not at
        #         step 1 of training) ----
        with torch.no_grad():
            was_training = self.training
            self.eval()
            out = self.forward(probe, sample=False)["x_hat"]
            self.train(was_training)
        if tuple(out.shape) != tuple(probe.shape):
            raise ValueError(
                f"Decoder output shape {tuple(out.shape)} != input shape "
                f"{tuple(probe.shape)}. Check that ae_up_block_types has the "
                f"same number of stages as ae_down_block_types."
            )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recognition network q(z | x).

        Args
            x : (B, C, H, W) float tensor. The cached, frozen IN-VAE latent.
                Must match cfg.in_channels/in_height/in_width.
        Returns
            mu     : (B, D)  posterior mean of the M1 latents.
            logvar : (B, D)  log of the posterior VARIANCE (not std),
                             clamped to [logvar_min, logvar_max] if set.

        Notes
            * mu is the quantity the BLAE losses must be computed on.
            * Which internal AutoencoderKL vector is used is controlled
              by cfg.internal_latent (see module docstring, decision 2).
        """
        dist = self.ae.encode(x).latent_dist  # DiagonalGaussianDistribution
        feat = dist.mean if self.cfg.internal_latent == "mean" else dist.sample()
        feat = feat.flatten(start_dim=1)  # (B, P)
        params = self.pre_linear(feat)  # (B, 2*D)
        mu, logvar = params.chunk(2, dim=1)  # first D = mu, last D = logvar
        if self.cfg.logvar_min is not None or self.cfg.logvar_max is not None:
            logvar = logvar.clamp(min=self.cfg.logvar_min, max=self.cfg.logvar_max)
        return mu, logvar

    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = mu + sigma * eps, eps ~ N(0, I), sigma = exp(0.5 * logvar).

        Args
            mu, logvar : (B, D) each.
        Returns
            z : (B, D). Differentiable w.r.t. mu and logvar.
        Uses torch's global RNG (seed it from the caller for
        reproducibility).
        """
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    # ------------------------------------------------------------------
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decoder: z -> reconstruction of the head input.

        Args
            z : (B, D) M1 latents (a sample, or mu for a deterministic
                reconstruction).
        Returns
            x_hat : (B, C, H, W), same shape as the encoder input.
        """
        flat = self.post_linear(z)  # (B, P)
        c, h, w = self.bottleneck_shape
        return self.ae.decode(flat.view(-1, c, h, w)).sample

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """
        Full pass: encode, (optionally) sample, decode.

        Args
            x      : (B, C, H, W) cached IN-VAE latent.
            sample : True  -> z is a reparameterized sample (training).
                     False -> z = mu, no noise (evaluation / caching mu).
        Returns a dict with EXACTLY these keys:
            "mu"     (B, D)        posterior mean            -> BLAE losses,
                                                                M1 prior, KL
            "logvar" (B, D)        log posterior variance    -> KL
            "z"      (B, D)        the latent that was decoded and that the
                                   M1 prior/adjacency should be applied to
            "x_hat"  (B, C, H, W)  reconstruction of x       -> recon loss
                                                                against x
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return {"mu": mu, "logvar": logvar, "z": z, "x_hat": self.decode(z)}

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable description of the built module."""
        c, h, w = self.bottleneck_shape
        n_params = sum(p.numel() for p in self.parameters())
        n_ae = sum(p.numel() for p in self.ae.parameters())
        cfg = self.cfg
        return (
            f"HEAD\n"
            f"  input            : ({cfg.in_channels}, {cfg.in_height}, {cfg.in_width})\n"
            f"  AutoencoderKL    : stages={len(cfg.ae_block_out_channels)}, "
            f"channels={tuple(cfg.ae_block_out_channels)}, "
            f"layers/block={cfg.ae_layers_per_block}, params={n_ae:,}\n"
            f"  AE bottleneck    : ({c}, {h}, {w}) = {self.bottleneck_numel} scalars "
            f"(internal_latent='{cfg.internal_latent}')\n"
            f"  pre_linear       : {self.bottleneck_numel} -> {2 * cfg.n_latents} "
            f"(= mu[{cfg.n_latents}] + logvar[{cfg.n_latents}])\n"
            f"  post_linear      : {cfg.n_latents} -> {self.bottleneck_numel}\n"
            f"  logvar clamp     : [{cfg.logvar_min}, {cfg.logvar_max}]\n"
            f"  total params     : {n_params:,}\n"
        )


# ----------------------------------------------------------------------
# Usage example (TerraInc numbers appear ONLY here)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cfg = HeadConfig(
        in_channels=32,
        in_height=16,
        in_width=16,
        n_latents=10,
        ae_down_block_types=("DownEncoderBlock2D",) * 3,
        ae_up_block_types=("UpDecoderBlock2D",) * 3,
        ae_block_out_channels=(32, 64, 128),
        ae_layers_per_block=1,
        ae_latent_channels=4,
        ae_norm_num_groups=32,
    )
    head = HEAD(cfg)
    print(head.summary())

    x = torch.randn(8, 32, 16, 16)  # stand-in for cached IN-VAE latents
    out = head(x, sample=True)
    for k, v in out.items():
        print(f"{k:7s} {tuple(v.shape)}")

    # Caller-side sketch of the loss (NOT part of HEAD):
    #   rec = F.mse_loss(out["x_hat"], x)
    #   kl  = KL(N(out["mu"], exp(out["logvar"])) || M1 prior)   # Trainer
    #   blae = inj_loss(out["mu"]) + bilip_loss(out["mu"], GeoD) # on mu only
