"""
invae.py -- the frozen, pretrained REPA-E IN-VAE (f16d32) used as feature
extractor. Replaces the FrozenInVAE that used to live in m1net.py (which
wrongly assumed the diffusers AutoencoderKL interface).

=======================================================================
FACTS ABOUT THE RELEASED IN-VAE (checked in github.com/End2End-Diffusion/REPA-E)
=======================================================================
* Class: REPA-E's own `AutoencoderKL` in models/autoencoder.py (a port of the
  CompVis latent-diffusion VAE), NOT diffusers.AutoencoderKL. Weights are a
  plain torch state dict (.pt). Config f16d32: embed_dim=32,
  ch_mult=[1,1,2,2,4].
* encode(x) returns a DiagonalGaussianDistribution directly (no
  `.latent_dist`); its `.mean` is the deterministic latent.
* Normalisation layers: GroupNorm(32) only -> per-sample, no running
  statistics, NO BatchNorm, nothing to recalibrate to TI images.
  (REPA-E's BatchNorm sits in the diffusion model, outside the VAE, and is
  not used here.)
* Input range: [-1, 1], i.e. uint8 x / 127.5 - 1 (utils.preprocess_imgs_vae).
  NO ImageNet mean/std normalisation.
* Resolution: trained at 256x256; latent for 256x256 input is (32, 16, 16)
  = 8192 values. Self-attention is built in at latent-resolution 16 and the
  config assumes resolution=256; feed 256x256 images.
* The class imports `dictdot` (pip install dictdot).

=======================================================================
CONTENTS
=======================================================================
load_repae_vae(repae_dir, ckpt_path, arch, device)  -> nn.Module, eval, frozen
FrozenInVAE(vae)                                     -> images in [-1,1] ->
                                                        latent means (B,32,16,16)
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional

import torch
import torch.nn as nn


def load_repae_vae(
    repae_dir: str,
    ckpt_path: str,
    arch: str = "f16d32",
    device: str = "cpu",
) -> nn.Module:
    """
    Build REPA-E's AutoencoderKL and load pretrained weights.

    Args
        repae_dir : path to a clone of github.com/End2End-Diffusion/REPA-E
                    (the file <repae_dir>/models/autoencoder.py is imported
                    directly under a private module name, so it cannot clash
                    with other packages called `models`).
        ckpt_path : path to the downloaded VAE state dict (.pt) of the
                    pretrained IN-VAE (HF: REPA-E/invae).
        arch      : "f16d32" (IN-VAE, VA-VAE) or "f8d4" (SD-VAE).
        device    : where to place the module.
    Returns the module in eval mode with requires_grad=False.

    Checkpoint handling: the file may be the raw state dict or a dict that
    wraps it under "vae", "state_dict" or "model". Loading is STRICT: any
    missing or unexpected key raises (REPA-E's own training script loads with
    strict=False; here a silent partial load would give garbage features).
    """
    src = os.path.join(repae_dir, "models", "autoencoder.py")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} not found; is repae_dir a REPA-E clone?")
    spec = importlib.util.spec_from_file_location("repae_autoencoder", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if arch not in mod.vae_models:
        raise ValueError(f"arch must be one of {list(mod.vae_models)}")
    vae = mod.vae_models[arch]()

    sd = torch.load(ckpt_path, map_location="cpu")
    for key in ("vae", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    vae.load_state_dict(sd, strict=True)

    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


class FrozenInVAE(nn.Module):
    """
    Frozen image -> latent-mean extractor around a loaded REPA-E VAE.

    Args
        vae          : module from load_repae_vae (anything whose encode(x)
                       returns an object with `.mean`, or with
                       `.latent_dist.mean` for a diffusers VAE).
    forward(images)
        images : (B, 3, 256, 256) float32 in [-1, 1]  (RGB).
        returns (B, 32, 16, 16): the posterior MEAN (deterministic), no grad.
    The module is kept in eval mode permanently.
    """

    def __init__(
        self,
        vae: nn.Module,
    ):
        super().__init__()
        self.vae = vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # always stay in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.vae.encode(images)
        z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.mean
        return z
