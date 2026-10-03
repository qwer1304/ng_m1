"""
extract_features.py -- run the frozen IN-VAE encoder once over images and
cache the result as a feats.pt file that main.py reads.

This is Stage 2 of the pipeline (feature extraction, done once per choice of
backbone). No gradient, no training, no sampling: the cached vector is the
posterior MEAN of the pretrained REPA-E IN-VAE (f16d32) encoder.

=======================================================================
THE ENCODER (facts verified in github.com/End2End-Diffusion/REPA-E)
=======================================================================
* Loaded with invae.load_repae_vae: REPA-E's own AutoencoderKL class (from a
  clone of their repo, --repae_dir) + the pretrained state dict (--vae_ckpt),
  STRICT loading (missing/unexpected keys raise).
* No BatchNorm anywhere in the VAE (GroupNorm only, per-sample): nothing to
  recalibrate to TI.
* Input range is [-1, 1] (uint8 / 127.5 - 1). NO ImageNet mean/std
  normalisation. This script converts whatever --input_range you declare to
  [-1, 1]; do not pre-normalise with ImageNet statistics.
* Trained at 256x256 -> latent (32, 16, 16) = 8192 values.

=======================================================================
WHAT THE SCRIPT DOES NOT DO
=======================================================================
It does not read TerraIncognita, resize, crop, or decide which locations go
into which file. Your image pipeline does that and is plugged in through a
LOADER FACTORY (below). The script only encodes what the loader yields.

=======================================================================
LOADER CONTRACT
=======================================================================
--loader MODULE:FUNC   imports MODULE and calls FUNC(**loader_args); FUNC must
                       return an iterable (list, generator, DataLoader) that
                       yields, in a fixed order, tuples
                           (images, y, t, e)            or
                           (images, y, t, e, image_id)
    images    (B, 3, 256, 256) float tensor, CPU or GPU, RGB.
              Value range given by --input_range:
                  "01"  -> values in [0, 1]   (default)
                  "pm1" -> values in [-1, 1]
                  "255" -> values in [0, 255] (uint8 or float)
              The script converts to the [-1, 1] range the VAE expects.
              Size must be the size the checkpoint should see (256 here; it
              is NOT resized by this script; --image_size 0 disables the check).
    y, t, e   (B,) integer tensors: species, time-of-day, location indices,
              exactly the indices main.py will use (contiguous from 0).
    image_id  optional (B,) integer tensor or list of strings/ints; stored
              as-is (tensor -> "image_id" tensor, otherwise a python list).
--loader_arg K=V       (repeatable) passed to FUNC as string keyword
                       arguments, e.g. --loader_arg split=train.

Row order of the output = order of iteration. The GeodesicTable built later
by main.py uses that same order, so do not shuffle between runs.

=======================================================================
OUTPUT FILE (torch.save dict), read by main.py
=======================================================================
  x           (N, C, H, W)  latent means, dtype per --dtype
  y, t, e     (N,) int64
  brightness  (N,) float32  mean luminance in [0,1] of the INPUT image,
              0.299 R + 0.587 G + 0.114 B averaged over all pixels, computed
              on the image as delivered by the loader (before any AE
              normalization). Cached for the time-of-day proxy of the note;
              use the same measure on every location, L100 included.
  image_id    only if the loader provides it
  meta        dict: vae_ckpt, arch, input_range,
              image_size, latent_shape, n, dtype, dropped_locations

Run it once for the training locations and once for the test location (L100)
with different loader arguments and different --out files, using the SAME
--vae_ckpt / --input_range. Never mix L100 into the training
file (main.py trains on whatever is in the file it is given).

=======================================================================
USAGE
=======================================================================
python extract_features.py --repae_dir ~/REPA-E \\
    --vae_ckpt pretrained/invae/<the .pt you downloaded from REPA-E/invae> \\
    --loader my_pipeline:make_loader --loader_arg split=train \\
    --out feats_train.pt --device cuda --amp bf16

The script prints the latent shape (C, H, W); check it against the
(32, 16, 16) assumed in the note. main.py reads (C, H, W) from the file.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from invae import FrozenInVAE, load_repae_vae

LUMA = (0.299, 0.587, 0.114)


def brightness(images01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> (B,) mean luminance (BT.601 weights)."""
    w = torch.tensor(LUMA, device=images01.device, dtype=images01.dtype).view(1, 3, 1, 1)
    return (images01 * w).sum(1).mean(dim=(1, 2))


@torch.no_grad()
def extract_features(
    vae: nn.Module,
    batches: Iterable,
    device: str = "cuda",
    input_range: str = "01",
    amp: str = "off",
    dtype: torch.dtype = torch.float32,
    drop_locations: Sequence[int] = (),
    expected_hw: Optional[int] = 256,
) -> Dict:
    """
    Encode every batch from `batches` with the frozen encoder.

    Args
        vae           module from invae.load_repae_vae.
        batches       iterable of (images, y, t, e[, image_id]) as in the
                      module docstring.
        device        where the encoder runs.
        input_range   "01", "pm1" or "255" (range of the incoming images).
        amp           "off" | "fp16" | "bf16": autocast for the encoder only.
                      The stored latents are cast back to float32 first.
        dtype         storage dtype of x (float32 or float16).
        drop_locations location indices to skip entirely (safety net, e.g. the
                      index of L100 if it could leak into a training run).
        expected_hw   if not None, images must be expected_hw x expected_hw.
    Returns dict as described in the module docstring (without "meta").
    """
    if input_range not in ("01", "pm1", "255"):
        raise ValueError("input_range must be '01', 'pm1' or '255'.")
    enc = FrozenInVAE(vae.to(device)).to(device)
    ac = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]

    xs: List[torch.Tensor] = []
    ys, ts, es, bs = [], [], [], []
    ids: List = []
    have_ids = None
    drop = set(int(v) for v in drop_locations)

    for batch in batches:
        images, y, t, e = batch[:4]
        image_id = batch[4] if len(batch) > 4 else None
        if have_ids is None:
            have_ids = image_id is not None
        if (image_id is not None) != have_ids:
            raise ValueError("image_id must be present in all batches or none.")
        images = images.to(device, non_blocking=True).float()
        if expected_hw is not None and tuple(images.shape[-2:]) != (expected_hw, expected_hw):
            raise ValueError(
                f"images are {tuple(images.shape[-2:])}, expected "
                f"{expected_hw}x{expected_hw} (script does not resize)."
            )
        img01 = {"01": images, "pm1": (images + 1.0) / 2.0, "255": images / 255.0}[input_range]
        if float(img01.min()) < -1e-3 or float(img01.max()) > 1.0 + 1e-3:
            raise ValueError(
                f"image values [{float(img01.min()):.3f},{float(img01.max()):.3f}] "
                f"(after mapping to [0,1]) violate --input_range {input_range}."
            )
        if ac is None:
            lat = enc(img01 * 2.0 - 1.0)
        else:
            with torch.autocast(device_type=torch.device(device).type, dtype=ac):
                lat = enc(img01 * 2.0 - 1.0)
        lat = lat.float()
        y, t, e = (v.detach().cpu().long() for v in (y, t, e))
        keep = torch.tensor([int(v) not in drop for v in e], dtype=torch.bool)
        if not keep.all():
            lat, img01 = lat[keep.to(lat.device)], img01[keep.to(img01.device)]
            y, t, e = y[keep], t[keep], e[keep]
            if image_id is not None:
                image_id = (
                    image_id[keep] if torch.is_tensor(image_id)
                    else [v for v, kk in zip(image_id, keep.tolist()) if kk]
                )
        xs.append(lat.to(dtype).cpu())
        ys.append(y), ts.append(t), es.append(e)
        bs.append(brightness(img01).cpu())
        if image_id is not None:
            ids.append(image_id)

    if not xs:
        raise ValueError("loader yielded no data.")
    out = {
        "x": torch.cat(xs),
        "y": torch.cat(ys),
        "t": torch.cat(ts),
        "e": torch.cat(es),
        "brightness": torch.cat(bs).float(),
    }
    if have_ids:
        out["image_id"] = (
            torch.cat(ids) if torch.is_tensor(ids[0]) else [v for b in ids for v in b]
        )
    return out


def _parse_kv(items: List[str]) -> Dict[str, str]:
    d = {}
    for s in items:
        if "=" not in s:
            raise SystemExit(f"--loader_arg expects K=V, got {s!r}")
        k, v = s.split("=", 1)
        d[k] = v
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repae_dir", required=True, help="clone of github.com/End2End-Diffusion/REPA-E")
    p.add_argument("--vae_ckpt", required=True, help="pretrained IN-VAE state dict (.pt)")
    p.add_argument("--arch", default="f16d32", choices=["f16d32", "f8d4"])
    p.add_argument("--loader", required=True, help="MODULE:FUNC returning the batch iterable")
    p.add_argument("--loader_arg", action="append", default=[], help="K=V (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input_range", default="01", choices=["01", "pm1", "255"])
    p.add_argument("--amp", default="off", choices=["off", "fp16", "bf16"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--image_size", type=int, default=256, help="expected input size; 0 disables the check")
    p.add_argument("--drop_e", type=int, nargs="*", default=[], help="location indices to skip")
    a = p.parse_args()

    vae = load_repae_vae(a.repae_dir, a.vae_ckpt, a.arch, a.device)

    sys.path.insert(0, os.getcwd())  # so a loader module in the current dir is found
    mod_name, fn_name = a.loader.split(":")
    batches = getattr(importlib.import_module(mod_name), fn_name)(**_parse_kv(a.loader_arg))

    res = extract_features(
        vae, batches, device=a.device, input_range=a.input_range,
        amp=a.amp,
        dtype=torch.float16 if a.dtype == "float16" else torch.float32,
        drop_locations=a.drop_e, expected_hw=a.image_size or None,
    )
    x = res["x"]
    res["meta"] = {
        "vae_ckpt": a.vae_ckpt, "arch": a.arch,
        "input_range": a.input_range,
        "image_size": a.image_size, "latent_shape": tuple(x.shape[1:]), "n": x.shape[0],
        "dtype": a.dtype, "dropped_locations": list(a.drop_e),
    }
    torch.save(res, a.out)
    print(f"saved {a.out}: N={x.shape[0]} latent (C,H,W)={tuple(x.shape[1:])} "
          f"mean={x.float().mean():.4f} std={x.float().std():.4f}")


if __name__ == "__main__":
    main()
