"""
ti_dataset.py -- builds (and loads) the TerraIncognita dataset of IN-VAE latents
with all labels and metadata, as a TensorDataset, optionally saved to disk.

It integrates two earlier tools:
  * terra_build_annotations.py  -> metadata (location, time of day, synthetic
    flag, source location, bounding boxes). The metadata logic is that file's,
    kept as-is (section "Annotation metadata"); changes are marked "# CHANGED".
  * extract_features.py         -> IN-VAE latents (called, not re-implemented).
No annotation JSON is parsed anywhere else.

=======================================================================
WHAT YOU GIVE IT
=======================================================================
root          directory of ALREADY RESIZED TI images (the tool does not resize).
              Each image path must END with  <location>/<species>/<file>.jpg
              (any number of directories may sit above <location>). The species
              folder name becomes the label (classes = sorted species names, as
              torchvision.ImageFolder would); it must equal a category name in
              the annotation JSONs (e.g. bird, bobcat, ..., empty).
annotations   the TI annotation JSON files.
target_size   side of the (square) input images, 256 for the IN-VAE. Used to
              (a) check every image has exactly this size, (b) transform the
              bounding boxes with the old utility's rule: proportional resize
              of the annotation space (1494 x 2048) then center crop. If your
              external resize does something else, the boxes are wrong.
vae_ckpt, repae_dir   the pretrained REPA-E IN-VAE (see invae.py).

=======================================================================
POLICY OPTIONS
=======================================================================
tod_bins      "2" (night 0-5 / day 6-23), "4" (night, dawn 6-8, day 9-17,
              dusk 18-23) or a custom spec "night:0-5+22-23;day:6-21".
              The hour is also stored, so bins can be changed later without
              re-extracting (load_ti_dataset(tod_bins=...)).
holdout_locations  raw ids of the test location(s) (e.g. [100]). Decides the index
              order of e (see output below). Default none: pure alphabetical.
synthetic     "include" (default) | "exclude" | "only". Synthetic images are
              the ones transported from another location. For them location =
              TARGET location, time of day and boxes come from the SOURCE
              image (utility's rule).
multi_animal  "keep" (default) | "drop". Images with more than one box of the
              labelled species. With "keep", ALL boxes are stored and n_boxes
              says how many; downstream code decides what to do.
Samples with no match in the annotations are dropped with a warning
(not an error). Policies are applied BEFORE extraction, so dropped images
are never encoded.

=======================================================================
THE OUTPUT (one row per kept image, N rows)
=======================================================================
  x           (N, 32, 16, 16) float   IN-VAE latent means
  y           (N,) long   species index (index into meta["class_names"])
  t           (N,) long   time-of-day bin (meta["tod_names"])
  e           (N,) long   location index 0.. (meta["location_ids"][e] = raw id).
              Training locations first, alphabetical by name; the holdout_
              locations (e.g. 100) last, also alphabetical -- the usual
              ImageFolder result when the test location lives in another
              parent directory (so L100 gets the last label). Without
              holdout_locations it is one plain alphabetical sort.
  loc         (N,) long   raw location id (target location if synthetic)
  hour        (N,) long   capture hour 0..23
  synthetic   (N,) bool
  src_loc     (N,) long   source location for synthetic, -1 otherwise
  n_boxes     (N,) long   boxes of the labelled species (0 for 'empty')
  boxes       (N, M, 4)   [x1,y1,x2,y2] in target_size space, padded with -1
  brightness  (N,) float  mean luminance of the input image in [0,1]
  uuid, src_uuid, path, loc_dir   python lists (src_uuid None for originals;
              path relative to root; loc_dir = the location folder name)
  meta        dict of the settings used.
As a TensorDataset the tensors come in the order FIELDS:
    x, y, t, e, ids, loc, hour, synthetic, src_loc, n_boxes, boxes
(ids = row position in the returned dataset, for the geodesic table).

=======================================================================
USE
=======================================================================
  ds, info = build_ti_dataset(root, annotations, vae_ckpt, repae_dir,
                              tod_bins="2", save_path="ti.pt")
  ds, info = load_ti_dataset("ti.pt", locations=[<training location ids>])   # later
CLI: python ti_dataset.py --help
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import TensorDataset

FIELDS = ("x", "y", "t", "e", "ids", "loc", "hour", "synthetic", "src_loc", "n_boxes", "boxes")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".tif", ".tiff", ".webp")

# ======================================================================
# Time-of-day binning
# ======================================================================
TOD_PRESETS = {
    "4": [("night", range(0, 6)), ("dawn", range(6, 9)), ("day", range(9, 18)), ("dusk", range(18, 24))],
    "2": [("night", range(0, 6)), ("day", range(6, 24))],
}


def parse_tod_spec(spec: str) -> List[Tuple[str, List[int]]]:
    """Parse "2", "4" or a custom spec "night:0-5+22-23;day:6-21"
    (bins separated by ';', ranges by '+', 'a-b' inclusive or a single hour)."""
    if spec in TOD_PRESETS:
        return [(n, list(h)) for n, h in TOD_PRESETS[spec]]
    bins = []
    for part in spec.split(";"):
        name, rngs = part.split(":")
        hours: List[int] = []
        for r in rngs.split("+"):
            a, _, b = r.partition("-")
            hours += list(range(int(a), int(b or a) + 1))
        bins.append((name.strip(), hours))
    return bins


def make_hour_lut(bins) -> List[int]:
    """hour (0..23) -> bin index. Raises if an hour is missing or doubled."""
    lut = [-1] * 24
    for i, (name, hours) in enumerate(bins):
        for h in hours:
            if not 0 <= h <= 23:
                raise ValueError(f"hour {h} out of range in bin {name}")
            if lut[h] != -1:
                raise ValueError(f"Hour {h} assigned to more than one bin")
            lut[h] = i
    missing = [h for h in range(24) if lut[h] == -1]
    if missing:
        raise ValueError(f"Hours not covered by any bin: {missing}")
    return lut


def rebin(hour: torch.Tensor, spec: str) -> Tuple[torch.Tensor, List[str]]:
    """Recompute time-of-day indices from hours. Returns (t, bin names)."""
    bins = parse_tod_spec(spec)
    lut = torch.tensor(make_hour_lut(bins), dtype=torch.long)
    return lut[hour.long()], [n for n, _ in bins]


# ======================================================================
# Annotation metadata  (from terra_build_annotations.py; logic unchanged,
# changes marked "# CHANGED")
# ======================================================================
ANN_H = 1494
ANN_W = 2048


def _resolve(p, root):
    p = Path(p)
    return p if p.is_absolute() else root / p


def _make_bbox_transform(target_size):
    """[x,y,w,h] in annotation space -> [x1,y1,x2,y2] in target_size space
    (proportional resize, center crop). None if target_size is None."""
    if target_size is None:
        return None
    scale = target_size / ANN_H
    new_w = int(ANN_W * scale)
    new_h = int(ANN_H * scale)
    crop_off_w = (new_w - target_size) / 2.0
    crop_off_h = (new_h - target_size) / 2.0

    def transform(x, y, w, h):
        x1 = max(0.0, min(float(target_size), x * scale - crop_off_w))
        y1 = max(0.0, min(float(target_size), y * scale - crop_off_h))
        x2 = max(0.0, min(float(target_size), (x + w) * scale - crop_off_w))
        y2 = max(0.0, min(float(target_size), (y + h) * scale - crop_off_h))
        return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]

    return transform


def _build_annotation_index(json_paths, hour_lut, bbox_transform):
    """img_index: uuid -> {location, tod_bin, hour};
    bbox_index: (uuid, category_id) -> [[x1,y1,x2,y2], ...]"""
    img_index = {}
    bbox_index = defaultdict(list)
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        for img in data["images"]:
            uid = img["id"]
            dc = img.get("date_captured", "")
            if not dc:
                continue
            dt = datetime.strptime(dc, "%Y-%m-%d %H:%M:%S")
            if uid not in img_index:
                img_index[uid] = {
                    "location": int(img["location"]),
                    "tod_bin": hour_lut[dt.hour],
                    "hour": dt.hour,  # CHANGED: keep the hour for later re-binning
                }
        for ann in data["annotations"]:
            uid, cat_id, bbox = ann["image_id"], ann["category_id"], ann.get("bbox")
            if bbox is None or uid not in img_index:
                continue
            x, y, w, h = bbox
            if bbox_transform is not None:
                entry = bbox_transform(x, y, w, h)
            else:
                entry = [round(x, 3), round(y, 3), round(x + w, 3), round(y + h, 3)]
            if entry not in bbox_index[(uid, cat_id)]:
                bbox_index[(uid, cat_id)].append(entry)
    return img_index, bbox_index


_SYN_RE = re.compile(
    r"^syn_L(\d+)_to_L(\d+)_([^_]+)_"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"_s(\d+)\.jpg$",
    re.IGNORECASE,
)


def _parse_synthetic(fname):
    m = _SYN_RE.match(fname)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(4)


def _build_category_index(json_paths):
    """name -> category_id (TerraInc COCO ids)."""
    cats = {}
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        for c in data["categories"]:
            cats[c["name"]] = c["id"]
    return cats


def build_annotation_dict(samples, annotation_jsons, bins, root_dir=None,
                          target_size=None, species=None):
    """
    samples          list of (path, label_idx)        # CHANGED: in-memory list,
                     (was: path of a torch.save'd list)
    annotation_jsons list of annotation JSON paths
    bins             list of (name, hours)
    root_dir         base dir for relative paths (default cwd)
    target_size      transform bboxes to this square size; None = annotation space
    species          {label_idx: name}
    Returns {sample index -> {label, location, tod, tod_bin, hour, synthetic,
             src_uuid, src_location, bboxes}}.   # CHANGED: + hour, src_location
    Samples with no match in the annotations are DROPPED WITH A WARNING     # CHANGED
    (was: KeyError), so their index is simply absent from the result.
    """
    root = Path(root_dir) if root_dir is not None else Path.cwd()
    hour_lut = make_hour_lut(bins)
    names = [n for n, _ in bins]
    bbox_transform = _make_bbox_transform(target_size)
    json_paths = [str(_resolve(p, root)) for p in annotation_jsons]
    img_index, bbox_index = _build_annotation_index(json_paths, hour_lut, bbox_transform)
    cat_index = _build_category_index(json_paths)

    unmatched = sorted({n for n in species.values() if n not in cat_index})  # CHANGED
    if unmatched:
        warnings.warn(f"species folder names without a COCO category (their boxes "
                      f"will be empty): {unmatched}")

    result, errors = {}, []
    for idx, (abs_path, label) in enumerate(samples):
        fname = Path(_resolve(abs_path, root)).name
        species_name = species.get(label, None)
        cat_id = cat_index.get(species_name)
        parsed = _parse_synthetic(fname)
        if parsed is not None:  # ---- synthetic ----
            src_loc, tgt_loc, uid = parsed
            if uid not in img_index:
                errors.append((idx, fname, f"src_uuid {uid} not found"))
                continue
            rec = img_index[uid]
            result[idx] = {
                "label": label, "location": tgt_loc, "tod": names[rec["tod_bin"]],
                "tod_bin": rec["tod_bin"], "hour": rec["hour"], "synthetic": True,
                "src_uuid": uid, "src_location": src_loc,
                "bboxes": bbox_index.get((uid, cat_id), []) if cat_id else [],
            }
        else:  # ---- original ----
            uid = Path(fname).stem
            if uid not in img_index:
                errors.append((idx, fname, f"uuid {uid} not found"))
                continue
            rec = img_index[uid]
            result[idx] = {
                "label": label, "location": rec["location"], "tod": names[rec["tod_bin"]],
                "tod_bin": rec["tod_bin"], "hour": rec["hour"], "synthetic": False,
                "src_uuid": None, "src_location": -1,
                "bboxes": bbox_index.get((uid, cat_id), []) if cat_id else [],
            }
    if errors:  # CHANGED: warning instead of KeyError
        msg = f"{len(errors)} sample(s) not found in annotations (DROPPED):\n"
        msg += "".join(f"  [{i}] {f}: {r}\n" for i, f, r in errors[:10])
        if len(errors) > 10:
            msg += f"  ... and {len(errors) - 10} more\n"
        warnings.warn(msg)
    return result


# ======================================================================
# Scanning and reading images
# ======================================================================
def scan_images(root: str):
    """
    Find every image under root whose path ends <location>/<species>/<file>.
    Returns (rel_paths, labels, loc_dirs, class_names): class_names = species
    folder names EXPLICITLY sorted alphabetically (os.walk/scandir order is
    never relied on); labels index into it. Row order is sorted by path too.
    Files with fewer than two directory levels below root are skipped with a
    warning.
    """
    root = os.path.abspath(root)
    found, skipped = [], 0
    for dp, dn, fn in os.walk(root, followlinks=True):
        dn.sort()
        for f in sorted(fn):
            if not f.lower().endswith(IMG_EXT):
                continue
            rel = os.path.relpath(os.path.join(dp, f), root)
            parts = Path(rel).parts
            if len(parts) < 3:
                skipped += 1
                continue
            found.append((rel, parts[-2], parts[-3]))
    if skipped:
        warnings.warn(f"{skipped} image(s) not under <location>/<species>/ skipped")
    if not found:
        raise ValueError(f"no usable images under {root}")
    class_names = sorted({s for _, s, _ in found})
    cidx = {c: i for i, c in enumerate(class_names)}
    found.sort(key=lambda r: r[0])
    return [r[0] for r in found], [cidx[r[1]] for r in found], [r[2] for r in found], class_names


def _read_images(paths: Sequence[str], size: int, workers: int) -> torch.Tensor:
    """Read already-resized RGB images -> (B,3,size,size) float in [0,1].
    Raises if any image is not exactly size x size."""
    import numpy as np
    from PIL import Image

    def one(p):
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            raise ValueError(f"{p} is {im.size[0]}x{im.size[1]}, expected {size}x{size} "
                             f"(this tool does not resize)")
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return torch.stack(list(ex.map(one, paths))).float() / 255.0


def order_locations(ids, holdout=()) -> List[int]:
    """
    Order of location ids = the index order. Non-holdout (training) locations
    first, sorted ALPHABETICALLY by name (the raw id as a string); then the
    holdout (test) locations, also alphabetically. This reproduces the usual
    ImageFolder outcome where the test location (L100) sits in a different
    parent directory and so gets the LAST label. With holdout empty it is a
    plain alphabetical sort of all names (L100 would come first).
    """
    h = {int(v) for v in holdout}
    return sorted({int(v) for v in ids}, key=lambda v: (v in h, str(v)))


def _pad_boxes(boxes: List[List[List[float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    n = torch.tensor([len(b) for b in boxes], dtype=torch.long)
    m = max(1, int(n.max()) if len(n) else 1)
    out = torch.full((len(boxes), m, 4), -1.0)
    for i, b in enumerate(boxes):
        if b:
            out[i, : len(b)] = torch.tensor(b, dtype=torch.float32)
    return out, n


# ======================================================================
# Build
# ======================================================================
def build_ti_dataset(
    root: str,
    annotations: Sequence[str],
    vae_ckpt: str,
    repae_dir: str,
    *,
    target_size: int = 256,
    tod_bins: str = "2",
    synthetic: str = "include",
    multi_animal: str = "keep",
    holdout_locations: Sequence[int] = (),
    arch: str = "f16d32",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    amp: str = "off",
    batch_size: int = 64,
    workers: int = 8,
    limit: Optional[int] = None,
    save_path: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[TensorDataset, Dict]:
    """
    Build the dataset (see module docstring for every argument's meaning).
    holdout_locations: raw ids of the test location(s) (e.g. 100); they get the
    LAST indices (see order_locations), training locations come first.
    Extra arguments: arch / amp as in extract_features.py;
    batch_size, workers (image-reading threads); limit (keep only the first
    `limit` usable rows, for tests); save_path (write the dataset file there;
    refuses to overwrite an existing file unless overwrite=True).
    Returns (TensorDataset in FIELDS order, info dict) exactly like
    load_ti_dataset on the saved file.
    """
    if synthetic not in ("include", "exclude", "only"):
        raise ValueError("synthetic must be include|exclude|only")
    if multi_animal not in ("keep", "drop"):
        raise ValueError("multi_animal must be keep|drop")
    if save_path and os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f"{save_path} exists; pass overwrite=True or load it "
                              f"with load_ti_dataset (avoids silently reusing a stale cache).")
    from extract_features import extract_features
    from invae import load_repae_vae

    # 1. scan, 2. metadata (from the old utility)
    rel, labels, loc_dirs, class_names = scan_images(root)
    root_abs = os.path.abspath(root)
    meta_by_idx = build_annotation_dict(
        [(os.path.join(root_abs, r), l) for r, l in zip(rel, labels)],
        annotations, parse_tod_spec(tod_bins), target_size=target_size,
        species=dict(enumerate(class_names)),
    )

    # 3. policies (before any image is read)
    keep, n_syn_drop, n_multi_drop = [], 0, 0
    for i in sorted(meta_by_idx):
        m = meta_by_idx[i]
        if (synthetic == "exclude" and m["synthetic"]) or (synthetic == "only" and not m["synthetic"]):
            n_syn_drop += 1
            continue
        if multi_animal == "drop" and len(m["bboxes"]) > 1:
            n_multi_drop += 1
            continue
        keep.append(i)
    if limit:
        keep = keep[:limit]
    if not keep:
        raise ValueError("no samples left after matching and policies.")
    print(f"location order (index = e): {order_locations([meta_by_idx[i]['location'] for i in keep], holdout_locations)}")
    print(f"scanned {len(rel)}; matched {len(meta_by_idx)}; dropped by synthetic="
          f"{synthetic}: {n_syn_drop}; by multi_animal={multi_animal}: {n_multi_drop}; "
          f"kept {len(keep)}")

    loc = torch.tensor([meta_by_idx[i]["location"] for i in keep])
    # ids come from an EXPLICIT sort of the location names (never from disk /
    # annotation order); see order_locations.
    location_ids = order_locations(loc.tolist(), holdout_locations)
    e_idx = torch.tensor([location_ids.index(int(v)) for v in loc])
    y = torch.tensor([meta_by_idx[i]["label"] for i in keep])
    t = torch.tensor([meta_by_idx[i]["tod_bin"] for i in keep])

    # 4. extraction (extract_features.py)
    def batches():
        for s in range(0, len(keep), batch_size):
            sel = list(range(s, min(s + batch_size, len(keep))))
            imgs = _read_images([os.path.join(root_abs, rel[keep[j]]) for j in sel], target_size, workers)
            yield imgs, y[sel], t[sel], e_idx[sel], torch.tensor(sel)

    vae = load_repae_vae(repae_dir, vae_ckpt, arch, device)
    ex = extract_features(vae, batches(), device=device,
                          input_range="01", amp=amp, expected_hw=target_size)
    if not torch.equal(ex["image_id"], torch.arange(len(keep))):
        raise RuntimeError("row order lost during extraction")

    # 5. assemble
    boxes, n_boxes = _pad_boxes([meta_by_idx[i]["bboxes"] for i in keep])
    d = {
        "x": ex["x"], "y": y, "t": t, "e": e_idx, "loc": loc,
        "hour": torch.tensor([meta_by_idx[i]["hour"] for i in keep]),
        "synthetic": torch.tensor([meta_by_idx[i]["synthetic"] for i in keep]),
        "src_loc": torch.tensor([meta_by_idx[i]["src_location"] for i in keep]),
        "n_boxes": n_boxes, "boxes": boxes, "brightness": ex["brightness"],
        "uuid": [Path(rel[i]).stem for i in keep],
        "src_uuid": [meta_by_idx[i]["src_uuid"] for i in keep],
        "path": [rel[i] for i in keep], "loc_dir": [loc_dirs[i] for i in keep],
        "meta": {
            "class_names": class_names, "location_ids": location_ids,
            "tod_names": [n for n, _ in parse_tod_spec(tod_bins)], "tod_spec": tod_bins,
            "holdout_locations": [int(v) for v in holdout_locations],
            "synthetic_policy": synthetic, "multi_animal_policy": multi_animal,
            "target_size": target_size, "bbox_space": f"{target_size}x{target_size}",
            "vae_ckpt": os.path.abspath(vae_ckpt), "arch": arch,
            "root": root_abs, "n": len(keep),
        },
    }
    if save_path:
        torch.save(d, save_path)
        print("saved", save_path)
    return _to_dataset(d)


# ======================================================================
# Dataset view / loading
# ======================================================================
def _to_dataset(
    d: Dict,
    locations: Optional[Sequence[int]] = None,
    include_synthetic: bool = True,
    tod_bins: Optional[str] = None,
) -> Tuple[TensorDataset, Dict]:
    meta = d["meta"]
    N = d["x"].shape[0]
    keep = torch.ones(N, dtype=torch.bool)
    if locations is not None:
        unknown = set(int(v) for v in locations) - set(meta["location_ids"])
        if unknown:
            raise ValueError(f"locations {sorted(unknown)} not in file (has {meta['location_ids']})")
        keep &= torch.isin(d["loc"], torch.tensor(list(locations), dtype=torch.long))
    if not include_synthetic:
        keep &= ~d["synthetic"]
    if not keep.any():
        raise ValueError("filters removed every row.")
    if tod_bins is None:
        t, tod_names = d["t"], list(meta["tod_names"])
    else:
        t, tod_names = rebin(d["hour"], tod_bins)
    loc = d["loc"][keep]
    kept_ids = order_locations(loc.tolist(), meta.get("holdout_locations", ()))
    e = torch.tensor([kept_ids.index(int(v)) for v in loc], dtype=torch.long)
    n = int(keep.sum())
    tensors = {
        "x": d["x"][keep], "y": d["y"][keep], "t": t[keep], "e": e, "ids": torch.arange(n),
        "loc": loc, "hour": d["hour"][keep], "synthetic": d["synthetic"][keep],
        "src_loc": d["src_loc"][keep], "n_boxes": d["n_boxes"][keep], "boxes": d["boxes"][keep],
    }
    rows = keep.nonzero().flatten()
    sel = lambda lst: [lst[int(i)] for i in rows]
    info = {
        "class_names": list(meta["class_names"]), "tod_names": tod_names,
        "location_ids": kept_ids, "n_species": len(meta["class_names"]),
        "n_times": len(tod_names), "n_locations": len(kept_ids), "rows": rows,
        "brightness": d["brightness"][keep], "uuid": sel(d["uuid"]),
        "src_uuid": sel(d["src_uuid"]), "path": sel(d["path"]),
        "loc_dir": sel(d["loc_dir"]), "meta": meta,
    }
    return TensorDataset(*[tensors[k] for k in FIELDS]), info


def load_ti_dataset(path: str, locations: Optional[Sequence[int]] = None,
                    include_synthetic: bool = True, tod_bins: Optional[str] = None):
    """
    Load a saved dataset. Optional filters:
      locations         raw location ids to KEEP (e.g. the training
                        locations as raw ids); e is re-indexed 0..k-1 over the kept ones.
      include_synthetic False drops synthetic rows.
      tod_bins          None keeps the saved bins; "2", "4" or a custom spec
                        re-bins from the stored hour.
    Returns (TensorDataset in FIELDS order, info) with info: class_names,
    tod_names, location_ids (raw ids, index = e), n_species, n_times,
    n_locations, rows (row numbers in the saved file), brightness, uuid,
    src_uuid, path, loc_dir (filtered lists), meta.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    return _to_dataset(d, locations, include_synthetic, tod_bins)


# ======================================================================
# CLI
# ======================================================================
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--annotations", nargs="+", required=True)
    p.add_argument("--repae_dir", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--arch", default="f16d32", choices=["f16d32", "f8d4"])
    p.add_argument("--target_size", type=int, default=256)
    p.add_argument("--tod_bins", default="2", help='"2", "4" or "night:0-5;day:6-23"')
    p.add_argument("--synthetic", default="include", choices=["include", "exclude", "only"])
    p.add_argument("--multi_animal", default="keep", choices=["keep", "drop"])
    p.add_argument("--holdout_locations", type=int, nargs="*", default=[],
                   help="raw ids of the test location(s), e.g. 100; indexed last")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", default="off", choices=["off", "fp16", "bf16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None, help="save the dataset here")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    ds, info = build_ti_dataset(
        a.root, a.annotations, a.vae_ckpt, a.repae_dir, target_size=a.target_size,
        tod_bins=a.tod_bins, synthetic=a.synthetic, multi_animal=a.multi_animal, holdout_locations=a.holdout_locations,
        arch=a.arch, device=a.device, amp=a.amp,
        batch_size=a.batch_size, workers=a.workers, limit=a.limit,
        save_path=a.out, overwrite=a.overwrite,
    )
    print(f"rows={len(ds)} classes={info['class_names']} tod={info['tod_names']} "
          f"locations={info['location_ids']}")
