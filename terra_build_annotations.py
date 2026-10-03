"""
build_annotations.py
====================
Builds a flat JSON annotation file keyed by ImageFolder sample index.

For each sample in a saved ImageFolder samples list, looks up its metadata
from TerraInc COCO-style annotation JSONs and emits one record per index.

Output format
-------------
{
    "0": {
        "label"    : 8,
        "location" : 38,
        "tod"      : "dusk",
        "tod_bin"  : 3,
        "synthetic": false,
        "src_uuid" : null,
        "bboxes"   : [[x1, y1, x2, y2], ...]   # in target_size space, empty for 'empty' class
    },
    ...
}

Bbox coordinate transform (when target_size is provided)
---------------------------------------------------------
Annotation space: 1494 x 2048 (H x W)
Preprocessing: proportional resize (short side -> target_size) then center crop.
  scale      = target_size / 1494
  new_w      = int(2048 * scale)
  crop_off_w = (new_w - target_size) / 2
  crop_off_h = (int(1494 * scale) - target_size) / 2  (= 0 for target_size=224)

  x_new = x_ann * scale - crop_off_w   (clipped to [0, target_size])
  y_new = y_ann * scale - crop_off_h   (clipped to [0, target_size])
  w_new = w_ann * scale
  h_new = h_ann * scale

Multi-animal images
-------------------
A single physical file may contain multiple animals of the same species,
each with its own bbox. All bboxes for the matching category are stored
as a list. For the 'empty' class, bboxes is an empty list.

Usage
-----
python build_annotations.py \
    --samples      dataset_samples.pt \
    --annotations  train_annotations.json cis_val_annotations.json \
                   cis_test_annotations.json trans_val_annotations.json \
                   trans_test_annotations.json \
    --bins         4 \
    --target_size  224 \
    --root_dir     /path/to/data \
    --out          dataset_annotations.json
"""

import json
import re
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ---------------------------------------------------------------------------
# Binning presets
# ---------------------------------------------------------------------------

BINS_4 = [
    ('night', range( 0,  6)),
    ('dawn',  range( 6,  9)),
    ('day',   range( 9, 18)),
    ('dusk',  range(18, 24)),
]

BINS_2 = [
    ('night', range( 0,  6)),
    ('day',   range( 6, 24)),
]


def make_hour_lut(bins):
    lut = [-1] * 24
    for bin_idx, (name, hours) in enumerate(bins):
        for h in hours:
            if lut[h] != -1:
                raise ValueError(f"Hour {h} assigned to more than one bin")
            lut[h] = bin_idx
    missing = [h for h in range(24) if lut[h] == -1]
    if missing:
        raise ValueError(f"Hours not covered by any bin: {missing}")
    return lut


def bin_names(bins):
    return [name for name, _ in bins]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve(p, root):
    p = Path(p)
    return p if p.is_absolute() else root / p


# ---------------------------------------------------------------------------
# Bbox transform: annotation space -> target_size x target_size
# ---------------------------------------------------------------------------

ANN_H = 1494
ANN_W = 2048

def _make_bbox_transform(target_size):
    """
    Returns a function that maps [x, y, w, h] in annotation space
    to [x1, y1, x2, y2] in target_size x target_size space.
    Returns None if target_size is None (no transform).
    """
    if target_size is None:
        return None

    scale      = target_size / ANN_H
    new_w      = int(ANN_W * scale)
    new_h      = int(ANN_H * scale)
    crop_off_w = (new_w - target_size) / 2.0
    crop_off_h = (new_h - target_size) / 2.0

    def transform(x, y, w, h):
        x1 = max(0.0, min(float(target_size), x * scale - crop_off_w))
        y1 = max(0.0, min(float(target_size), y * scale - crop_off_h))
        x2 = max(0.0, min(float(target_size), (x + w) * scale - crop_off_w))
        y2 = max(0.0, min(float(target_size), (y + h) * scale - crop_off_h))
        return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]

    return transform


# ---------------------------------------------------------------------------
# Annotation index
# ---------------------------------------------------------------------------

def _build_annotation_index(json_paths, hour_lut, bbox_transform):
    """
    Build two indices:
      img_index : uuid -> {location, tod_bin}
      bbox_index: (uuid, category_id) -> [[x1,y1,x2,y2], ...]
    """
    img_index  = {}
    bbox_index = defaultdict(list)

    for p in json_paths:
        with open(p) as f:
            data = json.load(f)

        for img in data['images']:
            uid = img['id']
            dc  = img.get('date_captured', '')
            if not dc:
                continue
            dt      = datetime.strptime(dc, '%Y-%m-%d %H:%M:%S')
            tod_bin = hour_lut[dt.hour]
            if uid not in img_index:
                img_index[uid] = {
                    'location': int(img['location']),
                    'tod_bin':  tod_bin,
                }

        for ann in data['annotations']:
            uid    = ann['image_id']
            cat_id = ann['category_id']
            bbox   = ann.get('bbox')
            if bbox is None or uid not in img_index:
                continue
            key = (uid, cat_id)
            x, y, w, h = bbox
            if bbox_transform is not None:
                entry = bbox_transform(x, y, w, h)
            else:
                entry = [round(x,3), round(y,3),
                         round(x+w,3), round(y+h,3)]
            # avoid duplicates across files
            if entry not in bbox_index[key]:
                bbox_index[key].append(entry)

    return img_index, bbox_index


# ---------------------------------------------------------------------------
# Synthetic filename parser
# ---------------------------------------------------------------------------

_SYN_RE = re.compile(
    r'^syn_L(\d+)_to_L(\d+)_([^_]+)_'
    r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
    r'_s(\d+)\.jpg$',
    re.IGNORECASE,
)

def _parse_synthetic(fname):
    m = _SYN_RE.match(fname)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(4)


# ---------------------------------------------------------------------------
# Category id lookup  (built from annotation jsons)
# ---------------------------------------------------------------------------

def _build_category_index(json_paths):
    """name -> category_id  (TerraInc COCO ids, not ImageFolder label indices)"""
    cats = {}
    for p in json_paths:
        with open(p) as f:
            data = json.load(f)
        for c in data['categories']:
            cats[c['name']] = c['id']
    return cats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ImageFolder label index -> species name (alphabetical)
DEFAULT_SPECIES = {
    0: 'bird', 1: 'bobcat', 2: 'cat', 3: 'coyote', 4: 'dog',
    5: 'empty', 6: 'opossum', 7: 'rabbit', 8: 'raccoon', 9: 'squirrel',
}

def build_annotation_dict(samples_path, annotation_jsons, bins,
                          root_dir=None, target_size=None, species=None):
    """
    Parameters
    ----------
    samples_path : str or Path
        Path to torch.save'd dataset.samples list of (path, label_idx).
    annotation_jsons : list of str or Path
        All available TerraInc annotation JSON files.
    bins : list of (name, hours) tuples
        Binning strategy. Use BINS_4 or BINS_2, or define your own.
    root_dir : str or Path, optional
        Base dir for resolving relative paths. Defaults to cwd.
    target_size : int, optional
        If provided, bboxes are transformed to target_size x target_size space.
        If None, bboxes are stored as [x1,y1,x2,y2] in annotation space.
    species : dict {label_idx: name}, optional
        Maps ImageFolder label index to species name.
        Defaults to DEFAULT_SPECIES.

    Returns
    -------
    dict : {int index -> dict with keys:
              label, location, tod, tod_bin, synthetic, src_uuid, bboxes}
    """
    if species is None:
        species = DEFAULT_SPECIES

    root           = Path(root_dir) if root_dir is not None else Path.cwd()
    hour_lut       = make_hour_lut(bins)
    names          = bin_names(bins)
    bbox_transform = _make_bbox_transform(target_size)
    json_paths     = [str(_resolve(p, root)) for p in annotation_jsons]

    img_index, bbox_index = _build_annotation_index(
                                json_paths, hour_lut, bbox_transform)
    cat_index = _build_category_index(json_paths)  # name -> coco category_id

    samples = torch.load(str(_resolve(samples_path, root)), weights_only=False)

    result = {}
    errors = []

    for idx, (abs_path, label) in enumerate(samples):
        fname        = Path(_resolve(abs_path, root)).name
        species_name = species.get(label, None)
        parsed       = _parse_synthetic(fname)

        if parsed is not None:
            # ---- synthetic ----
            src_loc, tgt_loc, src_uuid = parsed

            if src_uuid not in img_index:
                errors.append((idx, fname, f"src_uuid {src_uuid} not found"))
                continue

            rec      = img_index[src_uuid]
            cat_id   = cat_index.get(species_name)
            bboxes   = bbox_index.get((src_uuid, cat_id), []) if cat_id else []

            result[idx] = {
                'label':     label,
                'location':  tgt_loc,
                'tod':       names[rec['tod_bin']],
                'tod_bin':   rec['tod_bin'],
                'synthetic': True,
                'src_uuid':  src_uuid,
                'bboxes':    bboxes,
            }

        else:
            # ---- original ----
            uid = Path(fname).stem

            if uid not in img_index:
                errors.append((idx, fname, f"uuid {uid} not found"))
                continue

            rec    = img_index[uid]
            cat_id = cat_index.get(species_name)
            bboxes = bbox_index.get((uid, cat_id), []) if cat_id else []

            result[idx] = {
                'label':     label,
                'location':  rec['location'],
                'tod':       names[rec['tod_bin']],
                'tod_bin':   rec['tod_bin'],
                'synthetic': False,
                'src_uuid':  None,
                'bboxes':    bboxes,
            }

    if errors:
        msg = f"{len(errors)} sample(s) not found in annotation index:\n"
        for i, fname, reason in errors[:10]:
            msg += f"  [{i}] {fname}: {reason}\n"
        if len(errors) > 10:
            msg += f"  ... and {len(errors) - 10} more\n"
        raise KeyError(msg)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',      required=True,
                        help='torch.save()d dataset.samples (.pt)')
    parser.add_argument('--annotations',  nargs='+', required=True,
                        help='TerraInc annotation JSON files')
    parser.add_argument('--bins',         choices=['4', '2'], default='4',
                        help='4=night/dawn/day/dusk  2=night/day')
    parser.add_argument('--target_size',  type=int, default=None,
                        help='Rescale bboxes to this image size (e.g. 224)')
    parser.add_argument('--root_dir',     default=None,
                        help='Base dir for resolving relative paths')
    parser.add_argument('--out',          default='dataset_annotations.json',
                        help='Output JSON path')
    args = parser.parse_args()

    bins = BINS_4 if args.bins == '4' else BINS_2

    ann = build_annotation_dict(
        samples_path     = args.samples,
        annotation_jsons = args.annotations,
        bins             = bins,
        root_dir         = args.root_dir,
        target_size      = args.target_size,
    )

    out = {str(k): v for k, v in ann.items()}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"Written {len(ann)} entries to {args.out}")

    from collections import Counter
    locs  = Counter(v['location'] for v in ann.values())
    tods  = Counter(v['tod']      for v in ann.values())
    syns  = sum(1 for v in ann.values() if v['synthetic'])
    multi = sum(1 for v in ann.values() if len(v['bboxes']) > 1)
    empty = sum(1 for v in ann.values() if len(v['bboxes']) == 0)
    print(f"Locations    : {dict(sorted(locs.items()))}")
    print(f"TOD          : {dict(tods)}")
    print(f"Synthetic    : {syns} / {len(ann)}")
    print(f"Multi-bbox   : {multi}")
    print(f"Empty/no-bbox: {empty}")
