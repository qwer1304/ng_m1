"""
main.py -- sets everything up from command-line arguments and runs training.

Pipeline built here (each piece lives in its own file):
    data (cached IN-VAE latents)   -> this file (load / split / synthetic)
    configuration tree             -> config.py (RunConfig)
    geodesic table (if needed)     -> geodesic.py
    network                        -> m1net.py (+ head.py)
    losses                         -> losses.py
    model = network + losses       -> model.py
    training loop, checkpoints     -> trainer.py

=======================================================================
WHERE EACH SETTING LIVES
=======================================================================
  * Loss weights, ramps, caps, gate numbers, max_epochs, the global switch
    schedule.enabled: the RunConfig tree of config.py. Set them with
        --config run.json          (optional JSON, partial is fine)
        --set loss.lambda_kl=2     (repeatable; dotted overrides)
    Nothing about them is a flag of this file.
  * Network shape (n_content, n_style, ae_*, ...), data selection, optimizer
    settings, batch size, validation split: flags of this file.

=======================================================================
INPUT DATA
=======================================================================
--features PATH  dataset file written by ti_dataset.py (IN-VAE latents with all
                 labels/metadata, test location included). Filtered by:
    --train_locations IDS   raw location ids to train on (REQUIRED with
                            --features; leave the test location out, e.g.
                            "--train_locations 36 38 43"). e is re-indexed
                            0..k-1 over these.
    --exclude_synthetic     drop synthetic (transported) images
    --tod_bins SPEC         re-bin time of day from the stored hour.
--synthetic      ignore --features and generate small random data (smoke test)

Validation: --val_frac of the rows (default 0.1), chosen stratified by the
(species, time, location) cell, from the TRAINING locations only. Never a
held-out location. If synthetic images are included the split is NOT by
src_uuid (deferred), so a source image and its transported copies can
straddle the split; a warning is printed.

The geodesic table (needed iff model.needs_geod()) is built on ALL rows of the
selected training locations (train and validation rows), indexed by the row
ids of the dataset; the validation rows only act as extra geometry for the
regulariser targets.

REC baseline: main measures the trivial reconstruction error on the TRAINING
rows (variance of x around its per-feature mean, averaged over all entries,
the best constant predictor) and gives it to the model
(model.set_rec_baseline), which uses it at the KL cap.

=======================================================================
OUTPUT (--out_dir, default runs/<timestamp>)
=======================================================================
args.json, run_cfg.json (written at the start); events.log, log.jsonl,
last.pt, report.json, failure.json (only on failure) written by the Trainer.
Exit status is 1 when the run ended in a failure state, 0 otherwise.

=======================================================================
RESUME
=======================================================================
--resume out_dir/last.pt restarts from the checkpoint: the arguments the run
was started with (saved in the checkpoint) and the saved RunConfig are used;
only --device and any NEW --set overrides (applied on top of the saved
RunConfig, e.g. a larger schedule.max_epochs) are taken from the command
line. The Trainer restores model, schedule state, optimizers and RNG.

=======================================================================
EXAMPLES
=======================================================================
python main.py --features ti.pt --train_locations 36 38 43 --n_content 8 --n_style 2 \\
    --geod_mode full --geod_jobs 8 --set schedule.max_epochs=300
python main.py --synthetic --set schedule.max_epochs=4 --set loss.lambda_inj=0
python main.py --resume runs/20261004-101500/last.pt --set schedule.max_epochs=400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from config import RunConfig
from geodesic import GeodesicTable, build_geodesic_table
from head import HeadConfig
from ti_dataset import load_ti_dataset
from m1net import M1NetConfig
from model import M1Model
from trainer import Trainer, TrainerConfig


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # data
    p.add_argument("--features", type=str, default=None)
    p.add_argument("--train_locations", type=int, nargs="+", default=None)
    p.add_argument("--exclude_synthetic", action="store_true")
    p.add_argument("--tod_bins", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n_species", type=int, default=None)
    p.add_argument("--n_times", type=int, default=None)
    p.add_argument("--n_locations", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_frac", type=float, default=0.1, help="0 = no validation")
    # network
    p.add_argument("--n_content", type=int, default=8)
    p.add_argument("--n_style", type=int, default=2)
    p.add_argument("--ae_channels", type=str, default="32,64,128",
                   help="AutoencoderKL block_out_channels; N entries give N-1 halvings")
    p.add_argument("--ae_layers_per_block", type=int, default=1)
    p.add_argument("--ae_latent_channels", type=int, default=4)
    p.add_argument("--internal_latent", type=str, default="mean", choices=["mean", "sample"])
    p.add_argument("--emb_dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--mechanism_sharing", type=str, default="shared", choices=["shared", "per_node"])
    # run configuration tree (config.py)
    p.add_argument("--config", type=str, default=None, help="JSON file with (part of) the RunConfig tree")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                   help="dotted override of the RunConfig, repeatable")
    # geodesic table
    p.add_argument("--geod_path", type=str, default=None, help="load if exists, else build and save here")
    p.add_argument("--geod_mode", type=str, default="full", choices=["full", "landmark"])
    p.add_argument("--geod_k", type=int, default=10)
    p.add_argument("--geod_landmarks", type=int, default=2000)
    p.add_argument("--geod_pca", type=int, default=256, help="0 disables PCA")
    p.add_argument("--geod_chunk", type=int, default=2048)
    p.add_argument("--geod_jobs", type=int, default=1)
    p.add_argument("--geod_fp16", action="store_true")
    # optimization (Trainer)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adj_lr", type=float, default=1e-3)
    p.add_argument("--adj_momentum", type=float, default=0.9)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--ckpt_every", type=int, default=1)
    p.add_argument("--no_val_acc", action="store_true")
    # run
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_data(a):
    """Return (dataset, info). dataset yields (x, y, t, e, ids, ...) with
    x (N,C,H,W); info is the dict of load_ti_dataset (None for --synthetic)."""
    if a.synthetic:
        g = torch.Generator().manual_seed(a.seed)
        N = 512
        x = torch.randn(N, 32, 16, 16, generator=g)
        y = torch.randint(0, 10, (N,), generator=g)
        t = torch.randint(0, 2, (N,), generator=g)
        e = torch.randint(0, 3, (N,), generator=g)
        return TensorDataset(x, y, t, e, torch.arange(N)), None
    if a.features is None:
        raise SystemExit("give --features PATH or --synthetic")
    if not a.train_locations:
        raise SystemExit("--train_locations is required with --features "
                         "(otherwise the test location would be trained on)")
    return load_ti_dataset(
        a.features, locations=a.train_locations,
        include_synthetic=not a.exclude_synthetic, tod_bins=a.tod_bins,
    )


def split_indices(y, t, e, val_frac, seed):
    """Stratified train/validation split by the (y, t, e) cell. A cell with
    fewer than 2 rows goes entirely to training. Returns (train_idx, val_idx)
    as sorted numpy arrays. Reproducible from (labels, val_frac, seed)."""
    n = len(y)
    if val_frac <= 0:
        return np.arange(n), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    cell = (y.numpy().astype(np.int64) * 1000 + t.numpy()) * 1000 + e.numpy()
    val = []
    for c in np.unique(cell):
        idx = np.nonzero(cell == c)[0]
        k = int(round(val_frac * len(idx))) if len(idx) >= 2 else 0
        if k > 0:
            val.extend(rng.choice(idx, k, replace=False).tolist())
    val = np.array(sorted(val), dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[val] = False
    return np.nonzero(mask)[0], val


def rec_baseline(x):
    """Trivial REC: variance of x around its per-feature mean, averaged over
    all entries (the MSE of the best constant predictor)."""
    return float(x.float().var(dim=0, unbiased=False).mean())


# ----------------------------------------------------------------------
# geodesic table
# ----------------------------------------------------------------------
def get_geod(a, x):
    """Load or build the geodesic table (only called if model.needs_geod())."""
    if a.geod_path and os.path.exists(a.geod_path):
        print("loading geodesic table", a.geod_path)
        return GeodesicTable.load(a.geod_path)
    print(f"building geodesic table ({a.geod_mode}, k={a.geod_k}) ...")
    tab = build_geodesic_table(
        x.numpy(), k=a.geod_k, mode=a.geod_mode, n_landmarks=a.geod_landmarks,
        pca_dim=(a.geod_pca or None), chunk=a.geod_chunk,
        device=a.device, n_jobs=a.geod_jobs,
        dtype=np.float16 if a.geod_fp16 else np.float32, seed=a.seed,
    )
    if a.geod_path:
        tab.save(a.geod_path)
    return tab


# ----------------------------------------------------------------------
# resume: arguments and RunConfig
# ----------------------------------------------------------------------
def resolve_args_and_cfg(a):
    """Return (args, run_cfg, ckpt_or_None). Fresh run: CLI args, config file
    plus --set. Resume: saved args and saved RunConfig, plus new --set only."""
    if a.resume is None:
        cfg = RunConfig()
        if a.config:
            cfg = RunConfig.load(a.config, base=cfg)
        cfg = cfg.with_overrides(a.set).finalize()
        if a.out_dir is None:
            a.out_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
        return a, cfg, None
    ck = torch.load(a.resume, map_location="cpu", weights_only=False)
    saved = dict(ck["run_args"])
    new_set, device = list(a.set), a.device
    if a.config:
        print("note: --config ignored on resume (the saved RunConfig is used); use --set for changes")
    b = argparse.Namespace(**saved)
    b.resume, b.device, b.config = a.resume, device, saved.get("config")
    b.set = list(saved.get("set", [])) + new_set
    cfg = RunConfig.from_dict(ck["run_cfg"]).with_overrides(new_set).finalize()
    print(f"resuming from {a.resume} (epoch {ck['epoch']}); using the saved arguments"
          + (f", new overrides {new_set}" if new_set else ""))
    return b, cfg, ck


# ----------------------------------------------------------------------
def main():
    a, run_cfg, ck = resolve_args_and_cfg(parse())
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    run_cfg.save(os.path.join(a.out_dir, "run_cfg.json"))
    with open(os.path.join(a.out_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2, default=str)

    # ---- data ----
    ds, info = load_data(a)
    x, y, t, e = ds.tensors[:4]
    N, C, H, W = x.shape
    if info is not None:
        n_sp, n_t, n_e = info["n_species"], info["n_times"], info["n_locations"]
        print(f"classes={info['class_names']} tod={info['tod_names']} "
              f"locations(e order)={info['location_ids']}")
        if not a.exclude_synthetic and a.val_frac > 0:
            print("WARNING: synthetic images are included and the validation split is not by "
                  "src_uuid; validation numbers may be optimistic.")
    else:
        n_sp, n_t, n_e = int(y.max()) + 1, int(t.max()) + 1, int(e.max()) + 1
    n_sp, n_t, n_e = a.n_species or n_sp, a.n_times or n_t, a.n_locations or n_e
    print(f"data: N={N} x=({C},{H},{W}) species={n_sp} times={n_t} locations={n_e}")

    tr_idx, va_idx = split_indices(y, t, e, a.val_frac, a.seed)
    print(f"split: train={len(tr_idx)} val={len(va_idx)}")
    train_loader = DataLoader(Subset(ds, tr_idx.tolist()), batch_size=a.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = (DataLoader(Subset(ds, va_idx.tolist()), batch_size=a.batch_size, shuffle=False)
                  if len(va_idx) else None)
    if len(train_loader) == 0:
        raise SystemExit(f"batch_size {a.batch_size} exceeds the {len(tr_idx)} training rows "
                         f"(drop_last=True would give no batch)")

    # ---- network + model ----
    channels = tuple(int(c) for c in a.ae_channels.split(","))
    head_cfg = HeadConfig(
        in_channels=C, in_height=H, in_width=W, n_latents=a.n_content + a.n_style,
        ae_down_block_types=("DownEncoderBlock2D",) * len(channels),
        ae_up_block_types=("UpDecoderBlock2D",) * len(channels),
        ae_block_out_channels=channels,
        ae_layers_per_block=a.ae_layers_per_block,
        ae_latent_channels=a.ae_latent_channels,
        internal_latent=a.internal_latent,
    )
    net_cfg = M1NetConfig(
        head=head_cfg, n_content=a.n_content, n_style=a.n_style,
        n_species=n_sp, n_times=n_t, n_locations=n_e,
        emb_dim=a.emb_dim, hidden=a.hidden, mechanism_sharing=a.mechanism_sharing,
    )
    model = M1Model(net_cfg, run_cfg)
    print(model.net.head.summary())
    model.set_rec_baseline(rec_baseline(x[torch.as_tensor(tr_idx)]))
    print(f"REC baseline (trivial, train rows): {model.losses.terms['rec'].baseline:.5f}")

    geod = get_geod(a, x) if model.needs_geod() else None

    # ---- trainer ----
    tr_cfg = TrainerConfig(
        lr=a.lr, adj_lr=a.adj_lr, adj_momentum=a.adj_momentum, grad_clip=a.grad_clip,
        device=a.device, log_every=a.log_every, ckpt_every=a.ckpt_every,
        val_acc=not a.no_val_acc,
    )
    trainer = Trainer(model, tr_cfg, out_dir=a.out_dir, geod=geod, run_args=vars(a))
    if ck is not None:
        trainer.load(a.resume)
    res = trainer.fit(train_loader, val_loader)

    print("outcome:", res["outcome"], "|", res["reason"])
    print("final content adjacency:\n", model.net.prior.get_adj().int())
    print("outputs in", a.out_dir)
    return 1 if res["outcome"].startswith("failure") else 0


if __name__ == "__main__":
    sys.exit(main())
