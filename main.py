"""
main.py -- sets everything up from command-line arguments and runs training.

Pipeline built here (each piece lives in its own file):
    data (cached IN-VAE latents)   -> this file (load / synthetic)
    geodesic table (if needed)     -> geodesic.py
    network                        -> m1net.py (+ head.py)
    losses                         -> losses.py
    model = network + losses       -> model.py
    training loop                  -> trainer.py

=======================================================================
INPUT DATA
=======================================================================
--features PATH  the dataset file written by ti_dataset.py (IN-VAE latents with
                 all labels/metadata, test location included). Loaded with
                 ti_dataset.load_ti_dataset, filtered by:
    --train_locations IDS   raw location ids to train on (REQUIRED with
                            --features; leave the test location out, e.g.
                            "--train_locations 36 38 43"). e is re-indexed
                            0..k-1 over these.
    --exclude_synthetic     drop synthetic (transported) images
    --tod_bins SPEC         re-bin time of day from the stored hour ("2", "4"
                            or a custom spec); default keeps the saved bins.
n_species / n_times / n_locations come from the loaded file (override with
the flags below only for debugging). Multi-animal handling is a build-time
choice of ti_dataset.py (keep/drop), not applied here.
--synthetic      ignore --features and generate small random data (smoke test)

C, H, W are read from x (nothing is hard-coded).

=======================================================================
EXAMPLE
=======================================================================
python main.py --features ti.pt --train_locations 36 38 43 --n_content 8 --n_style 2 \\
    --ae_channels 32,64,128 --epochs 100 --lambda_inj 0.1 \\
    --geod_mode full --geod_jobs 8
python main.py --synthetic --epochs 4 --check_epoch 2
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from geodesic import GeodesicTable, build_geodesic_table
from head import HeadConfig
from ti_dataset import load_ti_dataset
from losses import LossConfig
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
    # losses
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_rec", type=float, default=10.0)
    p.add_argument("--lambda_sparse", type=float, default=0.01)
    p.add_argument("--lambda_moral", type=float, default=0.0)
    p.add_argument("--lambda_inj", type=float, default=0.0)
    p.add_argument("--lambda_bilip", type=float, default=0.0)
    p.add_argument("--inj_thresh", type=float, default=0.3)
    p.add_argument("--bilip_L", type=float, default=2.0)
    p.add_argument("--bilip_subset_frac", type=float, default=0.3)
    # geodesic table
    p.add_argument("--geod_path", type=str, default=None, help="load if exists, else build and save here")
    p.add_argument("--geod_mode", type=str, default="full", choices=["full", "landmark"])
    p.add_argument("--geod_k", type=int, default=10)
    p.add_argument("--geod_landmarks", type=int, default=2000)
    p.add_argument("--geod_pca", type=int, default=256, help="0 disables PCA")
    p.add_argument("--geod_chunk", type=int, default=2048)
    p.add_argument("--geod_jobs", type=int, default=1)
    p.add_argument("--geod_fp16", action="store_true")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adj_lr", type=float, default=1e-3)
    p.add_argument("--check_epoch", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="m1_run.pt")
    return p.parse_args()


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
    ds, info = load_ti_dataset(
        a.features, locations=a.train_locations,
        include_synthetic=not a.exclude_synthetic, tod_bins=a.tod_bins,
    )
    return ds, info


def get_geod(a, x):
    """Load or build the geodesic table (only called if lambda_inj > 0)."""
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


def main():
    a = parse()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ds, info = load_data(a)
    x, y, t, e = ds.tensors[:4]
    N, C, H, W = x.shape
    if info is not None:
        n_sp, n_t, n_e = info["n_species"], info["n_times"], info["n_locations"]
        print(f"classes={info['class_names']} tod={info['tod_names']} "
              f"locations(e order)={info['location_ids']}")
    else:
        n_sp, n_t, n_e = int(y.max()) + 1, int(t.max()) + 1, int(e.max()) + 1
    n_sp, n_t, n_e = a.n_species or n_sp, a.n_times or n_t, a.n_locations or n_e
    print(f"data: N={N} x=({C},{H},{W}) species={n_sp} times={n_t} locations={n_e}")

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
    loss_cfg = LossConfig(
        lambda_kl=a.lambda_kl, lambda_rec=a.lambda_rec, lambda_sparse=a.lambda_sparse,
        lambda_moral=a.lambda_moral, lambda_inj=a.lambda_inj, lambda_bilip=a.lambda_bilip,
        inj_thresh=a.inj_thresh, bilip_L=a.bilip_L, bilip_subset_frac=a.bilip_subset_frac,
    )
    model = M1Model(net_cfg, loss_cfg)
    print(model.net.head.summary())

    geod = get_geod(a, x) if a.lambda_inj > 0 else None

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True)

    tr_cfg = TrainerConfig(
        epochs=a.epochs, lr=a.lr, adj_lr=a.adj_lr, check_epoch=a.check_epoch,
        grad_clip=a.grad_clip, device=a.device,
    )
    trainer = Trainer(model, tr_cfg, geod=geod)
    trainer.fit(loader)
    trainer.save(a.out)
    print("final content adjacency:\n", model.net.prior.get_adj().int())
    print("saved", a.out)


if __name__ == "__main__":
    main()
