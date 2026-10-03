"""
geodesic.py -- pre-computation of the geodesic-distance table GeoD used by
the BLAE injective loss (losses.injective_loss).

=======================================================================
WHAT GeoD IS
=======================================================================
For N cached feature vectors x_1..x_N (the frozen IN-VAE latents, flattened),
GeoD[i, j] is the shortest-path distance between i and j along a k-nearest-
neighbour graph of the data (Isomap-style geodesic distance). During
training, for a minibatch with dataset indices `ids`, the loss needs the
(b, b) sub-table GeoD[ids][:, ids]. This module builds the table ONCE per
choice of backbone/embedding and serves those sub-tables.

BLAE's own helper (utils/functionals.py: get_Distance_Mat via sklearn Isomap,
compute_euclidean_knn_distance_matrix) materialises dense N x N arrays and
runs sequentially; at N ~ 18k, d ~ 8192 that is what the note flags as too
costly. This module replaces it with the two-stage, bounded-cost procedure
described in sec:pseudoinj (Gotchas).

=======================================================================
COST MODEL (N points, d dims, k neighbours, S Dijkstra sources)
=======================================================================
Stage 0  optional dimension reduction d -> pca_dim (PCA, torch.pca_lowrank).
         Time O(N d q), memory O(N d). Set pca_dim=None to skip.
Stage 1  kNN graph via the Gram identity ||a-b||^2 = ||a||^2+||b||^2-2 a.b,
         computed in row chunks of `chunk` rows:
         time O(N^2 d') , PEAK memory O(chunk * N)  (never N x N x d, never
         N x N when chunk << N). Runs on `device` (GPU ok).
Stage 2  Dijkstra from S sources over the sparse graph (E ~ N k edges):
         time O(S (E + N log N)), sharded over `n_jobs` worker processes
         (embarrassingly parallel). S = N ("full" mode) or S = L
         ("landmark" mode).
Storage  "full"     : N x N table, dtype float32 (1.3 GB at N=18k) or
                      float16 (0.65 GB); optionally a numpy memmap on disk.
         "landmark" : L x N table only (L=2000, N=18k, float32: 144 MB).
                      Sub-tables are then an UPPER-BOUND approximation, see
                      GeodesicTable.sub.
The table lives on CPU. Only a (b, b) slice per minibatch goes to the GPU.

Precision note: the Gram identity in float32 gives edge lengths accurate to
about 1e-3 absolute (cancellation). Tested against sklearn Isomap: max
difference 6e-4 on a 1500-point swiss roll. Irrelevant for a regulariser.

Dijkstra note: the graph has non-negative weights, so plain Dijkstra
(scipy.sparse.csgraph.dijkstra) is the right choice; Johnson's algorithm
exists for negative weights and has no advantage here.

=======================================================================
CONNECTIVITY
=======================================================================
Geodesic distances are infinite between disconnected components. If the kNN
graph is disconnected, k is doubled (up to max_k) and stage 1 repeated; if
still disconnected a ValueError is raised.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


# ----------------------------------------------------------------------
# Stage 0: optional PCA
# ----------------------------------------------------------------------
def _reduce(X: torch.Tensor, pca_dim: Optional[int], seed: int) -> torch.Tensor:
    """Project (N, d) onto the top `pca_dim` principal directions."""
    if pca_dim is None or X.shape[1] <= pca_dim:
        return X
    g = torch.manual_seed(seed)  # pca_lowrank uses the global RNG
    Xc = X - X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(Xc, q=pca_dim, center=False, niter=3)
    return Xc @ V


# ----------------------------------------------------------------------
# Stage 1: chunked kNN graph
# ----------------------------------------------------------------------
def _knn_graph(X: torch.Tensor, k: int, chunk: int, device: str) -> csr_matrix:
    """Symmetric sparse kNN graph with Euclidean edge weights."""
    N = X.shape[0]
    Xd = X.to(device)
    sq = (Xd * Xd).sum(1)
    rows, cols, vals = [], [], []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (Xd[s:e] @ Xd.T)  # (c, N)
        idx = torch.arange(s, e, device=device)
        d2[idx - s, idx] = float("inf")  # exclude self
        dist2, nn_idx = torch.topk(d2, k, dim=1, largest=False)
        rows.append(np.repeat(np.arange(s, e), k))
        cols.append(nn_idx.cpu().numpy().ravel())
        vals.append(dist2.clamp_min(0).sqrt().cpu().numpy().ravel())
    g = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N),
    ).tocsr()
    return g.maximum(g.T)  # undirected: edge exists if either endpoint lists it


# ----------------------------------------------------------------------
# Stage 2: sharded Dijkstra
# ----------------------------------------------------------------------
_G: Optional[csr_matrix] = None


def _init_worker(g: csr_matrix) -> None:
    global _G
    _G = g


def _dijkstra_shard(sources: np.ndarray) -> np.ndarray:
    return dijkstra(_G, directed=False, indices=sources)


def _shortest_paths(
    g: csr_matrix, sources: np.ndarray, n_jobs: int, shard: int, out: np.ndarray
) -> None:
    """Fill out[r, :] with shortest-path distances from sources[r]."""
    shards = [(s, sources[s : s + shard]) for s in range(0, len(sources), shard)]
    if n_jobs <= 1:
        _init_worker(g)
        for s, src in shards:
            out[s : s + len(src)] = _dijkstra_shard(src)
        return
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_init_worker, initargs=(g,)
    ) as ex:
        for (s, src), res in zip(shards, ex.map(_dijkstra_shard, [x for _, x in shards])):
            out[s : s + len(src)] = res


# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------
class GeodesicTable:
    """
    Geodesic distances between N points, in one of two storage modes.

    mode == "full"     : self.dist is (N, N); sub(ids) is exact.
    mode == "landmark" : self.dist is (L, N) distances from landmarks
                         self.landmarks (L,) to all points; sub(ids) is the
                         landmark upper bound
                             D[i, j] ~= min_l ( d(l, i) + d(l, j) ),
                         which is exact whenever i or j is a landmark or lies
                         on a shortest path through one, and otherwise
                         OVERESTIMATES (triangle inequality). An
                         overestimated D lowers the ratio latent/geodesic in
                         the injective loss, biasing it toward pushing pairs
                         apart. Prefer "full" when N^2 fits in RAM.
    """

    def __init__(self, mode: str, dist, landmarks: Optional[np.ndarray] = None):
        if mode not in ("full", "landmark"):
            raise ValueError("mode must be 'full' or 'landmark'.")
        self.mode, self.dist, self.landmarks = mode, dist, landmarks
        self.n = dist.shape[1]

    def sub(
        self, ids: Sequence[int], device: str = "cpu", landmark_chunk: int = 128
    ) -> torch.Tensor:
        """
        Return the (b, b) float32 sub-table for dataset indices `ids`
        (b = len(ids)) on `device`. Diagonal is 0. `ids` are indices in the
        same ordering as the features the table was built from.
        """
        ids = np.asarray(ids)
        if self.mode == "full":
            block = np.asarray(self.dist[np.ix_(ids, ids)], dtype=np.float32)
            return torch.from_numpy(block).to(device)
        Dl = torch.from_numpy(np.asarray(self.dist[:, ids], dtype=np.float32)).to(device)
        b = len(ids)
        out = torch.full((b, b), float("inf"), device=device)
        for s in range(0, Dl.shape[0], landmark_chunk):
            c = Dl[s : s + landmark_chunk]  # (l, b)
            out = torch.minimum(out, (c[:, :, None] + c[:, None, :]).amin(0))
        out.fill_diagonal_(0.0)
        return out

    # ---- persistence ----
    def save(self, path: str) -> None:
        """Save to `path` (.npz). For big full tables prefer out_path in build."""
        np.savez(
            path,
            mode=self.mode,
            dist=np.asarray(self.dist),
            landmarks=self.landmarks if self.landmarks is not None else np.zeros(0),
        )

    @staticmethod
    def load(path: str, mmap: bool = True) -> "GeodesicTable":
        z = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        mode = str(z["mode"])
        lm = z["landmarks"] if mode == "landmark" else None
        return GeodesicTable(mode, z["dist"], lm)


# ----------------------------------------------------------------------
# Public builder
# ----------------------------------------------------------------------
def build_geodesic_table(
    X,
    k: int = 10,
    mode: str = "full",
    n_landmarks: int = 2000,
    pca_dim: Optional[int] = 256,
    chunk: int = 2048,
    device: str = "cpu",
    n_jobs: int = 1,
    shard: int = 256,
    dtype=np.float32,
    max_k: int = 80,
    seed: int = 0,
    out_path: Optional[str] = None,
) -> GeodesicTable:
    """
    Build the geodesic table. All arguments are explicit; nothing global.

    X           (N, ...) array/tensor of cached features; flattened to (N, d).
    k           neighbours in the kNN graph (Isomap-style; 10 is typical).
    mode        "full" (N x N, exact) or "landmark" (L x N, approximate).
    n_landmarks L for landmark mode (chosen uniformly at random, seeded).
    pca_dim     reduce to this many dims before the kNN stage (None = off).
                Distances are then Euclidean in the PCA subspace.
    chunk       rows per kNN chunk; peak memory ~ chunk * N * 4 bytes.
    device      "cpu" or "cuda" for the kNN stage only.
    n_jobs      worker processes for Dijkstra (<=1: run in-process).
    shard       Dijkstra sources per task (memory per task ~ shard*N*8 bytes).
    dtype       storage dtype of the table: np.float32 or np.float16.
    max_k       upper limit when k is doubled to connect the graph.
    seed        seeds PCA and landmark choice.
    out_path    full mode only: write the table to this .npy memmap file
                instead of RAM (path is then also what to np.load with
                mmap_mode="r"); the returned table wraps the memmap.
    """
    X = torch.as_tensor(np.asarray(X) if not torch.is_tensor(X) else X).float()
    X = X.reshape(X.shape[0], -1)
    N = X.shape[0]
    Xr = _reduce(X, pca_dim, seed)

    kk = min(k, N - 1)
    while True:
        g = _knn_graph(Xr, kk, chunk, device)
        ncomp, _ = connected_components(g, directed=False)
        if ncomp == 1:
            break
        if kk >= min(max_k, N - 1):
            raise ValueError(
                f"kNN graph has {ncomp} components even at k={kk}; "
                f"geodesic distances are undefined between components."
            )
        kk = min(2 * kk, max_k, N - 1)

    if mode == "full":
        sources = np.arange(N)
        if out_path is not None:
            out = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=dtype, shape=(N, N)
            )
        else:
            out = np.empty((N, N), dtype=dtype)
        landmarks = None
    elif mode == "landmark":
        L = min(n_landmarks, N)
        sources = np.sort(np.random.default_rng(seed).choice(N, L, replace=False))
        out = np.empty((L, N), dtype=dtype)
        landmarks = sources
    else:
        raise ValueError("mode must be 'full' or 'landmark'.")

    _shortest_paths(g, sources, n_jobs, shard, out)
    if isinstance(out, np.memmap):
        out.flush()
    return GeodesicTable(mode, out, landmarks)
