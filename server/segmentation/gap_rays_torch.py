"""The ray judges of the co-visible split on the card (USER 2026-10-15: observatorio's SAM3 stage
spent hours in `_gap_rays` — a cKDTree over the 99 M-point cloud queried for every ray sample of
every component pair, single-threaded).

The SAME test, exactly: "is there a measured point within gap/2 of this ray sample". The cloud
is indexed ONCE on the declared device (`surface_fit.evidence_device`) as a voxel grid at cell =
gap/2: a dense occupancy bitmap plus a CSR of the points sorted by cell. A sample whose 3×3×3
block holds no occupied cell has no point within gap/2 (a point that close lies in one of those
cells — the exact negative); the others are checked against every point of their block, distance
for distance. Rays, samples and judges are produced in the order of the numpy recipe
(`segmentation.pipeline._gap_rays`), so the verdicts are the same; deterministic: sorts, gathers,
no scatter."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


class FreeSpaceIndex:
    """The cloud as a voxel grid at ``cell`` on one device, for exact radius tests up to ``cell``."""

    def __init__(self, xyz: np.ndarray, cell: float, device: Optional[str] = None):
        if device is None:
            from reconstruction.surface_fit.hole_audit import evidence_device
            device = evidence_device()
        self.device = str(device)
        dev = self.device
        self.cell = float(cell)
        P = torch.as_tensor(np.ascontiguousarray(np.asarray(xyz, np.float32)), device=dev)
        self.P = P
        self.origin = P.min(dim=0).values.to(torch.float64) - 2.0 * self.cell    # a margin ring
        c = torch.floor((P.to(torch.float64) - self.origin) / self.cell).to(torch.int64)
        self.dims = (c.max(dim=0).values + 3).tolist()                               # a margin ring
        self.strides = (int(self.dims[1]) * int(self.dims[2]), int(self.dims[2]), 1)
        lin = c[:, 0] * self.strides[0] + c[:, 1] * self.strides[1] + c[:, 2]
        self.order = torch.sort(lin, stable=True).indices
        lin_s = lin[self.order]
        self.cells, self.counts = torch.unique_consecutive(lin_s, return_counts=True)
        self.starts = torch.searchsorted(lin_s, self.cells)
        # NO dense bitmap over the bbox: one stray point 146 m away made it billions of cells
        # (pccr 2026-10-15); occupancy is a searchsorted hit on the sorted occupied cells
        self.n_cells = int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])
        off = torch.arange(-1, 2, device=dev)
        dz, dy, dx = torch.meshgrid(off, off, off, indexing="ij")
        self.block = (dz.reshape(-1) * self.strides[0] + dy.reshape(-1) * self.strides[1] + dx.reshape(-1))

    def _cell_of(self, S: torch.Tensor) -> torch.Tensor:
        c = torch.floor((S.to(torch.float64) - self.origin) / self.cell).to(torch.int64)
        inside = ((c >= 0) & (c < torch.as_tensor(self.dims, device=S.device))).all(dim=1)
        c = torch.clamp(c, min=torch.zeros(3, dtype=torch.int64, device=S.device),
                        max=torch.as_tensor(self.dims, device=S.device) - 1)
        return c[:, 0] * self.strides[0] + c[:, 1] * self.strides[1] + c[:, 2], inside

    def any_within(self, S: np.ndarray, r: float) -> np.ndarray:
        """bool (M,): a cloud point within ``r`` (≤ cell) of each sample — exact."""
        if float(r) > self.cell + 1e-12:
            raise ValueError(f"radius {r} exceeds the index cell {self.cell}")
        S_t = torch.as_tensor(np.ascontiguousarray(np.asarray(S, np.float64)), device=self.device)
        M = int(len(S_t))
        out = torch.zeros(M, dtype=torch.bool, device=self.device)
        if M == 0:
            return out.cpu().numpy()
        lin, inside = self._cell_of(S_t)
        # the margin ring keeps every neighbour id of an inside cell inside the grid
        nb = lin[:, None] + self.block[None, :]                                        # (M, 27)
        nb = torch.clamp(nb, 0, self.n_cells - 1)
        pos_all = torch.clamp(torch.searchsorted(self.cells, nb), max=len(self.cells) - 1)   # (M, 27)
        hit_all = self.cells[pos_all] == nb
        maybe = inside & hit_all.any(dim=1)
        idx_m = maybe.nonzero()[:, 0]
        if len(idx_m) == 0:
            return out.cpu().numpy()
        pos = pos_all[idx_m]
        hit = hit_all[idx_m]
        st = self.starts[pos]
        ct = torch.where(hit, self.counts[pos], torch.zeros_like(pos))
        K = int(ct.max())
        kk = torch.arange(K, device=self.device)
        r2 = float(r) * float(r)
        step = max(1, int(2e7 // max(27 * K, 1)))
        res = torch.zeros(len(idx_m), dtype=torch.bool, device=self.device)
        Pf = self.P.to(torch.float64)
        for a in range(0, len(idx_m), step):
            sl = slice(a, a + step)
            idx = st[sl][:, :, None] + kk[None, None, :]                                 # (v, 27, K)
            valid = kk[None, None, :] < ct[sl][:, :, None]
            pidx = self.order[torch.clamp(idx, max=len(self.order) - 1)]
            diff = Pf[pidx] - S_t[idx_m[sl]][:, None, None, :]
            d2 = (diff * diff).sum(dim=-1)
            d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
            res[sl] = (d2.reshape(d2.shape[0], -1) <= r2).any(dim=1)
        out[idx_m] = res
        return out.cpu().numpy()


def closest_pair(A: np.ndarray, B: np.ndarray, tree_a=None, tree_b=None):
    """The closest points (a ∈ A, b ∈ B): the SMALLER set queries the other's tree (the pair is
    the same either way; the trees may come cached from the caller — one per component instead
    of one per pair, the CPU cost that was left in the split)."""
    from scipy.spatial import cKDTree
    if len(B) <= len(A):
        tree = tree_a if tree_a is not None else cKDTree(A)
        d, ia = tree.query(B, k=1)
        jb = int(np.argmin(d))
        return A[ia[jb]], B[jb]
    tree = tree_b if tree_b is not None else cKDTree(B)
    d, ib = tree.query(A, k=1)
    ja = int(np.argmin(d))
    return A[ja], B[ib[ja]]


def gap_rays_t(A: np.ndarray, B: np.ndarray, index: FreeSpaceIndex, cams: np.ndarray,
               gap_m: float, reach_m: float, tree_a=None, tree_b=None) -> List[int]:
    """:func:`segmentation.pipeline._gap_rays` on the index: the same samples, rays, steps and
    judge order; the radius tests exact on the device."""
    gap_m = float(gap_m)
    a, b = closest_pair(A, B, tree_a, tree_b)
    n = int(np.floor(np.linalg.norm(b - a) / gap_m))
    if n < 2 or len(cams) == 0:
        return []
    samples = a + (b - a) * (np.linspace(1, n - 1, min(n - 1, 20))[:, None] / n)
    steps = np.arange(1, max(int(reach_m / gap_m), 1) + 1) * gap_m
    cams = np.asarray(cams, np.float64)
    nC, nS = len(cams), len(samples)
    rays = samples[None, :, :] - cams[:, None, :]                       # (C, S, 3)
    L = np.linalg.norm(rays, axis=2)                                     # (C, S)
    ok = L > gap_m
    dirs = rays / np.maximum(L, 1e-12)[:, :, None]
    # the FRONT samples of every ray at once: C + dir · t for t in (gap, L − gap) step gap,
    # padded to the longest ray and masked
    n_front = np.maximum(np.ceil((L - 2.0 * gap_m) / gap_m), 0).astype(np.int64)     # len(arange(gap, L-gap, gap))
    n_front = np.where(ok, n_front, 0)
    Kf = int(n_front.max()) if n_front.size else 0
    blocked = np.zeros((nC, nS), dtype=bool)
    if Kf > 0:
        t = (np.arange(Kf) + 1.0) * gap_m                                            # (Kf,)
        pts = cams[:, None, None, :] + dirs[:, :, None, :] * t[None, None, :, None]  # (C, S, Kf, 3)
        valid = (np.arange(Kf)[None, None, :] < n_front[:, :, None])
        flat = pts[valid]
        within = index.any_within(flat, gap_m / 2.0)
        acc = np.zeros(valid.shape, dtype=bool)
        acc[valid] = within
        blocked = acc.any(axis=2)
    # the BEHIND samples: sample + dir · steps
    pts_b = samples[None, :, None, :] + dirs[:, :, None, :] * steps[None, None, :, None]   # (C, S, T, 3)
    hit = index.any_within(pts_b.reshape(-1, 3), gap_m / 2.0).reshape(nC, nS, -1).any(axis=2)
    out: List[int] = []
    for ci in range(nC):
        for si in range(nS):
            if not ok[ci, si] or blocked[ci, si]:
                continue
            out.append(int(hit[ci, si]))
    return out
