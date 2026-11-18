"""The exact cloud-to-mesh distance on torch tensors — float64, ONE declared device.

The twin of ``bim.comparison.compute_c2m_distances``'s numpy path, which stays the path of record
(USER 2026-11-01: "todo debe correr en GPU, en modo determinista"; 2026-11-11: "todo debe ser GPU
nada en cpu"). MEASURED 2026-11-18 on pccr 2026-08-31 against its own IFC (18.6 M points, 281
elements, 1.4 M triangles): the numpy path 8.5 minutes in 8 threads — its floor alone (4.67 M points
that lie 2 – 5 cm off a mesh that is a shell around them: ~80 triangles tried per point) over three.

What runs where:
* THE SEARCH OF THE CANDIDATES stays scipy's k-d tree on the CPU (in threads): which triangles to
  try, nearest centroid first — integers, the same on every path;
* THE EXACT DISTANCE to each candidate (Ericson's closest point on a triangle, region by region)
  and the bookkeeping of the proof (which points are done, which need a wider look) run here, on
  the device, with the SAME operations in the same order as ``_point_to_tri_batch`` — sums of three
  products left to right, one division, one correctly rounded square root (``_sqrt``: torch's own
  is not, on the CPU), no fused multiply-add across calls: the distances are the record's, bit for
  bit (tests/test_bim_scan_comparison.py — 1e-9 is the rule; equality is what is held).

Selected by ``bim.deviation.device``: numpy (the record) | cpu (this module on the CPU — the tests)
| cuda. The server never calls this on the card itself: the comparison is a child process
(bim/compare_job.py) that ends.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DEVICES = ("numpy", "cpu", "cuda")


def _torch():
    import torch
    return torch


def resolve_device(name: str) -> str:
    """'numpy' | 'cpu' | 'cuda', validated; cuda with no visible card FAILS naming the key."""
    name = str(name)
    if name not in DEVICES:
        raise ValueError(f"bim.deviation.device must be one of {list(DEVICES)}, got {name!r}")
    if name == "cuda" and not _torch().cuda.is_available():
        raise RuntimeError("bim.deviation.device is 'cuda' and torch sees no card — the comparison does not "
                           "fall back to the CPU by itself; declare cpu or numpy")
    return name


def tensor(a, device: str, dtype=None):
    torch = _torch()
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(torch.float64 if dtype is None else dtype)


def _dot(u, v):
    return u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]


def point_to_tri(P, A, B, C):
    """(N,) distances from N points to N triangles (A, B, C) — ``_point_to_tri_batch``, the same
    regions decided by the same comparisons and, where two regions meet, the same one winning (the
    record writes vertex A, B, C, then edge AB, AC, BC, then the inside: the last one written)."""
    torch = _torch()
    tiny = 1e-30
    AB, AC, AP = B - A, C - A, P - A
    d1, d2 = _dot(AB, AP), _dot(AC, AP)
    BP = P - B
    d3, d4 = _dot(AB, BP), _dot(AC, BP)
    CP = P - C
    d5, d6 = _dot(AB, CP), _dot(AC, CP)
    mask_a = (d1 <= 0) & (d2 <= 0)
    mask_b = (d3 >= 0) & (d4 <= d3)
    mask_c = (d6 >= 0) & (d5 <= d6)
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / torch.clamp(d1 - d3, min=tiny)
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / torch.clamp(d2 - d6, min=tiny)
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / torch.clamp((d4 - d3) + (d5 - d6), min=tiny)
    denom = 1.0 / torch.clamp(va + vb + vc, min=tiny)
    v_in, w_in = vb * denom, vc * denom
    closest = A + v_in[:, None] * AB + w_in[:, None] * AC          # inside the triangle, unless …
    closest = torch.where(mask_a[:, None], A, closest)
    closest = torch.where(mask_b[:, None], B, closest)
    closest = torch.where(mask_c[:, None], C, closest)
    closest = torch.where(mask_ab[:, None], A + v_ab[:, None] * AB, closest)
    closest = torch.where(mask_ac[:, None], A + w_ac[:, None] * AC, closest)
    closest = torch.where(mask_bc[:, None], B + w_bc[:, None] * (C - B), closest)
    D = P - closest
    return _sqrt(D[:, 0] * D[:, 0] + D[:, 1] * D[:, 1] + D[:, 2] * D[:, 2])


def _sqrt(x):
    """The square root, CORRECTLY ROUNDED on every device. torch's own on the CPU is not (measured
    2026-11-18, torch 2.9.1: its vectorised float64 sqrt is one unit in the last place off
    math.sqrt for 1.4 % of 200,000 values — numpy's and the card's for none): on the CPU the root
    is numpy's, over the tensor's own memory."""
    if x.device.type == "cpu":
        return _torch().from_numpy(np.sqrt(x.numpy()))
    return _torch().sqrt(x)


def nearest_in_class(P, best, A, B, C, tree, reach: float, radius, points_np: np.ndarray, first_k: int, pairs: int,
                     threads: int, blocks):
    """``bim.comparison._nearest_in_class`` with the distances on the device: ``P``, ``best``, the
    triangles and their ``radius`` are tensors there; ``points_np`` is P on the CPU, for the tree;
    ``blocks(k)`` the record's column blocks. Every triangle the record measures is measured (a
    block's are decided from the distances it began with, as there), so the minimum is the record's."""
    torch = _torch()
    F = int(A.shape[0])
    device = P.device
    todo = np.arange(len(points_np))
    k = min(first_k, F)
    while len(todo):
        step = max(1, pairs // k)
        left = []
        for s0 in range(0, len(todo), step):
            part = todo[s0:s0 + step]
            cd_np, ci_np = tree.query(points_np[part], k=k, workers=threads)
            if k == 1:
                cd_np, ci_np = cd_np[:, None], ci_np[:, None]
            part_t = torch.as_tensor(part, device=device)
            cd = torch.as_tensor(cd_np, device=device)
            ci = torch.as_tensor(ci_np, device=device)
            d = best[part_t]
            live = torch.arange(len(part), device=device)
            for j0, j1 in blocks(k):
                cdb, cib = cd[live, j0:j1], ci[live, j0:j1]
                bound = d[live][:, None]
                r, c = torch.nonzero((cdb < bound + reach) & (cdb - radius[cib] < bound), as_tuple=True)
                if r.numel():
                    f = cib[r, c]
                    rows = live[r]
                    d.scatter_reduce_(0, rows, point_to_tri(P[part_t[rows]], A[f], B[f], C[f]), reduce="amin", include_self=True)
                live = live[cd[live, j1 - 1] < d[live] + reach]
                if live.numel() == 0:
                    break
            best[part_t] = d
            if k < F and live.numel():
                left.append(part[live.cpu().numpy()])
        if k >= F or not left:
            break
        todo = np.concatenate(left)
        k = min(k * 4, F)
    return best


def distances(points: np.ndarray, tri: np.ndarray, centroids: np.ndarray, radius: np.ndarray, size_class: np.ndarray,
              class_order: np.ndarray, device: str, chunk: int, first_k: int, pairs: int, threads: int, blocks,
              progress: Optional[callable] = None) -> np.ndarray:
    """The class loop of ``compute_c2m_distances`` on ``device``: (N,) float64, on the CPU."""
    from scipy.spatial import cKDTree
    torch = _torch()
    if device == "cpu":
        torch.set_num_threads(max(1, threads))
    n = len(points)
    P = tensor(points, device)
    best = torch.full((n,), float("inf"), dtype=torch.float64, device=device)
    done = 0
    for c in class_order:
        sel = np.flatnonzero(size_class == c)
        T = tensor(tri[sel], device)
        A, B, C = T[:, 0], T[:, 1], T[:, 2]
        own = tensor(radius[sel], device)
        tree = cKDTree(centroids[sel])
        reach = float(radius[sel].max())
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            best[start:end] = nearest_in_class(P[start:end], best[start:end], A, B, C, tree, reach, own,
                                               points[start:end], first_k, pairs, threads, blocks)
            done += end - start
            if progress is not None and n > chunk:
                progress(round(done / (n * len(class_order)) * 100))
    return best.cpu().numpy()
