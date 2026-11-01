"""26-connected components of OCCUPIED voxels on the card (USER 2026-11-01: "todo debe correr en GPU").

Three places answer "which points are joined by a chain of occupied cells" on the CPU, one instance at a
time: the OBB's core points (`segmentation.pipeline._voxel_components`, a Python union-find over every
voxel adjacency — 1.3 s on a 970 k-point wall, ~5 s on pccr's floor; run for every instance of the
projection and again for the 484 OBBs the certification republishes), the co-visible split's components
(`_label_occupied_cells`, scipy's connected components on the cell graph, ~0.2 s × 818 instances on pccr
audit5 — 2 min 49 s of the projection) and the loop detector's `disjoint_clusters`. One kernel here:

* the cells: the integer coordinates packed into one int64 key with a margin ring (the numpy recipes'
  packing: lexicographic order = the order of `np.unique(key, axis=0)`), `torch.unique` sorted;
* the adjacency: for each of the 13 half-offsets the neighbour's packed key, `searchsorted` on the sorted
  cells — exactly the numpy recipes' lookup;
* the components by min-label hooking with pointer jumping (Shiloach–Vishkin): every edge hooks the larger
  root onto the smaller (`scatter_reduce amin` — integers: the result does not depend on the order of the
  atomics), labels shortcut to their roots, until no edge joins two roots. The label of a component is
  then its SMALLEST cell index — unique, so the output is deterministic and equal to the union-find's
  (`parent[max] = min`) and to scipy's labels relabelled by smallest cell.

tests/test_voxel_components_torch.py holds the three twins equal to the numpy functions label for label.
The device is the projection's evidence device (`surface_fit.evidence_device`, cpu in the tests).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

HALF_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if (dx, dy, dz) > (0, 0, 0)]


def _torch():
    import torch
    return torch


def _pack(ki, base, span):
    d = ki - base + 1
    return (d[:, 0] * span[1] + d[:, 1]) * span[2] + d[:, 2]


def components_of_cells(cells, span) -> "object":
    """Root (the smallest cell index of its component) per cell, for the SORTED packed cells ``cells``
    (M,) of a grid of extent ``span`` (3,) — 26-connectivity."""
    torch = _torch()
    m = int(cells.shape[0])
    dev = cells.device
    idx = torch.arange(m, device=dev, dtype=torch.int64)
    a_list, b_list = [], []
    for ox, oy, oz in HALF_OFFSETS:
        off = (ox * int(span[1]) + oy) * int(span[2]) + oz
        nb = cells + off
        pos = torch.searchsorted(cells, nb)
        ok = pos < m
        pos_c = torch.where(ok, pos, torch.zeros_like(pos))
        hit = ok & (cells[pos_c] == nb)
        a_list.append(idx[hit])
        b_list.append(pos_c[hit])
    a = torch.cat(a_list) if a_list else idx[:0]
    b = torch.cat(b_list) if b_list else idx[:0]
    lab = idx.clone()
    if int(a.shape[0]) == 0:
        return lab
    while True:
        la, lb = lab[a], lab[b]
        lo = torch.minimum(la, lb)
        hi = torch.maximum(la, lb)
        new = lab.clone()
        new.scatter_reduce_(0, hi, lo, reduce="amin", include_self=True)     # hook the larger root
        while True:                                                          # shortcut to the roots
            nn = new[new]
            if torch.equal(nn, new):
                break
            new = nn
        if torch.equal(new, lab):
            return lab
        lab = new


def _cells_of(ki, device) -> Tuple["object", "object", "object", bool]:
    """(sorted packed cells, inverse per point, span, degenerate): the unique cells of the integer
    coordinates ``ki`` (N, 3) with the numpy recipes' margin packing; ``degenerate`` when the packing
    would overflow an int64 (the recipes return one component then)."""
    torch = _torch()
    k = torch.as_tensor(np.ascontiguousarray(np.asarray(ki, np.int64)), device=device)
    base = k.min(0).values
    span = (k.max(0).values - base + 3)
    if float(span[0]) * float(span[1]) * float(span[2]) > 9.0e18:
        return None, None, span, True
    packed = _pack(k, base, span)
    cells, inv = torch.unique(packed, sorted=True, return_inverse=True)
    return cells, inv, span, False


def voxel_components(pts: np.ndarray, voxel_m: float, device: str) -> np.ndarray:
    """`segmentation.pipeline._voxel_components` on ``device``: per point the smallest unique-voxel index
    of its 26-connected component of occupied voxels at ``voxel_m``; all zeros for one voxel or a
    degenerate extent (the recipe's answers)."""
    key = np.floor(np.asarray(pts, np.float64) / float(voxel_m)).astype(np.int64)
    cells, inv, span, degenerate = _cells_of(key, device)
    if degenerate or int(cells.shape[0]) <= 1:
        return np.zeros(len(pts), np.int64)
    roots = components_of_cells(cells, span)
    return roots[inv].cpu().numpy()


def label_occupied_cells(ki: np.ndarray, device: str) -> Tuple[np.ndarray, int]:
    """`segmentation.pipeline._label_occupied_cells` on ``device``: (component per point, n), the labels
    0..n−1 in the order of each component's smallest cell key."""
    torch = _torch()
    cells, inv, span, degenerate = _cells_of(ki, device)
    if degenerate:
        raise OverflowError("the cell grid's extent does not pack into an int64")
    roots = components_of_cells(cells, span)
    uniq, dense = torch.unique(roots, sorted=True, return_inverse=True)
    return dense[inv].cpu().numpy(), int(uniq.shape[0])
