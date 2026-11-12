"""GenRecon's input preparation on a device (USER 2026-11-11: "todo lo que me marcaste que puede pasarse a GPU hay que
hacerlo en GPU … Nada debe hacerse en cpu"): the torch twins of adapter.py's arithmetic over the published cloud — the
points in GenRecon's frame, the levels of their heights, the active voxels of the cubes' lattice and the objects'
stands.

THE RECORD stays numpy (``genrecon.adapter_device: numpy`` — adapter.py's own lines, genrecon_levels.height_levels,
np.percentile, presence.plan_layout's per-point lines). The twins, float64 on ONE device:

* :func:`to_frame` — ``xyz @ A.T + t`` written row by row, ``((x·a0 + y·a1) + z·a2) + t``: one IEEE rounding per
  operation, so the same bits on the cpu and on the card. numpy's ``@`` is BLAS (fused multiply-adds, its own
  order): the two agree to the last bits of a float64, not bit for bit — the ONE place where the device's products
  can differ from the record's, and by that much.
* :func:`levels_of_sorted`, :func:`percentile_of_sorted`, :func:`group_percentiles` — numpy's histogram and numpy's
  linear percentile read from heights SORTED on the device: the counts are numpy's own cumulative search of the
  edges, the order statistics are exact selections and the interpolation between the two neighbours is numpy's own
  arithmetic (its private functions, on the two values). From the same heights, the record's numbers.
* :func:`active_keys` — the voxel of every point and the voxels holding at least ``min_points`` of them: from the
  same points, the record's integers. The division is a TRUE division on the card too (torch divides a tensor by a
  Python number there by multiplying with its inverse, which moves a point lying on a voxel's face to the voxel
  before it).

Measured 2026-11-12 on the real sessions, the adapter run whole by the record and on the card and every file it
writes compared: pccr 2026-08-31 (18,770,254 points, 547 keyframes, 1,424 object boxes) and pccr 2026-08-24
(114,451,841 points, 2,765 keyframes, 8,400 object boxes), both with a floor transform — cloud_genrecon.npy,
active_cells.npy, the COLMAP text files, layout.json, levels.json and objects.json BYTE-IDENTICAL (0 values differ;
two runs on the card identical too). The adapter's wall 28.0 → 21.6 s and 121.2 → 63.5 s (what is left is the
imports, the PLY / JSON reads and the text files), the card's peak 2.3 and 14.6 GB.

Stays on the CPU, declared: reading the PLY / JSON, the index arrays of the segmentation's instances, the COLMAP
text files, the symlinks, np.save, and plan_layout's arithmetic over the VOXELS (the solid ground, the cubes'
centres: thousands of voxels, not points)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

try:                                              # numpy's own percentile arithmetic (private: numpy 1.x, then 2.x)
    from numpy.lib import function_base as _npq
    _LINEAR = _npq._QuantileMethods["linear"]
    _npq._get_indexes, _npq._get_gamma, _npq._lerp
except (ImportError, AttributeError):
    from numpy.lib import _function_base_impl as _npq
    _LINEAR = _npq._QuantileMethods["linear"]
    _npq._get_indexes, _npq._get_gamma, _npq._lerp

DEVICES = ("numpy", "cpu", "cuda")
KEY = "genrecon.adapter_device"


def adapter_device(genrecon_cfg: dict) -> str:
    """``genrecon.adapter_device``, typed: where adapter.py's arithmetic over the cloud runs — "numpy" (THE RECORD),
    "cpu" or "cuda" (this module). A missing key or another value fails naming the key; "cuda" with no card FAILS —
    nothing is computed somewhere else."""
    if "adapter_device" not in (genrecon_cfg or {}):
        raise RuntimeError(f"config.yaml: {KEY} is missing ({' | '.join(DEVICES)})")
    dev = genrecon_cfg["adapter_device"]
    if dev not in DEVICES:
        raise RuntimeError(f"config.yaml: {KEY} must be one of {' | '.join(DEVICES)} (got {dev!r})")
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.yaml: {KEY} is cuda and no CUDA device is visible — nothing is computed "
                           f"somewhere else")
    return dev


def as_points(xyz, device: str) -> torch.Tensor:
    """An (N, 3) array of points as a float64 tensor on ``device`` (a tensor already there is returned as it is)."""
    if isinstance(xyz, torch.Tensor):
        return xyz.to(device=device, dtype=torch.float64)
    return torch.from_numpy(np.ascontiguousarray(xyz, np.float64)).to(device)


def to_frame(xyz: torch.Tensor, A, t) -> torch.Tensor:
    """``xyz @ A.T + t`` for (N, 3) points: every row of ``A`` as ``((x·a0 + y·a1) + z·a2) + t`` — one rounding per
    operation, the same bits on every device (module docstring)."""
    A = np.asarray(A, np.float64)
    t = np.asarray(t, np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return torch.stack([((x * float(A[r, 0]) + y * float(A[r, 1])) + z * float(A[r, 2])) + float(t[r])
                        for r in range(3)], 1)


def extent(P: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """``P.min(0), P.max(0)`` of (N, 3) points on a device, as numpy."""
    return P.amin(0).cpu().numpy(), P.amax(0).cpu().numpy()


def sorted_heights(y: torch.Tensor) -> torch.Tensor:
    """The heights in ascending order, on their device (what the levels and the percentiles below read)."""
    ys = torch.sort(y).values
    if ys.numel() and bool(torch.isnan(ys[-1])):
        raise ValueError("a height is not a number — the cloud holds a non-finite coordinate")
    return ys


def levels_of_sorted(ys: torch.Tensor, bin_m: float, min_share: float) -> List[float]:
    """genrecon_levels.height_levels of heights already SORTED on a device: the same edges (numpy's arange from the
    lowest and the highest height) and numpy's own count of them (np.histogram of explicit edges: the cumulative
    search of every edge in the sorted heights, the last edge closed)."""
    if int(ys.numel()) == 0:
        return []
    inv = 1.0 / bin_m
    y_min, y_max = np.float64(ys[0].item()), np.float64(ys[-1].item())
    edges = np.arange(np.floor(y_min * inv) / inv, y_max + bin_m, bin_m)
    if len(edges) < 2:
        return []
    e = torch.from_numpy(edges).to(ys.device)
    cum = torch.cat([torch.searchsorted(ys, e[:-1], right=False), torch.searchsorted(ys, e[-1:], right=True)])
    h = (cum[1:] - cum[:-1]).cpu().numpy()
    return [float(edges[i]) for i in range(len(h)) if h[i] / h.sum() >= min_share]


def _neighbours(n, q):
    """numpy's linear percentile for samples of ``n`` values (an int, or one int per sample): the positions of the
    two order statistics and the weight between them — numpy's own functions."""
    quantiles = np.asanyarray(np.true_divide(q, 100))
    virtual = np.asanyarray(_LINEAR["get_virtual_index"](n, quantiles))
    prev, nxt = _npq._get_indexes(np.empty(0, np.float64), virtual, n)
    return prev, nxt, _npq._get_gamma(virtual, prev, _LINEAR)


def percentile_of_sorted(ys: torch.Tensor, q):
    """``np.percentile(y, q)`` (linear) of heights already SORTED on a device; ``q`` a number or a sequence."""
    n = int(ys.numel())
    if n == 0:
        raise ValueError("no height has no percentile")
    prev, nxt, gamma = _neighbours(n, q)
    at = np.concatenate([np.atleast_1d(prev), np.atleast_1d(nxt)]).astype(np.int64)
    v = ys[torch.from_numpy(np.where(at < 0, at + n, at)).to(ys.device)].cpu().numpy()
    k = len(v) // 2
    return _npq._lerp(v[:k].reshape(np.shape(prev)), v[k:].reshape(np.shape(nxt)), gamma)


def group_percentiles(values: torch.Tensor, counts, qs: Sequence[float]) -> np.ndarray:
    """``np.percentile`` (linear) of every GROUP of ``values`` at each of ``qs``: the groups are the consecutive
    runs of ``counts`` values. (groups, len(qs)) float64. Every group is ranked on the device in two stable sorts
    (by value, then by group)."""
    counts = np.asarray(counts, np.int64)
    out = np.empty((len(counts), len(qs)), np.float64)
    if not len(counts):
        return out
    total = int(values.numel())
    if (counts <= 0).any() or int(counts.sum()) != total:
        raise ValueError(f"the groups ({int(counts.sum())} values, the smallest of {int(counts.min())}) are not "
                         f"the {total} values, each group holding one at least")
    dev = values.device
    ends = np.cumsum(counts)
    group = torch.searchsorted(torch.from_numpy(ends).to(dev), torch.arange(total, device=dev), right=True)
    by_value = torch.argsort(values, stable=True)
    ranked = values[by_value][torch.argsort(group[by_value], stable=True)]
    if bool(torch.isnan(ranked).any()):
        raise ValueError("a value is not a number — the cloud holds a non-finite coordinate")
    first = ends - counts
    for j, q in enumerate(qs):
        prev, nxt, gamma = _neighbours(counts, q)
        at = np.concatenate([np.where(prev < 0, prev + counts, prev) + first, np.where(nxt < 0, nxt + counts, nxt) + first])
        v = ranked[torch.from_numpy(at.astype(np.int64)).to(dev)].cpu().numpy()
        out[:, j] = _npq._lerp(v[:len(counts)], v[len(counts):], gamma)
    return out


def active_keys(P: torch.Tensor, origin, v: float, min_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """presence.plan_layout's per-point lines on the device: the voxel of every point of ``P`` on the lattice of
    ``origin`` and side ``v``, the lattice's dims and the keys ``(i·dims[1] + j)·dims[2] + k`` of the voxels holding
    at least ``min_points`` points, ascending. Returns (keys, dims) as numpy int64."""
    side = torch.tensor(float(v), dtype=torch.float64, device=P.device)      # a tensor: a true division on the card
    ijk = [torch.clamp(torch.floor((P[:, a] - float(origin[a])) / side).to(torch.int64), min=0) for a in range(3)]
    dims = [int(c.max()) + 1 for c in ijk]
    key = (ijk[0] * dims[1] + ijk[1]) * dims[2] + ijk[2]
    act, cnt = torch.unique(key, return_counts=True)
    return act[cnt >= int(min_points)].cpu().numpy(), np.asarray(dims, np.int64)
