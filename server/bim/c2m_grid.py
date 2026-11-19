"""The exact cloud-to-mesh distance with THE SEARCH ON THE DEVICE TOO, EVERY ELEMENT IN ONE PASS —
float64, one declared device.

USER 2026-11-19: "la comparacion debe hacerse en GPU" / "Debe funcionar al instante porque en
corridas largas va a tarda mucho, hace que todo se procese en una sola pasada". Until then the
candidates of each point came from scipy's k-d tree on the CPU (95 of the 171 s of pccr 2026-08-31)
and each element was a run of its own (280 elements: 6,012 small passes, whose fixed cost was most
of the time left).

WHAT IS COMPUTED: for every point, the distance to the nearest triangle OF ITS OWN ELEMENT
(``point_elem`` / ``tri_elem``: the element of each point and of each triangle) — the minimum over
every triangle of that element, proven, never a nearest-k guess.

HOW:
* The triangles of all the elements go in SIZE CLASSES (a factor of two in the centroid-to-vertex
  distance; ``reach`` is the class's largest).
* Each class has a hash grid of its CENTROIDS (one entry a triangle, keyed by cell AND element) at
  levels L = 0, 1, …: the cell is h = CELL_REACHES · reach · 2^L.
* A point looks at the 27 cells around its own, of its own element: every centroid within
  H = h · (1 − SLACK) of it is in them. A triangle that was NOT looked at has its centroid farther
  than H, so no point of it nearer than H − reach: once the best distance found is ≤ H − reach the
  point is done. Otherwise it is looked at once more, at the level whose H covers best + reach — or
  one level up while it has found no triangle at all. A level whose grid is at most 2 cells a side
  holds every triangle.
* A point that already has a best distance does not open the cells farther than best + reach from
  it: a centroid there is of a triangle that cannot be nearer.
* Of the candidates, only those whose bounding sphere can still be nearer (centroid distance −
  radius ≤ the smallest centroid distance, or the best so far) get their exact distance:
  ``c2m_torch.point_to_tri``, the record's operations in the record's order. The minimum of the
  same float64 values is the same value whatever the order they are found in: the distances are
  the record's, bit for bit (tests/test_bim_scan_comparison.py).

The slacks (SLACK on the cell, 1e-9 on the bounds) only make the search look at a few triangles
more: they cover the rounding of a centroid, of a radius and of a cell index, and cannot change a
distance.

THE COVERAGE's question (``within_all``: has each surface sample a point of its element within the
proximity?) is asked the same way, the samples in the grid; the points are taken in rounds and a
sample already covered leaves the grid — most are covered by the first sixteenth of the points.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from bim.c2m_torch import _sqrt, _torch, point_to_tri, tensor

CELL_REACHES = 2.0           # a level-0 cell: this many times the class's reach
SLACK = 1e-6                 # the share of a cell not counted on (rounding of the cell index)
CLASS_SHARE = 0.02           # a size class under this share of the triangles joins the next larger one
PAIR_BUDGET = 48_000_000     # point × centroid pairs held at once (their centroid distance)
EXACT_BUDGET = 8_000_000     # exact point-to-triangle distances held at once
POINT_BLOCK = 1_000_000      # points whose 27 cells are looked up at once
SUPER_BLOCK = 32_000_000     # points on the device at once (a cloud of hundreds of millions goes in turns)
MAX_CELLS_A_SIDE = 2 ** 19   # a level-0 cell is never smaller than the extent over this
COVER_ROUNDS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16))   # the points of a coverage, by index mod 16


def _offsets(device):
    torch = _torch()
    r = torch.arange(-1, 2, device=device, dtype=torch.int64)
    return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(27, 3)


class _Grid:
    """Points (the centroids of one class, or a coverage's samples) in cells of side ``h``, each
    with its element: ``order`` lists them cell by cell, ``cells`` the occupied (element, cell)
    keys ascending, ``starts`` / ``counts`` each one's run."""
    __slots__ = ("h", "origin", "dims", "per_elem", "cells", "starts", "counts", "order", "covers_all")

    def __init__(self, cent, elem, origin, hi, h: float):
        torch = _torch()
        self.h = float(h)
        self.origin = origin
        self.dims = torch.floor((hi - origin) / self.h).to(torch.int64) + 1
        self.per_elem = int(self.dims[0]) * int(self.dims[1]) * int(self.dims[2])      # cells of one element's grid
        if self.per_elem * (int(elem.max()) + 1) >= 2 ** 62:
            raise ValueError(f"c2m_grid: {int(elem.max()) + 1} grids of {self.dims.tolist()} cells have no 64-bit key")
        ijk = torch.floor((cent - origin) / self.h).to(torch.int64)
        key = ijk[:, 0] + self.dims[0] * (ijk[:, 1] + self.dims[1] * ijk[:, 2]) + self.per_elem * elem
        key, self.order = torch.sort(key)
        self.cells, self.counts = torch.unique_consecutive(key, return_counts=True)
        self.starts = torch.cumsum(self.counts, 0) - self.counts
        self.covers_all = bool((self.dims <= 2).all())

    def lookup(self, Pb, eb, offsets, limit=None):
        """For each point (of element ``eb``), its 27 cells: (m, 27) how many entries each holds
        and where its run starts. ``limit`` (m,): the cells farther than it from the point are not
        opened (inf: all of them)."""
        torch = _torch()
        q = (Pb - self.origin) / self.h
        ijk = torch.floor(q)
        frac = q - ijk
        ijk = ijk.to(torch.int64)
        nb = ijk[:, None, :] + offsets[None, :, :]
        inside = ((nb >= 0) & (nb < self.dims)).all(dim=-1)
        key = nb[..., 0] + self.dims[0] * (nb[..., 1] + self.dims[1] * nb[..., 2]) + self.per_elem * eb[:, None]
        pos = torch.searchsorted(self.cells, key.reshape(-1)).reshape(key.shape).clamp_(max=self.cells.numel() - 1)
        hit = inside & (self.cells[pos] == key)
        if limit is not None:
            # how far the point is from each of the 27 cells' boxes
            gap = torch.where(offsets[None, :, :] < 0, frac[:, None, :],
                              torch.where(offsets[None, :, :] > 0, 1.0 - frac[:, None, :], torch.zeros((), dtype=frac.dtype, device=frac.device)))
            hit &= self.h * _sqrt((gap * gap).sum(dim=-1).reshape(-1)).reshape(hit.shape) <= limit[:, None]
        return torch.where(hit, self.counts[pos], torch.zeros((), dtype=torch.int64, device=Pb.device)), self.starts[pos]


def _expand(cnt, st):
    """Every entry of the cells (cnt, st) of each row: (row, slot) — the row it belongs to and its
    place in the grid's order — or None when the cells are all empty."""
    torch = _torch()
    dev = cnt.device
    c = cnt.reshape(-1)
    nz = torch.nonzero(c, as_tuple=True)[0]
    if nz.numel() == 0:
        return None
    c = c[nz]
    ends = torch.cumsum(c, 0)
    grp = torch.repeat_interleave(torch.arange(c.numel(), device=dev), c)
    slot = st.reshape(-1)[nz][grp] + (torch.arange(int(ends[-1]), device=dev) - (ends - c)[grp])
    return torch.div(nz, 27, rounding_mode="floor")[grp], slot


def _batches(cnt):
    """The rows of ``cnt`` (m, 27) in runs of at most PAIR_BUDGET entries (a row is never split)."""
    torch = _torch()
    csum = torch.cumsum(cnt.sum(dim=1), 0)
    m, start, base = int(cnt.shape[0]), 0, 0
    while start < m:
        limit = torch.tensor([base + PAIR_BUDGET], dtype=csum.dtype, device=cnt.device)
        end = min(m, max(int(torch.searchsorted(csum, limit, right=True)), start + 1))
        yield start, end
        base, start = int(csum[end - 1]), end


def _measure(Pb, bb, cnt, st, order, A, B, C, cent, rad) -> None:
    """Lower ``bb`` (the best distance of each point of Pb, in place) with the triangles of the
    cells (cnt, st) of each point."""
    torch = _torch()
    pairs = _expand(cnt, st)
    if pairs is None:
        return
    row, slot = pairs
    f = order[slot]
    del slot, pairs
    Pp = Pb[row]
    D = Pp - cent[f]
    dc = _sqrt(D[:, 0] * D[:, 0] + D[:, 1] * D[:, 1] + D[:, 2] * D[:, 2])
    del D
    # a centroid is a point of its triangle: the smallest centroid distance bounds the answer
    ub = bb.scatter_reduce(0, row, dc, reduce="amin", include_self=True)[row]
    rf = rad[f]
    keep = torch.nonzero(dc - rf <= ub + 1e-9 * (ub + rf + 1.0), as_tuple=True)[0]
    del dc, ub, rf
    for e0 in range(0, int(keep.numel()), EXACT_BUDGET):
        k = keep[e0:e0 + EXACT_BUDGET]
        fk = f[k]
        bb.scatter_reduce_(0, row[k], point_to_tri(Pp[k], A[fk], B[fk], C[fk]), reduce="amin", include_self=True)


def _pass(P, pe, best, sel, grid: _Grid, A, B, C, cent, rad, reach: float, offsets) -> None:
    """One look of the points ``sel`` at their 27 cells of ``grid``: ``best`` lowered in place."""
    for b0 in range(0, int(sel.numel()), POINT_BLOCK):
        rows = sel[b0:b0 + POINT_BLOCK]
        Pb, bb = P[rows], best[rows]
        # (a point with no distance yet opens every cell: inf)
        cnt, st = grid.lookup(Pb, pe[rows], offsets, limit=bb + reach + 1e-9 * (bb + reach + grid.h))
        for start, end in _batches(cnt):
            _measure(Pb[start:end], bb[start:end], cnt[start:end], st[start:end], grid.order, A, B, C, cent, rad)
        best[rows] = bb


def _nearest_in_class(P, pe, best, A, B, C, cent, rad, te, reach: float, lo, hi) -> None:
    """Lower ``best`` with the triangles (A, B, C; elements ``te``) of ONE size class — exactly
    (the module's text). ``lo`` / ``hi``: a box that holds every point and every centroid."""
    torch = _torch()
    dev = P.device
    n = int(P.shape[0])
    h0 = max(CELL_REACHES * float(reach), float((hi - lo).max()) / MAX_CELLS_A_SIDE, 1e-9)
    H0 = h0 * (1.0 - SLACK)

    def needed(b):
        """The lowest level whose 27 cells prove a point whose best distance is b."""
        L = torch.ceil(torch.log2(torch.clamp((b + reach) / H0, min=1.0))).to(torch.int64)
        return L + (H0 * torch.pow(2.0, L.to(torch.float64)) < b + reach).to(torch.int64)

    level = torch.zeros(n, dtype=torch.int64, device=dev)
    known = torch.isfinite(best)
    if bool(known.any()):
        level[known] = needed(best[known])
    active = torch.ones(n, dtype=torch.bool, device=dev)
    offsets = _offsets(dev)
    grids = {}
    while True:
        idx = torch.nonzero(active, as_tuple=True)[0]
        if idx.numel() == 0:
            break
        L = int(level[idx].min())
        sel = idx[level[idx] == L]
        grid = grids.get(L)
        if grid is None:
            grid = grids[L] = _Grid(cent, te, lo, hi, h0 * 2.0 ** L)
        _pass(P, pe, best, sel, grid, A, B, C, cent, rad, reach, offsets)
        if grid.covers_all:                                       # every triangle of the class was looked at
            active[sel] = False
            continue
        b = best[sel]
        done = b + reach <= grid.h * (1.0 - SLACK)
        active[sel[done]] = False
        rest = sel[~done]
        if rest.numel():
            br = best[rest]
            found = torch.isfinite(br)
            nxt = torch.where(found, needed(torch.where(found, br, torch.zeros_like(br))), torch.full_like(rest, L + 1))
            level[rest] = torch.clamp(nxt, min=L + 1)


def _size_classes(radius: np.ndarray):
    """Each triangle's size class and the classes, the one with most triangles first."""
    size_class = np.floor(np.log2(np.maximum(radius, 1e-12))).astype(np.int64)
    classes, counts = np.unique(size_class, return_counts=True)
    for i in range(len(classes) - 1):
        if counts[i] < CLASS_SHARE * len(radius):
            size_class[size_class == classes[i]] = classes[i + 1]
            counts[i + 1] += counts[i]
            counts[i] = 0
    classes, counts = classes[counts > 0], counts[counts > 0]
    return size_class, classes[np.argsort(-counts, kind="stable")]


def distances_all(points: np.ndarray, point_elem: np.ndarray, tri: np.ndarray, tri_elem: np.ndarray, device: str,
                  progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """(N,) float64 on the CPU: the distance from each of ``points`` to the nearest of the triangles
    ``tri`` (F, 3, 3) of ITS element (``point_elem`` (N,), ``tri_elem`` (F,): integers ≥ 0). A point
    whose element has no triangle keeps inf."""
    torch = _torch()
    n = len(points)
    out = np.full(n, np.inf)
    if n == 0 or len(tri) == 0:
        return out
    points = np.ascontiguousarray(points, dtype=np.float64)
    tri = np.ascontiguousarray(tri, dtype=np.float64)
    centroids = tri.mean(axis=1)
    radius = np.linalg.norm(tri - centroids[:, None, :], axis=2).max(axis=1)
    size_class, order = _size_classes(radius)
    lo = tensor(np.minimum(points.min(axis=0), centroids.min(axis=0)), device)
    hi = tensor(np.maximum(points.max(axis=0), centroids.max(axis=0)), device)
    classes = []
    for c in order:
        sel = np.flatnonzero(size_class == c)
        T = tensor(tri[sel], device)
        classes.append((T[:, 0], T[:, 1], T[:, 2], tensor(centroids[sel], device), tensor(radius[sel], device),
                        torch.as_tensor(np.ascontiguousarray(tri_elem[sel]), device=device).to(torch.int64), float(radius[sel].max())))
    steps, step = len(classes) * -(-n // SUPER_BLOCK), 0
    for s0 in range(0, n, SUPER_BLOCK):
        s1 = min(n, s0 + SUPER_BLOCK)
        P = tensor(points[s0:s1], device)
        pe = torch.as_tensor(np.ascontiguousarray(point_elem[s0:s1]), device=device).to(torch.int64)
        best = torch.full((s1 - s0,), float("inf"), dtype=torch.float64, device=device)
        for A, B, C, cent, rad, te, reach in classes:
            _nearest_in_class(P, pe, best, A, B, C, cent, rad, te, reach, lo, hi)
            step += 1
            if progress is not None:
                progress(round(step / steps * 100))
        out[s0:s1] = best.cpu().numpy()
    return out


def within_all(points: np.ndarray, point_elem: np.ndarray, samples: np.ndarray, sample_elem: np.ndarray, radius: float,
               device: str) -> np.ndarray:
    """(S,) bool on the CPU: has each of ``samples`` a point OF ITS ELEMENT within ``radius``? — the
    coverage's question (bim.comparison.compute_coverage_pct), exactly. The samples go in a grid of
    cells of at least ``radius`` (the 27 cells around a point hold every sample within it); the
    points are taken in rounds and the samples already covered leave the grid."""
    torch = _torch()
    covered = torch.zeros(len(samples), dtype=torch.bool, device=device)
    if len(points) == 0 or len(samples) == 0:
        return covered.cpu().numpy()
    points = np.ascontiguousarray(points, dtype=np.float64)
    S = tensor(samples, device)
    se = torch.as_tensor(np.ascontiguousarray(sample_elem), device=device).to(torch.int64)
    lo = tensor(np.minimum(points.min(axis=0), np.asarray(samples).min(axis=0)), device)
    hi = tensor(np.maximum(points.max(axis=0), np.asarray(samples).max(axis=0)), device)
    h = max(float(radius) / (1.0 - SLACK), float((hi - lo).max()) / MAX_CELLS_A_SIDE, 1e-9)
    reach_of_a_cell = float(radius) + 1e-9 * (float(radius) + h)
    offsets = _offsets(S.device)
    for s0 in range(0, len(points), SUPER_BLOCK):
        s1 = min(len(points), s0 + SUPER_BLOCK)
        P = tensor(points[s0:s1], device)
        pe = torch.as_tensor(np.ascontiguousarray(point_elem[s0:s1]), device=device).to(torch.int64)
        turn = torch.arange(s1 - s0, device=P.device) % 16
        for a, b in COVER_ROUNDS:
            left = torch.nonzero(~covered, as_tuple=True)[0]      # the samples still to cover
            if left.numel() == 0:
                break
            grid = _Grid(S[left], se[left], lo, hi, h)
            rows_all = torch.nonzero((turn >= a) & (turn < b), as_tuple=True)[0]
            for b0 in range(0, int(rows_all.numel()), POINT_BLOCK):
                rows = rows_all[b0:b0 + POINT_BLOCK]
                Pb = P[rows]
                cnt, st = grid.lookup(Pb, pe[rows], offsets,
                                      limit=torch.full((int(rows.numel()),), reach_of_a_cell, dtype=torch.float64, device=P.device))
                for start, end in _batches(cnt):
                    pairs = _expand(cnt[start:end], st[start:end])
                    if pairs is None:
                        continue
                    row, slot = pairs
                    f = left[grid.order[slot]]
                    D = Pb[start:end][row] - S[f]
                    d = _sqrt(D[:, 0] * D[:, 0] + D[:, 1] * D[:, 1] + D[:, 2] * D[:, 2])
                    covered[f[d <= radius]] = True
    return covered.cpu().numpy()


# ── one element (the calls of bim.comparison's functions) ─────────────────────────────────────────
def distances(points: np.ndarray, tri: np.ndarray, device: str, progress: Optional[Callable[[int], None]] = None) -> np.ndarray:
    """``distances_all`` for one mesh."""
    return distances_all(points, np.zeros(len(points), dtype=np.int64), tri, np.zeros(len(tri), dtype=np.int64), device, progress)


def within(points: np.ndarray, samples: np.ndarray, radius: float, device: str) -> np.ndarray:
    """``within_all`` for one element."""
    return within_all(points, np.zeros(len(points), dtype=np.int64), samples, np.zeros(len(samples), dtype=np.int64), radius, device)
