"""The chat's measurements on the card (USER 2026-11-13: "el vlm y todas sus herramientas deben
funcionar en GPU"; the chat "tarda muchisimo en contestar con la placa a disposición").

MEASURED 2026-11-13 on pccr 2026-08-31 (18.6 M points in 1,360 instances), the numpy path:
the clearance between the floor (1,062,834 points) and a wall (743,629) did not end in 8 minutes
of scipy's k-d tree on one core; floor against a work surface (59,366 points) 14.5 s; two
objects of 59 k points 4.8 s. The other measurements (bands, extent, plumb, level, height
profile, flatness, the volumes) are 0.01 – 1 s each once the points are in memory.

Every point-based kernel of ``phase5_qa.tools.SpatialTools`` here, on torch tensors in float64
on ONE device, the numpy code in tools.py KEPT as the path of record:

* ``closest_pair`` — the EXACT minimum distance between two point sets and its pair. Small
  problems are measured outright (every pair, in blocks). Large ones go down an implicit octree
  of each set (Morton order; the cells of a level are runs of the sorted points, each with the
  tight box of its own points): a pair of cells is kept only while the distance between its two
  boxes — a rigorous lower bound of every point pair inside, also in floating point: every
  operation of it is monotone — does not exceed an upper bound that is the distance of a REAL
  pair of points (strided samples of the most promising cells). The pairs of cells that survive
  the finest level are measured outright. Nothing is approximated: the pair that realises the
  minimum is never pruned (its bound cannot exceed its own distance), so the distance is the
  k-d tree's. Squared distances are dx² + dy² + dz² in that order (scipy's). Among pairs at
  EXACTLY the same distance the first point of the first set wins, then the first of the second
  (the record: numpy's first argmin over the first set; within the second set the tree breaks
  an exact tie its own way — the one place the pair, never the distance, can differ);
* the surface bands, extents and extreme points; the principal axes (QR of the centred points,
  then the SVD of its 3 × 3 factor: the right singular vectors of the points themselves);
  numpy's median and linear percentile, reproduced on sorted values; the height profile's
  stations; the oriented-box coordinates of the volumes, their occupancy voxels, the per-object
  counts and the summed-area search of a free placement (the same first origin in i, j, k
  order as the record's three nested loops).

Selected by ``semantic.chat.tools_device``: numpy (the path of record) | cpu (this module on the
CPU — the tests) | cuda. tests/test_chat_tools_torch.py holds the two paths equal: exact for
counts, indices and verdicts, 1e-9 for geometry. What stays on the CPU, declared: sqlite (the
store's reads), the Python that formats an answer, and the scalar trigonometry of one tilt.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np

DEVICES = ("numpy", "cpu", "cuda")
KEY = "semantic.chat.tools_device"

# bounds on memory and work — none of them changes a result (the search is exact)
PAIR_BUDGET = 24_000_000      # pair distances held at once
ENUM_TOTAL = 200_000_000      # point pairs measured outright; above it the cells are refined first
MAX_LEVEL = 10                # the finest cells: 2**10 along the longest side of a set's own box
START_LEVEL = 3               # the first level compared cell against cell (at most 512 × 512)
PROBE_PAIRS = 256             # pairs of cells sampled for the upper bound at every level
PROBE_SAMPLES = 128           # strided points of a cell in one sample


def _torch():
    import torch
    return torch


def resolve_device(name: str) -> str:
    """'numpy' | 'cpu' | 'cuda', validated; a cuda with no visible card FAILS naming the key —
    nothing runs somewhere else by itself."""
    name = str(name)
    if name not in DEVICES:
        raise ValueError(f"{KEY} must be one of {list(DEVICES)}, got {name!r}")
    if name == "cuda" and not _torch().cuda.is_available():
        raise RuntimeError(f"{KEY} is 'cuda' and torch sees no card — the chat's tools do not "
                           f"fall back to the CPU by themselves; declare cpu or numpy")
    return name


def tensor(a: np.ndarray, device: str):
    """(n, 3) points as a float64 tensor on the device."""
    return _torch().as_tensor(np.ascontiguousarray(a, dtype=np.float64), device=device)


def release(device: str) -> None:
    """Give the card's cached blocks back when a question ends (the process keeps only its
    context): the next stage finds the memory free."""
    if device == "cuda":
        torch = _torch()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


# ── the closest pair of two point sets ──────────────────────────────────────────────────────
def _cross_sqdist(a, B):
    """(r, n) squared distances of r points against n, dx² + dy² + dz² in that order."""
    d = a[:, None, 0] - B[None, :, 0]
    s = d * d
    d = a[:, None, 1] - B[None, :, 1]
    s = s + d * d
    d = a[:, None, 2] - B[None, :, 2]
    return s + d * d


def _pair_sqdist(a, b):
    """(m,) squared distances of paired points, dx² + dy² + dz² in that order."""
    d = a - b
    return (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]


class _Best:
    """The smallest squared distance seen and, among the pairs at exactly it, the first point
    of the first set, then the first of the second (original indices)."""

    def __init__(self, n_b: int):
        self.d2 = float("inf")
        self.code = -1
        self.n_b = int(n_b)

    def offer(self, d2, oa, ob) -> None:
        if d2.numel() == 0:
            return
        m = d2.min()
        mv = float(m)
        if mv > self.d2:
            return
        sel = d2 == m
        code = int((oa[sel] * self.n_b + ob[sel]).min())
        if mv < self.d2 or self.code < 0 or code < self.code:
            self.d2, self.code = mv, code

    def result(self) -> Tuple[float, int, int]:
        return math.sqrt(self.d2), self.code // self.n_b, self.code % self.n_b


def _brute(A, B, best: _Best, pair_budget: int) -> None:
    torch = _torch()
    n_a, n_b = int(A.shape[0]), int(B.shape[0])
    rows = max(1, pair_budget // max(n_b, 1))
    cols = torch.arange(n_b, device=A.device)
    for r0 in range(0, n_a, rows):
        d2 = _cross_sqdist(A[r0:r0 + rows], B)
        m = d2.min()
        if float(m) > best.d2:
            continue
        rr, cc = torch.nonzero(d2 == m, as_tuple=True)
        best.offer(d2[rr, cc], rr + r0, cols[cc])


def _spread3(v):
    """The bits of v (at most 21) spread to every third position."""
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


class _Level:
    __slots__ = ("keys", "starts", "counts", "lo", "hi")

    def __init__(self, keys, starts, counts, lo, hi):
        self.keys, self.starts, self.counts, self.lo, self.hi = keys, starts, counts, lo, hi


class _Tree:
    """A point set sorted along the Morton curve of a grid of 2**MAX_LEVEL cells per side over
    its own box: the cells of every coarser level are runs of the sorted points."""

    def __init__(self, P):
        torch = _torch()
        lo = P.min(0).values
        ext = float((P.max(0).values - lo).max())
        side = 1 << MAX_LEVEL
        if ext > 0.0:
            c = torch.clamp(torch.floor((P - lo) / (ext / side)).to(torch.int64), 0, side - 1)
        else:
            c = torch.zeros(P.shape, dtype=torch.int64, device=P.device)
        m = (_spread3(c[:, 0]) << 2) | (_spread3(c[:, 1]) << 1) | _spread3(c[:, 2])
        self.order = torch.argsort(m, stable=True)
        self.m = m[self.order]
        self.P = P[self.order]

    def level(self, lev: int) -> _Level:
        torch = _torch()
        key = self.m >> (3 * (MAX_LEVEL - lev))
        keys, counts = torch.unique_consecutive(key, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        seg = torch.repeat_interleave(torch.arange(len(keys), device=key.device), counts)
        idx = seg[:, None].expand(-1, 3)
        lo = torch.full((len(keys), 3), float("inf"), dtype=self.P.dtype, device=key.device)
        hi = torch.full((len(keys), 3), float("-inf"), dtype=self.P.dtype, device=key.device)
        lo = lo.scatter_reduce(0, idx, self.P, "amin", include_self=True)
        hi = hi.scatter_reduce(0, idx, self.P, "amax", include_self=True)
        return _Level(keys, starts, counts, lo, hi)


def _box_sqdist(la: _Level, lb: _Level, i, j):
    """The squared distance between the boxes of cell i of one set and cell j of the other: no
    pair of their points is closer (in floating point too — every step is monotone)."""
    torch = _torch()
    g = torch.maximum(la.lo[i] - lb.hi[j], lb.lo[j] - la.hi[i]).clamp_min(0.0)
    return (g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1]) + g[:, 2] * g[:, 2]


def _chunks(cnt, budget: int) -> Iterator[Tuple[int, int]]:
    """Consecutive runs of rows whose counts add up to at most ``budget`` (a larger row alone)."""
    c = _torch().cumsum(cnt, 0).cpu().numpy()
    n, s, base = len(c), 0, 0
    while s < n:
        e = int(np.searchsorted(c, base + budget, side="right"))
        if e <= s:
            e = s + 1
        yield s, e
        base = int(c[e - 1])
        s = e


def _expand(start_a, cnt_a, start_b, cnt_b):
    """Every (a, b) with a in [start_a, start_a + cnt_a) and b in [start_b, start_b + cnt_b),
    row after row, a-major."""
    torch = _torch()
    cnt = cnt_a * cnt_b
    tot = int(cnt.sum())
    dev = cnt.device
    rep = torch.repeat_interleave(torch.arange(len(cnt), device=dev), cnt)
    offs = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(tot, device=dev) - offs[rep]
    return start_a[rep] + local // cnt_b[rep], start_b[rep] + local % cnt_b[rep]


def _probe(ta: _Tree, tb: _Tree, la: _Level, lb: _Level, i, j, samples: int) -> float:
    """The smallest squared distance among strided samples of the given pairs of cells — the
    distance of a real pair of points, hence an upper bound of the minimum."""
    torch = _torch()
    ar = torch.arange(samples, device=i.device)

    def sample(level: _Level, cells):
        n = level.counts[cells][:, None]
        pos = torch.where(n > samples, (ar[None, :] * n) // samples,
                          torch.minimum(ar[None, :], n - 1))
        return level.starts[cells][:, None] + pos

    pa = ta.P[sample(la, i)]                                 # (k, S, 3)
    pb = tb.P[sample(lb, j)]
    d = pa[:, :, None, 0] - pb[:, None, :, 0]
    s = d * d
    d = pa[:, :, None, 1] - pb[:, None, :, 1]
    s = s + d * d
    d = pa[:, :, None, 2] - pb[:, None, :, 2]
    s = s + d * d
    return float(s.min())


def _children(next_keys, parent_keys):
    torch = _torch()
    lo = torch.searchsorted(next_keys, parent_keys << 3)
    hi = torch.searchsorted(next_keys, (parent_keys << 3) + 8)
    return lo, hi - lo


def _enumerate(ta: _Tree, tb: _Tree, la: _Level, lb: _Level, i, j, best: _Best,
               pair_budget: int) -> None:
    """Every point pair of the given pairs of cells, in blocks of at most ``pair_budget``."""
    torch = _torch()
    sa, na = la.starts[i], la.counts[i]
    sb, nb = lb.starts[j], lb.counts[j]
    # a pair of cells larger than the budget is cut along its first set
    rows_per = torch.clamp(pair_budget // torch.clamp(nb, min=1), min=1)
    parts = (na + rows_per - 1) // rows_per
    if int(parts.max()) > 1:
        rep = torch.repeat_interleave(torch.arange(len(parts), device=parts.device), parts)
        k = torch.arange(int(parts.sum()), device=parts.device) - (torch.cumsum(parts, 0) - parts)[rep]
        sa = sa[rep] + k * rows_per[rep]
        na = torch.minimum(rows_per[rep], la.counts[i][rep] - k * rows_per[rep])
        sb, nb = sb[rep], nb[rep]
    cnt = na * nb
    for s, e in _chunks(cnt, pair_budget):
        a, b = _expand(sa[s:e], na[s:e], sb[s:e], nb[s:e])
        best.offer(_pair_sqdist(ta.P[a], tb.P[b]), ta.order[a], tb.order[b])


def closest_pair(A, B, *, enum_total: int = ENUM_TOTAL, pair_budget: int = PAIR_BUDGET,
                 start_level: int = START_LEVEL, probe_pairs: int = PROBE_PAIRS,
                 probe_samples: int = PROBE_SAMPLES) -> Tuple[float, int, int]:
    """(distance, index in A, index in B) of the closest pair of two (n, 3) float64 tensors on
    one device — exact (module docstring). The bounds only move work between the two phases."""
    torch = _torch()
    n_a, n_b = int(A.shape[0]), int(B.shape[0])
    if n_a == 0 or n_b == 0:
        raise ValueError("closest_pair needs two non-empty point sets")
    best = _Best(n_b)
    if n_a * n_b <= enum_total:
        _brute(A, B, best, pair_budget)
        return best.result()
    ta, tb = _Tree(A), _Tree(B)
    lev = min(max(int(start_level), 0), MAX_LEVEL)
    la, lb = ta.level(lev), tb.level(lev)
    ka, kb = len(la.keys), len(lb.keys)
    dev = A.device
    i = torch.arange(ka, device=dev).repeat_interleave(kb)
    j = torch.arange(kb, device=dev).repeat(ka)
    low = _box_sqdist(la, lb, i, j)
    ub2 = float("inf")
    while True:
        top = torch.topk(low, min(int(probe_pairs), int(low.numel())), largest=False).indices
        ub2 = min(ub2, _probe(ta, tb, la, lb, i[top], j[top], int(probe_samples)))
        keep = low <= ub2
        i, j, low = i[keep], j[keep], low[keep]
        total = int((la.counts[i] * lb.counts[j]).sum())
        if total <= enum_total or lev >= MAX_LEVEL:
            break
        lev += 1
        na, nb = ta.level(lev), tb.level(lev)
        ca_s, ca_n = _children(na.keys, la.keys[i])
        cb_s, cb_n = _children(nb.keys, lb.keys[j])
        ni: List = []
        nj: List = []
        nl: List = []
        for s, e in _chunks(ca_n * cb_n, pair_budget):
            a, b = _expand(ca_s[s:e], ca_n[s:e], cb_s[s:e], cb_n[s:e])
            v = _box_sqdist(na, nb, a, b)
            kk = v <= ub2
            ni.append(a[kk])
            nj.append(b[kk])
            nl.append(v[kk])
        i, j, low = torch.cat(ni), torch.cat(nj), torch.cat(nl)
        la, lb = na, nb
    _enumerate(ta, tb, la, lb, i, j, best, pair_budget)
    return best.result()


# ── bands, extents, axes ────────────────────────────────────────────────────────────────────
def band_stats(P) -> dict:
    """What the record reads off an object's points: min / max corner, centroid, the centres of
    its lowest and highest surface bands (5 % of the height, at least 5 cm) and its extreme
    points — numpy arrays."""
    torch = _torch()
    y = P[:, 1]
    mn, mx = P.min(0).values, P.max(0).values
    y_min, y_max = float(mn[1]), float(mx[1])
    band = max(0.05, 0.05 * (y_max - y_min))
    return {"min": mn.cpu().numpy(), "max": mx.cpu().numpy(), "band": band,
            "centroid": P.mean(0).cpu().numpy(),
            "top": P[y >= y_max - band].mean(0).cpu().numpy(),
            "bottom": P[y <= y_min + band].mean(0).cpu().numpy(),
            "highest": P[int(torch.argmax(y))].cpu().numpy(),
            "lowest": P[int(torch.argmin(y))].cpu().numpy()}


def principal_axes(P) -> Tuple[np.ndarray, np.ndarray]:
    """(centroid, vt) — vt's rows are the right singular vectors of the centred points, largest
    singular value first (numpy.linalg.svd's vt up to the sign of a row): the QR of the centred
    points and the SVD of its 3 × 3 triangular factor."""
    torch = _torch()
    c = P.mean(0)
    r = torch.linalg.qr(P - c, mode="r").R
    vt = torch.linalg.svd(r).Vh
    return c.cpu().numpy(), vt.cpu().numpy()


def median(v) -> float:
    """numpy.median of a 1-D tensor: the middle value, or the mean of the two middle ones."""
    torch = _torch()
    s = torch.sort(v).values
    n = int(s.numel())
    if n % 2:
        return float(s[n // 2])
    return float(np.mean([float(s[n // 2 - 1]), float(s[n // 2])]))


def percentile(v, q: float) -> float:
    """numpy.percentile(v, q) with its default linear interpolation, on sorted values: the same
    virtual index and the same two-sided lerp."""
    torch = _torch()
    s = torch.sort(v).values
    n = int(s.numel())
    quant = np.true_divide(q, 100)
    virtual = n * quant + (1 + quant * (1 - 1 - 1)) - 1          # alpha = beta = 1
    prev = int(np.floor(virtual))
    gamma = float(virtual - prev)
    prev = min(max(prev, 0), n - 1)
    nxt = min(prev + 1, n - 1)
    a, b = float(s[prev]), float(s[nxt])
    diff = b - a
    out = a + diff * gamma
    if gamma >= 0.5:
        out = b - diff * (1 - gamma)
    if gamma == 0.0:
        out = a
    return float(out)


def plane_residuals_mm(P) -> dict:
    """The least-squares plane's residuals in millimetres: rms, 95th percentile, maximum."""
    torch = _torch()
    c, vt = principal_axes(P)
    n = vt[2]
    n = n / (np.linalg.norm(n) + 1e-12)
    d0 = -float(n @ c)
    res = torch.abs(P @ torch.as_tensor(n, device=P.device) + d0) * 1000.0
    return {"rms": float(torch.sqrt((res * res).mean())), "p95": percentile(res, 95),
            "max": float(res.max())}


# ── oriented boxes (the object's own box, the user's volumes) ───────────────────────────────
def local_coords(P, T: np.ndarray):
    """World points in the frame of a rigid 4 × 4 (phase_r.geometry.obb_local_coords)."""
    torch = _torch()
    from phase_r.geometry import invert_rigid
    ti = torch.as_tensor(np.asarray(invert_rigid(np.asarray(T, float)), np.float64), device=P.device)
    return P @ ti[:3, :3].T + ti[:3, 3]


def height_profile(P, T: np.ndarray, aabb: np.ndarray, n_stations: int,
                   floor=None) -> Tuple[int, float, float, list]:
    """The record's stations along the object's longest horizontal box axis: per station
    (count, y_min, y_max, floor clearance or None)."""
    torch = _torch()
    ext_x, ext_z = aabb[1] - aabb[0], aabb[5] - aabb[4]
    axis_i = 0 if ext_x >= ext_z else 2
    local = local_coords(P, T)
    s_lo, s_hi = (aabb[0], aabb[1]) if axis_i == 0 else (aabb[4], aabb[5])
    span = max(float(s_hi - s_lo), 1e-9)
    station_of = torch.clamp(((local[:, axis_i] - float(s_lo)) / span * n_stations).to(torch.int64),
                             0, n_stations - 1)
    rows = []
    for k in range(n_stations):
        sel = station_of == k
        cnt = int(sel.sum())
        if cnt < 10:
            rows.append((cnt, None, None, None))
            continue
        pts = P[sel]
        ys = pts[:, 1]
        y_min, y_max = float(ys.min()), float(ys.max())
        clearance = None
        if floor is not None and int(floor.shape[0]) >= 30:
            xz = pts[:, [0, 2]].mean(0)
            d = floor[:, [0, 2]] - xz
            near = floor[torch.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])
                         < max(span / n_stations, 0.5)]
            if int(near.shape[0]) >= 10:
                clearance = y_min - median(near[:, 1])
        rows.append((cnt, y_min, y_max, clearance))
    return axis_i, float(s_lo), span, rows


def box_clearance(P, T: np.ndarray, half: np.ndarray) -> dict:
    """An object's points against a box: the nearest point (first of the nearest), its
    counterpart on the box, its distance, how many points lie inside, the centroid."""
    torch = _torch()
    local = local_coords(P, T)
    h = torch.as_tensor(np.asarray(half, np.float64), device=P.device)
    outside = torch.clamp(torch.abs(local) - h[None, :], min=0.0)
    d = torch.sqrt((outside[:, 0] * outside[:, 0] + outside[:, 1] * outside[:, 1])
                   + outside[:, 2] * outside[:, 2])
    j = int(torch.argmin(d))
    return {"j": j, "distance": float(d[j]), "n_inside": int((d <= 1e-9).sum()),
            "local_j": local[j].cpu().numpy(), "point": P[j].cpu().numpy(),
            "centroid": P.mean(0).cpu().numpy()}


def inside_mask(P, T: np.ndarray, half: np.ndarray):
    torch = _torch()
    if int(P.shape[0]) == 0:
        return torch.zeros(0, dtype=torch.bool, device=P.device)
    local = local_coords(P, T)
    h = torch.as_tensor(np.asarray(half, np.float64), device=P.device)
    return torch.all(torch.abs(local) <= h[None, :] + 1e-9, dim=1)


def voxel_keys(P_inside, T: np.ndarray, half: np.ndarray, voxel_m: float, n_vox: np.ndarray):
    """(m, 3) integer voxel of every inside point in the box's own grid."""
    torch = _torch()
    h = torch.as_tensor(np.asarray(half, np.float64), device=P_inside.device)
    local = local_coords(P_inside, T) + h
    keys = torch.floor(local / voxel_m).to(torch.int64)
    hi = torch.as_tensor(np.asarray(n_vox, np.int64) - 1, device=P_inside.device)
    return torch.minimum(torch.clamp(keys, min=0), hi[None, :])


def occupied_voxels(keys, n_vox: np.ndarray) -> int:
    torch = _torch()
    ny, nz = int(n_vox[1]), int(n_vox[2])
    return int(torch.unique((keys[:, 0] * ny + keys[:, 1]) * nz + keys[:, 2]).numel())


def counts_by_owner(owners) -> Tuple[np.ndarray, np.ndarray]:
    """(ids ascending, counts) — numpy.unique(owners, return_counts=True)."""
    torch = _torch()
    ids, cnt = torch.unique(owners, return_counts=True)
    return ids.cpu().numpy(), cnt.cpu().numpy()


def free_placement(keys, n_vox: np.ndarray, need: np.ndarray) -> Tuple[Optional[tuple], int]:
    """(first origin in i, j, k order of a ``need``-sized block with no occupied voxel, or None;
    the number of free voxels) — the record's summed-area table and three nested loops."""
    torch = _torch()
    nx, ny, nz = (int(v) for v in n_vox)
    dx, dy, dz = (int(v) for v in need)
    dev = keys.device
    occ = torch.zeros((nx, ny, nz), dtype=torch.bool, device=dev)
    if int(keys.shape[0]):
        # written at UNIQUE flat indices: a deterministic index_put
        occ.view(-1)[torch.unique((keys[:, 0] * ny + keys[:, 1]) * nz + keys[:, 2])] = True
    free = (~occ).to(torch.int64)
    n_free = int(free.sum())
    ii = free.cumsum(0).cumsum(1).cumsum(2)
    p = torch.zeros((nx + 1, ny + 1, nz + 1), dtype=torch.int64, device=dev)
    p[1:, 1:, 1:] = ii
    tot = (p[dx:, dy:, dz:] - p[:nx + 1 - dx, dy:, dz:] - p[dx:, :ny + 1 - dy, dz:]
           - p[dx:, dy:, :nz + 1 - dz] + p[:nx + 1 - dx, :ny + 1 - dy, dz:]
           + p[:nx + 1 - dx, dy:, :nz + 1 - dz] + p[dx:, :ny + 1 - dy, :nz + 1 - dz]
           - p[:nx + 1 - dx, :ny + 1 - dy, :nz + 1 - dz])
    ok = (tot == dx * dy * dz).flatten()
    if not bool(ok.any()):
        return None, n_free
    first = int(torch.argmax(ok.to(torch.uint8)))
    sj, sk = ny - dy + 1, nz - dz + 1
    return (first // (sj * sk), (first // sk) % sj, first % sk), n_free
