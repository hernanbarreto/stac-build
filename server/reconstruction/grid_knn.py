"""
Exact k-nearest-neighbour search on a uniform cell grid (torch, CUDA or CPU).

Shared by the cloud-cleaning SOR (``gpu_cloud_clean``) and the scene
consolidation (``surface_fit.consolidate``). It replaces two grid searches that
were NOT kNN (2026-09-28, "identical inputs must give bit-identical outputs"):

  * they looked only inside the 27 cells around a point and averaged over
    whatever they found there — a point with fewer than k neighbours in that
    box got the mean of the few it had, so the SOR statistic of an isolated
    floater was not its kNN distance (and often was +inf);
  * they kept at most 48/64 points per cell, the first ones of a NON-stable
    sort — which neighbours a dense cell contributed was an accident;
  * they tiled the cloud by the card's FREE VRAM, each tile with its own cell
    origin and a one-cell halo — the answer depended on what else was running
    on the GPU.

What this module guarantees, for the same input on the same device:

  * the k nearest points (self excluded), EXACT: the candidate block around a
    query is grown level by level (cell size doubling, ONE global origin) until
    the k-th distance is certified — every point outside the block is farther
    than the block's inscribed radius;
  * with ``radius``: the k nearest among the points within ``radius`` (fewer
    when fewer exist), every point within ``radius`` being a candidate;
  * ties broken by a fixed order (distance, then candidate order, which is the
    global point index inside a cell), never by a sort's whim;
  * the result of a query depends only on the coordinates — never on how the
    queries were blocked, on ``query_block`` / ``candidate_budget`` (memory
    BOUNDS) or on free memory. The whole cloud's sorted grid is resident on the
    device (~20 B per point); a card too small FAILS, it never answers
    differently.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

# Relative safety margin against the ulp-level rounding of floor((x-o)/h): a
# point is certified as "farther than the block" only beyond (1 - margin)·h,
# and a radius search uses cells (1 + margin) wider than the radius. Numerical
# guard, not a threshold — 1e-6 of a cell is far above float64 rounding
# (~1e-12 of a cell for any scene) and far below any geometric scale.
_MARGIN = 1e-6

_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def config_section(path: Sequence[str], keys: Sequence[str]) -> dict:
    """``keys`` of the config.yaml section at ``path``, read FRESH from disk.
    A missing section or key FAILS naming it — no default is ever assumed."""
    import yaml
    sec = yaml.safe_load(_CONFIG.read_text()) or {}
    for i, p in enumerate(path):
        if not isinstance(sec, dict) or p not in sec:
            raise KeyError(f"config.yaml: missing section {'.'.join(path[:i + 1])}")
        sec = sec[p]
    out = {}
    for k in keys:
        if not isinstance(sec, dict) or k not in sec:
            raise KeyError(f"config.yaml: missing key {'.'.join(path)}.{k}")
        out[k] = sec[k]
    return out


def resolve_device(device: Optional[str]) -> str:
    """``None`` = CUDA when present, else CPU — chosen ONCE, up front, and the
    algorithm is the same on both. An explicit ``"cuda"`` that is not there
    FAILS: a caller that asked for the card never gets a silent CPU run."""
    import torch
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    return str(device)


def _cells(p, origin, h: float):
    """Integer cell index of each point — the ONE arithmetic every caller uses
    (float64 subtraction and division are IEEE-exact-rounded on every device)."""
    import torch
    return torch.floor((p.double() - origin) / h).long()


class _Grid:
    """Every point of the cloud bucketed by cell (stable sort → inside a cell
    the points keep ascending global index)."""

    def __init__(self, pts, origin, h: float, extent: np.ndarray, chunk: int):
        import torch
        self.origin, self.h = origin, h
        n_cells = [int(np.floor(float(e) / h)) + 1 for e in extent]
        # shifted by +1 so a neighbour offset never goes negative; +3 leaves
        # room for the +1 neighbour of the last cell and one cell of slack
        self.dims = [nc + 3 for nc in n_cells]
        if self.dims[0] * self.dims[1] * self.dims[2] >= 2 ** 62:
            raise ValueError(f"grid of {self.dims} cells overflows an int64 key "
                             f"(cell {h} m) — the scene extent is not plausible")
        self.final = all(nc <= 2 for nc in n_cells)     # every block holds all
        dev = pts.device
        n = pts.shape[0]
        key = torch.empty(n, dtype=torch.int64, device=dev)
        for s in range(0, n, chunk):
            key[s:s + chunk] = self.key(_cells(pts[s:s + chunk], origin, h) + 1)
        skey, self.perm = torch.sort(key, stable=True)
        del key
        self.uniq, self.counts = torch.unique_consecutive(skey, return_counts=True)
        del skey
        self.starts = torch.cumsum(self.counts, 0) - self.counts
        dy, dz = self.dims[1], self.dims[2]
        self.offs = torch.tensor([(dx_ * dy + dy_) * dz + dz_
                                  for dx_ in (-1, 0, 1) for dy_ in (-1, 0, 1)
                                  for dz_ in (-1, 0, 1)], dtype=torch.int64, device=dev)

    def key(self, c1):
        return (c1[:, 0] * self.dims[1] + c1[:, 1]) * self.dims[2] + c1[:, 2]

    def search(self, pts, qb, k: int, r2: Optional[float], budget: int):
        """Yield (q, idx, d2) for the query block ``qb``, split into sub-blocks
        of at most ``budget`` candidate pairs (one query alone may exceed it)."""
        import torch
        qp = pts[qb]
        nkey = self.key(_cells(qp, self.origin, self.h) + 1)[:, None] + self.offs[None, :]
        pos = torch.searchsorted(self.uniq, nkey).clamp(max=len(self.uniq) - 1)
        hit = self.uniq[pos] == nkey
        cnt = torch.where(hit, self.counts[pos], torch.zeros_like(pos))
        st = self.starts[pos]
        cum = torch.cumsum(cnt.sum(1), 0).cpu().numpy()
        s, n_q = 0, len(qb)
        while s < n_q:
            base = int(cum[s - 1]) if s else 0
            e = max(int(np.searchsorted(cum, base + budget, side="right")), s + 1)
            yield self._sub(pts, qb[s:e], qp[s:e], cnt[s:e], st[s:e], k, r2)
            s = e

    def _sub(self, pts, q, qp, cnt, st, k: int, r2: Optional[float]):
        import torch
        dev = pts.device
        n_q, n_off = cnt.shape
        idx_out = torch.full((n_q, k), -1, dtype=torch.int64, device=dev)
        d2_out = torch.full((n_q, k), float("inf"), dtype=torch.float64, device=dev)
        cf, sf = cnt.reshape(-1), st.reshape(-1)
        m = int(cf.sum())
        if m == 0:
            return q, idx_out, d2_out
        pair = torch.repeat_interleave(torch.arange(len(cf), device=dev), cf)
        first = torch.cumsum(cf, 0) - cf
        gi = self.perm[sf[pair] + (torch.arange(m, device=dev) - first[pair])]
        ql = pair // n_off
        del pair, first
        # d2 = (dx·dx + dy·dy) + dz·dz in float64, elementwise — the same bits
        # whatever block the query sits in
        d2 = None
        for a in range(3):
            da = pts[gi, a].double() - qp[ql, a].double()
            d2 = da * da if d2 is None else d2 + da * da
        keep = gi != q[ql]
        if r2 is not None:
            keep &= d2 <= r2
        gi, ql, d2 = gi[keep], ql[keep], d2[keep]
        if gi.numel() == 0:
            return q, idx_out, d2_out
        # order by (query, distance, candidate order): two stable sorts
        o = torch.sort(d2, stable=True).indices
        o = o[torch.sort(ql[o], stable=True).indices]
        gi, ql, d2 = gi[o], ql[o], d2[o]
        cq = torch.bincount(ql, minlength=n_q)
        s0 = torch.cumsum(cq, 0) - cq
        j = torch.arange(k, device=dev)
        valid = j[None, :] < cq[:, None]
        p = (s0[:, None] + j[None, :]).clamp(max=gi.numel() - 1)
        return (q, torch.where(valid, gi[p], idx_out),
                torch.where(valid, d2[p], d2_out))


def grid_knn(pts, k: int, cell: Optional[float], *, radius: Optional[float] = None,
             queries=None, query_block: int, candidate_budget: int
             ) -> Iterator[Tuple["object", "object", "object"]]:
    """Exact k nearest neighbours (self excluded) of ``queries`` (default: every
    point) among ``pts`` — a (N,3) float tensor already on its device.

    Yields ``(q, idx, d2)`` device tensors: query indices (B,), neighbour
    indices (B,k) and squared distances (B,k) float64, ascending; ``-1`` /
    ``+inf`` where fewer than k exist (within ``radius`` when given, in the
    whole cloud otherwise). Every query is yielded exactly once, in no promised
    order.

    ``cell``: first-level cell size for the unbounded search — performance
    only, the answer does not depend on it. With ``radius`` the cell is the
    radius (one level, no expansion)."""
    import torch
    dev = pts.device
    n = pts.shape[0]
    if queries is None:
        queries = torch.arange(n, device=dev)
    if n == 0 or len(queries) == 0:
        return
    lo = pts.min(0).values.double()
    extent = (pts.max(0).values.double() - lo).cpu().numpy()
    if radius is not None:
        h, r2 = float(radius) * (1.0 + _MARGIN), float(radius) ** 2
    else:
        h, r2 = float(cell), None
    pending = queries
    while len(pending):
        grid = _Grid(pts, lo, h, extent, query_block)
        final = radius is not None or grid.final
        thr2 = (h * (1.0 - _MARGIN)) ** 2
        unresolved = []
        for b0 in range(0, len(pending), query_block):
            for q, idx, d2 in grid.search(pts, pending[b0:b0 + query_block], k, r2,
                                          candidate_budget):
                if final:
                    yield q, idx, d2
                    continue
                # certified: k found and the k-th is nearer than anything the
                # 27-cell block left out
                ok = (idx[:, k - 1] >= 0) & (d2[:, k - 1] <= thr2)
                if bool(ok.all()):
                    yield q, idx, d2
                    continue
                if bool(ok.any()):
                    yield q[ok], idx[ok], d2[ok]
                unresolved.append(q[~ok])
        del grid
        pending = (torch.cat(unresolved) if unresolved
                   else pending.new_empty(0))
        h *= 2.0
