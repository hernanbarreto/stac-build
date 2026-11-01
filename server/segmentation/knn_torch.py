"""Exact bounded neighbour queries on a hashed voxel grid, on the device (USER 2026-11-01: "todo debe
correr en GPU, nada debe quedar en CPU").

The projection asked scipy's cKDTree, on the CPU, for the neighbours of millions of points within a
fixed radius: the attach step votes among the ``votes`` nearest labelled points within ``attach_dist_m``
of every unsegmented point (pccr audit5: 20 M queries, 38 s of tree queries plus the Python rebuild).
Here the reference set is sorted by its cell on a grid whose cell IS the radius (one int64 key per cell
with a margin ring; CSR: first point and count of every occupied cell), and a query reads the 27 cells
around its own: every point within the radius lies in those cells, so the answers are EXACT.

* ``radius_pairs`` — every (query, point, squared distance) within the radius, in chunks of queries
  whose candidate pairs fit ``pair_budget``;
* ``knn_within_radius`` — the k nearest within the radius per query (distances ascending, -1 / +inf
  where fewer), the k smallest taken by a stable sort by (query, distance): ties by the fixed order the
  candidates are generated in (offset by offset, cell by cell, points in cell order) — the one place a
  cKDTree answer can differ (it breaks exact distance ties its own way; the DISTANCES agree).

Squared distances are the sum of the three squared coordinate differences in x, y, z order, in float64
(what scipy's tree computes before its square root). Deterministic: sorts, gathers, integer counts.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch

OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
PAIR_BUDGET = 24_000_000          # candidate pairs held at once (a bound on memory, not a decision)


class GridIndex:
    """The reference points ``P`` (m, 3) float64 sorted by cell at ``cell``: CSR over the occupied cells."""

    def __init__(self, P: torch.Tensor, cell: float):
        self.P = P
        self.cell = float(cell)
        self.device = P.device
        lo = P.min(0).values
        self.origin = lo - 2.0 * self.cell                               # a margin ring
        c = torch.floor((P - self.origin) / self.cell).to(torch.int64)
        self.dims = (c.max(0).values + 3).tolist()
        self.strides = (int(self.dims[1]) * int(self.dims[2]), int(self.dims[2]), 1)
        key = c[:, 0] * self.strides[0] + c[:, 1] * self.strides[1] + c[:, 2]
        self.order = torch.argsort(key, stable=True)
        self.cells, self.counts = torch.unique_consecutive(key[self.order], return_counts=True)
        self.starts = torch.cumsum(self.counts, 0) - self.counts

    def cell_coords(self, Q: torch.Tensor) -> torch.Tensor:
        return torch.floor((Q - self.origin) / self.cell).to(torch.int64)

    def lookup(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(start, count) of the cell at integer ``coords`` (n, 3); count 0 off the grid or empty."""
        d0, d1, d2 = (int(x) for x in self.dims)
        valid = ((coords[:, 0] >= 0) & (coords[:, 0] < d0) & (coords[:, 1] >= 0) & (coords[:, 1] < d1)
                 & (coords[:, 2] >= 0) & (coords[:, 2] < d2))
        key = coords[:, 0] * self.strides[0] + coords[:, 1] * self.strides[1] + coords[:, 2]
        m = int(self.cells.shape[0])
        pos = torch.searchsorted(self.cells, key)
        pos_c = torch.clamp(pos, max=max(m - 1, 0))
        hit = valid & (pos < m) & (self.cells[pos_c] == key)
        start = torch.where(hit, self.starts[pos_c], torch.zeros_like(pos_c))
        count = torch.where(hit, self.counts[pos_c], torch.zeros_like(pos_c))
        return start, count

    def block_counts(self, Q: torch.Tensor) -> torch.Tensor:
        """Candidate points in the 27 cells around every query (n,)."""
        cq = self.cell_coords(Q)
        tot = torch.zeros(int(Q.shape[0]), dtype=torch.int64, device=self.device)
        for off in OFFSETS:
            _, cnt = self.lookup(cq + torch.tensor(off, device=self.device, dtype=torch.int64))
            tot += cnt
        return tot

    def pairs(self, Q: torch.Tensor, off: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """(query index, reference index) of every reference point in the cell at ``off`` from each
        query's cell — queries in order, the cell's points in cell order."""
        cq = self.cell_coords(Q) + torch.tensor(off, device=self.device, dtype=torch.int64)
        start, count = self.lookup(cq)
        tot = int(count.sum())
        if tot == 0:
            e = torch.zeros(0, dtype=torch.int64, device=self.device)
            return e, e
        q = torch.repeat_interleave(torch.arange(int(Q.shape[0]), device=self.device), count)
        first = torch.cumsum(count, 0) - count
        local = torch.arange(tot, device=self.device) - first[q]
        p = self.order[start[q] + local]
        return q, p


def _sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    d = A - B
    return d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]


def _chunks(tot: torch.Tensor, pair_budget: int) -> Iterator[Tuple[int, int]]:
    """[lo, hi) query ranges whose candidate counts ``tot`` sum to at most ``pair_budget`` (at least one
    query per chunk)."""
    n = int(tot.shape[0])
    cum = torch.cumsum(tot, 0)
    lo = 0
    while lo < n:
        base = int(cum[lo - 1]) if lo > 0 else 0
        hi = int(torch.searchsorted(cum, torch.tensor(base + int(pair_budget), device=tot.device, dtype=cum.dtype)))
        hi = min(max(hi, lo + 1), n)
        yield lo, hi
        lo = hi


def radius_pairs(P: torch.Tensor, Q: torch.Tensor, radius: float, pair_budget: int = PAIR_BUDGET
                 ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(query index, reference index, squared distance) of every pair within ``radius`` (inclusive), by
    chunks of queries. float64."""
    P = P.to(torch.float64)
    Q = Q.to(torch.float64)
    n = int(Q.shape[0])
    if n == 0 or int(P.shape[0]) == 0:
        return
    g = GridIndex(P, float(radius))
    r2 = float(radius) * float(radius)
    tot = g.block_counts(Q)
    for lo, hi in _chunks(tot, pair_budget):
        Qc = Q[lo:hi]
        qs, ps, ds = [], [], []
        for off in OFFSETS:
            q, p = g.pairs(Qc, off)
            if int(q.shape[0]) == 0:
                continue
            d = _sqdist(Qc[q], P[p])
            keep = d <= r2
            qs.append(q[keep] + lo)
            ps.append(p[keep])
            ds.append(d[keep])
        if qs:
            yield torch.cat(qs), torch.cat(ps), torch.cat(ds)


def _segment_topk(q: torch.Tensor, d: torch.Tensor, i: torch.Tensor, n: int, k: int
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The k smallest ``d`` per query ``q`` (0..n-1) as (n, k) distances and indices, ascending, +inf and
    -1 where a query has fewer; ties by their order in the input (stable sorts)."""
    dev = q.device
    o = torch.argsort(d, stable=True)
    q, d, i = q[o], d[o], i[o]
    o = torch.argsort(q, stable=True)
    q, d, i = q[o], d[o], i[o]
    first = torch.searchsorted(q, torch.arange(n, device=dev))
    rank = torch.arange(int(q.shape[0]), device=dev) - first[q]
    keep = rank < k
    D = torch.full((n, k), float("inf"), dtype=d.dtype, device=dev)
    I = torch.full((n, k), -1, dtype=torch.int64, device=dev)
    D[q[keep], rank[keep]] = d[keep]
    I[q[keep], rank[keep]] = i[keep]
    return D, I


def knn_within_radius(P: torch.Tensor, Q: torch.Tensor, k: int, radius: float,
                      pair_budget: int = PAIR_BUDGET) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``k`` nearest reference points ``P`` (m, 3) within ``radius`` (inclusive) of every query ``Q``
    (n, 3), exact: (distances (n, k) ascending, indices (n, k)); +inf / -1 where fewer. float64."""
    P = P.to(torch.float64)
    Q = Q.to(torch.float64)
    n = int(Q.shape[0])
    dev = Q.device
    D = torch.full((n, int(k)), float("inf"), dtype=torch.float64, device=dev)
    I = torch.full((n, int(k)), -1, dtype=torch.int64, device=dev)
    if n == 0 or int(P.shape[0]) == 0 or int(k) <= 0:
        return D, I
    g = GridIndex(P, float(radius))
    r2 = float(radius) * float(radius)
    tot = g.block_counts(Q)
    for lo, hi in _chunks(tot, pair_budget):
        Qc = Q[lo:hi]
        qs, ps, ds = [], [], []
        for off in OFFSETS:
            q, p = g.pairs(Qc, off)
            if int(q.shape[0]) == 0:
                continue
            d = _sqdist(Qc[q], P[p])
            keep = d <= r2
            qs.append(q[keep])
            ps.append(p[keep])
            ds.append(d[keep])
        if not qs:
            continue
        Dc, Ic = _segment_topk(torch.cat(qs), torch.cat(ds), torch.cat(ps), hi - lo, int(k))
        D[lo:hi] = Dc
        I[lo:hi] = Ic
    return torch.sqrt(D), I


def knn_within_radius_numpy(P: np.ndarray, Q: np.ndarray, k: int, radius: float, device: str
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`knn_within_radius` for numpy arrays."""
    Pt = torch.as_tensor(np.ascontiguousarray(np.asarray(P, np.float64)), device=device)
    Qt = torch.as_tensor(np.ascontiguousarray(np.asarray(Q, np.float64)), device=device)
    D, I = knn_within_radius(Pt, Qt, k, radius)
    return D.cpu().numpy(), I.cpu().numpy()
