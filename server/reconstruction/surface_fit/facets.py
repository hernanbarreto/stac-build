"""PLANE SEGMENTS — the object's surface as the set of planar pieces its points describe at a
resolution (USER 2026-10-16: "con planos podés reconstruir geometrías, curvaturas … una pared
curvada: planos que van describiendo la geometría, segmentos de plano, no único plano que describe
todo … por ahí perdés algo de detalle, depende qué objeto sea y el tamaño de voxelización").

Region growing over a 3-D grid of cells of size ``cell`` (the resolution):

1. every cell with enough points whose scatter across its plane is under a third of the
   tolerance (a surface, not a volume) is a candidate seed, the flattest first;
2. a region starts from a seed's plane and grows to the 26-connected neighbouring cells that
   hold a surface in that plane: a seed's worth of points (or all of a sparse cell's) within
   ``tol``, thin across it (under a third of the tolerance — a volume's cell fills the slab
   and stays out; a corner cell gives each surface its own points); after
   every ring of growth the plane is REFITTED to the region's points and the points that no
   longer fit are released — a region cannot drift along a curve: it stops where the surface
   bends more than the tolerance, and the next seed starts the next facet;
3. what no region holds at this resolution is the residue, declared.

So a flat wall is ONE segment, a curved wall a chain of facets as wide as its curvature allows
(chord ≈ 2·√(2·r·tol)), a step two parallel segments with their exact offset, a ramp its own
tilted segment. The tolerance is what the resolution cannot resolve (half a cell) — never under
the object's own measured noise. Deterministic (seeded subsample for the refits), tool_measured.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _plane_of(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(unit normal, centroid, thickness σ) of the PCA plane through ``Q``."""
    c = Q.mean(0)
    D = Q - c
    w, v = np.linalg.eigh(D.T @ D / max(len(Q), 1))
    return v[:, 0], c, float(np.sqrt(max(float(w[0]), 0.0)))


def grow_plane_segments(P: np.ndarray, cell: float, tol: float, seed_points: int = 12,
                        refit_max_points: int = 200_000, seed: int = 0
                        ) -> Tuple[np.ndarray, List[Dict]]:
    """Segment ``P`` (N, 3) into planar regions. Returns (region id per point, −1 = residue;
    the regions, largest first: {id, n, normal, centroid, rms_m, thickness_m, cells})."""
    rng = np.random.default_rng(int(seed))
    P = np.asarray(P, np.float64)
    n = len(P)
    region = np.full(n, -1, np.int64)
    if n < seed_points:
        return region, []
    cell = float(cell)
    tol = float(tol)
    o = P.min(0) - 1e-9
    C = np.floor((P - o) / cell).astype(np.int64)
    dims = C.max(0) + 1
    key = (C[:, 0] * dims[1] + C[:, 1]) * dims[2] + C[:, 2]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uniq, start, counts = np.unique(ks, return_index=True, return_counts=True)
    cell_pts: Dict[int, np.ndarray] = {int(k): order[s:s + c] for k, s, c in zip(uniq, start, counts)}
    coord = {int(k): (int(k // (dims[1] * dims[2])), int((k // dims[2]) % dims[1]), int(k % dims[2]))
             for k in uniq}
    d1, d2 = int(dims[1]), int(dims[2])

    def neighbours(k: int) -> List[int]:
        i, j, l = coord[k]
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dl in (-1, 0, 1):
                    if di == 0 and dj == 0 and dl == 0:
                        continue
                    kk = ((i + di) * d1 + (j + dj)) * d2 + (l + dl)
                    if kk in cell_pts and 0 <= j + dj < d1 and 0 <= l + dl < d2 and i + di >= 0:
                        out.append(kk)
        return out

    def fit(idx: np.ndarray):
        Q = P[idx]
        if len(Q) > int(refit_max_points):
            Q = Q[np.sort(rng.choice(len(Q), int(refit_max_points), replace=False))]
        return _plane_of(Q)

    # a cell SEEDS a plane when its points are a surface at this resolution: enough of them, and
    # their scatter across the plane under half the tolerance the facet will use — a volume (a
    # blob, a heap) scatters like the cell itself (≈ 0.29·cell every way) and seeds nothing; it
    # is residue, declared. Growth needs no such test: a sparse cell joins when its points lie
    # in a plane already known.
    seeds = []
    for k, idx in cell_pts.items():
        if len(idx) >= int(seed_points):
            _, _, th = _plane_of(P[idx])
            if th <= tol / 3.0:
                seeds.append((th, -len(idx), k))
    seeds.sort()
    regions: List[Dict] = []
    for _th, _negn, k0 in seeds:
        free0 = cell_pts[k0][region[cell_pts[k0]] < 0]
        if len(free0) < int(seed_points):
            continue
        nrm, cen, _ = _plane_of(P[free0])
        d0 = np.abs((P[free0] - cen) @ nrm)
        first = free0[d0 <= tol]
        if len(first) < int(seed_points):
            continue
        rid = len(regions)
        region[first] = rid
        members = [first]
        visited = {k0}
        frontier = neighbours(k0)
        while frontier:
            nxt: List[int] = []
            joined = False
            tested = set()
            for kk in frontier:
                if kk in visited or kk in tested:
                    continue
                tested.add(kk)
                idx = cell_pts[kk]
                free = idx[region[idx] < 0]
                if len(free) == 0:
                    continue
                ds = (P[free] - cen) @ nrm
                d = np.abs(ds)
                inl_m = d <= tol
                n_in = int(inl_m.sum())
                # the cell joins when the points it holds in the plane ARE a surface there: enough
                # of them (a seed's worth, or all of a sparse cell's) and thin across the plane —
                # under a third of the tolerance. A corner cell holding two surfaces gives each
                # its own points; a volume's cell, whose points fill the slab (spread 0.58 of the
                # tolerance), joins nothing. A cell rejected now is tested again when a later
                # ring reaches it with the refitted plane.
                if n_in >= 3 and (n_in >= int(seed_points) or n_in == len(free)) \
                        and float(np.std(ds[inl_m])) <= tol / 3.0:
                    inl = free[inl_m]
                    region[inl] = rid
                    members.append(inl)
                    visited.add(kk)
                    joined = True
                    nxt.extend(neighbours(kk))
            if not joined:
                break
            allm = np.concatenate(members)
            nrm, cen, _ = fit(allm)
            d = np.abs((P[allm] - cen) @ nrm)
            drop = allm[d > tol]
            if len(drop):
                region[drop] = -1
                allm = allm[d <= tol]
            members = [allm]
            frontier = [kk for kk in nxt if kk not in visited]
        allm = np.concatenate(members)
        if len(allm) < int(seed_points):
            region[allm] = -1
            continue
        nrm, cen, th = fit(allm)
        d = (P[allm] - cen) @ nrm
        regions.append({"id": rid, "n": int(len(allm)), "normal": nrm, "centroid": cen,
                        "rms_m": float(np.sqrt(np.mean(d ** 2))), "thickness_m": th,
                        "cells": int(len(visited))})
    regions.sort(key=lambda r: -r["n"])
    remap = np.full(max([r["id"] for r in regions], default=-1) + 1, -1, np.int64)
    for i, r in enumerate(regions):
        remap[r["id"]] = i
        r["id"] = i
    m = region >= 0
    region[m] = remap[region[m]]
    return region, regions


def object_cell(extent: np.ndarray, spacing_p99: float, max_cell: float, facets_across: int) -> float:
    """The object's own resolution: at least ``facets_across`` facets across its smallest
    extent (a pipe's section gets facets, not one slab), never under four sampling gaps (a cell
    needs points for a plane), never over the user's ``max_cell``."""
    ext = float(np.min(np.asarray(extent, np.float64)))
    return float(np.clip(ext / max(int(facets_across), 1), 4.0 * float(spacing_p99), float(max_cell)))


def cell_occupancy_median(P: np.ndarray, cell: float) -> int:
    """The median number of points an occupied cell of ``P`` holds at this resolution — what a
    facet of one cell is worth in this object's sampling. A segment with fewer points than that is
    under-sampled at the resolution (a mask leak's handful of points metres away, a sliver) and
    is declared, not meshed: pccr's pipe carried 300 leaked points up to 5 m from it, in cells
    of 12–20 points, each a facet the hole audit then bridged to (2026-10-16)."""
    P = np.asarray(P, np.float64)
    if len(P) == 0:
        return 0
    o = P.min(0) - 1e-9
    C = np.floor((P - o) / float(cell)).astype(np.int64)
    dims = C.max(0) + 1
    key = (C[:, 0] * dims[1] + C[:, 1]) * dims[2] + C[:, 2]
    _, counts = np.unique(key, return_counts=True)
    return int(np.median(counts))
