"""The floor as a per-keyframe constraint — place against moment.

USER 2026-09-19: *"si podemos detectar rampas y escalones de verdad, y
discriminarlos para no romperlos, podemos hacer un concenso para ajustar
siempre el piso"*.

THE DISCRIMINATOR. Real terrain is a property of the PLACE; a pose error is a
property of the MOMENT. The same cell of floor measured from two keyframes far
apart in the walk gives the same height if the difference is a ramp or a step,
and different heights if the difference is drift. So terrain and error separate
without ever assuming the floor is flat.

Measured on pccr epoch 0, 6,852 cells of 10 cm over 5.7 M floor points:

    separation between the keyframes    cells    spread of that cell's height
    that see the SAME cell
    0.0 - 0.5 m                         1,875    median   18.3 mm
    0.5 - 2.0 m                         3,003    median   44.3 mm
    2.0 - 5.0 m                           633    median   81.1 mm
    5.0 - 20.0 m                        1,129    median  295.6 mm

Terrain cannot do that: a ramp does not disperse 18 mm when looked at half a
metre apart and 296 mm when looked at fifteen metres apart. The growth IS the
pose error, measured directly and without a single duplicated object. The
18.3 mm of the first row is the floor's own measurement noise, over 1,875 cases.

THE MODEL. Every observation of a cell from a keyframe is

    h[cell, kf] = T[cell] + dz[kf] + noise

``T`` is the terrain — ramps and steps live here and survive untouched — and
``dz`` is the vertical pose error of that keyframe, which is what the
correction removes. Solved by alternating medians, so a patch of furniture
misread as floor moves nothing.

WHAT IT CANNOT DO. Only the vertical component: a drift along the floor is
invisible to the floor. Horizontal still needs the objects. And the global
level is free by construction — ``dz`` is centred, and the display levelling
decides where zero is.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class FloorObservations:
    """One height per (cell, keyframe): what that moment saw of that place."""

    cell: np.ndarray                  # cell index per observation
    kf: np.ndarray                    # keyframe per observation
    h: np.ndarray                     # height along `up`, metres
    n: np.ndarray                     # points behind each observation
    n_cells: int
    cell_m: float

    def as_dict(self) -> dict:
        return {"observations": int(len(self.h)), "cells": int(self.n_cells),
                "keyframes": int(len(np.unique(self.kf))),
                "cell_m": self.cell_m, "provenance": "tool_measured"}


def floor_index(output_dir, n_points: int,
                log: Callable[[str], None] = lambda m: None) -> np.ndarray:
    """The cloud indices of the session's floor — the same set
    :func:`floor_points` returns, so any per-point field can be taken with it."""
    output_dir = Path(output_dir)
    seg = json.loads((output_dir / "segmentation_result.json").read_text())
    inst = seg.get("instances") or []
    label = None
    fl = output_dir / "floor_level.json"
    if fl.exists():
        try:
            sel = int(json.loads(fl.read_text())["selected_instance_id"])
            label = next((str(i["label"]) for i in inst
                          if int(i["instance_id"]) == sel), None)
        except (KeyError, ValueError, TypeError):
            label = None
    if label is None:
        cand = [i for i in inst if "floor" in str(i.get("label", "")).lower()]
        if not cand:
            raise RuntimeError("no floor instance in the segmentation")
        label = str(max(cand, key=lambda i: len(i.get("globalIndices") or []))["label"])
    gi = np.concatenate([np.asarray(i.get("globalIndices") or [], np.int64)
                         for i in inst if str(i.get("label")) == label]
                        or [np.zeros(0, np.int64)])
    return gi[(gi >= 0) & (gi < int(n_points))]


def floor_points(output_dir, xyz: np.ndarray, ks_of_point: np.ndarray,
                 log: Callable[[str], None] = print
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(points, keyframe) of the session's floor.

    The floor is the instance the levelling already chose (``floor_level.json``)
    plus every other instance carrying the same label — the floor of a real
    session is segmented in pieces, and all of them measure the same surface.
    """
    output_dir = Path(output_dir)
    seg = json.loads((output_dir / "segmentation_result.json").read_text())
    inst = seg.get("instances") or []
    label = None
    fl = output_dir / "floor_level.json"
    if fl.exists():
        try:
            sel = int(json.loads(fl.read_text())["selected_instance_id"])
            label = next((str(i["label"]) for i in inst
                          if int(i["instance_id"]) == sel), None)
        except (KeyError, ValueError, TypeError):
            label = None
    if label is None:
        cand = [i for i in inst if "floor" in str(i.get("label", "")).lower()]
        if not cand:
            raise RuntimeError("no floor instance in the segmentation — the "
                               "floor cannot be used as a constraint")
        label = str(max(cand, key=lambda i: len(i.get("globalIndices") or []))["label"])
    gi = np.concatenate([np.asarray(i.get("globalIndices") or [], np.int64)
                         for i in inst if str(i.get("label")) == label]
                        or [np.zeros(0, np.int64)])
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    k = ks_of_point[gi]
    ok = k >= 0
    log(f"[floor] {label}: {int(ok.sum()):,} points over "
        f"{len(np.unique(k[ok]))} keyframe(s)")
    return xyz[gi[ok]], k[ok]


def observe(P: np.ndarray, kf: np.ndarray, up: np.ndarray, cell_m: float,
            min_points_per_obs: int = 1) -> FloorObservations:
    """Collapse the floor into one height per (cell, keyframe).

    The cell grid is horizontal — perpendicular to ``up`` — so a ramp is a
    smooth change of ``T`` across cells and a step is a sharp one, and neither
    is confused with a keyframe moving.
    """
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    a, b, h = P @ e1, P @ e2, P @ up
    ia = np.floor((a - a.min()) / cell_m).astype(np.int64)
    ib = np.floor((b - b.min()) / cell_m).astype(np.int64)
    cell = ia * (int(ib.max()) + 1) + ib

    order = np.lexsort((kf, cell))
    c_s, k_s, h_s = cell[order], kf[order], h[order]
    new = np.ones(len(c_s), bool)
    new[1:] = (c_s[1:] != c_s[:-1]) | (k_s[1:] != k_s[:-1])
    start = np.flatnonzero(new)
    end = np.append(start[1:], len(c_s))
    med = np.empty(len(start))
    for i, (s, e) in enumerate(zip(start, end)):
        med[i] = np.median(h_s[s:e])
    cnt = (end - start).astype(np.int64)
    keep = cnt >= int(min_points_per_obs)
    return FloorObservations(cell=c_s[start][keep], kf=k_s[start][keep],
                             h=med[keep], n=cnt[keep],
                             n_cells=int(len(np.unique(cell))), cell_m=float(cell_m))


# REMOVED 2026-09-19, USER: *"eliminalo, ese paso ya no hace falta verdad?"*.
# The per-keyframe version of this — solve `h = T[cell] + dz[kf]` by alternating
# medians and move every keyframe by its own `dz` — worked on the floor and
# BROKE the objects: on pccr it pushed the desk's first visit down 32.8 mm and
# its second up 30.0 mm, opening a closed object by 62.8 mm, all of it
# vertical. The floor and the objects constrain the same keyframes, and
# applying one after the other always breaks the second.
# The CHUNK version below does the same job where it is well conditioned —
# seven unknowns instead of 216 — and leaves the objects alone.


# ── THE PER-KEYFRAME OFFSET — a MEASUREMENT, never applied on its own ────
#
# This came back on 2026-09-19 after the joint system, holding only the six
# chunk edges, closed the desk and broke the floor anyway: the floor's own
# dispersion went 66.4 -> 123.2 mm at long separations while every chunk edge
# stayed exact. Six differences between seven chunk centres are too coarse —
# the desk's correction acts on keyframes 0-12 and 200-215 and no constraint
# lived there.
#
# So the per-keyframe offset is measured again — but ONLY as a measurement.
# Applied on its own it pushed the desk's two visits apart by 62.8 mm
# (2026-09-19, the epoch that had to be deleted). Inside the joint system it
# cannot: the objects are solved at the same time, over the same unknowns.


def keyframe_offsets(obs: FloorObservations, n_kf: int, max_iters: int = 30,
                     tol_m: float = 1e-5, log: Callable[[str], None] = print
                     ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """``h[cell, kf] = T[cell] + dz[kf]``, by alternating medians.

    Returns ``(dz, constrained, report)``. Medians, not means: a patch of
    furniture misread as floor, or one keyframe with a bad depth, moves
    nothing. Only cells seen from TWO OR MORE keyframes separate the two — a
    cell seen once explains itself with its own terrain and says nothing about
    any moment. The global level is free and centred out; the display levelling
    decides the datum.
    """
    cells, inv = np.unique(obs.cell, return_inverse=True)
    n_per_cell = np.bincount(inv, minlength=len(cells))
    usable = n_per_cell[inv] >= 2
    if not usable.any():
        raise RuntimeError("no floor cell is seen from two keyframes — the "
                           "floor cannot separate terrain from pose error")
    inv_u, kf_u, h_u = inv[usable], obs.kf[usable], obs.h[usable]
    dz = np.zeros(int(n_kf))
    T = np.zeros(len(cells))
    seen_kf = np.unique(kf_u)
    it = 0
    for it in range(int(max_iters)):
        prev = dz.copy()
        r = h_u - dz[kf_u]
        for c in range(len(cells)):
            m = inv_u == c
            if m.any():
                T[c] = np.median(r[m])
        r = h_u - T[inv_u]
        for k in seen_kf:
            dz[k] = np.median(r[kf_u == k])
        dz -= np.median(dz[seen_kf])
        if np.max(np.abs(dz - prev)) <= tol_m:
            break
    constrained = np.zeros(int(n_kf), bool)
    constrained[seen_kf] = True
    resid = h_u - T[inv_u] - dz[kf_u]
    rep = {"iterations": it + 1, "cells_usable": int(len(np.unique(obs.cell[usable]))),
           "observations_usable": int(usable.sum()),
           "keyframes_constrained": int(constrained.sum()),
           "dz_span_mm": round(float(dz[seen_kf].max() - dz[seen_kf].min()) * 1000, 2),
           "terrain_span_mm": round(float(T.max() - T.min()) * 1000, 2),
           "residual_mm": {"median": round(float(np.median(np.abs(resid))) * 1000, 2),
                           "p90": round(float(np.percentile(np.abs(resid), 90)) * 1000, 2)},
           "provenance": "tool_measured"}
    log(f"[floor] per keyframe: {rep['cells_usable']:,} cell(s), "
        f"{rep['observations_usable']:,} observation(s), "
        f"{rep['keyframes_constrained']} keyframe(s); dz span "
        f"{rep['dz_span_mm']:.1f} mm, terrain span {rep['terrain_span_mm']:.1f} mm, "
        f"residual median {rep['residual_mm']['median']:.1f} mm")
    return dz, constrained, rep


# ── THE CHUNK LOOP CLOSURE — step 0 of the correction ────────────────────
#
# USER 2026-09-19, after seeing it work: *"guarda el codigo de correccion de
# piso que aplicaste porque funciona muy bien, y debe ser parte del pipeline …
# yo creo que antes practicamente de empezar la correccion, porque ya te deja
# muy bien acotado los errores"*.
#
# The SAME discriminator as above — place against moment — but between CHUNKS
# instead of keyframes, which is where it is best conditioned: seven unknowns
# against thousands of shared cells.
#
# WHAT NOT TO DO, measured the hard way. Levelling every chunk's floor to y=0
# by its OWN average height moves each chunk by a number that is mostly the
# building's real relief: pccr asked for 306 to 632 mm, and since consecutive
# chunks overlap by half their keyframes, the shared area ended up split into
# two heights — vertical duplicates CREATED where there were none, and every
# object crossing a seam stretched.
#
# What works is to compare ONLY the cells both chunks saw:
#
#     chunk pair   shared cells   height difference
#     0 <-> 6          1,118         +247.8 mm      <- the walk closing on itself
#     1 <-> 2            381          +50.9 mm
#     2 <-> 3            541          +22.1 mm
#     3 <-> 4             61           +8.1 mm
#     4 <-> 5             53           -8.0 mm
#     5 <-> 6            288          -12.0 mm
#
# Consecutive chunks differ by 8 to 51 mm; the start and the end of the walk,
# which see the same floor because the walk returns, differ by 248 mm. That is
# the accumulated vertical drift, and the relief never enters the measurement.
#
# The offsets are applied with a linear blend across each seam, over the
# chunk plan's OWN overlap — so nothing is torn: pccr's largest step between
# neighbouring keyframes came out at 6.2 mm.
#
# USER's verdict on the result: *"el piso quedo nivelado perfecto en altura, y
# se respeto la rampa que tenia"*.


def chunk_of_keyframe(chunk_ranges, n_kf: int) -> np.ndarray:
    """Which chunk each keyframe belongs to — the one whose centre is nearest.

    The plan's chunks overlap by half, so a keyframe belongs to two of them; a
    correction has to pick one or the same points would move twice.
    """
    cen = np.array([(a + b) / 2.0 for a, b in chunk_ranges], float)
    return np.array([int(np.argmin(np.abs(cen - k))) for k in range(int(n_kf))])


def chunk_edges(obs: FloorObservations, own_of_kf: np.ndarray, n_chunks: int,
                min_shared_cells: int) -> List[Tuple[int, int, float, int]]:
    """(i, j, height difference, shared cells) for every pair of chunks that
    saw the same floor. The difference is a MEDIAN over the shared cells, so
    the relief cancels: both chunks are looking at the same places."""
    oc = own_of_kf[obs.kf]
    per: List[Dict[int, List[float]]] = [{} for _ in range(n_chunks)]
    for c, h, o in zip(obs.cell, obs.h, oc):
        per[o].setdefault(int(c), []).append(float(h))
    med = [{c: float(np.median(v)) for c, v in d.items()} for d in per]
    out = []
    for i in range(n_chunks):
        for j in range(i + 1, n_chunks):
            common = set(med[i]) & set(med[j])
            if len(common) < int(min_shared_cells):
                continue
            d = np.array([med[j][c] - med[i][c] for c in common])
            out.append((i, j, float(np.median(d)), len(common)))
    return out


def solve_chunk_offsets(edges, n_chunks: int) -> Tuple[np.ndarray, dict]:
    """Per-chunk vertical offset from the pairwise differences.

    ``c_i - c_j = d_ij`` for every measured pair, plus a zero-mean gauge so the
    scene is not lifted as a whole. Least squares weighted by shared cells: a
    pair measured on a thousand cells outweighs one measured on fifty, and
    where the graph is a tree — pccr's is — every edge is satisfied exactly.
    """
    if not edges:
        raise RuntimeError("no two chunks share floor cells — the chunks "
                           "cannot be levelled against each other")
    A = np.zeros((len(edges) + 1, n_chunks))
    y = np.zeros(len(edges) + 1)
    w = np.zeros(len(edges) + 1)
    for r, (i, j, d, n) in enumerate(edges):
        A[r, i], A[r, j], y[r], w[r] = 1.0, -1.0, d, np.sqrt(n)
    A[-1, :], y[-1], w[-1] = 1.0, 0.0, np.sqrt(sum(e[3] for e in edges))
    c = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    resid = [{"pair": [i, j], "measured_mm": round(d * 1000, 2),
              "solved_mm": round(float(c[i] - c[j]) * 1000, 2),
              "residual_mm": round(float(d - (c[i] - c[j])) * 1000, 2),
              "shared_cells": int(n)} for i, j, d, n in edges]
    return c, {"offsets_mm": [round(float(x) * 1000, 2) for x in c],
               "edges": resid, "provenance": "tool_measured"}


def measure_chunks(output_dir, xyz: np.ndarray, ks_of_point: np.ndarray,
                   poses: np.ndarray, chunk_ranges, cell_m: float,
                   min_shared_cells: int, blend_kf: int,
                   log: Callable[[str], None] = print
                   ) -> Tuple[np.ndarray, dict]:
    """The per-keyframe vertical correction that levels the chunks against
    each other, blended across the seams so nothing is torn."""
    up = -poses[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    n_kf = len(poses)
    P, kf = floor_points(output_dir, xyz, ks_of_point, log=log)
    obs = observe(P, kf, up, cell_m)
    own = chunk_of_keyframe(chunk_ranges, n_kf)
    edges = chunk_edges(obs, own, len(chunk_ranges), min_shared_cells)
    for i, j, d, n in edges:
        log(f"[chunk-floor] {i} <-> {j}: {d * 1000:+.1f} mm over {n} shared cell(s)")
    c, rep = solve_chunk_offsets(edges, len(chunk_ranges))

    dz = -c[own]
    # blend across each seam over the plan's own overlap, so no object that
    # crosses a boundary is cut in two
    starts = [int(np.flatnonzero(own == i).min()) for i in range(len(chunk_ranges))]
    half = max(int(blend_kf), 0)
    for k0 in starts[1:]:
        lo, hi = max(k0 - half, 0), min(k0 + half, n_kf - 1)
        if hi <= lo:
            continue
        a, b = dz[lo], dz[hi]
        for k in range(lo, hi + 1):
            dz[k] = a + (b - a) * (k - lo) / (hi - lo)
    t_kf = np.outer(dz, up)
    rep.update({"chunks": len(chunk_ranges), "blend_kf": half,
                "max_correction_mm": round(float(np.abs(dz).max()) * 1000, 2),
                "max_step_between_keyframes_mm": round(
                    float(np.abs(np.diff(dz)).max()) * 1000, 2) if n_kf > 1 else 0.0,
                "observations": obs.as_dict(),
                "up": [round(float(x), 6) for x in up]})
    log(f"[chunk-floor] {len(chunk_ranges)} chunk(s), {len(edges)} measured "
        f"pair(s); max correction {rep['max_correction_mm']:.1f} mm, step "
        f"between keyframes {rep['max_step_between_keyframes_mm']:.2f} mm")
    return t_kf, rep


# ── THE FLOOR'S TEXTURE — the horizontal the geometry cannot see ─────────
#
# USER 2026-09-19, looking at the joint epoch: *"hay desacuerdo pero entre las
# lineas del piso"*.
#
# Everything above is vertical by construction: a drift ALONG a floor moves no
# point off the floor, so the geometry of the floor says nothing about it. What
# does say something is what is PAINTED on it — the tile joints are a ruler.
#
# Measured on pccr epoch 8, between the first stretch of the walk (kf 0-18) and
# the last (kf 195-215), over 46,462 shared cells of 1 cm:
#
#     window +-10 cm    -9.0 x -1.0 cm   peak 0.301   (clipped by the window)
#     window +-20 cm   -15.0 x -1.0 cm   peak 0.303
#     window +-30 cm   -15.0 x -1.0 cm   peak 0.303
#     window +-40 cm   -15.0 x -1.0 cm   peak 0.303
#
# 15 cm of residual horizontal offset, the same answer at every window — that
# is what "the lines do not line up" is.
#
# THE PERIODICITY. A tiled floor repeats, so its correlation repeats with it: a
# shift of exactly one period matches as well as no shift at all. That is why
# the same measurement over the UNCORRECTED cloud was worthless (peaks 0.283
# and 0.274, a ratio of 0.97). It works now because the residual is one tenth
# of the period, and the search is bounded well under half of it. The period is
# measured, not assumed, and reported with the answer.
#
# The correlation peak is low (0.303): a floor is a weak correlation target.
# The weight carries it, so a pair that did not lock cannot outvote one that
# did.


def _intensity_raster(P: np.ndarray, I: np.ndarray, ax: np.ndarray,
                      ay: np.ndarray, lo: np.ndarray, shape, cell_m: float):
    u = ((P @ ax - lo[0]) / cell_m).astype(np.int64)
    v = ((P @ ay - lo[1]) / cell_m).astype(np.int64)
    ok = (u >= 0) & (u < shape[0]) & (v >= 0) & (v < shape[1])
    s = np.zeros(shape)
    c = np.zeros(shape)
    np.add.at(s, (u[ok], v[ok]), I[ok])
    np.add.at(c, (u[ok], v[ok]), 1.0)
    return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


def tile_period_m(g: np.ndarray, cell_m: float, max_m: float = 2.0):
    """The repeat of whatever is painted on the floor, per axis, or None."""
    f = np.where(np.isfinite(g), g, np.nanmean(g))
    f = (f - f.mean()) / (f.std() + 1e-9)
    ac = np.fft.irfft2(np.abs(np.fft.rfft2(f)) ** 2, s=f.shape) / f.size
    out = []
    n = int(max_m / cell_m)
    for prof in (ac[:, 0][:n], ac[0, :][:n]):
        pk = [i for i in range(5, len(prof) - 1)
              if prof[i] > prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.03]
        out.append(round(pk[0] * cell_m, 3) if pk else None)
    return out


def chunk_texture_edges(P: np.ndarray, I: np.ndarray, kf: np.ndarray,
                        up: np.ndarray, own_of_kf: np.ndarray, n_chunks: int,
                        cell_m: float, window_m: float, min_shared_cells: int,
                        log: Callable[[str], None] = print) -> List[dict]:
    """Horizontal offset between every pair of chunks that share floor, from
    what is painted on it. Returns one dict per pair that locked."""
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    oc = own_of_kf[kf]
    W = int(round(window_m / cell_m))
    out: List[dict] = []
    for i in range(n_chunks):
        for j in range(i + 1, n_chunks):
            mi, mj = oc == i, oc == j
            if mi.sum() < 1000 or mj.sum() < 1000:
                continue
            a = np.concatenate([P[mi] @ e1, P[mj] @ e1])
            b = np.concatenate([P[mi] @ e2, P[mj] @ e2])
            lo = np.array([a.min(), b.min()])
            shape = tuple(((np.array([a.max(), b.max()]) - lo) / cell_m).astype(int) + 1)
            if min(shape) < 4 * W or max(shape) > 4000:
                continue
            g1 = _intensity_raster(P[mi], I[mi], e1, e2, lo, shape, cell_m)
            g2 = _intensity_raster(P[mj], I[mj], e1, e2, lo, shape, cell_m)
            both = np.isfinite(g1) & np.isfinite(g2)
            if int(both.sum()) < int(min_shared_cells):
                continue
            f1 = np.where(np.isfinite(g1), g1, np.nanmean(g1))
            f2 = np.where(np.isfinite(g2), g2, np.nanmean(g2))
            f1 = (f1 - f1.mean()) / (f1.std() + 1e-9)
            f2 = (f2 - f2.mean()) / (f2.std() + 1e-9)
            c = np.fft.irfft2(np.fft.rfft2(f1) * np.conj(np.fft.rfft2(f2)),
                              s=shape) / f1.size
            n0, n1 = shape
            du = np.where(2 * np.arange(n0) > n0, np.arange(n0) - n0, np.arange(n0))
            dv = np.where(2 * np.arange(n1) > n1, np.arange(n1) - n1, np.arange(n1))
            inside = (np.abs(du)[:, None] <= W) & (np.abs(dv)[None, :] <= W)
            cw = np.where(inside, c, -np.inf)
            pidx = np.unravel_index(int(np.argmax(cw)), shape)
            sdu, sdv = int(du[pidx[0]]), int(dv[pidx[1]])
            peak = float(cw[pidx])
            per = tile_period_m(g1, cell_m)
            shift = (sdu * cell_m) * e1 + (sdv * cell_m) * e2
            out.append({"pair": [i, j], "shift_m": [float(x) for x in shift],
                        "du_m": round(sdu * cell_m, 4), "dv_m": round(sdv * cell_m, 4),
                        "peak": round(peak, 4), "shared_cells": int(both.sum()),
                        "period_m": per, "window_m": float(window_m)})
            log(f"[floor-texture] {i} <-> {j}: {sdu * cell_m * 100:+.1f} x "
                f"{sdv * cell_m * 100:+.1f} cm, peak {peak:.3f}, "
                f"{int(both.sum())} shared cell(s), period {per}")
    return out
