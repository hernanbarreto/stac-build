"""GenRecon's presence decided by the keyframes, never by a count of points (USER 2026-11-05: "que decidan las
fotos, no la cuenta de puntos … el límite es la nube").

Every point of the published cloud was born in ONE keyframe at ONE pixel (cleaned_cloud.ply: frame_global,
pixel_row, pixel_col), so the depth each keyframe measured is rebuilt from the cloud itself — nothing outside the
cloud enters. Each point is then put to every OTHER keyframe that frames it:

  AT       that keyframe measured a surface where the point is (within the tolerance) — it confirms it;
  THROUGH  that keyframe measured only surfaces BEHIND the point — it saw through it: the point floats in air;
  hidden   that keyframe measured only surfaces in front of the point (occluded), or nothing there — no vote.

support = 1 (the keyframe it was born in) + AT. A point stays when ``support >= min_views`` and THROUGH does not
outnumber its support. The tolerance is the voxel or the session's own depth disagreement at the point's depth,
whichever is larger: κ·z with κ = 3σ, σ the robust spread of the relative disagreement between keyframes that
measured the same surface (measured here, on this cloud). Votes only REMOVE points: the cloud is the limit.

The cubes (``plan_layout``) are GenRecon's own lattice laid over the WHOLE of what stays — its exact extent, no
percentile —: x and y by GenRecon's own rule (inference/get_chunks.BaseChunker._define_centers: the floor plan padded
by 3.5/16 of the cube, then _axis_centers), z stacked from the lowest point (genrecon_levels.stacked_centers), every cube's 32³ grid one lattice; a voxel is ACTIVE when it holds
a point that stayed, and a cube is kept when it holds an active voxel. Deterministic: element-wise ops, gathers and
min/max scatters on one declared device."""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np
import torch

from genrecon_levels import stacked_centers

ATOMIC = 1.0 / 16.0            # GenRecon's lattice: strides are multiples of 1/16 of the cube
MIN_OVERLAP = 4.0 / 16.0       # adjacent cubes overlap by at least 4/16 of the cube (BaseChunker min_overlap_factor 4)
MIN_PADDING = 3.5 / 16.0       # the floor plan is padded by 3.5/16 of the cube on each side (BaseChunker.min_padding):
                               # the outermost walls sit inside a cube, never on its face (as the model was trained)


def _project(X: torch.Tensor, w2c: torch.Tensor, intr: Sequence[float]):
    """Points (n, 3) → (z, u, v) in one keyframe with the OPENCV lens (fx fy cx cy k1 k2 p1 p2)."""
    fx, fy, cx, cy, k1, k2, p1, p2 = intr
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return z, fx * xd + cx, fy * yd + cy


def vote_points(xyz: np.ndarray, frame_global: np.ndarray, pixel_row: np.ndarray, pixel_col: np.ndarray,
                frames: Sequence[int], c2w: np.ndarray, intr: Sequence[float], width: int, height: int,
                voxel: float, *, min_views: int = 2, block: int = 4, device: str = "cuda",
                batch: int = 8_000_000, log: Callable[[str], None] = print) -> Dict:
    """The keyframes' vote on every point (module docstring). ``xyz`` in the frame of ``c2w`` (the raw frame);
    pixel rows/cols on the ``width`` x ``height`` grid of ``intr``. ``block`` px squares hold each keyframe's
    measured depth range (its own points' min/max) so a sparse depth map still answers."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("presence: device cuda requested and no card is available")
    N, K = len(xyz), len(frames)
    frames = np.asarray(frames, np.int64)
    lut = np.full(int(max(frames.max(), np.max(frame_global, initial=0))) + 1, -1, np.int64)
    lut[frames] = np.arange(K)
    birth = lut[np.asarray(frame_global, np.int64)]                       # keyframe index, -1 = not a keyframe
    w2c = torch.as_tensor(np.linalg.inv(np.asarray(c2w, np.float64)), dtype=torch.float32, device=dev)
    P = torch.as_tensor(np.asarray(xyz, np.float32), device=dev)
    bk = torch.as_tensor(birth, device=dev)
    nbr, nbc = (height + block - 1) // block, (width + block - 1) // block
    nb = nbr * nbc
    # each keyframe's measured depth, rebuilt from the points born in it: per block of pixels its min and max
    zmin = torch.full((K * nb,), float("inf"), device=dev)
    zmax = torch.full((K * nb,), float("-inf"), device=dev)
    pr = torch.as_tensor(np.asarray(pixel_row, np.int64), device=dev)
    pc = torch.as_tensor(np.asarray(pixel_col, np.int64), device=dev)
    for k in range(K):
        m = bk == k
        if not bool(m.any()):
            continue
        z, _, _ = _project(P[m], w2c[k], intr)
        idx = k * nb + (pr[m] // block).clamp(0, nbr - 1) * nbc + (pc[m] // block).clamp(0, nbc - 1)
        zmin.scatter_reduce_(0, idx, z, reduce="amin", include_self=True)
        zmax.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
    del pr, pc

    def lookup(k: int, X: torch.Tensor):
        z, u, v = _project(X, w2c[k], intr)
        ok = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        bi = k * nb + (v.clamp(0, height - 1).long() // block) * nbc + (u.clamp(0, width - 1).long() // block)
        lo, hi = zmin[bi], zmax[bi]
        return z, ok & torch.isfinite(lo), lo, hi

    # the session's own depth disagreement between keyframes measuring one surface: a fixed stride of the points,
    # every keyframe, blocks that hold one surface (spread ≤ a voxel); σ = 1.4826·MAD of the relative difference
    sub = torch.arange(0, N, max(1, N // 2_000_000), device=dev)
    rel = []
    for k in range(K):
        X = P[sub]
        z, ok, lo, hi = lookup(k, X)
        ok &= (bk[sub] != k) & ((hi - lo) <= voxel)
        r = ((lo + hi) * 0.5 / z - 1.0)[ok]
        rel.append(r[r.abs() < 0.25])
    rel = torch.cat(rel)
    if len(rel) < 1000:
        raise RuntimeError(f"presence: only {len(rel)} keyframe pairs see one surface — nothing to measure the "
                           "session's depth disagreement on")
    med = rel.median()
    sigma = float(1.4826 * (rel - med).abs().median())
    kappa = 3.0 * sigma
    log(f"[presence] the session's depth disagreement between keyframes: σ {sigma * 100:.2f} % of the depth "
        f"(median {float(med) * 100:+.2f} %, {len(rel):,} pairs); tolerance max(voxel {voxel * 100:.1f} cm, "
        f"{kappa * 100:.2f} % of the depth)")

    at = torch.zeros(N, dtype=torch.int16, device=dev)
    through = torch.zeros(N, dtype=torch.int16, device=dev)
    for s in range(0, N, batch):
        X = P[s:s + batch]
        bks = bk[s:s + batch]
        a = torch.zeros(len(X), dtype=torch.int16, device=dev)
        t = torch.zeros(len(X), dtype=torch.int16, device=dev)
        for k in range(K):
            z, ok, lo, hi = lookup(k, X)
            ok &= bks != k
            tol = torch.clamp(kappa * z, min=voxel)
            a += (ok & (z >= lo - tol) & (z <= hi + tol)).to(torch.int16)
            t += (ok & (z < lo - tol)).to(torch.int16)
        at[s:s + batch], through[s:s + batch] = a, t
    support = at.to(torch.int32) + 1
    thr = through.to(torch.int32)
    keep = (support >= int(min_views)) & (thr <= support)
    n_keep = int(keep.sum())
    n_single = int((support < int(min_views)).sum())
    n_through = int(((support >= int(min_views)) & (thr > support)).sum())
    log(f"[presence] {n_keep:,} of {N:,} points stay ({100.0 * n_keep / max(N, 1):.1f} %): "
        f"{n_single:,} measured by fewer than {min_views} keyframe(s), {n_through:,} seen through by more "
        f"keyframes than measure them; {K} keyframes, blocks of {block} px")
    return {"keep": keep.cpu().numpy(), "support": support.cpu().numpy().astype(np.int16),
            "through": thr.cpu().numpy().astype(np.int16), "sigma_rel": sigma, "kappa": kappa,
            "min_views": int(min_views), "block_px": int(block),
            "removed_single_view": n_single, "removed_seen_through": n_through}


def axis_centers(lo: float, hi: float, cube: float) -> list:
    """GenRecon's own axis rule (inference/get_chunks.BaseChunker._axis_centers) over [lo, hi]: the fewest cubes,
    centred on the midpoint, strides multiples of cube/16 leaving at least a quarter of overlap."""
    dim = hi - lo
    if dim <= cube:
        return [lo + 0.5 * dim]
    atomic, max_stride = ATOMIC * cube, cube * (1.0 - MIN_OVERLAP)
    for n in range(2, 10_000):
        k = max(1, math.ceil((dim - cube) / (n - 1) / atomic - 1e-9))
        stride = k * atomic
        if stride > max_stride + 1e-9:
            continue
        start = 0.5 * dim - 0.5 * (n - 1) * stride
        return [lo + start + i * stride for i in range(n)]
    raise ValueError(f"cannot place cubes of {cube} m over {dim} m")


def plan_layout(P: np.ndarray, cube: float, R: int = 32, log: Callable[[str], None] = print, min_points: int = 1,
                z_centers: Sequence[float] = None, support_levels: Sequence[float] = None,
                level_bin: float = 0.05, device: str = None) -> Dict:
    """The cubes over the WHOLE of ``P`` (GenRecon's frame, z up) and its active voxels (module docstring).

    ``z_centers``: the vertical layout (genrecon_levels.anchored_centers — one base exactly on the lowest
    support level), ONE source with levels.json; absent, the layers stand on the cloud's lowest point.
    ``support_levels``: SOLID GROUND (USER 2026-11-08: "el suelo es sólido … rellena su columna hacia abajo
    hasta la base del cubo"): every column holding an active voxel in the ``level_bin`` slice of a support
    level (the floor, a ramp's landing, the platform — the levels the floor step measures) is filled active
    from under that slice down to the base of the lattice, so no floor has air below it in ANY cube and the
    model can only paint its top face — the condition it was trained on (below a floor there is earth).

    ``device``: where the lines that read EVERY point run (the cloud's extent, each point's voxel, the count of
    every voxel) — absent or "numpy", THE RECORD below; "cpu" / "cuda", layout_gpu.py's twins (``P`` may then be
    a tensor already on that device). Everything after them is the same numpy over the voxels."""
    on_device = device not in (None, "numpy")
    if on_device:
        import layout_gpu
        P = layout_gpu.as_points(P, device)
        lo, hi = layout_gpu.extent(P)
    else:
        P = np.asarray(P, np.float64)
        lo, hi = P.min(0), P.max(0)
    pad = MIN_PADDING * cube         # GenRecon's own padding of the floor plan (inference/get_chunks._define_centers)
    xs, ys = axis_centers(lo[0] - pad, hi[0] + pad, cube), axis_centers(lo[1] - pad, hi[1] + pad, cube)
    if z_centers is not None:
        zs = [float(z) for z in z_centers]
        if min(zs) - 0.5 * cube > lo[2] + 1e-9 or max(zs) + 0.5 * cube < hi[2] - 1e-9:
            raise RuntimeError(f"z_centers {min(zs):.3f}..{max(zs):.3f} (cube {cube}) do not cover the "
                               f"cloud's z {lo[2]:.3f}..{hi[2]:.3f} — the cubes must cover the WHOLE cloud")
    else:
        zs = stacked_centers(lo[2], hi[2], cube)
    v = cube / R
    axes = [np.asarray(xs), np.asarray(ys), np.asarray(zs)]
    origin = np.array([a.min() - 0.5 * cube for a in axes])
    corners = [np.round((a - 0.5 * cube - o) / v).astype(np.int64) for a, o in zip(axes, origin)]
    for a, o, c in zip(axes, origin, corners):
        if np.abs((a - 0.5 * cube - o) / v - c).max() > 1e-6:
            raise RuntimeError("the cubes' grids are not one lattice")
    if on_device:
        act, dims = layout_gpu.active_keys(P, origin, v, min_points)
    else:
        ijk = np.clip(np.floor((P - origin) / v).astype(np.int64), 0, None)   # the guard above bounds the clip to float noise
        dims = ijk.max(0) + 1
        key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
        act, cnt = np.unique(key, return_counts=True)
        act = act[cnt >= int(min_points)]        # a voxel is active from min_points points (GenRecon's own rule: 10)
    cells = np.stack([act // (dims[1] * dims[2]), (act // dims[2]) % dims[1], act % dims[2]], 1)
    # SOLID GROUND (docstring): columns of the support levels' active voxels filled down to the lattice base
    n_ground = 0
    if support_levels:
        oz, fills = float(origin[2]), []
        for L in sorted({float(l) for l in support_levels}):
            k0 = int(math.floor((L - oz) / v + 1e-9))                 # the slice's first voxel layer
            k1 = min(int(math.floor((L + level_bin - oz) / v + 1e-9)), int(dims[2]) - 1)
            if k0 <= 0 or k0 >= int(dims[2]):                         # at the lattice base: nothing below it
                continue
            layer = cells[(cells[:, 2] >= k0) & (cells[:, 2] <= k1)]
            if not len(layer):
                continue
            cols = np.unique(layer[:, :2], axis=0)
            depth = np.arange(0, k0, dtype=np.int64)
            fills.append(np.column_stack([np.repeat(cols, len(depth), axis=0), np.tile(depth, len(cols))]))
        if fills:
            before = len(cells)
            cells = np.unique(np.concatenate([cells] + fills), axis=0)
            n_ground = int(len(cells) - before)
            log(f"[presence] solid ground (USER 2026-11-08): {n_ground:,} voxels filled under "
                f"{len(set(float(l) for l in support_levels))} support level(s), down to the lattice base — "
                f"no floor keeps air below it")
    # a cube is kept when its R³ cells hold an active one: the count by a summed-area table over the lattice
    G = np.zeros(tuple(int(d) + R + 1 for d in dims), np.int32)
    G[cells[:, 0] + 1, cells[:, 1] + 1, cells[:, 2] + 1] = 1
    S = G.cumsum(0).cumsum(1).cumsum(2)

    def box(a0, b0, c0):
        a1, b1, c1 = a0 + R, b0 + R, c0 + R
        a0, b0, c0, a1, b1, c1 = [int(np.clip(t, 0, S.shape[i] - 1)) for i, t in
                                  zip((0, 1, 2, 0, 1, 2), (a0, b0, c0, a1, b1, c1))]
        return (S[a1, b1, c1] - S[a0, b1, c1] - S[a1, b0, c1] - S[a1, b1, c0]
                + S[a0, b0, c1] + S[a0, b1, c0] + S[a1, b0, c0] - S[a0, b0, c0])

    centers = []
    for i, cx in enumerate(xs):
        for j, cy in enumerate(ys):
            for k, cz in enumerate(zs):
                if box(corners[0][i], corners[1][j], corners[2][k]) > 0:
                    centers.append([float(cx), float(cy), float(cz)])
    log(f"[presence] cubes of {cube} m (voxel {v * 100:.2f} cm) over the whole of what stays: x {lo[0]:.2f}..{hi[0]:.2f} "
        f"y {lo[1]:.2f}..{hi[1]:.2f} z {lo[2]:.2f}..{hi[2]:.2f} m — {len(xs)}x{len(ys)}x{len(zs)} on the lattice, "
        f"{len(centers)} hold an active voxel ({len(cells):,} active voxels)")
    return {"chunk_size": float(cube), "R": int(R), "voxel": float(v), "origin": origin.tolist(),
            "centers": centers, "x_centers": [float(a) for a in xs], "y_centers": [float(a) for a in ys],
            "z_centers": [float(a) for a in zs], "extent": [lo.tolist(), hi.tolist()],
            "ground_filled": int(n_ground), "support_levels": [float(l) for l in (support_levels or [])],
            "cells": cells.astype(np.int32)}
