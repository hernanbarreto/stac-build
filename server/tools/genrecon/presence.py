"""Presence for GenRecon from what the chain MEASURED, not only from the points that survived the filters
(USER 2026-11-01 02:15 CDMX: the mesh "se comió" different parts in the original and in the audit — a voxel
exists only where ≥ min_points published points remain, so every region a filter thinned loses its mesh).

``cloud+depth2``: the per-keyframe depth maps of the chain (dense, before any filter) are unprojected with
the certified poses and voted on GenRecon's voxel grid — a voxel counts for presence when TWO different
keyframes put measured depth in it ("one neighbour is an opinion, two agreeing are evidence": the chain's
own repair rule, tsdf_weight_thresh 2.0). The points of those voxels join the cloud's as presence; the
cloud itself is untouched and the model is still conditioned on the images. Everything here is numpy,
deterministic (fixed strides, no draws) and declared in levels.json / the log.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

GENRECON_VOXEL_M = 3.7 / 32.0      # GenRecon's chunk cube / its 32³ active grid ≈ 11.6 cm


def unproject_depth(depth: np.ndarray, valid: np.ndarray, K: np.ndarray, c2w: np.ndarray,
                    stride: int = 3, min_depth_m: float = 0.05) -> np.ndarray:
    """(N, 3) world points of a depth map on ITS grid (K is that grid's camera): every ``stride``-th pixel
    with a valid, positive depth, through the pinhole K and the keyframe's c2w (the lens is ignored: at
    GenRecon's 11.6 cm voxel the distortion of a phone frame moves nothing across a cell)."""
    d = np.asarray(depth, np.float64)
    ok = np.asarray(valid, bool) & np.isfinite(d) & (d > float(min_depth_m))
    h, w = d.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    sel = ok[vv, uu]
    u = uu[sel].astype(np.float64); v = vv[sel].astype(np.float64); z = d[vv, uu][sel]
    K = np.asarray(K, np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    c2w = np.asarray(c2w, np.float64)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def voxel_keys(xyz: np.ndarray, voxel_m: float) -> np.ndarray:
    """One int64 per point: its voxel on a grid of ``voxel_m`` anchored at the origin (exact, no hash)."""
    c = np.floor(np.asarray(xyz, np.float64) / float(voxel_m)).astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def vote_cells(keys: np.ndarray, kf: np.ndarray, min_views: int = 2) -> np.ndarray:
    """The voxel keys that at least ``min_views`` DIFFERENT keyframes put measured depth in — sorted,
    unique. ``kf`` is the keyframe index of every point (same length as ``keys``)."""
    pairs = np.unique(np.stack([np.asarray(keys, np.int64), np.asarray(kf, np.int64)], 1), axis=0)
    cells, counts = np.unique(pairs[:, 0], return_counts=True)
    return cells[counts >= int(min_views)]


def depth_presence(points: np.ndarray, kf: np.ndarray, cloud_xyz: np.ndarray, voxel_m: float = GENRECON_VOXEL_M,
                   min_views: int = 2, max_per_cell: int = 24) -> Tuple[np.ndarray, Dict[str, int]]:
    """The depth-map points that join the cloud for presence: those in voxels confirmed by ≥ ``min_views``
    keyframes, at most ``max_per_cell`` per voxel (the first ones in the deterministic order given — never a
    draw). Returns (points, record) with the census: cloud voxels, confirmed depth voxels, the NEW ones the
    cloud did not hold, and the points appended."""
    keys = voxel_keys(points, voxel_m)
    kept_cells = vote_cells(keys, kf, min_views)
    cloud_cells = np.unique(voxel_keys(cloud_xyz, voxel_m)) if len(cloud_xyz) else np.zeros(0, np.int64)
    new_cells = np.setdiff1d(kept_cells, cloud_cells, assume_unique=True)
    in_kept = np.isin(keys, kept_cells)
    idx = np.nonzero(in_kept)[0]
    # at most max_per_cell per voxel, by the points' own order (a stable sort by cell, then position)
    order = idx[np.argsort(keys[idx], kind="stable")]
    k_sorted = keys[order]
    first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    rank = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    chosen = order[rank < int(max_per_cell)]
    chosen.sort()
    rec = {"cloud_voxels": int(len(cloud_cells)), "depth_voxels_confirmed": int(len(kept_cells)),
           "depth_voxels_new": int(len(new_cells)), "points_appended": int(len(chosen)),
           "min_views": int(min_views), "voxel_m": float(voxel_m), "max_per_cell": int(max_per_cell)}
    return np.asarray(points, np.float64)[chosen], rec
