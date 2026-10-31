"""The instance of every gaussian — so a click on a splat selects the object the segmentation
named, and an object switched off in the Instances list takes its splats with it.

USER 2026-10-31: the segmentation is not fixed — the Segmentation Manager merges instances, creates
new ones, erases points — so NO instance id is baked into the gaussians. Every gaussian carries the
ROW of its seed in the published cloud (``seed_index.bin``: its own row for a seed, the nearest
seed's within ``label.nearest_radius_m`` for a gaussian born in the densification, NO_SEED beyond
it) and the instance is looked up LIVE in the cloud's ``instance_ids.npy`` — the file every edit
rewrites and the octree colours by. A new cloud (another point count) makes the mapping stale, said.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

NO_SEED = np.uint32(0xFFFFFFFF)


def seed_index_by_nearest(xyz: np.ndarray, seeds_xyz: np.ndarray, seeds_cloud_index: np.ndarray, radius_m: float) -> np.ndarray:
    """(N,) uint32: the cloud row of the nearest seed within ``radius_m`` of every gaussian, NO_SEED
    beyond it — geometry only, no semantics."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, np.float64)
    out = np.full(len(xyz), NO_SEED, np.uint32)
    if len(seeds_xyz) == 0 or len(xyz) == 0:
        return out
    tree = cKDTree(np.asarray(seeds_xyz, np.float64))
    d, j = tree.query(xyz, k=1, distance_upper_bound=float(radius_m))
    ok = np.isfinite(d) & (j < len(seeds_xyz))
    out[ok] = np.asarray(seeds_cloud_index, np.uint32)[j[ok]]
    return out


def instances_from_seed_index(seed_index: np.ndarray, instance_ids: np.ndarray) -> np.ndarray:
    """(N,) uint32: the CURRENT instance of every gaussian — ``instance_ids`` is the cloud's live
    ``instance_ids.npy``; NO_SEED and a row beyond the cloud read 0 (unsegmented)."""
    si = np.asarray(seed_index, np.uint32)
    ids = np.asarray(instance_ids)
    out = np.zeros(len(si), np.uint32)
    ok = (si != NO_SEED) & (si < len(ids))
    out[ok] = ids[si[ok]].astype(np.uint32)
    return out


def instance_census(ids: np.ndarray) -> Dict[str, int]:
    ids = np.asarray(ids)
    seg = ids[ids > 0]
    return {"n_gaussians": int(len(ids)), "n_segmented": int(len(seg)),
            "n_instances": int(len(np.unique(seg))) if len(seg) else 0}
