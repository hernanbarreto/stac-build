"""The chunk cubes' floor and ceiling for GenRecon (adapter.py → levels.json → the STAC patch of
inference/get_chunks.py). GenRecon covers the scene with ONE layer of cubes whose base is z_floor and
whose height is (z_ceiling - z_floor) x chunk_size_factor; its model paints the floor at the cube's base
and the ceiling near its top, and whatever lies below the base is in no cube at all.

A LEVEL is a bin_m slice of heights holding at least min_share of the points it is counted over:
  support_levels  over the segmentation's support surfaces (autoprompt.support_surfaces) — the floor
                  the cameras walk on (2026-10-16: a percentile of a subsample must not decide it);
  cloud_levels    over the WHOLE published cloud — every surface the scene holds, labelled or not.
z_floor = the LOWEST of both. Station 2026-11-02: the tracks lie 1.15 m under the platform and are
'track' / 'rail' / 'gravel', none a support noun — the base on the platform left them under every cube
(USER: "corregí eso"). With no level at all, the percentile 0.2 of the cloud (the vendor's own anchor).
"""
from typing import Dict, List, Sequence

import numpy as np

SUPPORT_MIN_POINTS = 1000      # under this many support points the support says nothing (as before)


def height_levels(y, bin_m: float, min_share: float) -> List[float]:
    """The lower edges of the bin_m slices of ``y`` holding at least ``min_share`` of it, ascending."""
    y = np.asarray(y, np.float64)
    if y.size == 0:
        return []
    inv = 1.0 / bin_m                  # 20.0 at 5 cm: the arithmetic levels.json was written with
    h, e = np.histogram(y, bins=np.arange(np.floor(y.min() * inv) / inv, y.max() + bin_m, bin_m))
    return [float(e[i]) for i in range(len(h)) if h[i] / h.sum() >= min_share]


def chunk_levels(y_cloud, y_support, y_ceilings: Sequence, bin_m: float = 0.05,
                 min_share: float = 0.01) -> Dict:
    """levels.json: z_floor (the lowest support or cloud level), z_ceiling (the highest ceiling
    instance's p95, else the cloud's p99.97) and where each came from."""
    y_cloud = np.asarray(y_cloud, np.float64)
    levels: Dict = {"bin_m": bin_m, "level_min_share": min_share}
    floors = []
    if len(y_support) > SUPPORT_MIN_POINTS:
        sup = height_levels(y_support, bin_m, min_share)
        levels["support_levels"] = sup
        floors.append((min(sup), "segmentation support surfaces"))
    cloud = height_levels(y_cloud, bin_m, min_share)
    levels["cloud_levels"] = cloud
    if cloud:
        floors.append((min(cloud), "the whole cloud's lowest level (below every support surface)"))
    if floors:
        z, src = min(floors, key=lambda f: f[0])
        levels.update(z_floor=float(z), source=src)
    else:
        levels.update(z_floor=float(np.percentile(y_cloud, 0.2)),
                      source="percentile 0.2 of the cloud (no level)")
    ceil = [float(np.percentile(np.asarray(g, np.float64), 95)) for g in y_ceilings if len(g)]
    if ceil:
        levels.update(z_ceiling=max(ceil), ceiling_source="segmentation ceiling instances p95")
    else:
        levels.update(z_ceiling=float(np.percentile(y_cloud, 99.97)),
                      ceiling_source="percentile 99.97 of the cloud (no ceiling instance)")
    return levels
