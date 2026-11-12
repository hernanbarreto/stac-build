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
import math
from typing import Callable, Dict, List, Optional, Sequence

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
                 min_share: float = 0.01, levels_of: Optional[Callable] = None,
                 percentile_of: Optional[Callable] = None) -> Dict:
    """levels.json: z_floor (the lowest support or cloud level), z_ceiling (the highest ceiling
    instance's p95, else the cloud's p99.97) and where each came from.

    ``levels_of(y, bin_m, min_share)`` and ``percentile_of(y, q)`` read the levels and a percentile of a set of
    heights: absent, THE RECORD (height_levels and np.percentile over numpy heights); layout_gpu.py passes its
    twins, which read heights sorted on a device."""
    if levels_of is None:
        levels_of = height_levels
    if percentile_of is None:
        def percentile_of(y, q):
            return np.percentile(np.asarray(y, np.float64), q)
    levels: Dict = {"bin_m": bin_m, "level_min_share": min_share}
    floors = []
    if len(y_support) > SUPPORT_MIN_POINTS:
        sup = levels_of(y_support, bin_m, min_share)
        levels["support_levels"] = sup
        floors.append((min(sup), "segmentation support surfaces"))
    cloud = levels_of(y_cloud, bin_m, min_share)
    levels["cloud_levels"] = cloud
    if cloud:
        floors.append((min(cloud), "the whole cloud's lowest level (below every support surface)"))
    if floors:
        z, src = min(floors, key=lambda f: f[0])
        levels.update(z_floor=float(z), source=src)
    else:
        levels.update(z_floor=float(percentile_of(y_cloud, 0.2)),
                      source="percentile 0.2 of the cloud (no level)")
    ceil = [float(percentile_of(g, 95)) for g in y_ceilings if len(g)]
    top = float(percentile_of(y_cloud, 99.97))
    if ceil:
        levels.update(z_ceiling=max(ceil), ceiling_source="segmentation ceiling instances p95")
    else:
        levels.update(z_ceiling=top, ceiling_source="percentile 99.97 of the cloud (no ceiling instance)")
    levels["z_top"] = max(levels["z_ceiling"], top)      # the cloud's highest surface (stacked cubes cover it)
    return levels


# GenRecon's own lattice (inference/get_chunks.py BaseChunker): strides are multiples of 1/16 of the cube, adjacent
# cubes overlap by at least 4/16 of it (ScannetChunker's min_overlap_factor 4) — the joint generation places every
# chunk on that lattice, so the layers follow the same rule as the floor plan does
ATOMIC = 1.0 / 16.0
MIN_OVERLAP = 4.0 / 16.0


def stacked_centers(lo: float, hi: float, cube: float) -> List[float]:
    """The centres of the fewest cubes of side ``cube`` that cover [lo, hi] along the vertical: the first one STANDS
    on ``lo`` (the lowest level — the floor at a cube's base, as the model was trained), the stride the smallest
    multiple of cube/16 that reaches ``hi`` with at least a quarter of overlap between layers (USER 2026-11-02:
    "los tamaños de cubo con los que se entrenó, y deberían ser tantos como sean necesarios para cubrir la altura
    desde la parte más baja a la más alta de la nube")."""
    if cube <= 0:
        raise ValueError(f"cube side must be positive, got {cube}")
    dim = hi - lo
    if dim <= cube:
        return [lo + 0.5 * cube]
    atomic, max_stride = ATOMIC * cube, cube * (1.0 - MIN_OVERLAP)
    for n in range(2, 10_000):
        k = max(1, math.ceil((dim - cube) / (n - 1) / atomic - 1e-9))
        stride = k * atomic
        if stride > max_stride + 1e-9:
            continue
        return [lo + 0.5 * cube + i * stride for i in range(n)]
    raise ValueError(f"cannot stack cubes of {cube} m over {dim} m")


def anchored_centers(floor: float, lo: float, hi: float, cube: float) -> List[float]:
    """The centres of the cubes that cover [lo, hi] along the vertical with one base EXACTLY on ``floor``,
    the lowest measured support level (USER 2026-11-08: the model was trained with the floor at the cube's
    base and nothing below it — anchored there it paints the floor's top face only, never the slab's
    underside). Layers rise from ``floor`` by stacked_centers' own rule, and where the cloud reaches lower
    (lo < floor: station's tracks 1.15 m under the platform) layers continue DOWN — the fewest of them,
    their stride a multiple of cube/16 leaving at least a quarter of overlap — so every cube's grid stays
    on the one lattice ``plan_layout`` verifies. Full coverage of the cloud is kept (USER 2026-11-05)."""
    if cube <= 0:
        raise ValueError(f"cube side must be positive, got {cube}")
    up = stacked_centers(floor, max(hi, floor), cube)
    if lo >= floor:
        return up
    atomic, max_stride = ATOMIC * cube, cube * (1.0 - MIN_OVERLAP)
    span = floor - lo
    n = max(1, math.ceil(span / max_stride - 1e-9))        # the fewest layers below the anchor
    k = max(1, math.ceil(span / n / atomic - 1e-9))        # their stride, on the 1/16 lattice
    stride = k * atomic
    return [floor + 0.5 * cube - j * stride for j in range(n, 0, -1)] + up


def stack_layers(levels: Dict, cube: float, centers: Optional[Sequence[float]] = None) -> Dict:
    """levels.json with the stacked layout the patched chunker reads: ``chunk_size_m`` and ``chunk_z_centers``
    (``centers`` when given — the anchored layout, ONE source with layout.json — else stacked from the lowest
    level to the cloud's top). ``cube`` 0 = GenRecon's single layer, floor to ceiling."""
    if cube and cube > 0:
        levels["chunk_size_m"] = float(cube)
        levels["chunk_z_centers"] = ([float(c) for c in centers] if centers is not None
                                     else stacked_centers(levels["z_floor"], levels["z_top"], float(cube)))
    return levels
