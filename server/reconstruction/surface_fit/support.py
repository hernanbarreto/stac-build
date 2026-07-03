"""
No-extrapolation guarantee: a fitted surface is only meshed where measured
points exist within ``support_radius``. Where the segment has no data, the
surface is trimmed — a hole in the measurements stays a hole in the model.

Works in the surface's 2-D UV parametrization (shared by plane, quadric
[(s, r·θ)] and swept models): occupancy grid of the projected points, dilated
by the support radius, then a regular quad mesh keeping only supported quads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class TrimmedMesh:
    vertices_uv: np.ndarray     # (V,2)
    faces: np.ndarray           # (F,3) int
    support_fraction: float     # supported quads / quads in the UV bounding box
    area_m2: float              # supported area (quads × resolution²)


def support_grid(uv: np.ndarray, cell: float,
                 support_radius: float) -> Tuple[np.ndarray, float, float]:
    """Boolean occupancy of the UV domain, dilated by ``support_radius``.
    Returns (grid[h,w], u0, v0) with grid cell size = ``cell``."""
    uv = np.asarray(uv, dtype=np.float64)
    pad = max(support_radius, cell)
    u0 = float(uv[:, 0].min() - pad)
    v0 = float(uv[:, 1].min() - pad)
    ii = np.floor((uv[:, 0] - u0) / cell).astype(np.int64)
    jj = np.floor((uv[:, 1] - v0) / cell).astype(np.int64)
    w = int(ii.max()) + 1 + int(np.ceil(pad / cell))
    h = int(jj.max()) + 1 + int(np.ceil(pad / cell))
    occ = np.zeros((h, w), dtype=bool)
    occ[jj, ii] = True
    r = int(round(support_radius / cell))
    if r > 0:
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        occ = ndimage.binary_dilation(occ, structure=(xx * xx + yy * yy) <= r * r)
    return occ, u0, v0


def trimmed_quad_mesh(uv: np.ndarray, resolution: float,
                      support_radius: float) -> TrimmedMesh:
    """Regular grid mesh over the UV bounds at ``resolution``, keeping only
    quads whose center is supported (points within ``support_radius``).

    The occupancy test runs on a finer grid (¼ of the support radius, capped
    by resolution) so thin gaps larger than the radius are honoured even when
    the render resolution is coarse.
    """
    uv = np.asarray(uv, dtype=np.float64)
    occ_cell = max(min(resolution, support_radius / 2.0), 1e-4)
    occ, ou0, ov0 = support_grid(uv, occ_cell, support_radius)

    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    nu = max(int(np.ceil((u1 - u0) / resolution)), 1)
    nv = max(int(np.ceil((v1 - v0) / resolution)), 1)

    # quad centers → supported?
    uc = u0 + (np.arange(nu) + 0.5) * resolution
    vc = v0 + (np.arange(nv) + 0.5) * resolution
    ci = np.clip(np.floor((uc - ou0) / occ_cell).astype(np.int64), 0, occ.shape[1] - 1)
    cj = np.clip(np.floor((vc - ov0) / occ_cell).astype(np.int64), 0, occ.shape[0] - 1)
    keep = occ[np.ix_(cj, ci)]                       # (nv, nu)
    n_keep = int(keep.sum())
    if n_keep == 0:
        return TrimmedMesh(np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64), 0.0, 0.0)

    # grid nodes (nv+1, nu+1); only nodes referenced by kept quads survive
    jj, ii = np.nonzero(keep)
    corners = np.stack([jj * (nu + 1) + ii,
                        jj * (nu + 1) + ii + 1,
                        (jj + 1) * (nu + 1) + ii,
                        (jj + 1) * (nu + 1) + ii + 1], axis=1)   # (K,4) node ids
    used = np.unique(corners)
    remap = np.full((nv + 1) * (nu + 1), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    c = remap[corners]
    faces = np.concatenate([c[:, [0, 1, 3]], c[:, [0, 3, 2]]], axis=0)

    gj, gi = np.divmod(used, nu + 1)
    verts_uv = np.column_stack([u0 + gi * resolution, v0 + gj * resolution])
    return TrimmedMesh(vertices_uv=verts_uv, faces=faces,
                       support_fraction=float(n_keep) / float(nu * nv),
                       area_m2=float(n_keep) * resolution * resolution)


def mesh_on_surface(uv: np.ndarray, uv_to_world: Callable[[np.ndarray], np.ndarray],
                    resolution: float, support_radius: float
                    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Convenience: trimmed UV mesh lifted to world via ``uv_to_world``.
    Returns (vertices_world (V,3), faces (F,3), support_fraction, area_m2)."""
    tm = trimmed_quad_mesh(uv, resolution, support_radius)
    if len(tm.vertices_uv) == 0:
        return (np.zeros((0, 3)), tm.faces, 0.0, 0.0)
    return (uv_to_world(tm.vertices_uv), tm.faces, tm.support_fraction, tm.area_m2)
