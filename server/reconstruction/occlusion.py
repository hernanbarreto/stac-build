"""
Occlusion reasoning → opening detection for structural surfaces.
================================================================

A wall with an object in front of it has an *occlusion shadow* in the point
cloud — a region with no points that is **not** a real hole. A real opening
(door / window / passage / pit mouth) also has no points there, but you can see
*through* it / there is nothing in the way. This module tells the two apart with
the camera poses + a ray-cast against the other scene elements:

  1. rasterise the surface's outline into a grid in its (u,v) parameter space;
  2. cells with ≥1 high-confidence point ⇒ "supported"; the rest ⇒ "unsupported";
  3. connected components of unsupported cells = candidate holes;
  4. for each candidate, for every camera that should see the surface there:
     cast a ray camera→surface-point; if it hits another element first ⇒ that
     view is occluded. Majority occluded (or no view can verify) ⇒ shadow, keep
     it filled. Otherwise ⇒ real opening → an `OpeningElement` cut from the outline.

Planar surfaces only for now (curved-surface openings are a later refinement).
Returns the list of `OpeningElement`s (also mutates `surf.openings`).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .elements import SurfaceElement, OpeningElement
from .visual_hull import CameraView, project_points
from .geometry.primitives import _unit

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None
try:
    from shapely.geometry import Polygon, Point, box as _shp_box
    from shapely.ops import unary_union
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _HAS_SHAPELY = False


class _Raycaster:
    """Thin wrapper over Open3D's CPU RaycastingScene over a set of occluder meshes."""

    def __init__(self, meshes):
        self.scene = None
        if o3d is None:
            return
        sc = o3d.t.geometry.RaycastingScene()
        added = 0
        for m in meshes:
            try:
                v = np.asarray(m.vertices, dtype=np.float32)
                f = np.asarray(m.faces, dtype=np.uint32)
                if len(v) >= 3 and len(f) >= 1:
                    sc.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
                    added += 1
            except Exception:
                pass
        self.scene = sc if added else None

    def first_hit_dist(self, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """(N,3) origins + (N,3) unit dirs → (N,) distance to first hit (inf if none)."""
        if self.scene is None:
            return np.full(len(origins), np.inf)
        rays = np.concatenate([np.asarray(origins, np.float32),
                               np.asarray(dirs, np.float32)], axis=1)
        res = self.scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        return res["t_hit"].numpy().astype(np.float64)


def _cells_to_polygon(cells_ij: np.ndarray, umin: float, vmin: float, res: float):
    """Boundary polygon (M,2 in (u,v)) of a set of grid cells."""
    if not _HAS_SHAPELY:
        ci, cj = cells_ij[:, 0], cells_ij[:, 1]
        return np.array([[umin + ci.min() * res, vmin + cj.min() * res],
                         [umin + (ci.max() + 1) * res, vmin + cj.min() * res],
                         [umin + (ci.max() + 1) * res, vmin + (cj.max() + 1) * res],
                         [umin + ci.min() * res, vmin + (cj.max() + 1) * res]])
    boxes = [_shp_box(umin + i * res, vmin + j * res, umin + (i + 1) * res, vmin + (j + 1) * res)
             for i, j in cells_ij]
    poly = unary_union(boxes)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = poly.simplify(0.25 * res, preserve_topology=True)
    return np.asarray(poly.exterior.coords)[:-1]


def _classify_opening(region_poly: np.ndarray, outline_uv: np.ndarray,
                      surf: SurfaceElement, edge_tol: float) -> str:
    if surf.role in ("floor", "slab", "deck", "platform_edge"):
        return "pit_mouth"
    rv0, rv1 = region_poly[:, 1].min(), region_poly[:, 1].max()
    ov0, ov1 = outline_uv[:, 1].min(), outline_uv[:, 1].max()
    touches_bottom = (rv0 - ov0) <= edge_tol
    touches_top = (ov1 - rv1) <= edge_tol
    if touches_bottom and touches_top:
        return "passage"
    if touches_bottom:
        return "door"
    return "window"


def detect_openings(surf: SurfaceElement, hc_cloud: Optional[np.ndarray],
                    views: Optional[List[CameraView]] = None,
                    occluder_meshes=None, grid_res: float = 0.08,
                    min_area: float = 0.12, near_plane: float = 0.12,
                    min_pts_per_cell: int = 2, min_verifying_views: int = 2,
                    add_to_surface: bool = True) -> List[OpeningElement]:
    """Detect real openings in a planar structural surface. See module docstring."""
    if surf.surface_type != "plane" or surf.outline is None or len(surf.outline) < 3 \
            or not _HAS_SHAPELY:
        return []
    n = _unit(surf.plane_normal)
    origin = np.asarray(surf.basis_origin, dtype=np.float64)
    u = _unit(surf.basis_u)
    v = _unit(surf.basis_v)
    outline = np.asarray(surf.outline, dtype=np.float64)
    poly = Polygon(outline)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 4 * min_area:
        return []
    umin, vmin = outline.min(0)
    umax, vmax = outline.max(0)
    nu = max(2, int(np.ceil((umax - umin) / grid_res)))
    nv = max(2, int(np.ceil((vmax - vmin) / grid_res)))
    if nu * nv > 200_000:               # cap grid size
        grid_res = np.sqrt((umax - umin) * (vmax - vmin) / 150_000.0)
        nu = max(2, int(np.ceil((umax - umin) / grid_res)))
        nv = max(2, int(np.ceil((vmax - vmin) / grid_res)))

    # supported cells (from high-conf points near the plane) — require a few
    # points per cell so a couple of stray noisy points don't "fill" a real hole
    supported = np.zeros((nu, nv), dtype=bool)
    if hc_cloud is not None and len(hc_cloud):
        rel = np.asarray(hc_cloud, dtype=np.float64) - origin
        sd = rel @ n
        near = np.abs(sd) < near_plane
        pu = (rel[near] @ u - umin) / grid_res
        pv = (rel[near] @ v - vmin) / grid_res
        iu = np.clip(pu.astype(int), 0, nu - 1)
        iv = np.clip(pv.astype(int), 0, nv - 1)
        counts = np.zeros((nu, nv), dtype=np.int32)
        np.add.at(counts, (iu, iv), 1)
        supported = counts >= max(1, int(min_pts_per_cell))

    # cells inside the outline
    cu = umin + (np.arange(nu) + 0.5) * grid_res
    cv = vmin + (np.arange(nv) + 0.5) * grid_res
    inside = np.zeros((nu, nv), dtype=bool)
    # vectorise the point-in-polygon a bit: test a coarse mask then refine via prepared geometry
    from shapely.prepared import prep
    pp = prep(poly)
    for i in range(nu):
        for j in range(nv):
            if pp.contains(Point(cu[i], cv[j])):
                inside[i, j] = True
    unsupported = inside & ~supported
    # morphological opening: drop thin / tiny `unsupported` blobs — 1-cell
    # statistical pinholes and the 1-cell strip along the outline edge — while
    # keeping chunky regions (a real door / window / pit mouth survives intact).
    try:
        from scipy.ndimage import binary_opening
        unsupported = binary_opening(unsupported, structure=np.ones((3, 3)), iterations=1)
    except Exception:
        pass
    if not unsupported.any():
        return []

    from scipy.ndimage import label as cc_label
    lab, ncomp = cc_label(unsupported)
    rc = _Raycaster(occluder_meshes) if occluder_meshes else None
    cam_pos = np.array([np.asarray(v_.c2w, dtype=np.float64)[:3, 3] for v_ in (views or [])]) \
        if views else np.zeros((0, 3))
    edge_tol = 1.5 * grid_res

    openings: List[OpeningElement] = []
    for comp in range(1, ncomp + 1):
        cells = np.argwhere(lab == comp)
        if len(cells) * grid_res * grid_res < min_area:
            continue
        ci, cj = cells[:, 0], cells[:, 1]
        cuu = umin + (ci.mean() + 0.5) * grid_res
        cvv = vmin + (cj.mean() + 0.5) * grid_res
        p3 = origin + cuu * u + cvv * v

        # shadow vs real opening
        is_shadow = True
        if views:
            n_check = 0
            n_occ = 0
            for v_ in views:
                cam = np.asarray(v_.c2w, dtype=np.float64)[:3, 3]
                dvec = p3 - cam
                dist = float(np.linalg.norm(dvec))
                if dist < 1e-3:
                    continue
                dir_ = dvec / dist
                if abs(float(n @ dir_)) < 0.18:           # too edge-on to see the surface here
                    continue
                uv, z, inb = project_points(p3[None, :], v_.c2w, v_.K, v_.width, v_.height)
                if not bool(inb[0]):
                    continue
                n_check += 1
                if rc is not None:
                    hit = float(rc.first_hit_dist(cam[None, :], dir_[None, :])[0])
                    if hit < dist * 0.98:
                        n_occ += 1
            # need a few cameras that *can* see the surface there and agree it's
            # unobstructed before calling it a real hole (otherwise: keep filled)
            is_shadow = (n_check < max(1, int(min_verifying_views))) or (n_occ >= 0.5 * max(n_check, 1))
        if is_shadow:
            continue

        region_poly = _cells_to_polygon(cells, umin, vmin, grid_res)
        if region_poly is None or len(region_poly) < 3:
            continue
        # clip the region to the outline (so it can't poke past the surface)
        try:
            rp = Polygon(region_poly).intersection(poly)
            if rp.geom_type == "MultiPolygon":
                rp = max(rp.geoms, key=lambda g: g.area)
            if rp.is_empty or rp.area < min_area:
                continue
            region_poly = np.asarray(rp.exterior.coords)[:-1]
        except Exception:
            pass
        # a real door / window / passage is bounded by surface on *both* u-sides
        # (the wall continues past it). An unsupported region that runs into a
        # left/right edge of the outline is the (loose) outline reaching past
        # where the surface actually extends — not an opening. Skip it.
        rp_u = region_poly[:, 0]
        ou0, ou1 = float(outline[:, 0].min()), float(outline[:, 0].max())
        if (float(rp_u.min()) - ou0) <= edge_tol or (ou1 - float(rp_u.max())) <= edge_tol:
            continue
        kind = _classify_opening(region_poly, outline, surf, edge_tol)
        conf = float(min(1.0, 0.5 + 0.5 * len(cells) * grid_res * grid_res / max(poly.area, 1e-6) * 4))
        openings.append(OpeningElement(polygon=region_poly, kind=kind, confidence=conf))

    if add_to_surface:
        surf.openings.extend(openings)
        if openings:
            surf.meta["n_openings_detected"] = len(openings)
    return openings
