"""
Stage 2.1 — plane fit with an in-plane UV parametrization.

Thin wrapper over ``reconstruction.geometry.primitives.fit_plane_ransac``
(RANSAC + total-least-squares refine on inliers) that adds what the rest of
surface_fit needs: a deterministic UV basis on the plane (all residual grids,
heatmaps, support masks and meshes live in this 2-D frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from reconstruction.geometry.primitives import PlaneFit, fit_plane_ransac, _orthobasis
from .models import orient_normal

_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class PlaneModel:
    """n·x + d = 0 with an anchored orthonormal in-plane basis (origin, u, v).
    world = origin + s*u + t*v ;  uv = [(x-origin)·u, (x-origin)·v]."""
    normal: np.ndarray        # (3,) unit, oriented (see models.orient_normal)
    d: float
    origin: np.ndarray        # (3,) point on plane (projected inlier centroid)
    u: np.ndarray             # (3,) unit, in-plane
    v: np.ndarray             # (3,) unit, in-plane (n × u)
    rms: float                # inlier RMS from the fit (m)
    inlier_frac: float
    n_points: int

    DOF = 3  # normal (2) + offset (1)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) @ self.normal + self.d

    def to_uv(self, pts: np.ndarray) -> np.ndarray:
        q = np.asarray(pts, dtype=np.float64) - self.origin
        return np.column_stack([q @ self.u, q @ self.v])

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        return self.origin + uv[:, :1] * self.u + uv[:, 1:2] * self.v

    def params_dict(self) -> dict:
        return {
            "normal": self.normal.tolist(),
            "d": float(self.d),
            "origin": self.origin.tolist(),
            "basis_u": self.u.tolist(),
            "basis_v": self.v.tolist(),
            "rms_m": float(self.rms),
            "inlier_frac": float(self.inlier_frac),
        }


def _plane_basis(normal: np.ndarray, world_up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """In-plane (u, v) chosen so heatmaps read naturally: for walls, v is the
    projection of world-up onto the plane ("height" axis) and u is horizontal;
    for floors/ceilings the generic orthobasis is fine (no privileged axis)."""
    n = np.asarray(normal, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    up_in_plane = up - (up @ n) * n
    ln = np.linalg.norm(up_in_plane)
    if ln > 0.3:  # meaningfully non-horizontal surface → align v with up
        v = up_in_plane / ln
        u = np.cross(v, n)
        u /= np.linalg.norm(u)
        return u, v
    return _orthobasis(n)


def fit_plane(points: np.ndarray,
              dist_thresh: float = 0.012,
              ransac_iters: int = 500,
              min_inlier_frac: float = 0.30,
              world_up: Optional[np.ndarray] = None,
              scene_centroid: Optional[np.ndarray] = None) -> Optional[PlaneModel]:
    """RANSAC + LS plane with oriented normal and anchored UV basis.
    Returns ``None`` when no plane gathers ``min_inlier_frac`` of the points."""
    pts = np.asarray(points, dtype=np.float64)
    pf: Optional[PlaneFit] = fit_plane_ransac(
        pts, dist_thresh=float(dist_thresh), iters=int(ransac_iters),
        min_inlier_frac=float(min_inlier_frac), measure_curvature=False)
    if pf is None:
        return None

    up = _UP if world_up is None else np.asarray(world_up, dtype=np.float64)
    normal = orient_normal(pf.normal, up, points=pts, centroid_hint=scene_centroid)
    d = float(pf.d) if normal @ pf.normal > 0 else float(-pf.d)

    ctr = pts[pf.inliers].mean(0)
    origin = ctr - (ctr @ normal + d) * normal      # project centroid onto plane
    u, v = _plane_basis(normal, up)
    return PlaneModel(normal=normal, d=d, origin=origin, u=u, v=v,
                      rms=pf.rms, inlier_frac=pf.inlier_frac, n_points=pf.n_points)
