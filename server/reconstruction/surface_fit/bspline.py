"""
Stage 2.4 — free-form B-spline surface, the last rung of the ladder.

A tensor-product CUBIC B-spline height field h(u,v) over the segment's total-
least-squares base plane, with knot (≈ control point) spacing configurable
(default 50 cm). That spacing is the whole point: the spline is a geometric
low-pass — it follows real curvature at the decimetre-and-up scale and
physically cannot follow high-frequency sensor/registration noise. Nothing is
generated: the surface is a least-squares fit to measured heights only, and
the support trim removes any UV area without measurements.

Implementation: ``scipy.interpolate.LSQBivariateSpline`` (least-squares with
explicit interior knots). scipy is already a hard dependency of the repo —
chosen over adding ``geomdl``, which offers the same tensor-product B-spline
math for this use case at the cost of a new dependency.

Limitation (by design): a height field over a plane cannot represent closed
or strongly folded surfaces — those are the swept-profile/quadric stages' job;
by the time escalation reaches the B-spline, the segment failed those either
because it is a gently free-form sheet (deformed slab, warped wall), which a
height field represents exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import LSQBivariateSpline

from .plane import PlaneModel, fit_plane

logger = logging.getLogger("SurfaceFit")

_MIN_PTS_PER_CELL = 4     # knot cells need support or the LS system is singular


@dataclass
class BSplineModel:
    """Base plane + cubic B-spline height field h(u,v) along the normal."""
    base: PlaneModel
    spline: LSQBivariateSpline
    dof: int
    rms: float
    inlier_frac: float
    n_points: int

    @property
    def DOF(self) -> int:
        return self.dof

    # expose the base-plane normal so plumb findings still work for walls
    @property
    def normal(self) -> np.ndarray:
        return self.base.normal

    def _height(self, uv: np.ndarray) -> np.ndarray:
        return self.spline.ev(uv[:, 0], uv[:, 1])

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        # residual along the normal vs the spline height at the same (u,v) —
        # exact for a height field (the point and its surface foot share UV)
        uv = self.base.to_uv(pts)
        return self.base.signed_distance(pts) - self._height(uv)

    def to_uv(self, pts: np.ndarray) -> np.ndarray:
        return self.base.to_uv(pts)

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        flat = self.base.uv_to_world(uv)
        return flat + self._height(uv)[:, None] * self.base.normal

    def params_dict(self) -> dict:
        tx, ty = self.spline.get_knots()
        return {
            "base_plane": self.base.params_dict(),
            "knots_u": np.asarray(tx).tolist(),
            "knots_v": np.asarray(ty).tolist(),
            "coeffs": np.asarray(self.spline.get_coeffs()).tolist(),
            "degree": [3, 3],
            "rms_m": float(self.rms),
            "inlier_frac": float(self.inlier_frac),
        }


def fit_bspline(points: np.ndarray,
                ctrl_point_spacing_m: float = 0.5,
                dist_thresh: float = 0.02,
                min_inlier_frac: float = 0.5,
                world_up: Optional[np.ndarray] = None,
                scene_centroid: Optional[np.ndarray] = None) -> Optional[BSplineModel]:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 200:
        return None

    # base plane: loose threshold — it only anchors the UV chart; the spline
    # models the actual shape. Half the extent keeps ~all points inliers.
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    base = fit_plane(pts, dist_thresh=max(0.05, extent / 4.0), ransac_iters=200,
                     min_inlier_frac=0.5, world_up=world_up,
                     scene_centroid=scene_centroid)
    if base is None:
        return None

    uv = base.to_uv(pts)
    h = base.signed_distance(pts)
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    if (u1 - u0) < ctrl_point_spacing_m or (v1 - v0) < ctrl_point_spacing_m:
        return None

    # interior knots every ctrl spacing, only where the data can support them
    eps = 1e-6 * max(u1 - u0, v1 - v0, 1.0)
    tx = np.arange(u0 + ctrl_point_spacing_m, u1 - ctrl_point_spacing_m / 2,
                   ctrl_point_spacing_m)
    ty = np.arange(v0 + ctrl_point_spacing_m, v1 - ctrl_point_spacing_m / 2,
                   ctrl_point_spacing_m)
    n_coef = (len(tx) + 4) * (len(ty) + 4)
    if n < _MIN_PTS_PER_CELL * n_coef:
        logger.info("bspline: %d points can't support %d coefficients — skipping",
                    n, n_coef)
        return None
    try:
        spl = LSQBivariateSpline(uv[:, 0], uv[:, 1], h, tx, ty, kx=3, ky=3,
                                 bbox=[u0 - eps, u1 + eps, v0 - eps, v1 + eps])
    except Exception as e:
        logger.info("bspline: LSQBivariateSpline failed (%s)", e)
        return None

    model = BSplineModel(base=base, spline=spl, dof=int(n_coef),
                         rms=0.0, inlier_frac=0.0, n_points=n)
    resid = model.signed_distance(pts)
    inl = np.abs(resid) < dist_thresh
    model.inlier_frac = float(inl.mean())
    model.rms = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.any() else float("inf")
    if model.inlier_frac < min_inlier_frac:
        logger.info("bspline: inlier fraction %.2f < %.2f — rejected",
                    model.inlier_frac, min_inlier_frac)
        return None
    return model
