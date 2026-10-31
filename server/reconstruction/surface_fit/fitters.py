"""The analytic fitters the cloud detector of the precision edge metric still uses — plane, cylinder,
sphere — with the context they read. What remains of the surface-fit escalation ladder that left with
the meshing stack on 2026-10-17 (``escalate.py``; its swept-profile and B-spline stages went with their
modules): ``segmentation/perfect_object`` kept importing ``FITTERS`` / ``FitContext`` from the deleted
module, so the acta's edge metric failed at import on every certification since (pccr 2026-10-31).

Every fitter returns a model with the common surface protocol (``signed_distance``, ``to_uv``,
``uv_to_world``, ``params_dict``, ``DOF``) or None when the points do not hold that form."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .plane import PlaneModel, fit_plane


@dataclass
class FitContext:
    """Everything a fitter may need besides the points."""
    world_up: np.ndarray
    scene_centroid: Optional[np.ndarray] = None
    camera_path: Optional[np.ndarray] = None      # (M,3) — the walk, an axis hint for cylinders
    dist_thresh: float = 0.012
    ransac_iters: int = 500
    min_inlier_frac: float = 0.30


def _fit_plane_stage(pts: np.ndarray, ctx: FitContext) -> Optional[PlaneModel]:
    return fit_plane(pts, dist_thresh=ctx.dist_thresh, ransac_iters=ctx.ransac_iters,
                     min_inlier_frac=ctx.min_inlier_frac, world_up=ctx.world_up,
                     scene_centroid=ctx.scene_centroid)


def _fit_cylinder_stage(pts: np.ndarray, ctx: FitContext):
    from .quadric import fit_cylinder
    axis_hint = None
    if ctx.camera_path is not None and len(ctx.camera_path) >= 2:
        d = np.asarray(ctx.camera_path[-1], float) - np.asarray(ctx.camera_path[0], float)
        ln = np.linalg.norm(d)
        axis_hint = d / ln if ln > 1e-9 else None
    return fit_cylinder(pts, dist_thresh=max(ctx.dist_thresh, 0.02),
                        min_inlier_frac=0.5, axis_hint=axis_hint)


def _fit_sphere_stage(pts: np.ndarray, ctx: FitContext):
    from .quadric import fit_sphere
    return fit_sphere(pts, dist_thresh=max(ctx.dist_thresh, 0.02),
                      min_inlier_frac=0.5)


FITTERS: Dict[str, Callable[[np.ndarray, FitContext], Optional[object]]] = {
    "plane": _fit_plane_stage,
    "cylinder": _fit_cylinder_stage,
    "sphere": _fit_sphere_stage,
}
