"""
Stage 2 core — hierarchical model escalation.

Rule: always the model with the FEWEST degrees of freedom that explains the
data. A model "explains the data" when its residual field is spatially
unstructured (white noise) per Moran's I on the gridded residuals — see
``spatial_test``. If structure remains, escalate:

    plane → quadric (cylinder/sphere) → swept profile → B-spline

The escalation test runs against the *fitting* cloud (consolidated, if stage 1
ran); the stage-4 fidelity report is computed elsewhere against the original
cloud. Using the original here would re-detect the double layers that
consolidation exists to collapse, and over-escalate.

Every fitter returns a model object with the common surface protocol:
    signed_distance(pts) -> (N,)   metric residuals
    to_uv(pts) -> (N,2)            2-D parametrization for grids/meshing
    uv_to_world(uv) -> (N,3)
    params_dict() -> dict          JSON-safe parameters
    DOF: int
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .models import MoranResult
from .plane import PlaneModel, fit_plane
from .spatial_test import GridField, assess_field, grid_scalar

logger = logging.getLogger("SurfaceFit")

# Escalation ladder, lowest DOF first. Fitters registered by the stage modules;
# entries missing from FITTERS are skipped (lets stage 2.1 ship before 2.2-2.4).
MODEL_LADDER = ("plane", "cylinder", "sphere", "swept_profile", "bspline")


@dataclass
class FitContext:
    """Everything a stage fitter may need besides the points."""
    world_up: np.ndarray
    scene_centroid: Optional[np.ndarray] = None
    camera_path: Optional[np.ndarray] = None      # (M,3) for swept-axis hints
    dist_thresh: float = 0.012
    ransac_iters: int = 500
    min_inlier_frac: float = 0.30
    ctrl_point_spacing_m: float = 0.5             # B-spline low-pass knob
    section_spacing_m: float = 0.20               # swept-profile sampling


@dataclass
class EscalationResult:
    model: object                 # accepted surface-protocol model
    kind: str
    path: List[str]               # kinds tried, in order
    moran: MoranResult            # verdict of the accepted model (vs fit cloud)
    field: GridField              # residual grid of the accepted model (fit cloud)
    residual_structured: bool     # True = even the last model left structure


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


def _fit_swept_stage(pts: np.ndarray, ctx: FitContext):
    from .profile_sweep import fit_swept_profile
    return fit_swept_profile(pts, dist_thresh=max(ctx.dist_thresh, 0.02),
                             min_inlier_frac=0.5,
                             section_spacing_m=ctx.section_spacing_m,
                             ctrl_point_spacing_m=ctx.ctrl_point_spacing_m,
                             world_up=ctx.world_up, camera_path=ctx.camera_path)


def _fit_bspline_stage(pts: np.ndarray, ctx: FitContext):
    from .bspline import fit_bspline
    return fit_bspline(pts, ctrl_point_spacing_m=ctx.ctrl_point_spacing_m,
                       dist_thresh=max(ctx.dist_thresh, 0.02),
                       min_inlier_frac=0.5, world_up=ctx.world_up,
                       scene_centroid=ctx.scene_centroid)


FITTERS: Dict[str, Callable[[np.ndarray, FitContext], Optional[object]]] = {
    "plane": _fit_plane_stage,
    "cylinder": _fit_cylinder_stage,
    "sphere": _fit_sphere_stage,
    "swept_profile": _fit_swept_stage,
    "bspline": _fit_bspline_stage,
}


def escalate_fit(pts: np.ndarray, ctx: FitContext,
                 models: Sequence[str] = MODEL_LADDER,
                 grid_cell_m: float = 0.05,
                 morans_z_max: float = 4.0,
                 morans_i_min: float = 0.05,
                 structure_min_mm: float = 1.5) -> Optional[EscalationResult]:
    """Try models lowest-DOF-first; accept the first whose residuals carry no
    RELEVANT spatial structure. Two gates must both fire to escalate:
      - statistical: Moran's I says the residual field is not white noise;
      - metric: the structure's amplitude exceeds ``structure_min_mm`` —
        sub-millimetre systematic wiggles are honest but irrelevant against
        construction tolerances, and escalating on them trades a compact
        model for hundreds of DOF with no metric gain.
    If every model leaves relevant structure, keep the LOWEST-RMS one (not
    the last tried) and flag the report."""
    pts = np.asarray(pts, dtype=np.float64)
    tried: List[str] = []
    best: Optional[EscalationResult] = None
    best_rms = float("inf")

    for kind in models:
        fitter = FITTERS.get(kind)
        if fitter is None:
            logger.info("escalate: model '%s' not available yet — skipping", kind)
            continue
        model = fitter(pts, ctx)
        if model is None:
            logger.info("escalate: model '%s' did not fit — skipping", kind)
            continue
        tried.append(kind)
        signed = model.signed_distance(pts)
        field = grid_scalar(model.to_uv(pts), signed, grid_cell_m)
        moran = assess_field(field, z_max=morans_z_max, i_min=morans_i_min,
                             structure_min_mm=structure_min_mm)
        rms_mm = float(np.sqrt(np.mean(signed ** 2))) * 1000.0
        logger.info("escalate: %-13s dof=%-3d rms=%.2fmm  Moran I=%.3f z=%.1f "
                    "amp=%.1fmm → %s",
                    kind, getattr(model, "DOF", -1), rms_mm, moran.i, moran.z,
                    moran.amplitude_mm,
                    "structured, escalating" if moran.relevant else "ACCEPT")
        res = EscalationResult(model=model, kind=kind, path=list(tried),
                               moran=moran, field=field,
                               residual_structured=moran.relevant)
        if not moran.relevant:
            res.path = list(tried)
            return res
        if rms_mm < best_rms:
            best, best_rms = res, rms_mm

    if best is not None:
        best.path = list(tried)
        logger.warning("escalate: relevant residual structure remains — keeping "
                       "lowest-RMS model '%s' (%.2fmm) and flagging the report",
                       best.kind, best_rms)
    return best
