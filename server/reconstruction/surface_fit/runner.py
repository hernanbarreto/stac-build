"""
Per-segment orchestration: consolidated-or-raw points in → FittedSurface out,
with all stage-4 artifacts written next to the mesh.

Config plumbing follows the repo convention (see tsdf_export.build_tsdf_scene_
kwargs): the valid config keys ARE the keyword parameters of ``fit_segment``,
derived by introspection, so config.yaml's ``surface_fit:`` section and the
function signature can never drift apart.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from reconstruction.elements import json_safe
from .escalate import FitContext, escalate_fit
from .heatmap import pick_vmax_mm, write_deviation_ply, write_heatmap_png
from .models import FittedSurface, ResidualReport
from .residuals import compute_stats, detect_findings
from .spatial_test import assess_field, grid_scalar
from .support import mesh_on_surface

logger = logging.getLogger("SurfaceFit")

_UP = np.array([0.0, 0.0, 1.0])

# runtime-only parameters — everything else in fit_segment's signature is a
# config key (single source of truth, mirrored into config.yaml `surface_fit:`)
_RUNTIME_ARGS = {"xyz", "instance_id", "label", "original_xyz", "world_up",
                 "scene_centroid", "camera_path", "out_dir", "progress_cb"}


def fit_segment(xyz: np.ndarray,
                *,
                instance_id: Optional[int] = None,
                label: str = "",
                original_xyz: Optional[np.ndarray] = None,
                world_up: Optional[np.ndarray] = None,
                scene_centroid: Optional[np.ndarray] = None,
                camera_path: Optional[np.ndarray] = None,
                out_dir: Optional[Path] = None,
                progress_cb: Optional[Callable] = None,
                # ── config knobs (config.yaml `surface_fit:`) ──
                models: Sequence[str] = ("plane", "cylinder", "sphere",
                                         "swept_profile", "bspline"),
                consolidate_method: str = "auto",   # auto|wlop|mls|none (stage 1)
                consolidate_radius_m: float = 0.06,
                wlop_select_percentage: float = 25.0,
                wlop_iterations: int = 30,
                plane_dist_thresh: float = 0.012,
                ransac_iters: int = 500,
                min_inlier_frac: float = 0.30,
                max_fit_points: int = 400_000,
                grid_cell_m: float = 0.05,
                morans_z_max: float = 4.0,
                morans_i_min: float = 0.05,
                structure_min_mm: float = 1.5,
                flatness_tol_mm: float = 5.0,
                flatness_span_m: float = 2.0,
                finding_dev_mm: float = 5.0,
                finding_min_area_m2: float = 0.05,
                tilt_mm_per_m: float = 3.0,
                support_radius_m: float = 0.08,
                mesh_resolution_m: float = 0.05,
                heatmap_vmax_mm: Optional[float] = None,
                ctrl_point_spacing_m: float = 0.5,
                section_spacing_m: float = 0.20,
                ) -> Optional[FittedSurface]:
    """Fit the lowest-DOF smooth surface to one SAM3 segment.

    ``xyz`` is the cloud to FIT (consolidated when stage 1 ran); ``original_xyz``
    is the cloud to MEASURE against (defaults to ``xyz`` — always pass the raw
    calibrated segment when they differ; residual fidelity is the deliverable).
    Returns None when no model fits (segment too small / degenerate).
    """
    t0 = time.time()
    xyz = np.asarray(xyz, dtype=np.float64)
    orig = xyz if original_xyz is None else np.asarray(original_xyz, dtype=np.float64)
    up = _UP if world_up is None else np.asarray(world_up, dtype=np.float64)
    name = f"{label or 'segment'}_{instance_id if instance_id is not None else '?'}"

    def _cb(phase: str):
        if progress_cb:
            progress_cb(instance_id=instance_id, phase=phase,
                        elapsed=time.time() - t0)

    if len(xyz) < 50:
        logger.warning("%s: only %d points — skipping", name, len(xyz))
        return None

    # ── stage 1: consolidate the FITTING cloud (residuals still use `orig`) ──
    if consolidate_method != "none" and len(xyz) >= 500:
        _cb("consolidate")
        from .consolidate import consolidate
        xyz = consolidate(xyz, method=consolidate_method,
                          neighbor_radius_m=consolidate_radius_m,
                          wlop_select_percentage=wlop_select_percentage,
                          wlop_iterations=wlop_iterations)

    # fit on a bounded random subsample (deterministic); measure on everything
    fit_pts = xyz
    if len(xyz) > max_fit_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_fit_points, replace=False)
        fit_pts = xyz[sel]
        logger.info("%s: fitting on %d of %d points (residuals use all)",
                    name, max_fit_points, len(xyz))

    _cb("fit")
    ctx = FitContext(world_up=up, scene_centroid=scene_centroid,
                     camera_path=camera_path, dist_thresh=plane_dist_thresh,
                     ransac_iters=ransac_iters, min_inlier_frac=min_inlier_frac,
                     ctrl_point_spacing_m=ctrl_point_spacing_m,
                     section_spacing_m=section_spacing_m)
    esc = escalate_fit(fit_pts, ctx, models=models, grid_cell_m=grid_cell_m,
                       morans_z_max=morans_z_max, morans_i_min=morans_i_min,
                       structure_min_mm=structure_min_mm)
    if esc is None:
        logger.warning("%s: no model could be fitted", name)
        return None

    _cb("residuals")
    fitted = evaluate_model(
        esc.model, kind=esc.kind, escalation_path=esc.path, orig=orig,
        instance_id=instance_id, label=label, world_up=up,
        grid_cell_m=grid_cell_m, morans_z_max=morans_z_max,
        morans_i_min=morans_i_min, structure_min_mm=structure_min_mm,
        flatness_tol_mm=flatness_tol_mm, flatness_span_m=flatness_span_m,
        finding_dev_mm=finding_dev_mm, finding_min_area_m2=finding_min_area_m2,
        tilt_mm_per_m=tilt_mm_per_m, support_radius_m=support_radius_m,
        mesh_resolution_m=mesh_resolution_m)

    if out_dir is not None:
        _cb("export")
        export_artifacts(fitted, orig, Path(out_dir),
                         grid_cell_m=grid_cell_m, heatmap_vmax_mm=heatmap_vmax_mm)
    _cb("done")
    return fitted


def evaluate_model(model, *, kind: str, escalation_path: List[str],
                   orig: np.ndarray, instance_id=None, label: str = "",
                   world_up=None, grid_cell_m: float = 0.05,
                   morans_z_max: float = 4.0, morans_i_min: float = 0.05,
                   structure_min_mm: float = 1.5, flatness_tol_mm: float = 5.0,
                   flatness_span_m: float = 2.0, finding_dev_mm: float = 5.0,
                   finding_min_area_m2: float = 0.05, tilt_mm_per_m: float = 3.0,
                   support_radius_m: float = 0.08,
                   mesh_resolution_m: float = 0.05) -> FittedSurface:
    """Stage 4 + per-segment mesh for a given surface model: full residual
    report vs the ORIGINAL cloud and the support-trimmed mesh. Reused by
    fit_segment and by the scene-level regularization rebuild (stage 3)."""
    up = _UP if world_up is None else np.asarray(world_up, dtype=np.float64)
    name = f"{label or 'segment'}_{instance_id if instance_id is not None else '?'}"

    signed = model.signed_distance(orig)
    uv = model.to_uv(orig)
    field = grid_scalar(uv, signed, grid_cell_m)
    moran = assess_field(field, z_max=morans_z_max, i_min=morans_i_min,
                         structure_min_mm=structure_min_mm)
    stats = compute_stats(signed, field, flatness_tol_mm=flatness_tol_mm,
                          flatness_span_m=flatness_span_m, moran=moran)
    findings = detect_findings(
        field, uv_to_world=model.uv_to_world, world_up=up,
        normal=getattr(model, "normal", up),
        finding_dev_mm=finding_dev_mm,
        finding_min_area_m2=finding_min_area_m2,
        tilt_mm_per_m=tilt_mm_per_m)
    report = ResidualReport(stats=stats, findings=findings)
    logger.info("%s: %s dof=%d rms=%.2fmm p95=%.2fmm max=%.2fmm flatness(%0.0fmm/%.1fm)=%s findings=%d",
                name, kind, getattr(model, "DOF", -1), stats.rms_mm,
                stats.p95_mm, stats.max_mm, flatness_tol_mm, flatness_span_m,
                {True: "PASS", False: "FAIL", None: "n/a"}[stats.flatness_pass],
                len(findings))

    verts, faces, support_frac, area = mesh_on_surface(
        uv, model.uv_to_world, resolution=mesh_resolution_m,
        support_radius=support_radius_m)
    logger.info("%s: mesh %d verts / %d faces, %.2f m² supported (%.0f%% of UV bbox)",
                name, len(verts), len(faces), area, 100.0 * support_frac)

    return FittedSurface(
        kind=kind, params=model.params_dict(),
        mesh_vertices=verts, mesh_faces=faces, report=report,
        instance_id=instance_id, label=label, n_input_points=int(len(orig)),
        support_fraction=support_frac, escalation_path=list(escalation_path),
        dof=int(getattr(model, "DOF", 0)), model=model)


# ── artifact export ─────────────────────────────────────────────────

def export_artifacts(fitted: FittedSurface, orig_pts: np.ndarray, out_dir: Path,
                     grid_cell_m: float = 0.05,
                     heatmap_vmax_mm: Optional[float] = None) -> None:
    """surface.ply + deviation.ply + heatmap.png + residuals.json + meta.json"""
    import open3d as o3d
    model = fitted.model
    signed_m = model.signed_distance(orig_pts)
    uv = model.to_uv(orig_pts)
    field = grid_scalar(uv, signed_m, grid_cell_m)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{fitted.label or 'segment'}_{fitted.instance_id}"

    if len(fitted.mesh_vertices):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(fitted.mesh_vertices),
            o3d.utility.Vector3iVector(fitted.mesh_faces.astype(np.int32)))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / "surface.ply"), mesh)

    signed_mm = signed_m * 1000.0
    vmax = pick_vmax_mm(signed_mm, heatmap_vmax_mm)
    fitted.report.deviation_ply = write_deviation_ply(
        orig_pts, signed_m, out_dir / "deviation.ply", vmax_mm=vmax)

    st = fitted.report.stats
    subtitle = (f"{len(orig_pts):,} pts   rms {st.rms_mm:.2f} mm   p95 {st.p95_mm:.2f} mm   "
                f"max {st.max_mm:.1f} mm   flatness {st.flatness_worst_mm:.1f} mm"
                f"/{st.flatness_span_m:.0f} m ({'PASS' if st.flatness_pass else 'FAIL'})"
                if st.flatness_worst_mm is not None else
                f"{len(orig_pts):,} pts   rms {st.rms_mm:.2f} mm   p95 {st.p95_mm:.2f} mm")
    f_uv = None
    if fitted.report.findings:
        f_uv = model.to_uv(np.array([f.center_xyz for f in fitted.report.findings]))
    fitted.report.heatmap_png = write_heatmap_png(
        field, out_dir / "heatmap.png", vmax_mm=vmax,
        title=f"{name} — {fitted.kind} (dof {fitted.dof})", subtitle=subtitle,
        findings=fitted.report.findings, findings_uv=f_uv)

    rj = out_dir / "residuals.json"
    rj.write_text(json.dumps(fitted.report.to_dict(), indent=2))
    fitted.report.json_path = str(rj)
    (out_dir / "meta.json").write_text(json.dumps(fitted.to_meta(), indent=2))
    logger.info("%s: artifacts → %s", name, out_dir)


# ── config plumbing (repo pattern: signature = single source of truth) ──

def surface_fit_config_keys() -> List[str]:
    sig = inspect.signature(fit_segment)
    return [p for p in sig.parameters if p not in _RUNTIME_ARGS]


def build_surface_fit_kwargs(config: Optional[dict],
                             overrides: Optional[dict] = None) -> Dict:
    keys = surface_fit_config_keys()
    scfg = (config or {}).get("surface_fit", {}) or {}
    kw = {k: scfg[k] for k in keys if k in scfg}
    if overrides:
        kw.update({k: v for k, v in overrides.items() if k in keys})
    return kw


# ── session loading (CLI / worker entry) ────────────────────────────

def load_instances(session_dir: Path):
    """Yield per-instance segments from a session's cleaned cloud.

    Returns (instances, points, scene_centroid, raw_points): the parsed
    instance dicts of segmentation_result.json, the full cleaned_cloud points
    (N,3), its centroid (normal-orientation hint for walls), and — when the
    scene-level consolidation ran — the UNTOUCHED measurement from
    cleaned_cloud_raw.ply (same count/order, so globalIndices index both).
    Stage-4 residuals must use the raw cloud (charter: never measure fidelity
    against the consolidated one).
    """
    import open3d as o3d
    session_dir = Path(session_dir)
    out = session_dir / "output"
    seg_path = out / "segmentation_result.json"
    cloud_path = out / "cleaned_cloud.ply"
    if not seg_path.exists():
        raise FileNotFoundError(f"missing {seg_path} — run SAM3 segmentation first")
    if not cloud_path.exists():
        raise FileNotFoundError(f"missing {cloud_path} — run CloudComPy postprocess first")
    seg = json.loads(seg_path.read_text())
    instances = seg.get("instances") or []
    pts = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
    raw_path = out / "cleaned_cloud_raw.ply"
    raw = None
    if raw_path.exists():
        raw = np.asarray(o3d.io.read_point_cloud(str(raw_path)).points)
        if len(raw) != len(pts):
            logger.warning("cleaned_cloud_raw.ply size mismatch (%d vs %d) — "
                           "ignoring raw reference", len(raw), len(pts))
            raw = None
    logger.info("session %s: %d instances, cloud %s%s", session_dir.name,
                len(instances), cloud_path.name,
                " (+raw reference)" if raw is not None else "")
    return instances, pts, pts.mean(0), raw


def segment_points(instance: dict, cloud_pts: np.ndarray) -> np.ndarray:
    idx = np.asarray(instance.get("globalIndices") or [], dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(cloud_pts))]
    return cloud_pts[idx]
