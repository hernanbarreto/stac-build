"""
Stage 5 — hybrid scene orchestration.

Architectural classes (configurable label list: walls, floors, ceilings,
vaults, columns…) go through the surface_fit pipeline: consolidate → escalate
→ (scene-wide plane regularization + clean edges) → support-trimmed mesh +
fidelity record. Every other class keeps the existing TSDF crop path
untouched (``segmentation.tsdf_export.crop_scene_mesh_to_instances``) — this
module only fits what it owns and reports what it skipped.

Outputs per fitted instance (``output/surface_fit/<label>_<id>/``):
    surface.ply / surface.glb   support-trimmed fitted mesh (mvs-texturing
                                consumes plain triangle geometry)
    deviation.ply               original points colored by signed residual
    heatmap.png                 deviation map over the surface UV frame
    residuals.json              stats + findings (the fidelity deliverable)
    meta.json                   model params + mesh info + report
Plus ``output/surface_fit/scene_report.json`` with the regularization log and
the fitted/skipped split.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from reconstruction.elements import json_safe
from .models import FittedSurface
from .plane import PlaneModel
from .regularize import RegularizeReport, regularize_planes, snap_edges
from .runner import (build_surface_fit_kwargs, evaluate_model, export_artifacts,
                     fit_segment, load_instances, segment_points)

logger = logging.getLogger("SurfaceFit")

# SAM3 labels that get a fitted surface (matched as substrings, lowercase, so
# "partition wall" and "retaining wall" ride the "wall" entry)
DEFAULT_FITTED_ROLES = ("wall", "floor", "ceiling", "slab", "vault", "tunnel",
                        "column", "beam", "deck", "platform", "ramp")


def _role_is_fitted(label: str, fitted_roles: Sequence[str]) -> bool:
    ll = (label or "").lower()
    return any(r in ll for r in fitted_roles)


def _safe_name(label: str, instance_id) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (label or "segment"))
    return f"{safe}_{instance_id}"


def scene_config_kwargs(config: Optional[dict]) -> dict:
    """Scene-level knobs from config.yaml `surface_fit:` (the per-segment keys
    ride build_surface_fit_kwargs; these govern stages 3/5)."""
    scfg = (config or {}).get("surface_fit", {}) or {}
    out = {}
    for k in ("fitted_roles", "regularize", "regularize_angle_tol_deg",
              "coplanar_tol_mm", "snap_edge_dist_m"):
        if k in scfg:
            out[k] = scfg[k]
    return out


def fit_scene(session_dir: Path,
              out_base: Optional[Path] = None,
              config: Optional[dict] = None,
              overrides: Optional[dict] = None,
              fitted_roles: Sequence[str] = DEFAULT_FITTED_ROLES,
              regularize: bool = True,
              regularize_angle_tol_deg: float = 1.0,
              coplanar_tol_mm: float = 10.0,
              snap_edge_dist_m: float = 0.10,
              instance_ids: Optional[Sequence[int]] = None,
              progress_cb: Optional[Callable] = None) -> dict:
    """Fit every architectural instance of a session, regularize the planes
    scene-wide, and export the hybrid results. Returns the scene report."""
    t0 = time.time()
    session_dir = Path(session_dir)
    out_base = Path(out_base) if out_base else session_dir / "output" / "surface_fit"
    kwargs = build_surface_fit_kwargs(config, overrides)

    instances, cloud_pts, centroid, raw_pts = load_instances(session_dir)
    fitted: List[FittedSurface] = []
    seg_points: Dict[int, np.ndarray] = {}   # RAW measurement per instance
    skipped: List[dict] = []

    def _cb(**kw):
        if progress_cb:
            progress_cb(**kw)

    for inst in instances:
        iid = inst.get("instance_id", inst.get("id"))
        label = inst.get("label", "")
        if instance_ids is not None and iid not in instance_ids:
            continue
        if not _role_is_fitted(label, fitted_roles):
            skipped.append({"instance_id": iid, "label": label,
                            "reason": "non-architectural → TSDF path"})
            logger.info("scene: %s_%s → TSDF path (label not in fitted_roles)",
                        label, iid)
            continue
        pts = segment_points(inst, cloud_pts)
        raw_seg = segment_points(inst, raw_pts) if raw_pts is not None else pts
        _cb(instance_id=iid, phase="fit", elapsed=time.time() - t0)
        try:
            fs = fit_segment(pts, instance_id=iid, label=label,
                             original_xyz=raw_seg,
                             scene_centroid=centroid,
                             progress_cb=progress_cb, **kwargs)
        except Exception as e:
            logger.exception("scene: %s_%s failed: %s", label, iid, e)
            fs = None
        if fs is None:
            skipped.append({"instance_id": iid, "label": label,
                            "reason": "no model fitted → TSDF path"})
            continue
        fitted.append(fs)
        seg_points[iid] = raw_seg   # stage-4 reference = raw measurement

    # ── stage 3: scene-wide plane regularization + honest re-evaluation ──
    reg_report = RegularizeReport()
    if regularize and fitted:
        plane_idx = [i for i, f in enumerate(fitted)
                     if isinstance(f.model, PlaneModel)]
        if len(plane_idx) >= 2:
            models = [fitted[i].model for i in plane_idx]
            weights = [float(fitted[i].n_input_points) for i in plane_idx]
            snapped, reg_report = regularize_planes(
                models, weights=weights,
                angle_tol_deg=regularize_angle_tol_deg,
                coplanar_tol_mm=coplanar_tol_mm)
            eval_keys = {k: v for k, v in kwargs.items() if k in (
                "grid_cell_m", "morans_z_max", "morans_i_min", "structure_min_mm",
                "flatness_tol_mm", "flatness_span_m", "finding_dev_mm",
                "finding_min_area_m2", "tilt_mm_per_m", "support_radius_m",
                "mesh_resolution_m")}
            for i, m_new in zip(plane_idx, snapped):
                old = fitted[i]
                moved = float(np.rad2deg(np.arccos(np.clip(
                    old.model.normal @ m_new.normal, -1, 1))))
                d_shift = abs(m_new.d - (old.model.d if old.model.normal @ m_new.normal > 0
                                         else -old.model.d))
                if moved < 1e-9 and d_shift < 1e-9:
                    continue
                _cb(instance_id=old.instance_id, phase="regularize",
                    elapsed=time.time() - t0)
                fs2 = evaluate_model(
                    m_new, kind="plane", escalation_path=old.escalation_path,
                    orig=seg_points[old.instance_id],
                    instance_id=old.instance_id, label=old.label, **eval_keys)
                fs2.regularized = True
                fitted[i] = fs2
        snap_edges([f for f in fitted if isinstance(f.model, PlaneModel)],
                   snap_dist_m=snap_edge_dist_m, report=reg_report)

    # ── export ──
    results = []
    for fs in fitted:
        out_dir = out_base / _safe_name(fs.label, fs.instance_id)
        _cb(instance_id=fs.instance_id, phase="export", elapsed=time.time() - t0)
        export_artifacts(fs, seg_points[fs.instance_id], out_dir,
                         grid_cell_m=kwargs.get("grid_cell_m", 0.05),
                         heatmap_vmax_mm=kwargs.get("heatmap_vmax_mm"))
        _export_glb(fs, out_dir)
        results.append({"instance_id": fs.instance_id, "label": fs.label,
                        "kind": fs.kind, "regularized": fs.regularized,
                        "rms_mm": fs.report.stats.rms_mm,
                        "p95_mm": fs.report.stats.p95_mm,
                        "flatness_pass": fs.report.stats.flatness_pass,
                        "findings": len(fs.report.findings),
                        "dir": str(out_dir)})

    scene_report = json_safe({
        "fitted": results,
        "skipped_to_tsdf": skipped,
        "regularization": reg_report.to_dict(),
        "elapsed_s": round(time.time() - t0, 1),
    })
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "scene_report.json").write_text(json.dumps(scene_report, indent=2))
    logger.info("scene: %d fitted, %d → TSDF path, report → %s",
                len(results), len(skipped), out_base / "scene_report.json")
    return scene_report


def _export_glb(fs: FittedSurface, out_dir: Path) -> None:
    """surface.glb next to surface.ply (UI + downstream tooling parity with
    the TSDF crop outputs)."""
    if not len(fs.mesh_vertices):
        return
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=fs.mesh_vertices, faces=fs.mesh_faces,
                               process=False)
        mesh.export(str(out_dir / "surface.glb"))
    except Exception as e:  # GLB is a convenience copy; PLY is the artifact
        logger.warning("%s_%s: GLB export failed: %s", fs.label, fs.instance_id, e)
