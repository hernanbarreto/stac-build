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

# USER DOCTRINE (2026-08-29, reaffirmed same day): the label is NEVER
# indicative of geometry — it may be a BIM object name, a bare code, a number.
# There is NO role/name routing here: every instance gets the same `models`
# ladder and GEOMETRY alone decides acceptance (accept_p95_mm gate).


def _safe_name(label: str, instance_id) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (label or "segment"))
    return f"{safe}_{instance_id}"


def scene_config_kwargs(config: Optional[dict]) -> dict:
    """Scene-level knobs from config.yaml `surface_fit:` (the per-segment keys
    ride build_surface_fit_kwargs; these govern stages 3/5)."""
    scfg = (config or {}).get("surface_fit", {}) or {}
    out = {}
    for k in ("regularize", "regularize_angle_tol_deg",
              "coplanar_tol_mm", "snap_edge_dist_m", "accept_p95_mm",
              "decompose_see_through", "see_through_behind_ratio",
              "unexplained_dist_m", "hole_audit", "hole_fill_ratio",
              "hole_open_ratio", "hole_occluded_ratio", "hole_min_votes",
              "hole_interp_max_cells", "texture_objects", "decompose",
              "decompose_min_points", "decompose_max_primitives",
              "decompose_inlier_dist_m", "decompose_min_frac",
              "contour_regularize", "contour_tol_m"):
        if k in scfg:
            out[k] = scfg[k]
    return out


def fit_scene(session_dir: Path,
              out_base: Optional[Path] = None,
              config: Optional[dict] = None,
              overrides: Optional[dict] = None,
              regularize: bool = True,
              regularize_angle_tol_deg: float = 1.0,
              coplanar_tol_mm: float = 10.0,
              snap_edge_dist_m: float = 0.10,
              accept_p95_mm: float = 60.0,
              decompose_see_through: bool = True,
              see_through_behind_ratio: float = 0.5,
              unexplained_dist_m: float = 0.10,
              hole_audit: bool = True,
              hole_fill_ratio: float = 0.6,
              hole_open_ratio: float = 0.3,
              hole_occluded_ratio: float = 0.5,
              hole_min_votes: int = 2,
              hole_interp_max_cells: int = 30,
              texture_objects: bool = True,
              decompose: bool = True,
              decompose_min_points: int = 8000,
              decompose_max_primitives: int = 6,
              decompose_inlier_dist_m: float = 0.04,
              decompose_min_frac: float = 0.10,
              contour_regularize: bool = True,
              contour_tol_m: float = 0.08,
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
    gi_map: Dict[int, np.ndarray] = {}       # global cloud indices per instance
    extra_parts: Dict[int, List[FittedSurface]] = {}  # decomposition primitives
    forced_leftover: Dict[int, np.ndarray] = {}       # decomposition residue (global idx)
    skipped: List[dict] = []

    _EVAL_KEYS = ("grid_cell_m", "morans_z_max", "morans_i_min",
                  "structure_min_mm", "flatness_tol_mm", "flatness_span_m",
                  "finding_dev_mm", "finding_min_area_m2", "tilt_mm_per_m",
                  "support_radius_m", "mesh_resolution_m", "support_dist_m")
    eval_kwargs = {k: v for k, v in kwargs.items() if k in _EVAL_KEYS}

    def _decompose_points(fit_pts: np.ndarray, orig_pts: np.ndarray,
                          iid: int, label: str, tag: str):
        """Iterative primitive extraction over a point set → list of
        FittedSurfaces (largest first) + boolean residue mask. None when
        nothing parametric was found."""
        from .decompose import extract_primitives
        from .escalate import FitContext
        # PERMISSIVE fitter gates — extraction has its OWN acceptance
        # (min_points / min_frac_remaining). With the fitter's internal gate at
        # the same 10% as ours, the train's first plane (~10.5% inliers) passed
        # or failed on the RANSAC seed; and 1.2 cm is too tight for large-scale
        # noise — the membership distance is the honest tolerance here.
        ctx = FitContext(world_up=np.array([0.0, 1.0, 0.0]),
                         scene_centroid=centroid,
                         dist_thresh=float(decompose_inlier_dist_m),
                         ransac_iters=int(kwargs.get("ransac_iters", 500)),
                         min_inlier_frac=0.02)
        parts, rem = extract_primitives(
            fit_pts, ctx,
            inlier_dist=float(decompose_inlier_dist_m),
            min_points=int(decompose_min_points),
            min_frac_remaining=float(decompose_min_frac),
            max_primitives=int(decompose_max_primitives),
            tag=tag)
        surfs: List[FittedSurface] = []
        for kind, model, mask in parts:
            try:
                surfs.append(evaluate_model(
                    model, kind=kind, escalation_path=[kind],
                    orig=orig_pts[mask], instance_id=iid, label=label,
                    **eval_kwargs))
            except Exception as e:  # noqa: BLE001
                logger.warning("decompose %s: %s part eval failed: %s",
                               tag, kind, e)
        if not surfs:
            return None
        surfs.sort(key=lambda s: -int(s.n_input_points))
        return surfs, rem

    def _cb(**kw):
        if progress_cb:
            progress_cb(**kw)

    for inst in instances:
        iid = inst.get("instance_id", inst.get("id"))
        label = inst.get("label", "")
        if instance_ids is not None and iid not in instance_ids:
            continue
        # USER DOCTRINE (2026-08-29): the label NEVER decides the technique —
        # it may be a code or a number. Every instance gets the same config
        # ladder, and GEOMETRY alone accepts: p95 of the residuals vs the
        # ORIGINAL points under accept_p95_mm — otherwise decomposition/TSDF.
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
        if fs is not None:
            p95 = float(fs.report.stats.p95_mm)
            if p95 > float(accept_p95_mm):
                logger.info("scene: %s_%s — %s REJECTED by geometry "
                            "(p95 %.1fmm > %.0fmm) — trying primitive "
                            "decomposition",
                            label, iid, fs.kind, p95, accept_p95_mm)
                fs = None
            else:
                logger.info("scene: %s_%s — %s accepted by geometry "
                            "(p95 %.1fmm)", label, iid, fs.kind, p95)
        # PRIMITIVE DECOMPOSITION (user 2026-08-29, the train): a complex
        # object that no single model explains still CONTAINS primitives —
        # extract them iteratively; only the residue goes to Poisson.
        if fs is None and decompose and len(pts) >= int(decompose_min_points):
            r2 = None
            try:
                r2 = _decompose_points(pts, raw_seg, iid, label,
                                       f"{label}_{iid}")
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s_%s decomposition failed: %s",
                               label, iid, e)
            if r2 is not None:
                surfs, rem = r2
                fs = surfs[0]
                fs.from_decompose = True   # phantom check target (see-through)
                if len(surfs) > 1:
                    extra_parts[int(iid)] = surfs[1:]
                gi_full = np.asarray(inst.get("globalIndices") or [],
                                     dtype=np.int64)
                if len(gi_full) == len(rem):
                    forced_leftover[int(iid)] = gi_full[rem]
                logger.info("scene: %s_%s decomposed into %d primitive(s) %s "
                            "+ %s residue pts → poisson", label, iid,
                            len(surfs), [s.kind for s in surfs],
                            f"{int(rem.sum()):,}")
        if fs is None:
            skipped.append({"instance_id": iid, "label": label,
                            "reason": "no model fitted → TSDF path"})
            continue
        fitted.append(fs)
        seg_points[iid] = raw_seg   # stage-4 reference = raw measurement
        gi_map[iid] = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)

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
                "mesh_resolution_m", "support_dist_m")}
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
        # ── Stage-1 hole audit (user 2026-08-29): every hole in the point
        # support is voted against the scan-frame masks — reconstruction gaps
        # get image-supported fills, real openings stay open with mask-shaped
        # borders. The residual REPORT below still uses only measured points.
        audit_rep = None
        audit_grid = None
        if hole_audit and fs.model is not None:
            try:
                from .hole_audit import audit_and_fill, save_report
                raw_seg = seg_points[fs.instance_id]
                signed = np.abs(np.asarray(fs.model.signed_distance(raw_seg)))
                sd = float(kwargs.get("support_dist_m", 0.04))
                uv_sup = np.asarray(fs.model.to_uv(raw_seg[signed <= sd]))
                r = audit_and_fill(
                    fs.model, uv_sup,
                    output_dir=session_dir / "output", session_dir=session_dir,
                    instance_id=int(fs.instance_id),
                    resolution=float(kwargs.get("mesh_resolution_m", 0.05)),
                    support_radius=float(kwargs.get("support_radius_m", 0.08)),
                    fill_ratio=float(hole_fill_ratio),
                    open_ratio=float(hole_open_ratio),
                    occluded_ratio=float(hole_occluded_ratio),
                    min_votes=int(hole_min_votes),
                    interp_max_cells=int(hole_interp_max_cells),
                    # planes interpolate safely anywhere on the model; curved
                    # models (bspline/swept/cylinder) only near their data
                    max_fill_dist_cells=None if fs.kind == "plane" else 3)
                if r is not None:
                    fs.mesh_vertices, fs.mesh_faces, audit_rep = r
                    audit_grid = audit_rep.pop("_grid", None)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_report(out_dir, audit_rep)
                    logger.info(
                        "scene: %s_%s hole audit — %d hole cells: %d filled "
                        "(%d image-supported + %d occlusion-inferred + %d "
                        "interpolated, %.2f m²), %d open (real openings, "
                        "%.2f m²), %d ambiguous",
                        fs.label, fs.instance_id, audit_rep["hole_cells"],
                        audit_rep["filled_cells"],
                        audit_rep["filled_image_supported"],
                        audit_rep["filled_occlusion_inferred"],
                        audit_rep["filled_interpolated"],
                        audit_rep["filled_area_m2"],
                        audit_rep["open_cells"], audit_rep["open_area_m2"],
                        audit_rep["ambiguous_cells"])
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s_%s hole audit failed (mesh keeps "
                               "point support only): %s",
                               fs.label, fs.instance_id, e)
        # ── Contour regularization (user 2026-08-29: "detectar la tendencia
        # de las formas para perfeccionarlas") — the audited occupancy is
        # vectorized and each contour tried against the 2-D shape ladder
        # (circle → rectangle → rounded rect → arch → snapped polygon → raw);
        # the mesh is rebuilt with CAD-crisp boundaries from whatever passed
        # its deviation gate. Planes + bsplines (user 2026-08-29: the
        # all-bspline regime lost the crisp opening cuts — the machinery is
        # pure UV-domain, so the curved surface gets the same clean borders;
        # uv_to_world lands the rebuilt grid back on the curved model).
        contour_rep = None
        if (contour_regularize and fs.kind in ("plane", "bspline")
                and audit_grid is not None and fs.model is not None):
            try:
                from .contours import regularize_mesh
                keepg, openg, gu0, gv0, gres = audit_grid
                rc = regularize_mesh(fs.model, keepg, openg, gu0, gv0, gres,
                                     tol=float(contour_tol_m))
                if rc is not None:
                    v_uv, f2, contour_rep = rc
                    fs.mesh_vertices = np.asarray(fs.model.uv_to_world(v_uv))
                    fs.mesh_faces = np.asarray(f2)
                    logger.info(
                        "scene: %s_%s contours — %s",
                        fs.label, fs.instance_id,
                        ", ".join(f"{c['role']}:{c['shape']}"
                                  f"(p95 {c.get('p95_dev_m', 0)*1000:.0f}mm)"
                                  for c in contour_rep))
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s_%s contour regularization failed "
                               "(grid mesh kept): %s", fs.label,
                               fs.instance_id, e)
        # 3-D sanity crop for CURVED models (user 2026-08-29: the ceiling kept
        # showing spikes): a mesh vertex farther than 25 cm (3-D) from any
        # measured on-surface point is spline behaviour, not measurement.
        if fs.kind != "plane" and fs.model is not None and len(fs.mesh_vertices):
            try:
                from scipy.spatial import cKDTree as _KD
                raw_seg = seg_points[fs.instance_id]
                signed = np.abs(np.asarray(fs.model.signed_distance(raw_seg)))
                sd = float(kwargs.get("support_dist_m", 0.04))
                onsurf = raw_seg[signed <= max(sd, 0.02)]
                if len(onsurf) > 100:
                    d3, _ = _KD(onsurf).query(np.asarray(fs.mesh_vertices), k=1)
                    bad = d3 > 0.25
                    if bad.any():
                        keep_v = ~bad
                        remap = np.full(len(fs.mesh_vertices), -1, dtype=np.int64)
                        remap[keep_v] = np.arange(int(keep_v.sum()))
                        f = np.asarray(fs.mesh_faces)
                        okf = keep_v[f].all(axis=1)
                        fs.mesh_faces = remap[f[okf]]
                        fs.mesh_vertices = np.asarray(fs.mesh_vertices)[keep_v]
                        logger.info("scene: %s_%s spike crop — dropped %d verts "
                                    ">25cm from measurement", fs.label,
                                    fs.instance_id, int(bad.sum()))
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s_%s spike crop failed: %s",
                               fs.label, fs.instance_id, e)
        # Unexplained remainder (2026-08-29, wall3): points the accepted model
        # does NOT explain. If large, DECOMPOSE it into primitives first (user
        # 2026-08-29: "donde se pueda aplicar ransac debe aplicarse"); only
        # what no primitive claims stays for Poisson.
        rest_n, rest_frac = 0, 0.0
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            gi = gi_map.get(fs.instance_id)
            raw_seg = seg_points[fs.instance_id]
            if fs.instance_id in forced_leftover:
                rest_idx = forced_leftover[fs.instance_id]
                rest_n = int(len(rest_idx))
                rest_frac = rest_n / max(len(raw_seg), 1)
                if rest_n:
                    np.save(out_dir / "unexplained_idx.npy", rest_idx)
            elif fs.model is not None and gi is not None and len(gi) == len(raw_seg):
                signed = np.abs(np.asarray(fs.model.signed_distance(raw_seg)))
                rest_mask = signed > float(unexplained_dist_m)
                rest_idx = gi[rest_mask]
                if (decompose and fs.instance_id not in extra_parts
                        and int(rest_mask.sum()) >= int(decompose_min_points)):
                    r2 = None
                    try:
                        r2 = _decompose_points(
                            raw_seg[rest_mask], raw_seg[rest_mask],
                            int(fs.instance_id), fs.label,
                            f"{fs.label}_{fs.instance_id}:rest")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("scene: %s_%s remainder decomposition "
                                       "failed: %s", fs.label, fs.instance_id, e)
                    if r2 is not None:
                        surfs2, rem2 = r2
                        extra_parts.setdefault(int(fs.instance_id), []).extend(surfs2)
                        rest_idx = rest_idx[rem2]
                        logger.info("scene: %s_%s remainder → %d primitive(s) "
                                    "%s; %s residue pts → poisson",
                                    fs.label, fs.instance_id, len(surfs2),
                                    [s.kind for s in surfs2],
                                    f"{len(rest_idx):,}")
                rest_n = int(len(rest_idx))
                rest_frac = rest_n / max(len(raw_seg), 1)
                if rest_n:
                    np.save(out_dir / "unexplained_idx.npy", rest_idx)
                    logger.info("scene: %s_%s: %d pts (%.0f%%) unexplained → "
                                "unexplained_idx.npy", fs.label, fs.instance_id,
                                rest_n, rest_frac * 100)
        except Exception as e:  # noqa: BLE001
            logger.warning("scene: %s_%s unexplained-remainder export failed: %s",
                           fs.label, fs.instance_id, e)
        # ── See-through phantom cut (user 2026-08-29: a decomposition plane
        # covered wall3's access arch). A RANSAC primitive's mesh can bridge
        # across open space between its real points; every decomposition-born
        # face is checked against the cloud Z-buffer — cameras consistently
        # seeing measured geometry BEHIND it (and never AT its depth) refute
        # it. Applies ONLY to decomposition surfaces: direct fits follow their
        # own data and don't bridge.
        def _phantom_cut(verts, faces, tag):
            if not decompose_see_through or not len(faces):
                return verts, faces
            try:
                from .hole_audit import see_through_filter
                r = see_through_filter(
                    np.asarray(verts), np.asarray(faces),
                    session_dir / "output", session_dir,
                    behind_ratio=float(see_through_behind_ratio))
                if r is None:
                    return verts, faces
                keepf, strep = r
                if keepf.all():
                    return verts, faces
                f2 = np.asarray(faces)[keepf]
                used = np.unique(f2)
                remap = np.full(len(verts), -1, dtype=np.int64)
                remap[used] = np.arange(len(used))
                logger.info("scene: %s see-through cut — %d/%d phantom "
                            "face(s) dropped", tag, strep["refuted"],
                            strep["faces"])
                return np.asarray(verts)[used], remap[f2]
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s see-through cut failed: %s", tag, e)
                return verts, faces

        if getattr(fs, "from_decompose", False):
            fs.mesh_vertices, fs.mesh_faces = _phantom_cut(
                fs.mesh_vertices, fs.mesh_faces,
                f"{fs.label}_{fs.instance_id}:primary")
        # merge the decomposition primitives into the instance mesh
        parts_meta = []
        if extra_parts.get(int(fs.instance_id)):
            vlist = [np.asarray(fs.mesh_vertices)]
            flist = [np.asarray(fs.mesh_faces)]
            off = len(vlist[0])
            for p in extra_parts[int(fs.instance_id)]:
                v = np.asarray(p.mesh_vertices)
                f = np.asarray(p.mesh_faces)
                if not len(v) or not len(f):
                    continue
                v, f = _phantom_cut(v, f,
                                    f"{fs.label}_{fs.instance_id}:{p.kind}")
                if not len(f):
                    logger.info("scene: %s_%s part %s fully refuted by "
                                "see-through — dropped", fs.label,
                                fs.instance_id, p.kind)
                    continue
                v = np.asarray(v)
                f = np.asarray(f)
                vlist.append(v)
                flist.append(f + off)
                off += len(v)
                parts_meta.append({"kind": p.kind,
                                   "points": int(p.n_input_points),
                                   "rms_mm": p.report.stats.rms_mm})
            if len(vlist) > 1:
                fs.mesh_vertices = np.concatenate(vlist)
                fs.mesh_faces = np.concatenate(flist)
                logger.info("scene: %s_%s merged %d primitive part(s) into "
                            "the mesh", fs.label, fs.instance_id, len(parts_meta))
        # Stage-2 silhouette agreement: measure (never edit) how the FINAL
        # surface matches the frames — low agreement flags a shape/curve that
        # does not look like what the images show.
        sil = None
        if hole_audit and fs.model is not None and len(fs.mesh_vertices):
            try:
                from .hole_audit import silhouette_report, save_report
                sil = silhouette_report(np.asarray(fs.mesh_vertices),
                                        session_dir / "output", session_dir,
                                        int(fs.instance_id))
                if sil:
                    logger.info("scene: %s_%s silhouette agreement — IoU %.2f, "
                                "precision %.2f, recall %.2f (%d frames)",
                                fs.label, fs.instance_id, sil["mean_iou"],
                                sil["mean_precision"], sil["mean_recall"],
                                len(sil["frames"]))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    save_report(out_dir, {**(audit_rep or {}),
                                          "contours": contour_rep,
                                          "silhouette": sil})
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s_%s silhouette metric failed: %s",
                               fs.label, fs.instance_id, e)
        export_artifacts(fs, seg_points[fs.instance_id], out_dir,
                         grid_cell_m=kwargs.get("grid_cell_m", 0.05),
                         heatmap_vmax_mm=kwargs.get("heatmap_vmax_mm"))
        _export_glb(fs, out_dir)
        # Stage-3: real texture from the scan frames (user: "siempre con
        # textura, con los colores correctos") — texrecon atlas over the
        # fitted (and hole-audited) mesh, in place on surface.glb.
        if texture_objects and (out_dir / "surface.glb").exists():
            try:
                from reconstruction.texture_bake import bake_object_glb
                _cb(instance_id=fs.instance_id, phase="texture",
                    elapsed=time.time() - t0)
                bake_object_glb(out_dir / "surface.glb", session_dir,
                                session_dir / "output")
            except Exception as e:  # noqa: BLE001
                logger.warning("scene: %s_%s texture bake failed (mesh kept "
                               "untextured): %s", fs.label, fs.instance_id, e)
        results.append({"instance_id": fs.instance_id, "label": fs.label,
                        "kind": fs.kind, "regularized": fs.regularized,
                        "rms_mm": fs.report.stats.rms_mm,
                        "p95_mm": fs.report.stats.p95_mm,
                        "flatness_pass": fs.report.stats.flatness_pass,
                        "findings": len(fs.report.findings),
                        "unexplained_points": rest_n,
                        "unexplained_frac": round(rest_frac, 4),
                        "parts": parts_meta,
                        "contours": contour_rep,
                        "hole_audit": ({k: audit_rep[k] for k in
                                        ("hole_cells", "filled_cells",
                                         "filled_image_supported",
                                         "filled_occlusion_inferred",
                                         "open_cells", "ambiguous_cells",
                                         "filled_area_m2", "open_area_m2")}
                                       if audit_rep else None),
                        "silhouette": ({k: sil[k] for k in
                                        ("mean_iou", "mean_precision",
                                         "mean_recall")} if sil else None),
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
