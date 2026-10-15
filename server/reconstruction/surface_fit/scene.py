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
from .runner import (_UP, build_surface_fit_kwargs, evaluate_model, export_artifacts,
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
              "contour_regularize", "contour_tol_m", "manhattan", "manhattan_snap_deg", "manhattan_min_inlier_frac", "manhattan_part_min_frac",
              "manhattan_bar_aspect_min", "manhattan_bar_round_tol"):
        if k in scfg:
            out[k] = scfg[k]
    return out


def _measured_part(fs, kind: str, V: np.ndarray, F: np.ndarray, n_points: int, rec: dict):
    """A mesh that joins the object's parts as measured (no fitted model, no snap)."""
    from types import SimpleNamespace
    p = SimpleNamespace(kind=kind, mesh_vertices=np.asarray(V, np.float64), mesh_faces=np.asarray(F, np.int64),
                        n_input_points=int(n_points), model=None, instance_id=fs.instance_id, label=fs.label,
                        report=SimpleNamespace(stats=SimpleNamespace(rms_mm=None)), manhattan=rec)
    return p


def _residue_components(pts: np.ndarray, mask: np.ndarray, cell: float):
    """Connected pieces of the residue points: 3-D occupancy at ``cell``, 26-connected."""
    from scipy import ndimage as _ndi
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    P = np.asarray(pts[idx], np.float64)
    o = P.min(0)
    c = np.floor((P - o) / cell).astype(np.int64)
    occ = np.zeros(tuple(int(x) + 1 for x in c.max(0)), dtype=bool)
    occ[c[:, 0], c[:, 1], c[:, 2]] = True
    lab, n = _ndi.label(occ, structure=np.ones((3, 3, 3), dtype=bool))
    comp = lab[c[:, 0], c[:, 1], c[:, 2]]
    out = []
    for k in range(1, n + 1):
        sel = comp == k
        m = np.zeros(len(pts), dtype=bool)
        m[idx[sel]] = True
        out.append(m)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def _bar_cylinder(P: np.ndarray, frame_axes: dict, sd: float, snap_deg: float,
                  aspect_min: float, round_tol: float):
    """A BAR (one long axis, a section of centimetres) as a straight cylinder on the frame:
    axis = the PCA axis snapped to the nearest frame axis within snap_deg, radius = the median
    radial distance; None when the piece is not a bar or its section is not round (radial
    p90 − p10 over round_tol support distances). Returns (CylinderModel, record)."""
    from reconstruction.geometry.primitives import _angular_extent, _orthobasis
    from .quadric import CylinderModel
    P = np.asarray(P, np.float64)
    if len(P) < 50:
        return None
    ctr = P.mean(0)
    _, _, vt = np.linalg.svd(P - ctr, full_matrices=False)
    e1 = vt[0] / max(float(np.linalg.norm(vt[0])), 1e-12)
    s_ = (P - ctr) @ e1
    length = float(np.percentile(s_, 95) - np.percentile(s_, 5))
    rad0 = np.linalg.norm((P - ctr) - np.outer(s_, e1), axis=1)
    diameter = 2.0 * float(np.percentile(rad0, 95))
    if diameter <= 1e-6 or length < float(aspect_min) * diameter:
        return None
    best = None
    for name, a in frame_axes.items():
        ang = float(np.rad2deg(np.arccos(np.clip(abs(e1 @ a), 0.0, 1.0))))
        if best is None or ang < best[2]:
            best = (name, a * (1.0 if e1 @ a >= 0 else -1.0), ang)
    axis_name, axis, moved = best
    snapped = moved <= float(snap_deg)
    if not snapped:
        axis, axis_name = e1, None
    q = P - ctr
    s2 = q @ axis
    rad = np.linalg.norm(q - np.outer(s2, axis), axis=1)
    radius = float(np.median(rad))
    spread = float(np.percentile(rad, 90) - np.percentile(rad, 10))
    if radius <= 1e-4 or spread > float(round_tol) * sd:
        return None
    u, v = _orthobasis(axis)
    ang_pts = np.arctan2((q - np.outer(s2, axis)) @ v, (q - np.outer(s2, axis)) @ u)
    t0, t1 = _angular_extent(ang_pts)
    d = rad - radius
    inl = np.abs(d) <= sd
    model = CylinderModel(axis_point=ctr.copy(), axis_dir=np.asarray(axis, np.float64), radius=radius,
                          theta_ref=u, theta0=float(t0), theta_span=float(t1 - t0),
                          rms=float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("inf"),
                          inlier_frac=float(inl.mean()), n_points=int(len(P)))
    rec = {"kind": "cylinder", "n_points": int(len(P)), "axis": axis_name, "snapped": bool(snapped),
           "moved_deg": round(moved, 4), "radius_m": round(radius, 4), "length_m": round(length, 4),
           "radial_spread_m": round(spread, 4)}
    return model, rec


def _manhattan_residue(pts: np.ndarray, rmask: np.ndarray, frame_axes: dict, sd: float, snap_deg: float,
                       part_min_frac: float, aspect_min: float, round_tol: float,
                       iid: int, label: str, eval_kwargs: dict):
    """The rest of a marked object: its connected pieces, each a straight cylinder when it is a
    round bar (evaluated honestly on its own points), the others pooled for the TSDF.
    Returns (parts, record) — record['_tsdf_mask'] = the points left for the TSDF."""
    n_obj = max(len(pts), 1)
    parts, cyl_recs = [], []
    tsdf_mask = np.zeros(len(pts), dtype=bool)
    for comp in _residue_components(pts, rmask, 2.0 * sd):
        if int(comp.sum()) < max(int(part_min_frac * n_obj), 100):
            tsdf_mask |= comp
            continue
        r = _bar_cylinder(pts[comp], frame_axes, sd, snap_deg, aspect_min, round_tol)
        if r is None:
            tsdf_mask |= comp
            continue
        model, rec = r
        try:
            fk = evaluate_model(model, kind="cylinder", escalation_path=["manhattan_bar"],
                                orig=pts[comp], instance_id=iid, label=label, **eval_kwargs)
        except Exception as e:  # noqa: BLE001 — the piece goes to the TSDF, declared
            rec["unmeshed"] = f"{type(e).__name__}: {e}"
            cyl_recs.append(rec)
            tsdf_mask |= comp
            continue
        fk.from_decompose = True
        fk.manhattan = rec
        parts.append(fk)
        cyl_recs.append(rec)
    return parts, {"cylinders": cyl_recs, "_tsdf_mask": tsdf_mask}


def _plane_components(pts: np.ndarray, mask: np.ndarray, model, cell: float):
    """The connected pieces of a plane part: occupancy of its points' UV at ``cell``,
    8-connected labelling (scipy.ndimage), one boolean mask over ``pts`` per piece, the
    largest first."""
    from scipy import ndimage as _ndi
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    uv = np.asarray(model.to_uv(np.asarray(pts[idx], np.float64)), np.float64)
    u0, v0 = uv.min(0)
    ii = np.floor((uv[:, 0] - u0) / cell).astype(np.int64)
    jj = np.floor((uv[:, 1] - v0) / cell).astype(np.int64)
    occ = np.zeros((int(jj.max()) + 1, int(ii.max()) + 1), dtype=bool)
    occ[jj, ii] = True
    lab, n = _ndi.label(occ, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return [mask.copy()]
    comp_of_pt = lab[jj, ii]
    out = []
    for c in range(1, n + 1):
        sel = comp_of_pt == c
        m = np.zeros(len(pts), dtype=bool)
        m[idx[sel]] = True
        out.append(m)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def _is_thin(pts: np.ndarray, model, sd: float) -> bool:
    """A plane part is a SURFACE of the object, not a slice through a lump: the object's points
    on the plane's shoulder (sd < |d| ≤ 2·sd) are at most half of those in its core (|d| ≤ sd)."""
    d = np.abs(np.asarray(model.signed_distance(np.asarray(pts, np.float64))))
    core = int((d <= sd).sum())
    shoulder = int(((d > sd) & (d <= 2.0 * sd)).sum())
    return core > 0 and shoulder <= 0.5 * core


def _panel_width(P: np.ndarray, model) -> float:
    """The narrowest in-plane extent (p5–p95 along the in-plane principal directions) of a
    plane part's points — what tells a panel from a bar lying within the tolerance."""
    uv = np.asarray(model.to_uv(np.asarray(P, np.float64)), np.float64)
    if len(uv) < 10:
        return 0.0
    c = uv - uv.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ext = []
    for ax in vt:
        q = c @ ax
        ext.append(float(np.percentile(q, 95) - np.percentile(q, 5)))
    return float(min(ext))


def _session_up(output_dir: Path) -> np.ndarray:
    """The session's UP in the cloud's own frame: the display frame is three.js Y-up (the floor
    leveling writes floor_transform.npz so that display = s·R·raw + t puts the floor at y = 0),
    so raw up = Rᵀ·ŷ — the identity on a baked-orientation session (pccr: the raw cloud IS Y-up,
    and the fit's default z-up read its floor as a wall, 2026-10-15). A session with no floor
    transform keeps the fit's default (the synthetic tests)."""
    p = Path(output_dir) / "floor_transform.npz"
    if not p.exists():
        return _UP
    d = np.load(p)
    R = np.asarray(d["R"], np.float64)
    up = R.T @ np.array([0.0, 1.0, 0.0])
    return up / max(float(np.linalg.norm(up)), 1e-12)


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
              manhattan: bool = True,
              manhattan_snap_deg: float = 15.0,
              manhattan_min_inlier_frac: float = 0.75,
              manhattan_part_min_frac: float = 0.05,
              manhattan_bar_aspect_min: float = 4.0,
              manhattan_bar_round_tol: float = 2.0,
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

    # MANHATTAN (USER 2026-10-15): every object is marked unless the user unmarked it
    # (segmentation.manhattan); a marked object is fitted as a PLANE and, below, snapped to the
    # scene frame; one that is not planar (p95 over the gate) says so and follows the ladder
    from segmentation import manhattan as _MAN
    man_off = _MAN.off_set(session_dir / "output") if manhattan else set()
    man_up = _session_up(session_dir / "output")
    man_planes: set = set()
    man_rejected: Dict[int, float] = {}
    man_report = None

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
        man_on = bool(manhattan) and int(iid) not in man_off
        fs = None
        try:
            if man_on:
                # a PLANE needs no consolidation: RANSAC + the median offset are robust to the
                # double layers WLOP collapses for the splines, and the residual report uses
                # the raw points anyway — WLOP was 171 s of the floor's 3 min (pccr 2026-10-15);
                # the session's own up (floor_transform.npz) orients it
                fs = fit_segment(pts, instance_id=iid, label=label,
                                 original_xyz=raw_seg,
                                 scene_centroid=centroid, world_up=man_up,
                                 progress_cb=progress_cb,
                                 **{**kwargs, "models": ["plane"], "consolidate_method": "none",
                                    "max_fit_points": 0})
                if fs is None:
                    man_rejected[int(iid)] = 0.0     # no plane gathers its points at all
                    logger.info("scene: %s_%s — marked Manhattan but no plane explains it: the "
                                "configured ladder decides", label, iid)
                if fs is not None:
                    # A marked object IS a surface when its planes explain most of its points
                    # within the support distance — the SHARE, not the p95: pccr's floor is 87 %
                    # within 4 cm of its plane and its p95 is 88 mm (chair bases, clutter, the
                    # 7.6 cm undulation the certification measured) — that p95 is the FINDING a
                    # straight BIM floor reports, not a reason to model the floor as a spline.
                    sd = float(kwargs.get("support_dist_m", 0.04))
                    within = (float(np.mean(np.abs(np.asarray(fs.model.signed_distance(raw_seg))) <= sd))
                              if len(raw_seg) else 0.0)
                    # MANHATTAN DECOMPOSITION (USER 2026-10-15, the S-shaped wall), ALWAYS tried:
                    # planes, the largest first, down to manhattan_part_min_frac of the object
                    # (a desk's side panels, an S-wall's returns — never a floor's clutter);
                    # each part is snapped to the frame below (parallel runs on one axis with
                    # their exact offsets, the returns perpendicular), coplanar runs one plane,
                    # exact edges between them.
                    from .decompose import extract_primitives
                    from .escalate import FitContext
                    n_obj = max(len(pts), 1)
                    part_min = max(int(float(manhattan_part_min_frac) * n_obj), 100)
                    ctx_m = FitContext(world_up=man_up, scene_centroid=centroid,
                                       dist_thresh=sd,
                                       ransac_iters=int(kwargs.get("ransac_iters", 500)),
                                       min_inlier_frac=0.02)
                    parts, rem = extract_primitives(
                        pts, ctx_m, inlier_dist=sd, min_points=part_min,
                        min_frac_remaining=float(manhattan_part_min_frac),
                        max_primitives=int(decompose_max_primitives),
                        tag=f"manhattan {label}_{iid}", kinds=("plane",))
                    # a plane claims whatever lies within the tolerance — two legs 1.4 m apart
                    # in one vertical plane, a run and its coplanar twin — so each part is split
                    # into its CONNECTED pieces in the plane (occupancy at 2 support distances),
                    # and a piece is a PANEL when it is at least four support distances wide in
                    # its narrowest in-plane direction (p5–p95): a 6 cm tubular leg lies within
                    # 4 cm of a plane through its axis and is NOT a plane — it is residue
                    # … and THIN: a plane sliced through a solid lump holds as many of the
                    # object's points on its shoulder (sd..2·sd) as in its core; a panel's
                    # shoulder is its noise tail
                    split = []
                    for _k, m_, mask_ in parts:
                        if not _is_thin(pts, m_, sd):
                            continue
                        for comp in _plane_components(pts, mask_, m_, 2.0 * sd):
                            if int(comp.sum()) >= part_min and _panel_width(pts[comp], m_) >= 4.0 * sd:
                                split.append(("plane", m_, comp))
                    # the greedy extraction let a rejected slice claim points first (a blob's
                    # slice took 250 of the plate's points): every kept panel RECLAIMS, largest
                    # first, the object's points within the tolerance of its plane that are
                    # connected to it
                    claimed = np.zeros(n_obj, dtype=bool)
                    reclaimed = []
                    for _k, m_, comp in sorted(split, key=lambda t: -int(t[2].sum())):
                        mem = (np.abs(np.asarray(m_.signed_distance(pts))) <= sd) & ~claimed
                        grown = np.zeros(n_obj, dtype=bool)
                        for c2 in _plane_components(pts, mem, m_, 2.0 * sd):
                            if (c2 & comp).any():
                                grown |= c2
                        if int(grown.sum()) >= part_min:
                            claimed |= grown
                            reclaimed.append(("plane", m_, grown))
                    parts = sorted(reclaimed, key=lambda t: -int(t[2].sum()))
                    explained = float(sum(int(m.sum()) for _k, _m, m in parts)) / n_obj
                    if not parts:
                        man_rejected[int(iid)] = within
                        logger.info("scene: %s_%s — marked Manhattan but no plane holds %.0f%% of "
                                    "its points (one plane: %.0f%% within %.0f mm): the configured "
                                    "ladder decides", label, iid, 100 * manhattan_part_min_frac,
                                    100 * within, 1000 * sd)
                        fs = None
                    else:
                        # THE MODEL: the planes found, straight; the rest of the object —
                        # whatever no plane holds — is declared, and when the planes do not
                        # make a SURFACE (under manhattan_min_inlier_frac: a desk top with
                        # tubular legs) it is meshed AS MEASURED next to them (the published
                        # GLB fuses unexplained_idx.npy), never bent into a spline. A surface's
                        # leftovers (a floor's clutter) are declared only.
                        residue_mask = np.ones(n_obj, dtype=bool)
                        for _k, _m, m in parts:
                            residue_mask &= ~m
                        residue = float(residue_mask.mean())
                        partial = explained < float(manhattan_min_inlier_frac)
                        if len(parts) >= 2:
                            k0, m0, mask0 = parts[0]
                            fs = evaluate_model(m0, kind="plane",
                                                escalation_path=["plane", "manhattan_parts"],
                                                orig=raw_seg[mask0], instance_id=iid, label=label,
                                                **eval_kwargs)
                            fs.from_decompose = True
                            fs.manhattan_parts = [(m, raw_seg[mask]) for _k, m, mask in parts]
                        fs.manhattan_residue = residue
                        fs.manhattan_partial = partial
                        # the rest (USER 2026-10-15 "probar con cilindros, y si no, tsdf sin
                        # poisson"): handled in the Manhattan stage below — bars become
                        # cylinders on the frame, what is neither a plane nor a bar gets the
                        # live-cloud TSDF; NEVER Poisson (forced_leftover stays empty)
                        fs.manhattan_residue_mask = residue_mask if (partial and residue >= float(manhattan_part_min_frac)) else None
                        forced_leftover[int(iid)] = np.zeros(0, dtype=np.int64)
                        logger.info("scene: %s_%s — marked Manhattan: %d plane(s) explain %.0f%% "
                                    "(one plane %.0f%%), residue %.0f%% %s", label, iid, len(parts),
                                    100 * explained, 100 * within, 100 * residue,
                                    "→ bars / TSDF next to the planes (not a surface)"
                                    if fs.manhattan_residue_mask is not None else "declared")
                if fs is not None:
                    man_planes.add(int(iid))
                elif [str(m) for m in (kwargs.get("models") or [])] != ["plane"]:
                    fs = fit_segment(pts, instance_id=iid, label=label,
                                     original_xyz=raw_seg,
                                     scene_centroid=centroid,
                                     progress_cb=progress_cb, **kwargs)
            else:
                fs = fit_segment(pts, instance_id=iid, label=label,
                                 original_xyz=raw_seg,
                                 scene_centroid=centroid,
                                 progress_cb=progress_cb, **kwargs)
        except Exception as e:
            logger.exception("scene: %s_%s failed: %s", label, iid, e)
            fs = None
        if fs is not None:
            p95 = float(fs.report.stats.p95_mm)
            if int(iid) in man_planes:
                logger.info("scene: %s_%s — Manhattan plane kept (p95 %.1fmm: the deviation is "
                            "the finding, declared in its residual report)", label, iid, p95)
            elif p95 > float(accept_p95_mm):
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

    # ── Manhattan: the marked planes snap to the scene frame, re-evaluated honestly ──
    if manhattan and fitted:
        m_idx = [i for i, f in enumerate(fitted)
                 if int(f.instance_id) in man_planes and isinstance(f.model, PlaneModel)]
        if m_idx:
            from .manhattan import regularize_manhattan
            # every marked plane of the scene: the single-plane objects and every part of the
            # decomposed ones — ONE frame, one snap, coplanar merge across objects
            entries = []           # (fitted index, part index or None, model, points)
            for i in m_idx:
                f = fitted[i]
                parts_i = getattr(f, "manhattan_parts", None)
                if parts_i:
                    for k, (m, pk) in enumerate(parts_i):
                        entries.append((i, k, m, pk))
                else:
                    entries.append((i, None, f.model, seg_points[f.instance_id]))
            snapped, man_report = regularize_manhattan(
                [e[2] for e in entries], [float(len(e[3])) for e in entries],
                [e[3] for e in entries], up_hint=man_up, snap_deg=float(manhattan_snap_deg),
                coplanar_tol_mm=float(coplanar_tol_mm),
                ids=[int(fitted[e[0]].instance_id) for e in entries])
            by_obj: Dict[int, list] = {}
            for e, m_new, rec in zip(entries, snapped, man_report["objects"]):
                by_obj.setdefault(e[0], []).append((e[1], m_new, rec, e[3]))
            for i, rows in by_obj.items():
                old = fitted[i]
                _cb(instance_id=old.instance_id, phase="manhattan", elapsed=time.time() - t0)
                if rows[0][0] is None:                       # a single plane
                    _k, m_new, rec, pts_i = rows[0]
                    fs2 = evaluate_model(
                        m_new, kind="plane", escalation_path=old.escalation_path,
                        orig=seg_points[old.instance_id],
                        instance_id=old.instance_id, label=old.label, **eval_kwargs)
                    fs2.regularized = bool(rec.get("snapped"))
                    fs2.manhattan_residue_mask = getattr(old, "manhattan_residue_mask", None)
                    fs2.manhattan = {**rec, "decomposed": False,
                                     "residue_frac": round(float(getattr(old, "manhattan_residue", 0.0)), 4),
                                     "partial": bool(getattr(old, "manhattan_partial", False)),
                                     "residue_meshed": bool(len(forced_leftover.get(int(old.instance_id), ())))}
                    fitted[i] = fs2
                    continue
                # a decomposed object: every part evaluated on its OWN points with its snapped
                # plane; the largest is the instance's surface, the others its parts
                surfs = []
                for k, m_new, rec, pts_k in sorted(rows, key=lambda r: -len(r[3])):
                    fk = evaluate_model(m_new, kind="plane",
                                        escalation_path=["plane", "manhattan_parts"],
                                        orig=pts_k, instance_id=old.instance_id, label=old.label,
                                        **eval_kwargs)
                    fk.regularized = bool(rec.get("snapped"))
                    fk.from_decompose = True
                    fk.manhattan = {**rec, "part": int(k), "n_points": int(len(pts_k))}
                    surfs.append(fk)
                primary = surfs[0]
                primary.manhattan_residue_mask = getattr(old, "manhattan_residue_mask", None)
                primary.manhattan = {**primary.manhattan, "decomposed": True,
                                     "parts": [sf.manhattan for sf in surfs],
                                     "residue_frac": round(float(getattr(old, "manhattan_residue", 0.0)), 4),
                                     "partial": bool(getattr(old, "manhattan_partial", False)),
                                     "residue_meshed": bool(len(forced_leftover.get(int(old.instance_id), ())))}
                fitted[i] = primary
                if len(surfs) > 1:
                    extra_parts[int(old.instance_id)] = surfs[1:]
                logger.info("scene: %s_%s — Manhattan decomposition: %d plane(s) on the frame, "
                            "residue %.0f%% declared", old.label, old.instance_id, len(surfs),
                            100 * float(getattr(old, "manhattan_residue", 0.0)))
            # THE REST of a marked object that is not a surface: bars → straight cylinders on
            # the frame, the remainder → the live-cloud TSDF, never Poisson (USER 2026-10-15)
            frame_axes = {k: np.asarray(man_report["frame"][k], np.float64) for k in ("up", "X", "Z")}
            origins = None
            for i in m_idx:
                f = fitted[i]
                rmask = getattr(f, "manhattan_residue_mask", None)
                if rmask is None or not rmask.any():
                    continue
                pts_i = seg_points[f.instance_id]
                sd_i = float(kwargs.get("support_dist_m", 0.04))
                rest_parts, rest_rec = _manhattan_residue(
                    pts_i, rmask, frame_axes, sd_i, float(manhattan_snap_deg),
                    float(manhattan_part_min_frac), float(manhattan_bar_aspect_min),
                    float(manhattan_bar_round_tol), f.instance_id, f.label, eval_kwargs)
                tsdf_mask = rest_rec.pop("_tsdf_mask", None)
                if tsdf_mask is not None and tsdf_mask.any():
                    gi_i = gi_map.get(f.instance_id)
                    try:
                        if origins is None:
                            from segmentation.pipeline import _load_ply_origins
                            origins = _load_ply_origins(session_dir / "output" / "cleaned_cloud.ply")
                        if origins is None or gi_i is None or len(gi_i) != len(pts_i):
                            raise RuntimeError("no per-point provenance for the TSDF frames")
                        from segmentation.tsdf_export import object_tsdf_config, tsdf_mesh_of_points
                        oc = object_tsdf_config(config or {})
                        frames = np.unique(np.asarray(origins[1])[gi_i[tsdf_mask]]).tolist()
                        r = tsdf_mesh_of_points(session_dir / "output", pts_i[tsdf_mask], frames, oc)
                        if r is None:
                            raise RuntimeError("the TSDF integrated nothing")
                        V_t, F_t = r
                        rest_parts.append(_measured_part(f, "tsdf", V_t, F_t, int(tsdf_mask.sum()),
                                                         {"kind": "tsdf", "n_points": int(tsdf_mask.sum()),
                                                          "frames": len(frames), "source": "live cloud"}))
                        rest_rec["tsdf"] = {"n_points": int(tsdf_mask.sum()), "frames": len(frames),
                                            "vertices": int(len(V_t)), "faces": int(len(F_t))}
                    except Exception as e:  # noqa: BLE001 — declared, never Poisson
                        rest_rec["tsdf"] = {"n_points": int(tsdf_mask.sum()), "unmeshed": f"{type(e).__name__}: {e}"}
                        logger.warning("scene: %s_%s residue TSDF not meshed (declared): %s", f.label, f.instance_id, e)
                if rest_parts:
                    extra_parts.setdefault(int(f.instance_id), []).extend(rest_parts)
                f.manhattan = {**f.manhattan, "residue": rest_rec}
                logger.info("scene: %s_%s — residue: %d cylinder(s), tsdf %s", f.label, f.instance_id,
                            len(rest_rec.get("cylinders", [])),
                            "meshed" if "vertices" in (rest_rec.get("tsdf") or {}) else
                            ("none" if "tsdf" not in rest_rec else "declared"))
            man_report["not_a_surface"] = {str(k): round(v, 3) for k, v in man_rejected.items()}
    elif manhattan:
        man_report = {"objects": [], "not_a_surface": {str(k): round(v, 3) for k, v in man_rejected.items()}}

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
                if (getattr(old, "manhattan", None) or {}).get("snapped"):
                    continue    # the Manhattan frame is the authority for a marked plane: it
                    #             anchors the others' clusters and is never moved by them
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
                if getattr(old, "manhattan", None):
                    fs2.manhattan = old.manhattan
                fitted[i] = fs2
        # the decomposition parts of the Manhattan objects meet their neighbours — and each
        # other (the S-wall's returns) — on exact intersection lines too
        snap_edges([f for f in fitted if isinstance(f.model, PlaneModel)]
                   + [pp for parts in extra_parts.values() for pp in parts
                      if isinstance(pp.model, PlaneModel)],
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
        if getattr(fs, "manhattan", None):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "manhattan.json").write_text(json.dumps(
                {**fs.manhattan, "frame": (man_report or {}).get("frame")}, indent=2))
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
                        "manhattan": getattr(fs, "manhattan", None),
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
        "manhattan": man_report,
    })
    out_base.mkdir(parents=True, exist_ok=True)
    from correction.epoch import stamp_nearest as _stamp_epoch
    _stamp_epoch(scene_report, out_base)
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
