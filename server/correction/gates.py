"""Acceptance gates — pure, testable, and ALL of them veto the application.

USER 2026-09-06: "the correction is computed from the marked segments but the
REST of the scene is the exam". MEJORAS_OBLIGATORIAS §3 (box1): a degenerate
solve applied 92°/9 m while the floor witness collapsed — every gate here
BLOCKS, none merely reports.

Each gate returns ``{"name", "passed", "detail", ...measurements}``; the full
list lands in the run report, also on rejection, so the user always gets the
numbers and the exact reason.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from correction.config import CorrectionConfig
from correction.distribute import warp_subset
from correction.session import CorrectionSession
from correction.solve import fit_plane, rot_deg


def gate_object_collapse(res_before_m: float, res_after_m: float,
                         cfg: CorrectionConfig,
                         centroid_offsets_m: Optional[Dict[str, float]]
                         = None) -> dict:
    """The copies must COLLAPSE onto the reference. Median NN alone is not
    enough — a wall can slide along its own plane with a tiny NN residual —
    so each object's corrected CENTROID must also land on its reference
    centroid within the same limit."""
    worst_centroid = max(centroid_offsets_m.values()) \
        if centroid_offsets_m else 0.0
    # the improvement ratio only applies when the copies started truly
    # apart; below the noise floor a "worse ratio" is sampling noise
    ratio_ok = (res_after_m <= cfg.gates.residual_improvement_ratio
                * res_before_m
                or res_after_m <= cfg.gates.collapse_floor_m)
    passed = (res_after_m <= cfg.gates.max_object_residual_m
              and ratio_ok
              and worst_centroid <= cfg.gates.max_object_residual_m)
    return {"name": "object_collapse", "passed": passed,
            "before_cm": round(res_before_m * 100, 1),
            "after_cm": round(res_after_m * 100, 1),
            "centroid_offsets_cm": ({k: round(v * 100, 1) for k, v in
                                     centroid_offsets_m.items()}
                                    if centroid_offsets_m else {}),
            "detail": f"median NN {res_before_m*100:.1f} → "
                      f"{res_after_m*100:.1f} cm, worst centroid offset "
                      f"{worst_centroid*100:.1f} cm (limits: ≤ "
                      f"{cfg.gates.max_object_residual_m*100:.0f} cm, ≤ "
                      f"{cfg.gates.residual_improvement_ratio:g} × before)"}


def gate_plausibility(visit_solutions: List[dict],
                      cfg: CorrectionConfig) -> dict:
    """MEJORAS §3.2: rot beyond max_rot_deg or |t| beyond max_translation_m
    means broken anchors, not drift — reject."""
    worst_rot = worst_t = 0.0
    for sol in visit_solutions:
        worst_rot = max(worst_rot, rot_deg(np.asarray(sol["R"])))
        worst_t = max(worst_t, float(np.linalg.norm(sol["t"])))
    passed = (worst_rot <= cfg.gates.max_rot_deg
              and worst_t <= cfg.gates.max_translation_m)
    return {"name": "plausibility", "passed": passed,
            "max_rot_deg": round(worst_rot, 2),
            "max_t_m": round(worst_t, 3),
            "detail": f"largest solved transform: {worst_rot:.2f}° / "
                      f"{worst_t:.3f} m (caps {cfg.gates.max_rot_deg:g}° / "
                      f"{cfg.gates.max_translation_m:g} m — measured real "
                      f"drift is ~1-2° / ~1.5 m)"}


def gate_continuity(distribution_report: dict, cfg: CorrectionConfig) -> dict:
    step_mm = float(distribution_report["max_step_between_keyframes_mm"])
    step_deg = float(distribution_report["max_step_between_keyframes_deg"])
    passed = (step_mm <= cfg.gates.max_step_mm
              and step_deg <= cfg.gates.max_step_deg)
    return {"name": "continuity", "passed": passed,
            "max_step_mm": step_mm, "max_step_deg": step_deg,
            "detail": f"max step between neighbouring keyframes "
                      f"{step_mm:.1f} mm / {step_deg:.3f}° (limits "
                      f"{cfg.gates.max_step_mm:g} mm / "
                      f"{cfg.gates.max_step_deg:g}°)"}


def gate_integrity(session: CorrectionSession) -> dict:
    """Every point must resolve to a keyframe — an unresolvable point would
    silently stay behind while its neighbours move."""
    n_bad = int((session.ks < 0).sum())
    return {"name": "integrity", "passed": n_bad == 0,
            "unresolvable_points": n_bad,
            "detail": ("all points resolve to a keyframe" if n_bad == 0 else
                       f"{n_bad:,} points carry a frame_global that maps to "
                       f"no keyframe — the session's camera_frames.txt does "
                       f"not cover the cloud; fix the reconstruction "
                       f"artifacts before correcting")}


def _floor_band(xyz: np.ndarray, mask: np.ndarray,
                cfg: CorrectionConfig,
                rng: np.random.Generator):
    """(band_indices, plane) of the floor inside ``mask``: low percentile
    seeds the band, RANSAC+SVD fits the reference plane."""
    idx = np.where(mask)[0]
    if len(idx) < cfg.floor.min_inliers:
        return None, None
    y = xyz[idx, 1]
    y0 = np.percentile(y, cfg.floor.low_band_pct)
    band = idx[y < y0 + cfg.floor.band_m]
    if len(band) < cfg.floor.min_inliers:
        return None, None
    n, c0 = fit_plane(xyz[band], rng, cfg)
    if n[1] < 0:
        n = -n
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    if tilt > cfg.floor.max_tilt_deg:
        return band, None
    return band, (n, c0)


def gate_scene_exam(session: CorrectionSession, marked_iids: List[int],
                    ref_kf_end: int, affected_kfs: List[int],
                    R_kf: np.ndarray, t_kf: np.ndarray, k_kf: np.ndarray,
                    cfg: CorrectionConfig,
                    rng: np.random.Generator) -> dict:
    """Held-out exam: (a) the FLOOR of the affected keyframes against the
    reference floor plane fitted on the identity region; (b) every UNMARKED
    instance that exists both in the affected keyframes and in the reference
    span — its displaced part must land on its reference part. Witnesses come
    from segmentation_result.json automatically; the user never marks them."""
    xyz, ks = session.xyz, session.ks
    affected_set = np.zeros(session.n_kf, dtype=bool)
    for k in affected_kfs:
        affected_set[k] = True
    pt_affected = (ks >= 0) & affected_set[np.clip(ks, 0, session.n_kf - 1)]
    pt_identity = (ks >= 0) & (ks <= ref_kf_end)

    # (a) floor witness -----------------------------------------------------
    floor_result: Dict = {"witness": "floor"}
    _, plane = _floor_band(xyz, pt_identity, cfg, rng)
    if plane is None:
        floor_result.update({
            "passed": False,
            "detail": "no trustworthy reference floor plane in the identity "
                      "region (too few band points or tilt beyond "
                      f"{cfg.floor.max_tilt_deg:g}°) — the scene exam cannot "
                      "certify this correction"})
    else:
        n, c0 = plane
        band_aff, _ = _floor_band(xyz, pt_affected, cfg, rng)
        if band_aff is None:
            floor_result.update({
                "passed": True,
                "detail": "affected keyframes contain no floor band — floor "
                          "witness not applicable"})
        else:
            sub = band_aff if len(band_aff) <= cfg.solve.eval_sample else \
                rng.choice(band_aff, cfg.solve.eval_sample, replace=False)
            before = float(np.median(np.abs((xyz[sub] - c0) @ n)))
            warped = warp_subset(xyz, session.fg, ks, session.cam_center,
                                 sub, R_kf, t_kf, k_kf)
            after = float(np.median(np.abs((warped - c0) @ n)))
            passed = ((after - before) <= cfg.gates.heldout_floor_tol_m
                      and after <= cfg.gates.heldout_floor_abs_m)
            floor_result.update({
                "passed": passed,
                "before_cm": round(before * 100, 1),
                "after_cm": round(after * 100, 1),
                "detail": f"floor band vs reference plane: "
                          f"{before*100:.1f} → {after*100:.1f} cm (may not "
                          f"worsen > {cfg.gates.heldout_floor_tol_m*100:.0f} "
                          f"cm, must end ≤ "
                          f"{cfg.gates.heldout_floor_abs_m*100:.0f} cm)"})

    # (b) unmarked witness instances ---------------------------------------
    witnesses: List[dict] = []
    res_path = session.output_dir / "segmentation_result.json"
    instances = []
    if res_path.exists():
        instances = json.loads(res_path.read_text()).get("instances", [])
    from scipy.spatial import cKDTree
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id", -1)))
        if iid in marked_iids:
            continue
        gidx = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gidx = gidx[(gidx >= 0) & (gidx < len(xyz))]
        if len(gidx) < cfg.evidence.min_object_points_solve:
            continue
        part_disp = gidx[pt_affected[gidx]]
        part_ref = gidx[pt_identity[gidx]]
        if len(part_disp) < cfg.evidence.min_object_points_solve \
                or len(part_ref) < cfg.evidence.min_object_points_solve:
            continue
        tree = cKDTree(xyz[part_ref])
        sub = part_disp if len(part_disp) <= cfg.solve.eval_sample else \
            rng.choice(part_disp, cfg.solve.eval_sample, replace=False)
        d0, _ = tree.query(xyz[sub], workers=cfg.runtime.workers)
        warped = warp_subset(xyz, session.fg, ks, session.cam_center,
                             sub, R_kf, t_kf, k_kf)
        d1, _ = tree.query(warped, workers=cfg.runtime.workers)
        before, after = float(np.median(d0)), float(np.median(d1))
        w_passed = (after <= before
                    or after <= cfg.gates.heldout_object_tol_m)
        witnesses.append({
            "witness": inst.get("label") or str(iid), "iid": iid,
            "passed": w_passed,
            "before_cm": round(before * 100, 1),
            "after_cm": round(after * 100, 1)})

    passed = bool(floor_result.get("passed")) \
        and all(w["passed"] for w in witnesses)
    failed_w = [w["witness"] for w in witnesses if not w["passed"]]
    detail = floor_result.get("detail", "")
    if witnesses:
        detail += (f"; {len(witnesses)} unmarked witness instance(s)"
                   + (f", FAILED: {failed_w}" if failed_w else ", all pass"))
    else:
        detail += "; no unmarked witness instance spans both visits"
    return {"name": "scene_exam", "passed": passed, "floor": floor_result,
            "witness_objects": witnesses, "detail": detail}
