"""Floor alignment against an explicit reference MODEL, per keyframe.

Redesign of the retired ``align_floor_y0`` (H6: taking every anchor to y=0
with a vertical normal flattened ramps and drainage slopes). The reference
floor is now a declared model, chosen by the user in the UI and recorded in
the ledger:

  * ``level``   — horizontal plane at y=0 (the previous behaviour, now
                  declared: it FLATTENS slopes by design).
  * ``plane``   — one RANSAC plane fitted to the anchors' floor points; a real
                  slope survives, only per-keyframe drift is removed.
  * ``profile`` — longitudinal slope: per-anchor floor height regressed
                  against trajectory chainage (ramps / platform drainage where
                  a single plane underfits).

Anchors are KEYFRAMES whose local floor RANSAC passes the guards
(``max_tilt_deg``, ``min_inliers``); for ``plane``/``profile`` an anchor whose
local floor sits farther than ``step_demote_m`` from the reference model is a
REAL level change and is demoted to interpolated — the step is preserved.
Non-anchor keyframes interpolate (slerp+lerp); the alignment then passes the
same plausibility/continuity gates and the same transactional apply as the
object correction.

Scene-exam note (declared): a global floor alignment has no held-out identity
region — its exam is the post-alignment residual of every anchor floor
against the chosen model (reported per keyframe) plus the continuity and
plausibility gates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from correction.config import CorrectionConfig
from correction.session import CorrectionSession

UP = np.array([0.0, 1.0, 0.0])
FLOOR_MODELS = ("level", "plane", "profile")


def _keyframe_floor(session: CorrectionSession, k: int,
                    cfg: CorrectionConfig, rng: np.random.Generator
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                               dict]:
    """Local floor plane of keyframe k: (normal, centroid, info). Normal is
    None when the keyframe fails the guards (info says why)."""
    fl = cfg.floor
    idx = np.where(session.ks == k)[0]
    if len(idx) < fl.min_inliers:
        return None, None, {"kf": k, "role": "demoted",
                            "why": f"only {len(idx)} points"}
    P = session.xyz[idx]
    y0 = np.percentile(P[:, 1], fl.low_band_pct)
    band = P[P[:, 1] < y0 + fl.band_m]
    if len(band) < fl.min_inliers:
        return None, None, {"kf": k, "role": "demoted", "why": "no low band"}
    S = band if len(band) <= fl.ransac_sample else \
        band[rng.choice(len(band), fl.ransac_sample, replace=False)]
    cos_max = np.cos(np.radians(fl.max_tilt_deg))
    best, bn = None, -1
    for _ in range(fl.ransac_iters):
        a, b_, c_ = S[rng.choice(len(S), 3, replace=False)]
        nrm = np.cross(b_ - a, c_ - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm /= ln
        if nrm[1] < 0:
            nrm = -nrm
        if nrm @ UP < cos_max:
            continue
        cnt = int((np.abs((S - a) @ nrm) < fl.ransac_tol_m).sum())
        if cnt > bn:
            bn, best = cnt, (nrm, a)
    if best is None or bn < fl.min_inliers * fl.min_inlier_ratio:
        return None, None, {"kf": k, "role": "demoted",
                            "why": f"floor RANSAC failed ({bn} inliers)"}
    nrm, a = best
    inl = np.abs((S - a) @ nrm) < fl.ransac_refit_band_m
    c_f = S[inl].mean(0)
    nrm = np.linalg.svd(S[inl] - c_f, full_matrices=False)[2][2]
    if nrm[1] < 0:
        nrm = -nrm
    tilt = float(np.degrees(np.arccos(np.clip(nrm @ UP, -1, 1))))
    return nrm, c_f, {"kf": k, "role": "anchor", "tilt_deg": round(tilt, 3),
                      "floor_y_m": round(float(c_f[1]), 4),
                      "floor_inliers": int(inl.sum())}


def _chainage(session: CorrectionSession) -> np.ndarray:
    """Cumulative walked distance at each keyframe (for the profile model)."""
    centers = session.poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b."""
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(axis / s * np.arctan2(s, float(a @ b))
                                ).as_matrix()


def solve_floor(session: CorrectionSession, cfg: CorrectionConfig,
                model: str, keyframes: Optional[List[int]],
                rng: np.random.Generator, log=print) -> dict:
    """Solve the per-keyframe floor alignment. Returns
    {R_kf, t_kf, k_kf, anchors, per_kf_report, model, model_params,
    floor_npz, exam} — the caller runs the gates and the transactional
    apply."""
    if model not in FLOOR_MODELS:
        raise RuntimeError(f"unknown floor model {model!r} — valid: "
                           f"{FLOOR_MODELS}")
    n_kf = session.n_kf
    candidates = (sorted(set(int(k) for k in keyframes))
                  if keyframes else list(range(n_kf)))
    for k in candidates:
        if not (0 <= k < n_kf):
            raise RuntimeError(f"keyframe {k} out of range 0..{n_kf - 1}")

    # 1) local floor per candidate keyframe --------------------------------
    locals_: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    per_kf_report: List[dict] = []
    for k in candidates:
        nrm, c_f, info = _keyframe_floor(session, k, cfg, rng)
        per_kf_report.append(info)
        if nrm is not None:
            locals_[k] = (nrm, c_f)
    if not locals_:
        raise RuntimeError(
            "no candidate keyframe produced a trustworthy local floor plane "
            "— nothing to align (check the tilt/inlier guards in "
            "correction.floor)")
    log(f"  {len(locals_)} anchor candidate(s) of {len(candidates)} "
        f"keyframes")

    # smooth the anchor normals along the walk (drift is smooth; per-patch
    # normal noise at a long lever arm becomes inter-keyframe steps the
    # continuity gate would veto)
    if cfg.floor.normal_smooth_kf > 0 and len(locals_) > 1:
        ks_sorted = sorted(locals_)
        smoothed = {}
        for k in ks_sorted:
            nbrs = [locals_[j][0] for j in ks_sorted
                    if abs(j - k) <= cfg.floor.normal_smooth_kf]
            n_avg = np.mean(nbrs, axis=0)
            n_avg /= np.linalg.norm(n_avg)
            smoothed[k] = (n_avg, locals_[k][1])
        locals_ = smoothed

    # 2) reference model ---------------------------------------------------
    model_params: dict = {"model": model}
    chain = _chainage(session)
    if model == "level":
        def ref_normal(k):
            return UP

        def ref_signed_dist(k, c):
            return float(c[1])
        model_params["plane"] = {"normal": UP.tolist(), "y0": 0.0}
    elif model == "plane":
        pts = np.concatenate([session.xyz[session.ks == k][
            session.xyz[session.ks == k][:, 1]
            < np.percentile(session.xyz[session.ks == k][:, 1],
                            cfg.floor.low_band_pct) + cfg.floor.band_m]
            for k in locals_])
        if len(pts) > cfg.floor.ransac_sample:
            pts = pts[rng.choice(len(pts), cfg.floor.ransac_sample,
                                 replace=False)]
        # robust plane over the union of anchor floor bands (near-horizontal
        # guard as for local planes — a "floor" 40° off is not a floor)
        cos_max = np.cos(np.radians(cfg.floor.max_tilt_deg))
        best, bn = None, -1
        for _ in range(cfg.floor.ransac_iters):
            a, b_, c_ = pts[rng.choice(len(pts), 3, replace=False)]
            nrm = np.cross(b_ - a, c_ - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            nrm /= ln
            if nrm[1] < 0:
                nrm = -nrm
            if nrm @ UP < cos_max:
                continue
            cnt = int((np.abs((pts - a) @ nrm) < cfg.floor.ransac_tol_m).sum())
            if cnt > bn:
                bn, best = cnt, (nrm, a)
        if best is None:
            raise RuntimeError("reference plane RANSAC failed over the "
                               "anchor floor points")
        nrm0, a0 = best
        inl = np.abs((pts - a0) @ nrm0) < cfg.floor.ransac_refit_band_m
        c0 = pts[inl].mean(0)
        n_ref = np.linalg.svd(pts[inl] - c0, full_matrices=False)[2][2]
        if n_ref[1] < 0:
            n_ref = -n_ref
        model_params["plane"] = {
            "normal": [round(float(x), 6) for x in n_ref],
            "point": [round(float(x), 4) for x in c0],
            "slope_deg": round(float(np.degrees(np.arccos(
                np.clip(n_ref @ UP, -1, 1)))), 3),
            "inliers": int(inl.sum())}

        def ref_normal(k):
            return n_ref

        def ref_signed_dist(k, c):
            return float((c - c0) @ n_ref)
    else:  # profile
        ks_a = sorted(locals_)
        s_a = np.array([chain[k] for k in ks_a])
        y_a = np.array([locals_[k][1][1] for k in ks_a])
        # robust linear fit y = a + b·s (one trimming pass)
        A = np.stack([np.ones_like(s_a), s_a], axis=1)
        coef, *_ = np.linalg.lstsq(A, y_a, rcond=None)
        resid = y_a - A @ coef
        keep = np.abs(resid - np.median(resid)) <= \
            3 * (np.median(np.abs(resid - np.median(resid))) + 1e-9)
        if keep.sum() >= 2:
            coef, *_ = np.linalg.lstsq(A[keep], y_a[keep], rcond=None)
        a_c, b_c = float(coef[0]), float(coef[1])
        model_params["profile"] = {
            "y_at_0": round(a_c, 4), "slope_m_per_m": round(b_c, 6),
            "slope_pct": round(b_c * 100, 3)}

        centers = session.poses[:, :3, 3]

        def _walk_dir(k):
            k2 = min(k + 1, n_kf - 1)
            k1 = max(k - 1, 0)
            d = centers[k2] - centers[k1]
            d[1] = 0.0
            ln = np.linalg.norm(d)
            return d / ln if ln > 1e-9 else np.array([1.0, 0.0, 0.0])

        def ref_normal(k):
            n = UP - b_c * _walk_dir(k)
            return n / np.linalg.norm(n)

        def ref_signed_dist(k, c):
            return float(c[1] - (a_c + b_c * chain[k]))

    # 3) anchors: straighten local normal onto the model normal, land the
    #    local floor centroid on the model surface; step-demote for
    #    plane/profile (a real level change is preserved, H6) -------------
    anchors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k, (nrm, c_f) in sorted(locals_.items()):
        dist = ref_signed_dist(k, c_f)
        if model != "level" and abs(dist) > cfg.floor.step_demote_m:
            for info in per_kf_report:
                if info.get("kf") == k:
                    info["role"] = "demoted"
                    info["why"] = (f"floor sits {dist:+.3f} m off the "
                                  f"reference model — real step/level "
                                  f"change, preserved by interpolation")
            continue
        n_t = ref_normal(k)
        tilt_off = float(np.degrees(np.arccos(
            np.clip(float(nrm @ n_t), -1, 1))))
        if tilt_off >= cfg.floor.min_tilt_deg:
            R = _rot_between(nrm, n_t)
            t = c_f - R @ c_f        # rotate about the local floor centroid
        else:
            # below min_tilt_deg the measured tilt is patch noise: a noisy
            # rotation at a long lever arm turns into inter-keyframe steps
            # the continuity gate would (rightly) veto — height only
            R = np.eye(3)
            t = np.zeros(3)
        t = t - dist * n_t           # land the centroid on the model
        anchors[k] = (R, t)
        for info in per_kf_report:
            if info.get("kf") == k and info.get("role") == "anchor":
                info["dy_m"] = round(-dist, 4)
    if not anchors:
        raise RuntimeError(
            "every candidate keyframe was demoted (steps/guards) — nothing "
            "to anchor; pick keyframes on the reference floor level")

    # 4) interpolate every keyframe between/past the anchors ---------------
    a_kfs = sorted(anchors)
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    slerp = None
    if len(a_kfs) > 1:
        slerp = Slerp(a_kfs, Rotation.from_matrix(
            np.stack([anchors[k][0] for k in a_kfs])))
    for k in range(n_kf):
        if k in anchors:
            R_kf[k], t_kf[k] = anchors[k]
        elif k <= a_kfs[0]:
            R_kf[k], t_kf[k] = anchors[a_kfs[0]]
        elif k >= a_kfs[-1]:
            R_kf[k], t_kf[k] = anchors[a_kfs[-1]]
        else:
            lo = max(a for a in a_kfs if a < k)
            hi = min(a for a in a_kfs if a > k)
            w = (k - lo) / (hi - lo)
            R_kf[k] = slerp([k]).as_matrix()[0]
            t_kf[k] = (1 - w) * anchors[lo][1] + w * anchors[hi][1]

    # 5) exam: post-alignment residual of every anchor floor vs the model --
    exam: List[dict] = []
    worst = 0.0
    for k, (nrm, c_f) in sorted(locals_.items()):
        if k not in anchors:
            continue
        c_new = R_kf[k] @ c_f + t_kf[k]
        resid = abs(ref_signed_dist(k, c_new))
        worst = max(worst, resid)
        exam.append({"kf": k, "residual_mm": round(resid * 1000, 2)})

    # 6) display transform: level puts the floor at y=0 by construction →
    #    identity npz (stale leveling shifted the scene, USER 2026-09-06);
    #    plane/profile keep the geometry sloped → existing transform stays.
    floor_npz = ({"s": np.float64(1.0), "R": np.eye(3), "t": np.zeros(3)}
                 if model == "level" else None)

    return {"R_kf": R_kf, "t_kf": t_kf, "k_kf": np.ones(n_kf),
            "anchors": [{"kf": int(k),
                         "rot_deg": round(float(np.degrees(np.arccos(
                             np.clip((np.trace(anchors[k][0]) - 1) / 2,
                                     -1, 1)))), 3),
                         "t_m": round(float(np.linalg.norm(anchors[k][1])), 4)}
                        for k in a_kfs],
            "per_kf_report": per_kf_report, "model": model,
            "model_params": model_params,
            "exam": {"anchor_floor_residuals": exam,
                     "worst_residual_mm": round(worst * 1000, 2)},
            "floor_npz": floor_npz}
