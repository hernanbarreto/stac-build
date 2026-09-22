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
floor REALLY changes level — a jump between neighbours that the walk's own
drift rate cannot explain and that the session can tell apart from its own
repeatability — is demoted to interpolated, and the step is preserved.
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

import time

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


def _moving_median(values: np.ndarray, keys: np.ndarray, window: int
                   ) -> np.ndarray:
    """Per-key moving median of ``values`` over keys within ±window."""
    out = np.empty_like(values, dtype=np.float64)
    for i, k in enumerate(keys):
        m = np.abs(keys - k) <= window
        out[i] = float(np.median(values[m]))
    return out


# MAD → sigma for a normal distribution: 1/Phi^-1(3/4). A mathematical
# constant, not a threshold — nothing decides by changing it.
MAD_TO_SIGMA = 1.0 / 0.6744897501960817


def _session_repeatability(session: CorrectionSession, cfg) -> float:
    """What this session can repeat, measured by itself. The same cascade the
    correction already uses (uncertainty.json → elastic seam residual →
    intra-chunk agreement), with the config fallback only for a session that
    wrote no evidence at all."""
    fb = float(cfg.visit_drift.default_repeatability_m)
    try:
        from reconstruction.certify.repeatability import session_repeatability
        # it returns a RECORD, not a number — `float(dict)` raised a TypeError
        # the bare except below swallowed, so this always fell back to the
        # config and never used what the session measured, against its own
        # docstring (found 2026-09-21). `visit_drift_run:56` reads it right.
        v = session_repeatability(session.output_dir, fallback_m=fb,
                                  log=lambda m: None)
        sig = float((v or {}).get("sigma_floor_m") or 0.0)
        return sig if sig > 0 else fb
    except Exception:  # noqa: BLE001 — a session with no evidence keeps the fallback
        return fb


def _robust_sigma(v: np.ndarray) -> float:
    """MAD → sigma. The session's own scatter, measured, never assumed."""
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return 1e-6
    return max(float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v)))), 1e-6)


def _level_changes(trend: np.ndarray, walk: np.ndarray, rep_m: float
                   ) -> Tuple[np.ndarray, float]:
    """Where the floor REALLY changes level, and the drift rate it is judged
    against.

    A step is a DISCONTINUITY: the height jumps between two adjacent anchors
    and stays at the new level. Drift ACCUMULATES: it grows with the distance
    walked. Distance to the reference MODEL cannot tell them apart — it IS the
    drift being corrected, which is what `step_demote_m` got wrong (pccr epoch
    0: 49.8 cm of offset built out of jumps of at most 4.9 cm, median 1.2 mm,
    accumulating at +18.5 mm/m).

    So the jump between neighbours is compared against what the walk's own
    drift rate explains over that stretch, and what is left over has to clear
    the session's OWN repeatability before it may be called a level change.
    Measured: pccr's largest excess is 41.7 mm against a 47.7 mm repeatability
    — no step, and the whole floor is corrected; a synthetic 15 cm step is
    147.1 mm and a 50 cm one 494.5 mm.

    Returns (cut, rate): `cut[i]` is True when a real level change happens
    between anchor i and i+1 (so `cut` has one entry fewer than `trend`).
    """
    trend = np.asarray(trend, np.float64)
    walk = np.asarray(walk, np.float64)
    if len(trend) < 3:
        return np.zeros(max(0, len(trend) - 1), bool), 0.0
    A = np.vstack([walk, np.ones_like(walk)]).T
    rate = float(np.linalg.lstsq(A, trend, rcond=None)[0][0])
    excess = np.abs(np.diff(trend)) - abs(rate) * np.abs(np.diff(walk))
    return excess > float(rep_m), rate


def _segments(cut: np.ndarray) -> np.ndarray:
    """Anchor → the level segment it belongs to; a cut starts a new one."""
    seg = np.zeros(len(cut) + 1, np.int64)
    for i, c in enumerate(cut):
        seg[i + 1] = seg[i] + (1 if c else 0)
    return seg


def solve_floor(session: CorrectionSession, cfg: CorrectionConfig,
                model: str, keyframes: Optional[List[int]],
                rng: np.random.Generator, log=print,
                fixed_plane: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                identity_until: int = -1) -> dict:
    """Solve the per-keyframe floor alignment. Returns
    {R_kf, t_kf, k_kf, anchors, per_kf_report, model, model_params,
    floor_npz, exam} — the caller runs the gates and the transactional
    apply.

    ``fixed_plane`` (normal, point): use this reference plane instead of
    fitting one (the object correction passes the plane of its identity
    region — the floor is then a CONSTRAINT of the loop closure, prompt
    §5.4 "1 objeto + plano de piso"). ``identity_until``: keyframes up to
    this index stay identity (the reference visit); anchors only after it.

    Real-data lesson (pccr 2026-09-08: 197 mm / 2.7° steps between
    neighbouring keyframes → continuity veto in every model): a per-keyframe
    floor patch is NOISY — furniture caught in the low band, unstable
    normals on a 2 m patch, long lever arms from the origin. The drift is
    SMOOTH along the walk, so every anchor uses the TREND: heights and
    normals are moving medians/means over ±``smooth_window_kf``, and an
    anchor whose raw height sits more than ``local_mad_k`` robust sigmas from
    its own local trend is demoted (a table top is not the floor).
    """
    if model not in FLOOR_MODELS:
        raise RuntimeError(f"unknown floor model {model!r} — valid: "
                           f"{FLOOR_MODELS}")
    n_kf = session.n_kf
    candidates = (sorted(set(int(k) for k in keyframes))
                  if keyframes else list(range(n_kf)))
    candidates = [k for k in candidates if k > identity_until]
    for k in candidates:
        if not (0 <= k < n_kf):
            raise RuntimeError(f"keyframe {k} out of range 0..{n_kf - 1}")
    if not candidates:
        raise RuntimeError("no candidate keyframe after the identity region")

    # 1) local floor per candidate keyframe --------------------------------
    locals_: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    per_kf_report: List[dict] = []
    # RANSAC on every candidate keyframe is minutes of work. It used to print
    # nothing until it was over, so the user could not tell it apart from a
    # hang — "no se esta imprimiendo nada en consola" (USER 2026-09-21). No
    # stage may go more than a handful of seconds without a magnitude.
    _REPORTS = 8          # how many progress lines this stage prints
    _step = max(1, int(len(candidates) / _REPORTS))
    _t0 = time.time()
    for _n, k in enumerate(candidates, 1):
        nrm, c_f, info = _keyframe_floor(session, k, cfg, rng)
        per_kf_report.append(info)
        if nrm is not None:
            locals_[k] = (nrm, c_f)
        if _n % _step == 0 or _n == len(candidates):
            log(f"    floor planes: {_n}/{len(candidates)} keyframes, "
                f"{len(locals_)} usable ({time.time() - _t0:.0f}s)")
    if not locals_:
        raise RuntimeError(
            "no candidate keyframe produced a trustworthy local floor plane "
            "— nothing to align (check the tilt/inlier guards in "
            "correction.floor)")
    log(f"  {len(locals_)} anchor candidate(s) of {len(candidates)} "
        f"keyframes")

    # 2) reference model ---------------------------------------------------
    model_params: dict = {"model": model}
    chain = _chainage(session)
    if fixed_plane is not None:
        n_ref = np.asarray(fixed_plane[0], dtype=np.float64)
        n_ref = n_ref / np.linalg.norm(n_ref)
        if n_ref[1] < 0:
            n_ref = -n_ref
        c0 = np.asarray(fixed_plane[1], dtype=np.float64)
        model = "plane"
        model_params = {"model": "plane", "fixed": True, "plane": {
            "normal": [round(float(x), 6) for x in n_ref],
            "point": [round(float(x), 4) for x in c0]}}

        def ref_normal(k):
            return n_ref

        def ref_signed_dist(k, c):
            return float((c - c0) @ n_ref)
    elif model == "level":
        def ref_normal(k):
            return UP

        def ref_signed_dist(k, c):
            return float(c[1])
        model_params["plane"] = {"normal": UP.tolist(), "y0": 0.0}
    elif model == "plane":
        # USER 2026-09-09 (pccr): a plane fitted over the WHOLE drifted floor
        # followed the drift itself (3.0° tilt) and the correction became
        # invisible. The reference is the floor at the START of the walk —
        # the first `reference_span_kf` anchors, where drift is zero — and
        # every later keyframe is brought onto it.
        ref_ks = sorted(locals_)[:cfg.floor.reference_span_kf]
        pts = np.concatenate([session.xyz[session.ks == k][
            session.xyz[session.ks == k][:, 1]
            < np.percentile(session.xyz[session.ks == k][:, 1],
                            cfg.floor.low_band_pct) + cfg.floor.band_m]
            for k in ref_ks])
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
            "inliers": int(inl.sum()),
            "reference_keyframes": [int(ref_ks[0]), int(ref_ks[-1])]}

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

    # 3) TREND along the walk: raw per-keyframe distances to the model and
    #    normals → moving median / mean over ±smooth_window_kf; anchors off
    #    the local trend (furniture caught as floor) are demoted, and a run
    #    of anchors beyond a real DISCONTINUITY keeps its own level (H6) ---
    ks_sorted = np.array(sorted(locals_), dtype=np.int64)
    raw_dist = np.array([ref_signed_dist(int(k), locals_[int(k)][1])
                         for k in ks_sorted])
    W = cfg.floor.smooth_window_kf
    trend_dist = _moving_median(raw_dist, ks_sorted, W) if W > 0 \
        else raw_dist.copy()
    trend_norm: Dict[int, np.ndarray] = {}
    for k in ks_sorted:
        nbrs = [locals_[int(j)][0] for j in ks_sorted if abs(j - k) <= W]
        n_avg = np.mean(nbrs, axis=0)
        trend_norm[int(k)] = n_avg / np.linalg.norm(n_avg)

    # TWO tests, each against a MEASURED scale — never one number for both
    # (USER 2026-09-20; see `_level_changes` and config `floor.local_mad_k`).
    #  (1) FURNITURE: the anchor disagrees with its own neighbours.
    #  (2) LEVEL CHANGE: the floor jumps and stays jumped, by more than the
    #      drift rate explains and more than the session can repeat.
    resid_local = raw_dist - trend_dist
    sigma_local = _robust_sigma(resid_local)
    # ...and the same for the NORMALS: how much each keyframe's own measured
    # floor normal disagrees with the trend its neighbours draw. That angle IS
    # the normal-estimation noise of this session, measured on its own data —
    # it replaces `min_tilt_deg`, a constant that decided whether the scene
    # gets levelled at all (USER 2026-09-22: pccr's floor drifted 16.7 mm/m =
    # 0.957 deg and lost to the 1.0 deg constant by four hundredths of a
    # degree, so 216 keyframes were rotated by exactly 0.000 deg and 32 cm of
    # slope over the walk stayed in the cloud).
    _tilt_resid = np.array([
        float(np.degrees(np.arccos(np.clip(
            float(locals_[int(k)][0] @ trend_norm[int(k)]), -1, 1))))
        for k in ks_sorted])
    sigma_tilt = float(_robust_sigma(_tilt_resid - np.median(_tilt_resid)))
    rep_m = _session_repeatability(session, cfg)
    walk_of = _chainage(session)[ks_sorted]
    cut, drift_rate = _level_changes(trend_dist, walk_of, rep_m)
    seg_of = _segments(cut)
    ref_seg = int(seg_of[0])
    step_seg: Dict[int, float] = {}
    if cut.any() and model != "level" and fixed_plane is None:
        ref_level = float(np.median(trend_dist[seg_of == ref_seg]))
        ref_walk = float(np.median(walk_of[seg_of == ref_seg]))
        for sg in set(int(x) for x in seg_of) - {ref_seg}:
            m_ = seg_of == sg
            expl = abs(drift_rate) * abs(float(np.median(walk_of[m_])) - ref_walk)
            off = float(np.median(trend_dist[m_])) - ref_level
            if abs(off) - expl > rep_m:
                step_seg[sg] = off
                # A PRESERVED segment is height this stage decides NOT to
                # correct, so it has to say what it keeps and on what evidence
                # — the pccr epoch of 2026-09-22 preserved two of them and
                # delivered a floor 2.8° tilted with 606 mm end to end, and
                # nothing in the log said which stretch held it (USER: *"no
                # dejes cabos sueltos"*).
                _kfs = ks_sorted[m_]
                log(f"    floor: LEVEL CHANGE kept — keyframes "
                    f"{int(_kfs.min())}-{int(_kfs.max())} "
                    f"({int(m_.sum())} anchor(s), walk "
                    f"{float(np.median(walk_of[m_])):.1f} m) sit "
                    f"{off*1000:+.0f} mm from the reference level; the "
                    f"{drift_rate*1000:+.1f} mm/m drift explains "
                    f"{expl*1000:.0f} mm of it, the remaining "
                    f"{(abs(off)-expl)*1000:.0f} mm clears the session's "
                    f"{rep_m*1000:.0f} mm repeatability → NOT corrected")
    log(f"  floor: local scatter {sigma_local*1000:.1f} mm (k "
        f"{cfg.floor.local_mad_k}), drift {drift_rate*1000:+.1f} mm/m, "
        f"repeatability {rep_m*1000:.1f} mm, normal scatter "
        f"{sigma_tilt:.3f}° → tilt bar "
        f"{max(float(cfg.floor.min_tilt_deg), float(cfg.floor.local_mad_k)*sigma_tilt):.3f}° "
        f"— {int(cut.sum())} level change(s), {len(step_seg)} segment(s) "
        f"preserved")

    anchors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i, k in enumerate(ks_sorted):
        k = int(k)
        nrm_raw, c_f = locals_[k]
        why = None
        if abs(resid_local[i]) > cfg.floor.local_mad_k * sigma_local:
            why = (f"local floor {raw_dist[i]:+.3f} m vs trend "
                   f"{trend_dist[i]:+.3f} m — "
                   f"{abs(resid_local[i])/sigma_local:.1f} sigma off its own "
                   f"neighbours, not the floor (furniture / outlier patch), "
                   f"interpolated")
        elif int(seg_of[i]) in step_seg:
            why = (f"floor sits {step_seg[int(seg_of[i])]:+.3f} m from the "
                   f"reference level, more than the {drift_rate*1000:+.1f} "
                   f"mm/m drift explains and more than the session repeats "
                   f"({rep_m*1000:.1f} mm) — real step/level change, "
                   f"preserved by interpolation")
        if why:
            for info in per_kf_report:
                if info.get("kf") == k:
                    info["role"] = "demoted"
                    info["why"] = why
            continue
        dist = float(trend_dist[i])
        nrm = trend_norm[k]
        n_t = ref_normal(k)
        tilt_off = float(np.degrees(np.arccos(
            np.clip(float(nrm @ n_t), -1, 1))))
        # A tilt is REAL when it is bigger than what this session's own normals
        # scatter by — the same discipline the two tests above already follow
        # (a MEASURED scale, never one number for both). `min_tilt_deg` is the
        # floor of that bar, not the bar: set it to 0 and the measurement
        # decides alone.
        _tilt_bar = max(float(cfg.floor.min_tilt_deg),
                        float(cfg.floor.local_mad_k) * sigma_tilt)
        if tilt_off >= _tilt_bar:
            R = _rot_between(nrm, n_t)
            t = c_f - R @ c_f        # rotate about the local floor centroid
        else:
            # under the session's own normal noise the trend tilt is noise too:
            # height only
            R = np.eye(3)
            t = np.zeros(3)
        t = t - dist * n_t           # land the trend height on the model
        anchors[k] = (R, t)
        for info in per_kf_report:
            if info.get("kf") == k and info.get("role") == "anchor":
                info["dy_m"] = round(-dist, 4)
                info["tilt_trend_deg"] = round(tilt_off, 3)
    if not anchors:
        raise RuntimeError(
            "every candidate keyframe was demoted (steps/guards) — nothing "
            "to anchor; pick keyframes on the reference floor level")

    # 4) interpolate every keyframe between/past the anchors (identity up to
    #    identity_until when an identity region exists) -------------------
    a_kfs = sorted(anchors)
    if identity_until >= 0:
        anchors[identity_until] = (np.eye(3), np.zeros(3))
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
    if identity_until >= 0:
        del anchors[identity_until]
        a_kfs = sorted(anchors)

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
            "n_demoted": int(sum(1 for i in per_kf_report
                                 if i.get("role") == "demoted")),
            "per_kf_report": per_kf_report, "model": model,
            "model_params": model_params,
            "exam": {"anchor_floor_residuals": exam,
                     "worst_residual_mm": round(worst * 1000, 2)},
            "floor_npz": floor_npz}
