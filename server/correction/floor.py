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

import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from correction.config import CorrectionConfig
from correction.session import CorrectionSession

UP = np.array([0.0, 1.0, 0.0])
FLOOR_MODELS = ("level", "plane", "profile", "chunk")


def _vendor_path() -> None:
    """The fork on sys.path (``loop_utils.metric_lock.decide_change`` — the user's rule)."""
    import sys
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)


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
    if model == "chunk":
        return solve_floor_by_chunk(session, cfg, rng, log=log)
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


# ── model "chunk" (USER 2026-09-30: "tenemos la segmentación de piso, el piso por chunk
#    llevarlo a cero") ─────────────────────────────────────────────────────────────────

def _omega_chunk_of_keyframes(session: CorrectionSession) -> np.ndarray:
    """The Omega chunk that OWNS each keyframe (the reconstruction's own records)."""
    out = np.full(session.n_kf, -1, np.int64)
    rec = Path(session.output_dir) / "omega_run" / "results_output"
    for i, f in enumerate(session.frames):
        p = rec / f"frame_{int(f)}.npz"
        if p.exists():
            with np.load(p) as z:
                out[i] = int(z["chunk"]) if "chunk" in z.files else 0
    if (out < 0).any():
        raise RuntimeError(f"{int((out < 0).sum())} keyframe(s) have no Omega record in {rec} — "
                           f"the chunk of a keyframe is unknown")
    return out


def _segmented_floor_rows(session: CorrectionSession, labels) -> np.ndarray:
    """Rows of the cloud the projected segmentation labels as floor — a HINT since
    2026-10-08 (docs/plan_determinismo.md point 128): the floor of a chunk is measured
    geometrically; these rows only say whether the VLM's floor agrees. Empty when there is
    no segmentation or no instance carries one of ``labels``."""
    import json as _json
    p = Path(session.output_dir) / "segmentation_result.json"
    if not p.exists():
        return np.zeros(0, np.int64)
    try:
        inst = _json.loads(p.read_text()).get("instances") or []
    except (OSError, ValueError):
        return np.zeros(0, np.int64)
    want = {str(l).strip().lower() for l in labels}
    rows = [np.asarray(i.get("globalIndices") or [], np.int64) for i in inst
            if str(i.get("label", "")).strip().lower() in want]
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    rows = rows[(rows >= 0) & (rows < len(session.xyz))]
    return np.unique(rows)


# ── the geometric floor of a chunk (points 128 / 129, USER 2026-10-08) ───────────────────────
#
# THE FLOOR IS THE SURFACE THE CAMERA WALKS OVER — "piso" is not the label 'floor' (USER
# 2026-10-08): the horizontal SUPPORT plane under the cameras, whatever it is called (floor,
# platform, andén, walkway, slab, deck). Labels are a hint; geometry decides. A session can hold
# SEVERAL floors at different heights (a platform and the track bed below it): each is a floor,
# each stays a separate segmented object (this stage moves chunks rigidly — it never merges or
# flattens levels), and a real change of level (stepping down from the platform) is preserved as
# a STEP, never corrected as drift.
#
#   1. the candidates are the chunk's points BELOW its camera centres (a floor supports the
#      walk; ceilings, fixtures and anything above the lens are not candidates), seeded by the
#      user's low band: y < p(low_band_pct) + band_m (the same keys the per-keyframe models read);
#   2. the dominant support plane is fitted by IRLS with Tukey's biweight from that band, the
#      scale continued from band_m down to ransac_refit_band_m (halved at every converged stage:
#      the end points are the user's bands, the schedule only decides how the band tightens);
#      its support = the inliers within the refit band; the NEXT level is fitted the same way
#      on what the previous planes left, up to MAX_LEVELS per chunk (a bound, declared);
#   3. a further level is a REAL FLOOR only by THE USER'S RULE (metric_lock.decide_change):
#      the judges are the chunk's keyframes (their low-band points), ``before`` = the median
#      distance to the one-plane model, ``after`` = to the nearest of the levels, the error =
#      the session's own repeatability (a layer the reconstruction cannot repeat apart from
#      the floor is the onion, not a floor) — ≥ 5 judges, 95 % CI on the improving side,
#      median improvement ≥ the factor × that error; otherwise it is not a level;
#   4. the floor UNDER THE CAMERA of a chunk: among its real levels, the one whose
#      camera-over-floor height is continuous with the session's (the median over chunks of
#      the height over the nearest surface below the camera — a person keeps one height over
#      what they walk on); the margin to the runner-up is recorded;
#   5. the chunk's own error is a bootstrap over its inliers (Poisson weights, seeded by the
#      chunk: the large-n form of the multinomial resample, the resolution of decide_change's
#      interval): σ of the height and of the tilt; the tilt is corrected (to horizontal, about
#      the floor centroid) when it clears the factor × its σ;
#   6. the HEIGHT, surfaces by IDENTITY: the walk starts on its floor at y = 0. Every chunk moves
#      rigidly, so all its levels turn with its floor's rotation. At every seam (the first
#      keyframe the next chunk owns) each real level of the next chunk is matched to the NEAREST
#      real level of the previous chunk, both evaluated at that (x, z); the distance, the margin
#      to the second-nearest and the seam's error (the session repeatability and the two planes'
#      σ, times the factor) are recorded. The chunk lands with its floor under the camera
#      continuous with the previous chunk's corrected copy of the SAME surface: the previous
#      chunk's own floor → the jump is drift and is corrected; ANOTHER level of the previous
#      chunk (the camera stepped from the platform down to the track bed) → the chunk lands on
#      that level and the step between the two surfaces is kept;
#   7. a chunk whose tilt and height corrections do not clear the factor × their σ takes the
#      motion blended between its nearest measurable neighbours (never identity: the 5000-point
#      cut left pccr's chunk still while its neighbours turned 5.8°) — unless its own
#      measurement contradicts that blend beyond its error (its own value stands, declared);
#      the segmented floor, when there is one, only reports whether it agrees.
#   DECLARED LIMIT: a level the previous chunk never saw as a level cannot be told from drift of
#   the one it did see — it is matched to the nearest known surface; the acta names every match
#   with its distance and margin, and every step with its size.

N_BOOT_FLOOR = 2000      # resamples: the resolution of decide_change's interval (loop_utils.metric_lock)
IRLS_MAX_ITER = 100      # a BOUND on the robust fit's iterations, declared in the record when hit
IRLS_TOL = 1e-9          # convergence: the normal (rad) and the centroid (m) stop moving
MAX_LEVELS = 3           # a BOUND on the floor levels searched per chunk, declared when reached


def _weighted_plane(P: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(normal with y ≥ 0, centroid) of the weighted PCA plane through ``P``."""
    sw = float(w.sum())
    cen = (P * w[:, None]).sum(0) / sw
    Q = P - cen
    C = (Q * w[:, None]).T @ Q / sw
    vals, vecs = np.linalg.eigh(C)
    nrm = vecs[:, int(np.argmin(vals))]
    if nrm[1] < 0:
        nrm = -nrm
    return nrm, cen


def _plane_y_at(nrm: np.ndarray, cen: np.ndarray, x: float, z: float) -> float:
    """The plane's y at (x, z) — its height where a camera stands."""
    ny = float(nrm[1]) if abs(float(nrm[1])) > IRLS_TOL else IRLS_TOL
    return float(cen[1] - (nrm[0] * (x - cen[0]) + nrm[2] * (z - cen[2])) / ny)


def _irls_plane(P: np.ndarray, start_band_m: float, end_band_m: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Tukey IRLS plane from the horizontal level of ``P``'s band, the biweight scale continued
    from ``start_band_m`` down to ``end_band_m``. Returns (normal, centroid, inliers within the
    end band, record)."""
    nrm = UP.copy()
    cen = np.array([P[:, 0].mean(), float(np.median(P[:, 1])), P[:, 2].mean()])
    c = float(start_band_m)
    stages, iters, converged = 0, 0, True
    while True:
        for _ in range(IRLS_MAX_ITER):
            iters += 1
            d = (P - cen) @ nrm
            w = np.zeros(len(P))
            inside = np.abs(d) < c
            w[inside] = (1.0 - (d[inside] / c) ** 2) ** 2
            if w.sum() <= 0 or int(inside.sum()) < 3:
                break
            nrm2, cen2 = _weighted_plane(P, w)
            moved_n = float(np.arccos(np.clip(abs(float(nrm2 @ nrm)), -1.0, 1.0)))
            moved_c = float(np.linalg.norm(cen2 - cen))
            nrm, cen = nrm2, cen2
            if moved_n < IRLS_TOL and moved_c < IRLS_TOL:
                break
        else:
            converged = False
        stages += 1
        if c <= float(end_band_m) * (1.0 + IRLS_TOL):
            break
        c = max(c * 0.5, float(end_band_m))
    d = (P - cen) @ nrm
    inl = np.abs(d) < float(end_band_m)
    return nrm, cen, inl, {"stages": stages, "iterations": iters, "converged": converged}


# the CDF of Poisson(1) up to k = 20 (P(X > 20) ~ 4e-20, below the resolution of a uniform double):
# the bootstrap's weights are drawn by inverting it — exact Poisson(1) draws at a quarter of the
# cost of Generator.poisson (13 → 3 ms per 400 k points)
_POISSON1_CDF = np.cumsum([math.exp(-1.0) / math.factorial(k) for k in range(21)])
_POISSON1_CASCADE = 5            # values below this are counted by comparison; the rare rest by search


def _poisson1_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """n Poisson(1) draws as float64 — the inverse CDF of one uniform per point (the value is
    the number of CDF entries ≤ u, i.e. ``searchsorted(cdf, u, 'right')``)."""
    u = rng.random(n)
    w = (u >= _POISSON1_CDF[0]).astype(np.float64)
    for k in range(1, _POISSON1_CASCADE):
        w += (u >= _POISSON1_CDF[k])
    tail = np.flatnonzero(u >= _POISSON1_CDF[_POISSON1_CASCADE - 1])
    if len(tail):
        w[tail] = np.searchsorted(_POISSON1_CDF, u[tail], side="right")
    return w


def _bootstrap_plane(P: np.ndarray, seed: int, n_boot: int = N_BOOT_FLOOR
                     ) -> Tuple[float, float]:
    """(σ of the tilt in degrees, σ of the height in metres) of the plane through ``P``, by a
    Poisson bootstrap over its points (weights ~ Poisson(1), seeded): the height is the plane's
    y at the points' mean (x, z). Each resample's weighted PCA is computed from the points'
    second moments (one weighted sum of ten per-point products about the points' mean), so a
    resample costs its weight draw and one matrix-vector product — the same plane as
    :func:`_weighted_plane`, without re-reading the points three times."""
    P = np.asarray(P, np.float64)
    n = len(P)
    if n < 4:
        return float("inf"), float("inf")
    rng = np.random.default_rng(int(seed))
    c0 = P.mean(0)
    Q = P - c0                                         # about the mean: well-conditioned moments
    cx, cz = float(c0[0]), float(c0[2])
    Mt = np.ascontiguousarray(np.stack([np.ones(n), Q[:, 0], Q[:, 1], Q[:, 2],
                                        Q[:, 0] * Q[:, 0], Q[:, 0] * Q[:, 1], Q[:, 0] * Q[:, 2],
                                        Q[:, 1] * Q[:, 1], Q[:, 1] * Q[:, 2], Q[:, 2] * Q[:, 2]]))
    tilts = np.empty(n_boot)
    heights = np.empty(n_boot)
    for b in range(n_boot):
        w = _poisson1_weights(rng, n)
        if w.sum() <= 0:
            w[:] = 1.0
        s = Mt @ w
        sw = float(s[0])
        mu = s[1:4] / sw
        C = np.array([[s[4], s[5], s[6]], [s[5], s[7], s[8]], [s[6], s[8], s[9]]]) / sw \
            - np.outer(mu, mu)
        vals, vecs = np.linalg.eigh(C)
        nrm = vecs[:, int(np.argmin(vals))]
        if nrm[1] < 0:
            nrm = -nrm
        tilts[b] = np.degrees(np.arccos(np.clip(float(nrm @ UP), -1.0, 1.0)))
        heights[b] = _plane_y_at(nrm, c0 + mu, cx, cz)
    return float(np.std(tilts, ddof=1)), float(np.std(heights, ddof=1))


def _chunk_seed(chunk: int, rank: int) -> int:
    """The bootstrap seed of a chunk's level — derived, never a draw (repro.stable_id)."""
    from repro import stable_id
    return int(stable_id("floor_chunk_bootstrap", int(chunk), int(rank), n_hex=8), 16)


def _judge_of(cfg) -> Tuple[float, float]:
    from correction.config import judge_of
    fac, conf = judge_of(cfg)
    return float(fac), float(conf)


def _slerp_blend(Ra: np.ndarray, ta: np.ndarray, Rb: np.ndarray, tb: np.ndarray, w: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    sl = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))
    return sl([float(w)]).as_matrix()[0], (1.0 - w) * ta + w * tb


def _level_judge(P: np.ndarray, ks: np.ndarray, planes: List[Tuple[np.ndarray, np.ndarray]],
                 rep_m: float, fac: float, conf: float) -> dict:
    """THE USER'S RULE on whether the LAST of ``planes`` is a real floor level: per keyframe
    (judge) the median distance of its low-band points to the one-plane model (``before``)
    against the nearest of all the planes (``after``), error = the session's repeatability."""
    _vendor_path()
    from loop_utils.metric_lock import decide_change
    d = np.stack([np.abs((P - cen) @ nrm) for nrm, cen in planes], 1)
    d1 = d[:, 0]
    dn = d.min(1)
    before, after = [], []
    for k in np.unique(ks):
        sel = ks == k
        if not sel.any():
            continue
        before.append(float(np.median(d1[sel])))
        after.append(float(np.median(dn[sel])))
    return decide_change(np.asarray(before), np.asarray(after), error=float(rep_m),
                         error_factor=float(fac), confidence=float(conf))


def _chunk_candidates(P: np.ndarray, ks: np.ndarray, fl, chunk: int, rep_m: float, fac: float,
                      conf: float) -> List[dict]:
    """The floor LEVELS of a chunk's points below its cameras: the dominant support plane and,
    on what each level leaves, the next ones (up to MAX_LEVELS), each with its inliers,
    support, tilt, bootstrap errors and — from the second on — the user's-rule verdict on
    whether it is a real floor (``real``). The search stops at the first level that is not."""
    out: List[dict] = []
    rest = np.ones(len(P), bool)
    planes: List[Tuple[np.ndarray, np.ndarray]] = []
    for rank in range(MAX_LEVELS):
        Q = P[rest]
        if len(Q) < 3:
            break
        y0 = float(np.percentile(Q[:, 1], fl.low_band_pct))
        band = Q[Q[:, 1] < y0 + fl.band_m]
        if len(band) < 3:
            break
        nrm, cen, inl, fit = _irls_plane(band, float(fl.band_m), float(fl.ransac_refit_band_m))
        n_in = int(inl.sum())
        if n_in < 3:
            break
        tilt = float(np.degrees(np.arccos(np.clip(float(nrm @ UP), -1.0, 1.0))))
        s_tilt, s_h = _bootstrap_plane(band[inl], _chunk_seed(chunk, rank))
        planes.append((nrm, cen))
        cand = {"rank": rank, "normal": nrm, "centroid": cen, "support": n_in,
                "tilt_deg": tilt, "floor_y_m": float(cen[1]), "sigma_tilt_deg": s_tilt,
                "sigma_height_m": s_h, "fit": fit, "real": True, "judge": None,
                "points": band[inl]}
        if rank >= 1:
            # the judged points: those within the user's band of ANY level found so far —
            # the lowest band of the chunk alone never holds a higher level's points
            near = np.min(np.stack([np.abs((P - c_) @ n_) for n_, c_ in planes], 1), 1) < float(fl.band_m)
            dc = _level_judge(P[near], ks[near], planes, rep_m, fac, conf)
            cand["real"] = bool(dc["improves"])
            cand["judge"] = dc
        out.append(cand)
        if not cand["real"]:
            break
        # what this level claims leaves the pool: the next level is fitted on the rest
        d_all = (P - cen) @ nrm
        rest &= ~(np.abs(d_all) < float(fl.ransac_refit_band_m))
    return out


def solve_floor_by_chunk(session: CorrectionSession, cfg: CorrectionConfig,
                         rng: Optional[np.random.Generator] = None, log=print) -> dict:
    """ONE rigid motion per Omega chunk: the floor UNDER ITS CAMERAS → horizontal, and on the
    reference level of the surface it is (rotation about the chunk's floor centroid, then the
    vertical offset). Every keyframe of a chunk gets the same transform, so no two keyframes of
    a chunk are moved apart (the per-keyframe models failed the continuity gate on pccr and
    layered the floor) and a level change INSIDE a chunk is never flattened. GEOMETRIC and
    DETERMINISTIC (points 128 / 129, the user's floor concept of 2026-10-08; ``rng`` is
    accepted for the callers and unused): see the block comment above. A chunk whose correction
    does not clear its own measured error takes the motion blended between its measurable
    neighbours; a session without one measurable floor is declared and moves nothing (the depth
    composed with it is not lost)."""
    fl = cfg.floor
    fac, conf = _judge_of(cfg)
    rep_m = _session_repeatability(session, cfg)
    ck = _omega_chunk_of_keyframes(session)
    hint_rows = _segmented_floor_rows(session, fl.chunk_floor_labels)
    hint_mask = np.zeros(len(session.xyz), bool)
    hint_mask[hint_rows] = True
    cams = np.asarray(session.poses, np.float64)[:, :3, 3]
    n_kf = session.n_kf
    chunks = sorted(set(int(c) for c in ck.tolist()))
    per_chunk: List[dict] = []
    cands_of: Dict[int, List[dict]] = {}
    cam_xz: Dict[int, Tuple[float, float]] = {}
    cam_y_of: Dict[int, float] = {}
    first_kf: Dict[int, int] = {}
    for c in chunks:
        kfs = np.flatnonzero(ck == c)
        first_kf[c] = int(kfs[0])
        sel = np.isin(session.ks, kfs)
        P = session.xyz[sel]
        ks_c = session.ks[sel]
        cam_y_min = float(cams[kfs, 1].min())
        cam_y_med = float(np.median(cams[kfs, 1]))
        cam_y_of[c] = cam_y_med
        cam_xz[c] = (float(np.median(cams[kfs, 0])), float(np.median(cams[kfs, 2])))
        below = P[:, 1] < cam_y_min
        info = {"chunk": int(c), "keyframes": [int(kfs[0]), int(kfs[-1])],
                "points": int(len(P)), "points_below_cameras": int(below.sum()),
                "hint_points": int((hint_mask[sel] & below).sum()),
                "camera_y_median_m": cam_y_med, "repeatability_m": float(rep_m)}
        cands = (_chunk_candidates(P[below], ks_c[below], fl, c, rep_m, fac, conf)
                 if int(below.sum()) >= 3 else [])
        for q in cands:
            q["camera_over_level_m"] = cam_y_med - _plane_y_at(q["normal"], q["centroid"], *cam_xz[c])
        cands_of[c] = cands
        info["levels"] = [{"rank": q["rank"], "real": q["real"], "support": q["support"],
                           "floor_y_m": round(q["floor_y_m"], 4), "tilt_deg": round(q["tilt_deg"], 3),
                           "sigma_tilt_deg": round(q["sigma_tilt_deg"], 4),
                           "sigma_height_m": round(q["sigma_height_m"], 5),
                           "camera_over_level_m": round(q["camera_over_level_m"], 4),
                           "fit": q["fit"],
                           "judge": ({k: q["judge"][k] for k in ("improves", "n_judges", "min_judges",
                                                                 "median_delta", "ci_low", "ci_high",
                                                                 "required_delta", "reason")}
                                     if q["judge"] else None)} for q in cands]
        info["levels_bound_reached"] = bool(len(cands) >= MAX_LEVELS and cands[-1]["real"])
        if info["hint_points"] >= 3:
            info["hint_floor_y_m"] = round(float(np.median(P[(hint_mask[sel] & below)][:, 1])), 4)
        per_chunk.append(info)
    by_chunk = {int(i["chunk"]): i for i in per_chunk}

    # the floor UNDER THE CAMERA: a level stands under a camera when it has support within
    # the camera's own height over it of the camera's (x, z) — a 45° footprint, derived, no
    # radius chosen (a platform edge 60 cm below a camera standing on the track bed is below
    # the camera but not under it); among those, the one whose camera-over-floor height is
    # continuous with the session's (the median over chunks of the nearest such level — a
    # person keeps one height over what they walk on)
    from scipy.spatial import cKDTree
    for c in chunks:
        kfs = np.flatnonzero(ck == c)
        cam_tree = cKDTree(cams[kfs][:, [0, 2]])
        for q in cands_of[c]:
            pts = q["points"][:, [0, 2]]
            h = max(float(q["camera_over_level_m"]), 0.0)
            if not len(pts) or h <= 0.0:
                q["footprint_support"] = 0
                continue
            # the horizontal distance of every support point to its NEAREST camera of the chunk
            # (exact, a k-d tree over the chunk's cameras: no point x camera matrix)
            d, _ = cam_tree.query(pts, k=1, workers=1)
            q["footprint_support"] = int((d <= h).sum())
    nearest = {c: min(q["camera_over_level_m"] for q in cands_of[c] if q["real"] and q["footprint_support"])
               for c in chunks if any(q["real"] and q["footprint_support"] for q in cands_of[c])}
    H_ref = float(np.median(list(nearest.values()))) if nearest else None
    chosen: Dict[int, dict] = {}
    for c in chunks:
        info = by_chunk[c]
        for lv, q in zip(info["levels"], cands_of[c]):
            lv["footprint_support"] = int(q["footprint_support"])
        real = [q for q in cands_of[c] if q["real"] and q["footprint_support"]]
        if not real or H_ref is None:
            info.update(role="unmeasurable", why="no support plane under the cameras")
            continue
        ranked = sorted(real, key=lambda q: abs(q["camera_over_level_m"] - H_ref))
        pick = ranked[0]
        info.update(chosen_rank=int(pick["rank"]), reference_camera_height_m=H_ref,
                    camera_height_margin_m=(round(abs(ranked[1]["camera_over_level_m"] - H_ref)
                                                  - abs(pick["camera_over_level_m"] - H_ref), 4)
                                            if len(ranked) > 1 else None))
        if "hint_floor_y_m" in info:
            info["hint_agrees"] = bool(abs(info["hint_floor_y_m"] - pick["floor_y_m"]) < float(fl.band_m))
        chosen[c] = pick

    # the TILT of every chunk's floor under the camera, by the user's rule on its own error
    R_of: Dict[int, np.ndarray] = {}
    for c in chunks:
        q = chosen.get(c)
        if q is None:
            continue
        info = by_chunk[c]
        nrm, cen = q["normal"], q["centroid"]
        tilt = float(q["tilt_deg"])
        m_tilt = tilt - fac * float(q["sigma_tilt_deg"])
        R = _rot_between(nrm, UP) if m_tilt > 0.0 else np.eye(3)
        R_of[c] = R
        info.update(tilt_deg=round(tilt, 3), sigma_tilt_deg=round(float(q["sigma_tilt_deg"]), 4),
                    tilt_margin_deg=round(m_tilt, 4), tilt_applied=bool(m_tilt > 0.0),
                    inliers=int(q["support"]), error_factor=fac)

    # the HEIGHT along the walk: surfaces by IDENTITY (the user's floor concept, 2026-10-08).
    # Every chunk moves RIGIDLY, so all its levels turn with the rotation of its floor about that
    # floor's centroid. At each seam — the first keyframe the next chunk owns — every real level
    # of the next chunk is matched to the NEAREST real level of the previous chunk, both
    # evaluated at that same (x, z) and turned by their chunk's rotation; the distance, the
    # two-level margin and the seam's error (the session repeatability and the two planes'
    # sampling σ) are recorded. The walk starts on its floor at y = 0. The chunk then lands with
    # the floor UNDER ITS CAMERA continuous with the previous chunk's corrected copy of the SAME
    # surface at the seam: when that surface was the previous chunk's floor too, the jump between
    # them is drift and is corrected; when it was ANOTHER level of the previous chunk (the camera
    # stepped from the platform down to the track bed) the chunk lands on that level and the step
    # between the two surfaces stays — never corrected as drift.
    # DECLARED LIMIT: a level the previous chunk never saw cannot be told from drift of the one it
    # did see — it is matched to the nearest known surface (the record names the match, its
    # distance and margin).
    def _turned(c_, r_):
        """Level ``r_`` of chunk ``c_`` turned with the chunk's floor rotation (about the floor's
        centroid): (normal, point)."""
        R_ = R_of[c_]
        piv = chosen[c_]["centroid"]
        return R_ @ r_["normal"], R_ @ (r_["centroid"] - piv) + piv

    offset: Dict[int, float] = {}                   # the vertical offset applied to each chunk
    measurable: Dict[int, bool] = {}
    name_of: Dict[Tuple[int, int], str] = {}        # (chunk, level rank) → surface name
    surfaces: Dict[str, float] = {}                 # surface name → its corrected level where first seen
    prev_c: Optional[int] = None
    n_surf = 0
    for c in chunks:
        info = by_chunk[c]
        q = chosen.get(c)
        if q is None:
            continue
        x_s, z_s = float(cams[first_kf[c], 0]), float(cams[first_kf[c], 2])
        real = [r for r in cands_of[c] if r["real"]]
        prev_real = ([r for r in cands_of[prev_c] if r["real"]] if prev_c is not None else [])
        prev_at = [(name_of[(prev_c, int(r["rank"]))], _plane_y_at(*_turned(prev_c, r), x_s, z_s),
                    float(r["sigma_height_m"]), int(r["rank"])) for r in prev_real]
        match_rec: List[dict] = []
        matched_prev: Dict[int, Tuple[str, float]] = {}      # rank → (surface, prev corrected y)
        for r in real:
            y_r = _plane_y_at(*_turned(c, r), x_s, z_s)
            r["y_at_seam"] = y_r
            if prev_at:
                d = sorted((abs(y_r - y_p), nm, y_p, s_p) for nm, y_p, s_p, _rk in prev_at)
                best = d[0]
                err = float(np.sqrt(rep_m ** 2 + r["sigma_height_m"] ** 2 + best[3] ** 2))
                margin = (d[1][0] - best[0]) - fac * err if len(d) > 1 else None
                name = best[1]
                matched_prev[int(r["rank"])] = (name, best[2] + offset[prev_c])
                match_rec.append({"rank": int(r["rank"]), "y_at_seam_m": round(y_r, 4), "matched": name,
                                  "distance_m": round(best[0], 4), "seam_error_m": round(err, 5),
                                  "within_seam_error": bool(best[0] <= fac * err),
                                  "margin_m": (round(margin, 4) if margin is not None else None),
                                  "ambiguous": bool(margin is not None and margin <= 0.0)})
            else:
                n_surf += 1
                name = f"S{n_surf}"
                match_rec.append({"rank": int(r["rank"]), "y_at_seam_m": round(y_r, 4), "matched": None,
                                  "new_surface": name})
            name_of[(c, int(r["rank"]))] = name
        floor_name = name_of[(c, int(q["rank"]))]
        y_raw = q["y_at_seam"]
        s_h = float(q["sigma_height_m"])
        if prev_c is None:
            # the walk starts on its floor at y = 0: the reference surface
            dy = -y_raw
            seam = None
        else:
            prev_floor = by_chunk[prev_c]["surface"]
            dy = matched_prev[int(q["rank"])][1] - y_raw
            verdict = "same_surface" if floor_name == prev_floor else "level_change"
            seam = {"keyframe": int(first_kf[c]), "floor_surface_prev": prev_floor,
                    "floor_surface": floor_name, "verdict": verdict,
                    "jump_m": round(y_raw - (next(y for nm, y, _s, _rk in prev_at if nm == prev_floor)), 4)}
            if verdict != "same_surface":
                log(f"  floor[chunk {c}]: the floor under the camera is {floor_name} ({verdict}; chunk "
                    f"{prev_c} walked on {prev_floor}) — the level change is kept as a step")
        for r in real:
            surfaces.setdefault(name_of[(c, int(r["rank"]))], round(float(r["y_at_seam"] + dy), 5))
        m_h = abs(dy) - fac * s_h
        offset[c] = dy
        measurable[c] = bool(m_h > 0.0 or (seam or {}).get("verdict") == "level_change")
        info.update(surface=floor_name, levels_matched=match_rec, seam=seam,
                    floor_y_at_first_kf_m=round(y_raw, 4), offset_m=round(dy, 5),
                    sigma_height_m=round(s_h, 5), height_margin_m=round(m_h, 5),
                    height_applied=measurable[c])
        prev_c = c

    # the motions: a chunk whose tilt or height clears its own error is an ANCHOR and moves by
    # its own correction. A component that does not clear it cannot tell its own value from
    # the neighbours' (point 129: never identity by default): the neighbours' blended motion is
    # taken for it — UNLESS the chunk's own measurement contradicts that motion beyond its error
    # (a floor measured level to the millimetre next to a chunk that tilts 3° is level: the
    # neighbours' tilt is not within its error), in which case its own value stands, declared.
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    motion: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    own: Dict[int, Tuple[np.ndarray, float]] = {}        # (R, dy) the chunk itself measured
    anchors = []
    for c in chunks:
        info = by_chunk[c]
        q = chosen.get(c)
        if q is None:
            continue
        own[c] = (R_of[c], float(offset[c]))
        if not info.get("tilt_applied") and not measurable.get(c):
            info.update(role="neighbours", why="neither the tilt nor the height clears the error "
                                                "factor x the chunk's own bootstrap error")
            continue
        cen = q["centroid"]
        R = R_of[c]
        t = cen - R @ cen                                     # rotate about the floor centroid
        if measurable.get(c):
            t[1] += offset[c]                                 # … and land on the surface's level
        motion[c] = (R, t)
        info.update(role="anchor")
        kfs = np.flatnonzero(ck == c)
        anchors.append({"kf": int(np.percentile(kfs, 50, method="nearest")),
                        "rot_deg": round(float(q["tilt_deg"]) if info.get("tilt_applied") else 0.0, 3),
                        "t_m": round(abs(offset[c]) if measurable.get(c) else 0.0, 4), "chunk": int(c),
                        "surface": info.get("surface")})
        log(f"  floor[chunk {c}] kf {kfs[0]}-{kfs[-1]}: {q['support']:,} support pts of "
            f"{info['points_below_cameras']:,} below the cameras ({len(cands_of[c])} level(s), "
            f"surface {info.get('surface')}), tilt {q['tilt_deg']:.2f}° ± {q['sigma_tilt_deg']:.3f} "
            f"{'→ horizontal' if info.get('tilt_applied') else '(within error)'}, floor at "
            f"{info['floor_y_at_first_kf_m'] * 100:+.1f} cm ± {q['sigma_height_m'] * 100:.2f} → "
            + (f"offset {offset[c] * 100:+.1f} cm" if measurable.get(c) else "height within error"))
    done = sorted(motion)
    for c in chunks:
        info = by_chunk[c]
        if c in motion:
            continue
        prev = [p for p in done if p < c]
        nxt = [p for p in done if p > c]
        if prev and nxt:
            a, b = prev[-1], nxt[0]
            w = (c - a) / float(b - a)
            R, t = _slerp_blend(motion[a][0], motion[a][1], motion[b][0], motion[b][1], w)
            info.update(blended_from=[int(a), int(b)], blend_weight=round(w, 4))
        elif prev or nxt:
            a = prev[-1] if prev else nxt[0]
            R, t = motion[a]
            info.update(blended_from=[int(a)], blend_weight=None)
        else:
            R, t = np.eye(3), np.zeros(3)
            info.update(blended_from=[], why=(info.get("why") or "") + " — no measurable chunk in "
                        "the session: identity, DECLARED")
        if c in own and info.get("blended_from"):
            # the chunk's own measurement has the last word where it contradicts the blend
            q = chosen[c]
            R_own, dy_own = own[c]
            ang = float(np.degrees(np.arccos(np.clip((np.trace(R @ R_own.T) - 1.0) / 2.0, -1.0, 1.0))))
            cen = q["centroid"]
            dy_blend = float((R @ cen + t)[1] - cen[1]) if not info.get("height_applied") else None
            take_R, take_dy = R, None
            contradicted = []
            if ang > fac * float(q["sigma_tilt_deg"]):
                take_R = R_own
                contradicted.append(f"tilt (blend {ang:.3f}° from the measured, σ {q['sigma_tilt_deg']:.3f}°)")
            if dy_blend is not None and abs(dy_blend - dy_own) > fac * float(q["sigma_height_m"]):
                take_dy = dy_own
                contradicted.append(f"height (blend {dy_blend * 100:+.2f} cm vs measured "
                                    f"{dy_own * 100:+.2f} cm, σ {q['sigma_height_m'] * 100:.2f} cm)")
            if contradicted:
                t_new = cen - take_R @ cen
                t_new[1] += (take_dy if take_dy is not None else (float((R @ cen + t)[1] - cen[1]) +
                                                                  float((take_R @ cen)[1] - (R @ cen)[1])))
                R, t = take_R, t_new
                info.update(role="own_within_error", blend_contradicted=contradicted)
        motion[c] = (R, t)
        log(f"  floor[chunk {c}]: {info.get('role')} ({info.get('why')}) → motion "
            f"{'blended from chunks ' + str(info.get('blended_from')) if info.get('blended_from') else 'identity'}"
            + (f"; the chunk's own measurement contradicts the blend on {info['blend_contradicted']} — "
               f"its own value stands" if info.get("blend_contradicted") else ""))
    for c in chunks:
        kfs = np.flatnonzero(ck == c)
        R_kf[kfs] = motion[c][0]
        t_kf[kfs] = motion[c][1]
    R_kf, t_kf, n_blend = _blend_across_overlaps(session, ck, R_kf, t_kf)
    if n_blend:
        log(f"  floor[chunk]: {n_blend} keyframe(s) in chunk overlaps blended between their two chunks' "
            f"motions — no step at the seams")
    if not anchors:
        log("  floor[chunk]: NO chunk measured a floor beyond its own error — the floor stage "
            "moves nothing (DECLARED); the depth composed with it still publishes")
    steps = [{"chunk": int(i["chunk"]), **i["seam"]} for i in per_chunk
             if (i.get("seam") or {}).get("verdict") in ("level_change", "new_surface")]
    if steps:
        log(f"  floor[chunk]: {len(steps)} level change(s) kept as steps: {steps}")
    return {"R_kf": R_kf, "t_kf": t_kf, "k_kf": np.ones(n_kf), "anchors": anchors,
            "n_demoted": sum(1 for i in per_chunk if i.get("role") != "anchor"),
            "per_kf_report": per_chunk, "model": "chunk",
            "model_params": {"labels_hint": list(fl.chunk_floor_labels), "error_factor": fac,
                             "confidence": conf, "repeatability_m": float(rep_m),
                             "n_boot": N_BOOT_FLOOR, "max_levels": MAX_LEVELS,
                             "surfaces": surfaces, "level_changes": steps, "per_chunk": per_chunk,
                             "measurable_chunks": [int(c) for c in done]},
            "exam": {"anchor_floor_residuals": [], "worst_residual_mm": 0.0},
            "floor_npz": {"s": np.float64(1.0), "R": np.eye(3), "t": np.zeros(3)}}


def _blend_across_overlaps(session: CorrectionSession, owner: np.ndarray, R_kf: np.ndarray,
                           t_kf: np.ndarray):
    """USER 2026-10-01 ("revisá las posiciones de cámara, hay un escalón que antes no existía"): one rigid
    motion per OWNER chunk cut a step at every seam — pccr kf 62→63 jumped 15.0 cm where F5 had 6.9 cm, and
    from a camera there the neighbouring chunk's points no longer matched the image. Omega's chunks overlap
    (chunk_plan.json): a keyframe inside the overlap of chunks a and b takes a·(1−w) + b·w — rotation by
    slerp, translation linearly — w its position across that overlap, so the motion changes gradually and
    each chunk's floor still lands at y = 0 where it alone owns the walk. Without a chunk plan nothing is
    blended. Returns (R_kf, t_kf, n_blended)."""
    import json as _json
    from scipy.spatial.transform import Rotation, Slerp
    p = Path(session.output_dir) / "chunk_plan.json"
    if not p.exists():
        return R_kf, t_kf, 0
    ranges = [tuple(r) for r in _json.loads(p.read_text()).get("chunk_ranges") or []]
    motion = {}
    for c in sorted(set(owner.tolist())):
        k = int(np.flatnonzero(owner == c)[0])
        motion[c] = (R_kf[k].copy(), t_kf[k].copy())
    R_out, t_out, n = R_kf.copy(), t_kf.copy(), 0
    for a in range(len(ranges) - 1):
        b = a + 1
        if a not in motion or b not in motion:
            continue
        lo, hi = ranges[b][0], ranges[a][1]            # the overlap [start of b, end of a)
        if hi - lo < 2:
            continue
        sl = Slerp([0.0, 1.0], Rotation.from_matrix([motion[a][0], motion[b][0]]))
        for i in range(max(lo, 0), min(hi, len(owner))):
            w = (i - lo) / (hi - 1 - lo)
            R_out[i] = sl([w]).as_matrix()[0]
            t_out[i] = (1 - w) * motion[a][1] + w * motion[b][1]
            n += 1
    return R_out, t_out, n

