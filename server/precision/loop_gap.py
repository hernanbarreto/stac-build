"""The loop gap — drift measured in METRES where the walk revisits (audit 2026-10-23, phase 3b;
USER: "sin absolutamente ninguna error de poses escala sin nada de drift").

Drift is unobservable from the images except where the walk sees the same surface twice. F4's
loop windows (the keyframes around both ends of every SALAD pair and every instance loop) carry
tracks seen from BOTH visits. Under poses without drift the same track triangulates to the same
point from either visit; under drifted poses the two triangulations differ by the drift between
the visits. Per loop track — HELD-OUT tracks only, which no rung optimised:
  X_A      triangulated from the first visit's observations alone (side A: the lower keyframe
           indices of the window; the sides are the two index runs of the window's observations,
           split at their widest index gap — a pair whose ends lie within one window has no
           two sides and is declared ``adjacent``),
  X_B      from the second visit's alone (side B),
  gap      |X_B − X_A|: the loop's residual at these poses, in metres,
  control  |X_A2 − X_A1|: the same statistic between the two halves of side A (alternate
           keyframes) — the triangulation noise of THAT track at THAT place, measured the same
           way, never a declared floor. Each half has half the observations, so the control
           overstates the noise of the gap by up to √2: a loop within √2 of its noise reads closed.
Per loop window:
  * ``open``: the gap is worse than the control by THE USER'S RULE (metric_lock.decide_change
    with before = gap, after = control, error = the control's median): significant under the
    bootstrap, at least min_judge_closures(confidence) tracks judging, and a median excess of at
    least error_factor × the control's median. A loop that is not open is CLOSED at these poses.
  * ``coherent``: a rigid fit X_B ≈ R·X_A + t (Kabsch, one Huber reweighting at the median
    residual) leaves a residual that reads CLOSED by the same rule (before = the fit's residual,
    after = control: not open). An open AND coherent loop is a rigid offset between the visits —
    drift the solve left in place (the offset and the angle reported); an open and INCOHERENT one
    is tracks that disagree with each other beyond their noise even once the best rigid offset is
    removed — not a rigid revisit (a false pair, a moving object), or a revisit with more than an
    offset in it (``explained_fraction`` = 1 − residual / gap says how much the offset explains).
    The offset reported is the surface's displacement between the visits at its centroid, the
    rotation about that centroid apart.
Measured at the initial poses (what Omega and the gauge left) and at the applied rung (what F5
leaves); reported in refine.json["loops"]. An instrument, never a correction by itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ADJACENT = "adjacent"           # the window's observations form one index run: no two visits
VERSION = 1


def loop_sides(indices: Sequence[int]) -> Optional[Tuple[set, set]]:
    """(side A, side B) keyframe-index sets of a loop window from the indices its observations
    cover: the two runs either side of the widest gap in the sorted indices; None when the
    indices are one contiguous run (the pair's ends lie within one window)."""
    idx = np.unique(np.asarray(list(indices), np.int64))
    if idx.size < 2:
        return None
    gaps = np.diff(idx)
    k = int(np.argmax(gaps))
    if gaps[k] <= 1:
        return None
    return set(idx[:k + 1].tolist()), set(idx[k + 1:].tolist())


def _halves(obs: List[Tuple[int, np.ndarray]]) -> Tuple[List, List]:
    """Alternate keyframes (by index order) of one side: the two halves span the same baseline."""
    srt = sorted(obs, key=lambda o: o[0])
    return srt[0::2], srt[1::2]


def kabsch(P: np.ndarray, Q: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The rigid (R, t) minimising Σ w |R·P + t − Q|² (Kabsch / Umeyama without scale)."""
    P = np.asarray(P, np.float64); Q = np.asarray(Q, np.float64)
    w = np.ones(len(P)) if w is None else np.asarray(w, np.float64)
    w = w / w.sum()
    pc = (w[:, None] * P).sum(0); qc = (w[:, None] * Q).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rigid_gap_fit(XA: np.ndarray, XB: np.ndarray) -> Dict[str, Any]:
    """X_B ≈ R·X_A + t with one Huber reweighting at the median residual of the plain fit
    (weights min(1, median/|r|): no declared scale). Returns the offset (|t| in m, the angle in
    degrees) and the residuals after the fit."""
    R, t = kabsch(XA, XB)
    r = np.linalg.norm((XA @ R.T + t) - XB, axis=1)
    med = float(np.median(r))
    if med > 0.0:
        w = np.minimum(1.0, med / np.maximum(r, np.finfo(float).tiny))
        R, t = kabsch(XA, XB, w)
        r = np.linalg.norm((XA @ R.T + t) - XB, axis=1)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    # the offset of the surface between the visits at its own centroid (the rotation about that
    # centroid separated out as the angle): R·x̄_A + t − x̄_A
    xa = XA.mean(axis=0)
    off = float(np.linalg.norm(R @ xa + t - xa))
    return {"translation_m": off, "rotation_deg": ang, "residual_m": r, "R": R, "t": t}


def _rule(before: np.ndarray, after: np.ndarray, error: float, error_factor: float, confidence: float,
          n_boot: int, seed: int) -> dict:
    from precision.refine import _vendor_path
    _vendor_path()
    from loop_utils.metric_lock import decide_change
    v = decide_change(before, after, error=float(error), error_factor=float(error_factor),
                      confidence=float(confidence), n_boot=int(n_boot), seed=int(seed))
    return {k: v[k] for k in ("improves", "significant", "enough_judges", "beyond_error", "n_obs",
                              "n_judges", "min_judges", "median_delta", "ci_low", "ci_high", "error",
                              "required_delta", "reason")}


def measure_loops(held_g: Dict[int, List[Tuple[int, np.ndarray]]], window_of: Dict[int, int],
                  loop_windows: Sequence[int], w2c: np.ndarray, params: Sequence[float], solver: dict,
                  min_tri_deg: float, *, error_factor: float, confidence: float, n_boot: int,
                  seed: int) -> Dict[str, Any]:
    """The instrument at one state (poses ``w2c``, camera ``params``). ``held_g``: the held-out
    tracks grouped by keyframe index (refine.group_tracks); ``window_of``: track → F4 window id;
    ``loop_windows``: the ids of the loop windows. Returns the per-window report and a summary."""
    from precision.refine import triangulate_tracks
    by_win: Dict[int, List[int]] = {}
    for t in held_g:
        w = window_of.get(t)
        if w is not None and w in set(loop_windows):
            by_win.setdefault(int(w), []).append(t)
    windows: Dict[str, Any] = {}
    gA: Dict[Tuple[int, int], List] = {}; gB: Dict[Tuple[int, int], List] = {}
    gA1: Dict[Tuple[int, int], List] = {}; gA2: Dict[Tuple[int, int], List] = {}
    sides: Dict[int, Tuple[set, set]] = {}
    for w in sorted(set(int(x) for x in loop_windows)):
        tracks = by_win.get(w, [])
        idx = [i for t in tracks for i, _ in held_g[t]]
        s = loop_sides(idx) if idx else None
        if s is None:
            windows[str(w)] = {"status": ADJACENT if idx else "no_heldout_track", "n_tracks": 0}
            continue
        sides[w] = s
        A, B = s
        for t in tracks:
            oa = [o for o in held_g[t] if o[0] in A]
            ob = [o for o in held_g[t] if o[0] in B]
            if len(oa) < 2 or len(ob) < 2:
                continue
            a1, a2 = _halves(oa)
            if len(a1) < 2 or len(a2) < 2:
                continue
            key = (w, t)
            gA[key] = oa; gB[key] = ob; gA1[key] = a1; gA2[key] = a2
    per_track: Dict[int, Dict[str, Any]] = {}
    tri = {}
    for name, g in (("A", gA), ("B", gB), ("A1", gA1), ("A2", gA2)):
        tri[name] = triangulate_tracks(g, w2c, params, solver, min_tri_deg) if g else {}
    for w, (A, B) in sides.items():
        keys = sorted(k for k in gA if k[0] == w and all(k in tri[n] for n in ("A", "B", "A1", "A2")))
        if not keys:
            windows[str(w)] = {"status": "no_track_triangulates_on_both_sides", "n_tracks": 0,
                               "side_a": sorted(A), "side_b": sorted(B)}
            continue
        XA = np.array([tri["A"][k] for k in keys]); XB = np.array([tri["B"][k] for k in keys])
        XA1 = np.array([tri["A1"][k] for k in keys]); XA2 = np.array([tri["A2"][k] for k in keys])
        gap = np.linalg.norm(XB - XA, axis=1)
        ctrl = np.linalg.norm(XA2 - XA1, axis=1)
        fit = rigid_gap_fit(XA, XB)
        err = float(np.median(ctrl))
        open_v = _rule(gap, ctrl, err, error_factor, confidence, n_boot, seed)
        # coherent = the residual after the rigid offset is not open (the same rule, residual vs control)
        coh_v = _rule(fit["residual_m"], ctrl, err, error_factor, confidence, n_boot, seed)
        per_track[w] = {"keys": [k[1] for k in keys], "gap": gap, "control": ctrl}
        windows[str(w)] = {
            "status": "judged", "n_tracks": int(len(keys)), "side_a": sorted(A), "side_b": sorted(B),
            "gap_median_m": float(np.median(gap)), "gap_p90_m": float(np.percentile(gap, 90)),
            "control_median_m": err,
            "rigid_fit": {"translation_m": fit["translation_m"], "rotation_deg": fit["rotation_deg"],
                          "residual_median_m": float(np.median(fit["residual_m"])),
                          "explained_fraction": float(1.0 - np.median(fit["residual_m"]) / np.median(gap))
                          if float(np.median(gap)) > 0.0 else None},
            "open": bool(open_v["improves"]), "open_verdict": open_v,
            "coherent": not bool(coh_v["improves"]), "residual_open_verdict": coh_v}
    judged = [v for v in windows.values() if v.get("status") == "judged"]
    summary = {"n_loop_windows": len(windows), "n_judged": len(judged),
               "n_adjacent": sum(v.get("status") == ADJACENT for v in windows.values()),
               "n_open": sum(v["open"] for v in judged),
               "n_open_coherent": sum(v["open"] and v["coherent"] for v in judged),
               "n_open_incoherent": sum(v["open"] and not v["coherent"] for v in judged),
               "gap_median_m": (float(np.median([v["gap_median_m"] for v in judged])) if judged else None),
               "gap_max_m": (float(max(v["gap_median_m"] for v in judged)) if judged else None),
               "control_median_m": (float(np.median([v["control_median_m"] for v in judged])) if judged else None),
               "largest_offset_m": (float(max(v["rigid_fit"]["translation_m"] for v in judged)) if judged else None)}
    return {"windows": windows, "summary": summary, "_per_track": per_track}


def judge_gap_change(before: Dict[str, Any], after: Dict[str, Any], *, error_factor: float, confidence: float,
                     n_boot: int, seed: int) -> dict:
    """THE USER'S RULE on the loop gaps of two states: paired per held-out loop track (the tracks
    judged in both), before − after in metres, error = the larger of the two states' median
    controls, clustered by loop window (the tracks of one window share its drift). ``improves``
    = the loops closed further; ``worsens`` = they opened."""
    from precision.refine import _vendor_path
    _vendor_path()
    from loop_utils.metric_lock import decide_change
    b_all, a_all, cl, errs = [], [], [], []
    for w, pb in before.get("_per_track", {}).items():
        pa = after.get("_per_track", {}).get(w)
        if pa is None:
            continue
        ib = {t: i for i, t in enumerate(pb["keys"])}
        common = [t for t in pa["keys"] if t in ib]
        if not common:
            continue
        ia = {t: i for i, t in enumerate(pa["keys"])}
        b_all.extend(float(pb["gap"][ib[t]]) for t in common)
        a_all.extend(float(pa["gap"][ia[t]]) for t in common)
        cl.extend([int(w)] * len(common))
        errs.append(max(float(np.median(pb["control"])), float(np.median(pa["control"]))))
    if not b_all:
        return {"improves": False, "worsens": False, "n_obs": 0, "n_judges": 0,
                "reason": "no held-out loop track judged in both states"}
    v = decide_change(b_all, a_all, error=float(max(errs)), error_factor=float(error_factor),
                      confidence=float(confidence), clusters=cl, n_boot=int(n_boot), seed=int(seed))
    v["unit"] = "m"
    v["clustered_by"] = "loop window"
    # the mirror of ``improves``: worse by the same rule (significant, enough judges, beyond the error) —
    # decide_change's own ``worsens`` is the bare interval, which one judge alone makes a point
    v["worsens_by_rule"] = bool(v["worsens"]) and bool(v["enough_judges"]) and bool(
        -float(v["median_delta"]) >= float(v["required_delta"]))
    return {k: v[k] for k in ("improves", "worsens", "worsens_by_rule", "significant", "enough_judges",
                              "beyond_error", "n_obs", "n_judges", "min_judges", "median_delta", "ci_low",
                              "ci_high", "error", "required_delta", "reason", "unit", "clustered_by")}


def prior_relaxation(state: Dict[str, Any], centres: np.ndarray, sigma_rel: float) -> Dict[str, Any]:
    """How far the OPEN loops stand from the gauge priors: per open window, its gap median over the
    prior's own random-walk σ across the loop's span, σ_rel·sqrt(Σ d² from the centre of side A to
    the centre of side B) — the factor the priors would need to let that loop close. The factor
    proposed is the median over the open windows (≥ 1); none open = nothing to relax."""
    c = np.asarray(centres, np.float64)
    d2 = np.sum(np.diff(c, axis=0) ** 2, axis=1)
    acc = np.concatenate([[0.0], np.cumsum(d2)])
    per: Dict[str, Any] = {}
    ratios = []
    for w, rep in state["windows"].items():
        if rep.get("status") != "judged" or not rep.get("open"):
            continue
        ia = int(np.median(rep["side_a"])); ib = int(np.median(rep["side_b"]))
        span = float(np.sqrt(max(acc[max(ia, ib)] - acc[min(ia, ib)], 0.0)))
        sig = float(sigma_rel) * span
        ratio = (float(rep["gap_median_m"]) / sig) if sig > 0.0 else float("nan")
        per[w] = {"span_m": span, "prior_sigma_m": sig, "gap_median_m": rep["gap_median_m"], "gap_over_sigma": ratio}
        if np.isfinite(ratio):
            ratios.append(ratio)
    factor = float(max(1.0, np.median(ratios))) if ratios else 1.0
    return {"per_open_window": per, "n_open": len(per), "factor": factor,
            "method": "median over the open loop windows of gap / (sigma_rel x sqrt(sum d^2 across the loop)), >= 1"}


def _public(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"windows": state["windows"], "summary": state["summary"]}


def loop_report(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """refine.json["loops"]: the instrument at the initial poses and at the applied rung, the
    reading in words per window, the counts."""
    verdicts: Dict[str, str] = {}
    for w, a in after["windows"].items():
        b = before["windows"].get(w, {})
        if a.get("status") != "judged":
            verdicts[w] = a.get("status", "not_judged")
        elif not a["open"]:
            verdicts[w] = "closed" if b.get("status") != "judged" or not b.get("open") else "closed_by_f5"
        elif a["coherent"]:
            verdicts[w] = "open_coherent: a rigid offset between the visits remains — drift F5 left in place"
        else:
            verdicts[w] = "open_incoherent: the visits' tracks disagree with each other — not a rigid revisit"
    sb, sa = before["summary"], after["summary"]
    return {"version": VERSION, "unit": "m",
            "method": "held-out loop tracks triangulated from each visit alone; gap |X_B - X_A| vs the "
                      "control |X_A2 - X_A1| (alternate keyframes of visit A); open / coherent by "
                      "metric_lock.decide_change (USER 2026-10-07)",
            "before": _public(before), "after": _public(after), "verdict_by_window": verdicts,
            "summary": {"n_loop_windows": sa["n_loop_windows"], "n_judged": sa["n_judged"],
                        "n_adjacent": sa["n_adjacent"],
                        "open_before": sb["n_open"], "open_after": sa["n_open"],
                        "open_coherent_after": sa["n_open_coherent"],
                        "open_incoherent_after": sa["n_open_incoherent"],
                        "gap_median_before_m": sb["gap_median_m"], "gap_median_after_m": sa["gap_median_m"],
                        "gap_max_after_m": sa["gap_max_m"], "control_median_after_m": sa["control_median_m"],
                        "largest_offset_after_m": sa["largest_offset_m"]}}
