"""The scale lock — the solved trajectory's local scale re-measured against the gauge's and
re-imposed (audit 2026-10-23, phase 3b; USER: "sin ... error de ... escala").

Reprojection is blind to scale: a bundle adjustment holds the metre only through its position
priors, which also hold the SHAPE Omega left (drift included). Relaxing the priors so the tracks
may close the loops (refine's σ ladder) lets the metre float with them. This instrument separates
the two: the metre is read again from the STEP LENGTHS of the walk, which the gauge measured
(F2: a per-keyframe scale from DA3 and the other instruments) and which drift leaves alone — a
rotation drift turns the steps, a position drift displaces them, neither changes their length.

  r_k      = |C^g_{k+1} − C^g_k| / |C_{k+1} − C_k|      gauge step over solved step, per step
  log s    = the median of log r_k weighted by the gauge step length (a longer step measures
             the metre better); its error σ and interval by a bootstrap over the steps (seeded)
  trend    = the weighted least-squares slope of log r_k along the gauge chainage (per 100 m),
             with its bootstrap interval — a residual scale DRIFT between the solve and the gauge,
             REPORTED (F2 owns the scale along the walk; a significant trend is a finding)
  applied  by THE USER'S RULE: the interval of log s excludes 0, at least min_judge_closures
             (confidence) steps judge, and |log s| ≥ error_factor × σ — then the whole solution
             (poses, landmarks) is scaled by s about the gauge trajectory's centroid; otherwise the
             solve's metre stands. Reported either way (refine.json["scale_lock"]).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

VERSION = 1


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Sort-based weighted median (deterministic): the smallest x whose cumulative weight reaches
    half of the total."""
    x = np.asarray(x, np.float64); w = np.asarray(w, np.float64)
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, 0.5 * cw[-1], side="left"))
    return float(x[order][min(k, len(x) - 1)])


def centres_of(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, np.float64)
    return -np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3])


def measure(c_gauge: np.ndarray, c_solved: np.ndarray, *, error_factor: float, confidence: float,
            n_boot: int, seed: int, min_judges: Optional[int] = None) -> Dict[str, Any]:
    """The instrument on two trajectories (N, 3) with the same keyframes."""
    cg = np.asarray(c_gauge, np.float64); cs = np.asarray(c_solved, np.float64)
    if cg.shape != cs.shape or cg.ndim != 2 or cg.shape[1] != 3:
        raise ValueError(f"scale_lock: trajectories {cg.shape} vs {cs.shape}")
    if min_judges is None:
        from precision.refine import _vendor_path
        _vendor_path()
        from loop_utils.loop_judge import min_judge_closures
        min_judges = min_judge_closures(float(confidence))
    dg = np.linalg.norm(np.diff(cg, axis=0), axis=1)
    ds = np.linalg.norm(np.diff(cs, axis=0), axis=1)
    ok = (dg > 0.0) & (ds > 0.0) & np.isfinite(dg) & np.isfinite(ds)
    out: Dict[str, Any] = {"version": VERSION, "n_steps": int(ok.sum()), "n_steps_total": int(dg.size),
                           "min_judges": int(min_judges), "error_factor": float(error_factor),
                           "confidence": float(confidence), "n_boot": int(n_boot), "seed": int(seed)}
    if int(ok.sum()) < 2:
        out.update({"log_s": 0.0, "s": 1.0, "sigma": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "apply": False, "reason": "fewer than two usable steps"})
        return out
    lr = np.log(dg[ok] / ds[ok])
    w = dg[ok]
    chain = np.concatenate([[0.0], np.cumsum(dg)])
    mid = (0.5 * (chain[:-1] + chain[1:]))[ok]
    log_s = weighted_median(lr, w)

    def slope(y, x, ww):
        xm = np.sum(ww * x) / np.sum(ww); ym = np.sum(ww * y) / np.sum(ww)
        den = np.sum(ww * (x - xm) ** 2)
        return float(np.sum(ww * (x - xm) * (y - ym)) / den) if den > 0.0 else float("nan")

    tr = slope(lr, mid, w)
    rng = np.random.default_rng(int(seed))
    n = int(lr.size)
    bs = np.empty(int(n_boot)); bt = np.empty(int(n_boot))
    for q in range(int(n_boot)):
        pick = rng.integers(0, n, n)
        bs[q] = weighted_median(lr[pick], w[pick])
        bt[q] = slope(lr[pick], mid[pick], w[pick])
    alpha = (1.0 - float(confidence)) / 2.0
    lo, hi = float(np.percentile(bs, 100.0 * alpha)), float(np.percentile(bs, 100.0 * (1.0 - alpha)))
    sigma = float(np.std(bs))
    bt = bt[np.isfinite(bt)]
    tlo, thi = ((float(np.percentile(bt, 100.0 * alpha)), float(np.percentile(bt, 100.0 * (1.0 - alpha))))
                if bt.size else (float("nan"), float("nan")))
    significant = bool(lo > 0.0 or hi < 0.0)
    enough = bool(n >= int(min_judges))
    beyond = bool(abs(log_s) >= float(error_factor) * sigma)
    apply = significant and enough and beyond
    failed = [k for k, v in (("judges", enough), ("significance", significant), ("error", beyond)) if not v]
    out.update({"log_s": float(log_s), "s": float(np.exp(log_s)), "sigma": sigma, "ci_low": lo, "ci_high": hi,
                "significant": significant, "enough_judges": enough, "beyond_error": beyond,
                "required_log_s": float(error_factor) * sigma, "apply": bool(apply), "failed": failed,
                "trend_per_100m": {"slope": (tr * 100.0) if np.isfinite(tr) else None,
                                   "ci_low": (tlo * 100.0) if np.isfinite(tlo) else None,
                                   "ci_high": (thi * 100.0) if np.isfinite(thi) else None,
                                   "significant": bool(np.isfinite(tlo) and np.isfinite(thi) and (tlo > 0.0 or thi < 0.0))},
                "reason": ("applied: the solve's metre differs from the gauge's by the user's rule" if apply
                           else "not applied: " + ", ".join(failed))})
    return out


def apply_similarity(w2c: np.ndarray, X: Dict[int, np.ndarray], s: float,
                     centre: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Scale the solution by ``s`` about ``centre`` (world): camera centres C' = s(C − c̄) + c̄,
    rotations unchanged, landmarks likewise — every reprojection is the same pixel."""
    w2c = np.asarray(w2c, np.float64)
    c = np.asarray(centre, np.float64)
    C = centres_of(w2c)
    Cn = float(s) * (C - c) + c
    out = w2c.copy()
    out[:, :3, 3] = -np.einsum("nij,nj->ni", w2c[:, :3, :3], Cn)
    Xn = {t: float(s) * (np.asarray(P, np.float64) - c) + c for t, P in X.items()}
    return out, Xn
