"""Rolling shutter — from declared to corrected, judged (audit 2026-10-23, phase 3f; USER: the
variables the chain only declared must be certified too).

A phone's sensor reads its rows one after another: row v of keyframe i is exposed at a time
offset (v − c_y) / H · τ from the frame's reference row, τ the READOUT TIME. While the camera
moves, the pose that made the observation is not the keyframe's but the keyframe's displaced by
that fraction of its motion. F5 solved every keyframe as one pose (a global shutter): the error
grows with the row's distance from the centre and with the keyframe's speed — the signature the
chain measured as a correlation (``refine.rolling_shutter_diag``) and declared.

THE MODEL. The keyframe's motion over its neighbours' interval, (ΔR, ΔC) from keyframe i−1 to
i+1 over Δf frames, is the velocity; the pose at row v is the keyframe's moved by the fraction
s(v) = (v − c_y) / H · τ_f / Δf of that motion, τ_f the readout in FRAME PERIODS (a device
constant; no clock is needed, only frame numbers). The predicted pixel displacement of an
observation per unit τ_f, d(p) = project(X, P_i(s)) − project(X, P_i) at s for τ_f = 1, is the
derivative of the model.

THE MEASUREMENT. On the held-out tracks (landmarks triangulated with the best rung) the residual
r = project(X, P_i) − p of every observation is regressed on its displacement d: τ_f =
−Σ r·d / Σ d·d — one scalar, its error and interval by a bootstrap over keyframes (the
observations of one keyframe share its motion). MEASURED, never declared.

THE CORRECTION. Every observation is RECTIFIED to the global-shutter pose: p_GS = p − τ_f · d(p),
its own noise kept (X enters only through the smooth displacement), and the bundle adjustment of
the best rung is re-solved on the rectified fit observations (the same priors, the same camera
freedom). A converged solve ABSORBS most of a rolling-shutter effect into its landmarks and
poses (a point seen at similar rows in consecutive keyframes carries the same shift in every
view: its landmark takes it; measured on the synthetic walk: the residuals show 17 % of τ_f), so
τ_f is not read off the residuals — it is SEARCHED: every candidate on a grid up to
``rs_max_frames`` rectifies and re-solves, the held-out median (held-out tracks rectified the
same way) is the criterion, the grid refined ``rs_passes`` times around the minimum. The winner
is judged against the global-shutter best by THE USER'S RULE (metric_lock.decide_change,
keyframe clusters, the solver error as the bar). Taken only when it improves; reported either way
with the search, the residuals' own reading (a lower bound), the Spearman correlation before and
after.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _log_so3(R: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


def _exp_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def keyframe_motion(w2c: np.ndarray, frames: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per keyframe: (ω — the world rotation vector from keyframe i−1 to i+1, ΔC — the centre
    motion, Δf — the frame gap of that interval). The ends use their one neighbour."""
    N = len(w2c)
    W = np.asarray(w2c, np.float64)
    R = W[:, :3, :3]; C = -np.einsum("nji,nj->ni", R, W[:, :3, 3])
    f = np.arange(N, dtype=np.float64) if frames is None else np.asarray(frames, np.float64)
    omega = np.zeros((N, 3)); dC = np.zeros((N, 3)); df = np.ones(N)
    for i in range(N):
        a, b = max(0, i - 1), min(N - 1, i + 1)
        if a == b:
            continue
        # c2w = Rᵀ; the world rotation taking camera a's orientation to camera b's: c2w_b · c2w_aᵀ
        omega[i] = _log_so3(R[b].T @ R[a])
        dC[i] = C[b] - C[a]
        df[i] = float(f[b] - f[a]) if f[b] > f[a] else float(b - a)
    return omega, dC, df


def pose_at(w2c_i: np.ndarray, omega_i: np.ndarray, dC_i: np.ndarray, s: float) -> np.ndarray:
    """The keyframe's pose moved by the fraction ``s`` of its neighbours' motion (world)."""
    R = w2c_i[:3, :3]; C = -R.T @ w2c_i[:3, 3]
    Rw = _exp_so3(s * omega_i)
    c2w_R = Rw @ R.T
    C2 = C + s * dC_i
    out = np.eye(4)
    out[:3, :3] = c2w_R.T
    out[:3, 3] = -c2w_R.T @ C2
    return out


def displacements(groups: Dict[int, List], X: Dict[int, np.ndarray], w2c: np.ndarray, params,
                  H: int, motion, cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every observation of a triangulated track: (keyframe index, residual r (2), the
    displacement d (2) per unit τ_f, the row fraction). ``project`` is refine's."""
    from precision.refine import project
    omega, dC, df = motion
    diag = float(np.hypot(2.0 * float(params[2]), 2.0 * float(params[3])))   # the frame's diagonal, from the principal point
    ks, rs, ds, rsp = [], [], [], []
    for t, P in X.items():
        for i, p in groups[t]:
            p = np.asarray(p, np.float64)
            s = (float(p[1]) - cy) / float(H) / float(df[i])      # per unit τ_f
            q0 = project(P[None], w2c[i], params)[0]
            q1 = project(P[None], pose_at(w2c[i], omega[i], dC[i], s), params)[0]
            if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(q1))):
                continue
            if np.linalg.norm(q0 - p) > diag or np.linalg.norm(q1 - q0) > diag:
                continue                    # a degenerate projection (z → 0), never a vote
            ks.append(i); rs.append(q0 - p); ds.append(q1 - q0)
            rsp.append(abs(float(p[1]) - cy) * float(np.linalg.norm(omega[i])) / float(df[i]))
    if not ks:
        return np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.asarray(ks), np.asarray(rs), np.asarray(ds), np.asarray(rsp)


def measure_readout(groups: Dict[int, List], X: Dict[int, np.ndarray], w2c: np.ndarray, params, H: int,
                    motion, *, n_boot: int, seed: int, confidence: float) -> Dict[str, Any]:
    """τ_f by least squares of the residuals on the displacements, bootstrapped by keyframe."""
    cy = float(params[3])
    ks, r, d, _ = displacements(groups, X, w2c, params, H, motion, cy)
    out: Dict[str, Any] = {"n_obs": int(len(ks)), "n_keyframes": int(len(np.unique(ks))) if len(ks) else 0}
    if len(ks) < 2 or float(np.sum(d * d)) <= 0.0:
        out.update({"tau_frames": 0.0, "sigma": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                    "significant": False, "reason": "no displacement to regress on (the keyframes do not move)"})
        return out
    num = -np.sum(r * d, axis=1); den = np.sum(d * d, axis=1)
    tau = float(num.sum() / den.sum())
    # one Huber reweighting at the median residual of the fit: a track that is not a rolling-shutter
    # effect (an outlier) weighs no more than a typical one
    e = np.linalg.norm(r + tau * d, axis=1)
    med = float(np.median(e))
    w = np.minimum(1.0, med / np.maximum(e, np.finfo(float).tiny)) if med > 0.0 else np.ones(len(e))
    num = num * w; den = den * w
    tau = float(num.sum() / den.sum()) if den.sum() > 0 else tau
    labels, inv = np.unique(ks, return_inverse=True)
    k = len(labels)
    num_k = np.bincount(inv, weights=num, minlength=k); den_k = np.bincount(inv, weights=den, minlength=k)
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_boot))
    for q in range(int(n_boot)):
        pick = rng.integers(0, k, k)
        dd = den_k[pick].sum()
        boots[q] = num_k[pick].sum() / dd if dd > 0 else tau
    alpha = (1.0 - float(confidence)) / 2.0
    lo, hi = float(np.percentile(boots, 100.0 * alpha)), float(np.percentile(boots, 100.0 * (1.0 - alpha)))
    out.update({"tau_frames": tau, "sigma": float(np.std(boots)), "ci_low": lo, "ci_high": hi,
                "significant": bool(lo > 0.0 or hi < 0.0),
                "explained_px": float(np.sqrt(np.mean(np.sum((tau * d) ** 2, axis=1)))),
                "residual_px_before": float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))),
                "residual_px_after_linear": float(np.sqrt(np.mean(np.sum((r + tau * d) ** 2, axis=1))))})
    return out


def rectify(groups: Dict[int, List], X: Dict[int, np.ndarray], w2c: np.ndarray, params, H: int, motion,
            tau: float) -> Dict[int, List]:
    """Every observation of a triangulated track moved to its global-shutter pixel, p − τ_f·d(p);
    a track without a landmark keeps its observations as they are."""
    from precision.refine import project
    omega, dC, df = motion
    cy = float(params[3])
    out: Dict[int, List] = {}
    for t, obs in groups.items():
        P = X.get(t)
        if P is None:
            out[t] = list(obs)
            continue
        new = []
        for i, p in obs:
            p = np.asarray(p, np.float64)
            s = (float(p[1]) - cy) / float(H) / float(df[i])
            q0 = project(P[None], w2c[i], params)[0]
            q1 = project(P[None], pose_at(w2c[i], omega[i], dC[i], s), params)[0]
            d = q1 - q0
            new.append((i, p - float(tau) * d if np.all(np.isfinite(d)) else p))
        out[t] = new
    return out


def spearman(groups: Dict[int, List], X: Dict[int, np.ndarray], w2c: np.ndarray, params, H: int, motion) -> Optional[float]:
    """The declared diagnostic: residual magnitude vs |row offset| × angular speed."""
    from scipy.stats import spearmanr
    cy = float(params[3])
    ks, r, _d, rowspeed = displacements(groups, X, w2c, params, H, motion, cy)
    if len(ks) < 3:
        return None
    rho, _p = spearmanr(rowspeed, np.linalg.norm(r, axis=1))
    return float(rho) if np.isfinite(rho) else None


def _evaluate(tau: float, best, fit_g, held_g, H: int, cfg, solver, solve_fn, heldout_fn, frames, name: str):
    """One candidate readout: the fit and held-out observations rectified with ``tau`` from the
    global-shutter best's landmarks and poses, one solve from the best, the held-out of the solve
    on the held-out observations rectified with the SOLVE's landmarks and poses."""
    from precision.refine import triangulate_tracks
    params = best.params_by_block[0]
    motion = keyframe_motion(best.w2c, frames)
    Xf = triangulate_tracks(fit_g, best.w2c, params, solver, cfg.min_tri_deg)
    fit_r = rectify(fit_g, Xf, best.w2c, params, H, motion, tau)
    rung = solve_fn(best, fit_r, name)
    Xh = triangulate_tracks(held_g, rung.w2c, rung.params_by_block[0], solver, cfg.min_tri_deg)
    held_r = rectify(held_g, Xh, rung.w2c, rung.params_by_block[0], H, keyframe_motion(rung.w2c, frames), tau)
    held = heldout_fn(rung, held_r)
    med = float(np.median(list(held.values()))) if held else float("inf")
    return rung, held, med


def judge_rolling_shutter(best, fit_g, held_g, wh, cfg, solver, *, fac: float, solve_fn: Callable,
                          heldout_fn: Callable, judge_fn: Callable, clusters, solver_error: float,
                          frames: Optional[np.ndarray] = None, log: Callable = print) -> Tuple[Any, Dict[int, float], Dict[str, Any]]:
    """The stage. The readout is a device constant the residuals of a converged solve only show a
    FRACTION of (the solve absorbs the rest into its landmarks and poses — measured: 17 % on the
    synthetic walk), so τ_f is not read off the residuals: it is SEARCHED. Every candidate τ_f on a
    grid up to ``rs_max_frames`` (step ``rs_grid_frames``) rectifies the observations and re-solves
    from the global-shutter best; the held-out median of each candidate is the criterion; the
    grid is refined ``rs_passes`` times around the minimum (the step halving). The regression on
    the global-shutter residuals is reported as the first reading (a lower bound). The winner is
    judged against the global-shutter best by the user's rule; taken only when it improves.
    Returns (the rung taken — the RS solve or ``best`` —, its held-out, the report)."""
    from precision.refine import triangulate_tracks
    H = int(wh[1])
    params = best.params_by_block[0]
    motion = keyframe_motion(best.w2c, frames)
    Xh = triangulate_tracks(held_g, best.w2c, params, solver, cfg.min_tri_deg)
    meas = measure_readout(held_g, Xh, best.w2c, params, H, motion, n_boot=int(cfg.permutations),
                           seed=int(cfg.seed), confidence=float(cfg.heldout_confidence))
    h_best = heldout_fn(best, held_g)
    med_gs = float(np.median(list(h_best.values()))) if h_best else float("inf")
    rep: Dict[str, Any] = {"measured_on_global_shutter_residuals": meas,
                           "spearman_before": spearman(held_g, Xh, best.w2c, params, H, motion),
                           "search": [], "taken": False, "corrected": False, "heldout_median_px_global_shutter": med_gs}
    if not meas["significant"]:
        rep["reason"] = ("the readout read on the global-shutter residuals is not significant — no rolling-shutter "
                         "signature to search for")
        log(f"[rs] readout on the residuals {meas['tau_frames']:.4f} frame(s) ± {meas.get('sigma', float('nan')):.4f}: "
            f"not significant — the global shutter stands")
        return best, None, rep
    step = float(cfg.rs_grid_frames)
    n_grid = int(round(float(cfg.rs_max_frames) / step))
    evaluated: Dict[float, Tuple[Any, Dict[int, float], float]] = {0.0: (best, h_best, med_gs)}
    order = []

    def ev(tau: float, name: str):
        key = round(float(tau), 6)
        if key not in evaluated:
            if key <= 0.0:
                evaluated[key] = (best, h_best, med_gs)
            else:
                evaluated[key] = _evaluate(key, best, fit_g, held_g, H, cfg, solver, solve_fn, heldout_fn, frames, name)
            rep["search"].append({"tau_frames": key, "heldout_median_px": evaluated[key][2], "name": name})
            order.append(key)
        return evaluated[key][2]
    sign = 1.0 if float(meas["tau_frames"]) >= 0.0 else -1.0       # the readout's sign from the residuals
    cands = [sign * k * step for k in range(1, n_grid + 1)]
    for k, t in enumerate(cands):
        ev(t, f"RS-grid{k + 1}")
    best_tau = min(evaluated, key=lambda t: (evaluated[t][2], abs(t)))
    for level in range(int(cfg.rs_passes)):
        step = step / 2.0
        for t in (best_tau - sign * step, best_tau + sign * step):
            if sign * t < 0.0 or abs(t) > float(cfg.rs_max_frames):
                continue
            ev(t, f"RS-refine{level + 1}")
        best_tau = min(evaluated, key=lambda t: (evaluated[t][2], abs(t)))
    rs_rung, held_rs, med_rs = evaluated[best_tau]
    rep["tau_frames"] = float(best_tau)
    if best_tau == 0.0:
        rep["reason"] = "no readout on the grid improves the held-out median over the global shutter"
        log(f"[rs] readout searched up to {cfg.rs_max_frames:g} frame(s): the global shutter (held-out median "
            f"{med_gs:.4f} px) is the minimum — not corrected")
        return best, None, rep
    verdict = judge_fn(h_best, held_rs, float(solver_error))
    rep["verdict_vs_best"] = verdict
    Xh2 = triangulate_tracks(held_g, rs_rung.w2c, rs_rung.params_by_block[0], solver, cfg.min_tri_deg)
    m2 = keyframe_motion(rs_rung.w2c, frames)
    held_rect = rectify(held_g, Xh2, rs_rung.w2c, rs_rung.params_by_block[0], H, m2, best_tau)
    Xh3 = triangulate_tracks(held_rect, rs_rung.w2c, rs_rung.params_by_block[0], solver, cfg.min_tri_deg)
    rep["spearman_after"] = spearman(held_rect, Xh3, rs_rung.w2c, rs_rung.params_by_block[0], H, m2)
    take = bool(verdict.get("improves"))
    rep["taken"] = take
    rep["corrected"] = take
    rep["heldout_median_px_rolling_shutter"] = med_rs
    log(f"[rs] readout {best_tau:.3f} frame(s) (searched {len(order)} candidate(s); the residuals read "
        f"{meas['tau_frames']:.3f} ± {meas['sigma']:.3f}): held-out median {med_rs:.4f} px vs {med_gs:.4f} px "
        f"global shutter: {'TAKEN' if take else 'not taken'} ({verdict.get('reason', '')})")
    return (rs_rung, held_rs, rep) if take else (best, None, rep)
