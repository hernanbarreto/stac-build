"""Accumulated drift as a smooth function of the distance walked (§4.3).

USER's formulation, 2026-09-09: the measured position of every keyframe carries
the accumulated error of the distance WALKED since the start — E(0) = 0, the
start is exact, and the error grows along the walk. A duplicate seen twice pins
that curve; more duplicates pin more of it.

Why this is a MODEL and not a σ. The pose graph expresses "the chain may be
wrong" with one knob: the odometry σ. Turn it down and the chain will not bend
at all — pccr 2026-09-14 applied 43 cm where the desks measured 112–215 cm, the
duplicates the user could see stayed put and the closure came out WORSE than
before (0.946 → 1.228). Turn it up and every link becomes independently free:
the closure does spread, but the per-keyframe structural constraints are then
satisfied LOCALLY — each keyframe shifts a little so its points land on the
wall plane — and the global bend survives. Measured on the synthetic corridor:
the walls took the lateral bend 4.15 → 1.14 cm with a stiff chain and only
4.31 → 3.01 cm with a loose one.

Both are right about something, because drift is not white noise: it is stiff
locally and free over the long run. One σ per link cannot say that. A few
coefficients over the chainage can.

The vendor's own loop optimiser (``loop_utils/sim3loop.py``) gets the same
shape for free by stacking the sequential and the loop residuals with NO
weights at all: with every link equally stiff, minimising Σ|Δ_k|² under a total
closure δ puts exactly δ/n on each link — a linear ramp, E(d) = ε·d. This
module states that ramp explicitly instead of hoping a weighting reproduces it,
and can carry a curvature term the uniform spread cannot.

    ξ(u) = Σ_m c_m · u^m,  u = walked(k)/walked(end) ∈ [0, 1],  m = 1..degree

Six unknowns per term (an se(3) vector) — 12 for the default linear+quadratic
against a keyframe graph's 6·n. u^m vanishes at u = 0, so the start stays exact
by construction, and the correction of keyframe k is exp(ξ(u_k)) in the world
frame, the same convention the graph's own corrections use.

The fit consumes ONLY the loop edges: the smooth basis IS the motion prior, so
there is no odometry term to fight it. What the model cannot explain is left to
the graph, which still runs afterwards with its stiff chain and its structural
constraints.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np


def chainage(centres: np.ndarray) -> np.ndarray:
    """Distance walked up to each keyframe (0 at the first)."""
    step = np.linalg.norm(np.diff(np.asarray(centres, np.float64), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def _basis(u: np.ndarray, degree: int) -> np.ndarray:
    """(n, degree) with column m = u^(m+1): every term vanishes at the start."""
    return np.stack([u ** (m + 1) for m in range(int(degree))], axis=1)


def _edge_weights(e: dict, sigma_floor_m: float, sigma_floor_deg: float):
    """(W_rot, W_t) 3×3 weight matrices of one loop edge — the anisotropic
    information when the edge carries it (a floor observes its normal and
    nothing else), else isotropic from its σ."""
    it, ir = e.get("info_t"), e.get("info_rot")
    if it is not None and ir is not None:
        return np.asarray(ir, np.float64), np.asarray(it, np.float64)
    st = max(float(e.get("sigma_m", sigma_floor_m)), sigma_floor_m)
    sr = np.radians(max(float(e.get("sigma_deg", sigma_floor_deg)), sigma_floor_deg))
    return np.eye(3) / sr ** 2, np.eye(3) / st ** 2


def fit_drift(poses: np.ndarray, loops: Sequence[dict], degree: int,
              iters: int, sigma_floor_m: float, sigma_floor_deg: float,
              max_step: float, log: Callable[[str], None] = print) -> Optional[dict]:
    """Least-squares fit of ξ(u) to the loop closures.

    ``poses``: (n, 4, 4) keyframe poses. ``loops``: edges {i, j, Z, sigma_m,
    sigma_deg, info_t?, info_rot?} where Z is the MEASURED relative pose
    T_i⁻¹·T_j. Returns {"coeffs", "u", "corrections", "residual_before_m",
    "residual_after_m", "rate_m_per_m"} or None when nothing can be fitted.
    """
    from loop_utils.lie import se3_exp, se3_inv, se3_log

    n = len(poses)
    edges = [e for e in loops
             if 0 <= int(e["i"]) < n and 0 <= int(e["j"]) < n and int(e["i"]) != int(e["j"])]
    if n < 3 or not edges:
        return None
    d = chainage(np.asarray(poses, np.float64)[:, :3, 3])
    D = float(d[-1])
    if D <= 1e-6:
        return None
    u = d / D

    # The degree is capped by the EVIDENCE, not by the config. USER 2026-09-09:
    # "one duplicate observes the net translation/yaw; heading curvature needs a
    # second". Each loop contributes one independent observation of the curve;
    # asking for a quadratic with a single closure leaves 12 unknowns against 6
    # equations and the fit invents the curvature of the middle — on the
    # synthetic corridor that took the lateral bend from 4.15 cm to 7.97 cm
    # before the walls were even consulted. Lowest order that the loops can
    # actually pin, and the report says which.
    n_span = len({(min(int(e["i"]), int(e["j"])), max(int(e["i"]), int(e["j"])))
                  for e in edges})
    M = max(1, min(int(degree), n_span))
    B = _basis(u, M)                           # (n, M)
    W = [_edge_weights(e, sigma_floor_m, sigma_floor_deg) for e in edges]

    def corrections(c: np.ndarray) -> np.ndarray:
        """exp(ξ(u_k)) for every keyframe, world frame."""
        xi = B @ c.reshape(M, 6)               # (n, 6)
        return np.stack([se3_exp(x) for x in xi])

    def residuals(c: np.ndarray) -> np.ndarray:
        X = corrections(c)
        out = np.empty(6 * len(edges))
        for k, e in enumerate(edges):
            i, j = int(e["i"]), int(e["j"])
            Zc = se3_inv(X[i] @ poses[i]) @ (X[j] @ poses[j])
            r = se3_log(se3_inv(np.asarray(e["Z"], np.float64)) @ Zc)
            Wr, Wt = W[k]
            # se3_log returns (rot, trans); whiten each half with its own
            # information so an unobserved direction cannot pull the fit
            out[6 * k:6 * k + 3] = _chol(Wr) @ r[:3]
            out[6 * k + 3:6 * k + 6] = _chol(Wt) @ r[3:]
        return out

    c = np.zeros(6 * M)
    r0 = residuals(c)
    cost0 = float(r0 @ r0)
    cost = cost0
    for _ in range(int(iters)):
        # numerical Jacobian: 6·degree unknowns (12 by default), a handful of
        # edges — the cost is nothing against one graph solve
        J = np.empty((len(r0), len(c)))
        h = 1e-6
        base = residuals(c)
        for p in range(len(c)):
            cp = c.copy(); cp[p] += h
            J[:, p] = (residuals(cp) - base) / h
        try:
            step = np.linalg.lstsq(J, -base, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        nrm = float(np.linalg.norm(step))
        if nrm > max_step:
            step *= max_step / nrm
        c_new = c + step
        r_new = residuals(c_new)
        cost_new = float(r_new @ r_new)
        if not np.isfinite(cost_new) or cost_new >= cost:
            break
        c, cost = c_new, cost_new
        if nrm < 1e-9:
            break

    X = corrections(c)

    def _edge_offset(Xa) -> float:
        tot = 0.0
        for e in edges:
            i, j = int(e["i"]), int(e["j"])
            Zc = se3_inv(Xa[i] @ poses[i]) @ (Xa[j] @ poses[j])
            tot += float(np.linalg.norm(
                (se3_inv(np.asarray(e["Z"], np.float64)) @ Zc)[:3, 3]))
        return tot

    eye = np.repeat(np.eye(4)[None], n, axis=0)
    before, after = _edge_offset(eye), _edge_offset(X)
    moved = np.linalg.norm(X[:, :3, 3], axis=1)
    rate = float(moved[-1] / D) if D > 0 else 0.0
    log(f"[drift] {M}-term model (degree {degree} asked, {n_span} independent "
        f"loop span(s)) over {D:.1f} m / {n} keyframes, {len(edges)} loop(s): "
        f"loop offset {before * 100:.0f} → {after * 100:.0f} cm, "
        f"keyframes moved 0 → {moved.max() * 100:.0f} cm "
        f"({rate * 100:.2f} cm/m at the end)")
    return {"coeffs": c.reshape(M, 6).tolist(), "u": u.tolist(), "corrections": X,
            "residual_before_m": before, "residual_after_m": after,
            "rate_m_per_m": rate, "degree": M, "degree_requested": int(degree),
            "n_independent_spans": n_span, "walk_m": D, "n_loops": len(edges)}


def _chol(W: np.ndarray) -> np.ndarray:
    """Whitening factor L with LᵀL = W (W symmetric positive semi-definite)."""
    try:
        return np.linalg.cholesky(W + 1e-12 * np.eye(3)).T
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(W)
        return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def apply_drift(poses: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Corrected poses X_k · T_k (world-frame correction, the graph's own
    convention)."""
    return np.einsum("nij,njk->nik", X, np.asarray(poses, np.float64))
