"""Geometric solvers: trimmed yaw-planar ICP, plane fit, depth-ray expansion,
DOF projection.

What it decides: the candidate transform (yaw R, translation t, depth k) of
one displaced visit against the reference evidence. The rotation is a YAW
about the vertical axis only, translation free in 3-D (USER 2026-09-06: a full
3-D rotation solved on two objects tilted chunk 6 and lifted its floor 11 cm).
Depth expansion happens along each point's own shooting ray from its own
camera. Unobservable DOF are removed by ``project_solution`` — the solver
never returns a component the evidence does not constrain.

Everything here is pure/deterministic (seeded RNG passed in); nothing touches
disk.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from correction.config import CorrectionConfig


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def fit_plane(P: np.ndarray, rng: np.random.Generator,
              cfg: CorrectionConfig) -> Tuple[np.ndarray, np.ndarray]:
    """RANSAC + SVD-refit plane (unit normal, point on plane)."""
    tol = cfg.solve.plane_ransac_tol_m
    S = P if len(P) <= cfg.solve.plane_ransac_sample else \
        P[rng.choice(len(P), cfg.solve.plane_ransac_sample, replace=False)]
    best, bn = None, -1
    for _ in range(cfg.solve.plane_ransac_iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        cnt = int((np.abs((S - a) @ n) < tol).sum())
        if cnt > bn:
            bn, best = cnt, (n, a)
    if best is None:
        raise RuntimeError("plane RANSAC found no valid triple — degenerate "
                           "evidence points")
    n, a = best
    inl = np.abs((S - a) @ n) < tol * 2
    c0 = S[inl].mean(0)
    n = np.linalg.svd(S[inl] - c0, full_matrices=False)[2][2]
    return n, c0


def trimmed_icp(src: np.ndarray, tree, target: np.ndarray,
                cfg: CorrectionConfig,
                rotation: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                float]:
    """Trimmed point-to-point ICP src→target; yaw-planar rotation.
    Returns (R, t, trimmed rms).

    rotation=False solves TRANSLATION ONLY. Used when the observability
    analysis restricts the visit's DOF: on a rotationally-symmetric object
    the yaw wanders freely and its compensating translation survives a
    naive post-hoc projection — the constraint must hold DURING the solve.
    """
    R = np.eye(3)
    t = np.zeros(3)
    S = src.copy()
    rms = float("inf")
    workers = cfg.runtime.workers
    for _ in range(cfg.solve.icp_iters):
        d, j = tree.query(S, workers=workers)
        k = max(cfg.solve.icp_min_corr, int(len(S) * cfg.solve.icp_trim))
        sel = np.argsort(d)[:k]
        P, Q = S[sel], target[j[sel]]
        mp, mq = P.mean(0), Q.mean(0)
        if rotation:
            # 2-D Kabsch in the XZ plane → yaw only
            P2 = (P - mp)[:, [0, 2]]
            Q2 = (Q - mq)[:, [0, 2]]
            H2 = P2.T @ Q2
            U2, _s2, Vt2 = np.linalg.svd(H2)
            D2 = np.diag([1, np.sign(np.linalg.det(Vt2.T @ U2.T))])
            R2 = Vt2.T @ D2 @ U2.T
            Ri = np.eye(3)
            Ri[0, 0], Ri[0, 2] = R2[0, 0], R2[0, 1]
            Ri[2, 0], Ri[2, 2] = R2[1, 0], R2[1, 1]
        else:
            Ri = np.eye(3)
        ti = mq - Ri @ mp
        S = S @ Ri.T + ti
        R = Ri @ R
        t = Ri @ t + ti
        rms = float(np.sqrt(
            (np.linalg.norm(S[sel] - Q, axis=1) ** 2).mean()))
        if rot_deg(Ri) < cfg.solve.icp_converge_deg \
                and np.linalg.norm(ti) < cfg.solve.icp_converge_m:
            break
    return R, t, rms


def expand_depth(points: np.ndarray, cam_centers: np.ndarray,
                 k: float) -> np.ndarray:
    """Depth correction along each point's own shooting ray from its own
    camera: p' = c + (p − c)·k."""
    return cam_centers + (points - cam_centers) * k


def project_solution(R: np.ndarray, t: np.ndarray, projection: dict,
                     about: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the DOF the evidence does not observe (observability.py spec).
    Unobservable components go to identity — never a guessed value.

    What gets projected is the MOTION the solution produces, and ``t`` is that
    motion only while ``R`` is the identity — which is the case for every
    caller that runs its ICP with ``rotation=full``: the projection then only
    ever sees a pure translation. Pass ``about`` (a point of the object) when
    ``R`` can carry a real rotation: ``t`` is then the translation component of
    a rotation about a DISTANT origin and is not a displacement at all.

    pccr 2026-09-17, desk#201: a closure of 165.3° that moves its copy 60 cm
    onto its twin has |t| = 10.1 m. Projecting that raw t perpendicular to the
    desk's axis asked for a 14 m translation, the greedy loop measured "its own
    copies 60.7 → 823.5 cm", and with 19 edges like it the pose graph closed 0%
    and the session stayed at epoch 0. Projecting the displacement about the
    object's own centroid asks for 17 cm.
    """
    mode = projection.get("mode")
    if mode == "full":
        return R, t
    if about is not None:
        c = np.asarray(about, dtype=np.float64)
        t = np.asarray(R, np.float64) @ c + np.asarray(t, np.float64) - c
    if mode == "translation":
        return np.eye(3), t
    if mode == "normal":
        n = np.asarray(projection["normal"], dtype=np.float64)
        n = n / np.linalg.norm(n)
        return np.eye(3), float(t @ n) * n
    if mode == "perp_axis":
        a = np.asarray(projection["axis"], dtype=np.float64)
        a = a / np.linalg.norm(a)
        return np.eye(3), t - float(t @ a) * a
    raise RuntimeError(f"unobservable visit reached the solver "
                       f"(projection mode {mode!r}) — the observability gate "
                       f"must reject it first")


def eval_residual(points: np.ndarray, tree, rng: np.random.Generator,
                  cfg: CorrectionConfig) -> float:
    """Median NN distance of (a sample of) points against the reference
    KD-tree."""
    n = min(cfg.solve.eval_sample, len(points))
    sub = points if n == len(points) else \
        points[rng.choice(len(points), n, replace=False)]
    d, _ = tree.query(sub, workers=cfg.runtime.workers)
    return float(np.median(d))
