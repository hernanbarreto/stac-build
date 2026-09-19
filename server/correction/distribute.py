"""Per-keyframe distribution of a loop closure — the DRIFT-RATE model.

USER 2026-09-09 (his formulation): the measured position of every keyframe
carries the ACCUMULATED error of the walk, p_n = p_n(true) + E_n with
E_n = Σ δE_i. With a constant per-metre error, E(d) = ε·d where d is the
distance walked since the start of the scan (E(0) = 0 — the start is exact).
A duplicated object seen at the reference visit and again at a later visit
measures the closure t_j = E(d_ref) − E(d_j); with one duplicate the error
curve is the straight line through (0, 0) and that closure (ε = t_j /
(d_j − d_ref)); with more duplicates it is piecewise-linear through their
knots. The correction at ANY keyframe is −E(d_k): small near the start,
growing along the walk. Beyond the last knot it continues at the rate
measured from the START, which is the sentence E(d) = ε·d itself — the rate
with the longest lever arm behind it, not a local slope between the two
closest closures. The reference copy moves too (by −E(d_ref)) — it is not exact either,
only closer to the start.

Superseded and removed (both smeared the closure over keyframes that were
right): the linear-in-keyframes spread with identity up to the reference,
and the per-chunk seam-weighted blocks.

Rotation (yaw) follows the same rate model on the rotation vector; depth k
stays a STEP over each displaced visit's keyframe span (a per-frame
acquisition error, not accumulated). Declared limitation: a heading drift
bends the accumulated error into an arc — one duplicate observes only the
net translation; a second object pins the curvature.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def chainage(poses: np.ndarray) -> np.ndarray:
    """Distance walked from the start of the scan at each keyframe (m)."""
    centers = np.asarray(poses)[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _V(w: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): the matrix that turns a screw's linear part into
    the translation of the rigid motion, t = V(w) @ rho."""
    th = float(np.linalg.norm(w))
    W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-9:                       # V -> I + W/2 as th -> 0
        return np.eye(3) + 0.5 * W
    return (np.eye(3) + ((1.0 - np.cos(th)) / th ** 2) * W
            + ((th - np.sin(th)) / th ** 3) * (W @ W))


def screw_log(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm: the screw [w(3), rho(3)] with exp(xi) == T.

    NOT the small-motion convention used elsewhere in the repo (``R=exp(w)``
    with the translation taken as-is): that one is exactly the separable model
    this module had, and it is what broke the distribution (pccr 2026-09-17).
    """
    T = np.asarray(T, np.float64)
    w = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    rho = np.linalg.solve(_V(w), T[:3, 3])
    return np.concatenate([w, rho])


def screw_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential: the rigid motion of the screw ``xi = [w, rho]``."""
    xi = np.asarray(xi, np.float64)
    w, rho = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(w).as_matrix()
    T[:3, 3] = _V(w) @ rho
    return T


def _screw_interp(x: float, xs: np.ndarray, Ts: np.ndarray) -> np.ndarray:
    """The correction curve at chainage ``x``, moving between knots along the
    ONE-PARAMETER SUBGROUP that joins them — the screw motion.

    The old curve interpolated the rotation on the manifold and the translation
    LINEARLY, as if the rotation were not there. That is only valid for small
    rotations. pccr 2026-09-17, desk#201: a closure of 148.5 deg that moves its
    copy 60.6 cm onto its twin was spread over the chain as displacements of
    up to 6-7 m in the middle of the walk — measured by the greedy loop as
    "its own copies 60.8 -> 783.2 cm". The two ENDS agreed (at f=1 the model
    reproduces the closure), which is why loops_posthoc predicted 60.6 -> 1.3
    cm for the same edge and nothing looked wrong until the frames were asked.

    BEYOND THE LAST KNOT the curve continues along the screw measured from the
    START — ``E(d) = ε·d``, the model's own sentence — not along the last
    segment's. Past the last closure there is nothing to measure, so the rate
    to continue with is the one with the longest lever arm behind it: a
    segment's rate carries the closure uncertainty divided by that segment's
    length, and the last two closures are usually the closest together. The
    synthetic corridor measured the cost of the other choice: knots at 57.3 m
    and 61.0 m whose 15.8 and 15.2 cm closures point ~35 deg apart gave a
    local rate that, extended over the remaining 3.7 m, moved the walk's end
    33 cm for a 15 cm closure (step between keyframes 2.57 mm -> 53.19 mm).
    With one knot — the common case — the two readings are identical, because
    the only segment IS the whole walk.
    """
    if len(xs) == 1:
        return Ts[0].copy()
    if x <= xs[0]:
        i, span = 0, max(xs[1] - xs[0], 1e-9)
        u = (x - xs[0]) / span
        j = 1
    elif x >= xs[-1]:
        i, j = 0, len(xs) - 1
        span = max(xs[-1] - xs[0], 1e-9)
        u = 1.0 + (x - xs[-1]) / span
    else:
        i = int(np.searchsorted(xs, x, side="right")) - 1
        j = i + 1
        u = (x - xs[i]) / max(xs[j] - xs[i], 1e-9)
    rel = np.linalg.inv(Ts[i]) @ Ts[j]
    return Ts[i] @ screw_exp(u * screw_log(rel))



def _interp_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of vector knots ys over xs; linear
    extrapolation with the last (or first) segment's slope."""
    if x <= xs[0]:
        return ys[0].copy()
    if x >= xs[-1]:
        if len(xs) == 1:
            return ys[-1].copy()
        slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-9)
        return ys[-1] + slope * (x - xs[-1])
    i = int(np.searchsorted(xs, x, side="right")) - 1
    w = (x - xs[i]) / max(xs[i + 1] - xs[i], 1e-9)
    return (1 - w) * ys[i] + w * ys[i + 1]


def distribute(n_kf: int, d_kf: np.ndarray, ref_kf: int,
               visit_solutions: List[dict]) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, dict]:
    """Per-keyframe R_kf (n,3,3), t_kf (n,3), k_kf (n,) from the solved
    visits under the drift-rate model. ``d_kf``: chainage per keyframe;
    ``ref_kf``: representative keyframe of the reference visit. Each
    visit_solution: {"anchor_kf", "kf_span": [a,b], "R", "t", "k"} where
    (R, t) maps the displaced copy onto the reference copy."""
    d_kf = np.asarray(d_kf, dtype=np.float64)
    d_ref = float(d_kf[int(ref_kf)])
    sols = sorted([s for s in visit_solutions
                   if float(d_kf[int(s["anchor_kf"])]) > d_ref],
                  key=lambda s: float(d_kf[int(s["anchor_kf"])]))
    if not sols:
        raise RuntimeError(
            "no displaced visit lies farther along the walk than the "
            "reference visit — nothing to distribute")
    # The closure T_j maps the displaced copy onto the reference copy in the
    # ORIGINAL coordinates. After the correction both copies must coincide:
    #   C(d_j) ∘ x_d = C(d_ref) ∘ x_r  with  x_r = T_j ∘ x_d
    #   ⇒ C(d_j) = C(d_ref) ∘ T_j
    # and the reference copy is itself on the curve, C(d_ref) = C at
    # chainage d_ref. For the separable rate model C(d) = (exp(f·r), f·τ),
    # f = d/d_1, the first closure gives it in closed form:
    #   exp((1−f_r)·r) = R_1  →  r = rotvec(R_1)/(1−f_r)
    #   τ(1−f_r) = R(d_ref)·t_1  →  τ = exp(f_r·r)·t_1/(1−f_r)
    s1 = sols[0]
    d1 = float(d_kf[int(s1["anchor_kf"])])
    f_r = d_ref / d1
    R1 = np.asarray(s1["R"], dtype=np.float64)
    t1 = np.asarray(s1["t"], dtype=np.float64)
    T1 = np.eye(4); T1[:3, :3] = R1; T1[:3, 3] = t1
    # The model is C(d) = exp(f·xi) on SE(3), f = d/d_1. Because exp(a·xi) and
    # exp(b·xi) share the screw they COMMUTE, so the closed form is one line:
    #   exp(f_r·xi) ∘ T_1 = exp(xi)  ⇒  T_1 = exp((1−f_r)·xi)
    #   ⇒  xi = log(T_1) / (1 − f_r)
    # The old derivation did the same split on (rotvec, t) separately, which
    # is the separable model — exact only while the rotation is small.
    xi_full = screw_log(T1) / (1.0 - f_r)
    C_ref = screw_exp(f_r * xi_full)
    C_ref_R, C_ref_t = C_ref[:3, :3], C_ref[:3, 3]
    # knots of the CORRECTION curve C(d): C(0)=identity, then every closure
    # composed with the reference's own correction
    ds = [0.0]
    Cs = [np.eye(4)]
    for s in sols:
        Rj = np.asarray(s["R"], dtype=np.float64)
        tj = np.asarray(s["t"], dtype=np.float64)
        Tj = np.eye(4); Tj[:3, :3] = Rj; Tj[:3, 3] = tj
        ds.append(float(d_kf[int(s["anchor_kf"])]))
        Cs.append(C_ref @ Tj)
    xs = np.array(ds)
    Cs_a = np.stack(Cs)
    span1 = d1 - d_ref
    tau_full = screw_exp(xi_full)[:3, 3]      # the full-walk correction, for the report

    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    for k in range(n_kf):
        Ck = _screw_interp(float(d_kf[k]), xs, Cs_a)
        R_kf[k] = Ck[:3, :3]
        t_kf[k] = Ck[:3, 3]

    k_kf = np.ones(n_kf)
    for sol in visit_solutions:
        kv = float(sol.get("k", 1.0))
        if kv != 1.0:
            a, b = sol["kf_span"]
            k_kf[int(a):int(b) + 1] = kv

    from correction.solve import rot_deg
    rate_t = np.linalg.norm(tau_full) / d1
    report = {
        "mode": "drift_rate_per_metre",
        "identity_until_kf": -1,
        "reference_kf": int(ref_kf),
        "reference_chainage_m": round(d_ref, 3),
        "reference_correction_m": round(float(np.linalg.norm(C_ref_t)), 4),
        "drift_rate_mm_per_m": round(float(rate_t) * 1000, 2),
        "drift_rate_deg_per_m": round(float(np.degrees(
            np.linalg.norm(xi_full[:3]))) / d1, 4),
        "knots": [{"chainage_m": round(float(x), 3),
                   "correction_m": round(float(np.linalg.norm(c)), 4)}
                  for x, c in zip(xs, Cs_a[:, :3, 3])],
        "anchors": [{"kf": int(s["anchor_kf"]),
                     "rot_deg": round(rot_deg(np.asarray(s["R"])), 3),
                     "t_m": round(float(np.linalg.norm(s["t"])), 4)}
                    for s in sols],
        "walk_m": round(float(d_kf[-1]), 3),
        "keyframes_warped": int(n_kf),
        "depth_keyframes": int((k_kf != 1.0).sum()),
    }
    report.update(steps_report(R_kf, t_kf))
    return R_kf, t_kf, k_kf, report


def steps_report(R_kf: np.ndarray, t_kf: np.ndarray) -> dict:
    """Continuity numbers of a per-keyframe transform set."""
    from correction.solve import rot_deg
    n_kf = len(R_kf)
    steps_t = np.linalg.norm(np.diff(t_kf, axis=0), axis=1)
    steps_r = [rot_deg(R_kf[i + 1] @ R_kf[i].T) for i in range(n_kf - 1)]
    return {
        "max_step_between_keyframes_mm":
            round(float(steps_t.max()) * 1000, 2) if len(steps_t) else 0.0,
        "max_step_between_keyframes_deg":
            round(float(max(steps_r)), 4) if steps_r else 0.0,
    }


def warp_subset(xyz: np.ndarray, fg: np.ndarray, ks: np.ndarray,
                cam_center: Dict[int, np.ndarray], idx: np.ndarray,
                R_kf: np.ndarray, t_kf: np.ndarray,
                k_kf: np.ndarray) -> np.ndarray:
    """The per-keyframe correction applied to a SUBSET of cloud points
    (returns new coordinates; input untouched). Depth k first, along each
    point's own camera ray, then the keyframe's rigid transform. Points with
    unresolvable keyframes are the integrity gate's job — here they raise."""
    sub = xyz[idx].copy()
    sk = ks[idx]
    if (sk < 0).any():
        raise RuntimeError(f"{int((sk < 0).sum())} points with unresolvable "
                           f"keyframe reached warp_subset — the integrity "
                           f"gate must veto first")
    if (k_kf != 1.0).any():
        kv = k_kf[sk]
        m = kv != 1.0
        if m.any():
            cams = np.stack([cam_center[int(f)] for f in fg[idx][m]])
            sub[m] = cams + (sub[m] - cams) * kv[m][:, None]
    Rm = R_kf[sk]
    sub = np.einsum('nij,nj->ni', Rm, sub) + t_kf[sk]
    return sub
