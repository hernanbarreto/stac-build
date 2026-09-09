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
growing along the walk, extrapolated with the last slope beyond the last
knot. The reference copy moves too (by −E(d_ref)) — it is not exact either,
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
    r_full = Rotation.from_matrix(R1).as_rotvec() / (1.0 - f_r)
    R_refc = Rotation.from_rotvec(f_r * r_full).as_matrix()
    tau_full = R_refc @ t1 / (1.0 - f_r)
    C_ref_R, C_ref_t = R_refc, f_r * tau_full
    # knots of the CORRECTION curve C(d): C(0)=identity, then every closure
    # composed with the reference's own correction
    ds = [0.0]
    Ct = [np.zeros(3)]
    Cr = [np.zeros(3)]
    for s in sols:
        Rj = np.asarray(s["R"], dtype=np.float64)
        tj = np.asarray(s["t"], dtype=np.float64)
        ds.append(float(d_kf[int(s["anchor_kf"])]))
        Cr.append(Rotation.from_matrix(C_ref_R @ Rj).as_rotvec())
        Ct.append(C_ref_R @ tj + C_ref_t)
    xs = np.array(ds)
    Ct_a = np.stack(Ct)
    Cr_a = np.stack(Cr)
    span1 = d1 - d_ref

    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    for k in range(n_kf):
        t_kf[k] = _interp_extrap(float(d_kf[k]), xs, Ct_a)
        R_kf[k] = Rotation.from_rotvec(
            _interp_extrap(float(d_kf[k]), xs, Cr_a)).as_matrix()

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
            np.linalg.norm(r_full))) / d1, 4),
        "knots": [{"chainage_m": round(float(x), 3),
                   "correction_m": round(float(np.linalg.norm(c)), 4)}
                  for x, c in zip(xs, Ct_a)],
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
