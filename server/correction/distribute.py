"""Per-keyframe distribution of the solved corrections.

USER 2026-09-06 (after two failed variants — per-chunk interpolation
multiplied the seam, moving only the displaced chunks duplicated shared
geometry): the revisit's error ACCUMULATED along the trajectory, so the
correction is spread over KEYFRAMES: identity up to the last keyframe of the
reference visit, the solved transform anchored at each displaced visit's
first evidence keyframe, slerp(yaw) + lerp(t) in between, the last anchor's
transform extended to the end. Neighbouring keyframes differ by millimetres —
no seam anywhere; the copies still land exactly on the reference.

Depth k is a STEP function over each displaced visit's keyframe span
(1.0 elsewhere): depth error is a per-frame acquisition error, genuinely
local to the frames that measured it; interpolating k across unrelated
keyframes would distort geometry no evidence touched. The continuity gate
governs the rigid part.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def distribute(n_kf: int, ref_kf_end: int,
               visit_solutions: List[dict]) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, dict]:
    """Build per-keyframe R_kf (n,3,3), t_kf (n,3), k_kf (n,) from the solved
    visits. Each visit_solution: {"anchor_kf", "kf_span": [a,b], "R", "t",
    "k"}. Returns (R_kf, t_kf, k_kf, distribution_report)."""
    anchors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        int(ref_kf_end): (np.eye(3), np.zeros(3))}
    for sol in visit_solutions:
        a_kf = int(sol["anchor_kf"])
        if a_kf <= ref_kf_end:
            continue
        anchors[a_kf] = (np.asarray(sol["R"]), np.asarray(sol["t"]))
    a_kfs = sorted(anchors)
    if len(a_kfs) < 2:
        raise RuntimeError(
            "no anchor keyframe after the reference visit — every solved "
            "visit precedes or overlaps the reference; nothing to distribute")

    rots = Rotation.from_matrix(np.stack([anchors[k][0] for k in a_kfs]))
    slerp = Slerp(a_kfs, rots)
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    for k in range(n_kf):
        if k <= a_kfs[0]:
            continue
        if k >= a_kfs[-1]:
            R_kf[k], t_kf[k] = anchors[a_kfs[-1]]
            continue
        lo = max(a for a in a_kfs if a <= k)
        hi = min(a for a in a_kfs if a > k)
        w = (k - lo) / (hi - lo)
        R_kf[k] = slerp([k]).as_matrix()[0]
        t_kf[k] = (1 - w) * anchors[lo][1] + w * anchors[hi][1]

    k_kf = np.ones(n_kf)
    for sol in visit_solutions:
        kv = float(sol.get("k", 1.0))
        if kv != 1.0:
            a, b = sol["kf_span"]
            k_kf[int(a):int(b) + 1] = kv

    from correction.solve import rot_deg
    steps_t = np.linalg.norm(np.diff(t_kf, axis=0), axis=1)
    steps_r = [rot_deg(R_kf[i + 1] @ R_kf[i].T) for i in range(n_kf - 1)]
    report = {
        "identity_until_kf": int(a_kfs[0]),
        "anchors": [{"kf": int(k),
                     "rot_deg": round(rot_deg(anchors[k][0]), 3),
                     "t_m": round(float(np.linalg.norm(anchors[k][1])), 4)}
                    for k in a_kfs],
        "keyframes_warped": int(n_kf - 1 - a_kfs[0]),
        "max_step_between_keyframes_mm": round(float(steps_t.max()) * 1000, 2)
        if len(steps_t) else 0.0,
        "max_step_between_keyframes_deg": round(float(max(steps_r)), 4)
        if steps_r else 0.0,
        "depth_keyframes": int((k_kf != 1.0).sum()),
    }
    return R_kf, t_kf, k_kf, report


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
