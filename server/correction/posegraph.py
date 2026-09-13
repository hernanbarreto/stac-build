"""Per-chunk pose graph — the revisit closures land on the chunks that
produced them (USER 2026-09-09: "el problema va ocurriendo en chunks
elevados en número"; the drift is born at the seams of the chunked
reconstruction, so the correction is solved per CHUNK, rigid inside a chunk
and blended across the overlap the reconstruction itself blended).

Graph: one node per chunk of ``chunk_plan.json`` with a yaw + translation
correction X_c; node 0 is fixed to identity (the start of the walk is exact
— the same premise as the drift-rate model). Edges:
  * SEAM prior between consecutive chunks: X_{c+1} ≈ X_c, weighted by
    1 / (distance walked between the two chunk centres) — a long walk may
    drift more than a short one. With a single closure this reproduces the
    drift-rate line sampled at the chunk centres; with several closures the
    graph solves them jointly.
  * LOOP edge per accepted joint closure T (later visit → earlier visit):
    after the correction both visits coincide, X_late · x = X_early · T · x
    for every point, i.e. X_late = X_early ∘ T.
Solved by non-linear least squares on (θ_c, t_c) (scipy, tiny problem).

Per keyframe: the chunk's X_c where the keyframe belongs to one chunk; a
linear blend (yaw angle + translation) across the overlap where it belongs
to two (the same place the reconstruction's blend_copies stitched). A
session without a chunk plan has no seams: the closure is then handed to the
drift-rate distribution (distribute.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from correction.config import CorrectionConfig
from correction.distribute import chainage, steps_report


def _yaw_R(theta: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, theta, 0.0]).as_matrix()


def yaw_of(R: np.ndarray) -> float:
    """Yaw angle (about +y) of a yaw-only rotation matrix."""
    return float(np.arctan2(R[0, 2], R[0, 0]))


def chunk_of_visit(plan: dict, kfs: List[int]) -> int:
    """The chunk that owns most of a visit's keyframes (ties → the later
    chunk, whose exclusive part the visit is closer to)."""
    ranges = plan["chunk_ranges"]
    counts = [sum(1 for k in kfs if a <= k < b) for a, b in ranges]
    best = max(counts)
    if best == 0:
        raise RuntimeError(f"visit kf {kfs[0]}..{kfs[-1]} lies in no chunk of "
                           f"the plan {ranges}")
    return max(i for i, c in enumerate(counts) if c == best)


def solve_chunk_graph(plan: dict, d_kf: np.ndarray, loops: List[dict],
                      cfg: CorrectionConfig) -> Tuple[np.ndarray, np.ndarray,
                                                      dict]:
    """Per-chunk (theta_c, t_c). ``loops``: [{"early_chunk", "late_chunk",
    "R", "t"}] with (R, t) mapping the later visit's points onto the
    earlier visit's. Returns (theta (C,), t (C,3), report)."""
    ranges = plan["chunk_ranges"]
    C = len(ranges)
    if C < 2:
        raise RuntimeError("the chunk plan holds a single chunk — no seams "
                           "to distribute a closure over")
    centres = [int(np.median(np.arange(a, b))) for a, b in ranges]
    d_c = np.array([float(d_kf[min(c, len(d_kf) - 1)]) for c in centres])
    seam_len = np.maximum(np.diff(d_c), 1e-6)
    seam_w = 1.0 / seam_len
    seam_w /= seam_w.mean()
    lever = cfg.posegraph.rot_lever_m
    lw = cfg.posegraph.loop_weight
    L = [(int(l["early_chunk"]), int(l["late_chunk"]),
          yaw_of(np.asarray(l["R"], dtype=np.float64)),
          np.asarray(l["t"], dtype=np.float64)) for l in loops]

    def unpack(x):
        th = np.concatenate([[0.0], x[:C - 1]])
        tt = np.vstack([np.zeros(3), x[C - 1:].reshape(C - 1, 3)])
        return th, tt

    def resid(x):
        th, tt = unpack(x)
        r = []
        for c in range(C - 1):
            w = np.sqrt(seam_w[c])
            r.append(w * lever * (th[c + 1] - th[c]))
            r.extend(w * (tt[c + 1] - tt[c]))
        for e, l_, thT, tT in L:
            w = np.sqrt(lw)
            r.append(w * lever * (th[l_] - th[e] - thT))
            r.extend(w * (tt[l_] - (_yaw_R(th[e]) @ tT + tt[e])))
        return np.asarray(r)

    x0 = np.zeros(C - 1 + 3 * (C - 1))
    res = least_squares(resid, x0, method="lm" if len(x0) <= len(resid(x0))
                        else "trf")
    th, tt = unpack(res.x)
    loop_res = []
    for e, l_, thT, tT in L:
        loop_res.append({
            "early_chunk": e, "late_chunk": l_,
            "residual_deg": round(float(np.degrees(th[l_] - th[e] - thT)), 4),
            "residual_mm": round(float(np.linalg.norm(
                tt[l_] - (_yaw_R(th[e]) @ tT + tt[e]))) * 1000, 2)})
    report = {
        "mode": "chunk_pose_graph",
        "n_chunks": C,
        "chunk_centres_kf": centres,
        "chunk_centre_chainage_m": [round(float(x), 3) for x in d_c],
        "nodes": [{"chunk": c, "yaw_deg": round(float(np.degrees(th[c])), 4),
                   "t_m": [round(float(v), 4) for v in tt[c]],
                   "t_norm_m": round(float(np.linalg.norm(tt[c])), 4)}
                  for c in range(C)],
        "loops": loop_res,
        "cost": round(float(res.cost), 6),
    }
    return th, tt, report


def per_keyframe(plan: dict, n_kf: int, theta: np.ndarray,
                 t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid inside a chunk's exclusive part, linear blend across an
    overlap shared by two chunks."""
    ranges = plan["chunk_ranges"]
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    for k in range(n_kf):
        owners = [i for i, (a, b) in enumerate(ranges) if a <= k < b]
        if not owners:
            # beyond the plan (should not happen): the last chunk's transform
            owners = [len(ranges) - 1]
        if len(owners) == 1:
            c = owners[0]
            R_kf[k], t_kf[k] = _yaw_R(float(theta[c])), t[c]
            continue
        if len(owners) > 2:
            raise RuntimeError(f"keyframe {k} belongs to {len(owners)} chunks "
                               f"— the plan's overlap exceeds half a chunk")
        c0, c1 = sorted(owners)
        a1 = ranges[c1][0]
        b0 = ranges[c0][1]
        w = (k - a1 + 0.5) / max(b0 - a1, 1)
        th = (1 - w) * theta[c0] + w * theta[c1]
        R_kf[k] = _yaw_R(float(th))
        t_kf[k] = (1 - w) * t[c0] + w * t[c1]
    return R_kf, t_kf


def distribute_closures(plan: Optional[dict], poses: np.ndarray,
                        closures: List[dict], cfg: CorrectionConfig,
                        log=print) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Per-keyframe (R_kf, t_kf) from accepted joint closures. Each closure:
    {"earlier_kfs": [a, b], "later_kfs": [a, b], "R", "t"} (later → earlier)."""
    n_kf = len(poses)
    d_kf = chainage(poses)
    if plan is None:
        from correction.distribute import distribute
        log("  no chunk plan (single pass): drift-rate distribution")
        sols = [{"anchor_kf": int(np.median(range(c["later_kfs"][0],
                                                   c["later_kfs"][1] + 1))),
                 "kf_span": c["later_kfs"], "R": c["R"], "t": c["t"], "k": 1.0}
                for c in closures]
        ref_kf = int(np.median(range(closures[0]["earlier_kfs"][0],
                                     closures[0]["earlier_kfs"][1] + 1)))
        R_kf, t_kf, _k, rep = distribute(n_kf, d_kf, ref_kf, sols)
        return R_kf, t_kf, rep
    loops = []
    for c in closures:
        e = chunk_of_visit(plan, list(range(c["earlier_kfs"][0], c["earlier_kfs"][1] + 1)))
        l_ = chunk_of_visit(plan, list(range(c["later_kfs"][0], c["later_kfs"][1] + 1)))
        if e == l_:
            raise RuntimeError(
                f"closure kf {c['earlier_kfs']} ↔ {c['later_kfs']} lies inside "
                f"chunk {e} — an intra-chunk duplicate is not a seam problem")
        loops.append({"early_chunk": e, "late_chunk": l_,
                      "R": c["R"], "t": c["t"]})
    theta, t, rep = solve_chunk_graph(plan, d_kf, loops, cfg)
    R_kf, t_kf = per_keyframe(plan, n_kf, theta, t)
    rep.update(steps_report(R_kf, t_kf))
    rep["walk_m"] = round(float(d_kf[-1]), 3)
    rep["keyframes_warped"] = int(n_kf)
    rep["depth_keyframes"] = 0
    rep["identity_until_kf"] = -1
    for n in rep["nodes"]:
        log(f"  chunk {n['chunk']}: yaw {n['yaw_deg']}°, |t| {n['t_norm_m']} m")
    return R_kf, t_kf, rep
