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


def seam_weights_from_reconstruction(output_dir, plan: dict) -> List[float]:
    """How badly each chunk seam glued, from the reconstruction's OWN
    evidence (maplong_run/elastic_seams.json: median disagreement of the two
    copies of the shared frames before any adjustment). One weight per seam
    (len = n_chunks − 1); uniform when the file is absent (declared in the
    report by the caller)."""
    import json
    from pathlib import Path as _P
    n_seams = max(len(plan["chunk_ranges"]) - 1, 0)
    p = _P(output_dir) / "maplong_run" / "elastic_seams.json"
    if not p.exists() or n_seams == 0:
        return [1.0] * n_seams
    data = json.loads(p.read_text())
    seams = data.get("seams") or {}
    w = []
    for i in range(n_seams):
        entries = seams.get(str(i)) or {}
        vals = [float(e["before_m"]) for e in entries.values()
                if isinstance(e, dict) and "before_m" in e]
        w.append(float(np.median(vals)) if vals else 1.0)
    if not any(x > 0 for x in w):
        return [1.0] * n_seams
    return w


def distribute_chunks(n_kf: int, ref_kf_end: int, visit_solutions: List[dict],
                      plan: dict, seam_weights: List[float]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """USER 2026-09-09 (pccr): a loop closure of 0.98 m spread linearly over
    180 keyframes displaced sectors that were CORRECT and duplicated them.
    The drift of a chunked reconstruction is produced at the SEAMS (SE(3)
    gluing), so the closure is applied per CHUNK: identity for the chunks
    that hold the reference visit, the full solved transform for the chunk
    of the displaced visit, and in between each seam absorbs a share of the
    closure proportional to how badly it glued (``seam_weights``). Every
    chunk is a RIGID SE(3) block (a Sim3 when depth k was diagnosed);
    keyframes inside a chunk overlap belong to both chunks and interpolate
    (slerp+lerp) between their two transforms — the step lands smoothly
    inside the overlap instead of at a hard boundary."""
    from scipy.spatial.transform import Rotation, Slerp
    ranges = [(int(a), int(b)) for a, b in plan["chunk_ranges"]]
    n_ch = len(ranges)

    def chunk_of(kf: int) -> int:
        """Primary chunk of a keyframe: the LAST chunk whose range holds it
        (the writer of the frame in an overlap is the later chunk)."""
        c = 0
        for i, (a, b) in enumerate(ranges):
            if a <= kf < b:
                c = i
        return c

    ref_chunk = chunk_of(int(ref_kf_end))
    sols = sorted([s for s in visit_solutions
                   if int(s["anchor_kf"]) > ref_kf_end],
                  key=lambda s: int(s["anchor_kf"]))
    if not sols:
        raise RuntimeError(
            "no anchor keyframe after the reference visit — nothing to "
            "distribute")
    # per-chunk target transform: identity up to ref_chunk, then the
    # closure shared over the seams up to each solved visit's chunk
    chunk_R = [np.eye(3) for _ in range(n_ch)]
    chunk_t = [np.zeros(3) for _ in range(n_ch)]
    prev_chunk, prev_R, prev_t = ref_chunk, np.eye(3), np.zeros(3)
    for sol in sols:
        tgt_chunk = chunk_of(int(sol["anchor_kf"]))
        R_s, t_s = np.asarray(sol["R"]), np.asarray(sol["t"])
        if tgt_chunk <= prev_chunk:
            chunk_R[tgt_chunk], chunk_t[tgt_chunk] = R_s, t_s
            continue
        seams = list(range(prev_chunk, tgt_chunk))     # seam i joins i, i+1
        wts = np.array([max(seam_weights[i], 0.0) for i in seams])
        wts = wts / wts.sum() if wts.sum() > 0 else np.ones(len(seams)) / len(seams)
        cum = np.cumsum(wts)
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([prev_R, R_s])))
        for j, c in enumerate(range(prev_chunk + 1, tgt_chunk + 1)):
            f = float(cum[j])
            chunk_R[c] = slerp([f]).as_matrix()[0]
            chunk_t[c] = (1 - f) * prev_t + f * t_s
        prev_chunk, prev_R, prev_t = tgt_chunk, R_s, t_s
    for c in range(prev_chunk + 1, n_ch):              # constant after last
        chunk_R[c], chunk_t[c] = prev_R, prev_t

    # per-keyframe: single chunk → its transform; overlap → blend
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    for kf in range(n_kf):
        owners = [i for i, (a, b) in enumerate(ranges) if a <= kf < b]
        if not owners:
            owners = [chunk_of(kf)]
        if len(owners) == 1 or kf <= ref_kf_end:
            c = owners[-1] if kf > ref_kf_end else ref_chunk
            R_kf[kf], t_kf[kf] = chunk_R[c], chunk_t[c]
            continue
        a_c, b_c = owners[0], owners[-1]
        lo, hi = ranges[b_c][0], ranges[a_c][1]        # overlap [lo, hi)
        w = (kf - lo + 1) / max(hi - lo + 1, 1)
        sl = Slerp([0.0, 1.0], Rotation.from_matrix(
            np.stack([chunk_R[a_c], chunk_R[b_c]])))
        R_kf[kf] = sl([w]).as_matrix()[0]
        t_kf[kf] = (1 - w) * chunk_t[a_c] + w * chunk_t[b_c]
    for kf in range(0, ref_kf_end + 1):
        R_kf[kf], t_kf[kf] = np.eye(3), np.zeros(3)

    k_kf = np.ones(n_kf)
    for sol in visit_solutions:
        kv = float(sol.get("k", 1.0))
        if kv != 1.0:
            a, b = sol["kf_span"]
            k_kf[int(a):int(b) + 1] = kv

    from correction.solve import rot_deg
    report = {
        "mode": "chunk_graph",
        "identity_until_kf": int(ref_kf_end),
        "anchors": [{"kf": int(s["anchor_kf"]),
                     "rot_deg": round(rot_deg(np.asarray(s["R"])), 3),
                     "t_m": round(float(np.linalg.norm(s["t"])), 4)}
                    for s in sols],
        "chunks": [{"chunk": c, "range": list(ranges[c]),
                    "rot_deg": round(rot_deg(chunk_R[c]), 3),
                    "t_m": round(float(np.linalg.norm(chunk_t[c])), 4)}
                   for c in range(n_ch)],
        "seam_weights": [round(float(x), 4) for x in seam_weights],
        "keyframes_warped": int(n_kf - 1 - ref_kf_end),
        "depth_keyframes": int((k_kf != 1.0).sum()),
    }
    report.update(steps_report(R_kf, t_kf))
    return R_kf, t_kf, k_kf, report


def steps_report(R_kf: np.ndarray, t_kf: np.ndarray) -> dict:
    """Continuity numbers of a final per-keyframe transform set (used after
    composing the floor pre-correction with the object transform)."""
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
