"""§6.1 ``mv_votes`` on the cloud: ``mv_consistency.compute_frame_consistency``
with the FINAL poses, dumped PER POINT through the provenance (frame_global,
pixel_row, pixel_col). A point's vote count is the number of neighbouring
keyframes (nearest camera centres) that agree with its own frame's depth at
its pixel within ``tau_rel``. Frames with no depth (or no neighbour with
depth) leave their points UNOBSERVED — a count of zero would be a claim.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def mv_votes_per_frame(frames: Dict[int, dict], n_neighbors: int, tau_rel: float,
                       device=None) -> Dict[int, dict]:
    """{real_frame: {"votes": uint8 (H,W), "valid": bool (H,W), "neighbors": [...]}}.
    GPU when available (the module's own choice), one frame + its
    neighbours resident at a time."""
    from reconstruction.mv_consistency import compute_frame_consistency, _neighbor_indices
    nums = sorted(frames)
    if len(nums) < 2:
        raise RuntimeError("mv_votes: at least two keyframes with depth are needed")
    centres = np.stack([np.asarray(frames[n]["T"], np.float64)[:3, 3] for n in nums])
    nbrs = _neighbor_indices(centres, int(n_neighbors))
    out: Dict[int, dict] = {}
    for i, n in enumerate(nums):
        f = frames[n]
        js = [nums[j] for j in nbrs[i]]
        votes, _est, valid = compute_frame_consistency(
            f["depth"], f["K"], f["T"],
            [frames[j]["depth"] for j in js], [frames[j]["K"] for j in js],
            [frames[j]["T"] for j in js], float(tau_rel), device=device)
        out[int(n)] = {"votes": np.asarray(votes, np.uint8), "valid": np.asarray(valid, bool),
                       "neighbors": [int(j) for j in js]}
    return out


def point_mv_votes(per_frame: Dict[int, dict], fg: np.ndarray, pr: np.ndarray,
                   pc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point (mv_votes uint8, observed bool) through the provenance. A
    point whose frame has no witness data, or whose pixel falls outside the
    frame's depth grid, is not observed."""
    fg = np.asarray(fg, np.int64); pr = np.asarray(pr, np.int64); pc = np.asarray(pc, np.int64)
    n = len(fg)
    votes = np.zeros(n, np.uint8)
    observed = np.zeros(n, bool)
    order = np.argsort(fg, kind="stable")
    sfg = fg[order]
    bounds = np.flatnonzero(np.diff(sfg)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    for s, e in zip(starts, ends):
        frame = int(sfg[s])
        rec = per_frame.get(frame)
        if rec is None:
            continue
        idx = order[s:e]
        H, W = rec["votes"].shape
        r, c = pr[idx], pc[idx]
        inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        sel = idx[inb]
        votes[sel] = rec["votes"][r[inb], c[inb]]
        observed[sel] = rec["valid"][r[inb], c[inb]]
    return votes, observed
