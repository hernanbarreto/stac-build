"""What a keyframe MEASURED, after the owner fusion (audit 2026-10-23, phase 3e).

``visit_drift.Visibility`` builds a keyframe's z-buffer from "the points born in this keyframe"
— its own depth map, internally consistent, never a drifted copy of anything (the reason the
whole cloud is not used). Since the owner fusion a keyframe's published points are only its
OWNER observations; what it measured of the elements other keyframes own was consumed, and lives
in ``rejected_points.npz`` (reason ``fused``, with the fused depth along its own ray since v5).
A z-buffer of the owners alone reads "seen and empty" over every consumed pixel and lets a
duplicate pass as unoccluded. Here the consumed observations come back as world points of their
own keyframe — the full depth map the camera measured, at the epoch the caller works in (the
correction's per-keyframe depth warp z' = k z + b and rigid motion applied through the poses
handed in, exactly as ``correction.apply.warp_full_cloud`` moves the cloud's points).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

REASON_FUSED = 10


def consumed_observations(output_dir: Path, kf_of_frame: Dict[int, int]
                          ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(keyframe index, row, col, depth) of every consumed observation with a depth, or None
    when the session holds no fusion record of version >= 5."""
    p = Path(output_dir) / "rejected_points.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        if "depth" not in z.files or "reason" not in z.files:
            return None
        m = (np.asarray(z["reason"]) == REASON_FUSED) & (np.asarray(z["depth"]) > 0)
        fg = np.asarray(z["frame_global"], np.int64)[m]
        r = np.asarray(z["pixel_row"], np.int64)[m]; c = np.asarray(z["pixel_col"], np.int64)[m]
        d = np.asarray(z["depth"], np.float64)[m]
    ks = np.array([kf_of_frame.get(int(f), -1) for f in fg], np.int64)
    ok = ks >= 0
    return ks[ok], r[ok], c[ok], d[ok]


def world_points(ks: np.ndarray, row: np.ndarray, col: np.ndarray, depth: np.ndarray, poses_c2w: np.ndarray,
                 K: np.ndarray, k_kf: Optional[np.ndarray] = None, b_kf: Optional[np.ndarray] = None) -> np.ndarray:
    """The observations as world points of their keyframe: depth (warped z' = k z + b when the
    caller's epoch carries one) along the ray of the bend-grid pixel (the undistorted native K),
    through the keyframe's c2w handed in."""
    K = np.asarray(K, np.float64)
    z = np.asarray(depth, np.float64)
    if k_kf is not None:
        z = np.asarray(k_kf, np.float64)[ks] * z + (np.asarray(b_kf, np.float64)[ks] if b_kf is not None else 0.0)
    x = (np.asarray(col, np.float64) - K[0, 2]) / K[0, 0] * z
    y = (np.asarray(row, np.float64) - K[1, 2]) / K[1, 1] * z
    P = np.stack([x, y, z], 1)
    R = np.asarray(poses_c2w, np.float64)[ks, :3, :3]
    t = np.asarray(poses_c2w, np.float64)[ks, :3, 3]
    return np.einsum("nij,nj->ni", R, P) + t
