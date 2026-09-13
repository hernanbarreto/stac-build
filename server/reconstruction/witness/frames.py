"""Per-keyframe depth + intrinsics + final c2w pose for the witnesses.

Production: the SAME sources the TSDF and ``mv_consistency`` consume — the
aligned chunk depth (served through the session's depth-correction sidecar)
and the canonical ``camera_poses.txt`` — keyed by REAL frame number.
Synthetic sessions hand the arrays in directly (``frames_from_arrays``); the
witness functions never care where a frame came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np


def load_session_frames(output_dir, log: Optional[Callable[[str], None]] = None) -> Dict[int, dict]:
    """{real_frame: {"depth": (H,W) float32, "K": (3,3), "T": (4,4) c2w}} from
    the session's aligned chunks + camera_poses.txt. Raises when fewer than
    three keyframes carry depth (nothing to cross-check)."""
    from reconstruction.mv_consistency import load_keyframe_data
    return load_keyframe_data(Path(output_dir), log or (lambda m: None))


def frames_from_arrays(depths: np.ndarray, Ks, Ts: np.ndarray, frame_numbers) -> Dict[int, dict]:
    """Synthetic / in-memory frames in the production layout."""
    out: Dict[int, dict] = {}
    for i, n in enumerate(frame_numbers):
        K = Ks[i] if np.ndim(Ks) == 3 else Ks
        out[int(n)] = {"depth": np.asarray(depths[i], np.float32),
                       "K": np.asarray(K, np.float64),
                       "T": np.asarray(Ts[i], np.float64)}
    return out


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H,W,3) camera-frame points of a depth map (invalid depth → NaN)."""
    d = np.asarray(depth, np.float64)
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], axis=-1)
    P[~(np.isfinite(d) & (d > 0))] = np.nan
    return P


def world_points(frame: dict) -> np.ndarray:
    """(H,W,3) world points of one frame (NaN where the depth is invalid)."""
    P = unproject(frame["depth"], frame["K"])
    T = np.asarray(frame["T"], np.float64)
    return P @ T[:3, :3].T + T[:3, 3]


def camera_depth(frame: dict, xyz: np.ndarray) -> np.ndarray:
    """Depth (camera z) of world points in this frame."""
    T = np.asarray(frame["T"], np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return (np.asarray(xyz, np.float64) - t) @ R[:, 2]
