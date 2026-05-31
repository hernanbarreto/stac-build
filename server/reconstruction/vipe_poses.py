#!/usr/bin/env python3
"""
ViPE pose loading + global metric scaling + camera_poses.txt writer.

ViPE provides continuous c2w poses for ALL frames (its BA). We scale the pose
translations to the DA3 metric (the calibration reference) by a single robust
global factor derived from the per-frame depth-calibration scales, then write
``camera_poses.txt`` (one row-major 4x4 per frame) + ``intrinsic.txt`` in the
exact format the downstream TSDF/texture path expects (``_load_da3_refined_poses``).

Pure-numpy (runs in the da3 env). No ViPE/OpenEXR import.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_vipe_poses(vipe_out: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (inds [N] original-frame#, c2w [N,4,4]). ViPE inds are positional
    in the sorted file list it was given; with an all-frames contiguous run they
    equal the original frame numbers."""
    d = np.load(vipe_out / "pose" / "frames.npz")
    return np.asarray(d["inds"]).astype(np.int64), np.asarray(d["data"], np.float64)


def global_scale_from_file(scale_path: Path) -> float:
    """Read the single global ViPE→DA3-metric pose scale g written by
    vipe_calibrate.py (median DA3/ViPE depth ratio). ViPE pose+depth share ViPE's
    native scale, so g converts pose translations to the DA3 metric. We use DA3
    depth DIRECTLY for the cloud/TSDF — only the pose translations are scaled."""
    try:
        return float(json.load(open(scale_path))["pose_scale"])
    except Exception:
        return 1.0


def build_pose_map(vipe_out: Path, frames: Optional[list] = None,
                   scale: float = 1.0) -> Dict[int, np.ndarray]:
    """frame# -> scaled c2w. ``frames`` (original frame numbers) restricts/orders
    the output; default = all ViPE frames."""
    inds, mats = load_vipe_poses(vipe_out)
    row = {int(fi): i for i, fi in enumerate(inds)}
    keys = frames if frames is not None else sorted(row)
    pm: Dict[int, np.ndarray] = {}
    for f in keys:
        f = int(f)
        if f in row:
            c2w = mats[row[f]].copy()
            c2w[:3, 3] *= scale
            pm[f] = c2w
    return pm


def write_pose_files(pose_map: Dict[int, np.ndarray], intrinsics_map: Dict[int, np.ndarray],
                     out_paths: list, intrinsic_path: Optional[Path] = None) -> None:
    """Write camera_poses.txt (16 floats/line, row-major c2w, frame order) to each
    path in ``out_paths``, and optionally intrinsic.txt (fx fy cx cy per frame).
    A companion ``camera_frames.txt`` lists the frame number per line (since we are
    no longer keyframe-indexed)."""
    frames = sorted(pose_map)
    for p in out_paths:
        p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for fr in frames:
                f.write(" ".join(f"{v:.10f}" for v in pose_map[fr].reshape(-1)) + "\n")
        # frame-number sidecar (all-frames is not keyframe-indexed anymore)
        with open(p.parent / "camera_frames.txt", "w") as f:
            f.write("\n".join(str(fr) for fr in frames) + "\n")
    if intrinsic_path is not None and intrinsics_map:
        intrinsic_path = Path(intrinsic_path)
        with open(intrinsic_path, "w") as f:
            for fr in frames:
                K = intrinsics_map.get(fr)
                if K is None:
                    continue
                f.write(f"{K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")
