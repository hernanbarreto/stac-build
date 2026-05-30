#!/usr/bin/env python3
"""
da3-env entry: build ViPE pose_map (scaled to DA3 metric) + compose chunk PLYs
with traceability + write camera_poses.txt / intrinsic.txt. Spawned by map_worker
after ViPE + calibration have produced vipe_run/ + depth_calibrated/.

    python run_vipe_compose.py --output-dir <out> --frames-dir <frames>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from reconstruction import vipe_poses, vipe_compose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--frames-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.output_dir
    vipe_out = out / "vipe_run"
    cal_dir = vipe_out / "depth_calibrated"

    # global metric scale for ViPE pose translations (robust median of good fits)
    scale = vipe_poses.global_scale_from_calibration(cal_dir)
    print(f"[vipe-compose] global pose scale (ViPE->DA3 metric) = {scale:.4f}", flush=True)

    # frame numbers that have a calibrated depth
    cal_frames = sorted(int(p.stem) for p in cal_dir.glob("[0-9]*.npz"))
    pose_map = vipe_poses.build_pose_map(vipe_out, frames=cal_frames, scale=scale)
    print(f"[vipe-compose] pose_map: {len(pose_map)} frames", flush=True)

    # per-frame intrinsics for the pose files (from the calibrated npz K)
    intr = {}
    for fr in pose_map:
        z = np.load(cal_dir / f"{fr:06d}.npz")
        intr[fr] = np.asarray(z["intrinsics"], np.float64)
    vipe_poses.write_pose_files(
        pose_map, intr,
        out_paths=[out / "camera_poses.txt", out / "da3_run" / "camera_poses.txt"],
        intrinsic_path=out / "intrinsic.txt",
    )
    # text copy that the tolerant parser also accepts
    (out / "camera_poses_mapanything.json").write_text(
        (out / "camera_poses.txt").read_text())
    print("[vipe-compose] wrote camera_poses.txt + intrinsic.txt", flush=True)

    def cb(n, msg):
        print(f"[vipe-compose] {msg}", flush=True)

    n = vipe_compose.compose_chunks_from_vipe(
        output_dir=out, frames_dir=args.frames_dir, cal_dir=cal_dir,
        pose_map=pose_map, progress_cb=cb,
    )
    print(f"[vipe-compose] DONE: {n} chunks written", flush=True)


if __name__ == "__main__":
    main()
