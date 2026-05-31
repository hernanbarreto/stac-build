#!/usr/bin/env python3
"""
da3-env entry: build ViPE pose_map (scaled to DA3 metric) + compose chunk PLYs
with traceability + write camera_poses.txt / intrinsic.txt. Spawned by map_worker
after ViPE + the global pose-scale step have produced vipe_run/ + da3_depth/.

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
    da3_dir = out / "da3_depth"

    # single global metric scale for ViPE pose translations (median DA3/ViPE ratio)
    scale = vipe_poses.global_scale_from_file(vipe_out / "pose_scale.json")
    print(f"[vipe-compose] global pose scale (ViPE->DA3 metric) = {scale:.4f}", flush=True)

    # frame numbers that have a DA3 depth map (used directly, no calibration)
    da3_frames = sorted(int(p.stem) for p in da3_dir.glob("[0-9]*.npz"))
    pose_map = vipe_poses.build_pose_map(vipe_out, frames=da3_frames, scale=scale)
    print(f"[vipe-compose] pose_map: {len(pose_map)} frames", flush=True)

    # per-frame intrinsics for the pose files (from the DA3 depth npz K)
    intr = {}
    for fr in pose_map:
        z = np.load(da3_dir / f"{fr:06d}.npz")
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
        output_dir=out, frames_dir=args.frames_dir, da3_dir=da3_dir,
        pose_map=pose_map, progress_cb=cb,
    )
    print(f"[vipe-compose] DONE: {n} chunks written", flush=True)


if __name__ == "__main__":
    main()
