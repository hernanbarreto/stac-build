#!/usr/bin/env python3
"""
Stray Scanner → Direct PLY (no SLAM tracking)

Back-projects LiDAR depth maps directly using ARKit odometry poses.
This validates whether the ARKit poses are geometrically consistent,
bypassing GauS-SLAM's tracking/submap alignment entirely.

Output: direct_cloud.ply (colored, metric)

Usage:
  python stray_direct_ply.py \
    --da3_dir /path/to/da3_full \
    --output /path/to/direct_cloud.ply \
    --conf_threshold 0.5
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("Stray direct PLY generator (no SLAM)")
    parser.add_argument("--da3_dir", required=True, help="da3_full/ directory")
    parser.add_argument("--output", required=True, help="Output PLY path")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Confidence threshold 0-1 (0=low, 0.5=medium+, 1=high only)")
    parser.add_argument("--frame_step", type=int, default=3,
                        help="Use every Nth frame")
    parser.add_argument("--max_depth", type=float, default=5.0,
                        help="Max depth in meters")
    args = parser.parse_args()

    da3_dir = Path(args.da3_dir)

    # Load manifest
    with open(da3_dir / "da3_manifest.json") as f:
        manifest = json.load(f)

    frame_files = manifest["frame_files"]
    depth_shape = manifest["depth_shape"]   # [H, W]
    depth_h, depth_w = depth_shape

    # Load all poses (w2c [N, 3, 4]) and intrinsics ([N, 3, 3])
    extrinsics = np.load(da3_dir / "extrinsics.npy")  # w2c [N, 3, 4]
    intrinsics = np.load(da3_dir / "intrinsics.npy")  # [N, 3, 3]

    K = intrinsics[0]  # same for all frames
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print(f"[Direct PLY] Frames: {len(frame_files)}, Resolution: {depth_w}x{depth_h}")
    print(f"[Direct PLY] Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"[Direct PLY] Frame step: {args.frame_step}, Conf threshold: {args.conf_threshold}")

    # Build pixel grid for backprojection
    u_grid, v_grid = np.meshgrid(np.arange(depth_w), np.arange(depth_h))  # (H, W)

    all_points = []
    all_colors = []
    all_frame_globals = []
    all_pixel_rows = []
    all_pixel_cols = []

    selected_indices = list(range(0, len(frame_files), args.frame_step))
    print(f"[Direct PLY] Processing {len(selected_indices)} frames...")

    for count, idx in enumerate(selected_indices):
        stem = frame_files[idx].replace(".jpg", "").replace(".png", "")

        # Load depth (meters)
        depth_path = da3_dir / f"{stem}_depth.npy"
        if not depth_path.exists():
            continue
        depth = np.load(str(depth_path))  # (H, W) float32 meters

        # Load confidence
        conf_path = da3_dir / f"{stem}_conf.npy"
        if conf_path.exists():
            conf = np.load(str(conf_path))  # (H, W) float32 [0, 1]
            valid_mask = (conf >= args.conf_threshold)
        else:
            valid_mask = np.ones((depth_h, depth_w), dtype=bool)

        # Depth validity
        valid_mask = valid_mask & (depth > 0.1) & (depth < args.max_depth)

        if not np.any(valid_mask):
            continue

        # Back-project to camera frame: p_cam = [(u-cx)/fx * d, (v-cy)/fy * d, d]
        d = depth[valid_mask]
        u = u_grid[valid_mask].astype(np.float32)
        v = v_grid[valid_mask].astype(np.float32)

        X = (u - cx) / fx * d
        Y = (v - cy) / fy * d
        Z = d

        # Traceability: which frame and pixel each point came from
        n_valid = len(d)
        frame_idx_int = int(stem)  # original frame index from filename
        all_frame_globals.append(np.full(n_valid, frame_idx_int, dtype=np.int32))
        all_pixel_rows.append(v_grid[valid_mask].astype(np.int32))
        all_pixel_cols.append(u_grid[valid_mask].astype(np.int32))

        # p_cam: (N, 3)
        p_cam = np.stack([X, Y, Z], axis=1)  # (N, 3)

        # w2c [3, 4] → c2w [4, 4]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = extrinsics[idx].astype(np.float64)
        c2w = np.linalg.inv(w2c)

        # Transform to world
        p_cam_h = np.concatenate([p_cam, np.ones((p_cam.shape[0], 1))], axis=1)  # (N, 4)
        p_world = (c2w @ p_cam_h.T).T[:, :3]  # (N, 3)

        # Load RGB from processed_images for color
        rgb_path = da3_dir / "processed_images" / f"{stem}.png"
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            colors = rgb[valid_mask]  # (N, 3) uint8
        else:
            colors = np.full((p_world.shape[0], 3), 128, dtype=np.uint8)

        all_points.append(p_world)
        all_colors.append(colors)
        # frame_globals, pixel_rows, pixel_cols already appended above

        if (count + 1) % 50 == 0 or count == len(selected_indices) - 1:
            print(f"  [{count+1}/{len(selected_indices)}] {stem}: {p_world.shape[0]} pts")

    if not all_points:
        print("[Direct PLY] No points generated!")
        return

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0)
    frame_globals = np.concatenate(all_frame_globals, axis=0)
    pixel_rows = np.concatenate(all_pixel_rows, axis=0)
    pixel_cols = np.concatenate(all_pixel_cols, axis=0)

    print(f"\n[Direct PLY] Total: {len(points):,} points")

    # Write PLY
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"property int frame_global\n"
        f"property int pixel_row\n"
        f"property int pixel_col\n"
        f"end_header\n"
    )

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        data = np.zeros(len(points), dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
            ("frame_global", np.int32), ("pixel_row", np.int32), ("pixel_col", np.int32),
        ])
        data["x"] = points[:, 0]
        data["y"] = points[:, 1]
        data["z"] = points[:, 2]
        data["red"]   = colors[:, 0]
        data["green"] = colors[:, 1]
        data["blue"]  = colors[:, 2]
        data["frame_global"] = frame_globals
        data["pixel_row"] = pixel_rows
        data["pixel_col"] = pixel_cols
        f.write(data.tobytes())

    print(f"[Direct PLY] ✅ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
