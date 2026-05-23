#!/usr/bin/env python3
"""
Stray Scanner LiDAR → Direct PLY with traceability

Back-projects native LiDAR depth maps (192x256 uint16 mm) using ARKit poses.
Includes frame_global, pixel_row, pixel_col for SAM3/ShapeR integration.

Usage:
  python stray_lidar_ply.py \
    --stray_dir /path/to/38153efc83 \
    --frames_dir /path/to/frames \
    --output /path/to/cleaned_cloud.ply \
    --conf_min 2 --frame_step 1 --max_depth 5.0
"""

import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_odometry(csv_path):
    """Load ARKit poses from odometry.csv → dict of frame_idx → c2w [4,4]"""
    poses = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            vals = [v.strip() for v in row]
            frame_idx = int(vals[1])
            x, y, z = float(vals[2]), float(vals[3]), float(vals[4])
            qx, qy, qz, qw = float(vals[5]), float(vals[6]), float(vals[7]), float(vals[8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses[frame_idx] = T
    return poses


def main():
    parser = argparse.ArgumentParser("Stray LiDAR → PLY with traceability")
    parser.add_argument("--stray_dir", required=True,
                        help="Stray Scanner dir (has depth/, confidence/, odometry.csv)")
    parser.add_argument("--frames_dir", required=True,
                        help="RGB frames dir (to get frame resolution for pixel mapping)")
    parser.add_argument("--output", required=True, help="Output PLY path")
    parser.add_argument("--conf_min", type=int, default=2,
                        help="Min ARKit confidence (0=low, 1=medium, 2=high)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Use every Nth frame")
    parser.add_argument("--max_depth", type=float, default=5.0,
                        help="Max depth in meters")
    args = parser.parse_args()

    stray_dir = Path(args.stray_dir)
    frames_dir = Path(args.frames_dir)
    depth_dir = stray_dir / "depth"
    conf_dir = stray_dir / "confidence"

    # Load camera intrinsics (RGB resolution)
    K_rgb = np.genfromtxt(stray_dir / "camera_matrix.csv", delimiter=",")
    fx_rgb, fy_rgb = K_rgb[0, 0], K_rgb[1, 1]
    cx_rgb, cy_rgb = K_rgb[0, 2], K_rgb[1, 2]

    # Detect RGB resolution from first frame
    first_frame = sorted(frames_dir.glob("*.jpg"))[0]
    rgb_sample = cv2.imread(str(first_frame))
    rgb_h, rgb_w = rgb_sample.shape[:2]
    print(f"[LiDAR PLY] RGB resolution: {rgb_w}x{rgb_h}")
    print(f"[LiDAR PLY] RGB intrinsics: fx={fx_rgb:.2f} fy={fy_rgb:.2f} cx={cx_rgb:.2f} cy={cy_rgb:.2f}")

    # Detect depth resolution from first depth map
    first_depth = cv2.imread(str(depth_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
    depth_h, depth_w = first_depth.shape[:2]
    print(f"[LiDAR PLY] Depth resolution: {depth_w}x{depth_h}")

    # Scale intrinsics to depth resolution
    sx = depth_w / rgb_w
    sy = depth_h / rgb_h
    fx_d = fx_rgb * sx
    fy_d = fy_rgb * sy
    cx_d = cx_rgb * sx
    cy_d = cy_rgb * sy
    print(f"[LiDAR PLY] Depth intrinsics: fx={fx_d:.2f} fy={fy_d:.2f} cx={cx_d:.2f} cy={cy_d:.2f}")

    # Scale factors for mapping depth pixels → RGB pixels (for traceability)
    px_scale_x = rgb_w / depth_w  # 1920/256 = 7.5
    px_scale_y = rgb_h / depth_h  # 1440/192 = 7.5
    print(f"[LiDAR PLY] Pixel scale: {px_scale_x:.2f}x (depth→RGB)")

    # Load ARKit poses
    poses = load_odometry(stray_dir / "odometry.csv")
    print(f"[LiDAR PLY] Loaded {len(poses)} ARKit poses")

    # Build pixel grid for backprojection (depth resolution)
    u_grid, v_grid = np.meshgrid(np.arange(depth_w), np.arange(depth_h))

    # Find available depth frames
    depth_files = sorted(depth_dir.glob("*.png"))
    frame_indices = []
    for df in depth_files:
        idx = int(df.stem)
        if idx in poses:
            frame_indices.append(idx)

    # Apply frame step
    frame_indices = frame_indices[::args.frame_step]
    print(f"[LiDAR PLY] Processing {len(frame_indices)} frames (step={args.frame_step})")
    print(f"[LiDAR PLY] Conf threshold: >={args.conf_min}, Max depth: {args.max_depth}m")

    all_points = []
    all_colors = []
    all_frame_globals = []
    all_pixel_rows = []
    all_pixel_cols = []

    for count, idx in enumerate(frame_indices):
        stem = f"{idx:06d}"

        # Load depth (uint16, millimeters)
        depth_path = depth_dir / f"{stem}.png"
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) / 1000.0  # mm → meters

        # Load confidence
        conf_path = conf_dir / f"{stem}.png"
        if conf_path.exists():
            conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
            valid_mask = (conf >= args.conf_min)
        else:
            valid_mask = np.ones((depth_h, depth_w), dtype=bool)

        # Depth validity
        valid_mask = valid_mask & (depth_m > 0.1) & (depth_m < args.max_depth)

        if not np.any(valid_mask):
            continue

        # Back-project to camera frame
        d = depth_m[valid_mask]
        u = u_grid[valid_mask].astype(np.float32)
        v = v_grid[valid_mask].astype(np.float32)

        X = (u - cx_d) / fx_d * d
        Y = (v - cy_d) / fy_d * d
        Z = d

        # Traceability: map depth pixels → RGB pixel coords
        n_valid = len(d)
        # frame_global = original frame index (matches frames/ directory)
        all_frame_globals.append(np.full(n_valid, idx, dtype=np.int32))
        # pixel coords in RGB resolution (for SAM3 mask matching)
        all_pixel_rows.append((v_grid[valid_mask] * px_scale_y).astype(np.int32))
        all_pixel_cols.append((u_grid[valid_mask] * px_scale_x).astype(np.int32))

        # Camera → World
        p_cam = np.stack([X, Y, Z], axis=1)
        c2w = poses[idx]
        p_cam_h = np.concatenate([p_cam, np.ones((n_valid, 1))], axis=1)
        p_world = (c2w @ p_cam_h.T).T[:, :3]

        # Load RGB color from frames
        rgb_path = frames_dir / f"{idx}.jpg"
        if not rgb_path.exists():
            rgb_path = frames_dir / f"{stem}.jpg"
        if rgb_path.exists():
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            # Sample colors at depth pixel locations (scaled to RGB resolution)
            rgb_rows = np.clip((v_grid[valid_mask] * px_scale_y).astype(int), 0, rgb_h - 1)
            rgb_cols = np.clip((u_grid[valid_mask] * px_scale_x).astype(int), 0, rgb_w - 1)
            colors = rgb[rgb_rows, rgb_cols]
        else:
            colors = np.full((n_valid, 3), 128, dtype=np.uint8)

        all_points.append(p_world.astype(np.float32))
        all_colors.append(colors)

        if (count + 1) % 100 == 0 or count == len(frame_indices) - 1:
            print(f"  [{count+1}/{len(frame_indices)}] frame {idx}: {n_valid:,} pts")

    if not all_points:
        print("[LiDAR PLY] No points generated!")
        return

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    frame_globals = np.concatenate(all_frame_globals, axis=0)
    pixel_rows = np.concatenate(all_pixel_rows, axis=0)
    pixel_cols = np.concatenate(all_pixel_cols, axis=0)

    print(f"\n[LiDAR PLY] Total: {len(points):,} points")

    # Write PLY with traceability
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
        data["red"] = colors[:, 0]
        data["green"] = colors[:, 1]
        data["blue"] = colors[:, 2]
        data["frame_global"] = frame_globals
        data["pixel_row"] = pixel_rows
        data["pixel_col"] = pixel_cols
        f.write(data.tobytes())

    print(f"[LiDAR PLY] ✅ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
