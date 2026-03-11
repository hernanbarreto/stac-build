#!/usr/bin/env python3
"""
MapAnything Processing Script

Processes WebXR-captured frames through MapAnything (Meta FAIR)
to produce dense metric 3D point clouds.

Usage:
    python mapanything_process.py <scan_dir> [--apache] [--max_views N]

Example:
    python mapanything_process.py server/projects/test2/scans/2026-03-09/src_webxr --apache
"""

import os
import sys
import json
import time
import struct
import argparse
import numpy as np
from pathlib import Path

# Memory efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch


# ─────────────────────────────────────────────────────────────
#  PLY I/O
# ─────────────────────────────────────────────────────────────

def save_ply(path: str, points: np.ndarray):
    """Save binary PLY. Points: Nx3 or Nx6 (with RGB 0-255)."""
    has_color = points.shape[1] >= 6
    n = len(points)

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        xyz = points[:, :3].astype(np.float32)
        if has_color:
            rgb = np.clip(points[:, 3:6], 0, 255).astype(np.uint8)
            for i in range(n):
                f.write(struct.pack('<fff', *xyz[i]))
                f.write(struct.pack('<BBB', *rgb[i]))
        else:
            xyz.tofile(f)


# ─────────────────────────────────────────────────────────────
#  Main Processing
# ─────────────────────────────────────────────────────────────

def process_scan(scan_dir: str, use_apache: bool = True, max_views: int = None, device: str = None):
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    scan_path = Path(scan_dir)
    frames_dir = scan_path / 'frames'
    clouds_dir = scan_path / 'clouds_ma'
    depth_dir = scan_path / 'depth_ma'

    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found")
        sys.exit(1)

    clouds_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    # Collect frames
    frame_files = sorted(frames_dir.glob('*.jpg'))
    if not frame_files:
        frame_files = sorted(frames_dir.glob('*.png'))
    if not frame_files:
        print(f"ERROR: No frames in {frames_dir}")
        sys.exit(1)

    if max_views and max_views < len(frame_files):
        frame_files = frame_files[:max_views]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'=' * 60}")
    print(f"MapAnything Processing")
    print(f"{'=' * 60}")
    print(f"  Scan dir:  {scan_path}")
    print(f"  Frames:    {len(frame_files)}")
    print(f"  Device:    {device}")
    print(f"  Model:     {'apache (commercial)' if use_apache else 'default (CC-BY-NC)'}")
    print(f"  Output:    {clouds_dir}")
    print()

    # Load model
    model_name = "facebook/map-anything-apache" if use_apache else "facebook/map-anything"
    print(f"Loading model: {model_name}...")
    t0 = time.time()
    model = MapAnything.from_pretrained(model_name).to(device)
    use_amp_flag = device != "cpu"
    amp_type = "bf16" if device != "cpu" else "fp32"
    print(f"  Loaded in {time.time() - t0:.1f}s on {device}")
    print()

    # Load images
    print("Loading and preprocessing images...")
    frame_paths = [str(f) for f in frame_files]
    views = load_images(frame_paths)
    print(f"  Loaded {len(views)} views")

    # Run inference
    print("\nRunning inference...")
    t1 = time.time()
    predictions = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=use_amp_flag,
        amp_dtype=amp_type,
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
    t_infer = time.time() - t1
    print(f"  Inference done in {t_infer:.1f}s ({t_infer/len(frame_files):.2f}s/frame)")

    # Process predictions
    print("\nExtracting point clouds...")
    total_points = 0
    all_points = []
    all_colors = []

    for i, pred in enumerate(predictions):
        frame_name = frame_files[i].stem

        # 3D points in world coordinates (H, W, 3)
        pts3d = pred["pts3d"].squeeze().cpu().numpy()
        # Mask (H, W, 1)
        mask = pred["mask"].squeeze().cpu().numpy().astype(bool)
        # Confidence
        conf = pred["conf"].squeeze().cpu().numpy()
        # Color (H, W, 3) - denormalized
        img = pred["img_no_norm"].squeeze().cpu().numpy()  # 0-1 float
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Depth
        depth_z = pred["depth_z"].squeeze().cpu().numpy()

        # Camera pose
        cam_pose = pred["camera_poses"].squeeze().cpu().numpy()

        # Apply mask - handle different shapes
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        valid_pts = pts3d[mask_2d]
        valid_rgb = img[mask_2d]
        n_pts = len(valid_pts)

        # Save per-frame point cloud
        if n_pts > 0:
            points_with_color = np.hstack([
                valid_pts.astype(np.float32),
                valid_rgb.astype(np.float32)
            ])
            save_ply(str(clouds_dir / f"{frame_name}.ply"), points_with_color)

            all_points.append(valid_pts)
            all_colors.append(valid_rgb)

        # Save depth
        np.save(str(depth_dir / f"{frame_name}.npy"), depth_z)

        # Save pose
        pose_path = depth_dir / f"{frame_name}_pose.npy"
        np.save(str(pose_path), cam_pose)

        total_points += n_pts
        d_valid = depth_z[mask_2d]
        d_min = d_valid.min() if len(d_valid) > 0 else 0
        d_max = d_valid.max() if len(d_valid) > 0 else 0

        print(f"  [{i+1}/{len(frame_files)}] {frame_name}: "
              f"{n_pts:,} pts, depth={d_min:.2f}-{d_max:.2f}m")

    # Save merged cloud
    if all_points:
        print("\nMerging all point clouds...")
        merged_pts = np.vstack(all_points)
        merged_rgb = np.vstack(all_colors)
        merged = np.hstack([merged_pts.astype(np.float32),
                           merged_rgb.astype(np.float32)])
        merged_path = scan_path / 'cloud_mapanything.ply'
        save_ply(str(merged_path), merged)
        print(f"  Merged cloud: {len(merged):,} points → {merged_path}")

    # Save poses summary
    poses_summary = {}
    for i, pred in enumerate(predictions):
        frame_name = frame_files[i].stem
        cam_pose = pred["camera_poses"].squeeze().cpu().numpy()
        intr = pred["intrinsics"].squeeze().cpu().numpy()
        poses_summary[frame_name] = {
            "camera_pose": cam_pose.tolist(),
            "intrinsics": intr.tolist(),
        }
    with open(scan_path / 'mapanything_poses.json', 'w') as f:
        json.dump(poses_summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done! {len(frame_files)} frames in {elapsed:.1f}s")
    print(f"  Total points:  {total_points:,}")
    print(f"  Per-frame:     {clouds_dir}")
    print(f"  Merged cloud:  {scan_path / 'cloud_mapanything.ply'}")
    print(f"  Poses:         {scan_path / 'mapanything_poses.json'}")
    print(f"{'=' * 60}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MapAnything 3D reconstruction')
    parser.add_argument('scan_dir', help='WebXR scan directory')
    parser.add_argument('--apache', action='store_true', default=True,
        help='Use Apache 2.0 licensed model (default: True)')
    parser.add_argument('--full', action='store_true',
        help='Use CC-BY-NC model (better quality, non-commercial)')
    parser.add_argument('--max_views', type=int, default=None,
        help='Limit number of views to process')
    parser.add_argument('--cpu', action='store_true',
        help='Force CPU execution')
    args = parser.parse_args()

    use_apache = not args.full
    device = "cpu" if args.cpu else None
    process_scan(args.scan_dir, use_apache=use_apache, max_views=args.max_views, device=device)
