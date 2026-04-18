#!/usr/bin/env python3
"""
Unified Hybrid Pipeline:
  - Poses:  ARKit → refined by VGGT-Long loop closure (SALAD)
  - Depth:  LiDAR (<2m, deterministic) + DA3 calibrated (>2m, metric neural)
  - Output: Clean manual backprojection (same method as the proven vggt_lidar)

Usage:
    cd /home/hernan/stac-builder/server
    bash tests/run_hybrid.sh
"""
import os
import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
server_dir = tests_dir.parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

vggt_dir = server_dir.parent / "vendor" / "VGGT-Long"
if str(vggt_dir) not in sys.path:
    sys.path.insert(0, str(vggt_dir))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Configuration ──
LIDAR_TRUST_M = 2.0       # LiDAR ground truth up to this range
DA3_MAX_RANGE_M = 20.0     # Clamp DA3 extrapolation
CALIB_MIN_DEPTH = 0.3      # Min depth for calibration overlap
INLIER_ROUNDS = 2          # Outlier rejection iterations
INLIER_SIGMA = 3.0         # MAD sigma threshold


def robust_linear_fit(da3_vals, lidar_vals):
    """depth_corrected = da3 * scale + offset, with outlier rejection."""
    mask = np.ones(len(da3_vals), dtype=bool)
    for _ in range(INLIER_ROUNDS):
        if mask.sum() < 20:
            break
        coeffs = np.polyfit(da3_vals[mask], lidar_vals[mask], 1)
        residuals = lidar_vals - np.polyval(coeffs, da3_vals)
        mad = np.median(np.abs(residuals[mask]))
        mask = np.abs(residuals) < INLIER_SIGMA * mad * 1.4826

    if mask.sum() > 20:
        scale, offset = np.polyfit(da3_vals[mask], lidar_vals[mask], 1)
    else:
        scale, offset = np.polyfit(da3_vals, lidar_vals, 1)

    if scale <= 0:
        scale, offset = 1.0, 0.0
    return scale, offset


def run_pipeline(data_dir: str, output_dir: str, stride: int = 4,
                 max_frames: int = 0):
    from ingestors.stray_scanner import prepare_stray_data

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    vggt_output = out / "vggt_loop_output"

    # ================================================================
    # STEP 1: Ingest Stray Scanner (LiDAR depth + ARKit poses)
    # ================================================================
    print("=" * 60)
    print("STEP 1: Ingesting Stray Scanner data (LiDAR + ARKit)")
    print("=" * 60)

    t0 = time.time()
    stray = prepare_stray_data(
        data_dir=data_dir,
        frames_output_dir=str(frames_dir),
        stride=stride,
        max_frames=max_frames,
        confidence_threshold=1,
    )
    print(f"Ingestion took {time.time() - t0:.1f}s")

    n_frames = len(stray['frame_paths'])
    print(f"  Frames: {n_frames}")
    if n_frames == 0:
        print("ERROR: No frames found!")
        return

    K_depth = stray['intrinsics']
    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]

    # Build lookup
    img_list = sorted(frames_dir.glob("*.jpg"))
    fname_to_idx = {Path(p).name: i for i, p in enumerate(stray['frame_paths'])}

    # ================================================================
    # STEP 2: Extract DA3 Giant depth maps (subprocess, env da3)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Extracting DA3 Giant depth maps")
    print("=" * 60)

    da3_dir = out / "da3_depths"
    import subprocess
    cmd = [
        "bash", str(server_dir / "run_da3_extractor.sh"),
        "--image_dir", str(frames_dir),
        "--output_dir", str(da3_dir)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # ================================================================
    # STEP 3: VGGT-Long for loop closure (poses only, no depth)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: VGGT-Long loop closure → refined poses")
    print("=" * 60)
    print("  (Using StrayLiDARAdapter — no neural depth, only SALAD loop detector)")

    from loop_utils.config_utils import load_config
    from vggt_long import VGGT_Long

    config_path = str(vggt_dir / "configs" / "stac_lidar.yaml")
    config = load_config(config_path)

    t0 = time.time()
    prev_cwd = os.getcwd()
    os.chdir(str(vggt_dir))

    pipeline = VGGT_Long(
        image_dir=str(frames_dir),
        save_dir=str(vggt_output),
        config=config,
    )
    # Inject LiDAR data — adapter uses ARKit poses + LiDAR depth
    pipeline.model.stray_data = stray
    pipeline.run()

    os.chdir(prev_cwd)
    print(f"VGGT-Long loop closure took {time.time() - t0:.1f}s")

    # ================================================================
    # STEP 4: Load VGGT-Long's refined poses
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Loading refined poses from VGGT-Long")
    print("=" * 60)

    poses_path = vggt_output / "camera_poses.txt"
    if not poses_path.exists():
        print("ERROR: camera_poses.txt not found! VGGT-Long may have failed.")
        return

    refined_poses = []
    with open(str(poses_path), 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            refined_poses.append(np.array(vals).reshape(4, 4))

    assert len(refined_poses) == len(img_list), \
        f"Pose count {len(refined_poses)} != image count {len(img_list)}"
    print(f"  Loaded {len(refined_poses)} refined poses")

    # Quick comparison: ARKit vs refined
    print("\n  Pose drift (ARKit → VGGT-Long):")
    drifts = []
    for i in range(len(img_list)):
        fname = img_list[i].name
        if fname not in fname_to_idx:
            continue
        idx = fname_to_idx[fname]
        arkit_pos = stray['poses'][idx][:3, 3]
        vggt_pos = refined_poses[i][:3, 3]
        drifts.append(np.linalg.norm(arkit_pos - vggt_pos))
    drifts = np.array(drifts)
    print(f"    Mean: {drifts.mean()*100:.1f}cm, Max: {drifts.max()*100:.1f}cm, "
          f"Std: {drifts.std()*100:.1f}cm")

    # ================================================================
    # STEP 5: Hybrid backprojection (refined poses + LiDAR + DA3)
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Hybrid backprojection")
    print(f"  LiDAR: 0–{LIDAR_TRUST_M}m (deterministic)")
    print(f"  DA3:   >{LIDAR_TRUST_M}m (calibrated against LiDAR)")
    print(f"  Poses: VGGT-Long refined (with loop closure)")
    print("=" * 60)

    all_pts = []
    all_cols = []
    diag_lidar = 0
    diag_da3 = 0
    diag_total = 0

    for i, img_path in enumerate(img_list):
        fname = img_path.name
        if fname not in fname_to_idx:
            continue

        idx = fname_to_idx[fname]
        depth_lidar = stray['depths'][idx]       # (192, 256) float32 m
        pose = refined_poses[i]                  # Refined C2W
        H, W = depth_lidar.shape

        # Load DA3 depth
        da3_path = da3_dir / (img_path.stem + ".npy")
        if not da3_path.exists():
            depth_da3_small = np.zeros_like(depth_lidar)
        else:
            depth_da3 = np.load(str(da3_path))
            depth_da3_small = cv2.resize(depth_da3, (W, H),
                                         interpolation=cv2.INTER_LINEAR)

        # ── Calibrate DA3 against LiDAR ──
        valid_lidar = (depth_lidar > 0) & (depth_lidar < LIDAR_TRUST_M)
        calib_mask = valid_lidar & (depth_lidar > CALIB_MIN_DEPTH)
        if calib_mask.sum() < 100:
            calib_mask = valid_lidar

        if calib_mask.sum() > 50 and depth_da3_small.max() > 0.01:
            scale, offset = robust_linear_fit(
                depth_da3_small[calib_mask], depth_lidar[calib_mask])
            da3_corrected = depth_da3_small * scale + offset
        else:
            da3_corrected = depth_da3_small.copy()
            scale, offset = 1.0, 0.0

        da3_corrected = np.clip(da3_corrected, 0, DA3_MAX_RANGE_M)

        # ── Fusion ──
        depth_hybrid = da3_corrected.copy()
        depth_hybrid[valid_lidar] = depth_lidar[valid_lidar]

        # ── Filter DA3 artifacts at depth edges ──
        # DA3 blends foreground/background at object boundaries, creating
        # ghost points. Detect and remove these using depth gradient.
        da3_only = ~valid_lidar & (depth_hybrid > 0)
        if da3_only.sum() > 0:
            # Sobel gradient magnitude on depth
            grad_x = cv2.Sobel(depth_hybrid, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_hybrid, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Threshold: reject DA3 pixels where gradient > 15% of local depth
            grad_relative = grad_mag / (depth_hybrid + 1e-6)
            edge_mask = grad_relative > 0.15

            # Dilate edge mask to also remove neighbors of edges
            kernel = np.ones((3, 3), np.uint8)
            edge_mask_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel,
                                           iterations=2).astype(bool)

            # Only discard DA3 pixels at edges (keep LiDAR untouched)
            da3_edge_reject = da3_only & edge_mask_dilated
            depth_hybrid[da3_edge_reject] = 0

        valid = (depth_hybrid > 0) & (depth_hybrid < DA3_MAX_RANGE_M)

        # Diagnostics
        n_lidar = int(valid_lidar.sum())
        da3_mask = ~valid_lidar & valid
        n_da3 = int(da3_mask.sum())
        diag_lidar += n_lidar
        diag_da3 += n_da3
        diag_total += H * W

        # ── Backproject ──
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = (u[valid] - cx) * depth_hybrid[valid] / fx
        y = (v[valid] - cy) * depth_hybrid[valid] / fy
        z = depth_hybrid[valid]
        pts_cam = np.stack([x, y, z], axis=-1)

        R, t = pose[:3, :3], pose[:3, 3]
        pts_world = (pts_cam @ R.T) + t

        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (W, H))

        all_pts.append(pts_world.astype(np.float32))
        all_cols.append(rgb_small[valid].astype(np.uint8))

        if (i + 1) % 20 == 0 or i == len(img_list) - 1:
            total_pts = sum(len(p) for p in all_pts)
            print(f"  {i+1}/{len(img_list)} | {total_pts:,} pts | "
                  f"S={scale:.4f} off={offset:.4f}")

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)

    pct_lidar = 100.0 * diag_lidar / max(diag_total, 1)
    pct_da3 = 100.0 * diag_da3 / max(diag_total, 1)
    print(f"\n  Total: {len(pts):,} points from {len(all_pts)} frames")
    print(f"  Source: LiDAR={pct_lidar:.1f}%, DA3={pct_da3:.1f}%")

    # ================================================================
    # STEP 6: Save PLY
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Saving PLY")
    print("=" * 60)

    ply_path = out / "hybrid.ply"
    _save_ply(str(ply_path), pts, cols)
    print(f"  Saved → {ply_path} ({len(pts):,} vertices)")

    desktop = "/mnt/c/Users/ingerop/Desktop/hybrid.ply"
    import shutil
    shutil.copy2(str(ply_path), desktop)
    print(f"  Copied → {desktop}")
    print("\n✅ Done!")


def _save_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Save binary PLY."""
    n = len(points)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    vd = np.empty(n, dtype=dtype)
    vd['x'], vd['y'], vd['z'] = points[:, 0], points[:, 1], points[:, 2]
    vd['r'], vd['g'], vd['b'] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        vd.tofile(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Pipeline: VGGT-Long poses + LiDAR + DA3 depth")
    parser.add_argument("--data_dir", type=str, default="./test2")
    parser.add_argument("--output_dir", type=str, default="./test2_hybrid_output")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames (0=all)")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
    )
