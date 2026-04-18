#!/usr/bin/env python3
"""
Unified Cloud: Hybrid DA3 + LiDAR backprojected with DA3-streaming poses.

Steps:
  1. Load DA3-streaming's camera_poses.txt (post-loop-closure C2W matrices)
  2. Load Stray Scanner LiDAR depth maps (native 192x256)
  3. Backproject LiDAR depth using DA3-streaming's poses → lidar_da3_poses.ply
  4. CloudCompPy: MergeEntities(combined_pcd.ply, lidar_da3_poses.ply) + filter
     → unified_cleaned.ply

Usage:
    cd /home/hernan/stac-builder/server
    bash tests/run_unified_cloud.sh
"""
import os
import sys
import argparse
import subprocess
import time
import shutil
import numpy as np
import cv2
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
server_dir = tests_dir.parent

if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))


def load_camera_poses(poses_path):
    """Load C2W 4x4 matrices from camera_poses.txt (16 floats per line)."""
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 16:
                poses.append(np.array([float(v) for v in vals]).reshape(4, 4))
    return poses


def save_ply(path, points, colors):
    """Save binary PLY."""
    n = len(points)
    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                      ('r','u1'),('g','u1'),('b','u1')])
    vd = np.empty(n, dtype=dtype)
    vd['x'], vd['y'], vd['z'] = points[:,0], points[:,1], points[:,2]
    vd['r'], vd['g'], vd['b'] = colors[:,0], colors[:,1], colors[:,2]
    with open(path, 'wb') as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"end_header\n".encode('ascii'))
        vd.tofile(f)
    print(f"  Saved {n:,} pts → {path} ({os.path.getsize(path)/1048576:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./test2")
    parser.add_argument("--hybrid_dir", default="./test2_da3_hybrid")
    parser.add_argument("--output_dir", default="./test2_da3_hybrid/unified")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--skip_backproject", action="store_true",
                        help="Skip backprojection if lidar_da3_poses.ply exists")
    parser.add_argument("--voxel_size", type=float, default=0.002)
    parser.add_argument("--sor_knn", type=int, default=8)
    parser.add_argument("--sor_sigma", type=float, default=3.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hybrid_dir = Path(args.hybrid_dir)
    da3_output = hybrid_dir / "da3_output"
    frames_dir = hybrid_dir / "frames"
    combined_ply = da3_output / "pcd" / "combined_pcd.ply"
    poses_path = da3_output / "camera_poses.txt"
    lidar_ply = out / "lidar_da3_poses.ply"

    # ── STEP 1-3: Generate LiDAR cloud (skip if already exists) ──
    if args.skip_backproject and lidar_ply.exists():
        print(f"Skipping backprojection — {lidar_ply} exists ({lidar_ply.stat().st_size/1048576:.0f} MB)")
    else:
        # STEP 1: Load poses
        print("=" * 60)
        print("STEP 1: Loading DA3-streaming poses")
        print("=" * 60)
        poses = load_camera_poses(str(poses_path))
        print(f"  {len(poses)} poses loaded")

        # STEP 2: Load LiDAR data
        print("\n" + "=" * 60)
        print("STEP 2: Loading Stray Scanner LiDAR data")
        print("=" * 60)
        from ingestors.stray_scanner import prepare_stray_data
        stray = prepare_stray_data(
            data_dir=args.data_dir,
            frames_output_dir=str(frames_dir),
            stride=args.stride,
            max_frames=args.max_frames,
            confidence_threshold=1,
        )
        print(f"  {len(stray['frame_paths'])} frames")

        # STEP 3: Backproject LiDAR
        print("\n" + "=" * 60)
        print("STEP 3: Backprojecting LiDAR with DA3-streaming poses")
        print("=" * 60)
        K = stray['intrinsics']
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))
        n = min(len(stray['depths']), len(poses), len(frame_files))

        all_pts, all_cols = [], []
        t0 = time.time()
        for i in range(n):
            depth = stray['depths'][i]
            c2w = poses[i]
            rgb = cv2.cvtColor(cv2.imread(str(frame_files[i])), cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
            H, W = depth.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            valid = depth > 0
            pts_cam = np.stack([
                (u[valid] - cx) * depth[valid] / fx,
                (v[valid] - cy) * depth[valid] / fy,
                depth[valid]
            ], axis=-1)
            pts_world = (pts_cam @ c2w[:3,:3].T) + c2w[:3,3]
            all_pts.append(pts_world.astype(np.float32))
            all_cols.append(rgb_small[valid].astype(np.uint8))
            if (i+1) % 100 == 0 or i == n-1:
                print(f"  {i+1}/{n} frames → {sum(len(p) for p in all_pts):,} pts")

        save_ply(str(lidar_ply), np.concatenate(all_pts), np.concatenate(all_cols))
        print(f"  ({time.time()-t0:.1f}s)")

    # ── STEP 4: CloudCompPy merge + filter ──
    print("\n" + "=" * 60)
    print("STEP 4: CloudCompPy — merge combined_pcd.ply + lidar_da3_poses.ply + filter")
    print("=" * 60)

    if not combined_ply.exists():
        print(f"  ERROR: {combined_ply} not found!")
        sys.exit(1)

    print(f"  PLY 1: {combined_ply.name} ({combined_ply.stat().st_size/1048576:.0f} MB)")
    print(f"  PLY 2: {lidar_ply.name} ({lidar_ply.stat().st_size/1048576:.0f} MB)")

    # Create clean input dir with ONLY the 2 PLYs (as *_pcd.ply for cloudcompy glob)
    cc_dir = out / "_cc_input"
    if cc_dir.exists():
        shutil.rmtree(cc_dir)
    cc_dir.mkdir()
    os.symlink(str(combined_ply.resolve()), str(cc_dir / "a_combined_pcd.ply"))
    os.symlink(str(lidar_ply.resolve()), str(cc_dir / "b_lidar_pcd.ply"))

    # Verify only 2 files
    cc_files = list(cc_dir.glob("*_pcd.ply"))
    print(f"  Input dir: {cc_dir} ({len(cc_files)} files)")
    for f in cc_files:
        print(f"    {f.name}")

    cleaned = str(out / "unified_cleaned.ply")
    script = server_dir / "run_cloudcompy.sh"

    if not script.exists():
        print(f"  ERROR: {script} not found!")
        sys.exit(1)

    cmd = ["bash", str(script),
           "--input-dir", str(cc_dir),
           "--output", cleaned,
           "--voxel-size", str(args.voxel_size),
           "--sor-knn", str(args.sor_knn),
           "--sor-sigma", str(args.sor_sigma),
           "--skip-noise", "--skip-normals"]
    print(f"  Command: run_cloudcompy.sh --input-dir _cc_input --output unified_cleaned.ply")
    print(f"  Params: voxel={args.voxel_size*1000:.0f}mm, SOR knn={args.sor_knn} σ={args.sor_sigma}")

    proc = subprocess.run(cmd)

    if proc.returncode != 0:
        print(f"  ❌ CloudCompPy failed (exit code {proc.returncode})")
        sys.exit(1)

    if not os.path.exists(cleaned):
        print("  ❌ No output produced")
        sys.exit(1)

    size_mb = os.path.getsize(cleaned) / 1048576
    print(f"\n  ✅ unified_cleaned.ply ({size_mb:.0f} MB)")

    # Copy to Desktop
    desktop = "/mnt/c/Users/ingerop/Desktop/unified_cloud.ply"
    shutil.copy2(cleaned, desktop)
    print(f"  ✅ Copied → {desktop}")


if __name__ == "__main__":
    main()
