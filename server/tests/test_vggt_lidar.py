#!/usr/bin/env python3
"""
VGGT-Long (poses with loop closure) + LiDAR depth (raw backprojection).

Pipeline:
  1. Ingest Stray Scanner data (LiDAR depth + ARKit initial poses)
  2. Run VGGT-Long on RGB frames → globally-consistent poses with loop closure
  3. Backproject LiDAR depth using VGGT-Long's refined poses
  4. Save colored PLY

Usage:
    cd /home/hernan/stac-builder/server
    bash run_vggt_lidar.sh
"""
import os
import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path

server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))

vggt_dir = server_dir.parent / "vendor" / "VGGT-Long"
if str(vggt_dir) not in sys.path:
    sys.path.insert(0, str(vggt_dir))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_vggt_lidar(data_dir: str, output_dir: str, stride: int = 4,
                   max_frames: int = 0, device_str: str = "cuda"):
    """Run VGGT-Long for poses + LiDAR for depth."""
    from ingestors.stray_scanner import prepare_stray_data
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    vggt_output = out / "vggt_long_output"
    
    # ── Step 1: Ingest Stray Scanner data ──
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
    print(f"  RGB: {stray['rgb_shape']}, Depth: {stray['depth_shape']}")
    
    if n_frames == 0:
        print("ERROR: No frames found!")
        return
    
    # Build frame index → data mapping for LiDAR backprojection later
    K_depth = stray['intrinsics']  # Already scaled to depth resolution
    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx, cy = K_depth[0, 2], K_depth[1, 2]
    
    frame_data = {}
    for i in range(n_frames):
        fname = Path(stray['frame_paths'][i]).name
        frame_data[fname] = {
            'depth': stray['depths'][i],
            'arkit_pose': stray['poses'][i],
            'index': i,
        }
    
    # ── Step 2: Run VGGT-Long for globally-consistent poses ──
    print("\n" + "=" * 60)
    print("STEP 2: Running VGGT-Long (loop closure + global optimization)")
    print("=" * 60)
    
    from loop_utils.config_utils import load_config
    from vggt_long import VGGT_Long
    
    config_path = str(vggt_dir / "configs" / "stac_lidar.yaml")
    config = load_config(config_path)
    
    t0 = time.time()
    pipeline = VGGT_Long(
        image_dir=str(frames_dir),
        save_dir=str(vggt_output),
        config=config,
    )
    
    # Inject real sensor data into the StrayLiDAR adapter
    # (replaces MapAnything neural inference with LiDAR + ARKit)
    pipeline.model.stray_data = stray
    
    pipeline.run()
    print(f"VGGT-Long took {time.time() - t0:.1f}s")
    
    # ── Step 3: Extract VGGT-Long's refined poses ──
    print("\n" + "=" * 60)
    print("STEP 3: Extracting VGGT-Long refined poses")
    print("=" * 60)
    
    poses_path = vggt_output / "camera_poses.txt"
    if not poses_path.exists():
        print("ERROR: camera_poses.txt not found! VGGT-Long may have failed.")
        return
    
    vggt_poses = []
    with open(str(poses_path), 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            pose_4x4 = np.array(vals).reshape(4, 4)
            vggt_poses.append(pose_4x4)
    
    # VGGT-Long processes img_list which is sorted(glob("*.jpg"))
    img_list = sorted(Path(frames_dir).glob("*.jpg"))
    assert len(vggt_poses) == len(img_list), \
        f"Pose count {len(vggt_poses)} != image count {len(img_list)}"
    
    print(f"  Loaded {len(vggt_poses)} refined poses")
    
    # Compare ARKit vs VGGT-Long poses
    print("\n  Pose comparison (first 5 frames):")
    for i in range(min(5, len(img_list))):
        fname = img_list[i].name
        if fname in frame_data:
            arkit_pos = frame_data[fname]['arkit_pose'][:3, 3]
            vggt_pos = vggt_poses[i][:3, 3]
            diff = np.linalg.norm(arkit_pos - vggt_pos)
            print(f"    {fname}: ARKit=({arkit_pos[0]:+.3f}, {arkit_pos[1]:+.3f}, {arkit_pos[2]:+.3f}) "
                  f"VGGT=({vggt_pos[0]:+.3f}, {vggt_pos[1]:+.3f}, {vggt_pos[2]:+.3f}) "
                  f"Δ={diff*100:.1f}cm")
    
    # ── Step 4: Backproject LiDAR depth with VGGT-Long poses ──
    print("\n" + "=" * 60)
    print("STEP 4: Backprojecting LiDAR depth with refined poses")
    print("=" * 60)
    
    all_pts = []
    all_cols = []
    
    for i, img_path in enumerate(img_list):
        fname = img_path.name
        if fname not in frame_data:
            continue
        
        depth = frame_data[fname]['depth']  # Native 192x256
        vggt_pose = vggt_poses[i]          # VGGT-Long's refined C2W
        
        # Resize RGB to depth resolution for coloring
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid = depth > 0
        
        x = (u[valid] - cx) * depth[valid] / fx
        y = (v[valid] - cy) * depth[valid] / fy
        z = depth[valid]
        pts_cam = np.stack([x, y, z], axis=-1)
        
        R, t = vggt_pose[:3, :3], vggt_pose[:3, 3]
        pts_world = (pts_cam @ R.T) + t
        
        all_pts.append(pts_world.astype(np.float32))
        all_cols.append(rgb_small[valid].astype(np.uint8))
        
        if (i + 1) % 20 == 0:
            total = sum(len(p) for p in all_pts)
            print(f"  {i + 1}/{len(img_list)} frames, {total} points")
    
    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    
    print(f"\nTotal: {len(all_pts)} frames, {len(pts)} points")
    
    # ── Step 5: Save PLY ──
    ply_path = out / "vggt_lidar.ply"
    _save_ply(str(ply_path), pts, cols)
    print(f"\nSaved → {ply_path}")
    
    # Copy to desktop
    desktop_path = "/mnt/c/Users/ingerop/Desktop/vggt_lidar.ply"
    import shutil
    shutil.copy2(str(ply_path), desktop_path)
    print(f"Copied → {desktop_path}")
    
    # Also generate ARKit-only baseline for comparison
    print("\nGenerating ARKit-only baseline for comparison...")
    all_pts_arkit = []
    all_cols_arkit = []
    for i, img_path in enumerate(img_list):
        fname = img_path.name
        if fname not in frame_data:
            continue
        depth = frame_data[fname]['depth']
        arkit_pose = frame_data[fname]['arkit_pose']
        rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid = depth > 0
        x = (u[valid] - cx) * depth[valid] / fx
        y = (v[valid] - cy) * depth[valid] / fy
        z = depth[valid]
        pts_cam = np.stack([x, y, z], axis=-1)
        R, t = arkit_pose[:3, :3], arkit_pose[:3, 3]
        pts_world = (pts_cam @ R.T) + t
        all_pts_arkit.append(pts_world.astype(np.float32))
        all_cols_arkit.append(rgb_small[valid].astype(np.uint8))
    
    pts_arkit = np.concatenate(all_pts_arkit)
    cols_arkit = np.concatenate(all_cols_arkit)
    arkit_ply = out / "arkit_lidar.ply"
    _save_ply(str(arkit_ply), pts_arkit, cols_arkit)
    arkit_desktop = "/mnt/c/Users/ingerop/Desktop/arkit_lidar.ply"
    shutil.copy2(str(arkit_ply), arkit_desktop)
    print(f"ARKit baseline → {arkit_desktop}")
    
    print("\n✅ Done! Compare both on Desktop:")
    print("  vggt_lidar.ply  → VGGT-Long refined poses + LiDAR depth")
    print("  arkit_lidar.ply → Raw ARKit poses + LiDAR depth")


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
    parser = argparse.ArgumentParser(description="VGGT-Long poses + LiDAR depth")
    parser.add_argument("--data_dir", type=str, default="./test2")
    parser.add_argument("--output_dir", type=str, default="./test2_output")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames (0=all)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    run_vggt_lidar(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        device_str=args.device,
    )
