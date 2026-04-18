#!/usr/bin/env python3
"""
VGGT-Long Hybrid Pipeline (DA3 + LiDAR + ARKit + SALAD Loop Closure)

Pipeline:
  1. Ingest Stray Scanner data (LiDAR depth + ARKit initial poses)
  2. Run VGGT-Long with HybridDepthAdapter:
     - Detects loops with SALAD
     - Infers depth with DA3 Giant (>5m)
     - Calibrates DA3 scale using LiDAR overlap (0-5m)
     - Aligns with SIM3 optimization
  3. Backproject final hybrid depth using VGGT-Long's refined poses
  4. Save colored PLY

Usage:
    cd /home/hernan/stac-builder/server
    python test_vggt_hybrid.py --data_dir ./test2
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

# Force CPU (to avoid OOM, or comment out to use GPU if applicable)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_vggt_hybrid(data_dir: str, output_dir: str, stride: int = 4,
                   max_frames: int = 0, device_str: str = "cpu"):
    """Run VGGT-Long Hybrid Pipeline."""
    from ingestors.stray_scanner import prepare_stray_data
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    vggt_output = out / "vggt_hybrid_output"
    
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
    
    if n_frames == 0:
        print("ERROR: No frames found!")
        return
        
    # ── Step 1.5: Extract DA3 Depth ──
    print("\n" + "=" * 60)
    print("STEP 1.5: Harvesting DA3 Giant depth maps in native environment")
    print("=" * 60)
    
    da3_depth_dir = out / "da3_depths"
    import subprocess
    cmd = [
        "bash", str(server_dir / "run_da3_extractor.sh"),
        "--image_dir", str(frames_dir),
        "--output_dir", str(da3_depth_dir)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Store path in stray dict for the adapter
    stray['da3_dir'] = str(da3_depth_dir)
    
    # We will need these later to map the outputs
    img_list = sorted(Path(frames_dir).glob("*.jpg"))
    fname_to_idx = {Path(p).name: i for i, p in enumerate(stray['frame_paths'])}
    
    # ── Step 2: Run VGGT-Long for Hybrid Depth & Globally-consistent poses ──
    print("\n" + "=" * 60)
    print("STEP 2: Running VGGT-Long Hybrid (DA3 + LiDAR + Loop Closure)")
    print("=" * 60)
    
    from loop_utils.config_utils import load_config
    from vggt_long import VGGT_Long
    
    config_path = str(vggt_dir / "configs" / "stac_hybrid.yaml")
    config = load_config(config_path)
    
    t0 = time.time()
    # Change working directory to VGGT-Long for relative dependencies (DINOv2)
    os.chdir(str(vggt_dir))
    
    pipeline = VGGT_Long(
        image_dir=str(frames_dir),
        save_dir=str(vggt_output),
        config=config,
    )
    
    # Inject real sensor data into the Hybrid adapter
    pipeline.model.stray_data = stray
    
    pipeline.run()
    
    # Restore working directory
    os.chdir(str(server_dir))
    print(f"VGGT-Long Hybrid took {time.time() - t0:.1f}s")
    
    # ── Step 3: Extract VGGT-Long's refined poses AND the depth maps ──
    print("\n" + "=" * 60)
    print("STEP 3: Extracting Hybrid Depth and Refined Poses")
    print("=" * 60)
    
    # In VGGT-Long, the chunks save points and colors, but because we want
    # clean backprojection we read the saved camera poses and we can re-evaluate
    # the hybrid depth (or we can just let VGGT-Long's save_pointcloud export it).
    # Since VGGT-Long saves a .ply per chunk, we can just merge them, or 
    # we can rebuild it exactly by re-reading the depth. 
    # Let's rely on the PLYs that VGGT-Long's `save_pointcloud()` already generated 
    # in `vggt_output/pcd/` since it handles all the masking correctly!
    
    pcd_dir = vggt_output / "pcd"
    if not pcd_dir.exists():
        print(f"ERROR: {pcd_dir} not found! VGGT-Long may have failed.")
        return
        
    import open3d as o3d
    ply_files = list(pcd_dir.glob("*.ply"))
    print(f"Found {len(ply_files)} chunk PLY files. Merging...")
    
    merged_pcd = o3d.geometry.PointCloud()
    for ply in ply_files:
        pcd = o3d.io.read_point_cloud(str(ply))
        merged_pcd += pcd
        
    # Downsample slightly to clean up overlap points
    print("Downsampling merged point cloud...")
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01) # 1cm voxels
    
    ply_path = out / "vggt_hybrid.ply"
    o3d.io.write_point_cloud(str(ply_path), merged_pcd)
    print(f"\nSaved Hybrid PLY → {ply_path}")
    
    # Copy to desktop
    desktop_path = "/mnt/c/Users/ingerop/Desktop/vggt_hybrid.ply"
    import shutil
    shutil.copy2(str(ply_path), desktop_path)
    print(f"Copied → {desktop_path}")
    print("\n✅ Done! The Hybrid Point Cloud is ready to inspect.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT-Long Hybrid Pipeline")
    parser.add_argument("--data_dir", type=str, default="./test2")
    parser.add_argument("--output_dir", type=str, default="./test2_hybrid_output")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=30, # testing small first
                        help="Max frames (0=all)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    run_vggt_hybrid(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        device_str=args.device,
    )
