#!/usr/bin/env python3
"""
DA3 Simple - Direct API for small image sets (no streaming pipeline)
Uses the Depth Anything 3 API directly without chunking/streaming overhead.
"""

import argparse
import glob
import os
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from depth_anything_3.api import DepthAnything3


def depth_to_pointcloud(depth, intrinsics, extrinsics, color_image):
    """
    Convert depth map to 3D point cloud in world coordinates.
    
    Args:
        depth: [H, W] depth map
        intrinsics: [3, 3] camera intrinsic matrix
        extrinsics: [3, 4] world-to-camera matrix (w2c)
        color_image: [H, W, 3] RGB image
    
    Returns:
        points: [N, 3] 3D points in world coordinates
        colors: [N, 3] RGB colors
    """
    H, W = depth.shape
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # Unproject to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack to [H, W, 3]
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates (invert w2c to get c2w)
    w2c = np.eye(4)
    w2c[:3, :] = extrinsics
    c2w = np.linalg.inv(w2c)
    
    # Reshape for transformation
    points_flat = points_cam.reshape(-1, 3)
    points_homo = np.concatenate([points_flat, np.ones((points_flat.shape[0], 1))], axis=1)
    points_world = (c2w @ points_homo.T).T[:, :3]
    
    # Get colors
    colors = color_image.reshape(-1, 3)
    
    # Filter invalid points (depth = 0 or too far)
    valid_mask = (depth.flatten() > 0) & (depth.flatten() < 100)
    
    return points_world[valid_mask], colors[valid_mask]


def save_ply(points, colors, output_path):
    """Save point cloud as PLY file."""
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                   f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")


def main():
    parser = argparse.ArgumentParser(description="DA3 Simple - Direct API for small image sets")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model", type=str, default="depth-anything/DA3-BASE", 
                       help="Model name (default: DA3-BASE for speed)")
    parser.add_argument("--sample_ratio", type=float, default=0.1, 
                       help="Point cloud sampling ratio (0-1)")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = os.path.join("./exps_simple", os.path.basename(args.image_dir), timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find images
    images = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg")) +
        glob.glob(os.path.join(args.image_dir, "*.png"))
    )
    
    if len(images) == 0:
        print(f"No images found in {args.image_dir}")
        return
    
    print(f"Found {len(images)} images")
    print(f"Output directory: {args.output_dir}")
    
    # Load model
    print(f"Loading model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)
    print("Model loaded!")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        prediction = model.inference(images)
    
    print(f"  Images shape: {prediction.processed_images.shape}")
    print(f"  Depth shape: {prediction.depth.shape}")
    print(f"  Conf shape: {prediction.conf.shape}")
    print(f"  Extrinsics shape: {prediction.extrinsics.shape}")
    print(f"  Intrinsics shape: {prediction.intrinsics.shape}")
    
    # Debug confidence
    print(f"  Confidence - min: {np.min(prediction.conf):.4f}, max: {np.max(prediction.conf):.4f}, mean: {np.mean(prediction.conf):.4f}")
    
    # Generate point cloud for each image
    all_points = []
    all_colors = []
    
    for i in range(len(images)):
        print(f"Processing image {i+1}/{len(images)}: {os.path.basename(images[i])}")
        
        depth = prediction.depth[i]
        intrinsics = prediction.intrinsics[i]
        extrinsics = prediction.extrinsics[i]
        color = prediction.processed_images[i]
        
        points, colors = depth_to_pointcloud(depth, intrinsics, extrinsics, color)
        
        # Sample points
        if args.sample_ratio < 1.0 and len(points) > 0:
            n_samples = int(len(points) * args.sample_ratio)
            indices = np.random.choice(len(points), n_samples, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        print(f"  Generated {len(points)} points")
        all_points.append(points)
        all_colors.append(colors)
    
    # Combine all points
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    print(f"\nTotal points: {len(all_points)}")
    
    # Save point cloud
    output_path = os.path.join(args.output_dir, "pointcloud.ply")
    print(f"Saving point cloud to {output_path}")
    save_ply(all_points, all_colors, output_path)
    
    # Save camera poses
    poses_path = os.path.join(args.output_dir, "camera_poses.txt")
    with open(poses_path, 'w') as f:
        for i in range(len(images)):
            w2c = np.eye(4)
            w2c[:3, :] = prediction.extrinsics[i]
            c2w = np.linalg.inv(w2c)
            f.write(" ".join([str(x) for x in c2w.flatten()]) + "\n")
    print(f"Camera poses saved to {poses_path}")
    
    # Save intrinsics
    intrinsics_path = os.path.join(args.output_dir, "intrinsics.txt")
    with open(intrinsics_path, 'w') as f:
        for i in range(len(images)):
            K = prediction.intrinsics[i]
            f.write(f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")
    print(f"Intrinsics saved to {intrinsics_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
