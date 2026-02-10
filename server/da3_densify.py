#!/usr/bin/env python3
"""
DA3 Single-Frame Densification Script
Runs in DA3 conda environment, called as subprocess from main server.

Usage:
    python da3_densify.py --image <path> --pose <pose.npy> --output <output.npz> [--intrinsics <K.npy>]

Input:
    - image: path to keyframe image (JPEG/PNG)
    - pose: 4x4 T_WC transformation matrix (NPY file)
    - intrinsics: optional 3x3 camera matrix (NPY file)

Output:
    - NPZ file with: points (N,3), colors (N,3), confidence (N)

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

import argparse
import sys
import os
import numpy as np
import cv2

# Add DA3 source paths
DA3_ROOT = os.path.expanduser("~/Depth-Anything-3")
sys.path.insert(0, os.path.join(DA3_ROOT, "src"))
sys.path.insert(0, DA3_ROOT)

# Cached model to avoid reloading on repeated calls
_cached_model = None


def get_model():
    """Load DA3 model (cached after first call)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    
    import torch
    from depth_anything_3.api import DepthAnything3
    
    print("[DA3 Densify] Loading model da3-large...")
    model = DepthAnything3(model_name="da3-large")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.device = device
    
    _cached_model = model
    print(f"[DA3 Densify] Model loaded on {device}")
    return model


def densify_single_frame(
    image_path: str,
    pose_T_WC: np.ndarray,
    intrinsics: np.ndarray = None,
    confidence_threshold: float = 0.3,
):
    """
    Generate dense point cloud from single image using DA3 depth + provided pose.
    
    Returns:
        Tuple of (points [N,3] world-space, colors [N,3] 0-1, confidence [N])
    """
    import torch
    
    model = get_model()
    
    # Load original image for colors (DA3 processed_images may be resized)
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Run DA3 inference on single image
    with torch.no_grad():
        prediction = model.inference(
            image=[image_path],
        )
    
    # Extract outputs: depth (1, H, W), conf (1, H, W)
    depth = prediction.depth[0]    # [H, W]
    
    H, W = depth.shape
    print(f"[DA3 Densify] Depth map: {H}x{W}, range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # Handle confidence - DA3 conf is in log-space (exp mode starts at 1)
    if prediction.conf is not None:
        conf = prediction.conf[0]  # [H, W]
        print(f"[DA3 Densify] Confidence: range [{conf.min():.2f}, {conf.max():.2f}]")
        # Normalize confidence to [0, 1] range using percentile-based approach
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    else:
        conf_norm = np.ones_like(depth)
    
    # Use DA3's predicted intrinsics if we don't have them
    if intrinsics is None:
        if prediction.intrinsics is not None:
            intrinsics = prediction.intrinsics[0]  # [3, 3]
            print(f"[DA3 Densify] Using DA3 predicted intrinsics: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")
        else:
            fx = fy = W / (2 * np.tan(np.radians(30)))
            cx, cy = W / 2, H / 2
            intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            print(f"[DA3 Densify] Using estimated intrinsics (60° FOV)")
    
    # Generate point cloud in camera space via back-projection
    u = np.arange(W).reshape(1, W).repeat(H, axis=0).astype(np.float32)
    v = np.arange(H).reshape(H, 1).repeat(W, axis=1).astype(np.float32)
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    X_cam = (u - cx) * depth / fx
    Y_cam = (v - cy) * depth / fy
    Z_cam = depth
    
    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1).reshape(-1, 3)
    
    # Transform to world space using MASt3R's pose
    R = pose_T_WC[:3, :3]
    t = pose_T_WC[:3, 3]
    points_world = (R @ points_cam.T).T + t
    
    # Get colors from original image (resized to depth map resolution)
    color_img = cv2.resize(orig_img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    colors = color_img.reshape(-1, 3).astype(np.float32) / 255.0
    
    # Filter by confidence and valid depth
    conf_flat = conf_norm.flatten()
    depth_flat = depth.flatten()
    valid = (conf_flat > confidence_threshold) & (depth_flat > 0.01) & (depth_flat < 100.0)
    
    n_valid = valid.sum()
    n_total = len(conf_flat)
    print(f"[DA3 Densify] Filtering: {n_valid:,}/{n_total:,} points pass (conf>{confidence_threshold}, depth=[0.01, 100])")
    
    return points_world[valid], colors[valid], conf_flat[valid]


def main():
    parser = argparse.ArgumentParser(description="DA3 Single-Frame Densification")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--pose", required=True, help="Path to pose matrix (4x4 NPY)")
    parser.add_argument("--output", required=True, help="Path to output NPZ")
    parser.add_argument("--intrinsics", help="Path to intrinsics matrix (3x3 NPY)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Load pose
    pose = np.load(args.pose)
    if pose.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4, got {pose.shape}")
    
    # Load intrinsics if provided
    intrinsics = None
    if args.intrinsics:
        intrinsics = np.load(args.intrinsics)
    
    # Run densification
    points, colors, confidence = densify_single_frame(
        args.image,
        pose,
        intrinsics,
        args.confidence
    )
    
    # Save output
    np.savez(args.output, 
             points=points.astype(np.float32),
             colors=colors.astype(np.float32),
             confidence=confidence.astype(np.float32))
    
    print(f"[DA3 Densify] ✅ {len(points):,} points -> {args.output}")
    

if __name__ == "__main__":
    main()
