#!/usr/bin/env python3
"""
Simple script to extract DA3 Giant depth maps to NumPy arrays.
Designed to run in the `da3` conda environment to avoid VGGT-Long dependency clashes.
"""
import argparse
import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image

def main():
    parser = argparse.ArgumentParser("Extract DA3 relative depth to NPY")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + 
                    glob.glob(os.path.join(args.image_dir, "*.png")))

    # Skip entirely if all outputs already exist (before loading model)
    missing = []
    for img_path in images:
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")
        if not os.path.exists(depth_path) or not os.path.exists(conf_path):
            missing.append(img_path)

    if not missing:
        print(f"[DA3 Extractor] All {len(images)} depth+conf maps already exist. Skipping.")
        return

    print(f"[DA3 Extractor] {len(missing)} of {len(images)} need processing")

    # Only now load the heavy model
    import sys
    da3_src = os.path.join(os.path.dirname(__file__), "../vendor/depth-anything-3/src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
        
    from depth_anything_3.api import DepthAnything3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DA3 Extractor] Loading {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model, device=device)
    model.eval()

    # Run inference in a batch using the provided list of paths
    with torch.no_grad():
        prediction = model.inference(images)
    
    # prediction.depth has shape [N, H, W], prediction.conf has shape [N, H, W]
    depths = prediction.depth
    confs = prediction.conf
    if isinstance(depths, torch.Tensor):
        depths = depths.cpu().numpy()
    if isinstance(confs, torch.Tensor):
        confs = confs.cpu().numpy()

    # DA3 conf uses expp1 activation (exp(x)+1), range ~1-60+
    # Subtract 1.0 so minimum is 0 (same as DA3-streaming does)
    confs = confs - 1.0
    confs = np.clip(confs, 0, None)

    print(f"[DA3 Extractor] Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
    print(f"[DA3 Extractor] Conf range:  [{confs.min():.3f}, {confs.max():.3f}]")

    for i, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")

        if os.path.exists(depth_path) and os.path.exists(conf_path):
            continue

        print(f"[{i+1}/{len(images)}] {basename} → depth + conf")
        np.save(depth_path, depths[i])
        np.save(conf_path, confs[i])

    print("[DA3 Extractor] Finished successfully.")

if __name__ == "__main__":
    main()
