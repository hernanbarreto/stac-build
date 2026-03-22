#!/usr/bin/env python3
"""
Test 1: PE Spatial G14 — Dense Feature Extraction
Extracts spatial features from a construction frame and saves a PCA visualization.

Usage:
  conda activate pe_spatial
  cd /home/hernan/stac-builder
  python tests/test_pe_spatial.py --image server/data/projects/test1/scans/2026-02-24/src_legacy/frames/00045.jpg
"""
import argparse
import sys
import time
import os

# Add perception_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'perception_models'))

import torch
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Test PE Spatial G14 feature extraction")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default="PE-Spatial-G14-448", 
                       help="Model config name")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Local path to .pt checkpoint (skips HuggingFace download)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, cpu")
    parser.add_argument("--output", type=str, default="/tmp/pe_spatial_test",
                       help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[PE Spatial] Using device: {device}")

    # Load model
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    print(f"[PE Spatial] Available configs: {pe.VisionTransformer.available_configs()}")
    print(f"[PE Spatial] Loading {args.model}...")
    t0 = time.time()
    model = pe.VisionTransformer.from_config(
        args.model, pretrained=True, checkpoint_path=args.checkpoint
    )
    model = model.to(device).eval()
    t_load = time.time() - t0
    print(f"[PE Spatial] Model loaded in {t_load:.1f}s")
    print(f"[PE Spatial] Image size: {model.image_size}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"[PE Spatial] Parameters: {n_params/1e9:.2f}B ({n_params_mb:.0f}MB)")

    # Load and preprocess image
    preprocess = transforms.get_image_transform(model.image_size)
    img = Image.open(args.image).convert("RGB")
    print(f"[PE Spatial] Input image: {img.size[0]}x{img.size[1]}")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Extract features  
    print(f"[PE Spatial] Extracting features...")
    t0 = time.time()
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        features = model.forward_features(img_tensor, strip_cls_token=True)
    t_feat = time.time() - t0
    print(f"[PE Spatial] Features shape: {features.shape} (took {t_feat:.2f}s)")
    # Expected: [1, N_tokens, D] where N_tokens = (img_size/patch_size)^2

    # Save raw features
    feat_np = features.cpu().float().numpy()
    np.save(os.path.join(args.output, "features.npy"), feat_np)
    print(f"[PE Spatial] Saved features.npy: {feat_np.shape}")

    # PCA visualization — map features to RGB
    print(f"[PE Spatial] Computing PCA visualization...")
    feat_2d = feat_np[0]  # [N_tokens, D]
    n_tokens = feat_2d.shape[0]
    grid_size = int(np.sqrt(n_tokens))
    
    # Center and PCA
    feat_centered = feat_2d - feat_2d.mean(axis=0)
    U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
    pca_3 = U[:, :3] * S[:3]  # project onto top-3 PCs
    
    # Normalize to [0, 255] for RGB
    pca_min = pca_3.min(axis=0)
    pca_max = pca_3.max(axis=0)
    pca_norm = ((pca_3 - pca_min) / (pca_max - pca_min + 1e-8) * 255).astype(np.uint8)
    pca_img = pca_norm.reshape(grid_size, grid_size, 3)
    
    # Upscale to original image size for comparison
    pca_pil = Image.fromarray(pca_img).resize(img.size, Image.NEAREST)
    pca_path = os.path.join(args.output, "pca_features.png")
    pca_pil.save(pca_path)
    print(f"[PE Spatial] Saved PCA visualization: {pca_path}")
    
    # Save original image for comparison
    img.save(os.path.join(args.output, "input_image.jpg"))
    
    print(f"\n[PE Spatial] ✅ Test complete!")
    print(f"  Input:    {args.image}")
    print(f"  Features: {feat_np.shape} ({feat_np.nbytes/1e6:.1f}MB)")
    print(f"  PCA vis:  {pca_path}")
    print(f"  Time:     load={t_load:.1f}s, inference={t_feat:.2f}s")
    print(f"  Device:   {device}")
    print(f"  VRAM:     {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == "cuda" else "  VRAM:     N/A (CPU)")

if __name__ == "__main__":
    main()
