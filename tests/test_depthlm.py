#!/usr/bin/env python3
"""
Test 3: DepthLM Pixtral 12B — Metric Depth Estimation
Predicts per-pixel metric depth from a construction frame.

Usage:
  conda activate depthlm
  cd /home/hernan/stac-builder
  python tests/test_depthlm.py --image server/data/projects/test1/scans/2026-02-24/src_legacy/frames/00045.jpg

Note: DepthLM (12B params) needs ~24GB. On CPU this takes 5-15 min per image.
You must first install: pip install transformers accelerate
"""
import argparse
import sys
import time
import os

import torch
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Test DepthLM metric depth estimation")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default="facebook/DepthLM",
                       help="HuggingFace model path")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: cuda or cpu. DepthLM needs ~24GB, use cpu for 6GB GPU")
    parser.add_argument("--output", type=str, default="/tmp/depthlm_test",
                       help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"[DepthLM] Using device: {args.device}")
    print(f"[DepthLM] ⚠️  Pixtral 12B needs ~24GB. On CPU this will take 5-15 min.")

    from transformers import AutoProcessor, LlavaForConditionalGeneration

    # Load model
    print(f"[DepthLM] Loading {args.model}...")
    t0 = time.time()
    
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()
    
    t_load = time.time() - t0
    print(f"[DepthLM] Model loaded in {t_load:.1f}s")

    # Load image  
    img = Image.open(args.image).convert("RGB")
    print(f"[DepthLM] Input image: {img.size[0]}x{img.size[1]}")

    # DepthLM uses a special prompt for depth estimation
    # Based on the eval.py in the repo
    prompt = "<s>[INST][IMG]Estimate the depth map of this image. For each pixel, provide the metric depth in meters.[/INST]"
    
    print(f"[DepthLM] Running depth estimation...")
    t0 = time.time()
    
    inputs = processor(
        text=prompt,
        images=[img],
        return_tensors="pt",
    )
    if args.device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
    
    t_gen = time.time() - t0
    
    # Decode output 
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"[DepthLM] Response ({t_gen:.1f}s):")
    print(f"  {response[:500]}...")
    
    # Save results
    output_path = os.path.join(args.output, "depthlm_response.txt")
    with open(output_path, 'w') as f:
        f.write(response)
    
    # Save input image
    img.save(os.path.join(args.output, "input_image.jpg"))
    
    print(f"\n[DepthLM] ✅ Test complete!")
    print(f"  Input:    {args.image}")
    print(f"  Response: {output_path}")
    print(f"  Time:     load={t_load:.1f}s, inference={t_gen:.1f}s")
    print(f"  Device:   {args.device}")

if __name__ == "__main__":
    main()
