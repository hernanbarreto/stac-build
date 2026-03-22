#!/usr/bin/env python3
"""
Test 2: PLM-8B — Perception Language Model
Scene analysis on a construction frame: material ID, element detection, state classification.

PLM-8B uses PE-Lang-G14 as visual encoder + Llama 3.1 8B as language decoder.
The repo's generate.py hardcodes .cuda(), so we patch it for CPU support.

Usage:
  conda activate pe_spatial
  cd /home/hernan/stac-builder

  # GPU (needs ~16GB VRAM):
  python tests/test_plm.py --image "tests/obra 1.jpeg" --device cuda

  # CPU (needs ~16GB RAM, takes 5-15 min per query):
  python tests/test_plm.py --image "tests/obra 1.jpeg" --device cpu
"""
import argparse
import sys
import time
import os
import json

# Add perception_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'perception_models'))

import torch
from PIL import Image


def load_model_with_device(ckpt, device="cpu"):
    """
    Load PLM model with configurable device.
    Based on generate.py's load_consolidated_model_and_tokenizer but with
    device support instead of hardcoded .cuda().
    """
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from core.args import dataclass_from_dict
    from core.checkpoint import load_consolidated_checkpoint
    from apps.plm.tokenizer import build_tokenizer
    from apps.plm.transformer import LMTransformer, LMTransformerArgs

    # Download from HF if needed
    if os.path.exists(ckpt):
        ckpt_path = ckpt
    else:
        print(f"[PLM-8B] Downloading {ckpt} from HuggingFace Hub...")
        ckpt_path = snapshot_download(ckpt)
        ckpt_path = os.path.join(ckpt_path, "original")
        print(f"[PLM-8B] Downloaded to: {ckpt_path}")

    # Load config
    config = OmegaConf.load(os.path.join(ckpt_path, "params.json"))

    # Build tokenizer
    tokenizer_path = config.data.tokenizer_path
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(ckpt_path, tokenizer_path)
    tokenizer = build_tokenizer(
        config.data.tokenizer_name,
        tokenizer_path,
        pooling_ratio=config.model.pooling_ratio,
        patch_size=config.model.vision_model.patch_size,
    )

    # Build model
    model_args = dataclass_from_dict(LMTransformerArgs, config.model, strict=False)
    model = LMTransformer(model_args)
    load_consolidated_checkpoint(model, ckpt_path)

    # Use configurable device and dtype
    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    if device == "cpu":
        # CPU doesn't support bf16 well on all hardware, use fp32
        param_dtype = torch.float32
    
    model = model.to(device).eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Test PLM-8B scene analysis")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--ckpt", type=str, default="facebook/Perception-LM-8B",
                       help="Checkpoint path (HF repo or local)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: cuda (needs ~16GB VRAM) or cpu (slow but works)")
    parser.add_argument("--question", type=str, 
                       default="Describe this construction scene in detail. Identify all visible materials, equipment, workers, and the construction activity being performed. What is the current state of construction?",
                       help="Question to ask about the image")
    parser.add_argument("--output", type=str, default="/tmp/plm_test",
                       help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"[PLM-8B] Device: {args.device}")
    if args.device == "cpu":
        print(f"[PLM-8B] ⚠️  Running on CPU — expect 5-15 min per query (8B params in fp32)")

    # Load model
    print(f"[PLM-8B] Loading model from {args.ckpt}...")
    t0 = time.time()
    model, tokenizer, config = load_model_with_device(args.ckpt, device=args.device)
    t_load = time.time() - t0
    print(f"[PLM-8B] Model loaded in {t_load:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PLM-8B] Parameters: {n_params/1e9:.2f}B")

    # Prepare image
    from core.transforms.image_transform import get_image_transform
    transform = get_image_transform(
        vision_input_type=config.data.vision_input_type,
        image_res=model.vision_model.image_size,
        max_num_tiles=config.data.max_num_tiles,
    )
    
    img = Image.open(args.image).convert("RGB")
    print(f"[PLM-8B] Input image: {img.size[0]}x{img.size[1]}")
    image_tensor, _ = transform(img)

    # Create generator
    from apps.plm.generate import (
        PackedCausalTransformerGenerator,
        PackedCausalTransformerGeneratorArgs,
    )
    from core.args import dataclass_from_dict

    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, 
        {"device": args.device, "dtype": "fp32" if args.device == "cpu" else "bf16"},
        strict=False
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Run generation
    prompts = [(args.question, image_tensor)]
    
    print(f"\n[PLM-8B] Question: {args.question}")
    print(f"[PLM-8B] Generating response...")
    t0 = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    t_gen = time.time() - t0

    response = generation[0] if generation else "No response generated"
    print(f"\n[PLM-8B] Response ({t_gen:.1f}s):")
    print(f"  {response}")

    # Save results
    result = {
        "image": args.image,
        "question": args.question,
        "response": response,
        "time_load_s": t_load,
        "time_generate_s": t_gen,
        "device": args.device,
        "params_B": n_params / 1e9,
    }
    output_path = os.path.join(args.output, "plm_results.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[PLM-8B] ✅ Test complete!")
    print(f"  Results: {output_path}")
    print(f"  Time:    load={t_load:.1f}s, generate={t_gen:.1f}s")

if __name__ == "__main__":
    main()
