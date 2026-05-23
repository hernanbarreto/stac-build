#!/usr/bin/env python3
"""
DA3 Hybrid/LiDAR Runner — Production entry point.

Invoked by run_da3_hybrid.sh. Uses StrayDA3Streaming (hybrid) or
StrayLiDAROnly (lidar) to process Stray Scanner sessions.

Usage:
    python run_da3_hybrid_main.py \
        --mode hybrid \
        --image_dir /path/to/frames \
        --data_dir /path/to/stray_scanner_data \
        --config /path/to/da3_config.yaml \
        --output_dir /path/to/output

Authors: Hernán Barreto — Ingerop IN3
"""
import os
import sys
import gc
import argparse
from pathlib import Path

import torch

# ── Path setup ──
server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))


def main():
    parser = argparse.ArgumentParser(description="DA3 Hybrid/LiDAR Streaming")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["hybrid", "lidar"],
                        help="Reconstruction mode: 'hybrid' or 'lidar'")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with RGB frames")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Stray Scanner raw data directory")
    parser.add_argument("--config", type=str, required=True,
                        help="DA3 streaming config YAML")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--selected_frames", type=str, default=None,
                        help="Path to selected_frames.json")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stray Scanner frame stride")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames (0=all)")
    parser.add_argument("--confidence_threshold", type=int, default=0,
                        help="Stray Scanner confidence threshold (0-2)")
    parser.add_argument("--lidar_trust_range", type=float, default=5.0,
                        help="Max LiDAR depth in meters")
    args = parser.parse_args()

    # ── Load DA3 config ──
    from da3_streaming import load_config
    config = load_config(args.config)

    # ── Prepare Stray Scanner data ──
    from ingestors.stray_scanner import prepare_stray_data

    print(f"[DA3-{args.mode.upper()}] Loading Stray Scanner data from {args.data_dir}")
    stray_data = prepare_stray_data(
        data_dir=args.data_dir,
        frames_output_dir=args.image_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
        selected_frames_path=args.selected_frames,
    )
    print(f"[DA3-{args.mode.upper()}] {len(stray_data['frame_paths'])} frames prepared")

    # ── Create output dir ──
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # ── Instantiate appropriate subclass ──
    lidar_cfg = {"trust_range": args.lidar_trust_range}

    if args.mode == "hybrid":
        from stray_da3_streaming import StrayDA3Streaming
        streamer = StrayDA3Streaming(
            args.image_dir, save_dir, config,
            stray_data=stray_data,
            lidar_cfg=lidar_cfg,
        )
    else:  # lidar
        from stray_da3_streaming import StrayLiDAROnly
        streamer = StrayLiDAROnly(
            args.image_dir, save_dir, config,
            stray_data=stray_data,
            lidar_cfg=lidar_cfg,
        )

    # ── Run pipeline ──
    streamer.run(selected_frames_path=args.selected_frames)
    streamer.close()

    del streamer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── Merge chunk PLYs ──
    from loop_utils.sim3utils import merge_ply_files
    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print(f"DA3-{args.mode.upper()} done.")


if __name__ == "__main__":
    main()
