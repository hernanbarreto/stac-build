#!/usr/bin/env python3
"""
Test: DA3-streaming with Stray Scanner hybrid injection.

If Stray Scanner data is found in --data_dir, injects ARKit poses + LiDAR depth.
Otherwise, runs pure DA3-streaming on the images.

Usage:
    cd /home/hernan/stac-builder/server
    bash tests/run_da3_hybrid.sh --max_frames 40
"""
import os
import sys
import gc
import argparse
import time
import numpy as np
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
server_dir = tests_dir.parent
da3_streaming_dir = server_dir.parent / "vendor" / "depth-anything-3" / "da3_streaming"

if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))
if str(da3_streaming_dir) not in sys.path:
    sys.path.insert(0, str(da3_streaming_dir))


def main():
    parser = argparse.ArgumentParser(
        description="DA3-streaming + Stray Scanner hybrid")
    parser.add_argument("--data_dir", type=str, default="./test2",
                        help="Stray Scanner data directory (or image directory)")
    parser.add_argument("--output_dir", type=str, default="./test2_da3_hybrid")
    parser.add_argument("--config", type=str, default=None,
                        help="DA3-streaming config YAML")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"

    # ── Auto-detect Stray Scanner data ──
    stray_data = None
    data_path = Path(args.data_dir)
    has_stray = (
        (data_path / "odometry.csv").exists() or
        (data_path / "camera_matrix.csv").exists() or
        (data_path / "depth").exists()
    )

    if has_stray:
        print("=" * 60)
        print("Stray Scanner data detected → HYBRID mode")
        print("  Poses: ARKit (deterministic)")
        print("  Depth: LiDAR + DA3 calibrated")
        print("=" * 60)

        from ingestors.stray_scanner import prepare_stray_data
        stray_data = prepare_stray_data(
            data_dir=args.data_dir,
            frames_output_dir=str(frames_dir),
            stride=args.stride,
            max_frames=args.max_frames,
            confidence_threshold=1,
        )
        print(f"  Frames: {len(stray_data['frame_paths'])}")
        image_dir = str(frames_dir)
    else:
        print("=" * 60)
        print("No Stray Scanner data → PURE DA3-streaming mode")
        print("=" * 60)
        image_dir = args.data_dir

    # ── Load config ──
    from loop_utils.config_utils import load_config

    if args.config:
        config_path = args.config
    else:
        config_path = str(da3_streaming_dir / "configs" / "stac_da3_streaming.yaml")

    config = load_config(config_path)
    print(f"Config: {config_path}")
    print(f"  chunk_size={config['Model']['chunk_size']}, "
          f"overlap={config['Model']['overlap']}, "
          f"device={config['Model'].get('device', 'auto')}")

    # ── Warmup numba if needed ──
    if config["Model"]["align_lib"] == "numba":
        from loop_utils.sim3utils import warmup_numba
        warmup_numba()

    # ── Run pipeline ──
    from stray_da3_streaming import StrayDA3Streaming

    prev_cwd = os.getcwd()
    os.chdir(str(da3_streaming_dir))  # DA3 needs relative paths for weights

    t0 = time.time()
    pipeline = StrayDA3Streaming(
        image_dir=image_dir,
        save_dir=str(out / "da3_output"),
        config=config,
        stray_data=stray_data,
    )
    pipeline.run()
    pipeline.close()

    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.1f}s")

    os.chdir(prev_cwd)

    # ── Merge PLYs ──
    from loop_utils.sim3utils import merge_ply_files

    pcd_dir = str(out / "da3_output" / "pcd")
    combined_ply = str(out / "da3_output" / "pcd" / "combined_pcd.ply")
    merge_ply_files(pcd_dir, combined_ply)

    # ── Copy to Desktop ──
    import shutil
    desktop = "/mnt/c/Users/ingerop/Desktop/hybrid_da3.ply"
    if os.path.exists(combined_ply):
        shutil.copy2(combined_ply, desktop)
        print(f"✅ Copied → {desktop}")
    else:
        print("⚠️  No combined PLY generated")

    # Cleanup
    del pipeline
    gc.collect()

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
