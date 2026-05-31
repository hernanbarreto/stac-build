#!/usr/bin/env python3
"""
DA3 (pure) Runner — STAC entry point.

Invoked by run_da3.sh. Mirrors the vendored da3_streaming.py main(), but goes
through the STAC subclass StrayDA3Streaming (with stray_data=None, which behaves
exactly like DA3_Streaming) so the keyframe filter lives in our code instead of
patching the gitignored vendor. StrayDA3Streaming.run() honors --selected_frames.

Usage:
    python run_da3_main.py \
        --image_dir /path/to/frames \
        --config /path/to/da3_config.yaml \
        --output_dir /path/to/output \
        [--selected_frames /path/to/selected_frames.json]

Authors: Hernán Barreto — Ingerop IN3
"""
import os
import sys
import gc
import argparse
from datetime import datetime
from pathlib import Path

import torch

# ── Path setup (server dir on sys.path so we can import the STAC subclass) ──
server_dir = Path(__file__).resolve().parent
if str(server_dir) not in sys.path:
    sys.path.insert(0, str(server_dir))


def main():
    parser = argparse.ArgumentParser(description="DA3 Streaming (pure, STAC entry)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory with RGB frames")
    parser.add_argument("--config", type=str, required=False,
                        default="./configs/base_config.yaml",
                        help="DA3 streaming config YAML")
    parser.add_argument("--output_dir", type=str, required=False, default=None,
                        help="Output directory")
    parser.add_argument("--selected_frames", type=str, required=False, default=None,
                        help="Optional path to selected_frames.json (keyframe filter)")
    args = parser.parse_args()

    # da3_streaming re-exports load_config / copy_file; subclass lives in server/
    from da3_streaming import load_config, copy_file, warmup_numba
    from loop_utils.sim3utils import merge_ply_files
    from stray_da3_streaming import StrayDA3Streaming

    config = load_config(args.config)

    image_dir = args.image_dir
    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = os.path.join("./exps", image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    # stray_data=None → pure DA3 behaviour; run() applies the keyframe filter.
    da3_streaming = StrayDA3Streaming(image_dir, save_dir, config, stray_data=None)
    da3_streaming.run(selected_frames_path=args.selected_frames)
    da3_streaming.close()

    del da3_streaming
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Lean prior-source mode: we only need depth + poses (read directly from
    # da3_run), so skip merging the per-chunk PLYs into combined_pcd.ply.
    if os.environ.get("DA3_LEAN_PRIORS") == "1":
        print("DA3-Streaming done (lean priors: skipped combined-PLY merge).")
    else:
        all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
        input_dir = os.path.join(save_dir, "pcd")
        print("Saving all the point clouds")
        merge_ply_files(input_dir, all_ply_path)
        print("DA3-Streaming done.")


if __name__ == "__main__":
    main()
