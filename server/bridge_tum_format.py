#!/usr/bin/env python3
"""
TUM-RGBD Format Bridge — converts DA3 full output to TUM-RGBD directory structure.

Reads DA3 extract_da3_full.py output (metric depth .npy, extrinsics, intrinsics)
and produces a directory structure compatible with GauS-SLAM's TUM dataset loader:

  tum_bridge/
  ├── rgb.txt              # timestamp rgb/filename.png
  ├── depth.txt            # timestamp depth/filename.png
  ├── groundtruth.txt      # timestamp tx ty tz qx qy qz qw
  ├── rgb/                 # symlinks or copies of original frames
  │   ├── 0.000000.png
  │   └── ...
  └── depth/               # 16-bit PNG depth maps (value / scale = meters)
      ├── 0.000000.png
      └── ...

Hernán Barreto — Ingerop IN3 Session IV — STAC
"""
import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


def w2c_to_tum_pose(w2c_34: np.ndarray) -> str:
    """Convert a [3, 4] w2c extrinsic to TUM format: tx ty tz qx qy qz qw.

    TUM ground truth uses camera-to-world (c2w) translation and quaternion.
    DA3 extrinsics are world-to-camera (w2c) in OpenCV convention.
    """
    # Build 4x4
    w2c = np.eye(4)
    w2c[:3, :] = w2c_34

    # Invert to get c2w
    c2w = np.linalg.inv(w2c)

    # Extract translation
    tx, ty, tz = c2w[:3, 3]

    # Extract quaternion (scipy returns [x, y, z, w])
    quat = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [qx, qy, qz, qw]
    qx, qy, qz, qw = quat

    return f"{tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"


def main():
    parser = argparse.ArgumentParser("DA3 → TUM-RGBD Format Bridge")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Original frames directory (jpg/png)")
    parser.add_argument("--da3_dir", type=str, required=True,
                        help="DA3 full output directory (from extract_da3_full.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output TUM-RGBD directory")
    parser.add_argument("--png_depth_scale", type=float, default=5000.0,
                        help="TUM depth scale factor (depth_meters = pixel_value / scale)")
    parser.add_argument("--max_depth", type=float, default=65.0,
                        help="Maximum depth in meters to encode (prevents uint16 overflow)")
    parser.add_argument("--use_processed_images", action="store_true",
                        help="Use DA3 processed images instead of originals (same resolution)")
    parser.add_argument("--selected_frames", type=str, default=None,
                        help="Path to selected_frames JSON (blur filter). Only those frames will be bridged.")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Temporal subsampling: take 1 every N frames (default=1 = all frames).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    da3_dir = Path(args.da3_dir)
    frames_dir = Path(args.frames_dir)

    # Create output structure
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # ── Load DA3 manifest ──
    manifest_path = da3_dir / "da3_manifest.json"
    if not manifest_path.exists():
        print(f"[TUM Bridge] ❌ da3_manifest.json not found in {da3_dir}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    frame_files = manifest["frame_files"]

    # ── Apply selected_frames filter if provided (blur filter) ──
    if args.selected_frames and os.path.exists(args.selected_frames):
        with open(args.selected_frames) as f:
            sf = json.load(f)
        selected_set = set(sf.get("selected", sf.get("selected_files", [])))
        if selected_set:
            original_count = len(frame_files)
            frame_files = [f for f in frame_files if f in selected_set]
            print(f"[TUM Bridge] Selected frames filter: {len(frame_files)}/{original_count} frames")

    # ── Apply temporal subsampling (frame_step) ──
    if args.frame_step > 1:
        original_count = len(frame_files)
        frame_files = frame_files[::args.frame_step]
        print(f"[TUM Bridge] Frame step {args.frame_step}: {len(frame_files)}/{original_count} frames after subsampling")

    n_frames = len(frame_files)
    print(f"[TUM Bridge] Processing {n_frames} frames")

    # ── Load extrinsics and intrinsics ──
    extrinsics = np.load(da3_dir / "extrinsics.npy")  # [N, 3, 4]
    intrinsics = np.load(da3_dir / "intrinsics.npy")  # [N, 3, 3]

    print(f"[TUM Bridge] Extrinsics: {extrinsics.shape}, Intrinsics: {intrinsics.shape}")

    # Build filename → original manifest index map.
    # CRITICAL: after selected_frames + frame_step filtering, frame_files[i]
    # is NOT extrinsics[i]. E.g. frame_files[1]="000006.jpg" but extrinsics[1]
    # is the pose of frame 1, not frame 6. We must look up the original index.
    all_manifest_frames = manifest["frame_files"]
    frame_to_orig_idx = {f: idx for idx, f in enumerate(all_manifest_frames)}

    # ── Generate timestamps (1-indexed, 1.0 second apart) ──
    timestamps = [float(i + 1) for i in range(n_frames)]

    # ── Process each frame ──
    rgb_entries = []
    depth_entries = []
    gt_entries = []

    for i, frame_file in enumerate(frame_files):
        stem = os.path.splitext(frame_file)[0]
        ts = timestamps[i]
        ts_str = f"{ts:.6f}"

        # --- RGB ---
        rgb_filename = f"{ts_str}.png"
        if args.use_processed_images:
            src_rgb = da3_dir / "processed_images" / f"{stem}.png"
        else:
            src_rgb = frames_dir / frame_file

        dst_rgb = rgb_dir / rgb_filename

        if src_rgb.exists() and not dst_rgb.exists():
            # Read and re-save as PNG (ensures consistent format)
            img = cv2.imread(str(src_rgb))
            if img is not None:
                cv2.imwrite(str(dst_rgb), img)

        rgb_entries.append(f"{ts_str} rgb/{rgb_filename}")

        # --- Depth ---
        depth_npy = da3_dir / f"{stem}_depth.npy"
        depth_filename = f"{ts_str}.png"
        dst_depth = depth_dir / depth_filename

        if depth_npy.exists() and not dst_depth.exists():
            depth_meters = np.load(depth_npy)  # float32, metric meters

            # Clamp to max depth (prevent uint16 overflow)
            max_encodable = 65535.0 / args.png_depth_scale
            depth_clamped = np.clip(depth_meters, 0, min(args.max_depth, max_encodable))

            # Convert to 16-bit PNG: pixel_value = depth_meters * scale
            depth_16bit = (depth_clamped * args.png_depth_scale).astype(np.uint16)
            cv2.imwrite(str(dst_depth), depth_16bit)

        depth_entries.append(f"{ts_str} depth/{depth_filename}")

        # --- Pose (ground truth) ---
        orig_idx = frame_to_orig_idx.get(frame_file, i)
        pose_str = w2c_to_tum_pose(extrinsics[orig_idx])
        gt_entries.append(f"{ts_str} {pose_str}")

        if (i + 1) % 50 == 0 or i == n_frames - 1:
            print(f"  [{i+1}/{n_frames}] {frame_file} → ts={ts_str}")

    # ── Write association files ──
    # rgb.txt
    with open(output_dir / "rgb.txt", "w") as f:
        f.write("# TUM RGB association file (generated by STAC-Builder bridge_tum_format.py)\n")
        f.write("# timestamp filename\n")
        for entry in rgb_entries:
            f.write(entry + "\n")

    # depth.txt
    with open(output_dir / "depth.txt", "w") as f:
        f.write("# TUM Depth association file (generated by STAC-Builder bridge_tum_format.py)\n")
        f.write("# timestamp filename\n")
        for entry in depth_entries:
            f.write(entry + "\n")

    # groundtruth.txt
    with open(output_dir / "groundtruth.txt", "w") as f:
        f.write("# TUM ground truth (DA3 estimated poses, c2w)\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for entry in gt_entries:
            f.write(entry + "\n")

    # ── Write intrinsics YAML for GauS-SLAM ──
    K = intrinsics[0]  # Use first frame intrinsics (all should be identical)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Read image dimensions from first processed depth
    depth_shape = manifest["depth_shape"]  # [H, W]
    img_h, img_w = depth_shape[0], depth_shape[1]

    intrinsics_yaml = {
        "dataset_name": "tum",
        "camera_params": {
            "image_height": img_h,
            "image_width": img_w,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "png_depth_scale": args.png_depth_scale,
            "crop_edge": 0,
        }
    }

    import yaml
    intrinsics_yaml_path = output_dir / "camera_intrinsics.yaml"
    with open(intrinsics_yaml_path, "w") as f:
        yaml.dump(intrinsics_yaml, f, default_flow_style=False)

    print(f"\n[TUM Bridge] ✅ Generated TUM-RGBD structure:")
    print(f"  📁 {output_dir}")
    print(f"  ├── rgb.txt          ({n_frames} entries)")
    print(f"  ├── depth.txt        ({n_frames} entries)")
    print(f"  ├── groundtruth.txt  ({n_frames} poses)")
    print(f"  ├── camera_intrinsics.yaml (fx={fx:.1f}, fy={fy:.1f})")
    print(f"  ├── rgb/             ({n_frames} PNGs)")
    print(f"  └── depth/           ({n_frames} 16-bit PNGs, scale={args.png_depth_scale})")
    print(f"  Image size: {img_w}×{img_h}")

    return True


if __name__ == "__main__":
    main()
