#!/usr/bin/env python3
"""
Stray Scanner → DA3 format converter.

Converts Stray Scanner iPhone scan data to the exact format produced by
extract_da3_full.py, so that the pipeline detects the outputs and skips
the DA3 inference step entirely.

Stray Scanner data:
  - rgb.mp4            : RGB video (one frame per depth frame)
  - depth/NNNNNN.png   : uint16 depth maps in millimeters
  - confidence/        : uint8 confidence maps (0=low, 1=medium, 2=high)
  - odometry.csv       : timestamp, frame, x, y, z, qx, qy, qz, qw (c2w, ARKit)
  - camera_matrix.csv  : 3x3 intrinsics

Output (same format as extract_da3_full.py):
  da3_full/
  ├── da3_manifest.json
  ├── extrinsics.npy       [N, 3, 4] world-to-camera OpenCV
  ├── intrinsics.npy       [N, 3, 3]
  ├── {stem}_depth.npy     float32 meters
  ├── {stem}_conf.npy      float32 confidence
  └── processed_images/    RGB frames at depth resolution

Also outputs:
  frames/                  RGB frames at full resolution (for STAC pipeline)

Hernán Barreto — Ingerop IN3 Session IV — STAC
"""

import argparse
import os
import sys
import json
import csv
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_camera_matrix(csv_path: Path, depth_w: int, depth_h: int, mp4_path: Path) -> np.ndarray:
    """Read 3x3 camera intrinsics from Stray Scanner camera_matrix.csv.

    Stray Scanner provides intrinsics calibrated for the full RGB resolution.
    We scale them to match the LiDAR depth resolution (typically 256x192).
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            vals = [float(v.strip()) for v in line.strip().split(",")]
            if vals:
                rows.append(vals)
    K = np.array(rows[:3], dtype=np.float32)  # [3, 3] at full RGB resolution
    print(f"[Stray→DA3] Raw camera matrix (full RGB resolution):\n{K}")

    # Detect RGB resolution from video to compute scale factor
    cap = cv2.VideoCapture(str(mp4_path))
    rgb_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"[Stray→DA3] RGB resolution: {rgb_w}x{rgb_h}, Depth resolution: {depth_w}x{depth_h}")

    # Scale intrinsics proportionally to depth resolution
    sx = depth_w / rgb_w
    sy = depth_h / rgb_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[1, 2] *= sy  # cy
    print(f"[Stray→DA3] Scaled camera matrix for {depth_w}x{depth_h} (sx={sx:.4f}, sy={sy:.4f}):\n{K_scaled}")
    return K_scaled


def load_odometry(csv_path: Path) -> list:
    """
    Read odometry.csv from Stray Scanner.
    Columns: timestamp, frame, x, y, z, qx, qy, qz, qw
    Returns list of (frame_idx, x, y, z, qx, qy, qz, qw)
    Poses are camera-to-world in ARKit coordinate system.
    """
    poses = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values (Stray Scanner CSV has spaces after commas)
            row = {k.strip(): v.strip() for k, v in row.items()}
            poses.append({
                "frame": int(row["frame"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "qx": float(row["qx"]),
                "qy": float(row["qy"]),
                "qz": float(row["qz"]),
                "qw": float(row["qw"]),
            })
    return poses


def c2w_arkit_to_w2c(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """
    Convert ARKit c2w pose (x,y,z + quaternion) to w2c [3,4] matrix.

    ARKit poses work directly with standard pinhole backprojection
    (X-right, Y-down, Z-forward convention) — no coordinate flip needed.
    See: stray_scanner.py and StrayVisualizer reference implementation.
    """
    rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # [3,3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [x, y, z]

    # Invert to get w2c (no coordinate system conversion needed)
    w2c = np.linalg.inv(c2w)

    return w2c[:3, :].astype(np.float32)  # [3, 4]


def extract_rgb_frames(mp4_path: Path, depth_dir: Path, frames_out: Path, processed_out: Path,
                       depth_shape: tuple, frame_indices: list):
    """
    Extract RGB frames from mp4 at the exact indices matching depth frames.

    Stray Scanner guarantees 1:1 correspondence between video frames and depth files
    (numbered 000000, 000001, ...).
    """
    frames_out.mkdir(parents=True, exist_ok=True)
    processed_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Stray→DA3] Video has {total_video_frames} frames, extracting {len(frame_indices)} depth-aligned frames")

    depth_h, depth_w = depth_shape

    extracted = 0
    current_video_frame = 0

    for frame_idx in frame_indices:
        stem = f"{frame_idx:06d}"
        frames_dst = frames_out / f"{stem}.jpg"
        processed_dst = processed_out / f"{stem}.png"

        # Seek only if needed
        if current_video_frame != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_video_frame = frame_idx

        ret, frame = cap.read()
        if not ret:
            print(f"[Stray→DA3] ⚠️ Could not read video frame {frame_idx}, skipping")
            current_video_frame += 1
            continue

        current_video_frame += 1

        # Full-resolution frame for STAC frames/ dir
        if not frames_dst.exists():
            cv2.imwrite(str(frames_dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Resized to depth resolution for processed_images/
        if not processed_dst.exists():
            resized = cv2.resize(frame, (depth_w, depth_h), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(processed_dst), resized)

        extracted += 1
        if extracted % 100 == 0 or extracted == len(frame_indices):
            print(f"  Extracted {extracted}/{len(frame_indices)} frames")

    cap.release()
    return extracted


def convert_depth(depth_png_path: Path, out_npy_path: Path, conf_png_path: Path, out_conf_path: Path):
    """
    Convert Stray Scanner depth PNG (uint16 millimeters) to float32 meters numpy.
    Convert confidence PNG (uint8 0/1/2) to float32 normalized [0,1].
    """
    # Depth: Stray Scanner saves in millimeters as uint16
    depth_mm = cv2.imread(str(depth_png_path), cv2.IMREAD_ANYDEPTH)
    if depth_mm is None:
        print(f"[Stray→DA3] ⚠️ Could not read depth: {depth_png_path}")
        return False

    depth_m = depth_mm.astype(np.float32) / 1000.0  # mm → meters
    np.save(str(out_npy_path), depth_m)

    # Confidence: 0=low, 1=medium, 2=high → normalize to [0, 1]
    if conf_png_path.exists():
        conf_raw = cv2.imread(str(conf_png_path), cv2.IMREAD_GRAYSCALE)
        conf_f = conf_raw.astype(np.float32) / 2.0  # 0/1/2 → 0.0/0.5/1.0
    else:
        # No confidence: assume all valid
        conf_f = np.ones(depth_m.shape, dtype=np.float32)

    np.save(str(out_conf_path), conf_f)
    return True


def main():
    parser = argparse.ArgumentParser("Stray Scanner → DA3 format converter")
    parser.add_argument("--stray_dir", type=str, required=True,
                        help="Stray Scanner scan directory (contains rgb.mp4, depth/, odometry.csv, camera_matrix.csv)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Project src_default/ directory (frames/ and output/ will be created here)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Limit to first N frames (0 = all frames)")
    args = parser.parse_args()

    stray_dir = Path(args.stray_dir)
    output_dir = Path(args.output_dir)

    # ── Output directories ──
    frames_dir = output_dir / "frames"
    da3_dir = output_dir / "output" / "gaus_slam_run" / "da3_full"
    processed_dir = da3_dir / "processed_images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    da3_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load odometry ──
    odometry = load_odometry(stray_dir / "odometry.csv")
    print(f"[Stray→DA3] Loaded {len(odometry)} odometry entries")

    # ── Find depth frames ──
    mp4_path = stray_dir / "rgb.mp4"
    depth_dir = stray_dir / "depth"
    conf_dir = stray_dir / "confidence"
    depth_files = sorted(depth_dir.glob("*.png"))
    print(f"[Stray→DA3] Found {len(depth_files)} depth frames")

    # Build index: frame_number → depth path
    depth_by_idx = {int(p.stem): p for p in depth_files}
    odometry_by_idx = {p["frame"]: p for p in odometry}

    # Find frames that have both depth AND odometry
    valid_indices = sorted(set(depth_by_idx.keys()) & set(odometry_by_idx.keys()))
    print(f"[Stray→DA3] Frames with both depth + odometry: {len(valid_indices)}")

    if args.max_frames > 0:
        valid_indices = valid_indices[:args.max_frames]
        print(f"[Stray→DA3] Capped to {len(valid_indices)} frames (--max_frames={args.max_frames})")

    n_frames = len(valid_indices)

    # ── Get depth resolution from first frame ──
    first_depth = cv2.imread(str(depth_by_idx[valid_indices[0]]), cv2.IMREAD_ANYDEPTH)
    depth_h, depth_w = first_depth.shape
    print(f"[Stray→DA3] Depth resolution: {depth_w}×{depth_h}")

    # ── Load camera intrinsics (scaled from RGB resolution to depth resolution) ──
    K = load_camera_matrix(stray_dir / "camera_matrix.csv", depth_w, depth_h, mp4_path)

    # ── Extract RGB frames ──
    print(f"\n[Stray→DA3] Extracting RGB frames from {mp4_path}...")
    n_extracted = extract_rgb_frames(mp4_path, depth_dir, frames_dir, processed_dir,
                                     (depth_h, depth_w), valid_indices)
    print(f"[Stray→DA3] ✅ Extracted {n_extracted} RGB frames")


    # ── Convert depth + confidence ──
    print(f"\n[Stray→DA3] Converting depth and confidence maps...")
    frame_files = []
    extrinsics_list = []
    intrinsics_list = []

    for i, frame_idx in enumerate(valid_indices):
        stem = f"{frame_idx:06d}"
        depth_src = depth_by_idx[frame_idx]
        conf_src = conf_dir / f"{stem}.png"
        depth_npy = da3_dir / f"{stem}_depth.npy"
        conf_npy = da3_dir / f"{stem}_conf.npy"

        if not depth_npy.exists() or not conf_npy.exists():
            ok = convert_depth(depth_src, depth_npy, conf_src, conf_npy)
            if not ok:
                print(f"[Stray→DA3] ⚠️ Skipping frame {stem} (depth read failed)")
                continue

        # Pose: ARKit c2w → w2c [3,4] (no coordinate flip needed)
        odo = odometry_by_idx[frame_idx]
        w2c = c2w_arkit_to_w2c(odo["x"], odo["y"], odo["z"],
                                odo["qx"], odo["qy"], odo["qz"], odo["qw"])
        extrinsics_list.append(w2c)

        # Intrinsics: same for all frames from Stray Scanner
        intrinsics_list.append(K.copy())

        frame_files.append(f"{stem}.jpg")

        if (i + 1) % 100 == 0 or (i + 1) == n_frames:
            print(f"  Processed {i+1}/{n_frames} frames")

    # ── Save extrinsics and intrinsics ──
    extrinsics_arr = np.stack(extrinsics_list, axis=0)  # [N, 3, 4]
    intrinsics_arr = np.stack(intrinsics_list, axis=0)  # [N, 3, 3]
    np.save(str(da3_dir / "extrinsics.npy"), extrinsics_arr)
    np.save(str(da3_dir / "intrinsics.npy"), intrinsics_arr)
    print(f"\n[Stray→DA3] Saved extrinsics: {extrinsics_arr.shape}")
    print(f"[Stray→DA3] Saved intrinsics: {intrinsics_arr.shape}")

    # ── Save DA3 manifest ──
    manifest = {
        "frame_files": frame_files,
        "num_frames": len(frame_files),
        "depth_shape": [depth_h, depth_w],
        "source": "stray_scanner",
        "stray_dir": str(stray_dir),
    }
    with open(da3_dir / "da3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Stray→DA3] ✅ Conversion complete!")
    print(f"  frames/           → {len(frame_files)} RGB JPGs")
    print(f"  da3_full/         → {len(frame_files)} depth + conf NPYs")
    print(f"  da3_full/extrinsics.npy   → {extrinsics_arr.shape}")
    print(f"  da3_full/intrinsics.npy   → {intrinsics_arr.shape}")
    print(f"  Depth resolution: {depth_w}×{depth_h}")
    print(f"\nReady to run GauS-SLAM pipeline — DA3 step will be skipped automatically.")


if __name__ == "__main__":
    main()
