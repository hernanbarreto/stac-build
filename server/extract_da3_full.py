#!/usr/bin/env python3
"""
DA3 Full Extraction — depth (metric) + poses (extrinsics) + intrinsics.

Runs DA3 NESTED-GIANT-LARGE-1.1 inference on a directory of frames and saves
ALL prediction outputs needed by the GauS-SLAM pipeline:

  - {stem}_depth.npy   : float32 [H, W]  — metric depth in meters
  - {stem}_conf.npy    : float32 [H, W]  — confidence (expp1 − 1)
  - extrinsics.npy     : float32 [N, 3, 4] — world-to-camera (OpenCV/COLMAP)
  - intrinsics.npy     : float32 [N, 3, 3] — camera intrinsics (fx, fy, cx, cy)
  - processed_images/  : directory of processed RGB PNGs at DA3 resolution

Designed to run in the `da3` conda environment.

Hernán Barreto — Ingerop IN3 Session IV — STAC
"""
import argparse
import os
import sys
import glob
import json
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser("DA3 Full Extraction (depth + poses + intrinsics)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing input frames (jpg/png)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for depth/conf/poses")
    parser.add_argument("--model", type=str,
                        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        help="HuggingFace model ID")
    parser.add_argument("--selected_frames", type=str, default=None,
                        help="Path to selected_frames.json (optional filter)")
    parser.add_argument("--process_res", type=int, default=504,
                        help="DA3 processing resolution")
    parser.add_argument("--use_ray_pose", action="store_true",
                        help="Use ray-based pose estimation (slower but more accurate)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto' (auto-detect)")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Process frames in chunks of N to avoid OOM (0 = all at once)")
    parser.add_argument("--per_frame", action="store_true",
                        help="Process one frame at a time (hybrid mode: only depth needed, poses discarded)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Gather image paths ──
    images = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg")) +
        glob.glob(os.path.join(args.image_dir, "*.png"))
    )

    # Apply selected_frames filter if provided
    if args.selected_frames and os.path.exists(args.selected_frames):
        with open(args.selected_frames, "r") as f:
            sf = json.load(f)
        selected_set = set(sf.get("selected", sf.get("selected_files", [])))
        if selected_set:
            images = [p for p in images if os.path.basename(p) in selected_set]
            print(f"[DA3 Full] Filtered to {len(images)} selected frames")

    if not images:
        print("[DA3 Full] No images found. Exiting.")
        return

    # ── Check if all outputs already exist ──
    extrinsics_path = os.path.join(args.output_dir, "extrinsics.npy")
    intrinsics_path = os.path.join(args.output_dir, "intrinsics.npy")

    all_exist = os.path.exists(extrinsics_path) and os.path.exists(intrinsics_path)
    if all_exist:
        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if (not os.path.exists(os.path.join(args.output_dir, stem + "_depth.npy")) or
                    not os.path.exists(os.path.join(args.output_dir, stem + "_conf.npy"))):
                all_exist = False
                break

    if all_exist:
        print(f"[DA3 Full] All outputs for {len(images)} frames already exist. Skipping.")
        return

    print(f"[DA3 Full] Processing {len(images)} frames with {args.model}")

    # ── Load model ──
    da3_src = os.path.join(os.path.dirname(__file__), "../vendor/depth-anything-3/src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)

    from depth_anything_3.api import DepthAnything3
    import gc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[DA3 Full] Loading model on {device}")
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device=device)
    model.eval()

    # ── Choose processing strategy ──
    if args.per_frame:
        # Per-frame mode: process 1 image at a time (for hybrid — only depth needed)
        # With N=1, DA3 NESTED runs monocularly (~6GB RAM vs ~16GB for N=250)
        # Poses are garbage with N=1 but hybrid overwrites them with ARKit anyway
        all_extrinsics, all_intrinsics = _run_per_frame(
            model, images, args, device
        )
    else:
        # Chunk mode: process N frames together (for gaus_slam_da3 — needs multi-view poses)
        all_extrinsics, all_intrinsics = _run_chunked(
            model, images, args, device
        )

    # ── Save global extrinsics + intrinsics ──
    extrinsics_path = os.path.join(args.output_dir, "extrinsics.npy")
    intrinsics_path = os.path.join(args.output_dir, "intrinsics.npy")

    print(f"\n[DA3 Full] Final extrinsics shape: {all_extrinsics.shape}")
    print(f"[DA3 Full] Final intrinsics shape: {all_intrinsics.shape}")

    np.save(extrinsics_path, all_extrinsics)
    np.save(intrinsics_path, all_intrinsics)
    print(f"[DA3 Full] Saved extrinsics: {extrinsics_path}")
    print(f"[DA3 Full] Saved intrinsics: {intrinsics_path}")

    # ── Save frame list manifest ──
    depth_shape = list(all_extrinsics.shape[1:])  # fallback shape
    # Try to get actual depth shape from first saved file
    first_stem = os.path.splitext(os.path.basename(images[0]))[0]
    first_depth = os.path.join(args.output_dir, first_stem + "_depth.npy")
    if os.path.exists(first_depth):
        depth_shape = list(np.load(first_depth).shape)

    manifest = {
        "model": args.model,
        "process_res": args.process_res,
        "use_ray_pose": args.use_ray_pose,
        "per_frame": args.per_frame,
        "is_metric": 1,
        "num_frames": len(images),
        "depth_shape": depth_shape,
        "frame_files": [os.path.basename(p) for p in images],
    }
    manifest_path = os.path.join(args.output_dir, "da3_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[DA3 Full] Saved manifest: {manifest_path}")

    print("[DA3 Full] ✅ Finished successfully.")


def _run_per_frame(model, images, args, device):
    """Process one frame at a time — for hybrid mode where only depth is needed.

    With N=1, DA3 NESTED runs in monocular mode (~6GB RAM).
    Poses are identity (useless with 1 frame) but hybrid overwrites with ARKit.
    """
    import gc
    import cv2

    n = len(images)
    print(f"[DA3 Full] ⚡ Per-frame mode: processing {n} frames one at a time")
    print(f"[DA3 Full] (Poses will be identity — use ARKit for hybrid mode)")

    all_extrinsics = []
    all_intrinsics = []
    proc_dir = os.path.join(args.output_dir, "processed_images")
    os.makedirs(proc_dir, exist_ok=True)

    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")

        # Skip if already exists (resume support)
        if os.path.exists(depth_path) and os.path.exists(conf_path):
            # Still need extrinsics/intrinsics for the array — load from a dummy
            # calibrate_depth_hybrid.py will overwrite these anyway
            all_extrinsics.append(np.eye(3, 4, dtype=np.float32)[None])
            all_intrinsics.append(np.eye(3, dtype=np.float32)[None])
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{n}] {os.path.basename(img_path)} — already exists, skipping")
            continue

        # Run inference on single frame
        with torch.no_grad():
            prediction = model.inference(
                [img_path],
                process_res=args.process_res,
                use_ray_pose=False,  # pointless with 1 frame
            )

        # Extract depth + conf
        depth = prediction.depth
        conf = prediction.conf
        ext = prediction.extrinsics
        ixt = prediction.intrinsics

        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()
        if isinstance(ext, torch.Tensor):
            ext = ext.cpu().numpy()
        if isinstance(ixt, torch.Tensor):
            ixt = ixt.cpu().numpy()

        # DA3 conf: expp1 activation → subtract 1
        conf = conf - 1.0
        conf = np.clip(conf, 0, None)

        # Save depth + conf
        np.save(depth_path, depth[0])
        np.save(conf_path, conf[0])

        all_extrinsics.append(ext)
        all_intrinsics.append(ixt)

        # Save processed image
        if prediction.processed_images is not None:
            out_path = os.path.join(proc_dir, stem + ".png")
            rgb = prediction.processed_images[0]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, bgr)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] {os.path.basename(img_path)} "
                  f"→ depth [{depth[0].min():.2f}, {depth[0].max():.2f}]m")

        # Free memory
        del prediction, depth, conf, ext, ixt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_extrinsics, axis=0), np.concatenate(all_intrinsics, axis=0)


def _run_chunked(model, images, args, device):
    """Process frames in multi-frame chunks — for gaus_slam_da3 where poses are needed."""
    import gc
    import cv2

    chunk_size = args.chunk_size if args.chunk_size > 0 else len(images)
    num_chunks = (len(images) + chunk_size - 1) // chunk_size
    print(f"[DA3 Full] Processing in {num_chunks} chunk(s) of up to {chunk_size} frames")

    all_extrinsics = []
    all_intrinsics = []
    saved_count = 0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(images))
        chunk_images = images[start:end]

        print(f"\n[DA3 Full] ── Chunk {chunk_idx+1}/{num_chunks}: frames {start+1}-{end} ({len(chunk_images)} frames) ──")

        print(f"[DA3 Full] Running inference (process_res={args.process_res}, use_ray_pose={args.use_ray_pose})")
        with torch.no_grad():
            prediction = model.inference(
                chunk_images,
                process_res=args.process_res,
                use_ray_pose=args.use_ray_pose,
            )

        depths = prediction.depth
        confs = prediction.conf
        extrinsics = prediction.extrinsics
        intrinsics = prediction.intrinsics

        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        if isinstance(confs, torch.Tensor):
            confs = confs.cpu().numpy()
        if isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()

        confs = confs - 1.0
        confs = np.clip(confs, 0, None)

        print(f"[DA3 Full] Chunk depth range: [{depths.min():.3f}, {depths.max():.3f}] meters")

        for i, img_path in enumerate(chunk_images):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
            conf_path = os.path.join(args.output_dir, stem + "_conf.npy")

            print(f"  [{saved_count+1}/{len(images)}] {os.path.basename(img_path)} "
                  f"→ depth [{depths[i].min():.2f}, {depths[i].max():.2f}]m")
            np.save(depth_path, depths[i])
            np.save(conf_path, confs[i])
            saved_count += 1

        all_extrinsics.append(extrinsics)
        all_intrinsics.append(intrinsics)

        if prediction.processed_images is not None:
            proc_dir = os.path.join(args.output_dir, "processed_images")
            os.makedirs(proc_dir, exist_ok=True)
            for i, img_path in enumerate(chunk_images):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(proc_dir, stem + ".png")
                rgb = prediction.processed_images[i]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_path, bgr)

        del prediction, depths, confs, extrinsics, intrinsics
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[DA3 Full] Chunk {chunk_idx+1}/{num_chunks} done, memory freed.")

    if num_chunks > 1:
        print(f"[DA3 Full] ⚠️  Poses from {num_chunks} chunks are NOT globally consistent")

    return np.concatenate(all_extrinsics, axis=0), np.concatenate(all_intrinsics, axis=0)


if __name__ == "__main__":
    main()
