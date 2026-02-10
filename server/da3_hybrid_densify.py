#!/usr/bin/env python3
"""
DA3 Hybrid Densifier — DA3_Streaming + MASt3R Metric Scale

Architecture:
  1. DA3_Streaming processes all frames with SIM3 inter-chunk alignment
     → globally consistent point cloud at arbitrary scale
  2. Load DA3's estimated camera poses (camera_poses.txt)
  3. Single global Umeyama: DA3 poses → MASt3R metric poses → one scale factor
  4. Rescale all per-chunk PLYs by the metric scale
  5. Emit CHUNK_READY markers for real-time viewer streaming

Usage (subprocess from mast3r-slam env):
    python da3_hybrid_densify.py \
        --images-dir <dir> \
        --poses <mast3r_poses.npy> \
        --output <output.npz> \
        --config-json <config.json>

ALL configuration comes from config.yaml via the --config-json file.
No hardcoded values in this script.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

import argparse
import sys
import os
from pathlib import Path
import gc
import glob
import json
import time
import numpy as np

# Add DA3 source paths
DA3_ROOT = os.path.expanduser("~/Depth-Anything-3")
sys.path.insert(0, os.path.join(DA3_ROOT, "src"))
sys.path.insert(0, DA3_ROOT)
sys.path.insert(0, os.path.join(DA3_ROOT, "da3_streaming"))


def resolve_hf_model_paths(model_name):
    """
    Resolve model.safetensors and config.json from HuggingFace cache.
    
    Args:
        model_name: HuggingFace model name, e.g. "depth-anything/DA3-LARGE-1.1"
    
    Returns:
        (weights_path, config_path) absolute paths to the cached files
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_name.replace("/", "--")
    model_dir = os.path.join(cache_dir, model_dir_name)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model '{model_name}' not found in HF cache at {model_dir}. "
            f"Run: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_name}')\""
        )
    
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        raise FileNotFoundError(f"No snapshots dir found for {model_name}")
    
    snapshots = sorted(os.listdir(snapshots_dir))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found for {model_name}")
    
    snapshot_dir = os.path.join(snapshots_dir, snapshots[-1])
    
    weights_path = os.path.join(snapshot_dir, "model.safetensors")
    config_path = os.path.join(snapshot_dir, "config.json")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"model.safetensors not found in {snapshot_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {snapshot_dir}")
    
    return weights_path, config_path


def load_ply_points_colors(ply_path):
    """
    Load points and colors from a PLY file (ASCII or binary_little_endian).
    Returns (points [N,3], colors [N,3] in 0-1 float range).
    """
    import struct
    
    with open(ply_path, 'rb') as f:
        # Parse header
        vertex_count = 0
        ply_format = "ascii"
        properties = []
        
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("format"):
                ply_format = line.split()[1]
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))  # (type, name)
        
        if vertex_count == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        
        if ply_format == "binary_little_endian":
            # Build struct format from properties
            type_map = {
                "float": "f", "float32": "f",
                "double": "d", "float64": "d",
                "uchar": "B", "uint8": "B",
                "char": "b", "int8": "b",
                "ushort": "H", "uint16": "H",
                "short": "h", "int16": "h",
                "uint": "I", "uint32": "I",
                "int": "i", "int32": "i",
            }
            fmt = "<" + "".join(type_map.get(p[0], "f") for p in properties)
            stride = struct.calcsize(fmt)
            
            raw = f.read(vertex_count * stride)
            
            points = np.zeros((vertex_count, 3), dtype=np.float32)
            colors = np.zeros((vertex_count, 3), dtype=np.float32)
            
            prop_names = [p[1] for p in properties]
            x_i = prop_names.index("x") if "x" in prop_names else 0
            y_i = prop_names.index("y") if "y" in prop_names else 1
            z_i = prop_names.index("z") if "z" in prop_names else 2
            r_i = prop_names.index("red") if "red" in prop_names else 3
            g_i = prop_names.index("green") if "green" in prop_names else 4
            b_i = prop_names.index("blue") if "blue" in prop_names else 5
            
            for i in range(vertex_count):
                vals = struct.unpack_from(fmt, raw, i * stride)
                points[i] = [vals[x_i], vals[y_i], vals[z_i]]
                r, g, b = vals[r_i], vals[g_i], vals[b_i]
                # Normalize colors: if uchar (0-255) → 0-1
                if isinstance(r, int):
                    colors[i] = [r / 255.0, g / 255.0, b / 255.0]
                else:
                    colors[i] = [r, g, b]
        else:
            # ASCII format
            points = []
            colors = []
            for _ in range(vertex_count):
                parts = f.readline().decode('ascii', errors='ignore').strip().split()
                if len(parts) >= 6:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([float(parts[3]) / 255.0, float(parts[4]) / 255.0, float(parts[5]) / 255.0])
            points = np.array(points, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)
    
    return points, colors


def umeyama_alignment(x, y, with_scale=True):
    """
    Umeyama alignment: find s, R, t such that y ≈ s*R*x + t
    
    Args:
        x: [N, 3] source points (DA3 camera positions)
        y: [N, 3] target points (MASt3R camera positions)
    
    Returns:
        scale, R, t
    """
    assert x.shape == y.shape
    n, d = x.shape
    
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    
    xc = x - mx
    yc = y - my
    
    sx = np.mean(np.sum(xc ** 2, axis=1))
    H = (xc.T @ yc) / n
    
    U, D, Vt = np.linalg.svd(H)
    
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    
    R = Vt.T @ S @ U.T
    
    if with_scale:
        scale = np.trace(np.diag(D) @ S) / sx
    else:
        scale = 1.0
    
    t = my - scale * R @ mx
    
    return scale, R, t


def build_da3_streaming_config(app_cfg):
    """
    Build a DA3_Streaming-compatible config dict.
    
    ALL values come from app_cfg (our config.yaml serialized as JSON).
    Model weights and config.json are resolved from HuggingFace cache.
    """
    depth_cfg = app_cfg["models"]["depth"]
    da3_cfg = app_cfg["models"]["da3"]
    alignment_cfg = app_cfg["alignment"]
    server_cfg = app_cfg["server"]

    # Resolve model from HF cache based on model name
    model_name = depth_cfg["name"]
    weights_path, config_path = resolve_hf_model_paths(model_name)

    print(f"[DA3 Hybrid] Model: {model_name}")
    print(f"[DA3 Hybrid] Weights: {weights_path}")
    print(f"[DA3 Hybrid] Config: {config_path}")
    print(f"[DA3 Hybrid] Device: {depth_cfg['device']}")

    # SALAD weights for loop closure
    salad_path = os.path.join(app_cfg["paths"]["da3_weights_dir"], "dino_salad.ckpt")

    return {
        "Weights": {
            "DA3_CONFIG": config_path,
            "DA3": weights_path,
            "SALAD": salad_path,
        },
        "Model": {
            "chunk_size": server_cfg["chunk_size"],
            "overlap": server_cfg["chunk_overlap"],
            "loop_chunk_size": da3_cfg["loop_chunk_size"],
            "loop_enable": da3_cfg.get("loop_enable", False),
            "useDBoW": da3_cfg.get("useDBoW", False),
            "delete_temp_files": da3_cfg.get("delete_temp_files", False),
            "align_method": alignment_cfg["method"],
            "align_lib": alignment_cfg["align_lib"],
            "scale_compute_method": alignment_cfg["scale_compute_method"],
            "align_type": alignment_cfg["align_type"],
            "device": depth_cfg["device"],
            "use_ray_pose": depth_cfg.get("use_ray_pose", True),
            "ref_view_strategy": depth_cfg["ref_view_strategy"],
            "ref_view_strategy_loop": depth_cfg.get("ref_view_strategy_loop", depth_cfg["ref_view_strategy"]),
            "depth_threshold": da3_cfg["depth_threshold"],
            "save_depth_conf_result": da3_cfg.get("save_depth_conf_result", False),
            "save_debug_info": da3_cfg.get("save_debug_info", False),
            "Sparse_Align": {
                "keypoint_select": alignment_cfg["sparse_align"]["keypoint_select"],
                "keypoint_num": alignment_cfg["sparse_align"]["keypoint_num"],
            },
            "IRLS": {
                "delta": alignment_cfg["ransac"]["delta"],
                "max_iters": alignment_cfg["ransac"]["max_iters"],
                "tol": alignment_cfg["ransac"]["tolerance"],
            },
            "Pointcloud_Save": {
                "conf_threshold_coef": da3_cfg["pointcloud_save"]["conf_threshold_coef"],
                "sample_ratio": da3_cfg["pointcloud_save"]["sample_ratio"],
            },
        },
        "Loop": {
            "SALAD": da3_cfg["loop_salad"],
            "SIM3_Optimizer": da3_cfg["loop_sim3_optimizer"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="DA3 Hybrid: DA3_Streaming + MASt3R metric scale"
    )
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory with input frames")
    parser.add_argument("--poses", type=str, required=True,
                        help="Path to MASt3R metric c2w poses (.npy)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path for final metric point cloud")
    parser.add_argument("--config-json", type=str, required=True,
                        help="Path to JSON file with all config from config.yaml")
    args = parser.parse_args()

    # ── Load config (everything comes from here) ──────────────────────
    with open(args.config_json) as f:
        app_cfg = json.load(f)

    print(f"[DA3 Hybrid] Starting DA3_Streaming + MASt3R metric scale")
    print(f"[DA3 Hybrid] Images: {args.images_dir}")

    # ── Step 1: Load MASt3R metric poses ──────────────────────────────
    mast3r_c2w = np.load(args.poses)  # [N, 4, 4] camera-to-world
    n_frames = len(mast3r_c2w)
    print(f"[DA3 Hybrid] Loaded {n_frames} MASt3R metric c2w poses")
    print(f"[DA3 Hybrid] Trajectory span: {np.ptp(mast3r_c2w[:, :3, 3], axis=0)}")

    # ── Step 2: Run DA3_Streaming ─────────────────────────────────────
    from da3_streaming import DA3_Streaming

    output_dir = str(Path(args.output).parent / "da3_streaming_output")
    os.makedirs(output_dir, exist_ok=True)

    da3_config = build_da3_streaming_config(app_cfg)

    print(f"[DA3 Hybrid] Initializing DA3_Streaming...")
    t0 = time.time()

    da3 = DA3_Streaming(
        image_dir=args.images_dir,
        save_dir=output_dir,
        config=da3_config,
    )

    print(f"[DA3 Hybrid] Running DA3_Streaming.run() ...")
    da3.run()

    elapsed = time.time() - t0
    print(f"[DA3 Hybrid] DA3_Streaming complete in {elapsed:.1f}s")

    # ── Step 3: Load DA3 camera poses ─────────────────────────────────
    da3_poses_path = os.path.join(output_dir, "camera_poses.txt")
    if not os.path.exists(da3_poses_path):
        print(f"[DA3 Hybrid] ERROR: camera_poses.txt not found at {da3_poses_path}")
        sys.exit(1)

    da3_c2w = []
    with open(da3_poses_path, 'r') as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 16:
                da3_c2w.append(np.array(vals).reshape(4, 4))

    da3_c2w = np.array(da3_c2w, dtype=np.float64)
    print(f"[DA3 Hybrid] Loaded {len(da3_c2w)} DA3 camera poses")

    # Match pose counts
    n = min(len(da3_c2w), len(mast3r_c2w))
    if n < 3:
        print(f"[DA3 Hybrid] ERROR: Need at least 3 poses for Umeyama, got {n}")
        sys.exit(1)

    da3_c2w = da3_c2w[:n]
    mast3r_c2w_matched = mast3r_c2w[:n]

    # ── Step 4: Global Umeyama alignment ──────────────────────────────
    da3_positions = da3_c2w[:, :3, 3]
    mast3r_positions = mast3r_c2w_matched[:, :3, 3]

    # Filter out zero poses
    valid = ~(np.all(da3_positions == 0, axis=1) | np.all(mast3r_positions == 0, axis=1))
    da3_pos_valid = da3_positions[valid]
    mast3r_pos_valid = mast3r_positions[valid]

    print(f"[DA3 Hybrid] Valid poses for Umeyama: {len(da3_pos_valid)}/{n}")

    scale, R, t = umeyama_alignment(da3_pos_valid, mast3r_pos_valid, with_scale=True)

    print(f"[DA3 Hybrid] ══════════════════════════════════════")
    print(f"[DA3 Hybrid] GLOBAL Umeyama scale: {scale:.6f}")
    print(f"[DA3 Hybrid] (All chunks share this single scale)")
    print(f"[DA3 Hybrid] ══════════════════════════════════════")

    # Verify alignment quality
    da3_aligned = scale * (R @ da3_pos_valid.T).T + t
    residual = np.mean(np.linalg.norm(da3_aligned - mast3r_pos_valid, axis=1))
    print(f"[DA3 Hybrid] Alignment residual: {residual:.4f}m")

    # ── Step 5: Load PLYs, rescale, and emit CHUNK_READY ──────────────
    pcd_dir = os.path.join(output_dir, "pcd")
    ply_files = sorted(glob.glob(os.path.join(pcd_dir, "*_pcd.ply")))

    if not ply_files:
        print(f"[DA3 Hybrid] ERROR: No PLY files found in {pcd_dir}")
        sys.exit(1)

    print(f"[DA3 Hybrid] Found {len(ply_files)} chunk PLYs to rescale")

    all_points = []
    all_colors = []

    for chunk_idx, ply_path in enumerate(ply_files):
        points, colors = load_ply_points_colors(ply_path)

        if len(points) == 0:
            print(f"[DA3 Hybrid] Chunk {chunk_idx}: empty, skipping")
            continue

        # Apply metric transformation: p_metric = s * R @ p_da3 + t
        points_metric = scale * (R @ points.T).T + t
        points_metric = points_metric.astype(np.float32)

        all_points.append(points_metric)
        all_colors.append(colors)

        # Save chunk NPZ for streaming
        chunk_npz = Path(args.output).parent / f"chunk_{chunk_idx}.npz"
        np.savez_compressed(str(chunk_npz), points=points_metric, colors=colors)

        # Emit CHUNK_READY for viewer streaming
        print(f"CHUNK_READY:{chunk_npz}:{len(points_metric)}", flush=True)
        print(f"[DA3 Hybrid] Chunk {chunk_idx}: {len(points_metric):,} metric points")

    # ── Step 6: Save final concatenated result ────────────────────────
    if all_points:
        final_points = np.concatenate(all_points, axis=0)
        final_colors = np.concatenate(all_colors, axis=0)

        np.savez_compressed(args.output, points=final_points, colors=final_colors)

        print(f"[DA3 Hybrid] ✅ Final result: {len(final_points):,} metric points")
        print(f"[DA3 Hybrid] Saved to {args.output}")
    else:
        print(f"[DA3 Hybrid] ERROR: No points produced")
        sys.exit(1)

    # Cleanup
    del da3
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

    print(f"[DA3 Hybrid] Done.")


if __name__ == "__main__":
    main()
