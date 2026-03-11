# STAC-Builder: MapAnything Worker (Subprocess)
# Runs MapAnything 3D reconstruction via VGGT-Long in its own process.
# Reads frames from session directory, writes chunk PLYs + origins.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import json
import subprocess
import re
import shutil
import glob
import tempfile
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _map_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """MapAnything reconstruction via VGGT-Long — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe.send_log("Starting MapAnything reconstruction (VGGT-Long)")
    pipe.send_progress(0, "Initializing...", stage="reconstruction")

    # ── Step 1: Frame quality analysis ──
    pipe.send_progress(2, "Analyzing frame quality...", stage="reconstruction")
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    try:
        from frame_quality import analyze_frames, save_manifest
        fq = analyze_frames(str(frames_dir))
        if "error" not in fq:
            save_manifest(str(frames_dir), fq)
    except Exception as e:
        pipe.send_log(f"Frame quality analysis skipped: {e}", level="warning")

    # ── Step 2: Frame selection (visual novelty filter) ──
    selected_frames_path = None
    ma_cfg = config.get("mapanything", {})

    if ma_cfg.get("use_keyframes", True):
        from frame_selector import _load_valid_frame_list, select_keyframes, load_selected_frames
        frame_sel_cfg = config.get("frame_selection", {})
        if frame_sel_cfg.get("enabled", False):
            try:
                pipe.send_progress(5, "Selecting keyframes...", stage="reconstruction")
                sel = select_keyframes(str(frames_dir), frame_sel_cfg)
                pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes")
            except Exception as e:
                pipe.send_log(f"Frame selection failed: {e}", level="warning")

        # Check if selected_frames.json exists
        sf_path = frames_dir / "selected_frames.json"
        if sf_path.exists():
            selected_frames_path = str(sf_path)
            pipe.send_log(f"Using keyframes from {sf_path}")

    # ── Step 3: Generate VGGT-Long config ──
    pipe.send_progress(8, "Generating VGGT-Long config...", stage="reconstruction")

    vggt_config = _build_vggt_config(config)
    
    # Write temporary config for this run
    vggt_config_path = output_dir / "vggt_long_config.yaml"
    import yaml
    with open(vggt_config_path, 'w') as f:
        yaml.dump(vggt_config, f, default_flow_style=False)

    pipe.send_log(f"VGGT-Long config: {vggt_config_path}")

    # ── Step 4: Run VGGT-Long as subprocess ──
    pipe.send_progress(10, "Starting VGGT-Long reconstruction...", stage="reconstruction")

    # Determine VGGT-Long script path
    project_root = Path(__file__).resolve().parent.parent.parent  # server/workers -> server -> project_root
    vggt_script = project_root / "vendor" / "VGGT-Long" / "vggt_long.py"
    
    if not vggt_script.exists():
        raise FileNotFoundError(f"VGGT-Long script not found: {vggt_script}")

    # Build save directory (VGGT-Long outputs to its own structure)
    vggt_save_dir = output_dir / "maplong_run"

    # Build command using .sh launcher (same pattern as CloudCompPy)
    server_dir_path = Path(__file__).resolve().parent.parent
    script_path = server_dir_path / "run_mapanything.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"run_mapanything.sh not found: {script_path}")

    device = ma_cfg.get("device", "cpu")

    cmd = [
        "bash", str(script_path),
        "--image_dir", str(frames_dir),
        "--config", str(vggt_config_path),
        "--save_dir", str(vggt_save_dir),
    ]

    if selected_frames_path:
        cmd.extend(["--selected_frames", selected_frames_path])

    # Set CUDA visibility
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    pipe.send_log(f"Running: {' '.join(cmd[-6:])}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(vggt_script.parent),
    )

    # Parse progress from stdout
    chunk_pattern = re.compile(r'\[Progress\]:\s*(\d+)/(\d+)')
    
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        # Check for cancellation
        if pipe.check_cancel():
            proc.terminate()
            pipe.send_log("Cancelled by user", level="warning")
            return

        # Parse progress
        match = chunk_pattern.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            pct = 10 + (done / max(total, 1)) * 70
            pipe.send_progress(pct, f"Chunk {done}/{total}", stage="reconstruction")
        elif "Loading model" in line or "Loading MapAnything" in line:
            pipe.send_progress(12, "Loading MapAnything model...", stage="reconstruction")
        elif "Extracting features" in line:
            pipe.send_progress(15, "Loop detection (feature extraction)...", stage="reconstruction")
        elif "Apply alignment" in line:
            pipe.send_progress(82, "Applying alignment...", stage="reconstruction")

        # Forward all log lines (same pattern as cloudcompy_worker)
        pipe.send_log(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"VGGT-Long exited with code {proc.returncode}")

    pipe.send_progress(85, "VGGT-Long complete, post-processing...", stage="reconstruction")

    # ── Step 5: Post-process VGGT-Long output ──
    # Copy PLY files to output dir with chunk_XXX.ply naming
    pcd_dir = vggt_save_dir / "pcd"
    if not pcd_dir.exists():
        raise FileNotFoundError(f"VGGT-Long output not found: {pcd_dir}")

    ply_files = sorted(glob.glob(str(pcd_dir / "*_pcd.ply")))
    # Filter out combined_pcd.ply — it's the merged file, not a chunk
    ply_files = [f for f in ply_files if "combined" not in Path(f).name]
    if not ply_files:
        raise FileNotFoundError(f"No chunk PLY files found in {pcd_dir}")

    pipe.send_log(f"Found {len(ply_files)} chunk PLYs")

    for i, ply_src in enumerate(ply_files):
        ply_dst = output_dir / f"chunk_{i:03d}.ply"
        shutil.copy2(ply_src, ply_dst)
        pipe.send_log(f"Copied {Path(ply_src).name} → {ply_dst.name}")

    # ── Step 6: Generate origin traceability from saved chunk data ──
    pipe.send_progress(90, "Generating origin traceability...", stage="reconstruction")
    _generate_origins(vggt_save_dir, output_dir, vggt_config, pipe)

    # ── Step 7: Save camera poses metadata ──
    pipe.send_progress(95, "Saving metadata...", stage="reconstruction")
    cam_poses_src = vggt_save_dir / "camera_poses.json"
    if cam_poses_src.exists():
        shutil.copy2(cam_poses_src, output_dir / "camera_poses_mapanything.json")

    # Copy intrinsics if available
    intrinsic_src = vggt_save_dir / "intrinsic.txt"
    if intrinsic_src.exists():
        shutil.copy2(intrinsic_src, output_dir / "intrinsic.txt")

    pipe.send_progress(100, "MapAnything reconstruction complete", stage="reconstruction")
    pipe.send_log(f"MapAnything complete: {len(ply_files)} chunks")

    # Cleanup: delete heavy temp .npy files now that origins are generated
    for tmp_dir_name in ["_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop"]:
        tmp_dir = vggt_save_dir / tmp_dir_name
        if tmp_dir.exists():
            size_mb = sum(f.stat().st_size for f in tmp_dir.glob("*")) / (1024 * 1024)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            pipe.send_log(f"Cleaned up {tmp_dir_name}/ ({size_mb:.0f} MB freed)")

    import gc
    gc.collect()


def _build_vggt_config(config: dict) -> dict:
    """Load the tested stac_mapanything.yaml and override only user-configurable params."""
    import yaml as _yaml

    ma = config.get("mapanything", {})

    # Load the tested base config
    project_root = Path(__file__).resolve().parent.parent.parent
    base_cfg_path = project_root / "vendor" / "VGGT-Long" / "configs" / "stac_mapanything.yaml"

    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")

    with open(base_cfg_path, 'r') as f:
        cfg = _yaml.safe_load(f)

    # Override only the configurable parameters from config.yaml mapanything section
    cfg["Model"]["chunk_size"] = ma.get("chunk_size", cfg["Model"]["chunk_size"])
    cfg["Model"]["overlap"] = ma.get("chunk_overlap", cfg["Model"]["overlap"])
    cfg["Model"]["loop_enable"] = ma.get("loop_closure", cfg["Model"].get("loop_enable", True))
    cfg["Model"]["delete_temp_files"] = False  # Keep .npy for origin traceability

    pc = cfg["Model"].get("Pointcloud_Save", {})
    pc["sample_ratio"] = ma.get("sample_ratio", pc.get("sample_ratio", 1.0))
    pc["conf_threshold_coef"] = ma.get("conf_threshold_coef", pc.get("conf_threshold_coef", 0.75))
    cfg["Model"]["Pointcloud_Save"] = pc

    if ma.get("model_weights"):
        cfg["Weights"]["Map"] = ma["model_weights"]

    return cfg



def _read_ply_point_count(ply_path: Path) -> int:
    """Read vertex count from a PLY file header."""
    with open(ply_path, 'rb') as f:
        for line in f:
            line = line.decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                return int(line.split()[-1])
            if line == 'end_header':
                break
    return 0


def _generate_origins(vggt_save_dir: Path, output_dir: Path,
                      vggt_config: dict, pipe: WorkerPipe):
    """Generate chunk_*_origins.npz from VGGT-Long saved chunk .npy files.
    
    The unaligned chunk data contains world_points with shape (S, H, W, 3).
    S = frames in chunk, H/W = image dimensions.
    We reconstruct (frame_global, pixel_row, pixel_col) using the same
    math as alignment pipeline.
    """
    import numpy as np

    # VGGT-Long saves chunk data to _tmp_results_unaligned/chunk_N.npy
    unaligned_dir = vggt_save_dir / "_tmp_results_unaligned"
    if not unaligned_dir.exists():
        pipe.send_log("No unaligned chunk data found — origins not generated", level="warning")
        return

    chunk_npy_files = sorted(glob.glob(str(unaligned_dir / "chunk_*.npy")))
    if not chunk_npy_files:
        pipe.send_log("No chunk .npy files found — origins not generated", level="warning")
        return

    chunk_size = vggt_config["Model"]["chunk_size"]
    overlap = vggt_config["Model"]["overlap"]
    chunk_step = chunk_size - overlap
    conf_coef = vggt_config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
    sample_ratio = vggt_config["Model"]["Pointcloud_Save"]["sample_ratio"]

    for chunk_idx, npy_path in enumerate(chunk_npy_files):
        try:
            chunk_data = np.load(npy_path, allow_pickle=True).item()

            # world_points shape: (S, H, W, 3) or (1, S, H, W, 3)
            wp = chunk_data['world_points']
            if wp.ndim == 5:
                wp = wp[0]  # Remove batch dim
            
            confs = chunk_data['world_points_conf']
            if confs.ndim == 4:
                confs = confs[0]  # Remove batch dim

            S, H, W = wp.shape[:3]
            HW = H * W

            # Flatten
            points_flat = wp.reshape(-1, 3)
            confs_flat = confs.reshape(-1)

            # Apply same confidence filtering as save_confident_pointcloud_batch
            # MUST match exactly: (conf >= threshold) & (conf > 1e-5)
            conf_threshold = max(0.0, np.mean(confs_flat) * conf_coef)
            valid_mask = (confs_flat >= conf_threshold) & (confs_flat > 1e-5)

            valid_indices = np.where(valid_mask)[0]

            # Apply sampling
            if sample_ratio < 1.0 and len(valid_indices) > 0:
                n_samples = int(len(valid_indices) * sample_ratio)
                sample_local = np.sort(np.random.choice(
                    len(valid_indices), n_samples, replace=False
                ))
                final_indices = valid_indices[sample_local]
            else:
                final_indices = valid_indices

            if len(final_indices) == 0:
                pipe.send_log(f"Chunk {chunk_idx}: no valid points for origins", level="warning")
                continue

            # Compute origin for each surviving point
            frame_local = final_indices // HW
            pixel_row = (final_indices % HW) // W
            pixel_col = final_indices % W

            # Convert to global frame index
            frame_global = frame_local + chunk_idx * chunk_step

            # Read the actual PLY point count for this chunk
            ply_path = output_dir / f"chunk_{chunk_idx:03d}.ply"
            ply_point_count = _read_ply_point_count(ply_path) if ply_path.exists() else None

            # If there's a mismatch, truncate origins to match PLY
            if ply_point_count is not None and len(final_indices) != ply_point_count:
                pipe.send_log(
                    f"Chunk {chunk_idx}: origin/PLY mismatch ({len(final_indices)} vs {ply_point_count}), adjusting",
                    level="warning"
                )
                if ply_point_count < len(final_indices):
                    final_indices = final_indices[:ply_point_count]
                # Recompute after truncation
                frame_local = final_indices // HW
                pixel_row = (final_indices % HW) // W
                pixel_col = final_indices % W
                frame_global = frame_local + chunk_idx * chunk_step

            # Save origins
            origin_path = output_dir / f"chunk_{chunk_idx:03d}_origins.npz"
            np.savez_compressed(
                origin_path,
                frame_global=frame_global.astype(np.int32),
                pixel_row=pixel_row.astype(np.int16),
                pixel_col=pixel_col.astype(np.int16),
                scaled_resolution=[H, W],
            )
            pipe.send_log(f"Saved origins: {origin_path.name} ({len(frame_global)} points)")

            # Also save chunk metadata for segmentation compatibility
            meta_path = output_dir / f"chunk_{chunk_idx:03d}_meta.json"
            meta = {
                "chunk_id": chunk_idx,
                "frame_count": S,
                "scaled_resolution": [H, W],
                "chunk_step": chunk_step,
                "frame_global_start": chunk_idx * chunk_step,
                "frame_global_end": chunk_idx * chunk_step + S - 1,
                "backend": "mapanything",
                "ply_pre_aligned": True,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

        except Exception as e:
            pipe.send_log(f"Failed to generate origins for chunk {chunk_idx}: {e}", level="warning")
            import traceback
            traceback.print_exc()


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager as multiprocessing target."""
    run_worker_safe(_map_work, conn, session_dir, config)
