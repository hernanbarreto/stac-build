# STAC-Builder: Reconstruction Worker (Subprocess)
# Runs 3D reconstruction via DA3 Streaming or VGGT-Long (MapAnything) in its own process.
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
    """3D reconstruction — runs inside a dedicated subprocess.
    
    Dispatches to DA3 Streaming or VGGT-Long (MapAnything) based on config.
    """

    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read backend from new reconstruction config, fallback to legacy mapanything
    recon_cfg = config.get("reconstruction", {})
    backend = recon_cfg.get("backend", "mapanything")
    # Also support legacy config where 'mapanything' was a top-level key
    if not recon_cfg and "mapanything" in config:
        backend = "mapanything"
        recon_cfg = {"mapanything": config["mapanything"], "device": config["mapanything"].get("device", "cpu")}

    pipe.send_log(f"Starting 3D reconstruction (backend: {backend})")
    pipe.send_progress(0, "Initializing...", stage="reconstruction")

    # ── Step 1: Frame quality analysis ──
    # Skip for lidar-only mode (no neural inference, uses all frames)
    if backend != "lidar":
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
    else:
        pipe.send_log("Skipping frame quality/selection (lidar mode — all frames used)")
        server_dir = str(Path(__file__).resolve().parent.parent)
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)

    # ── Step 2: Frame selection (visual novelty filter) ──
    # Skip for lidar-only mode
    selected_frames_path = None
    use_keyframes = recon_cfg.get("use_keyframes", True) and backend != "lidar"

    if use_keyframes:
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
    if sf_path.exists() and backend != "lidar":
        selected_frames_path = str(sf_path)
        pipe.send_log(f"Using keyframes from {sf_path}")

    # ── Step 3: Dispatch to backend ──
    if backend == "da3":
        _run_da3(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
    elif backend in ("hybrid", "lidar"):
        _run_hybrid_or_lidar(
            pipe, frames_dir, output_dir, selected_frames_path,
            recon_cfg, config, session_path, mode=backend
        )
    else:
        _run_mapanything(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)


def _run_da3(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
             selected_frames_path: str, recon_cfg: dict, config: dict):
    """Run DA3 Streaming as subprocess."""
    import yaml

    da3_cfg = recon_cfg.get("da3", {})
    device = recon_cfg.get("device", "cpu")

    pipe.send_progress(8, "Generating DA3 config...", stage="reconstruction")

    # Build DA3 config YAML from our config.yaml settings
    da3_config = _build_da3_config(recon_cfg)

    # Write temporary config for this run
    da3_config_path = output_dir / "da3_streaming_config.yaml"
    with open(da3_config_path, 'w') as f:
        yaml.dump(da3_config, f, default_flow_style=False)

    pipe.send_log(f"DA3 config: {da3_config_path}")

    # ── Run DA3 Streaming as subprocess ──
    pipe.send_progress(10, "Starting DA3 Streaming reconstruction...", stage="reconstruction")

    project_root = Path(__file__).resolve().parent.parent.parent
    server_dir_path = Path(__file__).resolve().parent.parent
    script_path = server_dir_path / "run_da3.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"run_da3.sh not found: {script_path}")

    da3_save_dir = output_dir / "da3_run"

    # Build image dir — if selected_frames exist, we need to pass the frames dir
    # DA3 reads images from --image_dir directly
    image_dir = str(frames_dir)

    cmd = [
        "bash", str(script_path),
        "--image_dir", image_dir,
        "--config", str(da3_config_path),
        "--output_dir", str(da3_save_dir),
    ]

    # Pass selected keyframes filter if available
    if selected_frames_path and Path(selected_frames_path).exists():
        cmd.extend(["--selected_frames", str(selected_frames_path)])

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
    )

    # Parse progress from stdout
    chunk_pattern = re.compile(r'\[Progress\]:\s*(\d+)/(\d+)')

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        if pipe.check_cancel():
            proc.terminate()
            pipe.send_log("Cancelled by user", level="warning")
            return

        match = chunk_pattern.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            pct = 10 + (done / max(total, 1)) * 70
            pipe.send_progress(pct, f"Chunk {done}/{total}", stage="reconstruction")
        elif "Loading model" in line:
            pipe.send_progress(12, "Loading DA3 model...", stage="reconstruction")
        elif "Extracting features" in line:
            pipe.send_progress(15, "Loop detection (feature extraction)...", stage="reconstruction")
        elif "Apply alignment" in line:
            pipe.send_progress(82, "Applying alignment...", stage="reconstruction")

        pipe.send_log(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"DA3 Streaming exited with code {proc.returncode}")

    pipe.send_progress(85, "DA3 complete, post-processing...", stage="reconstruction")

    # ── Post-process DA3 output (same format as VGGT-Long) ──
    _postprocess_reconstruction(pipe, da3_save_dir, output_dir, da3_config, backend="da3")


def _run_hybrid_or_lidar(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                         selected_frames_path: str, recon_cfg: dict, config: dict,
                         session_path: Path, mode: str = "hybrid"):
    """Run DA3 Streaming with Stray Scanner data injection (hybrid or lidar mode).

    - hybrid: DA3 inference + LiDAR injection + LiDAR backprojection complement
    - lidar: DA3 SLAM only (no neural inference), uses LiDAR depth directly
    """
    import yaml

    lidar_cfg = recon_cfg.get("lidar", {})
    da3_cfg = recon_cfg.get("da3", {})
    fallback = lidar_cfg.get("fallback_to_da3", True)

    # ── Detect Stray Scanner data ──
    from ingestors.stray_detector import detect_stray_data
    stray_info = detect_stray_data(str(session_path))

    if not stray_info["is_stray_session"]:
        if fallback:
            pipe.send_log(
                f"No Stray Scanner data found — falling back to DA3 (from {mode})",
                level="warning"
            )
            _run_da3(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
            return
        else:
            raise FileNotFoundError(
                f"Backend '{mode}' requires Stray Scanner data, but not found in {session_path}"
            )

    pipe.send_log(f"Stray Scanner data detected: {stray_info['depth_count']} depth frames")
    pipe.send_progress(8, f"Generating DA3 config ({mode} mode)...", stage="reconstruction")

    # Build DA3 config
    da3_config = _build_da3_config(recon_cfg)
    da3_config_path = output_dir / "da3_streaming_config.yaml"
    with open(da3_config_path, 'w') as f:
        yaml.dump(da3_config, f, default_flow_style=False)

    # ── Run DA3 Hybrid/LiDAR as subprocess ──
    pipe.send_progress(10, f"Starting DA3 {mode.upper()} reconstruction...", stage="reconstruction")

    server_dir_path = Path(__file__).resolve().parent.parent
    script_path = server_dir_path / "run_da3_hybrid.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"run_da3_hybrid.sh not found: {script_path}")

    da3_save_dir = output_dir / "da3_run"

    cmd = [
        "bash", str(script_path),
        "--mode", mode,
        "--image_dir", str(frames_dir),
        "--data_dir", str(session_path),
        "--config", str(da3_config_path),
        "--output_dir", str(da3_save_dir),
        "--stride", str(lidar_cfg.get("stride", 4)),
        "--confidence_threshold", str(lidar_cfg.get("confidence_threshold", 1)),
        "--lidar_trust_range", str(lidar_cfg.get("trust_range", 5.0)),
    ]

    if selected_frames_path and Path(selected_frames_path).exists():
        cmd.extend(["--selected_frames", str(selected_frames_path)])

    env = os.environ.copy()
    device = recon_cfg.get("device", "cpu")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    pipe.send_log(f"Running: {mode} mode with LiDAR trust={lidar_cfg.get('trust_range', 5.0)}m")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    chunk_pattern = re.compile(r'\[Progress\]:\s*(\d+)/(\d+)')

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        if pipe.check_cancel():
            proc.terminate()
            pipe.send_log("Cancelled by user", level="warning")
            return

        match = chunk_pattern.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            pct = 10 + (done / max(total, 1)) * 60
            pipe.send_progress(pct, f"Chunk {done}/{total}", stage="reconstruction")
        elif "Loading model" in line:
            pipe.send_progress(12, "Loading DA3 model...", stage="reconstruction")
        elif "StrayDA3" in line or "StrayLiDAR" in line:
            pipe.send_progress(15, line[:80], stage="reconstruction")

        pipe.send_log(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"DA3 {mode} exited with code {proc.returncode}")

    pipe.send_progress(75, f"DA3 {mode} complete, post-processing...", stage="reconstruction")

    # ── Post-process reconstruction output (chunks → output/) ──
    _postprocess_reconstruction(pipe, da3_save_dir, output_dir, da3_config, backend=f"da3_{mode}")

    # ── Hybrid mode: generate LiDAR complement cloud ──
    if mode == "hybrid":
        pipe.send_progress(92, "Generating LiDAR complement cloud...", stage="reconstruction")
        try:
            _generate_lidar_complement(pipe, da3_save_dir, output_dir, session_path, lidar_cfg)
        except Exception as e:
            pipe.send_log(f"LiDAR complement generation failed: {e}", level="warning")
            import traceback
            traceback.print_exc()

    pipe.send_progress(100, f"{mode.upper()} reconstruction complete", stage="reconstruction")


def _generate_lidar_complement(pipe: WorkerPipe, da3_save_dir: Path, output_dir: Path,
                               session_path: Path, lidar_cfg: dict):
    """Generate a LiDAR-only point cloud using DA3-streaming's refined poses.

    Backprojects raw LiDAR depth maps using camera_poses.txt from the
    DA3-streaming run (post-loop-closure). This cloud is saved as
    lidar_complement.ply in output/ for CloudCompPy to merge.
    """
    import numpy as np
    import cv2

    poses_path = da3_save_dir / "camera_poses.txt"
    if not poses_path.exists():
        pipe.send_log("camera_poses.txt not found — skipping LiDAR complement", level="warning")
        return

    # Load poses
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 16:
                poses.append(np.array([float(v) for v in vals]).reshape(4, 4))

    # Load Stray Scanner data
    from ingestors.stray_scanner import prepare_stray_data
    frames_dir = da3_save_dir.parent.parent / "frames"  # session/frames/
    if not frames_dir.exists():
        frames_dir = session_path / "frames"

    stray = prepare_stray_data(
        data_dir=str(session_path),
        frames_output_dir=str(frames_dir),
        stride=lidar_cfg.get("stride", 4),
        max_frames=0,
        confidence_threshold=lidar_cfg.get("confidence_threshold", 1),
    )

    K = stray['intrinsics']
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    n = min(len(stray['depths']), len(poses), len(frame_files))

    pipe.send_log(f"Backprojecting {n} LiDAR frames with DA3-streaming poses")

    all_pts, all_cols = [], []
    for i in range(n):
        depth = stray['depths'][i]
        c2w = poses[i]
        rgb = cv2.cvtColor(cv2.imread(str(frame_files[i])), cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid = depth > 0
        pts_cam = np.stack([
            (u[valid] - cx) * depth[valid] / fx,
            (v[valid] - cy) * depth[valid] / fy,
            depth[valid]
        ], axis=-1)
        pts_world = (pts_cam @ c2w[:3,:3].T) + c2w[:3,3]
        all_pts.append(pts_world.astype(np.float32))
        all_cols.append(rgb_small[valid].astype(np.uint8))

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)

    # Save as lidar_complement.ply in output/
    complement_path = output_dir / "lidar_complement.ply"
    _n = len(pts)
    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                      ('r','u1'),('g','u1'),('b','u1')])
    vd = np.empty(_n, dtype=dtype)
    vd['x'], vd['y'], vd['z'] = pts[:,0], pts[:,1], pts[:,2]
    vd['r'], vd['g'], vd['b'] = cols[:,0], cols[:,1], cols[:,2]
    with open(complement_path, 'wb') as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {_n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"end_header\n".encode('ascii'))
        vd.tofile(f)

    size_mb = complement_path.stat().st_size / (1024 * 1024)
    pipe.send_log(f"LiDAR complement: {_n:,} pts ({size_mb:.0f} MB) → {complement_path.name}")


def _run_mapanything(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                     selected_frames_path: str, recon_cfg: dict, config: dict):
    """Run VGGT-Long (MapAnything) as subprocess — legacy backend."""
    import yaml

    ma_cfg = recon_cfg.get("mapanything", config.get("mapanything", {}))
    device = recon_cfg.get("device", ma_cfg.get("device", "cpu"))

    pipe.send_progress(8, "Generating VGGT-Long config...", stage="reconstruction")

    vggt_config = _build_vggt_config(config)
    vggt_config_path = output_dir / "vggt_long_config.yaml"
    with open(vggt_config_path, 'w') as f:
        yaml.dump(vggt_config, f, default_flow_style=False)

    pipe.send_log(f"VGGT-Long config: {vggt_config_path}")

    # ── Run VGGT-Long as subprocess ──
    pipe.send_progress(10, "Starting VGGT-Long reconstruction...", stage="reconstruction")

    project_root = Path(__file__).resolve().parent.parent.parent
    vggt_script = project_root / "vendor" / "VGGT-Long" / "vggt_long.py"

    if not vggt_script.exists():
        raise FileNotFoundError(f"VGGT-Long script not found: {vggt_script}")

    vggt_save_dir = output_dir / "maplong_run"
    server_dir_path = Path(__file__).resolve().parent.parent
    script_path = server_dir_path / "run_mapanything.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"run_mapanything.sh not found: {script_path}")

    cmd = [
        "bash", str(script_path),
        "--image_dir", str(frames_dir),
        "--config", str(vggt_config_path),
        "--save_dir", str(vggt_save_dir),
    ]

    if selected_frames_path:
        cmd.extend(["--selected_frames", selected_frames_path])

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

    chunk_pattern = re.compile(r'\[Progress\]:\s*(\d+)/(\d+)')

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        if pipe.check_cancel():
            proc.terminate()
            pipe.send_log("Cancelled by user", level="warning")
            return

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

        pipe.send_log(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"VGGT-Long exited with code {proc.returncode}")

    pipe.send_progress(85, "VGGT-Long complete, post-processing...", stage="reconstruction")

    _postprocess_reconstruction(pipe, vggt_save_dir, output_dir, vggt_config, backend="mapanything")


def _postprocess_reconstruction(pipe: WorkerPipe, save_dir: Path, output_dir: Path,
                                 run_config: dict, backend: str = "da3"):
    """Post-process reconstruction output (shared by DA3 and MapAnything).
    
    Both backends produce identical output layout:
      save_dir/pcd/N_pcd.ply, camera_poses.txt, intrinsic.txt
    """
    # ── Copy PLY files to output dir ──
    pcd_dir = save_dir / "pcd"
    if not pcd_dir.exists():
        raise FileNotFoundError(f"Reconstruction output not found: {pcd_dir}")

    ply_files = sorted(glob.glob(str(pcd_dir / "*_pcd.ply")))
    ply_files = [f for f in ply_files if "combined" not in Path(f).name]
    if not ply_files:
        raise FileNotFoundError(f"No chunk PLY files found in {pcd_dir}")

    pipe.send_log(f"Found {len(ply_files)} chunk PLYs")

    for i, ply_src in enumerate(ply_files):
        ply_dst = output_dir / f"chunk_{i:03d}.ply"
        shutil.copy2(ply_src, ply_dst)
        pipe.send_log(f"Copied {Path(ply_src).name} → {ply_dst.name}")

    # ── Copy GS PLY if available (DA3 with infer_gs=True) ──
    gs_dir = save_dir / "gs_ply"
    if gs_dir.exists():
        gs_files = sorted(glob.glob(str(gs_dir / "*.ply")))
        if gs_files:
            gs_output_dir = output_dir / "gs_ply"
            gs_output_dir.mkdir(exist_ok=True)
            for gs_src in gs_files:
                gs_dst = gs_output_dir / Path(gs_src).name
                shutil.copy2(gs_src, gs_dst)
            pipe.send_log(f"Copied {len(gs_files)} GS PLY files")

    # ── Generate origin traceability ──
    pipe.send_progress(90, "Generating origin traceability...", stage="reconstruction")
    run_config["_backend"] = backend
    _generate_origins(save_dir, output_dir, run_config, pipe)

    # ── Save camera poses metadata ──
    pipe.send_progress(95, "Saving metadata...", stage="reconstruction")
    for src_name, dst_name in [
        ("camera_poses.txt", "camera_poses_mapanything.json"),
        ("camera_poses.json", "camera_poses_mapanything.json"),
        ("intrinsic.txt", "intrinsic.txt"),
    ]:
        src = save_dir / src_name
        if src.exists():
            shutil.copy2(src, output_dir / dst_name)

    pipe.send_progress(100, f"{backend.upper()} reconstruction complete", stage="reconstruction")
    pipe.send_log(f"{backend} complete: {len(ply_files)} chunks")

    # Cleanup temp .npy files
    for tmp_dir_name in ["_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop"]:
        tmp_dir = save_dir / tmp_dir_name
        if tmp_dir.exists():
            size_mb = sum(f.stat().st_size for f in tmp_dir.glob("*")) / (1024 * 1024)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            pipe.send_log(f"Cleaned up {tmp_dir_name}/ ({size_mb:.0f} MB freed)")

    import gc
    gc.collect()


def _build_da3_config(recon_cfg: dict) -> dict:
    """Build DA3 Streaming config YAML from our reconstruction config."""
    da3 = recon_cfg.get("da3", {})
    device = recon_cfg.get("device", "cpu")

    # SALAD weights: use the one from VGGT-Long or DA3 streaming's weights dir
    project_root = Path(__file__).resolve().parent.parent.parent
    salad_path = project_root / "vendor" / "depth-anything-3" / "da3_streaming" / "weights" / "dino_salad.ckpt"
    if not salad_path.exists():
        # Fallback to VGGT-Long weights
        salad_path = project_root / "vendor" / "VGGT-Long" / "weights" / "dino_salad.ckpt"

    cfg = {
        "Weights": {
            "DA3_HF_MODEL": da3.get("model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
            "SALAD": str(salad_path),
        },
        "Model": {
            "device": device,
            "chunk_size": da3.get("chunk_size", 120),
            "overlap": da3.get("overlap", 60),
            "loop_chunk_size": 20,
            "loop_enable": da3.get("loop_enable", True),
            "infer_gs": da3.get("infer_gs", True),
            "useDBoW": False,
            "delete_temp_files": False,  # Keep .npy for origin traceability
            "align_lib": da3.get("align_lib", "numpy"),
            "align_method": da3.get("align_method", "sim3"),
            "scale_compute_method": "auto",
            "align_type": "dense",
            "ref_view_strategy": "saddle_balanced",
            "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": da3.get("depth_threshold", 15.0),
            "save_depth_conf_result": da3.get("save_depth_conf_result", True),
            "save_debug_info": False,
            "Sparse_Align": {
                "keypoint_select": "orb",
                "keypoint_num": 5000,
            },
            "IRLS": {
                "delta": 0.1,
                "max_iters": 5,
                "tol": 1e-9,
            },
            "Pointcloud_Save": {
                "sample_ratio": da3.get("sample_ratio", 1.0),
                "conf_threshold_coef": da3.get("conf_threshold_coef", 0.75),
            },
        },
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {
                "lang_version": "python",
                "max_iterations": 30,
                "lambda_init": 1e-6,
            },
        },
    }
    return cfg


def _build_vggt_config(config: dict) -> dict:
    """Load the tested stac_mapanything.yaml and override only user-configurable params."""
    import yaml as _yaml

    ma = config.get("reconstruction", {}).get("mapanything", config.get("mapanything", {}))

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
    sample_ratio = vggt_config["Model"]["Pointcloud_Save"]["sample_ratio"]

    # DA3 .npy files contain pickled Prediction objects from depth_anything_3
    # We need to temporarily add the module to sys.path for unpickling
    project_root = Path(__file__).resolve().parent.parent.parent
    da3_src = str(project_root / "vendor" / "depth-anything-3" / "src")
    _added_da3 = False
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
        _added_da3 = True

    for chunk_idx, npy_path in enumerate(chunk_npy_files):
        try:
            chunk_data = np.load(npy_path, allow_pickle=True).item()

            # Handle both DA3 Prediction objects and legacy VGGT-Long dicts
            if hasattr(chunk_data, 'depth'):
                # DA3 Prediction: compute point count from depth shape (N, H, W)
                depth = chunk_data.depth
                if depth.ndim == 3:
                    S, H, W = depth.shape
                else:
                    S, H, W = depth.shape[0], depth.shape[1], depth.shape[2]
            elif 'world_points' in chunk_data:
                # Legacy VGGT-Long dict
                wp = chunk_data['world_points']
                if wp.ndim == 5:
                    wp = wp[0]
                S, H, W = wp.shape[:3]
            else:
                pipe.send_log(f"Chunk {chunk_idx}: unrecognized data format", level="warning")
                continue
            HW = H * W
            total_points = S * HW

            # All points are kept (no confidence filtering — matches VGGT-Long behavior)
            all_indices = np.arange(total_points)

            # Apply sampling
            if sample_ratio < 1.0 and len(all_indices) > 0:
                n_samples = int(len(all_indices) * sample_ratio)
                sample_local = np.sort(np.random.choice(
                    len(all_indices), n_samples, replace=False
                ))
                final_indices = all_indices[sample_local]
            else:
                final_indices = all_indices

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
                "backend": vggt_config.get("_backend", "mapanything"),
                "ply_pre_aligned": True,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

        except Exception as e:
            pipe.send_log(f"Failed to generate origins for chunk {chunk_idx}: {e}", level="warning")
            import traceback
            traceback.print_exc()

    # Clean up temporary sys.path addition
    if _added_da3 and da3_src in sys.path:
        sys.path.remove(da3_src)


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager as multiprocessing target."""
    run_worker_safe(_map_work, conn, session_dir, config)
