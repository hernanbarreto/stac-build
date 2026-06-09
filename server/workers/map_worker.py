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

    # Replace mode (set by PipelineManager). When False, pre-existing frames/
    # artifacts (frame_quality.json, selected_frames.json) are reused as-is —
    # they belong to the outputs the user chose NOT to overwrite, so they must
    # not be re-computed either.
    replace = config.get("_pipeline_replace", True)

    # Ensure server/ is importable for frame_quality / frame_selector
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    # ── Step 1: Frame quality analysis (blur detection) ──
    # Toggleable via reconstruction.blur_filter (default ON). OFF = keep ALL frames,
    # no Laplacian cull, no frame_quality.json gating in Step 2.
    blur_on = bool(recon_cfg.get("blur_filter", True))
    fq_path = frames_dir / "frame_quality.json"
    if not blur_on:
        pipe.send_log("Blur filter OFF (reconstruction.blur_filter: false) — keeping ALL frames")
    elif not replace and fq_path.exists():
        pipe.send_log("Reusing existing frame_quality.json (replace=off — skipping blur analysis)")
    else:
        pipe.send_progress(2, "Analyzing frame quality...", stage="reconstruction")
        try:
            from frame_quality import analyze_frames, save_manifest
            fq = analyze_frames(str(frames_dir))
            if "error" not in fq:
                save_manifest(str(frames_dir), fq)
        except Exception as e:
            pipe.send_log(f"Frame quality analysis skipped: {e}", level="warning")

    # ── Step 2: Frame selection — ALWAYS produce frames/selected_frames.json, the SINGLE
    # source of truth for the processed frame set (read by SAM3 / scene_analyzer / TSDF /
    # da3 / mapanything backends). Mode = reconstruction.frames_selector:
    #   "dino"   → blur + DINO-cosine keyframes (params in config.frame_selection)
    #   "stride" → all blur-valid frames, uniform 1-of-N (mapanything.frame_stride)
    #   "none"   → all blur-valid frames (no decimation)
    # All three write the SAME file (none/stride write the FULL list explicitly), so the
    # frame set is always pinned and every backend consumes the same --selected_frames.
    selected_frames_path = None
    mode = str(recon_cfg.get("frames_selector", "none")).lower()
    sf_path = frames_dir / "selected_frames.json"
    if not replace and sf_path.exists():
        pipe.send_log("Reusing existing selected_frames.json (replace=off)")
    elif mode == "dino":
        from frame_selector import select_keyframes
        try:
            pipe.send_progress(5, "Selecting keyframes (blur + DINO cosine)...", stage="reconstruction")
            sel = select_keyframes(str(frames_dir), config.get("frame_selection", {}))
            pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes (dino)")
        except Exception as e:
            pipe.send_log(f"DINO frame selection failed: {e}", level="warning")
    else:
        # "stride" or "none": write the FULL blur-valid set, optionally strided. Writing
        # it explicitly (vs leaving it unset) keeps selected_frames.json the single source.
        try:
            if blur_on:
                from frame_selector import _load_valid_frame_list
                _files = _load_valid_frame_list(frames_dir)
            else:
                # blur OFF → ALL frames on disk, no quality cull.
                _files = (glob.glob(str(frames_dir / "*.jpg"))
                          + glob.glob(str(frames_dir / "*.png")))
            valid = sorted(_files,
                           key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
            valid = [os.path.basename(f) for f in valid]
            stride = (int(recon_cfg.get("mapanything", {}).get("frame_stride", 1) or 1)
                      if mode == "stride" else 1)
            chosen = valid[::stride] if stride > 1 else valid
            with open(sf_path, "w") as _f:
                json.dump({"version": "2.0",
                           "method": (f"stride_{stride}" if mode == "stride" else "none"),
                           "total_frames": len(valid), "selected_count": len(chosen),
                           "selected_files": chosen}, _f)
            pipe.send_log(f"Frame set: {len(chosen)}/{len(valid)} frames "
                          f"({mode}{f', stride {stride}' if mode == 'stride' else ''}, "
                          f"{'blur-valid' if blur_on else 'no blur'}) → selected_frames.json")
        except Exception as e:
            pipe.send_log(f"Could not build frame list: {e}", level="warning")
    if sf_path.exists():
        selected_frames_path = str(sf_path)
        pipe.send_log(f"Using frames from {sf_path}")

    # ── Step 2b: DA3-dense fusion frame set ──
    # The asymmetric design feeds DA3 the FULL blur-valid set (a superset of the VGGT
    # keyframes) so it produces per-frame depth for every sharp frame → the TSDF fuses
    # that dense set (DENSITY win), while VGGT/MapAnything still reconstructs only the
    # keyframes for the loop-closed poses. "full" = all blur-valid, NOT all frames on
    # disk (excludes the blurry ones) and NOT the keyframe decimation.
    da3_dense_frames_path = None
    try:
        if blur_on:
            from frame_selector import _load_valid_frame_list
            _valid = _load_valid_frame_list(frames_dir)  # blur-valid basenames
            _dpath = frames_dir / "da3_frames.json"
            with open(_dpath, "w") as _f:
                json.dump({"version": "2.0", "method": "blur_valid_dense",
                           "total_frames": len(_valid), "selected_count": len(_valid),
                           "selected_files": sorted(_valid,
                               key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))}, _f)
            da3_dense_frames_path = str(_dpath)
            pipe.send_log(f"DA3-dense frame set: {len(_valid)} blur-valid frames "
                          f"(excludes blurry) → da3_frames.json")
        else:
            pipe.send_log("blur_filter OFF → DA3 dense over ALL frames on disk")
    except Exception as e:
        pipe.send_log(f"Could not build DA3-dense frame list ({e}) — DA3 over all frames",
                      level="warning")

    # ── Step 3: Dispatch to backend ──
    if backend == "da3":
        _run_da3(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
    elif backend == "lidar":
        _run_lidar_only(pipe, frames_dir, output_dir, recon_cfg, session_path, selected_frames_path)
    elif backend == "hybrid":
        _run_hybrid_or_lidar(
            pipe, frames_dir, output_dir, selected_frames_path,
            recon_cfg, config, session_path, mode=backend
        )
    elif backend == "hybrid_cond":
        # Stray → DA3 (cam_enc pose conditioning + LiDAR depth calibration) → MapAnything
        # with the FULL prior (depth + intrinsics + poses). MapAnything still loop-closes.
        _run_mapanything(pipe, frames_dir, output_dir, selected_frames_path,
                         recon_cfg, config, session_path=session_path, cond=True)
    else:
        _run_mapanything(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)


def _find_stray_dir(session_path: Path) -> Path:
    """Find the Stray Scanner raw data dir (odometry.csv + depth/). Checks the session
    dir, a ``stray/`` subdir of it, and the same in sibling dirs (Stray data may sit in
    src_default/, src_default/stray/, or a sibling)."""
    def _ok(d: Path) -> bool:
        return (d / "odometry.csv").exists() and (d / "depth").is_dir()
    for cand in (session_path, session_path / "stray"):
        if _ok(cand):
            return cand
    parent = session_path.parent
    for child in parent.iterdir():
        if not child.is_dir() or child.name == session_path.name:
            continue
        for cand in (child, child / "stray"):
            if _ok(cand):
                return cand
    return None


def _run_lidar_only(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                    recon_cfg: dict, session_path: Path,
                    selected_frames_path: str = None):
    """Pure LiDAR reconstruction — backproject LiDAR depth maps with ARKit poses.

    No DA3, no neural inference. Uses native LiDAR (192x256) + odometry.csv.
    Only processes keyframes (from selected_frames.json) if available.
    Generates chunk_999_lidar.ply + chunk_999_lidar_origins.npz.
    CloudCompPy handles cleanup and traceability injection.
    """
    import numpy as np
    import cv2

    lidar_cfg = recon_cfg.get("lidar", {})

    # ── Find Stray Scanner data ──
    stray_dir = _find_stray_dir(session_path)
    if stray_dir is None:
        raise FileNotFoundError(
            f"Backend 'lidar' requires Stray Scanner data (depth/, odometry.csv), "
            f"but not found in {session_path} or siblings."
        )

    n_depth_files = len(list((stray_dir / 'depth').glob('*.png')))
    pipe.send_log(f"Stray Scanner data: {stray_dir.name}/ ({n_depth_files} depth frames)")
    pipe.send_progress(5, "Loading Stray Scanner data...", stage="reconstruction")

    # ── Load Stray Scanner data ──
    from ingestors.stray_scanner import prepare_stray_data

    stray = prepare_stray_data(
        data_dir=str(stray_dir),
        frames_output_dir=str(frames_dir),
        stride=lidar_cfg.get("stride", 4),
        max_frames=0,
        confidence_threshold=lidar_cfg.get("confidence_threshold", 1),
    )

    K = stray['intrinsics']
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # ── Filter to keyframes only ──
    all_frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    if selected_frames_path and Path(selected_frames_path).exists():
        with open(selected_frames_path) as f:
            sf_data = json.load(f)
        keyframe_names = set(sf_data.get("selected_files", []))
        frame_files = [fp for fp in all_frame_files if fp.name in keyframe_names]
        pipe.send_log(f"Using {len(frame_files)}/{len(all_frame_files)} keyframes")
    else:
        frame_files = all_frame_files
        pipe.send_log(f"No keyframe filter — using all {len(frame_files)} frames")

    # Build lookup: frame filename → stray index (for depth/pose access)
    stray_idx_map = {}  # frame_global_idx → stray array index
    for si, fidx in enumerate(stray['frame_indices']):
        stray_idx_map[fidx] = si

    n = len(frame_files)
    pipe.send_log(f"Backprojecting {n} LiDAR frames with ARKit poses")
    pipe.send_progress(10, f"Backprojecting {n} frames...", stage="reconstruction")

    # Scale factors: depth → RGB resolution for traceability
    rgb_h, rgb_w = stray['rgb_shape']
    depth_h, depth_w = stray['depth_shape']
    px_scale_y = rgb_h / depth_h
    px_scale_x = rgb_w / depth_w
    pipe.send_log(f"Pixel scale: depth({depth_w}x{depth_h}) → RGB({rgb_w}x{rgb_h}) = {px_scale_x:.1f}x")

    all_pts, all_cols = [], []
    all_fg, all_pr, all_pc = [], [], []
    all_conf = []
    skipped = 0

    for i, fp in enumerate(frame_files):
        # Extract frame index from filename (e.g., "000123.jpg" → 123)
        frame_global_idx = int(fp.stem)
        si = stray_idx_map.get(frame_global_idx)
        if si is None:
            skipped += 1
            continue  # No depth/pose for this frame

        depth = stray['depths'][si]
        raw_conf = stray.get('conf_masks', [None] * len(stray['depths']))[si]
        c2w = stray['poses'][si]

        rgb = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
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

        # Colors from full-res RGB
        rgb_rows = np.clip((v[valid] * px_scale_y).astype(int), 0, rgb_h - 1)
        rgb_cols = np.clip((u[valid] * px_scale_x).astype(int), 0, rgb_w - 1)
        all_cols.append(rgb[rgb_rows, rgb_cols].astype(np.uint8))

        # Confidence [0,1,2] → [0.0, 0.5, 1.0]
        if raw_conf is not None:
            all_conf.append(raw_conf[valid].astype(np.float32) / 2.0)
        else:
            all_conf.append(np.ones(valid.sum(), dtype=np.float32))

        # Traceability: real frame index + RGB-resolution pixel coords
        all_fg.append(np.full(valid.sum(), frame_global_idx, dtype=np.float32))
        all_pr.append(v[valid].astype(np.float32) * px_scale_y)
        all_pc.append(u[valid].astype(np.float32) * px_scale_x)

        if (i + 1) % 50 == 0 or i == n - 1:
            pct = 10 + (i / n) * 80
            pipe.send_progress(pct, f"Frame {i+1}/{n}", stage="reconstruction")

    if skipped > 0:
        pipe.send_log(f"Skipped {skipped} frames (no depth/pose data)")

    if not all_pts:
        raise RuntimeError("No valid points generated from LiDAR backprojection")

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    confs = np.concatenate(all_conf)
    fg = np.concatenate(all_fg)
    pr = np.concatenate(all_pr)
    pc = np.concatenate(all_pc)

    pipe.send_log(f"Total: {len(pts):,} points")
    pipe.send_progress(92, "Saving LiDAR cloud...", stage="reconstruction")

    # ── Save PLY + origins ──
    chunk_ply = output_dir / "chunk_999_lidar.ply"
    origins_npz = output_dir / "chunk_999_lidar_origins.npz"

    np.savez_compressed(origins_npz, frame_global=fg, pixel_row=pr, pixel_col=pc,
                        confidence=confs.astype(np.float32))

    _n = len(pts)
    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                      ('r','u1'),('g','u1'),('b','u1'),
                      ('confidence','<f4')])
    vd = np.empty(_n, dtype=dtype)
    vd['x'], vd['y'], vd['z'] = pts[:,0], pts[:,1], pts[:,2]
    vd['r'], vd['g'], vd['b'] = cols[:,0], cols[:,1], cols[:,2]
    vd['confidence'] = confs

    with open(chunk_ply, 'wb') as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {_n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"property float confidence\n"
                f"end_header\n".encode('ascii'))
        vd.tofile(f)

    size_mb = chunk_ply.stat().st_size / (1024 * 1024)
    pipe.send_log(f"LiDAR cloud: {_n:,} pts ({size_mb:.0f} MB) → {chunk_ply.name}")
    pipe.send_progress(100, "LiDAR reconstruction complete", stage="reconstruction")


def _run_da3(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
             selected_frames_path: str, recon_cfg: dict, config: dict, depth_only: bool = False,
             cond_stray_dir: str = None):
    """Run DA3 Streaming as subprocess. depth_only=True: just produce per-frame
    results_output/frame_*.npz (depth+conf+intrinsics) for MapAnything priors and skip
    the postprocess (no DA3 cloud/chunks). cond_stray_dir set (hybrid_cond): DA3 is
    conditioned on ARKit poses via cam_enc + its depth calibrated to LiDAR. Returns the
    da3 save dir."""
    import yaml

    da3_cfg = recon_cfg.get("da3", {})
    device = recon_cfg.get("device", "cpu")

    pipe.send_progress(8, "Generating DA3 config...", stage="reconstruction")

    # Build DA3 config YAML from our config.yaml settings
    da3_config = _build_da3_config(recon_cfg)
    # Sky removal via DA3's OWN sky head: ON for the standalone `da3` backend, OFF when
    # DA3 only feeds MapAnything priors (depth_only) — there MapAnything strips the sky
    # itself with skyseg. Toggle with reconstruction.da3.remove_sky (default True).
    da3_config["Model"]["remove_sky"] = (not depth_only) and bool(
        recon_cfg.get("da3", {}).get("remove_sky", True))

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

    # hybrid_cond: run_da3_main loads Stray from this dir and uses StrayDA3CondStreaming
    # (cam_enc pose conditioning + LiDAR depth calibration) instead of plain DA3.
    if cond_stray_dir:
        cmd.extend(["--cond-stray", str(cond_stray_dir)])

    # Set CUDA visibility and prevent CPU lockups
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"

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

    if depth_only:
        pipe.send_log("DA3 depth extraction complete (priors mode) — skipping postprocess")
        return da3_save_dir

    pipe.send_progress(85, "DA3 complete, post-processing...", stage="reconstruction")

    # ── Post-process DA3 output (same format as VGGT-Long) ──
    _postprocess_reconstruction(pipe, da3_save_dir, output_dir, da3_config, backend="da3")
    return da3_save_dir



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

    # ── Detect Stray Scanner data (searches session_path and siblings) ──
    stray_dir = _find_stray_dir(session_path)

    if stray_dir is None:
        if mode == "lidar":
            raise FileNotFoundError(
                f"Backend 'lidar' requires Stray Scanner data (depth/, odometry.csv), "
                f"but not found in {session_path} or siblings."
            )
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

    # Override session_path with the actual stray data location
    session_path = stray_dir
    n_depths = len(list((stray_dir / 'depth').glob('*.png')))
    pipe.send_log(f"Stray Scanner data found: {stray_dir.name}/ ({n_depths} depth frames)")
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
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"

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

    # ── Generate LiDAR cloud (hybrid: complement, lidar: primary) ──
    if mode in ("hybrid", "lidar"):
        label = "complement" if mode == "hybrid" else "primary"
        pipe.send_progress(92, f"Generating LiDAR {label} cloud...", stage="reconstruction")
        try:
            _generate_lidar_complement(pipe, da3_save_dir, output_dir, session_path,
                                       lidar_cfg, selected_frames_path)
        except Exception as e:
            pipe.send_log(f"LiDAR {label} generation failed: {e}", level="warning")
            import traceback
            traceback.print_exc()

    pipe.send_progress(100, f"{mode.upper()} reconstruction complete", stage="reconstruction")


def _generate_lidar_complement(pipe: WorkerPipe, da3_save_dir: Path, output_dir: Path,
                               session_path: Path, lidar_cfg: dict,
                               selected_frames_path: str = None):
    """Generate a LiDAR-only point cloud using DA3-streaming's refined poses.

    Backprojects raw LiDAR depth maps using camera_poses.txt from the
    DA3-streaming run (post-loop-closure). This cloud is saved as
    lidar_complement.ply in output/ for CloudCompPy to merge.

    CRITICAL: ``camera_poses.txt`` is indexed by the SAME keyframe set DA3
    processed. ``prepare_stray_data`` must therefore be restricted to those
    keyframes too — otherwise pose[i] gets paired with a different frame's
    depth (index misalignment) and the complement cloud lands in the wrong
    place. Hence ``selected_frames_path`` is threaded through.
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
        selected_frames_path=selected_frames_path,
    )

    K = stray['intrinsics']
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    # Use the frame list prepare_stray_data actually selected (keyframe-filtered
    # when selected_frames_path was given) so frame_files[i] / depths[i] /
    # poses[i] all index the same frame. Re-globbing the directory would pick
    # up every JPG and break the index alignment.
    frame_files = [Path(p) for p in stray['frame_paths']]
    n = min(len(stray['depths']), len(poses), len(frame_files))

    pipe.send_log(f"Backprojecting {n} LiDAR frames with DA3-streaming poses")

    # Compute scale factors: depth → RGB resolution for traceability
    rgb_h, rgb_w = stray['rgb_shape']
    depth_h, depth_w = stray['depth_shape']
    px_scale_y = rgb_h / depth_h  # e.g., 1440/192 = 7.5
    px_scale_x = rgb_w / depth_w  # e.g., 1920/256 = 7.5
    pipe.send_log(f"Pixel scale: depth({depth_w}x{depth_h}) → RGB({rgb_w}x{rgb_h}) = {px_scale_x:.1f}x")

    all_pts, all_cols = [], []
    all_fg, all_pr, all_pc = [], [], []
    all_conf = []

    for i in range(n):
        depth = stray['depths'][i]
        raw_conf = stray.get('conf_masks', [None] * n)[i]  # raw ARKit [0, 1, 2]
        
        c2w = poses[i]
        rgb = cv2.cvtColor(cv2.imread(str(frame_files[i])), cv2.COLOR_BGR2RGB)
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Keep all points where depth measurement exists (noise bounds)
        valid = depth > 0
        pts_cam = np.stack([
            (u[valid] - cx) * depth[valid] / fx,
            (v[valid] - cy) * depth[valid] / fy,
            depth[valid]
        ], axis=-1)
        pts_world = (pts_cam @ c2w[:3,:3].T) + c2w[:3,3]
        all_pts.append(pts_world.astype(np.float32))

        # Sample colors from full-res RGB at scaled pixel coordinates
        rgb_rows = np.clip((v[valid] * px_scale_y).astype(int), 0, rgb_h - 1)
        rgb_cols = np.clip((u[valid] * px_scale_x).astype(int), 0, rgb_w - 1)
        all_cols.append(rgb[rgb_rows, rgb_cols].astype(np.uint8))
        
        # Normalize ARKit confidence [0, 1, 2] -> [0.0, 0.5, 1.0]
        if raw_conf is not None:
            conf_val = raw_conf[valid].astype(np.float32) / 2.0
        else:
            conf_val = np.ones(valid.sum(), dtype=np.float32)
        all_conf.append(conf_val)
        
        # Traceability: real frame index + pixel coords in RGB resolution
        real_frame_idx = stray['frame_indices'][i]
        all_fg.append(np.full(valid.sum(), real_frame_idx, dtype=np.float32))
        all_pr.append((v[valid].astype(np.float32) * px_scale_y))
        all_pc.append((u[valid].astype(np.float32) * px_scale_x))

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    confs = np.concatenate(all_conf)
    fg = np.concatenate(all_fg)
    pr = np.concatenate(all_pr)
    pc = np.concatenate(all_pc)

    # Save as chunk_999_lidar.ply in output/
    complement_path = output_dir / "chunk_999_lidar.ply"
    origins_path = output_dir / "chunk_999_lidar_origins.npz"

    np.savez_compressed(origins_path, frame_global=fg, pixel_row=pr, pixel_col=pc,
                        confidence=confs.astype(np.float32))
    _n = len(pts)
    dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),
                      ('r','u1'),('g','u1'),('b','u1'),
                      ('confidence','<f4')])
    vd = np.empty(_n, dtype=dtype)
    vd['x'], vd['y'], vd['z'] = pts[:,0], pts[:,1], pts[:,2]
    vd['r'], vd['g'], vd['b'] = cols[:,0], cols[:,1], cols[:,2]
    vd['confidence'] = confs
    
    with open(complement_path, 'wb') as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {_n}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"property float confidence\n"
                f"end_header\n".encode('ascii'))
        vd.tofile(f)

    size_mb = complement_path.stat().st_size / (1024 * 1024)
    pipe.send_log(f"LiDAR complement: {_n:,} pts ({size_mb:.0f} MB) → {complement_path.name}")


def _run_mapanything(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                     selected_frames_path: str, recon_cfg: dict, config: dict,
                     session_path: Path = None, cond: bool = False):
    """Run VGGT-Long (MapAnything) as subprocess — legacy backend. cond=True (backend
    hybrid_cond): the DA3 priors are produced Stray-conditioned (ARKit poses via cam_enc
    + LiDAR-calibrated depth) and MapAnything gets the FULL prior (depth + K + poses)."""
    import yaml

    ma_cfg = recon_cfg.get("mapanything", config.get("mapanything", {}))
    device = recon_cfg.get("device", ma_cfg.get("device", "cpu"))

    pipe.send_progress(8, "Generating VGGT-Long config...", stage="reconstruction")

    vggt_config = _build_vggt_config(config)

    # hybrid_cond: force the full-prior path (poses too) regardless of config defaults.
    cond_stray_dir = None
    if cond:
        vggt_config["Model"]["da3_prior_use_poses"] = True
        if session_path is not None:
            try:
                _sd = _find_stray_dir(Path(session_path))
                cond_stray_dir = str(_sd) if _sd is not None else None
            except Exception as _e:
                pipe.send_log(f"hybrid_cond: Stray dir not found ({_e}) — DA3 runs "
                              f"image-only (no pose conditioning)", level="warning")

    # ── DA3 priors (multi-modal MapAnything) ── Opt-in. Run DA3 depth extraction first,
    # then feed its per-frame METRIC depth + intrinsics into MapAnything (poses are still
    # estimated by MapAnything — DA3 poses are intentionally NOT used). Image-only stays
    # the default. Consumed by MapAnythingAdapter.infer_chunk (base_model.py).
    if ma_cfg.get("use_da3_priors", False) or cond:
        try:
            # Resume: if DA3 depth was already extracted (e.g. a prior run that crashed
            # later in MapAnything), reuse it instead of re-running DA3 — unless replace.
            _replace = config.get("_pipeline_replace", True)
            _existing = output_dir / "da3_run" / "results_output"
            if not _replace and _existing.exists() and any(_existing.glob("frame_*.npz")):
                da3_dir = output_dir / "da3_run"
                pipe.send_log(f"DA3 priors already present ({_existing}) — skipping DA3 extraction")
            else:
                pipe.send_log("MapAnything DA3-priors mode: extracting DA3 metric depth first")
                # Run DA3 over ALL frames (selected_frames=None) → builds a metric-depth
                # dictionary keyed by real frame number for EVERY frame. MapAnything
                # reconstructs only the keyframes, but for each keyframe it always finds
                # its DA3 depth in that dictionary. DA3 stays the metric ANCHOR (only the
                # confident pixels, after the da3_prior_conf_percentile floor); MapAnything
                # infers the rest. Density comes from MapAnything's output, not DA3.
                # cond: pass the keyframe list so prepare_stray_data uses the frames
                # already on disk (no rgb.mp4 needed) and indexes depth/poses to them.
                # non-cond: the BLUR-VALID dense set (da3_frames.json, written by run()
                # Step 2b) → DA3 over all sharp frames (excludes blurry), the dense fusion
                # set for the TSDF. Read by PATH here because run()'s local is out of this
                # function's scope. Falls back to None (all frames on disk) if absent.
                _da3_dense = frames_dir / "da3_frames.json"
                _da3_dense_arg = str(_da3_dense) if _da3_dense.exists() else None
                da3_dir = _run_da3(pipe, frames_dir, output_dir,
                                   selected_frames_path if cond else _da3_dense_arg,
                                   recon_cfg, config, depth_only=True,
                                   cond_stray_dir=cond_stray_dir)
            priors_dir = Path(da3_dir) / "results_output"
            if priors_dir.exists() and any(priors_dir.glob("frame_*.npz")):
                vggt_config["Model"]["da3_priors_dir"] = str(priors_dir)
                pipe.send_log(f"DA3 priors → {priors_dir} (depth+intrinsics fed to MapAnything)")
            else:
                pipe.send_log("DA3 priors not produced — MapAnything runs image-only",
                              level="warning")
        except Exception as _e:
            pipe.send_log(f"DA3 priors extraction failed ({_e}) — MapAnything image-only",
                          level="warning")

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
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"

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


def _cleanup_recon_temps(save_dir: Path, output_dir: Path, backend: str, pipe: WorkerPipe):
    """Delete reconstruction temporaries that are dead once chunks + origins exist:
    maplong_run/{_tmp_results_unaligned,_tmp_results_loop,pcd} and (mapanything only)
    da3_run/da3_full except results_output (kept for the texture bake). KEEPS
    _tmp_results_aligned — the TSDF reads per-frame depth from it.

    Idempotent (skips what's already gone), so it is safe — and now called — on BOTH the
    normal path AND the resume early-exit. Previously the resume path returned before the
    cleanup, so temps (tens of GB) accumulated across resumed runs and never got freed."""
    for tmp_dir_name in ["_tmp_results_unaligned", "_tmp_results_loop", "pcd"]:
        tmp_dir = save_dir / tmp_dir_name
        if tmp_dir.exists():
            size_mb = sum(f.stat().st_size for f in tmp_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            pipe.send_log(f"Cleaned up {tmp_dir_name}/ ({size_mb:.0f} MB freed)")

    # DA3-priors cleanup: in the mapanything backend, da3_run/ holds ONLY the consumed
    # DA3 depth priors (+ DA3's own intermediate cloud) — MapAnything already ingested
    # them and nothing downstream needs them → delete (can be tens of GB). NOT for the
    # da3 backend, where da3_run IS the reconstruction output.
    if backend == "mapanything":
        for _d in ("da3_run", "da3_full"):
            prior_dir = output_dir / _d
            if not prior_dir.exists():
                continue
            # KEEP results_output/ (per-frame DA3 depth) — the vertex_gpu photo bake uses
            # it as the occlusion oracle (nvdiffrast_bake reads da3_run/results_output).
            freed = 0
            for child in prior_dir.iterdir():
                if child.name == "results_output":
                    continue
                try:
                    if child.is_dir():
                        freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        freed += child.stat().st_size
                        child.unlink()
                except OSError:
                    pass
            pipe.send_log(f"Cleaned up {_d}/ (kept results_output for texture bake; "
                          f"{freed / (1024 * 1024):.0f} MB freed)")


def _postprocess_reconstruction(pipe: WorkerPipe, save_dir: Path, output_dir: Path,
                                 run_config: dict, backend: str = "da3"):
    """Post-process reconstruction output (shared by DA3 and MapAnything).
    
    Both backends produce identical output layout:
      save_dir/pcd/N_pcd.ply, camera_poses.txt, intrinsic.txt
    """
    # ── Copy PLY files to output dir ──
    pcd_dir = save_dir / "pcd"

    def _chunk_idx(p):
        # Files are "<N>_pcd.ply" with N the chunk number. Sort NUMERICALLY — plain
        # sorted() is lexicographic ("10_pcd" < "1_pcd") and scrambles the chunk order
        # (chunk_001 ← 10_pcd, chunk_011 ← 1_pcd …), mismatching poses/frame ranges.
        stem = Path(p).name.split("_", 1)[0]
        return int(stem) if stem.isdigit() else (1 << 30)

    ply_files = (sorted(glob.glob(str(pcd_dir / "*_pcd.ply")), key=_chunk_idx)
                 if pcd_dir.exists() else [])
    ply_files = [f for f in ply_files if "combined" not in Path(f).name]
    if not ply_files:
        # No new chunk PLYs. EXPECTED when VGGT-Long early-exited a resume
        # (camera_poses.txt already present → reconstruction already complete) or when
        # the cascade cleanup already removed pcd/. If the products already exist there
        # is nothing to post-process → skip gracefully instead of crashing. To force a
        # full rebuild, reconstruct WITH replace (clears camera_poses → VGGT-Long re-runs).
        already_done = ((output_dir / "cleaned_cloud.ply").exists()
                        or any(output_dir.glob("chunk_*.ply"))
                        or (output_dir / "camera_poses.txt").exists()
                        or (save_dir / "camera_poses.txt").exists())
        if already_done:
            pipe.send_log("Reconstruction already complete (no new chunks produced) — "
                          "skipping post-process. Use replace to force a full rebuild.")
            # Complete origins if a prior run crashed mid-generation (e.g. on disk): the
            # cloud has N chunk_*.ply but fewer chunk_*_origins.npz → CloudComPy drops the
            # size-mismatched origins → lost traceability (frame_global) → TSDF can't mask
            # to the cloud. Regenerate BEFORE the cleanup, which deletes pcd/ (the source).
            _n_ply = len(list(output_dir.glob("chunk_*.ply")))
            _n_org = len(list(output_dir.glob("chunk_*_origins.npz")))
            if _n_ply and _n_org < _n_ply:
                pipe.send_log(f"Origins incomplete ({_n_org}/{_n_ply}) — regenerating "
                              f"before cleanup")
                run_config["_backend"] = backend
                try:
                    _generate_origins(save_dir, output_dir, run_config, pipe)
                except Exception as _e:
                    pipe.send_log(f"Origins regeneration failed ({_e})", level="warning")
            # Still reclaim dead temporaries on a resume — they accumulate (tens of GB)
            # across resumed runs because this path used to return before any cleanup.
            _cleanup_recon_temps(save_dir, output_dir, backend, pipe)
            pipe.send_progress(100, f"{backend.upper()} already complete", stage="reconstruction")
            return
        raise FileNotFoundError(f"No chunk PLY files found in {pcd_dir}")

    pipe.send_log(f"Found {len(ply_files)} chunk PLYs")

    for i, ply_src in enumerate(ply_files):
        ply_dst = output_dir / f"chunk_{i:03d}.ply"
        shutil.copyfile(ply_src, ply_dst)
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
                shutil.copyfile(gs_src, gs_dst)
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
            shutil.copyfile(src, output_dir / dst_name)

    # Real-frame pose traceability: keep camera_poses.txt under output/ AND emit
    # camera_frames.txt mapping each pose line -> REAL frame number (from
    # frame_list.json, the exact ordered frames the backend processed). The TSDF
    # then keys poses by real frame number, matching the per-point frame_global and
    # the per-frame depth loader. Only present for backends that write frame_list.json.
    _cp_txt = save_dir / "camera_poses.txt"
    _fl = save_dir / "frame_list.json"
    if _cp_txt.exists():
        shutil.copyfile(_cp_txt, output_dir / "camera_poses.txt")
    if _fl.exists():
        try:
            import re as _re
            _names = json.loads(_fl.read_text())
            _nums = []
            for _n in _names:
                _m = _re.search(r"(\d+)", str(_n))
                _nums.append(str(int(_m.group(1))) if _m else "-1")
            (output_dir / "camera_frames.txt").write_text("\n".join(_nums) + "\n")
            shutil.copyfile(_fl, output_dir / "frame_list.json")
            pipe.send_log(f"Wrote camera_frames.txt ({len(_nums)} frames) for real-frame "
                          f"pose/depth traceability")
        except Exception as _e:
            pipe.send_log(f"Could not write camera_frames.txt: {_e}", level="warning")

    pipe.send_progress(100, f"{backend.upper()} reconstruction complete", stage="reconstruction")
    pipe.send_log(f"{backend} complete: {len(ply_files)} chunks")

    # Cascade cleanup (step 1/3): chunks were copied to output/chunk_NNN.ply and origins
    # generated from _tmp_results_aligned, so the unaligned/loop/pcd temps + the consumed
    # DA3 priors are dead → free them (keeps _tmp_results_aligned for the TSDF depth).
    _cleanup_recon_temps(save_dir, output_dir, backend, pipe)

    import gc
    gc.collect()


def _build_da3_config(recon_cfg: dict) -> dict:
    """Build DA3 Streaming config YAML from our reconstruction config."""
    da3 = recon_cfg.get("da3", {})
    device = recon_cfg.get("device", "cpu")

    # SALAD weights (DINO-SALAD, used by DA3's loop detector). Search known
    # locations in order; the real file lives in weights/da3/ on this pod.
    project_root = Path(__file__).resolve().parent.parent.parent
    _salad_candidates = [
        project_root / "weights" / "da3" / "dino_salad.ckpt",
        project_root / "weights" / "dino_salad.ckpt",
        project_root / "vendor" / "depth-anything-3" / "da3_streaming" / "weights" / "dino_salad.ckpt",
        project_root / "vendor" / "VGGT-Long" / "weights" / "dino_salad.ckpt",
    ]
    salad_path = next((p for p in _salad_candidates if p.exists()), _salad_candidates[0])
    if not salad_path.exists():
        print(f"[map_worker] ⚠️ dino_salad.ckpt not found in any known location; "
              f"loop closure will fail. Looked in: {[str(p) for p in _salad_candidates]}")

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
                # String on purpose: the vendored sim3utils does eval(config[...]["tol"]),
                # so it must round-trip through YAML as a string, not a float.
                "tol": "1e-9",
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
                # String on purpose: vendored sim3loop does eval(config[...]["lambda_init"]).
                "lambda_init": "1e-6",
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
    # Stride is applied UPFRONT now (Step 2 bakes it into selected_frames.json), so
    # VGGT-Long must NOT stride again — it processes exactly the list it's given.
    cfg["Model"]["frame_stride"] = 1
    cfg["Model"]["delete_temp_files"] = False  # Keep .npy for origin traceability

    pc = cfg["Model"].get("Pointcloud_Save", {})
    pc["sample_ratio"] = ma.get("sample_ratio", pc.get("sample_ratio", 1.0))
    pc["conf_threshold_coef"] = ma.get("conf_threshold_coef", pc.get("conf_threshold_coef", 0.75))
    cfg["Model"]["Pointcloud_Save"] = pc

    # Confidence floors (consumed in base_models/base_model.py). Both configurable from
    # config.yaml — nothing hardcoded. da3_prior_conf_percentile filters the DA3 metric
    # depth prior per frame; map_conf_percentile is MapAnything's own inference floor.
    cfg["Model"]["da3_prior_conf_percentile"] = ma.get("da3_prior_conf_percentile", 0)
    cfg["Model"]["map_conf_percentile"] = ma.get("map_conf_percentile", 10)
    # "Full prior" path (hybrid_cond): also feed DA3's per-frame poses (camera_poses) to
    # MapAnything, not just depth+K. Off by default — only meaningful when the DA3 priors
    # were produced with real ARKit pose conditioning (Stray + StrayDA3CondStreaming).
    cfg["Model"]["da3_prior_use_poses"] = bool(ma.get("da3_prior_use_poses", False))

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
    """Generate chunk_NNN_origins.npz (frame_global, pixel_row, pixel_col,
    confidence) aligned 1:1 with each chunk PLY's points.

    DA3 writes each chunk PLY (K_pcd.ply) as the *confidence-thresholded* subset
    of its points, in frame-major order (see save_confident_pointcloud_batch:
    mask = conf >= mean(conf)*coef & conf > 1e-5). We replicate that EXACT mask on
    the chunk's saved Prediction (.npy) so the origins line up point-for-point
    with the PLY, and carry the per-point confidence. CloudComPy injects these as
    scalar fields at merge time and preserves them through dedup/voxel/SOR into
    cleaned_cloud.ply (so every final point is traceable to its keyframe + pixel
    + confidence).

    `frame_global` is the REAL frame number (numeric part of the original filename),
    resolved from the processed-list position via frame_list.json (written by
    VGGT-Long with the exact ordered frames it used, after any keyframe filter +
    stride). This keeps per-point traceability correct under stride/keyframe subsets.
    Falls back to the raw processed-list index if frame_list.json is absent (legacy).

    Legacy MapAnything/VGGT-Long chunks (dicts with 'world_points', no per-point
    confidence) keep the original all-points behavior.
    """
    import numpy as np

    # Read ALIGNED chunks (full dict; only world_points is sim3-transformed,
    # depth/conf/intrinsic/world_points_conf are identical to unaligned). Unaligned
    # is deleted incrementally during alignment to avoid keeping 2x copies on disk.
    chunks_dir = vggt_save_dir / "_tmp_results_aligned"
    if not chunks_dir.exists():
        chunks_dir = vggt_save_dir / "_tmp_results_unaligned"   # legacy fallback
    pcd_dir = vggt_save_dir / "pcd"
    if not chunks_dir.exists():
        pipe.send_log("No chunk data found — origins not generated", level="warning")
        return

    # _postprocess copied save_dir/pcd/K_pcd.ply (K = real chunk number), sorted
    # lexicographically, to output/chunk_{i:03d}.ply. Mirror that ordering so
    # chunk_{i:03d}_origins.npz pairs with chunk_{i:03d}.ply — but use the REAL
    # chunk number K for frame_global (lexicographic sort puts "10" before "2").
    def _pcd_num(p):
        # MUST match the NUMERIC ordering _postprocess uses to copy N_pcd.ply →
        # chunk_{i:03d}.ply, so chunk_{i:03d}_origins pairs with chunk_{i:03d}.ply.
        # (Plain sorted() is lexicographic → origins would pair with the wrong chunk.)
        s = Path(p).name.split("_", 1)[0]
        return int(s) if s.isdigit() else (1 << 30)
    pcd_files = sorted(glob.glob(str(pcd_dir / "*_pcd.ply")), key=_pcd_num)
    pcd_files = [f for f in pcd_files if "combined" not in Path(f).name]
    if not pcd_files:
        pipe.send_log("No chunk PLYs found — origins not generated", level="warning")
        return

    chunk_size = vggt_config["Model"]["chunk_size"]
    overlap = vggt_config["Model"]["overlap"]
    chunk_step = chunk_size - overlap

    # STAC: map img_list position -> REAL frame number (numeric part of the original
    # filename), so frame_global stays correct under stride / keyframe subsetting.
    # frame_list.json is written by VGGT-Long with the exact ordered frames it used.
    import re as _re
    frame_numbers = None
    _flp = vggt_save_dir / "frame_list.json"
    if _flp.exists():
        try:
            _names = json.loads(_flp.read_text())
            _fn = []
            for _nm in _names:
                _m = _re.search(r"(\d+)", str(_nm))
                _fn.append(int(_m.group(1)) if _m else -1)
            frame_numbers = np.asarray(_fn, dtype=np.int64)
            pipe.send_log(f"Origins: frame_global mapped to real frame numbers "
                          f"via frame_list.json ({len(frame_numbers)} frames)")
        except Exception as _e:
            pipe.send_log(f"Origins: frame_list.json unreadable ({_e}) — "
                          f"frame_global will be the processed-list index", level="warning")
    ps = vggt_config["Model"].get("Pointcloud_Save", {})
    coef = ps.get("conf_threshold_coef", 0.75)
    sample_ratio = ps.get("sample_ratio", 1.0)

    if sample_ratio < 1.0:
        pipe.send_log(
            f"sample_ratio={sample_ratio} (<1.0): DA3 randomly subsamples each PLY, "
            f"so per-point origins can't be reproduced exactly — set "
            f"reconstruction.da3.sample_ratio: 1.0 for full traceability.",
            level="warning")

    # DA3 .npy files contain pickled Prediction objects from depth_anything_3
    project_root = Path(__file__).resolve().parent.parent.parent
    da3_src = str(project_root / "vendor" / "depth-anything-3" / "src")
    _added_da3 = da3_src not in sys.path
    if _added_da3:
        sys.path.insert(0, da3_src)

    try:
        for i, pf in enumerate(pcd_files):
            try:
                try:
                    K = int(Path(pf).stem.split("_")[0])  # "10_pcd" -> 10
                except ValueError:
                    K = i  # fallback

                # STAC: prefer the INLINE origins VGGT-Long wrote with the SAME mask as
                # the PLY (guaranteed 1:1 — no cross-process float-boundary drift that
                # made CloudComPy drop origins → lost confidence + TSDF traceability).
                # Falls through to the re-derivation below for DA3 / legacy runs.
                inline = pcd_dir / f"{K}_origins.npz"
                if inline.exists():
                    z = np.load(inline)
                    n = int(len(z["frame_global"]))
                    out_ply = output_dir / f"chunk_{i:03d}.ply"
                    ply_n = _read_ply_point_count(out_ply) if out_ply.exists() else None
                    if ply_n is not None and ply_n != n:
                        pipe.send_log(f"Chunk {i:03d} (src {K}): inline origins {n} vs PLY "
                                      f"{ply_n} — mismatch (will be dropped at merge)", level="warning")
                    np.savez_compressed(output_dir / f"chunk_{i:03d}_origins.npz",
                                        **{k: z[k] for k in z.files})
                    with open(output_dir / f"chunk_{i:03d}_meta.json", "w") as f:
                        json.dump({"chunk_id": i, "source_chunk": int(K),
                                   "n_points": n, "chunk_step": int(chunk_step)}, f)
                    pipe.send_log(f"Saved origins chunk_{i:03d} (src {K}, inline 1:1): {n} pts")
                    continue

                npy_path = chunks_dir / f"chunk_{K}.npy"
                if not npy_path.exists():
                    pipe.send_log(f"Chunk {i:03d}: missing {npy_path.name}; origins skipped", level="warning")
                    continue

                chunk_data = np.load(npy_path, allow_pickle=True).item()

                conf_flat = None
                mapany_thr = None   # exact VGGT-Long threshold for the mapanything branch
                if hasattr(chunk_data, "conf") and getattr(chunk_data, "conf") is not None:
                    # DA3 Prediction — replicate the PLY's confidence mask
                    conf = np.asarray(chunk_data.conf, dtype=np.float32)
                    if conf.ndim == 4:
                        conf = conf.reshape(conf.shape[0], conf.shape[-2], conf.shape[-1])
                    S, H, W = conf.shape
                    conf_flat = conf.reshape(-1)
                elif hasattr(chunk_data, "depth"):
                    depth = np.asarray(chunk_data.depth)
                    S, H, W = depth.shape[0], depth.shape[-2], depth.shape[-1]
                elif isinstance(chunk_data, dict) and "world_points" in chunk_data:
                    wp = chunk_data["world_points"]
                    if wp.ndim == 5:
                        wp = wp[0]
                    S, H, W = wp.shape[:3]
                    # MapAnything/VGGT-Long: the PLY is conf-filtered by
                    # world_points_conf (save_confident_pointcloud_batch). Replicate
                    # that SAME mask below so origins line up 1:1 with the PLY points
                    # (otherwise CloudComPy drops the size-mismatched origins → no
                    # traceability). conf is pose-invariant, so the unaligned chunk's
                    # world_points_conf matches the aligned PLY's.
                    wpc = chunk_data.get("world_points_conf")
                    if wpc is None:
                        # da3 aligned chunk dicts store the per-point conf under "conf"
                        # (da3_streaming.py: aligned_chunk_data["conf"] = chunk_data.conf),
                        # and the PLY is masked with that SAME array. mapanything uses
                        # "world_points_conf". Read whichever the backend wrote.
                        wpc = chunk_data.get("conf")
                    if wpc is not None:
                        _wpc = np.asarray(wpc).reshape(-1)
                        # Threshold EXACTLY as VGGT-Long computes it: mean over the RAW
                        # dtype (float16/float32), NOT cast to float32 first. The float32
                        # cast shifted the mean → ±N points flipped at the boundary →
                        # origin/PLY count mismatch → CloudComPy dropped ALL origins (no
                        # confidence, no TSDF traceability). conf_flat stays float32 for
                        # the mask comparison itself (matches save_confident_pointcloud_batch).
                        mapany_thr = float(np.mean(_wpc)) * coef
                        conf_flat = _wpc.astype(np.float32)
                else:
                    pipe.send_log(f"Chunk {i:03d} (src {K}): unrecognized data format", level="warning")
                    continue

                HW = H * W

                if conf_flat is not None:
                    # Exact replica of save_confident_pointcloud_batch's mask. Use the
                    # raw-dtype threshold for mapanything (mapany_thr) so the count matches.
                    thr = mapany_thr if mapany_thr is not None else float(np.mean(conf_flat)) * coef
                    surviving = np.flatnonzero((conf_flat >= thr) & (conf_flat > 1e-5))
                    confidence = conf_flat[surviving].astype(np.float32)
                else:
                    # Legacy (MapAnything/VGGT-Long): keep all points, no confidence.
                    surviving = np.arange(S * HW)
                    confidence = None

                if len(surviving) == 0:
                    pipe.send_log(f"Chunk {i:03d} (src {K}): no points after conf mask", level="warning")
                    continue

                frame_local = surviving // HW
                within = surviving % HW
                pixel_row = (within // W).astype(np.int16)
                pixel_col = (within % W).astype(np.int16)
                abs_idx = frame_local + K * chunk_step          # position in processed img_list
                if frame_numbers is not None:
                    safe = np.clip(abs_idx, 0, len(frame_numbers) - 1)
                    if np.any(abs_idx >= len(frame_numbers)) or np.any(abs_idx < 0):
                        pipe.send_log(f"Chunk {i:03d} (src {K}): frame index out of "
                                      f"frame_list range — clipped", level="warning")
                    frame_global = frame_numbers[safe].astype(np.int32)
                else:
                    frame_global = abs_idx.astype(np.int32)

                # Sanity: must match the copied chunk_{i:03d}.ply point count, or
                # CloudComPy will drop the (size-mismatched) origins on merge.
                out_ply = output_dir / f"chunk_{i:03d}.ply"
                ply_n = _read_ply_point_count(out_ply) if out_ply.exists() else None
                if ply_n is not None and ply_n != len(surviving):
                    pipe.send_log(
                        f"Chunk {i:03d} (src {K}): origin/PLY count mismatch "
                        f"({len(surviving)} vs {ply_n}) — origins will be dropped at merge. "
                        f"Check conf mask / sample_ratio.", level="warning")

                save_kw = dict(
                    frame_global=frame_global,
                    pixel_row=pixel_row,
                    pixel_col=pixel_col,
                    scaled_resolution=np.array([H, W], dtype=np.int32),
                )
                if confidence is not None:
                    save_kw["confidence"] = confidence
                np.savez_compressed(output_dir / f"chunk_{i:03d}_origins.npz", **save_kw)

                conf_tag = (f", conf[{confidence.min():.2f},{confidence.max():.2f}]"
                            if confidence is not None else " (no conf)")
                pipe.send_log(f"Saved origins chunk_{i:03d} (src chunk {K}): {len(surviving)} pts{conf_tag}")

                with open(output_dir / f"chunk_{i:03d}_meta.json", "w") as f:
                    json.dump({
                        "chunk_id": i,
                        "source_chunk": K,
                        "frame_count": int(S),
                        "scaled_resolution": [int(H), int(W)],
                        "chunk_step": int(chunk_step),
                        "frame_global_start": int(K * chunk_step),
                        "frame_global_end": int(K * chunk_step + S - 1),
                        "backend": vggt_config.get("_backend", "da3"),
                        "has_confidence": confidence is not None,
                        "ply_pre_aligned": True,
                    }, f)

            except Exception as e:
                pipe.send_log(f"Failed to generate origins for chunk {i:03d}: {e}", level="warning")
                import traceback
                traceback.print_exc()
    finally:
        if _added_da3 and da3_src in sys.path:
            sys.path.remove(da3_src)



# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager as multiprocessing target."""
    run_worker_safe(_map_work, conn, session_dir, config)
