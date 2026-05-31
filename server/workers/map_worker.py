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

    pose_source = recon_cfg.get("pose_source", "da3")

    # ── Step 1: Frame quality analysis (blur detection) ──
    # Gated by config frame_quality.enabled. In the ViPE prior-driven architecture
    # ViPE consumes ALL frames, so blur filtering is off by default.
    fq_path = frames_dir / "frame_quality.json"
    if not config.get("frame_quality", {}).get("enabled", True):
        pipe.send_log("Frame quality (blur) DISABLED in config — not running")
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

    # ── Step 2: Frame selection (DINO cosine novelty) ──
    # Gated by reconstruction.use_keyframes AND frame_selection.enabled. In the ViPE
    # prior-driven architecture ViPE + DA3 priors run on ALL frames, so keyframe
    # selection is off by default → selected_frames_path stays None (all-frames fusion).
    selected_frames_path = None
    use_keyframes = recon_cfg.get("use_keyframes", True)
    sf_path = frames_dir / "selected_frames.json"
    if not use_keyframes:
        pipe.send_log("Keyframe selection (cosine) DISABLED — running on ALL frames")
    else:
        if not replace and sf_path.exists():
            pipe.send_log("Reusing existing selected_frames.json (replace=off — skipping keyframe selection)")
        else:
            frame_sel_cfg = config.get("frame_selection", {})
            if frame_sel_cfg.get("enabled", False):
                from frame_selector import select_keyframes
                try:
                    pipe.send_progress(5, "Selecting keyframes...", stage="reconstruction")
                    sel = select_keyframes(str(frames_dir), frame_sel_cfg)
                    pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes")
                except Exception as e:
                    pipe.send_log(f"Frame selection failed: {e}", level="warning")
        # only pick up keyframes when keyframe mode is on (ignore any stale file otherwise)
        if sf_path.exists():
            selected_frames_path = str(sf_path)
            pipe.send_log(f"Using keyframes from {sf_path}")

    # ── Step 3: Dispatch to backend ──
    # pose_source == "vipe": ViPE is the SLAM (poses, all frames), DA3 isolated is
    # the metric depth ref; the cloud is fused on the cosine keyframes (selected_frames).
    if pose_source == "vipe":
        _run_vipe_slam(pipe, frames_dir, output_dir, recon_cfg, config, selected_frames_path)
    elif backend == "da3":
        _run_da3(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
    elif backend == "lidar":
        _run_lidar_only(pipe, frames_dir, output_dir, recon_cfg, session_path, selected_frames_path)
    elif backend == "hybrid":
        _run_hybrid_or_lidar(
            pipe, frames_dir, output_dir, selected_frames_path,
            recon_cfg, config, session_path, mode=backend
        )
    else:
        _run_mapanything(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)


def _find_stray_dir(session_path: Path) -> Path:
    """Find the Stray Scanner raw data directory (sibling of src_default/)."""
    if (session_path / "odometry.csv").exists() and (session_path / "depth").is_dir():
        return session_path
    parent = session_path.parent
    for child in parent.iterdir():
        if not child.is_dir() or child.name == session_path.name:
            continue
        if (child / "odometry.csv").exists() and (child / "depth").is_dir():
            return child
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
             selected_frames_path: str, recon_cfg: dict, config: dict,
             lean: bool = False):
    """Run DA3 Streaming as subprocess. ``lean`` = prior-source mode (we only need
    depth + poses): minimize the per-chunk PLY (sample_ratio↓), skip the PLY merge
    and the chunk→output copy, so no point-cloud garbage is generated."""
    import yaml

    da3_cfg = recon_cfg.get("da3", {})
    device = recon_cfg.get("device", "cpu")

    pipe.send_progress(8, "Generating DA3 config...", stage="reconstruction")

    # Build DA3 config YAML from our config.yaml settings
    da3_config = _build_da3_config(recon_cfg)
    if lean:
        # world_points are still computed (needed for SIM3 loop-closure alignment),
        # but we shrink the saved per-chunk PLY to near-nothing (can't fully disable
        # without patching the vendor; 0 is invalid, so use a tiny ratio).
        da3_config.setdefault("Model", {}).setdefault("Pointcloud_Save", {})["sample_ratio"] = 0.0005

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

    # Set CUDA visibility and prevent CPU lockups
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
    if lean:
        env["DA3_LEAN_PRIORS"] = "1"   # run_da3_main skips the combined-PLY merge

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

    if lean:
        # prior-source: we read da3_run depth+poses directly — no chunk→output copy.
        pipe.send_log("DA3 complete (lean prior-source: skipping PLY post-process)")
        pipe.send_progress(85, "DA3 streaming complete (priors)", stage="reconstruction")
        return

    pipe.send_progress(85, "DA3 complete, post-processing...", stage="reconstruction")

    # ── Post-process DA3 output (same format as VGGT-Long) ──
    _postprocess_reconstruction(pipe, da3_save_dir, output_dir, da3_config, backend="da3")


# ────────────────────────────────────────────────────────────────────
#  ViPE SLAM pose-source pipeline (pose_source == "vipe")
#  DA3(all frames, depth ref) → ViPE(all frames, poses+depth) →
#  per-frame calibrate ViPE→DA3 metric → compose cloud + traceability.
#  ViPE IS the SLAM. Cleans its own scratch — no temporaries/leftovers.
# ────────────────────────────────────────────────────────────────────

def _vipe_cfg(recon_cfg: dict) -> dict:
    v = recon_cfg.get("vipe", {})
    return {
        "venv": v.get("venv", "/workspace/stac-build/vendor/vipe/.venv"),
        "vipe_dir": v.get("vipe_dir", "/workspace/stac-build/vendor/vipe"),
        "pipeline": v.get("pipeline", "dav3"),
        "skip_if_exists": v.get("skip_if_exists", True),
    }


def _conda_root() -> str:
    return os.environ.get("CONDA_ROOT", "/workspace/miniforge3")


def _stream_proc(pipe: WorkerPipe, proc, tag: str):
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            pipe.send_log(f"[{tag}] {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{tag} failed (exit {proc.returncode})")


def _clean_da3_scratch(da3_dir: Path):
    """Remove the streaming scratch so nothing temporary is left behind."""
    for name in ("_tmp_results_unaligned", "_tmp_results_aligned",
                 "_tmp_results_loop", "pcd", "gs_ply"):
        shutil.rmtree(da3_dir / name, ignore_errors=True)


def _extract_da3_priors(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                        conf_percentile: float = 0.0):
    """Extract per-frame depth + c2w poses from the DA3 streaming run (da3_run/)
    into vipe_priors/ (depth/*.npy + poses.txt + frames.txt) for ViPE injection.
    conf_percentile > 0 zeroes-out the bottom N% lowest-confidence DA3 depth pixels
    per frame (so ViPE isn't corrupted by far/unreliable depth → no onion).
    We take ONLY DA3's depth + poses, not its chunks. Runs in the da3 env (numpy)."""
    server_dir = Path(__file__).resolve().parent.parent
    inner = (f"source {_conda_root()}/etc/profile.d/conda.sh && "
             f"conda activate {os.environ.get('DA3_CONDA_ENV', 'da3')} && "
             f"python -u {server_dir / 'reconstruction' / 'extract_da3_priors.py'} "
             f"--da3-run {output_dir / 'da3_run'} --frames-dir {frames_dir} "
             f"--out {output_dir / 'vipe_priors'} --conf-percentile {conf_percentile}")
    pipe.send_progress(40, "Extracting DA3 depth+poses → ViPE priors...", stage="reconstruction")
    proc = subprocess.Popen(["bash", "-lc", inner], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    _stream_proc(pipe, proc, "da3-priors")


def _clean_vipe_inputs(pipe: WorkerPipe, output_dir: Path):
    """After vipe_depth/ (metric ViPE depth npz, exported from ViPE's refined EXR)
    exists, the inputs are redundant: the raw ViPE EXR, the DA3 streaming run, and
    the injected priors. KEEP vipe_depth/ — compose AND TSDF read it. Keep poses +
    intrinsics + pose_scale."""
    targets = [
        output_dir / "vipe_run" / "depth" / "frames.zip",
        output_dir / "vipe_run" / "mask",
        output_dir / "da3_run",
        output_dir / "vipe_priors",
    ]
    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
            elif t.exists():
                t.unlink()
            pipe.send_log(f"[cleanup] removed reconstruction scratch: {t.name}")
        except Exception as e:
            pipe.send_log(f"[cleanup] could not remove {t}: {e}", level="warning")


def _write_stride_selection(frames_dir: Path, output_dir: Path, stride: int) -> str:
    """Write a selected_frames.json (selected_files = basenames) of 1-of-`stride`
    frames, sorted — the SAME set ViPE strides via FrameDirStream. Used to make DA3
    streaming process only those frames so DA3 priors and ViPE align 1:1."""
    import json
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    files = sorted({f for e in exts for f in
                    list(frames_dir.glob(f"*{e}")) + list(frames_dir.glob(f"*{e.upper()}"))})
    keep = [f.name for f in files[::stride]]
    out = output_dir / "stride_frames.json"
    with open(out, "w") as f:
        json.dump({"selected_files": keep, "selected": keep,
                   "total_frames": len(files), "stride": stride}, f)
    return str(out)


def _run_vipe_with_priors(pipe: WorkerPipe, frames_dir: Path, output_dir: Path, vcfg: dict,
                          inject_poses: bool = False, stride: int = 1):
    """Run ViPE fed with per-frame priors from vipe_priors/. ALWAYS injects metric
    DEPTH (anchors scale in the BA). POSES are injected ONLY for stray/lidar
    (inject_poses=True) — in monocular (da3) we let ViPE solve poses freely, because
    DA3's poses can drag ViPE toward their own drift. Output → vipe_run/."""
    vipe_out = output_dir / "vipe_run"
    if vcfg["skip_if_exists"] and (vipe_out / "pose" / "frames.npz").exists():
        pipe.send_log("[vipe] reusing existing vipe_run/pose/frames.npz")
        return
    vipe_out.mkdir(parents=True, exist_ok=True)
    server_dir = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PATH"] = f"{vcfg['venv']}/bin:" + env.get("PATH", "")
    cmd = [f"{vcfg['venv']}/bin/python", "-u",
           str(server_dir / "reconstruction" / "run_vipe_with_priors.py"),
           "--frames-dir", str(frames_dir),
           "--priors", str(output_dir / "vipe_priors"),
           "--output", str(vipe_out),
           "--pipeline", vcfg["pipeline"]]
    if inject_poses:
        cmd.append("--inject-poses")
    if stride > 1:
        cmd += ["--frame-stride", str(stride)]
    pipe.send_progress(50, f"Running ViPE with priors (depth{'+poses' if inject_poses else ' only'})...",
                       stage="reconstruction")
    pipe.send_log(f"[vipe] {' '.join(cmd[1:])}")
    proc = subprocess.Popen(cmd, cwd=vcfg["vipe_dir"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    _stream_proc(pipe, proc, "vipe")
    (vipe_out / "rgb" / "frames.mp4").unlink(missing_ok=True)


def _run_vipe_depth_export(pipe: WorkerPipe, output_dir: Path, vcfg: dict,
                           selected_frames_path: str = None):
    """Transcode ViPE's refined (metric) depth EXR → vipe_depth/{frame}.npz for the
    fusion keyframes. ViPE is already metric (priors), so no scale — pose_scale=1."""
    server_dir = Path(__file__).resolve().parent.parent
    cmd = [f"{vcfg['venv']}/bin/python", "-u",
           str(server_dir / "reconstruction" / "vipe_calibrate.py"),
           "--vipe-out", str(output_dir / "vipe_run"),
           "--depth-out", str(output_dir / "vipe_depth"),
           "--out", str(output_dir / "vipe_run" / "pose_scale.json")]
    if selected_frames_path:
        cmd += ["--selected-frames", selected_frames_path]
    pipe.send_progress(78, "Exporting ViPE metric depth → npz (keyframes)...",
                       stage="reconstruction")
    pipe.send_log(f"[vipe-depth] {' '.join(cmd[1:])}")
    proc = subprocess.Popen(cmd, cwd=vcfg["vipe_dir"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    _stream_proc(pipe, proc, "vipe-depth")


def _run_vipe_compose(pipe: WorkerPipe, frames_dir: Path, output_dir: Path):
    """Compose chunks + poses in the da3 env (open3d)."""
    server_dir = Path(__file__).resolve().parent.parent
    script = server_dir / "reconstruction" / "run_vipe_compose.py"
    inner = (f"source {_conda_root()}/etc/profile.d/conda.sh && "
             f"conda activate {os.environ.get('DA3_CONDA_ENV', 'da3')} && "
             f"PYTHONPATH={server_dir}:$PYTHONPATH python -u {script} "
             f"--output-dir {output_dir} --frames-dir {frames_dir}")
    pipe.send_progress(82, "Composing cloud (ViPE poses + metric ViPE depth)...",
                       stage="reconstruction")
    proc = subprocess.Popen(["bash", "-lc", inner], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    _stream_proc(pipe, proc, "vipe-compose")


def _run_vipe_slam(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                   recon_cfg: dict, config: dict, selected_frames_path: str = None):
    """Unified ViPE reconstruction. ViPE is the universal refinement engine, fed
    per-frame PRIORS (metric depth + poses) from the configured source:
      - da3   : DA3 streaming over ALL frames (blur-only) → its depth + poses.
      - stray : ARKit poses + LiDAR depth.  [future]
    ViPE consumes them (METRIC_DEPTH = BA constraint → continuous metric anchor,
    kills drift; POSE = trajectory seed → smooths chunk jumps) and outputs refined
    metric poses + depth. Cloud/TSDF fuse the keyframes from ViPE's output (no
    global scale — already metric)."""
    vcfg = _vipe_cfg(recon_cfg)
    prior_source = recon_cfg.get("prior_source", "da3")
    stride = max(1, int(recon_cfg.get("frame_stride", 1)))   # 1-of-N temporal subsample
    # replace=True (UI "Replace existing outputs") → run from scratch.
    # replace=False → RESUME: detect what's already on disk and run only the missing
    # sub-steps. Each step's output is the checkpoint for the next.
    replace = config.get("_pipeline_replace", True)
    pipe.send_log(f"Reconstruction: ViPE (universal engine) + priors from '{prior_source}' "
                  f"(replace={replace})")

    # ── Detect existing artifacts (resume points), newest → oldest ──
    n_chunks = len(list(output_dir.glob("chunk_*.ply")))
    n_vdepth = len(list((output_dir / "vipe_depth").glob("*.npz"))) if (output_dir / "vipe_depth").exists() else 0
    have_vipe_run = (output_dir / "vipe_run" / "pose" / "frames.npz").exists()
    have_vipe_depth = n_vdepth > 0
    have_priors = (output_dir / "vipe_priors" / "poses.txt").exists()
    have_chunks = n_chunks > 0
    # ALWAYS log what the resume detector sees, so we can tell where it (re)starts.
    pipe.send_log(f"[resume] detected: chunks={n_chunks}, vipe_depth={n_vdepth}, "
                  f"vipe_run={have_vipe_run}, priors={have_priors} | replace={replace} → "
                  f"{'FULL run' if replace else 'resume from latest artifact'}")

    if not replace and have_chunks:
        pipe.send_log(f"[resume] {n_chunks} chunks present → reconstruction already complete, skipping to cloudcompy")
        pipe.send_progress(90, "Reconstruction complete (reused existing chunks)", stage="reconstruction")
        return

    # 1. Prior source → vipe_priors/. Skip if ViPE output (run/depth) or priors exist.
    if replace or not (have_vipe_run or have_vipe_depth or have_priors):
        if prior_source == "da3":
            # DA3 streaming on the strided frame set (1-of-N) — depth+poses only.
            sel = _write_stride_selection(frames_dir, output_dir, stride) if stride > 1 else None
            if stride > 1:
                pipe.send_log(f"Frame stride: 1-of-{stride} (DA3 + ViPE use the same subset)")
            _run_da3(pipe, frames_dir, output_dir, sel, recon_cfg, config, lean=True)
            _cp = float(recon_cfg.get("da3", {}).get("prior_conf_percentile", 35))
            _extract_da3_priors(pipe, frames_dir, output_dir, conf_percentile=_cp)
            # da3_run fully consumed (vipe_priors is self-contained) → free it BEFORE
            # the long ViPE run instead of holding ~25GB of DA3 scratch on disk.
            shutil.rmtree(output_dir / "da3_run", ignore_errors=True)
            shutil.rmtree(output_dir / "da3_full", ignore_errors=True)
            (output_dir / "da3_streaming_config.yaml").unlink(missing_ok=True)
            pipe.send_log("[cleanup] removed da3_run scratch (priors extracted, self-contained)")
        elif prior_source == "stray":
            raise NotImplementedError("prior_source='stray' (ARKit+LiDAR injection) not wired yet")
        else:
            raise ValueError(f"unknown prior_source: {prior_source}")
    else:
        pipe.send_log("[resume] ViPE output / priors present → skipping DA3 prior source")

    # 2. ViPE with priors → refined metric poses + depth. Skip if its output exists.
    #    Poses prior ONLY for stray/lidar (ARKit poses are trustworthy); for da3
    #    (monocular) we feed ONLY depth and let ViPE solve poses freely.
    if replace or not (have_vipe_run or have_vipe_depth):
        _run_vipe_with_priors(pipe, frames_dir, output_dir, vcfg,
                              inject_poses=(prior_source == "stray"), stride=stride)
    else:
        pipe.send_log("[resume] vipe_run present → skipping ViPE")

    # 3. Export ViPE's refined metric depth → vipe_depth/. Skip if already exported.
    if replace or not have_vipe_depth:
        _run_vipe_depth_export(pipe, output_dir, vcfg, selected_frames_path)
        _clean_vipe_inputs(pipe, output_dir)   # drop raw EXR + priors; KEEP vipe_depth/
    else:
        pipe.send_log("[resume] vipe_depth present → skipping depth export")

    # 4. Compose chunks (cloud + origins) from metric ViPE depth + ViPE poses.
    #    (Only reached when chunks were absent; cheap — reads vipe_depth + poses.)
    _run_vipe_compose(pipe, frames_dir, output_dir)

    pipe.send_progress(90, "ViPE SLAM reconstruction complete", stage="reconstruction")


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

    `frame_global` indexes self.img_list, i.e. the SELECTED KEYFRAMES — map it
    through selected_frames.json["selected_files"][frame_global] to get the
    source frame file.

    Legacy MapAnything/VGGT-Long chunks (dicts with 'world_points', no per-point
    confidence) keep the original all-points behavior.
    """
    import numpy as np

    unaligned_dir = vggt_save_dir / "_tmp_results_unaligned"
    pcd_dir = vggt_save_dir / "pcd"
    if not unaligned_dir.exists():
        pipe.send_log("No unaligned chunk data found — origins not generated", level="warning")
        return

    # _postprocess copied save_dir/pcd/K_pcd.ply (K = real chunk number), sorted
    # lexicographically, to output/chunk_{i:03d}.ply. Mirror that ordering so
    # chunk_{i:03d}_origins.npz pairs with chunk_{i:03d}.ply — but use the REAL
    # chunk number K for frame_global (lexicographic sort puts "10" before "2").
    pcd_files = sorted(glob.glob(str(pcd_dir / "*_pcd.ply")))
    pcd_files = [f for f in pcd_files if "combined" not in Path(f).name]
    if not pcd_files:
        pipe.send_log("No chunk PLYs found — origins not generated", level="warning")
        return

    chunk_size = vggt_config["Model"]["chunk_size"]
    overlap = vggt_config["Model"]["overlap"]
    chunk_step = chunk_size - overlap
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

                npy_path = unaligned_dir / f"chunk_{K}.npy"
                if not npy_path.exists():
                    pipe.send_log(f"Chunk {i:03d}: missing {npy_path.name}; origins skipped", level="warning")
                    continue

                chunk_data = np.load(npy_path, allow_pickle=True).item()

                conf_flat = None
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
                else:
                    pipe.send_log(f"Chunk {i:03d} (src {K}): unrecognized data format", level="warning")
                    continue

                HW = H * W

                if conf_flat is not None:
                    # Exact replica of save_confident_pointcloud_batch's mask.
                    thr = float(np.mean(conf_flat)) * coef
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
                frame_global = (frame_local + K * chunk_step).astype(np.int32)

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
