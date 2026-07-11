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
import time
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
        # NO FALLBACK: blur/quality analysis feeds frame selection — if it fails, fail.
        from frame_quality import analyze_frames, save_manifest
        fq = analyze_frames(str(frames_dir))
        if "error" in fq:
            raise RuntimeError(f"Frame quality analysis failed: {fq['error']}")
        save_manifest(str(frames_dir), fq)

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
    # SIMPLE pipeline (reconstruction.simple.enabled): sparse temporal sampling is the
    # first pillar of the one-pass design — it overrides whatever selector is configured.
    _simple_cfg = recon_cfg.get("simple") or {}
    _simple_on = bool(_simple_cfg.get("enabled", False))
    if _simple_on and mode not in ("fps", "motion"):
        _sel = str(_simple_cfg.get("frame_selection", "motion")).lower()
        pipe.send_log(f"SIMPLE pipeline ON → frame selection '{_sel}' "
                      f"overrides frames_selector '{mode}'")
        mode = _sel if _sel in ("fps", "motion") else "motion"
    sf_path = frames_dir / "selected_frames.json"
    if not replace and sf_path.exists():
        pipe.send_log("Reusing existing selected_frames.json (replace=off)")
    elif mode == "fps":
        # Temporal sampling: keep ~target_fps frames of the blur-valid set, spaced by
        # ORIGINAL frame number (frames are extracted 1:1 from the video, so the numeric
        # stem is the video frame index). Native fps read from the source video; 30 as
        # the safe fallback.
        target_fps = float(_simple_cfg.get("target_fps", 1.0) or 1.0)
        native_fps = 30.0
        _vid = next((p for ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v")
                     for p in [frames_dir.parent / f"source_video{ext}"] if p.exists()), None)
        if _vid is not None:
            try:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(str(_vid))
                _f = _cap.get(_cv2.CAP_PROP_FPS)
                _cap.release()
                if _f and _f > 0:
                    native_fps = float(_f)
            except Exception as _e:
                pipe.send_log(f"could not read native fps ({_e}) — assuming 30", level="warning")
        # Sharpest-per-bin sampling — the cadence is GUARANTEED. The old walk sampled
        # the blur-VALID list, so a blurry stretch became a hole in the sequence
        # (measured: 6 s missing on one scan → a 9 m jump between adjacent keyframes
        # on another). Zero overlap between neighbours is far worse for Omega's joint
        # attention than a soft frame: uniform baselines are what the web demo feeds
        # it. So: one frame per 1/target_fps bin, preferring the sharpest VALID frame
        # (frame_quality.json FFT score), falling back to the least-blurry one when
        # the whole bin failed the filter. Blur stays a preference, never a gate.
        step = max(1, int(round(native_fps / max(target_fps, 0.01))))
        _all = sorted((os.path.basename(f) for f in
                       (glob.glob(str(frames_dir / "*.jpg")) + glob.glob(str(frames_dir / "*.png")))),
                      key=lambda f: int(os.path.splitext(f)[0]))
        _quality = {}   # file -> (fft_score, valid)
        _fq_path = frames_dir / "frame_quality.json"
        if blur_on and _fq_path.exists():
            try:
                for _e in json.loads(_fq_path.read_text()).get("frames", []):
                    _quality[_e["file"]] = (float(_e.get("fft_score", 0.0)),
                                            bool(_e.get("valid", True)))
            except Exception as _e:
                pipe.send_log(f"frame_quality.json unreadable ({_e}) — uniform sampling",
                              level="warning")
        bins = {}
        for f in _all:
            bins.setdefault(int(os.path.splitext(f)[0]) // step, []).append(f)
        chosen, soft_bins = [], 0
        for b in sorted(bins):
            frames_in_bin = bins[b]
            if _quality:
                valid_in_bin = [f for f in frames_in_bin if _quality.get(f, (0, True))[1]]
                pool = valid_in_bin or frames_in_bin
                if not valid_in_bin:
                    soft_bins += 1
                chosen.append(max(pool, key=lambda f: _quality.get(f, (0.0, True))[0]))
            else:
                chosen.append(frames_in_bin[0])
        if len(chosen) < 2:
            raise RuntimeError(f"fps sampling produced {len(chosen)} frame(s) "
                               f"(target_fps={target_fps}, native={native_fps:.1f}) — "
                               f"not enough to reconstruct")
        with open(sf_path, "w") as _f:
            json.dump({"version": "2.0", "method": f"fps_{target_fps:g}",
                       "total_frames": len(_all), "selected_count": len(chosen),
                       "selected_files": chosen}, _f)
        _soft = f", {soft_bins} bin(s) all-blurry → kept least-blurry" if soft_bins else ""
        pipe.send_log(f"Frame set: {len(chosen)}/{len(_all)} frames "
                      f"(~{target_fps:g} fps of native {native_fps:.1f}, step {step}, "
                      f"sharpest per bin{_soft}) → selected_frames.json")
    elif mode == "motion":
        # PARALLAX-uniform keyframes: cut one keyframe per fixed quantum of ACCUMULATED
        # inter-frame pixel motion (frame_quality.json inter_frame_diff), picking the
        # sharpest frame inside each quantum window. Pixel motion ≈ parallax, which is
        # what the multi-view estimator actually consumes — NOT meters and NOT seconds:
        #   · fast walking → more keyframes (no 9 m jumps between neighbours)
        #   · standing still / slow drift-prone stretches → almost none (redundant
        #     low-baseline frames amplify feed-forward drift — FastVGGT)
        #   · near scenes cut denser than far scenes automatically (measured: 734
        #     units/m close-range vs 160 units/m in a big hall — same walking speed)
        # DINO-cosine is superseded: dissimilarity is enforced geometrically here.
        quantum = float(_simple_cfg.get("keyframe_motion_quantum", 250.0))
        chosen, _n_total, soft_windows = _motion_keyframes(frames_dir, quantum)
        if len(chosen) < 2:
            raise RuntimeError(f"motion sampling produced {len(chosen)} keyframe(s) "
                               f"(quantum={quantum:g}) — not enough to reconstruct; "
                               f"lower keyframe_motion_quantum")
        with open(sf_path, "w") as _f:
            json.dump({"version": "2.0", "method": f"motion_{quantum:g}",
                       "total_frames": _n_total, "selected_count": len(chosen),
                       "selected_files": chosen}, _f)
        _soft = f", {soft_windows} window(s) all-blurry → kept least-blurry" if soft_windows else ""
        pipe.send_log(f"Frame set: {len(chosen)}/{_n_total} keyframes "
                      f"(parallax-uniform, quantum {quantum:g} motion units, sharpest "
                      f"per window{_soft}) → selected_frames.json")
    elif mode == "dino":
        from frame_selector import select_keyframes
        # NO FALLBACK: keyframe selection is foundational (writes selected_frames.json).
        pipe.send_progress(5, "Selecting keyframes (blur + DINO cosine)...", stage="reconstruction")
        sel = select_keyframes(str(frames_dir), config.get("frame_selection", {}))
        pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes (dino)")
    elif mode == "parallax":
        # GEOMETRIC keyframe selection for the SLAM backbone: triangulation angle, not
        # appearance. Needs per-frame depth+pose → DA3 runs depth-only on ALL blur-valid
        # frames FIRST (the reorder), then we select by parallax. NO cosine fallback.
        from frames.selector import _load_valid_frame_list, select_keyframes_parallax
        pipe.send_progress(4, "Parallax selection: DA3 depth on all blur-valid frames...",
                           stage="reconstruction")
        if blur_on:
            _bv = _load_valid_frame_list(frames_dir)
        else:
            _bv = sorted([os.path.basename(f) for f in
                          (glob.glob(str(frames_dir / "*.jpg")) + glob.glob(str(frames_dir / "*.png")))],
                         key=lambda f: int(os.path.splitext(f)[0]))
        _bv_path = frames_dir / "_parallax_blur_valid.json"
        with open(_bv_path, "w") as _f:
            json.dump({"version": "2.0", "method": "blur_valid", "total_frames": len(_bv),
                       "selected_count": len(_bv), "selected_files": _bv}, _f)
        # da3 backbone: run DA3 FULL on the DENSE set (poses + loop closure + chunks = the
        # backbone). Otherwise depth-only (just priors for the maplong backbone).
        _da3_backbone = (backend == "da3")
        _da3_dir = _run_da3(pipe, frames_dir, output_dir, str(_bv_path), recon_cfg, config,
                            depth_only=not _da3_backbone)
        _npz_dir = Path(_da3_dir) / "results_output"
        pipe.send_progress(6, "Parallax selection: triangulation-angle keyframes...",
                           stage="reconstruction")
        sel = select_keyframes_parallax(str(frames_dir), str(_npz_dir),
                                        config.get("frame_selection", {}))
        if not sel.get("geometric_ok", False):
            # Decision #1: abort with a clear English message the UI surfaces.
            raise RuntimeError("Reconstruction not possible: " + (sel.get("reason") or
                               "insufficient geometric parallax (camera rotation only)"))
        pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes "
                      f"(parallax, baseline {sel['parallax_stats']['global_baseline_m']}m)")
        # DA3 already ran on the full blur-valid set → the backend must NOT re-run it.
        recon_cfg.setdefault("mapanything", {})["_da3_already_extracted"] = True
        if _da3_backbone:
            # DA3 FULL already produced the dense backbone (poses+loops+chunks, postprocessed
            # to output/) → the backend dispatch must NOT re-run it on the sparse keyframes.
            recon_cfg.setdefault("da3", {})["_da3_backbone_done"] = True
    else:
        # "stride" or "none": write the FULL blur-valid set, optionally strided. Writing
        # it explicitly (vs leaving it unset) keeps selected_frames.json the single source.
        # NO FALLBACK: if the frame set can't be built, fail.
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
    # selected_frames.json is the single source of truth downstream → it MUST exist now.
    if not sf_path.exists():
        raise RuntimeError("selected_frames.json was not produced — frame selection failed")
    selected_frames_path = str(sf_path)
    pipe.send_log(f"Using frames from {sf_path}")

    # ── Step 2b: DA3-dense fusion frame set ──
    # The asymmetric design feeds DA3 the FULL blur-valid set (a superset of the VGGT
    # keyframes) so it produces per-frame depth for every sharp frame → the TSDF fuses
    # that dense set (DENSITY win), while VGGT/MapAnything still reconstructs only the
    # keyframes for the loop-closed poses. "full" = all blur-valid, NOT all frames on
    # disk (excludes the blurry ones) and NOT the keyframe decimation.
    # DA3 runs on this DENSER set: DINO at dino_threshold_dense (0.99) → keyframes PLUS the
    # extra inter-keyframe frames that add NEW coverage, but DEDUPED (a camera that filmed the
    # same spot for minutes is NOT included). MapAnything still uses the 0.98 keyframes
    # (selected_frames.json) for poses. The 0.99 set is a SUPERSET-in-spirit: it gives DA3 depth
    # for every frame the dense-fusion step will need, without the redundancy of all-blur-valid.
    da3_dense_frames_path = None
    try:
        fcfg = dict(config.get("frame_selection", {}) or {})
        _dense_thr = fcfg.get("dino_threshold_dense", 0.99)
        _dpath = frames_dir / "da3_frames.json"
        if mode == "dino" and blur_on:
            from frame_selector import dino_select_keyframes
            fcfg["dino_threshold"] = _dense_thr     # 0.99 — denser than the 0.98 keyframes
            # segment_id="dense" → writes selected_frames_segdense.json (a throwaway), NOT the main
            # selected_frames.json. Without it dino_select_keyframes CLOBBERS selected_frames.json
            # (the 0.98 keyframes) with the 0.99 set → MapAnything would run on 0.99 (the bug).
            _sel = dino_select_keyframes(str(frames_dir), fcfg, segment_id="dense")
            (frames_dir / "selected_frames_segdense.json").unlink(missing_ok=True)  # drop throwaway
            _files = list(_sel.get("selected_files", []))
            # UNION with the 0.98 keyframes: the two DINO selections are INDEPENDENT sequential
            # runs, so a 0.98 keyframe is NOT guaranteed to be in the 0.99 set. DA3 must cover
            # every keyframe (else dense_fusion has no ICP target there → fillers skipped). Add
            # any missing keyframes so da3_frames is a true superset of selected_frames.
            try:
                _kf = json.load(open(sf_path)).get("selected_files", []) if sf_path.exists() else []
                _have = set(_files)
                _files = _files + [n for n in _kf if n not in _have]
            except Exception:
                pass
            _files = sorted(set(_files), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
            with open(_dpath, "w") as _f:
                json.dump({"version": "2.0", "method": f"dino_dense_{_dense_thr}",
                           "total_frames": _sel.get("total_frames", len(_files)),
                           "selected_count": len(_files), "selected_files": _files}, _f)
            da3_dense_frames_path = str(_dpath)
            pipe.send_log(f"DA3-dense set: DINO {_dense_thr} → {len(_files)} frames "
                          f"(keyframes + deduped inter-keyframe) → da3_frames.json")
        elif blur_on:
            from frame_selector import _load_valid_frame_list
            _valid = _load_valid_frame_list(frames_dir)  # blur-valid basenames
            with open(_dpath, "w") as _f:
                json.dump({"version": "2.0", "method": "blur_valid_dense",
                           "total_frames": len(_valid), "selected_count": len(_valid),
                           "selected_files": sorted(_valid,
                               key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))}, _f)
            da3_dense_frames_path = str(_dpath)
            pipe.send_log(f"DA3-dense set: {len(_valid)} blur-valid frames → da3_frames.json")
        else:
            pipe.send_log("blur_filter OFF → DA3 dense over ALL frames on disk")
    except Exception as e:
        # NO FALLBACK: the DA3-dense set drives TSDF density — don't silently degrade.
        raise RuntimeError(f"Could not build DA3-dense frame list: {e}") from e

    # ── Step 3: Dispatch to backend ──
    if backend == "da3":
        if not recon_cfg.get("da3", {}).get("_da3_backbone_done"):
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
    elif backend == "vggtomega":
        # SOTA pose backbone (CVPR 2026), dynamic-scene robust (worksite default). DA3
        # per-frame metric depth (NO streaming) is the metric anchor; VGGT-Long[Omega]
        # gives up-to-scale poses; scale_align makes them metric. No ICP dense-fusion.
        _run_vggtomega(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
    else:
        _run_mapanything(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)

    # ── Step 4 (opt-in): dense pose densification + fusion ("ventana-VGGT") ──
    # Anchor the non-keyframe DA3 depths to the VGGT keyframe poses and back-project
    # them → extra cloud points with inter-keyframe coverage, written as a chunk PLY
    # that CloudCompPy merges. Runs BEFORE CloudCompPy → the cleaned cloud (and hence
    # the TSDF, which is masked to it) comes out MORE COMPLETE. Opt-in + non-fatal.
    # The VGGT-Omega path does NOT use ICP dense-fusion (its poses are SOTA + globally
    # optimised; the local ICP was a stopgap that also caused the texture mis-mapping).
    if backend != "vggtomega" and (recon_cfg.get("dense_fusion", {}) or {}).get("enabled"):
        try:
            _run_dense_fusion(pipe, frames_dir, output_dir, config, recon_cfg)
        except Exception as e:
            pipe.send_log(f"[dense-fusion] skipped ({e}) — cloud unchanged", level="warning")

    # ── Step 5 (opt-in): GLOBAL POSE REFINEMENT (BA) ──
    # Refine the backbone's keyframe poses with VGGSfM learned correspondences + COLMAP's
    # pose-prior Ceres bundle adjustment → refined camera_poses.txt → re-project the cloud.
    # The TSDF then integrates depth at the REFINED poses. Runs BEFORE CloudCompPy. ALL
    # chunked backbones (da3, mapanything, AND vggtomega) get the BA: a Sim3-aligned chunked
    # reconstruction still drifts per-window, and the BA over VGGSfM tracks reconciles it.
    # For vggtomega the order is Omega → scale_align (metric via DA3) → BA, so the BA refines
    # already-metric poses. NO FALLBACK: if the BA fails, the reconstruction FAILS (we never
    # silently ship un-refined poses presented as refined).
    if (recon_cfg.get("bundle_adjust", {}) or {}).get("enabled"):
        _run_bundle_adjust_step(pipe, frames_dir, output_dir, recon_cfg)

    # ── Step 6: structure-assisted FINE registration between chunks (surface_fit
    # stage 0). Independent of the BA (which stays OFF for vggtomega — it degrades
    # the VGGT poses): the mm→cm inter-chunk bias that becomes TSDF double layers
    # is corrected directly on the chunk PLYs via plane-constrained point-to-plane
    # alignment, and the affected frame poses get the same rigid correction so
    # cloud↔TSDF stay consistent. Runs BEFORE CloudCompPy merges the chunks.
    # Best-effort: a failure logs and continues (the chunks are still valid,
    # just layered — the surface_fit stage-1 WLOP will partially compensate). ──
    if (recon_cfg.get("fine_register", {}) or {}).get("enabled", True):
        _run_fine_register_step(pipe, output_dir, recon_cfg)


def _run_fine_register_step(pipe: WorkerPipe, output_dir: Path, recon_cfg: dict):
    """surface_fit stage 0 over output/chunk_*.ply (reproject_chunks contract).
    Skips itself when there are <2 backbone chunks (nothing to register)."""
    server_dir = Path(__file__).resolve().parent.parent
    py = sys.executable
    pipe.send_progress(68, "Fine inter-chunk registration (plane-constrained)...",
                       stage="reconstruction")
    cmd = [py, "-m", "reconstruction.surface_fit.fine_register",
           "--output-dir", str(output_dir)]
    fcfg = recon_cfg.get("fine_register", {}) or {}
    if fcfg.get("accept_sep_m") is not None:
        cmd += ["--accept-sep", str(fcfg["accept_sep_m"])]
    if fcfg.get("max_correction_m") is not None:
        cmd += ["--max-correction", str(fcfg["max_correction_m"])]
    if fcfg.get("pieces_per_chunk") is not None:
        cmd += ["--pieces", str(fcfg["pieces_per_chunk"])]
    if fcfg.get("ground_datum") is False:
        cmd += ["--no-ground-datum"]
    if fcfg.get("capture_m") is not None:
        cmd += ["--capture", str(fcfg["capture_m"])]
    if fcfg.get("iters") is not None:
        cmd += ["--iters", str(fcfg["iters"])]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=str(server_dir))
    for line in p.stdout:
        pipe.send_log("[finereg] " + line.rstrip())
    if p.wait() != 0:
        pipe.send_log("[finereg] ⚠ fine registration failed — continuing with "
                      "unregistered chunks", level="warning")


def _run_dense_fusion(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                      config: dict, recon_cfg: dict):
    """Run reconstruction/dense_pose_fusion.py in the mapanything env (it needs the
    MapAnything model). Streams its stdout to the pipe. Writes chunk_998_densefusion.*
    for CloudCompPy to merge."""
    import subprocess, sys, tempfile, json as _json
    dcfg = recon_cfg.get("dense_fusion", {}) or {}
    py = dcfg.get("python", "/workspace/miniforge3/envs/mapanything/bin/python")
    script = Path(__file__).resolve().parent.parent / "reconstruction" / "dense_pose_fusion.py"
    # Pass the live config via a temp JSON (dense_fusion + frame_selection params).
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        _json.dump(config, tf); cfg_path = tf.name
    pipe.send_progress(50, "Dense fusion: densifying non-keyframe poses...", stage="reconstruction")
    pipe.send_log("[dense-fusion] starting (ventana-VGGT) — anchors non-keyframe DA3 depths")
    proc = subprocess.Popen(
        [py, str(script), "--output-dir", str(output_dir), "--frames-dir", str(frames_dir),
         "--config", cfg_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        pipe.send_log(line.rstrip())
    rc = proc.wait()
    pipe.send_log(f"[dense-fusion] exit={rc}")


def _run_bundle_adjust_step(pipe: WorkerPipe, frames_dir: Path, output_dir: Path, recon_cfg: dict):
    """Global pose refinement: VGGSfM learned correspondences → COLMAP/Ceres pose-prior
    bundle adjustment → refined camera_poses.txt (all copies). Runs in the mapanything env
    (GPU tracker + pycolmap). The TSDF then integrates depth at the REFINED poses."""
    import subprocess
    bcfg = recon_cfg.get("bundle_adjust", {}) or {}
    py = bcfg.get("python", "/workspace/miniforge3/envs/mapanything/bin/python")
    server_dir = Path(__file__).resolve().parent.parent

    pipe.send_progress(50, "Pose refinement: extracting VGGSfM tracks (dense)...", stage="reconstruction")
    pipe.send_log("[bundle-adjust] step 1/3 — learned correspondences (VGGSfM, dense set)")
    _cmd = [py, "-m", "reconstruction.vggt_tracks", "--output-dir", str(output_dir),
            "--frames-dir", str(frames_dir),
            "--win", str(bcfg.get("track_window", 24)),
            "--stride", str(bcfg.get("track_stride", 12)),
            "--grid-side", str(bcfg.get("track_grid", 48))]
    # DENSE two-pass BA: track keyframes + fillers (the da3_frames set) so the fillers can be
    # localised against the keyframe map. Falls back to keyframe-only if da3_frames.json is absent.
    _da3_dense = frames_dir / "da3_frames.json"
    if _da3_dense.exists():
        _cmd += ["--frame-list", str(_da3_dense)]
    p1 = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, cwd=str(server_dir))
    for line in p1.stdout:
        pipe.send_log("[ba-tracks] " + line.rstrip())
    if p1.wait() != 0:
        raise RuntimeError("VGGSfM track extraction failed — aborting (no fallback)")

    pipe.send_progress(62, "Pose refinement: COLMAP/Ceres bundle adjustment...", stage="reconstruction")
    pipe.send_log("[bundle-adjust] step 2/2 — pose-prior BA (pycolmap/Ceres)")
    p2 = subprocess.Popen(
        [py, "-m", "reconstruction.run_colmap_ba", "--output-dir", str(output_dir),
         "--prior-stddev-m", str(bcfg.get("prior_stddev_m", 0.10))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(server_dir))
    for line in p2.stdout:
        pipe.send_log("[ba] " + line.rstrip())
    if p2.wait() != 0:
        raise RuntimeError("COLMAP/Ceres bundle adjustment failed — aborting (no fallback)")
    pipe.send_log("[bundle-adjust] refined keyframe poses + localised filler poses written")

    # ── step 3/3a: DENSIFY — back-project the BA-localised filler poses → extra cloud points
    # (replaces the old ICP dense-fusion; the fillers are now globally consistent with the
    # keyframe map). Writes chunk_997_densefusion.ply that CloudCompPy merges. ──
    pipe.send_progress(64, "Pose refinement: densifying cloud (filler back-projection)...",
                       stage="reconstruction")
    p_d = subprocess.Popen(
        [py, "-m", "reconstruction.densify_fillers", "--output-dir", str(output_dir),
         "--stride", str(bcfg.get("densify_stride", 2))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(server_dir))
    for line in p_d.stdout:
        pipe.send_log("[ba-densify] " + line.rstrip())
    if p_d.wait() != 0:
        raise RuntimeError("filler densification failed — aborting (no fallback)")

    # ── step 3/3b: re-project the KEYFRAME chunk clouds to the refined keyframe poses so the
    # CLOUD matches the TSDF (which integrates at the refined poses) — consistent. ──
    pipe.send_progress(66, "Pose refinement: re-projecting keyframe cloud to refined poses...",
                       stage="reconstruction")
    p3 = subprocess.Popen(
        [py, "-m", "reconstruction.reproject_chunks", "--output-dir", str(output_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(server_dir))
    for line in p3.stdout:
        pipe.send_log("[ba-reproj] " + line.rstrip())
    if p3.wait() != 0:
        raise RuntimeError("cloud re-projection to refined poses failed — aborting (no fallback)")
    pipe.send_log("[bundle-adjust] cloud densified + re-projected (cloud↔TSDF consistent, dense)")


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
        # depth_only: only results_output/ (per-frame depth) is consumed downstream
        # (scale_align anchor). DA3's vendor streaming still wrote its OWN chunk byproducts
        # (_tmp_results_*, pcd) ~10x bigger than results_output → free them NOW (before omega)
        # so they don't sit through the whole run and overflow the disk (~22GB on test1).
        for _sub in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd"):
            _d = da3_save_dir / _sub
            if _d.exists():
                _mb = sum(f.stat().st_size for f in _d.rglob("*") if f.is_file()) / (1024 * 1024)
                shutil.rmtree(_d, ignore_errors=True)
                pipe.send_log(f"[da3 depth_only] freed da3_run/{_sub} ({_mb:.0f} MB) — not used by vggtomega")
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
            # Resume: if DA3 depth was already extracted (a prior crashed run, OR the parallax
            # keyframe selector already ran DA3 on all blur-valid frames this session), reuse
            # it instead of re-running DA3 — unless replace AND it wasn't this session.
            _replace = config.get("_pipeline_replace", True)
            _already = ma_cfg.get("_da3_already_extracted", False)  # parallax selector ran DA3
            _existing = output_dir / "da3_run" / "results_output"
            if (_already or not _replace) and _existing.exists() and any(_existing.glob("frame_*.npz")):
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
            # NO FALLBACK: use_da3_priors was requested AND the per-frame DA3 depth npz are
            # also what the bundle adjustment + TSDF (da3_frames) depend on — don't silently
            # drop to image-only.
            if not (priors_dir.exists() and any(priors_dir.glob("frame_*.npz"))):
                raise RuntimeError(f"DA3 priors not produced at {priors_dir} (no frame_*.npz)")
            vggt_config["Model"]["da3_priors_dir"] = str(priors_dir)
            pipe.send_log(f"DA3 priors → {priors_dir} (depth+intrinsics fed to MapAnything)")
        except Exception as _e:
            raise RuntimeError(f"DA3 priors extraction failed: {_e}") from _e

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


def _build_vggtomega_config(config: dict) -> dict:
    """Load stac_vggtomega.yaml and override the same user-configurable params as the
    MapAnything path (chunk size/overlap/loop), keeping the Omega-specific keys."""
    import yaml as _yaml
    ma = config.get("reconstruction", {}).get("mapanything", config.get("mapanything", {}))
    project_root = Path(__file__).resolve().parent.parent.parent
    base = project_root / "vendor" / "VGGT-Long" / "configs" / "stac_vggtomega.yaml"
    if not base.exists():
        raise FileNotFoundError(f"Omega base config not found: {base}")
    with open(base) as f:
        cfg = _yaml.safe_load(f)
    om = config.get("reconstruction", {}).get("vggtomega", {})
    cfg["Model"]["chunk_size"] = om.get("chunk_size", ma.get("chunk_size", cfg["Model"]["chunk_size"]))
    cfg["Model"]["overlap"] = om.get("chunk_overlap", ma.get("chunk_overlap", cfg["Model"]["overlap"]))
    cfg["Model"]["loop_enable"] = om.get("loop_closure", cfg["Model"].get("loop_enable", True))
    cfg["Model"]["frame_stride"] = 1
    cfg["Model"]["delete_temp_files"] = False
    cfg["Model"]["omega_resolution"] = om.get("resolution", cfg["Model"].get("omega_resolution", 512))
    cfg["Model"]["omega_mode"] = om.get("mode", cfg["Model"].get("omega_mode", "balanced"))
    return cfg


def _emit_omega_depth(save_dir: Path, output_dir: Path, chunk_size: int, overlap: int,
                      selected_frames_path: str, pipe: WorkerPipe) -> None:
    """Write per-frame VGGT-Omega depth (globally scale-consistent) to
    omega_run/results_output/frame_<num>.npz so scale_align can compare it to DA3.
    The omega 'depth' is recovered from the ALIGNED world_points projected onto each
    camera's forward axis → it carries the same global (up-to-scale) units as the poses."""
    import numpy as np, json, glob
    sel = json.load(open(selected_frames_path))
    files = sorted(sel.get("selected_files", sel if isinstance(sel, list) else []))
    stems = [int(Path(f).stem) for f in files]
    N = len(stems)
    step = max(1, chunk_size - overlap)
    out_dir = output_dir / "omega_run" / "results_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned = save_dir / "_tmp_results_aligned"

    # STAC fix: the omega depth MUST use the ALIGNED per-frame pose, NOT the raw chunk
    # extrinsic. world_points are aligned (per-chunk Sim3) but cd['extrinsic'] is raw →
    # mixing them gave a wrong omega depth → scale_align underestimated s by ~1.37x
    # (measured: gauge came out <1m instead of ~1.4m). Use camera_poses.txt (already
    # aligned at this point, up-to-scale like world_points) keyed by camera_frames.txt.
    pose_map = {}
    for base in (output_dir, save_dir):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if pp.exists() and fp.exists():
            plines = [l for l in pp.read_text().splitlines() if len(l.split()) == 16]
            pnums = [int(float(x)) for x in fp.read_text().split()]
            if len(plines) == len(pnums) and plines:
                pose_map = {n: np.array(list(map(float, l.split())), np.float64).reshape(4, 4)
                            for n, l in zip(pnums, plines)}
                break
    if not pose_map:
        pipe.send_log("[omega-depth] no aligned camera_poses.txt found — falling back to "
                      "raw chunk extrinsic (scale may be off)", level="warning")

    n_written = 0
    for cp in sorted(glob.glob(str(aligned / "chunk_*.npy"))):
        try:
            k = int(Path(cp).stem.split("_")[1])
            cd = np.load(cp, allow_pickle=True).item()
            wp = np.asarray(cd["world_points"])            # [S,H,W,3] aligned world
            ext = np.asarray(cd["extrinsic"])              # [S,4,4] c2w (RAW — fallback only)
            S = wp.shape[0]
            start = k * step
            for j in range(S):
                gi = start + j
                if gi >= N:
                    break
                c2w = pose_map.get(stems[gi], ext[j])     # ALIGNED pose; raw extrinsic fallback
                cam_c = c2w[:3, 3]
                fwd = c2w[:3, 2]                           # camera +z in world
                d = (wp[j] - cam_c) @ fwd                  # [H,W] depth along view axis
                np.savez_compressed(out_dir / f"frame_{stems[gi]}.npz",
                                    depth=d.astype(np.float32))
                n_written += 1
        except Exception as e:
            pipe.send_log(f"[omega-depth] chunk {cp} skipped ({e})", level="warning")
    pipe.send_log(f"[omega-depth] wrote {n_written} per-frame omega depths for scale align")


from workers.base import gpu_free_gb as _gpu_free_gb, stop_semantic_service


def _motion_keyframes(frames_dir: Path, quantum: float):
    """Parallax-uniform keyframes: one per `quantum` of accumulated inter-frame pixel
    motion (frame_quality.json), sharpest frame per window (all-blurry windows keep
    their least-blurry frame — a soft frame beats a hole). Returns (files, n_total,
    soft_windows). Shared by the frame-selection stage and the chunked-metric phase 2,
    which re-selects DENSER so a 12 m chunk still holds enough keyframes to align."""
    fq_path = frames_dir / "frame_quality.json"
    if not fq_path.exists():
        raise RuntimeError("frame selection 'motion' needs frame_quality.json "
                           "(inter_frame_diff) — run with the quality analysis enabled")
    entries = json.loads(fq_path.read_text()).get("frames", [])
    entries.sort(key=lambda e: int(os.path.splitext(e["file"])[0]))
    if not entries:
        raise RuntimeError("frame_quality.json has no per-frame entries")
    window, chosen, soft = [], [], [0]

    def _flush(win):
        valid_w = [e for e in win if e.get("valid", True)]
        pool = valid_w or win
        if not valid_w:
            soft[0] += 1
        chosen.append(max(pool, key=lambda e: float(e.get("fft_score", 0.0)))["file"])

    acc = 0.0
    for e in entries:
        window.append(e)
        acc += float(e.get("inter_frame_diff", 0.0))
        if acc >= quantum:
            _flush(window)
            window, acc = [], 0.0
    if window:                      # tail: whatever motion was left still gets a view
        _flush(window)
    return chosen, len(entries), soft[0]


def _run_da3_anchor(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                    anchor_files: list, recon_cfg: dict) -> None:
    """ISOLATED per-frame DA3 metric depth on the K scale-anchor frames — NO streaming.
    The streaming pipeline chains poses across consecutive frames; anchor frames are
    seconds apart, the chain breaks (pose=None → crash) and none of its machinery is
    needed: the scale is a per-pixel depth RATIO, poses don't participate. Runs
    extract_da3_depth.py (--per_frame) and converts its output to the exact layout
    scale_align consumes: da3_run/results_output/frame_<num>.npz (depth + conf)."""
    import numpy as np
    server_dir = Path(__file__).resolve().parent.parent
    tmp = output_dir / "_da3_anchor_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in anchor_files:
        src = frames_dir / f
        if src.exists():
            os.symlink(str(src), str(tmp / f))
    raw = output_dir / "da3_run" / "anchor_raw"
    model_id = str((recon_cfg.get("da3", {}) or {}).get(
        "model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"))
    cmd = [sys.executable, str(server_dir / "extract_da3_depth.py"),
           "--image_dir", str(tmp), "--output_dir", str(raw),
           "--model", model_id, "--per_frame"]
    pipe.send_log(f"DA3 anchor: isolated per-frame depth on {len(anchor_files)} frames "
                  f"({model_id}) — no streaming")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.strip()
        if line:
            pipe.send_log(line)
        if pipe.check_cancel():
            proc.terminate()
            return
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"DA3 anchor extraction exited with code {proc.returncode}")

    ro = output_dir / "da3_run" / "results_output"
    ro.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in anchor_files:
        stem = os.path.splitext(f)[0]
        dp, cp = raw / f"{stem}_depth.npy", raw / f"{stem}_conf.npy"
        if not dp.exists():
            continue
        num = int(stem)
        arrays = {"depth": np.load(dp).astype(np.float32)}
        if cp.exists():
            arrays["conf"] = np.load(cp).astype(np.float32)
        kp = raw / f"{stem}_intrinsics.npy"
        if kp.exists():
            arrays["intrinsics"] = np.load(kp).astype(np.float64)
        np.savez_compressed(ro / f"frame_{num}.npz", **arrays)
        n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    if n == 0:
        raise RuntimeError("DA3 anchor produced no depth maps — scale cannot be estimated")
    pipe.send_log(f"DA3 anchor: {n} metric depth maps → {ro}")


def _run_vggtomega(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                   selected_frames_path: str, recon_cfg: dict, config: dict):
    """VGGT-Omega backbone: DA3 per-frame metric depth (anchor) + VGGT-Long[Omega] poses
    (up-to-scale) + metric scale alignment. No ICP dense-fusion."""
    import yaml, re as _re
    device = recon_cfg.get("device", "cpu")

    # ── SIMPLE pipeline knobs (reconstruction.simple) ──
    _simple_cfg = recon_cfg.get("simple") or {}
    _simple_on = bool(_simple_cfg.get("enabled", False))
    _n_selected = 0
    try:
        _sel = json.load(open(selected_frames_path))
        _sel_files = _sel.get("selected_files", _sel if isinstance(_sel, list) else [])
        _n_selected = len(_sel_files)
    except Exception:
        _sel_files = []

    # Exclusive GPU: reconstruction and the semantic service never share the card.
    # vLLM's resident ~40 GB would cap the Omega pass; stop it here — the VLM stage
    # brings it back up on its own once reconstruction is done.
    if _simple_on and bool(_simple_cfg.get("exclusive_gpu", True)):
        stop_semantic_service(pipe, stage="Omega reconstruction")

    # ── DA3 per-frame metric depth (NO streaming) on the dense/keyframe set ──
    # DA3 here is ONLY the metric anchor consumed by scale_align (and the TSDF depth
    # source). The Omega backbone itself does NOT use it. So when scale_align is OFF
    # (testing raw Omega), skip DA3 entirely — otherwise it re-runs for hours for nothing.
    _scale_align_on = bool((recon_cfg.get("vggtomega", {}) or {}).get("scale_align", True))
    # SIMPLE: the metric scale is ONE scalar — a handful of evenly-spread anchor frames
    # is statistically equivalent to the whole set (measured: an 11-frame re-check moved
    # s by only -0.91%). Stray sessions already skip DA3 inference entirely (their depth
    # is converted to the DA3 layout by convert_stray_to_da3.py and detected as done).
    _anchor_files = None
    if _simple_on and _scale_align_on and _sel_files:
        _k = int(_simple_cfg.get("scale_anchor_frames", 12) or 12)
        if 1 < _k < _n_selected:
            _idx = sorted({round(i * (_n_selected - 1) / (_k - 1)) for i in range(_k)})
            _anchor_files = [_sel_files[int(i)] for i in _idx]
            _anchor_path = output_dir / "scale_anchor_frames.json"
            with open(_anchor_path, "w") as _f:
                json.dump({"version": "2.0", "method": f"scale_anchor_{_k}",
                           "total_frames": _n_selected,
                           "selected_count": len(_anchor_files),
                           "selected_files": _anchor_files}, _f)
            pipe.send_log(f"SIMPLE: DA3 metric anchor on {len(_anchor_files)}/{_n_selected} "
                          f"evenly-spread frames (scale is one scalar — the rest is waste)")
    if _scale_align_on:
        pipe.send_progress(6, "VGGT-Omega: extracting DA3 metric depth (per-frame)...",
                           stage="reconstruction")
        # DA3's ONLY role in the vggtomega path is per-frame metric depth VALUES
        # (scale_align + metric-lock anchors). ISOLATED per-frame extraction, NEVER
        # streaming: the streaming machinery chains poses across frames — poses
        # nothing here consumes — and crashes when the chain starves (test3:
        # 12 keyframes → single chunk → save_camera_poses pose=None). The old
        # streaming fallback fired exactly when the selection was SMALLER than
        # scale_anchor_frames; a small selection simply means every frame anchors.
        if not _sel_files:
            raise RuntimeError("scale_align needs selected frames to anchor DA3 depth "
                               "(selected_frames.json empty or unreadable)")
        if _anchor_files is None:
            _anchor_files = list(_sel_files)
            pipe.send_log(f"DA3 metric anchor on ALL {_n_selected} selected frames "
                          f"(isolated per-frame — no streaming)")
        _ro = output_dir / "da3_run" / "results_output"
        _missing = [f for f in _anchor_files
                    if not (_ro / f"frame_{int(os.path.splitext(f)[0])}.npz").exists()]
        if _missing:
            _run_da3_anchor(pipe, frames_dir, output_dir, sorted(set(_missing)), recon_cfg)
        else:
            # Stray sessions (depth pre-converted to the DA3 layout) and resumed
            # runs land here: everything already extracted.
            pipe.send_log("DA3 anchor: all per-frame depths already on disk — skipped")
    else:
        pipe.send_log("scale_align OFF → skipping DA3 (its only role here is the metric "
                      "anchor for scale_align) — running Omega ONLY")

    # ── VGGT-Long with the Omega backbone — TWO-PHASE ──
    # Phase 1: ONE pass over the motion keyframes. For short walks this IS the result
    # (no windows → no seams → no onion; validated vs the web demo on test2). It is
    # ALSO the probe: the metric trajectory it yields measures the real walk length.
    # Phase 2 (walk > max_walk_single_pass_m): Omega drifts ~1.3 cm/m on long walks
    # (feed-forward, gauge-anchored at frame 0, no global correction — measured on
    # test4). Re-run CHUNKED-METRIC: chunks sized by walked meters, each metric-locked
    # to DA3 anchors BEFORE alignment, glued SE(3) (scale is not negotiable — the Sim3
    # scale freedom is what produced the onion), SALAD loop closure + pose graph on.
    from reconstruction.chunk_plan import (walk_length_m, plan_chunks,
                                           plan_anchor_indices, trim_static_ends)
    vggt_config = _build_vggtomega_config(config)
    _va_cfg = recon_cfg.get("vggtomega", {}) or {}
    _max_walk = float(_simple_cfg.get("max_walk_single_pass_m", 25.0))
    _chunk_walk = float(_simple_cfg.get("chunk_walk_m", 12.0))
    _anch_per_chunk = int(_simple_cfg.get("chunk_anchors", 3))
    _anchor_dir = output_dir / "da3_run" / "results_output"

    def _apply_conf_filter(cfg_v):
        # Point-confidence filter, same knob the web demo exposes: drop the bottom P%
        # of the valid points by confidence (scene-independent; a mean-relative coef
        # kept 53% of one scan and 89% of another). The origins generator replicates
        # this exact mask, so traceability stays 1:1.
        _pct = _simple_cfg.get("conf_percentile")
        _coef = _simple_cfg.get("conf_threshold_coef")
        if _pct is not None or _coef:
            _ps = cfg_v["Model"].setdefault("Pointcloud_Save", {})
            _ps["use_conf_filter"] = True
            if _pct is not None:
                _ps["conf_percentile"] = float(_pct)
                pipe.send_log(f"SIMPLE: confidence filter — drop the bottom {float(_pct):g}% "
                              f"of valid points (keeps {100 - float(_pct):g}%)")
            else:
                _ps["conf_threshold_coef"] = float(_coef)
                pipe.send_log(f"SIMPLE: point confidence filter conf >= mean*{float(_coef):g}")

    def _apply_chunked_metric(cfg_v, _chunk, _ov):
        cfg_v["Model"]["chunk_size"] = int(_chunk)
        cfg_v["Model"]["overlap"] = int(_ov)
        cfg_v["Model"]["loop_enable"] = True
        cfg_v["Model"]["using_sim3"] = False       # SE(3): scale locked by the anchors
        cfg_v["Model"]["metric_lock"] = {
            "enable": True,
            "anchor_dir": str(_anchor_dir),
            "near_frac": float(_va_cfg.get("scale_near_frac", 0.25)),
            # per-chunk LINEAR scale drift (self-gated on held-out seam obs):
            # one scalar per chunk cannot represent a chunk whose internal
            # scale drifts (test4: 48% anchor spread inside chunk 3; the
            # leftover warp was the z-drift on the chimney/cones)
            "scale_drift": bool(_va_cfg.get("scale_drift", True)),
            # soft health tier: anchor spread beyond this ⇒ SUSPECT (chunk
            # argues more quietly in elastic/finereg, still writes points)
            "suspect_spread": float(_va_cfg.get("suspect_spread", 0.30)),
        }
        cfg_v["Model"]["exact_seam_align"] = True   # rigid seams from EXACT pixel
                                                    # correspondences (mm), not the
                                                    # coarse point-map fit (25-30cm)
        cfg_v["Model"]["frame_ownership"] = True    # one frame → one writer: overlap
                                                    # frames stop entering the cloud twice
        cfg_v["Model"]["elastic_seam"] = bool(      # per-frame seam CONSENSUS: the same
            _va_cfg.get("elastic_seam", True))      # pixel seen by two chunks lands at
                                                    # ONE 3D position (rigid residual per
                                                    # shared frame, blended across the
                                                    # overlap, poses moved with points)
        cfg_v["Model"]["intra_chunk"] = bool(       # INTRA-CHUNK per-frame consensus:
            _va_cfg.get("intra_chunk", True))       # bounded fields, endpoints clamped
                                                    # to the seam consensus — corrections
                                                    # longer than one chunk are impossible
                                                    # by construction; per-chunk held-out
                                                    # gates; worst case = identity per
                                                    # chunk (run 3 behaviour).
        cfg_v["Model"]["depth_graph"] = True        # per-frame DEPTH graph: different
                                                    # frames agree on the depth of the
                                                    # same surface (kills the in-depth
                                                    # object duplication: 1.5% intra +
                                                    # 2x at chunk crossings, measured)
        cfg_v["Model"]["blend_copies"] = True       # two-copy consensus: overlap frames
                                                    # keep the MEAN of their two chunks'
                                                    # fields instead of discarding one
                                                    # (measured: cross-owner depth
                                                    # disagreement 1.51% -> 1.01%)
        pipe.send_log(f"CHUNKED-METRIC: chunks {int(_chunk)}/{int(_ov)} (50% overlap), "
                      f"scale graph (seams+DA3) + self-gated per-chunk scale DRIFT, "
                      f"EXACT-correspondence seam gluing, "
                      f"frame ownership (one writer per frame), ELASTIC per-frame "
                      f"seam consensus (shared pixels share one 3D position), "
                      f"DEPTH graph (frames agree on shared-surface depth)")

    def _ensure_anchors(_files):
        """Isolated DA3 depth for every anchor file not already extracted."""
        _missing = [f for f in _files
                    if not (_anchor_dir / f"frame_{int(os.path.splitext(f)[0])}.npz").exists()]
        if _missing:
            _run_da3_anchor(pipe, frames_dir, output_dir, sorted(set(_missing)), recon_cfg)

    def _omega_pass(cfg_v, tag):
        vggt_config_path = output_dir / "vggt_omega_config.yaml"
        with open(vggt_config_path, "w") as f:
            yaml.dump(cfg_v, f, default_flow_style=False)
        pipe.send_log(f"VGGT-Long[Omega] config ({tag}): {vggt_config_path}")

        project_root = Path(__file__).resolve().parent.parent.parent
        vggt_script = project_root / "vendor" / "VGGT-Long" / "vggt_long.py"
        vggt_save_dir = output_dir / "maplong_run"
        script_path = Path(__file__).resolve().parent.parent / "run_mapanything.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"run_mapanything.sh not found: {script_path}")
        cmd = ["bash", str(script_path), "--image_dir", str(frames_dir),
               "--config", str(vggt_config_path), "--save_dir", str(vggt_save_dir)]
        if selected_frames_path:
            cmd.extend(["--selected_frames", selected_frames_path])
        env = os.environ.copy()
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        pipe.send_progress(10, f"Starting VGGT-Long[Omega] ({tag})...", stage="reconstruction")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, cwd=str(vggt_script.parent))
        chunk_pattern = _re.compile(r'\[Progress\]:\s*(\d+)/(\d+)')
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if pipe.check_cancel():
                proc.terminate(); pipe.send_log("Cancelled by user", level="warning")
                return False
            m = chunk_pattern.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                pipe.send_progress(10 + (done / max(total, 1)) * 65, f"Chunk {done}/{total}",
                                   stage="reconstruction")
            pipe.send_log(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"VGGT-Long[Omega] exited with code {proc.returncode}")

        pipe.send_progress(78, f"VGGT-Long[Omega] {tag} complete, post-processing...",
                           stage="reconstruction")
        _postprocess_reconstruction(pipe, vggt_save_dir, output_dir, cfg_v, backend="mapanything")
        return True

    _chunked_already = False
    if _simple_on and _n_selected:
        _max_sp = int(_simple_cfg.get("max_frames_single_pass", 600) or 600)
        _free = _gpu_free_gb()
        _fits = max(2, int((_free - 4.0) / 0.086)) if _free is not None else _max_sp
        if _n_selected <= min(_max_sp, _fits):
            vggt_config["Model"]["chunk_size"] = max(_n_selected, 2)
            vggt_config["Model"]["overlap"] = 0
            vggt_config["Model"]["loop_enable"] = False
            pipe.send_log(f"SIMPLE single-pass: {_n_selected} frames in ONE chunk "
                          f"(no windows → no seams). Walk length measured after — "
                          f"a walk over {_max_walk:g} m re-runs chunked-metric.")
        elif _scale_align_on:
            # too many frames for one pass even as a probe → chunked-metric DIRECTLY
            # (no meters yet: size by keyframe count; keyframes are parallax-uniform)
            _chunk = max(24, min(60, _fits))
            _ov = _chunk // 2
            _chunked_already = True
            _anchor_idx = plan_anchor_indices(_n_selected, _chunk, _ov, _anch_per_chunk)
            _ensure_anchors([_sel_files[i] for i in _anchor_idx])
            _apply_chunked_metric(vggt_config, _chunk, _ov)
        else:
            _chunk = min(int(vggt_config["Model"]["chunk_size"]), _fits)
            _chunk = max(_chunk, 50)
            vggt_config["Model"]["chunk_size"] = _chunk
            vggt_config["Model"]["overlap"] = _chunk // 2
            pipe.send_log(f"SIMPLE: {_n_selected} frames exceed one pass and scale_align "
                          f"is OFF → legacy windowed mode ({_chunk}/{_chunk // 2})",
                          level="warning")
        _apply_conf_filter(vggt_config)
    if not _omega_pass(vggt_config, "chunked-metric" if _chunked_already else "single-pass"):
        return

    # ── metric scale + orientation (runs after EVERY pass) ──
    # Opt-out (scale_align: false) leaves poses UP-TO-SCALE — used to isolate whether a
    # bad result comes from the scale alignment vs the raw Omega backbone.
    vggt_save_dir = output_dir / "maplong_run"
    if not _scale_align_on:
        pipe.send_log("[scale-align] DISABLED (vggtomega.scale_align: false) — poses stay up-to-scale",
                      level="warning")
        return

    def _metricize_and_orient(cfg_v, tag):
        """Emit omega depth → scale_align (global; in chunked-metric mode the chunks are
        already locked, so this is the residual/VERIFICATION pass — its spread is the
        health metric) → bake upright orientation. Returns the walk length in meters."""
        pipe.send_progress(86, f"VGGT-Omega ({tag}): aligning metric scale to DA3...",
                           stage="reconstruction")
        _emit_omega_depth(vggt_save_dir, output_dir,
                          int(cfg_v["Model"]["chunk_size"]),
                          int(cfg_v["Model"]["overlap"]),
                          selected_frames_path, pipe)
        from reconstruction.scale_align import run as _scale_run
        _s = _scale_run(output_dir, dry_run=False,
                        log=lambda m: pipe.send_log(f"[scale-align] {m}"),
                        near_frac=float(_va_cfg.get("scale_near_frac", 0.25)),
                        conf_top_frac=float(_va_cfg.get("scale_conf_top_frac", 0.10)))
        # METRIC IS MANDATORY: a non-metric cloud is useless (BIM comparison needs real
        # units). If scale_align could not estimate s, FAIL — never ship up-to-scale.
        if _s is None:
            raise RuntimeError(
                "metric scale alignment FAILED — scale_align could not estimate s. Refusing to "
                "produce a NON-METRIC reconstruction (it can't be compared against BIM). See the "
                "[scale-align] lines above for the exact reason (frame match / ratios / inputs).")

        # ── SIMPLE: bake the upright orientation (gravity from the camera poses) ──
        if _simple_on and bool(_simple_cfg.get("orient_from_poses", True)):
            pipe.send_progress(92, "Baking upright orientation from camera poses...",
                               stage="reconstruction")
            from reconstruction.orient import run as _orient_run
            _T = _orient_run(output_dir, log=lambda m: pipe.send_log(f"[orient] {m}"))
            if _T is None:
                pipe.send_log("[orient] orientation NOT applied (no poses or weak camera-down "
                              "consensus) — the floor leveler downstream is the fallback",
                              level="warning")
        try:
            _walk = walk_length_m(output_dir / "camera_poses.txt")
        except Exception as _e:  # noqa: BLE001
            pipe.send_log(f"[chunk-plan] could not measure walk length ({_e})", level="warning")
            _walk = 0.0
        pipe.send_log(f"[chunk-plan] measured walk: {_walk:.1f} m ({tag})")
        return _walk

    _walk_m = _metricize_and_orient(vggt_config, "chunked-metric" if _chunked_already
                                    else "single-pass")

    # ── PHASE 2: the probe says the walk exceeds Omega's comfort range → re-run
    # chunked-metric. The comfort limit is a config parameter (25 m default): Omega is
    # excellent on short walks and drifts ~1.3 cm/m past them (measured, test4).
    if (_simple_on and not _chunked_already and _walk_m > _max_walk
            and _n_selected >= 24):
        # ── coverage trim (probe-informed): drop rotation-only ENDS before chunking ──
        # The probe already walked the scene: keyframes whose camera does not move
        # hold no parallax, so their chunks would be born rotten at ANY chunking
        # (test4: 3/13 chunks — 23% of the phase-2 GPU — spent on a tail with
        # 0.24 m of walk in 24 kf, geometry garbage by construction) and their
        # anchors/seams pollute the scale graph. Static HEAD/TAIL are trimmed in
        # real-frame space and declared as ranges WITHOUT 3D coverage; mid-walk
        # weak stretches stay — cutting them would split the sequence into islands
        # with no shared frames to glue, and the vendor's chunk health gate covers
        # them. Trim failure falls open (no trim) LOUDLY: the health gate is the
        # safety net either way.
        _trim_lo_num = _trim_hi_num = None
        try:
            import numpy as _np
            _probe_frames = json.loads(
                (output_dir / "maplong_run" / "frame_list.json").read_text())
            _rows = [l.split() for l in open(output_dir / "camera_poses.txt") if l.strip()]
            _ctr = _np.array([[float(x) for x in r] for r in _rows]).reshape(-1, 4, 4)[:, :3, 3]
            # per-keyframe estimated focal: the ZOOM detector's direct signal
            _fx = None
            for _ip in (output_dir / "intrinsic.txt",
                        output_dir / "maplong_run" / "intrinsic.txt"):
                if _ip.exists():
                    _fx = _np.array([float(l.split()[0]) for l in
                                     _ip.read_text().splitlines() if l.strip()])
                    break
            if _fx is not None and len(_fx) != len(_ctr):
                pipe.send_log(f"[coverage-trim] intrinsic.txt has {len(_fx)} rows vs "
                              f"{len(_ctr)} poses — zoom detection skipped", level="warning")
                _fx = None
            if len(_ctr) != len(_probe_frames):
                pipe.send_log(f"[coverage-trim] probe poses ({len(_ctr)}) != frame list "
                              f"({len(_probe_frames)}) — trim skipped", level="warning")
            if len(_ctr) == len(_probe_frames) and len(_ctr) >= 3:
                _plo, _phi = trim_static_ends(_ctr, fx=_fx)
                if (_plo, _phi) != (0, len(_ctr)):
                    _pnums = [int(os.path.splitext(f)[0]) for f in _probe_frames]
                    _trim_lo_num, _trim_hi_num = _pnums[_plo], _pnums[_phi - 1]
                    _kept = [f for f in _sel_files
                             if _trim_lo_num <= int(os.path.splitext(f)[0]) <= _trim_hi_num]
                    _walk_kept = float(_np.linalg.norm(
                        _np.diff(_ctr[_plo:_phi], axis=0), axis=1).sum())
                    with open(output_dir / "coverage_trim.json", "w") as _f:
                        json.dump({"frame_lo": _trim_lo_num, "frame_hi": _trim_hi_num,
                                   "probe_kf_trimmed_head": int(_plo),
                                   "probe_kf_trimmed_tail": int(len(_ctr) - _phi),
                                   "keyframes_dropped": _n_selected - len(_kept),
                                   "walk_m_total": _walk_m, "walk_m_kept": _walk_kept,
                                   "reason": "static and/or optical-zoom ends: neither "
                                             "adds baseline -> no parallax -> no 3D "
                                             "information (steps a decade below the "
                                             "walking pace, or focal at robust z>3.5)"},
                                  _f, indent=1)
                    pipe.send_log(f"[coverage-trim] static ends: keeping real frames "
                                  f"[{_trim_lo_num}, {_trim_hi_num}] — dropped "
                                  f"{_n_selected - len(_kept)} keyframe(s) "
                                  f"({int(_plo)} head / {int(len(_ctr) - _phi)} tail probe kf); "
                                  f"those ranges are declared WITHOUT 3D coverage",
                                  level="warning")
                    _sel_files = _kept
                    _n_selected = len(_kept)
                    _walk_m = _walk_kept
                    try:
                        _prev_sel = json.load(open(selected_frames_path))
                        _tot = int(_prev_sel.get("total_frames", len(_kept)))
                        _meth = str(_prev_sel.get("method", "motion")) + "+trim"
                    except Exception:
                        _tot, _meth = len(_kept), "motion+trim"
                    with open(selected_frames_path, "w") as _f:
                        json.dump({"version": "2.0", "method": _meth,
                                   "total_frames": _tot,
                                   "selected_count": len(_kept),
                                   "selected_files": _kept}, _f)
        except Exception as _e:  # noqa: BLE001
            pipe.send_log(f"[coverage-trim] SKIPPED ({_e}) — the chunk health gate "
                          f"remains the safety net for parallax-starved chunks",
                          level="warning")

        # Re-select keyframes DENSER for the chunked pass: with sparse keyframes
        # (big m/kf) a minimum-size chunk covers far more walk than chunk_walk_m —
        # e.g. 66 kf over 81 m gives 1.2 m/kf, so a 24-kf chunk spans 29 m. Target
        # ~30 kf per chunk: quantum scales linearly with the desired kf spacing.
        # Within a 12 m chunk this added density is HARMLESS (the drift-amplifying
        # redundancy is a long-sequence effect; 30-frame passes are the demo regime).
        # 45 kf per 12 m chunk (user-requested density bump from 30, 2026-07-11):
        # cloud density scales with keyframes/metre; chunk_size grows with it so
        # each chunk still spans chunk_walk_m. VRAM is ample (~0.086 GB/frame);
        # inference cost grows ~quadratically per chunk. Density is only useful
        # WITH the fusion stages active (blend/consensus) — otherwise extra frames
        # add fuzz layers, not signal.
        _kf_per_chunk = 45
        _m_per_kf = _walk_m / max(_n_selected, 1)
        _desired_m_per_kf = _chunk_walk / _kf_per_chunk
        if _desired_m_per_kf < _m_per_kf * 0.95:
            _q1 = float(_simple_cfg.get("keyframe_motion_quantum", 250.0))
            _q2 = max(20.0, _q1 * _desired_m_per_kf / _m_per_kf)
            _chosen2, _n_total2, _ = _motion_keyframes(frames_dir, _q2)
            if _trim_lo_num is not None:
                # the dense re-selection sweeps the WHOLE video — keep it inside
                # the trimmed coverage window
                _chosen2 = [f for f in _chosen2
                            if _trim_lo_num <= int(os.path.splitext(f)[0]) <= _trim_hi_num]
            if len(_chosen2) > _n_selected:
                with open(selected_frames_path, "w") as _f:
                    json.dump({"version": "2.0", "method": f"motion_{_q2:g}_chunked",
                               "total_frames": _n_total2,
                               "selected_count": len(_chosen2),
                               "selected_files": _chosen2}, _f)
                pipe.send_log(f"[chunk-plan] re-selected {len(_chosen2)} keyframes "
                              f"(quantum {_q1:g}→{_q2:.0f}) so each {_chunk_walk:g} m "
                              f"chunk holds ~{_kf_per_chunk} keyframes")
                _sel_files = _chosen2
                _n_selected = len(_chosen2)
        _chunk, _ov = plan_chunks(_n_selected, _walk_m, _chunk_walk)
        if _chunk < _n_selected:
            pipe.send_log(f"[chunk-plan] walk {_walk_m:.1f} m > comfort "
                          f"{_max_walk:g} m → phase 2: chunked-metric re-run "
                          f"({_chunk} kf/chunk ≈ {_chunk_walk:g} m walked each)")
            _anchor_idx = plan_anchor_indices(_n_selected, _chunk, _ov, _anch_per_chunk)
            _ensure_anchors([_sel_files[i] for i in _anchor_idx])
            # wipe phase-1 reconstruction artifacts (NOT da3_run — the anchors live there)
            for _pat in ("chunk_*.ply", "chunk_*_origins.npz", "chunk_*_meta.json"):
                for _f in output_dir.glob(_pat):
                    _f.unlink(missing_ok=True)
            for _name in ("maplong_run", "omega_run", "frame_list.json", "intrinsic.txt",
                          "camera_poses.txt", "camera_poses.txt.prescale",
                          "camera_poses.txt.preorient", "camera_frames.txt",
                          "camera_poses_mapanything.json",
                          ".metric_scale_applied", ".orientation_applied"):
                _t = output_dir / _name
                if _t.is_dir():
                    shutil.rmtree(_t, ignore_errors=True)
                elif _t.exists():
                    _t.unlink()
            vggt_config = _build_vggtomega_config(config)
            _apply_chunked_metric(vggt_config, _chunk, _ov)
            _apply_conf_filter(vggt_config)
            _chunked_already = True
            if not _omega_pass(vggt_config, "chunked-metric"):
                return
            _walk_m = _metricize_and_orient(vggt_config, "chunked-metric")
        else:
            pipe.send_log(f"[chunk-plan] walk {_walk_m:.1f} m > comfort {_max_walk:g} m "
                          f"but only {_n_selected} keyframes (one chunk) — keeping the "
                          f"single pass", level="warning")

    # Success → free da3_run when the TSDF won't use it (depth_source not DA3-based).
    _ds = str((config.get("tsdf", {}) or {}).get("depth_source", "auto")).lower()
    if _ds not in ("da3", "da3_frames", "auto"):
        _da3_run = output_dir / "da3_run"
        if _da3_run.exists():
            _mb = sum(f.stat().st_size for f in _da3_run.rglob("*") if f.is_file()) / (1024 * 1024)
            shutil.rmtree(_da3_run, ignore_errors=True)
            pipe.send_log(f"[scale-align] freed da3_run/ ({_mb:.0f} MB) — TSDF uses omega "
                          f"depth (depth_source={_ds}), DA3 no longer needed")


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
                        # Fail HERE, not an hour later inside CloudCompPy: a desync means
                        # the cleaned cloud cannot carry per-point traceability, and the
                        # merge step aborts anyway (cloudcompy_postprocess: no fallback).
                        raise RuntimeError(
                            f"Chunk {i:03d} (src {K}): inline origins {n} vs PLY {ply_n} — "
                            f"points and origins are out of sync; the confidence mask used "
                            f"for the PLY and for the origins must be identical.")
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
