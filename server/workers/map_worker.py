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
from typing import Optional

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

    # Frame selection 'parallax_lk' (reconstruction.simple.frame_selection, claude_stac.txt
    # §4-F1) owns the frame quality too: intake I0 measures FEATURES (no percentile cull)
    # and writes frames/frame_quality.json itself in the legacy shape, so the blur
    # analysis below must not overwrite it. Resolved here, before Step 1; an unknown
    # value fails now, naming itself.
    _parallax_lk = _resolve_frame_selection(recon_cfg) == "parallax_lk"

    # ── Step 1: Frame quality analysis (blur detection) ──
    # Toggleable via reconstruction.blur_filter (default ON). OFF = keep ALL frames,
    # no Laplacian cull, no frame_quality.json gating in Step 2.
    blur_on = bool(recon_cfg.get("blur_filter", True))
    fq_path = frames_dir / "frame_quality.json"
    if _parallax_lk:
        pipe.send_log("Frame selection 'parallax_lk' → frame quality comes from intake I0 "
                      "(frames/quality_features.json + legacy frame_quality.json) — "
                      "skipping the blur analysis")
    elif not blur_on:
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
        # parallax_lk | motion | fps | dino | hf — anything else is a RuntimeError naming
        # the value (the old coercion turned every other value into a silent 'motion').
        _sel = _resolve_frame_selection(recon_cfg)
        pipe.send_log(f"SIMPLE pipeline ON → frame selection '{_sel}' "
                      f"overrides frames_selector '{mode}'")
        mode = _sel
    sf_path = frames_dir / "selected_frames.json"
    if mode == "parallax_lk":
        # Intake I0 → I1 → I2 in-process (CPU; I2's VLM / SAM3 only when
        # intake.content.enabled). replace on or off, the intake's marker decides step
        # by step what is already measured (intake/run.py): nothing measured is redone,
        # a missing or incomplete step runs.
        _run_intake_selection(pipe, session_path, frames_dir, config, replace)
    elif not replace and sf_path.exists():
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
    elif mode == "hf":
        # The legacy H/F-ratio selector (ORB-SLAM style), selectable by name
        # (claude_stac.txt §4-F1). NO FALLBACK, the dino pattern: it writes
        # selected_frames.json itself and its failure is the stage's failure.
        from frames.selector import select_keyframes_hf
        pipe.send_progress(5, "Selecting keyframes (blur + H/F ratio)...", stage="reconstruction")
        sel = select_keyframes_hf(str(frames_dir), config.get("frame_selection", {}))
        pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes (hf)")
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

    # ── NO SEMANTICS HERE (USER 2026-10-05: "intake, da3 para medir, omega, f0 a f6,
    # octree, época publicada, vlm, sam3, máscaras, correcciones"): the VLM and SAM3
    # are pipeline STAGES after the cloud stage (pipeline_manager.DEFAULT_STAGE_ORDER:
    # reconstruction → cloudcompy → vlm → sam3 → certify), so the published cloud
    # reaches the viewer before the hours of segmentation; the SAM3 stage projects
    # its masks on the cloud it finds on disk. (2026-09-28 → 2026-10-05 they ran
    # here, before any geometry.)

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
        if mode == "parallax_lk":
            # The witness frames ARE the dense set (claude_stac.txt §3: keyframes for
            # Ω / BA, witnesses for depth): witness_frames.json ∪ selected_frames.json,
            # regardless of blur_filter (intake I0 replaced the blur cull). DECLARED:
            # witnesses have no cap (§4-F1), and every legacy consumer of this file
            # runs over ALL of them — the dense BA tracking (bundle_adjust, off) and
            # the mapanything backend's non-cond DA3 set. The production vggtomega path
            # does not read it (its DA3 anchors come from the keyframes).
            _dpath, _doc = _write_intake_da3_frames(frames_dir)
            da3_dense_frames_path = str(_dpath)
            pipe.send_log(f"DA3-dense set: {_doc['selected_count']} frames = "
                          f"{_doc['n_witness']} witness ∪ {_doc['n_keyframes']} keyframes "
                          f"of {_doc['total_frames']} (parallax_lk) → da3_frames.json — a "
                          f"legacy DA3 / dense-BA consumer runs over all of them")
        elif mode == "dino" and blur_on:
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
    elif backend == "vggtomega_pgsr":
        # PRECISION MODE (precision task, Phase D): the full vggtomega pipeline runs
        # first and its output initializes a per-scene PGSR photometric optimization
        # at NATIVE resolution (fixed initial poses + cloud as Gaussian seed; planar +
        # multi-view geometric regularization; SAM3 dynamic masks excluded from the
        # loss; optional photometric pose refinement by flag). The trainer exports
        # rendered depths → the TSDF integrates them via depth_source "pgsr_render".
        # NOTE: the PGSR stage itself runs AFTER CloudCompPy (it seeds from the
        # cleaned cloud) — pipeline_manager triggers _run_pgsr_stage there.
        _run_vggtomega(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
    elif backend == "mapanything":
        _run_mapanything(pipe, frames_dir, output_dir, selected_frames_path, recon_cfg, config)
    else:
        # FAIL LOUD: an unknown backend used to fall through silently to the
        # legacy mapanything path — a typo or a stale UI selection would run a
        # visibly worse pipeline with no error. Removed backends (gaus_slam*,
        # nerfstudio) land here too.
        raise RuntimeError(
            f"unknown reconstruction backend '{backend}' — valid: vggtomega_pgsr "
            f"(default), vggtomega, da3, mapanything, hybrid, hybrid_cond, lidar")

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
    # ── E-full pose_refine: JOINT point-to-plane optimization of all frame
    # poses (multi-view global ICP + smoothness priors) — attacks the root
    # defect Phase A isolated: feed-forward per-frame poses never reconciled.
    # Self-gated: a fresh post-solve measurement must show the cloud disagrees
    # with itself less, else it applies nothing. Best-effort like finereg. ──
    if (recon_cfg.get("pose_refine", {}) or {}).get("enabled", True):
        _run_pose_refine_step(pipe, output_dir, recon_cfg)
    # (DINOv3 fases 3/5 — feature-metric refine, scene/object anchor —
    # DELETED by USER ORDER 2026-09-05: "eliminá todo lo de dinov3 y fase 5")

    # ── THE PRECISION CORE, INSIDE THE RECONSTRUCTION (USER 2026-09-28: "f0 a f7 es
    # etapa de reconstrucción, antes de cloudcompy"): F0 → F5 refine camera and poses
    # against the images, the depth stage (F6 bend: Omega's depth bent to F5 per
    # keyframe + multi-view vote, USER 2026-10-01) publishes THE cloud with its
    # octree, the chunk check reports on it.
    _run_precision_core(pipe, session_path, output_dir, config)


def _run_precision_core(pipe: WorkerPipe, session_path: Path, output_dir: Path,
                        config: dict) -> None:
    """The precision core on the reconstruction this stage just produced, then keep
    only the published cloud.

    The core needs NO cloud until its depth stage (USER 2026-09-29: "¿por qué
    filtramos una nube que no vamos a usar todavía?"): F0 reads Omega's K, F2 the
    DA3 windows and Omega's depth, F4 the frames, F5 the tracks — F2 and F5 publish
    epochs of POSES only (precision/poses_epoch.py) — and the depth stage (F6 bend,
    cloud.source omega_bent; or the F6 sweep → F7 chain) builds the cloud and
    publishes it with its octree. Omega's chunk PLYs (its raw cloud) are never
    merged, filtered or shown; they are deleted with the pose-only epochs once the
    cloud is published (USER 2026-09-28: "no quiero ninguna época 0, la única para
    visualizar debe ser la N, el resto deben descartarse"). No Omega comparison
    cloud is built any more (USER 2026-10-05: the ORIGINAL the final is compared
    with is the published cloud itself — the certification keeps its epoch stored
    and selectable unless `certify.single_final_epoch` is on).
    workers/precision_worker.py runs the core (precision/runner.py's step list,
    each step its own subprocess, resumable). Gated by
    reconstruction.precision.enabled."""
    from precision.config import load_precision_config
    from workers.base import run_stage_inline
    if not load_precision_config(config).enabled:
        return

    pipe.send_progress(84, "Precision core F0 → F6 bend...", stage="reconstruction")
    run_stage_inline(pipe, "workers.precision_worker", str(session_path), config,
                     label="precision", pct_range=(84.0, 99.0))
    from precision.product import product_is_live
    _live, _why = product_is_live(output_dir)
    if not _live:
        raise RuntimeError(f"the precision core ended without a published cloud — {_why}")
    pipe.send_log(f"[precision] product: {_why}")

    freed = _discard_previous_epochs(output_dir)
    pipe.send_log(f"[epochs] the published cloud is the ORIGINAL epoch the viewer shows — "
                  f"{freed / 1048576:.0f} MB of pose-only epochs and Omega chunks discarded")
    pipe.send_progress(99, "Published cloud is the reconstruction", stage="reconstruction")


def _discard_previous_epochs(output_dir: Path) -> int:
    """Delete every stored epoch (`_epoch_*/`: the gauge's and the refine's pose-only
    states — no cloud in them), `_tx_epoch_*/` leftovers and Omega's raw chunk PLYs
    (`chunk_*.ply` with its origins/meta — never merged). The live epoch — the
    published cloud — is the ORIGINAL the certification's epoch is compared with
    (USER 2026-10-05); nothing older is worth a byte. Returns the bytes freed. The
    ledger (corrections.jsonl) and geometry_epoch.json stay: the live epoch is the
    published one and the record says so."""
    freed = 0
    for pattern in ("_epoch_*", "_tx_epoch_*"):
        for d in output_dir.glob(pattern):
            if d.is_dir() and not d.is_symlink():
                freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
    for pattern in ("chunk_*.ply", "chunk_*_origins.npz", "chunk_*_meta.json"):
        for f in output_dir.glob(pattern):
            if f.is_file():
                freed += f.stat().st_size
                f.unlink(missing_ok=True)
    # corrections/epoch_*.npz STAY: they are small, and selecting epoch 0 walks the
    # transforms of every epoch between it and the live one (pccr 2026-09-30)
    return freed


def _run_pose_refine_step(pipe: WorkerPipe, output_dir: Path, recon_cfg: dict):
    """E-full over output/chunk_*.ply (reproject_chunks contract): global
    per-frame pose refinement. Skips itself when inputs are missing."""
    server_dir = Path(__file__).resolve().parent.parent
    py = sys.executable
    pipe.send_progress(69, "Global pose refinement (multi-view consensus)...",
                       stage="reconstruction")
    cmd = [py, "-m", "reconstruction.pose_refine",
           "--output-dir", str(output_dir)]
    pcfg = recon_cfg.get("pose_refine", {}) or {}
    for key, flag in (("pair_window", "--pair-window"),
                      ("samples_per_frame", "--samples"),
                      ("rel_tol", "--rel-tol"),
                      ("max_depth", "--max-depth"),
                      ("near_ref", "--near-ref"),
                      ("odo_weight", "--odo-weight"),
                      ("leash_weight", "--leash-weight"),
                      ("min_gain", "--min-gain"),
                      ("outer_iters", "--outer-iters")):
        if pcfg.get(key) is not None:
            cmd += [flag, str(pcfg[key])]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=str(server_dir))
    for line in p.stdout:
        pipe.send_log("[pose-refine] " + line.rstrip())
    if p.wait() != 0:
        pipe.send_log("[pose-refine] ⚠ pose refinement failed — continuing "
                      "with unrefined poses", level="warning")


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
    if fcfg.get("anneal_rounds") is not None:
        cmd += ["--anneal-rounds", str(fcfg["anneal_rounds"])]
    if fcfg.get("max_total_correction_m") is not None:
        cmd += ["--max-total-correction", str(fcfg["max_total_correction_m"])]
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


def _apply_stac_model_keys(cfg: dict, config: dict) -> dict:
    """Every STAC-level Model key that must reach the vendor WHATEVER the layout.

    It lives in the BUILDERS, not at a call site: the confidence floor first went
    into one branch of `_run_mapanything` (which the production backend never
    takes) and three full reconstructions ran with it at 0. Anything written here
    travels on every path by construction.

    ── confidence floor ──
    USER 2026-09-23: ONE confidence floor governs the whole reconstruction.

    *"deben desaparecer de la nube eh!, porque no quiero que se hagan ajustes de
    pose sobre ruido, sobre puntos de baja confianza que es lo que tal vez rompe
    los ajustes de pose y piso"*. `reconstruction.simple.conf_min_norm` is a
    min-max fraction of each chunk's own valid confidences — the same arithmetic
    the viewer slider uses — and the vendor applies it in the two places that
    matter: the pose-fit correspondence sampler and the PLY writer.

    ── sky mask ──
    `skyseg.onnx` zeroes the confidence of every pixel it calls sky, and those
    pixels then fall under the percentile and LEAVE THE CLOUD. The vendor reads
    `Model.mask_sky` with a default of True and NOTHING in this repo ever wrote
    the key — not a builder, not a YAML — so the filter ran on every frame of
    every scene, indoor ones included. Measured 2026-09-23 over the cached masks:
    pccr 1.7 % of pixels on average (peak 12 % of one frame), observatorio 1.2 %,
    test2 0.5 %, and on observatorio's worst frame the masked band is the CONCRETE
    WALKWAY between the rails, not sky. It is now a declared key with its current
    behaviour as the default — turning it off is a geometry decision, and the
    defect was that it could not be decided at all.
    """
    _simple = ((config.get("reconstruction", {}) or {}).get("simple", {}) or {})
    m = cfg.setdefault("Model", {})
    m["pose_fit_conf_min_norm"] = float(_simple.get("conf_min_norm", 0.0) or 0.0)
    m["mask_sky"] = bool(_simple.get("mask_sky", True))
    return cfg


def _omega_grid_wh(frames_dir) -> tuple:
    """(w, h) of the Omega grid for this session's frames in max_size mode: the native frame
    rounded to Omega's patch (omega_native_resolution) and the short side by the aspect."""
    import cv2
    from intake.quality import list_frames
    paths = list_frames(Path(frames_dir))
    if not paths:
        raise RuntimeError(f"no frame in {frames_dir} — the Omega grid cannot be sized")
    img = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {paths[0]}")
    h, w = img.shape[:2]
    res = omega_native_resolution(frames_dir)
    s_ = float(res) / float(max(w, h))
    return int(round(w * s_)), int(round(h * s_))


def omega_native_resolution(frames_dir) -> int:
    """``reconstruction.vggtomega.resolution: native`` — the frames' long side rounded
    up to Omega's patch: with mode ``max_size`` the Omega grid IS the native frame
    (pccr 464x832 → 832, grid 464x832, scale 1.0; the old 512 balanced saw 384x688)."""
    import cv2 as _cv2m
    from precision.camera import OMEGA_PATCH_SIZE
    from intake.quality import list_frames
    paths = list_frames(Path(frames_dir))
    if not paths:
        raise RuntimeError(f"no frame in {frames_dir} — the native resolution cannot be read")
    img = _cv2m.imread(str(paths[0]), _cv2m.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {paths[0]}")
    long_side = max(img.shape[:2])
    return int(-(-long_side // OMEGA_PATCH_SIZE) * OMEGA_PATCH_SIZE)


def _build_vggtomega_config(config: dict, frames_dir=None) -> dict:
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
    _res = om.get("resolution", cfg["Model"].get("omega_resolution", 512))
    _mode = om.get("mode", cfg["Model"].get("omega_mode", "balanced"))
    if _res == "native":
        if frames_dir is None:
            raise RuntimeError("reconstruction.vggtomega.resolution is 'native': the Omega "
                               "config needs the session's frames to read it")
        _res, _mode = omega_native_resolution(frames_dir), "max_size"
    cfg["Model"]["omega_resolution"] = int(_res)
    cfg["Model"]["omega_mode"] = _mode
    return _apply_stac_model_keys(cfg, config)


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
        # the raw chunk extrinsic is in another frame and scale (it underestimated s
        # by ~1.37x): records built on it would be wrong, not approximate
        raise RuntimeError("[omega-depth] no aligned camera_poses.txt / camera_frames.txt "
                           "pair — the Omega records cannot be written in the aligned frame")

    # the chunk layout exactly as the fork builds it (vggt_long.py), and every frame's
    # OWNER — the chunk whose centre is nearest (loop_utils.metric_lock.frame_owner):
    # the record of a shared frame carries the depth, conf, K and pose of the chunk
    # that writes its points (traceability), never of whichever chunk came last
    if N <= chunk_size or step <= 0:
        _chunks = [(0, N)]
    else:
        _chunks = [(i * step, min(i * step + chunk_size, N))
                   for i in range((N - overlap + step - 1) // step)]
    _centres = [(a + b) / 2.0 for a, b in _chunks]

    def _owner(g):
        best, bd = -1, None
        for kk, (a, b) in enumerate(_chunks):
            if a <= g < b and (bd is None or abs(g - _centres[kk]) < bd):
                best, bd = kk, abs(g - _centres[kk])
        return best

    n_written = 0
    # numeric chunk order (a lexicographic glob put chunk_10 before chunk_2)
    for cp in sorted(glob.glob(str(aligned / "chunk_*.npy")),
                     key=lambda q: int(Path(q).stem.split("_")[1])):
        try:
            k = int(Path(cp).stem.split("_")[1])
            cd = np.load(cp, allow_pickle=True).item()
            wp = np.asarray(cd["world_points"])            # [S,H,W,3] aligned world
            if wp.ndim == 5:
                wp = wp[0]
            ext = np.asarray(cd["extrinsic"])              # [S,4,4] c2w (RAW — fallback only)
            if ext.ndim == 4:
                ext = ext[0]
            # claude_stac.txt §4-F3: every keyframe's omega record carries what the
            # later stages read — depth, conf, the grid K and the aligned c2w
            wconf = cd.get("world_points_conf")
            wconf = (np.asarray(wconf).reshape(wp.shape[:3]) if wconf is not None else None)
            Kin = cd.get("intrinsic")
            Kin = np.asarray(Kin) if Kin is not None else None
            if Kin is not None and Kin.ndim == 4:
                Kin = Kin[0]
            S = wp.shape[0]
            start = k * step
            for j in range(S):
                gi = start + j
                if gi >= N:
                    break
                if _owner(gi) != k:
                    continue
                if stems[gi] not in pose_map:
                    raise RuntimeError(f"[omega-depth] frame {stems[gi]} has no aligned pose")
                c2w = pose_map[stems[gi]]                 # the ALIGNED pose of record
                cam_c = c2w[:3, 3]
                fwd = c2w[:3, 2]                           # camera +z in world
                d = (wp[j] - cam_c) @ fwd                  # [H,W] depth along view axis
                rec = {"depth": d.astype(np.float32),
                       "pose_c2w": np.asarray(c2w, np.float64),
                       "chunk": np.int64(k), "frame_global": np.int64(gi)}
                if wconf is not None:
                    rec["conf"] = wconf[j].astype(np.float32)
                if Kin is not None:
                    rec["K_omega"] = Kin[j].astype(np.float64)
                np.savez_compressed(out_dir / f"frame_{stems[gi]}.npz", **rec)
                n_written += 1
        except Exception as e:
            raise RuntimeError(f"[omega-depth] chunk {cp} could not be recorded: {e}") from e
    pipe.send_log(f"[omega-depth] wrote {n_written} per-frame omega records (depth, conf, "
                  f"K_omega, pose_c2w)")


from workers.base import (gpu_free_gb as _gpu_free_gb, gpu_total_gb as _gpu_total_gb,
                          stop_semantic_service,
                          stop_semantic_service_verified)


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


# ── frame selection 'parallax_lk' (intake I0 → I1 → I2; claude_stac.txt §4-F1) ──

FRAME_SELECTIONS = ("parallax_lk", "motion", "fps", "dino", "hf")


def _resolve_frame_selection(recon_cfg: dict) -> str:
    """The effective frame-selection mode of a reconstruction config (pure, no I/O).

    ``reconstruction.frames_selector`` (legacy: none | stride | dino | parallax | fps |
    motion) — unless the SIMPLE pipeline is on and that selector is neither fps nor
    motion: then ``reconstruction.simple.frame_selection`` overrides it and must be one
    of FRAME_SELECTIONS. Any other value, or a missing key, is a RuntimeError naming
    it. Until 2026-09-27 anything but fps|motion was coerced to 'motion' silently, so
    dino/hf could never be chosen from the SIMPLE block and a typo ran the wrong
    selector without a word.

    DELIBERATE CONTRACT CHANGE (2026-09-27): with the SIMPLE pipeline on, the key is
    MANDATORY — the old code defaulted a missing key to 'motion'. A session YAML or a
    config without it now fails at Step 1 for every backend, naming the key: that is
    a configuration error to fix, not a regression (the repo rule: no silent
    default). config.yaml carries it (parallax_lk)."""
    recon_cfg = recon_cfg or {}
    mode = str(recon_cfg.get("frames_selector", "none")).lower()
    simple = recon_cfg.get("simple") or {}
    if not bool(simple.get("enabled", False)) or mode in ("fps", "motion"):
        return mode
    if "frame_selection" not in simple:
        raise RuntimeError("reconstruction.simple.frame_selection is missing — set one of "
                           f"{FRAME_SELECTIONS} (there is no silent default)")
    sel = str(simple.get("frame_selection")).lower()
    if sel not in FRAME_SELECTIONS:
        raise RuntimeError(f"reconstruction.simple.frame_selection = {sel!r} is not a frame "
                           f"selection — valid: {FRAME_SELECTIONS}")
    return sel


def _ensure_semantic_or_fail(pipe: WorkerPipe, config: dict) -> None:
    """Intake I2 (VLM content tags) needs the semantic service: healthcheck, auto-start
    and wait exactly as the VLM worker does (semantic.service.ensure_service). When it
    does not come up the stage FAILS with the reason the service reported — the intake
    never tags nothing and calls it done."""
    from semantic.service import ensure_service
    said = []

    def _log(m):
        said.append(str(m))
        pipe.send_log(m)

    svc = (config.get("semantic") or {}).get("service") or {}
    pipe.send_progress(4, "Intake I2: semantic service (Qwen3-VL) for the content tags...",
                       stage="reconstruction")
    if ensure_service(config, log=_log, cancelled=pipe.check_cancel):
        return
    if pipe.check_cancel():
        raise RuntimeError("cancelled while waiting for the semantic service (intake I2)")
    reason = said[-1] if said else "ensure_service returned False without a message"
    raise RuntimeError(
        f"intake I2 (content tags; intake.content.enabled: true) needs the semantic service "
        f"at http://{svc.get('host', '127.0.0.1')}:{svc.get('port', 8799)} and it is not "
        f"reachable — {reason}. Start it (bash scripts/serve_semantic.sh) or set "
        f"intake.content.enabled: false; the intake does not fall back to untagged frames")


def _run_intake_selection(pipe: WorkerPipe, session_path: Path, frames_dir: Path,
                          config: dict, replace: bool) -> None:
    """Frame selection 'parallax_lk': intake I0 → I1 → I2 in-process.

    ALWAYS through intake.run.run_intake and its marker (<session>/intake/
    intake_state.json): a step whose parameters, frame inventory and upstream
    inputs match the marker is skipped — nothing already measured is re-measured
    or overwritten — and a step that is missing or incomplete runs. So replace=off
    reuses a finished intake as-is, and a previous run that died in I2 (the
    semantic service did not come up, SAM3 failed) gets its I2 on the retry
    instead of silently going on without content tags or exclusion masks (the old
    shortcut reused selected_frames.json + witness_frames.json without reading the
    marker). With intake.content.enabled the semantic service is ensured right
    before I2 runs (a service that does not come up fails the stage with the
    reason) and stopped — and verified gone — after the last tag, before the first
    SAM3 call. A cancel is honoured inside every intake loop
    (intake.quality.IntakeCancelled names where)."""
    from intake.config import load_intake_config
    from intake.run import run_intake

    pipe.send_log(f"Frame selection 'parallax_lk' (replace={'on' if replace else 'off'}): the "
                  f"intake marker decides step by step what is already measured")
    icfg = load_intake_config(config)
    pipe.send_progress(3, "Intake: quality features → parallax keyframes"
                       + (" → content tags" if icfg.content.enabled else "") + "...",
                       stage="reconstruction")
    # I2 (content tags + exclusion masks) is OFF by default since 2026-10-05 (USER:
    # "olvidate de la exclusión de personas y objetos en movimiento"); it stays
    # selectable (intake.content.enabled). When on, GPU exclusivity is the SAM3
    # stage's own rule: vLLM serves the I2 tags, then is stopped — VERIFIED (no
    # 'vllm serve' left) — before SAM3 segments the exclusion masks; the VLM stage
    # after the cloud stage restarts it (ensure_service).

    def _before_sam3():
        # the verification travels into content_tags.json (sam3_handover)
        return stop_semantic_service_verified(pipe, stage="intake I2 SAM3")

    # the FIRST GPU step of the intake is the DA3 focal probe (zaragoza 2026-10-04: it ran with
    # vLLM loading beside it and died in OOM) — the card is handed over there, verified; I2's
    # VLM brings vLLM back (before_content) and hands it over again before SAM3
    from intake.focal import default_probe
    _probe = default_probe()

    def _focal(*a, **k):
        stop_semantic_service_verified(pipe, stage="intake focal probe (DA3)")
        return _probe(*a, **k)

    res = run_intake(session_path, icfg, log=pipe.send_log, progress=None,
                     before_content=lambda: _ensure_semantic_or_fail(pipe, config),
                     before_sam3=_before_sam3, focal=_focal,
                     cancelled=pipe.check_cancel)
    ran = [k for k, v in res["steps"].items() if v.get("ran")]
    pipe.send_log(f"Intake steps run this time: {ran or 'none (every marker matched)'}")
    s = res["summary"]
    if s["n_keyframes"] < 2:
        # two views is the structural minimum of a multi-view reconstruction (the fps
        # and motion branches stop at the same count)
        raise RuntimeError(f"parallax_lk selection produced {s['n_keyframes']} keyframe(s) "
                           f"(quantum {icfg.parallax.parallax_quantum_px:g} px of measured "
                           f"parallax) — not enough to reconstruct; see "
                           f"{res['artifacts']['coverage_warnings']}")
    pipe.send_log(f"Frame set: {s['n_keyframes']}/{s['n_frames']} keyframes (parallax_lk, "
                  f"quantum {icfg.parallax.parallax_quantum_px:g} px), {s['n_witness']} witness "
                  f"frames, {s['n_warnings']} coverage warning(s) → selected_frames.json / "
                  f"witness_frames.json")


def _intake_da3_frames(frames_dir: Path) -> dict:
    """da3_frames.json for frame selection 'parallax_lk': the witness frames ∪ the
    keyframes (frames/witness_frames.json ∪ frames/selected_frames.json, both written by
    intake I1; keyframes ⊂ witnesses by construction, the union guards the contract).
    v2 frame-list shape (``version`` "2.0", method 'parallax_lk_witness') plus the
    artifact stamps (provenance, geometry_epoch, camera_epoch — carried over from the
    intake's selected_frames.json — and intake_version); extra keys every v2 reader
    ignores. ``total_frames`` is COUNTED on disk (the frame inventory) and must equal
    what both intake documents recorded. RuntimeError naming a missing file or key, or
    an inventory that disagrees."""
    from intake.quality import QualityError, list_frames
    docs = {}
    for name in ("selected_frames.json", "witness_frames.json"):
        p = frames_dir / name
        if not p.exists():
            raise RuntimeError(f"{p} does not exist — intake I1 did not run")
        doc = json.loads(p.read_text())
        for key in ("selected_files", "total_frames"):
            if key not in doc:
                raise RuntimeError(f"{p} lacks '{key}' (the v2 frame-list contract)")
        if not isinstance(doc["selected_files"], list):
            raise RuntimeError(f"{p} 'selected_files' is not a list (the v2 frame-list "
                               f"contract)")
        docs[name] = doc
    sel = docs["selected_frames.json"]
    for key in ("provenance", "geometry_epoch", "camera_epoch"):
        if key not in sel:
            raise RuntimeError(f"{frames_dir / 'selected_frames.json'} lacks '{key}' — it "
                               f"was not written by intake I1 (re-run the intake)")
    try:
        total = len(list_frames(frames_dir))
    except QualityError as e:
        raise RuntimeError(str(e)) from e
    for name, doc in docs.items():
        if int(doc["total_frames"]) != total:
            raise RuntimeError(f"{frames_dir / name} records total_frames "
                               f"{doc['total_frames']} but {frames_dir} holds {total} frame(s) "
                               f"— the intake ran on another frame inventory; re-run it")
    kf = sel["selected_files"]
    wit = docs["witness_frames.json"]["selected_files"]
    files = sorted(set(kf) | set(wit), key=lambda f: int(os.path.splitext(f)[0]))
    return {"version": "2.0", "method": "parallax_lk_witness", "total_frames": total,
            "selected_count": len(files), "selected_files": files,
            "provenance": sel["provenance"], "geometry_epoch": int(sel["geometry_epoch"]),
            "camera_epoch": int(sel["camera_epoch"]),
            "intake_version": docs["witness_frames.json"].get("version"),
            "n_keyframes": len(kf), "n_witness": len(wit)}


def _write_intake_da3_frames(frames_dir: Path) -> tuple:
    """Build :func:`_intake_da3_frames` and write frames/da3_frames.json atomically
    (tmp + os.replace — a reader never sees half a file). Returns (path, doc)."""
    doc = _intake_da3_frames(frames_dir)
    path = frames_dir / "da3_frames.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, path)
    return path, doc


def _run_da3_anchor(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                    anchor_files: list, recon_cfg: dict) -> None:
    """ISOLATED per-frame DA3 metric depth on the K scale-anchor frames — NO streaming.
    The streaming pipeline chains poses across consecutive frames; anchor frames are
    seconds apart, the chain breaks (pose=None → crash) and none of its machinery is
    needed: the scale is a per-pixel depth RATIO, poses don't participate. Runs
    extract_da3_depth.py (--per_frame) and converts its output to the exact layout
    scale_align consumes: da3_run/results_output/frame_<num>.npz (depth + conf)."""
    from reconstruction.da3_anchor import extract_anchor_depths
    model_id = str((recon_cfg.get("da3", {}) or {}).get(
        "model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"))
    n = extract_anchor_depths(frames_dir, output_dir, anchor_files, model_id,
                              python=sys.executable, log=pipe.send_log,
                              check_cancel=pipe.check_cancel)
    if n == 0:
        pipe.send_log("DA3 anchor extraction cancelled by user", level="warning")


def math_deg(rad: float) -> float:
    import math
    return math.degrees(float(rad))


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
        # 0 (or >= the selection) = EVERY selected keyframe anchors — the
        # `or 12` that used to sit here turned a deliberate 0 back into 12.
        _k = int(_simple_cfg.get("scale_anchor_frames", 12))
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
        # ONE pass, ONE DA3 round — USER ORDER 2026-09-22: *"no quiero que haga
        # dos pasadas de da3, despues vggt omega para luego ir otra vez a da3 y
        # vggt omega pero con chunks, quiero que lo haga de una"*. With a FIXED
        # chunk size the chunk layout, and therefore the per-chunk metric
        # anchors, is known BEFORE any inference — so they are extracted in THIS
        # round instead of costing a second DA3 launch (model load included) in
        # between the two Omega passes the walk probe used to need.
        _cf_anchor = int(_simple_cfg.get("chunk_frames", 0) or 0)
        if _anchor_files and _cf_anchor and _n_selected > _cf_anchor:
            from reconstruction.chunk_plan import plan_anchor_indices as _pai
            _chunk_anchor_files = [
                _sel_files[i] for i in _pai(_n_selected, _cf_anchor, _cf_anchor // 2,
                                            int(_simple_cfg.get("chunk_anchors", 3)))]
            _extra = sorted(set(_chunk_anchor_files) - set(_anchor_files))
            if _extra:
                _anchor_files = sorted(set(_anchor_files) | set(_extra))
                with open(output_dir / "scale_anchor_frames.json", "w") as _f:
                    json.dump({"version": "2.0",
                               "method": f"scale_anchor_{_k}+chunk_{_cf_anchor}",
                               "total_frames": _n_selected,
                               "selected_count": len(_anchor_files),
                               "selected_files": _anchor_files}, _f)
                pipe.send_log(f"SIMPLE: + {len(_extra)} per-chunk anchor(s) for the "
                              f"{_cf_anchor}/{_cf_anchor // 2} layout in the SAME DA3 "
                              f"round → {len(_anchor_files)} frames, ONE extraction")
    # ── F2 (claude_stac.txt §4-F2): I3 DA3 windows → the metric WALK, BEFORE Omega ──
    # The walk sizes the chunks (I4 below) and the windows leave every keyframe's
    # metric anchor on disk — the per-frame DA3 round that follows finds them all
    # and skips. A single Omega pass is no instrument for it: over pccr 2026-08-24
    # it read 1526.6 m for a walk its chunked run measured at 104.8 m.
    _walk_doc = None
    if _scale_align_on and _sel_files:
        from precision.config import load_precision_config
        _pc = load_precision_config(config)
        if _pc.enabled:
            from intake.walk import run_da3_windows, measure_walk
            pipe.send_progress(5, "Gauge I3: DA3 multi-view windows → metric walk...",
                               stage="reconstruction")
            run_da3_windows(output_dir.parent, _pc.gauge, sys.executable, log=pipe.send_log,
                            check_cancel=pipe.check_cancel, frames_dir=frames_dir,
                            files=_sel_files)
            _walk_doc = measure_walk(output_dir.parent, _pc.gauge, log=pipe.send_log)
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

    # ── VGGT-Long with the Omega backbone — ONE PASS ──
    # USER ORDER 2026-09-22: *"quiero que lo haga de una, si hay muchos kf lo chunkee
    # y son menos que lo haga en uno solo siempre 60/30"*. `chunk_frames` decides,
    # from the KEYFRAME COUNT alone, before any inference:
    #   n_kf <= chunk_frames → ONE chunk, overlap 0, loop closure off: no seams.
    #   n_kf >  chunk_frames → chunked-metric DIRECTLY at chunk_frames/2 overlap,
    #     each chunk metric-locked to DA3 anchors BEFORE alignment, glued SE(3)
    #     (scale is not negotiable — the Sim3 freedom is what produced the onion),
    #     SALAD loop closure + pose graph on.
    # There is NO second pass and NO walk comfort limit. The walk USED to decide
    # both (`max_walk_single_pass_m`, `chunk_walk_m` — both REMOVED): it measured
    # 44.1 m on pccr's ~19 m walk and that one number re-ran the whole
    # reconstruction AND sized its chunks from the error. The walk is still
    # measured and reported — it is evidence, not a verdict.
    from reconstruction.chunk_plan import (walk_length_m, plan_anchor_indices,
                                           plan_chunks, chunk_ranges)
    vggt_config = _build_vggtomega_config(config, frames_dir)
    _va_cfg = recon_cfg.get("vggtomega", {}) or {}
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

    def _persist_chunk_plan(_chunk, _ov, _n_kf, _phase, _walk=None):
        """Persist the REAL chunk plan (USER 2026-09-08: the correction module
        may group evidence only by the reconstruction's actual chunks, never by
        a fixed divisor). Written whenever a chunked run is configured; a
        single-pass session has no plan (the corrector then works purely per
        keyframe)."""
        plan = {
            "version": 1,
            "phase": _phase,
            "n_keyframes": int(_n_kf),
            "chunk_size": int(_chunk),
            "overlap": int(_ov),
            "chunk_ranges": [[int(a), int(b)] for a, b in
                             chunk_ranges(int(_n_kf), int(_chunk), int(_ov))],
            "walk_m": (round(float(_walk), 2) if _walk is not None else None),
        }
        _invalidate_on_new_chunk_plan(output_dir, plan, pipe.send_log)
        (output_dir / "chunk_plan.json").write_text(json.dumps(plan, indent=1))
        pipe.send_log(f"[chunk-plan] persisted output/chunk_plan.json: "
                      f"{len(plan['chunk_ranges'])} chunk(s), size {_chunk}, "
                      f"overlap {_ov}")

    def _apply_chunked_metric(cfg_v, _chunk, _ov):
        cfg_v["Model"]["chunk_size"] = int(_chunk)
        cfg_v["Model"]["overlap"] = int(_ov)
        cfg_v["Model"]["loop_enable"] = True
        cfg_v["Model"]["using_sim3"] = False       # SE(3): scale locked by the anchors
        # USER ORDER 2026-09-04 ("deja omegalong como corresponde, sin
        # agregados adicionales de ajustes; la escala de DA3 queda; si se
        # detecta zoom se corrige la escala"): every adjustment stage is now
        # CONFIG-GATED and OFF by default — nothing deleted, everything
        # selectable in config.yaml. What stays on: the vendor baseline
        # (chunked alignment + loop closure) and the DA3 metric scale
        # (metric_lock + seam graph). Zoom chunks get their broken DA3
        # anchors excluded and their scale from the seam graph, and the
        # health gate is diagnostic-only (no more declared holes).
        cfg_v["Model"]["metric_lock"] = {
            "enable": True,
            "anchor_dir": str(_anchor_dir),
            "near_frac": float(_va_cfg.get("scale_near_frac", 0.25)),
            # per-chunk LINEAR scale drift (self-gated) — an adjustment: off
            "scale_drift": bool(_va_cfg.get("scale_drift", False)),
            "suspect_spread": float(_va_cfg.get("suspect_spread", 0.30)),
            # zoom → correct the scale (anchor exclusion + seam graph)
            "zoom_scale_fix": bool(_va_cfg.get("zoom_scale_fix", True)),
            # THE WEIGHTS OF THE IN-RUN SCALE LADDER. Nothing ever wrote these,
            # so the lock solved with the vendor's own fallbacks (0.003 / 0.08)
            # while the POST-HOC solver of the same quantity used the configured
            # pair (certify.scale sigma_seam_log 0.02 / sigma_anchor_log 0.03) —
            # a seam 27x stiffer and an anchor 2.7x softer, undeclared. The
            # defaults below ARE the vendor's, so this changes no geometry; what
            # changes is that the numbers now exist where they can be decided.
            # CLAUDE.md's ladder budget ("6 x sigma_seam_log ~ 12 %") is written
            # against the post-hoc pair, so the two do not match by construction.
            "sigma_seam": float(_va_cfg.get("sigma_seam", 0.003)),
            "sigma_anchor": float(_va_cfg.get("sigma_anchor", 0.08)),
        }
        cfg_v["Model"]["exact_seam_align"] = bool(
            _va_cfg.get("exact_seam_align", False))
        cfg_v["Model"]["frame_ownership"] = bool(
            _va_cfg.get("frame_ownership", False))
        cfg_v["Model"]["ownership_backfill"] = bool(
            _va_cfg.get("ownership_backfill", False))
        cfg_v["Model"]["elastic_seam"] = bool(
            _va_cfg.get("elastic_seam", False))
        cfg_v["Model"]["elastic_smooth_win"] = int(
            _va_cfg.get("elastic_smooth_win", 5))
        # null/0 = NO absolute cap (USER 2026-09-16: proportion decides, not
        # size — metric_lock.demote_disproportionate_fits). Only a real number
        # still applies the legacy |t| ceiling on top of it.
        _emt = _va_cfg.get("elastic_max_t_m", None)
        cfg_v["Model"]["elastic_max_t_m"] = (float(_emt) if _emt not in (None, "", False)
                                             else None)
        cfg_v["Model"]["intra_chunk"] = bool(
            _va_cfg.get("intra_chunk", False))
        cfg_v["Model"]["depth_graph"] = bool(
            _va_cfg.get("depth_graph", False))
        cfg_v["Model"]["blend_copies"] = bool(
            _va_cfg.get("blend_copies", False))
        # ── claude_stac.txt F1: exact loop bridges + closed scale graph ──
        # loops:/scale:/correction_graph: are validated here (a missing key
        # aborts naming it) and handed to the fork as Model.loops/Model.scale;
        # the fork imports the spatial gate + the DA3 anchor extractor from
        # stac_server_dir. Absolute scale rows (VIO per chunk, regulated
        # dimensions, a user measurement) join the same solve.
        from reconstruction.loops.config import (load_loops_config, fork_model_loops,
                                                 fork_model_scale, fork_model_graph,
                                                 fork_model_authority, fork_model_certify,
                                                 fork_loop_salad)
        _mg = load_loops_config(config)
        _server_dir = str(Path(__file__).resolve().parent.parent)
        cfg_v["Model"]["loops"] = fork_model_loops(_mg, _server_dir)
        cfg_v["Model"]["scale"] = fork_model_scale(_mg)
        # §4.4 visual candidates: the SALAD retrieval thresholds come from
        # config.yaml loops.salad — the fork's base_config carries the
        # vendor's per-video-frame values (0.85 / NMS 25), which over 216
        # KEYFRAMES proposed zero pairs on pccr (2026-09-13) → no bridge, no
        # loop, duplicates untouched.
        cfg_v.setdefault("Loop", {})["SALAD"] = fork_loop_salad(_mg)
        pipe.send_log(f"[loops] SALAD candidates over keyframes: similarity ≥ "
                      f"{_mg.loops.salad.similarity_threshold:g}, top-{_mg.loops.salad.top_k}, "
                      f"gap ≥ {_mg.loops.salad.min_gap_keyframes} kf, NMS {_mg.loops.salad.nms_threshold} kf")
        # F2: keyframe SE(3) graph replaces the vendor sim3loop; authority
        # budgets per stage; optional ensemble witness (§4.3/§4.7/§4.8)
        cfg_v["Model"]["graph"] = fork_model_graph(_mg)
        cfg_v["Model"]["authority"] = fork_model_authority(_mg)
        cfg_v["Model"]["certify"] = fork_model_certify(_mg)
        cfg_v["Model"]["metric_lock"]["anchor_extract"] = {
            "python": sys.executable, "frames_dir": str(frames_dir),
            "output_dir": str(output_dir),
            "model_id": str((recon_cfg.get("da3", {}) or {}).get(
                "model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"))}
        _abs_path = output_dir / "scale_absolute_rows.json"
        _abs_rows = []
        if _abs_path.exists():
            _abs_rows = list(json.loads(_abs_path.read_text()).get("rows", []))
            pipe.send_log(f"[scale-graph] {len(_abs_rows)} absolute scale row(s) from "
                          f"{_abs_path.name} enter the chunk scale graph")
        cfg_v["Model"]["metric_lock"]["absolute_rows"] = _abs_rows
        if bool(_va_cfg.get("scale_vio", True)):
            from ingestors.vio_detector import detect_vio_data
            _det = detect_vio_data(frames_dir.parent)
            if _det["has_vio"]:
                from reconstruction.vio_scale import video_fps
                _fps = video_fps(frames_dir.parent)
                if not _fps:
                    raise RuntimeError("VIO present but the video fps could not be read — "
                                       "VIO rows need a time base (docs/VIO_FORMAT.md)")
                cfg_v["Model"]["metric_lock"]["vio"] = {
                    "path": str(_det["vio_path"]), "fps": float(_fps),
                    "min_coverage": float(_va_cfg.get("vio_min_coverage", 0.5))}
                pipe.send_log(f"[scale-graph] VIO {_det['vio_path'].name} → per-chunk "
                              f"absolute scale rows (σ {_mg.scale.sigma_vio})")
        _adj = [k for k in ("exact_seam_align", "frame_ownership",
                            "ownership_backfill", "elastic_seam", "intra_chunk",
                            "depth_graph", "blend_copies")
                if cfg_v["Model"].get(k)]
        pipe.send_log(f"CHUNKED-METRIC: chunks {int(_chunk)}/{int(_ov)}, DA3 metric "
                      f"scale (anchors + seam graph; zoom chunks: anchors excluded, "
                      f"scale from seams), chunk health flags diagnostic-only; "
                      f"adjustment stages ON: {_adj if _adj else 'NONE (clean omegalong)'}")

    def _ensure_anchors(_files):
        """Isolated DA3 depth for every anchor file not already extracted."""
        _missing = [f for f in _files
                    if not (_anchor_dir / f"frame_{int(os.path.splitext(f)[0])}.npz").exists()]
        if _missing:
            _run_da3_anchor(pipe, frames_dir, output_dir, sorted(set(_missing)), recon_cfg)

    def _omega_pass(cfg_v, tag, sel_path=None):
        # sel_path: the keyframe list this pass runs on — the walk probe runs on
        # an evenly-strided subset of selected_frames.json, every other pass on
        # the full set
        sel_path = sel_path or selected_frames_path
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
        if sel_path:
            cmd.extend(["--selected_frames", str(sel_path)])
        env = os.environ.copy()
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        # deterministic numerics (USER 2026-09-28: identical keyframes → bit-identical
        # output): cuBLAS workspace, a FIXED thread count for every CPU library the
        # fork uses (the pose graph, lstsq, SVD — a reduction's order must not depend on
        # the machine's 252 cores or the cgroup's 30), MKL in its reproducible mode
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[_k] = "8"
        env["MKL_CBWR"] = "COMPATIBLE"
        env["PYTHONHASHSEED"] = "0"

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
    _probe_sel = None             # set when the first pass is the strided walk probe
    # AS MANY KEYFRAMES PER CHUNK AS THE CARD ALLOWS — USER ORDER 2026-09-23:
    # *"vamos a armar los chunk de la mayor cantidad de frames posibles, si hay
    # mas de uno, con el solape del 50% ... eso lo va a determinar el GPU, lo
    # que el GPU permita"*, on his visual verdict over many runs: *"yo se como
    # queda observatorio con un solo chunk, y es mucho mejor que lo que tenemos
    # ahora, lo mismo el test2"*.
    #
    # WHY IT BEATS A FIXED SIZE, read off this repo's own measurements: the
    # damage lands on the SEAMS. test2's epoch 1 tore at seam 6->7; the elastic
    # stage starts from 8.4 cm of disagreement between the two copies of a
    # shared frame; the in-run pose graph worsens its held-out by 0.6 cm. A
    # chunk that holds the whole scene has none of those. Omega's feed-forward
    # drift is the reason chunking exists, and on walks of ~11-13 m it is
    # smaller than what the seams cost.
    #
    # `chunk_frames: 0` = ask the card: (free VRAM - 4 GB base) / 0.086 GB per
    # frame, the vendor's own measured footprint (500 frames ~ 43 GB, the paper's
    # number). A positive value overrides it, for A/B work.
    _chunk_cfg = int(_simple_cfg.get("chunk_frames", 0) or 0)
    if _simple_on and _n_selected:
        # the card's TOTAL memory decides the layout (a property of the card); FREE
        # memory at this instant depends on vLLM teardown and fragmentation and made
        # the chunk layout — the whole geometry — a function of transient GPU state
        _free = _gpu_total_gb()
        if _chunk_cfg:
            _cap = _chunk_cfg
            _need = 4.0 + 0.086 * _cap
            if _free is not None and _free < _need:
                pipe.send_log(f"WARNING: free VRAM {_free:.1f} GB < {_need:.1f} GB "
                              f"needed for {_cap}-frame chunks — NOT resizing "
                              f"(explicit chunk_frames): free the GPU or lower "
                              f"reconstruction.simple.chunk_frames", level="warning")
            pipe.send_log(f"SIMPLE: chunk capacity {_cap} frames "
                          f"(reconstruction.simple.chunk_frames, explicit)")
        elif _free is None:
            raise RuntimeError("the GPU's total memory cannot be read (nvidia-smi) — the "
                               "chunk capacity cannot be decided")
        else:
            # 0.086 GB/frame was MEASURED at pccr's grid (464x832); a frame's footprint grows with its
            # pixels (tokens) — zaragoza's 1920x1080 frame holds 5.4x more (2026-10-04)
            _ref_px = 464.0 * 832.0
            _gw, _gh = _omega_grid_wh(frames_dir)
            _per_frame = 0.086 * (float(_gw) * float(_gh)) / _ref_px
            _cap = max(24, int((_free - 4.0) / _per_frame))
            pipe.send_log(f"SIMPLE: chunk capacity {_cap} frames — {_free:.1f} GB "
                          f"total on the card, 4.0 GB base + {_per_frame:.3f} GB/frame "
                          f"(0.086 measured at 464x832, scaled to this session's {_gw}x{_gh} grid). "
                          f"{_n_selected} keyframe(s) to place.")
        _chunk_cfg = _cap
        _max_walk0 = float(_simple_cfg.get("max_walk_single_pass_m", 0) or 0)
        _cw0 = float(_simple_cfg.get("chunk_walk_m", 12.0) or 12.0)
        _walk0 = float(_walk_doc["walk_length_m"]) if _walk_doc else None
        # THE WALK DECIDES (USER 2026-09-30: "Omega deriva, está probado que no más de 5 m"):
        # a walk over max_walk_single_pass_m is chunked at chunk_walk_m of REAL walk. The
        # Omega coherence probe that tried to measure it was DELETED the same day — it
        # chose one 289-kf pass on pccr although 128 kf had drifted +301 %.
        # An explicit pin (chunk_frames_over_walk > 0) still wins: it is the A/B knob.
        _pin0 = int(_simple_cfg.get("chunk_frames_over_walk", 0) or 0)
        _chunk_it = _walk0 is not None and _scale_align_on and not (
            _n_selected <= _chunk_cfg and (_max_walk0 <= 0 or _walk0 <= _max_walk0))
        if _chunk_it:
            # I4 (claude_stac.txt §4-F2): the chunks are decided before Omega runs —
            # one pass, no re-run. Precedence: pinned size > metres.
            if _pin0 > 0:
                _fx = max(24, min(_pin0, int(_chunk_cfg)))
                _ov = _fx // 2
                _how = f"{_fx} keyframes pinned (chunk_frames_over_walk)"
            else:
                _fx, _ov = plan_chunks(_n_selected, _walk0, _cw0, max_size=max(_chunk_cfg, 24))
                _how = f"{_cw0:g} m of REAL walk"
            _chunked_already = True
            _anchor_idx = plan_anchor_indices(_n_selected, _fx, _ov, _anch_per_chunk)
            _ensure_anchors([_sel_files[i] for i in _anchor_idx])
            _apply_chunked_metric(vggt_config, _fx, _ov)
            _visit_m = float(((config.get("correction") or {}).get("visit_drift") or {})
                             .get("min_walk_m", 0) or 0)
            _sal = (vggt_config.get("Loop") or {}).get("SALAD")
            if _sal is not None and _visit_m > 0 and _walk0 > 0:
                import math as _math
                _band = max(int(_sal["min_gap"]),
                            int(_math.ceil(_visit_m / (_walk0 / _n_selected))))
                pipe.send_log(f"[loops] SALAD non-local band: {_band} kf = {_visit_m:g} m of "
                              f"the measured walk")
                _sal["min_gap"] = int(_band)
                _sal["min_gap_frac"] = 0.0
            if _sal is not None:
                # SALAD's appearance bar calibrated on the session's GEOMETRIC revisits
                # (the DA3-window walk) — LoopModels.LoopModel.calibrate_threshold
                from intake.walk import revisit_reference, REVISIT_REFERENCE_NAME
                _ref = revisit_reference(output_dir.parent)
                _sal["revisit_reference"] = str(output_dir / REVISIT_REFERENCE_NAME)
                pipe.send_log(f"[loops] SALAD revisit reference: {len(_ref['frames'])} "
                              f"keyframes, revisit = cameras < {_ref['dist_bar_m']:.2f} m "
                              f"(scene median depth) and < "
                              f"{math_deg(_ref['hfov_rad']) / 2.0:.1f}° apart (half the FOV)")
            _persist_chunk_plan(_fx, _ov, _n_selected, "walk-planned", _walk=_walk0)
            pipe.send_log(f"SIMPLE chunked-metric (I4): walk {_walk0:.1f} m measured by the "
                          f"DA3 windows → {len(chunk_ranges(_n_selected, _fx, _ov))} chunks of "
                          f"{_fx} keyframes ({_how}, overlap {_ov}); ONE Omega pass")
        elif _n_selected <= _chunk_cfg:
            vggt_config["Model"]["chunk_size"] = max(_n_selected, 2)
            vggt_config["Model"]["overlap"] = 0
            vggt_config["Model"]["loop_enable"] = False
            # single pass = no chunks: a stale plan from a previous chunked run
            # would lie to the correction module
            (output_dir / "chunk_plan.json").unlink(missing_ok=True)
            # THE ONE ADJUSTMENT STAGE A SINGLE CHUNK CAN STILL RUN. Every seam
            # stage guards on `len(chunk_indices) < 2` and disables itself here;
            # `_stac_intra_chunk` guards on `< 1` — it was WRITTEN to work on one
            # chunk, because it corrects the warp BETWEEN FRAMES OF THE SAME
            # chunk, which is exactly what omega's feed-forward drift is when the
            # chunk holds the whole scene. It was only ever written inside
            # `_apply_chunked_metric`, so the single-pass layout dropped it in
            # silence: measured 2026-09-23 — `intra_chunk` appears in pccr's
            # session YAML (chunked) and NOT in test2's or observatorio's.
            # config.yaml says `intra_chunk: true  # (KEEP ON)` with its A4
            # verdict; the flag is what decides, not the layout.
            vggt_config["Model"]["intra_chunk"] = bool(_va_cfg.get("intra_chunk", False))
            pipe.send_log(f"SIMPLE single-pass: {_n_selected} keyframes ≤ "
                          f"{_chunk_cfg} → ONE chunk, no overlap, no seams. "
                          f"Nothing measured afterwards re-runs it. "
                          f"Adjustment stage ON: "
                          f"{'intra_chunk' if vggt_config['Model']['intra_chunk'] else 'NONE'} "
                          f"(the seam stages need 2+ chunks and stand down by themselves).")
        elif _scale_align_on and int(_simple_cfg.get("chunk_frames_over_walk", 0) or 0) > 0:
            # A PINNED size (chunk_frames_over_walk) needs no walk probe: the probe only
            # sizes the chunks, and on pccr 2026-08-24 it read 1526.6 m over a walk the
            # chunked run measured at 104.8 m (14.6x — one Omega pass over ~105 m drifts
            # past any use), which made 110 chunks of 24 kf (~1.9 m each). Chunked
            # directly at the pinned size; SALAD's band in keyframes is the one
            # definition of a visit (correction.visit_drift.min_walk_m) at the pin's own
            # density — chunk_frames_over_walk keyframes per chunk_walk_m metres.
            _fx = int(_simple_cfg.get("chunk_frames_over_walk"))
            _cw = float(_simple_cfg.get("chunk_walk_m", 12.0) or 12.0)
            if _fx > _chunk_cfg:
                raise RuntimeError(
                    f"reconstruction.simple.chunk_frames_over_walk = {_fx} keyframes per chunk "
                    f"exceeds what the card holds ({_chunk_cfg}) — lower it")
            _ov = _fx // 2
            _chunked_already = True
            _anchor_idx = plan_anchor_indices(_n_selected, _fx, _ov, _anch_per_chunk)
            _ensure_anchors([_sel_files[i] for i in _anchor_idx])
            _apply_chunked_metric(vggt_config, _fx, _ov)
            _visit_m = float(((config.get("correction") or {}).get("visit_drift") or {})
                             .get("min_walk_m", 0) or 0)
            _sal = (vggt_config.get("Loop") or {}).get("SALAD")
            if _sal is not None and _visit_m > 0:
                import math as _math
                _band = max(int(_sal["min_gap"]), int(_math.ceil(_visit_m * _fx / _cw)))
                pipe.send_log(f"[loops] SALAD non-local band: {_band} kf = {_visit_m:g} m of "
                              f"walk at the pinned {_fx} kf / {_cw:g} m")
                _sal["min_gap"] = int(_band)
                _sal["min_gap_frac"] = 0.0
            _persist_chunk_plan(_fx, _ov, _n_selected, "pinned-chunked")
            pipe.send_log(f"SIMPLE chunked-metric (pinned): {_n_selected} keyframes > "
                          f"{_chunk_cfg} (card capacity) → "
                          f"{len(chunk_ranges(_n_selected, _fx, _ov))} chunks of {_fx} "
                          f"(overlap {_ov}) from reconstruction.simple.chunk_frames_over_walk; "
                          f"no walk probe")
        elif _scale_align_on:
            # THE WALK DECIDES EVEN WHEN THE WHOLE SET DOES NOT FIT THE CARD —
            # USER 2026-09-28: *"que no sea por memoria sino los 12m, siempre"*.
            # This branch used to chunk DIRECTLY at the card's capacity: pccr
            # 2026-08-24 (1329 kf, A100 80 GB) became 3 chunks of 870 frames,
            # seams 3.5-5.1 m apart, and the run was OOM-killed. The walk that
            # sizes the chunks needs a metric pass, so the first pass is a PROBE
            # over an evenly-strided subset that fits: the strided walk traces
            # the same trajectory, and its over-measurement when it drifts is the
            # same self-correcting signal as the single pass's (see THE WALK
            # DECIDES below). The probe is never the result — the full set is
            # always re-run chunked at chunk_walk_m.
            _stride = -(-_n_selected // _chunk_cfg)
            _probe_files = list(_sel_files[::_stride])
            _probe_doc = dict(_sel) if isinstance(_sel, dict) else {}
            _probe_doc.update({"method": f"walk_probe_stride_{_stride}",
                               "total_frames": _n_selected,
                               "selected_count": len(_probe_files),
                               "selected_files": _probe_files})
            _probe_path = output_dir / "walk_probe_frames.json"
            _probe_path.write_text(json.dumps(_probe_doc))
            _probe_sel = str(_probe_path)
            vggt_config["Model"]["chunk_size"] = max(len(_probe_files), 2)
            vggt_config["Model"]["overlap"] = 0
            vggt_config["Model"]["loop_enable"] = False
            vggt_config["Model"]["intra_chunk"] = bool(_va_cfg.get("intra_chunk", False))
            (output_dir / "chunk_plan.json").unlink(missing_ok=True)
            pipe.send_log(f"SIMPLE walk probe: {_n_selected} keyframes > {_chunk_cfg} "
                          f"(card capacity) → ONE pass over every {_stride}th keyframe "
                          f"({len(_probe_files)} frames) to MEASURE the walk; the full "
                          f"set is then re-run chunked at "
                          f"{float(_simple_cfg.get('chunk_walk_m', 12.0) or 12.0):g} m "
                          f"of walk per chunk (the probe is not the result)")
        else:
            vggt_config["Model"]["chunk_size"] = _chunk_cfg
            vggt_config["Model"]["overlap"] = _chunk_cfg // 2
            pipe.send_log(f"SIMPLE: {_n_selected} keyframes > {_chunk_cfg} and "
                          f"scale_align is OFF → windowed mode without the metric "
                          f"lock ({_chunk_cfg}/{_chunk_cfg // 2})", level="warning")
        _apply_conf_filter(vggt_config)
    _tag1 = ("walk-probe" if _probe_sel else
             "chunked-metric" if _chunked_already else "single-pass")
    if not _omega_pass(vggt_config, _tag1, sel_path=_probe_sel):
        return

    # ── metric scale + orientation (runs after EVERY pass) ──
    # Opt-out (scale_align: false) leaves poses UP-TO-SCALE — used to isolate whether a
    # bad result comes from the scale alignment vs the raw Omega backbone.
    vggt_save_dir = output_dir / "maplong_run"
    if not _scale_align_on:
        pipe.send_log("[scale-align] DISABLED (vggtomega.scale_align: false) — poses stay up-to-scale",
                      level="warning")
        return

    def _metricize_and_orient(cfg_v, tag, sel_path=None):
        """Emit omega depth → scale_align (global; in chunked-metric mode the chunks are
        already locked, so this is the residual/VERIFICATION pass — its spread is the
        health metric) → bake upright orientation. Returns the walk length in meters."""
        pipe.send_progress(86, f"VGGT-Omega ({tag}): aligning metric scale to DA3...",
                           stage="reconstruction")
        _emit_omega_depth(vggt_save_dir, output_dir,
                          int(cfg_v["Model"]["chunk_size"]),
                          int(cfg_v["Model"]["overlap"]),
                          sel_path or selected_frames_path, pipe)
        from reconstruction.scale_align import run as _scale_run
        _s = _scale_run(output_dir, dry_run=False,
                        log=lambda m: pipe.send_log(f"[scale-align] {m}"),
                        near_frac=float(_va_cfg.get("scale_near_frac", 0.25)),
                        conf_top_frac=float(_va_cfg.get("scale_conf_top_frac", 0.10)),
                        cfg={"mode": str(_va_cfg.get("scale_mode", "global_median")),
                             "vio": bool(_va_cfg.get("scale_vio", True)),
                             "vio_segment_s": float(_va_cfg.get("vio_segment_s", 5.0)),
                             "vio_min_segments": int(_va_cfg.get("vio_min_segments", 8)),
                             "vio_min_coverage": float(_va_cfg.get("vio_min_coverage", 0.5))},
                        session_dir=frames_dir.parent)
        # METRIC IS MANDATORY: a non-metric cloud is useless (BIM comparison needs real
        # units). If scale_align could not estimate s, FAIL — never ship up-to-scale.
        if _s is None:
            raise RuntimeError(
                "metric scale alignment FAILED — scale_align could not estimate s. Refusing to "
                "produce a NON-METRIC reconstruction (it can't be compared against BIM). See the "
                "[scale-align] lines above for the exact reason (frame match / ratios / inputs).")
        # claude_stac.txt §5.4: in chunked-metric mode the chunks are already
        # locked, so scale_align is the VERIFIER — s ≈ 1 expected. A deviation
        # beyond scale.verify_max_dev is a session failure, not a warning.
        if tag == "chunked-metric":
            from reconstruction.loops.config import load_loops_config
            _vmax = load_loops_config(config).scale.verify_max_dev
            _dev = abs(float(_s) - 1.0)
            pipe.send_log(f"[scale-verify] chunked-metric residual s={float(_s):.4f} "
                          f"(|s-1|={_dev:.4f}, limit {_vmax:g})")
            if _dev > _vmax:
                raise RuntimeError(
                    f"scale VERIFICATION FAILED: the locked chunks disagree with the DA3/VIO "
                    f"verifier by {_dev*100:.1f}% (> {_vmax*100:.0f}% = scale.verify_max_dev). "
                    f"The scale graph (scale_graph.json / metric_lock.json) does not close — "
                    f"session failure, not a warning.")

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
        # streaming clean (user 2026-09-03, fixed 09-04): the aligned chunk
        # npys' LAST consumer is the omega-depth/scale step that just
        # succeeded — delete them NOW (tens of GB) instead of keeping them
        # for the optional TSDF depth fallback. Runs only on scale SUCCESS
        # (the fail path raises above); a phase-2 chunked re-run regenerates
        # the dir from scratch.
        _aligned = Path(vggt_save_dir) / "_tmp_results_aligned"
        from reconstruction.loops.config import load_loops_config as _llc
        if _llc(config).certify.keep_aligned_chunks:
            pipe.send_log("[cleanup] _tmp_results_aligned KEPT (certify.keep_aligned_chunks: "
                          "the post-hoc graph / A/B harness / certification read it)")
        elif _aligned.exists():
            _mb = sum(f.stat().st_size for f in _aligned.rglob("*")
                      if f.is_file()) / 1048576
            shutil.rmtree(_aligned, ignore_errors=True)
            pipe.send_log(f"[cleanup] _tmp_results_aligned deleted after scale "
                          f"({_mb:.0f} MB freed)")
        return _walk

    _walk_m = _metricize_and_orient(vggt_config, _tag1, sel_path=_probe_sel)

    # The walk lands in the plan as EVIDENCE (the plan is written before the pass,
    # so the size never waited for it). Reading it back against the real walk is
    # how the 2.3x over-measurement became visible at all.
    _plan_path = output_dir / "chunk_plan.json"
    if _plan_path.exists():
        try:
            _plan = json.loads(_plan_path.read_text())
            _plan["walk_m"] = round(float(_walk_m), 2)
            _plan_path.write_text(json.dumps(_plan, indent=1))
        except Exception as _e:  # noqa: BLE001
            pipe.send_log(f"[chunk-plan] could not stamp the measured walk ({_e})",
                          level="warning")

    # ── THE WALK DECIDES, AND THE SINGLE PASS IS ITS PROBE (USER 2026-09-23) ──
    # *"da3 sobre todos los kf, chunk unico, medida de recorrido, menos de 15m
    # un chunk, mas de 15m, 60/30"*.
    #
    # THE OVER-MEASUREMENT IS THE SIGNAL, NOT A BUG. I removed this on
    # 2026-09-22 having read it backwards: a single pass that measured 44 m over
    # a ~19 m walk looked like a broken instrument. It is not — a pass whose
    # frames still agree measures the real length; one that DRIFTED measures
    # long, because the drift stretches the trajectory it is summing. Either way
    # the answer is the same: chunk it. Proven on the very run that led here —
    # pccr in one chunk measured 43.7 m and dropped 60.1 % of the cloud as
    # single_witness (frames disagreeing about where surfaces are), against
    # 18.8 m and 11.6 % on the same scene in 60/30.
    #
    # WHY IT IS NOT ARBITRARY: below the limit raw Omega beats Omega + the seam
    # machinery, above it the machinery wins. Almost every adjustment stage
    # lives on the SEAMS (exact_seam_align, elastic_seam, frame_ownership,
    # blend_copies, ownership_backfill, and scale_drift, whose judge IS the seam
    # ratios), so a single chunk runs none of them: what is delivered is raw
    # Omega plus one global scale. On short walks that is better — the user's
    # verdict on observatorio (11.2 m) and test2 (12.9 m). On long ones the
    # drift the machinery exists to fight is what dominates.
    _walk_probe = float(_walk_m)
    _max_walk = float(_simple_cfg.get("max_walk_single_pass_m", 0) or 0)
    # The re-run is sized in WALKED METRES, not in frames: 60 frames is 5.2 m on
    # one scene and 3.1 m on another, and a 3 m chunk gives Omega no baseline
    # (USER 2026-09-23: "implementemos el chunk walk 12, por algo estaban no?").
    # A positive `chunk_frames_over_walk` overrides it with a fixed size.
    _fixed2 = int(_simple_cfg.get("chunk_frames_over_walk", 0) or 0)
    _chunk_walk = float(_simple_cfg.get("chunk_walk_m", 12.0) or 12.0)
    _phase2, _ov2 = 0, 0          # 0 = the single pass stands, nothing re-runs
    # The chunk is sized by the WALK alone; the card's capacity is only the
    # physical ceiling a 12 m chunk cannot exceed (it binds only when 12 m of
    # this walk holds more keyframes than the card can take — logged if so).
    _max_chunk = int(_chunk_cfg or _n_selected)
    # with the walk MEASURED before Omega (F2 I4) the pass's own walk is evidence only —
    # it never re-runs anything
    if (_simple_on and not _chunked_already and _scale_align_on and _walk_doc is None
            and (_probe_sel or (_max_walk > 0 and _walk_m > _max_walk))):
        if _fixed2:
            _phase2, _ov2 = _fixed2, _fixed2 // 2
        else:
            _phase2, _ov2 = plan_chunks(_n_selected, _walk_m, _chunk_walk,
                                        max_size=max(_max_chunk, 24))
            _want = int(round(_chunk_walk * _n_selected / max(_walk_m, 1e-6)))
            if _want > _phase2 and _phase2 < _n_selected:
                pipe.send_log(f"[chunk-plan] {_chunk_walk:g} m of this walk is {_want} "
                              f"keyframes; the card holds {_phase2} per chunk — chunks "
                              f"of {_phase2} ({_phase2 * _walk_m / _n_selected:.1f} m)",
                              level="warning")
        if _phase2 >= _n_selected and not _probe_sel:
            pipe.send_log(f"[chunk-plan] walk {_walk_m:.1f} m > {_max_walk:g} m but "
                          f"{_chunk_walk:g} m per chunk needs {_phase2} keyframes of "
                          f"{_n_selected} — one chunk already covers it, keeping the "
                          f"single pass")
            _phase2 = 0
    if _phase2:
        pipe.send_log(f"[chunk-plan] walk {_walk_m:.1f} m "
                      + ("(strided probe) → the full set" if _probe_sel else
                         f"> {_max_walk:g} m → the single pass either covers a long "
                         f"walk or drifted; either way it")
                      + f" is re-run CHUNKED at {_phase2}/{_ov2} "
                      f"({len(chunk_ranges(_n_selected, _phase2, _ov2))} chunks, "
                      f"{_chunk_walk:g} m of walk each)")
        pipe.send_progress(40, f"Walk {_walk_m:.1f} m — re-running chunked "
                               f"({_phase2}/{_ov2})...", stage="reconstruction")
        _anchor_idx = plan_anchor_indices(_n_selected, _phase2, _ov2, _anch_per_chunk)
        _ensure_anchors([_sel_files[i] for i in _anchor_idx])   # already on disk: a no-op
        # wipe what phase 1 produced — NOT da3_run, the anchors live there and
        # every keyframe already has one (scale_anchor_frames: 0)
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
        vggt_config = _build_vggtomega_config(config, frames_dir)
        _apply_chunked_metric(vggt_config, _phase2, _ov2)
        # SALAD's non-local band in METRES OF WALK, not in a share of the keyframe
        # count: the walk is known now. `min_gap_frac` x n was 22 kf on pccr's 216
        # and 133 kf on 1329 (pccr 2026-08-24: 0 candidates, nothing closed). The
        # bar is the system's ONE definition of two visits,
        # correction.visit_drift.min_walk_m (USER 2026-09-25), translated to
        # keyframes through this walk's measured m/kf; the configured floor stays.
        _visit_m = float(((config.get("correction") or {}).get("visit_drift") or {})
                         .get("min_walk_m", 0) or 0)
        _sal = (vggt_config.get("Loop") or {}).get("SALAD")
        if _sal is not None and _visit_m > 0 and _walk_m > 0:
            import math as _math
            _band = max(int(_sal["min_gap"]),
                        int(_math.ceil(_visit_m / (_walk_m / _n_selected))))
            pipe.send_log(f"[loops] SALAD non-local band from the walk: {_band} kf = "
                          f"{_visit_m:g} m of walk at {_walk_m / _n_selected * 100:.1f} cm/kf "
                          f"(was {_sal['min_gap_frac']:g} x {_n_selected} kf)")
            _sal["min_gap"] = int(_band)
            _sal["min_gap_frac"] = 0.0
        _persist_chunk_plan(_phase2, _ov2, _n_selected, "chunked-metric", _walk=_walk_m)
        _apply_conf_filter(vggt_config)
        _chunked_already = True
        if not _omega_pass(vggt_config, "chunked-metric"):
            return
        _walk_m = _metricize_and_orient(vggt_config, "chunked-metric")
        pipe.send_log(f"[chunk-plan] chunked re-run measured walk: {_walk_m:.1f} m "
                      f"(the single pass read {_walk_probe:.1f} m — the gap between "
                      f"the two IS the drift the chunking removed)")

    # Success → free da3_run when the TSDF won't use it (depth_source not DA3-based).
    _ds = str((config.get("tsdf", {}) or {}).get("depth_source", "auto")).lower()
    if _ds not in ("da3", "da3_frames", "auto"):
        _da3_run = output_dir / "da3_run"
        if _da3_run.exists():
            _mb = sum(f.stat().st_size for f in _da3_run.rglob("*") if f.is_file()) / (1024 * 1024)
            shutil.rmtree(_da3_run, ignore_errors=True)
            pipe.send_log(f"[scale-align] freed da3_run/ ({_mb:.0f} MB) — TSDF uses omega "
                          f"depth (depth_source={_ds}), DA3 no longer needed")


# What a NEW chunk plan leaves standing (USER 2026-10-05: "si lo rechaza, debe eliminar todo lo
# derivado de la inferencia, y de ahí hacia adelante también"): only what was computed BEFORE
# Omega and does not depend on how the keyframes are chunked.
_PLAN_INDEPENDENT_OUTPUT = {"chunk_plan.json", "da3_run", "da3_windows", "intake",
                            "salad_revisit_reference.json", "vggt_omega_config.yaml", "maplong_run"}
_PLAN_INDEPENDENT_MAPLONG = {"loop_closures.txt", "salad_calibration.json", "sky_masks",
                             "frame_list.json", "vggt_omega_config.yaml"}


def _invalidate_on_new_chunk_plan(output_dir: Path, plan: dict, log=print) -> bool:
    """A chunked run whose plan differs from the one the outputs on disk were made with
    (another walk measured, another chunk size) wipes EVERY product of the old plan before
    Omega runs: the chunk predictions, the bridges, the aligned copies, the per-chunk PLYs
    and stamps, and everything downstream (omega_run, the precision core, the cloud, the
    segmentation projected on it, the epochs). Old and new never live side by side — a
    resume would otherwise load chunk 11 of the old plan next to chunk 10 of the new one
    (USER 2026-10-05, pccr 2408: 296/148 on disk, 293/146 planned). What stays: the
    frames and the intake, DA3 per keyframe and its windows, the SALAD candidates and
    calibration, the sky masks — all computed before Omega, independent of the chunking.
    Returns True when something was wiped."""
    out = Path(output_dir)
    ml = out / "maplong_run"
    old_path = out / "chunk_plan.json"
    old = None
    if old_path.exists():
        try:
            old = json.loads(old_path.read_text())
        except (OSError, ValueError):
            old = {"unreadable": True}
    has_chunks = any((ml / "_tmp_results_unaligned").glob("chunk_*.npy")) if ml.is_dir() else False
    same = (old is not None
            and old.get("chunk_ranges") == plan["chunk_ranges"]
            and int(old.get("n_keyframes", -1)) == int(plan["n_keyframes"]))
    if same:
        return False
    if old is None and not has_chunks:
        return False                  # a first run: nothing of any plan on disk yet
    why = ("chunk files on disk with no chunk plan recorded for them" if old is None else
           f"the outputs on disk are of chunk plan {old.get('chunk_size')}/{old.get('overlap')} "
           f"over {old.get('n_keyframes')} keyframes, this run plans "
           f"{plan['chunk_size']}/{plan['overlap']} over {plan['n_keyframes']}")
    freed = 0
    doomed = [p for p in out.iterdir() if p.name not in _PLAN_INDEPENDENT_OUTPUT]
    if ml.is_dir():
        doomed += [p for p in ml.iterdir() if p.name not in _PLAN_INDEPENDENT_MAPLONG]
    for p in doomed:
        try:
            if p.is_dir() and not p.is_symlink():
                freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
            else:
                freed += p.stat().st_size if p.exists() else 0
                p.unlink(missing_ok=True)
        except OSError as e:
            raise RuntimeError(f"a new chunk plan must start clean and {p} could not be deleted ({e}) "
                               f"— old and new chunk products would live side by side") from e
    log(f"[chunk-plan] NEW PLAN — {why}: every product of the old plan and everything downstream "
        f"deleted ({len(doomed)} item(s), {freed / 1e9:.1f} GB); kept: frames, intake, DA3 per "
        f"keyframe + windows, SALAD candidates, sky masks")
    return True


def _cleanup_recon_temps(save_dir: Path, output_dir: Path, backend: str, pipe: WorkerPipe):
    """Delete reconstruction temporaries that are dead once chunks + origins exist:
    maplong_run/{_tmp_results_unaligned,_tmp_results_loop,pcd} and (mapanything only)
    da3_run/da3_full except results_output (kept for the texture bake). KEEPS
    _tmp_results_aligned — the TSDF reads per-frame depth from it.

    Idempotent (skips what's already gone), so it is safe — and now called — on BOTH the
    normal path AND the resume early-exit. Previously the resume path returned before the
    cleanup, so temps (tens of GB) accumulated across resumed runs and never got freed."""
    _keep_loop = False
    try:
        from reconstruction.loops.config import load_loops_config as _llc
        from config import cfg as _cfg_all
        _keep_loop = bool(_llc(_cfg_all).certify.keep_aligned_chunks)
    except Exception as _e:  # noqa: BLE001 — a cleanup must never abort a finished reconstruction
        pipe.send_log(f"[cleanup] certify config unavailable ({_e}) — loop bridges deleted",
                      level="warning")
    # uncert/ (per-frame uncertainty maps) is read by nothing after the run: uncertainty.json carries
    # the per-frame numbers the certification reads, and the fork's own resume reads that json too
    for tmp_dir_name in ["_tmp_results_unaligned", "pcd"] + ([] if _keep_loop else ["_tmp_results_loop", "uncert"]):
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

    return _apply_stac_model_keys(cfg, config)



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
                    # streaming clean (user 2026-09-03: minimize expansion):
                    # this chunk is now fully mirrored in output/ — its pcd/
                    # source ply + inline npz are dead; end-of-stage cleanup
                    # used to keep the 2x copy alive for the whole post-process
                    for _dead in (Path(pcd_files[i]), inline):
                        try:
                            _dead.unlink()
                        except OSError:
                            pass
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
