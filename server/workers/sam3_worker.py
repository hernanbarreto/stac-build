# STAC-Builder: SAM3 Worker (Subprocess)
# Runs SAM3 segmentation in its own process.
# Reads VLM analysis + frames, writes segmentation.json + masks.
#
# DETERMINISM (docs/plan_determinismo.md points 90, 91, 92, 154, 164, 165 — 2026-10-08):
#   * the cuBLAS workspace is pinned before the first CUDA call of this process, and the
#     model runs under repro's deterministic torch (TF32 off, strict algorithms) —
#     sam3_wrapper.load_model;
#   * the card is EXCLUSIVE before SAM3 starts: the semantic service is stopped with the
#     VERIFIED stop and repro.require_exclusive_gpu refuses a shared card — never a run on
#     whatever memory was left (an OOM used to be retried once, then the prompt skipped);
#   * NO partial result: a prompt SAM3 did not run to the end (failed, not reached, an OOM
#     recovery) fails the stage naming it; the mask→cloud projection is part of the
#     deliverable and its failure fails the stage; the per-object mesh crop runs only from
#     a scene mesh STAMPED for this reconstruction and its failure fails the stage too;
#   * the model built (version, checkpoint sha256, device, numerics, the card and the
#     environment) is recorded in the census.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import json
import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe

SCENE_MESH_STAMP = "scene.glb.stamp.json"      # beside tsdf/scene/scene.glb: its reconstruction


def _sam3_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """SAM3 segmentation — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    # the deterministic environment of this process BEFORE torch touches the card (point
    # 91): the cuBLAS workspace can only be set while CUDA is still untouched
    import repro
    repro.ensure_cublas_workspace()

    # The census (below) splits masklets into visits by the mask filter's own key:
    # read it BEFORE SAM3 runs, so a config without it fails in seconds, not after.
    from segmentation.census import visit_gap_kf
    gap_kf = visit_gap_kf(config)
    # Same for the SAM3 detection / confirmation thresholds of the model that will
    # be built (models.segmentation.sam3_thresholds): the wrapper loads the model
    # lazily inside the first prompt, and a bad key must fail the stage here,
    # naming it — not end as every prompt at zero masklets.
    from segmentation.sam3_wrapper import check_sam3_thresholds
    check_sam3_thresholds()

    # Read VLM analysis if available (written by vlm_worker)
    vlm_path = output_dir / "vlm_analysis.json"
    boxes_map = None
    fallback_prompts = None
    vlm_data = None
    if vlm_path.exists():
        vlm_data = json.loads(vlm_path.read_text())
        prompt = vlm_data.get("prompt", "")
        frame_map = vlm_data.get("frame_map", {})
        boxes_map = vlm_data.get("boxes") or None  # Phase 1 per-instance box seeds
        fallback_prompts = vlm_data.get("fallback_prompts") or None   # USER 2026-09-30
        # SIMPLE pipeline: the VLM contributes ONLY the concept vocabulary. SAM3 alone
        # searches, identifies and tracks each concept over the WHOLE sampled frame set
        # (empty frame_map → all frames; with ~1 fps sampling that is a single SAM3
        # session per concept → SAM3's own tracking IS the instance identity). Box
        # seeds are NOT passed: injected boxes pinned duplicate instances of the same
        # physical object and (vendor behavior) each add_prompt resets the session
        # anyway, so they cost identity for near-zero benefit.
        _simple = ((config.get("reconstruction", {}) or {}).get("simple", {}) or {})
        if bool(_simple.get("enabled", False)):
            frame_map = {}
            boxes_map = None
            pipe.send_log(f"SIMPLE segmentation: text concepts only, all frames per "
                          f"concept — '{prompt}'")
        else:
            pipe.send_log(f"Using VLM prompt: '{prompt}'"
                          + (f" (+box seeds for {len(boxes_map)} labels)" if boxes_map else ""))
    else:
        # No scene understanding on disk. An explicit config prompt is
        # honored (operator override); otherwise FAIL — a canned category
        # list is not scene analysis, and writing it to segmentation.json
        # would permanently satisfy the resume probes with garbage.
        prompt = config.get("segmentation_prompt") or ""
        frame_map = {}
        if prompt:
            pipe.send_log(f"No VLM analysis found — using explicit config prompt: '{prompt}'")
        else:
            raise RuntimeError(
                "No vlm_analysis.json (scene understanding) for this session — "
                "the VLM stage must run first. Re-run the pipeline; it resumes "
                "from the missing stage automatically.")

    if pipe.check_cancel():
        return

    # EXCLUSIVE GPU (points 92 / 154): the VLM analysis is already on disk — vLLM is not
    # needed during SAM3, and long single-session tracking (batch_size ≥ frame count)
    # wants the VRAM. The stop is VERIFIED (a vLLM left alive fails the stage) and the
    # card must then be free of every foreign process — nothing is lowered to fit.
    # `reconstruction.simple.exclusive_gpu: false` skips both (tests only — a shared
    # card is then declared, never silently run on).
    _simple_gpu = ((config.get("reconstruction", {}) or {}).get("simple", {}) or {})
    gpu_record = None
    if bool(_simple_gpu.get("enabled", False)) and bool(_simple_gpu.get("exclusive_gpu", True)):
        from workers.base import stop_semantic_service_verified
        stop_semantic_service_verified(pipe, stage="SAM3")
        gpu_record = repro.require_exclusive_gpu(log=pipe.send_log)
    else:
        pipe.send_log("[gpu] DECLARED: reconstruction.simple.exclusive_gpu is off — the card "
                      "was not checked before SAM3", level="warning")

    pipe.send_progress(0, "Loading SAM3 model...", stage="sam3")

    from segmentation_pipeline import run_segmentation

    pipe.send_progress(10, "Running segmentation...", stage="sam3")

    def _seg_progress(pct, msg):
        # Map internal 0-100% to pipeline range 10-80%
        mapped_pct = 10 + (pct / 100) * 70
        pipe.send_progress(mapped_pct, msg, stage="sam3")

    # THE CENSUS (USER 2026-09-29: "debe segmentar todo, absolutamente preciso y
    # completo"): what the VLM looked at, every concept it proposed and its fate,
    # and per prompt the SAM3 masklets with their keyframe spans and visits, and
    # whether SAM3 RAN the prompt at all (prompt_status) — output/
    # segmentation_census.json, from what is already on disk. Written on every
    # run, a failed segmentation included — one that RAISED too, so the file on
    # disk never describes another run's prompts. Since 2026-10-08 it carries the
    # model built (version, checkpoint sha256, device, numerics) and the environment
    # of this process (card, versions) under ``sam3.thresholds`` (points 91 / 165).
    prompt_status: dict = {}

    def _sam3_record():
        from segmentation.sam3_wrapper import get_sam3_wrapper
        w = get_sam3_wrapper()
        rec = dict(getattr(w, "applied_thresholds", None) or {})
        rec["model"] = getattr(w, "model_record", None)
        rec["exclusive_gpu"] = gpu_record
        try:
            import torch
            if torch.cuda.is_initialized():
                rec["environment"] = repro.environment_record(gpu=True)
        except Exception as e:  # noqa: BLE001 — recorded as unreadable, never invented
            rec["environment"] = {"error": f"{type(e).__name__}: {e}"}
        return rec

    def _census(seg_error):
        try:
            from segmentation.census import build_census
            build_census(output_dir, prompt=prompt, vlm_doc=vlm_data, gap_kf=gap_kf,
                         frames_dir=frames_dir, seg_error=seg_error,
                         prompt_status=prompt_status,
                         sam3_thresholds=_sam3_record(),
                         log=pipe.send_log)
        except Exception as e:  # noqa: BLE001 — a report bug never costs the masks
            pipe.send_log(f"segmentation census FAILED ({type(e).__name__}: {e}) — the "
                          f"masks are kept, output/segmentation_census.json is not written",
                          level="warning")

    try:
        result = run_segmentation(
            frames_dir=str(frames_dir),
            output_dir=str(output_dir),
            prompt=prompt,
            frame_map=frame_map,
            boxes_map=boxes_map,
            on_progress=_seg_progress,
            prompt_status=prompt_status,
            fallback_prompts=fallback_prompts,
        )
    except Exception as e:
        _census(f"{type(e).__name__}: {e}")
        raise

    if pipe.check_cancel():
        return

    _census(result.get("error"))

    if "error" in result:
        raise RuntimeError(f"Segmentation failed: {result['error']}")

    # NEVER A PARTIAL RESULT (point 92): every prompt SAM3 received must have RUN to the
    # end on the first attempt. A prompt that failed, was skipped or never reached, or
    # one that ran only after an out-of-memory recovery (the work done then depends on
    # the card's free memory), fails the stage naming it — the census above already
    # describes what happened.
    incomplete = []
    for p, st in sorted((prompt_status or {}).items()):
        st = st or {}
        if st.get("status") != "ran":
            incomplete.append(f"'{p}': {st.get('status')} ({st.get('reason')})")
        elif st.get("note"):
            incomplete.append(f"'{p}': ran only after a recovery ({st.get('note')})")
    if incomplete:
        raise RuntimeError(f"SAM3 did not run {len(incomplete)} prompt(s) to the end on the "
                           f"first attempt — no partial segmentation is delivered (docs/"
                           f"plan_determinismo.md point 92): " + "; ".join(incomplete))

    n_instances = len(result.get("instances", []))
    pipe.send_log(f"Segmentation complete: {n_instances} instances")

    # Segmented at the INTAKE (USER 2026-09-28: VLM + SAM3 run once, before any
    # geometry): there is no cloud yet. The 2-D masks are the deliverable of this
    # run; the cloud stage projects them onto the cloud the precision core publishes.
    if not (output_dir / "cleaned_cloud.ply").exists():
        pipe.send_log("No cloud on disk yet — the 2-D masks are kept; the mask→cloud "
                      "projection runs once a cloud exists")
        pipe.send_progress(100, f"Segmentation complete: {n_instances} objects (2-D)",
                           stage="sam3")
        return

    # Apply to cleaned cloud
    pipe.send_progress(80, "Applying to point cloud...", stage="sam3")

    # The mask→cloud projection belongs HERE: the scene cloud is on disk and the masks
    # are projected onto it the moment they exist. It is part of this stage's
    # deliverable (segmentation_result.json, classification, the store): its failure
    # FAILS the stage with the reason (point 164) — it used to be "non-fatal", and the
    # stage went green with whatever an earlier run had left on disk.
    # On the reconstruction's PRODUCT epoch — the cloud the certification starts from (point 150).
    from correction.chain import select_product_epoch
    select_product_epoch(output_dir, log=pipe.send_log)
    from segmentation_pipeline import map_segmentation_to_cloud
    seg_data = map_segmentation_to_cloud(output_dir)
    if seg_data.get("error"):
        raise RuntimeError(f"mask→cloud projection failed: {seg_data['error']}")
    n_applied = len(seg_data.get("instances", []))
    cov = seg_data.get("coverage")
    pipe.send_log(f"Mapped {n_applied} instances onto the cloud"
                  + (f" ({cov * 100:.1f}% coverage)" if cov is not None else ""))

    # Save seg_data for broadcast
    from atomic_io import atomic_write_json
    atomic_write_json(output_dir / "seg_broadcast.json", seg_data)

    # ── Per-object textured TSDF mesh ──
    # Carve each instance's textured surface mesh out of the scene TSDF
    # (output/tsdf/scene/scene.glb) — ONLY from a scene mesh STAMPED for this
    # reconstruction (point 164: a leftover scene.glb of another geometry is never
    # cropped; the TSDF stage is out of the automatic chain since 2026-08-28, so today
    # no such stamp exists and the step is declared skipped). When it runs, its failure
    # fails the stage.
    if not pipe.check_cancel():
        scene_dir = output_dir / "tsdf" / "scene"
        glb = scene_dir / "scene.glb"
        stamp_p = scene_dir / SCENE_MESH_STAMP
        if not glb.exists() and not (scene_dir / "scene.glb.orig").exists():
            pipe.send_log("No scene TSDF mesh — the per-object mesh crop does not run "
                          "(the TSDF stage is out of the automatic chain)")
        elif not stamp_p.exists():
            pipe.send_log(f"DECLARED: {glb.name} carries no {SCENE_MESH_STAMP} naming its "
                          f"reconstruction — a mesh of another geometry is never cropped "
                          f"(point 164); the per-object mesh crop is skipped", level="warning")
        else:
            from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
            rid = reconstruction_id(output_dir)
            try:
                got = json.loads(stamp_p.read_text()).get(RECONSTRUCTION_ID_KEY)
            except (OSError, ValueError) as e:
                raise RuntimeError(f"{stamp_p} is unreadable ({e}) — the scene mesh cannot be "
                                   f"attributed to a reconstruction") from e
            if got != rid:
                raise RuntimeError(f"{glb} was built for another reconstruction "
                                   f"({str(got)[:12]}…, this one is {rid[:12]}…) — never cropped")
            pipe.send_progress(85, "Carving per-object TSDF meshes...", stage="sam3")
            from segmentation.tsdf_export import crop_scene_mesh_to_instances
            result_path = output_dir / "segmentation_result.json"
            with open(result_path) as f:
                segments_result = json.load(f)
            n_tot = len(segments_result.get("instances", []))

            def _crop_progress(inst_id, phase, elapsed, mesh_path):
                if phase == "done":
                    _crop_progress.done += 1
                    pct = 85 + (_crop_progress.done / max(n_tot, 1)) * 13
                    pipe.send_progress(min(pct, 98),
                                       f"TSDF mesh {_crop_progress.done}/{n_tot}",
                                       stage="sam3")
            _crop_progress.done = 0

            written = crop_scene_mesh_to_instances(
                output_dir=output_dir,
                segments_result=segments_result,
                progress_cb=_crop_progress,
            )
            pipe.send_log(f"Per-object TSDF: wrote {len(written)} textured mesh(es)")

    pipe.send_progress(100, f"Segmentation complete: {n_instances} objects", stage="sam3")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_sam3_work, conn, session_dir, config)
