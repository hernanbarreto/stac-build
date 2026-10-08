# STAC-Builder: VLM Worker (Subprocess)
# Scene understanding stage of the automatic pipeline: Qwen3-VL comprehends
# the scene (no domain assumption) and builds the SAM3 prompts.
#
# DETERMINISM (docs/plan_determinismo.md points 80, 81, 155, 156 — 2026-10-08): the stage
# runs on ITS OWN vLLM — launched from the job's FROZEN configuration with the deterministic
# flags of semantic.serve, after the verified stop of whatever engine was up and with no
# other process on the card, its identity verified before the first call and written into
# vlm_analysis.json, stopped (verified) when the stage ends (semantic.service.job_engine).
# There is NO alternative answer: the InternVL3 fallback is gone — a service that does not
# come up within its bound, a call that fails, a configuration error: the stage FAILS,
# naming the reason, and the pipeline resumes here on the next run.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import contextlib
import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _ensure_semantic_service(pipe: WorkerPipe, config: dict):
    """THIS JOB'S engine (semantic.service.job_engine): a context manager that yields the
    verified identity of the vLLM it launched from the frozen run configuration and stops
    it on exit. The frozen file (output/run_config.yaml, written by the pipeline manager at
    job start) is what the launcher reads its semantic block from; its block must be the one
    this worker was given — two configurations never run one stage. The session directory
    travels in ``config["_session_dir"]`` (a private key, like the manager's
    ``_pipeline_replace``); without it (a CLI, a test) no frozen file is read."""
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from intake.run_config import load_run_config, run_config_path
    from semantic.service import job_engine

    run_cfg_path = None
    session_path = Path(config["_session_dir"]) if config.get("_session_dir") else None
    if session_path is not None:
        frozen, sha = load_run_config(session_path)          # raises: never an unfrozen job
        if (frozen.get("semantic") or {}) != (config.get("semantic") or {}):
            raise RuntimeError(
                "the 'semantic' block of the frozen run configuration (output/run_config.yaml, "
                f"sha256 {sha[:12]}) differs from the one this worker received — a stage never "
                "runs on two configurations (point 155)")
        run_cfg_path = run_config_path(session_path)
    backend = (config.get("autoprompt", {}) or {}).get("backend", "qwen_local")
    pipe.send_progress(0, "Starting this job's semantic service (Qwen3-VL)...", stage="vlm")
    return job_engine(config, backend=backend, owner="vlm_worker", stage="VLM stage",
                      session_dir=session_path, run_config_path=run_cfg_path,
                      log=lambda m: pipe.send_log(m), cancelled=pipe.check_cancel)


def _vlm_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """VLM scene analysis — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    autoprompt_cfg = config.get("autoprompt", {})

    def _on_progress(pct, msg):
        pipe.send_progress(pct, msg, stage="vlm")
        pipe.send_log(msg)

    if pipe.check_cancel():
        return

    # The auto-prompter's strict keys (which frames / crops the VLM is shown, the
    # BOUND on SAM3 prompts) are read HERE, before anything is launched: a missing or
    # renamed key fails the stage naming it, before the minutes of a model load.
    from segmentation.autoprompt.vlm_sampling import (load_max_sam3_prompts,
                                                      load_vlm_sampling)
    if not autoprompt_cfg.get("enabled", True):
        raise RuntimeError(
            "autoprompt.enabled is false and the InternVL3 fallback is gone (docs/"
            "plan_determinismo.md point 156): the VLM stage has no other way to understand "
            "the scene — set autoprompt.enabled: true")
    load_vlm_sampling(config)
    load_max_sam3_prompts(config)

    # Qwen3-VL grounded auto-prompter over THIS job's engine. It writes
    # output/vlm_analysis.json itself and returns the (prompt, frame_map) contract the
    # SAM3 worker consumes. Any failure fails the stage: there is no fallback.
    engine = _ensure_semantic_service(pipe, dict(config, _session_dir=str(session_path)))
    if not hasattr(engine, "__enter__"):
        # a plain truthy value (a test's stub): no engine of this job's own to manage
        engine = contextlib.nullcontext(engine)
    with engine as identity:
        if not identity:
            raise RuntimeError("semantic service unavailable")
        service = identity if isinstance(identity, dict) else None
        pipe.send_progress(0, "Auto-prompting with Qwen3-VL...", stage="vlm")
        pipe.send_log("Starting Qwen3-VL auto-prompter (Phase 1)")
        from segmentation.autoprompt.session_builder import AutoPrompter
        ap = AutoPrompter(session_path, session_path / "output",
                          backend=autoprompt_cfg.get("backend", "qwen_local"),
                          config=config)
        result = ap.run(run_sam3=False, on_progress=_on_progress, service=service)
        auto_prompt = result.prompt
        pipe.send_log(
            f"Auto-prompter: {result.n_accepted} instances accepted, "
            f"{result.n_review} in review queue; classes={result.per_class_counts}"
        )

    if pipe.check_cancel():
        return

    # The auto-prompter already wrote output/vlm_analysis.json (the SAM3 worker reads it).
    pipe.send_progress(100, f"Auto-prompt complete: {auto_prompt}", stage="vlm")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_vlm_work, conn, session_dir, config)
