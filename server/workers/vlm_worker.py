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
# REUSE WITHOUT A LAUNCH (point 82, 2026-10-08 close of wave 2): when the session's
# vlm_analysis.json is reusable in every part of its stamp but the engine's identity, that
# identity is computed WITHOUT starting vLLM (semantic.serve --identity-only in the semantic
# env: the weights verified by sha256, the card, the versions — exactly what a launch
# records) and, when the stamp then matches, no vLLM is launched, hashed or stopped: the
# analysis is the answer, declared in the log. When it does not match, the engine is
# launched as before and its live identity must equal the expected one.
# UNDERSTANDING REUSED, MERGE RE-DERIVED (2026-10-09): when the analysis is not reusable whole
# but its per-image understanding is (AutoPrompter.understanding_awaits_engine_identity — only
# what follows the per-image calls changed), the engine is still launched (the merge pass
# needs it), declared in the log; AutoPrompter.run decides on the live identity.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import contextlib
import sys
from pathlib import Path
from multiprocessing.connection import Connection
from typing import Optional

from workers.base import WorkerPipe, run_worker_safe


def _engine_plan(pipe: WorkerPipe, config: dict, session_path: Optional[Path],
                 run_cfg_path, backend: str) -> dict:
    """Whether this job needs its own vLLM at all: ``{"launch": bool, "expected": identity
    or None}``. The identity of the engine this job WOULD launch is computed without
    launching it (semantic.serve.expected_identity — the weights' sha256, the card, the
    versions) ONLY when it is the last thing standing between the session and a whole
    reuse of its analysis (AutoPrompter.reuse_awaits_engine_identity: every other stamped
    input already matches); otherwise the launch verifies the weights once, as always. A
    reuse under the expected identity needs no engine: ``launch`` False. An identity that
    cannot be computed FAILS the stage (semantic.serve.ServeError) — the launch would fail
    the same way."""
    if session_path is None:
        return {"launch": True, "expected": None}
    from segmentation.autoprompt.session_builder import AutoPrompter
    from semantic.serve import expected_identity
    ap = AutoPrompter(session_path, session_path / "output", backend=backend, config=config)
    pending, why = ap.reuse_awaits_engine_identity()
    if not pending:
        # the analysis is not reusable whole; its PER-IMAGE understanding may still be
        # (2026-10-09: only the merge changed) — then the engine is launched for the merge
        # pass alone and run() takes the decision on the live identity, declared
        u_ok, u_why = ap.understanding_awaits_engine_identity()
        if u_ok:
            pipe.send_log(f"[vlm] DECLARED: the analysis is derived again ({why}), but its "
                          f"per-image understanding is reusable once this job's engine shows "
                          f"the identity it was made with ({u_why}) — this job's vLLM is "
                          f"launched for the merge pass; the per-image calls are not re-run")
        else:
            pipe.send_log(f"[vlm] the analysis is derived again ({why}; the per-image "
                          f"understanding too: {u_why}) — this job's vLLM is launched")
        return {"launch": True, "expected": None}
    ident = expected_identity(backend, run_cfg_path, log=pipe.send_log)
    doc, why = ap.analysis_reusable(ident)
    if doc is None:
        pipe.send_log(f"[vlm] the analysis is derived again ({why}) — this job's vLLM is launched")
        return {"launch": True, "expected": ident}
    pipe.send_log(f"[vlm] DECLARED: vlm_analysis.json is reusable WHOLE under its stamp "
                  f"({why}; engine identity {ident.get('sha256', '')[:12]} verified without a "
                  f"launch: weights sha256, card, versions) — this job's vLLM is NOT launched, "
                  f"hashed again or stopped; the understanding is not re-run")
    return {"launch": False, "expected": ident}


class _ReusedIdentity(contextlib.nullcontext):
    """The identity of an engine this job did NOT launch: the session's analysis is reused
    whole under it (:func:`_engine_plan`), so the stage auto-prompts nothing."""
    launched = False


def _ensure_semantic_service(pipe: WorkerPipe, config: dict):
    """THIS JOB'S engine (semantic.service.job_engine): a context manager that yields the
    verified identity of the vLLM it launched from the frozen run configuration and stops
    it on exit — or, when the session's analysis is reusable whole under the identity that
    engine would have (:func:`_engine_plan`), that identity WITHOUT a launch. The frozen
    file (output/run_config.yaml, written by the pipeline manager at job start) is what the
    launcher reads its semantic block from; its block must be the one this worker was given
    — two configurations never run one stage. The session directory travels in
    ``config["_session_dir"]`` (a private key, like the manager's ``_pipeline_replace``);
    without it (a CLI, a test) no frozen file is read and nothing is reused."""
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
    plan = _engine_plan(pipe, config, session_path, run_cfg_path, backend)
    if not plan["launch"]:
        pipe.send_progress(0, "VLM analysis reused under its stamp — no engine launched",
                           stage="vlm")
        return _ReusedIdentity(plan["expected"])
    pipe.send_progress(0, "Starting this job's semantic service (Qwen3-VL)...", stage="vlm")
    engine = job_engine(config, backend=backend, owner="vlm_worker", stage="VLM stage",
                        session_dir=session_path, run_config_path=run_cfg_path,
                        log=lambda m: pipe.send_log(m), cancelled=pipe.check_cancel)
    if plan["expected"] is None:
        return engine
    # the engine that comes up must be the one the reuse was refused on (one check for
    # every consumer of an identity computed without a launch: semantic.serve)
    from semantic.serve import engine_as_expected
    return engine_as_expected(engine, plan["expected"], stage="VLM stage")


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

    seg_round = int(config.get("_seg_round") or 1)
    if seg_round >= 2:
        _vlm_round(pipe, session_path, config, seg_round)
        return

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
        if getattr(engine, "launched", True):
            pipe.send_progress(0, "Auto-prompting with Qwen3-VL...", stage="vlm")
            pipe.send_log("Starting Qwen3-VL auto-prompter (Phase 1)")
        else:
            # no engine of this job's own (the analysis is reused under its stamp): the
            # auto-prompter below returns the session's analysis, it asks nothing
            pipe.send_log("the session's analysis is reused under its stamp — nothing is "
                          "auto-prompted (no engine)")
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


def _vlm_round(pipe: WorkerPipe, session_path: Path, config: dict, k: int):
    """Round ``k`` >= 2 of the segmentation (segmentation.autoprompt.rounds): THIS job's engine,
    the keyframes with what is already segmented painted grey, the new prompts recorded in
    output/seg_rounds.json for the SAM3 stage that follows."""
    from intake.run_config import load_run_config, run_config_path
    from semantic.service import job_engine
    from segmentation.autoprompt.rounds import run_round
    frozen, sha = load_run_config(session_path)
    if (frozen.get("semantic") or {}) != (config.get("semantic") or {}):
        raise RuntimeError(
            "the 'semantic' block of the frozen run configuration (output/run_config.yaml, "
            f"sha256 {sha[:12]}) differs from the one this worker received — a stage never "
            "runs on two configurations (point 155)")
    backend = (config.get("autoprompt", {}) or {}).get("backend", "qwen_local")
    pipe.send_progress(0, f"Round {k}: starting this job's semantic service (Qwen3-VL)...",
                       stage="vlm")
    engine = job_engine(config, backend=backend, owner="vlm_worker", stage=f"VLM round {k}",
                        session_dir=session_path, run_config_path=run_config_path(session_path),
                        log=lambda m: pipe.send_log(m), cancelled=pipe.check_cancel)
    with engine as identity:
        if not identity:
            raise RuntimeError("semantic service unavailable")
        rec = run_round(session_path, config, k, log=pipe.send_log,
                        progress=lambda pct, msg: pipe.send_progress(pct, msg, stage="vlm"),
                        cancelled=pipe.check_cancel)
    new = rec.get("new_prompts") or []
    pipe.send_progress(100, f"Round {k}: {len(new)} new prompt(s)"
                       + (f": {'; '.join(new)}" if new else " — nothing left to name"),
                       stage="vlm")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_vlm_work, conn, session_dir, config)
