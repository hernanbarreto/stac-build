# STAC-Builder: VLM Worker (Subprocess)
# Runs InternVL3 scene analysis in its own process.
# Reads frames, writes scene_analysis.json / auto prompt.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _vlm_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """VLM scene analysis — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    scene_cfg = config.get("scene_analysis", {})
    autoprompt_cfg = config.get("autoprompt", {})

    def _on_progress(pct, msg):
        pipe.send_progress(pct, msg, stage="vlm")
        pipe.send_log(msg)

    if pipe.check_cancel():
        return

    # Phase 1 DEFAULT: Qwen3-VL grounded auto-prompter over the shared semantic
    # service. It writes output/vlm_analysis.json itself (richer, box-aware) and
    # returns the (prompt, frame_map) contract the SAM3 worker consumes.
    # InternVL3's analyze_scene stays as the fallback (autoprompt disabled or on
    # error / unreachable service).
    auto_prompt, frame_map = None, {}
    used_autoprompt = False
    if autoprompt_cfg.get("enabled", True):
        try:
            pipe.send_progress(0, "Auto-prompting with Qwen3-VL...", stage="vlm")
            pipe.send_log("Starting Qwen3-VL auto-prompter (Phase 1)")
            from segmentation.autoprompt.session_builder import AutoPrompter
            ap = AutoPrompter(session_path, session_path / "output",
                              backend=autoprompt_cfg.get("backend", "qwen_local"),
                              config=config)
            result = ap.run(run_sam3=False, on_progress=_on_progress)
            auto_prompt, frame_map = result.prompt, result.frame_map
            used_autoprompt = True
            pipe.send_log(
                f"Auto-prompter: {result.n_accepted} instances accepted, "
                f"{result.n_review} in review queue; classes={result.per_class_counts}"
            )
        except Exception as e:  # noqa: BLE001
            pipe.send_log(f"Auto-prompter failed ({e}); falling back to InternVL3",
                          level="warning")

    if not used_autoprompt:
        pipe.send_progress(0, "Loading InternVL3 model...", stage="vlm")
        pipe.send_log("Starting VLM scene analysis (InternVL3 fallback)")
        from scene_analyzer import analyze_scene
        auto_prompt, frame_map = analyze_scene(str(frames_dir), scene_cfg, on_progress=_on_progress)

    if pipe.check_cancel():
        return

    # The auto-prompter already wrote output/vlm_analysis.json; the fallback path
    # writes it below. Either way the SAM3 worker reads the same file.
    if used_autoprompt:
        pipe.send_progress(100, f"Auto-prompt complete: {auto_prompt}", stage="vlm")
        return

    if auto_prompt:
        pipe.send_log(f"Auto-detected prompt: '{auto_prompt}'")
        categories = [c.strip() for c in auto_prompt.split(";") if c.strip()]
        pipe.send_log(f"Categories: {len(categories)}, frame mappings: {len(frame_map)}")
    else:
        auto_prompt = "floor;wall;ceiling;door;window;furniture;object"
        frame_map = {}
        pipe.send_log("No categories detected, using fallback", level="warning")

    # Write results to disk so SAM3 worker can read them
    import json
    output_dir = (session_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm_result = {
        "prompt": auto_prompt,
        "frame_map": frame_map,
    }
    result_path = output_dir / "vlm_analysis.json"
    result_path.write_text(json.dumps(vlm_result, indent=2))

    pipe.send_log(f"VLM result saved to {result_path.name}")
    pipe.send_progress(100, f"Scene analysis complete: {auto_prompt}", stage="vlm")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_vlm_work, conn, session_dir, config)
