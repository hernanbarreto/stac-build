# STAC-Builder — Auto-prompter batch CLI (Phase 1).
#
# Headless: keyframes in -> tracked masks out, no UI. This is the masklet
# PRODUCER for Phase R (run it early, in parallel with DA3-Streaming).
#
#   conda activate da3 && cd server
#   python -m segmentation.autoprompt.cli --session <session_dir> [--run-sam3]
#
# Without --run-sam3 it emits only the auto-prompt contract + review queue
# (fast, no SAM3/VRAM). With --run-sam3 it also produces segmentation.json +
# seg_masks.npz (loads SAM3).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[2]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from segmentation.autoprompt.session_builder import AutoPrompter  # noqa: E402


def _load_config() -> dict:
    try:
        from config import cfg
        return dict(cfg)
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="SAM3 auto-prompter (Qwen3-VL) — batch")
    ap.add_argument("--session", required=True, help="session dir (contains frames/)")
    ap.add_argument("--output", default=None, help="output dir (default <session>/output)")
    ap.add_argument("--backend", default="qwen_local")
    ap.add_argument("--run-sam3", action="store_true", help="also run SAM3 to produce masks")
    ap.add_argument("--max-keyframes", type=int, default=0, help="0 = all selected keyframes")
    ap.add_argument("--vocabulary", default=None, help="override vocabulary.yaml path")
    args = ap.parse_args()

    session = Path(args.session)
    output = Path(args.output) if args.output else session / "output"
    config = _load_config()
    # allow CLI to override a couple of config knobs
    config.setdefault("autoprompt", {})
    if args.max_keyframes:
        config["autoprompt"]["max_keyframes"] = args.max_keyframes

    ap_runner = AutoPrompter(session, output, backend=args.backend, config=config,
                             vocab_path=args.vocabulary)

    def on_progress(pct, msg):
        print(f"[autoprompt {pct:3d}%] {msg}", flush=True)

    result = ap_runner.run(run_sam3=args.run_sam3, on_progress=on_progress)
    print("\n=== AUTO-PROMPT RESULT ===")
    print(result.summary())
    print(f"prompt: {result.prompt}")
    print(f"vlm_analysis: {result.vlm_analysis_path}")
    print(f"review_queue: {result.review_queue_path}  ({result.n_review} dubious)")
    print(f"sam3_ran: {result.sam3_ran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
