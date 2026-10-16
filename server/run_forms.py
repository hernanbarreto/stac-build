#!/usr/bin/env python3
"""FORMS worker — the Mesh button's per-object stage (USER 2026-10-16: the VLM decides the form,
the cloud measures pose and dimensions, the masks of every view correct it). A separate process:
the FastAPI loop stays free, torch runs strict-deterministic on the card, vLLM is brought up for
the form query if it is down.

Usage:
    python run_forms.py --session-dir <.../src_default> [--instance-id N ...] [--no-query] [--no-texture]

Protocol (stdout):
    [FORMS-PROGRESS]{"instance_id": 3, "phase": "fit|mesh|tsdf|publish|done|error"}
    [FORMS-RESULT]{"ok": [...], "failed": [...], "skipped": [...]}     (single line, JSON)
"""
import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass
faulthandler.enable()
os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
except Exception:  # noqa: BLE001 — affinity is a mitigation, not a requirement
    pass


def _progress(iid: int, phase: str) -> None:
    print("[FORMS-PROGRESS]" + json.dumps({"instance_id": int(iid), "phase": phase}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="the forms stage of a session")
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--instance-id", type=int, action="append", default=None,
                    help="instance to mesh (repeatable); default: every instance")
    ap.add_argument("--no-query", action="store_true", help="use the form records on disk only")
    ap.add_argument("--no-texture", action="store_true")
    args = ap.parse_args()
    import repro
    from config import cfg
    from reconstruction.forms.catalog import load_forms_config
    from reconstruction.forms.run import run_session
    fcfg = load_forms_config(cfg)
    repro.enable_deterministic_torch(int(fcfg.seed))
    client = None
    if not args.no_query:
        from segmentation.object_captioner import _service_client
        client = _service_client(None, log=lambda m: print(m, flush=True))
    summary = run_session(Path(args.session_dir), instance_ids=args.instance_id, fcfg=fcfg, raw_cfg=cfg,
                          log=lambda m: print(m, flush=True), progress=_progress, client=client,
                          texture=(False if args.no_texture else None), query=not args.no_query)
    print("[FORMS-RESULT]" + json.dumps({"ok": summary["ok"], "failed": summary["failed"],
                                         "skipped": summary["skipped"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
