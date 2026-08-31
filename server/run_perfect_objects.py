#!/usr/bin/env python3
"""Per-object PERFECTION worker — separate process so the FastAPI loop stays
free AND pinned to 8 cores (Open3D/TBB thrashes unpinned on this 252-core
box — same lesson as the Poisson worker, 2026-08-29).

Spawned by /api/segmentation/perfect/export. Protocol (stdout):
    [PERF-PROGRESS]{"instance_id": i, "phase": "fit|done|error", "detail": ...}
    [PERF-RESULT]{"written": ["...glb", ...], "skipped": [...]}
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
except Exception:  # noqa: BLE001 — affinity is a mitigation, not a requirement
    pass

import argparse
import json
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass


def _progress(iid, phase, detail=""):
    print("[PERF-PROGRESS]" + json.dumps(
        {"instance_id": int(iid), "phase": phase, "detail": detail}),
        flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--instance-id", type=int, action="append", required=True)
    ap.add_argument("--mode", default="perfect", choices=["perfect", "parts", "model"],
                    help="'parts' = flat-colored diagnosis view + inventory")
    ap.add_argument("--source", action="append", default=[],
                    help="'<iid>:poisson' or '<iid>:pgsr' — which mesh of the "
                         "segment to perfect (default: poisson, else pgsr)")
    args = ap.parse_args()
    source_by_iid = {}
    for s in args.source:
        try:
            iid_s, suf = s.split(":", 1)
            source_by_iid[int(iid_s)] = suf.strip()
        except ValueError:
            pass

    from config import cfg
    sfc = (cfg.get("surface_fit") or {})
    output_dir = Path(args.output_dir)

    from segmentation.perfect_object import (build_model_object,
                                             diagnose_object, perfect_object)
    fn = {"parts": diagnose_object, "model": build_model_object,
          "perfect": perfect_object}[args.mode]
    written, skipped = [], []
    for iid in args.instance_id:
        t0 = time.time()
        _progress(iid, "fit", "decomposing + snapping")
        try:
            glb = fn(output_dir, int(iid), cfg=sfc,
                     source=source_by_iid.get(int(iid)),
                     log=lambda m: print(m, flush=True))
            if glb is not None:
                written.append(str(glb))
                _progress(iid, "done", f"{round(time.time() - t0, 1)}s")
            else:
                skipped.append({"instance_id": iid, "reason": "no output"})
                _progress(iid, "error", "no output")
        except Exception as e:  # noqa: BLE001 — one object must not sink the rest
            import traceback
            traceback.print_exc()
            skipped.append({"instance_id": iid, "reason": str(e)})
            _progress(iid, "error", str(e))
    print("[PERF-RESULT]" + json.dumps(
        {"written": written, "skipped": skipped}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
