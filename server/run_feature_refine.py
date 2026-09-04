#!/usr/bin/env python3
"""Offline runner — DINOv3 plan FASE 4 on an existing session.

Flags background-contaminated points of segmented instances by feature
affinity. Writes reversible sidecars only (never mutates the segmentation).

    python run_feature_refine.py --output-dir <session>/output \
        --instance-id 1 [--instance-id 2 ...]     (omit ids = all instances)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
except Exception:  # noqa: BLE001
    pass

import argparse
import json
import logging
import sys
from pathlib import Path

_SERVER = Path(__file__).resolve().parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--instance-id", type=int, action="append", default=[])
    args = ap.parse_args()
    out = Path(args.output_dir)

    from config import cfg
    dcfg = dict(cfg.get("dino_features") or {})
    icfg = {**dcfg, **(dcfg.get("instance_refine") or {})}

    ids = args.instance_id
    if not ids:
        res = json.loads((out / "segmentation_result.json").read_text())
        ids = [int(i.get("instance_id", i.get("id")))
               for i in res.get("instances", [])]
    from segmentation.instance_feature_refine import refine_instance
    rc = 0
    for iid in ids:
        try:
            refine_instance(out, iid, cfg=icfg, log=print)
        except Exception as e:  # noqa: BLE001 — one instance must not sink all
            import traceback
            traceback.print_exc()
            print(f"[feat-refine] instance {iid} FAILED: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
