#!/usr/bin/env python3
"""Offline runner — DINOv3 plan FASE 1/2 on an existing session.

Scores cleaned_cloud.ply (fase 1). The fase-2 filter runs ONLY with
--filter AND is refused when a segmentation already maps globalIndices
into this cloud (filtering would desynchronize every instance).

    python run_dino_score.py --output-dir <session>/output [--filter]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
except Exception:  # noqa: BLE001
    pass

import argparse
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
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--filter", action="store_true",
                    help="fase 2: apply the two-witness filter (refused on "
                         "sessions with a mapped segmentation)")
    args = ap.parse_args()
    out = Path(args.output_dir)

    from config import cfg
    dcfg = dict(cfg.get("dino_features") or {})
    if args.filter:
        seg = out / "segmentation_result.json"
        if seg.exists():
            print("REFUSED: segmentation_result.json exists — filtering the "
                  "cloud now would break every instance's globalIndices. "
                  "Re-run segmentation after filtering, or score only.")
            return 2
        dcfg["filter"] = dict(dcfg.get("filter") or {}, enabled=True)
    else:
        dcfg["filter"] = dict(dcfg.get("filter") or {}, enabled=False)

    from reconstruction.cloud_feature_score import run
    rep = run(out, Path(args.frames_dir) if args.frames_dir else None,
              cfg=dcfg, log=print)
    return 0 if rep else 1


if __name__ == "__main__":
    raise SystemExit(main())
