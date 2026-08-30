#!/usr/bin/env python3
"""Per-object Poisson worker — separate process so the FastAPI loop stays free
AND so we can pin CPU affinity: Open3D's Poisson thrashes/hangs on this
252-core box (TBB ignores OMP_NUM_THREADS; verified 2026-08-29 — hangs forever
unpinned, ~1 min/object pinned to 8 cores).

Spawned by /api/segmentation/tsdf/export. Protocol (stdout):
    [POBJ-PROGRESS]{"instance_id": i, "phase": "poisson|done|error"}
    [POBJ-RESULT]{"written": ["...glb", ...]}
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
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--instance-id", type=int, action="append", default=None)
    ap.add_argument("--depth", type=int, default=0,
                    help="Poisson octree depth; 0 = auto from object size")
    ap.add_argument("--indices-file", default=None,
                    help="mesh an arbitrary cloud-index subset instead of instances")
    ap.add_argument("--out-glb", default=None,
                    help="output GLB path (required with --indices-file)")
    ap.add_argument("--texture", action="store_true",
                    help="bake a texrecon atlas from the scan frames")
    ap.add_argument("--regularize", action="store_true",
                    help="REGULARIZED POISSON: iron primitive regions onto "
                         "their fitted models and cut image-confirmed openings")
    args = ap.parse_args()

    depth = args.depth if args.depth > 0 else None
    if args.indices_file:
        from segmentation.poisson_object import poisson_mesh_indices
        ok = poisson_mesh_indices(Path(args.output_dir), Path(args.indices_file),
                                  Path(args.out_glb), depth=depth,
                                  texture=args.texture)
        print("[POBJ-RESULT]" + json.dumps(
            {"written": [args.out_glb] if ok else []}), flush=True)
        return 0 if ok else 1

    def _progress(iid, phase, elapsed, mesh):
        print("[POBJ-PROGRESS]" + json.dumps(
            {"instance_id": int(iid), "phase": phase,
             "elapsed": elapsed, "mesh": mesh}), flush=True)

    from segmentation.poisson_object import export_poisson_objects
    written = export_poisson_objects(
        Path(args.output_dir), obj_ids=args.instance_id,
        depth=depth, texture=args.texture, regularize=args.regularize,
        progress_cb=_progress)
    print("[POBJ-RESULT]" + json.dumps({"written": [str(w) for w in written]}),
          flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
