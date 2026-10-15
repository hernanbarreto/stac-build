#!/usr/bin/env python3
"""Per-object TSDF worker (USER 2026-10-15 "todo tiene que correr sobre todo lo corregido"): the
PUBLISHED cloud of the live epoch is the depth (segmentation.tsdf_export, depth_source live_cloud),
on the card, torch STRICT deterministic. Protocol (stdout):
    [TOBJ-PROGRESS]{"instance_id": i, "phase": "starting|integrating|extracting|done|error", "elapsed": s}
    [TOBJ-RESULT]{"written": ["...glb", ...]}
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

_KEYS = ("depth_source", "device", "voxel_length", "sdf_trunc", "depth_min", "depth_trunc",
         "fill_px", "weight_threshold", "block_count")


def object_tsdf_config(cfg) -> dict:
    """`tsdf.object`, every key required — a missing one fails naming it."""
    obj = ((cfg.get("tsdf") or {}).get("object") or {}) if hasattr(cfg, "get") else {}
    missing = [k for k in _KEYS if k not in obj]
    if missing:
        raise RuntimeError(f"tsdf.object is missing {missing} — nothing is assumed")
    return {k: obj[k] for k in _KEYS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--instance-id", type=int, action="append", required=True)
    ap.add_argument("--suffix", default="_tsdf", help="tsdf/<label>_<id><suffix>/")
    args = ap.parse_args()
    import repro
    repro.enable_deterministic_torch(0)   # STRICT, before anything touches CUDA (F6's entry)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from config import cfg
    from segmentation.tsdf_export import export_tsdf_meshes
    oc = object_tsdf_config(cfg)
    out = Path(args.output_dir)
    session_dir = out.parent
    seg = json.loads((out / "segmentation_result.json").read_text())
    t0 = time.time()

    def _cb(i, phase, elapsed, mesh):
        print("[TOBJ-PROGRESS]" + json.dumps({"instance_id": int(i), "phase": phase,
                                               "elapsed": None if elapsed is None else round(float(elapsed), 1),
                                               "mesh": None if mesh is None else str(mesh)}), flush=True)

    glbs = export_tsdf_meshes(
        out, session_dir / "frames", seg, session_dir=session_dir,
        obj_ids=[int(i) for i in args.instance_id],
        voxel_length=float(oc["voxel_length"]), sdf_trunc=float(oc["sdf_trunc"]),
        depth_trunc=float(oc["depth_trunc"]), depth_min=float(oc["depth_min"]),
        progress_cb=_cb, name_suffix=str(args.suffix),
        depth_source=str(oc["depth_source"]), device=str(oc["device"]),
        live_fill_px=int(oc["fill_px"]), weight_threshold=float(oc["weight_threshold"]),
        block_count=int(oc["block_count"]))
    print("[TOBJ-RESULT]" + json.dumps({"written": [str(g) for g in glbs],
                                         "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    return 0 if glbs else 1


if __name__ == "__main__":
    sys.exit(main())
