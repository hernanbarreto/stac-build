#!/usr/bin/env python3
"""
Whole-scene TSDF worker — runs export_tsdf_scene in a SEPARATE PROCESS so the
GPU/Open3D work never blocks the FastAPI server's event loop (which made the UI
falsely report "server down" during reconstruction).

Spawned by the /api/segmentation/tsdf/scene_export endpoint via
asyncio.create_subprocess_exec(sys.executable, run_tsdf_scene.py, ...). Runs in
the same conda env as the server (da3, which has Open3D-CUDA). Progress is
streamed back as parseable stdout lines the endpoint forwards to /tsdf/progress.

Protocol (stdout):
    [TSDF-PROGRESS]{"phase": "...", "elapsed": 12.3, "mesh": "..."}
    [TSDF-RESULT]<glb_path>      (or [TSDF-RESULT]NONE on failure)
Everything else printed is plain log forwarded to the server console.
"""
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

# Make the server package importable (segmentation.tsdf_export, project_paths…).
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# Route the export_tsdf_scene logger ("TSDFExport") to stdout so its coverage
# diagnostics (per-frame source, mask points, n_integrated / skipped) reach the
# server console — main.py forwards this subprocess's stdout line by line.
# Without this the INFO records are dropped (no handler) and we fly blind.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def main():
    ap = argparse.ArgumentParser(description="Whole-scene TSDF worker (GPU)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("--params", default="{}",
                    help="JSON of export_tsdf_scene kwargs (voxel_length, sdf_trunc, ...)")
    args = ap.parse_args()

    try:
        params = json.loads(args.params) if args.params else {}
    except Exception:
        params = {}

    def _cb(phase, elapsed=None, mesh=None):
        upd = {"phase": phase}
        if elapsed is not None:
            upd["elapsed"] = float(elapsed)
        if mesh:
            upd["mesh"] = str(mesh)
        print("[TSDF-PROGRESS]" + json.dumps(upd), flush=True)

    try:
        from segmentation.tsdf_export import export_tsdf_scene
        path = export_tsdf_scene(
            output_dir=Path(args.output_dir),
            frames_dir=Path(args.frames_dir),
            session_dir=Path(args.session_dir) if args.session_dir else None,
            progress_cb=_cb,
            **params,
        )
        print("[TSDF-RESULT]" + (str(path) if path else "NONE"), flush=True)
        sys.exit(0 if path else 1)
    except Exception as e:
        traceback.print_exc()
        print(f"[TSDF-PROGRESS]{json.dumps({'phase': 'error', 'error': str(e)})}", flush=True)
        print("[TSDF-RESULT]NONE", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
