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
import os
# OMP cap BEFORE any numeric import: on this 252-core box unbounded OpenMP
# (Open3D Poisson, scipy) THRASHES — same load-bearing lesson as PGSR's
# torch.set_num_threads(8). Must be set before the first parallel region.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import faulthandler
import json
import logging
import sys
import traceback
from pathlib import Path

# Make the server package importable (segmentation.tsdf_export, project_paths…).
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# ── Never fly blind again ───────────────────────────────────────────────────
# Real-time, lossless logging so the server sees EVERY line live and on disk:
#   1. line-buffered stdout/stderr → each line hits the pipe immediately (the
#      server forwards this subprocess's stdout line by line).
#   2. faulthandler → on a NATIVE crash (Open3D/CUDA segfault/abort, which leaves
#      no Python traceback) the C-level stack of every thread is dumped to stderr
#      before the process dies — that is how we locate a hard crash.
#   3. a flushing StreamHandler attached DIRECTLY to the module loggers. They set
#      propagate=False (TSDFExport, TextureBake) so basicConfig's root handler
#      NEVER receives their records — the real cause of the silent log loss in the
#      manual-button path. Attaching the handler to each logger fixes it.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
faulthandler.enable()


class _FlushingStream(logging.StreamHandler):
    """StreamHandler that flushes after every record so nothing is lost on crash."""
    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


# Loggers we MUST capture (each may set propagate=False → invisible to root).
_CAPTURED_LOGGERS = ("TSDFExport", "TextureBake", "NvdiffrastBake", "CameraSource",
                     "CloudMesh", "MeshFusion")  # cloud_delaunay stage — a silent
                     # death in the pipeline path proved these MUST be captured
_LOG_FORMAT = logging.Formatter("[%(name)s] %(message)s")

_stdout_handler = _FlushingStream(sys.stdout)
_stdout_handler.setLevel(logging.INFO)
_stdout_handler.setFormatter(_LOG_FORMAT)
logging.basicConfig(level=logging.INFO, handlers=[_stdout_handler])  # root (other libs)
for _name in _CAPTURED_LOGGERS:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.INFO)
    _lg.addHandler(_stdout_handler)
    _lg.propagate = False  # handler attached directly → don't double-log via root


def _attach_file_log(output_dir: Path) -> None:
    """Tee all captured logs to output/tsdf/scene/run.log so a HARD crash (which
    can lose the in-flight server pipe) still leaves the full run on disk."""
    try:
        log_dir = output_dir / "tsdf" / "scene"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "run.log", mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        logging.getLogger().addHandler(fh)
        for _name in _CAPTURED_LOGGERS:
            logging.getLogger(_name).addHandler(fh)
        # Also stream native-crash tracebacks into the same file.
        faulthandler.enable(file=open(log_dir / "run.log.fault", "w"))
        logging.getLogger("TSDFExport").info(f"[worker] run.log → {log_dir / 'run.log'}")
    except Exception as e:
        print(f"[worker] could not attach file log: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Whole-scene TSDF worker (GPU)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("--params", default="{}",
                    help="JSON of export_tsdf_scene kwargs (voxel_length, sdf_trunc, ...)")
    args = ap.parse_args()
    _attach_file_log(Path(args.output_dir))

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
