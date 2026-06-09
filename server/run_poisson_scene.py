#!/usr/bin/env python3
"""
Whole-scene Poisson worker — runs export_poisson_scene in a SEPARATE PROCESS so the
GPU/Open3D work never blocks the FastAPI server's event loop (same rationale as
run_tsdf_scene.py).

Spawned by the /api/segmentation/poisson/scene_export endpoint via
asyncio.create_subprocess_exec(sys.executable, run_poisson_scene.py, ...). Runs in
the same conda env as the server (da3, which has Open3D-CUDA). Progress is streamed
back as the SAME parseable stdout lines the TSDF worker uses, so the endpoint reuses
one parser.

Protocol (stdout):
    [TSDF-PROGRESS]{"phase": "...", "elapsed": 12.3, "mesh": "..."}
    [TSDF-RESULT]<glb_path>      (or [TSDF-RESULT]NONE on failure)
Everything else printed is plain log forwarded to the server console.
"""
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

# Same real-time, lossless logging setup as run_tsdf_scene.py (see its header).
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


_CAPTURED_LOGGERS = ("TSDFExport", "TextureBake", "NvdiffrastBake", "CameraSource")
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
    """Tee all captured logs to output/tsdf/scene_poisson/run.log so a HARD crash
    still leaves the full run on disk (mirrors run_tsdf_scene.py)."""
    try:
        log_dir = output_dir / "tsdf" / "scene_poisson"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "run.log", mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        logging.getLogger().addHandler(fh)
        for _name in _CAPTURED_LOGGERS:
            logging.getLogger(_name).addHandler(fh)
        faulthandler.enable(file=open(log_dir / "run.log.fault", "w"))
        logging.getLogger("TSDFExport").info(f"[worker] run.log → {log_dir / 'run.log'}")
    except Exception as e:
        print(f"[worker] could not attach file log: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Whole-scene Poisson worker (GPU)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("--params", default="{}",
                    help="JSON of export_poisson_scene kwargs (poisson_depth, ...)")
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
        from segmentation.tsdf_export import export_poisson_scene
        path = export_poisson_scene(
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
