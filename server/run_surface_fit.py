#!/usr/bin/env python3
"""
surface_fit standalone CLI — fit measurement-backed smooth surfaces to SAM3
segments (plane → quadric → swept profile → B-spline) and export the fidelity
record (deviation PLY + heatmap PNG + residual stats JSON) per instance.

Runs in the same conda env as the server (da3: Open3D + scipy + matplotlib).

Usage:
    # one instance of a session
    python run_surface_fit.py --session-dir <.../src_default> --instance-id 3
    # every architectural instance of a session
    python run_surface_fit.py --session-dir <.../src_default> --all
    # ad-hoc: a bare segment PLY, no session needed
    python run_surface_fit.py --ply wall.ply --label wall

Protocol (stdout), same family as run_tsdf_scene.py:
    [SFIT-PROGRESS]{"instance_id": 3, "phase": "fit", "elapsed": 1.2}
    [SFIT-RESULT]{"ok": [...], "failed": [...]}     (single line, JSON)
"""
import argparse
import faulthandler
import json
import logging
import sys
import traceback
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

# Same lossless-logging setup as run_tsdf_scene.py: line-buffered stdout,
# faulthandler for native crashes, flushing handler attached directly to the
# named module logger (propagate=False).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
faulthandler.enable()


class _FlushingStream(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


_CAPTURED_LOGGERS = ("SurfaceFit",)
_stdout_handler = _FlushingStream(sys.stdout)
_stdout_handler.setLevel(logging.INFO)
_stdout_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_stdout_handler])
for _name in _CAPTURED_LOGGERS:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.INFO)
    _lg.addHandler(_stdout_handler)
    _lg.propagate = False


def _attach_file_log(base_dir: Path) -> None:
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(base_dir / "run.log", mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
        logging.getLogger().addHandler(fh)
        for _name in _CAPTURED_LOGGERS:
            logging.getLogger(_name).addHandler(fh)
        logging.getLogger("SurfaceFit").info("run.log → %s", base_dir / "run.log")
    except Exception as e:
        print(f"[worker] could not attach file log: {e}", flush=True)


def _progress(instance_id=None, phase="", elapsed=None, **_):
    upd = {"phase": phase}
    if instance_id is not None:
        upd["instance_id"] = instance_id
    if elapsed is not None:
        upd["elapsed"] = round(float(elapsed), 2)
    print("[SFIT-PROGRESS]" + json.dumps(upd), flush=True)


def _safe_name(label: str, instance_id) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (label or "segment"))
    return f"{safe}_{instance_id}" if instance_id is not None else safe


def main():
    ap = argparse.ArgumentParser(description="surface_fit CLI (plane/quadric/swept/bspline + residual deliverables)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--session-dir", help="session dir containing output/cleaned_cloud.ply + segmentation_result.json")
    src.add_argument("--ply", help="ad-hoc mode: fit a bare segment point cloud")
    ap.add_argument("--instance-id", type=int, action="append", default=None,
                    help="instance to fit (repeatable); default with --session-dir: --all required")
    ap.add_argument("--all", action="store_true", help="fit every instance of the session")
    ap.add_argument("--label", default="", help="label for --ply mode (used in outputs)")
    ap.add_argument("--output-dir", default=None,
                    help="override artifact dir (default: <session>/output/surface_fit or <ply_dir>/surface_fit)")
    ap.add_argument("--params", default="{}",
                    help="JSON of fit_segment config kwargs (overrides config.yaml surface_fit section)")
    args = ap.parse_args()

    import numpy as np
    from config import cfg
    from reconstruction.surface_fit import fit_segment, build_surface_fit_kwargs

    try:
        overrides = json.loads(args.params) if args.params else {}
    except Exception:
        overrides = {}
    kwargs = build_surface_fit_kwargs(cfg, overrides)

    ok, failed, skipped = [], [], []
    try:
        if args.ply:
            import open3d as o3d
            pts = np.asarray(o3d.io.read_point_cloud(args.ply).points)
            base = Path(args.output_dir) if args.output_dir else Path(args.ply).resolve().parent / "surface_fit"
            _attach_file_log(base)
            out = base / _safe_name(args.label, 0)
            fs = fit_segment(pts, instance_id=0, label=args.label or Path(args.ply).stem,
                             out_dir=out, progress_cb=_progress, **kwargs)
            (ok if fs is not None else failed).append({"instance_id": 0, "label": args.label,
                                                       "dir": str(out) if fs else None})
        elif args.all:
            # scene mode: fit every architectural instance + stage-3 scene-wide
            # regularization (plane snapping, clean edges) + hybrid report
            from reconstruction.surface_fit.scene import fit_scene, scene_config_kwargs
            session_dir = Path(args.session_dir)
            base = Path(args.output_dir) if args.output_dir else session_dir / "output" / "surface_fit"
            _attach_file_log(base)
            rep = fit_scene(session_dir, out_base=base, config=cfg,
                            overrides=overrides, progress_cb=_progress,
                            **scene_config_kwargs(cfg))
            ok = rep["fitted"]
            failed = [s for s in rep["skipped_to_tsdf"]
                      if "no model" in s.get("reason", "")]
            skipped = [s for s in rep["skipped_to_tsdf"]
                       if "no model" not in s.get("reason", "")]
        else:
            # selective mode — SAME scene pipeline as --all (uniform config
            # ladder — NO name routing, geometry gate accept_p95_mm, stage-3
            # regularization among the fitted planes, hybrid report), restricted
            # to the requested instances. The old per-segment loop here skipped
            # regularization and ignored roles, so a selective run produced
            # different (worse) geometry than a scene run — now they match.
            from reconstruction.surface_fit.scene import fit_scene, scene_config_kwargs
            session_dir = Path(args.session_dir)
            base = Path(args.output_dir) if args.output_dir else session_dir / "output" / "surface_fit"
            _attach_file_log(base)
            wanted = [int(i) for i in (args.instance_id or [])]
            if not wanted:
                ap.error("--session-dir requires --instance-id or --all")
            rep = fit_scene(session_dir, out_base=base, config=cfg,
                            overrides=overrides, progress_cb=_progress,
                            instance_ids=wanted, **scene_config_kwargs(cfg))
            ok = rep["fitted"]
            failed = [s for s in rep["skipped_to_tsdf"]
                      if "no model" in s.get("reason", "")]
            skipped = [s for s in rep["skipped_to_tsdf"]
                       if "no model" not in s.get("reason", "")]

        print("[SFIT-RESULT]" + json.dumps({"ok": ok, "failed": failed,
                                            "skipped": skipped}), flush=True)
        sys.exit(0 if ok and not failed else (0 if ok else 1))
    except Exception as e:
        traceback.print_exc()
        print("[SFIT-PROGRESS]" + json.dumps({"phase": "error", "error": str(e)}), flush=True)
        print("[SFIT-RESULT]" + json.dumps({"ok": ok, "failed": failed, "error": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
