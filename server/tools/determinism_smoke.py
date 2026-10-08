#!/usr/bin/env python3
"""GPU determinism smoke test of the two STRICT inferences of the reconstruction: DA3 (intake I3 /
F2 windows, docs/plan_determinismo.md point 27) and PointDiT (F6 mono detail, point 54).

Both run under ``torch.use_deterministic_algorithms(True)`` without warn-only: an op with no
deterministic CUDA kernel RAISES, and on a real run that raise would kill the reconstruction at
the intake (DA3) or at F6 (PointDiT). This tool finds out BEFORE a run, on the real weights, the
real code path and real frames, in minutes:

1. STRICT RAISE — each model is loaded by the pipeline's own loader (``extract_da3_depth._load_model``
   with the pinned, sha-checked weights; ``precision.pointdit_runner.PointDiTRunner`` from the real
   ``reconstruction.precision.mono_detail`` config) and run on real frames of a session at the
   pipeline's resolution / tiles, in bf16 autocast, strict. A missing deterministic kernel shows
   up as the exception torch raises, naming the op.
2. OP CENSUS — the first pass runs under a TorchDispatchMode that records every op executed. Ops
   outside torch's own namespaces (custom CUDA extensions, e.g. xformers) are LISTED APART: torch's
   deterministic check cannot see inside them, only repetition can.
3. BIT-EXACT REPETITION — ``--repeat`` passes in the same process, and (mode ``all``) two fresh
   processes per model launched with the chain's step environment (repro.deterministic_env + the
   precision runner's BLAS pins): every output array's sha256 must be identical everywhere.

Run it with the GPU FREE (no reconstruction, no vLLM) — it never writes into the session:

    cd server && /workspace/miniforge3/envs/da3/bin/python tools/determinism_smoke.py all \\
        --session projects/pccr/scans/2026-08-31/src_default --out-dir /tmp/det_smoke

Exit code 0 = both models deterministic; 1 = a model raised or differed (the report says which).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import numpy as np  # noqa: E402

import repro  # noqa: E402

DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
# torch's own op namespaces: their ops are covered by use_deterministic_algorithms (an op with no
# deterministic kernel raises); anything else is a custom kernel torch cannot vouch for
TORCH_NAMESPACES = ("aten", "prims", "prim", "_c10d_functional", "c10d", "profiler", "quantized")
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")


def _sha(a: Any) -> Optional[str]:
    if a is None:
        return None
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _frames(session: Path) -> List[Path]:
    fdir = session / "frames"
    out = sorted(p for p in fdir.iterdir() if p.suffix.lower() in FRAME_SUFFIXES) if fdir.is_dir() else []
    if not out:
        raise SystemExit(f"no frames under {fdir}")
    return out


def _spread(items: List[Path], n: int) -> List[Path]:
    """``n`` items evenly spread over the list (first and last included) — a fixed choice."""
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    return [items[round(k * (len(items) - 1) / (n - 1))] for k in range(n)]


class _Census:
    """Every op dispatched inside the block, counted by name (TorchDispatchMode)."""

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        census = collections.Counter()

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                census[str(func)] += 1
                return func(*args, **(kwargs or {}))

        self.counts = census
        self.mode = _Mode()

    def __enter__(self):
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self.mode.__exit__(*exc)

    def report(self) -> Dict[str, Any]:
        ops = dict(sorted(self.counts.items()))
        custom = {k: v for k, v in ops.items() if k.split(".", 1)[0] not in TORCH_NAMESPACES}
        return {"n_distinct_ops": len(ops), "ops": ops, "custom_ops": custom}


def _strict_error(e: BaseException) -> Dict[str, Any]:
    msg = str(e)
    return {"type": type(e).__name__, "message": msg[-2000:],
            "is_determinism_raise": "deterministic" in msg.lower(),
            "traceback": traceback.format_exc()[-4000:]}


def _env_record() -> Dict[str, Any]:
    import torch
    rec: Dict[str, Any] = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                           "cudnn": int(torch.backends.cudnn.version() or 0),
                           "python": sys.executable,
                           "env": {k: os.environ.get(k) for k in repro.ENV_KEYS}}
    try:
        rec["card"] = repro.card_identity(0)
    except Exception as e:  # noqa: BLE001 — informative only here
        rec["card"] = f"unreadable: {e}"
    rec["numerics"] = repro.torch_numerics_record()
    return rec


def _gpu_cotenants() -> List[str]:
    try:
        me = os.getpid()
        return [f"pid {p['pid']} {p.get('name')} {p.get('used_mib')} MiB"
                for p in repro.gpu_compute_processes() if p["pid"] != me]
    except Exception as e:  # noqa: BLE001
        return [f"unreadable: {e}"]


# ── DA3 ──────────────────────────────────────────────────────────────────────────────────────

def run_da3(session: Path, n_frames: int, process_res: int, repeat: int, model_id: str) -> Dict[str, Any]:
    """The pipeline's DA3 window inference (extract_da3_depth.run_windows' own calls) on
    ``n_frames`` real frames, ``1 + repeat`` passes: pass 0 under the op census, then ``repeat``
    compared passes. ≥ 3 frames exercise the reference-view selection (point 42)."""
    import extract_da3_depth as X
    out: Dict[str, Any] = {"model": "da3", "model_id": model_id, "process_res": int(process_res),
                           "cotenants_at_start": _gpu_cotenants()}
    out["numerics_requested"] = X._deterministic()          # strict, before CUDA (as the extractor)
    paths = [str(p) for p in _spread(_frames(session), n_frames)]
    out["frames"] = [Path(p).name for p in paths]
    import torch
    t0 = time.time()
    model, _ = X._load_model(model_id)
    out["load_s"] = round(time.time() - t0, 1)
    captured: Dict[str, Any] = {}
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "_apply_depth_alignment"):
        _orig_align = inner._apply_depth_alignment

        def _capture(output, metric_output):
            captured["mono"] = metric_output.depth.detach().float().cpu().numpy()
            return _orig_align(output, metric_output)

        inner._apply_depth_alignment = _capture
    import depth_anything_3.model.dinov2.vision_transformer as _vt
    _orig_select = _vt.select_reference_view

    def _select(x, *a, **k):
        b_idx = _orig_select(x, *a, **k)
        captured.setdefault("ref", []).extend(int(v) for v in b_idx.reshape(-1).tolist())
        return b_idx

    _vt.select_reference_view = _select

    def _one() -> Dict[str, Optional[str]]:
        captured.clear()
        torch.manual_seed(0)
        with torch.no_grad():
            pred = model.inference(paths, process_res=int(process_res))
        return {"depth": _sha(X._np(pred.depth).astype(np.float32)),
                "conf": _sha(X._np(pred.conf)),
                "extrinsics": _sha(X._np(pred.extrinsics)),
                "intrinsics": _sha(X._np(pred.intrinsics)),
                "depth_mono": _sha(captured.get("mono")),
                "ref_views": json.dumps(captured.get("ref", [])),
                "scale_factor": repr(None if pred.scale_factor is None else float(pred.scale_factor)),
                "is_metric": repr(int(pred.is_metric))}

    return _passes(out, _one, repeat)


# ── PointDiT ─────────────────────────────────────────────────────────────────────────────────

def run_pointdit(session: Path, n_frames: int, repeat: int) -> Dict[str, Any]:
    """F6's PointDiT calls (precision.mono_detail.run_stage: plan_tiles → runner.depth per tile,
    at the configured tile size, steps, model and seed) on every tile of ``n_frames`` real frames."""
    import cv2
    from precision.config import load_precision_config
    from precision.mono_detail import plan_tiles
    from precision.pointdit_runner import PointDiTRunner
    repro.ensure_cublas_workspace()                 # f6_bend runs under the step env: same here
    out: Dict[str, Any] = {"model": "pointdit", "cotenants_at_start": _gpu_cotenants()}
    md = load_precision_config().mono_detail
    out["config"] = {"model": md.model, "steps": int(md.steps), "seed": int(md.seed),
                     "tile_px": int(md.tile_px), "tile_overlap_frac": float(md.tile_overlap_frac),
                     "context_scale": float(md.context_scale), "enabled": bool(md.enabled)}
    runner = PointDiTRunner(md)
    if not runner.device.startswith("cuda"):
        raise SystemExit("PointDiT smoke test needs CUDA (the pipeline runs it on the GPU)")
    t0 = time.time()
    runner.load()
    out["load_s"] = round(time.time() - t0, 1)
    images = []
    for p in _spread(_frames(session), n_frames):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"unreadable frame {p}")
        images.append((p.name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    out["frames"] = [n for n, _ in images]
    plans = {n: plan_tiles(img.shape[0], img.shape[1], int(md.tile_px), float(md.tile_overlap_frac),
                           float(md.context_scale)) for n, img in images}
    out["tiles"] = {n: [[t.y0, t.x0, t.h, t.w, t.run_h, t.run_w] for t in ts] for n, ts in plans.items()}

    def _one() -> Dict[str, Optional[str]]:
        res: Dict[str, Optional[str]] = {}
        for n, img in images:
            for k, t in enumerate(plans[n]):
                z, v = runner.depth(img[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w], size=(t.run_h, t.run_w))
                res[f"{n}#t{k}.z"] = _sha(z)
                res[f"{n}#t{k}.valid"] = _sha(v)
        return res

    out = _passes(out, _one, repeat)
    try:
        out["footprint"] = runner.footprint()
    except Exception as e:  # noqa: BLE001
        out["footprint"] = f"unreadable: {e}"
    return out


# ── shared ───────────────────────────────────────────────────────────────────────────────────

def _passes(out: Dict[str, Any], one, repeat: int) -> Dict[str, Any]:
    """Pass 0 under the op census (strict), then ``repeat`` plain passes; every pass's hashes."""
    out["environment"] = _env_record()
    hashes: List[Dict[str, Optional[str]]] = []
    census = _Census()
    try:
        t0 = time.time()
        with census:
            hashes.append(one())
        out["census_pass_s"] = round(time.time() - t0, 1)
        for _ in range(int(repeat)):
            t0 = time.time()
            hashes.append(one())
            out.setdefault("pass_s", []).append(round(time.time() - t0, 1))
        out["status"] = "ok"
    except Exception as e:  # noqa: BLE001 — the point of the test: report what raised
        out["status"] = "raised"
        out["error"] = _strict_error(e)
    out["census"] = census.report()
    out["passes"] = hashes
    compared = hashes[1:]
    out["in_process_identical"] = (len(compared) >= 2 and all(h == compared[0] for h in compared[1:])) \
        if out["status"] == "ok" else False
    out["census_pass_matches"] = bool(compared) and hashes[0] == compared[0]
    if compared and not out["in_process_identical"]:
        out["differs"] = sorted(k for k in compared[0] if any(h.get(k) != compared[0][k] for h in compared[1:]))
    return out


def _write(path: Optional[str], doc: Dict[str, Any]) -> None:
    txt = json.dumps(doc, indent=1, sort_keys=True, default=str)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(txt)
    else:
        print(txt)


def _child_env() -> Dict[str, str]:
    """The chain's step environment (cuBLAS workspace, hash seed, BLAS threads and kernel pin) with
    the one Hugging Face cache, offline (da3_weights.hf_env — every DA3 subprocess's)."""
    import da3_weights
    from precision.runner import step_env
    return da3_weights.hf_env(step_env(threads=int(os.environ.get("DET_SMOKE_THREADS", "8"))))


def run_all(args) -> int:
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = _child_env()
    verdict: Dict[str, Any] = {"session": str(args.session), "children": {}}
    ok_all = True
    for model in args.models:
        runs = []
        for k in range(2):
            out = outdir / f"{model}_proc{k}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), model, "--session", str(args.session),
                   "--repeat", str(args.repeat), "--out", str(out)]
            if model == "da3":
                cmd += ["--frames", str(args.da3_frames), "--process-res", str(args.process_res),
                        "--model-id", args.model_id]
            else:
                cmd += ["--frames", str(args.pointdit_frames)]
            print(f"[smoke] {model} process {k + 1}/2: {' '.join(cmd[2:])}", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, cwd=str(_SERVER_DIR), env=env)
            doc = json.loads(out.read_text()) if out.exists() else {"status": "no_report", "returncode": r.returncode}
            doc["returncode"] = r.returncode
            doc["wall_s"] = round(time.time() - t0, 1)
            runs.append(doc)
            print(f"[smoke] {model} process {k + 1}/2: status {doc.get('status')}, in-process identical "
                  f"{doc.get('in_process_identical')}, {doc.get('wall_s')} s", flush=True)
            if doc.get("status") != "ok":
                break
        ok = (len(runs) == 2 and all(d.get("status") == "ok" and d.get("in_process_identical") for d in runs)
              and runs[0]["passes"][1:] == runs[1]["passes"][1:])
        cross = None
        if len(runs) == 2 and runs[0].get("passes") and runs[1].get("passes"):
            a, b = runs[0]["passes"][-1], runs[1]["passes"][-1]
            cross = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        verdict["children"][model] = {
            "deterministic": bool(ok),
            "statuses": [d.get("status") for d in runs],
            "error": next((d.get("error") for d in runs if d.get("error")), None),
            "in_process_identical": [d.get("in_process_identical") for d in runs],
            "cross_process_differs": cross,
            "custom_ops": runs[0].get("census", {}).get("custom_ops") if runs else None,
            "n_distinct_ops": runs[0].get("census", {}).get("n_distinct_ops") if runs else None,
            "cotenants_at_start": [d.get("cotenants_at_start") for d in runs],
        }
        ok_all &= bool(ok)
    verdict["deterministic"] = ok_all
    _write(str(outdir / "verdict.json"), verdict)
    print(json.dumps({m: {k: v for k, v in d.items() if k != "custom_ops"} | {"custom_ops": sorted((d.get("custom_ops") or {}))}
                      for m, d in verdict["children"].items()}, indent=1, default=str))
    print(f"[smoke] verdict: {'DETERMINISTIC' if ok_all else 'NOT DETERMINISTIC / RAISED'} → {outdir / 'verdict.json'}")
    return 0 if ok_all else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("da3")
    d.add_argument("--session", required=True)
    d.add_argument("--frames", type=int, default=4, help="frames in the window (>= 3 exercises the reference view)")
    d.add_argument("--process-res", type=int, default=840)
    d.add_argument("--model-id", default=DA3_MODEL)
    d.add_argument("--repeat", type=int, default=2)
    d.add_argument("--out")
    p = sub.add_parser("pointdit")
    p.add_argument("--session", required=True)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--out")
    a = sub.add_parser("all")
    a.add_argument("--session", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--models", nargs="+", default=["da3", "pointdit"], choices=["da3", "pointdit"])
    a.add_argument("--da3-frames", type=int, default=4)
    a.add_argument("--pointdit-frames", type=int, default=1)
    a.add_argument("--process-res", type=int, default=840)
    a.add_argument("--model-id", default=DA3_MODEL)
    a.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args(argv)
    if args.cmd == "all":
        return run_all(args)
    session = Path(args.session).resolve()
    if args.cmd == "da3":
        doc = run_da3(session, args.frames, args.process_res, args.repeat, args.model_id)
    else:
        doc = run_pointdit(session, args.frames, args.repeat)
    _write(args.out, doc)
    return 0 if doc.get("status") == "ok" and doc.get("in_process_identical") else 1


if __name__ == "__main__":
    sys.exit(main())
