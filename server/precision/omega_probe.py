"""P1 — Omega's resolution probe (claude_stac.txt §4-F3).

The same window of keyframes (the central ``window_frames`` of the session) is
inferred by Omega at every probed resolution; at each one, every pair of frames in
the window is measured on the EXACT surfaces both see — the geometry of one frame
projected into the other's camera, compared with the other's own depth there
(``loop_utils.metric_lock.depth_pair_samples``) — and the pair's median relative
depth mismatch is read before any fit (``pair_depth_relation``). A resolution
whose frames agree better about where the surfaces are wins the comparison.

It REPORTS (``output/omega_probe.json``); the value is the user's to set in
``reconstruction.vggtomega.resolution`` / ``.mode``. The held-out reprojection per
resolution joins the report once the correspondence (F4) and refinement (F5)
stages exist. GPU; Omega runs on the keyframes only.

USER 2026-10-07 (docs/plan_determinismo.md point 40): kept as a REPORT, deterministic —
every inference inside ``repro.deterministic_torch`` (strict deterministic kernels, TF32
off), on the card only (no CPU fallback), the card checked free before the step (point 4)
and the environment recorded in the report (point 37); the wall-clock seconds and the
peak VRAM of each resolution go to ``omega_probe.timing.json``, outside the report.

CLI: ``python -m precision.omega_probe --session <dir>``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

PROBE_NAME = "omega_probe.json"
TIMING_NAME = "omega_probe.timing.json"
PROBE_VERSION = 1
# the torch seed of the probe's deterministic block: Omega's inference draws no random number
# (eval mode, no dropout) — the seed only pins torch's RNG state, as deterministic_torch requires
PROBE_TORCH_SEED = 0
PROVENANCE = "tool_measured"
LOG_TAG = "[omega-probe]"

Infer = Callable[[List[str], int, str], Dict[str, np.ndarray]]


class ProbeError(RuntimeError):
    """A structural impossibility of the probe — with the exact reason."""


def _vendor_path() -> None:
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)


def probe_window(files: Sequence[str], window_frames: int) -> List[str]:
    """The central ``window_frames`` keyframes (all of them when fewer)."""
    files = list(files)
    if len(files) <= window_frames:
        return files
    start = (len(files) - int(window_frames)) // 2
    return files[start:start + int(window_frames)]


def window_disagreement(pred: Dict[str, np.ndarray], pair_samples: int) -> Dict[str, Any]:
    """Every ordered pair (i < j) of the window: the median relative depth mismatch
    of the surfaces both see. ``pred``: world_points [S,H,W,3], conf [S,H,W],
    c2w [S,4,4], K [S,3,3] (one resolution's inference)."""
    _vendor_path()
    from loop_utils.metric_lock import depth_pair_samples, pair_depth_relation
    wp, cf, c2w, K = pred["world_points"], pred["conf"], pred["c2w"], pred["K"]
    S = len(wp)
    w2c = np.linalg.inv(np.asarray(c2w, np.float64))
    mism, n_starved, n_broken = [], 0, 0
    for i in range(S):
        for j in range(i + 1, S):
            smp = depth_pair_samples(wp[i], cf[i], wp[j], cf[j], w2c[j], K[j],
                                     max_samples=int(pair_samples), seed=i * S + j)
            if smp is None:
                n_starved += 1
                continue
            rel = pair_depth_relation(*smp)
            if rel is None:
                n_broken += 1
                continue
            mism.append(rel[2])
    m = np.asarray(mism, np.float64)
    return {"n_pairs": S * (S - 1) // 2, "n_measured": int(m.size),
            "n_starved": n_starved, "n_broken": n_broken,
            "median_rel_mismatch": float(np.median(m)) if m.size else None,
            "p90_rel_mismatch": float(np.percentile(m, 90)) if m.size else None}


def omega_infer(session_dir: Path) -> Infer:
    """The production Omega (vendor/VGGT-Long VGGTOmegaAdapter, the session's own
    vendor config), loaded once; each call sets the resolution and mode."""
    _vendor_path()
    import torch
    from config import cfg as raw_cfg
    from workers.map_worker import _build_vggtomega_config
    from base_models.vggtomega_adapter import VGGTOmegaAdapter
    if not torch.cuda.is_available():
        # no CPU fallback (other kernels, another product — point 12's rule, applied to F3)
        raise ProbeError("torch sees no CUDA device — Omega's probe runs on the card, never on "
                         "the CPU")
    vcfg = _build_vggtomega_config(raw_cfg, Path(session_dir) / "frames")
    ad = VGGTOmegaAdapter(vcfg, device="cuda")
    ad.load()

    def infer(paths: List[str], resolution: int, mode: str) -> Dict[str, np.ndarray]:
        from repro import deterministic_torch
        ad.image_resolution, ad.preproc_mode = int(resolution), str(mode)
        with deterministic_torch(PROBE_TORCH_SEED) as num:
            out = ad.infer_chunk(paths)
        infer.numerics = num
        f = lambda t: t.detach().float().cpu().numpy()[0]           # noqa: E731
        conf = out["world_points_conf"]
        return {"world_points": f(out["world_points"]), "conf": f(conf),
                "c2w": f(out["extrinsic"]), "K": f(out["intrinsic"])}
    return infer


def run_probe(session_dir: Path, pcfg, infer: Optional[Infer] = None,
              log: Callable = print) -> Dict[str, Any]:
    """Probe every configured resolution on the central window; write the report."""
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    sel = frames_dir / "selected_frames.json"
    if not sel.exists():
        raise ProbeError(f"{sel} does not exist — Omega's probe windows the KEYFRAMES")
    doc = json.loads(sel.read_text())
    files = sorted(doc.get("selected_files") or [], key=lambda f: int(Path(f).stem))
    if len(files) < 2:
        raise ProbeError(f"{sel} lists {len(files)} keyframe(s) — a pair is the minimum")
    window = probe_window(files, pcfg.window_frames)
    paths = [str(frames_dir / f) for f in window]
    on_card = infer is None
    infer = infer or omega_infer(session_dir)
    results, timing = [], []
    for res in pcfg.resolutions:
        t0 = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        pred = infer(paths, int(res), pcfg.mode)
        dt = time.time() - t0
        m = window_disagreement(pred, pcfg.pair_samples)
        vram = None
        if torch is not None and torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / 1024 ** 3
        rec = {"resolution": int(res), "mode": pcfg.mode,
               "grid_hw": [int(pred["world_points"].shape[1]), int(pred["world_points"].shape[2])],
               **m}
        results.append(rec)
        # time and memory pressure are not the measurement (point 36): they go to the timing file
        timing.append({"resolution": int(res), "seconds": round(dt, 2), "peak_vram_gb": vram})
        log(f"{LOG_TAG} {res} ({pcfg.mode}, grid {rec['grid_hw'][0]}x{rec['grid_hw'][1]}): "
            f"median pair mismatch "
            + (f"{rec['median_rel_mismatch'] * 100:.2f} %" if rec["median_rel_mismatch"] is not None
               else "n/a")
            + f" over {rec['n_measured']}/{rec['n_pairs']} pair(s), {dt:.1f} s")
    ranked = [r for r in results if r["median_rel_mismatch"] is not None]
    best = min(ranked, key=lambda r: r["median_rel_mismatch"])["resolution"] if ranked else None
    from intake.quality import read_session_epochs
    from repro import environment_record
    report = {"version": PROBE_VERSION, "provenance": PROVENANCE,
              **read_session_epochs(session_dir),
              # point 37: what the probe's numbers depend on beyond their inputs — no clock
              "environment": environment_record(gpu=on_card),
              "numerics": getattr(infer, "numerics", None),
              "params": {"resolutions": list(pcfg.resolutions), "mode": pcfg.mode,
                         "window_frames": pcfg.window_frames,
                         "pair_samples": pcfg.pair_samples},
              "window": [int(Path(f).stem) for f in window],
              "results": results, "lowest_mismatch_resolution": best,
              "decides": "nothing — the user sets reconstruction.vggtomega.resolution / .mode"}
    out = session_dir / "output" / PROBE_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    (out.parent / TIMING_NAME).write_text(json.dumps({"resolutions": timing}, indent=1))
    log(f"{LOG_TAG} lowest mismatch at {best} → {out} (report only)")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.omega_probe",
                                 description="Omega resolution probe on the central window.")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    pc = load_precision_config().omega.resolution_probe
    if not pc.enabled:
        print(f"{LOG_TAG} reconstruction.precision.omega.resolution_probe.enabled is false")
        return 0
    # the card is this step's alone (point 4), checked before torch touches CUDA — also when the
    # step is launched by hand; the cuBLAS workspace is pinned before CUDA initialises
    from repro import ensure_cublas_workspace, require_exclusive_gpu
    require_exclusive_gpu(log=print)
    ensure_cublas_workspace()
    run_probe(Path(args.session), pc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
