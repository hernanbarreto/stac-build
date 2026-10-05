"""DA3 windows sized by the card — at NATIVE resolution (USER 2026-10-04: "debe ser nativa").

zaragoza 2026-10-04 (1920x1080): the focal probe ran DA3 NESTED-GIANT on 16 frames at process_res
1932 in one multi-view window and died in CUDA OOM — 56.8 GB in the process, vLLM loading beside
it. MEASURED the same day on this card (A100 80 GB): 2 frames at 1932 → 25.8 GiB peak. pccr's
frames (464x832 → 840) hold ~2 000 tokens each; a 1080p frame holds ~10 800. DA3's footprint grows
with the TOKENS of the window, so the resolution stays native and the WINDOW shrinks.

The pipeline asks the card, every session, with no number of its own:
  1. ``footprint``: ONE calibration window of ``calibration_frames`` frames at the session's own
     process_res (extract_da3_depth.py prints the weights' allocation after the load and the
     window's peak) → GiB of weights and GiB per token, cached in output/intake/da3_vram.json
     for the same model / resolution / card.
  2. ``max_frames_per_window``: the largest window whose predicted peak
     (weights + per_token × tokens_per_frame × N) fits the card's free memory less ``margin_frac``.
  3. A window that still does not fit exits extract_da3_depth.py with code OOM_EXIT; the caller
     halves the window and says so (the safety net, never the plan).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

DA3_PATCH = 14
LOG_TAG = "[intake.vram]"
CACHE_NAME = "da3_vram.json"
OOM_EXIT = 3                      # extract_da3_depth.py's exit code when a window does not fit
_PEAK_RE = re.compile(r"peak ([0-9.]+) GiB allocated")
_WEIGHTS_RE = re.compile(r"model loaded: ([0-9.]+) GiB allocated")


class VramError(RuntimeError):
    pass


def da3_grid(native_w: int, native_h: int, process_res: int) -> tuple:
    """(grid_w, grid_h) DA3 runs at: the long side resized to ``process_res``, the short side by
    the aspect, both rounded to the patch."""
    long_side = max(native_w, native_h)
    s = float(process_res) / float(long_side)
    w = int(round(native_w * s / DA3_PATCH)) * DA3_PATCH
    h = int(round(native_h * s / DA3_PATCH)) * DA3_PATCH
    return max(w, DA3_PATCH), max(h, DA3_PATCH)


def tokens_per_frame(native_w: int, native_h: int, process_res: int) -> int:
    w, h = da3_grid(native_w, native_h, process_res)
    return (w // DA3_PATCH) * (h // DA3_PATCH)


def free_vram_gb() -> Optional[float]:
    """The card's FREE memory now (nvidia-smi), None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def card_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def parse_footprint(lines: Sequence[str]) -> Dict[str, float]:
    """The weights' allocation and the window's peak from extract_da3_depth.py's lines."""
    weights = peak = None
    for ln in lines:
        m = _WEIGHTS_RE.search(ln)
        if m:
            weights = float(m.group(1))
        m = _PEAK_RE.search(ln)
        if m:
            peak = float(m.group(1))
    if weights is None or peak is None:
        raise VramError("the calibration window printed no footprint (model loaded / peak lines)")
    return {"weights_gb": weights, "peak_gb": peak}


def footprint(session_dir: Path, frames: Sequence[str], frames_dir: Path, model_id: str, process_res: int,
              native_wh: Sequence[int], python: str, calibration_frames: int, log: Callable = print,
              cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, object]:
    """GiB of weights and GiB per token of ``model_id`` at ``process_res`` on THIS card, measured on
    one window of ``calibration_frames`` frames (reused from output/intake/da3_vram.json while the
    model, the resolution and the card are the same)."""
    session_dir = Path(session_dir)
    cache = session_dir / "output" / "intake" / CACHE_NAME
    card = card_name()
    tpf = tokens_per_frame(int(native_wh[0]), int(native_wh[1]), int(process_res))
    key = {"model_id": model_id, "process_res": int(process_res), "card": card, "tokens_per_frame": tpf,
           "calibration_frames": int(calibration_frames)}
    if cache.exists():
        try:
            doc = json.loads(cache.read_text())
            if doc.get("key") == key:
                log(f"{LOG_TAG} footprint reused: {doc['weights_gb']:.1f} GiB weights + "
                    f"{doc['per_token_gb'] * 1e3:.3f} MiB/token ({card})")
                return doc
        except (OSError, ValueError):
            pass
    n = max(2, min(int(calibration_frames), len(frames)))
    idx = [int(round(k * (len(frames) - 1) / max(n - 1, 1))) for k in range(n)]
    window = [str(Path(frames_dir) / frames[i]) for i in sorted(set(idx))]
    wdir = session_dir / "output" / "intake" / "da3_vram_probe"
    wdir.mkdir(parents=True, exist_ok=True)
    for p in wdir.glob("window_*.npz"):
        p.unlink()
    spec_path = wdir / "windows.json"
    spec_path.write_text(json.dumps({"windows": [window], "process_res": int(process_res), "model_id": model_id}))
    server_dir = Path(__file__).resolve().parent.parent
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--image_dir", str(frames_dir),
           "--output_dir", str(wdir), "--model", model_id, "--process_res", str(int(process_res)),
           "--windows_json", str(spec_path)]
    log(f"{LOG_TAG} calibrating DA3's footprint on this card: {len(window)} frame(s) at process_res "
        f"{process_res} ({tpf:,} tokens/frame) — the window size follows from it")
    t0 = time.time()
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines: List[str] = []
    for line in proc.stdout:
        if line.strip():
            lines.append(line.strip())
        if cancelled is not None and cancelled():
            proc.terminate()
            raise VramError("cancelled")
    proc.wait()
    if proc.returncode == OOM_EXIT:
        raise VramError(f"even {len(window)} frame(s) at process_res {process_res} do not fit this card "
                        f"({card}) — the native resolution cannot be run here")
    if proc.returncode != 0:
        tail = " | ".join(lines[-3:])
        raise VramError(f"the calibration window failed (exit {proc.returncode}): {tail}")
    fp = parse_footprint(lines)
    for p in wdir.glob("window_*.npz"):          # the footprint is measured; the depth is dead
        p.unlink()
    tokens = tpf * len(window)
    per_token = max(fp["peak_gb"] - fp["weights_gb"], 0.0) / float(tokens)
    doc = {"version": 1, "provenance": "tool_measured", "key": key, "weights_gb": fp["weights_gb"],
           "peak_gb": fp["peak_gb"], "frames": len(window), "tokens": tokens, "per_token_gb": per_token,
           "seconds": round(time.time() - t0, 1), "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(doc, indent=1))
    log(f"{LOG_TAG} measured: {fp['weights_gb']:.1f} GiB weights, peak {fp['peak_gb']:.1f} GiB on "
        f"{len(window)} frame(s) → {per_token * 1e3:.3f} MiB/token ({card}, {doc['seconds']} s)")
    return doc


def max_frames_per_window(fp: Dict[str, object], tokens_per_frame_: int, free_gb: float, margin_frac: float,
                          at_most: int) -> int:
    """The largest window (≤ ``at_most``) whose predicted peak fits ``free_gb`` less the margin."""
    budget = float(free_gb) * (1.0 - float(margin_frac)) - float(fp["weights_gb"])
    per_frame = float(fp["per_token_gb"]) * float(tokens_per_frame_)
    if per_frame <= 0:
        return int(at_most)
    n = int(math.floor(budget / per_frame))
    return max(1, min(int(at_most), n))


def window_size(session_dir: Path, frames: Sequence[str], frames_dir: Path, model_id: str, process_res: int,
                native_wh: Sequence[int], python: str, requested: int, calibration_frames: int,
                margin_frac: float, log: Callable = print,
                cancelled: Optional[Callable[[], bool]] = None) -> int:
    """``requested`` frames per window, or fewer when the card cannot hold them at this resolution —
    the decision and its reasons in the log."""
    fp = footprint(session_dir, frames, frames_dir, model_id, process_res, native_wh, python,
                   calibration_frames, log, cancelled)
    free = free_vram_gb()
    if free is None:
        raise VramError("the card's free memory cannot be read (nvidia-smi) — the window cannot be sized")
    tpf = int(fp["key"]["tokens_per_frame"])
    n = max_frames_per_window(fp, tpf, free, margin_frac, requested)
    pred = float(fp["weights_gb"]) + float(fp["per_token_gb"]) * tpf * n
    if n < int(requested):
        log(f"{LOG_TAG} window {requested} → {n} frame(s): {free:.1f} GiB free, {tpf:,} tokens/frame at "
            f"process_res {process_res} (native), predicted peak {pred:.1f} GiB with a {margin_frac:.0%} margin")
    else:
        log(f"{LOG_TAG} window {requested} frame(s) fits: predicted peak {pred:.1f} GiB of {free:.1f} GiB free")
    return n
