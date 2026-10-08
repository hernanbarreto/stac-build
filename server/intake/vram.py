"""DA3 windows sized by the card — at NATIVE resolution (USER 2026-10-04: "debe ser nativa"),
from the COMMITTED card table (docs/plan_determinismo.md points 5, 6, 13, 25, 41).

zaragoza 2026-10-04 (1920x1080): the focal probe ran DA3 NESTED-GIANT on 16 frames at process_res
1932 in one multi-view window and died in CUDA OOM — 56.8 GB in the process, vLLM loading beside
it. MEASURED the same day on this card (A100 80 GB): 2 frames at 1932 → 25.8 GiB peak. pccr's
frames (464x832 → 840) hold ~2 000 tokens each; a 1080p frame holds ~10 800. DA3's footprint grows
with the TOKENS of the window, so the resolution stays native and the WINDOW shrinks.

THE SIZE IS A FUNCTION OF COMMITTED NUMBERS (2026-10-07). It used to be floor() of a 2-frame
calibration window measured per session and cached in output/intake/da3_vram.json (no software
version in its key): pccr's I3 window was floor(26.06) — a 2-frame peak 12 MiB higher after a
torch / cuDNN / driver update would have made it 25 and changed the windows, the walk, the chunk
plan and the anchors together. Now:
  1. the card is read through torch + its board memory from nvidia-smi (repro.card_identity →
     repro.card_key, the card MODEL: name | board MiB | sm), never an 'unknown' sentinel (a slow
     nvidia-smi used to turn it into 'unknown' and change the decision; a failing one now FAILS);
  2. its entry in server/card_table.json gives the card's total memory and DA3's weights + peak
     of a 2-frame window at this process_res, measured ONCE on that card by the calibration CLI
     (``python -m intake.vram --calibrate``) and committed — no run measures or writes it;
  3. ``window_sizing``: the largest window (≤ the configured one) whose predicted peak
     (weights + per_token × tokens_per_frame × N) fits the card's total less ``margin_frac`` —
     the same formula, over constants. A card or resolution with no entry FAILS naming the CLI;
  4. a window that still does not fit exits extract_da3_depth.py with OOM_EXIT and the caller
     FAILS (point 4) — it is never halved: another window size is another walk.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

DA3_PATCH = 14
LOG_TAG = "[intake.vram]"
# extract_da3_depth.py's exit codes (defined there; a test keeps the two equal)
OOM_EXIT = 3                      # a window does not fit the card
REF_VIEW_EXIT = 5                 # a regenerated window picked another reference view
IDENTITY_EXIT = 6                 # the extracting process is not the planned DA3 environment
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


def max_frames_per_window(fp: Dict[str, object], tokens_per_frame_: int, total_gb: float,
                          margin_frac: float, at_most: int) -> int:
    """The largest window (≤ ``at_most``) whose predicted peak fits ``total_gb`` less the margin
    (``fp``: weights_gb, per_token_gb — GiB)."""
    budget = float(total_gb) * (1 - float(margin_frac)) - float(fp["weights_gb"])
    per_frame = float(fp["per_token_gb"]) * float(tokens_per_frame_)
    if per_frame <= 0:
        return int(at_most)
    n = int(math.floor(budget / per_frame))
    return max(1, min(int(at_most), n))


def window_sizing(model_id: str, process_res: int, native_wh: Sequence[int], requested: int,
                  margin_frac: float, *, card_key: str,
                  table_path: Optional[Path] = None) -> Dict[str, object]:
    """The window size for ``model_id`` at ``process_res`` on the card ``card_key``, from the
    committed card table, with every number behind it (recorded in windows.json): the card's
    total, the footprint and its provenance, the predicted peak and the HEADROOM — how far the
    exact quotient sits above the integer chosen (pccr: 26.06 → 26, 0.06 frames)."""
    import card_table
    ent = card_table.card_entry(card_key, table_path)
    total = card_table.sizing_total_gib(ent)
    fp = card_table.da3_footprint(ent, model_id, int(process_res), card_key)
    tpf = tokens_per_frame(int(native_wh[0]), int(native_wh[1]), int(process_res))
    req = int(requested)
    budget = total * (1 - float(margin_frac)) - fp["weights_gib"]
    per_frame = fp["per_token_gib"] * tpf
    exact = budget / per_frame if per_frame > 0 else float("inf")
    n = max_frames_per_window({"weights_gb": fp["weights_gib"], "per_token_gb": fp["per_token_gib"]},
                              tpf, total, margin_frac, req)
    return {"window_frames": int(n), "requested": req, "limited_by": "card" if n < req else "requested",
            "card": str(card_key), "model_id": str(model_id), "process_res": int(process_res),
            "tokens_per_frame": int(tpf), "card_total_gib": total, "margin_frac": float(margin_frac),
            "weights_gib": fp["weights_gib"], "per_token_gib": fp["per_token_gib"],
            "frames_exact": exact, "headroom_frames": (exact - n) if n < req else None,
            "predicted_peak_gib": fp["weights_gib"] + per_frame * n,
            "footprint_provenance": fp["provenance"], "source": "card_table"}


def window_size(model_id: str, process_res: int, native_wh: Sequence[int], requested: int,
                margin_frac: float, *, card_key: str, log: Callable = print,
                table_path: Optional[Path] = None) -> Dict[str, object]:
    """:func:`window_sizing`, declared in the log. DETERMINISTIC (USER 2026-10-05, "debe ser
    determinista"): committed constants of the card, never the memory FREE at that moment (pccr
    2408 measured its walk over 221 windows of 18 keyframes on one run — an orphan SAM3 held
    21 GB — and 153 of 25 on the next) nor a fresh measurement (2026-10-07)."""
    s = window_sizing(model_id, process_res, native_wh, requested, margin_frac, card_key=card_key,
                      table_path=table_path)
    n, req = int(s["window_frames"]), int(s["requested"])
    if n < req:
        log(f"{LOG_TAG} window {req} → {n} frame(s): {s['card']} ({s['card_total_gib']:.1f} GiB "
            f"total, card table), {s['tokens_per_frame']:,} tokens/frame at process_res "
            f"{process_res} (native), predicted peak {s['predicted_peak_gib']:.1f} GiB with a "
            f"{float(margin_frac):.0%} margin; {s['frames_exact']:.3f} would fit "
            f"({s['headroom_frames']:.3f} frame(s) of headroom)")
    else:
        log(f"{LOG_TAG} window {req} frame(s) fits: predicted peak "
            f"{s['predicted_peak_gib']:.1f} GiB of {s['card']}'s {s['card_total_gib']:.1f} GiB")
    return s


# ── the calibration CLI: the ONLY writer of a card's DA3 entry ─────────────────────────────

def calibrate(session_dir: Path, model_id: str, process_res, python: str,
              calibration_frames: int, log: Callable = print,
              table_path: Optional[Path] = None) -> Dict[str, object]:
    """Measure DA3's weights and 2-frame peak at ``process_res`` on THIS card (exclusive GPU) and
    write its card-table entry. Run by hand on a new card or resolution; the entry is committed."""
    import repro
    import card_table
    import da3_weights
    from intake.walk import da3_process_res, keyframe_files
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    res = da3_process_res(process_res, frames_dir)
    try:
        frames = keyframe_files(frames_dir)
    except Exception:  # noqa: BLE001 — any frame of the scan measures the footprint
        from intake.quality import list_frames
        frames = [p.name for p in list_frames(frames_dir)]
    if len(frames) < 2:
        raise VramError(f"{frames_dir} holds {len(frames)} frame(s) — the calibration needs two")
    import cv2
    img = cv2.imread(str(frames_dir / frames[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise VramError(f"cannot read {frames_dir / frames[0]}")
    native_wh = (int(img.shape[1]), int(img.shape[0]))
    tpf = tokens_per_frame(native_wh[0], native_wh[1], res)
    n = max(2, min(int(calibration_frames), len(frames)))
    idx = sorted({int(round(k * (len(frames) - 1) / max(n - 1, 1))) for k in range(n)})
    window = [str(frames_dir / frames[i]) for i in idx]
    card_table.require_one_visible_card()                    # point 78: THE device, one card
    # the card MODEL key and its board memory (nvidia-smi memory.total of this uuid) — the same
    # identity every run looks the entry up with; card_identity raises when nvidia-smi does not
    # list the card
    ident = repro.card_identity(0)
    key = repro.card_key(ident)
    repro.require_exclusive_gpu(log=log)
    wdir = session_dir / "output" / "intake" / "da3_vram_probe"
    wdir.mkdir(parents=True, exist_ok=True)
    for p in wdir.glob("window_*.npz"):
        p.unlink()
    spec_path = wdir / "windows.json"
    spec_path.write_text(json.dumps({"windows": [window], "process_res": int(res), "model_id": model_id}))
    server_dir = Path(__file__).resolve().parent.parent
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--image_dir", str(frames_dir),
           "--output_dir", str(wdir), "--model", model_id, "--process_res", str(int(res)),
           "--windows_json", str(spec_path)]
    log(f"{LOG_TAG} calibrating DA3's footprint on {key}: {len(window)} frame(s) at process_res "
        f"{res} ({tpf:,} tokens/frame)")
    env = da3_weights.hf_env(repro.deterministic_env())
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    lines = [ln.strip() for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if ln.strip()]
    for ln in lines:
        log(ln)
    if proc.returncode == OOM_EXIT:
        raise VramError(f"even {len(window)} frame(s) at process_res {res} do not fit {key}")
    if proc.returncode != 0:
        raise VramError(f"the calibration window failed (exit {proc.returncode}): {' | '.join(lines[-3:])}")
    fp = parse_footprint(lines)
    for p in wdir.glob("window_*.npz"):          # the footprint is measured; the depth is dead
        p.unlink()
    meas = {"weights_gib": fp["weights_gb"], "peak_gib": fp["peak_gb"], "frames": len(window),
            "tokens_per_frame": int(tpf),
            "provenance": (f"intake.vram --calibrate on {session_dir.name} "
                           f"({native_wh[0]}x{native_wh[1]}, process_res {res}), "
                           f"{time.strftime('%Y-%m-%d')}: {len(window)}-frame window, weights "
                           f"{fp['weights_gb']:.2f} GiB, peak {fp['peak_gb']:.2f} GiB")}
    path = card_table.write_da3_entry(key, int(ident["memory_total_mib"]), model_id, int(res),
                                      meas, table_path)
    log(f"{LOG_TAG} {key}: DA3 {model_id} @ {res} → {path} (commit it)")
    return meas


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m intake.vram",
                                 description="Measure DA3's footprint on THIS card and write its "
                                             "entry in server/card_table.json (then commit it).")
    ap.add_argument("--calibrate", action="store_true", required=True)
    ap.add_argument("--session", required=True, help="a scan whose frames are measured")
    ap.add_argument("--process-res", default="native")
    ap.add_argument("--model", default=None, help="default: intake.parallax.focal_probe_model")
    ap.add_argument("--frames", type=int, default=None,
                    help="calibration window size (default intake.parallax.vram_calibration_frames)")
    a = ap.parse_args(argv)
    from config import cfg as raw
    from intake.config import load_intake_config
    p = load_intake_config(raw).parallax
    calibrate(Path(a.session), str(a.model or p.focal_probe_model),
              a.process_res if a.process_res == "native" else int(a.process_res), sys.executable,
              int(a.frames or p.vram_calibration_frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
