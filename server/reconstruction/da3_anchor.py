"""Isolated per-frame DA3 metric depth for scale-anchor frames — NO streaming.

The streaming pipeline chains poses across consecutive frames; anchor frames
are seconds apart, the chain breaks and none of its machinery is needed: the
scale is a per-pixel depth RATIO, poses do not participate. Runs
``extract_da3_depth.py --per_frame`` in the given Python (the ``da3`` env) and
converts its output to the layout every scale consumer reads:
``<output_dir>/da3_run/results_output/frame_<num>.npz`` (depth + conf [+
intrinsics]).

Shared by workers/map_worker.py (chunk anchors) and, through
``Model.metric_lock.anchor_extract``, by the VGGT-Long fork (loop-bridge
anchors, claude_stac.txt §4.1).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


def extract_anchor_depths(frames_dir, output_dir, anchor_files: Iterable[str],
                          model_id: str, python: str, log: Callable[[str], None] = print,
                          check_cancel: Optional[Callable[[], bool]] = None) -> int:
    """Extract DA3 depth for ``anchor_files`` (basenames inside frames_dir).
    Returns the number of npz written; raises on extractor failure or when
    nothing could be produced. Returns 0 without error only when the caller
    was cancelled (check_cancel)."""
    frames_dir, output_dir = Path(frames_dir), Path(output_dir)
    anchor_files = list(anchor_files)
    server_dir = Path(__file__).resolve().parent.parent
    tmp = output_dir / "_da3_anchor_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in anchor_files:
        src = frames_dir / f
        if src.exists():
            os.symlink(str(src), str(tmp / f))
    raw = output_dir / "da3_run" / "anchor_raw"
    # the extractor writes straight into this directory and does not create it:
    # every run until 2026-09-19 inherited it from a previous one, so the first
    # reconstruction of a session with no da3_run/ died on the first np.save
    raw.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"),
           "--image_dir", str(tmp), "--output_dir", str(raw),
           "--model", str(model_id), "--per_frame"]
    log(f"DA3 anchor: isolated per-frame depth on {len(anchor_files)} frames "
        f"({model_id}) — no streaming")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        line = line.strip()
        if line:
            log(line)
        if check_cancel is not None and check_cancel():
            proc.terminate()
            return 0
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"DA3 anchor extraction exited with code {proc.returncode}")

    ro = output_dir / "da3_run" / "results_output"
    ro.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in anchor_files:
        stem = os.path.splitext(f)[0]
        dp, cp = raw / f"{stem}_depth.npy", raw / f"{stem}_conf.npy"
        if not dp.exists():
            continue
        num = int("".join(ch for ch in stem if ch.isdigit()))
        arrays = {"depth": np.load(dp).astype(np.float32)}
        if cp.exists():
            arrays["conf"] = np.load(cp).astype(np.float32)
        kp = raw / f"{stem}_intrinsics.npy"
        if kp.exists():
            arrays["intrinsics"] = np.load(kp).astype(np.float64)
        np.savez_compressed(ro / f"frame_{num}.npz", **arrays)
        n += 1
    shutil.rmtree(tmp, ignore_errors=True)
    if n == 0:
        raise RuntimeError("DA3 anchor produced no depth maps — scale cannot be estimated")
    log(f"DA3 anchor: {n} metric depth maps → {ro}")
    return n
