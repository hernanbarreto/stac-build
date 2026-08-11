#!/usr/bin/env python3
"""Offline DA3 anchor (re-)extraction for the scale A/B (precision task, Phase A).

Sessions whose da3_run/results_output was freed after the run cannot feed
tools/scale_ab.py. This tool re-extracts ISOLATED per-frame DA3 metric depth —
the exact conversion map_worker._run_da3_anchor performs — for:

  - K anchor keyframes evenly spread over selected_frames.json (default 32, a
    superset of the 12/24/32 counts the A/B sweeps), and
  - P PAIRS of CONSECUTIVE keyframes at even positions along the walk (default
    3) — the eval pairs for the depth-reprojection metric. Pair frames are
    excluded from the anchor subsets by scale_ab only via the evenly-spread
    pick; they exist primarily as evaluation data.

Run in the da3 env (the extractor's env):
    conda run -n da3 python tools/extract_da3_anchors.py <session_dir_or_output_dir> \
        [--count 32] [--pairs 3] [--model depth-anything/DA3NESTED-GIANT-LARGE-1.1]

READ-ONLY over everything except output/da3_run/ (which it repopulates).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _resolve_dirs(arg: Path):
    """Accept either the session dir (src_*/) or its output/ dir."""
    arg = arg.resolve()
    if arg.name == "output":
        return arg.parent, arg
    if (arg / "output").is_dir():
        return arg, arg / "output"
    sys.exit(f"FAIL: {arg} is neither a session dir (with output/) nor an output dir")


def pick_frames(sel_files, count, pairs):
    """Evenly-spread anchors + consecutive-keyframe eval pairs. Deterministic."""
    n = len(sel_files)
    chosen = []
    if 1 < count < n:
        idx = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
        chosen = [sel_files[int(i)] for i in idx]
    else:
        chosen = list(sel_files)
    pair_files = []
    if pairs > 0 and n >= 2:
        for k in range(pairs):
            i = round((k + 1) * (n - 2) / (pairs + 1))
            pair_files += [sel_files[int(i)], sel_files[int(i) + 1]]
    return sorted(set(chosen) | set(pair_files)), pair_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    a = ap.parse_args()

    session, output = _resolve_dirs(Path(a.session_dir))
    frames_dir = session / "frames"
    sel_path = frames_dir / "selected_frames.json"
    if not sel_path.exists():
        sys.exit(f"FAIL: {sel_path} not found")
    sel = json.loads(sel_path.read_text())
    sel_files = sel.get("selected_files", sel if isinstance(sel, list) else [])
    if not sel_files:
        sys.exit("FAIL: selected_frames.json has no selected_files")

    todo, pair_files = pick_frames(sel_files, a.count, a.pairs)
    ro = output / "da3_run" / "results_output"
    ro.mkdir(parents=True, exist_ok=True)
    missing = [f for f in todo
               if not (ro / f"frame_{int(os.path.splitext(f)[0])}.npz").exists()]
    print(f"anchors+pairs planned: {len(todo)} ({len(pair_files)} pair frames), "
          f"already on disk: {len(todo) - len(missing)}, to extract: {len(missing)}")
    if missing:
        tmp = output / "_da3_anchor_frames"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for f in missing:
            src = frames_dir / f
            if src.exists():
                os.symlink(str(src), str(tmp / f))
        raw = output / "da3_run" / "anchor_raw"
        server_dir = Path(__file__).resolve().parent.parent
        cmd = [sys.executable, str(server_dir / "extract_da3_depth.py"),
               "--image_dir", str(tmp), "--output_dir", str(raw),
               "--model", a.model, "--per_frame"]
        print("running:", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit(f"FAIL: extract_da3_depth.py exited with {rc}")
        n = 0
        for f in missing:
            stem = os.path.splitext(f)[0]
            dp, cp = raw / f"{stem}_depth.npy", raw / f"{stem}_conf.npy"
            if not dp.exists():
                continue
            arrays = {"depth": np.load(dp).astype(np.float32)}
            if cp.exists():
                arrays["conf"] = np.load(cp).astype(np.float32)
            kp = raw / f"{stem}_intrinsics.npy"
            if kp.exists():
                arrays["intrinsics"] = np.load(kp).astype(np.float64)
            np.savez_compressed(ro / f"frame_{int(stem)}.npz", **arrays)
            n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        if n == 0:
            sys.exit("FAIL: extraction produced no depth maps")
        print(f"extracted {n} DA3 depth maps → {ro}")
    pairs_json = output / "scale_ab_eval_pairs.json"
    pairs_json.write_text(json.dumps({"pair_files": pair_files}, indent=2))
    print(f"eval pairs recorded → {pairs_json}")


if __name__ == "__main__":
    main()
