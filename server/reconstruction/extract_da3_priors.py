#!/usr/bin/env python3
"""
Extract per-frame PRIORS (metric depth + c2w poses) from a DA3 streaming run into
the source-agnostic layout that run_vipe_with_priors.py consumes. We take ONLY
DA3's depth + poses — NOT its chunks/point clouds.

DA3 streaming final outputs (after alignment + loop closure):
  - poses: camera_poses.txt — one row-major 4x4 C2W per processed frame (verified
    c2w via npz_output_process.read_camera_poses), aligned 1:1 with the depth.
  - depth, by availability:
      A) output/da3_full/{stem}_depth.npy   (only when LiDAR is injected — Stray)
      B) results_output/frame_<idx>.npz      (plain DA3, save_depth_conf_result),
         the 'depth' key, sequence-indexed and aligned 1:1 with camera_poses.txt.
  - frame numbers (to map to the ViPE frames): camera_frames.txt sidecar →
    da3_full stems → else the sorted frames_dir stems (assumes DA3 processed every
    frame 1:1; the len assert below catches any subsampling).

Output (vipe_priors/):  depth/{frame:06d}.npy + poses.txt + frames.txt

Runs in the da3 conda env (numpy only).

    python extract_da3_priors.py --da3-run <out>/da3_run --frames-dir <frames> \
        --out <out>/vipe_priors
"""
import argparse
import os
from pathlib import Path

import numpy as np

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def _find_depth(da3_run: Path, out_dir: Path):
    """Return (kind, sorted_list). Prefer per-frame *_depth.npy; else results npz."""
    for d in (out_dir / "da3_full", da3_run / "da3_full", da3_run):
        files = sorted(d.glob("*_depth.npy"))
        if files:
            return "npy", files
    for d in (da3_run / "results_output", out_dir / "results_output", da3_run):
        files = sorted(d.glob("frame_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        if files:
            return "npz", files
    return None, []


def _load_depth(kind: str, p: Path):
    if kind == "npy":
        return np.load(p).astype(np.float32)
    z = np.load(p)
    return z["depth"].astype(np.float32) if "depth" in z else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da3-run", required=True, type=Path)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out_dir = args.da3_run.parent  # = .../output

    poses_file = _first_existing([args.da3_run / "camera_poses.txt",
                                  out_dir / "camera_poses.txt"])
    if poses_file is None:
        raise FileNotFoundError(f"[da3-priors] camera_poses.txt not found under {args.da3_run}")
    pose_lines = [ln for ln in poses_file.read_text().splitlines() if ln.strip()]

    kind, depth_files = _find_depth(args.da3_run, out_dir)
    if not depth_files:
        raise FileNotFoundError(
            f"[da3-priors] no per-frame depth found (da3_full/*_depth.npy or "
            f"results_output/frame_*.npz) under {args.da3_run} / {out_dir}")

    # frame numbers, INDEX-ALIGNED with camera_poses.txt (= processing order)
    cframes = _first_existing([args.da3_run / "camera_frames.txt",
                               out_dir / "camera_frames.txt"])
    if cframes is not None:
        frame_nums = [int(x) for x in cframes.read_text().split()]
    elif kind == "npy":
        frame_nums = [int(p.stem.replace("_depth", "")) for p in depth_files]
    else:
        # plain-DA3 results npz: assume DA3 processed every input frame 1:1
        frame_nums = [int(p.stem) for p in
                      sorted({f for e in _EXTS for f in
                              list(args.frames_dir.glob(f"*{e}")) +
                              list(args.frames_dir.glob(f"*{e.upper()}"))})]

    # FAIL LOUD on misalignment rather than silently corrupting the reconstruction
    if not (len(pose_lines) == len(depth_files) == len(frame_nums)):
        raise RuntimeError(
            f"[da3-priors] misaligned priors: {len(pose_lines)} poses, "
            f"{len(depth_files)} depth maps, {len(frame_nums)} frame numbers — "
            f"refusing to emit (would corrupt the pose↔depth↔frame mapping). "
            f"depth kind={kind}")

    out = args.out
    (out / "depth").mkdir(parents=True, exist_ok=True)
    for old in (out / "depth").glob("*.npy"):   # clear stale priors
        old.unlink()

    # SYMLINK npy depth (no duplication); copy npz-extracted depth.
    with open(out / "poses.txt", "w") as fp, open(out / "frames.txt", "w") as ff:
        for line, fr, dpath in zip(pose_lines, frame_nums, depth_files):
            fp.write(line.strip() + "\n")
            ff.write(f"{fr}\n")
            link = out / "depth" / f"{fr:06d}.npy"
            if kind == "npy":
                try:
                    os.symlink(os.path.abspath(dpath), link)
                    continue
                except FileExistsError:
                    continue
                except OSError:
                    pass
            d = _load_depth(kind, dpath)
            if d is not None:
                np.save(link, d.astype(np.float32))

    print(f"[da3-priors] {len(frame_nums)} priors ({kind}) → {out} "
          f"(poses.txt + frames.txt + depth/*.npy)", flush=True)


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        print("[da3-priors] FATAL — traceback follows:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
