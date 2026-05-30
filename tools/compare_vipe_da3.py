#!/usr/bin/env python3
"""
Compare ViPE vs DA3 camera trajectories (PoC for the robust-SLAM migration).

ViPE is run on the FULL continuous frame stream; this script samples its poses
at our keyframe indices and compares against DA3's chunk-based poses
(camera_poses.txt). It reports:

  1. SIM3 (Umeyama) alignment of ViPE → DA3 + ATE (how close the trajectories
     are overall, scale-resolved).
  2. Per-step discontinuity analysis on BOTH trajectories — the key question:
     does ViPE remove the chunk-handoff jumps (e.g. test1's kf79→80 = 3.7m)?

Usage:
    python tools/compare_vipe_da3.py \
        --vipe-out  /path/to/vipe_results \
        --da3-poses /path/.../output/da3_run/camera_poses.txt \
        --selected  /path/.../frames/selected_frames.json

ViPE output layout (read directly, no vipe import needed):
    <vipe-out>/pose/<name>.npz  with keys: inds [N], data [N,4,4] (SE3, c2w)
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


# ── Loaders ─────────────────────────────────────────────────────────

def load_da3_poses(path):
    """camera_poses.txt → C2W [N,4,4], keyed by keyframe-sequence index 0..N-1."""
    mats = []
    for line in open(path):
        v = line.split()
        if len(v) >= 16:
            mats.append(np.array([float(x) for x in v[:16]], dtype=np.float64).reshape(4, 4))
    return np.array(mats)


def load_vipe_poses(vipe_out):
    """Find pose/<name>.npz; return (inds [N], C2W [N,4,4]).

    ViPE `data` is an SE3 matrix array. Convention is normalized below by the
    SIM3 alignment, so c2w vs w2c does not affect ATE — but we try to return c2w
    (camera centers) for the discontinuity analysis (camera positions)."""
    cands = sorted(glob.glob(os.path.join(vipe_out, "pose", "*.npz")))
    if not cands:
        raise FileNotFoundError(f"No pose/*.npz under {vipe_out}")
    d = np.load(cands[0])
    inds = np.asarray(d["inds"]).astype(np.int64)
    data = np.asarray(d["data"], dtype=np.float64)
    if data.ndim == 3 and data.shape[1:] == (4, 4):
        mats = data
    elif data.ndim == 2 and data.shape[1] == 16:
        mats = data.reshape(-1, 4, 4)
    else:
        raise ValueError(f"Unexpected ViPE pose data shape {data.shape}")
    print(f"[vipe] {Path(cands[0]).name}: {len(inds)} poses, ind range "
          f"{inds.min()}..{inds.max()}")
    return inds, mats


def keyframe_original_numbers(selected_json):
    """selected_frames.json → sorted original frame numbers (e.g. 0,20,42,...)."""
    sf = json.load(open(selected_json))
    files = sf if isinstance(sf, list) else sf.get("selected_files", [])
    return sorted(int(Path(f).stem) for f in files)


# ── Geometry ────────────────────────────────────────────────────────

def umeyama_sim3(src, dst):
    """Similarity transform (s,R,t) mapping src→dst (Nx3 each). Returns aligned src."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    cov = Xd.T @ Xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return (s * (R @ src.T)).T + t


def step_stats(pos, label):
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    med = np.median(d); mad = np.median(np.abs(d - med)) * 1.4826
    thr = med + 6 * mad
    out = np.where(d > thr)[0]
    print(f"\n[{label}] pasos: median={med*1000:.1f}mm  p95={np.percentile(d,95)*1000:.1f}mm  "
          f"MAX={d.max()*1000:.0f}mm  |  outliers(>med+6MAD={thr*1000:.0f}mm): {len(out)}")
    for i in out[:8]:
        print(f"    kf {i}->{i+1}: {d[i]*1000:.0f}mm")
    return d, out


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vipe-out", required=True)
    ap.add_argument("--da3-poses", required=True)
    ap.add_argument("--selected", required=True)
    args = ap.parse_args()

    da3 = load_da3_poses(args.da3_poses)                 # [K,4,4] keyframe-index
    kf_orig = keyframe_original_numbers(args.selected)   # original frame numbers
    vinds, vmats = load_vipe_poses(args.vipe_out)
    vidx = {int(fi): i for i, fi in enumerate(vinds)}    # original frame -> row

    if len(da3) != len(kf_orig):
        print(f"⚠️ DA3 poses ({len(da3)}) != keyframes ({len(kf_orig)})")

    # Match ViPE poses at our keyframe original frame numbers.
    rows, da3_keep = [], []
    for i, on in enumerate(kf_orig):
        if on in vidx and i < len(da3):
            rows.append(vidx[on]); da3_keep.append(i)
    if not rows:
        print("❌ No overlap between ViPE inds and keyframe frame numbers — "
              "did ViPE run on the full frame stream?")
        return
    vpos = vmats[rows][:, :3, 3]
    dpos = da3[da3_keep][:, :3, 3]
    print(f"\nmatched {len(rows)}/{len(kf_orig)} keyframes")

    # SIM3 align ViPE → DA3, ATE.
    vpos_al = umeyama_sim3(vpos, dpos)
    ate = np.sqrt(((vpos_al - dpos) ** 2).sum(1)).mean()
    print(f"\n=== ATE (ViPE alineado a DA3, SIM3) = {ate*1000:.1f} mm (mean) / "
          f"{np.sqrt(((vpos_al-dpos)**2).sum(1)).max()*1000:.0f} mm (max) ===")

    # Discontinuity analysis on BOTH (the real question).
    step_stats(dpos, "DA3 (chunk-based)")
    step_stats(vpos, "ViPE (dense BA continuo)")
    print("\n→ Si ViPE NO tiene el salto grande que DA3 sí tiene, confirma que el "
          "SLAM continuo elimina las juntas de chunk.")


if __name__ == "__main__":
    main()
