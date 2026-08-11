"""
Multi-view geometric consistency filter — pre-TSDF (precision task, Phase B).
================================================================================
The TSDF integrates each keyframe's omega depth as-is; per-frame surface noise
is averaged INTO the mesh. This module runs the standard MVS fusion check
BEFORE integration, without touching poses: every pixel of every keyframe depth
map (the exact source the TSDF consumes: the aligned chunk depth) is projected
into its N nearest neighbour keyframes at the FINAL metric poses; the pixel is
valid only if at least `min_consistent_views` neighbours agree on the surface
within `tau_rel` relative depth. Inconsistent pixels are masked out of the TSDF
(or, flagged variant, replaced by the median of the consistent estimates).

Geometry per neighbour j of frame i (all torch, GPU when available):
    X   = unproject(depth_i, K_i) → world via T_i (c2w)
    z_ij = depth of X in camera j;  (u,v) = project(X, K_j)
    d_j  = bilinear sample of depth_j at (u,v)
    vote_j = |z_ij − d_j| / d_j ≤ tau_rel        (in-bounds, both depths valid)
    est_j  = depth_i · d_j / z_ij                (j's surface pulled along i's ray)

Outputs (resume-aware, fail-fast):
    output/mv_consistency/frame_<num>.npz   {valid: bool HxW, votes: uint8,
                                             depth: float32 (median variant only)}
    output/mv_consistency/report.json       params + per-frame/global discard %

TELEMETRY IS A HEALTH SIGNAL: a global discard above `warn_discard_pct`
(default 40%) means POSES OR SCALE ARE WRONG, not that the filter is working —
it is logged as a strong warning and flagged in the report. Never silent.

The TSDF applies the masks via _resolve_mapanything_depth(mv_dir=…) — see
segmentation/tsdf_export.py. Enabled by config.yaml `tsdf.mv_consistency`
(default OFF until it wins its A/B, per the evidence rule).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("MVConsistency")

MV_DIRNAME = "mv_consistency"
REPORT_NAME = "report.json"


def _neighbor_indices(centres: np.ndarray, n_neighbors: int) -> List[np.ndarray]:
    """N nearest keyframes by CAMERA-CENTRE distance (excluding self). Spatial —
    not temporal — proximity, so revisits/overlap passes also cross-check each
    other, which is where the double surfaces live."""
    n = len(centres)
    k = min(n_neighbors, n - 1)
    out = []
    for i in range(n):
        d = np.linalg.norm(centres - centres[i], axis=1)
        d[i] = np.inf
        out.append(np.argsort(d, kind="stable")[:k])
    return out


def compute_frame_consistency(depth_i, K_i, T_i, nbr_depths, nbr_Ks, nbr_Ts,
                              tau_rel: float, device=None):
    """Votes + per-neighbour ray estimates for ONE keyframe against its
    neighbours. Returns (votes uint8 HxW, est float32 (n_nbr,HxW-shaped), valid_i
    bool HxW). est is NaN where the neighbour cast no vote."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    di = torch.as_tensor(depth_i, dtype=torch.float32, device=dev)
    H, W = di.shape
    valid_i = torch.isfinite(di) & (di > 1e-4)

    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                          torch.arange(W, device=dev, dtype=torch.float32),
                          indexing="ij")
    Ki = torch.as_tensor(K_i, dtype=torch.float32, device=dev)
    x = (u - Ki[0, 2]) / Ki[0, 0] * di
    y = (v - Ki[1, 2]) / Ki[1, 1] * di
    ones = torch.ones_like(di)
    Ti = torch.as_tensor(T_i, dtype=torch.float32, device=dev)
    Pw = Ti @ torch.stack([x, y, di, ones]).reshape(4, -1)     # world, (4, HW)

    votes = torch.zeros(H * W, dtype=torch.uint8, device=dev)
    ests = []
    for dj, Kj, Tj in zip(nbr_depths, nbr_Ks, nbr_Ts):
        djt = torch.as_tensor(dj, dtype=torch.float32, device=dev)
        Hj, Wj = djt.shape
        Tj_inv = torch.linalg.inv(torch.as_tensor(Tj, dtype=torch.float32, device=dev))
        Pc = (Tj_inv @ Pw)[:3]                                  # camera j
        z = Pc[2]
        Kjt = torch.as_tensor(Kj, dtype=torch.float32, device=dev)
        uu = Kjt[0, 0] * Pc[0] / z.clamp(min=1e-6) + Kjt[0, 2]
        vv = Kjt[1, 1] * Pc[1] / z.clamp(min=1e-6) + Kjt[1, 2]
        # bilinear sample of neighbour depth (grid_sample expects [-1,1] coords)
        gx = (uu / max(Wj - 1, 1)) * 2.0 - 1.0
        gy = (vv / max(Hj - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
        samp = torch.nn.functional.grid_sample(
            djt.reshape(1, 1, Hj, Wj), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True).reshape(-1)
        ok = (z > 1e-4) & (samp > 1e-4) & torch.isfinite(samp) \
            & (uu >= 0) & (uu <= Wj - 1) & (vv >= 0) & (vv <= Hj - 1) \
            & valid_i.reshape(-1)
        rel = torch.abs(z - samp) / samp.clamp(min=1e-6)
        vote = ok & (rel <= tau_rel)
        votes += vote.to(torch.uint8)
        est = di.reshape(-1) * samp / z.clamp(min=1e-6)
        est = torch.where(vote, est, torch.full_like(est, float("nan")))
        ests.append(est)
    est_stack = (torch.stack(ests) if ests
                 else torch.empty(0, H * W, device=dev)).reshape(-1, H, W)
    return (votes.reshape(H, W).cpu().numpy(),
            est_stack.cpu().numpy(),
            valid_i.cpu().numpy())


def _load_keyframe_data(output_dir: Path, log) -> Dict[int, dict]:
    """All keyframes' (depth, K, T c2w) from the SAME sources the TSDF uses:
    aligned chunk depth via tsdf_export's loader + final camera_poses.txt.
    Depths live in CPU RAM (~1 MB/frame at the omega grid — hundreds of
    keyframes fit easily); the GPU only ever holds one frame + its neighbours."""
    from segmentation.tsdf_export import _resolve_mapanything_depth
    from reconstruction.scale_align import _read_poses

    src = _resolve_mapanything_depth(output_dir, conf_percentile=None)
    if src is None:
        raise RuntimeError("mv_consistency: no MapAnything/omega chunk depth found "
                           "(maplong_run/_tmp_results_aligned) — nothing to filter")
    loader, _hw = src
    lines, nums, _ = _read_poses(output_dir)
    mats = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
            for ln in lines if len(ln.split()) == 16]
    frames: Dict[int, dict] = {}
    for n, T in zip(nums, mats):
        d = loader(int(n))
        if d is None or d.get("K") is None:
            continue
        frames[int(n)] = {"depth": np.asarray(d["depth"], np.float32),
                          "K": np.asarray(d["K"], np.float64), "T": T}
    if len(frames) < 3:
        raise RuntimeError(f"mv_consistency: only {len(frames)} keyframes with "
                           f"depth+pose+intrinsics — cannot cross-check")
    log(f"loaded {len(frames)} keyframes (depth+K from chunk npy, poses from "
        f"camera_poses.txt)")
    return frames


def run(output_dir: Path, n_neighbors: int = 4, min_consistent_views: int = 2,
        tau_rel: float = 0.02, replace_median: bool = False,
        warn_discard_pct: float = 40.0, log=None, device=None,
        _frames_override: Optional[Dict[int, dict]] = None) -> dict:
    """Generate per-keyframe consistency masks + report. Resume-aware: if the
    report exists with identical params and every npz is present, it is a no-op.
    Fail-fast on missing inputs. Returns the report dict."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir = Path(output_dir)
    mv_dir = output_dir / MV_DIRNAME
    params = {"n_neighbors": int(n_neighbors),
              "min_consistent_views": int(min_consistent_views),
              "tau_rel": float(tau_rel), "replace_median": bool(replace_median)}

    rp = mv_dir / REPORT_NAME
    if rp.exists():
        try:
            old = json.loads(rp.read_text())
        except Exception:
            old = None
        if old and old.get("params") == params and all(
                (mv_dir / f"frame_{n}.npz").exists() for n in old.get("frames", {})):
            _log(f"masks already on disk with identical params — skipped "
                 f"(global discard {old.get('global_discard_pct', 0):.1f}%)")
            return old

    t0 = time.time()
    frames = _frames_override or _load_keyframe_data(output_dir, _log)
    nums = sorted(frames.keys())
    centres = np.stack([frames[n]["T"][:3, 3] for n in nums])
    nbrs = _neighbor_indices(centres, n_neighbors)

    mv_dir.mkdir(parents=True, exist_ok=True)
    per_frame: Dict[str, dict] = {}
    tot_valid = 0
    tot_drop = 0
    for i, n in enumerate(nums):
        f = frames[n]
        js = [nums[j] for j in nbrs[i]]
        votes, est, valid_i = compute_frame_consistency(
            f["depth"], f["K"], f["T"],
            [frames[j]["depth"] for j in js],
            [frames[j]["K"] for j in js],
            [frames[j]["T"] for j in js],
            tau_rel, device=device)
        passed = valid_i & (votes >= min_consistent_views)
        n_valid = int(valid_i.sum())
        n_drop = int((valid_i & ~passed).sum())
        tot_valid += n_valid
        tot_drop += n_drop
        arrays = {"valid": passed, "votes": votes}
        if replace_median:
            # median over the CONSISTENT estimates + the pixel's own depth —
            # only where the pixel passed (masked-out pixels never integrate)
            stack = np.concatenate([f["depth"][None], est], 0)
            med = np.nanmedian(stack, axis=0).astype(np.float32)
            arrays["depth"] = np.where(passed, med, f["depth"]).astype(np.float32)
        np.savez_compressed(mv_dir / f"frame_{n}.npz", **arrays)
        per_frame[str(n)] = {"valid_px": n_valid, "dropped_px": n_drop,
                             "discard_pct": round(100.0 * n_drop / max(n_valid, 1), 2),
                             "neighbors": [int(j) for j in js]}

    global_pct = 100.0 * tot_drop / max(tot_valid, 1)
    warned = global_pct > float(warn_discard_pct)
    report = {"version": 1, "params": params, "n_keyframes": len(nums),
              "frames": per_frame, "global_discard_pct": round(global_pct, 2),
              "warn_discard_pct": float(warn_discard_pct),
              "pose_scale_warning": bool(warned),
              "elapsed_s": round(time.time() - t0, 1)}
    rp.write_text(json.dumps(report, indent=2))
    _log(f"{len(nums)} keyframes filtered in {report['elapsed_s']}s — global "
         f"discard {global_pct:.1f}% "
         f"({'median-replace variant' if replace_median else 'mask variant'})")
    if warned:
        _log(f"⚠️  STRONG WARNING: {global_pct:.1f}% of valid pixels are "
             f"multi-view INCONSISTENT (> {warn_discard_pct:g}%). This level of "
             f"disagreement indicates BAD POSES OR SCALE, not a working filter — "
             f"inspect the reconstruction before trusting this mesh.")
    return report
