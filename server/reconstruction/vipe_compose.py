#!/usr/bin/env python3
"""
Pose-driven cloud composer for the ViPE SLAM pipeline. Back-projects each frame's
metric ViPE depth (vipe_depth/, already × global g) with its ViPE c2w pose
(translations also × g) into world space, emitting chunk PLYs + chunk_*_origins.npz
(frame_global, pixel_row, pixel_col, confidence) 1:1 with the points — the SAME
schema map_worker._generate_origins / CloudComPy expect, so the downstream merge +
cleaned_cloud + TSDF are unchanged. ViPE depth is multi-view consistent (co-optimized
with the poses) so it fuses cleanly, unlike per-image DA3 isolated depth.

Replaces DA3's world_points chunks when pose_source == "vipe". frame_global is the
ORIGINAL frame number, not a keyframe index.

Runs in the da3 env (numpy + open3d; vipe_depth is plain npz). No EXR/ViPE.
"""
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import open3d as o3d


def _load_rgb_at(frames_dir: Path, frame: int, H: int, W: int) -> Optional[np.ndarray]:
    p = frames_dir / f"{frame:06d}.jpg"
    if not p.exists():
        return None
    try:
        import cv2
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return cv2.resize(im, (W, H)).astype(np.uint8)
    except Exception:
        return None


def compose_chunks_from_vipe(
    output_dir: Path,
    frames_dir: Path,
    depth_dir: Path,                     # vipe_depth (npz: depth(metric, ×g), intrinsics)
    pose_map: Dict[int, np.ndarray],     # frame# -> scaled c2w (DA3 metric)
    depth_min: float = 0.3,
    depth_max: float = 8.0,
    edge_thresh: float = 0.04,
    conf_percentile: float = 25.0,       # drop per-frame bottom % conf (only if npz has conf)
    voxel_size: float = 0.010,           # voxel-dedup per chunk (m). 10mm: balance
                                         # density vs keeping chunks light (5mm=~360M
                                         # choked CloudCompy; 10mm → ~4x fewer ~90M).
                                         # Cloud is for viewing/TSDF-mask (TSDF uses
                                         # depth), so this density is fine.
    chunk_size: int = 120,               # frames per chunk PLY
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> int:
    """Compose chunk_*.ply + chunk_*_origins.npz from metric ViPE depth + scaled ViPE
    poses. Returns the number of chunks written. Points/origins are 1:1 from the same mask."""
    output_dir = Path(output_dir); frames_dir = Path(frames_dir); depth_dir = Path(depth_dir)
    # only frames with both a depth map and a pose
    frames = sorted(f for f in pose_map
                    if (depth_dir / f"{int(f):06d}.npz").exists())
    if not frames:
        raise RuntimeError("[vipe-compose] no frames with depth + pose")

    # clear any prior chunk artifacts (no leftovers / no duplicates)
    for old in output_dir.glob("chunk_*"):
        old.unlink()

    n_chunks = 0
    n_total = 0
    for ci in range(0, len(frames), chunk_size):
        grp = frames[ci:ci + chunk_size]
        pts, cols, fg, pr, pc, cf = [], [], [], [], [], []
        for fr in grp:
            z = np.load(depth_dir / f"{fr:06d}.npz")
            d = z["depth"].astype(np.float32); K = z["intrinsics"]
            conf = z["conf"].astype(np.float32) if "conf" in z else None
            H, W = d.shape
            m = (d > depth_min) & (d < depth_max) & np.isfinite(d)
            # flying-pixel cull at depth discontinuities (same as TSDF)
            dd = np.where(m, d, 0.0)
            gx = np.abs(np.diff(dd, axis=1, prepend=dd[:, :1]))
            gy = np.abs(np.diff(dd, axis=0, prepend=dd[:1, :]))
            m &= (gx < edge_thresh) & (gy < edge_thresh)
            # confidence filter: drop the per-frame bottom `conf_percentile`%
            if conf is not None and conf_percentile > 0 and m.any():
                thr = float(np.percentile(conf[m], conf_percentile))
                m &= (conf >= thr)
            if not m.any():
                continue
            vv, uu = np.nonzero(m); zz = d[vv, uu].astype(np.float64)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            Xc = np.stack([(uu - cx) / fx * zz, (vv - cy) / fy * zz, zz], 1)
            c2w = pose_map[fr]
            Xw = (c2w[:3, :3] @ Xc.T).T + c2w[:3, 3]
            pts.append(Xw.astype(np.float32))
            rgb = _load_rgb_at(frames_dir, fr, H, W)
            cols.append((rgb[vv, uu] if rgb is not None
                         else np.full((len(vv), 3), 180, np.uint8)))
            fg.append(np.full(len(vv), fr, np.int32))
            pr.append(vv.astype(np.int16)); pc.append(uu.astype(np.int16))
            cf.append((conf[vv, uu] if conf is not None
                       else np.ones(len(vv), np.float32)).astype(np.float32))
        if not pts:
            continue
        P = np.concatenate(pts); C = np.concatenate(cols)
        FG = np.concatenate(fg); PR = np.concatenate(pr)
        PC = np.concatenate(pc); CF = np.concatenate(cf)
        # voxel dedup: keep one point per voxel cell. np.unique on the voxel key
        # gives a single index per cell, applied to points AND origins so the
        # 1:1 traceability (frame_global, pixel, confidence) is preserved.
        if voxel_size > 0:
            keys = np.floor(P / voxel_size).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            idx.sort()
            P, C = P[idx], C[idx]
            FG, PR, PC, CF = FG[idx], PR[idx], PC[idx], CF[idx]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
        cloud.colors = o3d.utility.Vector3dVector(C.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(output_dir / f"chunk_{n_chunks:03d}.ply"), cloud)
        # origins 1:1 with the PLY points (same order, same voxel-dedup index)
        np.savez_compressed(
            output_dir / f"chunk_{n_chunks:03d}_origins.npz",
            frame_global=FG, pixel_row=PR, pixel_col=PC, confidence=CF,
            scaled_resolution=np.array([H, W], np.int32),
        )
        with open(output_dir / f"chunk_{n_chunks:03d}_meta.json", "w") as f:
            json.dump({"backend": "vipe", "pose_source": "vipe",
                       "frames": [int(x) for x in grp],
                       "n_points": int(len(P))}, f)
        n_total += len(P); n_chunks += 1
        if progress_cb:
            progress_cb(n_chunks, f"composed chunk {n_chunks} ({n_total:,} pts)")
    return n_chunks
