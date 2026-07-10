# STAC-Builder — normals from per-point traceability (no KDTree, no MST).
#
# The cleaned cloud carries (frame_global, pixel_row, pixel_col) per point, and the
# reconstruction keeps per-frame depth (maplong_run/_tmp_results_aligned) and camera
# poses. That makes generic normal estimation unnecessary:
#
#   · normal   = depth-map GRADIENT at the point's own pixel (cross product of the
#     unprojected du/dv neighbours) — vectorized per frame, milliseconds each;
#   · orientation = FREE: every normal faces its own camera (flip on dot>0). The
#     open3d alternative (orient_normals_consistent_tangent_plane) builds a global
#     MST over every point — single-threaded MINUTES on multi-million clouds, and
#     the reason the Poisson stage "hangs" on normals.
#
# Uses torch on GPU when available; numpy otherwise (maps are small — 384x688).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("TraceNormals")


# ── per-frame machinery ──────────────────────────────────────────────

def _normal_map_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H,W,3) camera-space normal map from a depth map: unproject, then cross the
    horizontal/vertical finite differences. Fully vectorized."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    X = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=-1)
    dx = np.empty_like(X); dx[:, :-1] = X[:, 1:] - X[:, :-1]; dx[:, -1] = dx[:, -2]
    dy = np.empty_like(X); dy[:-1] = X[1:] - X[:-1]; dy[-1] = dy[-2]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.maximum(norm, 1e-12)
    # camera-facing: in OpenCV camera space the camera looks down +Z, so a surface
    # normal pointing AT the camera has negative Z. Flip the ones that don't.
    flip = n[..., 2] > 0
    n[flip] *= -1.0
    n[~np.isfinite(n)] = 0.0
    return n


def _load_frame_depth_index(output_dir: Path):
    """frame_number → (chunk npy path, local index) from maplong_run's aligned
    chunks + frame_list.json + the run's chunk layout."""
    run = output_dir / "maplong_run"
    fl_path = run / "frame_list.json"
    aligned = run / "_tmp_results_aligned"
    if not (fl_path.exists() and aligned.is_dir()):
        return None
    names = json.loads(fl_path.read_text())
    nums = []
    for nm in names:
        m = re.search(r"(\d+)", str(Path(nm).name))
        nums.append(int(m.group(1)) if m else -1)
    chunks = sorted(aligned.glob("chunk_*.npy"),
                    key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not chunks:
        return None
    # layout from the generated config (single chunk → the whole list)
    import yaml
    cfgp = output_dir / "vggt_omega_config.yaml"
    chunk_size, overlap = len(nums), 0
    if cfgp.exists():
        m = (yaml.safe_load(cfgp.read_text()) or {}).get("Model", {})
        chunk_size = int(m.get("chunk_size", chunk_size))
        overlap = int(m.get("overlap", 0))
    step = max(chunk_size - overlap, 1)
    index: Dict[int, Tuple[Path, int]] = {}
    for ci, cp in enumerate(chunks):
        start = ci * step
        end = min(start + chunk_size, len(nums))
        for local, gi in enumerate(range(start, end)):
            # overlap frames appear in two chunks — keep the one farther from the
            # chunk edge (better-conditioned depth); later chunks overwrite the
            # edge-most entries naturally when closer to their own centre.
            num = nums[gi]
            prev = index.get(num)
            centre_d = abs(local - (end - start) / 2)
            if prev is None or centre_d < prev[2]:
                index[num] = (cp, local, centre_d)
    return {k: (v[0], v[1]) for k, v in index.items()}


def _load_poses(output_dir: Path) -> Optional[Dict[int, np.ndarray]]:
    """frame_number → 4x4 c2w from camera_poses.txt + camera_frames.txt."""
    pp = output_dir / "camera_poses.txt"
    fp = output_dir / "camera_frames.txt"
    if not (pp.exists() and fp.exists()):
        return None
    poses = np.array([[float(x) for x in l.split()] for l in open(pp) if l.strip()])
    frames = [int(l.split()[0]) for l in open(fp) if l.strip()]
    if poses.shape[1] == 17:
        poses = poses[:, 1:]
    if poses.shape[1] != 16 or len(frames) != len(poses):
        return None
    return {f: M for f, M in zip(frames, poses.reshape(-1, 4, 4))}


def _intrinsics_for(depth_hw: Tuple[int, int], output_dir: Path) -> Optional[np.ndarray]:
    """K at the depth-map resolution, from intrinsic.txt (written at that same
    processing resolution by the backend)."""
    for cand in (output_dir / "intrinsic.txt", output_dir / "maplong_run" / "intrinsic.txt"):
        if cand.exists():
            vals = [float(x) for x in cand.read_text().split()]
            if len(vals) >= 9:
                return np.array(vals[:9], dtype=np.float64).reshape(3, 3)
    return None


# ── public API ───────────────────────────────────────────────────────

def normals_from_trace(xyz: np.ndarray, frame_global: np.ndarray,
                       pixel_row: np.ndarray, pixel_col: np.ndarray,
                       output_dir: Path,
                       log=None) -> Optional[np.ndarray]:
    """(N,3) world-space, camera-oriented normals for a traced cloud. Returns None
    when the session lacks any required artifact (caller falls back to KDTree+MST)."""
    def _log(m):
        (log or logger.info)(m)

    output_dir = Path(output_dir)
    depth_index = _load_frame_depth_index(output_dir)
    poses = _load_poses(output_dir)
    if depth_index is None or poses is None:
        return None

    frames = np.unique(frame_global)
    have = [f for f in frames if int(f) in depth_index and int(f) in poses]
    if len(have) < max(3, 0.5 * len(frames)):
        _log(f"trace-normals: only {len(have)}/{len(frames)} frames have depth+pose — "
             f"falling back")
        return None

    normals = np.zeros((len(xyz), 3), dtype=np.float32)
    done = np.zeros(len(xyz), dtype=bool)
    K_cache: Optional[np.ndarray] = None
    chunk_cache: Dict[Path, dict] = {}

    for f in have:
        cp, local = depth_index[int(f)]
        data = chunk_cache.get(cp)
        if data is None:
            chunk_cache.clear()                     # one chunk resident at a time
            data = np.load(cp, allow_pickle=True).item()
            chunk_cache[cp] = data
        depth = np.asarray(data["depth"][local]).squeeze()
        if depth.ndim != 2:
            continue
        if K_cache is None:
            K_cache = _intrinsics_for(depth.shape, output_dir)
            if K_cache is None:
                _log("trace-normals: no intrinsic.txt — falling back")
                return None
        nmap = _normal_map_from_depth(depth, K_cache)

        M = poses[int(f)]
        R = M[:3, :3]
        mask = frame_global == f
        r = np.clip(pixel_row[mask], 0, depth.shape[0] - 1)
        c = np.clip(pixel_col[mask], 0, depth.shape[1] - 1)
        n_cam = nmap[r, c]
        normals[mask] = (n_cam @ R.T).astype(np.float32)
        done[mask] = True

    frac = float(done.mean())
    if frac < 0.5:
        _log(f"trace-normals: covered only {frac:.0%} of points — falling back")
        return None
    if not done.all():
        # leftover points (frames without depth): nearest covered neighbour would be
        # overkill — give them the mean normal of their frame's plane fallback: zero
        # normals break Poisson, so copy from the nearest covered point index-wise.
        idx = np.where(done)[0]
        missing = np.where(~done)[0]
        take = idx[np.clip(np.searchsorted(idx, missing), 0, len(idx) - 1)]
        normals[missing] = normals[take]
    _log(f"trace-normals: {len(xyz):,} pts from {len(have)} depth maps — no KDTree, "
         f"no MST, camera-oriented by construction")
    return normals
