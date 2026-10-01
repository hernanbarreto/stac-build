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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("TraceNormals")


# ── per-frame machinery ──────────────────────────────────────────────

def _one_sided_tangent(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(H,W,3) tangent along the image COLUMNS (u): per pixel, the forward or the
    backward difference of the unprojected points ``X``, whichever side lies on
    the pixel's own surface.

    The side is the one whose INVERSE-depth ``w`` second difference is smaller in
    magnitude. Inverse depth is affine in (u, v) on any plane, so that second
    difference is exactly 0 on a face and large across a crease or a depth step —
    a comparison of the two sides, no threshold. A plain forward difference gave
    the last pixel before a crease or an occluding edge the tangent of the OTHER
    surface (edge audit 2026-10-01 #10): a blended normal at every crease and a
    near-ray normal at every silhouette, the exact pixels the MLS normal gate is
    there to protect. Ties (a plane) keep the forward difference; the first column
    can only go forward, the last only backward.
    """
    H, W = w.shape
    tan = np.empty_like(X)
    if W < 2:
        tan[:] = 0.0
        return tan
    fwd = np.empty_like(X)
    fwd[:, :-1] = X[:, 1:] - X[:, :-1]
    fwd[:, -1] = fwd[:, -2]
    bwd = np.empty_like(X)
    bwd[:, 1:] = X[:, 1:] - X[:, :-1]
    bwd[:, 0] = bwd[:, 1]
    # |second difference| centred on each interior column j (1 … W-2); a side
    # whose centre falls outside, or touches a pixel with no depth, is +inf
    e_f = np.full((H, W), np.inf, dtype=np.float64)
    e_b = np.full((H, W), np.inf, dtype=np.float64)
    if W >= 3:
        with np.errstate(invalid="ignore"):
            s = np.abs(w[:, 2:].astype(np.float64) - 2.0 * w[:, 1:-1] + w[:, :-2])
        s = np.where(np.isfinite(s), s, np.inf)
        e_f[:, :-2] = s            # forward side of column c is centred on c + 1
        e_b[:, 2:] = s             # backward side of column c is centred on c - 1
    use_b = e_b < e_f
    use_b[:, -1] = True            # the last column has no forward neighbour
    tan[:] = np.where(use_b[..., None], bwd, fwd)
    return tan


def _normal_map_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H,W,3) camera-space normal map from a depth map: unproject, then cross the
    horizontal/vertical one-sided differences (``_one_sided_tangent``: the side on
    the pixel's own surface). Fully vectorized."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    z = depth.astype(np.float32)
    X = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(z > 0, 1.0 / z, np.nan)
    dx = _one_sided_tangent(X, w)
    dy = _one_sided_tangent(X.transpose(1, 0, 2), w.T).transpose(1, 0, 2)
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


def _load_frames(output_dir: Path) -> Optional[List[int]]:
    """The real frame number of each camera_poses.txt / intrinsic.txt row."""
    fp = output_dir / "camera_frames.txt"
    if not fp.exists():
        return None
    return [int(l.split()[0]) for l in open(fp) if l.strip()]


def _load_poses(output_dir: Path, poses_dir: Optional[Path] = None
                ) -> Optional[Dict[int, np.ndarray]]:
    """frame_number → 4x4 c2w from camera_poses.txt (in ``poses_dir`` when given —
    an epoch transaction stages its WARPED poses there) + the session's
    camera_frames.txt (``output_dir``; the transaction does not stage it, the
    keyframe set does not change)."""
    pp = Path(poses_dir or output_dir) / "camera_poses.txt"
    frames = _load_frames(output_dir)
    if not pp.exists() or frames is None:
        return None
    poses = np.array([[float(x) for x in l.split()] for l in open(pp) if l.strip()])
    if poses.shape[1] == 17:
        poses = poses[:, 1:]
    if poses.shape[1] != 16 or len(frames) != len(poses):
        return None
    return {f: M for f, M in zip(frames, poses.reshape(-1, 4, 4))}


def _read_intrinsic_rows(path: Path) -> np.ndarray:
    """(N,3,3) K per row of an intrinsic.txt. Two layouts exist:

    - ``fx fy cx cy`` per keyframe — what the vendor writes (vggt_long.py, one row
      per camera_poses.txt row; precision.camera.read_omega_intrinsics reads the
      same rows) and every session since;
    - a flattened 3x3 (legacy): 9 values on one row, or the matrix as 3 rows of 3.

    Anything else RAISES: a file that exists and cannot be read is not "no
    intrinsics". (This used to read the first 9 numbers of ANY file as one 3x3, so
    on the 4-column layout cy came out as the SECOND keyframe's fy — pccr: 363.6
    for 416, an 8° shear of every trace normal, edge audit 2026-10-01 #10.)
    """
    rows = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if rows.shape == (3, 3):
        return rows[None].copy()
    if rows.ndim == 2 and len(rows) and rows.shape[1] == 9:
        return rows.reshape(-1, 3, 3)
    if rows.ndim == 2 and len(rows) and rows.shape[1] == 4:
        K = np.zeros((len(rows), 3, 3), dtype=np.float64)
        K[:, 0, 0], K[:, 1, 1] = rows[:, 0], rows[:, 1]
        K[:, 0, 2], K[:, 1, 2] = rows[:, 2], rows[:, 3]
        K[:, 2, 2] = 1.0
        return K
    raise ValueError(f"{path}: rows of {rows.shape[1] if rows.ndim == 2 else '?'} "
                     f"value(s) — expected 'fx fy cx cy' rows or a 3x3 matrix")


def _intrinsics_for(output_dir: Path, frames: Sequence[int]
                    ) -> Optional[Dict[int, np.ndarray]]:
    """frame_number → K at the depth-map resolution, from intrinsic.txt (written at
    that same processing resolution by the backend, one row per keyframe in
    camera_frames.txt order; a single row — or the legacy 3x3 — serves every
    frame). None when the session has no intrinsic.txt, or when its rows cannot be
    attributed to the keyframes (a row count that is neither 1 nor one per frame):
    a K borrowed from another keyframe is the error this replaced (the omega focal
    varies per keyframe — pccr fx 354.7–400.3)."""
    for cand in (output_dir / "intrinsic.txt", output_dir / "maplong_run" / "intrinsic.txt"):
        if cand.exists():
            Ks = _read_intrinsic_rows(cand)
            if len(Ks) == 1:
                return {int(f): Ks[0] for f in frames}
            if len(Ks) == len(frames):
                return {int(f): K for f, K in zip(frames, Ks)}
            logger.warning("trace-normals: %s has %d rows for %d keyframes — the "
                           "intrinsics cannot be attributed to frames", cand,
                           len(Ks), len(frames))
            return None
    return None


# ── public API ───────────────────────────────────────────────────────

def normals_from_trace(xyz: np.ndarray, frame_global: np.ndarray,
                       pixel_row: np.ndarray, pixel_col: np.ndarray,
                       output_dir: Path,
                       log=None,
                       poses_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """(N,3) world-space, camera-oriented normals for a traced cloud. Returns None
    when the session lacks any required artifact (caller falls back to KDTree+MST).

    Each keyframe's depth is unprojected with ITS OWN intrinsic.txt row and rotated
    by its own pose. ``poses_dir``: where the camera_poses.txt that matches ``xyz``
    lives when it is not ``output_dir`` — an epoch transaction warps the cloud and
    stages the warped poses next to it, while the depth maps, intrinsics and frame
    list stay in the session; the pre-warp rotation would tilt every normal by its
    keyframe's correction."""
    def _log(m):
        (log or logger.info)(m)

    output_dir = Path(output_dir)
    depth_index = _load_frame_depth_index(output_dir)
    poses = _load_poses(output_dir, poses_dir)
    if depth_index is None or poses is None:
        return None
    K_by_frame = _intrinsics_for(output_dir, _load_frames(output_dir) or [])
    if K_by_frame is None:
        _log("trace-normals: no intrinsic.txt attributable to the keyframes — falling back")
        return None

    frames = np.unique(frame_global)
    have = [f for f in frames
            if int(f) in depth_index and int(f) in poses and int(f) in K_by_frame]
    if len(have) < max(3, 0.5 * len(frames)):
        _log(f"trace-normals: only {len(have)}/{len(frames)} frames have depth+pose — "
             f"falling back")
        return None

    normals = np.zeros((len(xyz), 3), dtype=np.float32)
    done = np.zeros(len(xyz), dtype=bool)
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
        nmap = _normal_map_from_depth(depth, K_by_frame[int(f)])

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
