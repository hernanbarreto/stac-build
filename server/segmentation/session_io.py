"""
Session camera / cloud I-O shared across export modules.
=========================================================

Generic loaders that several exporters (TSDF, texture bake, reconstruction
v2, mesh generation) need to answer: where are this session's camera poses,
intrinsics and frames, whatever the capture source (Stray/ARKit LiDAR, DA3
streaming, MapAnything/VGGT-Omega), plus the per-instance output-folder
naming/rename/delete helpers for ``output/shape/``.

Extracted verbatim from the retired ``shaper_export.py`` (ShapeR was replaced
by MeshFlow, which consumes plain segment PLYs instead of multi-view PKLs —
only these session-level helpers were generic).
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("ShaperExport")

# 90°-rotation matrices that map each cardinal "up" direction to +Z
# (ShapeR's training convention). One of these is selected at export time
# based on which world-up axis the camera trajectory implies. Backends
# emit different conventions (ARKit/Stray = Y-up, Aria MPS = Y-down,
# some lidar exporters = Z-up), and the wrong choice rotates the object
# upside-down or sideways relative to the model's priors.
_R_TO_ZUP = {
    # Each entry: world-up vector → rotation matrix sending it to (0,0,1)
    "+x": np.array([[ 0, 0,  1, 0], [ 0, 1, 0, 0], [-1, 0,  0, 0], [0, 0, 0, 1]], dtype=np.float64),
    "-x": np.array([[ 0, 0, -1, 0], [ 0, 1, 0, 0], [ 1, 0,  0, 0], [0, 0, 0, 1]], dtype=np.float64),
    "+y": np.array([[ 1, 0,  0, 0], [ 0, 0,-1, 0], [ 0, 1,  0, 0], [0, 0, 0, 1]], dtype=np.float64),  # ARKit / Stray
    "-y": np.array([[ 1, 0,  0, 0], [ 0, 0, 1, 0], [ 0,-1,  0, 0], [0, 0, 0, 1]], dtype=np.float64),  # Aria MPS
    "+z": np.eye(4, dtype=np.float64),                                                                # already Z-up
    "-z": np.array([[ 1, 0,  0, 0], [ 0,-1, 0, 0], [ 0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64),
}


def _detect_world_up(cam_positions: np.ndarray, points: np.ndarray) -> str:
    """Return one of {'+x','-x','+y','-y','+z','-z'} naming the world-up axis.

    Heuristic: in a typical scan the camera trajectory varies mostly
    horizontally — the axis with the smallest spread in camera positions is
    the gravity axis (up or down). Sign is resolved by checking that the
    point cloud lies *below* the camera centroid along that axis, which is
    the common case for a hand-held scan of objects/walls. This avoids
    hardcoding any per-backend assumption.
    """
    if len(cam_positions) < 4 or len(points) < 4:
        # Not enough data for reliable PCA — fall back to ARKit/Stray (+y)
        return "+y"

    # Spread along each cardinal axis. Real-world poses are nearly always
    # axis-aligned with gravity (any modern SLAM/AR backend orients its
    # world frame this way), so picking the cardinal axis with smallest
    # spread is more robust than a full PCA + cross-product gymnastics.
    spread = cam_positions.std(axis=0)
    up_idx = int(np.argmin(spread))

    # Sign: cameras above objects → cam_centroid[up_idx] > points_centroid[up_idx]
    cam_c = cam_positions.mean(axis=0)
    pts_c = points.mean(axis=0)
    sign = "+" if (cam_c[up_idx] - pts_c[up_idx]) > 0 else "-"
    return f"{sign}{'xyz'[up_idx]}"


# ── Camera data loader ──────────────────────────────────────────────

@dataclass
class CameraSource:
    """Per-frame poses and intrinsics, normalized to RGB image resolution."""
    pose_map: Dict[int, np.ndarray]            # frame_idx -> (4,4) c2w
    intrinsics_map: Dict[int, np.ndarray]      # frame_idx -> (3,3) K (RGB res)
    source_resolution: Optional[Tuple[int, int]]  # (H, W) of K reference, or None
    backend: str

    def K_for(self, frame_idx: int) -> Optional[np.ndarray]:
        return self.intrinsics_map.get(frame_idx)


def _find_stray_dir(session_dir: Path) -> Optional[Path]:
    """Stray Scanner data is a sibling directory containing odometry+depth."""
    candidates = [session_dir]
    if session_dir.parent.exists():
        candidates += [c for c in session_dir.parent.iterdir() if c.is_dir()]
    for c in candidates:
        if (c / "odometry.csv").exists() and (c / "camera_matrix.csv").exists():
            return c
    return None


def _load_stray_source(stray_dir: Path) -> Optional[CameraSource]:
    try:
        from ingestors.stray_scanner import load_intrinsics, load_odometry
    except Exception as e:
        logger.warning(f"ingestors.stray_scanner unavailable: {e}")
        return None

    K = load_intrinsics(str(stray_dir / "camera_matrix.csv"))
    frame_indices, poses = load_odometry(str(stray_dir / "odometry.csv"))
    pose_map = dict(zip(frame_indices, poses))
    intr_map = {fi: K for fi in frame_indices}

    # Stray RGB resolution from any frame. Camera_matrix.csv is at RGB res.
    H, W = None, None
    rgb_dir = stray_dir.parent
    # Probe a frame to confirm resolution
    for fi in frame_indices[:3]:
        for jpg_dir in [stray_dir.parent / "src_default" / "frames",
                        stray_dir / "frames"]:
            jpg_path = jpg_dir / f"{fi:06d}.jpg"
            if jpg_path.exists():
                with Image.open(str(jpg_path)) as im:
                    W, H = im.size
                break
        if H is not None:
            break

    logger.info(f"[CameraSource] Stray Scanner: {len(pose_map)} poses, "
                f"K={K[0,0]:.1f},{K[1,1]:.1f},{K[0,2]:.1f},{K[1,2]:.1f}, "
                f"res={W}x{H}")
    return CameraSource(pose_map=pose_map, intrinsics_map=intr_map,
                        source_resolution=(H, W) if H else None,
                        backend="lidar/stray")


def _parse_da3_poses_text(path: Path) -> Dict[int, np.ndarray]:
    """DA3/VGGT-Long camera_poses.txt: 16 floats per line, row-major c2w (or w2c
    in some versions). Convention test: heuristic — assume rows are c2w; if not,
    inversion is consistent across all frames so the relative geometry holds."""
    pose_map = {}
    with open(path) as f:
        for i, line in enumerate(f):
            vals = line.strip().split()
            if len(vals) == 16:
                pose_map[i] = np.array([float(v) for v in vals],
                                       dtype=np.float64).reshape(4, 4)
    return pose_map


def _parse_da3_poses_json(path: Path) -> Dict[int, np.ndarray]:
    # The reconstruction worker copies DA3's space-separated camera_poses.txt to
    # camera_poses_mapanything.json verbatim (canonical name, original format),
    # so this ".json" is often actually whitespace-matrix TEXT, not JSON. Be
    # format-agnostic: try JSON, fall back to the text parser on any decode error
    # — robust for any scan/backend regardless of which format produced it.
    with open(path) as f:
        head = f.read(64).lstrip()
    if not head.startswith(("{", "[")):
        return _parse_da3_poses_text(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return _parse_da3_poses_text(path)
    pose_map = {}
    if isinstance(data, dict):
        # frame_name -> {camera_pose: [[..]], intrinsics:...} OR frame_idx -> mat
        for k, v in data.items():
            try:
                idx = int(Path(str(k)).stem)  # "001234" -> 1234
            except ValueError:
                continue
            if isinstance(v, dict) and "camera_pose" in v:
                pose_map[idx] = np.array(v["camera_pose"], dtype=np.float64)
            elif isinstance(v, list):
                pose_map[idx] = np.array(v, dtype=np.float64).reshape(4, 4)
    elif isinstance(data, list):
        for i, mat in enumerate(data):
            pose_map[i] = np.array(mat, dtype=np.float64).reshape(4, 4)
    return pose_map


def _load_da3_intrinsics(output_dir: Path) -> Tuple[Optional[np.ndarray],
                                                    Optional[Tuple[int, int]]]:
    """DA3 / MapAnything intrinsics. Returns (K, (H, W)) at processed resolution."""
    # 1. intrinsic.txt copied to output/
    for path in [output_dir / "intrinsic.txt",
                 output_dir / "da3_run" / "intrinsic.txt"]:
        if path.exists():
            try:
                arr = np.loadtxt(str(path))
                if arr.shape == (3, 3):
                    return arr, None  # legacy single 3x3
                # DA3 native: one row per keyframe of [fx, fy, cx, cy]. Use the
                # median row as the representative K (intrinsics barely vary
                # frame-to-frame). This is the format DA3 actually writes — the
                # old (3,3)-only check silently failed here → "no camera source".
                if arr.ndim == 2 and arr.shape[1] >= 4:
                    fx, fy, cx, cy = np.median(arr[:, :4], axis=0)
                    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                    dtype=np.float64), None
                if arr.ndim == 1 and arr.shape[0] >= 4:
                    fx, fy, cx, cy = arr[:4]
                    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                    dtype=np.float64), None
            except Exception:
                pass

    # 2. extract_da3_full layout: intrinsics.npy [N,3,3]
    for path in [output_dir / "intrinsics.npy",
                 output_dir / "da3_run" / "intrinsics.npy"]:
        if path.exists():
            arr = np.load(str(path))
            if arr.ndim == 3 and arr.shape[1:] == (3, 3):
                return arr[0].astype(np.float64), None

    # 3. mapanything_poses.json embeds intrinsics per frame
    map_json = output_dir / "mapanything_poses.json"
    if map_json.exists():
        with open(map_json) as f:
            data = json.load(f)
        for v in data.values():
            if isinstance(v, dict) and "intrinsics" in v:
                return np.array(v["intrinsics"], dtype=np.float64), None
    return None, None


def _load_frame_index_map(output_dir: Path) -> Optional[List[int]]:
    """Map each ``camera_poses.txt`` line (a keyframe, in line order) to the REAL
    global frame number.

    The reconstruction writes ``camera_poses.txt`` (one c2w per keyframe, line
    order) plus a ``camera_frames.txt`` sidecar whose line *i* holds the real
    frame number of that keyframe. The cloud's per-point ``frame_global`` and the
    frame JPG filenames are keyed by that REAL frame number — NOT by the line
    ordinal. Consumers must cross-reference the two (the TSDF exporter already
    does, in ``tsdf_export._load_da3_refined_poses``). Returns the per-line frame
    numbers, or None when no sidecar exists (then line-ordinal keying is correct,
    e.g. backends whose ``frame_global`` is itself the keyframe ordinal).
    """
    for cand in (output_dir / "camera_frames.txt",
                 output_dir / "da3_run" / "camera_frames.txt"):
        if cand.exists():
            try:
                nums = [int(x) for x in cand.read_text().split()]
                if nums:
                    return nums
            except Exception:
                pass
    # Fallback: frame_list.json — the ordered list of processed frame filenames.
    for cand in (output_dir / "frame_list.json",
                 output_dir / "da3_run" / "frame_list.json"):
        if cand.exists():
            try:
                names = json.loads(cand.read_text())
                nums = []
                for n in names:
                    m = re.search(r"(\d+)", str(n))
                    nums.append(int(m.group(1)) if m else -1)
                if nums and all(v >= 0 for v in nums):
                    return nums
            except Exception:
                pass
    return None


def _key_poses_by_real_frame(pose_lines: Dict[int, np.ndarray],
                             frame_map: Optional[List[int]]) -> Dict[int, np.ndarray]:
    """Re-key a line-ordinal pose dict (keys ``0..N-1``) to real global frame
    numbers via ``frame_map`` (line -> real frame).

    No-op unless the keys are exactly the contiguous range ``0..N-1`` and the
    counts match — so poses already keyed by real frame (JSON dict-form keyed by
    filename stem) pass through untouched, and a count mismatch degrades safely to
    the old line-ordinal behaviour rather than silently corrupting the mapping.
    """
    if not pose_lines or not frame_map:
        return pose_lines
    keys = sorted(pose_lines)
    if keys != list(range(len(keys))):
        return pose_lines  # already keyed by something other than line ordinal
    if len(frame_map) != len(keys):
        logger.warning(f"[CameraSource] camera_frames.txt has {len(frame_map)} "
                       f"entries but {len(keys)} poses — keeping line-ordinal keys")
        return pose_lines
    return {int(frame_map[i]): pose_lines[i] for i in keys}


def _load_neural_source(output_dir: Path) -> Optional[CameraSource]:
    """DA3 / MapAnything / hybrid pose loader.

    Poses are returned keyed by REAL global frame number, matching the cloud's
    per-point ``frame_global`` and the frame JPG filenames. ``camera_poses.txt``
    lists one c2w per keyframe in line order; ``camera_frames.txt`` gives each
    line's real frame number, and the two are cross-referenced here. Keying by
    line ordinal (the old behaviour) handed every frame a wrong camera — hundreds
    of frames off — so the projected geometry no longer matched the cloud and
    ShapeR received inconsistent multi-view extrinsics.
    """
    frame_map = _load_frame_index_map(output_dir)
    pose_map: Dict[int, np.ndarray] = {}

    # 1) Canonical path: the scale-aligned, metric ``camera_poses.txt``.
    #    IMPORTANT: ``camera_poses_mapanything.json`` is a STALE copy taken
    #    *before* ``reconstruction.scale_align`` rewrites ``camera_poses.txt`` to
    #    metric — its translations are up-to-scale and no longer match the metric
    #    ``cleaned_cloud.ply`` (camera centres come out ~10x too small). It must
    #    NOT be preferred over the scale-aligned text file.
    for path in [output_dir / "camera_poses.txt",
                 output_dir / "da3_run" / "camera_poses.txt",
                 output_dir / "output" / "camera_poses.txt"]:
        if path.exists():
            try:
                lines = _parse_da3_poses_text(path)
                if lines:
                    pose_map = _key_poses_by_real_frame(lines, frame_map)
                    logger.info(
                        f"[CameraSource] Loaded {len(pose_map)} poses from "
                        f"{path.name} "
                        + ("(keyed by real frame via camera_frames.txt)"
                           if frame_map else "(keyed by line ordinal)"))
                    break
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")

    # 2) Fallback for older runs lacking a scale-aligned text file. Still re-key
    #    by camera_frames.txt when the parse produced line-ordinal keys.
    if not pose_map:
        for path in [output_dir / "camera_poses_mapanything.json",
                     output_dir / "mapanything_poses.json"]:
            if path.exists():
                try:
                    parsed = _parse_da3_poses_json(path)
                    if parsed:
                        pose_map = _key_poses_by_real_frame(parsed, frame_map)
                        logger.info(f"[CameraSource] Loaded {len(pose_map)} poses from {path.name}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to parse {path}: {e}")

    # 3) extrinsics.npy [N,4,4] (extract_da3_full layout)
    if not pose_map:
        for path in [output_dir / "extrinsics.npy",
                     output_dir / "da3_run" / "extrinsics.npy"]:
            if path.exists():
                arr = np.load(str(path))
                if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                    # extract_da3_full saves w2c (OpenCV); invert to c2w
                    lines = {i: np.linalg.inv(w2c).astype(np.float64)
                             for i, w2c in enumerate(arr)}
                    pose_map = _key_poses_by_real_frame(lines, frame_map)
                    logger.info(f"[CameraSource] Loaded {len(pose_map)} poses from {path.name} (inverted w2c)")
                    break

    if not pose_map:
        return None

    K, _ = _load_da3_intrinsics(output_dir)
    if K is None:
        logger.warning("Neural backend poses found but no intrinsics — skipping")
        return None

    intr_map = {fi: K for fi in pose_map}
    return CameraSource(pose_map=pose_map, intrinsics_map=intr_map,
                        source_resolution=None, backend="da3/mapanything")


def _load_camera_source(session_dir: Path, output_dir: Path) -> Optional[CameraSource]:
    """Auto-detect the active reconstruction backend and load camera data."""
    stray_dir = _find_stray_dir(session_dir)
    if stray_dir is not None:
        src = _load_stray_source(stray_dir)
        if src is not None:
            return src

    src = _load_neural_source(output_dir)
    if src is not None:
        return src

    logger.error(f"No camera data found for session={session_dir} output={output_dir}")
    return None


# ── Image / camera helpers ──────────────────────────────────────────

def _safe_label(label: str, instance_id: int) -> str:
    return f"{label.replace(' ', '_').replace('/', '_')[:30]}_{instance_id}"


def rename_shape_folder(output_dir: Path, old_label: str, new_label: str,
                        instance_id: int) -> bool:
    output_dir = Path(output_dir)
    shape_dir = output_dir / "shape"
    old_path = shape_dir / _safe_label(old_label, instance_id)
    new_path = shape_dir / _safe_label(new_label, instance_id)
    if old_path.exists() and old_path != new_path:
        old_path.rename(new_path)
        print(f"[Shape] Renamed {old_path.name} → {new_path.name}")
        return True
    return False


def delete_shape_folder(output_dir: Path, label: str, instance_id: int) -> bool:
    import shutil
    output_dir = Path(output_dir)
    folder = output_dir / "shape" / _safe_label(label, instance_id)
    if folder.exists():
        shutil.rmtree(folder)
        print(f"[Shape] Deleted {folder.name}")
        return True
    return False
