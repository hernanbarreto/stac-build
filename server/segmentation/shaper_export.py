"""
ShapeR PKL Export — Package segmented objects for ShapeR inference.
===================================================================

After SAM3 segmentation, exports one .pkl per instance with the schema that
vendor/ShapeR/dataset/shaper_dataset.py consumes:

    points_model            Tensor (N,3)        centered sub-cloud
    bounds                  Tensor (3,)         half-extent of bbox
    T_model_world           Tensor (4,4)        world->model centering transform
    T_zup_obj               Tensor (4,4)        Y-down (cv) -> Z-up (shaper)
    inv_dist_std/dist_std   Tensor (N,)         zeros (no per-point uncertainty)
    image_data              list[bytes]         PNG/JPEG per view
    Ts_camera_model         Tensor (V,4,4)      world->camera in centered model frame
    camera_params           Tensor (V,16)       Fisheye624 [fx,fy,cx,cy,k0..k5,p0,p1,s0..s3]
    visible_points_model    list[Tensor (Mi,3)] points visible in view i (model frame)
    object_point_projections list[Tensor (Mi,2)] (u,v) of those points
    caption                 str                 the object's description (ShapeR conditioning)
    caption_fields          dict|None            {category, shape, material, detail}; None only for the bare label
    caption_source          str                 manual | vlm_object | vlm_concept | vlm_on_demand | label
    category                str                 SAM3 label (fallback caption)
    label, instance_id, n_source_points, n_views, source_frames  metadata
    n_candidate_frames      int                 posed keyframes that SAW the object (the view pool)
    n_posed_frames          int                 posed keyframes with a frame on disk

Backend-agnostic — auto-detects pose source for lidar/Stray, DA3, MapAnything,
hybrid, gaus_slam_*. Pinhole intrinsics get embedded as Fisheye624 with zero
distortion so vendor `rectify_images` becomes ~identity.

Authors: Hernán Barreto — Ingerop IN3
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
    # precision sessions (F0/F5 camera.json): the traceability grid → native mapping and the
    # lens, so pixels map EXACTLY (no resolution guess) and images are undistorted
    grid: Optional[object] = None
    camera: Optional[object] = None

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


def _load_precision_source(output_dir: Path) -> Optional[CameraSource]:
    """The precision core's camera (2026-09-29): camera_poses.txt (c2w per keyframe, F5's
    refined poses, the baked +Y-up frame) keyed by camera_frames.txt (REAL video frame
    numbers), the session camera camera.json (F5's K on the NATIVE grid + OPENCV lens) and
    the traceability grid (camera.json omega_grid: the grid pixel_row/col live on)."""
    cj, fr, po = output_dir / "camera.json", output_dir / "camera_frames.txt", output_dir / "camera_poses.txt"
    if not (cj.exists() and fr.exists() and po.exists()):
        return None
    from precision.camera import load_camera_json
    cam = load_camera_json(cj)
    frames = [int(float(x)) for x in fr.read_text().split()]
    c2w = np.loadtxt(po).reshape(-1, 4, 4)
    if len(c2w) != len(frames):
        logger.error(f"[ShaperExport] {len(frames)} frames but {len(c2w)} poses — no camera")
        return None
    K = np.asarray(cam.K(), np.float64)
    return CameraSource(pose_map={f: c2w[i] for i, f in enumerate(frames)},
                        intrinsics_map={f: K for f in frames},
                        source_resolution=(int(cam.height), int(cam.width)),
                        backend="precision", grid=cam.omega_grid, camera=cam)


def _load_camera_source(session_dir: Path, output_dir: Path) -> Optional[CameraSource]:
    """Auto-detect the active reconstruction backend and load camera data."""
    src = _load_precision_source(output_dir)
    if src is not None:
        return src
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

def _png_encode(image: np.ndarray) -> bytes:
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _jpg_encode(image: np.ndarray, quality: int = 90) -> bytes:
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Fisheye624 is r = f·θ·(1 + k0θ² + … + k5θ¹²): with every k = 0 it is the EQUIDISTANT
# fisheye, NOT a pinhole — the vendor's rectify_images then warps a pinhole frame
# (measured 2026-09-29: 130 px / 27 % at the corner of a 464×832 frame at f = 392).
# The pinhole r = f·tanθ IS this polynomial with k = the Maclaurin coefficients of
# tanθ/θ (0.13 px max error at 50.6°), so the rectification becomes the identity.
_TAN_SERIES_K = (1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0, 1382.0 / 155925.0,
                 21844.0 / 6081075.0)


def _fisheye624_from_pinhole(K: np.ndarray) -> np.ndarray:
    """[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3] reproducing the PINHOLE camera K
    (radial terms = tan series, tangential / thin-prism = 0)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([fx, fy, cx, cy, *_TAN_SERIES_K] + [0.0] * 6, dtype=np.float32)


def _scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _sharpness(gray: np.ndarray) -> float:
    """Variance-of-Laplacian focus measure (higher = sharper). Pure numpy."""
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if g.size < 16 or g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (4.0 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def _pick_diverse_views(scores: np.ndarray, dirs: np.ndarray, n: int) -> np.ndarray:
    """Pick ~``n`` view indices: seed with the highest-scoring view, then greedily
    add whichever remaining view is most angularly distant (camera→object
    direction) from everything picked so far, lightly biased by score — so the
    subset is both high-quality and spread around the object."""
    m = len(scores)
    if m <= n:
        return np.arange(m)
    order = np.argsort(-scores)
    picked = [int(order[0])]
    cand = [int(i) for i in order[1:]]
    srng = float(scores.max() - scores.min()) + 1e-9
    while len(picked) < n and cand:
        D = dirs[picked]
        best_ci, best_v = 0, -1e18
        for ci, idx in enumerate(cand):
            ang = 1.0 - float(np.max(D @ dirs[idx]))      # 0..2; larger = more distinct
            sc = float((scores[idx] - scores.min()) / srng)
            v = ang + 0.35 * sc
            if v > best_v:
                best_v, best_ci = v, ci
        picked.append(cand.pop(best_ci))
    return np.array(picked, dtype=int)


# ── Manual caption parsing ──────────────────────────────────────────

def _parse_manual_caption(text: str) -> Dict[str, str]:
    """Best-effort split of a free-form manual caption into the four-field schema
    the reconstruction classifier consumes (``category, shape, material, detail``).

    The captioner's auto path produces a dict with these keys directly. Manual
    typing usually arrives as comma-separated phrases ("pipe, metallic, slightly
    curved") or a single noun ("trunk"). Both should still surface
    ``caption_fields["category"]`` so ``classify._caption_hint`` can match the
    role/profile regex. We never invent fields — anything past the third comma
    lands in ``detail`` verbatim; missing slots stay as empty strings.
    """
    fields = {"category": "", "shape": "", "material": "", "detail": ""}
    if not text:
        return fields
    parts = [p.strip().rstrip(".,;") for p in str(text).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return fields
    keys = ("category", "shape", "material")
    for i, key in enumerate(keys):
        if i < len(parts):
            fields[key] = parts[i]
    if len(parts) > 3:
        fields["detail"] = ", ".join(parts[3:])
    return fields


# ── Views: the camera on the image, the projection ─────────────────

def _K_on_image(cam: CameraSource, K_src: np.ndarray, W_img: int, H_img: int,
                frame_path: Path) -> np.ndarray:
    """The intrinsics on the JPG's pixel grid.

    Precision sessions: K is the session camera on the NATIVE grid and ``frames/``
    must BE the native video — a resampled frame would silently shift every
    projection, so it fails here. Other backends record K at their processing
    resolution (DA3 384×688 vs 360×640 JPGs): rescale to the image — in EITHER
    direction — from the declared source resolution (Stray) or the centred
    principal point (2·cx, 2·cy: exact for the pinhole DA3 / MapAnything models).
    """
    if cam.grid is not None and cam.camera is not None:
        if (W_img, H_img) != (int(cam.camera.width), int(cam.camera.height)):
            raise RuntimeError(f"{frame_path.name} is {W_img}x{H_img}, the session camera "
                               f"is {cam.camera.width}x{cam.camera.height} — frames/ is not "
                               f"the native video")
        return np.asarray(K_src, dtype=np.float64)
    if cam.source_resolution is not None:
        src_h, src_w = float(cam.source_resolution[0]), float(cam.source_resolution[1])
    else:
        src_w, src_h = 2.0 * float(K_src[0, 2]), 2.0 * float(K_src[1, 2])
    sx = (W_img / src_w) if src_w > 1.0 else 1.0
    sy = (H_img / src_h) if src_h > 1.0 else 1.0
    if abs(sx - 1.0) < 0.01:
        sx = 1.0
    if abs(sy - 1.0) < 0.01:
        sy = 1.0
    if sx != 1.0 or sy != 1.0:
        return _scale_K(np.asarray(K_src, dtype=np.float64), sx, sy)
    return np.asarray(K_src, dtype=np.float64)


def _project_pinhole(points_world: np.ndarray, c2w_4: np.ndarray, K: np.ndarray,
                     W_img: int, H_img: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, inside) of ``points_world`` through the pinhole ``K`` of the camera
    at ``c2w_4``: ``inside`` = in front of the camera and within the image."""
    w2c = np.linalg.inv(c2w_4)
    q = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    zf = q[:, 2] > 1e-6
    z = np.where(zf, q[:, 2], 1.0)
    u = K[0, 0] * q[:, 0] / z + K[0, 2]
    v = K[1, 1] * q[:, 1] / z + K[1, 2]
    inside = zf & (u >= 0) & (u <= W_img - 1) & (v >= 0) & (v <= H_img - 1)
    return u, v, inside


def published_depth_fn(output_dir: Path, log: Callable[[str], None] = logger.info
                       ) -> Optional[Callable[[int], Optional[np.ndarray]]]:
    """``keyframe -> the depth map the PUBLISHED cloud was built from`` (Omega's record ×
    the per-keyframe bend / scale of ``corrected_cloud.json``, composed with the
    certification's transform epochs — precision.chunk_check.load_inputs, the one
    reader of that model), or None when the session has no such model (a cloud not
    built from Omega's records: then no occlusion test can be made and the caller
    says so). Maps are cached: one npz read per keyframe for every object."""
    try:
        from precision.chunk_check import load_inputs, scale_map
        (frames, _c2w, _K, s_k, src, _chunk, _chain, rec, _da3, bend,
         composed) = load_inputs(Path(output_dir).parent, log=lambda m: None)
    except Exception as e:  # noqa: BLE001 — declared by the caller
        log(f"[ShaperExport] no published depth model ({e}) — no occlusion test")
        return None
    if not str(src).startswith("corrected_cloud.json"):
        log(f"[ShaperExport] the cloud's depth model is not on record ({src}) — no occlusion test")
        return None
    offset = composed.get("offset") or {}
    cache: Dict[int, Optional[np.ndarray]] = {}
    known = set(int(f) for f in frames)

    def depth(fidx: int) -> Optional[np.ndarray]:
        fidx = int(fidx)
        if fidx in cache:
            return cache[fidx]
        D = None
        p = Path(rec) / f"frame_{fidx}.npz"
        if fidx in known and p.exists():
            with np.load(p) as z:
                d0 = np.asarray(z["depth"], np.float64)
            D = d0 * scale_map(s_k[fidx], bend.get(fidx), d0.shape[0], d0.shape[1])
            off = float(offset.get(fidx, 0.0))
            if off:
                D = np.where(d0 > 0, D + off, D)
            D = np.where(np.isfinite(d0) & (d0 > 0), D, 0.0).astype(np.float32)
        cache[fidx] = D
        return D

    return depth


def occluded_points(pts_world: np.ndarray, c2w_4: np.ndarray, u: np.ndarray, v: np.ndarray,
                    inside: np.ndarray, W_img: int, H_img: int, D: Optional[np.ndarray],
                    tol_rel: float) -> np.ndarray:
    """Points of ``inside`` that lie BEHIND the keyframe's own surface by more than
    ``tol_rel`` of its depth (``loops.witness.occlusion_tol_rel``, the witness rule):
    the camera sees something else there — a wall between it and the object. A pixel
    with no depth says nothing (the point is not counted occluded)."""
    occ = np.zeros(len(pts_world), bool)
    if D is None or not inside.any():
        return occ
    w2c = np.linalg.inv(c2w_4)
    z = pts_world[inside] @ w2c[2, :3] + w2c[2, 3]
    Hd, Wd = D.shape
    uu = np.clip((u[inside] * Wd / float(W_img)).astype(np.int64), 0, Wd - 1)
    vv = np.clip((v[inside] * Hd / float(H_img)).astype(np.int64), 0, Hd - 1)
    d = D[vv, uu]
    occ[np.flatnonzero(inside)] = (d > 0) & (z > d * (1.0 + float(tol_rel)))
    return occ


# ── The description ─────────────────────────────────────────────────

CAPTION_FIELDS = ("category", "shape", "material", "detail")


def _stored_shape_caption(inst: dict) -> Optional[dict]:
    """The VLM's description the segmentation stored on the instance
    (``shape_caption`` = {caption, category, shape, material, detail, provenance,
    source: "concept" | "object", generated}). A list of candidates yields the
    "object" one over the "concept" one (USER 2026-10-01: the description of the
    object itself beats the description of the concept it was found by)."""
    raw = inst.get("shape_caption")
    if not raw:
        return None
    cands = raw if isinstance(raw, list) else [raw]
    cands = [c for c in cands if isinstance(c, dict) and str(c.get("caption") or "").strip()]
    if not cands:
        return None
    rank = {"object": 0, "concept": 1}
    cands.sort(key=lambda c: rank.get(str(c.get("source") or ""), 2))
    return cands[0]


def resolve_caption(inst_id: int, label: str, inst: dict,
                    captions: Optional[Dict[int, str]],
                    caption_fn: Optional[Callable],
                    caption_frames: List[str],
                    masks_fn: Optional[Callable[[], Dict[str, np.ndarray]]] = None,
                    on_phase: Optional[Callable[[int, str], None]] = None,
                    ) -> Tuple[str, Optional[Dict[str, str]], str]:
    """(caption, caption_fields, caption_source) for one instance, in this order
    of precedence (USER 2026-10-01):

    1. ``captions[inst_id]`` — typed in the UI (human_validated) → ``manual``;
    2. ``inst["shape_caption"]`` — the VLM's description stored by the
       segmentation, "object" over "concept" → ``vlm_object`` | ``vlm_concept``;
    3. ``caption_fn`` — the on-demand captioner, ONLY when nothing is stored (the
       endpoint hands it in only under ``shaper.auto_caption``) → ``vlm_on_demand``;
    4. the SAM3 label → ``label`` (caption_fields None: nothing was described).

    ``caption_fields`` is always filled from a stored description (empty strings
    for fields it lacks — never None when a description exists)."""
    if captions and inst_id in captions and str(captions[inst_id]).strip():
        text = str(captions[inst_id]).strip()
        return text, _parse_manual_caption(text), "manual"
    stored = _stored_shape_caption(inst)
    if stored is not None:
        fields = {k: str(stored.get(k) or "") for k in CAPTION_FIELDS}
        source = str(stored.get("source") or "")
        return (str(stored["caption"]).strip(), fields,
                "vlm_object" if source == "object" else "vlm_concept")
    if caption_fn is not None:
        try:
            if on_phase is not None:
                on_phase(int(inst_id), "captioning")
            masks = masks_fn() if masks_fn is not None else {}
            res = caption_fn(caption_frames, masks, label)
            if isinstance(res, dict) and str(res.get("caption") or "").strip():
                return (str(res["caption"]).strip(),
                        {k: str(res.get(k) or "") for k in CAPTION_FIELDS}, "vlm_on_demand")
            if isinstance(res, str) and res.strip():
                return res.strip(), _parse_manual_caption(res.strip()), "vlm_on_demand"
        except Exception as e:  # noqa: BLE001 — declared; the label stands in
            logger.warning(f"  Caption generation failed for {label}_{inst_id}: {e}")
    return label, None, "label"


def _sam3_view_masks(output_dir: Path, inst_id: int,
                     frame_paths: List[str]) -> Dict[str, np.ndarray]:
    """{frame filename: SAM3 mask} of this instance in the given views, from the
    session's ``seg_masks.npz`` — the real silhouettes for the captioner's crops,
    not the speckle of projected points. The instance's npz object ids are the
    ``segmentation.json`` instances that carry its ``instance_id`` (several
    masklets fused into one object are unioned); mask keys live in the store's
    own frame space (``mask_space``). Empty when the session has no store."""
    output_dir = Path(output_dir)
    store = output_dir / "seg_masks.npz"
    if not store.exists():
        return {}
    oids: List[int] = []
    seg_doc = output_dir / "segmentation.json"
    if seg_doc.exists():
        try:
            doc = json.loads(seg_doc.read_text())
            oids = [int(i["id"]) for i in doc.get("instances", [])
                    if int(i.get("instance_id", -1)) == int(inst_id) and "id" in i]
        except Exception as e:  # noqa: BLE001 — declared below
            logger.warning(f"[ShaperExport] segmentation.json unreadable ({e}) — "
                           f"using the store's own convention for instance {inst_id}")
    if not oids:
        oids = [int(inst_id) - 1]      # the store's convention (hole_audit.calibrate_oid)
    out: Dict[str, np.ndarray] = {}
    try:
        from segmentation import mask_space
        with np.load(store, allow_pickle=True) as z:
            ms = mask_space.resolve(output_dir, masks=z)
            files = set(z.files)
            for fp in frame_paths:
                stem = Path(fp).stem
                if not stem.isdigit():
                    continue
                acc = None
                for oid in oids:
                    key = ms.key(int(stem), oid)
                    if key is None or key not in files:
                        continue
                    m = np.asarray(z[key]) > 0
                    acc = m if acc is None else (acc | m)
                if acc is not None and acc.any():
                    out[Path(fp).name] = acc
    except Exception as e:  # noqa: BLE001 — declared; the caller falls back
        logger.warning(f"[ShaperExport] SAM3 masks unavailable for instance {inst_id}: {e}")
        return {}
    return out


# ── The published mesh of a shape folder ────────────────────────────

def pick_shape_glb(glb_files: List[Path]) -> Optional[Path]:
    """The GLB ``shape_list`` publishes for one object folder: the one whose own
    ``<stem>.meta.json`` sidecar exists (ShapeR's ``run_shaper_batch`` writes the
    pair; the newest sidecar wins when several do). Without any sidecar the
    legacy MeshFlow ``<stem>_visual.glb`` (folder-level meta.json), else the first
    GLB. Until 2026-10-01 ``_visual`` won over a ShapeR mesh sitting next to it."""
    files = [Path(g) for g in glb_files]
    if not files:
        return None
    with_meta = [g for g in files if g.with_suffix(".meta.json").exists()]
    if with_meta:
        return max(with_meta, key=lambda g: g.with_suffix(".meta.json").stat().st_mtime)
    return next((g for g in files if g.stem.endswith("_visual")), files[0])


# ── Main export ─────────────────────────────────────────────────────

def export_shaper_pkls(
    output_dir: Path,
    frames_dir: Path,
    segments_result: dict,
    session_dir: Optional[Path] = None,
    obj_ids: Optional[List[int]] = None,
    caption_fn: Optional[Callable] = None,
    captions: Optional[Dict[int, str]] = None,
    image_format: str = "png",       # "png" or "jpg"
    grayscale: bool = True,           # ShapeR released ckpt expects grayscale
    max_views: int = 0,               # 0 = auto (32 pool); else cap views per object
    min_view_points: int = 50,        # a keyframe is a view when ≥ N object points project inside it
    on_phase: Optional[Callable[[int, str], None]] = None,   # UI: (instance_id, "exporting_pkl" | "captioning")
    occlusion_tol_rel: Optional[float] = None,  # loops.witness.occlusion_tol_rel; None = no occlusion test
) -> List[Path]:
    """Export one ShapeR-compatible .pkl per segmented instance.

    Reads cleaned_cloud.ply + segmentation_result.json. Auto-detects the camera
    source (precision camera.json, Stray/lidar, DA3, MapAnything) from the
    session/output layout. Views: every posed keyframe that sees the object
    (see the candidate block below); caption: see :func:`resolve_caption`.
    """
    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)
    if session_dir is None:
        session_dir = frames_dir.parent
    session_dir = Path(session_dir)

    t0 = time.time()
    logger.info(f"[ShaperExport] start  output_dir={output_dir}  session_dir={session_dir}")

    # ── Load PLY traceability ──
    from segmentation.pipeline import _load_ply_origins

    ply_path = output_dir / "cleaned_cloud.ply"
    if not ply_path.exists():
        for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                ply_path = output_dir / alt
                break

    origins = _load_ply_origins(ply_path)
    if origins is None:
        logger.error(f"Cannot load PLY origins from {ply_path}")
        return []
    xyz, frame_global, pixel_row, pixel_col = origins
    logger.info(f"[ShaperExport] PLY: {len(xyz):,} points, "
                f"{len(np.unique(frame_global))} unique frames")

    # The traceability (birth frame / pixel) is REPORTED, no longer used to pick
    # views (USER 2026-10-01): the views are the posed keyframes that see the object.
    pr_all = pixel_row.astype(np.int32)
    pc_all = pixel_col.astype(np.int32)
    ply_src_h = int(pr_all.max()) + 1 if len(pr_all) else 1
    ply_src_w = int(pc_all.max()) + 1 if len(pc_all) else 1
    logger.info(f"[ShaperExport] PLY traceability resolution ≈ "
                f"{ply_src_w}x{ply_src_h}")

    # ── Load camera source ──
    cam = _load_camera_source(session_dir, output_dir)
    if cam is None:
        return []
    # the occlusion test of the candidate views needs the depth the cloud was built from
    depth_of = (published_depth_fn(output_dir, log=logger.info)
                if occlusion_tol_rel is not None else None)

    # ── Detect world-up convention (once per session) ──
    # Different reconstruction backends emit different conventions:
    # ARKit/Stray = Y-up, Aria MPS = Y-down, some lidar exports = Z-up.
    # Picking the wrong one feeds ShapeR an upside-down/sideways object
    # relative to its training priors and the generator falls apart.
    cam_positions = np.asarray([T[:3, 3] for T in cam.pose_map.values()],
                                dtype=np.float64)
    if (output_dir / ".orientation_applied").exists():
        world_up = "+y"          # reconstruction/orient.py baked +Y up from the camera gravity
    else:
        world_up = _detect_world_up(cam_positions, xyz)
    T_zup_session = _R_TO_ZUP[world_up].astype(np.float64)
    logger.info(f"[ShaperExport] backend={cam.backend}  "
                f"detected world-up={world_up}  → applying rotation to Z-up")
    # the session lens, once: every view is rectified to the undistorted image plane
    undist_maps = None
    if cam.camera is not None and np.any(cam.camera.dist()):
        from precision.camera import undistort_maps
        _mx, _my, _ = undistort_maps(cam.camera)
        undist_maps = (_mx, _my)

    # ── Get instances ──
    instances = segments_result.get("instances", [])
    if not instances:
        logger.warning("No instances in segmentation_result.json")
        return []

    shape_dir = output_dir / "shape"
    shape_dir.mkdir(exist_ok=True)

    encode = _png_encode if image_format == "png" else _jpg_encode
    exported: List[Path] = []

    for inst in instances:
        # In segmentation_result.json, "id" IS the unique instance_id (1-based).
        inst_id = inst.get("id", inst.get("instance_id", inst.get("globalId")))
        label = inst.get("label", f"object_{inst_id}")

        if obj_ids and inst_id not in obj_ids:
            continue

        gi = np.asarray(inst.get("globalIndices", []), dtype=np.int64)
        gi = gi[gi < len(xyz)]
        if len(gi) < 10:
            logger.warning(f"  [{label}_{inst_id}] too few points ({len(gi)}) — skipping")
            continue

        if on_phase is not None:
            on_phase(int(inst_id), "exporting_pkl")
        sub_pts = xyz[gi].astype(np.float64)

        # Center sub-cloud at bbox midpoint, then rotate Y-up (OpenCV/ARKit
        # world) into Z-up (ShapeR training convention). Vendor's
        # `project_points_to_image(points_zup, ...)` makes this requirement
        # explicit. Storing points_model in the original Y-up frame causes
        # the model to "see" the object rotated 90° relative to its priors.
        bb_min = sub_pts.min(axis=0)
        bb_max = sub_pts.max(axis=0)
        centroid = (bb_min + bb_max) / 2

        sub_pts_centered = sub_pts - centroid
        T_center = np.eye(4, dtype=np.float64)
        T_center[:3, 3] = -centroid

        # Apply per-session world-up→Z-up rotation. Recompute bounds in the
        # rotated frame because half-extents per axis change.
        T_zup = T_zup_session
        R_zup = T_zup[:3, :3]
        sub_pts_zup = sub_pts_centered @ R_zup.T
        half_size_zup = (sub_pts_zup.max(axis=0) - sub_pts_zup.min(axis=0)) / 2

        # Composed world → model_zup transform. This is what `rescale_back`
        # inverts to put the predicted mesh back into the original Y-up world.
        T_world_to_model = T_zup @ T_center

        # ── Candidate views: EVERY posed keyframe that SEES the object ──
        # USER 2026-10-01: "armar los pkl a máxima resolución posible". Until then a
        # candidate was a frame where the object's points were BORN, and it needed
        # ``min_view_points`` born there: pccr's backpack reached the PKL with 12
        # views for a preset that reads 32 (the vendor pads with repeats — no new
        # evidence). After voxel / witness / silhouette filtering a point's birth
        # frame says nothing about which cameras saw it. Now ALL of the object's
        # points are projected with every posed keyframe's camera and the frame is
        # a candidate when ≥ ``min_view_points`` land inside the image in front of
        # it; the score + parallax-diverse pick of ``max_views`` is unchanged. No
        # occlusion test exists in this exporter and none is invented here (the
        # vendor builds each view's mask and crop from these projections).
        n_keep = int(max_views) if (max_views and int(max_views) > 0) else 32
        obj_centroid_world = sub_pts.mean(axis=0)
        skipped_no_frame = skipped_no_K = 0
        n_posed_on_disk = 0
        n_occluded = 0
        seen: List[Dict] = []           # frames that see the object; images not decoded yet
        for fidx in sorted(int(f) for f in cam.pose_map.keys()):
            frame_path = frames_dir / f"{fidx:06d}.jpg"
            if not frame_path.exists():
                skipped_no_frame += 1
                continue
            K_src = cam.K_for(fidx)
            if K_src is None:
                skipped_no_K += 1
                continue
            n_posed_on_disk += 1
            with Image.open(str(frame_path)) as im:      # header only — no decode
                W_img, H_img = im.size
            K_frame = _K_on_image(cam, np.asarray(K_src, dtype=np.float64), W_img, H_img, frame_path)
            c2w = cam.pose_map[fidx]
            c2w_4 = np.eye(4, dtype=np.float64)
            c2w_4[:c2w.shape[0], :c2w.shape[1]] = c2w
            u_all, v_all, inside = _project_pinhole(sub_pts, c2w_4, K_frame, W_img, H_img)
            if depth_of is not None:
                # a camera behind a wall projects the object inside its image too: the
                # points its own surface hides are not seen from there
                occ = occluded_points(sub_pts, c2w_4, u_all, v_all, inside, W_img, H_img,
                                      depth_of(fidx), occlusion_tol_rel)
                n_occluded += int(occ.sum())
                inside = inside & ~occ
            n_in = int(inside.sum())
            if n_in < min_view_points:
                continue
            seen.append({"fidx": fidx, "frame_path": frame_path, "c2w_4": c2w_4,
                         "K_full": K_frame, "W": W_img, "H": H_img,
                         "u": u_all, "v": v_all, "inside": inside, "n_pts": n_in})
        n_candidates = len(seen)
        logger.info(f"[ShaperExport] {label}_{inst_id}: {len(gi):,} pts, seen by "
                    f"{n_candidates} of {n_posed_on_disk} posed keyframes "
                    f"(no_frame={skipped_no_frame} no_K={skipped_no_K}; occlusion test "
                    f"{'on' if depth_of is not None else 'OFF'}, {n_occluded:,} point-views hidden)")
        # Images are decoded only for the frames that can still be picked: the
        # ``4 × n_keep`` best-covered candidates (the pre-trim the exporter always ran).
        if len(seen) > 4 * n_keep:
            seen.sort(key=lambda d: -d["n_pts"])
            seen = seen[:4 * n_keep]

        per_frame: List[Dict] = []      # gathered conditioning, full frames
        for d in seen:
            fidx, frame_path, W_img, H_img = d["fidx"], d["frame_path"], d["W"], d["H"]
            with Image.open(str(frame_path)) as im:
                img_np = np.array(im.convert("L" if grayscale else "RGB"))
            if undist_maps is not None:
                # the frame through the session lens → the undistorted image plane (same K)
                import cv2 as _cv2
                img_np = _cv2.remap(img_np, undist_maps[0], undist_maps[1], _cv2.INTER_LINEAR)
            inside = d["inside"]
            uv_full = np.stack([d["u"][inside], d["v"][inside]], 1).astype(np.float32)

            # Feed the full frame to ShapeR. The vendor's dataset/image_processor
            # builds its own object-tight crop from ``object_point_projections``
            # (``rectified_point_masks`` → ``crop_and_resize``) so an external
            # pre-crop would only nudge the principal point in a way the released
            # ckpt wasn't trained on. Heterogeneous per-frame crop sizes also
            # break the batch stacker in `get_image_data_based_on_strategy`.
            c2w_4 = d["c2w_4"]
            T_cam_model = np.linalg.inv(T_world_to_model @ c2w_4).astype(np.float32)
            vis_pts_model = sub_pts_zup[inside].astype(np.float32)

            # full-frame mask of the projections: the captioner's FALLBACK when the
            # session holds no SAM3 mask for this view (see _sam3_view_masks)
            bin_mask_full = np.zeros((H_img, W_img), dtype=bool)
            bin_mask_full[uv_full[:, 1].astype(int), uv_full[:, 0].astype(int)] = True

            # frame-quality features (bbox of the object inside the full frame)
            bw, bh = float(uv_full[:, 0].ptp()), float(uv_full[:, 1].ptp())
            coverage = (bw * bh) / max(float(W_img) * float(H_img), 1.0)
            ucx, ucy = float(uv_full[:, 0].mean()), float(uv_full[:, 1].mean())
            cdist = np.hypot(ucx - W_img / 2.0, ucy - H_img / 2.0) / (0.5 * np.hypot(W_img, H_img))
            cdir = obj_centroid_world - c2w_4[:3, 3]
            cdir = cdir / (np.linalg.norm(cdir) + 1e-9)
            per_frame.append({
                "fidx": fidx, "img_full": img_np, "uv_full": uv_full, "K_full": d["K_full"],
                "T_cam_model": T_cam_model, "vis_pts": vis_pts_model,
                "frame_path": str(frame_path), "bin_mask_full": bin_mask_full,
                "n_pts": int(d["n_pts"]), "coverage": float(coverage),
                "centeredness": max(0.0, 1.0 - float(cdist)), "sharp": _sharpness(img_np),
                "cdir": cdir,
            })

        if not per_frame:
            logger.warning(
                f"  [{label}_{inst_id}] no posed keyframe sees ≥ {min_view_points} of its points "
                f"(posed={n_posed_on_disk} no_frame={skipped_no_frame} no_K={skipped_no_K}) — skipping"
            )
            continue

        # Score the candidates and pick a diverse, high-quality subset.
        if len(per_frame) > 1:
            npts = np.array([d["n_pts"] for d in per_frame], dtype=np.float64)
            cov = np.array([d["coverage"] for d in per_frame], dtype=np.float64)
            cen = np.array([d["centeredness"] for d in per_frame], dtype=np.float64)
            shp = np.array([d["sharp"] for d in per_frame], dtype=np.float64)
            shp_n = (shp - shp.min()) / (shp.ptp() + 1e-9)
            scores = (np.log1p(npts) + 1.4 * cen
                      + 0.6 * np.log1p(100.0 * np.clip(cov, 0.0, 0.8)) + 0.5 * shp_n)
            dirs = np.stack([d["cdir"] for d in per_frame])
            keep = _pick_diverse_views(scores, dirs, min(n_keep, len(per_frame)))
            per_frame = [per_frame[i] for i in keep.tolist()]   # best-first order
        logger.info(f"[ShaperExport] {label}_{inst_id}: {len(per_frame)} of "
                    f"{n_candidates} candidate views kept")

        image_data = [encode(d["img_full"]) for d in per_frame]
        Ts_cam_model = [d["T_cam_model"] for d in per_frame]
        cam_params = [_fisheye624_from_pinhole(np.asarray(d["K_full"])) for d in per_frame]
        visible_pts = [torch.from_numpy(d["vis_pts"]) for d in per_frame]
        obj_proj = [torch.from_numpy(d["uv_full"]) for d in per_frame]
        used_frames = [d["fidx"] for d in per_frame]
        caption_frames = [d["frame_path"] for d in per_frame]

        # The description (USER 2026-10-01: "la descripción de los objetos la que
        # armó el VLM"): manual > stored ``shape_caption`` > on-demand captioner >
        # label — see resolve_caption. The on-demand captioner gets the SAM3 masks
        # of the kept views (the projected-point speckle only where none exists).
        def _masks_for_captioner() -> Dict[str, np.ndarray]:
            masks = _sam3_view_masks(output_dir, int(inst_id), caption_frames)
            if not masks:
                masks = {Path(d["frame_path"]).name: d["bin_mask_full"] for d in per_frame}
            return masks

        caption_text, caption_fields, caption_source = resolve_caption(
            int(inst_id), label, inst, captions, caption_fn, caption_frames,
            _masks_for_captioner, on_phase=on_phase)
        print(f"  Caption ({caption_source}): {caption_text[:80]}")

        # Build PKL — schema mirrors vendor/ShapeR/dataset/shaper_dataset.py.
        # points_model + Ts_camera_model + visible_points_model are all in the
        # Z-up centered model frame. T_model_world encodes the full
        # world(Y-up) → model_zup transform so rescale_back can map predicted
        # meshes back to the original world frame.
        n_pts = len(sub_pts_zup)
        pkl_sample = {
            "points_model":            torch.from_numpy(sub_pts_zup.astype(np.float32)),
            "bounds":                  torch.from_numpy(half_size_zup.astype(np.float32)),
            "T_model_world":           torch.from_numpy(T_world_to_model.astype(np.float32)),
            "T_zup_obj":               torch.from_numpy(T_zup_session.astype(np.float32)),
            "inv_dist_std":            torch.zeros(n_pts, dtype=torch.float32),
            "dist_std":                torch.zeros(n_pts, dtype=torch.float32),

            "image_data":              image_data,
            "Ts_camera_model":         torch.from_numpy(np.stack(Ts_cam_model, axis=0)),
            "camera_params":           torch.from_numpy(np.stack(cam_params, axis=0)),
            "visible_points_model":    visible_pts,
            "object_point_projections": obj_proj,

            "caption":                 caption_text,
            "caption_fields":          caption_fields,
            "caption_source":          caption_source,
            "category":                label,
            "label":                   label,
            "instance_id":             int(inst_id),
            "global_id":               int(inst_id),
            "n_source_points":         int(len(gi)),
            "n_views":                 len(image_data),
            "source_frames":           [int(f) for f in used_frames],
            # the view pool (USER 2026-10-01): how many posed keyframes saw the object
            # and how many posed keyframes the session had on disk
            "n_candidate_frames":      int(n_candidates),
            "n_posed_frames":          int(n_posed_on_disk),
            "is_ariagen2":             False,
        }

        safe_label = label.replace(" ", "_").replace("/", "_")[:30]
        obj_dir = shape_dir / f"{safe_label}_{inst_id}"
        obj_dir.mkdir(exist_ok=True)
        # named by the folder: run_shaper_batch reports every item by the PKL STEM and
        # writes <stem>.glb — a shared 'data.pkl' made every object the same item
        pkl_path = obj_dir / f"{obj_dir.name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_sample, f)

        size_mb = pkl_path.stat().st_size / (1024 * 1024)
        print(f"  → {pkl_path.relative_to(output_dir)} "
              f"({size_mb:.1f} MB, {len(image_data)} views, {len(gi):,} pts)")
        exported.append(pkl_path)

    elapsed = time.time() - t0
    print(f"[Shape] Export complete: {len(exported)} PKLs in {elapsed:.1f}s")
    return exported


# ── Folder lifecycle helpers ────────────────────────────────────────

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
