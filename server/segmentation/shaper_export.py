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
    caption                 str                 narrative caption (ShapeR conditioning) or label
    caption_fields          dict|None            {category, shape, material, detail} or None (manual/fallback)
    category                str                 SAM3 label (fallback caption)
    label, instance_id, n_source_points, n_views, source_frames  metadata

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


def _fisheye624_from_pinhole(K: np.ndarray) -> np.ndarray:
    """[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3] with all distortion = 0.

    With zero distortion the vendor's Fisheye624 projection collapses to a
    pinhole model, so rectify_images becomes a near-identity operation.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([fx, fy, cx, cy] + [0.0] * 12, dtype=np.float32)


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
    min_view_points: int = 50,        # drop views with < N projected points
) -> List[Path]:
    """Export one ShapeR-compatible .pkl per segmented instance.

    Reads cleaned_cloud.ply traceability + segmentation_result.json. Auto-detects
    camera source (Stray/lidar, DA3, MapAnything) from session/output layout.
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

    # Cast pixel coords to int once
    pr_all = pixel_row.astype(np.int32)
    pc_all = pixel_col.astype(np.int32)
    fg_all = frame_global.astype(np.int64)

    # Global PLY traceability resolution. The previous per-object heuristic
    # used the object's max pixel in a frame as the source resolution proxy,
    # which fails when an object only spans a sub-region of the frame: the
    # heuristic falsely concludes the PLY was at lower res and rescales (u,v)
    # off the object. Compute it once from the whole cloud, where the max
    # span reaches every pixel reachable by the camera at PLY-traceability
    # resolution.
    ply_src_h = int(pr_all.max()) + 1 if len(pr_all) else 1
    ply_src_w = int(pc_all.max()) + 1 if len(pc_all) else 1
    logger.info(f"[ShaperExport] PLY traceability resolution ≈ "
                f"{ply_src_w}x{ply_src_h}")

    # ── Load camera source ──
    cam = _load_camera_source(session_dir, output_dir)
    if cam is None:
        return []

    # ── Detect world-up convention (once per session) ──
    # Different reconstruction backends emit different conventions:
    # ARKit/Stray = Y-up, Aria MPS = Y-down, some lidar exports = Z-up.
    # Picking the wrong one feeds ShapeR an upside-down/sideways object
    # relative to its training priors and the generator falls apart.
    cam_positions = np.asarray([T[:3, 3] for T in cam.pose_map.values()],
                                dtype=np.float64)
    world_up = _detect_world_up(cam_positions, xyz)
    T_zup_session = _R_TO_ZUP[world_up].astype(np.float64)
    logger.info(f"[ShaperExport] backend={cam.backend}  "
                f"detected world-up={world_up}  → applying rotation to Z-up")

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

        sub_pts = xyz[gi].astype(np.float64)
        sub_fg = fg_all[gi]
        sub_pr = pr_all[gi]
        sub_pc = pc_all[gi]

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

        unique_frames = np.unique(sub_fg).astype(int)
        logger.info(f"[ShaperExport] {label}_{inst_id}: {len(gi):,} pts, "
                    f"{len(unique_frames)} candidate frames")

        # How many views to keep in the PKL. ``max_views`` ≤ 0 → 32 — the batch
        # script's presets select num_views (16 for quality, 24 for `max`) from
        # this pool, so 32 leaves headroom to pick the best/most-diverse subset for
        # any preset (and to push --num_views up to 32 without re-exporting). Views
        # are cheap in the PKL relative to re-running the whole export. Pre-trim the
        # *candidates* by raw point count if there are far more than we'd ever keep.
        n_keep = int(max_views) if (max_views and int(max_views) > 0) else 32
        if len(unique_frames) > 4 * n_keep:
            counts = np.array([(sub_fg == f).sum() for f in unique_frames])
            unique_frames = unique_frames[np.argsort(-counts)[:4 * n_keep]]

        obj_centroid_world = sub_pts.mean(axis=0)
        per_frame: List[Dict] = []      # gathered, cropped-to-object conditioning
        skipped_no_frame = skipped_no_pose = skipped_no_K = 0

        for fidx in unique_frames:
            fidx = int(fidx)
            frame_path = frames_dir / f"{fidx:06d}.jpg"
            if not frame_path.exists():
                skipped_no_frame += 1
                continue
            c2w = cam.pose_map.get(fidx)
            if c2w is None:
                skipped_no_pose += 1
                continue
            K_src = cam.K_for(fidx)
            if K_src is None:
                skipped_no_K += 1
                continue

            with Image.open(str(frame_path)) as im:
                W_img, H_img = im.size
                pil = im.convert("L" if grayscale else "RGB")
                img_np = np.array(pil)

            # Subset of this object's points seen in this frame.
            mask = sub_fg == fidx
            if mask.sum() < min_view_points:
                continue

            # Bring the traceability pixels AND the intrinsics into the JPG's
            # resolution. The backend records BOTH at its processing resolution
            # (e.g. DA3 depth 384×688), which is frequently NOT the resolution of
            # the saved JPGs (e.g. 360×640). The two must be rescaled to the image
            # — in EITHER direction.
            #
            # Source resolution: prefer the backend's declared source_resolution
            # (Stray sets it from the RGB frames); otherwise use the intrinsics'
            # centered principal point (2·cx, 2·cy) — exact for the pinhole DA3/
            # MapAnything models and, unlike the old max-traceability-pixel guess,
            # independent of how much of the frame the object's cloud covers.
            #
            # The previous `ply_src * 1.2 < H_img` guard only ever UP-scaled (image
            # ≫ traceability) and silently ignored DA3's down-scale case → every
            # projection landed shifted toward the bottom-right and points past the
            # JPG bound got clipped. That is the mask-misalignment ShapeR saw.
            pr_obj = sub_pr[mask].astype(np.float32)
            pc_obj = sub_pc[mask].astype(np.float32)
            if cam.source_resolution is not None:
                src_h, src_w = (float(cam.source_resolution[0]),
                                float(cam.source_resolution[1]))
            else:
                src_w = 2.0 * float(K_src[0, 2])
                src_h = 2.0 * float(K_src[1, 2])
            sx = (W_img / src_w) if src_w > 1.0 else 1.0
            sy = (H_img / src_h) if src_h > 1.0 else 1.0
            if abs(sx - 1.0) < 0.01:
                sx = 1.0
            if abs(sy - 1.0) < 0.01:
                sy = 1.0
            if sx != 1.0 or sy != 1.0:
                pr_obj = pr_obj * sy
                pc_obj = pc_obj * sx
                K_frame = _scale_K(K_src, sx, sy)
            else:
                K_frame = np.asarray(K_src, dtype=np.float64)

            uv_full = np.stack([np.clip(pc_obj, 0, W_img - 1),
                                np.clip(pr_obj, 0, H_img - 1)], axis=1).astype(np.float32)

            # Feed the full frame to ShapeR. The vendor's dataset/image_processor
            # builds its own object-tight crop from ``object_point_projections``
            # (``rectified_point_masks`` → ``crop_and_resize``) so an external
            # pre-crop would only nudge the principal point in a way the released
            # ckpt wasn't trained on. Heterogeneous per-frame crop sizes also
            # break the batch stacker in `get_image_data_based_on_strategy`.
            c2w_4 = np.eye(4, dtype=np.float64)
            c2w_4[:c2w.shape[0], :c2w.shape[1]] = c2w
            T_cam_model = np.linalg.inv(T_world_to_model @ c2w_4).astype(np.float32)
            vis_pts_model = sub_pts_zup[mask].astype(np.float32)

            # full-frame mask for the captioner (it re-reads the full frame)
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
                "fidx": fidx, "img_full": img_np, "uv_full": uv_full, "K_full": K_frame,
                "T_cam_model": T_cam_model, "vis_pts": vis_pts_model,
                "frame_path": str(frame_path), "bin_mask_full": bin_mask_full,
                "n_pts": int(mask.sum()), "coverage": float(coverage),
                "centeredness": max(0.0, 1.0 - float(cdist)), "sharp": _sharpness(img_np),
                "cdir": cdir,
            })

        if not per_frame:
            logger.warning(
                f"  [{label}_{inst_id}] no usable frames "
                f"(no_frame={skipped_no_frame} no_pose={skipped_no_pose} "
                f"no_K={skipped_no_K}) — skipping"
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
        logger.info(f"[ShaperExport] {label}_{inst_id}: {len(per_frame)} views kept")

        image_data = [encode(d["img_full"]) for d in per_frame]
        Ts_cam_model = [d["T_cam_model"] for d in per_frame]
        cam_params = [_fisheye624_from_pinhole(np.asarray(d["K_full"])) for d in per_frame]
        visible_pts = [torch.from_numpy(d["vis_pts"]) for d in per_frame]
        obj_proj = [torch.from_numpy(d["uv_full"]) for d in per_frame]
        used_frames = [d["fidx"] for d in per_frame]
        caption_frames = [d["frame_path"] for d in per_frame]
        caption_masks = {Path(d["frame_path"]).name: d["bin_mask_full"] for d in per_frame}

        # Caption resolution. Manual captions (from the UI) are plain strings.
        # caption_fn — the structured InternVL captioner — returns a dict with
        # the narrative `caption` (ShapeR conditioning) plus the structured
        # fields {category, shape, material, detail} the reconstruction
        # classifier consumes downstream.
        caption_text = label
        caption_fields: Optional[Dict[str, str]] = None
        if captions and inst_id in captions:
            caption_text = captions[inst_id]
            caption_fields = _parse_manual_caption(caption_text)
            print(f"  Caption (manual): {caption_text[:80]}")
        elif caption_fn is not None:
            try:
                res = caption_fn(caption_frames, caption_masks, label)
                if isinstance(res, dict):
                    caption_text = res.get("caption") or label
                    caption_fields = {k: res.get(k, "") for k in
                                      ("category", "shape", "material", "detail")}
                else:
                    caption_text = res or label
                print(f"  Caption (auto):   {caption_text[:80]}")
            except Exception as e:
                logger.warning(f"  Caption generation failed: {e}")

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
            "category":                label,
            "label":                   label,
            "instance_id":             int(inst_id),
            "global_id":               int(inst_id),
            "n_source_points":         int(len(gi)),
            "n_views":                 len(image_data),
            "source_frames":           [int(f) for f in used_frames],
            "is_ariagen2":             False,
        }

        safe_label = label.replace(" ", "_").replace("/", "_")[:30]
        obj_dir = shape_dir / f"{safe_label}_{inst_id}"
        obj_dir.mkdir(exist_ok=True)
        pkl_path = obj_dir / "data.pkl"
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
