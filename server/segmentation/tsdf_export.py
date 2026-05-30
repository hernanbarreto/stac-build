"""
TSDF Mesh Export — Reconstruct per-segment meshes via Open3D ScalableTSDFVolume.
================================================================================

Sibling pipeline to ``shaper_export.py`` for comparing reconstruction methods on
the same segmented instances. ShapeR is an autoregressive object prior trained
on Aria-MPS scale objects and produces "onion layer" artifacts on planar
structures (walls, floors). TSDF integrates per-frame depth observations into a
volumetric SDF, reconciles contradictory observations by weighted average, and
extracts a single watertight surface via marching cubes — naturally avoiding
those artifacts on planar/architectural geometry.

Output layout mirrors ShapeR for easy A/B comparison:

    output/
        shape/<safe_label>_<id>/<safe_label>_<id>.glb   (ShapeR)
        tsdf/ <safe_label>_<id>/<safe_label>_<id>.glb   (this module)
        tsdf/ <safe_label>_<id>/<safe_label>_<id>.meta.json

Backend support: Stray/lidar (uint16 mm depth at 256x192) and DA3/MapAnything
(.npy float meters at processed res). Auto-detected via the existing
``_load_camera_source`` helper.

Authors: Hernán Barreto — Ingerop IN3
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Reuse loaders from shaper_export to keep camera/PLY handling consistent.
from segmentation.shaper_export import (
    CameraSource,
    _load_camera_source,
    _safe_label,
)

logger = logging.getLogger("TSDFExport")
logger.propagate = False  # avoid double-logging via root


# ── Depth source resolution ────────────────────────────────────────

def _resolve_stray_depth(stray_dir: Path, conf_min: int = 1
                         ) -> Optional[Tuple[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]],
                                             Tuple[int, int]]]:
    """Stray/ARKit native depth: ``depth/<idx>.png`` uint16 mm at 256x192.

    Returns (loader, (depth_h, depth_w)) where loader(frame_idx) yields
    (depth_meters_HW, valid_mask_HW). Confidence ≥ conf_min defines validity.
    """
    depth_dir = stray_dir / "depth"
    conf_dir = stray_dir / "confidence"
    if not depth_dir.exists():
        return None

    sample = sorted(depth_dir.glob("*.png"))
    if not sample:
        return None
    with Image.open(sample[0]) as im:
        w, h = im.size

    def _load(frame_idx: int):
        stem = f"{frame_idx:06d}"
        dpath = depth_dir / f"{stem}.png"
        if not dpath.exists():
            return None
        try:
            with Image.open(dpath) as im:
                d = np.array(im, dtype=np.uint16)
        except Exception:
            return None
        depth_m = d.astype(np.float32) / 1000.0  # mm → meters

        cpath = conf_dir / f"{stem}.png"
        if cpath.exists():
            with Image.open(cpath) as im:
                c = np.array(im, dtype=np.uint8)
            valid = c >= conf_min
        else:
            valid = np.ones_like(depth_m, dtype=bool)
        return depth_m, valid

    return _load, (h, w)


def _resolve_da3_depth(output_dir: Path, conf_percentile: Optional[float] = None
                       ) -> Optional[Tuple[Callable[[int], Optional[Tuple[np.ndarray, np.ndarray]]],
                                           Tuple[int, int]]]:
    """DA3/VGGT-Long: ``<out>/da3_run/<stem>_depth.npy`` float meters at processed res.

    Some pipelines also save next to ``output/`` directly.

    If ``conf_percentile`` is given, the loader also reads DA3's per-pixel
    confidence (``da3_run/results_output/frame_<idx>.npz``, same grid as the
    fused depth) and, per frame, drops the pixels whose confidence is below
    that percentile — only the most confident ``(100 - conf_percentile)%`` of
    each frame integrate into the TSDF.
    """
    # Standard locations where DA3 per-frame depths land:
    #   da3_full/        — written by hybrid backend (StrayDA3Streaming) and
    #                      by extract_da3_full.py (preferred for fresh runs)
    #   gaus_slam_run/da3_full/ — when the full GauS-SLAM hybrid path runs
    #   da3_run/         — DA3 streaming output dir for the legacy "da3" backend
    #   <output_dir>     — flat-layout fallback for hand-curated datasets
    candidates = [
        output_dir / "da3_full",
        output_dir / "gaus_slam_run" / "da3_full",
        output_dir / "da3_run",
        output_dir,
    ]
    depth_dir: Optional[Path] = None
    for d in candidates:
        if d.exists() and any(d.glob("*_depth.npy")):
            depth_dir = d
            break

    # Layout B (DA3 streaming "da3" backend): depth+conf packed inside
    # results_output/frame_<kf>.npz (save_depth_conf_result). Keyed by the
    # keyframe-sequence index — same space as frame_global and camera_poses.txt.
    if depth_dir is None:
        npz_dir: Optional[Path] = None
        for d in (output_dir / "da3_run" / "results_output",
                  output_dir / "results_output",
                  output_dir / "gaus_slam_run" / "results_output"):
            if d.exists() and any(d.glob("frame_*.npz")):
                npz_dir = d
                break
        if npz_dir is None:
            return None
        try:
            zs = np.load(str(next(iter(npz_dir.glob("frame_*.npz")))))
            if "depth" not in zs:
                return None
            h, w = zs["depth"].shape
        except Exception:
            return None

        def _load_npz(frame_idx: int):
            p = npz_dir / f"frame_{frame_idx}.npz"
            if not p.exists():
                return None
            try:
                z = np.load(str(p))
            except Exception:
                return None
            if "depth" not in z:
                return None
            d = z["depth"].astype(np.float32)
            if conf_percentile is not None and "conf" in z:
                conf = z["conf"].astype(np.float32)
                if conf.shape == d.shape:
                    thr = float(np.percentile(conf, conf_percentile))
                    return d, conf >= thr
            return d, np.ones_like(d, dtype=bool)

        return _load_npz, (h, w)

    # Probe shape from any sample
    sample = next(iter(depth_dir.glob("*_depth.npy")))
    arr = np.load(str(sample))
    if arr.ndim != 2:
        return None
    h, w = arr.shape

    # Optional per-pixel confidence — DA3 saves depth+conf together in
    # results_output/frame_<idx>.npz (same grid as the fused depth).
    conf_dir: Optional[Path] = None
    if conf_percentile is not None:
        for d in (output_dir / "da3_run" / "results_output",
                  output_dir / "results_output",
                  output_dir / "gaus_slam_run" / "results_output"):
            if d.exists() and any(d.glob("frame_*.npz")):
                conf_dir = d
                break

    def _load(frame_idx: int):
        d = None
        for stem in (f"{frame_idx:06d}", f"{frame_idx:05d}", str(frame_idx)):
            p = depth_dir / f"{stem}_depth.npy"
            if p.exists():
                try:
                    d = np.load(str(p)).astype(np.float32)
                except Exception:
                    return None
                break
        if d is None:
            return None
        if conf_dir is not None:
            cp = conf_dir / f"frame_{frame_idx}.npz"
            if cp.exists():
                try:
                    conf = np.load(str(cp))["conf"].astype(np.float32)
                except Exception:
                    conf = None
                if conf is not None and conf.shape == d.shape:
                    thr = float(np.percentile(conf, conf_percentile))
                    return d, conf >= thr
        return d, np.ones_like(d, dtype=bool)

    return _load, (h, w)


def _resolve_da3_frame_source(output_dir: Path, conf_percentile: Optional[float] = None
                              ) -> Optional[Tuple[Callable[[int], Optional[dict]],
                                                  Tuple[int, int]]]:
    """Data-driven per-frame source: ``results_output/frame_<kf>.npz`` bundles
    ``depth``, ``conf``, ``intrinsics`` (3x3) and ``image`` (H,W,3) ALL at the
    same native resolution DA3 produced — no median, no rescaling, no
    orientation guessing. The loader returns, per keyframe-index, a dict::

        {"depth": (H,W) float32, "valid": (H,W) bool,
         "K": (3,3) float64 | None, "rgb": (H,W,3) uint8 | None, "hw": (H,W)}

    Using each frame's OWN intrinsics+image (instead of a scaled global K and a
    re-decoded JPG) keeps the TSDF projection consistent with the cloud for any
    scan/resolution/orientation. Returns ``(loader, (H, W))`` or None.
    """
    npz_dir: Optional[Path] = None
    for d in (output_dir / "da3_run" / "results_output",
              output_dir / "results_output",
              output_dir / "gaus_slam_run" / "results_output"):
        if d.exists() and any(d.glob("frame_*.npz")):
            npz_dir = d
            break
    if npz_dir is None:
        return None
    try:
        probe = np.load(str(next(iter(npz_dir.glob("frame_*.npz")))))
        if "depth" not in probe or "intrinsics" not in probe:
            return None  # not the rich layout — caller falls back
        h, w = probe["depth"].shape
    except Exception:
        return None

    def _load(frame_idx: int) -> Optional[dict]:
        p = npz_dir / f"frame_{frame_idx}.npz"
        if not p.exists():
            return None
        try:
            z = np.load(str(p))
        except Exception:
            return None
        if "depth" not in z:
            return None
        d = z["depth"].astype(np.float32)
        valid = np.ones_like(d, dtype=bool)
        if conf_percentile is not None and "conf" in z:
            conf = z["conf"].astype(np.float32)
            if conf.shape == d.shape:
                thr = float(np.percentile(conf, conf_percentile))
                valid = conf >= thr
        K = (np.asarray(z["intrinsics"], dtype=np.float64).reshape(3, 3)
             if "intrinsics" in z else None)
        rgb = None
        if "image" in z:
            img = np.asarray(z["image"])
            if img.ndim == 3 and img.shape[2] == 3 and img.shape[:2] == d.shape:
                rgb = img.astype(np.uint8)
        return {"depth": d, "valid": valid, "K": K, "rgb": rgb, "hw": d.shape}

    return _load, (h, w)


# ── Per-frame mask construction ────────────────────────────────────

def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation via a single boolean-OR shift loop.

    Avoids a SciPy/OpenCV dependency for callers running in the lean ``da3``
    env. For radius ≤ 6 (typical: 3) the cost is trivial vs. depth I/O.
    """
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            y0_src = max(0, -dy); y1_src = min(h, h - dy)
            x0_src = max(0, -dx); x1_src = min(w, w - dx)
            y0_dst = max(0, dy);  y1_dst = min(h, h + dy)
            x0_dst = max(0, dx);  x1_dst = min(w, w + dx)
            out[y0_dst:y1_dst, x0_dst:x1_dst] |= mask[y0_src:y1_src, x0_src:x1_src]
    return out


def _build_segment_mask_at_depth(pr_rgb: np.ndarray, pc_rgb: np.ndarray,
                                 depth_h: int, depth_w: int,
                                 ply_h: int, ply_w: int,
                                 dilate_radius: int) -> np.ndarray:
    """Project segment pixel coords (at PLY traceability resolution) onto the
    depth grid and dilate to fill the gaps between sparse projections.

    PLY traceability records pixels at the camera-frame resolution that the
    point cloud was extracted from — for Stray/lidar that is RGB resolution
    (e.g. 1920x1440) while depth is at 256x192. Scaling is a simple
    floor-divide to put points into the depth grid.
    """
    sy = depth_h / float(ply_h) if ply_h else 1.0
    sx = depth_w / float(ply_w) if ply_w else 1.0

    yy = np.clip((pr_rgb.astype(np.float32) * sy).astype(np.int32), 0, depth_h - 1)
    xx = np.clip((pc_rgb.astype(np.float32) * sx).astype(np.int32), 0, depth_w - 1)

    seed = np.zeros((depth_h, depth_w), dtype=bool)
    seed[yy, xx] = True
    return _dilate_mask(seed, dilate_radius)


# ── Camera intrinsics scaling ──────────────────────────────────────

def _intrinsics_for_depth(K_rgb: np.ndarray, rgb_h: int, rgb_w: int,
                          depth_h: int, depth_w: int) -> np.ndarray:
    """Scale a pinhole K from RGB to depth resolution (per-axis ratio)."""
    sx = depth_w / float(rgb_w)
    sy = depth_h / float(rgb_h)
    K_d = K_rgb.copy().astype(np.float64)
    K_d[0, 0] *= sx; K_d[0, 2] *= sx
    K_d[1, 1] *= sy; K_d[1, 2] *= sy
    return K_d


# ── RGB colour for TSDF integration ────────────────────────────────

def _load_rgb_at_depth(frames_dir: Path, frame_idx: int,
                       depth_h: int, depth_w: int,
                       frame_name: Optional[str] = None) -> np.ndarray:
    """Load an RGB frame and resize to the depth grid.

    ``frame_name`` (e.g. the actual keyframe filename) takes precedence over the
    ``{idx:06d}.jpg`` convention — needed when ``frame_idx`` is a keyframe-
    sequence index (0..N) rather than the original frame number, so we load the
    real keyframe (e.g. 000020.jpg) instead of the wrong consecutive frame.

    Returns a uint8 (depth_h, depth_w, 3) array; neutral grey if missing.
    """
    jpg = frames_dir / (frame_name if frame_name else f"{frame_idx:06d}.jpg")
    if jpg.exists():
        try:
            with Image.open(jpg) as im:
                im.draft("RGB", (depth_w, depth_h))  # fast JPEG downscale
                im = im.convert("RGB").resize((depth_w, depth_h), Image.BILINEAR)
                return np.asarray(im, dtype=np.uint8)
        except Exception:
            pass
    return np.full((depth_h, depth_w, 3), 200, dtype=np.uint8)


# ── DA3 loop-closure-refined poses (Phase 2) ───────────────────────

def _load_da3_refined_poses(output_dir: Path, frames_dir: Path
                            ) -> Optional[Dict[int, np.ndarray]]:
    """DA3 loop-closure-refined c2w poses keyed by global frame index.

    The DA3 hybrid run writes ``camera_poses.txt`` — one row-major 4x4 C2W
    matrix per keyframe, in keyframe order — after loop closure + Sim3
    optimisation. Those poses are globally consistent; raw ARKit odometry
    drifts. Line i maps to the i-th keyframe in ``selected_frames.json`` and
    the keyframe filename gives the global frame index (hence its depth file).
    Returns None if the inputs are missing or inconsistent.
    """
    poses_txt = None
    for cand in (output_dir / "da3_run" / "camera_poses.txt",
                 output_dir / "camera_poses.txt"):
        if cand.exists():
            poses_txt = cand
            break
    if poses_txt is None:
        return None

    sel_json = frames_dir / "selected_frames.json"
    if not sel_json.exists():
        return None
    try:
        with open(sel_json) as f:
            kf_files = sorted(json.load(f).get("selected_files", []))
    except Exception:
        return None
    if not kf_files:
        return None

    mats: List[np.ndarray] = []
    with open(poses_txt) as f:
        for line in f:
            vals = line.split()
            if len(vals) == 16:
                mats.append(np.array([float(v) for v in vals],
                                     dtype=np.float64).reshape(4, 4))
    if len(mats) != len(kf_files):
        logger.warning(f"[TSDF-scene] camera_poses.txt has {len(mats)} poses but "
                       f"selected_frames.json lists {len(kf_files)} keyframes — "
                       f"cannot map; skipping refined poses")
        return None

    # Key by KEYFRAME-SEQUENCE index (line order), NOT the original frame number
    # from the filename. The depth npz (frame_<kf>.npz), the per-point
    # frame_global, and the base pose parser (_parse_da3_poses_text) are ALL
    # keyed by keyframe-sequence index — keying refined poses by the original
    # number (int(fname.stem)) made them mismatch the depth → no frames
    # integrated / wrong depth. Line i of camera_poses.txt is the i-th keyframe.
    refined: Dict[int, np.ndarray] = {i: mat for i, mat in enumerate(mats)}
    return refined or None


def _read_recon_backend(output_dir: Path) -> Optional[str]:
    """Reconstruction backend recorded in output/chunk_*_meta.json.

    Values: ``da3_hybrid`` (DA3+LiDAR fused depth available), ``da3`` (DA3
    neural depth), ``mapanything`` (VGGT-Long), ``da3_lidar``/``lidar`` (raw
    LiDAR only). Returns None if no chunk meta is found.
    """
    for cand in sorted(output_dir.glob("chunk_*_meta.json")):
        try:
            with open(cand) as f:
                b = json.load(f).get("backend")
            if b:
                return str(b)
        except Exception:
            continue
    return None


# ── Public API ─────────────────────────────────────────────────────

def export_tsdf_meshes(
    output_dir: Path,
    frames_dir: Path,
    segments_result: dict,
    session_dir: Optional[Path] = None,
    obj_ids: Optional[List[int]] = None,
    voxel_length: float = 0.015,        # 1.5 cm — wall/floor sweet spot
    sdf_trunc: float = 0.04,            # 4 cm — reconciliation window
    depth_trunc: float = 5.0,           # 5 m — drop noisy far returns
    depth_min: float = 0.15,            # 15 cm — drop near-camera artifacts
    dilate_radius: int = 3,             # px — seed→region for sparse traceability
    progress_cb: Optional[Callable[[int, str, Optional[float], Optional[str]], None]] = None,
) -> List[Path]:
    """Reconstruct one TSDF mesh per segmented instance.

    Args:
        output_dir: session output dir (contains ``cleaned_cloud.ply``).
        frames_dir: RGB frames dir (used only to detect resolution).
        segments_result: parsed ``segmentation_result.json``.
        session_dir: parent dir to detect Stray scan directory.
        obj_ids: optional filter — only reconstruct these instance IDs.
        voxel_length / sdf_trunc / depth_trunc / depth_min / dilate_radius:
            see Open3D ScalableTSDFVolume; defaults are tuned for room-scale
            indoor lidar scans.
        progress_cb: ``(instance_id, phase, elapsed, mesh_path)`` callback.
            Phases: ``starting`` → ``integrating`` → ``extracting`` →
            ``done`` | ``error``.

    Returns:
        List of ``.glb`` paths written.
    """
    import open3d as o3d  # local import — heavy and only needed here
    from segmentation.pipeline import _load_ply_origins

    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)
    if session_dir is None:
        session_dir = frames_dir.parent
    session_dir = Path(session_dir)

    t0 = time.time()
    logger.info(f"[TSDF] start  output_dir={output_dir}  session_dir={session_dir}")
    logger.info(f"[TSDF] params  voxel={voxel_length}m  trunc={sdf_trunc}m  "
                f"depth_max={depth_trunc}m  dilate={dilate_radius}px")

    # PLY traceability — per-point (frame, pr, pc) at PLY recording resolution.
    ply_path = output_dir / "cleaned_cloud.ply"
    if not ply_path.exists():
        for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                ply_path = output_dir / alt
                break
    origins = _load_ply_origins(ply_path)
    if origins is None:
        logger.error(f"[TSDF] cannot load PLY origins from {ply_path}")
        return []
    xyz, frame_global, pixel_row, pixel_col = origins
    pr_all = pixel_row.astype(np.int32)
    pc_all = pixel_col.astype(np.int32)
    fg_all = frame_global.astype(np.int64)
    ply_h = int(pr_all.max()) + 1 if len(pr_all) else 1
    ply_w = int(pc_all.max()) + 1 if len(pc_all) else 1
    logger.info(f"[TSDF] PLY: {len(xyz):,} points, "
                f"{len(np.unique(fg_all))} unique frames, traceability ≈ {ply_w}x{ply_h}")

    # Camera source (poses + RGB-resolution K). Stray preferred — has native depth.
    cam = _load_camera_source(session_dir, output_dir)
    if cam is None:
        logger.error("[TSDF] no camera source — aborting")
        return []
    logger.info(f"[TSDF] backend={cam.backend}  poses={len(cam.pose_map)}")

    # Depth source — first try Stray (sibling dir), then DA3 (output dir).
    stray_depth = None
    from segmentation.shaper_export import _find_stray_dir
    stray_dir = _find_stray_dir(session_dir)
    if stray_dir is not None:
        stray_depth = _resolve_stray_depth(stray_dir)
        if stray_depth is not None:
            logger.info(f"[TSDF] depth source: Stray {stray_dir} "
                        f"(shape={stray_depth[1][1]}x{stray_depth[1][0]})")
    da3_depth = _resolve_da3_depth(output_dir) if stray_depth is None else None
    if da3_depth is not None:
        logger.info(f"[TSDF] depth source: DA3 .npy "
                    f"(shape={da3_depth[1][1]}x{da3_depth[1][0]})")

    if stray_depth is None and da3_depth is None:
        logger.error("[TSDF] no depth source found (no Stray depth/, no DA3 *_depth.npy)")
        return []

    depth_loader, (depth_h, depth_w) = stray_depth or da3_depth

    # RGB resolution — needed to scale K from RGB→depth and PLY pixel coords→depth.
    rgb_h: Optional[int] = None
    rgb_w: Optional[int] = None
    for jpg_path in sorted(frames_dir.glob("*.jpg"))[:1]:
        with Image.open(jpg_path) as im:
            rgb_w, rgb_h = im.size
        break
    if rgb_h is None or rgb_w is None:
        # Fall back to PLY trace resolution which equals RGB res for Stray.
        rgb_h, rgb_w = ply_h, ply_w
        logger.warning(f"[TSDF] no JPG in {frames_dir} — assuming RGB res = PLY trace ({rgb_w}x{rgb_h})")

    instances = segments_result.get("instances", [])
    if not instances:
        logger.warning("[TSDF] no instances in segmentation_result.json")
        return []

    tsdf_root = output_dir / "tsdf"
    tsdf_root.mkdir(exist_ok=True)
    exported: List[Path] = []

    for inst in instances:
        inst_id = inst.get("id", inst.get("instance_id", inst.get("globalId")))
        label = inst.get("label", f"object_{inst_id}")
        if obj_ids and inst_id not in obj_ids:
            continue

        gi = np.asarray(inst.get("globalIndices", []), dtype=np.int64)
        gi = gi[gi < len(xyz)]
        if len(gi) < 10:
            logger.warning(f"[TSDF] {label}_{inst_id}: too few points ({len(gi)}) — skipping")
            continue

        sub_fg = fg_all[gi]
        sub_pr = pr_all[gi]
        sub_pc = pc_all[gi]
        unique_frames = np.unique(sub_fg).astype(int)
        logger.info(f"[TSDF] {label}_{inst_id}: {len(gi):,} pts, {len(unique_frames)} frames")

        if progress_cb:
            progress_cb(int(inst_id), "starting", None, None)

        # Per-instance TSDF volume — Open3D recommends new volume per object so
        # voxel hashes don't bleed across segments.
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.Gray32,
        )

        t_inst = time.time()
        n_integrated = 0
        skipped_no_depth = skipped_no_pose = skipped_empty_mask = 0

        for fidx in unique_frames:
            fidx = int(fidx)
            c2w = cam.pose_map.get(fidx)
            if c2w is None:
                skipped_no_pose += 1
                continue
            depth_pair = depth_loader(fidx)
            if depth_pair is None:
                skipped_no_depth += 1
                continue
            depth_m, depth_valid = depth_pair

            # Build the per-frame mask in depth-grid coords from this segment's
            # projected points — sparse seeds + dilation gives a usable region.
            seg_mask_pix = sub_fg == fidx
            mask = _build_segment_mask_at_depth(
                sub_pr[seg_mask_pix], sub_pc[seg_mask_pix],
                depth_h, depth_w, ply_h, ply_w, dilate_radius,
            )
            mask &= depth_valid
            mask &= depth_m > depth_min
            mask &= depth_m < depth_trunc

            # Drop "flying pixels" at depth discontinuities (silhouette edges).
            # The mask dilation purposely overshoots to fill gaps in the sparse
            # PLY traceability — but it also picks up background pixels right at
            # the object silhouette, where depth jumps from the object to the
            # wall behind. Those pixels project onto neither surface accurately
            # and create the "extra layer" artifact behind the object. We drop
            # any pixel where a 1-step neighbour differs by > edge_thresh metres.
            edge_thresh = 0.08  # 8 cm jump = silhouette boundary
            d = np.where(mask, depth_m, 0.0)
            gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
            gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
            mask &= (gx < edge_thresh) & (gy < edge_thresh)

            if not mask.any():
                skipped_empty_mask += 1
                continue

            # Apply mask: zero outside the segment so TSDF only fuses our object.
            depth_masked = np.where(mask, depth_m, 0.0).astype(np.float32)

            # Intrinsics at depth resolution.
            K_rgb = cam.K_for(fidx)
            if K_rgb is None:
                continue
            K_d = _intrinsics_for_depth(K_rgb, rgb_h, rgb_w, depth_h, depth_w)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=depth_w, height=depth_h,
                fx=float(K_d[0, 0]), fy=float(K_d[1, 1]),
                cx=float(K_d[0, 2]), cy=float(K_d[1, 2]),
            )

            # Open3D RGBD wants depth scale=1 when depth is already in meters.
            # Flat gray color — TSDF needs *some* color channel and we don't
            # care about texturing the comparison mesh. Gray32 color_type
            # requires float32 single-channel (uint8 single-channel triggers
            # "Unsupported image format" in ScalableTSDFVolume::Integrate).
            color_np = np.full((depth_h, depth_w), 0.78, dtype=np.float32)
            o3d_color = o3d.geometry.Image(color_np)
            o3d_depth = o3d.geometry.Image(depth_masked)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth,
                depth_scale=1.0, depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )

            c2w_4 = np.eye(4, dtype=np.float64)
            c2w_4[:c2w.shape[0], :c2w.shape[1]] = c2w
            extrinsic = np.linalg.inv(c2w_4)  # world → cam, what Open3D expects
            volume.integrate(rgbd, intrinsic, extrinsic)
            n_integrated += 1

            if progress_cb and n_integrated % 25 == 0:
                progress_cb(int(inst_id), "integrating",
                            time.time() - t_inst, None)

        if n_integrated == 0:
            logger.warning(f"[TSDF] {label}_{inst_id}: no frames integrated "
                           f"(no_depth={skipped_no_depth} no_pose={skipped_no_pose} "
                           f"empty_mask={skipped_empty_mask}) — skipping")
            if progress_cb:
                progress_cb(int(inst_id), "error", time.time() - t_inst,
                            None)
            continue

        if progress_cb:
            progress_cb(int(inst_id), "extracting", time.time() - t_inst, None)

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        # Drop floating shells: cluster the mesh and keep only components with
        # >= 5% of the largest cluster's triangles (or 200 tris min). TSDF
        # commonly produces small parallel "ghost" sheets when the same pixel
        # is observed from frames whose poses disagree slightly — they show up
        # as disconnected shells. Cluster-area filtering removes them without
        # touching the main surface.
        try:
            tri_clusters, n_per_cluster, _ = (
                mesh.cluster_connected_triangles()
            )
            tri_clusters = np.asarray(tri_clusters)
            n_per_cluster = np.asarray(n_per_cluster)
            if len(n_per_cluster) > 1:
                min_tri = max(200, int(0.05 * int(n_per_cluster.max())))
                keep = n_per_cluster >= min_tri
                drop_tris = ~keep[tri_clusters]
                n_drop = int(drop_tris.sum())
                if n_drop > 0:
                    mesh.remove_triangles_by_mask(drop_tris)
                    mesh.remove_unreferenced_vertices()
                    logger.info(
                        f"[TSDF] {label}_{inst_id}: dropped "
                        f"{int((~keep).sum())}/{len(n_per_cluster)} small "
                        f"cluster(s) ({n_drop:,} tris, min_tri={min_tri})"
                    )
        except Exception as e:
            logger.warning(
                f"[TSDF] {label}_{inst_id}: cluster cleanup skipped ({e})"
            )

        # Tiny smooth pass to soften staircase artifacts from voxelization.
        # Conservative — 1 iter, lambda 0.5 — preserves wall planarity.
        try:
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh.compute_vertex_normals()
        except Exception as e:
            logger.warning(f"[TSDF] {label}_{inst_id}: smoothing skipped ({e})")

        n_v = len(mesh.vertices)
        n_t = len(mesh.triangles)

        obj_dir = tsdf_root / _safe_label(label, int(inst_id))
        obj_dir.mkdir(exist_ok=True)
        glb_path = obj_dir / f"{_safe_label(label, int(inst_id))}.glb"
        meta_path = glb_path.with_suffix(".meta.json")

        # Open3D 0.19 supports .glb writes natively via write_triangle_mesh
        # when extension is .glb. Use it directly.
        ok = o3d.io.write_triangle_mesh(str(glb_path), mesh, write_ascii=False)
        if not ok:
            logger.error(f"[TSDF] failed to write GLB at {glb_path}")
            if progress_cb:
                progress_cb(int(inst_id), "error", time.time() - t_inst, None)
            continue

        elapsed = time.time() - t_inst
        meta = {
            "instance_id": int(inst_id),
            "label": label,
            "method": "tsdf",
            "voxel_length": voxel_length,
            "sdf_trunc": sdf_trunc,
            "depth_trunc": depth_trunc,
            "depth_min": depth_min,
            "dilate_radius": dilate_radius,
            "n_frames_seen": int(len(unique_frames)),
            "n_frames_integrated": int(n_integrated),
            "skipped_no_depth": int(skipped_no_depth),
            "skipped_no_pose": int(skipped_no_pose),
            "skipped_empty_mask": int(skipped_empty_mask),
            "n_vertices": int(n_v),
            "n_triangles": int(n_t),
            "elapsed_s": float(elapsed),
            "depth_resolution": [int(depth_w), int(depth_h)],
            "rgb_resolution": [int(rgb_w), int(rgb_h)],
            "backend": cam.backend,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        size_mb = glb_path.stat().st_size / (1024 * 1024)
        logger.info(f"[TSDF] ✅ {label}_{inst_id}: {n_v:,} verts / {n_t:,} tris "
                    f"({size_mb:.1f} MB) in {elapsed:.1f}s "
                    f"({n_integrated}/{len(unique_frames)} frames)")
        exported.append(glb_path)

        if progress_cb:
            progress_cb(int(inst_id), "done", elapsed, str(glb_path))

    logger.info(f"[TSDF] complete: {len(exported)} mesh(es) in {time.time()-t0:.1f}s")
    return exported


def delete_tsdf_folder(output_dir: Path, label: str, instance_id: int) -> bool:
    """Remove a per-instance TSDF folder (mirrors ``delete_shape_folder``)."""
    import shutil
    output_dir = Path(output_dir)
    folder = output_dir / "tsdf" / _safe_label(label, instance_id)
    if folder.exists():
        shutil.rmtree(folder)
        logger.info(f"[TSDF] Deleted {folder.name}")
        return True
    return False


# ── Whole-scene TSDF (single mesh from entire scan) ─────────────────────────

def export_tsdf_scene(
    output_dir: Path,
    frames_dir: Path,
    session_dir: Optional[Path] = None,
    voxel_length: float = 0.015,         # 1.5 cm — matched to iPad-LiDAR footprint
    sdf_trunc: float = 0.05,             # 5 cm — reconciles noisy depth into one surface
    depth_trunc: float = 5.0,
    depth_min: float = 0.15,
    edge_thresh: float = 0.04,           # 4 cm — stricter discontinuity filter
    conf_min: int = 2,                   # Stray/ARKit conf 0/1/2 — 2=high only (raw-LiDAR path)
    da3_conf_percentile: float = 50.0,   # drop the lowest-conf % of each DA3 frame (0 = off)
    mask_to_cleaned_cloud: bool = True,  # integrate only pixels present in cleaned_cloud.ply
    cleaned_cloud_dilate: int = 3,       # px dilation of the cleaned-cloud pixel mask
    smooth_iterations: int = 2,          # smoothing iterations post-extract
    smooth_method: str = "simple",       # winner of visual A/B over taubin (LiDAR-noise data)
    fill_holes: bool = False,            # off: texrecon leaves filled holes untextured
                                         # → scattered grey patches that fragment the atlas
    fill_hole_size: float = 0.1,         # max hole size to fill when enabled
    decimate_target: int = 1_500_000,    # target tri count before texturing (0 = off)
    texture: bool = True,                # bake a UV-atlas texture via texrecon
    use_refined_poses: bool = True,      # DA3 loop-closure poses, not raw ARKit
    scene_name: str = "scene",           # output subfolder under output/tsdf/
    variant_label: Optional[str] = None, # human label for the viewer panel
    tsdf_block_count: int = 120_000,     # GPU VoxelBlockGrid hash slots (~10GB @120k)
    tsdf_weight_thresh: float = 2.0,     # min observations per voxel to extract
    progress_cb: Optional[Callable[[str, Optional[float], Optional[str]], None]] = None,
) -> Optional[Path]:
    """Integrate the entire scan's depth maps into a single TSDF volume.

    Same depth pipeline as ``export_tsdf_meshes`` but without per-instance
    masking: every valid pixel of every frame with a pose contributes to one
    global mesh. Output goes to ``output_dir/tsdf_scene/scene.glb`` with a
    ``scene.meta.json`` sidecar.

    Returns the path of the written GLB, or None if no frames could be
    integrated.
    """
    import open3d as o3d  # local — heavy

    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)
    if session_dir is None:
        session_dir = frames_dir.parent
    session_dir = Path(session_dir)

    t0 = time.time()
    logger.info(f"[TSDF-scene] start  output_dir={output_dir}  session_dir={session_dir}")
    logger.info(f"[TSDF-scene] params  voxel={voxel_length}m  trunc={sdf_trunc}m  "
                f"depth_max={depth_trunc}m  depth_min={depth_min}m  "
                f"edge={edge_thresh}m  conf_min={conf_min}  smooth_iters={smooth_iterations}")

    # Camera source (poses + RGB-resolution K).
    cam = _load_camera_source(session_dir, output_dir)
    if cam is None:
        logger.error("[TSDF-scene] no camera source — aborting")
        return None
    logger.info(f"[TSDF-scene] backend={cam.backend}  poses={len(cam.pose_map)}")

    # ── Phase 2: prefer DA3 loop-closure-refined poses over raw ARKit ──
    # Raw ARKit odometry drifts; a TSDF integrated from it is incoherent
    # (floaters, ghosting). DA3's camera_poses.txt holds the post-loop-closure
    # c2w poses for the keyframes — globally consistent → a coherent mesh.
    poses_src = cam.backend
    if use_refined_poses:
        refined = _load_da3_refined_poses(output_dir, frames_dir)
        if refined:
            cam.pose_map = refined
            poses_src = "da3_loop_closure"
            logger.info(f"[TSDF-scene] using {len(refined)} DA3 loop-closure-"
                        f"refined poses (raw ARKit odometry drifts)")
        else:
            logger.info("[TSDF-scene] no DA3 refined poses found — "
                        "using camera-source poses as-is")

    # Keyframe-index → real keyframe filename, so RGB colour loads the correct
    # frame (poses/depth are keyed by keyframe-sequence index, but the JPGs on
    # disk use the original frame numbers, e.g. kf 1 == 000020.jpg).
    kf_name_map: Dict[int, str] = {}
    _sel = frames_dir / "selected_frames.json"
    if _sel.exists():
        try:
            with open(_sel) as _f:
                _kf = sorted(json.load(_f).get("selected_files", []))
            kf_name_map = {i: n for i, n in enumerate(_kf)}
        except Exception:
            kf_name_map = {}

    # Depth source — pick the BEST available, driven by the reconstruction
    # backend (output/chunk_*_meta.json):
    #   da3_hybrid        → DA3+LiDAR *fused* depth (da3_full/, 504×378) — best
    #   da3 / mapanything → DA3 / VGGT-Long neural depth
    #   da3_lidar / lidar → raw Stray LiDAR (256×192) + confidence
    backend = _read_recon_backend(output_dir)
    from segmentation.shaper_export import _find_stray_dir
    stray_dir = _find_stray_dir(session_dir)
    stray_depth = (_resolve_stray_depth(stray_dir, conf_min=conf_min)
                   if stray_dir is not None else None)
    da3_depth = _resolve_da3_depth(
        output_dir,
        conf_percentile=(da3_conf_percentile if da3_conf_percentile > 0 else None),
    )

    # Depth source: DA3+LiDAR fused calibrated depth (da3_full/) preferred;
    # raw LiDAR as fallback when no fused/neural depth exists.
    if da3_depth is not None:
        depth_loader, (depth_h, depth_w) = da3_depth
        _base = ("DA3+LiDAR fused" if (backend or "") == "da3_hybrid"
                 else "DA3/neural")
        depth_kind = (f"{_base} · conf≥p{da3_conf_percentile:.0f}"
                      if da3_conf_percentile > 0 else _base)
    elif stray_depth is not None:
        depth_loader, (depth_h, depth_w) = stray_depth
        depth_kind = f"raw LiDAR (conf>={conf_min})"
    else:
        logger.error("[TSDF-scene] no depth source found")
        return None
    logger.info(f"[TSDF-scene] recon_backend={backend}  depth={depth_kind}  "
                f"({depth_w}x{depth_h})")

    # Data-driven per-frame source: each frame_<kf>.npz bundles depth +
    # intrinsics + image at the SAME native resolution. Preferred over the
    # scaled-global-K path because it removes all resolution/orientation
    # guessing — the integration uses exactly what DA3 saved for that frame.
    frame_src = _resolve_da3_frame_source(
        output_dir,
        conf_percentile=(da3_conf_percentile if da3_conf_percentile > 0 else None),
    ) if da3_depth is not None else None
    frame_loader = None
    if frame_src is not None:
        frame_loader, (depth_h, depth_w) = frame_src
        logger.info(f"[TSDF-scene] per-frame npz source (depth+K+rgb) "
                    f"@ {depth_w}x{depth_h} — no K rescaling")

    # RGB resolution — only needed for the LEGACY scaled-K fallback below.
    rgb_h: Optional[int] = None
    rgb_w: Optional[int] = None
    for jpg_path in sorted(frames_dir.glob("*.jpg"))[:1]:
        with Image.open(jpg_path) as im:
            rgb_w, rgb_h = im.size
        break
    if rgb_h is None or rgb_w is None:
        # Fall back to first available pose's K to infer.
        for fi in cam.pose_map:
            K = cam.K_for(fi)
            if K is not None:
                rgb_w = int(2 * K[0, 2])
                rgb_h = int(2 * K[1, 2])
                break
        logger.warning(f"[TSDF-scene] no JPG — guessed RGB res ≈ {rgb_w}x{rgb_h}")

    if progress_cb:
        progress_cb("starting", 0.0, None)

    # GPU TSDF via VoxelBlockGrid (cuda) — orders of magnitude faster than the
    # legacy CPU ScalableTSDFVolume, which is unviable for large scans with
    # hundreds of depth maps. Falls back to CPU VBG if no CUDA device.
    import open3d.core as o3c
    _dev_str = "cuda:0" if o3d.core.cuda.is_available() else "CPU:0"
    o3d_device = o3c.Device(_dev_str)
    trunc_mult = max(1.0, float(sdf_trunc) / float(voxel_length))
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=(o3c.SizeVector((1,)), o3c.SizeVector((1,)), o3c.SizeVector((3,))),
        voxel_size=float(voxel_length),
        block_resolution=16,
        block_count=int(tsdf_block_count),
        device=o3d_device,
    )
    logger.info(f"[TSDF-scene] VoxelBlockGrid on {_dev_str} "
                f"(voxel={voxel_length}m, trunc_mult={trunc_mult:.1f}, "
                f"blocks={tsdf_block_count})")

    # ── Option A: restrict integration to pixels present in cleaned_cloud.ply ──
    # cleaned_cloud is the CloudCompPy-cleaned merge (DA3 + LiDAR). Each point
    # carries (frame, pixel) traceability, so we integrate ONLY the depth
    # pixels that produced a surviving cloud point — the TSDF then contains
    # nothing the cleaned cloud doesn't.
    cc_frame_pix = None
    cc_ply_hw = None
    if mask_to_cleaned_cloud:
        from segmentation.pipeline import _load_ply_origins
        cc_path = output_dir / "cleaned_cloud.ply"
        if not cc_path.exists():
            for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
                if (output_dir / alt).exists():
                    cc_path = output_dir / alt
                    break
        origins = _load_ply_origins(cc_path) if cc_path.exists() else None
        if origins is None:
            logger.warning("[TSDF-scene] mask_to_cleaned_cloud on, but no PLY "
                           "traceability found — integrating unmasked")
        else:
            _xyz, fg, pr, pc = origins
            del _xyz
            fg = np.asarray(fg).astype(np.int64)
            pr = np.asarray(pr).astype(np.int32)
            pc = np.asarray(pc).astype(np.int32)
            cc_ply_hw = (int(pr.max()) + 1 if len(pr) else 1,
                         int(pc.max()) + 1 if len(pc) else 1)
            cc_frame_pix = {}
            for f in np.unique(fg):
                m = fg == f
                cc_frame_pix[int(f)] = (pr[m], pc[m])
            logger.info(f"[TSDF-scene] mask_to_cleaned_cloud: {len(fg):,} cloud "
                        f"points, {len(cc_frame_pix)} frames, "
                        f"trace {cc_ply_hw[1]}x{cc_ply_hw[0]}")

    sorted_frames = sorted(cam.pose_map.keys())
    n_integrated = 0
    skipped_no_depth = skipped_no_K = skipped_empty = 0
    native_K_map: Dict[int, np.ndarray] = {}  # per-frame npz K (depth res) for texturing

    for i, fidx in enumerate(sorted_frames):
        fidx = int(fidx)
        c2w = cam.pose_map.get(fidx)
        if c2w is None:
            continue
        # Prefer the data-driven per-frame npz (depth + native K + native rgb,
        # same resolution, zero rescaling). Fall back to the legacy depth_loader
        # + scaled-global-K + re-decoded JPG when the rich source is absent.
        frame_K = None
        frame_rgb = None
        if frame_loader is not None:
            fr = frame_loader(fidx)
            if fr is None:
                skipped_no_depth += 1
                continue
            depth_m, depth_valid = fr["depth"], fr["valid"]
            frame_K, frame_rgb = fr["K"], fr["rgb"]
        else:
            depth_pair = depth_loader(fidx)
            if depth_pair is None:
                skipped_no_depth += 1
                continue
            depth_m, depth_valid = depth_pair

        # Full-frame mask: depth validity + range. No per-instance filtering.
        mask = depth_valid & (depth_m > depth_min) & (depth_m < depth_trunc)

        # Drop flying pixels at depth discontinuities — same heuristic as the
        # per-instance path, prevents "ghost" surfaces between objects and
        # background.
        d = np.where(mask, depth_m, 0.0)
        gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
        gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
        mask &= (gx < edge_thresh) & (gy < edge_thresh)

        # Option A — keep only pixels that produced a cleaned_cloud point.
        if cc_frame_pix is not None:
            fp = cc_frame_pix.get(fidx)
            if fp is None:
                skipped_empty += 1
                continue
            mask &= _build_segment_mask_at_depth(
                fp[0], fp[1], depth_h, depth_w,
                cc_ply_hw[0], cc_ply_hw[1], cleaned_cloud_dilate,
            )

        if not mask.any():
            skipped_empty += 1
            continue

        depth_masked = np.where(mask, depth_m, 0.0).astype(np.float32)

        # Intrinsics: use the frame's OWN K (native to this depth grid) when the
        # rich npz source is active — no rescaling. Otherwise scale the global K
        # from RGB→depth resolution (legacy path).
        if frame_K is not None:
            K_d = frame_K
            native_K_map[fidx] = frame_K  # reuse for texturing (scaled to JPG res)
        else:
            K_rgb = cam.K_for(fidx)
            if K_rgb is None:
                skipped_no_K += 1
                continue
            K_d = _intrinsics_for_depth(K_rgb, rgb_h, rgb_w, depth_h, depth_w)

        # Color: the npz image is already aligned to the depth grid (same
        # source, same resolution). Fall back to re-decoding the full-res JPG.
        if frame_rgb is not None:
            color_np = frame_rgb
        else:
            color_np = _load_rgb_at_depth(frames_dir, fidx, depth_h, depth_w,
                                          frame_name=kf_name_map.get(fidx))

        c2w_4 = np.eye(4, dtype=np.float64)
        c2w_4[:c2w.shape[0], :c2w.shape[1]] = c2w
        extrinsic = np.linalg.inv(c2w_4)  # world → cam

        # GPU integrate: depth/color as device Images; K & extrinsic as host
        # float64 tensors (Open3D requires intrinsics/extrinsics on CPU).
        # VBG.integrate requires (float, float) or (uint16, uint8) for
        # (depth, color) — pass color as float32 to match the float depth.
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_masked, device=o3d_device))
        color_t = o3d.t.geometry.Image(
            o3c.Tensor(np.ascontiguousarray(color_np).astype(np.float32), device=o3d_device))
        K_t = o3c.Tensor(np.ascontiguousarray(K_d), o3c.float64)
        ext_t = o3c.Tensor(np.ascontiguousarray(extrinsic), o3c.float64)
        coords = volume.compute_unique_block_coordinates(
            depth_t, K_t, ext_t, depth_scale=1.0, depth_max=depth_trunc,
            trunc_voxel_multiplier=trunc_mult)
        volume.integrate(coords, depth_t, color_t, K_t, K_t, ext_t,
                         depth_scale=1.0, depth_max=depth_trunc,
                         trunc_voxel_multiplier=trunc_mult)
        n_integrated += 1

        if progress_cb and n_integrated % 50 == 0:
            progress_cb("integrating", time.time() - t0, None)

    if n_integrated == 0:
        logger.error(f"[TSDF-scene] no frames integrated "
                     f"(no_depth={skipped_no_depth} no_K={skipped_no_K} "
                     f"empty={skipped_empty})")
        if progress_cb:
            progress_cb("error", time.time() - t0, None)
        return None

    if progress_cb:
        progress_cb("extracting", time.time() - t0, None)

    # Extract from the GPU grid → legacy mesh for the existing cleanup/texture path.
    mesh = volume.extract_triangle_mesh(
        weight_threshold=float(tsdf_weight_thresh)).to_legacy()
    # VBG's float-color path stores colours in 0-255; legacy meshes and GLB
    # expect 0-1, so normalise (the smoke test confirmed the 0-255 range).
    if mesh.has_vertex_colors():
        _vc = np.asarray(mesh.vertex_colors)
        if _vc.size and _vc.max() > 1.5:
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(_vc / 255.0, 0.0, 1.0))
    mesh.compute_vertex_normals()

    # Optional: drop tiny floating shells (same heuristic as per-instance).
    try:
        tri_clusters, n_per_cluster, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        n_per_cluster = np.asarray(n_per_cluster)
        if len(n_per_cluster) > 1:
            min_tri = max(500, int(0.01 * int(n_per_cluster.max())))
            keep = n_per_cluster >= min_tri
            drop_tris = ~keep[tri_clusters]
            n_drop = int(drop_tris.sum())
            if n_drop > 0:
                mesh.remove_triangles_by_mask(drop_tris)
                mesh.remove_unreferenced_vertices()
                logger.info(f"[TSDF-scene] dropped {int((~keep).sum())} small clusters "
                            f"({n_drop:,} tris)")
    except Exception as e:
        logger.warning(f"[TSDF-scene] cluster cleanup skipped ({e})")

    # ── Phase 3: mesh cleanup ──
    # Fill small LiDAR dropout holes (Open3D tensor API). hole_size caps it so
    # only sensor gaps close — not real openings or the open scan boundary.
    if fill_holes and fill_hole_size > 0:
        try:
            tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            filled = tm.fill_holes(hole_size=float(fill_hole_size)).to_legacy()
            n_new = len(filled.triangles) - len(mesh.triangles)
            # Guard: a runaway fill (e.g. capping the whole room) is rejected.
            if 0 <= n_new <= 0.5 * max(1, len(mesh.triangles)):
                filled.compute_vertex_normals()
                mesh = filled
                if n_new > 0:
                    logger.info(f"[TSDF-scene] hole-fill: +{n_new:,} tris")
            else:
                logger.warning(f"[TSDF-scene] hole-fill rejected "
                               f"(Δ{n_new:,} tris looks runaway) — skipped")
        except Exception as e:
            logger.warning(f"[TSDF-scene] hole-fill skipped ({e})")

    # Smoothing. `smooth_method`: "taubin" = λ/μ alternation, shrinkage-free;
    # "simple" = neighbour averaging (stronger, slight shrinkage). Configurable
    # so the choice is validated visually, not by a proxy metric.
    try:
        if smooth_iterations > 0:
            if smooth_method == "taubin":
                mesh = mesh.filter_smooth_taubin(
                    number_of_iterations=int(smooth_iterations))
            else:
                mesh = mesh.filter_smooth_simple(
                    number_of_iterations=int(smooth_iterations))
            mesh.compute_vertex_normals()
            logger.info(f"[TSDF-scene] smoothing: {smooth_method} "
                        f"×{smooth_iterations}")
    except Exception as e:
        logger.warning(f"[TSDF-scene] smoothing skipped ({e})")

    # Decimate before writing/texturing. The 5 mm marching-cubes mesh is
    # heavily over-tessellated vs the ~1-2 cm real LiDAR detail, and texrecon's
    # per-face MRF needs a sane triangle count. Decimate-then-texture is also
    # how Polycam keeps a light mesh looking sharp (detail lives in the texture).
    n_tri_full = len(mesh.triangles)
    if decimate_target > 0 and n_tri_full > decimate_target:
        try:
            mesh = mesh.simplify_quadric_decimation(int(decimate_target))
            mesh.compute_vertex_normals()
            logger.info(f"[TSDF-scene] decimated {n_tri_full:,} → "
                        f"{len(mesh.triangles):,} tris")
        except Exception as e:
            logger.warning(f"[TSDF-scene] decimation skipped ({e})")

    # Sit under the same ``output/tsdf/`` root as the per-instance meshes so
    # the ``/tsdf/list/`` endpoint (which iterates subfolders) picks it up
    # automatically and the viewport loads it alongside everything else.
    scene_dir = output_dir / "tsdf" / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    glb_path = scene_dir / f"{scene_name}.glb"
    meta_path = glb_path.with_suffix(".meta.json")

    ok = o3d.io.write_triangle_mesh(str(glb_path), mesh, write_ascii=False)
    if not ok:
        logger.error(f"[TSDF-scene] failed to write GLB at {glb_path}")
        if progress_cb:
            progress_cb("error", time.time() - t0, None)
        return None

    n_v, n_t = int(len(mesh.vertices)), int(len(mesh.triangles))

    # ── UV-atlas texture bake (Phase 1B) — overwrites scene.glb with the
    #    textured mesh. On any failure the vertex-colour preview is kept. ──
    textured = False
    if texture:
        if progress_cb:
            progress_cb("texturing", time.time() - t0, None)
        tex_in = scene_dir / "_scene_geom.ply"
        try:
            # Per-view intrinsics for texrecon, at the FULL-RES JPG resolution it
            # normalises against. The native K (npz, preferred) is at depth
            # resolution, so scale depth→jpg — otherwise texrecon's principal
            # point/focal (ppx=cx/img_w …) are computed against the wrong size
            # and every view projects offset, smearing the texture. Same root
            # cause as the integrate intrinsics bug, on the texturing side.
            sx_t = rgb_w / float(depth_w)
            sy_t = rgb_h / float(depth_h)
            tex_intrinsics: Dict[int, np.ndarray] = {}
            for _fidx in sorted_frames:
                _K = native_K_map.get(_fidx)  # per-frame npz K, captured at integrate
                if _K is None:
                    _K = cam.K_for(_fidx)
                if _K is None:
                    continue
                _K = np.asarray(_K, dtype=np.float64).copy()
                _K[0, 0] *= sx_t; _K[0, 2] *= sx_t
                _K[1, 1] *= sy_t; _K[1, 2] *= sy_t
                tex_intrinsics[_fidx] = _K
            # Hand texrecon a robust binary PLY (read natively) rather than
            # round-tripping the geometry through GLB.
            o3d.io.write_triangle_mesh(str(tex_in), mesh, write_ascii=False)
            from reconstruction.texture_bake import bake_texture
            res = bake_texture(
                mesh_path=tex_in,
                frames_dir=frames_dir,
                pose_map=cam.pose_map,
                intrinsics_map=tex_intrinsics,
                out_glb=glb_path,
            )
            textured = res is not None
            if not textured:
                logger.warning("[TSDF-scene] texture bake failed — "
                               "keeping the vertex-colour preview")
        except Exception as e:
            logger.warning(f"[TSDF-scene] texture bake error ({e}) — "
                           "keeping the vertex-colour preview")
        finally:
            tex_in.unlink(missing_ok=True)

    elapsed = time.time() - t0
    meta = {
        "method": "tsdf_scene",
        **({"label": variant_label} if variant_label else {}),
        "color": "uv_texture" if textured else "rgb8_per_vertex",
        "textured": textured,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "depth_min": depth_min,
        "edge_thresh": edge_thresh,
        "conf_min": int(conf_min),
        "da3_conf_percentile": float(da3_conf_percentile),
        "mask_to_cleaned_cloud": bool(mask_to_cleaned_cloud),
        "smooth_iterations": int(smooth_iterations),
        "smooth_method": smooth_method,
        "fill_hole_size": float(fill_hole_size) if fill_holes else 0.0,
        "decimate_target": int(decimate_target),
        "n_triangles_full": int(n_tri_full),
        "n_poses_available": int(len(cam.pose_map)),
        "n_frames_integrated": int(n_integrated),
        "skipped_no_depth": int(skipped_no_depth),
        "skipped_no_K": int(skipped_no_K),
        "skipped_empty_mask": int(skipped_empty),
        "n_vertices": n_v,
        "n_triangles": n_t,
        "elapsed_s": float(elapsed),
        "depth_resolution": [int(depth_w), int(depth_h)],
        "rgb_resolution": [int(rgb_w), int(rgb_h)],
        "backend": cam.backend,
        "recon_backend": backend or "unknown",
        "depth_source": depth_kind,
        "poses_source": poses_src,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = glb_path.stat().st_size / (1024 * 1024)
    logger.info(f"[TSDF-scene] ✅ {n_v:,} verts / {n_t:,} tris  textured={textured}  "
                f"({size_mb:.1f} MB) in {elapsed:.1f}s "
                f"({n_integrated}/{len(sorted_frames)} frames)")

    if progress_cb:
        progress_cb("done", elapsed, str(glb_path))
    return glb_path


# ── Whole-scene mesh via Poisson (Option B — direct from cleaned_cloud) ──────

def export_poisson_scene(
    output_dir: Path,
    frames_dir: Path,
    session_dir: Optional[Path] = None,
    max_points: int = 4_000_000,         # cheap uniform pre-downsample cap (fits 12 GB)
    voxel_downsample: float = 0.005,     # m — downsample the cloud before Poisson
    poisson_depth: int = 10,             # octree depth (resolution)
    density_quantile: float = 0.03,      # trim the lowest-density Poisson vertices
    decimate_target: int = 1_500_000,
    texture: bool = True,
    use_refined_poses: bool = True,
    progress_cb: Optional[Callable[[str, Optional[float], Optional[str]], None]] = None,
) -> Optional[Path]:
    """Whole-scene mesh via screened Poisson reconstruction from cleaned_cloud.ply.

    Option B sibling of ``export_tsdf_scene``: reconstructs the surface directly
    from the CloudCompPy-cleaned point cloud (DA3+LiDAR merge) rather than
    integrating depth. Output: ``output_dir/tsdf/scene_poisson/scene_poisson.glb``
    — its own folder so it sits alongside the TSDF scene for A/B comparison.
    """
    import open3d as o3d

    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)
    if session_dir is None:
        session_dir = frames_dir.parent
    session_dir = Path(session_dir)

    t0 = time.time()
    logger.info(f"[Poisson-scene] start  output_dir={output_dir}")
    logger.info(f"[Poisson-scene] params  voxel_down={voxel_downsample}m  "
                f"depth={poisson_depth}  density_q={density_quantile}")

    # ── source cloud ──
    cc_path = output_dir / "cleaned_cloud.ply"
    if not cc_path.exists():
        for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                cc_path = output_dir / alt
                break
    if not cc_path.exists():
        logger.error("[Poisson-scene] cleaned_cloud.ply not found — aborting")
        return None

    if progress_cb:
        progress_cb("starting", 0.0, None)
    pcd = o3d.io.read_point_cloud(str(cc_path))
    n_loaded = len(pcd.points)
    logger.info(f"[Poisson-scene] loaded {n_loaded:,} pts from {cc_path.name}")
    if n_loaded == 0:
        logger.error("[Poisson-scene] empty cloud — aborting")
        return None
    # Cheap, memory-light pre-downsample FIRST. uniform_down_sample is a plain
    # stride select — no hash, no blow-up — unlike voxel_down_sample, which
    # OOMs a 12 GB box on the raw ~44 M-point cleaned_cloud. Cut to max_points
    # before any hash-based op or Poisson.
    if max_points > 0 and n_loaded > max_points:
        k = (n_loaded + max_points - 1) // max_points
        pcd = pcd.uniform_down_sample(k)
        logger.info(f"[Poisson-scene] uniform pre-downsample x1/{k} → "
                    f"{len(pcd.points):,} pts")
    if voxel_downsample > 0:
        pcd = pcd.voxel_down_sample(voxel_downsample)
        logger.info(f"[Poisson-scene] voxel-down {voxel_downsample}m → "
                    f"{len(pcd.points):,} pts")
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)

    # ── Poisson surface reconstruction ──
    if progress_cb:
        progress_cb("integrating", time.time() - t0, None)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth))
    densities = np.asarray(densities)
    if 0.0 < density_quantile < 1.0 and len(densities):
        thr = float(np.quantile(densities, density_quantile))
        mesh.remove_vertices_by_mask(densities < thr)
    mesh.compute_vertex_normals()
    logger.info(f"[Poisson-scene] poisson depth={poisson_depth} → "
                f"{len(mesh.triangles):,} tris (trimmed q{density_quantile})")

    # ── drop tiny floating shells ──
    try:
        tri_cl, n_per, _ = mesh.cluster_connected_triangles()
        tri_cl = np.asarray(tri_cl)
        n_per = np.asarray(n_per)
        if len(n_per) > 1:
            min_tri = max(500, int(0.01 * int(n_per.max())))
            drop = (~(n_per >= min_tri))[tri_cl]
            if drop.any():
                mesh.remove_triangles_by_mask(drop)
                mesh.remove_unreferenced_vertices()
                logger.info(f"[Poisson-scene] dropped small clusters "
                            f"({int(drop.sum()):,} tris)")
    except Exception as e:
        logger.warning(f"[Poisson-scene] cluster cleanup skipped ({e})")

    n_full = len(mesh.triangles)
    if decimate_target > 0 and n_full > decimate_target:
        try:
            mesh = mesh.simplify_quadric_decimation(int(decimate_target))
            mesh.compute_vertex_normals()
            logger.info(f"[Poisson-scene] decimated {n_full:,} → "
                        f"{len(mesh.triangles):,} tris")
        except Exception as e:
            logger.warning(f"[Poisson-scene] decimation skipped ({e})")

    # ── output — own folder, alongside the TSDF scene ──
    scene_dir = output_dir / "tsdf" / "scene_poisson"
    scene_dir.mkdir(parents=True, exist_ok=True)
    glb_path = scene_dir / "scene_poisson.glb"
    meta_path = glb_path.with_suffix(".meta.json")
    if not o3d.io.write_triangle_mesh(str(glb_path), mesh, write_ascii=False):
        logger.error(f"[Poisson-scene] GLB write failed at {glb_path}")
        if progress_cb:
            progress_cb("error", time.time() - t0, None)
        return None
    n_v, n_t = int(len(mesh.vertices)), int(len(mesh.triangles))

    # ── UV-atlas texture (texrecon) — same path as the TSDF scene ──
    textured = False
    if texture:
        if progress_cb:
            progress_cb("texturing", time.time() - t0, None)
        cam = _load_camera_source(session_dir, output_dir)
        if cam is not None and use_refined_poses:
            refined = _load_da3_refined_poses(output_dir, frames_dir)
            if refined:
                cam.pose_map = refined
        if cam is not None:
            tex_in = scene_dir / "_poisson_geom.ply"
            try:
                o3d.io.write_triangle_mesh(str(tex_in), mesh, write_ascii=False)
                from reconstruction.texture_bake import bake_texture
                res = bake_texture(
                    mesh_path=tex_in, frames_dir=frames_dir,
                    pose_map=cam.pose_map, intrinsics_map=cam.intrinsics_map,
                    out_glb=glb_path,
                )
                textured = res is not None
                if not textured:
                    logger.warning("[Poisson-scene] texture bake failed — "
                                   "keeping the vertex-colour preview")
            except Exception as e:
                logger.warning(f"[Poisson-scene] texture bake error ({e})")
            finally:
                tex_in.unlink(missing_ok=True)

    elapsed = time.time() - t0
    meta = {
        "method": "poisson_scene",
        "label": "🟣 Poisson",
        "color": "uv_texture" if textured else "rgb_per_vertex",
        "textured": textured,
        "source": cc_path.name,
        "max_points": int(max_points),
        "voxel_downsample": voxel_downsample,
        "poisson_depth": int(poisson_depth),
        "density_quantile": density_quantile,
        "decimate_target": int(decimate_target),
        "n_vertices": n_v,
        "n_triangles": n_t,
        "elapsed_s": float(elapsed),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = glb_path.stat().st_size / (1024 * 1024)
    logger.info(f"[Poisson-scene] ✅ {n_v:,} verts / {n_t:,} tris  "
                f"textured={textured}  ({size_mb:.1f} MB) in {elapsed:.1f}s")
    if progress_cb:
        progress_cb("done", elapsed, str(glb_path))
    return glb_path
