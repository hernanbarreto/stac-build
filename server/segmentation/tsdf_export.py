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
from segmentation.session_io import (
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


def _resolve_mapanything_depth(output_dir: Path, conf_percentile: Optional[float] = None
                               ) -> Optional[Tuple[Callable[[int], Optional[dict]],
                                                   Tuple[int, int]]]:
    """VGGT-Long ('mapanything' backend): per-frame depth lives INSIDE the per-chunk
    dicts ``maplong_run/_tmp_results_unaligned/chunk_K.npy`` (keys: ``depth`` (S,H,W),
    ``intrinsic`` (S,3,3), optional ``conf``). ``frame_list.json`` lists the exact
    ordered frames the backend processed (after any keyframe filter + stride), so we
    map each REAL frame number → (chunk K, frame_local) and load depth + native K on
    demand. Returns a per-frame loader dict ``{depth, valid, K, rgb, hw}`` keyed by
    REAL frame number. ``rgb`` is None → the caller re-decodes ``{frame:06d}.jpg``,
    which is correct because the key IS the real frame number.
    """
    run_dir = output_dir / "maplong_run"
    # ALIGNED chunks (kept after postproc); depth/conf/intrinsic identical to unaligned.
    chunks = run_dir / "_tmp_results_aligned"
    if not chunks.exists():
        chunks = run_dir / "_tmp_results_unaligned"   # legacy fallback
    flp = run_dir / "frame_list.json"
    if not flp.exists():
        flp = output_dir / "frame_list.json"
    if not chunks.exists() or not flp.exists():
        return None
    try:
        names = json.loads(flp.read_text())
    except Exception:
        return None
    import re as _re
    frame_to_pos: Dict[int, int] = {}
    for p, n in enumerate(names):
        m = _re.search(r"(\d+)", str(n))
        if m:
            frame_to_pos.setdefault(int(m.group(1)), p)
    if not frame_to_pos:
        return None

    # chunk_step (= chunk_size - overlap) — from our origin meta sidecar; else the
    # vendor config copy. Needed to map list position → (chunk K, frame_local).
    chunk_step: Optional[int] = None
    for mp in sorted(output_dir.glob("chunk_*_meta.json")):
        try:
            chunk_step = int(json.load(open(mp)).get("chunk_step"))
            if chunk_step:
                break
        except Exception:
            continue
    if not chunk_step:
        # The vendor config copy is named per backend: mapanything → vggt_long_config.yaml,
        # vggtomega → vggt_omega_config.yaml. Try BOTH — looking only for vggt_long_config
        # made this return None for vggtomega, so the TSDF fell back to DA3 depth (a
        # DIFFERENT model than the omega cloud) → mesh ~1.6× off / displaced vs the cloud.
        import yaml as _yaml
        for _cfg in ("vggt_long_config.yaml", "vggt_omega_config.yaml"):
            try:
                c = _yaml.safe_load(open(run_dir / _cfg))
                chunk_step = int(c["Model"]["chunk_size"]) - int(c["Model"]["overlap"])
                if chunk_step:
                    break
            except Exception:
                chunk_step = None
    if not chunk_step or chunk_step < 1:
        return None

    def _load_chunk(K: int):
        p = chunks / f"chunk_{K}.npy"
        if not p.exists():
            return None
        try:
            return np.load(str(p), allow_pickle=True).item()
        except Exception:
            return None

    def _depth_stack(arr):
        a = np.asarray(arr)
        while a.ndim > 3:
            a = a[0]
        if a.ndim == 2:
            a = a[None]
        return a

    probe = _load_chunk(0)
    if not isinstance(probe, dict) or "depth" not in probe:
        return None
    d0 = _depth_stack(probe["depth"])
    h, w = int(d0.shape[-2]), int(d0.shape[-1])

    # How many chunks exist (chunk_0 … chunk_{n-1}). The LAST chunk is longer than
    # chunk_step (it keeps its full overlap tail with no successor), so its tail frames
    # live at positions ≥ n_chunks*chunk_step. The naive K = pos // chunk_step sends those
    # to a non-existent chunk → they get NO depth and are skipped, leaving the mesh short
    # of the cloud's end (test3: 50 of 230 frames dropped → last section uncovered). Clamp
    # K to the last chunk so the tail is read from it (local index stays < its depth len).
    n_chunks = 0
    while (chunks / f"chunk_{n_chunks}.npy").exists():
        n_chunks += 1
    if n_chunks == 0:
        return None

    cache: Dict[str, object] = {"K": None, "data": None}

    def _load(frame_idx: int) -> Optional[dict]:
        pos = frame_to_pos.get(int(frame_idx))
        if pos is None:
            return None
        K = min(pos // chunk_step, n_chunks - 1)
        local = pos - K * chunk_step
        if cache["K"] != K:
            data = _load_chunk(K)
            if data is None:
                return None
            cache["K"], cache["data"] = K, data
        cd = cache["data"]
        depth = _depth_stack(cd.get("depth"))
        if local >= depth.shape[0]:
            return None
        d = depth[local].astype(np.float32)
        valid = np.ones_like(d, dtype=bool)
        conf = cd.get("conf", cd.get("depth_conf"))
        if conf_percentile is not None and conf is not None:
            cstack = _depth_stack(conf)
            if local < cstack.shape[0] and cstack[local].shape == d.shape:
                cf = cstack[local].astype(np.float32)
                thr = float(np.percentile(cf, conf_percentile))
                valid = cf >= thr
        K_intr = None
        ki = cd.get("intrinsic", cd.get("intrinsics"))
        if ki is not None:
            ki = np.asarray(ki, dtype=np.float64)
            while ki.ndim > 3:
                ki = ki[0]
            if ki.ndim == 3 and local < ki.shape[0]:
                K_intr = ki[local].reshape(3, 3)
            elif ki.shape == (3, 3):
                K_intr = ki.reshape(3, 3)
        return {"depth": d, "valid": valid, "K": K_intr, "rgb": None, "hw": d.shape}

    return _load, (h, w)


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


def _rasterize_cloud_depth(xyz: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                           depth_h: int, depth_w: int,
                           splat_radius: int = 0) -> np.ndarray:
    """Z-buffer a frame's cleaned-cloud points back into its depth grid.

    This is the faithful inverse of how the cloud was built, and the core of the
    "TSDF must copy the cloud" fix. Integrating the RAW neural/LiDAR depth fuses
    the very outliers (flying pixels at silhouettes, far depth guesses) that
    CloudCompPy's SOR + noise filter already stripped from ``cleaned_cloud.ply``,
    so the mesh grows streaks and phantom surfaces the cloud doesn't have. Here
    we instead project the SURVIVING cloud points this frame observed through its
    own pose+K and keep the nearest z per pixel — every depth sample IS a cleaned
    cloud point, so the TSDF can only ever mesh what the cloud contains.

    Projecting + unprojecting with the SAME (pose, K) is self-consistent
    regardless of any global pose drift: Open3D unprojects pixel (u,v) at depth z
    back to exactly the world point we projected. The stored traceability pixels
    are therefore only used upstream to decide WHICH points a frame saw, never to
    place them.

    Args:
        xyz: (N,3) world points this frame observed (a cleaned-cloud subset).
        c2w: (≤4,4) camera-to-world pose.
        K:   (3,3) pinhole intrinsics at the (depth_h, depth_w) grid.
        splat_radius: optional px splat (same z) to fill single-point gaps so the
            rasterized depth matches the cloud's surface density.

    Returns (depth_h, depth_w) float32 depth in metres, 0 where empty.
    """
    empty = np.zeros((depth_h, depth_w), dtype=np.float32)
    if xyz is None or len(xyz) == 0:
        return empty
    c2w_4 = np.eye(4, dtype=np.float64)
    c2w_4[:c2w.shape[0], :c2w.shape[1]] = c2w
    w2c = np.linalg.inv(c2w_4)
    cam = np.asarray(xyz, dtype=np.float64) @ w2c[:3, :3].T + w2c[:3, 3]  # world→cam
    z = cam[:, 2]
    front = z > 1e-6
    if not front.any():
        return empty
    cam, z = cam[front], z[front]
    u = (K[0, 0] * cam[:, 0]) / z + K[0, 2]
    v = (K[1, 1] * cam[:, 1]) / z + K[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inb = (ui >= 0) & (ui < depth_w) & (vi >= 0) & (vi < depth_h)
    if not inb.any():
        return empty
    ui, vi, z = ui[inb], vi[inb], z[inb]

    depth = np.full((depth_h, depth_w), np.inf, dtype=np.float64)
    if splat_radius and splat_radius > 0:
        for dy in range(-splat_radius, splat_radius + 1):
            for dx in range(-splat_radius, splat_radius + 1):
                yy, xx = vi + dy, ui + dx
                ok = (yy >= 0) & (yy < depth_h) & (xx >= 0) & (xx < depth_w)
                np.minimum.at(depth, (yy[ok], xx[ok]), z[ok])  # nearest-z wins
    else:
        np.minimum.at(depth, (vi, ui), z)
    depth[np.isinf(depth)] = 0.0
    return depth.astype(np.float32)


def _joint_bilateral_upsample_depth(
        depth_lo: np.ndarray, valid_lo: Optional[np.ndarray],
        guide_hi: np.ndarray, factor: int,
        radius: int = 2, sigma_space: float = 2.0, sigma_range: float = 0.06,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Edge-aware ×``factor`` depth upsample guided by the full-res photo (A).

    The neural depth (≈518×294) is the geometry ceiling — the TSDF can't be sharper
    than it. This lifts that ceiling WITHOUT re-running recon: nearest-upsample the
    low-res depth to ``factor``× and cross-bilateral filter it using the high-res
    RGB as a guide, so depth EDGES snap to image edges (no bilinear smear across
    silhouettes) while flat regions are smoothed. Classic joint-bilateral upsampling
    (Kopf et al.), in its filtering form, on the GPU via torch.

    Hallucination-free: each output pixel averages ONLY valid upsampled depth
    samples, weighted by guide similarity — where no valid neighbour shares the
    pixel's colour, the output stays 0 (invalid), so it never invents geometry.

    Args:
        depth_lo: (h,w) float32 metres, 0 = invalid.
        valid_lo: (h,w) bool or None (None → depth>0).
        guide_hi: (H,W,3) or (H,W) uint8 photo at the TARGET resolution (H=factor·h,
                  W=factor·w). The edge guide AND the integration colour.
        factor:   integer upsample factor (≥2).
        radius / sigma_space / sigma_range: bilateral window + falloffs (sigma_range
                  on 0..1 guide intensity).

    Returns (depth_hi (H,W) float32, valid_hi (H,W) bool), or None on any failure
    (caller falls back to the native-resolution depth).
    """
    try:
        import torch
        import torch.nn.functional as F
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        h, w = depth_lo.shape
        H, W = h * factor, w * factor
        if guide_hi.shape[0] != H or guide_hi.shape[1] != W:
            return None  # guide must already be at the target grid

        d = torch.from_numpy(np.ascontiguousarray(depth_lo)).to(dev).float()[None, None]
        if valid_lo is None:
            v = (d > 0).float()
        else:
            v = torch.from_numpy(np.ascontiguousarray(valid_lo)).to(dev).float()[None, None]
            v = v * (d > 0).float()

        d_nn = F.interpolate(d, size=(H, W), mode="nearest")
        v_nn = F.interpolate(v, size=(H, W), mode="nearest")
        d_nn = d_nn * v_nn  # zero the invalid samples so they never bias the average

        g = torch.from_numpy(np.ascontiguousarray(guide_hi)).to(dev).float()
        if g.ndim == 3:
            g = g.mean(dim=2)
        g = (g / 255.0)[None, None]

        k = 2 * radius + 1
        off = torch.arange(-radius, radius + 1, device=dev).float()
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        w_space = torch.exp(-(oy ** 2 + ox ** 2) / (2 * sigma_space ** 2)).reshape(1, k * k, 1, 1)

        d_u = F.unfold(d_nn, k, padding=radius).reshape(1, k * k, H, W)
        v_u = F.unfold(v_nn, k, padding=radius).reshape(1, k * k, H, W)
        g_u = F.unfold(g, k, padding=radius).reshape(1, k * k, H, W)
        w_range = torch.exp(-((g_u - g) ** 2) / (2 * sigma_range ** 2))
        wgt = w_space * w_range * v_u
        num = (wgt * d_u).sum(dim=1)
        den = wgt.sum(dim=1)
        out = torch.where(den > 1e-6, num / den, torch.zeros_like(num))
        valid = (den > 1e-6) & (out > 0)
        return (out[0].detach().cpu().numpy().astype(np.float32),
                valid[0].detach().cpu().numpy())
    except Exception as e:
        logger.warning(f"[TSDF-scene] guided depth upsample failed ({e}) — "
                       "falling back to native-resolution depth")
        return None


def _group_rows_by_frame(fg: np.ndarray, pr: np.ndarray, pc: np.ndarray,
                         xyz: np.ndarray) -> Dict[int, tuple]:
    """Group per-point ``(pr, pc, xyz)`` by frame with ONE argsort.

    The obvious ``for f in unique(fg): m = fg == f`` does a full 67M-element scan
    PER frame → O(frames · N) on a single core (minutes, no GPU, no progress — the
    "everything at 0%" stall). Sorting once by frame and slicing contiguous runs is
    O(N log N): seconds for the same 67M points × ~1500 frames.
    Returns ``{frame: (pr_slice, pc_slice, xyz_slice)}``.
    """
    if len(fg) == 0:
        return {}
    order = np.argsort(fg, kind="stable")
    fs = fg[order]
    uniq, starts = np.unique(fs, return_index=True)
    ends = np.append(starts[1:], len(fs))
    pr_s, pc_s, xyz_s = pr[order], pc[order], xyz[order]
    return {int(uniq[i]): (pr_s[starts[i]:ends[i]],
                           pc_s[starts[i]:ends[i]],
                           xyz_s[starts[i]:ends[i]])
            for i in range(len(uniq))}


def _load_ply_confidence(ply_path: Path) -> Optional[np.ndarray]:
    """Read the per-point ``confidence`` scalar from a binary PLY (same layout as
    ``_load_ply_origins``). Returns an (N,) float32 array, or None if the field is
    absent. Used by the TSDF confidence gate to drop the noisy low/mid-confidence
    tail before meshing."""
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    try:
        with open(ply_path, 'rb') as f:
            n_pts = 0
            props = []
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_pts = int(line.split()[-1])
                elif line.startswith('property') and n_pts > 0:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] in _ply_type:
                        props.append((parts[2], _ply_type[parts[1]]))
                elif line == 'end_header':
                    break
            if n_pts == 0 or 'confidence' not in {p[0] for p in props}:
                return None
            data = np.frombuffer(f.read(), dtype=np.dtype(props))
            return data['confidence'].astype(np.float32)
    except Exception as e:
        logger.warning(f"[TSDF] could not read confidence from {ply_path}: {e}")
        return None


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

    # selected_frames.json is OPTIONAL: when camera_frames.txt is present (backends
    # that record the exact processed frames, e.g. mapanything/VGGT-Long), poses are
    # keyed by real frame number from that sidecar — no keyframe list needed.
    sel_json = frames_dir / "selected_frames.json"
    kf_files: List[str] = []
    if sel_json.exists():
        try:
            with open(sel_json) as f:
                kf_files = sorted(json.load(f).get("selected_files", []))
        except Exception:
            kf_files = []

    mats: List[np.ndarray] = []
    with open(poses_txt) as f:
        for line in f:
            vals = line.split()
            if len(vals) == 16:
                mats.append(np.array([float(v) for v in vals],
                                     dtype=np.float64).reshape(4, 4))

    # camera_frames.txt sidecar (written for mapanything) gives the REAL frame number
    # per pose line → key poses by that (matches the per-point frame_global). This is
    # the all-frames-indexed path.
    frames_txt = poses_txt.parent / "camera_frames.txt"
    if not frames_txt.exists():
        frames_txt = output_dir / "camera_frames.txt"
    if frames_txt.exists():
        try:
            nums = [int(x) for x in open(frames_txt).read().split()]
        except Exception:
            nums = []
        if len(nums) == len(mats) and mats:
            logger.info(f"[TSDF-scene] refined poses keyed by frame number "
                        f"(camera_frames.txt, {len(mats)} frames)")
            return {nums[i]: mats[i] for i in range(len(mats))}

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


# ── Asymmetric design: DA3 dense fusion (DA3 over all blur-valid frames, VGGT
#    keyframes for the loop-closed poses) ─────────────────────────────────────

def _list_da3_frame_indices(output_dir: Path) -> List[int]:
    """Real frame numbers DA3 produced dense depth for — the npz keys in
    ``da3_run/results_output/frame_<N>.npz``. This is the DENSE fusion frame set
    (all blur-valid frames), a superset of the VGGT keyframes."""
    import re as _re
    for d in (output_dir / "da3_run" / "results_output",
              output_dir / "results_output",
              output_dir / "gaus_slam_run" / "results_output"):
        if not d.exists():
            continue
        ids = []
        for p in d.glob("frame_*.npz"):
            m = _re.search(r"frame_(\d+)\.npz", p.name)
            if m:
                ids.append(int(m.group(1)))
        if ids:
            return sorted(ids)
    return []


def _interpolate_poses(pose_map: Dict[int, np.ndarray],
                       target_frames: List[int]) -> Dict[int, np.ndarray]:
    """Fill poses for the DENSE frames that have no VGGT keyframe pose, by
    interpolating the loop-closed keyframe trajectory (SLERP on rotation, linear
    on translation, by real frame number). Keyframes are chosen with high overlap
    (DINO 0.975), so inter-keyframe motion is small and interpolation is accurate.
    Frames outside the keyframe span clamp to the nearest keyframe (extrapolation
    is unsafe). Returns the keyframe poses PLUS the interpolated ones, all 4x4 c2w."""
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except Exception as e:
        logger.warning(f"[TSDF-scene] scipy unavailable for pose interpolation ({e}) "
                       "— dense frames without a keyframe pose will be skipped")
        return {int(k): np.asarray(v, dtype=np.float64) for k, v in pose_map.items()}

    def _to4(m):
        M = np.eye(4, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        M[:m.shape[0], :m.shape[1]] = m
        return M

    kf = sorted(int(k) for k in pose_map.keys())
    out: Dict[int, np.ndarray] = {int(k): _to4(pose_map[k]) for k in kf}
    if len(kf) < 2:
        return out
    kf_arr = np.asarray(kf, dtype=np.float64)
    rots = Rotation.from_matrix([out[k][:3, :3] for k in kf])
    slerp = Slerp(kf_arr, rots)
    trans = {k: out[k][:3, 3] for k in kf}
    n_interp = 0
    for t in target_frames:
        t = int(t)
        if t in out:
            continue
        if t <= kf[0]:
            out[t] = out[kf[0]].copy(); continue
        if t >= kf[-1]:
            out[t] = out[kf[-1]].copy(); continue
        j = int(np.searchsorted(kf_arr, t))     # kf[j-1] < t < kf[j]
        k0, k1 = kf[j - 1], kf[j]
        a = (t - k0) / float(k1 - k0)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = slerp([float(t)]).as_matrix()[0]
        M[:3, 3] = (1.0 - a) * trans[k0] + a * trans[k1]
        out[t] = M
        n_interp += 1
    logger.info(f"[TSDF-scene] pose interpolation: {len(kf)} keyframe poses → "
                f"{len(out)} frames (+{n_interp} interpolated for DA3 dense fusion)")
    return out


def _depth_axis_world_coord(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray,
                            axis: int) -> np.ndarray:
    """Per-pixel WORLD coordinate along ``axis`` of each depth sample, for the
    spatial tile crop in DA3-dense mode (the cloud-pixel mask can't cover the
    non-keyframe dense frames, so each tile keeps only the depth that unprojects
    into its axis-slab). Returns (H,W) float64; meaningless where depth==0."""
    H, W = depth.shape
    vv, uu = np.indices((H, W))
    z = depth.astype(np.float64)
    x = (uu - K[0, 2]) * z / K[0, 0]
    y = (vv - K[1, 2]) * z / K[1, 1]
    c2w = np.asarray(c2w, dtype=np.float64)
    return c2w[axis, 0] * x + c2w[axis, 1] * y + c2w[axis, 2] * z + c2w[axis, 3]


def _depth_world_xyz(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Per-pixel WORLD (x,y,z) of each depth sample → (H,W,3) float64, for the 3D
    CUBE crop (tsdf_tile_length_m): a dense frame keeps only the depth that
    unprojects INTO this cube's box on ALL three axes. Meaningless where depth==0."""
    H, W = depth.shape
    vv, uu = np.indices((H, W))
    z = depth.astype(np.float64)
    x = (uu - K[0, 2]) * z / K[0, 0]
    y = (vv - K[1, 2]) * z / K[1, 1]
    c2w = np.asarray(c2w, dtype=np.float64)
    out = np.empty((H, W, 3), dtype=np.float64)
    for a in range(3):
        out[..., a] = c2w[a, 0] * x + c2w[a, 1] * y + c2w[a, 2] * z + c2w[a, 3]
    return out


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
    from segmentation.session_io import _find_stray_dir
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

def _drop_long_edge_tris(mesh, max_edge: float, logger_prefix: str = ""):
    """Remove triangles with any edge longer than max_edge metres — the spurious
    'bridge'/spike triangles (e.g. near↔far) that sparse regions + decimation create.
    Real TSDF surfaces have edges ≈ voxel size, so this only touches the garbage."""
    if not max_edge or max_edge <= 0 or len(mesh.triangles) == 0:
        return mesh
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    e0 = np.linalg.norm(v[t[:, 0]] - v[t[:, 1]], axis=1)
    e1 = np.linalg.norm(v[t[:, 1]] - v[t[:, 2]], axis=1)
    e2 = np.linalg.norm(v[t[:, 2]] - v[t[:, 0]], axis=1)
    bad = (e0 > max_edge) | (e1 > max_edge) | (e2 > max_edge)
    n = int(bad.sum())
    if n > 0:
        mesh.remove_triangles_by_mask(bad)
        mesh.remove_unreferenced_vertices()
        logger.info(f"[TSDF-scene]{logger_prefix} dropped {n:,} long-edge tris "
                    f"(>{max_edge:.2f}m — bridge/spike lines)")
    return mesh


def _icp_mesh_to_cloud(mesh, output_dir: Path):
    """Compute the gated rigid transform that registers the (untextured) TSDF mesh onto the
    cleaned cloud. Returns ``(T, metrics)`` where T is a 4x4 np.ndarray to apply to the mesh
    AND to the camera poses (so texrecon then bakes texture in the cloud frame), or
    ``(None, metrics)`` if the snap is skipped.

    Why: the cloud (CloudCompPy of omega world_points) and the mesh (TSDF of omega
    depth+poses) are SUPPOSED to share omega's aligned frame, but an intermittent
    loop-closure inconsistency can leave the mesh rigidly offset — measured on test3: 11.46°
    + 0.57 m, while test7 (same code) was perfect. The cloud is the authoritative reference
    (what the user aligns to BIM, what the viewer floor-levels), so we snap the mesh onto it.

    Done BEFORE texturing and applied to BOTH the mesh vertices and the camera poses: a
    rigid move of mesh+cameras together leaves the texrecon projection geometry unchanged,
    so the texture bakes correctly and the output is already in the cloud frame with BAKED
    vertices (no post-texture GLB round-trip, which would corrupt texrecon's UV atlas, and
    no node-matrix that BIM-side trimesh/open3d readers might ignore).

    GATED — only when the fit is good AND the move is small enough to be a genuine
    correction, never a large ICP slide along a repetitive railway, and only if it improves.
    """
    metrics = {"applied": False, "fitness": 0.0, "rmse": None,
               "rot_deg": None, "trans_m": None, "reason": ""}
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                cloud_path = output_dir / alt
                break
    try:
        if not cloud_path.exists():
            metrics["reason"] = "no cleaned_cloud.ply"; return None, metrics
        import open3d as o3d  # local import — heavy, matches the rest of this module
        from scipy.spatial import cKDTree as _KDT
        tgt = o3d.io.read_point_cloud(str(cloud_path))
        if len(tgt.points) == 0:
            metrics["reason"] = "empty cloud"; return None, metrics
        V = np.asarray(mesh.vertices)
        if len(V) < 100:
            metrics["reason"] = "mesh too small"; return None, metrics
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(V.astype(np.float64))
        tgt_d = tgt.voxel_down_sample(0.05)
        src_d = src.voxel_down_sample(0.05)
        A = np.asarray(src_d.points); B = np.asarray(tgt_d.points)
        if len(A) < 100 or len(B) < 100:
            metrics["reason"] = "too few points after downsample"; return None, metrics

        def _median_nn(P, Q):
            return float(np.median(_KDT(Q).query(P, k=1)[0]))
        before = _median_nn(A, B)
        reg = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, 0.6, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
        T = np.asarray(reg.transformation)
        R, t = T[:3, :3], T[:3, 3]
        rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        trans_m = float(np.linalg.norm(t))
        after = _median_nn((R @ A.T).T + t, B)
        metrics.update(fitness=float(reg.fitness), rmse=float(reg.inlier_rmse),
                       rot_deg=rot_deg, trans_m=trans_m,
                       before_median=before, after_median=after)
        # Gates: good fit, genuine (small) correction, and it must actually improve.
        if reg.fitness < 0.7:
            metrics["reason"] = f"low fitness {reg.fitness:.2f}"; return None, metrics
        if reg.inlier_rmse > 0.06:
            metrics["reason"] = f"high rmse {reg.inlier_rmse:.3f}"; return None, metrics
        if rot_deg > 30.0 or trans_m > 2.5:
            metrics["reason"] = (f"transform too large ({rot_deg:.1f}deg/{trans_m:.2f}m) — "
                                 "refusing (possible ICP slide)"); return None, metrics
        if after > before + 1e-4:
            metrics["reason"] = f"would not improve ({before:.3f}->{after:.3f} m)"
            return None, metrics
        metrics["applied"] = True
        metrics["reason"] = "ok"
        return T, metrics
    except Exception as e:
        metrics["reason"] = f"error: {e}"
        return None, metrics


def _compress_scene_glb(glb_path: Path) -> None:
    """Best-effort shrink of a textured scene .glb in place (meshopt + WebP).

    A textured TSDF scene is ~270 MB (243 PNG atlases + ~4 M tris); served raw it
    floods the browser connection pool and renders only after a long stall. This
    runs tools/glb/compress_glb.mjs (≈6× smaller) — the viewer's GLTFLoader is
    wired with a Meshopt decoder, so the compressed output loads transparently.

    Never fatal: if node / the script / its node_modules are missing, or it errors,
    we log and keep the original uncompressed .glb (the script writes atomically,
    so a failure leaves the original intact).
    """
    import shutil
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tools" / "glb" / "compress_glb.mjs"
    node = shutil.which("node") or "/workspace/miniforge3/envs/nodejs/bin/node"
    if not script.exists():
        logger.warning(f"[TSDF-scene] GLB compressor not found at {script} — serving raw .glb")
        return
    if not (script.parent / "node_modules").exists():
        logger.warning("[TSDF-scene] tools/glb/node_modules missing (run `npm install` there) — serving raw .glb")
        return
    if not Path(node).exists():
        logger.warning("[TSDF-scene] node not found — serving raw .glb")
        return
    try:
        before = glb_path.stat().st_size / (1024 * 1024)
        r = subprocess.run(
            [node, str(script), str(glb_path), str(glb_path), "2048", "85"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            logger.warning(f"[TSDF-scene] GLB compression failed (rc={r.returncode}): "
                           f"{(r.stderr or r.stdout)[-500:]} — serving raw .glb")
            return
        after = glb_path.stat().st_size / (1024 * 1024)
        logger.info(f"[TSDF-scene] 🗜️  GLB compressed {before:.1f} MB → {after:.1f} MB "
                    f"(meshopt + WebP)")
    except Exception as e:
        logger.warning(f"[TSDF-scene] GLB compression error ({e}) — serving raw .glb")


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
                                         # (legacy raw-depth path only)
    rasterize_cloud_depth: bool = True,  # FAITHFUL mode: integrate the cloud's OWN points
                                         # (z-buffered back into each frame) instead of the
                                         # raw neural depth, so the TSDF meshes EXACTLY the
                                         # cleaned cloud — no SOR-stripped outliers/streaks/
                                         # phantoms. Needs PLY traceability; else falls back.
    cloud_splat_radius: int = 1,         # px splat (same z) when rasterizing — fills single-
                                         # point gaps so coverage matches the cloud's density
    conf_min_norm: float = 0.0,          # confidence gate in [0,1] — SAME min-max normalisation
                                         # as the UI slider: keep cloud points whose
                                         # (conf-min)/(max-min) ≥ this before meshing. 0 = off.
                                         # 0.1 ≈ the UI slider at 0.1 — drops the noisy low/mid-
                                         # confidence tail (vertical smear). Needs the cloud's
                                         # 'confidence' field (present in cleaned_cloud.ply).
    upsample_depth: int = 1,             # A — guided ×N joint-bilateral upsample of the neural
                                         # depth using the full-res photo as an edge guide,
                                         # integrated at the finer grid. Lifts the depth-
                                         # resolution ceiling (sharper geometry at distance)
                                         # without re-running recon. 1 = off. Dense-depth path
                                         # only (moot in raster mode).
    smooth_iterations: int = 2,          # smoothing iterations post-extract
    smooth_method: str = "simple",       # winner of visual A/B over taubin (LiDAR-noise data)
    fill_holes: bool = False,            # off: texrecon leaves filled holes untextured
                                         # → scattered grey patches that fragment the atlas
    fill_hole_size: float = 0.1,         # max hole size to fill when enabled
    decimate_target: int = 1_500_000,    # target tri count before texturing (0 = off)
    texture: bool = True,                # apply photo colour/texture
    texture_max_views: int = 400,        # texrecon view cap (evenly subsampled). 400 over a
                                         # 62m corridor thins to ~1 view/2.4 frames → patchy
                                         # texture on long scans. With auto_tune ON this is
                                         # scaled UP ∝ scene diagonal (vs a 10m reference),
                                         # capped at texture_max_views_ceiling, so long scenes
                                         # keep view density. More views ⇒ better coverage but
                                         # slower texrecon MRF (the bottleneck). 0 = use bake default.
    texture_max_views_ceiling: int = 1200,  # hard cap on the auto-scaled view count (texrecon
                                         # MRF cost grows with it — keep runtime sane).
    texture_mode: str = "vertex_gpu",    # "vertex_gpu" (full-mesh multi-view photo
                                         # blend, GPU, keeps geometry) | "texrecon"
                                         # (CPU UV atlas) | "none"
    use_refined_poses: bool = True,      # DA3 loop-closure poses, not raw ARKit
    auto_tune: bool = False,             # measure the cleaned cloud BEFORE integrating and adapt
                                         # scale-sensitive params to the scene — a 5 m room and a
                                         # 26 m corridor can't share fixed params. Currently tunes
                                         # depth_trunc (cap to the neural-depth reliable range so
                                         # far noisy depth isn't fused → the "fuzzy/thick" fix) and
                                         # decimate_target (scale with scene size so big scenes keep
                                         # detail). Manual values are the UPPER bounds it tightens.
    reliable_depth_m: float = 8.0,       # neural depth (≈294×518) is reliable to ~this range;
                                         # auto_tune caps depth_trunc to min(this, depth_trunc).
    max_decimate: int = 4_000_000,       # ceiling for the auto-scaled decimate_target (viewer/RAM).
    depth_source: str = "auto",          # which per-frame depth the TSDF integrates:
                                         #   "auto"        → MapAnything chunk depth if present,
                                         #                   else DA3, else LiDAR (legacy default)
                                         #   "da3"         → force DA3 dense per-frame depth
                                         #                   (da3_run/results_output/frame_*.npz);
                                         #                   ~same res as MapAnything but DA3 runs
                                         #                   on ALL frames (asymmetric design). NOTE:
                                         #                   with mask_to_cleaned_cloud the cloud is
                                         #                   keyframe-based, so only frames present in
                                         #                   the cloud integrate — for the full dense
                                         #                   benefit also needs pose interpolation +
                                         #                   unmasked tiling (not yet implemented).
                                         #   "mapanything" → force MapAnything chunk depth
    scene_name: str = "scene",           # output subfolder under output/tsdf/
    variant_label: Optional[str] = None, # human label for the viewer panel
    tsdf_block_count: int = 120_000,     # GPU VoxelBlockGrid hash slots (~10GB @120k)
    tsdf_weight_thresh: float = 2.0,     # min observations per voxel to extract
    tsdf_tiling: object = "auto",        # "auto" | "off" | int N — spatial tiling for big
                                         # scenes: split along the longest axis into N grids
                                         # so each stays under block_count (auto sizes N from
                                         # the cleaned-cloud occupied-block count)
    tsdf_tile_halo: float = 0.6,         # m of overlap between adjacent tiles — keeps the
                                         # seam surface continuous before the core-crop merge
    tsdf_tile_length_m: float = 0.0,     # if >0, tile by FIXED physical length (m) along the
                                         # longest axis (equal-length SECTIONS) instead of the
                                         # equal-block "auto" split. Short sections (~10 m) keep
                                         # each TSDF LOCAL → far fewer global marching-cubes/merge
                                         # artifacts on long scans. The sections are still welded
                                         # into ONE mesh (the shared-voxel seams are identical).
    tsdf_max_edge_m: float = 0.0,        # if >0, drop any triangle with an edge longer than this
                                         # (m). Kills the spurious long "bridge"/spike triangles
                                         # (near↔far) that sparse regions + global decimation create
                                         # — the "lines" across the mesh. Applied after merge AND
                                         # after decimation. Real surfaces have edges ≈ voxel size.
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
    # The name map MUST be keyed in the SAME space as cam.pose_map / sorted_frames,
    # because the integrate loop and the texture bakers both do name_map.get(fidx)
    # with fidx ∈ pose_map.keys():
    #   • mapanything / VGGT-Long (camera_frames.txt): poses keyed by the REAL frame
    #     number → key the name map by the JPG's numeric stem.
    #   • keyframe-sequence path: poses keyed by 0..N-1 → key by enumerate index.
    # With stride>1 these two spaces DIVERGE: enumerate-keying made frame 2·i's JPG
    # load for camera i, mis-texturing the first half of the trajectory while the
    # tail (no map entry → {fidx:06d}.jpg fallback) came out correct.
    kf_name_map: Dict[int, str] = {}
    _sel = frames_dir / "selected_frames.json"
    if _sel.exists():
        try:
            with open(_sel) as _f:
                _kf = sorted(json.load(_f).get("selected_files", []))
            _pose_keys = {int(k) for k in cam.pose_map.keys()}
            _stem_map: Dict[int, str] = {}
            for _n in _kf:
                try:
                    _stem_map[int(Path(_n).stem)] = _n
                except ValueError:
                    _stem_map = {}
                    break
            # Use REAL-frame-number keying whenever pose_map is real-number-keyed — detected
            # by the largest keyframe number being a pose key. The old `issubset` test broke
            # the moment dense-fusion FILLER frames (real numbers NOT in selected_frames.json)
            # were appended to pose_map: issubset→False flipped this to SEQUENCE-INDEX keying,
            # which then handed every frame whose real number ≤ #keyframes the WRONG image
            # (the n-th keyframe by order) → "texture from elsewhere in the video" exactly
            # where fillers exist. Fillers now correctly fall back to {num:06d}.jpg in the bake.
            if _stem_map and max(_stem_map) in _pose_keys:    # real-frame-number keyed
                kf_name_map = _stem_map
            else:
                kf_name_map = {i: n for i, n in enumerate(_kf)}  # sequence-index keyed
        except Exception:
            kf_name_map = {}

    # Depth source — pick the BEST available, driven by the reconstruction
    # backend (output/chunk_*_meta.json):
    #   da3_hybrid        → DA3+LiDAR *fused* depth (da3_full/, 504×378) — best
    #   da3 / mapanything → DA3 / VGGT-Long neural depth
    #   da3_lidar / lidar → raw Stray LiDAR (256×192) + confidence
    backend = _read_recon_backend(output_dir)
    from segmentation.session_io import _find_stray_dir
    stray_dir = _find_stray_dir(session_dir)
    stray_depth = (_resolve_stray_depth(stray_dir, conf_min=conf_min)
                   if stray_dir is not None else None)
    da3_depth = _resolve_da3_depth(
        output_dir,
        conf_percentile=(da3_conf_percentile if da3_conf_percentile > 0 else None),
    )

    # Depth source priority:
    #   1. MapAnything (maplong_run chunk .npy)
    #   2. DA3 / DA3+LiDAR fused (da3_run / da3_full)
    #   3. raw LiDAR
    _cp = da3_conf_percentile if da3_conf_percentile > 0 else None
    _ds = str(depth_source or "auto").lower()
    # DA3-DENSE FUSION (asymmetric design): integrate DA3's per-frame depth for EVERY
    # blur-valid frame (a superset of the VGGT keyframes), not the keyframe-only
    # MapAnything depth. The cloud/poses are keyframe-based, so this also drives pose
    # interpolation (below) and a spatial tile crop instead of the cloud-pixel mask.
    dense_da3 = (_ds == "da3")
    # Backend-driven: VGGT-Long ('mapanything') depth lives in maplong_run chunk .npy.
    # depth_source="da3" forces the DA3 dense per-frame source (skip MapAnything);
    # "mapanything" forces the chunk depth; "auto" keeps the legacy priority.
    # depth_source="da3_frames": use DA3's per-frame depth for ALL posed frames (da3_run/
    # results_output, which now covers the DINO-0.99 set: keyframes + the inter-keyframe
    # frames localized by dense_fusion), MASKED to the cloud, with the REAL poses (keyframe
    # poses + the dense_fusion-appended ones). NO interpolation, NO spatial tiling — unlike
    # "da3" (dense_da3). This is what lets the TSDF integrate the extra frames' depth so the
    # mesh gains inter-keyframe coverage. MapAnything's keyframe-only depth is skipped.
    mapany_src = None
    if _ds != "da3" and ((backend or "").startswith("mapanything")
                         or (output_dir / "maplong_run").exists()):
        mapany_src = _resolve_mapanything_depth(output_dir, conf_percentile=_cp)
    if _ds in ("da3", "da3_frames") and da3_depth is None:
        logger.warning(f"[TSDF-scene] depth_source='{_ds}' but no DA3 per-frame depth found "
                       "(da3_run/results_output/frame_*.npz) — falling back to auto")
    depth_loader = None
    frame_loader = None

    def _resize_nn_2d(a, TH, TW):
        H, W = a.shape
        yi = np.clip((np.arange(TH) * H / TH).astype(np.int64), 0, H - 1)
        xi = np.clip((np.arange(TW) * W / TW).astype(np.int64), 0, W - 1)
        return a[yi][:, xi]

    # HYBRID (depth_source="da3_frames"): MapAnything depth for the KEYFRAMES (consistent with
    # the cloud, which was built from MapAnything depth+poses), DA3 depth (resized to the
    # MapAnything grid) for the inter-keyframe FILLERS that MapAnything never reconstructed.
    # Each frame's own pose is already in pose_map (keyframes + dense_fusion-appended fillers).
    if _ds == "da3_frames" and mapany_src is not None:
        _ma_loader, (depth_h, depth_w) = mapany_src
        _da3_src = _resolve_da3_frame_source(output_dir, conf_percentile=_cp)
        if _da3_src is None:
            frame_loader = _ma_loader
            depth_kind = "MapAnything (no DA3 per-frame source — keyframes only)"
            logger.warning("[TSDF-scene] da3_frames: no DA3 per-frame source → keyframes only")
        else:
            _da3_loader, _ = _da3_src
            _mh, _mw = depth_h, depth_w

            def _hybrid(fidx, _ma=_ma_loader, _da3=_da3_loader, _mh=_mh, _mw=_mw):
                fr = _ma(fidx)                      # keyframe → MapAnything depth (native grid)
                if fr is not None:
                    return fr
                fr = _da3(fidx)                     # filler → DA3 depth, resized to MapAnything
                if fr is None:
                    return None
                d, v, K = fr.get("depth"), fr.get("valid"), fr.get("K")
                if d is not None and d.shape != (_mh, _mw):
                    H0, W0 = d.shape
                    d = _resize_nn_2d(d, _mh, _mw)
                    if v is not None:
                        v = _resize_nn_2d(v.astype(np.uint8), _mh, _mw).astype(bool)
                    if K is not None:
                        K = np.asarray(K, np.float64).copy()
                        K[0, 0] *= _mw / W0; K[0, 2] *= _mw / W0
                        K[1, 1] *= _mh / H0; K[1, 2] *= _mh / H0
                return {"depth": d, "valid": v if v is not None else (d > 0),
                        "K": K, "rgb": None, "hw": (_mh, _mw)}

            frame_loader = _hybrid
            depth_kind = f"HYBRID: MapAnything (keyframes) + DA3 resized→{_mw}x{_mh} (fillers)"
            logger.info(f"[TSDF-scene] per-frame source: {depth_kind}")
    elif mapany_src is not None:
        frame_loader, (depth_h, depth_w) = mapany_src
        depth_kind = ("MapAnything/VGGT-Long neural depth"
                      + (f" · conf≥p{da3_conf_percentile:.0f}" if _cp else ""))
        logger.info(f"[TSDF-scene] per-frame source: MapAnything depth (chunk .npy) "
                    f"@ {depth_w}x{depth_h} — native per-frame K, no rescaling")
    elif da3_depth is not None:
        depth_loader, (depth_h, depth_w) = da3_depth
        _base = ("DA3+LiDAR fused" if (backend or "") == "da3_hybrid"
                 else "DA3/neural")
        depth_kind = (f"{_base} · conf≥p{da3_conf_percentile:.0f}"
                      if _cp else _base)
        frame_src = _resolve_da3_frame_source(output_dir, conf_percentile=_cp)
        if frame_src is not None:
            frame_loader, (depth_h, depth_w) = frame_src
            logger.info(f"[TSDF-scene] per-frame npz source (depth+K+rgb) "
                        f"@ {depth_w}x{depth_h} — no K rescaling")
    elif stray_depth is not None:
        depth_loader, (depth_h, depth_w) = stray_depth
        depth_kind = f"raw LiDAR (conf>={conf_min})"
    else:
        logger.error("[TSDF-scene] no depth source found")
        return None
    logger.info(f"[TSDF-scene] recon_backend={backend}  depth={depth_kind}  "
                f"({depth_w}x{depth_h})")

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

    def _make_vbg():
        # One fresh bounded grid per tile (a single grid overflows on big scans).
        return o3d.t.geometry.VoxelBlockGrid(
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
                f"blocks={tsdf_block_count}/grid)")

    # ── Option A: restrict integration to pixels present in cleaned_cloud.ply ──
    # cleaned_cloud is the CloudCompPy-cleaned merge (DA3 + LiDAR). Each point
    # carries (frame, pixel) traceability, so we integrate ONLY the depth
    # pixels that produced a surviving cloud point — the TSDF then contains
    # nothing the cleaned cloud doesn't.
    cc_frame_pix = None
    cc_ply_hw = None
    cc_xyz = cc_fg = cc_pr = cc_pc = None  # kept for spatial tiling (point→tile assignment)
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
            # No per-point (frame,pixel) trace → can't mask depth to the cloud. Without
            # it the integrate runs unmasked out to depth_trunc (100m), so the TSDF
            # meshes a far larger volume than the cloud (the test3 symptom). Load the
            # cloud POINTS anyway (no trace needed) so we can still (a) tile by bbox and
            # (b) clamp depth + crop the final mesh to the cloud's extent.
            logger.warning("[TSDF-scene] mask_to_cleaned_cloud on, but no PLY "
                           "traceability found — integrating UNMASKED, bounding the "
                           "TSDF to the cloud bbox instead (regenerate the cloud with "
                           "sample_ratio=1.0 + conf filters off to restore traceability)")
            if cc_path.exists():
                try:
                    _cc = o3d.io.read_point_cloud(str(cc_path))
                    if len(_cc.points):
                        cc_xyz = np.asarray(_cc.points, dtype=np.float64)
                except Exception as _e:
                    logger.warning(f"[TSDF-scene] could not read cloud points for "
                                   f"bbox bound ({_e})")
        else:
            _xyz, fg, pr, pc = origins
            cc_fg = np.asarray(fg).astype(np.int64)
            cc_pr = np.asarray(pr).astype(np.int32)
            cc_pc = np.asarray(pc).astype(np.int32)
            cc_xyz = np.asarray(_xyz, dtype=np.float64)  # (N,3) world points → tile assign

            # ── Confidence gate (matches the UI slider) ──
            # Keep only points whose min-max-normalised confidence ≥ conf_min_norm —
            # the SAME (conf-min)/(max-min) normalisation the viewer's slider uses, so
            # `conf_min_norm: 0.1` means exactly the UI slider at 0.1. Drops the noisy
            # low/mid-confidence tail (the vertical smear) BEFORE meshing.
            if conf_min_norm and conf_min_norm > 0:
                conf = _load_ply_confidence(cc_path)
                if conf is not None and len(conf) == len(cc_fg):
                    cmin, cmax = float(conf.min()), float(conf.max())
                    thr = cmin + float(conf_min_norm) * max(cmax - cmin, 1e-6)
                    keep = conf >= thr
                    logger.info(f"[TSDF-scene] confidence gate: conf_min_norm={conf_min_norm} "
                                f"→ raw conf ≥ {thr:.1f} (range {cmin:.1f}..{cmax:.1f}) → kept "
                                f"{int(keep.sum()):,}/{len(conf):,} ({100*keep.mean():.1f}%)")
                    cc_xyz = cc_xyz[keep]; cc_fg = cc_fg[keep]
                    cc_pr = cc_pr[keep]; cc_pc = cc_pc[keep]
                else:
                    logger.warning("[TSDF-scene] conf_min_norm set but no usable 'confidence' "
                                   "field in the cloud — skipping the gate")

            cc_ply_hw = (int(cc_pr.max()) + 1 if len(cc_pr) else 1,
                         int(cc_pc.max()) + 1 if len(cc_pc) else 1)
            if dense_da3:
                # DA3-dense bypasses the per-frame cloud mask (spatial tile crop instead)
                # — keep cc_xyz for bbox/tiling but SKIP the 57M-point argsort it'd waste.
                logger.info(f"[TSDF-scene] dense-DA3: loaded {len(cc_fg):,} cloud points for "
                            f"bbox/tiling only (per-frame cloud mask skipped)")
            else:
                logger.info(f"[TSDF-scene] indexing {len(cc_fg):,} cloud points by frame "
                            f"(single argsort)…")
                cc_frame_pix = _group_rows_by_frame(cc_fg, cc_pr, cc_pc, cc_xyz)
                logger.info(f"[TSDF-scene] mask_to_cleaned_cloud: {len(cc_fg):,} cloud "
                            f"points, {len(cc_frame_pix)} frames, "
                            f"trace {cc_ply_hw[1]}x{cc_ply_hw[0]}")

    # FAITHFUL cloud-raster active? (computed here, BEFORE auto_tune, because auto_tune's
    # depth_trunc cap must NOT apply in raster mode — the z-buffered depth IS clean cloud
    # geometry, so capping it to the neural-depth reliable range would silently drop the
    # real far surface the cloud captured from >reliable_depth_m away.)
    raster_active = bool(rasterize_cloud_depth and cc_frame_pix is not None)

    # Cloud bbox — used to bound the TSDF to the cloud's extent (esp. in the unmasked
    # fallback, where depth_trunc=100m would otherwise mesh far beyond the cloud).
    cloud_bbox = None
    if cc_xyz is not None and len(cc_xyz):
        _mn, _mx = cc_xyz.min(0), cc_xyz.max(0)
        cloud_bbox = (_mn, _mx)
        if cc_frame_pix is None:  # unmasked fallback — clamp the integration distance
            _diag = float(np.linalg.norm(_mx - _mn))
            if 0 < _diag * 1.1 < depth_trunc:
                logger.info(f"[TSDF-scene] clamping depth_trunc {depth_trunc:.1f}m → "
                            f"{_diag * 1.1:.1f}m (cloud diagonal — unmasked bound)")
                depth_trunc = _diag * 1.1

    # ── AUTO-TUNE: adapt scale-sensitive params to the scene the cloud reveals ──
    # A fixed config can't serve a 5 m room AND a 26 m corridor: the neural depth
    # (≈294×518) is noisy past a few metres, so a 100 m depth_trunc fuses far junk →
    # "fuzzy/thick" walls on big scenes; and a 1.5 M-tri decimate over-coarsens a big
    # scene. Measure the cloud once and tighten depth_trunc + scale decimate_target.
    if auto_tune and cloud_bbox is not None:
        _mn, _mx = cloud_bbox
        diag = float(np.linalg.norm(_mx - _mn))
        ext = _mx - _mn
        # depth_trunc:
        #  • RAW-depth mode: cap to the neural-depth reliable range — far noisy per-frame
        #    depth is what blurs big scenes (the "fuzzy walls" fix).
        #  • RASTER mode: the integrated depth IS the already-clean cloud z-buffered back
        #    in, so there is NO far-noise to cap against — capping it to reliable_depth_m
        #    silently DROPS real far surface the cloud captured from farther away (test4:
        #    a 2.36M-pt cluster 18-30m from the cameras was culled, the far half missing).
        #    Instead open it up to cover the whole cloud (diagonal +10%).
        if raster_active:
            _dt_new = max(float(depth_trunc), diag * 1.1)
        else:
            _dt_new = min(float(depth_trunc), max(float(reliable_depth_m), 2.0))
        # decimate_target: scale ∝ scene diagonal vs a ~10 m reference, floor at the
        # configured value, ceil at max_decimate → big scenes keep detail, small don't bloat.
        _REF_DIAG = 10.0
        _dec_new = int(np.clip(decimate_target * (diag / _REF_DIAG),
                               decimate_target, max_decimate)) if decimate_target > 0 else 0
        # texture_max_views: scale ∝ scene diagonal too (same reference), floor at the
        # configured value, ceil at texture_max_views_ceiling — a long corridor needs more
        # views than a small room or texrecon leaves large patchy/untextured swaths.
        _tmv_new = (int(np.clip(texture_max_views * (diag / _REF_DIAG),
                                texture_max_views, max(texture_max_views, texture_max_views_ceiling)))
                    if texture_max_views > 0 else texture_max_views)
        logger.info(f"[TSDF-scene] AUTO-TUNE: scene diag={diag:.1f}m "
                    f"extent={ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f}m  "
                    f"→ depth_trunc {depth_trunc:.1f}→{_dt_new:.1f}m  "
                    f"decimate {decimate_target:,}→{_dec_new:,}  "
                    f"tex_views {texture_max_views}→{_tmv_new}")
        depth_trunc = _dt_new
        decimate_target = _dec_new
        texture_max_views = _tmv_new

    # FAITHFUL cloud-raster integration is active only when we have per-frame cloud
    # traceability (cc_frame_pix); otherwise we fall back to raw-depth integration.
    # (raster_active was computed above, before auto_tune, so the depth_trunc cap could
    # be skipped in raster mode.)
    if raster_active:
        depth_kind = (f"cleaned-cloud raster (splat={cloud_splat_radius}px) "
                      f"[{depth_kind} → K/RGB only]")
        logger.info("[TSDF-scene] integration mode: FAITHFUL cloud-raster — z-buffering "
                    "cleaned_cloud points per frame (raw depth used only for K/RGB); "
                    "edge/conf/dilate/depth_trunc cutoffs bypassed → the mesh copies the cloud")
    elif rasterize_cloud_depth and mask_to_cleaned_cloud:
        logger.warning("[TSDF-scene] rasterize_cloud_depth requested but no cloud "
                       "traceability available — using raw-depth integration "
                       "(regenerate the cloud with traceability to enable faithful mode)")

    if upsample_depth and upsample_depth > 1 and not raster_active:
        logger.info(f"[TSDF-scene] (A) guided depth upsample ×{int(upsample_depth)} ON — "
                    f"integrating at {depth_w*int(upsample_depth)}x{depth_h*int(upsample_depth)} "
                    f"(joint-bilateral, photo-guided)")
    elif upsample_depth and upsample_depth > 1 and raster_active:
        logger.info("[TSDF-scene] (A) upsample_depth set but raster mode active → skipped "
                    "(cloud points have no grid to upsample)")

    # ── DA3-dense fusion setup ──────────────────────────────────────────────────
    # Integrate DA3's per-frame depth for every blur-valid frame (a superset of the
    # VGGT keyframes). The loop-closed poses cover only the keyframes, so interpolate
    # them to the in-between dense frames; and ensure the cloud points are loaded for
    # the bbox/tiling bounds even though the per-frame cloud mask is bypassed (the
    # spatial tile crop replaces it — see _integrate).
    if dense_da3:
        da3_frames = _list_da3_frame_indices(output_dir)
        if not da3_frames:
            logger.warning("[TSDF-scene] depth_source='da3' but no DA3 dense frames found "
                           "(da3_run/results_output/) — falling back to keyframe poses only")
        else:
            cam.pose_map = _interpolate_poses(cam.pose_map, da3_frames)
            poses_src = poses_src + "+interp"
            # If the two-pass BA localised the filler frames, use those ACCURATE poses
            # instead of the SLERP interpolation (so the integrated filler depth lands
            # exactly where the densified cloud points are → cloud↔mesh consistent).
            _fp = output_dir / "ba_run" / "filler_poses.npz"
            if _fp.exists():
                try:
                    _z = np.load(_fp)
                    _ff = _z["frames"].astype(int); _fc = _z["c2w"].astype(np.float64)
                    _n = 0
                    for _k in range(len(_ff)):
                        cam.pose_map[int(_ff[_k])] = _fc[_k]; _n += 1
                    logger.info(f"[TSDF-scene] using {_n} BA-localised filler poses "
                                f"(filler_poses.npz) over SLERP interpolation")
                    poses_src = poses_src + "+ba_fillers"
                except Exception as _e:
                    logger.warning(f"[TSDF-scene] could not load filler_poses.npz ({_e})")
        if cc_xyz is None and not mask_to_cleaned_cloud:
            # Cloud points weren't loaded (mask off) — load them just for bbox/tiling.
            cc_path = output_dir / "cleaned_cloud.ply"
            for alt in ("cleaned_cloud.ply", "cleaned_cloud_symlink.ply", "merged.ply"):
                if (output_dir / alt).exists():
                    cc_path = output_dir / alt
                    break
            try:
                _cc = o3d.io.read_point_cloud(str(cc_path))
                if len(_cc.points):
                    cc_xyz = np.asarray(_cc.points, dtype=np.float64)
            except Exception as _e:
                logger.warning(f"[TSDF-scene] dense-DA3: could not load cloud for bbox/tiling ({_e})")
        # Re-derive the cloud bbox if it wasn't set (mask-off path).
        if cloud_bbox is None and cc_xyz is not None and len(cc_xyz):
            cloud_bbox = (cc_xyz.min(0), cc_xyz.max(0))
        # Clamp the integration distance to the cloud's extent — DA3 raw depth has no
        # cloud mask now, so an over-long depth_trunc would mesh far phantoms.
        if cloud_bbox is not None:
            _diag = float(np.linalg.norm(cloud_bbox[1] - cloud_bbox[0]))
            if 0 < _diag * 1.1 < depth_trunc:
                logger.info(f"[TSDF-scene] dense-DA3: clamping depth_trunc {depth_trunc:.1f}m → "
                            f"{_diag * 1.1:.1f}m (cloud diagonal)")
                depth_trunc = _diag * 1.1
        logger.info(f"[TSDF-scene] integration mode: DA3-DENSE — {len(_list_da3_frame_indices(output_dir))} "
                    f"DA3 frames, poses interpolated to dense, cloud-pixel mask OFF "
                    f"(spatial tile crop + edge_thresh={edge_thresh}m + weight≥{tsdf_weight_thresh})")

    sorted_frames = sorted(cam.pose_map.keys())
    skipped_no_depth = skipped_no_K = skipped_empty = 0
    native_K_map: Dict[int, np.ndarray] = {}  # per-frame npz K (depth res) for texturing
    integrated_frames: set = set()            # distinct frames integrated across all tiles
    # DA3-dense: per-frame [min,max] world coord along the tiling axis, so each tile
    # integrates ONLY the frames whose depth reaches into its slab (else every tile
    # re-loads all 1245 frames). Populated after the tiling split (axis known).
    frame_axis_extent: Dict[int, tuple] = {}
    frame_box_extent: Dict[int, tuple] = {}   # cube mode: fidx → (min_xyz, max_xyz)
    # Extract-safe active-block ceiling: Open3D's marching-cubes ABORTS the process on
    # a too-large grid. PROBED on CPU extract (Open3D 0.19, vol.cpu().extract_…):
    # clean at 51,703 active blocks, SEGFAULTS at 61,201 → real ceiling ≈ 55k (the old
    # ~36k figure was the GPU-extract era, before the move to CPU extract). 45k leaves a
    # safe margin under the 52k-clean / 61k-crash band while letting tiles grow far bigger
    # than the old 12k → fewer tiles on long scenes. Applies to ALL paths (a too-big
    # legacy tile crashes the extract exactly like a dense one), not just dense_da3.
    _EXTRACT_SAFE_ACTIVE = 45_000

    def _integrate(volume, cc_fp, tile_bounds=None):
        """Integrate every posed frame into `volume`. `cc_fp` maps frame → (rows,
        cols, world_xyz) for the cleaned-cloud points this frame observed (and,
        under tiling, only this tile's slice). In FAITHFUL mode (raster_active) the
        frame's world_xyz is z-buffered back into its depth grid and THAT is fused
        — so the mesh copies the cloud. In legacy mode the (rows, cols) pixels mask
        the raw neural/LiDAR depth instead. In DA3-DENSE mode (dense_da3) there is no
        cloud mask: every posed frame's DA3 depth integrates, cropped per tile by
        `tile_bounds`=(axis, lo, hi, halo) — the unprojected world axis-coord must lie
        in the tile slab. Returns the number of frames that touched this grid."""
        nonlocal skipped_no_depth, skipped_no_K, skipped_empty
        n_local = 0
        for fidx in sorted_frames:
            fidx = int(fidx)
            c2w = cam.pose_map.get(fidx)
            if c2w is None:
                continue
            # Whether THIS frame integrates the faithful cloud-raster depth: needs
            # rasterize_cloud_depth ON and per-frame cloud traceability (cc_fp).
            use_raster = rasterize_cloud_depth and cc_fp is not None

            # In raster mode, a frame with no points in THIS tile contributes nothing —
            # skip it BEFORE loading its npz. With many small tiles (the extract-safe
            # split) most frames miss most tiles, so this avoids thousands of wasted
            # per-frame npz reads.
            if use_raster and cc_fp.get(fidx) is None:
                skipped_empty += 1
                continue

            # LEGACY MASKED mode (the default scene path: raster OFF, not dense): a
            # frame with no cleaned-cloud points in THIS tile is masked away later
            # (its mask comes out empty), but only AFTER frame_loader() has paid to
            # np.load its depth chunk. With N tiles every tile re-loaded ALL frames
            # (test7: 4×978 = 3,912 loads for 935 integrated). Skip it up front —
            # equivalent to the post-load empty-mask skip below, minus the I/O.
            if (cc_fp is not None and not use_raster and not dense_da3
                    and cc_fp.get(fidx) is None):
                skipped_empty += 1
                continue

            # DA3-DENSE: skip a frame BEFORE loading its npz if its precomputed world
            # extent doesn't reach this tile (+halo). This is what makes many small tiles
            # viable — without it every tile re-loads+integrates ALL frames.
            if dense_da3 and tile_bounds is not None:
                if tile_bounds[0] == "box3d":
                    _, _lo, _hi, _halo = tile_bounds          # _lo,_hi are xyz vectors
                    _bx = frame_box_extent.get(fidx)
                    if _bx is not None and (np.any(_bx[1] < _lo - _halo)
                                            or np.any(_bx[0] >= _hi + _halo)):
                        skipped_empty += 1
                        continue
                else:
                    _ax, _lo, _hi, _halo = tile_bounds
                    _ext = frame_axis_extent.get(fidx)
                    if _ext is not None and (_ext[1] < _lo - _halo or _ext[0] >= _hi + _halo):
                        skipped_empty += 1
                        continue

            # Resolve the frame's native intrinsics + RGB first. The rich per-frame
            # npz source carries both at the depth grid (zero rescaling); the raster
            # path needs K up front to project the cloud, the legacy path needs the
            # neural depth too. Fall back to the scaled global K + re-decoded JPG.
            frame_K = None
            frame_rgb = None
            depth_m = depth_valid = None
            if frame_loader is not None:
                fr = frame_loader(fidx)
                if fr is not None:
                    depth_m, depth_valid = fr["depth"], fr["valid"]
                    frame_K, frame_rgb = fr["K"], fr["rgb"]
                elif not use_raster:
                    skipped_no_depth += 1   # legacy path can't proceed without depth
                    continue
            elif not use_raster:
                depth_pair = depth_loader(fidx)
                if depth_pair is None:
                    skipped_no_depth += 1
                    continue
                depth_m, depth_valid = depth_pair

            # Intrinsics at the depth grid: prefer the frame's OWN K (no rescaling).
            if frame_K is not None:
                K_d = frame_K
                native_K_map[fidx] = frame_K  # reuse for texturing (scaled to JPG res)
            else:
                K_rgb = cam.K_for(fidx)
                if K_rgb is None:
                    skipped_no_K += 1
                    continue
                K_d = _intrinsics_for_depth(K_rgb, rgb_h, rgb_w, depth_h, depth_w)

            # ── A: guided depth upsample ──────────────────────────────────────
            # Lift the depth-resolution ceiling for the DENSE path: ×factor joint-
            # bilateral upsample of the neural depth using the full-res photo as an
            # edge guide, then integrate at the finer grid with a correspondingly-
            # scaled K. Native K is already stashed in native_K_map above, so
            # texturing is unaffected. Skipped in raster mode (cloud points, no grid).
            cur_h, cur_w = depth_h, depth_w
            up_color = None
            if (upsample_depth and upsample_depth > 1 and not use_raster
                    and depth_m is not None):
                f = int(upsample_depth)
                Hh, Ww = depth_h * f, depth_w * f
                guide = _load_rgb_at_depth(frames_dir, fidx, Hh, Ww,
                                           frame_name=kf_name_map.get(fidx))
                up = _joint_bilateral_upsample_depth(depth_m, depth_valid, guide, f)
                if up is not None:
                    depth_m, depth_valid = up
                    K_d = np.asarray(K_d, dtype=np.float64).copy()
                    K_d[:2, :] *= f            # fx,fy,cx,cy (+skew) scale with the grid
                    cur_h, cur_w = Hh, Ww
                    up_color = guide           # reuse the high-res photo as integ. colour

            if use_raster:
                # FAITHFUL path: z-buffer the cleaned cloud's own surviving points
                # (this frame's slice, restricted to this tile via cc_fp) back into
                # the frame. No raw neural depth, so no flying-pixel streaks and no
                # far phantoms — the TSDF can only mesh what the cloud contains. The
                # edge/conf/dilate cutoffs are moot here (the cloud is already clean).
                fp = cc_fp.get(fidx)
                if fp is None:
                    skipped_empty += 1
                    continue
                depth_m = _rasterize_cloud_depth(
                    fp[2], c2w, K_d, depth_h, depth_w, cloud_splat_radius)
                mask = (depth_m > depth_min) & (depth_m < depth_trunc)
            else:
                # LEGACY path: raw neural/LiDAR depth, masked to the cloud's pixels.
                mask = depth_valid & (depth_m > depth_min) & (depth_m < depth_trunc)
                # Drop flying pixels at depth discontinuities — prevents "ghost"
                # surfaces between objects and background.
                d = np.where(mask, depth_m, 0.0)
                gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
                gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
                mask &= (gx < edge_thresh) & (gy < edge_thresh)
                # Keep only pixels that produced a cleaned_cloud point (and, under
                # tiling, only this tile's slice of them via cc_fp).
                if cc_fp is not None:
                    fp = cc_fp.get(fidx)
                    if fp is None:
                        skipped_empty += 1
                        continue
                    mask &= _build_segment_mask_at_depth(
                        fp[0], fp[1], cur_h, cur_w,
                        cc_ply_hw[0], cc_ply_hw[1], cleaned_cloud_dilate,
                    )

                # DA3-DENSE spatial tile crop: no cloud mask, so keep only the depth
                # whose unprojected WORLD coord lands in this tile (+halo). Bounds the
                # tile's grid without the cloud pixels.
                if dense_da3 and tile_bounds is not None:
                    if tile_bounds[0] == "box3d":
                        # 3D CUBE: the pixel must be inside the cube box on ALL three axes.
                        _, _lo, _hi, _halo = tile_bounds
                        _wxyz = _depth_world_xyz(depth_m, K_d, c2w)
                        mask &= np.all((_wxyz >= _lo - _halo)
                                       & (_wxyz < _hi + _halo), axis=2)
                    else:
                        _ax, _lo, _hi, _halo = tile_bounds
                        _wa = _depth_axis_world_coord(depth_m, K_d, c2w, _ax)
                        mask &= (_wa >= _lo - _halo) & (_wa < _hi + _halo)

            if not mask.any():
                skipped_empty += 1
                continue

            depth_masked = np.where(mask, depth_m, 0.0).astype(np.float32)

            # Color at the integration grid (cur_h×cur_w). Upsample (A) reuses the
            # already-decoded high-res guide; the npz image is used only when it
            # matches the grid; else re-decode the JPG at the grid resolution.
            if up_color is not None:
                color_np = up_color
            elif frame_rgb is not None and frame_rgb.shape[:2] == (cur_h, cur_w):
                color_np = frame_rgb
            else:
                color_np = _load_rgb_at_depth(frames_dir, fidx, cur_h, cur_w,
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
            # Open3D raises "No block is touched in TSDF volume" (a RuntimeError) when
            # a single frame unprojects to zero active blocks — degenerate K/pose, or
            # all depth out of (depth_min, depth_trunc). Guard PER FRAME so one bad
            # frame is skipped instead of aborting the entire scene mesh.
            try:
                coords = volume.compute_unique_block_coordinates(
                    depth_t, K_t, ext_t, depth_scale=1.0, depth_max=depth_trunc,
                    trunc_voxel_multiplier=trunc_mult)
                volume.integrate(coords, depth_t, color_t, K_t, K_t, ext_t,
                                 depth_scale=1.0, depth_max=depth_trunc,
                                 trunc_voxel_multiplier=trunc_mult)
            except RuntimeError as e:
                if "No block is touched" in str(e):
                    skipped_empty += 1
                    if skipped_empty <= 5:
                        _dv = depth_masked[depth_masked > 0]
                        logger.warning(
                            f"[TSDF-scene] frame {fidx}: no block touched — skipping "
                            f"(valid_px={int((depth_masked > 0).sum())}, "
                            f"depth_min={_dv.min():.3f} max={_dv.max():.3f} "
                            f"K_fx={K_d[0, 0]:.1f} hw={depth_h}x{depth_w})"
                            if _dv.size else
                            f"[TSDF-scene] frame {fidx}: no block touched, empty depth")
                    continue
                raise
            n_local += 1
            integrated_frames.add(fidx)
            # Last-resort overflow net: VBG.integrate corrupts the CUDA context (illegal
            # memory access) the moment the active block set exceeds the hash capacity,
            # and that error is async — it surfaces later in extract, killing the whole
            # process. Poll the live block count and stop THIS tile before it overflows.
            # Conservative tile sizing should keep this from ever firing.
            if n_local % 5 == 0:   # poll every 5 frames — bound the overshoot between
                                   # checks well under the 52k-clean / 61k-crash band
                try:
                    _active = int(volume.hashmap().size())
                except Exception:
                    _active = -1
                # Stop at the EXTRACT-safe ceiling (ALL paths, not just dense) — the
                # extract aborts long before the hash fills, and a too-big legacy tile
                # crashes it exactly like a dense one. A stopped tile loses a few late
                # frames' contribution to that slab (minor) vs crashing the run. The
                # occ-based pre-split sizes tiles well under this, so it rarely fires.
                if 0 <= _EXTRACT_SAFE_ACTIVE < _active:
                    logger.warning(
                        f"[TSDF-scene] tile hit extract-safe ceiling "
                        f"({_active:,}/{_EXTRACT_SAFE_ACTIVE:,} blocks) after {n_local} "
                        f"frames — stopping tile (finer tiling would avoid this)")
                    break
                if 0 <= int(tsdf_block_count) * 0.92 < _active:
                    logger.warning(
                        f"[TSDF-scene] grid near capacity ({_active:,}/"
                        f"{int(tsdf_block_count):,} blocks) after {n_local} frames — "
                        f"stopping this tile early to avoid overflow (raise tsdf_block_count "
                        f"or add tiles)")
                    break
                if progress_cb:
                    progress_cb("integrating", time.time() - t0, None)
        try:
            logger.info(f"[TSDF-scene]   tile integrated {n_local} frames → "
                        f"{int(volume.hashmap().size()):,} active blocks "
                        f"(cap {int(tsdf_block_count):,})")
        except Exception:
            pass
        return n_local

    # ── Spatial tiling: split the scene along its longest axis so each grid stays
    #    under block_count. A single VoxelBlockGrid has a FIXED hash capacity; a
    #    144m scan overflowed it (CUDA illegal memory access). Each tile is its own
    #    bounded grid; adjacent tiles share a `tsdf_tile_halo` overlap so the seam
    #    surface is meshed continuously, then each tile mesh is cropped to its core
    #    (triangles whose centroid lies inside the tile) and the cores concatenated.
    #    VBG voxel coords are GLOBAL (anchored at world origin, fixed voxel size), so
    #    the shared-halo surface is identical in adjacent tiles → cropping at the same
    #    plane gives gapless, non-overlapping joins. ──
    block_side = float(voxel_length) * 16.0
    vox = float(voxel_length)
    cube_mode = bool(tsdf_tile_length_m and tsdf_tile_length_m > 0
                     and cc_xyz is not None and len(cc_xyz))
    cubes = None            # list of (lo_xyz, hi_xyz) np arrays when cube_mode
    tiles = [(-np.inf, np.inf)]
    tile_axis = 0
    ax = None
    if cube_mode:
        # ── 3D CUBE tiling: cover the cloud's bbox with cubes of side ~tsdf_tile_length_m.
        # The VBG voxel grid is GLOBAL (anchored at the world origin, fixed voxel size),
        # so adjacent cubes that share the `tsdf_tile_halo` overlap emit IDENTICAL vertices
        # on the shared face → the weld is exact. Each cube is bounded on ALL three axes,
        # so no cube can overflow the grid regardless of the cloud's shape. Empty cubes
        # (no cloud points) are skipped. Bounds are FINITE (the bbox outer faces are padded
        # by a margin, not ±inf) so an oversized cube can be split cleanly at a midpoint.
        L = max(vox, round(float(tsdf_tile_length_m) / vox) * vox)   # cube side, voxel-aligned
        gmin = np.floor(cc_xyz.min(0) / vox) * vox                   # snapped global origin
        gmax = cc_xyz.max(0)
        _mrg = float(tsdf_tile_halo) + float(sdf_trunc) + 2.0 * vox  # outer-face padding
        ncell = np.maximum(1, np.ceil((gmax - gmin) / L).astype(np.int64))  # cubes per axis
        bnds = []
        for a in range(3):
            e = gmin[a] + L * np.arange(ncell[a] + 1)
            e[0] = e[0] - _mrg                       # extend outer faces (no neighbour there)
            e[-1] = e[-1] + _mrg                     # → nothing is clipped at the bbox edges
            bnds.append(e)
        # only emit cubes that actually contain cloud points (the rest are empty volume)
        cell = np.clip(np.floor((cc_xyz - gmin) / L).astype(np.int64), 0, ncell - 1)
        occ_keys = sorted(map(tuple, np.unique(cell, axis=0)))
        cubes = []
        for (i, j, k) in occ_keys:
            lo = np.array([bnds[0][i], bnds[1][j], bnds[2][k]], dtype=np.float64)
            hi = np.array([bnds[0][i + 1], bnds[1][j + 1], bnds[2][k + 1]], dtype=np.float64)
            cubes.append((lo, hi))
        n_tiles = len(cubes)
        logger.info(f"[TSDF-scene] tiling: 3D CUBES side={L:.2f}m  grid="
                    f"{ncell[0]}×{ncell[1]}×{ncell[2]}  occupied={n_tiles} cubes "
                    f"(empty skipped)")
    elif cc_xyz is not None and len(cc_xyz):
        ext = cc_xyz.max(0) - cc_xyz.min(0)
        tile_axis = int(np.argmax(ext))
        ax = cc_xyz[:, tile_axis]
        # Occupied 16³ blocks, and the axis block-index of each UNIQUE block — the
        # tile boundaries are balanced by BLOCK count, not point count. Point density
        # varies wildly (close walls = many pts/block, far surfaces = few), so an
        # equal-population split leaves the sparse-but-spread tiles overloaded with
        # blocks and they still overflow. Equal-block boundaries keep every grid level.
        blk = np.floor(cc_xyz / block_side).astype(np.int64)
        key = blk[:, 0] * 73856093 ^ blk[:, 1] * 19349663 ^ blk[:, 2] * 83492791
        _, _uidx = np.unique(key, return_index=True)
        ublk_axis = blk[_uidx, tile_axis].astype(np.float64)  # one per unique block
        occ = int(len(_uidx))
        _t = tsdf_tiling
        if isinstance(_t, bool):
            n_tiles = 1
        elif isinstance(_t, int):
            n_tiles = max(1, _t)
        elif isinstance(_t, str) and _t.strip().lstrip("-").isdigit():
            n_tiles = max(1, int(_t))
        elif isinstance(_t, str) and _t.strip().lower() == "off":
            n_tiles = 1
        else:  # "auto" — the binding limit is NOT the hash capacity but Open3D's
               # extract_triangle_mesh(), which SEGFAULTS on a large grid. So cap each
               # tile at an extract-safe occupied-block count — NOT 0.25·block_count,
               # which was sized for the (much larger) hash capacity and let tiles grow
               # until extract died.
            _EXTRACT_SAFE_OCC = 20_000  # occupied 16³ blocks/tile. RE-PROBED on the CPU
                                        # extract path (Open3D 0.19): clean to 51,703
                                        # active, crash at 61,201 → ceiling ≈55k (the old
                                        # 7k cap was the GPU-extract era and over-tiled
                                        # long scans). In the non-dense masked path active
                                        # ≈ occ (test7: 7k occ → ≤7.2k active), so 20k occ
                                        # → ≤~25k active, well under 45k EXTRACT_SAFE. This
                                        # cuts test7 from 4 tiles → 2 (fewer seams + less
                                        # per-tile re-scan); dense inflates via _dense_mult.
            usable = max(1.0, min(float(tsdf_block_count) * 0.25,
                                  float(_EXTRACT_SAFE_OCC)))
            # DA3-DENSE fills FAR more blocks than the sparse cloud's occupied set —
            # measured ≈15× (a tile sized for 6.7k cloud blocks integrated to 103.8k
            # active and crashed the extract). Inflate the block estimate so the tiles
            # are split fine enough to stay extract-safe. The per-frame extent skip
            # keeps the extra tiles cheap; the active-block break is the final net.
            _dense_mult = 16.0 if dense_da3 else 1.0
            n_tiles = max(1, int(np.ceil(occ * _dense_mult / usable)))
        if n_tiles > 1:
            # quantiles over per-unique-block axis positions → equal blocks per tile
            qb = np.quantile(np.sort(ublk_axis), np.linspace(0.0, 1.0, n_tiles + 1))
            qs = qb * block_side  # block-index → world coordinate
            qs[0], qs[-1] = -np.inf, np.inf
            tiles = [(float(qs[k]), float(qs[k + 1])) for k in range(n_tiles)]
        logger.info(f"[TSDF-scene] tiling: {len(tiles)} tile(s) along "
                    f"{'XYZ'[tile_axis]}  (~{occ:,} occupied blocks @ {block_side:.2f}m, "
                    f"cap {int(tsdf_block_count):,}/tile)")
    else:
        logger.info("[TSDF-scene] single grid (no cleaned-cloud xyz to tile by)")

    # DA3-DENSE: precompute each frame's world extent (one cheap sub-sampled
    # unprojection per frame) so the tile loop can skip frames that don't reach a tile
    # BEFORE loading their npz. Cube mode stores the full 3D box; slab mode the 1D axis
    # span. Only worthwhile when tiling.
    _n_tiles_now = len(cubes) if cube_mode else len(tiles)
    if dense_da3 and (cube_mode or _n_tiles_now > 1) and frame_loader is not None:
        logger.info(f"[TSDF-scene] dense-DA3: precomputing world extents for "
                    f"{len(sorted_frames)} frames…")
        _t_pre = time.time()
        for _fi in sorted_frames:
            _fi = int(_fi)
            _c2w = cam.pose_map.get(_fi)
            if _c2w is None:
                continue
            _fr = frame_loader(_fi)
            if not _fr or _fr.get("K") is None:
                continue
            _d = _fr["depth"][::8, ::8]                 # sub-sample 8× for speed
            _m = _d > depth_min
            if not _m.any():
                continue
            # K must match the sub-sampled grid: scale fx,fy,cx,cy by 1/8.
            _K = np.asarray(_fr["K"], dtype=np.float64).copy()
            _K[:2, :] /= 8.0
            if cube_mode:
                _w = _depth_world_xyz(_d, _K, _c2w)[_m]      # (n,3) valid world pts
                frame_box_extent[_fi] = (_w.min(0), _w.max(0))
            else:
                _wa = _depth_axis_world_coord(_d, _K, _c2w, tile_axis)[_m]
                frame_axis_extent[_fi] = (float(_wa.min()), float(_wa.max()))
        _ne = len(frame_box_extent) if cube_mode else len(frame_axis_extent)
        logger.info(f"[TSDF-scene] dense-DA3: extents ready ({_ne} frames, "
                    f"{time.time() - _t_pre:.0f}s)")

    halo = float(tsdf_tile_halo)
    # Active 16³-block ceiling the CPU marching-cubes extracts WITHOUT aborting (probed:
    # clean ≤51,703, crash ≥61,201). A cube above this is split into sub-cubes and re-queued.
    _CUBE_SAFE_ACTIVE = 45_000
    _CUBE_MIN_SIDE = 1.0                     # don't split below ~1 m (recursion floor)
    tile_meshes = []
    # Work QUEUE (not a fixed list): cube_mode can PUSH sub-cubes onto it when a cube's grid
    # is too big to extract — the subdivision guard. Slab mode never grows the queue.
    work = list(cubes) if cube_mode else list(tiles)
    n_tiles_tot = len(work)
    need_crop = cube_mode or n_tiles_tot > 1
    ti = 0
    wi = 0
    while wi < len(work):
        tile = work[wi]
        wi += 1
        ti += 1
        lo, hi = tile                       # cube: lo,hi are xyz vectors; slab: scalars
        tile_bounds = None
        if dense_da3:
            # No cloud-pixel mask: every posed frame integrates, cropped to this tile by
            # the spatial test in _integrate (tile_bounds).
            cc_fp = None
            if cube_mode:
                tile_bounds = ("box3d", lo, hi, halo)
                logger.info(f"[TSDF-scene] cube {ti} ({len(work)} queued) DA3-DENSE "
                            f"x∈[{lo[0]:.1f},{hi[0]:.1f}) y∈[{lo[1]:.1f},{hi[1]:.1f}) "
                            f"z∈[{lo[2]:.1f},{hi[2]:.1f}) +{halo:.2f}m halo")
            elif n_tiles_tot > 1:
                tile_bounds = (tile_axis, lo, hi, halo)
                logger.info(f"[TSDF-scene] tile {ti}/{n_tiles_tot} DA3-DENSE "
                            f"{'XYZ'[tile_axis]}∈[{lo:.2f},{hi:.2f}) +{halo:.2f}m halo "
                            f"(spatial crop)")
        elif cube_mode:
            if cc_fg is None:
                # UNMASKED fallback (cloud has no traceability): integrate every
                # posed frame spatially cropped to this cube — same treatment as
                # dense-DA3 — instead of dereferencing the missing (frame,pixel)
                # trace (this crashed with 'NoneType' object is not subscriptable).
                cc_fp = None
                tile_bounds = ("box3d", lo, hi, halo)
                logger.info(f"[TSDF-scene] cube {ti} ({len(work)} queued) UNMASKED "
                            f"x∈[{lo[0]:.1f},{hi[0]:.1f}) y∈[{lo[1]:.1f},{hi[1]:.1f}) "
                            f"z∈[{lo[2]:.1f},{hi[2]:.1f}) +{halo:.2f}m halo (spatial crop)")
            else:
                sel = np.all((cc_xyz >= lo - halo) & (cc_xyz < hi + halo), axis=1)
                cc_fp = _group_rows_by_frame(cc_fg[sel], cc_pr[sel], cc_pc[sel], cc_xyz[sel])
                logger.info(f"[TSDF-scene] cube {ti} ({len(work)} queued) "
                            f"x∈[{lo[0]:.1f},{hi[0]:.1f}) y∈[{lo[1]:.1f},{hi[1]:.1f}) "
                            f"z∈[{lo[2]:.1f},{hi[2]:.1f}) — {len(cc_fp)} frames")
        elif n_tiles_tot > 1:
            if cc_fg is None:
                cc_fp = None
                tile_bounds = (tile_axis, lo, hi, halo)
                logger.info(f"[TSDF-scene] tile {ti}/{n_tiles_tot} UNMASKED "
                            f"{'XYZ'[tile_axis]}∈[{lo:.2f},{hi:.2f}) +{halo:.2f}m halo "
                            f"(spatial crop)")
            else:
                sel = (ax >= lo - halo) & (ax < hi + halo)
                cc_fp = _group_rows_by_frame(cc_fg[sel], cc_pr[sel], cc_pc[sel], cc_xyz[sel])
                logger.info(f"[TSDF-scene] tile {ti}/{n_tiles_tot} "
                            f"{'XYZ'[tile_axis]}∈[{lo:.2f},{hi:.2f}) +{halo:.2f}m halo — "
                            f"{len(cc_fp)} frames")
        else:
            cc_fp = cc_frame_pix  # None (unmasked) or the full dict

        if progress_cb:
            progress_cb("integrating", time.time() - t0, None)
        vol = _make_vbg()
        ni = _integrate(vol, cc_fp, tile_bounds)
        if ni == 0:
            logger.warning(f"[TSDF-scene] tile {ti}: 0 frames integrated — skipped")
            del vol
            continue
        try:
            _active = int(vol.hashmap().size())
        except Exception:
            _active = -1

        # ── SUBDIVISION GUARD: the CPU marching-cubes ABORTS the process on a grid with
        # too many active blocks. If this cube is over the ceiling, split it into ≤8
        # sub-cubes (halving each axis longer than _CUBE_MIN_SIDE, at a voxel-snapped
        # midpoint) and re-queue them INSTEAD of extracting. The global voxel grid + the
        # halo overlap keep the sub-seams weldable. ──
        if cube_mode and _active > _CUBE_SAFE_ACTIVE:
            ext = hi - lo
            if np.any(ext > _CUBE_MIN_SIDE):
                mid = np.where(ext > _CUBE_MIN_SIDE,
                               lo + np.maximum(vox, np.round(ext / 2.0 / vox) * vox), hi)
                subs = []
                for dx in (0, 1):
                    for dy in (0, 1):
                        for dz in (0, 1):
                            slo = np.array([lo[0] if dx == 0 else mid[0],
                                            lo[1] if dy == 0 else mid[1],
                                            lo[2] if dz == 0 else mid[2]])
                            shi = np.array([mid[0] if dx == 0 else hi[0],
                                            mid[1] if dy == 0 else hi[1],
                                            mid[2] if dz == 0 else hi[2]])
                            if np.all(shi - slo > 1e-6):        # skip degenerate (unsplit axis)
                                subs.append((slo, shi))
                logger.info(f"[TSDF-scene]   cube {ti} too big ({_active:,} active blocks "
                            f"> {_CUBE_SAFE_ACTIVE:,}) → split into {len(subs)} sub-cubes")
                work.extend(subs)
                del vol
                try:
                    o3d.core.cuda.release_cache()
                except Exception:
                    pass
                continue

        if progress_cb:
            progress_cb("extracting", time.time() - t0, None)
        # Extract on CPU (marching cubes in RAM). estimated_vertex_number stays default
        # (-1): passing a value pads the mesh with garbage unreferenced verts.
        logger.info(f"[TSDF-scene]   extracting tile ({_active:,} active blocks)")
        try:
            tmesh = vol.cpu().extract_triangle_mesh(
                weight_threshold=float(tsdf_weight_thresh)).to_legacy()
        finally:
            del vol  # free the tile's GPU grid before building the next one
            try:
                o3d.core.cuda.release_cache()   # release Open3D's cached CUDA memory
            except Exception:
                pass

        if need_crop and len(tmesh.triangles):
            # Crop to the tile core: keep triangles whose centroid is in [lo,hi) — on the
            # tiling axis for slabs, or inside the box on ALL 3 axes for cubes (the halo
            # belongs to the neighbour). The shared global voxel grid makes seam vertices
            # coincide across neighbours, so cropping here gives gapless, weldable joins.
            v = np.asarray(tmesh.vertices)
            tri = np.asarray(tmesh.triangles)
            if cube_mode:
                cen = v[tri].mean(axis=1)                      # (n,3) centroids
                drop = ~np.all((cen >= lo) & (cen < hi), axis=1)
            else:
                cax = v[:, tile_axis][tri].mean(axis=1)
                drop = ~((cax >= lo) & (cax < hi))
            if drop.any():
                tmesh.remove_triangles_by_mask(drop)
                tmesh.remove_unreferenced_vertices()
        if len(tmesh.triangles):
            tile_meshes.append(tmesh)

    n_integrated = len(integrated_frames)
    if not tile_meshes:
        logger.error(f"[TSDF-scene] no frames integrated "
                     f"(no_depth={skipped_no_depth} no_K={skipped_no_K} "
                     f"empty={skipped_empty})")
        if progress_cb:
            progress_cb("error", time.time() - t0, None)
        return None

    # Concatenate tile cores → one mesh; weld the seam vertices (identical across
    # tiles thanks to the global voxel grid) so the joins are watertight.
    mesh = tile_meshes[0]
    for _m in tile_meshes[1:]:
        mesh += _m
    if len(tile_meshes) > 1:
        mesh = mesh.merge_close_vertices(float(voxel_length) * 0.5)
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        logger.info(f"[TSDF-scene] merged {len(tile_meshes)} tiles → "
                    f"{len(mesh.vertices):,} verts / {len(mesh.triangles):,} tris")

    # Kill the spurious 'bridge'/spike triangles (near↔far long edges) BEFORE the
    # global decimate — quadric simplification would otherwise smear them around.
    mesh = _drop_long_edge_tris(mesh, tsdf_max_edge_m, " post-weld")

    # VBG's float-color path stores colours in 0-255; legacy meshes and GLB
    # expect 0-1, so normalise (the smoke test confirmed the 0-255 range).
    if mesh.has_vertex_colors():
        _vc = np.asarray(mesh.vertex_colors)
        if _vc.size and _vc.max() > 1.5:
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(_vc / 255.0, 0.0, 1.0))
    mesh.compute_vertex_normals()

    # Bound the mesh to the cloud's volume (+margin). In the unmasked fallback the TSDF
    # can extend well past the cloud (depth ballooning); cropping guarantees the mesh
    # stays within the cloud's extent — fixes "TSDF volume larger than the cloud".
    if cloud_bbox is not None and (cc_frame_pix is None or dense_da3) and len(mesh.vertices):
        _mn, _mx = cloud_bbox
        _margin = max(0.1, float(sdf_trunc) * 4.0)
        _v = np.asarray(mesh.vertices)
        _keep = np.all((_v >= _mn - _margin) & (_v <= _mx + _margin), axis=1)
        if not _keep.all():
            mesh.remove_vertices_by_mask(~_keep)
            logger.info(f"[TSDF-scene] cropped {int((~_keep).sum()):,} verts outside "
                        f"cloud bbox (+{_margin:.2f}m margin)")

    # Drop ONLY true noise specks — NOT real disconnected geometry. LITERATURE
    # (deep-research, verified): small-cluster removal is non-standard and deletes
    # real surface; BundleFusion repairs by re-integration, not deletion. The old
    # `0.01 × largest-cluster` threshold scaled with mesh size (→ 50k tris on a 5M
    # mesh) and deleted whole ~2m regions that fragmented off (low truncation). Use a
    # tiny FIXED floor so only genuine specks go, and CAP the total fraction removed
    # so a fragmented mesh never loses a big chunk silently.
    try:
        tri_clusters, n_per_cluster, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        n_per_cluster = np.asarray(n_per_cluster)
        if len(n_per_cluster) > 1:
            min_tri = 200                       # absolute speck floor (was max(500, 1% of max))
            keep = n_per_cluster >= min_tri
            drop_tris = ~keep[tri_clusters]
            n_drop = int(drop_tris.sum())
            # Safety: never drop more than 5% of the mesh as "specks" — if we would,
            # the mesh is fragmented (raise sdf_trunc), don't silently delete coverage.
            if 0 < n_drop <= 0.05 * len(tri_clusters):
                mesh.remove_triangles_by_mask(drop_tris)
                mesh.remove_unreferenced_vertices()
                logger.info(f"[TSDF-scene] dropped {int((~keep).sum())} speck clusters "
                            f"({n_drop:,} tris, <{min_tri} tris each)")
            elif n_drop > 0.05 * len(tri_clusters):
                logger.warning(f"[TSDF-scene] speck cleanup would drop {n_drop:,} tris "
                               f"(>5% — mesh fragmented, raise sdf_trunc) — SKIPPING to "
                               f"preserve coverage")
    except Exception as e:
        logger.warning(f"[TSDF-scene] cluster cleanup skipped ({e})")

    # ── Phase 3: mesh cleanup ──
    # Fill small LiDAR dropout holes (Open3D tensor API). hole_size caps it so
    # only sensor gaps close — not real openings or the open scan boundary.
    if fill_holes and fill_hole_size > 0:
        try:
            # SANITIZE first: fill_holes() triangulates every boundary loop via VTK's
            # vtkPolygon, which ABORTS (vtkPolygon.cxx:956 "start>=end") on a degenerate
            # / non-manifold boundary. Welding tiles + dropping long-edge & speck tris
            # opens exactly those ragged loops, so clean the mesh to simple, manifold
            # boundaries before handing it to VTK.
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.remove_unreferenced_vertices()
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

    # Final long-edge pass: decimation can MERGE short edges into a few long ones,
    # so re-run the line-killer after simplifying.
    mesh = _drop_long_edge_tris(mesh, tsdf_max_edge_m, " post-decimate")

    # ── Snap mesh to the cleaned cloud (ICP, gated) BEFORE texturing ──
    # Guarantees the mesh shares the cloud's frame even if an upstream loop-closure
    # inconsistency left it rigidly offset (test3: 11.46°/0.57 m). Applied to the mesh AND
    # the camera poses together, so texturing below projects in the SAME (cloud) frame —
    # the texture bakes correctly and the GLB ships with BAKED vertices (no post-texture
    # round-trip that corrupts texrecon's UV atlas; no node-matrix that BIM-side readers
    # might ignore).
    if progress_cb:
        progress_cb("aligning", time.time() - t0, None)
    _icp_T, icp_snap = _icp_mesh_to_cloud(mesh, output_dir)
    if _icp_T is not None:
        mesh.transform(_icp_T)
        def _T_pose(p):
            p = np.asarray(p, dtype=np.float64)
            if p.shape != (4, 4):
                M = np.eye(4); M[:p.shape[0], :p.shape[1]] = p; p = M
            return _icp_T @ p
        cam.pose_map = {k: _T_pose(v) for k, v in cam.pose_map.items()}
        logger.info(f"[TSDF-scene] ICP-snap → cloud: rot {icp_snap['rot_deg']:.2f}° "
                    f"trans {icp_snap['trans_m']:.3f} m (fitness {icp_snap['fitness']:.2f}, "
                    f"rmse {icp_snap['rmse']:.3f}, median "
                    f"{icp_snap['before_median']:.3f}→{icp_snap['after_median']:.3f} m) — "
                    f"mesh+poses moved together before texturing")
    else:
        logger.info(f"[TSDF-scene] ICP-snap not applied ({icp_snap.get('reason')}) — "
                    "mesh left in pose frame")

    # Sit under the same ``output/tsdf/`` root as the per-instance meshes so
    # the ``/tsdf/list/`` endpoint (which iterates subfolders) picks it up
    # automatically and the viewport loads it alongside everything else.
    scene_dir = output_dir / "tsdf" / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    glb_path = scene_dir / f"{scene_name}.glb"
    meta_path = glb_path.with_suffix(".meta.json")

    # Per-view native intrinsics (each frame's OWN npz K, captured at integrate),
    # shared by both colouring backends.
    tex_intrinsics: Dict[int, np.ndarray] = {}
    for _fidx in sorted_frames:
        _K = native_K_map.get(_fidx)
        if _K is None:
            _K = cam.K_for(_fidx)
        if _K is not None:
            tex_intrinsics[_fidx] = np.asarray(_K, dtype=np.float64)

    # ── Colour mode "vertex_gpu": multi-view photo blend per vertex on the FULL
    #    mesh (GPU, ~10s, keeps geometry exact). Sets vertex colours BEFORE the
    #    GLB is written. Validated MAE ≈ 17/255 vs source photos. ──
    color_mode = "rgb8_per_vertex"
    textured = False
    if texture and texture_mode == "vertex_gpu":
        if progress_cb:
            progress_cb("texturing", time.time() - t0, None)
        try:
            from reconstruction.nvdiffrast_bake import (is_available,
                                                        bake_vertex_colors_gpu)
            if not is_available():
                raise RuntimeError("nvdiffrast/torch CUDA not available")
            mesh.compute_vertex_normals()
            _init = (np.asarray(mesh.vertex_colors)
                     if mesh.has_vertex_colors() else None)
            vc = bake_vertex_colors_gpu(
                vertices=np.asarray(mesh.vertices),
                vertex_normals=np.asarray(mesh.vertex_normals),
                faces=np.asarray(mesh.triangles),   # → TRUE mesh z-buffer occlusion
                frames_dir=frames_dir, pose_map=cam.pose_map,
                intrinsics_map=tex_intrinsics, output_dir=output_dir,
                name_map=kf_name_map, init_colors=_init,
                progress_cb=(lambda p: progress_cb("texturing", time.time() - t0, None))
                            if progress_cb else None,
            )
            if vc is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
                color_mode = "rgb8_per_vertex_photo"
                textured = True
            else:
                logger.warning("[TSDF-scene] vertex_gpu bake returned None — "
                               "keeping VBG vertex colour")
        except Exception as e:
            logger.warning(f"[TSDF-scene] vertex_gpu bake error ({e}) — "
                           "keeping VBG vertex colour")

    ok = o3d.io.write_triangle_mesh(str(glb_path), mesh, write_ascii=False)
    if not ok:
        logger.error(f"[TSDF-scene] failed to write GLB at {glb_path}")
        if progress_cb:
            progress_cb("error", time.time() - t0, None)
        return None

    n_v, n_t = int(len(mesh.vertices)), int(len(mesh.triangles))

    # ── UV-atlas texture bake (texrecon) — overwrites scene.glb with the
    #    textured mesh. On any failure the vertex-colour preview is kept. ──
    if texture and texture_mode == "texrecon":
        if progress_cb:
            progress_cb("texturing", time.time() - t0, None)
        tex_in = scene_dir / "_scene_geom.ply"
        try:
            # Hand texrecon a robust binary PLY (read natively) rather than
            # round-tripping the geometry through GLB.
            o3d.io.write_triangle_mesh(str(tex_in), mesh, write_ascii=False)
            from reconstruction.texture_bake import bake_texture
            res = bake_texture(
                mesh_path=tex_in,
                frames_dir=frames_dir,
                pose_map=cam.pose_map,
                intrinsics_map=tex_intrinsics,
                name_map=kf_name_map,  # keyframe-index → real JPG (sparse originals)
                out_glb=glb_path,
                **({"max_views": int(texture_max_views)} if texture_max_views > 0 else {}),
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

    # Keep an UNCOMPRESSED copy as ``scene.glb.orig`` before shrinking. The
    # compressed scene.glb (meshopt + WebP) is for the browser; trimesh can't read
    # those extensions, and crop_scene_mesh_to_instances (per-object textured
    # carve, run by the segmentation stage) needs a plain-GLB it can load. Without
    # this, fresh runs would have only the compressed mesh → the crop fails.
    try:
        import shutil as _shutil
        _shutil.copyfile(glb_path, str(glb_path) + ".orig")
    except Exception as e:
        logger.warning(f"[TSDF-scene] could not save scene.glb.orig ({e}) — "
                       f"per-object TSDF crop will fall back to the compressed mesh")

    # Shrink the final .glb (meshopt geometry + WebP textures) so the viewer can
    # stream it without saturating the browser. Done after texturing so the meta
    # below records the compressed on-disk size.
    _compress_scene_glb(glb_path)

    elapsed = time.time() - t0
    meta = {
        "method": "tsdf_scene",
        **({"label": variant_label} if variant_label else {}),
        "color": ("uv_texture" if texture_mode == "texrecon" and textured
                  else color_mode),
        "textured": textured,
        "texture_mode": texture_mode,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "depth_min": depth_min,
        "edge_thresh": edge_thresh,
        "conf_min": int(conf_min),
        "da3_conf_percentile": float(da3_conf_percentile),
        "mask_to_cleaned_cloud": bool(mask_to_cleaned_cloud),
        "integration_mode": "cloud_raster" if raster_active else "raw_depth",
        "rasterize_cloud_depth": bool(rasterize_cloud_depth),
        "cloud_splat_radius": int(cloud_splat_radius),
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
        "icp_snap_to_cloud": icp_snap,
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
        # NOTE: orient_normals_consistent_tangent_plane builds a global MST over
        # every point — the dominant cost (minutes, can look hung on >2-3M pts).
        # max_points/voxel_downsample keep the working set tractable. Emit a phase
        # so the UI shows it's WORKING, not stuck at "starting".
        n_norm = len(pcd.points)
        if progress_cb:
            progress_cb("normals", time.time() - t0, None)
        logger.info(f"[Poisson-scene] estimating + orienting normals on "
                    f"{n_norm:,} pts (global MST — the slow step)…")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        logger.info(f"[Poisson-scene] normals done ({time.time() - t0:.0f}s elapsed)")

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
            # keyframe-index → real JPG name (poses are keyframe-indexed but the
            # JPGs use sparse original frame numbers — see the TSDF-scene path).
            poisson_name_map: Dict[int, str] = {}
            _sel = frames_dir / "selected_frames.json"
            if _sel.exists():
                try:
                    with open(_sel) as _f:
                        _kf = sorted(json.load(_f).get("selected_files", []))
                    poisson_name_map = {i: n for i, n in enumerate(_kf)}
                except Exception:
                    poisson_name_map = {}
            try:
                o3d.io.write_triangle_mesh(str(tex_in), mesh, write_ascii=False)
                from reconstruction.texture_bake import bake_texture
                res = bake_texture(
                    mesh_path=tex_in, frames_dir=frames_dir,
                    pose_map=cam.pose_map, intrinsics_map=cam.intrinsics_map,
                    name_map=poisson_name_map,
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


# ── Single source of truth for the scene-TSDF config → kwargs mapping ───────
# The in-pipeline worker (workers/tsdf_worker.py) and the manual endpoint
# (main.py /tsdf/scene_export) MUST forward the same config.yaml `tsdf:` keys, or
# the two reconstruct differently. Keeping two hand-maintained allowlists made
# them silently drift. Instead, derive the forwardable keys from the
# export_tsdf_scene SIGNATURE so adding/removing a kwarg updates BOTH paths at
# once — they can never diverge again.
import inspect as _inspect

# Args provided by the caller at runtime, NOT taken from config.yaml `tsdf:`.
_TSDF_RUNTIME_ARGS = {"output_dir", "frames_dir", "session_dir", "progress_cb"}


def tsdf_scene_config_keys() -> set:
    """The config.yaml ``tsdf:`` keys forwarded to ``export_tsdf_scene`` — derived
    from its signature, so the pipeline worker and the manual endpoint stay
    identical automatically."""
    return {n for n in _inspect.signature(export_tsdf_scene).parameters
            if n not in _TSDF_RUNTIME_ARGS}


def build_tsdf_scene_kwargs(config: dict, overrides: Optional[dict] = None) -> dict:
    """Build ``export_tsdf_scene`` kwargs from ``config['tsdf']`` (+ optional
    per-request ``overrides``). The ONE place both run paths build their params."""
    keys = tsdf_scene_config_keys()
    tcfg = (config or {}).get("tsdf", {}) or {}
    kw = {k: tcfg[k] for k in keys if k in tcfg}
    if overrides:
        kw.update({k: overrides[k] for k in keys if k in overrides})
    return kw


# ── Poisson scene config → kwargs (mirror of the TSDF mapping) ──────────────
_POISSON_RUNTIME_ARGS = {"output_dir", "frames_dir", "session_dir", "progress_cb"}


def poisson_scene_config_keys() -> set:
    """The config.yaml ``poisson:`` keys forwarded to ``export_poisson_scene`` —
    derived from its signature, so the endpoint/worker stay in lockstep with it."""
    return {n for n in _inspect.signature(export_poisson_scene).parameters
            if n not in _POISSON_RUNTIME_ARGS}


def build_poisson_scene_kwargs(config: dict, overrides: Optional[dict] = None) -> dict:
    """Build ``export_poisson_scene`` kwargs from ``config['poisson']`` (+ optional
    per-request ``overrides``)."""
    keys = poisson_scene_config_keys()
    pcfg = (config or {}).get("poisson", {}) or {}
    kw = {k: pcfg[k] for k in keys if k in pcfg}
    if overrides:
        kw.update({k: overrides[k] for k in keys if k in overrides})
    return kw


# ── Per-instance crop of the scene TSDF mesh ────────────────────────────────
#
# The per-instance TSDF re-integration (``export_tsdf``) needs a recognised
# per-frame depth source. When the reconstruction backend stored its depth under
# a non-standard run dir (e.g. VGGTOMEGA → ``omega_run/`` / ``da3_run_probe/``),
# that resolver returns nothing and the per-object TSDF writes zero meshes.
#
# But the SCENE TSDF already ran successfully and produced a fully reconstructed,
# cloud-consistent mesh (``tsdf/scene/scene.glb``). For architectural geometry —
# arches, doorways, stairs — that scene mesh holds the REAL surface (concave,
# with true depth), which both ShapeR (a closed-object generative prior) and a
# from-scratch per-object TSDF struggle to reproduce.
#
# This function carves each instance straight out of that scene mesh: keep the
# scene vertices within ``proximity_m`` of the instance's segment sub-cloud
# (``globalIndices`` into ``cleaned_cloud.ply``), keep the faces whose three
# vertices all survive, drop tiny noise islands. No depth re-integration, no ICP,
# no generative prior — the output is a faithful slice of the scene surface.

def crop_scene_mesh_to_instances(
    output_dir: Path,
    segments_result: dict,
    obj_ids: Optional[List[int]] = None,
    proximity_m: float = 0.05,
    min_island_faces: int = 30,
    progress_cb: Optional[Callable[[int, str, Optional[float], Optional[str]], None]] = None,
) -> List[Path]:
    """Carve one mesh per segmented instance out of the scene TSDF mesh.

    Args:
        output_dir: session output dir (contains ``tsdf/scene/scene.glb`` and
            ``cleaned_cloud.ply``).
        segments_result: parsed ``segmentation_result.json``.
        obj_ids: optional filter — only crop these instance IDs.
        proximity_m: a scene vertex is kept if within this distance of the
            instance sub-cloud (5 cm matches the cloud↔mesh median of ~1 cm with
            margin; raise it to bridge sparse traceability, lower it to tighten).
        min_island_faces: connected components smaller than this (in faces) are
            dropped as noise. Components are otherwise ALL kept — never collapse
            to the single largest one, or open/stepped objects (stairs, frames)
            lose most of their geometry.
        progress_cb: ``(instance_id, phase, elapsed, mesh_path)`` callback.

    Returns:
        List of ``.glb`` paths written, under ``tsdf/<safe_label>_<id>/``.
    """
    import trimesh
    from scipy.spatial import cKDTree
    from segmentation.pipeline import _load_ply_origins

    output_dir = Path(output_dir)
    t0 = time.time()

    # ── Locate the scene mesh. Prefer the un-simplified textured ``.orig`` (the
    # web ``scene.glb`` is meshopt/draco-packed and trips trimesh's GLB reader).
    scene_dir = output_dir / "tsdf" / "scene"
    scene_path: Optional[Path] = None
    for cand in ("scene.glb.orig", "scene.glb"):
        if (scene_dir / cand).exists():
            scene_path = scene_dir / cand
            break
    if scene_path is None:
        logger.error("[TSDF-crop] no scene mesh found (tsdf/scene/scene.glb[.orig]) "
                     "— run the scene TSDF first")
        return []

    # Load as a SCENE (not force="mesh"): the textured scene mesh is a set of
    # per-chart sub-meshes, each with its own UV + baseColor image. Merging them
    # into one Trimesh (the old force="mesh") drops all textures — which is why
    # cropped objects came out untextured. ``dump`` bakes the node transforms so
    # every part's vertices are already in world frame.
    try:
        with open(scene_path, "rb") as f:
            scene = trimesh.load(f, file_type="glb", force="scene")
        if isinstance(scene, trimesh.Trimesh):
            parts = [scene]
        else:
            parts = scene.dump(concatenate=False)
        parts = [p for p in parts
                 if isinstance(p, trimesh.Trimesh) and len(p.faces) > 0]
    except Exception as e:
        logger.error(f"[TSDF-crop] failed to load scene mesh {scene_path.name}: {e}")
        return []
    if not parts:
        logger.error(f"[TSDF-crop] scene mesh {scene_path.name} is empty/unreadable")
        return []
    part_bounds = [np.asarray(p.bounds, dtype=np.float64) for p in parts]
    n_tex = sum(1 for p in parts
                if getattr(getattr(p, "visual", None), "uv", None) is not None)
    tot_f = sum(len(p.faces) for p in parts)
    logger.info(f"[TSDF-crop] scene mesh {scene_path.name}: {len(parts)} parts / "
                f"{tot_f:,} faces ({n_tex} textured)")

    # ── Cloud xyz — same indexing space as ``globalIndices``. Use the shared
    # PLY origin loader so this matches export_tsdf / shaper_export exactly.
    ply_path = output_dir / "cleaned_cloud.ply"
    if not ply_path.exists():
        for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                ply_path = output_dir / alt
                break
    origins = _load_ply_origins(ply_path)
    if origins is None:
        logger.error(f"[TSDF-crop] cannot load PLY origins from {ply_path}")
        return []
    xyz = origins[0]
    logger.info(f"[TSDF-crop] cloud: {len(xyz):,} points  (proximity={proximity_m*100:.0f}cm)")

    instances = segments_result.get("instances", [])
    if not instances:
        logger.warning("[TSDF-crop] no instances in segmentation_result.json")
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
        gi = gi[(gi >= 0) & (gi < len(xyz))]
        if len(gi) < 10:
            logger.warning(f"[TSDF-crop] {label}_{inst_id}: too few points ({len(gi)}) — skipping")
            continue
        sub = xyz[gi].astype(np.float64)

        if progress_cb:
            progress_cb(int(inst_id), "starting", None, None)

        t_inst = time.time()
        subtree = cKDTree(sub)
        # AABB pre-filter: only query parts whose bbox overlaps the instance's
        # (expanded by proximity). Skips the far charts → big speed-up on large
        # scenes without changing the result.
        inst_lo = sub.min(axis=0) - proximity_m
        inst_hi = sub.max(axis=0) + proximity_m

        # Per-part surviving-face masks + the centroids of those faces (for a
        # single global island filter across charts, since charts don't share
        # vertices and per-chart connectivity would wrongly split one object).
        part_face_masks: List[Optional[np.ndarray]] = [None] * len(parts)
        cent_list: List[np.ndarray] = []
        owner_list: List[np.ndarray] = []
        for pi, p in enumerate(parts):
            lo, hi = part_bounds[pi][0], part_bounds[pi][1]
            if np.any(hi < inst_lo) or np.any(lo > inst_hi):
                continue  # no overlap
            Vp = np.asarray(p.vertices, dtype=np.float64)
            dist, _ = subtree.query(Vp, k=1, workers=-1)
            keep_v = dist < proximity_m
            kf = keep_v[p.faces].all(axis=1)
            if not kf.any():
                continue
            part_face_masks[pi] = kf
            fc = p.vertices[p.faces[kf]].mean(axis=1)
            cent_list.append(fc)
            fids = np.where(kf)[0]
            owner_list.append(np.stack([np.full(len(fids), pi), fids], axis=1))

        if not cent_list:
            logger.warning(f"[TSDF-crop] {label}_{inst_id}: 0 faces within "
                           f"{proximity_m*100:.0f}cm — skipping")
            if progress_cb:
                progress_cb(int(inst_id), "error", None, None)
            continue

        # ── Global island filter. Cluster surviving face centroids by spatial
        # proximity (radius graph → connected components) and drop clusters with
        # < min_island_faces faces. Operates across chart boundaries so a single
        # object split into many texture charts is never broken up. min<=0 skips.
        cent = np.concatenate(cent_list)
        owner = np.concatenate(owner_list)
        keep_face = np.ones(len(cent), dtype=bool)
        if min_island_faces > 0 and len(cent) > 1:
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
                ct = cKDTree(cent)
                pairs = ct.query_pairs(r=max(2.5 * proximity_m, 0.03),
                                       output_type="ndarray")
                n = len(cent)
                if len(pairs):
                    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
                    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
                    graph = csr_matrix((np.ones(len(rows)), (rows, cols)),
                                       shape=(n, n))
                    _, lab = connected_components(graph, directed=False)
                else:
                    lab = np.arange(n)
                sizes = np.bincount(lab)
                keep_face = sizes[lab] >= min_island_faces
                if not keep_face.any():        # all clusters tiny → keep largest
                    keep_face = lab == int(np.argmax(sizes))
            except Exception as e:
                logger.warning(f"[TSDF-crop] {label}_{inst_id}: island filter failed "
                               f"({e}) — keeping all")

        # Restrict each part's face mask to islands we kept.
        owner_keep = owner[keep_face]
        kept_by_part: Dict[int, np.ndarray] = {}
        for pi in np.unique(owner_keep[:, 0]):
            kept_by_part[int(pi)] = owner_keep[owner_keep[:, 0] == pi, 1]

        # Build the cropped, still-textured sub-meshes.
        cropped: List[trimesh.Trimesh] = []
        for pi, fids in kept_by_part.items():
            g = parts[pi].copy()
            fm = np.zeros(len(g.faces), dtype=bool)
            fm[fids] = True
            g.update_faces(fm)               # preserves UV + material
            g.remove_unreferenced_vertices()  # also remaps the UV array
            if len(g.faces) > 0:
                # Force-recompute vertex normals: update_faces/remove_unreferenced
                # invalidate the cache, and the GLB exporter only emits NORMAL when
                # they're present — without this the crop ships flat-shaded.
                try:
                    _ = g.vertex_normals
                except Exception:
                    pass
                cropped.append(g)
        if not cropped:
            logger.warning(f"[TSDF-crop] {label}_{inst_id}: nothing left after island filter")
            if progress_cb:
                progress_cb(int(inst_id), "error", None, None)
            continue

        out_scene = trimesh.Scene(cropped)
        safe = _safe_label(label, inst_id)   # already "<label>_<id>"
        obj_dir = tsdf_root / safe
        obj_dir.mkdir(parents=True, exist_ok=True)
        out_path = obj_dir / f"{safe}.glb"
        out_scene.export(str(out_path))      # multi-material GLB keeps textures
        # Stream-compress (meshopt + WebP) like the scene mesh; best-effort.
        _compress_scene_glb(out_path)

        n_faces_out = int(sum(len(g.faces) for g in cropped))
        n_verts_out = int(sum(len(g.vertices) for g in cropped))
        n_tex_out = sum(1 for g in cropped
                        if getattr(getattr(g, "visual", None), "uv", None) is not None)
        ext = (out_scene.bounds[1] - out_scene.bounds[0]).tolist()
        elapsed = time.time() - t_inst
        meta = {
            "method": "tsdf_scene_crop",
            "source_mesh": scene_path.name,
            "instance_id": int(inst_id),
            "label": label,
            "glb_file": out_path.name,
            "textured": bool(n_tex_out > 0),
            "n_texture_charts": int(n_tex_out),
            "proximity_m": float(proximity_m),
            "min_island_faces": int(min_island_faces),
            "n_sub_points": int(len(gi)),
            "n_vertices": n_verts_out,
            "n_faces": n_faces_out,
            "bbox_extent": [round(float(v), 4) for v in ext],
            "elapsed_s": round(float(elapsed), 2),
        }
        with open(obj_dir / f"{safe}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[TSDF-crop] {label}_{inst_id}: sub={len(gi):,}pts → "
                    f"{n_verts_out:,} verts / {n_faces_out:,} faces "
                    f"({n_tex_out} textured charts) ext={[round(v,2) for v in ext]} "
                    f"→ {out_path.name}")
        if progress_cb:
            progress_cb(int(inst_id), "done", elapsed, str(out_path))
        exported.append(out_path)

    logger.info(f"[TSDF-crop] wrote {len(exported)} mesh(es) in {time.time()-t0:.1f}s")
    return exported
