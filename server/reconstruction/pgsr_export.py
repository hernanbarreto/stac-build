"""
PGSR scene exporter (precision task, Phase D).
================================================================================
Builds the COLMAP-text scene layout the PGSR trainer consumes, FROM the finished
vggtomega pipeline output — poses fixed, cloud as Gaussian seed:

    output/pgsr_scene/
        images/               symlinks to the NATIVE-resolution keyframe JPGs
        masks/                optional dynamic-class masks (frame_<num>.png, 255 =
                              EXCLUDED from the photometric loss)
        sparse/cameras.txt    one PINHOLE camera per keyframe (intrinsics scaled
                              from the omega grid to the native frame)
        sparse/images.txt     world-to-camera quaternions/translations (COLMAP
                              convention) from the final metric camera_poses.txt
        sparse/points3D.ply   subsampled cleaned cloud (xyz+rgb) — the seed

The poses are the pipeline's FINAL metric poses (post scale_align v2 + orient):
PGSR optimizes the scene photometrically FROM this initialization; poses move
only if the (flagged) photometric pose refinement is enabled in the trainer.
Fail-fast: missing poses/intrinsics/cloud abort with the exact reason.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("PGSRExport")

SCENE_DIRNAME = "pgsr_scene"


def _rot_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → COLMAP qvec (w, x, y, z)."""
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q[:] = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, \
            (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    if q[0] < 0:
        q = -q
    return q


def _read_intrinsics_rows(output_dir: Path):
    p = output_dir / "intrinsic.txt"
    if not p.exists():
        raise RuntimeError(f"pgsr_export: {p} not found (per-keyframe intrinsics)")
    rows = [[float(x) for x in ln.split()] for ln in p.read_text().splitlines()
            if ln.strip()]
    return rows


def _omega_grid_hw(output_dir: Path):
    from reconstruction.scale_align import _omega_depth
    om = _omega_depth(output_dir)
    if not om:
        raise RuntimeError("pgsr_export: no omega per-frame depth (needed to know "
                           "the grid the intrinsics live on)")
    d = next(iter(om.values()))
    return d.shape  # (H, W)


def _write_points3d_ply(cloud_ply: Path, dst: Path, max_pts: int) -> int:
    """Subsample cleaned_cloud.ply → points3D.ply (xyz + rgb, binary LE)."""
    from reconstruction.scale_align import _PLY_NP
    raw = cloud_ply.read_bytes()
    hend = raw.find(b"end_header")
    nl = raw.find(b"\n", hend)
    header = raw[:nl + 1].decode("ascii", "replace")
    fmt, props, n, in_v = None, [], 0, False
    for ln in header.splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_v = t[1] == "vertex"
            if in_v:
                n = int(t[2])
        elif t[0] == "property" and in_v and t[1] != "list":
            props.append((t[-1], t[1]))
    if fmt == "ascii":
        raise RuntimeError("pgsr_export: ascii cleaned_cloud.ply unsupported")
    endian = "<" if "little" in fmt else ">"
    dt = np.dtype([(nm, endian + _PLY_NP[ty]) for nm, ty in props])
    arr = np.frombuffer(raw[nl + 1:nl + 1 + n * dt.itemsize], dtype=dt)
    # UNIFORM seed (REVERTED 2026-08-19): the 08-18 confidence-first seed left
    # mid-confidence regions (far walls, grazing floor) seedless — densification
    # grew unconstrained Gaussians there and the mesh got displaced floating
    # parts ("partes de otro lado", user verdict). The known-good 08-11 recipe
    # seeds uniformly over the WHOLE cloud.
    step = max(1, n // max_pts)
    arr = arr[::step]
    names = arr.dtype.names
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((len(arr), 3), 128, np.uint8)
    out = np.zeros(len(arr), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                             ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                                             ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(out)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    dst.write_bytes(hdr.encode("ascii") + out.tobytes())
    return len(out)


def export_scene(output_dir: Path, frames_dir: Path, max_seed_pts: int = 1_500_000,
                 masks_dir: Optional[Path] = None, log=None) -> Path:
    """Build output/pgsr_scene from the finished pipeline output. Returns the
    scene path. Idempotent (rebuilds cheap text/symlinks every call)."""
    _log = log if log is not None else (lambda m: logger.info(m))
    # resolve() both: the image symlinks embed frames_dir verbatim — a relative
    # caller path would produce symlinks that only work from one cwd
    output_dir, frames_dir = Path(output_dir).resolve(), Path(frames_dir).resolve()
    from reconstruction.scale_align import _read_poses
    from PIL import Image

    scene = output_dir / SCENE_DIRNAME
    (scene / "sparse").mkdir(parents=True, exist_ok=True)
    img_dir = scene / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir()

    lines, nums, _ = _read_poses(output_dir)
    mats = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
            for ln in lines if len(ln.split()) == 16]
    K_rows = _read_intrinsics_rows(output_dir)
    if len(K_rows) != len(nums):
        raise RuntimeError(f"pgsr_export: intrinsic.txt rows ({len(K_rows)}) != "
                           f"keyframes ({len(nums)})")
    Hd, Wd = _omega_grid_hw(output_dir)

    cam_lines = ["# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    img_lines = ["# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "#"]
    n_img = 0
    native_wh = None
    for i, (n, T) in enumerate(zip(nums, mats)):
        src = frames_dir / f"{n:06d}.jpg"
        if not src.exists():
            raise RuntimeError(f"pgsr_export: keyframe image {src.name} missing")
        if native_wh is None:
            with Image.open(src) as im:
                native_wh = im.size                       # (W, H)
        Wn, Hn = native_wh
        fx, fy, cx, cy = K_rows[i][:4]
        sx, sy = Wn / Wd, Hn / Hd
        cam_lines.append(f"{i + 1} PINHOLE {Wn} {Hn} "
                         f"{fx * sx:.8g} {fy * sy:.8g} {cx * sx:.8g} {cy * sy:.8g}")
        Rw2c = T[:3, :3].T
        tw2c = -Rw2c @ T[:3, 3]
        q = _rot_to_qvec(Rw2c)
        name = f"{n:06d}.jpg"
        os.symlink(str(src), str(img_dir / name))
        img_lines.append(f"{i + 1} {q[0]:.9g} {q[1]:.9g} {q[2]:.9g} {q[3]:.9g} "
                         f"{tw2c[0]:.9g} {tw2c[1]:.9g} {tw2c[2]:.9g} {i + 1} {name}")
        img_lines.append("")                              # empty POINTS2D line
        n_img += 1
    (scene / "sparse" / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (scene / "sparse" / "images.txt").write_text("\n".join(img_lines) + "\n")

    cloud = None
    for cand in ("cleaned_cloud.ply", "cleaned_cloud_raw.ply"):
        if (output_dir / cand).exists():
            cloud = output_dir / cand
            break
    if cloud is None:
        raise RuntimeError("pgsr_export: no cleaned_cloud.ply — the Gaussian seed "
                           "is the pipeline's cloud, run the pipeline first")
    n_pts = _write_points3d_ply(cloud, scene / "sparse" / "points3D.ply", max_seed_pts)

    if masks_dir is not None and Path(masks_dir).exists():
        dst = scene / "masks"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(masks_dir, dst)
        _log(f"dynamic masks staged: {len(list(dst.glob('*')))} files")

    meta = {"n_images": n_img, "native_wh": list(native_wh), "seed_points": n_pts,
            "omega_grid_hw": [int(Hd), int(Wd)], "source_cloud": cloud.name}
    (scene / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    _log(f"PGSR scene: {n_img} keyframes at {native_wh[0]}x{native_wh[1]}, "
         f"{n_pts:,} seed points → {scene}")
    return scene


ANCHOR_DIRNAME = "cloud_anchor"


def export_cloud_anchor_depths(output_dir: Path, frames_dir: Path,
                               spacing_m: float = 0.006, half_res: bool = True,
                               log=None,
                               point_indices=None,
                               dst_dir: Optional[Path] = None) -> Path:
    """MAX-PRECISION anchor (user 2026-08-18: "PGSR tomando como premisa la
    nube"): z-buffer the FULL cleaned cloud into every keyframe camera and save
    the depth maps under pgsr_scene/cloud_anchor/<name>.npz. The trainer adds a
    loss that penalises rendered depth that leaves the cloud's noise band —
    PGSR refines the surface INSIDE the band, it can never drift from the cloud.

    Runs in the SERVER env (open3d available), reusing the SAME rasterizer the
    TSDF integrates with — anchor and mesh share one definition of "the cloud
    seen from this camera". Fail-fast on missing inputs."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir, frames_dir = Path(output_dir).resolve(), Path(frames_dir).resolve()
    from reconstruction.scale_align import _read_poses
    from segmentation.tsdf_export import _rasterize_cloud_depth
    import open3d as o3d
    from PIL import Image

    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        raise RuntimeError("cloud_anchor: cleaned_cloud.ply missing")
    xyz = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points, np.float64)
    if not len(xyz):
        raise RuntimeError("cloud_anchor: cleaned_cloud.ply is empty")
    # PER-OBJECT anchor (user 2026-08-30): rasterize ONLY the instance's own
    # points — the depth prior for masked object training (sparse-view cure)
    _sel = None
    if point_indices is not None:
        _sel = np.asarray(point_indices, dtype=np.int64)
        _sel = _sel[(_sel >= 0) & (_sel < len(xyz))]
        if not len(_sel):
            raise RuntimeError("cloud_anchor: point_indices selected 0 points")
        xyz = xyz[_sel]
        _log(f"cloud anchor: OBJECT mode — {len(xyz):,} points")
    # CONFIDENCE-WEIGHTED anchor (2026-08-18): the anchor pulls hard toward the
    # clean upper half of the cloud (weight 1.0) and only softly (0.3) toward
    # low-confidence points — where the cloud is unreliable, the photometric
    # consensus decides. Hierarchical raster: clean surface first, low-conf
    # fills only empty pixels.
    from segmentation.tsdf_export import _load_ply_confidence
    conf = _load_ply_confidence(cloud_path)
    if conf is not None and _sel is not None and len(conf) >= (_sel.max() + 1):
        conf = conf[_sel]
    hi_mask = None
    if conf is not None and len(conf) == len(xyz):
        hi_mask = conf >= float(np.percentile(conf, 50.0))
        _log(f"cloud anchor: confidence-weighted (primary {int(hi_mask.sum()):,} "
             f"pts w=1.0, filler {int((~hi_mask).sum()):,} pts w=0.3)")

    lines, nums, _ = _read_poses(output_dir)
    mats = [np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
            for ln in lines if len(ln.split()) == 16]
    K_rows = _read_intrinsics_rows(output_dir)
    Hd, Wd = _omega_grid_hw(output_dir)

    dst = Path(dst_dir) if dst_dir is not None \
        else output_dir / SCENE_DIRNAME / ANCHOR_DIRNAME
    dst.mkdir(parents=True, exist_ok=True)
    native_wh = None
    n_done = 0
    for i, (n, T) in enumerate(zip(nums, mats)):
        src = frames_dir / f"{n:06d}.jpg"
        if not src.exists():
            continue
        if native_wh is None:
            with Image.open(src) as im:
                native_wh = im.size
        Wn, Hn = native_wh
        if half_res:                      # anchor tolerance is cm-scale — half
            Wn, Hn = Wn // 2, Hn // 2     # res halves disk/VRAM at no cost
        fx, fy, cx, cy = K_rows[i][:4]
        K = np.array([[fx * Wn / Wd, 0, cx * Wn / Wd],
                      [0, fy * Hn / Hd, cy * Hn / Hd], [0, 0, 1.0]])
        if hi_mask is not None:
            d_hi = _rasterize_cloud_depth(xyz[hi_mask], T, K, Hn, Wn, 1,
                                          spacing_m=float(spacing_m), band_avg_m=0.02)
            d_lo = _rasterize_cloud_depth(xyz[~hi_mask], T, K, Hn, Wn, 1,
                                          spacing_m=float(spacing_m), band_avg_m=0.02)
            hi_v = d_hi > 0
            depth = np.where(hi_v, d_hi, d_lo)
            weight = np.where(hi_v, 1.0, 0.3).astype(np.float32)
        else:
            depth = _rasterize_cloud_depth(xyz, T, K, Hn, Wn, 1,
                                           spacing_m=float(spacing_m), band_avg_m=0.02)
            weight = np.ones_like(depth, np.float32)
        # silhouette-edge cull (same gradient rule as the TSDF/pgsr_cloud):
        # at object boundaries the raster jumps between fore/background — an
        # anchor there would penalise legitimate renders of either side
        gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        depth[(gx > 0.04) | (gy > 0.04)] = 0.0
        np.savez_compressed(dst / f"{n:06d}.npz",
                            depth=depth.astype(np.float32),
                            weight=weight)
        n_done += 1
    if n_done == 0:
        raise RuntimeError("cloud_anchor: no keyframe produced an anchor raster")
    _log(f"cloud anchor: {n_done} keyframe rasters ({Wn}x{Hn}, full cloud "
         f"{len(xyz):,} pts) → {dst}")
    return dst
