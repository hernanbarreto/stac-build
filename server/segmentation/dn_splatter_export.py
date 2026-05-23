"""
DN-Splatter dataset exporter — Stray Scanner → nerfstudio format.
==================================================================

Converts a session's Stray Scanner output (RGB frames + LiDAR depth +
ARKit poses) into the directory layout DN-Splatter (and any
``normal-nerfstudio`` dataparser) expects::

    <output_dir>/dn_splatter/
    ├── transforms.json          # nerfstudio camera + frame metadata
    ├── images/<NNNNNN>.jpg      # full-res RGB (symlink/copy from frames/)
    └── depths/<NNNNNN>.png      # uint16 mm depth upsampled to RGB res

Conversions performed
---------------------
- **Camera convention**: Stray exports c2w in OpenCV (+X right, +Y down,
  +Z forward). Nerfstudio expects OpenGL (+X right, +Y up, -Z forward).
  We flip the camera's Y and Z basis vectors (``c2w @ diag(1,-1,-1,1)``).
- **Depth resolution**: Stray depth is 256×192 uint16 mm. Nerfstudio
  expects depth at the SAME resolution as the RGB image. We upsample
  nearest-neighbour to 1920×1440 (no interpolation artefacts; each block
  of 7.5×7.5 RGB pixels gets the single real LiDAR measurement).
- **Depth units**: mm preserved; ``depth_unit_scale_factor: 0.001`` is
  written into ``transforms.json`` so the loader converts to metres.

Authors: Hernán Barreto — Ingerop IN3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("DNSplatterExport")


# ── Coordinate conversion ─────────────────────────────────────────────

_FLIP_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

def _opencv_c2w_to_opengl(c2w_cv: np.ndarray) -> np.ndarray:
    """Stray OpenCV camera (+Y down, +Z forward) → nerfstudio OpenGL
    (+Y up, -Z forward) c2w. World frame untouched.
    """
    return c2w_cv.astype(np.float64) @ _FLIP_CV_TO_GL


# ── Depth resizing ─────────────────────────────────────────────────────

def _upsample_depth_nn(depth_uint16: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour upsample uint16 depth. Preserves the discrete
    LiDAR measurements without inventing interpolated values that would
    create false surfaces between real samples.
    """
    if depth_uint16.shape == (target_h, target_w):
        return depth_uint16
    img = Image.fromarray(depth_uint16, mode="I;16")
    img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.array(img, dtype=np.uint16)


# ── Stray Scanner I/O (mirrors ingestors.stray_scanner; standalone) ───

def _load_stray_odometry(odometry_csv: Path) -> Dict[int, np.ndarray]:
    """Read odometry.csv → {frame_idx: 4x4 c2w in OpenCV}."""
    pose_map: Dict[int, np.ndarray] = {}
    with open(odometry_csv) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if len(parts) < 9:
                continue
            try:
                fidx = int(float(parts[1]))
                x, y, z = (float(p) for p in parts[2:5])
                qx, qy, qz, qw = (float(p) for p in parts[5:9])
            except ValueError:
                continue
            # Quaternion → rotation matrix (Hamilton, w-first → x,y,z,w stored).
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ], dtype=np.float64)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            pose_map[fidx] = T
    return pose_map


def _load_stray_intrinsics(camera_matrix_csv: Path) -> np.ndarray:
    """Read 3×3 pinhole K (single line CSV or 3-line CSV)."""
    rows = []
    with open(camera_matrix_csv) as f:
        for line in f:
            vals = [v.strip() for v in line.replace(",", " ").split() if v.strip()]
            if vals:
                rows.append([float(v) for v in vals])
    K = np.array(rows, dtype=np.float64)
    if K.shape == (9,):
        K = K.reshape(3, 3)
    if K.shape != (3, 3):
        raise ValueError(f"Bad camera_matrix.csv shape {K.shape}")
    return K


# ── Main export ────────────────────────────────────────────────────────

def export_dn_splatter_dataset(
    session_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    use_keyframes_only: bool = False,
    use_symlinks: bool = False,
) -> Path:
    """Build ``output_dir/dn_splatter/`` for DN-Splatter (nerfstudio).

    Parameters
    ----------
    session_dir
        Session root. We look for a sibling directory containing the Stray
        export (``odometry.csv``, ``depth/``, ``camera_matrix.csv``).
    frames_dir
        Directory with full-resolution RGB frames (``NNNNNN.jpg``).
    output_dir
        Where the ``dn_splatter/`` subdir will be created. Typical:
        ``<session>/output``.
    use_keyframes_only
        If True, restrict to frames present in ``cleaned_cloud.ply``'s
        ``frame_global`` field (the cosine-filtered keyframes). If False
        (default), use every Stray frame with a valid pose AND a
        corresponding JPG AND depth PNG.
    use_symlinks
        Symlink the source JPGs into ``images/`` instead of copying. Saves
        disk; fails on FS that doesn't support symlinks.

    Returns
    -------
    Path to ``transforms.json``.
    """
    session_dir = Path(session_dir).resolve()
    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    t0 = time.time()

    # Find the Stray sibling dir.
    stray_dir: Optional[Path] = None
    for c in [session_dir] + [p for p in session_dir.parent.iterdir() if p.is_dir()]:
        if (c / "odometry.csv").exists() and (c / "camera_matrix.csv").exists():
            stray_dir = c
            break
    if stray_dir is None:
        raise FileNotFoundError(
            f"No Stray sibling directory found near {session_dir} "
            f"(need odometry.csv + camera_matrix.csv)."
        )
    print(f"[DNSplatter] Stray dir: {stray_dir}")

    # Load Stray data.
    pose_map = _load_stray_odometry(stray_dir / "odometry.csv")
    K = _load_stray_intrinsics(stray_dir / "camera_matrix.csv")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"[DNSplatter] poses={len(pose_map)}  K=(fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f})")

    # Probe RGB resolution from the first JPG.
    jpgs = sorted(frames_dir.glob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No JPGs in {frames_dir}")
    with Image.open(jpgs[0]) as im:
        rgb_w, rgb_h = im.size
    print(f"[DNSplatter] RGB resolution {rgb_w}x{rgb_h}  ({len(jpgs)} jpgs)")

    # Probe Stray depth resolution.
    depth_dir = stray_dir / "depth"
    depth_samples = sorted(depth_dir.glob("*.png"))
    if not depth_samples:
        raise FileNotFoundError(f"No depth PNGs in {depth_dir}")
    with Image.open(depth_samples[0]) as im:
        depth_w0, depth_h0 = im.size
    print(f"[DNSplatter] Source depth resolution {depth_w0}x{depth_h0} → upsampling to {rgb_w}x{rgb_h}")

    # Optionally restrict to PLY-traceable keyframes.
    allowed: Optional[set] = None
    if use_keyframes_only:
        ply_path = (output_dir / "cleaned_cloud.ply")
        if not ply_path.exists():
            for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
                if (output_dir / alt).exists():
                    ply_path = output_dir / alt
                    break
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from segmentation.pipeline import _load_ply_origins
            origins = _load_ply_origins(ply_path)
            if origins is not None:
                _xyz, frame_global, _pr, _pc = origins
                allowed = set(int(x) for x in np.unique(frame_global))
                print(f"[DNSplatter] keyframes filter: {len(allowed)} unique frames "
                      f"in cleaned_cloud.ply")
        except Exception as e:
            logger.warning(f"keyframes filter failed ({e}) — falling back to all frames")

    # Create output structure.
    root = output_dir / "dn_splatter"
    images_out = root / "images"
    depths_out = root / "depths"
    for d in (root, images_out, depths_out):
        d.mkdir(parents=True, exist_ok=True)

    frames_meta: List[Dict] = []
    skipped_no_pose = skipped_no_depth = skipped_no_jpg = skipped_filtered = 0

    sorted_frames = sorted(pose_map.keys())
    for fidx in sorted_frames:
        if allowed is not None and fidx not in allowed:
            skipped_filtered += 1
            continue

        jpg_src = frames_dir / f"{fidx:06d}.jpg"
        if not jpg_src.exists():
            skipped_no_jpg += 1
            continue

        depth_src = depth_dir / f"{fidx:06d}.png"
        if not depth_src.exists():
            skipped_no_depth += 1
            continue

        # Symlink / copy RGB.
        jpg_dst = images_out / f"{fidx:06d}.jpg"
        if jpg_dst.exists() or jpg_dst.is_symlink():
            jpg_dst.unlink()
        if use_symlinks:
            os.symlink(str(jpg_src), str(jpg_dst))
        else:
            from shutil import copyfile
            copyfile(str(jpg_src), str(jpg_dst))

        # Upsample depth NN and write as uint16 PNG.
        with Image.open(depth_src) as im:
            depth_src_arr = np.array(im, dtype=np.uint16)
        depth_up = _upsample_depth_nn(depth_src_arr, rgb_h, rgb_w)
        depth_dst = depths_out / f"{fidx:06d}.png"
        Image.fromarray(depth_up, mode="I;16").save(str(depth_dst))

        # Convert pose to OpenGL c2w.
        c2w_gl = _opencv_c2w_to_opengl(pose_map[fidx])

        frames_meta.append({
            "file_path": f"images/{fidx:06d}.jpg",
            "depth_file_path": f"depths/{fidx:06d}.png",
            "transform_matrix": c2w_gl.tolist(),
        })

    if not frames_meta:
        raise RuntimeError(
            f"No frames kept (skipped: no_pose={skipped_no_pose} no_depth={skipped_no_depth} "
            f"no_jpg={skipped_no_jpg} filtered={skipped_filtered})"
        )

    # Init point cloud — nerfstudio's normal-nerfstudio dataparser hard-requires
    # a points3D source. Downsample the cleaned_cloud to ~500k pts and write as
    # init.ply. The trainer reads ply_file_path from transforms.json.
    init_ply_rel: Optional[str] = None
    ply_path = output_dir / "cleaned_cloud.ply"
    if not ply_path.exists():
        for alt in ("cleaned_cloud_symlink.ply", "merged.ply"):
            if (output_dir / alt).exists():
                ply_path = output_dir / alt
                break
    if ply_path.exists():
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from segmentation.pipeline import _load_ply_origins
            origins = _load_ply_origins(ply_path)
            if origins is not None:
                xyz, _fg, _pr, _pc = origins
                # Random downsample to 500k pts (deterministic).
                if len(xyz) > 500_000:
                    rng = np.random.default_rng(42)
                    sel = rng.choice(len(xyz), size=500_000, replace=False)
                    xyz_init = xyz[sel]
                else:
                    xyz_init = xyz
                # Minimal ASCII PLY (Nerfstudio accepts it).
                init_path = root / "init.ply"
                with open(init_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(xyz_init)}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("end_header\n")
                    for p in xyz_init:
                        f.write(f"{p[0]} {p[1]} {p[2]}\n")
                init_ply_rel = "init.ply"
                print(f"[DNSplatter] init point cloud: {init_path} ({len(xyz_init):,} pts)")
        except Exception as e:
            logger.warning(f"could not build init.ply: {e}")

    # Write transforms.json.
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": rgb_w,
        "h": rgb_h,
        # uint16 depth in mm → meters
        "depth_unit_scale_factor": 0.001,
        "frames": frames_meta,
    }
    if init_ply_rel:
        transforms["ply_file_path"] = init_ply_rel
    out_json = root / "transforms.json"
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)

    elapsed = time.time() - t0
    print(f"[DNSplatter] ✅ wrote {out_json}  ({len(frames_meta)} frames, {elapsed:.1f}s)")
    print(f"[DNSplatter]   skipped: filtered={skipped_filtered} "
          f"no_jpg={skipped_no_jpg} no_depth={skipped_no_depth}")
    return out_json


# ── CLI ────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(description="Convert a Stray session to DN-Splatter / nerfstudio format.")
    p.add_argument("--session-dir", required=True, type=Path,
                   help="Session root (contains frames/ and output/ subdirs).")
    p.add_argument("--keyframes-only", action="store_true",
                   help="Restrict to the keyframes present in cleaned_cloud.ply (PLY-traceable). "
                        "Default: use every Stray frame with pose+depth+JPG.")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink JPGs from frames/ instead of copying. Default: copy "
                        "(symlinks break when tar'd & sent to a remote machine).")
    args = p.parse_args()

    session = args.session_dir.resolve()
    out = export_dn_splatter_dataset(
        session_dir=session,
        frames_dir=session / "frames",
        output_dir=session / "output",
        use_keyframes_only=args.keyframes_only,
        use_symlinks=args.symlink,
    )
    print(f"\nReady. Train with:")
    print(f"  ns-train dn-splatter --pipeline.model.use-depth-loss True \\")
    print(f"      --pipeline.model.depth-lambda 0.2 \\")
    print(f"      --pipeline.model.use-normal-loss True \\")
    print(f"      --pipeline.model.use-normal-tv-loss True \\")
    print(f"      --pipeline.model.normal-supervision depth \\")
    print(f"      normal-nerfstudio --data {out.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
