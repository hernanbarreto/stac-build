"""
Consistent point cloud from the PGSR-rendered depths (precision mode).
================================================================================
The user-facing point cloud (Potree) was built from the RAW fused chunk clouds —
with all the per-frame pose/depth ghosting the later stages correct. The MESH,
in precision mode, integrates the PGSR-rendered depths at the refined poses —
so cloud and mesh disagreed exactly where the corrections did their job.

This module closes the loop ("ir para atrás con la nube"): it re-builds the
point cloud FROM the same photometrically-verified source the mesh uses —
per-keyframe PGSR depth + final (pose-refined) poses + photo colour — so what
you see in the Potree viewer IS the corrected geometry:

    output/pgsr_cloud.ply        (xyz + rgb, voxel-downsampled)

The Potree octree is then rebuilt from it (potree_converter.convert_ply_to_potree
with ply_override). cleaned_cloud.ply is NOT touched: downstream consumers
(segmentation mask→cloud mapping, BIM comparison) keep their source; the
consistent cloud is the VIEW artifact. Enabled by config
reconstruction.pgsr.consistent_cloud (default true; the pgsr_worker logs every
step — never silent).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("PGSRCloud")

CLOUD_NAME = "pgsr_cloud.ply"


def build_cloud(output_dir: Path, frames_dir: Path, stride: int = 2,
                voxel_m: float = 0.006, max_depth_m: Optional[float] = None,
                log=None) -> Path:
    """Unproject every pgsr_render depth at the FINAL poses → one coloured,
    voxel-downsampled cloud. Fail-fast on missing inputs. Returns the PLY path."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir, frames_dir = Path(output_dir), Path(frames_dir)
    render_dir = output_dir / "pgsr_render"
    files = sorted(render_dir.glob("frame_*.npz"))
    if not files:
        raise RuntimeError("pgsr_cloud: no rendered depths in pgsr_render/ — "
                           "run the PGSR stage first")

    from reconstruction.scale_align import _read_poses, _omega_depth
    from PIL import Image
    lines, nums, _ = _read_poses(output_dir)
    T_map = {n: np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
             for n, ln in zip(nums, lines) if len(ln.split()) == 16}
    rows = [[float(x) for x in ln.split()]
            for ln in (output_dir / "intrinsic.txt").read_text().splitlines()
            if ln.strip()]
    K_rows = {n: r for n, r in zip(nums, rows)}
    om = _omega_depth(output_dir)
    Hd, Wd = next(iter(om.values())).shape if om else (None, None)

    pts_all, rgb_all = [], []
    n_used = 0
    for p in files:
        n = int(p.stem.split("_")[1])
        T = T_map.get(n)
        kr = K_rows.get(n)
        if T is None or kr is None:
            continue
        d = np.load(p)
        depth = d["depth"].astype(np.float64)
        valid = d["valid"] if "valid" in d.files else depth > 1e-4
        H, W = depth.shape
        sx, sy = (W / Wd, H / Hd) if Hd else (1.0, 1.0)
        fx, fy, cx, cy = kr[0] * sx, kr[1] * sy, kr[2] * sx, kr[3] * sy
        jp = frames_dir / f"{n:06d}.jpg"
        if not jp.exists():
            raise RuntimeError(f"pgsr_cloud: frame image {jp.name} missing")
        rgb = np.asarray(Image.open(jp).convert("RGB").resize((W, H),
                                                              Image.BILINEAR))
        v, u = np.mgrid[0:H:stride, 0:W:stride]
        z = depth[::stride, ::stride]
        m = valid[::stride, ::stride] & np.isfinite(z) & (z > 1e-3)
        if max_depth_m:
            m &= z < float(max_depth_m)
        if not m.any():
            continue
        u, vv, z = u[m].astype(np.float64), v[m].astype(np.float64), z[m]
        x = (u - cx) / fx * z
        y = (vv - cy) / fy * z
        Pw = (T @ np.stack([x, y, z, np.ones_like(z)]))[:3].T
        pts_all.append(Pw.astype(np.float32))
        rgb_all.append(rgb[::stride, ::stride][m])
        n_used += 1
    if not pts_all:
        raise RuntimeError("pgsr_cloud: no valid points unprojected")
    pts = np.concatenate(pts_all)
    cols = np.concatenate(rgb_all)
    _log(f"unprojected {len(pts):,} points from {n_used} rendered keyframes "
         f"(stride {stride})")

    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    if voxel_m and voxel_m > 0:
        pc = pc.voxel_down_sample(float(voxel_m))
    out = output_dir / CLOUD_NAME
    o3d.io.write_point_cloud(str(out), pc, write_ascii=False, compressed=False)
    _log(f"consistent cloud: {len(pc.points):,} points (voxel {voxel_m * 1000:.0f} mm) "
         f"→ {out.name}")
    return out
