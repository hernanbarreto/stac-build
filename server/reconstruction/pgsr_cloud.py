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
                voxel_m: float = 0.006, max_depth_m: Optional[float] = 20.0,
                edge_thresh: float = 0.04, mv_neighbors: int = 4,
                mv_min_views: int = 2, mv_tau_rel: float = 0.02,
                sor: bool = True, log=None) -> Path:
    """Unproject the pgsr_render depths at the FINAL poses → one coloured,
    FILTERED, voxel-downsampled cloud. Three gates keep the render junk out
    (the mesh survives it because the TSDF averages/truncates — a raw
    unprojection does not):
      1. depth-edge mask (silhouette interpolation between object and
         background — same gradient cull the TSDF applies),
      2. MULTI-VIEW CONSISTENCY vote (reuses the Phase-B machinery on the
         renders: a point enters only if ≥ mv_min_views neighbour keyframes
         confirm the surface within mv_tau_rel — kills semi-transparent
         floater geometry),
      3. depth cap + statistical outlier removal after the downsample.
    Fail-fast on missing inputs. Returns the PLY path."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir, frames_dir = Path(output_dir), Path(frames_dir)
    render_dir = output_dir / "pgsr_render"
    files = sorted(render_dir.glob("frame_*.npz"))
    if not files:
        raise RuntimeError("pgsr_cloud: no rendered depths in pgsr_render/ — "
                           "run the PGSR stage first")

    from reconstruction.scale_align import _read_poses, _omega_depth
    from reconstruction.mv_consistency import (compute_frame_consistency,
                                               _neighbor_indices)
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

    # load all renders once (RAM: ~2 MB/frame) + per-frame K at the render grid
    frames = {}
    for p in files:
        n = int(p.stem.split("_")[1])
        T, kr = T_map.get(n), K_rows.get(n)
        if T is None or kr is None:
            continue
        d = np.load(p)
        depth = d["depth"].astype(np.float32)
        valid = (d["valid"] if "valid" in d.files else depth > 1e-4).astype(bool)
        H, W = depth.shape
        sx, sy = (W / Wd, H / Hd) if Hd else (1.0, 1.0)
        K = np.array([[kr[0] * sx, 0, kr[2] * sx],
                      [0, kr[1] * sy, kr[3] * sy], [0, 0, 1.0]])
        frames[n] = {"depth": depth, "valid": valid, "K": K, "T": T}
    if len(frames) < 3:
        raise RuntimeError(f"pgsr_cloud: only {len(frames)} usable renders")

    ordered = sorted(frames.keys())
    centres = np.stack([frames[n]["T"][:3, 3] for n in ordered])
    nbrs = _neighbor_indices(centres, mv_neighbors)

    pts_all, rgb_all = [], []
    raw_px = kept_px = 0
    for i, n in enumerate(ordered):
        f = frames[n]
        depth, valid, K, T = f["depth"], f["valid"], f["K"], f["T"]
        H, W = depth.shape
        # 1) silhouette/edge cull (gradient in metres, same rule as the TSDF)
        dz = np.where(valid, depth, 0.0)
        gx = np.abs(np.diff(dz, axis=1, prepend=dz[:, :1]))
        gy = np.abs(np.diff(dz, axis=0, prepend=dz[:1, :]))
        mask = valid & (gx < edge_thresh) & (gy < edge_thresh)
        # 2) multi-view consistency vote against the neighbour renders
        js = [ordered[j] for j in nbrs[i]]
        votes, _, _ = compute_frame_consistency(
            depth, K, T, [frames[j]["depth"] for j in js],
            [frames[j]["K"] for j in js], [frames[j]["T"] for j in js],
            mv_tau_rel)
        mask &= votes >= int(mv_min_views)
        # 3) depth cap
        if max_depth_m:
            mask &= depth < float(max_depth_m)

        jp = frames_dir / f"{n:06d}.jpg"
        if not jp.exists():
            raise RuntimeError(f"pgsr_cloud: frame image {jp.name} missing")
        rgb = np.asarray(Image.open(jp).convert("RGB").resize((W, H),
                                                              Image.BILINEAR))
        m = mask[::stride, ::stride]
        raw_px += int(valid[::stride, ::stride].sum())
        kept_px += int(m.sum())
        if not m.any():
            continue
        v, u = np.mgrid[0:H:stride, 0:W:stride]
        z = depth[::stride, ::stride][m].astype(np.float64)
        u, vv = u[m].astype(np.float64), v[m].astype(np.float64)
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (vv - K[1, 2]) / K[1, 1] * z
        Pw = (T @ np.stack([x, y, z, np.ones_like(z)]))[:3].T
        pts_all.append(Pw.astype(np.float32))
        rgb_all.append(rgb[::stride, ::stride][m])
    if not pts_all:
        raise RuntimeError("pgsr_cloud: no points survived the consistency gates")
    pts = np.concatenate(pts_all)
    cols = np.concatenate(rgb_all)
    _log(f"unprojected {len(pts):,} CONSISTENT points from {len(ordered)} renders "
         f"(kept {100.0 * kept_px / max(raw_px, 1):.1f}% — edge cull + "
         f"{mv_min_views}-view vote @ {mv_tau_rel:.0%} + <{max_depth_m:g} m)")

    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    if voxel_m and voxel_m > 0:
        pc = pc.voxel_down_sample(float(voxel_m))
    if sor:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
        _log(f"SOR: {len(pc.points):,} points after outlier removal")
    out = output_dir / CLOUD_NAME
    o3d.io.write_point_cloud(str(out), pc, write_ascii=False, compressed=False)
    _log(f"consistent cloud: {len(pc.points):,} points (voxel {voxel_m * 1000:.0f} mm) "
         f"→ {out.name}")
    return out
