"""
NKSR scene mesher — server-env orchestrator (research evaluation, 2026-08-18).
================================================================================
``tsdf.mesh_method: "nksr"`` → the cleaned cloud goes straight into NVIDIA's
Neural Kernel Surface Reconstruction (subprocess in the `nksr` env, see
run_nksr.sh) and the resulting mesh gets the SAME delivery as every other
mesher: anti-invention gate → speck removal → texrecon photo texture →
compressed scene.glb in the live slot.

Doctrine guard: NKSR is a LEARNED prior — it may extend surface beyond the
data. The ANTI-INVENTION GATE is deterministic and non-negotiable: any face
whose three vertices all sit farther than ``nksr_max_dist_m`` from the cleaned
cloud is dropped (counts logged, auditable). The cloud is never touched.

License: NVIDIA Source Code License-NC (research/evaluation) — user-approved
for this evaluation phase (2026-08-18); revisit before commercial use.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("NKSRScene")


def export_nksr_scene(
    output_dir: Path,
    frames_dir: Path,
    session_dir: Optional[Path] = None,
    texture: bool = True,
    texture_max_views: int = 400,
    nksr_detail_level: float = 0.0,
    nksr_voxel_size: float = 0.02,       # finest kernel voxel (m); 0 = detail_level
                                         # rules. 2 cm measured best on synthetic
                                         # (finer collapsed the ks model)
    nksr_chunk_size: float = 0.0,        # metres; >0 = out-of-core chunked mode
    nksr_max_points: int = 0,            # 0 = feed ALL cloud points
    nksr_max_dist_m: float = 0.05,       # ANTI-INVENTION gate: faces with all 3
                                         # verts farther than this from the cloud
                                         # are dropped (learned prior ≠ data)
    nksr_min_component_tris: int = 200,  # speck clusters below this are dropped
    progress_cb=None,
    **_ignored,                          # tsdf-only kwargs arrive here harmlessly
) -> Optional[str]:
    """cleaned_cloud.ply → NKSR mesh → gated + textured scene.glb.
    Returns the glb path (str) or None. Fail-fast with the exact reason."""
    import open3d as o3d
    t0 = time.time()
    output_dir, frames_dir = Path(output_dir), Path(frames_dir)
    cloud = output_dir / "cleaned_cloud.ply"
    if not cloud.exists():
        raise RuntimeError("nksr: cleaned_cloud.ply not found — run the pipeline first")

    if progress_cb:
        progress_cb("integrating", 0.0, None)
    server_dir = Path(__file__).resolve().parent.parent
    raw_ply = output_dir / "tsdf" / "nksr_raw.ply"
    cmd = ["bash", str(server_dir / "run_nksr.sh"),
           "--cloud", str(cloud),
           "--output-dir", str(output_dir),
           "--out", str(raw_ply),
           "--detail-level", str(float(nksr_detail_level)),
           "--voxel-size", str(float(nksr_voxel_size)),
           "--chunk-size", str(float(nksr_chunk_size)),
           "--max-points", str(int(nksr_max_points))]
    logger.info(f"[nksr-scene] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"[nksr-scene] {line}")
    proc.wait()
    if proc.returncode != 0 or not raw_ply.exists():
        raise RuntimeError(f"nksr subprocess failed (exit {proc.returncode}) — "
                           f"see the [nksr] lines above")

    mesh = o3d.io.read_triangle_mesh(str(raw_ply))
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if not len(F):
        raise RuntimeError("nksr: empty mesh")
    logger.info(f"[nksr-scene] raw mesh: {len(V):,} verts / {len(F):,} tris")

    # ── ANTI-INVENTION GATE (deterministic, doctrine) ──
    if progress_cb:
        progress_cb("extracting", time.time() - t0, None)
    from scipy.spatial import cKDTree
    cc = o3d.io.read_point_cloud(str(cloud))
    d_v, _ = cKDTree(np.asarray(cc.points)).query(V, k=1, workers=8)
    far_v = d_v > float(nksr_max_dist_m)
    drop_f = far_v[F].all(axis=1)
    if drop_f.any():
        mesh.remove_triangles_by_mask(drop_f)
        mesh.remove_unreferenced_vertices()
        logger.info(f"[nksr-scene] anti-invention gate: dropped {int(drop_f.sum()):,} "
                    f"faces with all verts >{nksr_max_dist_m * 1000:.0f}mm from the "
                    f"cloud ({100.0 * drop_f.mean():.1f}%)")
    # speck removal (same policy as the TSDF path)
    tri_c, _, _ = mesh.cluster_connected_triangles()
    tri_c = np.asarray(tri_c)
    sizes = np.bincount(tri_c) if len(tri_c) else np.array([])
    small = np.isin(tri_c, np.where(sizes < int(nksr_min_component_tris))[0])
    if small.any():
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
        logger.info(f"[nksr-scene] dropped {int(small.sum()):,} speck tris "
                    f"(<{nksr_min_component_tris} tris/component)")
    mesh.compute_vertex_normals()

    # ── deliverable: texture + compress + meta (same slot as the TSDF) ──
    if progress_cb:
        progress_cb("texturing", time.time() - t0, None)
    scene_dir = output_dir / "tsdf" / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    glb_path = scene_dir / "scene.glb"
    geom_ply = scene_dir / "_scene_geom.ply"
    o3d.io.write_triangle_mesh(str(geom_ply), mesh)
    textured = False
    if texture:
        try:
            from segmentation.session_io import _load_camera_source
            cam = _load_camera_source(session_dir or output_dir.parent, output_dir)
            from reconstruction.texture_bake import bake_texture
            res = bake_texture(
                mesh_path=geom_ply, frames_dir=frames_dir,
                pose_map=cam.pose_map, intrinsics_map=cam.intrinsics_map,
                out_glb=glb_path,
                **({"max_views": int(texture_max_views)}
                   if texture_max_views and int(texture_max_views) > 0 else {}))
            textured = res is not None
        except Exception as e:  # noqa: BLE001 — vertex-colour glb still ships
            logger.warning(f"[nksr-scene] texrecon failed ({e}) — vertex colours")
    if not textured:
        o3d.io.write_triangle_mesh(str(glb_path), mesh)
    geom_ply.unlink(missing_ok=True)
    try:
        shutil.copy2(glb_path, glb_path.with_suffix(".glb.orig"))
        from segmentation.tsdf_export import _compress_scene_glb
        _compress_scene_glb(glb_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[nksr-scene] glb compression skipped ({e})")
    (scene_dir / "scene_meta.json").write_text(json.dumps({
        "mesh_method": "nksr",
        "detail_level": float(nksr_detail_level),
        "chunk_size_m": float(nksr_chunk_size),
        "anti_invention_max_dist_m": float(nksr_max_dist_m),
        "n_vertices": len(mesh.vertices),
        "n_triangles": len(mesh.triangles),
        "textured": textured,
        "license_note": "NKSR: NVIDIA Source Code License-NC (research eval)",
    }, indent=2))
    raw_ply.unlink(missing_ok=True)
    if progress_cb:
        progress_cb("done", time.time() - t0, str(glb_path))
    logger.info(f"[nksr-scene] ✅ {len(mesh.vertices):,} verts / "
                f"{len(mesh.triangles):,} tris textured={textured} "
                f"in {time.time() - t0:.0f}s → {glb_path}")
    return str(glb_path)
