"""
GPU UV-atlas texture baker (nvdiffrast) — a fast, CUDA-native replacement for
texrecon/mvs-texturing. On an A6000 this bakes in ~minutes vs texrecon's ~hour,
with equal or better quality (per-texel best-view blending with photo-consistent
occlusion handling).

Pipeline:
  1. UV-unwrap the mesh with xatlas (per-vertex UVs, remapped topology).
  2. Rasterize the mesh IN UV SPACE with nvdiffrast → per atlas texel: the
     interpolated world-space position + normal (trivial ortho projection,
     uv∈[0,1] → ndc∈[-1,1], z=0).
  3. For each posed keyframe, project every valid texel into the camera using
     the frame's OWN native intrinsics (npz, depth resolution) to get
     NORMALISED image coords. Those coords sample BOTH:
        - the per-frame DEPTH map (occlusion oracle): a texel is visible iff its
          camera-space z matches the measured depth there (within tolerance) —
          no second rasterisation needed, and it reuses the real sensor/DA3
          depth we already trust.
        - the FULL-RES JPG (colour) for maximum texture detail.
     Accumulate colour weighted by view quality (front-facing cosine, proximity).
  4. Normalise → atlas; dilate to fill unseen texels; export a textured GLB.

Coordinate conventions match the TSDF/integrate path: pose_map is camera→world
(c2w), keyed by keyframe-INDEX; intrinsics are pinhole pixel K at the depth
resolution; JPGs are named by ORIGINAL frame number (resolve via name_map).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("NvdiffrastBake")


def is_available() -> bool:
    """True if the GPU baker can run (nvdiffrast + xatlas + CUDA torch)."""
    try:
        import torch
        import nvdiffrast.torch  # noqa: F401
        import xatlas  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_npz_dir(output_dir: Optional[Path]) -> Optional[Path]:
    if output_dir is None:
        return None
    for d in (output_dir / "da3_run" / "results_output",
              output_dir / "results_output",
              output_dir / "gaus_slam_run" / "results_output"):
        if d.exists() and any(d.glob("frame_*.npz")):
            return d
    return None


def bake_vertex_colors_gpu(
    vertices: np.ndarray,                    # (N,3) world coords (FULL mesh, no decimation)
    vertex_normals: np.ndarray,              # (N,3)
    frames_dir: Path,
    pose_map: Dict[int, np.ndarray],         # keyframe-idx → (4,4) c2w
    intrinsics_map: Dict[int, np.ndarray],   # keyframe-idx → (3,3) K (depth res)
    output_dir: Optional[Path] = None,       # to locate results_output npz (depth)
    name_map: Optional[Dict[int, str]] = None,
    init_colors: Optional[np.ndarray] = None,  # (N,3) [0,1] fallback for unseen verts
    frame_indices: Optional[List[int]] = None,
    max_views: int = 800,
    cos_power: float = 3.0,
    depth_tol: float = 0.06,
    device: str = "cuda",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[np.ndarray]:
    """Multi-view photo blend per VERTEX on the full mesh — keeps geometry exact
    (no decimation/UV unwrap), colours each vertex from the keyframes. Returns
    (N,3) float[0,1] vertex colours, or None on failure.

    Same projection/visibility/weighting as the atlas baker, but the elements are
    mesh vertices instead of atlas texels — so colour resolution is bounded by
    mesh density (fine for a dense TSDF mesh) and there is no atlas. Visibility
    uses the per-frame depth map as the occlusion oracle (reuses trusted depth).
    """
    import torch
    from PIL import Image

    t0 = time.time()
    frames_dir = Path(frames_dir)
    dev = torch.device(device)

    def _name(fi: int) -> Path:
        if name_map and fi in name_map:
            return frames_dir / Path(str(name_map[fi])).name
        return frames_dir / f"{fi:06d}.jpg"

    npz_dir = _resolve_npz_dir(output_dir)
    if npz_dir is None:
        logger.error("[VertBake] no results_output/*.npz (depth) found")
        return None

    txyz = torch.tensor(np.asarray(vertices, dtype=np.float32), device=dev)
    tnrm = torch.nn.functional.normalize(
        torch.tensor(np.asarray(vertex_normals, dtype=np.float32), device=dev), dim=-1)
    N = txyz.shape[0]
    col_sum = torch.zeros((N, 3), device=dev)
    w_sum = torch.zeros((N, 1), device=dev)

    if frame_indices is None:
        frame_indices = sorted(pose_map.keys())
    usable = [int(fi) for fi in frame_indices
              if pose_map.get(int(fi)) is not None and _name(int(fi)).exists()
              and (npz_dir / f"frame_{int(fi)}.npz").exists()]
    if len(usable) > max_views:
        step = len(usable) / float(max_views)
        usable = [usable[int(i * step)] for i in range(max_views)]
    if len(usable) < 2:
        logger.error(f"[VertBake] too few usable views ({len(usable)})")
        return None
    logger.info(f"[VertBake] {N:,} verts × {len(usable)} views")
    if progress_cb:
        progress_cb("baking")

    n_done = 0
    for fi in usable:
        try:
            c2w = np.asarray(pose_map[fi], dtype=np.float64)
            c2w4 = np.eye(4); c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
            w2c = np.linalg.inv(c2w4)
            R_t = torch.tensor(w2c[:3, :3], device=dev, dtype=torch.float32)
            tr = torch.tensor(w2c[:3, 3], device=dev, dtype=torch.float32)
            cam_c = torch.tensor(c2w4[:3, 3], device=dev, dtype=torch.float32)

            z = np.load(str(npz_dir / f"frame_{fi}.npz"))
            depth_np = z["depth"].astype(np.float32)
            Hd, Wd = depth_np.shape
            K = intrinsics_map.get(fi)
            if K is None and "intrinsics" in z:
                K = np.asarray(z["intrinsics"], dtype=np.float64).reshape(3, 3)
            if K is None:
                continue
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            Xc = txyz @ R_t.T + tr
            zc = Xc[:, 2]
            front = zc > 1e-3
            zc_safe = torch.clamp(zc, min=1e-3)
            u_n = (fx * Xc[:, 0] / zc_safe + cx) / Wd
            v_n = (fy * Xc[:, 1] / zc_safe + cy) / Hd
            inb = front & (u_n >= 0) & (u_n < 1) & (v_n >= 0) & (v_n < 1)
            grid = torch.stack([u_n * 2 - 1, v_n * 2 - 1], dim=-1)[None, :, None, :]

            depth_t = torch.tensor(depth_np, device=dev)[None, None]
            dsamp = torch.nn.functional.grid_sample(
                depth_t, grid, mode="bilinear", align_corners=False)[0, 0, :, 0]
            vis = inb & (dsamp > 1e-3) & ((zc - dsamp).abs() < depth_tol)

            with Image.open(_name(fi)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            rgb_t = torch.tensor(rgb, device=dev).permute(2, 0, 1)[None]
            csamp = torch.nn.functional.grid_sample(
                rgb_t, grid, mode="bilinear", align_corners=False)[0, :, :, 0].T

            vdir = torch.nn.functional.normalize(cam_c[None] - txyz, dim=-1)
            cosang = (tnrm * vdir).sum(-1).clamp(min=0.0)
            w = ((cosang ** cos_power) * vis.float())[:, None]
            col_sum += csamp * w
            w_sum += w
            n_done += 1
            if progress_cb and n_done % 100 == 0:
                progress_cb(f"baking {n_done}/{len(usable)}")
        except Exception as e:
            logger.warning(f"[VertBake] view {fi} skipped ({e})")
            continue

    seen = (w_sum[:, 0] > 1e-8)
    out = torch.zeros((N, 3), device=dev)
    out[seen] = col_sum[seen] / w_sum[seen]
    if init_colors is not None:
        ic = torch.tensor(np.asarray(init_colors, dtype=np.float32), device=dev)
        if ic.shape == out.shape:
            out[~seen] = ic[~seen]              # keep VBG colour where no photo saw it
    logger.info(f"[VertBake] ✅ {100.0*seen.float().mean().item():.1f}% verts photo-coloured "
                f"in {time.time()-t0:.1f}s")
    return out.clamp(0, 1).cpu().numpy()


def bake_texture_gpu(
    mesh_path: Path,
    frames_dir: Path,
    pose_map: Dict[int, np.ndarray],         # keyframe-idx → (4,4) c2w
    intrinsics_map: Dict[int, np.ndarray],   # keyframe-idx → (3,3) K (depth res)
    out_glb: Path,
    name_map: Optional[Dict[int, str]] = None,    # idx → real JPG filename
    output_dir: Optional[Path] = None,            # to locate results_output npz (depth)
    frame_indices: Optional[List[int]] = None,
    atlas_res: int = 4096,
    max_views: int = 800,
    unwrap_faces: int = 150_000,              # decimate before xatlas (it is CPU/slow)
    cos_power: float = 3.0,                   # higher → favour frontal views
    depth_tol: float = 0.06,                  # m; occlusion match tolerance
    dilate_iters: int = 16,                   # fill unseen texels
    device: str = "cuda",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Bake a UV-atlas texture onto ``mesh_path`` → ``out_glb``. Returns the GLB
    path on success, or None on failure (caller keeps the untextured mesh)."""
    import torch
    import nvdiffrast.torch as dr
    import xatlas
    import trimesh
    from PIL import Image

    t0 = time.time()
    mesh_path, frames_dir, out_glb = Path(mesh_path), Path(frames_dir), Path(out_glb)
    dev = torch.device(device)

    def _name(fi: int) -> Path:
        if name_map and fi in name_map:
            return frames_dir / Path(str(name_map[fi])).name
        return frames_dir / f"{fi:06d}.jpg"

    npz_dir = _resolve_npz_dir(output_dir)
    if npz_dir is None:
        logger.error("[NvBake] no results_output/*.npz (depth) found — cannot do "
                     "depth-based visibility")
        return None

    # ── 1. load geometry ──
    if progress_cb:
        progress_cb("unwrap")
    tm = trimesh.load(str(mesh_path), force="mesh", process=False)
    V = np.asarray(tm.vertices, dtype=np.float32)
    F = np.asarray(tm.faces, dtype=np.uint32)
    if len(V) == 0 or len(F) == 0:
        logger.error("[NvBake] empty mesh")
        return None

    # xatlas UV-unwrap is CPU-only and scales poorly (a 1.5M-tri mesh takes tens
    # of minutes). Decimate first — Polycam-style, the geometry stays light and
    # the detail lives in the baked atlas. The bake samples full-res photos, so a
    # coarser mesh barely affects texture sharpness.
    if unwrap_faces and len(F) > unwrap_faces:
        import open3d as o3d
        n0 = len(F)
        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V.astype(np.float64)),
            o3d.utility.Vector3iVector(F.astype(np.int32)))
        om = om.simplify_quadric_decimation(int(unwrap_faces))
        om.remove_unreferenced_vertices()
        V = np.asarray(om.vertices, dtype=np.float32)
        F = np.asarray(om.triangles, dtype=np.uint32)
        logger.info(f"[NvBake] decimated {n0:,} → {len(F):,} faces for unwrap")

    # ── 1b. UV unwrap (xatlas) — remaps topology ──
    # Speed: fix the pack resolution (default 0 → binary-searches resolutions,
    # each a full pack = very slow) and grow bigger charts (fewer charts → faster
    # segmentation + packing). Quality of the chart layout is irrelevant for
    # photo baking — we only need a valid bijective parametrization.
    t_uv = time.time()
    try:
        co = xatlas.ChartOptions()
        co.max_cost = 8.0
        co.max_iterations = 1
        po = xatlas.PackOptions()
        po.resolution = int(atlas_res)
        po.padding = 2
        po.bruteForce = False
        po.rotate_charts = False
        vmapping, indices, uvs = xatlas.parametrize(V, F, chart_options=co, pack_options=po)
    except TypeError:
        vmapping, indices, uvs = xatlas.parametrize(V, F)
    logger.info(f"[NvBake] xatlas unwrap {time.time()-t_uv:.1f}s")
    V2 = V[vmapping]                                  # (Nv,3) remapped verts
    uvs = np.asarray(uvs, dtype=np.float32)           # (Nv,2) in [0,1]
    faces2 = np.asarray(indices, dtype=np.int32)      # (M,3)
    # per-vertex normals on the remapped mesh
    tm2 = trimesh.Trimesh(vertices=V2, faces=faces2, process=False)
    N2 = np.asarray(tm2.vertex_normals, dtype=np.float32)
    logger.info(f"[NvBake] mesh {len(V):,}v/{len(F):,}f → atlas verts {len(V2):,}  "
                f"uv-charts unwrapped  atlas={atlas_res}²")

    V2_t = torch.tensor(V2, device=dev)
    N2_t = torch.tensor(N2, device=dev)
    uv_t = torch.tensor(uvs, device=dev)
    faces_t = torch.tensor(faces2, device=dev, dtype=torch.int32)

    # ── 2. rasterize the atlas in UV space → per-texel world pos + normal ──
    glctx = dr.RasterizeCudaContext(device=dev)
    # uv∈[0,1] → ndc∈[-1,1]; z=0, w=1 (orthographic in UV space)
    ndc = uv_t * 2.0 - 1.0
    pos_clip = torch.cat([ndc, torch.zeros_like(ndc[:, :1]),
                          torch.ones_like(ndc[:, :1])], dim=1)[None]  # (1,Nv,4)
    rast, _ = dr.rasterize(glctx, pos_clip, faces_t, resolution=[atlas_res, atlas_res])
    texel_mask = (rast[..., 3] > 0)[0]                # (R,R) bool — covered texels
    world_pos, _ = dr.interpolate(V2_t[None], rast, faces_t)   # (1,R,R,3)
    world_nrm, _ = dr.interpolate(N2_t[None], rast, faces_t)   # (1,R,R,3)
    world_pos = world_pos[0]
    world_nrm = torch.nn.functional.normalize(world_nrm[0], dim=-1)

    sel = texel_mask.reshape(-1)                      # (R*R,)
    P = int(sel.sum().item())
    if P == 0:
        logger.error("[NvBake] UV rasterisation produced no covered texels")
        return None
    txyz = world_pos.reshape(-1, 3)[sel]              # (P,3)
    tnrm = world_nrm.reshape(-1, 3)[sel]              # (P,3)
    col_sum = torch.zeros((P, 3), device=dev)
    w_sum = torch.zeros((P, 1), device=dev)
    logger.info(f"[NvBake] {P:,} covered texels ({100.0*P/(atlas_res*atlas_res):.1f}% of atlas)")

    # ── 3. accumulate colour over views ──
    if frame_indices is None:
        frame_indices = sorted(pose_map.keys())
    usable = [int(fi) for fi in frame_indices
              if pose_map.get(int(fi)) is not None and _name(int(fi)).exists()
              and (npz_dir / f"frame_{int(fi)}.npz").exists()]
    if len(usable) > max_views:
        step = len(usable) / float(max_views)
        usable = [usable[int(i * step)] for i in range(max_views)]
    if len(usable) < 2:
        logger.error(f"[NvBake] too few usable views ({len(usable)})")
        return None
    if progress_cb:
        progress_cb("baking")
    logger.info(f"[NvBake] baking from {len(usable)} views")

    n_done = 0
    for fi in usable:
        try:
            c2w = np.asarray(pose_map[fi], dtype=np.float64)
            c2w4 = np.eye(4); c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
            w2c = np.linalg.inv(c2w4)
            R_t = torch.tensor(w2c[:3, :3], device=dev, dtype=torch.float32)
            tr = torch.tensor(w2c[:3, 3], device=dev, dtype=torch.float32)
            cam_c = torch.tensor(c2w4[:3, 3], device=dev, dtype=torch.float32)

            z = np.load(str(npz_dir / f"frame_{fi}.npz"))
            depth_np = z["depth"].astype(np.float32)           # (Hd,Wd)
            Hd, Wd = depth_np.shape
            K = intrinsics_map.get(fi)
            if K is None and "intrinsics" in z:
                K = np.asarray(z["intrinsics"], dtype=np.float64).reshape(3, 3)
            if K is None:
                continue
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            # project texels → normalised image coords [0,1] (depth-res K)
            Xc = txyz @ R_t.T + tr                              # (P,3) cam space
            zc = Xc[:, 2]
            front = zc > 1e-3
            zc_safe = torch.clamp(zc, min=1e-3)
            u_n = (fx * Xc[:, 0] / zc_safe + cx) / Wd           # [0,1]
            v_n = (fy * Xc[:, 1] / zc_safe + cy) / Hd
            inb = front & (u_n >= 0) & (u_n < 1) & (v_n >= 0) & (v_n < 1)

            # grid for sampling (normalised → [-1,1])
            grid = torch.stack([u_n * 2 - 1, v_n * 2 - 1], dim=-1)[None, :, None, :]  # (1,P,1,2)

            # visibility via the depth map (occlusion oracle)
            depth_t = torch.tensor(depth_np, device=dev)[None, None]     # (1,1,Hd,Wd)
            dsamp = torch.nn.functional.grid_sample(
                depth_t, grid, mode="bilinear", align_corners=False)[0, 0, :, 0]  # (P,)
            vis = inb & (dsamp > 1e-3) & ((zc - dsamp).abs() < depth_tol)

            # colour from the full-res JPG
            with Image.open(_name(fi)) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0  # (Hj,Wj,3)
            rgb_t = torch.tensor(rgb, device=dev).permute(2, 0, 1)[None]       # (1,3,Hj,Wj)
            csamp = torch.nn.functional.grid_sample(
                rgb_t, grid, mode="bilinear", align_corners=False)[0, :, :, 0].T  # (P,3)

            # view-quality weight: front-facing cosine (favour frontal, sharp views)
            vdir = torch.nn.functional.normalize(cam_c[None] - txyz, dim=-1)   # (P,3)
            cosang = (tnrm * vdir).sum(-1).clamp(min=0.0)
            w = (cosang ** cos_power) * vis.float()
            w = w[:, None]
            col_sum += csamp * w
            w_sum += w
            n_done += 1
            if progress_cb and n_done % 100 == 0:
                progress_cb(f"baking {n_done}/{len(usable)}")
        except Exception as e:
            logger.warning(f"[NvBake] view {fi} skipped ({e})")
            continue

    seen = (w_sum[:, 0] > 1e-8)
    if int(seen.sum().item()) == 0:
        logger.error("[NvBake] no texel received any colour — check poses/depth")
        return None
    col = torch.zeros_like(col_sum)
    col[seen] = col_sum[seen] / w_sum[seen]
    logger.info(f"[NvBake] textured {100.0*seen.float().mean().item():.1f}% of covered texels")

    # ── 4. scatter back into the atlas, dilate to fill holes ──
    atlas = torch.zeros((atlas_res * atlas_res, 3), device=dev)
    atlas[sel] = col
    valid = torch.zeros((atlas_res * atlas_res, 1), device=dev)
    valid[sel] = seen.float()[:, None]
    atlas = atlas.reshape(atlas_res, atlas_res, 3).permute(2, 0, 1)[None]   # (1,3,R,R)
    valid = valid.reshape(atlas_res, atlas_res, 1).permute(2, 0, 1)[None]   # (1,1,R,R)
    # iterative nearest dilation: push valid colours into empty texels so the
    # GLB has no black seams at chart borders / unseen patches.
    if progress_cb:
        progress_cb("filling")
    for _ in range(dilate_iters):
        holes = (valid < 0.5)
        if not bool(holes.any()):
            break
        c_blur = torch.nn.functional.avg_pool2d(atlas, 3, 1, 1)
        v_blur = torch.nn.functional.avg_pool2d(valid, 3, 1, 1)
        fill = (v_blur > 1e-6) & holes
        atlas = torch.where(fill, c_blur / v_blur.clamp(min=1e-6), atlas)
        valid = torch.where(fill, torch.ones_like(valid), valid)

    atlas_np = (atlas[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # glTF texture origin is top-left with V↓; our atlas row 0 = uv.y=1 (ndc +1),
    # so flip vertically and keep UVs as-is.
    atlas_np = np.flipud(atlas_np).copy()
    tex_img = Image.fromarray(atlas_np, "RGB")

    # ── 5. export textured GLB ──
    if progress_cb:
        progress_cb("export")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_img, metallicFactor=0.0, roughnessFactor=1.0)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material, image=tex_img)
    out_mesh = trimesh.Trimesh(vertices=V2, faces=faces2, visual=visual, process=False)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(str(out_glb))
    logger.info(f"[NvBake] ✅ wrote {out_glb.name} "
                f"({out_glb.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")
    return out_glb
