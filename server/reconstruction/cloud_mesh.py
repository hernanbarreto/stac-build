#!/usr/bin/env python3
"""
Mesh FROM the cloud — Delaunay tetrahedra + visibility graph cut.
=================================================================

DOCTRINE (2026-08-16): the VGGT-Ω+DA3 cleaned cloud is the geometry — the
mesh must be built FROM it, never from a parallel path (TSDF re-integrating
depths discarded whole zones; Poisson meshes noise blindly). The mesh's ONLY
enhancement over the cloud is removing the onion/floater layers — and that is
exactly what per-point VISIBILITY does.

ALGORITHM (published, not invented — the RealityScan/RealityCapture lineage):
  Labatut, Pons, Keriven 2007 "Efficient Multi-View Reconstruction of
  Large-Scale Scenes using Interest Points, Delaunay Triangulation and Graph
  Cuts"; Jancosek & Pajdla 2011 (weakly-supported surfaces).
    1. Delaunay-tetrahedralize the cloud points (scipy/Qhull).
    2. Each point knows the CAMERA it was born from (our SAM3 traceability:
       frame_global → camera_poses.txt). The segment camera→point crosses
       tetrahedra: those get "free space" votes (source). The tetrahedron
       just BEHIND the point gets a "solid" vote (sink).
    3. Min s-t cut over the tet adjacency graph (PyMaxflow, face-area
       smoothness) labels every tet free/solid.
    4. The surface = triangles between a free and a solid tet — every vertex
       IS a cloud point. Onion/floating points are pierced by other cameras'
       rays → carved away by the cut, not meshed.
  Post: giant hull-closing triangles dropped by edge length; dust components
  dropped by size. Texture: the existing texrecon bake + meshopt compression.

Wired as `tsdf.mesh_method: "cloud_delaunay"` — same entry points (pipeline
TSDF stage + the UI's mesh button), same deliverable (tsdf/scene/scene.glb),
so the viewer needs nothing new. The cloud itself is NEVER modified.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("CloudMesh")

_MAX_WALK_STEPS = 2000       # per-ray cap (degenerate-geometry safety net)


# ──────────────────────────────────────────────────────────────────────────────
# numba kernels — ray/tetrahedra walking
# ──────────────────────────────────────────────────────────────────────────────

def _kernels():
    """Compile-on-first-use numba kernels (kept in a factory so importing the
    module stays cheap and numba is only required when the mesher runs)."""
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _orient(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz):
        # signed volume of tet (a,b,c,d): >0 if d on the positive side of (a,b,c)
        adx, ady, adz = ax - dx, ay - dy, az - dz
        bdx, bdy, bdz = bx - dx, by - dy, bz - dz
        cdx, cdy, cdz = cx - dx, cy - dy, cz - dz
        return (adx * (bdy * cdz - bdz * cdy)
                - ady * (bdx * cdz - bdz * cdx)
                + adz * (bdx * cdy - bdy * cdx))

    @njit(cache=True, fastmath=True)
    def _point_in_tet(pts, simplices, t, qx, qy, qz):
        a, b, c, d = simplices[t, 0], simplices[t, 1], simplices[t, 2], simplices[t, 3]
        ax, ay, az = pts[a, 0], pts[a, 1], pts[a, 2]
        bx, by, bz = pts[b, 0], pts[b, 1], pts[b, 2]
        cx, cy, cz = pts[c, 0], pts[c, 1], pts[c, 2]
        dx, dy, dz = pts[d, 0], pts[d, 1], pts[d, 2]
        o0 = _orient(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz)
        s = 1.0 if o0 > 0 else -1.0
        if s * _orient(qx, qy, qz, bx, by, bz, cx, cy, cz, dx, dy, dz) < -1e-14:
            return False
        if s * _orient(ax, ay, az, qx, qy, qz, cx, cy, cz, dx, dy, dz) < -1e-14:
            return False
        if s * _orient(ax, ay, az, bx, by, bz, qx, qy, qz, dx, dy, dz) < -1e-14:
            return False
        if s * _orient(ax, ay, az, bx, by, bz, cx, cy, cz, qx, qy, qz) < -1e-14:
            return False
        return True

    @njit(cache=True, fastmath=True)
    def _find_incident_tet(pts, simplices, inc_idx, inc_dat, vi, qx, qy, qz):
        """Among tets incident to vertex vi, the one containing point q (-1 if none
        — q stepped outside the hull)."""
        for j in range(inc_idx[vi], inc_idx[vi + 1]):
            t = inc_dat[j]
            if _point_in_tet(pts, simplices, t, qx, qy, qz):
                return t
        return -1

    @njit(cache=True, fastmath=True)
    def _walk_segment(pts, simplices, neighbors, t0, px, py, pz, tx, ty, tz,
                      free_cnt, do_count):
        """Walk tets along segment p→t starting at tet t0 (which contains p).
        Adds a free-space vote to every crossed tet when do_count. Returns the
        final tet (containing t) or -1 if the segment exits the hull."""
        cur = t0
        prev = -1
        for _ in range(_MAX_WALK_STEPS):
            if do_count:
                free_cnt[cur] += 1.0
            if _point_in_tet(pts, simplices, cur, tx, ty, tz):
                return cur
            nxt = -2
            for k in range(4):
                nb = neighbors[cur, k]
                if nb == prev and prev != -1:
                    continue
                # face opposite corner k = the other three vertices
                i0 = simplices[cur, (k + 1) % 4]
                i1 = simplices[cur, (k + 2) % 4]
                i2 = simplices[cur, (k + 3) % 4]
                f0x, f0y, f0z = pts[i0, 0], pts[i0, 1], pts[i0, 2]
                f1x, f1y, f1z = pts[i1, 0], pts[i1, 1], pts[i1, 2]
                f2x, f2y, f2z = pts[i2, 0], pts[i2, 1], pts[i2, 2]
                kx, ky, kz = (pts[simplices[cur, k], 0], pts[simplices[cur, k], 1],
                              pts[simplices[cur, k], 2])
                # target beyond this face? (opposite sides of the face plane)
                ot = _orient(f0x, f0y, f0z, f1x, f1y, f1z, f2x, f2y, f2z, tx, ty, tz)
                ok = _orient(f0x, f0y, f0z, f1x, f1y, f1z, f2x, f2y, f2z, kx, ky, kz)
                if ot * ok >= 0:
                    continue
                # does segment p→t actually pass through this face's triangle?
                s1 = _orient(px, py, pz, tx, ty, tz, f0x, f0y, f0z, f1x, f1y, f1z)
                s2 = _orient(px, py, pz, tx, ty, tz, f1x, f1y, f1z, f2x, f2y, f2z)
                s3 = _orient(px, py, pz, tx, ty, tz, f2x, f2y, f2z, f0x, f0y, f0z)
                if (s1 >= 0 and s2 >= 0 and s3 >= 0) or (s1 <= 0 and s2 <= 0 and s3 <= 0):
                    nxt = nb
                    break
            if nxt == -2:
                return cur            # numeric corner: stop here (target ~on face)
            if nxt == -1:
                return -1             # exited the convex hull
            prev = cur
            cur = nxt
        return cur

    @njit(cache=True, fastmath=True, parallel=False)
    def _cast_all(pts, simplices, neighbors, inc_idx, inc_dat,
                  ray_v, cam_xyz, eps, behind, free_cnt, sink_cnt):
        """For every (point, its camera): free votes along camera→point,
        one solid vote just behind the point."""
        n_ok = 0
        for r in range(ray_v.shape[0]):
            vi = ray_v[r]
            px, py, pz = pts[vi, 0], pts[vi, 1], pts[vi, 2]
            cx, cy, cz = cam_xyz[r, 0], cam_xyz[r, 1], cam_xyz[r, 2]
            dx, dy, dz = cx - px, cy - py, cz - pz
            norm = (dx * dx + dy * dy + dz * dz) ** 0.5
            if norm < 1e-9:
                continue
            dx, dy, dz = dx / norm, dy / norm, dz / norm
            # start tet: the incident tet the ray ENTERS. The tets around a
            # vertex can be much smaller than eps, so shrink the probe offset
            # until one contains it (fixes 47% of rays failing to start).
            t0 = -1
            sx, sy, sz = px, py, pz
            e = eps
            for _try in range(6):
                sx, sy, sz = px + dx * e, py + dy * e, pz + dz * e
                t0 = _find_incident_tet(pts, simplices, inc_idx, inc_dat,
                                        vi, sx, sy, sz)
                if t0 >= 0:
                    break
                e *= 0.25
            if t0 < 0:
                continue
            # FREE votes: walk start→camera
            _walk_segment(pts, simplices, neighbors, t0,
                          sx, sy, sz, cx, cy, cz, free_cnt, True)
            n_ok += 1
            # SOLID vote: the tet IMMEDIATELY behind the point — i.e., an
            # INCIDENT tet of vi on the far side from the camera. Going deeper
            # (walking 1.5×spacing in) landed in the giant interior-spanning
            # tets of the Delaunay, whose huge boundary faces are NOT the fine
            # surface (diagnosed: correct cut, 0 usable triangles). Shrink the
            # probe toward the vertex until an incident tet contains it.
            tb = -1
            e = behind
            for _try in range(6):
                bx, by, bz = px - dx * e, py - dy * e, pz - dz * e
                tb = _find_incident_tet(pts, simplices, inc_idx, inc_dat,
                                        vi, bx, by, bz)
                if tb >= 0:
                    break
                e *= 0.4
            if tb >= 0:
                sink_cnt[tb] += 1.0
        return n_ok

    return _cast_all


# ──────────────────────────────────────────────────────────────────────────────
# inputs: cloud + per-point camera (our SAM3 traceability)
# ──────────────────────────────────────────────────────────────────────────────

def _load_cloud_with_cameras(output_dir: Path):
    """cleaned_cloud.ply xyz+rgb + per-point ORIGIN camera center.
    Traceability: the PLY carries frame_global per point (injected by
    CloudCompy); camera_poses.txt + camera_frames.txt give that frame's c2w."""
    from segmentation.pipeline import _load_ply_origins
    import open3d as o3d

    cc = output_dir / "cleaned_cloud.ply"
    if not cc.exists():
        raise RuntimeError("cloud_mesh: cleaned_cloud.ply not found — run the "
                           "reconstruction first")
    pc = o3d.io.read_point_cloud(str(cc))
    xyz = np.asarray(pc.points, np.float64)
    rgb = (np.asarray(pc.colors) if pc.has_colors()
           else np.full((len(xyz), 3), 0.7))
    origins = _load_ply_origins(cc)
    if origins is None:
        raise RuntimeError("cloud_mesh: cleaned_cloud.ply has no per-point "
                           "frame traceability (frame_global) — the visibility "
                           "mesher needs it (regenerate the cloud)")
    _oxyz, fg, _pr, _pc = origins
    fg = np.asarray(fg).astype(np.int64)
    if len(fg) != len(xyz):
        raise RuntimeError(f"cloud_mesh: origins length {len(fg)} != cloud "
                           f"points {len(xyz)} — stale traceability")

    from reconstruction.scale_align import _read_poses
    lines, nums, _ = _read_poses(output_dir)
    cam_by_frame: Dict[int, np.ndarray] = {}
    for n, ln in zip(nums, lines):
        v = ln.split()
        if len(v) == 16:
            cam_by_frame[int(n)] = np.array(
                [float(x) for x in v], np.float64).reshape(4, 4)[:3, 3]
    return xyz, rgb, fg, cam_by_frame


def _subsample_with_origins(xyz, rgb, fg, cam_ok_mask, max_points: int,
                            target_voxel: float = 0.012):
    """Voxel-subsample: one vertex per voxel at the MEAN of its cloud points.
    Averaging cancels the per-point noise (~√N) — a surface through single raw
    points exposes every millimetre of error as a spike ("muy triangulada"),
    while the mean is exactly how the dense cloud LOOKS smooth to the eye.
    The representative's ORIGIN (frame) and index are kept for traceability/
    visibility; only its position/colour are the voxel average. Voxel grows
    until the count fits max_points.
    Returns (keep_indices, mean_xyz, mean_rgb, voxel)."""
    keep_pool = np.flatnonzero(cam_ok_mask)
    # START at the TARGET resolution (12 mm default — the TSDF-mesh resolution
    # the user validated visually). The old span/2000 start let an 11M-point
    # cloud (test2) collapse to 46 mm cells → giant facets + lost thin
    # structures ("macro mejor, micro mucho peor"). max_points is only a
    # safety net now, not the resolution driver.
    span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    voxel = max(float(target_voxel), 0.004)
    for _ in range(24):
        q = np.floor(xyz[keep_pool] / voxel).astype(np.int64)
        key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        if len(first) <= max_points or voxel > span:
            order = np.argsort(first)               # stable representative order
            rank = np.empty(len(first), np.int64)
            rank[order] = np.arange(len(first))
            cell = rank[inv]                        # pool-point → output row
            n = len(first)
            cnt = np.bincount(cell, minlength=n).astype(np.float64)
            mean_xyz = np.stack(
                [np.bincount(cell, weights=xyz[keep_pool][:, k], minlength=n)
                 for k in range(3)], 1) / cnt[:, None]
            mean_rgb = np.stack(
                [np.bincount(cell, weights=rgb[keep_pool][:, k], minlength=n)
                 for k in range(3)], 1) / cnt[:, None]
            return keep_pool[np.sort(first)], mean_xyz, mean_rgb, voxel
        voxel *= 1.3
    raise RuntimeError("cloud_mesh: subsample failed to converge")


def _implicit_finish(mesh, target_voxel: float, crop_dist: float, log=logger.info):
    """CONTINUOUS surface extraction over the visibility-validated points —
    the TSDF/PGSR finish with OUR coverage. The Delaunay cut already decided
    WHERE surface exists (complete, onion-free); its vertices + CUT-ORIENTED
    normals feed a screened-Poisson solve (implicit interpolation = the
    smoothness vertex polishing can never reach), and the result is CROPPED
    to within `crop_dist` of the Delaunay master surface, so Poisson can not
    invent geometry (the blindness that had it banned). Vertex colours are
    carried over by nearest master vertex."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    mesh.compute_vertex_normals()
    P = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    C = (np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    depth = int(np.clip(np.ceil(np.log2(
        float(np.linalg.norm(P.max(0) - P.min(0))) / max(target_voxel, 1e-4))),
        8, 12))
    # pymeshlab's screened Poisson, NOT Open3D's: o3d's (TBB, uncappable by
    # OMP_NUM_THREADS) thrashed >4 min on a 21k-point sphere on this 252-core
    # box; pymeshlab solved the same solve in seconds.
    import pymeshlab
    import tempfile
    with tempfile.TemporaryDirectory() as _td:
        _pin = _td + "/pts.ply"
        _pout = _td + "/poisson.ply"
        o3d.io.write_point_cloud(_pin, pcd, write_ascii=False)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(_pin)
        ms.generate_surface_reconstruction_screened_poisson(
            depth=depth, preclean=True)
        ms.save_current_mesh(_pout)
        sm = o3d.io.read_triangle_mesh(_pout)
    log(f"[cloud-mesh] implicit finish: screened Poisson depth={depth} → "
        f"{len(sm.vertices):,} verts (pre-crop)")
    # crop to the master surface: kill Poisson's blind closures/extrapolations
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(P.astype(np.float32)),
                     o3d.core.Tensor(np.asarray(mesh.triangles).astype(np.uint32)))
    SV = np.asarray(sm.vertices, np.float32)
    d = sc.compute_distance(o3d.core.Tensor(SV)).numpy()
    far = d > float(crop_dist)
    sm.remove_vertices_by_mask(far)
    log(f"[cloud-mesh] implicit finish: cropped {int(far.sum()):,} vertices "
        f">{crop_dist * 1000:.0f} mm from the master surface — "
        f"{len(sm.vertices):,} verts / {len(sm.triangles):,} tris kept")
    if len(sm.triangles) < 1000:
        raise RuntimeError("implicit finish collapsed — keeping the master mesh")
    if C is not None and len(sm.vertices):
        _t = cKDTree(P)
        _, ni = _t.query(np.asarray(sm.vertices), k=1)
        sm.vertex_colors = o3d.utility.Vector3dVector(C[ni])
    sm.remove_unreferenced_vertices()
    sm.compute_vertex_normals()
    return sm


def _taubin_target(mesh, iters: int, work_dir: Path) -> np.ndarray:
    """Smoothed vertex positions for the polish blend. pymeshlab's Taubin
    measured clearly better than Open3D's on the synthetic noisy sphere
    (roughness 7.0→1.07 mm vs →2.73 mm at ×30, same true-error ≈1.1 mm), so
    it is the primary; Open3D is the fallback. Vertex count must not change."""
    try:
        import open3d as o3d
        import pymeshlab
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp_in = work_dir / "_taubin_in.ply"
        tmp_out = work_dir / "_taubin_out.ply"
        try:
            o3d.io.write_triangle_mesh(str(tmp_in), mesh, write_ascii=False)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(tmp_in))
            ms.apply_coord_taubin_smoothing(stepsmoothnum=int(iters))
            ms.save_current_mesh(str(tmp_out))
            out = o3d.io.read_triangle_mesh(str(tmp_out))
            V1 = np.asarray(out.vertices)
            if len(V1) != len(mesh.vertices):
                raise RuntimeError("vertex count changed")
            logger.info(f"[cloud-mesh] polish target: pymeshlab taubin ×{iters}")
            return V1
        finally:
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
    except Exception as _e:  # noqa: BLE001 — o3d fallback keeps the polish alive
        logger.info(f"[cloud-mesh] polish target: o3d taubin ×{iters} "
                    f"(pymeshlab unavailable: {_e})")
        return np.asarray(mesh.filter_smooth_taubin(
            number_of_iterations=int(iters)).vertices)


# ──────────────────────────────────────────────────────────────────────────────
# core: points + per-point camera → surface triangles (pure, testable)
# ──────────────────────────────────────────────────────────────────────────────

def mesh_core(P: np.ndarray, cam_xyz: np.ndarray, spacing: float,
              cm_lambda: float = 0.6, cm_max_edge_m: float = 0.0,
              cm_min_component_tris: int = 60, cm_sink_w: float = 3.0,
              log=logger.info) -> Tuple[np.ndarray, dict]:
    """Delaunay + visibility graph cut. P (N,3) points, cam_xyz (N,3) each
    point's origin-camera center. Returns (faces (M,3) int, stats dict)."""
    t0 = time.time()
    from scipy.spatial import Delaunay
    P = np.ascontiguousarray(P, np.float64)
    dt = Delaunay(P)
    simplices = np.ascontiguousarray(dt.simplices.astype(np.int32))
    neighbors = np.ascontiguousarray(dt.neighbors.astype(np.int32))
    T = len(simplices)
    log(f"[cloud-mesh] Delaunay: {T:,} tetrahedra ({time.time() - t0:.0f}s)")

    # vertex → incident tets (CSR)
    flat = simplices.reshape(-1)
    order = np.argsort(flat, kind="stable")
    inc_dat = np.ascontiguousarray((order // 4).astype(np.int32))
    counts = np.bincount(flat, minlength=len(P))
    inc_idx = np.zeros(len(P) + 1, np.int64)
    np.cumsum(counts, out=inc_idx[1:])

    # visibility votes
    cast_all = _kernels()
    ray_v = np.arange(len(P), dtype=np.int64)
    free_cnt = np.zeros(T, np.float32)
    sink_cnt = np.zeros(T, np.float32)
    eps = max(spacing * 0.25, 1e-4)
    behind = max(spacing * 1.5, 0.01)
    n_ok = cast_all(P, simplices, neighbors, inc_idx, inc_dat,
                    ray_v, np.ascontiguousarray(cam_xyz, np.float64),
                    eps, behind, free_cnt, sink_cnt)
    log(f"[cloud-mesh] visibility: {n_ok:,}/{len(P):,} rays walked, free votes "
        f"{free_cnt.sum():,.0f}, solid votes {sink_cnt.sum():,.0f} "
        f"({time.time() - t0:.0f}s)")
    if n_ok < len(P) * 0.3:
        raise RuntimeError(f"cloud_mesh: only {n_ok}/{len(P)} visibility rays "
                           f"resolved — geometry/poses inconsistent")

    # min-cut
    import maxflow
    g = maxflow.Graph[float](T, T * 2)
    g.add_nodes(T)
    for t in range(T):
        if free_cnt[t] > 0 or sink_cnt[t] > 0:
            # cm_sink_w: one solid vote must outweigh the ~2 boundary faces
            # the cut needs around it (λ·area each) — with 1.0 the cut just
            # paid every sink and declared EVERYTHING free (0 triangles).
            g.add_tedge(t, float(free_cnt[t]), float(sink_cnt[t]) * cm_sink_w)
    tA, kA = np.nonzero(neighbors > np.arange(T)[:, None])
    nbA = neighbors[tA, kA]
    idx = np.stack([simplices[tA, (kA + 1) % 4],
                    simplices[tA, (kA + 2) % 4],
                    simplices[tA, (kA + 3) % 4]], 1)
    e1 = P[idx[:, 1]] - P[idx[:, 0]]
    e2 = P[idx[:, 2]] - P[idx[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    med_a = max(float(np.median(areas)), 1e-12)
    # weight floor 0.8: with a low floor the cut SNAKES through near-zero-area
    # sliver faces almost for free instead of paying the real surface (test:
    # flow 15.6k < true-surface 24k → 509 tris). Every face costs ≥0.8λ.
    w = (cm_lambda * np.clip(areas / med_a, 0.8, 20.0)).astype(np.float64)
    for i in range(len(tA)):
        g.add_edge(int(tA[i]), int(nbA[i]), w[i], w[i])
    flow = g.maxflow()
    seg = np.array([g.get_segment(t) for t in range(T)], np.uint8)  # 0=src(free)
    free = seg == 0
    log(f"[cloud-mesh] graph cut: flow {flow:,.0f} — {int(free.sum()):,} free / "
        f"{int((~free).sum()):,} solid tets ({time.time() - t0:.0f}s)")

    # surface faces between free and solid, oriented toward free space
    boundary = free[tA] != free[nbA]
    faces = idx[boundary]
    solid_is_tA = ~free[tA[boundary]]
    corner = simplices[tA[boundary], kA[boundary]]
    a, b, c = P[faces[:, 0]], P[faces[:, 1]], P[faces[:, 2]]
    n = np.cross(b - a, c - a)
    to_corner = P[corner] - a
    dot = np.einsum("ij,ij->i", n, to_corner)
    flip = np.where(solid_is_tA, dot > 0, dot < 0)
    faces[flip] = faces[flip][:, ::-1]

    # long-edge filter (hull-closing monsters)
    max_edge = float(cm_max_edge_m) if cm_max_edge_m > 0 else max(8.0 * spacing, 0.08)
    el = np.stack([np.linalg.norm(P[faces[:, 0]] - P[faces[:, 1]], axis=1),
                   np.linalg.norm(P[faces[:, 1]] - P[faces[:, 2]], axis=1),
                   np.linalg.norm(P[faces[:, 2]] - P[faces[:, 0]], axis=1)], 1)
    keep_f = el.max(1) <= max_edge
    n_long = int((~keep_f).sum())
    faces = faces[keep_f]

    # dust components (vertex connectivity)
    if len(faces):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        ii = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        jj = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        adj = coo_matrix((np.ones(len(ii), np.int8), (ii, jj)),
                         shape=(len(P), len(P)))
        n_comp, lab = connected_components(adj, directed=False)
        tri_lab = lab[faces[:, 0]]
        sizes = np.bincount(tri_lab, minlength=n_comp)
        faces = faces[sizes[tri_lab] >= int(cm_min_component_tris)]
    log(f"[cloud-mesh] surface: {len(faces):,} tris (dropped {n_long:,} "
        f"long-edge >{max_edge:.2f}m + dust) ({time.time() - t0:.0f}s)")
    stats = {"tets": T, "rays_ok": int(n_ok), "flow": float(flow),
             "free_tets": int(free.sum()), "max_edge_m": max_edge,
             "n_long_dropped": n_long}
    return faces.astype(np.int32), stats


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def export_cloud_mesh_scene(
    output_dir: Path,
    frames_dir: Path,
    session_dir: Optional[Path] = None,
    texture: bool = True,
    texture_max_views: int = 400,
    cm_max_points: int = 6_000_000,      # SAFETY net only (Qhull time/RAM); resolution is
                                         # driven by cm_target_voxel, not by this cap
    cm_target_voxel: float = 0.012,      # mesh vertex spacing (m) — the TSDF resolution the
                                         # user validated; 46mm cells (old 1.2M cap on an 11M
                                         # cloud) caused the giant-facet look
    cm_lambda: float = 0.6,              # smoothness vs visibility votes
    cm_max_edge_m: float = 0.0,          # 0 = auto: 8× the subsample voxel
    cm_min_component_tris: int = 60,     # dust components below this are dropped
    guide_glb: Optional[str] = None,     # BEST-OF-BOTH: path to the untextured TSDF guide mesh
                                         # (produced by the dispatcher). Where our surface lies
                                         # within cm_guide_max_dist of it, vertices are drawn
                                         # onto the TSDF surface — its smooth finish where it
                                         # exists, our coverage everywhere else. No topology
                                         # change, no seams (quadratic falloff to the edge).
    cm_guide_max_dist: float = 0.03,     # attraction radius (m)
    fill_holes: bool = True,             # small-hole fill (same machinery/guards as the TSDF
    fill_hole_size: float = 0.25,        # path) — closes gaps the long-edge filter opened
    cm_finish: str = "implicit",         # SURFACE FINISH: "implicit" = screened-Poisson over
                                         # the cut-validated points with cut-oriented normals,
                                         # cropped ≤cm_finish_crop to the master surface — the
                                         # CONTINUOUS TSDF/PGSR look with our coverage (vertex
                                         # polishing can never reach it). "vertex" = the taubin
                                         # polish path. Falls back to "vertex" on failure.
    cm_finish_crop: float = 0.02,        # implicit finish crop distance (m) to the master
    cm_polish: str = "taubin",           # POLISH on the finished mesh: strong Taubin low-pass
                                         # blended per-vertex by cloud CONFIDENCE. MEASURED on
                                         # a noisy synthetic sphere: roughness 7.0→1.1mm AND
                                         # true-surface error 3.2→1.1mm (recovers the real
                                         # shape, no shrink). TwoStep/bilateral was measured
                                         # WORSE (barely smooths; diverges when pushed) and
                                         # was removed. "off" disables.
    cm_polish_steps: int = 30,           # taubin iterations of the smooth target
    progress_cb=None,
    **_ignored,                          # tsdf-only kwargs arrive here harmlessly
) -> Optional[str]:
    """Delaunay+visibility mesh of the cleaned cloud → textured scene.glb.
    Returns the glb path (str) or None on failure. The cloud is never touched."""
    import open3d as o3d
    t0 = time.time()
    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)

    def _prog(phase):
        if progress_cb:
            progress_cb(phase, time.time() - t0, None)

    # ── 1. cloud + per-point camera centers ──
    _prog("loading")
    xyz, rgb, fg, cam_by_frame = _load_cloud_with_cameras(output_dir)
    frames_avail = np.array(sorted(cam_by_frame.keys()), np.int64)
    cam_ok = np.isin(fg, frames_avail)
    logger.info(f"[cloud-mesh] {len(xyz):,} pts, {cam_ok.mean() * 100:.1f}% with "
                f"a posed origin camera ({len(frames_avail)} cameras)")
    if cam_ok.mean() < 0.5:
        raise RuntimeError("cloud_mesh: less than half the cloud has a posed "
                           "origin camera — poses/traceability out of sync")

    keep, P, C, voxel = _subsample_with_origins(xyz, rgb, fg, cam_ok,
                                                int(cm_max_points),
                                                float(cm_target_voxel))
    P = np.ascontiguousarray(P)
    F = fg[keep]
    spacing = voxel if voxel > 0 else float(
        np.linalg.norm(P.max(0) - P.min(0))) / max(len(P), 1) ** (1 / 3)
    logger.info(f"[cloud-mesh] meshing {len(P):,} points "
                f"(subsample voxel {voxel * 1000:.1f} mm)")

    # ── 2-5. Delaunay + visibility + cut + surface (mesh_core, testable) ──
    _prog("meshing")
    cam_xyz = np.zeros((len(P), 3), np.float64)
    for fnum in frames_avail:
        m = F == fnum
        if m.any():
            cam_xyz[m] = cam_by_frame[int(fnum)]
    faces, _stats = mesh_core(P, cam_xyz, spacing,
                              cm_lambda=cm_lambda, cm_max_edge_m=cm_max_edge_m,
                              cm_min_component_tris=cm_min_component_tris)
    if len(faces) < 100:
        raise RuntimeError("cloud_mesh: surface came out empty — inspect "
                           "visibility votes / poses")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(P)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))

    # CONFIDENCE-ADAPTIVE Taubin smoothing. The surface interpolates every raw
    # cloud point exactly → per-point noise becomes spikes, and (user
    # observation, test2/test3) the spikes live where views/confidence are LOW
    # while well-observed zones are already smooth. So: blend each vertex
    # between its original and taubin-smoothed position by the cloud's own
    # per-point confidence — low confidence → full smoothing (spikes ironed),
    # high confidence → mostly original (detail preserved). Topology and
    # completeness untouched; positions move ≤ the local noise amplitude.
    _implicit_done = False
    if str(cm_finish).lower() == "implicit":
        _prog("smoothing")
        try:
            mesh = _implicit_finish(mesh, float(cm_target_voxel),
                                    float(cm_finish_crop))
            _implicit_done = True
        except Exception as _e:  # noqa: BLE001 — master mesh + vertex polish still ship
            logger.warning(f"[cloud-mesh] implicit finish failed ({_e}) — "
                           f"vertex-polish fallback")
    if not _implicit_done and \
            str(cm_polish).lower() == "taubin" and int(cm_polish_steps) > 0:
        _prog("smoothing")
        # POLISH (measured on a noisy synthetic sphere, taubin ×30: local
        # roughness 7.0→1.1 mm AND true-surface error 3.2→1.1 mm — it recovers
        # the real shape; TwoStep/bilateral measured worse and was removed).
        # The adaptive blend dosifies it: low-confidence vertices take the full
        # smooth target, well-observed ones keep ~70% of their original detail.
        V0 = np.asarray(mesh.vertices).copy()
        V1 = _taubin_target(mesh, int(cm_polish_steps), output_dir / "tsdf")
        w = np.full(len(V0), 0.7)            # fallback: uniform, mostly smooth
        try:
            from segmentation.tsdf_export import _load_ply_confidence
            _conf = _load_ply_confidence(output_dir / "cleaned_cloud.ply")
            if _conf is not None and len(_conf) == len(xyz):
                c = np.asarray(_conf)[keep].astype(np.float64)
                lo, hi = np.quantile(c, (0.2, 0.8))
                t_ = np.clip((c - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
                w = 1.0 - 0.7 * t_           # conf low→1.0 (smooth), high→0.3
                logger.info(f"[cloud-mesh] polish: taubin ×{int(cm_polish_steps)} "
                            f"CONFIDENCE-ADAPTIVE (weight {w.min():.2f}–"
                            f"{w.max():.2f})")
            else:
                logger.info(f"[cloud-mesh] polish: taubin ×{int(cm_polish_steps)} "
                            f"uniform (no per-point confidence found)")
        except Exception as _e:  # noqa: BLE001
            logger.info(f"[cloud-mesh] polish: taubin uniform (confidence load: {_e})")
        mesh.vertices = o3d.utility.Vector3dVector(
            V0 * (1.0 - w[:, None]) + V1 * w[:, None])

    # ── REGION FUSION with the guide mesh — THE CLOUD AS REFEREE ──
    # (user design 2026-08-17): per ~25cm cell, the mesh that best represents
    # the cloud (coverage × fidelity, roughness breaks ties) WINS the cell —
    # TSDF/PGSR keeps the complex objects it nails, Delaunay keeps the planes
    # and everything the other missed. Objective, per-cell, auditable
    # (fusion_report.json). Replaces the old vertex attraction.
    guided_pct = 0.0
    if guide_glb and Path(guide_glb).exists():
        try:
            import trimesh as _tm
            from reconstruction.mesh_fusion import fuse_meshes
            _gm = _tm.load(str(guide_glb), force="mesh")
            _Vb = np.asarray(mesh.vertices)
            _Fb = np.asarray(mesh.triangles)
            _Cb = (np.asarray(mesh.vertex_colors)
                   if mesh.has_vertex_colors() else None)
            # referee cloud subsampled for scoring speed (≤2M pts)
            _ref = xyz if len(xyz) <= 2_000_000 else \
                xyz[np.random.default_rng(0).choice(len(xyz), 2_000_000,
                                                    replace=False)]
            Vf, Ff, from_a, _fstats = fuse_meshes(
                np.asarray(_gm.vertices, np.float64),
                np.asarray(_gm.faces, np.int64),
                _Vb, _Fb, np.asarray(_ref, np.float64),
                report_path=output_dir / "tsdf" / "fusion_report.json",
                log=logger.info)
            _fused = o3d.geometry.TriangleMesh()
            _fused.vertices = o3d.utility.Vector3dVector(Vf)
            _fused.triangles = o3d.utility.Vector3iVector(Ff.astype(np.int32))
            # vertex colours: A side has none (untextured guide) → grey;
            # B side keeps the cloud colours. texrecon repaints everything.
            _cols = np.full((len(Vf), 3), 0.62)
            if _Cb is not None:
                _cols[len(_gm.vertices):] = _Cb
            _fused.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(_cols, 0, 1))
            _fused.remove_unreferenced_vertices()
            mesh = _fused
            guided_pct = 100.0 * _fstats["cells_a"] / max(_fstats["cells"], 1)
        except Exception as _e:  # noqa: BLE001 — the Delaunay mesh still ships
            logger.warning(f"[cloud-mesh] fusion skipped ({_e}) — Delaunay-only")

    # ── small-hole fill (same sanitize + runaway guard as the TSDF path) ──
    if fill_holes and fill_hole_size > 0:
        try:
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.remove_unreferenced_vertices()
            _tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            _filled = _tmesh.fill_holes(hole_size=float(fill_hole_size)).to_legacy()
            _n_new = len(_filled.triangles) - len(mesh.triangles)
            if 0 <= _n_new <= 0.5 * max(1, len(mesh.triangles)):
                mesh = _filled
                if _n_new > 0:
                    logger.info(f"[cloud-mesh] hole-fill: +{_n_new:,} tris "
                                f"(≤{fill_hole_size:g} m)")
            else:
                logger.warning(f"[cloud-mesh] hole-fill rejected "
                               f"(Δ{_n_new:,} tris looks runaway) — skipped")
        except Exception as _e:  # noqa: BLE001
            logger.warning(f"[cloud-mesh] hole-fill skipped ({_e})")
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    n_verts, n_tris = len(mesh.vertices), len(mesh.triangles)

    # ── 6. deliverable: texture + compress + meta (same slot as the TSDF) ──
    scene_dir = output_dir / "tsdf" / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    glb_path = scene_dir / "scene.glb"
    o3d.io.write_triangle_mesh(str(scene_dir / "_scene_geom.ply"), mesh,
                               write_ascii=False)
    textured = False
    if texture:
        _prog("texturing")
        try:
            from segmentation.session_io import _load_camera_source
            cam = _load_camera_source(session_dir or output_dir.parent, output_dir)
            from reconstruction.texture_bake import bake_texture
            res = bake_texture(
                mesh_path=scene_dir / "_scene_geom.ply",
                frames_dir=frames_dir,
                pose_map=cam.pose_map,
                intrinsics_map=cam.intrinsics_map,
                out_glb=glb_path,
                **({"max_views": int(texture_max_views)}
                   if texture_max_views and int(texture_max_views) > 0 else {}),
            )
            textured = res is not None
        except Exception as e:  # noqa: BLE001 — vertex-colour glb still ships
            logger.warning(f"[cloud-mesh] texture bake failed ({e}) — "
                           f"delivering vertex colours")
    if not textured:
        o3d.io.write_triangle_mesh(str(glb_path), mesh)
    (scene_dir / "_scene_geom.ply").unlink(missing_ok=True)
    try:
        import shutil
        shutil.copy2(glb_path, glb_path.with_suffix(".glb.orig"))
        from segmentation.tsdf_export import _compress_scene_glb
        _compress_scene_glb(glb_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[cloud-mesh] glb compression skipped ({e})")

    (scene_dir / "scene_meta.json").write_text(json.dumps({
        "method": "cloud_delaunay_visibility",
        "textured": textured,
        "n_vertices": n_verts, "n_triangles": n_tris,
        "source_cloud_points": int(len(xyz)),
        "meshed_points": int(len(P)),
        "subsample_voxel_m": round(voxel, 4),
        "max_edge_m": round(float(_stats["max_edge_m"]), 3),
        "lambda": cm_lambda,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))
    logger.info(f"[cloud-mesh] ✅ {n_verts:,} verts / {n_tris:,} tris "
                f"textured={textured} in {time.time() - t0:.0f}s → {glb_path}")
    return str(glb_path)
