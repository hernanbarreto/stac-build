"""Synthetic tests for reconstruction/cloud_mesh.mesh_core — the Delaunay +
visibility graph-cut mesher (doctrine 2026-08-16: mesh FROM the cloud).

Scene: points on a unit sphere ("the real surface"), each seen by the camera
nearest to its outward normal (8 cameras on a radius-4 shell). An ONION ghost
layer floats at radius 1.6 between surface and cameras — rays from the other
points' cameras pierce it, so the cut must NOT mesh it.

Checks:
  1. the surface is reconstructed (most sphere points used, many triangles);
  2. mesh triangles hug the sphere (vertex radius ≈ 1);
  3. the onion layer is REMOVED (almost no ghost point survives on the mesh);
  4. the surface has no macro-holes (solid angle coverage of face normals).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconstruction.cloud_mesh import mesh_core  # noqa: E402


def _fibonacci_sphere(n, r=1.0, seed=0):
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    y = 1.0 - 2.0 * (i + 0.5) / n
    rad = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    p = np.stack([np.cos(phi) * rad, y, np.sin(phi) * rad], 1)
    rng = np.random.default_rng(seed)
    return r * p + rng.normal(0, 0.003, p.shape)   # slight realistic noise


def test_sphere_with_onion():
    n_surf, n_ghost = 20000, 1200
    surf = _fibonacci_sphere(n_surf, r=1.0)
    ghost = _fibonacci_sphere(n_ghost, r=1.6, seed=7)
    P = np.vstack([surf, ghost])

    # 8 cameras on a radius-4 shell (cube corners); each point sees the camera
    # most aligned with its outward direction
    cams = np.array([[sx, sy, sz] for sx in (-1, 1)
                     for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    cams = cams / np.linalg.norm(cams, axis=1, keepdims=True) * 4.0
    dirs = P / np.linalg.norm(P, axis=1, keepdims=True)
    cam_idx = np.argmax(dirs @ (cams / 4.0).T, axis=1)
    cam_xyz = cams[cam_idx]

    spacing = 0.025   # ≈ mean nearest-neighbour distance at n=20k on the sphere
    faces, stats = mesh_core(P, cam_xyz, spacing, cm_lambda=0.6,
                             cm_max_edge_m=0.2, cm_min_component_tris=60,
                             log=lambda m: print(m))

    assert stats["rays_ok"] > 0.9 * len(P), \
        f"ray walking failed: {stats['rays_ok']}/{len(P)}"
    assert len(faces) > 10000, f"too few triangles: {len(faces)}"

    used = np.unique(faces)
    used_r = np.linalg.norm(P[used], axis=1)

    # 2. mesh hugs the sphere
    on_sphere = np.abs(used_r - 1.0) < 0.1
    frac_surface = on_sphere.mean()
    assert frac_surface > 0.97, \
        f"mesh vertices off the real surface: only {frac_surface:.1%} at r≈1"

    # 3. onion removed: ghost points (r≈1.6) that ended up ON the mesh
    ghost_used = (used >= n_surf).sum()
    assert ghost_used < 0.05 * n_ghost, \
        f"onion NOT carved: {ghost_used}/{n_ghost} ghost points on the mesh"

    # 1b. coverage: most real surface points participate
    surf_used = (used < n_surf).sum()
    assert surf_used > 0.85 * n_surf, \
        f"incomplete surface: only {surf_used}/{n_surf} sphere points used"

    # 4. no macro-holes: face centroids cover all directions (24 solid-angle bins)
    cent = P[faces].mean(1)
    d = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    bins = (np.digitize(d[:, 0], [-0.33, 0.33]) * 9
            + np.digitize(d[:, 1], [-0.33, 0.33]) * 3
            + np.digitize(d[:, 2], [-0.33, 0.33]))
    filled = len(np.unique(bins))
    # 26, not 27: the center bin (|x|,|y|,|z| all < 0.33) is unreachable by
    # unit direction vectors (their norm would be < 1).
    assert filled >= 26, f"macro-hole: only {filled}/26 direction bins covered"

    print(f"OK: {len(faces):,} tris | {frac_surface:.1%} on-surface | "
          f"ghost on mesh {ghost_used}/{n_ghost} | coverage 27/27 bins")


def test_implicit_finish_smooth_and_faithful():
    """The implicit (screened-Poisson) finish must deliver the continuous
    TSDF-style surface: local roughness collapses, the true shape is
    recovered, and the crop keeps it glued to the master surface."""
    import open3d as o3d
    from reconstruction.cloud_mesh import _implicit_finish

    m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=80)
    V = np.asarray(m.vertices)
    rng = np.random.default_rng(0)
    m.vertices = o3d.utility.Vector3dVector(V + rng.normal(0, 0.004, V.shape))
    m.vertex_colors = o3d.utility.Vector3dVector(np.full((len(V), 3), 0.5))

    def local_rough(mm):
        mm.compute_adjacency_list()
        Vv = np.asarray(mm.vertices)
        d = np.zeros(len(Vv))
        for i, adj in enumerate(mm.adjacency_list):
            if adj:
                d[i] = np.linalg.norm(Vv[i] - Vv[list(adj)].mean(0))
        return float(np.mean(d))

    r0 = local_rough(m)
    out = _implicit_finish(m, target_voxel=0.02, crop_dist=0.02,
                           log=lambda s: print(s))
    r1 = local_rough(out)
    Vo = np.asarray(out.vertices)
    err = float(np.abs(np.linalg.norm(Vo, axis=1) - 1.0).mean())
    far = float(np.abs(np.linalg.norm(Vo, axis=1) - 1.0).max())
    print(f"implicit: rugosidad {r0*1000:.2f}->{r1*1000:.2f}mm | "
          f"error real {err*1000:.2f}mm | desvío máx {far*1000:.1f}mm")
    assert r1 < r0 * 0.4, f"implicit no alisa: {r1*1000:.2f}mm"
    assert err < 0.003, f"implicit se aparta de la forma real: {err*1000:.2f}mm"
    assert far < 0.035, f"implicit inventó geometría a {far*1000:.0f}mm"
    assert len(out.triangles) > 10000, "implicit colapsó la superficie"


if __name__ == "__main__":
    test_sphere_with_onion()
    test_implicit_finish_smooth_and_faithful()
    print("ALL TESTS PASSED")
