"""
Stage 3 + 5 tests: GlobFit-style plane regularization (snap only within
tolerance, findings preserved beyond it), clean-edge snapping, and the hybrid
fit_scene orchestration over a synthetic on-disk session.
"""
import json

import numpy as np
import pytest

from reconstruction.surface_fit.plane import fit_plane
from reconstruction.surface_fit.regularize import regularize_planes, snap_edges
from reconstruction.surface_fit.runner import fit_segment

SIGMA = 0.002
UP = np.array([0.0, 0.0, 1.0])


def plane_cloud(normal, d0, size=(4.0, 3.0), n=30_000, seed=0):
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    ref = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1, 0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    su = rng.uniform(0, size[0], n)
    sv = rng.uniform(0, size[1], n)
    pts = -d0 * normal + su[:, None] * u + sv[:, None] * v
    return pts + rng.normal(0, SIGMA, n)[:, None] * normal


def tilted(base, deg, axis=(0, 1, 0)):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.deg2rad(deg) * np.asarray(axis, float)).apply(base)


class TestRegularizePlanes:
    def test_small_deviations_snap_orthogonal(self):
        """Floor + wall 0.4° off vertical + wall 0.5° off orthogonal, tol 1°
        → exact orthogonal triplet."""
        floor = fit_plane(plane_cloud(tilted([0, 0, 1], 0.3), 0.0, seed=1))
        wall1 = fit_plane(plane_cloud(tilted([1, 0, 0], 0.4), -2.0, seed=2))
        wall2 = fit_plane(plane_cloud(tilted([0, 1, 0], 0.5, axis=(1, 0, 0)), -3.0, seed=3))
        out, rep = regularize_planes([floor, wall1, wall2],
                                     weights=[3.0, 1.0, 1.0],
                                     angle_tol_deg=1.0)
        n0, n1, n2 = (o.normal for o in out)
        assert abs(abs(n0 @ UP) - 1.0) < 1e-9          # floor exactly horizontal
        assert abs(n1 @ UP) < 1e-9                     # walls exactly vertical
        assert abs(n2 @ UP) < 1e-9
        assert abs(n1 @ n2) < 1e-6                     # walls exactly orthogonal
        assert len(rep.snapped) >= 3

    def test_large_deviation_not_forced(self):
        """A wall 3° off vertical with tol 1° must KEEP its real orientation —
        it's a construction finding, not fit noise."""
        wall = fit_plane(plane_cloud(tilted([1, 0, 0], 3.0), -2.0, seed=4))
        out, _ = regularize_planes([wall], angle_tol_deg=1.0)
        ang = np.rad2deg(np.arccos(np.clip(abs(out[0].normal @ UP), 0, 1)))
        assert abs(90.0 - ang - 3.0) < 0.2             # still ~3° off plumb

    def test_coplanar_merge(self):
        """Two wall pieces 4 mm apart (tol 10 mm) → one common plane."""
        w1 = fit_plane(plane_cloud([1, 0, 0], -2.000, seed=5))
        w2 = fit_plane(plane_cloud([1, 0, 0], -2.004, seed=6))
        out, rep = regularize_planes([w1, w2], coplanar_tol_mm=10.0)
        d0 = out[0].d if out[0].normal[0] > 0 else -out[0].d
        d1 = out[1].d if out[1].normal[0] > 0 else -out[1].d
        assert abs(d0 - d1) < 1e-9
        assert rep.coplanar_groups

    def test_separate_walls_not_merged(self):
        """Parallel walls 3 m apart must obviously stay apart."""
        w1 = fit_plane(plane_cloud([1, 0, 0], -2.0, seed=7))
        w2 = fit_plane(plane_cloud([1, 0, 0], -5.0, seed=8))
        out, rep = regularize_planes([w1, w2], coplanar_tol_mm=10.0)
        assert not rep.coplanar_groups


class TestCleanEdges:
    def test_wall_floor_edge_closes_gap(self):
        """Floor reaching x∈[0,3.9], wall at x=4: after edge snapping the
        floor's boundary must touch the wall plane exactly (explicit coords —
        the generic cloud helper's in-plane basis is orientation-ambiguous)."""
        rng = np.random.default_rng(9)
        n = 40_000
        floor_pts = np.column_stack([rng.uniform(0, 3.9, n),
                                     rng.uniform(0, 3.0, n),
                                     rng.normal(0, SIGMA, n)])
        wall_pts = np.column_stack([4.0 + rng.normal(0, SIGMA, n),
                                    rng.uniform(0, 3.0, n),
                                    rng.uniform(0, 2.5, n)])
        f_floor = fit_segment(floor_pts, consolidate_method="none",
                              label="floor", models=["plane"])
        f_wall = fit_segment(wall_pts, consolidate_method="none",
                             label="wall", models=["plane"])
        gap_before = float(np.min(np.abs(f_wall.model.signed_distance(
            f_floor.mesh_vertices))))
        assert gap_before > 0.02                       # there IS a gap to close
        snap_edges([f_floor, f_wall], snap_dist_m=0.15)
        gap_after = float(np.min(np.abs(f_wall.model.signed_distance(
            f_floor.mesh_vertices))))
        assert gap_after < 1e-9                        # boundary ON the wall plane
        # interior did not deform: mesh still on the floor plane
        assert float(np.max(np.abs(f_floor.model.signed_distance(
            f_floor.mesh_vertices)))) < 1e-9


class TestFitScene:
    @pytest.fixture()
    def session(self, tmp_path):
        """Synthetic on-disk session: floor + wall (architectural) + a blob
        ('pipe valve') that must be skipped to the TSDF path."""
        rng = np.random.default_rng(30)
        floor = plane_cloud([0, 0, 1], 0.0, size=(4, 3), n=50_000, seed=11)
        wall = plane_cloud(tilted([1, 0, 0], 0.4), -4.0, size=(3, 2.5),
                           n=50_000, seed=12)
        blob = rng.normal(0, 0.3, (8_000, 3)) + np.array([2.0, 1.5, 1.0])
        cloud = np.vstack([floor, wall, blob])
        out = tmp_path / "output"
        out.mkdir()
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
        o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), pcd)
        n_f, n_w, n_b = len(floor), len(wall), len(blob)
        seg = {"instances": [
            {"id": 1, "instance_id": 1, "label": "floor",
             "globalIndices": list(range(n_f))},
            {"id": 2, "instance_id": 2, "label": "wall",
             "globalIndices": list(range(n_f, n_f + n_w))},
            {"id": 3, "instance_id": 3, "label": "valve",
             "globalIndices": list(range(n_f + n_w, n_f + n_w + n_b))},
        ]}
        (out / "segmentation_result.json").write_text(json.dumps(seg))
        return tmp_path

    def test_hybrid_scene(self, session):
        from reconstruction.surface_fit.scene import fit_scene
        rep = fit_scene(session, overrides={"consolidate_method": "none",
                                            "models": ["plane"]})
        assert len(rep["fitted"]) == 2
        assert len(rep["skipped_to_tsdf"]) == 1
        assert rep["skipped_to_tsdf"][0]["label"] == "valve"
        # regularization snapped the 0.4°-off wall
        wall_r = next(r for r in rep["fitted"] if r["label"] == "wall")
        assert wall_r["regularized"]
        # artifacts on disk
        out = session / "output" / "surface_fit"
        for r in rep["fitted"]:
            d = out / f"{r['label']}_{r['instance_id']}"
            for f in ("surface.ply", "surface.glb", "deviation.ply",
                      "heatmap.png", "residuals.json", "meta.json"):
                assert (d / f).exists(), f"missing {f} in {d}"
        assert (out / "scene_report.json").exists()
