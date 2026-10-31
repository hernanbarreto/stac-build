"""The cloud detector of the precision edge metric imports its fitters from a module that exists
(2026-10-31: ``reconstruction.surface_fit.escalate`` had left with the meshing stack and every acta's
edge metric failed at import): two walls meeting at a right angle are found as two planes."""
import numpy as np

from segmentation import perfect_object as PO


def test_the_cloud_detector_imports_its_fitters_and_finds_the_two_planes_of_an_edge():
    rng = np.random.default_rng(0)
    n = 6000
    wall_a = np.column_stack([rng.uniform(0, 2, n), rng.uniform(0, 2, n), rng.normal(0, 0.003, n)])          # z = 0
    wall_b = np.column_stack([rng.normal(0, 0.003, n), rng.uniform(0, 2, n), rng.uniform(0, 2, n)])          # x = 0
    pts = np.vstack([wall_a, wall_b])
    regions, min_pts = PO._detect_and_snap_cloud(pts, {"perfect_min_region_faces": 300, "p2c_inlier_dist_m": 0.02,
                                                       "p2c_max_primitives": 6, "p2c_freeform_min_pts": 2000,
                                                       "perfect_max_fit_pts": 20000}, "edge", lambda *a: None)
    planes = [r for r in regions if r["kind"] == "plane"]
    assert len(planes) >= 2 and min_pts == 300
    normals = np.array([np.abs(np.asarray(r["model"].normal, float)) for r in planes[:2]])
    # one normal along z, one along x (either order)
    assert np.isclose(normals[:, 2].max(), 1.0, atol=0.05) and np.isclose(normals[:, 0].max(), 1.0, atol=0.05)
    covered = sum(len(r["v_idx"]) for r in planes)
    assert covered >= 0.9 * len(pts)
