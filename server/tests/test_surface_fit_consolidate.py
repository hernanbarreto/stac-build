"""
Stage-1 consolidation tests: the double-layer ("onion") failure mode must
collapse to a thin sheet BEFORE fitting, so the escalation sees shape instead
of registration bias — while the stage-4 report still measures against the
original layered cloud.

The WLOP test needs the CloudComPy310 conda env (CGAL bindings); it skips
cleanly when the satellite launcher can't run.
"""
import numpy as np
import pytest

from reconstruction.surface_fit.consolidate import (consolidate_mls,
                                                    consolidate_wlop)
from reconstruction.surface_fit.runner import fit_segment

SIGMA = 0.002


def make_double_layer(n=40_000, sep=0.008, sigma=SIGMA, seed=21):
    """Horizontal plane at z=0 with the x>0 half duplicated `sep` above —
    the exact TSDF 'onion layer' pathology."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(0, sigma, n)
    pts = np.column_stack([x, y, z])
    ghost = pts[pts[:, 0] > 0].copy()
    ghost[:, 2] += sep
    return np.vstack([pts, ghost])


def thickness_mm(pts, region):
    """std (mm) of z inside a boolean region — layer collapse metric."""
    return float(pts[region][:, 2].std() * 1000.0)


class TestMLS:
    def test_double_layer_collapses(self):
        cloud = make_double_layer()
        out = consolidate_mls(cloud, radius=0.06, iterations=2)
        # right half held two sheets 8 mm apart (std ≈ 4 mm); must end thin
        assert thickness_mm(cloud, cloud[:, 0] > 0.5) > 3.0
        assert thickness_mm(out, out[:, 0] > 0.5) < 1.5

    def test_left_half_undamaged(self):
        """Where there was no ghost layer, MLS must only denoise, not move."""
        cloud = make_double_layer()
        out = consolidate_mls(cloud, radius=0.06, iterations=2)
        left = out[out[:, 0] < -0.5]
        assert abs(left[:, 2].mean()) * 1000.0 < 0.5    # plane stays at z≈0


@pytest.mark.skipif(
    consolidate_wlop(np.random.default_rng(0).uniform(0, 1, (300, 3))) is None,
    reason="CGAL WLOP satellite (CloudComPy310 env) not available")
class TestWLOP:
    def test_double_layer_collapses(self):
        cloud = make_double_layer(n=25_000)
        out = consolidate_wlop(cloud, neighbor_radius_m=0.06,
                               select_percentage=25.0, iterations=25)
        assert out is not None and len(out) > 1000
        assert thickness_mm(out, out[:, 0] > 0.5) < 1.0


def make_full_double_layer(n=40_000, sep=0.008, sigma=SIGMA, seed=22):
    """The whole plane seen twice, `sep` apart (two chunks, registration
    bias) — collapses to the mid-plane with no step."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(0, sigma, n)
    pts = np.column_stack([x, y, z])
    ghost = pts.copy()
    ghost[:, 2] += sep
    return np.vstack([pts, ghost])


class TestEndToEnd:
    def test_consolidated_fit_accepts_plane(self):
        """With consolidation ON, the fully-doubled segment must fit as a
        PLANE whose escalation residual is clean, while the stage-4 report
        (vs the ORIGINAL layered cloud) honestly keeps the bimodal deviation."""
        cloud = make_full_double_layer()
        fs = fit_segment(cloud, consolidate_method="mls", label="floor",
                         models=["plane", "bspline"])
        assert fs is not None and fs.kind == "plane"
        # report vs original: the two layers straddle the fitted plane → the
        # residual spread must reflect the ±half-separation, not be hidden
        assert fs.report.stats.p95_mm > 2.0
        assert fs.n_input_points == len(cloud)

    def test_partial_ghost_keeps_step_visible(self):
        """Ghost on HALF the area: consolidation leaves a genuine step at the
        coverage boundary — the fit must NOT silently call that a plane."""
        cloud = make_double_layer()
        fs = fit_segment(cloud, consolidate_method="mls", label="floor",
                         models=["plane"])
        assert fs.report.stats.moran.relevant or fs.report.stats.p95_mm > 2.0

    def test_without_consolidation_escalates(self):
        """Same cloud, stage 1 off → the plane leaves relevant structure."""
        cloud = make_double_layer()
        fs = fit_segment(cloud, consolidate_method="none", label="floor",
                         models=["plane"])
        assert fs.report.stats.moran.relevant


class TestNormalAware:
    def test_thin_double_sided_wall_not_merged(self):
        """Dos caras REALES de un panel a 4 cm, cámaras a ambos lados: la
        consolidación consciente de normales debe mantener cada cara en su
        plano (no inventar una superficie media inexistente)."""
        from reconstruction.surface_fit.consolidate import (
            consolidate_mls, estimate_oriented_normals)
        rng = np.random.default_rng(40)
        n = 30_000
        face_a = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 2, n),
                                  rng.normal(0.0, 0.002, n)])
        face_b = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 2, n),
                                  0.04 + rng.normal(0.0, 0.002, n)])
        cloud = np.vstack([face_a, face_b])
        cams = np.array([[1.5, 1.0, -2.0], [1.5, 1.0, 2.0]])   # one per side
        normals = estimate_oriented_normals(cloud, cams)
        out = consolidate_mls(cloud, radius=0.06, iterations=2,
                              max_points=None, normals=normals)
        za = out[:n, 2]
        zb = out[n:, 2]
        assert abs(za.mean()) < 0.004, f"cara A se movió a {za.mean()*1000:.1f}mm"
        assert abs(zb.mean() - 0.04) < 0.004, f"cara B se movió a {zb.mean()*1000:.1f}mm"

    def test_same_side_ghost_still_collapses(self):
        """Capa fantasma con la MISMA orientación (registro): sí colapsa."""
        from reconstruction.surface_fit.consolidate import (
            consolidate_mls, estimate_oriented_normals)
        rng = np.random.default_rng(41)
        n = 30_000
        real = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 2, n),
                                rng.normal(0.0, 0.002, n)])
        ghost = real + np.array([0, 0, 0.008])
        cloud = np.vstack([real, ghost])
        cams = np.array([[1.5, 1.0, 2.0]])                     # same side
        normals = estimate_oriented_normals(cloud, cams)
        out = consolidate_mls(cloud, radius=0.06, iterations=2,
                              max_points=None, normals=normals)
        assert out[:, 2].std() * 1000 < 1.5                    # thin sheet


class TestSceneConsolidate:
    def test_end_to_end(self, tmp_path):
        import json
        import open3d as o3d
        from reconstruction.surface_fit.consolidate import scene_consolidate

        rng = np.random.default_rng(42)
        n = 40_000
        pts = np.column_stack([rng.uniform(0, 4, n), rng.uniform(0, 3, n),
                               rng.normal(0, 0.002, n)])
        ghost = pts.copy(); ghost[:, 2] += 0.010
        cloud = np.vstack([pts, ghost])
        colors = rng.uniform(0, 1, (len(cloud), 3))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(tmp_path / "cleaned_cloud.ply"), pcd)
        eye = " ".join(f"{x:.9g}" for x in np.array(
            [[1,0,0,2],[0,1,0,1.5],[0,0,1,2.0],[0,0,0,1]], float).reshape(-1))
        (tmp_path / "camera_poses.txt").write_text(eye + "\n")
        (tmp_path / "fine_register_report.json").write_text(json.dumps(
            {"sep_after_m": {"0-1": 0.015}}))

        stats = scene_consolidate(tmp_path)
        assert stats is not None
        assert abs(stats["radius_m"] - 0.03) < 1e-9     # 2×0.015, adaptativo
        assert (tmp_path / "cleaned_cloud_raw.ply").exists()
        out = o3d.io.read_point_cloud(str(tmp_path / "cleaned_cloud.ply"))
        opts = np.asarray(out.points); ocol = np.asarray(out.colors)
        assert len(opts) == len(cloud)                  # orden/conteo intactos
        assert np.allclose(ocol, colors, atol=1/255)    # colores preservados
        assert opts[:, 2].std() * 1000 < 2.0            # capas colapsadas
        raw = np.asarray(o3d.io.read_point_cloud(
            str(tmp_path / "cleaned_cloud_raw.ply")).points)
        assert raw[:, 2].std() * 1000 > 4.0             # el crudo sigue bicapa
