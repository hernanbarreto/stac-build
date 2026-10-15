"""The plane RANSAC and the support mesh on torch (USER 2026-10-15) equal the numpy recipes they
replaced, on the CPU and on CUDA under strict deterministic mode, bit for bit across runs."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np                                                   # noqa: E402
import pytest                                                        # noqa: E402
import torch                                                         # noqa: E402

from reconstruction.geometry.primitives import fit_plane_ransac      # noqa: E402
from reconstruction.surface_fit import fit_torch as FT               # noqa: E402
from reconstruction.surface_fit.support import trimmed_quad_mesh_np  # noqa: E402


def _cloud():
    rng = np.random.default_rng(3)
    n = 60000
    on = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.normal(0, 0.003, n)])
    R = np.array([[0.96, 0.0, 0.28], [0.0, 1.0, 0.0], [-0.28, 0.0, 0.96]])
    on = on @ R.T + np.array([0.3, -0.2, 1.0])
    out = rng.uniform(-2, 2, (15000, 3)) + np.array([0.3, -0.2, 1.0])
    return np.vstack([on, out])


def test_the_torch_ransac_equals_the_seeded_numpy_recipe():
    pts = _cloud()
    ref = fit_plane_ransac(pts, dist_thresh=0.012, iters=200, min_inlier_frac=0.3,
                           measure_curvature=False, seed=0)
    got = FT.fit_plane_ransac_t(pts, dist_thresh=0.012, iters=200, min_inlier_frac=0.3, seed=0)
    assert ref is not None and got is not None
    assert np.allclose(got.normal, ref.normal, atol=1e-9) and abs(got.d - ref.d) < 1e-9
    assert np.array_equal(got.inliers, ref.inliers) and abs(got.rms - ref.rms) < 1e-9
    assert got.n_points == len(pts) and 0.75 < got.inlier_frac < 0.85
    again = FT.fit_plane_ransac_t(pts, dist_thresh=0.012, iters=200, min_inlier_frac=0.3, seed=0)
    assert again.normal.tobytes() == got.normal.tobytes() and again.inliers.tobytes() == got.inliers.tobytes()
    assert FT.fit_plane_ransac_t(np.random.default_rng(1).normal(0, 1, (5000, 3)), 0.012, 50, 0.3) is None


def _uv():
    rng = np.random.default_rng(7)
    uv = rng.uniform(0, 3, (40000, 2))
    hole = ((uv[:, 0] - 1.5) ** 2 + (uv[:, 1] - 1.5) ** 2) < 0.35 ** 2
    notch = (uv[:, 0] > 2.6) & (uv[:, 1] < 0.8)
    return uv[~hole & ~notch]


def test_the_torch_support_mesh_equals_the_numpy_recipe():
    uv = _uv()
    ref = trimmed_quad_mesh_np(uv, 0.05, 0.08)
    got = FT.trimmed_quad_mesh_t(uv, 0.05, 0.08)
    assert got.vertices_uv.shape == ref.vertices_uv.shape and got.faces.shape == ref.faces.shape
    assert np.allclose(got.vertices_uv, ref.vertices_uv, atol=1e-12)
    assert np.array_equal(got.faces, ref.faces)
    assert got.support_fraction == ref.support_fraction and got.area_m2 == ref.area_m2
    assert 0.8 < got.support_fraction < 0.98, "a hole and a notch in the support"


def test_the_device_is_declared():
    import config as C
    try:
        FT.use_device(None)
        import pytest as _p
        with _p.raises(RuntimeError, match="surface_fit.fit_device"):
            C.cfg, saved = {"surface_fit": {}}, C.cfg
            try:
                FT.fit_device()
            finally:
                C.cfg = saved
        assert (C.cfg.get("surface_fit") or {}).get("fit_device") in ("cuda", "cpu")
    finally:
        FT.use_device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_on_cuda_the_fit_and_the_mesh_match_the_cpu_under_strict_mode():
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        pytest.skip("CUDA initialised without the pinned cuBLAS workspace")
    import repro
    pts, uv = _cloud(), _uv()
    cpu_fit = FT.fit_plane_ransac_t(pts, 0.012, 200, 0.3, seed=0)
    cpu_mesh = FT.trimmed_quad_mesh_t(uv, 0.05, 0.08)
    try:
        FT.use_device("cuda")
        with repro.deterministic_torch(0):
            runs = [(FT.fit_plane_ransac_t(pts, 0.012, 200, 0.3, seed=0),
                     FT.trimmed_quad_mesh_t(uv, 0.05, 0.08)) for _ in range(2)]
    finally:
        FT.use_device("cpu")
    for fit, mesh in runs:
        assert np.allclose(fit.normal, cpu_fit.normal, atol=1e-9) and np.array_equal(fit.inliers, cpu_fit.inliers)
        assert np.allclose(mesh.vertices_uv, cpu_mesh.vertices_uv, atol=1e-12) and np.array_equal(mesh.faces, cpu_mesh.faces)
    assert runs[0][0].inliers.tobytes() == runs[1][0].inliers.tobytes()
    assert runs[0][1].vertices_uv.tobytes() == runs[1][1].vertices_uv.tobytes()
