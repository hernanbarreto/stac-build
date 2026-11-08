"""tools/amb3r: AMB3R's per-frame records carried onto the record grid F0 reads."""
import numpy as np

from precision.camera import grid_to_native as g2n_pipeline, omega_grid_for
from tools.amb3r.records import K_between, full_frame_grid, grid_to_native, native_to_grid, resample


def _plane_depth(K, n, d0, u, v):
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return d0 / (ray @ n)


def test_omega_at_832_is_already_the_record_grid():
    g = omega_grid_for(1024, 576, "max_size", 832).to_dict()
    a = full_frame_grid(1024, 576, 832, 464)
    assert all(g[k] == a[k] for k in ("w", "h", "crop_x", "crop_y", "crop_w", "crop_h", "content_w", "content_h"))


def test_a_da3_grid_is_carried_onto_the_record_grid_without_bending_a_plane():
    W, H = 1024, 576
    a = full_frame_grid(W, H, 826, 462)
    g = omega_grid_for(W, H, "max_size", 832).to_dict()
    Ka = np.array([[690.0, 0, 412.6], [0, 688.0, 230.4], [0, 0, 1]])
    n = np.array([0.2, -0.1, 1.0])
    n /= np.linalg.norm(n)
    yy, xx = np.mgrid[0:a["h"], 0:a["w"]].astype(np.float64)
    depth = _plane_depth(Ka, n, 3.0, xx, yy)
    d, ok = resample(depth, depth > 0, a, g)
    Kg = K_between(Ka, a, g)
    r, c = np.nonzero(ok)
    assert len(r) > 0.95 * g["w"] * g["h"]
    z = d[r, c].astype(np.float64)
    X = np.stack([(c - Kg[0, 2]) / Kg[0, 0] * z, (r - Kg[1, 2]) / Kg[1, 1] * z, z], -1)
    assert np.max(np.abs(X @ n - 3.0)) < 2e-3 * 3.0


def test_the_local_grid_maps_equal_the_pipelines():
    gm = omega_grid_for(1024, 576, "max_size", 832)
    uv = np.random.default_rng(0).uniform(0, 800, (100, 2))
    assert np.allclose(grid_to_native(uv, gm.to_dict()), g2n_pipeline(uv, gm))
    assert np.allclose(native_to_grid(grid_to_native(uv, gm.to_dict()), gm.to_dict()), uv)
