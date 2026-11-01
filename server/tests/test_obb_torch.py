"""The OBB's kernels on the device (segmentation/obb_torch.py) against the numpy references of
segmentation.pipeline: the vertical RANSAC yaws (same hypotheses, same supports), the footprint bootstrap
(same seeded uniforms, σ to 1e-9 relative, same seed), the extents."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from segmentation import obb_torch as OT  # noqa: E402
from segmentation import pipeline as SP  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")
VCFG = {"enabled": True, "ransac_iters": 200, "dist_m": 0.02, "vertical_tol_deg": 10.0, "max_planes": 4,
        "sample": 60000, "seed": 0}


def _cabinet(seed=0, yaw_deg=23.0):
    """A box with two vertical faces and a top, turned by ``yaw_deg``, with noise and strays."""
    rng = np.random.default_rng(seed)
    n = 5000
    front = np.column_stack([rng.uniform(0, 1.2, n), rng.uniform(0, 2.0, n), np.zeros(n)])
    side = np.column_stack([np.zeros(n), rng.uniform(0, 2.0, n), rng.uniform(0, 0.6, n)])
    top = np.column_stack([rng.uniform(0, 1.2, n // 2), np.full(n // 2, 2.0), rng.uniform(0, 0.6, n // 2)])
    strays = rng.uniform(-0.5, 2.5, (30, 3))
    P = np.concatenate([front, side, top, strays]) + rng.normal(0, 0.004, (3 * n - n // 2 + 30, 3))
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return P @ R.T + [3.0, 0.0, -2.0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vertical_plane_yaws_equal_the_numpy_recipe(seed):
    P = _cabinet(seed, yaw_deg=10.0 + 25.0 * seed)
    a = SP._vertical_plane_yaws_numpy(P, VCFG)
    b = SP._vertical_plane_yaws(P, VCFG, device=DEV)
    assert len(a) == len(b) >= 2
    for (ya, sa), (yb, sb) in zip(a, b):
        assert sa == sb
        assert abs(ya - yb) < 1e-9 or abs(abs(ya - yb) - np.pi / 2) < 1e-9


def test_footprint_error_equals_the_numpy_recipe():
    for seed in range(3):
        P = _cabinet(seed)[:, [0, 2]]
        ident = SP._footprint_identity(P)
        for yaw in (0.0, 0.3, 1.2):
            sa, na, ka = SP._footprint_error_numpy(P, yaw, identity=ident)
            sb, nb, kb = SP._footprint_error(P, yaw, identity=ident, device=DEV)
            assert na == nb and ka == kb
            assert np.isclose(sa, sb, rtol=1e-9, atol=1e-15)
            assert sa > 0


def test_extents_equal():
    P = _cabinet(1)[:, [0, 2]]
    yaws = [0.0, 0.4, 1.1, 1.5]
    E = OT.extents_t(P, yaws, DEV)
    for k, a in enumerate(yaws):
        c, s_ = np.cos(-a), np.sin(-a)
        r = P @ np.array([[c, -s_], [s_, c]]).T
        assert np.allclose(E[k], r.max(0) - r.min(0), atol=1e-12, rtol=0)
