"""F6's bend with every keyframe of a window fitted at once on the device (precision.bend_torch) against the numpy
path (depth_on_f5.bend_coefficients): the same coefficients and σ to 1e-9, the same kept terms, the same fallbacks,
on synthetic landmark rows with outliers, a tilt, empty keyframes and keyframes under min_rows."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from precision import bend_torch as BT  # noqa: E402
from precision import depth_on_f5 as D6  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")
W, H = 464, 832


def _rows(seed=0, n_kf=40):
    rng = np.random.default_rng(seed)
    rows = {}
    for i in range(n_kf):
        n = int(rng.integers(0, 400)) if i % 9 else int(rng.integers(0, 8))   # some keyframes nearly empty
        u = rng.uniform(0, W, n); v = rng.uniform(0, H, n)
        k = 1.02 + 0.03 * ((u - W / 2) / W) * (i % 3 == 0) - 0.02 * ((v - H / 2) / H) * (i % 4 == 0)
        r = k + rng.normal(0, 0.01, n)
        out = rng.random(n) < 0.05
        r[out] += rng.normal(0, 0.3, int(out.sum()))                        # outliers for the Huber weights
        rows[i] = (D6.design(u, v, W, H), r)
    rows[5] = (np.zeros((0, 3)), np.zeros(0))
    return rows


@pytest.mark.parametrize("window", [0, 1, 2, 4, 8])
def test_bend_coefficients_on_the_device_equal_numpy(window):
    rows = _rows(window)
    n = len(rows)
    a_c, a_i = D6.bend_coefficients(rows, n, window, 20, 1.345, 10, 1.1)
    b_c, b_i = BT.bend_coefficients_t(rows, n, window, 20, 1.345, 10, 1.1, DEV, budget=20000)   # several batches
    assert a_c.keys() == b_c.keys()
    for i in a_c:
        ia, ib = a_i[i], b_i[i]
        assert ia["fallback"] == ib["fallback"] and ia["rows"] == ib["rows"] and ia["window"] == ib["window"]
        assert np.allclose(a_c[i], b_c[i], rtol=1e-9, atol=1e-12), (i, a_c[i], b_c[i])
        if ia["fallback"] is None:
            assert ia["kept"] == ib["kept"], i
            for key in ("sigma_full", "sigma"):
                x, y = np.asarray(ia[key], float), np.asarray(ib[key], float)
                assert np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(np.isinf(x), np.isinf(y))
                f = np.isfinite(x)
                assert np.allclose(x[f], y[f], rtol=1e-8, atol=1e-14), (i, key, x, y)
    assert sum(1 for i in a_i if a_i[i]["fallback"] is None) > 10
