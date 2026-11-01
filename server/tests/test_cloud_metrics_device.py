"""The cloud metrics' stable point sample on the device (precision.cloud_metrics.stable_point_sample,
`reconstruction.precision.cloud_metrics.device` cpu | cuda) against the numpy path of record: the same rows,
ties among identical keys by row order, on provenance keys and on xyz-bit keys."""
import os

import numpy as np
import pytest

from precision import cloud_metrics as CM

torch = pytest.importorskip("torch")
DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _fields(n, rng, keyed=True):
    f = {"x": rng.normal(0, 3, n).astype(np.float32), "y": rng.normal(0, 3, n).astype(np.float32),
         "z": rng.normal(0, 3, n).astype(np.float32)}
    if keyed:
        f["frame_global"] = rng.integers(0, 300, n).astype(np.int32)
        f["pixel_row"] = rng.integers(0, 832, n).astype(np.int32)
        f["pixel_col"] = rng.integers(0, 464, n).astype(np.int32)
    # identical points (ties of the key): the stable order by row decides
    for k in f:
        f[k][::97] = f[k][0]
    return f


@pytest.mark.parametrize("keyed", [True, False])
def test_stable_point_sample_on_the_device_equals_numpy(keyed):
    rng = np.random.default_rng(7)
    n = 200_000
    f = _fields(n, rng, keyed)
    rows = np.sort(rng.choice(n, 150_000, replace=False))
    for n_max in (5, 1000, 50_000, 149_999):
        a = CM.stable_point_sample(f, rows, n_max)
        b = CM.stable_point_sample(f, rows, n_max, device=DEV)
        assert np.array_equal(a, b), n_max
        assert len(a) == n_max
    # no sampling needed: the rows themselves
    assert np.array_equal(CM.stable_point_sample(f, rows, 150_000, device=DEV), rows)


def test_the_uint64_order_is_kept_through_the_sign_flip():
    rng = np.random.default_rng(3)
    h = rng.integers(0, 2 ** 63 - 1, 50_000, dtype=np.int64).astype(np.uint64)
    h[::3] |= np.uint64(1 << 63)                    # keys with the top bit set sort AFTER the others in uint64
    want = np.argsort(h, kind="stable")
    key = torch.as_tensor(np.ascontiguousarray(h.view(np.int64)), device=DEV) ^ (-(1 << 63))
    got = torch.sort(key, stable=True).indices.cpu().numpy()
    assert np.array_equal(want, got)
