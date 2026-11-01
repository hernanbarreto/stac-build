"""F2's per-frame Huber gains in batches on the device (precision.gauge_torch.huber_location_batch) against the
reference (precision.gauge.huber_location): the same estimate to 1e-9, the same convergence and MAD-zero flags, on
rows of different sizes with outliers, ties and a majority of exact equal values."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from precision import gauge as G  # noqa: E402
from precision import gauge_torch as GT  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _rows(seed=0):
    rng = np.random.default_rng(seed)
    rs, ws = [], []
    for b in range(9):
        n = int(rng.integers(4, 6000))
        r = rng.normal(0.05 * b, 0.02, n)
        out = rng.random(n) < 0.1
        r[out] += rng.normal(0, 0.5, int(out.sum()))
        if b == 3:
            r[: n // 2 + 1] = 0.125                  # more than half exactly equal: the MAD vanishes
        if b == 5:
            r = np.round(r, 3)                       # ties
        rs.append(r); ws.append(rng.uniform(0.1, 2.0, n))
    return rs, ws


def test_batch_equals_the_reference():
    rs, ws = _rows()
    want = [G.huber_location(r, w, 1.345, 1e-12, 10000) for r, w in zip(rs, ws)]
    got = GT.huber_location_batch(rs, ws, 1.345, 1e-12, 10000, DEV)
    for a, b in zip(want, got):
        assert a.converged == b.converged and a.mad_zero == b.mad_zero
        assert abs(a.log - b.log) <= 1e-9 * max(1.0, abs(a.log)), (a, b)
        assert abs(a.iterations - b.iterations) <= 1
    assert any(g.mad_zero for g in want)


def test_a_capped_row_is_reported_not_converged():
    rs, ws = _rows(1)
    want = [G.huber_location(r, w, 1.345, 1e-30, 3) for r, w in zip(rs, ws)]
    got = GT.huber_location_batch(rs, ws, 1.345, 1e-30, 3, DEV)
    assert [g.converged for g in want] == [g.converged for g in got]
    for a, b in zip(want, got):
        assert abs(a.log - b.log) <= 1e-9 * max(1.0, abs(a.log))
