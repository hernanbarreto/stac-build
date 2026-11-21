"""The fork's rule with its bootstrap medians on the device (precision.judge_torch.decide_change_t) against the fork's
decide_change, key for key, on random inputs of both paths (unclustered and clustered, odd and even sizes, ties,
blocks), and the group bootstrap's standard error against depth_on_f5.cluster_bootstrap_se."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from precision import judge_torch as JT  # noqa: E402
from precision import depth_on_f5 as D6  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _cases():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 64, 301, 1000):
        b = rng.normal(0.02, 0.01, n); a = b - rng.normal(0.001, 0.003, n)
        yield b, a, None
        yield np.round(b, 3), np.round(a, 3), None                       # ties
        cl = rng.integers(0, max(1, n // 7), n)
        yield b, a, cl


def test_decide_change_t_equals_the_fork():
    ML = JT._vendor()
    for b, a, cl in _cases():
        for nb in (1, 37, 2000):
            want = ML.decide_change(b, a, error=0.0005, error_factor=1.1, confidence=0.95, clusters=cl, n_boot=nb, seed=3)
            got = JT.decide_change_t(b, a, error=0.0005, error_factor=1.1, confidence=0.95, clusters=cl, n_boot=nb,
                                     seed=3, device=DEV)
            assert got == want, (len(b), cl is not None, nb)


def test_decide_change_t_across_blocks(monkeypatch):
    ML = JT._vendor()
    monkeypatch.setattr(ML, "_BOOT_BLOCK_ELEMS", 997)                    # several blocks, an odd block size
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, 250); a = b - rng.normal(0.1, 1, 250)
    for cl in (None, rng.integers(0, 40, 250)):
        want = ML.decide_change(b, a, error=0.01, error_factor=1.1, confidence=0.95, clusters=cl, n_boot=500, seed=0)
        got = JT.decide_change_t(b, a, error=0.01, error_factor=1.1, confidence=0.95, clusters=cl, n_boot=500, seed=0,
                                 device=DEV)
        assert got == want


def test_cluster_bootstrap_se_t_equals_the_reference():
    rng = np.random.default_rng(2)
    groups = [rng.normal(0.03, 0.01, int(rng.integers(0, 60))) for _ in range(41)]
    groups[3] = np.round(groups[3], 2)
    for nb in (2, 50, 2000):
        want = D6.cluster_bootstrap_se(groups, nb, 5)
        got = JT.cluster_bootstrap_se_t(groups, nb, 5, DEV)
        assert got == want
    assert JT.cluster_bootstrap_se_t(groups[:1], 100, 0, DEV) == D6.cluster_bootstrap_se(groups[:1], 100, 0) == 0.0
