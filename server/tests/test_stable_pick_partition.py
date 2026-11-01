"""The fork's stable_pick by partition (vendor/VGGT-Long/loop_utils/stable_sample.py) against its
definition, the full stable sort (_stable_pick_reference): the same indices, in the same order,
for every size, every k, ties at the boundary and salts; and depth_pair_samples with the source
half precomputed (depth_pair_source) gives the samples it gives alone."""
import sys
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))
from loop_utils import stable_sample as SS  # noqa: E402
from loop_utils import metric_lock as ML  # noqa: E402


@pytest.mark.parametrize("n", [0, 1, 2, 7, 100, 5000, 70000])
def test_stable_pick_is_the_stable_sort(n):
    rng = np.random.default_rng(n)
    keys = rng.integers(0, 2 ** 63, n, dtype=np.uint64)
    for k in sorted({0, 1, 2, n // 3, n // 2, n - 1, n, n + 5} - {-1}):
        for salt in (0, 7):
            a = SS.stable_pick(keys, k, salt=salt)
            b = SS._stable_pick_reference(keys, k, salt=salt)
            assert a.dtype == b.dtype and np.array_equal(a, b), (n, k, salt)


def test_ties_at_the_boundary_fall_to_the_earlier_element(monkeypatch):
    # equal hashes (the salted hash of equal keys): duplicates straddling the k-th place
    rng = np.random.default_rng(3)
    base = rng.integers(0, 2 ** 63, 50, dtype=np.uint64)
    keys = np.concatenate([base, base, base[:10], base])          # every key 3-4 times
    for k in range(0, len(keys) + 1, 7):
        assert np.array_equal(SS.stable_pick(keys, k), SS._stable_pick_reference(keys, k)), k


def test_depth_pair_samples_with_the_source_precomputed():
    rng = np.random.default_rng(0)
    H, W = 60, 80
    wp_src = rng.normal(0, 1, (H, W, 3)); wp_src[..., 2] += 4.0
    conf_src = rng.uniform(0, 1, (H, W)).astype(np.float32)
    conf_src[:5] = 0.0
    wp_dst = wp_src + rng.normal(0, 0.01, (H, W, 3))
    conf_dst = np.ones((H, W), np.float32)
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    w2c = np.eye(4)
    for ms in (500, 1000, 100000):
        a = ML.depth_pair_samples(wp_src, conf_src, wp_dst, conf_dst, w2c, K, max_samples=ms, seed=11)
        src = ML.depth_pair_source(wp_src, conf_src, max_samples=ms, seed=11)
        b = ML.depth_pair_samples(wp_src, conf_src, wp_dst, conf_dst, w2c, K, max_samples=ms, seed=11,
                                  src=src)
        assert (a is None) == (b is None)
        if a is not None:
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    starved = np.zeros((H, W), np.float32)
    assert ML.depth_pair_source(wp_src, starved) is None
    assert ML.depth_pair_samples(wp_src, starved, wp_dst, conf_dst, w2c, K) is None
