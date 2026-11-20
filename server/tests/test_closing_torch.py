"""The closing pass's arithmetic on a device (USER 2026-11-19: "Todo debe hacerse en placa") against the record's
numpy lines — THE SAME BITS: an exact fused multiply-add built from plain float64 operations repeats the BLAS
product numpy's matmul runs (the placement, the record depth); the rest is elementwise."""
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

FORK = Path(os.environ.get("STAC_FORK_DIR") or Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
sys.path.insert(0, str(FORK))
from loop_utils import closing_torch as CT  # noqa: E402
from loop_utils import metric_lock as ML  # noqa: E402

T = torch.as_tensor


def _exact_fma(a, b, c):
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _families(rng, n):
    a, b = rng.normal(size=n), rng.normal(size=n)
    m = rng.integers(1 << 52, 1 << 53, n).astype(np.float64)
    x32 = rng.normal(size=n).astype(np.float32).astype(np.float64) * 7
    return {
        "random magnitudes": (rng.normal(size=n) * 10.0 ** rng.integers(-6, 6, n),
                              rng.normal(size=n) * 10.0 ** rng.integers(-6, 6, n),
                              rng.normal(size=n) * 10.0 ** rng.integers(-6, 6, n)),
        "cancellation": (a, b, -(a * b) * (1 + rng.integers(-3, 4, n) * 2.0 ** -52)),
        "a tiny addend": (a, b, rng.normal(size=n) * 1e-17),
        "a huge addend": (a, b, rng.normal(size=n) * 1e17),
        "a float32 coordinate by a rotation entry": (x32, rng.uniform(-1, 1, n), rng.normal(size=n) * 3),
        "exact halfway cases": (m, np.ones(n), 0.5 * rng.choice([-1.0, 1.0], n)),
        "halfway with a product error": (m, 1.0 + rng.integers(1, 9, n) * 2.0 ** -52, np.full(n, 0.5)),
        "zeros": (a * (rng.random(n) < 0.5), b, np.where(rng.random(n) < 0.5, 0.0, a)),
    }


def test_the_fused_multiply_add_is_exact():
    rng = np.random.default_rng(5)
    plain_differs = 0
    for name, (a, b, c) in _families(rng, 4000).items():
        got = CT.fma(T(a), T(b), T(c)).numpy()
        ref = np.array([_exact_fma(float(x), float(y), float(z)) for x, y, z in zip(a, b, c)])
        assert np.array_equal(got, ref), name
        plain_differs += int(((a * b + c) != ref).sum())
    assert plain_differs > 3000, "the families do exercise the difference between a fused and a plain a*b+c"


def test_infinities_and_nan_follow_the_plain_operations():
    a = T(np.array([np.inf, 1.0, np.nan, -np.inf, 2.0, np.inf]))
    b = T(np.array([2.0, np.inf, 1.0, 3.0, 5.0, 0.0]))
    c = T(np.array([1.0, -np.inf, 1.0, np.inf, np.inf, 1.0]))
    got, want = CT.fma(a, b, c).numpy(), (a * b + c).numpy()
    assert np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])


def _rot(rng):
    w = rng.normal(size=3) * 0.7
    th = np.linalg.norm(w)
    Kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def _record_place(wp, s, R, t):                       # loop_utils.sim3utils.apply_sim3_direct, then apply()'s cast
    return (s * np.matmul(R, wp[..., np.newaxis]).squeeze(-1) + t).astype(np.float32)


@pytest.mark.parametrize("shape", [(3, 40, 64), (2, 37, 61), (9, 16, 23)])
def test_the_placement_is_the_records_bit_for_bit(shape):
    rng = np.random.default_rng(sum(shape))
    wp = (rng.normal(size=shape + (3,)) * np.array([3.0, 1.5, 6.0])).astype(np.float32)
    wp[0, 0, 0] = np.nan
    wp[0, 1, 1] = [np.inf, 1.0, -2.0]
    s, R, t = float(np.exp(rng.normal() * 0.3)), _rot(rng), rng.normal(size=3) * 4
    rec = _record_place(wp, s, R, t)
    got = CT.place(T(wp), s, R, t).numpy()
    fin = np.isfinite(rec).all(-1)
    assert got.dtype == np.float32 and np.array_equal(got[fin], rec[fin]), (
        "the device's placement is not numpy's matmul bit for bit — numpy's BLAS kernel on this machine is not the "
        "one closing_torch repeats (its module docstring)")
    assert np.array_equal(np.isfinite(got), np.isfinite(rec))


@pytest.mark.parametrize("hw", [(40, 64), (37, 61), (16, 62), (5, 63), (7, 3)])
def test_the_record_depth_is_the_servers_bit_for_bit(hw):
    rng = np.random.default_rng(hw[1])
    wp = (rng.normal(size=hw + (3,)) * np.array([3.0, 1.5, 6.0]) + np.array([0, 0, 5.0])).astype(np.float32)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = _rot(rng), rng.normal(size=3)
    rec = ((wp - c2w[:3, 3]) @ c2w[:3, 2]).astype(np.float32)        # workers/map_worker._emit_omega_depth
    got = CT.record_depth(T(wp), c2w).numpy()
    assert got.dtype == np.float32 and np.array_equal(got, rec)


def test_the_uncertainty_map_the_range_and_the_backfill_are_the_records():
    rng = np.random.default_rng(8)
    H, W = 33, 47
    pa = (rng.normal(size=(H, W, 3)) * 4).astype(np.float32)
    pb = (pa + rng.normal(size=(H, W, 3)) * 0.02).astype(np.float32)
    ca = (rng.random((H, W)) + 1e-5).astype(np.float32)
    cb = (rng.random((H, W)) + 1e-5).astype(np.float32)
    for c in (ca, cb):
        c[rng.random((H, W)) < 0.2] = 0.0
        c[rng.random((H, W)) < 0.05] = np.float32(1e-5)                # exactly on the mask's edge
    # cube_run._unc_feed
    ok = (ca > 1e-5) & (cb > 1e-5)
    d = np.full((H, W), np.nan, np.float32)
    d[ok] = np.linalg.norm(pb[ok].astype(np.float64) - pa[ok].astype(np.float64), axis=1)
    got = CT.two_copy_distance(T(pb), T(cb), T(pa), T(ca)).numpy()
    assert np.array_equal(got, d, equal_nan=True) and np.isfinite(d).sum() > 500
    assert np.isnan(CT.two_copy_distance(T(pb), T(cb * 0), T(pa), T(ca)).numpy()).all()
    # cube_run._dg_feed
    cam = np.array([0.3, -0.2, 0.1])
    for rng_m in (6.0, 15.0, 0.0):
        want = (np.where(np.linalg.norm(pa - cam, axis=-1) <= rng_m, ca, 0.0).astype(np.float32) if rng_m > 0
                else ca.astype(np.float32))
        assert np.array_equal(CT.conf_within_range(T(pa), cam, T(ca), rng_m).numpy(), want), rng_m
    assert 0 < (np.linalg.norm(pa - cam, axis=-1) <= 6.0).sum() < H * W
    # metric_lock.backfill_mask
    for thr in (None, -1.0, 0.0, 0.31, float(np.float32(0.4)), float(np.nextafter(np.float32(0.4), np.float32(1)))):
        assert np.array_equal(CT.backfill_mask(T(ca), thr).numpy(), ML.backfill_mask(ca, thr)), thr


def test_a_float32_compared_with_a_python_number_as_numpy_compares():
    rng = np.random.default_rng(2)
    for t in list(rng.random(200) * 3) + [1e-5, 1e-6, 0.0, 0.1, 1.0, float(np.float32(0.7)), 1 / 3]:
        t32 = np.float32(t)
        c = np.array([np.nextafter(t32, np.float32(-9)), t32, np.nextafter(t32, np.float32(9))], np.float32)
        assert np.array_equal(c >= CT.f32_ge(t), c >= t) and np.array_equal(c > CT.f32_gt(t), c > t), t


def test_the_depth_correction_moves_the_points_as_the_record():
    rng = np.random.default_rng(12)
    H, W = 29, 41
    wp = (rng.normal(size=(H, W, 3)) * 3 + np.array([0, 0, 6.0])).astype(np.float32)
    dep = (rng.random((H, W)) * 8).astype(np.float32)
    dep[rng.random((H, W)) < 0.1] = 0.0
    dep[0, 0], dep[0, 1] = np.nan, np.float32(1e-6)
    cam = rng.normal(size=3)
    for a, b in ((1.013, -0.021), (0.82, 0.23), (1.0, 0.0)):
        want, _d = ML.apply_depth_correction(wp, dep, cam, a, b)
        got = CT.depth_correction(T(wp), T(dep), cam, a, b).numpy()
        assert got.dtype == np.float32 and np.array_equal(got, want), (a, b)


def test_the_device_is_typed():
    with pytest.raises(ValueError, match="unknown device"):
        CT.device_of("numpy")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cube_closing_device"):
            CT.device_of("cuda")
