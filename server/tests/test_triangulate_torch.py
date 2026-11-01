"""The batched DLT on torch (precision.refine._dlt_stack_t, `refine.triangulation_device` cpu | cuda) against
the numpy path of record: the same tracks kept, the same landmarks to 1e-9, through triangulate_tracks and
the leave-one-view-out held-out on the synthetic OPENCV scene."""
import os

import numpy as np
import pytest

from precision import refine as RF
from tests.test_ba_torch import CAM, WH, _scene

torch = pytest.importorskip("torch")
DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")
SOLVER = {"max_iter": 50, "eps_px": 1e-6, "roundtrip_ulps": 64}


class _Cfg:
    min_tri_deg = 1.0


def _on(dev, fn):
    RF.use_device(dev)
    try:
        return fn()
    finally:
        RF.use_device(None)


def test_dlt_stack_on_the_device_equals_numpy():
    rng = np.random.default_rng(0)
    w2c, pts, groups, X = _scene(n_cam=9, n_pts=400, noise_px=0.3)
    tracks, counts, starts, ii, uv = RF._track_arrays(groups)
    from precision.camera import undistort_normalized_tracks
    xn, ok = undistort_normalized_tracks(uv, RF.K_of(CAM), np.asarray(CAM[4:8], np.float64), starts, counts, **SOLVER)
    for k in np.unique(counts):
        sel = np.nonzero((counts == k) & ok)[0]
        idx = starts[sel][:, None] + np.arange(int(k))[None, :]
        Xa, ga = RF._dlt_stack(xn[idx], w2c[ii[idx]], 1.0)
        Xb, gb = RF._dlt_stack_t(xn[idx], w2c[ii[idx]], 1.0, DEV)
        assert np.array_equal(ga, gb)
        assert np.allclose(Xa[ga], Xb[gb], rtol=1e-9, atol=1e-9)
    # a degenerate stack: all rays from one centre → not kept on either path
    T = np.tile(np.eye(4), (3, 4, 1, 1))
    x = rng.normal(0, 0.1, (3, 4, 2))
    _, g1 = RF._dlt_stack(x, T, 1.0)
    _, g2 = RF._dlt_stack_t(x, T, 1.0, DEV)
    assert not g1.any() and np.array_equal(g1, g2)


def test_triangulate_tracks_and_leave_one_out_equal_across_devices():
    w2c, pts, groups, X = _scene(n_cam=12, n_pts=500, noise_px=0.4)
    a = _on(None, lambda: RF.triangulate_tracks(groups, w2c, CAM, SOLVER, 1.0))
    b = _on(DEV, lambda: RF.triangulate_tracks(groups, w2c, CAM, SOLVER, 1.0))
    assert a.keys() == b.keys() and len(a) > 400
    assert max(float(np.max(np.abs(a[t] - b[t]))) for t in a) < 1e-9
    held = {t: obs for k, (t, obs) in enumerate(groups.items()) if k % 2 == 0}
    r = RF.RungResult("R0", w2c, [list(CAM)], np.zeros(len(w2c), int), 0.0, 0, "CONVERGENCE")
    ha = _on(None, lambda: RF.heldout_leave_one_view_out(held, r, SOLVER, _Cfg, max_px=float(np.hypot(*WH))))
    hb = _on(DEV, lambda: RF.heldout_leave_one_view_out(held, r, SOLVER, _Cfg, max_px=float(np.hypot(*WH))))
    assert ha.keys() == hb.keys() and max(abs(ha[i] - hb[i]) for i in ha) < 1e-9


def test_use_device_rejects_unknown_names():
    with pytest.raises(RF.RefineError):
        RF.use_device("tpu")
    assert RF.device_of() == "numpy"
