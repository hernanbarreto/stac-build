"""The batched held-out of F5 (precision.refine.heldout_rms / heldout_rms_blocks / project_obs) against
the one-track-at-a-time references, on a synthetic OPENCV scene with a lens: the same tracks, the
same RMS per track to 1e-9 px, the same dropped ledger; the per-block case with two camera blocks."""
import numpy as np
import pytest

from precision import refine as RF
from tests.test_ba_torch import CAM, WH, _scene

SOLVER = {"max_iter": 50, "eps_px": 1e-6, "roundtrip_ulps": 64}


def test_project_obs_equals_project_point_by_point():
    rng = np.random.default_rng(0)
    w2c, pts, groups, X = _scene(n_cam=4, n_pts=40)
    Xs = np.array(list(X.values()))[:30]
    ii = rng.integers(0, len(w2c), len(Xs))
    got = RF.project_obs(Xs, w2c[ii], np.asarray(CAM))
    want = np.array([RF.project(Xs[k][None], w2c[ii[k]], CAM)[0] for k in range(len(Xs))])
    assert np.allclose(got, want, atol=1e-9, rtol=0)
    # per-observation cameras
    cam2 = list(CAM); cam2[0] *= 1.01; cam2[4] = -0.05
    P = np.where((ii % 2 == 0)[:, None], np.asarray(CAM)[None], np.asarray(cam2)[None])
    got2 = RF.project_obs(Xs, w2c[ii], P)
    want2 = np.array([RF.project(Xs[k][None], w2c[ii[k]], P[k])[0] for k in range(len(Xs))])
    assert np.allclose(got2, want2, atol=1e-9, rtol=0)


def test_heldout_rms_batched_equals_the_per_track_reference():
    w2c, pts, groups, X = _scene(n_cam=10, n_pts=400, noise_px=0.3)
    held = {t: obs for k, (t, obs) in enumerate(groups.items()) if k % 3 == 0}
    d1, d2 = {"tracks": 0, "observations": 0}, {"tracks": 0, "observations": 0}
    a = RF.heldout_rms(held, w2c, CAM, SOLVER, 1.0, dropped=d1)
    b = RF.heldout_rms_per_track(held, w2c, CAM, SOLVER, 1.0, dropped=d2)
    assert a.keys() == b.keys() and len(a) > 50
    assert max(abs(a[t] - b[t]) for t in a) < 1e-9
    assert d1 == d2


def test_heldout_rms_blocks_equals_the_per_track_reference():
    w2c, pts, groups, X = _scene(n_cam=10, n_pts=400, noise_px=0.3)
    held = {t: obs for k, (t, obs) in enumerate(groups.items()) if k % 3 == 1}
    cam2 = list(CAM); cam2[0] *= 1.02; cam2[1] *= 1.02; cam2[4] = 0.03
    block_of = np.array([0] * 5 + [1] * 5)
    r = RF.RungResult("R3", w2c, [list(CAM), cam2], block_of, 0.0, 0, "CONVERGENCE")

    class Cfg:
        min_tri_deg = 1.0
    a, da = RF._heldout_of(r, held, SOLVER, Cfg)
    b, db = RF._heldout_of_blocks_per_track(r, held, SOLVER, Cfg)
    assert a.keys() == b.keys() and len(a) > 50
    assert max(abs(a[t] - b[t]) for t in a) < 1e-9
    assert da == db
