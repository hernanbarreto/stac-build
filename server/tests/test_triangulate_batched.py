"""The batched triangulation (USER 2026-10-08, speed) equals the one-track-at-a-time reference: the same
tracks kept, in the same order, the same bits, the same dropped counts. Synthetic, CPU only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import refine as RF  # noqa: E402

SOLVER = {"max_iter": 100, "eps_px": 1e-9, "roundtrip_ulps": 64}
PARAMS = [388.07, 384.79, 232.03, 413.89, 0.00798, -0.00726, 0.00007, -0.00006]


def _cams(n):
    out = []
    for i in range(n):
        a = 0.05 * i
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        C = np.array([0.3 * i, 0.02 * i, 0.0])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = -R @ C
        out.append(T)
    return np.array(out)


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    w2c = _cams(12)
    groups = {}
    for t in range(600):
        X = np.array([rng.uniform(-2, 4), rng.uniform(-1, 1), rng.uniform(3, 9)])
        k = int(rng.integers(2, 7))
        views = sorted(rng.choice(12, k, replace=False).tolist())
        obs = []
        for i in views:
            uv = RF.project(X[None], w2c[i], PARAMS)[0] + rng.normal(0, 0.3, 2)
            obs.append((i, uv))
        groups[1000 + t] = obs
    groups[5] = [(0, np.array([100.0, 100.0]))]                          # one view: never triangulated
    groups[7] = [(0, np.array([200.0, 300.0])), (1, np.array([200.0, 300.0]))]   # tiny baseline
    groups[9] = [(2, np.array([1e7, 1e7])), (3, np.array([1e7, -1e7]))]  # no round trip through the lens
    behind = np.array([0.5, 0.0, -4.0])                                  # behind the cameras
    groups[11] = [(i, RF.project(behind[None], w2c[i], PARAMS)[0]) for i in (0, 4, 8)]
    return groups, w2c


def test_batched_equals_the_per_track_reference_bit_for_bit():
    groups, w2c = _scene()
    d_ref, d_bat = {}, {}
    A = RF.triangulate_tracks_per_track(groups, w2c, PARAMS, SOLVER, 1.0, dropped=d_ref)
    B = RF.triangulate_tracks(groups, w2c, PARAMS, SOLVER, 1.0, dropped=d_bat)
    assert list(A.keys()) == list(B.keys()) and len(A) > 400
    assert all(np.array_equal(A[t], B[t]) for t in A)
    assert d_ref == d_bat and d_bat.get("tracks") == 1          # the far track dropped and counted
    assert 5 not in B and 11 not in B


def test_batch_bound_does_not_change_the_result(monkeypatch):
    groups, w2c = _scene(1)
    full = RF.triangulate_tracks(groups, w2c, PARAMS, SOLVER, 1.0)
    monkeypatch.setattr(RF, "_TRI_BATCH_OBS", 7)
    small = RF.triangulate_tracks(groups, w2c, PARAMS, SOLVER, 1.0)
    assert list(full.keys()) == list(small.keys())
    assert all(np.array_equal(full[t], small[t]) for t in full)
