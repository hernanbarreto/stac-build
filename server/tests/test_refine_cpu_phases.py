"""F5's CPU phases around the solver, vectorised (precision.refine, 2026-11-01), against the loops they
replaced: the leave-one-view-out held-out per keyframe (the batched SVD stacks vs the per-track loop), the
split map by searchsorted, the rolling-shutter diagnostic's pairs — on the synthetic OPENCV scene of the
solver tests."""
import numpy as np

from precision import refine as RF
from tests.test_ba_torch import CAM, WH, _scene

SOLVER = {"max_iter": 50, "eps_px": 1e-6, "roundtrip_ulps": 64}


class _Cfg:
    min_tri_deg = 1.0
    seed = 0
    heldout_confidence = 0.95
    permutations = 200


def test_leave_one_view_out_batched_equals_the_loop():
    w2c, pts, groups, X = _scene(n_cam=12, n_pts=500, noise_px=0.4)
    held = {t: obs for k, (t, obs) in enumerate(groups.items()) if k % 2 == 0}
    r = RF.RungResult("R0", w2c, [list(CAM)], np.zeros(len(w2c), int), 0.0, 0, "CONVERGENCE")
    a = RF.heldout_leave_one_view_out(held, r, SOLVER, _Cfg, max_px=float(np.hypot(*WH)))
    b = RF.heldout_leave_one_view_out_per_track(held, r, SOLVER, _Cfg, max_px=float(np.hypot(*WH)))
    assert a.keys() == b.keys() and len(a) >= 10
    assert max(abs(a[i] - b[i]) for i in a) < 1e-9
    # a far cap leaves the frames whose residuals all exceed it out, as the loop does
    a2 = RF.heldout_leave_one_view_out(held, r, SOLVER, _Cfg, max_px=1e-12)
    b2 = RF.heldout_leave_one_view_out_per_track(held, r, SOLVER, _Cfg, max_px=1e-12)
    assert a2 == b2 == {}


def test_split_of_observations_is_the_dict_map():
    rng = np.random.default_rng(1)
    tid = rng.permutation(5000) + 10
    split = rng.integers(0, 2, len(tid)).astype(np.int8)
    obs_track = rng.choice(tid, 40000)
    want = np.array([dict(zip(tid.tolist(), split.tolist()))[int(t)] for t in obs_track], np.int8)
    got = RF.split_of_observations(tid, split, obs_track)
    assert np.array_equal(want, got) and got.dtype == np.int8


def test_rolling_shutter_diag_pairs_equal_the_loop():
    from scipy.stats import spearmanr
    w2c, pts, groups, X = _scene(n_cam=10, n_pts=300, noise_px=0.5)
    held = {t: obs for k, (t, obs) in enumerate(groups.items()) if k % 3 == 0}
    r = RF.RungResult("R0", w2c, [list(CAM)], np.zeros(len(w2c), int), 0.0, 0, "CONVERGENCE")
    rep = RF.rolling_shutter_diag(held, r, SOLVER, _Cfg)
    # the loop's pairs
    import math
    N = len(w2c); ang = np.zeros(N)
    for i in range(N):
        a_, b_ = max(0, i - 1), min(N - 1, i + 1)
        R = w2c[b_, :3, :3] @ w2c[a_, :3, :3].T
        ang[i] = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))) / (b_ - a_)
    Xt = RF.triangulate_tracks(held, w2c, CAM, SOLVER, _Cfg.min_tri_deg)
    xs, ys = [], []
    for t, P in Xt.items():
        for i, p in held[t]:
            ys.append(float(np.linalg.norm(RF.project(P[None], w2c[i], CAM)[0] - p)))
            xs.append(abs(float(p[1]) - CAM[3]) * ang[i])
    rho, pval = spearmanr(xs, ys)
    assert rep["n"] == len(xs) > 100
    assert abs(rep["spearman"] - rho) < 1e-9 and abs(rep["p"] - pval) < 1e-9


def test_track_groups_arrays_equal_the_loop_and_the_lists():
    rng = np.random.default_rng(7)
    frames = [100, 103, 106, 109, 112, 115, 118]
    n = 20000
    track = rng.integers(0, 3000, n); frame = rng.choice(frames + [999], n); uv = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    g = RF.group_tracks(track, frame, uv, frames)
    ref = RF._group_tracks_reference(track, frame, uv, frames)
    assert list(g.keys()) == list(ref.keys())
    for t in ref:
        assert [i for i, _ in g[t]] == [i for i, _ in ref[t]]
        assert np.array_equal(np.array([p for _, p in g[t]]), np.array([p for _, p in ref[t]]))
    # the cached arrays are the lists' content, and _track_arrays reads them (min_views 1, 2, 3)
    for mv in (1, 2, 3):
        a = RF._track_arrays(g, min_views=mv)
        b = RF._track_arrays(dict(ref), min_views=mv)
        assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
        assert np.array_equal(a[3], b[3]) and np.array_equal(a[4], b[4])
    assert RF.track_clusters(g) == RF.track_clusters(dict(ref))
    # _obs_of_tracks on a filtered subsequence of the groups' order equals the lists' arrays
    sub = [t for k, t in enumerate(g.tracks.tolist()) if k % 3 == 0]
    ca, ia, ua = RF._obs_of_tracks(g, sub)
    cb, ib, ub = RF._obs_of_tracks(dict(ref), sub)
    assert np.array_equal(ca, cb) and np.array_equal(ia, ib) and np.array_equal(ua, ub)


def test_torch_problem_inputs_equal_the_ceres_rule_loop():
    from precision import ba_torch as BA
    rng = np.random.default_rng(8)
    frames = list(range(50))
    n = 30000
    track = rng.integers(0, 4000, n); frame = rng.choice(frames, n); uv = rng.uniform(0, 400, (n, 2)).astype(np.float32)
    # some tracks observe the same image twice (the Ceres rule keeps the LAST observation there)
    g = RF.group_tracks(track, frame, uv, frames)
    X = {int(t): rng.normal(0, 1, 3) for k, t in enumerate(g.tracks.tolist()) if k % 4}
    a = BA.problem_inputs(g, X, len(frames))
    b = BA._problem_inputs_reference(g, X)
    assert a[0] == b[0] and len(a[0]) > 1000
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x, y)
