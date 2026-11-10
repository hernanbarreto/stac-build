"""F5's CPU phases around the solver, vectorised (precision.refine, 2026-11-01), against the loops they
replaced: the leave-one-view-out held-out per keyframe (the batched SVD stacks vs the per-track loop), the
split map by searchsorted, the rolling-shutter diagnostic's pairs — on the synthetic OPENCV scene of the
solver tests."""
import numpy as np
import pytest

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


def test_arrays_only_groups_read_the_same_as_the_lists():
    """F5's groups carry no Python lists (2026-11-10: 66 GB on pccr 2026-08-24, the pod's limit killed F5): every
    reader gives what it gave with the lists — listed() rebuilds them exactly, _obs_of_tracks serves any order,
    the triangulation, the held-out and the torch problem's inputs are unchanged."""
    from precision import ba_torch as BA
    rng = np.random.default_rng(11)
    frames = [100, 103, 106, 109, 112, 115, 118]
    n = 20000
    track = rng.integers(0, 3000, n); frame = rng.choice(frames + [999], n); uv = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    g = RF.group_tracks(track, frame, uv, frames)
    a = RF.group_tracks(track, frame, uv, frames, lists=False)
    assert g.listed and not a.listed and len(a) == 0 and len(g) > 1000
    la = RF.listed(a)
    assert list(la.keys()) == list(g.keys())
    for t in g:
        assert [i for i, _ in la[t]] == [i for i, _ in g[t]]
        assert np.array_equal(np.array([p for _, p in la[t]]), np.array([p for _, p in g[t]], np.float64))
    assert RF.listed(g) is g
    # any order (reversed, a shuffled subset): the same runs as the lists give
    for sub in (g.tracks.tolist()[::-1], rng.permutation(g.tracks)[:500].tolist()):
        ca, ia, ua = RF._obs_of_tracks(a, sub)
        cb, ib, ub = RF._obs_of_tracks(dict(g), sub)
        assert np.array_equal(ca, cb) and np.array_equal(ia, ib) and np.array_equal(ua, ub)
    with pytest.raises(RF.RefineError):
        RF._obs_of_tracks(a, [10 ** 9])
    # the scene's readers: triangulation, held-out, per-track reference, the torch problem
    w2c, pts, groups, X = _scene(n_cam=12, n_pts=400, noise_px=0.4)
    tr_, fr_, uv_ = [], [], []
    for t, obs in groups.items():
        for i, p in obs:
            tr_.append(t); fr_.append(i); uv_.append(p)
    full = RF.group_tracks(np.asarray(tr_), np.asarray(fr_), np.asarray(uv_), list(range(len(w2c))))
    arr = RF.group_tracks(np.asarray(tr_), np.asarray(fr_), np.asarray(uv_), list(range(len(w2c))), lists=False)
    Xf = RF.triangulate_tracks(full, w2c, CAM, SOLVER, _Cfg.min_tri_deg)
    Xa = RF.triangulate_tracks(arr, w2c, CAM, SOLVER, _Cfg.min_tri_deg)
    assert list(Xf) == list(Xa) and all(np.array_equal(Xf[t], Xa[t]) for t in Xf)
    assert RF.heldout_rms(full, w2c, CAM, SOLVER, _Cfg.min_tri_deg) == RF.heldout_rms(arr, w2c, CAM, SOLVER, _Cfg.min_tri_deg)
    assert (RF.heldout_rms_per_track(arr, w2c, CAM, SOLVER, _Cfg.min_tri_deg)
            == RF.heldout_rms_per_track(full, w2c, CAM, SOLVER, _Cfg.min_tri_deg))
    pa, pf = BA.problem_inputs(arr, Xf, len(w2c)), BA.problem_inputs(full, Xf, len(w2c))
    assert pa[0] == pf[0] and all(np.array_equal(x, y) for x, y in zip(pa[1:], pf[1:]))


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


def test_witness_pnp_is_the_same_whatever_the_split():
    pycolmap = pytest.importorskip("pycolmap")
    if not (hasattr(pycolmap, "estimate_and_refine_absolute_pose") and hasattr(pycolmap.Camera, "create_from_model_name")):
        pytest.skip("this pycolmap predates the PnP API F5 runs with (env mapanything)")
    rng = np.random.default_rng(9)
    w2c, pts, groups, X = _scene(n_cam=6, n_pts=300, noise_px=0.3)
    # witnesses: new frames 100..104 observing the landmarks through camera poses near the keyframes
    track, frame, uv = [], [], []
    ids = list(X)
    for f in range(100, 105):
        T = w2c[f % len(w2c)].copy(); T[:3, 3] += rng.normal(0, 0.05, 3)
        P = np.array([X[t] for t in ids])
        q = RF.project(P, T, CAM)
        ok = (q[:, 0] > 0) & (q[:, 0] < WH[0]) & (q[:, 1] > 0) & (q[:, 1] < WH[1])
        for t, p in zip(np.asarray(ids)[ok], q[ok] + rng.normal(0, 0.4, (int(ok.sum()), 2))):
            track.append(int(t)); frame.append(f); uv.append(p)
    track, frame, uv = np.asarray(track), np.asarray(frame), np.asarray(uv)
    r = RF.RungResult("R0", w2c, [list(CAM)], np.zeros(len(w2c), int), 0.0, 0, "CONVERGENCE")

    class Cfg:
        seed = 0
        min_witness_corr = 6
    a = RF.localize_witnesses(r, X, track, frame, uv, list(range(100, 105)), WH, 5.0, Cfg, workers=1)
    b = RF.localize_witnesses(r, X, track, frame, uv, list(range(100, 105)), WH, 5.0, Cfg, workers=3)
    assert list(a.keys()) == list(b.keys()) and len(a) == 5
    for f in a:
        assert a[f] == b[f], f
    assert sum(1 for v in a.values() if v["localized"]) >= 3


def test_undistortion_in_track_chunks_gives_the_same_bits(monkeypatch):
    """The lens' batched undistortion in chunks of whole tracks (2026-11-10: cv2.projectPoints' Jacobian alone was
    32 GB for pccr 2026-08-24's fit observations) equals the one call, bit for bit, verdicts included."""
    from precision.camera import undistort_normalized_tracks
    rng = np.random.default_rng(12)
    counts = rng.integers(2, 9, 3000)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    uv = rng.uniform(-40, 520, (int(counts.sum()), 2))
    K = RF.K_of(CAM)
    dist = np.asarray(CAM[4:8], np.float64) * 200.0          # a strong lens: 1800 of the 3000 tracks fail the round trip
    xa, oa = undistort_normalized_tracks(uv, K, dist, starts, counts, **SOLVER)
    for chunk in (7, 101, 5000):
        monkeypatch.setattr(RF, "_UNDISTORT_CHUNK_OBS", chunk)
        xb, ob = RF._undistort_tracks(uv, K, dist, starts, counts, SOLVER)
        assert np.array_equal(xa, xb, equal_nan=True) and np.array_equal(oa, ob)
    assert 0 < int((~oa).sum()) < len(oa)


def test_the_torch_problem_reads_groups_made_by_refine_run_as_main(monkeypatch):
    """The chain runs F5 as `python -m precision.refine`: its groups are __main__'s TrackGroups, a second class next to
    the precision.refine the solver imports (2026-11-10, pccr 2026-08-24: an isinstance test sent them to the
    per-observation reader, which found no lists — R0 solved an EMPTY problem and "converged"). The solver reads them by
    their arrays; groups without lists never reach the per-observation reader; an empty problem is refused."""
    import importlib.util
    from pathlib import Path
    from precision import ba_torch as BA
    spec = importlib.util.spec_from_file_location("refine_as_main", Path(RF.__file__))
    main_mod = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, "refine_as_main", main_mod)      # what runpy does for __main__
    spec.loader.exec_module(main_mod)
    assert main_mod.TrackGroups is not RF.TrackGroups
    rng = np.random.default_rng(8)
    frames = list(range(50))
    n = 30000
    track = rng.integers(0, 4000, n); frame = rng.choice(frames, n); uv = rng.uniform(0, 400, (n, 2)).astype(np.float32)
    g_main = main_mod.group_tracks(track, frame, uv, frames, lists=False)
    g_ref = RF.group_tracks(track, frame, uv, frames)
    assert not isinstance(g_main, RF.TrackGroups)
    X = {int(t): rng.normal(0, 1, 3) for k, t in enumerate(g_ref.tracks.tolist()) if k % 4}
    a = BA.problem_inputs(g_main, X, len(frames))
    b = BA._problem_inputs_reference(g_ref, X)
    assert a[0] == b[0] and len(a[0]) > 1000
    assert all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))
    with pytest.raises(BA.BATorchError):
        BA._problem_inputs_reference(g_main, X)
    # a rung whose landmarks reach no observation is refused, never solved on the priors alone
    from tests.test_ba_torch import CFG
    w2c = np.tile(np.eye(4), (len(frames), 1, 1))
    with pytest.raises(RF.RefineError):
        BA.run_rung_torch("R0", w2c, CAM, WH, g_main, {10 ** 9: np.zeros(3)}, np.full(len(frames), 0.5), CFG,
                          device="cpu")
