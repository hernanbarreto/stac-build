"""F4's per-window observation collector as arrays (precision.tracks.collect_window_observations) against
the per-observation loop it replaced (_collect_window_observations_reference): the same tracks, the same
observations in the same order, the same values (sigma in float64 as the loop's float(sig) * s_mean)."""
import numpy as np

from precision import tracks as TR


def test_collector_equals_the_loop():
    rng = np.random.default_rng(3)
    F, Q = 24, 300
    frames = np.arange(1000, 1000 + 3 * F, 3)
    wit = {frames[5], frames[17], 999999}
    for seed in range(4):
        rng = np.random.default_rng(seed)
        vis = rng.uniform(0, 1, (F, Q)).astype(np.float32)
        good = vis > 0.4
        good[7] = True                                        # the query frame itself
        good[:, :5] = False; good[7, :5] = True                 # queries seen once: no track
        uv = rng.uniform(0, 800, (F, Q, 2)).astype(np.float32)
        sig = rng.uniform(0.2, 3.0, (F, Q)).astype(np.float32)
        a = TR.collect_window_observations(good, uv, vis, sig, frames, 12345, 9, seed % 2 == 0, wit, 1.37)
        b = TR._collect_window_observations_reference(good, uv, vis, sig, frames, 12345, 9, seed % 2 == 0, wit, 1.37)
        for k in ("track", "frame", "window", "loop", "kind", "q_id", "q_col"):
            assert np.array_equal(a[k], b[k]), k
        assert np.array_equal(a["uv"], b["uv"]) and a["uv"].dtype == np.float32
        assert np.array_equal(a["sigma"], b["sigma"]) and np.array_equal(a["vis"], b["vis"])
        assert len(a["track"]) > 0 and (a["kind"] == TR.KIND_WITNESS).any()


def test_the_frames_of_the_observations_are_the_set_of_them():
    """`frames_of` (the report's coverage counts) is `set(obs_frame)`: frames repeated, missing, unordered, none,
    and a negative id (the plain way)."""
    from precision.tracks import frames_of
    rng = np.random.default_rng(31)
    for f in (rng.integers(0, 14000, 200_000), rng.integers(5, 9, 50), np.array([7]), np.zeros(0, np.int64),
              np.array([3, -2, 3, 11]), rng.integers(0, 5000, 3000).astype(np.int32)):
        got = frames_of(f)
        assert got == set(int(x) for x in f) and all(type(x) is int for x in got)
    kf = [3, 9, 11, 40]
    assert len(frames_of(np.array([11, 3, 3, 8])) & set(kf)) == len(set(np.array([11, 3, 3, 8])) & set(kf)) == 2
