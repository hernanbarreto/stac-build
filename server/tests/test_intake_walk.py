"""I3 → the metric walk (claude_stac.txt §4-F2): windows cover every keyframe and
chain through their shared frames; windows placed each in its own arbitrary frame
chain back into one trajectory whose length is the true walk within the local
error; the anchors come from the window where each frame is most central."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import walk as W                                     # noqa: E402


class G:                                   # the production window layout
    window_frames = 32
    window_overlap_frac = 0.5
    process_res = 504
    model_id = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _trajectory(n=150, step=0.08):
    """A walk that turns: straight, a 90° bend, straight — c2w per keyframe."""
    poses, pos, yaw = [], np.zeros(3), 0.0
    for k in range(n):
        if 50 <= k < 80:
            yaw += 3.0
        R = _rot("y", yaw)
        pos = pos + R @ np.array([0.0, 0.0, step])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        poses.append(T)
    return np.array(poses)


def _w2c(c2w):
    out = np.linalg.inv(c2w)
    return out[:, :3, :]                   # DA3 hands 3x4 world-to-camera


def _write_windows(session, truth, noise_m, seed=0):
    rng = np.random.default_rng(seed)
    n = len(truth)
    plan = W.plan_windows(n, G.window_frames, G.window_overlap_frac)
    wdir = session / "output" / W.WINDOWS_DIRNAME
    wdir.mkdir(parents=True)
    files = [f"{k:06d}.jpg" for k in range(n)]
    json.dump({"windows": [[str(session / "frames" / f) for f in files[a:b]] for a, b in plan],
               "process_res": G.process_res, "model_id": G.model_id},
              open(wdir / "windows.json", "w"))
    for i, (a, b) in enumerate(plan):
        gauge = np.eye(4)                   # every window in its own arbitrary frame
        gauge[:3, :3] = _rot("y", rng.uniform(-180, 180)) @ _rot("x", rng.uniform(-20, 20))
        gauge[:3, 3] = rng.normal(0, 5, 3)
        c2w = np.array([gauge @ T for T in truth[a:b]])
        c2w[:, :3, 3] += rng.normal(0, noise_m, (b - a, 3))
        h, w = 4, 6
        np.savez(wdir / f"window_{i:04d}.npz", frames=np.arange(a, b),
                 depth=np.full((b - a, h, w), float(i), np.float32),
                 conf=np.ones((b - a, h, w), np.float32), extrinsics=_w2c(c2w),
                 intrinsics=np.tile(np.eye(3), (b - a, 1, 1)),
                 scale_factor=np.float64(1.0), is_metric=np.int64(1))
    return plan


def test_windows_cover_every_keyframe_and_share_the_overlap():
    for n in (5, 32, 33, 100, 1329):
        plan = W.plan_windows(n, 32, 0.5)
        covered = set()
        for a, b in plan:
            covered |= set(range(a, b))
            assert b - a == min(32, n)
        assert covered == set(range(n))
        for (a0, b0), (a1, b1) in zip(plan, plan[1:]):
            assert b0 - a1 >= 16                # at least the planned overlap
    with pytest.raises(W.WalkError):
        W.plan_windows(10, 4, 0.2)              # shares fewer than 2 frames


def test_c2w_from_three_by_four_world_to_camera():
    truth = _trajectory(5)
    assert np.allclose(W._to_c2w(_w2c(truth)), truth)
    assert np.allclose(W._to_c2w(np.linalg.inv(truth)), truth)


def test_chained_walk_is_the_true_walk(tmp_path):
    truth = _trajectory()
    true_walk = float(np.sum(np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1)))
    _write_windows(tmp_path, truth, noise_m=0.0)
    doc = W.measure_walk(tmp_path, G, log=lambda *a: None)
    assert abs(doc["walk_length_m"] - true_walk) < 1e-6
    assert doc["local_error_max_m"] < 1e-9
    ch = [c["chainage_m"] for c in doc["chainage"]]
    assert ch[0] == 0.0 and ch == sorted(ch) and len(ch) == len(truth)


def test_noisy_windows_chain_within_their_local_error(tmp_path):
    truth = _trajectory()
    true_walk = float(np.sum(np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1)))
    _write_windows(tmp_path, truth, noise_m=0.005)
    doc = W.measure_walk(tmp_path, G, log=lambda *a: None)
    # 5 mm of centre noise per keyframe inflates a sum of 8 cm steps by the noise's
    # own contribution — bounded by the step count times the noise
    assert abs(doc["walk_length_m"] - true_walk) < len(truth) * 0.005 * 2
    assert 0.0 < doc["local_error_median_m"] < 0.05
    assert all(abs(s["span_ratio"] - 1.0) < 0.05 for s in doc["seams"])


def test_anchors_come_from_the_most_central_window(tmp_path):
    truth = _trajectory(64)
    plan = _write_windows(tmp_path, truth, noise_m=0.0)
    W.measure_walk(tmp_path, G, log=lambda *a: None)
    ro = tmp_path / "output" / "da3_run" / "results_output"
    assert len(list(ro.glob("frame_*.npz"))) == 64
    for f in (0, 20, 40, 63):
        centres = [abs(f - (a + b - 1) / 2.0) if a <= f < b else np.inf for a, b in plan]
        with np.load(ro / f"frame_{f}.npz") as z:
            assert float(z["depth"][0, 0]) == float(int(np.argmin(centres)))
            assert set(z.files) == {"depth", "conf", "intrinsics"}


def test_a_missing_window_names_itself(tmp_path):
    truth = _trajectory(64)
    _write_windows(tmp_path, truth, noise_m=0.0)
    (tmp_path / "output" / W.WINDOWS_DIRNAME / "window_0001.npz").unlink()
    with pytest.raises(W.WalkError, match="window_0001"):
        W.measure_walk(tmp_path, G, log=lambda *a: None)


def test_chunk_plan_from_the_walks_windows_is_reproducible(tmp_path):
    """I4 (USER 2026-10-06): the chunks are planned by CO-VISIBILITY on the same I3 windows the
    walk was chained from (reconstruction.chunk_covis) — the same windows and walk, the same
    plan bit for bit, and the persisted measurement is stamped by that walk."""
    from reconstruction.chunk_covis import plan_session
    truth = _trajectory(150)
    _write_windows(tmp_path, truth, noise_m=0.0)
    W.measure_walk(tmp_path, G, log=lambda *a: None)
    a = plan_session(tmp_path, log=lambda *a: None)
    b = plan_session(tmp_path, log=lambda *a: None)
    assert a["report"]["measurement"] == "measured" and b["report"]["measurement"] == "reused"
    assert a["ranges"] == b["ranges"]
    assert a["ranges"][0][0] == 0 and a["ranges"][-1][1] == 150
    doc = json.loads((tmp_path / "intake" / "covis.json").read_text())
    assert doc["stamp"] == a["report"]["input_stamp"] and doc["n"] == 150
