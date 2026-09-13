"""Chunk pose graph: a single closure between the first and the last chunk
distributes along the seams proportionally to the distance walked (chunk 0
fixed), the loop edge is satisfied, and keyframes blend across overlaps
without a step."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction import posegraph                              # noqa: E402
from correction.solve import rot_deg                          # noqa: E402
from tests.synth_correction import make_correction_cfg        # noqa: E402


def _plan(n_kf=120, size=40, overlap=20):
    ranges, a = [], 0
    while True:
        b = min(a + size, n_kf)
        ranges.append([a, b])
        if b >= n_kf:
            break
        a = b - overlap
    return {"version": 1, "n_keyframes": n_kf, "chunk_size": size,
            "overlap": overlap, "chunk_ranges": ranges}


def _straight_walk(n_kf, step=0.25):
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    poses[:, 0, 3] = np.arange(n_kf) * step
    return poses


def test_single_closure_reaches_the_loop_and_blends():
    cfg = make_correction_cfg()
    plan = _plan()
    poses = _straight_walk(plan["n_keyframes"])
    R = posegraph._yaw_R(np.radians(1.5))
    t = np.array([0.30, -0.05, 0.10])
    closures = [{"earlier_kfs": [2, 10], "later_kfs": [108, 118],
                 "R": R.tolist(), "t": t.tolist()}]
    R_kf, t_kf, rep = posegraph.distribute_closures(plan, poses, closures, cfg)
    assert rep["mode"] == "chunk_pose_graph"
    # the loop edge is satisfied: X_last = X_0 ∘ T = T
    assert rep["loops"][0]["residual_mm"] < 1.0
    assert abs(rep["loops"][0]["residual_deg"]) < 0.01
    last = rep["nodes"][-1]
    assert abs(last["yaw_deg"] - 1.5) < 0.01
    assert np.allclose(last["t_m"], t, atol=1e-3)
    # chunk 0 is identity, the correction grows monotonically along the walk
    assert rep["nodes"][0]["t_norm_m"] == 0.0
    norms = [n["t_norm_m"] for n in rep["nodes"]]
    assert all(b >= a for a, b in zip(norms, norms[1:]))
    # per keyframe: identity in chunk 0's exclusive part, T at the end,
    # no step larger than the seam blend
    assert np.allclose(R_kf[5], np.eye(3)) and np.allclose(t_kf[5], 0)
    assert np.allclose(t_kf[-1], t, atol=1e-3)
    assert abs(rot_deg(R_kf[-1]) - 1.5) < 0.01
    steps = np.linalg.norm(np.diff(t_kf, axis=0), axis=1)
    assert steps.max() < np.linalg.norm(t) / 10


def test_two_closures_solved_jointly():
    """A second closure in the middle of the walk pins the curve there."""
    cfg = make_correction_cfg()
    plan = _plan(n_kf=200, size=50, overlap=20)
    poses = _straight_walk(200)
    t_mid = np.array([0.10, 0.0, 0.0])
    t_end = np.array([0.40, 0.0, 0.0])
    closures = [
        {"earlier_kfs": [2, 8], "later_kfs": [95, 100],
         "R": np.eye(3).tolist(), "t": t_mid.tolist()},
        {"earlier_kfs": [2, 8], "later_kfs": [190, 198],
         "R": np.eye(3).tolist(), "t": t_end.tolist()},
    ]
    R_kf, t_kf, rep = posegraph.distribute_closures(plan, poses, closures, cfg)
    for l in rep["loops"]:
        assert l["residual_mm"] < 2.0
    c_mid = posegraph.chunk_of_visit(plan, list(range(95, 101)))
    assert np.allclose(rep["nodes"][c_mid]["t_m"], t_mid, atol=2e-3)
    assert np.allclose(t_kf[-1], t_end, atol=2e-3)


def test_intra_chunk_closure_is_refused():
    cfg = make_correction_cfg()
    plan = _plan()
    poses = _straight_walk(plan["n_keyframes"])
    closures = [{"earlier_kfs": [2, 5], "later_kfs": [12, 15],
                 "R": np.eye(3).tolist(), "t": [0.1, 0, 0]}]
    try:
        posegraph.distribute_closures(plan, poses, closures, cfg)
    except RuntimeError as e:
        assert "inside chunk" in str(e)
    else:
        raise AssertionError("an intra-chunk closure must be refused")
