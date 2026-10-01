"""F6's contradicted pixels are repaired from the neighbours, not erased (USER 2026-09-30).

pccr: F6 contradicted 32 % of the pixels of Omega's depth on F5's poses and the corrected cloud came
out eroded. Such a pixel still sees a surface: when >= min_views neighbours put the same surface on
its ray it takes their median depth; when they do not, it leaves as before."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.corrected_cloud import measured_tau, repair_contradicted  # noqa: E402

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])


def _cam(x):
    c2w = np.eye(4); c2w[0, 3] = x                       # cameras side by side, looking at z = 3 m
    return c2w


def _plane_depth():
    return np.full((H, W), 3.0, np.float32)


def _setup():
    order = [0, 1, 2]
    c2w = {0: _cam(-0.2), 1: _cam(0.0), 2: _cam(0.2)}
    depth = {f: _plane_depth() for f in order}
    enter = {f: np.ones((H, W), bool) for f in order}
    bad = {f: np.zeros((H, W), bool) for f in order}
    bad[1][20:40, 30:50] = True                          # contradicted in the middle camera
    depth[1][bad[1]] = 0.0; enter[1][bad[1]] = False     # F6 stores no depth there
    return order, c2w, depth, enter, bad


def test_a_contradicted_pixel_takes_the_depth_its_neighbours_agree_on():
    order, c2w, depth, enter, bad = _setup()
    rep = repair_contradicted(depth, enter, bad, K, c2w, order, (-1, 1), tau=0.02, min_views=2)
    r, c, z = rep[1]
    assert len(r) == int(bad[1].sum())
    assert np.allclose(z, 3.0, atol=1e-3)


def test_one_neighbour_is_not_enough():
    order, c2w, depth, enter, bad = _setup()
    enter[2][:] = False                                  # only camera 0 can speak
    rep = repair_contradicted(depth, enter, bad, K, c2w, order, (-1, 1), tau=0.02, min_views=2)
    assert 1 not in rep or len(rep[1][0]) == 0


def test_neighbours_that_disagree_give_nothing_back():
    order, c2w, depth, enter, bad = _setup()
    depth[2][:] = 4.5                                    # camera 2 sees another surface
    rep = repair_contradicted(depth, enter, bad, K, c2w, order, (-1, 1), tau=0.02, min_views=2)
    assert 1 not in rep or len(rep[1][0]) == 0


def test_the_tolerance_is_measured_from_the_session():
    order, c2w, depth, enter, _ = _setup()
    rng = np.random.default_rng(0)
    for f in order:
        depth[f] = (3.0 * (1 + rng.normal(0, 0.01, (H, W)))).astype(np.float32)
        enter[f][:] = True
    tau = measured_tau(depth, enter, K, c2w, order, (-1, 1), 75.0, stride=1)
    assert 0.005 < tau < 0.04
