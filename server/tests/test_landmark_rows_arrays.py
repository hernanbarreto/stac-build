"""F6's landmark rows per keyframe as arrays (precision.depth_on_f5.landmark_rows_by_keyframe) against the
per-observation loop they replace: the same rows per keyframe, in the same order, z to 1e-12."""
import numpy as np

from precision import depth_on_f5 as D6
from precision import refine as RF
from tests.test_ba_torch import _scene


def test_landmark_rows_equal_the_loop():
    w2c, pts, groups, X = _scene(n_cam=8, n_pts=300, noise_px=0.3)
    # a third of the tracks are not triangulated; a few single-observation tracks; one landmark behind a camera
    Xs = {t: x for k, (t, x) in enumerate(X.items()) if k % 3}
    some = list(groups)[:3]
    for t in some:
        groups[t] = groups[t][:1]
    t_back = list(Xs)[0]
    Xs[t_back] = np.array([0.0, 0.0, -50.0])
    a = D6.landmark_rows_by_keyframe(groups, Xs, w2c, len(w2c))
    b = D6._landmark_rows_by_keyframe_reference(groups, Xs, w2c, len(w2c))
    assert a.keys() == b.keys()
    n = 0
    for i in a:
        assert a[i].shape == b[i].shape, i
        if len(a[i]):
            assert np.array_equal(a[i][:, :2], b[i][:, :2])
            assert np.allclose(a[i][:, 2], b[i][:, 2], rtol=1e-12, atol=0)
            n += len(a[i])
    assert n > 100
    assert RF  # the groups are refine's shape
