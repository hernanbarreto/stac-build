"""GenRecon presence from the chain's depth maps (tools/genrecon/presence.py, USER 2026-11-01): a voxel counts
when TWO keyframes measured depth in it; the points of those voxels join the cloud; one keyframe alone adds
nothing; the unprojection round-trips a known point; deterministic."""
import numpy as np

from tools.genrecon import presence as P


def test_the_unprojection_round_trips_a_known_point():
    K = np.array([[300.0, 0, 231.5], [0, 300.0, 419.5], [0, 0, 1]])
    c2w = np.eye(4); c2w[:3, 3] = [1.0, 2.0, 3.0]
    depth = np.zeros((840, 462), np.float32); valid = np.zeros_like(depth, bool)
    depth[300, 150] = 2.0; valid[300, 150] = True
    pts = P.unproject_depth(depth, valid, K, c2w, stride=1)
    assert pts.shape == (1, 3)
    x = (150 + 0.5 - 231.5) / 300.0 * 2.0; y = (300 + 0.5 - 419.5) / 300.0 * 2.0
    assert np.allclose(pts[0], [1.0 + x, 2.0 + y, 3.0 + 2.0])


def test_two_keyframes_make_a_voxel_present_one_does_not_and_the_census_says_so():
    v = P.GENRECON_VOXEL_M
    a = np.array([[0.5 * v, 0.5 * v, 0.5 * v]] * 5)                  # keyframe 0 and 1 both see voxel A
    b = np.array([[5.5 * v, 0.5 * v, 0.5 * v]] * 5)                  # keyframe 0 alone sees voxel B
    pts = np.concatenate([a, a, b]); kf = np.array([0] * 5 + [1] * 5 + [0] * 5)
    cloud = np.array([[10.5 * v, 0.5 * v, 0.5 * v]])                 # the cloud holds a third voxel only
    add, rec = P.depth_presence(pts, kf, cloud, voxel_m=v, min_views=2, max_per_cell=3)
    assert rec["depth_voxels_confirmed"] == 1 and rec["depth_voxels_new"] == 1 and rec["cloud_voxels"] == 1
    assert rec["points_appended"] == 3 and len(add) == 3 and np.allclose(add, a[0])   # capped per voxel, A only
    add2, rec2 = P.depth_presence(pts, kf, cloud, voxel_m=v, min_views=2, max_per_cell=3)
    assert np.array_equal(add, add2) and rec == rec2                                  # deterministic
    _, rec3 = P.depth_presence(pts, kf, a[:1], voxel_m=v, min_views=2)
    assert rec3["depth_voxels_new"] == 0                                              # the cloud already held A
