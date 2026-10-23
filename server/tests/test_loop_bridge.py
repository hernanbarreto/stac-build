"""Phase 3d (audit 2026-10-23): the loop bridge's pieces — the pair verification by the session
camera, the merge of a bridged track, and the MASt3R image preparation reproducing dust3r's loader
EXACTLY (a match must come back to the native pixel)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import loop_bridge as LB                            # noqa: E402

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
DIST = np.array([-0.05, 0.01, 0.0, 0.0])


def _pair(n=300, seed=0):
    """Correspondences of a rigid pair through the lens: points 2-8 m ahead, the second camera
    0.4 m to the right and turned 6°."""
    import cv2
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), rng.uniform(2, 8, n)]
    R = cv2.Rodrigues(np.array([0.0, np.radians(6.0), 0.0]))[0]
    t = np.array([-0.4, 0.0, 0.0])
    a, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)
    b, _ = cv2.projectPoints(X.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K, DIST)
    return a.reshape(-1, 2) + rng.normal(0, 0.3, (n, 2)), b.reshape(-1, 2) + rng.normal(0, 0.3, (n, 2))


def test_the_camera_verifies_a_rigid_pair_and_refuses_a_scrambled_one():
    a, b = _pair()
    conf = np.ones(len(a))
    keep, why = LB.verify_pair(a, b, conf, K, DIST, ransac_px=1.5, min_inliers=30, max_matches=4000,
                               ransac_prob=0.999, seed=0, min_inlier_ratio=0.25)
    assert why == "" and keep.sum() > 0.9 * len(a)
    rng = np.random.default_rng(1)
    keep2, why2 = LB.verify_pair(a, b[rng.permutation(len(b))], conf, K, DIST, ransac_px=1.5, min_inliers=30,
                                 max_matches=4000, ransac_prob=0.999, seed=0, min_inlier_ratio=0.25)
    assert why2 and not keep2.any()
    keep3, why3 = LB.verify_pair(a[:10], b[:10], conf[:10], K, DIST, ransac_px=1.5, min_inliers=30, max_matches=4000,
                                 ransac_prob=0.999, seed=0)
    assert "fewer than" in why3 and not keep3.any()


def test_merge_bridged_joins_the_two_sides_into_one_track():
    # track 5 queried on frame 10 (side A = {10, 11}), track 9 on frame 40 (side B = {40, 41});
    # the tracker also saw track 5 on frame 40 (a cross-side observation of its own — tb's side decides)
    obs = {"track": [5, 5, 5, 9, 9, 9], "frame": [10, 11, 40, 40, 41, 11], "uv": [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
           "bridged": [False] * 6}
    tq = {"id": [5, 9], "frame": [10, 40], "uv": [[1, 1], [4, 4]]}
    rep = LB.merge_bridged(obs, tq, [(5, 9)], {10, 11}, {40, 41})
    assert rep["bridged_tracks"] == 1 and rep["links"] == 1
    assert obs["track"] == [5, 5, 5, 5] and obs["frame"] == [10, 11, 40, 41]
    assert obs["uv"][2] == [4, 4], "on side B the track queried there decides"
    assert tq["id"] == [5] and tq["frame"] == [10]


def test_mast3r_preparation_reproduces_dust3rs_loader_and_maps_back_to_native():
    """dust3r's load_images (resize long side 512, centre crop to multiples of 16) reproduced bit
    for bit on the tensor and the true_shape; the inverse mapping lands a cropped pixel on the
    native one it came from."""
    import importlib.util
    root = Path(__file__).resolve().parents[2] / "vendor" / "mast3r" / "dust3r"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if importlib.util.find_spec("dust3r") is None or importlib.util.find_spec("torchvision") is None:
        pytest.skip("vendor/mast3r/dust3r (and torchvision) are not on this machine")
    import tempfile
    import PIL.Image
    from dust3r.utils.image import load_images
    rng = np.random.default_rng(0)
    for W1, H1 in ((1280, 720), (720, 1280), (1000, 1000), (640, 480)):
        img = rng.integers(0, 255, (H1, W1, 3), np.uint8)
        t, sh, sc, off = LB.mast3r_prepare(img, 512)
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "a.png")
            PIL.Image.fromarray(img).save(p)
            ref = load_images([p], size=512, verbose=False)[0]
        assert tuple(ref["true_shape"][0]) == tuple(sh[0])
        assert np.allclose(ref["img"].numpy(), t.numpy(), atol=1e-6)
        # a cropped pixel back to native: within one native pixel of the resize's own mapping
        h, w = int(sh[0][0]), int(sh[0][1])
        u = np.array([0.0, w / 2, w - 1.0]); v = np.array([0.0, h / 2, h - 1.0])
        nu = (u + 0.5 + off[0]) * sc[0] - 0.5; nv = (v + 0.5 + off[1]) * sc[1] - 0.5
        assert nu.min() >= -0.5 and nu.max() <= W1 - 0.5 and nv.min() >= -0.5 and nv.max() <= H1 - 0.5
        assert abs(nu[1] - (W1 - 1) / 2) < sc[0] and abs(nv[1] - (H1 - 1) / 2) < sc[1]
