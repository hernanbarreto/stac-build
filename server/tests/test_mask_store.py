"""The inflated shared mask store of the closures (correction/mask_store.py) against the raw store read mask
by mask (segmentation.fuse_parent.FusedMasks): the same masks, the same emptiness, the same pixel lookups on
numpy and on the device, and the two readers of the closures (masklet_visits, points_of_masklets) giving the
same masklets and the same points through either."""
import os

import numpy as np
import pytest

from correction import mask_store as MS
from correction import visit_drift as VD
from segmentation import fuse_parent
from tests import test_certify_mask_filter_camera as CAM

torch = pytest.importorskip("torch")
DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def test_the_store_holds_what_the_npz_holds(tmp_path):
    out, xyz, fg, pr, pc, ks, _skirt, _label, poses = CAM._session(tmp_path)
    ref = fuse_parent.FusedMasks(out)
    store = MS.MaskStore(out, threads=3).preload()
    keys = [k for k in ref.files if k.startswith("f") and "_o" in k]
    assert keys and store.files == ref.files
    for k in keys:
        a = np.asarray(ref[k]) > 0
        b = store[k]
        assert b.dtype == np.uint8 and np.array_equal(a, b > 0)
        assert store.any(k) == bool(a.any())
    # a meta entry comes back as the raw member
    assert str(np.asarray(store["mask_frame_space"])) == str(np.asarray(ref["mask_frame_space"]))
    # the lookups: random pixels of a frame against all its masks, numpy and the device
    rng = np.random.default_rng(0)
    H, W = store.shape
    r = rng.integers(0, H, 5000)
    c = rng.integers(0, W, 5000)
    frame_keys = [k for k in keys if k.startswith("f3_")]
    want = np.stack([np.asarray(ref[k])[r, c] > 0 for k in frame_keys])
    assert np.array_equal(want, store.lookup(frame_keys, r, c, "numpy"))
    assert np.array_equal(want, store.lookup(frame_keys, r, c, DEV))
    assert store.lookup([], r, c, DEV).shape == (0, 5000) and store.lookup(frame_keys, r[:0], c[:0], DEV).shape == (len(frame_keys), 0)


def test_the_readers_give_the_same_masklets_and_points_through_either_store(tmp_path):
    out, xyz, fg, pr, pc, ks, _skirt, _label, poses = CAM._session(tmp_path)
    ref = fuse_parent.FusedMasks(out)
    cam = VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=CAM.MIN_DEPTH, log=lambda m: None)
    mp = cam.record_to_mask(pr, pc)
    m_ref = VD.masklet_visits(out, log=lambda m: None, masks=ref)
    m_new = VD.masklet_visits(out, log=lambda m: None)                      # the shared store
    assert [(m.oid, m.keyframes.tolist(), m.visits) for m in m_ref] == \
           [(m.oid, m.keyframes.tolist(), m.visits) for m in m_new]
    p_ref = VD.points_of_masklets(out, fg, pr, pc, log=lambda m: None, mask_pixels=mp, masks=ref)
    for dev in ("numpy", DEV):
        VD.use_device(dev)
        try:
            p_new = VD.points_of_masklets(out, fg, pr, pc, log=lambda m: None, mask_pixels=mp)
        finally:
            VD.use_device("cpu")
        assert p_ref.keys() == p_new.keys()
        for o in p_ref:
            assert np.array_equal(p_ref[o], p_new[o]), (dev, o)
    assert sum(len(v) for v in p_ref.values()) > 1000
    # the shared store is one object per session directory
    assert MS.shared_store(out) is MS.shared_store(out)
