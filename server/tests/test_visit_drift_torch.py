"""The mask filter's torch path (correction/visit_drift_torch.py) against the numpy path of record:
the projection through the session camera WITH a lens and a resized mask grid lands every point on the
same pixel, the sorted z-buffer equals np.minimum.at + scipy's minimum filter, the max-pool dilation
equals cv2.dilate, and on the rendered scene of test_certify_mask_filter_camera the two devices delete
the same points with the same ballot — twice, bit for bit. CPU only; the cuda path is the same code."""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# the torch device the equivalence tests run on: cpu (CI) or STAC_TEST_DEVICE=cuda on a box with a card
DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")
if DEV == "cuda" and not torch.cuda.is_available():
    pytest.skip("STAC_TEST_DEVICE=cuda without a visible card", allow_module_level=True)

from correction import visit_drift as VD                       # noqa: E402
from correction import visit_drift_torch as VT                 # noqa: E402
from correction.config import CorrectionConfigError, load_correction_config   # noqa: E402
from tests import test_certify_mask_filter_camera as CAM       # noqa: E402
from tests.synth_correction import make_correction_raw_cfg     # noqa: E402


def _lens_camera():
    from precision.camera import CameraModel, grid_full_frame_resize
    W, H = 640, 480
    omega = grid_full_frame_resize(W, H, W, H, "omega")
    mask = grid_full_frame_resize(W, H, 320, 240, "mask")      # the mask grid: half the native frame
    cam = CameraModel(width=W, height=H, params=(500.0, 498.0, 321.3, 239.2, -0.12, 0.03, 0.001, -0.0007),
                      source="synthetic", camera_epoch=0, omega_grid=omega, mask_grid=mask)
    return VD.SessionProjection(cam, omega, mask, source="test")


def _pose(seed):
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    a = rng.uniform(-0.3, 0.3)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    c2w = np.eye(4)
    c2w[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    c2w[:3, 3] = rng.normal(size=3) * 0.5
    return c2w


def test_the_projection_lands_every_point_on_the_same_pixel_with_a_lens():
    sp = _lens_camera()
    rng = np.random.default_rng(0)
    P = np.column_stack([rng.uniform(-3, 3, 20000), rng.uniform(-2, 2, 20000), rng.uniform(-1, 6, 20000)])
    tp = VT.TorchProjection(sp, torch.device(DEV))
    for seed in range(3):
        pose = _pose(seed)
        ok, r, c, z = sp.to_mask(P, pose, 0.05)
        ok_t, r_t, c_t, z_t = tp.to_mask(torch.as_tensor(P, dtype=torch.float64, device=DEV), pose, 0.05)
        ok_t, r_t, c_t, z_t = (t.cpu() for t in (ok_t, r_t, c_t, z_t))
        assert np.array_equal(ok, ok_t.numpy())
        assert ok.sum() > 1000 and ok.sum() < len(P)              # in frame AND out of frame both exercised
        assert np.array_equal(r, r_t.numpy()) and np.array_equal(c, c_t.numpy())
        assert np.allclose(z, z_t.numpy(), rtol=1e-12, atol=0)


def test_the_sorted_zbuffer_equals_minimum_at_and_the_scipy_filter():
    from scipy import ndimage as ndi
    rng = np.random.default_rng(1)
    H, W, n = 60, 80, 300                                          # sparse: inf regions survive the filter
    r = rng.integers(0, H, n)
    c = rng.integers(0, W, n)
    z = rng.uniform(0.5, 5.0, n)
    r[:100] = r[0]; c[:100] = c[0]                                 # one pixel hit 100 times
    zb = np.full((H, W), np.inf)
    np.minimum.at(zb, (r, c), z)
    zb = ndi.minimum_filter(zb, size=5, mode="nearest")
    zt = VT.zbuffer(torch.as_tensor(r, device=DEV), torch.as_tensor(c, device=DEV), torch.as_tensor(z, device=DEV), (H, W), 5).cpu().numpy()
    assert np.array_equal(zb, zt)                                 # inf == inf included
    assert np.isinf(zt).any() and np.isfinite(zt).any()
    # nothing landed: all inf
    e = torch.zeros(0, dtype=torch.int64, device=DEV)
    assert np.isinf(VT.zbuffer(e, e, torch.zeros(0, dtype=torch.float64, device=DEV), (H, W), 5).cpu().numpy()).all()


def test_the_max_pool_dilation_equals_cv2():
    import cv2
    rng = np.random.default_rng(2)
    m = (rng.uniform(size=(50, 70)) < 0.05).astype(np.uint8)
    m[0, 3] = 1; m[49, 69] = 1                                     # on the border
    for px in (1, 2, 3):
        ref = cv2.dilate(m, np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0
        assert np.array_equal(ref, VT.dilate(torch.as_tensor(m > 0, device=DEV), px).cpu().numpy())
    assert np.array_equal(m > 0, VT.dilate(torch.as_tensor(m > 0, device=DEV), 0).cpu().numpy())


def test_the_edge_distances_equal_cv2_at_the_query_pixels():
    rng = np.random.default_rng(3)
    H, W = 90, 120
    for kind in ("blobs", "border", "all", "none", "one"):
        m = np.zeros((H, W), bool)
        if kind == "blobs":
            m[(rng.uniform(size=(H, W)) < 0.002)] = True
            import cv2
            m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        elif kind == "border":
            m[:40, :] = True; m[60:, 100:] = True
        elif kind == "all":
            m[:] = True
        elif kind == "one":
            m[45, 60] = True
        ref = VD._edge_distance(m.astype(np.uint8))
        r = torch.as_tensor(rng.integers(0, H, 3000), device=DEV); c = torch.as_tensor(rng.integers(0, W, 3000), device=DEV)
        got = VT.edge_distance_at(torch.as_tensor(m, device=DEV), r, c).cpu().numpy()
        want = ref[r.cpu().numpy(), c.cpu().numpy()]
        assert got.dtype == np.float32
        # the same nearest pixel (the squared distance is an integer) and the same side; cv2's float32
        # transform is itself 1 ulp off the correctly rounded root on ~13 % of the pixels (measured)
        assert np.array_equal(np.rint(got.astype(np.float64) ** 2), np.rint(want.astype(np.float64) ** 2)), kind
        assert np.array_equal(np.sign(got), np.sign(want)), kind
        assert np.allclose(got, want, rtol=2e-6, atol=0), kind


def _run(out, xyz, fg, pr, pc, ks, poses, device, extra=None):
    cam = VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=CAM.MIN_DEPTH, log=lambda m: None)
    ex_xyz, ex_ks = (None, None) if extra is None else extra
    vis = VD.Visibility(out, xyz, ks, poses, cam, CAM.DEPTH_TOL_M, min_depth_m=CAM.MIN_DEPTH,
                        extra_xyz=ex_xyz, extra_ks=ex_ks, device=device)
    masklets = VD.masklet_visits(out, log=lambda m: None)
    pm = VD.points_of_masklets(out, fg, pr, pc, log=lambda m: None, mask_pixels=cam.record_to_mask(pr, pc))
    kill, rep = VD.cloud_filter_masklets(
        pm, masklets, ks, xyz, vis, 1, 0.0, 8, CAM.DILATE, occlusion_tol_rel=CAM.TOL_REL,
        min_votes=CAM.MIN_VOTES, min_inside_frac=CAM.MIN_FRAC, min_tri_deg=CAM.MIN_TRI,
        log=lambda m: None)
    return vis, kill, rep


def _same_votes(a, b):
    if a is None or b is None:
        assert a is None and b is None
        return
    assert set(a) == set(b)
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.shape == y.shape, k
        if x.dtype.kind == "f":
            # the recorded edge distances: cv2's float32 transform vs the exact root — 1 ulp apart at
            # most (the decisions — kill, counts, inside/on flags — are compared exactly below)
            assert np.array_equal(np.isnan(x), np.isnan(y)), k
            assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=2e-6, atol=0), k
        else:
            assert np.array_equal(x, y), k


def test_numpy_and_torch_delete_the_same_points_with_the_same_ballot(tmp_path):
    out, xyz, fg, pr, pc, ks, skirt, label, poses = CAM._session(tmp_path)
    # the consumed observations complete the z-buffers: a shifted subsample stands for them
    extra = (xyz[::7] + np.array([0.0, 0.0, 0.01]), ks[::7])
    vis_n, kill_n, rep_n = _run(out, xyz, fg, pr, pc, ks, poses, "numpy", extra)
    vis_t, kill_t, rep_t = _run(out, xyz, fg, pr, pc, ks, poses, DEV, extra)
    assert vis_n.torch is None and vis_t.torch is not None
    assert kill_n.any() and skirt[kill_n].all()                   # the scene's verdict still holds
    assert np.array_equal(kill_n, kill_t)
    assert rep_n.dropped_points == rep_t.dropped_points
    _same_votes(getattr(rep_n, "votes", None), getattr(rep_t, "votes", None))
    for kf in range(len(poses)):
        a, b = vis_n.zbuf(kf), vis_t.zbuf(kf)
        assert np.array_equal(np.isinf(a), np.isinf(b))
        assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-12, atol=0)
    # the torch path twice: bit-identical
    _, kill_2, rep_2 = _run(out, xyz, fg, pr, pc, ks, poses, DEV, extra)
    assert np.array_equal(kill_t, kill_2)
    _same_votes(getattr(rep_t, "votes", None), getattr(rep_2, "votes", None))


def test_the_ballots_agree_vote_for_vote_on_the_rendered_scene(tmp_path):
    """other_mask_votes and rule1_votes called directly on both Visibilities, with a rival mask and
    the record: the counts, and the recorded votes, are the same."""
    out, xyz, fg, pr, pc, ks, skirt, label, poses = CAM._session(tmp_path)
    cam = VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=CAM.MIN_DEPTH, log=lambda m: None)
    vn = VD.Visibility(out, xyz, ks, poses, cam, CAM.DEPTH_TOL_M, min_depth_m=CAM.MIN_DEPTH, device="numpy")
    vt = VD.Visibility(out, xyz, ks, poses, cam, CAM.DEPTH_TOL_M, min_depth_m=CAM.MIN_DEPTH, device=DEV)
    sel = np.flatnonzero(label == 0)[::3]                          # the plate's points (and the skirt)
    P, birth = xyz[sel], ks[sel]
    own = {kf: vn.masks[f"f{kf}_o0"] for kf in range(len(poses))}
    rival = {kf: vn.masks[f"f{kf}_o1"] > 0 for kf in range(len(poses))}
    rec_n: list = []
    rec_t: list = []
    nn = VD.other_mask_votes(vn, P, birth, own, lambda kf: rival[kf], CAM.DILATE, CAM.TOL_REL, CAM.MIN_TRI, record=rec_n)
    nt = VD.other_mask_votes(vt, P, birth, own, lambda kf: rival[kf], CAM.DILATE, CAM.TOL_REL, CAM.MIN_TRI, record=rec_t)
    for a, b in zip(nn, nt):
        assert np.array_equal(a, b)
    assert nn[0].sum() > 0 and nn[2].sum() > 0                    # votes cast, some on the wall
    assert len(rec_n) == len(rec_t)
    for ra, rb in zip(rec_n, rec_t):
        for k in ra:
            x, y = np.asarray(ra[k]), np.asarray(rb[k])
            if x.dtype.kind == "f":
                assert np.allclose(x, y, rtol=2e-6, atol=0, equal_nan=True), k
            else:
                assert np.array_equal(x, y), k
    views = [(2, 4, kf, own[kf]) for kf in range(len(poses))]      # "visit 2-4": kf 0, 1, 5, 6 judge the rest
    s_n, i_n = VD.rule1_votes(vn, P, birth, views, CAM.DILATE)
    s_t, i_t = VD.rule1_votes(vt, P, birth, views, CAM.DILATE)
    assert np.array_equal(s_n, s_t) and np.array_equal(i_n, i_t) and s_n.sum() > 0


def test_the_device_is_a_declared_choice_and_a_missing_card_fails():
    with pytest.raises(CorrectionConfigError, match="visit_drift.device"):
        load_correction_config(make_correction_raw_cfg(**{"visit_drift.device": "tpu"}))
    raw = make_correction_raw_cfg()
    del raw["correction"]["visit_drift"]["device"]
    with pytest.raises(CorrectionConfigError, match="visit_drift.device"):
        load_correction_config(raw)
    assert load_correction_config(make_correction_raw_cfg()).visit_drift.device == "cpu"
    assert VT.resolve_device("numpy") == "numpy" and VT.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="device"):
        VT.resolve_device("gpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="no card"):
            VT.resolve_device("cuda")
