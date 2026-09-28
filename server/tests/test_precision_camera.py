"""F0 — the grid-to-native mapping is the vendor's own preprocessing, exactly;
distortion round-trips; the session camera from omega's per-frame K; the
native-resolution guard."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import camera as C   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VENDOR_OMEGA = REPO / "vendor" / "vggt-omega"


def _vendor_load_fn():
    if str(VENDOR_OMEGA) not in sys.path:
        sys.path.insert(0, str(VENDOR_OMEGA))
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from vggt_omega.utils import load_fn
    return load_fn


def _vendor_crop_box(load_fn, w, h):
    """Exact crop offset the vendor applies, read back from an index image."""
    from PIL import Image
    col = np.tile(np.arange(w, dtype=np.int32), (h, 1))
    row = np.tile(np.arange(h, dtype=np.int32)[:, None], (1, w))
    ic = load_fn._crop_to_supported_aspect_ratio(Image.fromarray(col, mode="I"))
    ir = load_fn._crop_to_supported_aspect_ratio(Image.fromarray(row, mode="I"))
    cw, ch = ic.size
    return int(np.asarray(ic)[0, 0]), int(np.asarray(ir)[0, 0]), cw, ch


@pytest.mark.parametrize("mode", ["balanced", "max_size"])
@pytest.mark.parametrize("res", [512, 1024, 768])
def test_omega_grid_equals_vendor_arithmetic(mode, res):
    load_fn = _vendor_load_fn()
    rng = np.random.default_rng(0)
    sizes = [(464, 832), (832, 464), (1920, 1080), (1080, 1920), (1280, 720), (3000, 900),
             (900, 3000), (640, 640)]
    sizes += [tuple(int(v) for v in rng.integers(120, 2400, size=2)) for _ in range(150)]
    for (w, h) in sizes:
        g = C.omega_grid_for(w, h, mode, res)
        cx, cy, cw, ch = _vendor_crop_box(load_fn, w, h)
        assert (g.crop_x, g.crop_y, g.crop_w, g.crop_h) == (cx, cy, cw, ch), (w, h, mode, res)
        aspect = ch / max(cw, 1)
        if mode == "balanced":
            th, tw = load_fn._balanced_target_shape(aspect, res, 16)
        else:
            th, tw = load_fn._max_size_target_shape(aspect, res, 16)
        assert (g.w, g.h) == (tw, th), (w, h, mode, res)


def test_pccr_grid_is_384x688():
    g = C.omega_grid_for(464, 832, "balanced", 512)
    assert (g.w, g.h, g.crop_x, g.crop_y) == (384, 688, 0, 0)


def _gaussian_blob(img, x0, y0, sigma, amp):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def _centroid(a, x_pred, y_pred, r):
    h, w = a.shape
    x0, x1 = int(max(0, np.floor(x_pred - r))), int(min(w, np.ceil(x_pred + r) + 1))
    y0, y1 = int(max(0, np.floor(y_pred - r))), int(min(h, np.ceil(y_pred + r) + 1))
    win = a[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = win.sum()
    return (xx * win).sum() / m, (yy * win).sum() / m


@pytest.mark.parametrize("mode,res", [("balanced", 512), ("max_size", 1024), ("balanced", 1024)])
@pytest.mark.parametrize("native", [(464, 832), (1000, 300), (2200, 900)])
def test_markers_through_the_real_preprocessing(tmp_path, mode, res, native):
    """Blobs at known native positions pass through the VENDOR function; their
    centroids on the model grid land where native_to_grid predicts (to 0.05
    px, which separates the pixel-centre convention from the naive x/scale
    one by construction), and the crop is respected (a blob outside it is
    gone)."""
    from PIL import Image
    load_fn = _vendor_load_fn()
    w, h = native
    g = C.omega_grid_for(w, h, mode, res)
    rng = np.random.default_rng(1)
    img = np.zeros((h, w), dtype=np.float64)
    sig_g = 2.0                                   # blob σ on the GRID (px)
    win = 5.0 * sig_g                             # centroid window radius: 5σ
    sep = 4.0 * win                               # blobs never share a window
    sig_n = sig_g * max(g.scale_x, g.scale_y)     # → σ in native px
    pts = []
    tries = 0
    while len(pts) < 6 and tries < 2000:
        tries += 1
        x0 = rng.uniform(g.crop_x + 0.12 * g.crop_w, g.crop_x + 0.88 * g.crop_w)
        y0 = rng.uniform(g.crop_y + 0.12 * g.crop_h, g.crop_y + 0.88 * g.crop_h)
        u, v = C.native_to_grid(np.array([x0, y0]), g)
        if all(np.hypot(u - pu, v - pv) > sep for (_, _, pu, pv) in pts):
            pts.append((x0, y0, u, v))
            _gaussian_blob(img, x0, y0, sig_n, 250.0)
    assert len(pts) >= 4
    outside = None
    if g.crop_x > 0:
        outside = (g.crop_x / 2.0, h / 2.0)
        _gaussian_blob(img, *outside, sig_n, 250.0)
    p = tmp_path / "frame.png"
    Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8)).convert("RGB").save(p)
    t = load_fn.load_and_preprocess_images([str(p)], mode=mode, image_resolution=res)
    a = t[0, 0].numpy() * 255.0
    assert a.shape == (g.h, g.w)
    worst = 0.0
    worst_naive = 1e9
    for (x0, y0, u, v) in pts:
        cu, cv = _centroid(a, u, v, win)
        worst = max(worst, abs(cu - u), abs(cv - v))
        # the naive convention (x_native / scale) must be visibly wrong when
        # the scale is far from 1 — otherwise this test could not tell them apart
        nu, nv = (x0 - g.crop_x) / g.scale_x, (y0 - g.crop_y) / g.scale_y
        worst_naive = min(worst_naive, max(abs(cu - nu), abs(cv - nv)))
    assert worst < 0.05, (native, mode, res, worst)
    if max(g.scale_x, g.scale_y) > 1.5:
        assert worst_naive > 0.1, (native, mode, res, worst_naive)
    if outside is not None:
        assert not C.grid_valid_mask(np.array(outside), g)


def test_grid_round_trips_are_exact():
    rng = np.random.default_rng(2)
    for (w, h) in [(464, 832), (3000, 900), (900, 3000), (1920, 1080)]:
        for mode, res in [("balanced", 512), ("max_size", 1024)]:
            g = C.omega_grid_for(w, h, mode, res)
            uv = rng.uniform(-5, max(w, h) + 5, size=(500, 2))
            assert np.abs(C.native_to_grid(C.grid_to_native(uv, g), g) - uv).max() < 1e-9
            assert np.abs(C.grid_to_native(C.native_to_grid(uv, g), g) - uv).max() < 1e-9
            K = np.array([[300.0, 0.0, g.w / 2], [0.0, 305.0, g.h / 2], [0, 0, 1]])
            assert np.abs(C.K_native_to_grid(C.K_grid_to_native(K, g), g) - K).max() < 1e-9
            # a pixel centre of the grid maps to the centre of the native block it covers
            centre = C.grid_to_native(np.array([[0.0, 0.0]]), g)[0]
            assert np.isclose(centre[0], g.crop_x + g.scale_x / 2 - 0.5)
            assert np.isclose(centre[1], g.crop_y + g.scale_y / 2 - 0.5)


def test_padding_offsets_enter_the_map():
    g = C.GridMap(name="t", w=400, h=300, content_w=384, content_h=300, pad_left=8, pad_top=0,
                  crop_x=0, crop_y=0, crop_w=768, crop_h=600, native_w=768, native_h=600)
    assert np.allclose(C.grid_to_native(np.array([8.0, 0.0]), g), [0.5, 0.5])
    assert np.allclose(C.native_to_grid(C.grid_to_native(np.array([[11.3, 7.7]]), g), g), [[11.3, 7.7]])


def _cam(dist=(0.0, 0.0, 0.0, 0.0)):
    g = C.omega_grid_for(464, 832, "balanced", 512)
    return C.CameraModel(width=464, height=832, params=(382.0, 381.0, 231.5, 415.5) + tuple(dist),
                         source="synthetic", camera_epoch=0, omega_grid=g)


def _solver():
    """The production solver settings (reconstruction.precision.camera)."""
    import yaml
    from precision.config import load_precision_config
    with open(Path(__file__).resolve().parents[1] / "config.yaml") as f:
        return C.undistort_solver(load_precision_config(yaml.safe_load(f)).camera)


def test_distortion_round_trip_and_maps():
    pytest.importorskip("cv2")
    cam = _cam((-0.12, 0.03, 0.001, -0.0005))
    rng = np.random.default_rng(3)
    uv = np.stack([rng.uniform(10, 454, 2000), rng.uniform(10, 822, 2000)], axis=1)
    d = C.distort_points(uv, cam)
    solver = _solver()
    assert np.abs(C.undistort_points(d, cam, **solver) - uv).max() < 1e-6
    u = C.undistort_points(uv, cam, **solver)
    assert np.abs(C.distort_points(u, cam) - uv).max() < 1e-6
    assert np.abs(d - uv).max() > 1.0                     # the lens really moves pixels
    m1, m2, Kn = C.undistort_maps(cam)
    assert np.allclose(Kn, cam.K())
    ii = rng.integers(0, 832, 300); jj = rng.integers(0, 464, 300)
    pred = C.distort_points(np.stack([jj, ii], axis=1).astype(np.float64), cam)
    assert np.abs(m1[ii, jj] - pred[:, 0]).max() < 1e-3
    assert np.abs(m2[ii, jj] - pred[:, 1]).max() < 1e-3
    z = _cam()
    assert np.array_equal(C.undistort_points(uv, z, **solver), uv)
    assert np.array_equal(C.distort_points(uv, z), uv)


def test_camera_from_omega_median_native_and_spread():
    g = C.omega_grid_for(464, 832, "balanced", 512)
    rng = np.random.default_rng(4)
    n = 216
    fx = 316.0 * (1 + 0.004 * rng.standard_normal(n))
    fy = 315.0 * (1 + 0.004 * rng.standard_normal(n))
    rows = np.stack([fx, fy, np.full(n, 192.0), np.full(n, 344.0)], axis=1)
    cam = C.camera_from_omega(rows, g, fx_spread_warn_pct=2.0)
    assert cam.width == 464 and cam.height == 832 and cam.source == "omega"
    assert abs(cam.fx - np.median(fx) * g.scale_x) < 1e-9
    assert abs(cam.fy - np.median(fy) * g.scale_y) < 1e-9
    assert np.isclose(cam.cx, (192.0 + 0.5) * g.scale_x - 0.5)
    assert np.isclose(cam.cy, (344.0 + 0.5) * g.scale_y - 0.5)
    assert cam.params[4:] == (0.0, 0.0, 0.0, 0.0)
    r = cam.report
    assert 0.2 < r["fx_spread_pct"] < 0.6 and r["fx_spread_warning"] is False
    assert abs(r["pixel_aspect_fx_over_fy"] - 1.0) < 0.02
    K3 = np.zeros((n, 3, 3)); K3[:, 0, 0] = fx; K3[:, 1, 1] = fy; K3[:, 0, 2] = 192; K3[:, 1, 2] = 344; K3[:, 2, 2] = 1
    cam2 = C.camera_from_omega(K3, g, fx_spread_warn_pct=0.1)
    assert cam2.params == cam.params and cam2.report["fx_spread_warning"] is True
    assert np.allclose(cam.K_omega(), np.array([[np.median(fx), 0, 192.0], [0, np.median(fy), 344.0], [0, 0, 1]]))


def test_camera_json_round_trip_stamps_epochs(tmp_path):
    cam = _cam((0.01, 0.0, 0.0, 0.0))
    p = C.save_camera_json(tmp_path / "output" / "camera.json", cam, geometry_epoch=3)
    d = json.load(open(p))
    assert d["geometry_epoch"] == 3 and d["camera_epoch"] == 0 and d["model"] == "OPENCV"
    assert d["param_names"] == list(C.CAMERA_PARAM_NAMES) and d["provenance"] == "tool_measured"
    assert d["omega_grid"]["w"] == 384 and d["omega_grid"]["scale_x"] == pytest.approx(464 / 384)
    back = C.load_camera_json(p)
    assert back == cam
    with pytest.raises(C.CameraError):
        C.load_camera_json(tmp_path / "nope.json")


def test_rescale_K_and_stray():
    g = C.omega_grid_for(1440, 1920, "balanced", 512)
    K = np.array([[1400.0, 0, 720.0], [0, 1400.0, 960.0], [0, 0, 1]])
    same = C.rescale_K(K, (1440, 1920), (1440, 1920), aspect_tol=0.002)
    assert np.allclose(same, K)
    half = C.rescale_K(K, (1440, 1920), (720, 960), aspect_tol=0.002)
    assert np.isclose(half[0, 0], 700.0) and np.isclose(half[0, 2], (720.0 + 0.5) / 2 - 0.5)
    with pytest.raises(C.CameraError, match="aspect"):
        C.rescale_K(K, (1440, 1920), (720, 900), aspect_tol=0.002)
    cam = C.camera_from_stray(K, (1440, 1920), g, aspect_tol=0.002)
    assert cam.source == "stray" and cam.fx == 1400.0


def test_verify_native_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from PIL import Image
    sess = tmp_path / "s"
    (sess / "frames").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (320, 240), (i, i, i)).save(sess / "frames" / f"{i:06d}.jpg")
    rep = C.verify_native_frames(sess)                   # no video: the frames alone
    assert (rep["native_w"], rep["native_h"], rep["n_frames"]) == (320, 240, 4)
    vw = cv2.VideoWriter(str(sess / "source_video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    if vw.isOpened():
        for _ in range(3):
            vw.write(np.zeros((240, 320, 3), np.uint8))
        vw.release()
        assert C.verify_native_frames(sess)["video_w"] == 320
        Image.new("RGB", (160, 120)).save(sess / "frames" / "000004.jpg")
        with pytest.raises(C.NativeResolutionError, match="mixes"):
            C.verify_native_frames(sess)
        (sess / "frames" / "000004.jpg").unlink()
        for p in (sess / "frames").glob("*.jpg"):
            Image.new("RGB", (160, 120)).save(p)
        with pytest.raises(C.NativeResolutionError, match="rescaled"):
            C.verify_native_frames(sess)
    with pytest.raises(C.CameraError, match="does not match"):
        C.verify_omega_grid(C.omega_grid_for(464, 832, "balanced", 512), (384, 688))
    C.verify_omega_grid(C.omega_grid_for(464, 832, "balanced", 512), (688, 384))


def _fake_session(tmp_path, nw=464, nh=832, mode="balanced", res=512, stray=False, masks=True):
    """A session directory with the artifacts F0 reads: frames, the omega run
    config, intrinsic.txt on the omega grid, one omega depth npz, seg masks."""
    from PIL import Image
    import yaml
    sess = tmp_path / "scan" / "src_default"
    (sess / "frames").mkdir(parents=True)
    out = sess / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (nw, nh), (10, 20, 30)).save(sess / "frames" / f"{i:06d}.jpg")
    g = C.omega_grid_for(nw, nh, mode, res)
    with open(out / "vggt_omega_config.yaml", "w") as f:
        yaml.safe_dump({"Model": {"omega_mode": mode, "omega_resolution": res}}, f)
    rows = np.array([[316.0, 315.0, g.w / 2, g.h / 2], [317.0, 315.5, g.w / 2, g.h / 2],
                     [315.5, 314.8, g.w / 2, g.h / 2]])
    np.savetxt(out / "intrinsic.txt", rows)
    np.savez(out / "omega_run" / "results_output" / "frame_1.npz",
             depth=np.ones((g.h, g.w), np.float32))
    if masks:
        np.savez(out / "seg_masks.npz", scaled_res=np.array([g.h, g.w], np.int32))
    with open(out / "geometry_epoch.json", "w") as f:
        json.dump({"epoch": 2}, f)
    if stray:
        np.savetxt(sess / "camera_matrix.csv", np.array([[400.0, 0, nw / 2], [0, 400.0, nh / 2], [0, 0, 1]]),
                   delimiter=",")
        (sess / "odometry.csv").write_text("timestamp,frame,x,y,z,qx,qy,qz,qw\n")
    return sess, g, rows


class _CamCfg:
    def __init__(self, init_from="auto", warn=2.0, tol=0.002):
        self.init_from, self.fx_spread_warn_pct, self.aspect_tol = init_from, warn, tol


def test_build_session_camera_from_omega(tmp_path):
    sess, g, rows = _fake_session(tmp_path)
    cam = C.build_session_camera(sess, _CamCfg(), log=lambda *a: None)
    d = json.load(open(sess / "output" / "camera.json"))
    assert cam.source == "omega" and d["geometry_epoch"] == 2 and d["camera_epoch"] == 0
    assert (cam.omega_grid.w, cam.omega_grid.h) == (384, 688)
    assert cam.mask_grid == C.grid_full_frame_resize(464, 832, 384, 688, "mask")
    assert cam.mask_grid.scale_x == cam.omega_grid.scale_x       # no crop: same map
    assert np.isclose(cam.fx, np.median(rows[:, 0]) * g.scale_x)
    assert cam.report["omega_preprocessing"] == {"mode": "balanced", "resolution": 512,
                                                 "depth_shape_hw": [688, 384]}
    assert cam.report["frames"]["n_frames"] == 3
    back = C.load_session_camera(sess)
    assert back == cam
    # the mask grid maps masks (native, then full-frame resized) exactly
    uv_native = np.array([[100.0, 700.0]])
    assert np.allclose(C.mask_px_to_native(C.native_to_mask_px(uv_native, cam), cam), uv_native)


def test_build_session_camera_prefers_stray_and_cross_checks(tmp_path):
    sess, g, rows = _fake_session(tmp_path, stray=True)
    cam = C.build_session_camera(sess, _CamCfg("auto"), log=lambda *a: None)
    assert cam.source == "stray" and cam.fx == 400.0
    assert "cross_check_omega" in cam.report and cam.report["omega"]["n_frames"] == 3
    cam2 = C.build_session_camera(sess, _CamCfg("omega"), log=lambda *a: None)
    assert cam2.source == "omega"
    sess3, _, _ = _fake_session(tmp_path / "b")
    with pytest.raises(C.CameraError, match="stray"):
        C.build_session_camera(sess3, _CamCfg("stray"), log=lambda *a: None)


def test_build_session_camera_fails_on_grid_mismatch(tmp_path):
    sess, g, rows = _fake_session(tmp_path, masks=False)
    np.savez(sess / "output" / "omega_run" / "results_output" / "frame_1.npz",
             depth=np.ones((384, 688), np.float32))            # a transposed run
    with pytest.raises(C.CameraError, match="does not match"):
        C.build_session_camera(sess, _CamCfg(), log=lambda *a: None)
    (sess / "output" / "vggt_omega_config.yaml").unlink()
    with pytest.raises(C.CameraError, match="did not record"):
        C.build_session_camera(sess, _CamCfg(), log=lambda *a: None)


def test_mask_grid_differs_from_omega_grid_when_omega_cropped():
    # a 3000x900 frame: omega centre-crops to 1800x900; SAM3 masks are a
    # full-frame resize — the two grids must NOT share a map
    g = C.omega_grid_for(3000, 900, "balanced", 512)
    assert g.crop_x > 0
    m = C.mask_grid_for(3000, 900, (g.h, g.w))
    assert m.crop_x == 0 and m.scale_x != g.scale_x
    uv = np.array([[200.0, 300.0]])
    assert not np.allclose(C.grid_to_native(uv, g), C.grid_to_native(uv, m))


def test_undistort_solver_comes_from_config_and_is_the_one_used():
    """The iteration cap, the stop tolerance and the round-trip resolution are
    reconstruction.precision.camera keys (no literal in precision/camera.py);
    the cap never decides silently: one iteration leaves a residual the
    re-distortion check REFUSES (cv2 reports neither cap nor stop), the
    configured solver converges and passes it; a bad value is refused naming
    the key."""
    pytest.importorskip("cv2")
    solver = _solver()
    assert set(solver) == {"max_iter", "eps_px", "roundtrip_ulps"}
    assert isinstance(solver["max_iter"], int) and solver["max_iter"] >= 1
    assert solver["eps_px"] > 0
    assert isinstance(solver["roundtrip_ulps"], int) and solver["roundtrip_ulps"] >= 1
    cam = _cam((-0.12, 0.03, 0.001, -0.0005))
    rng = np.random.default_rng(5)
    uv = np.stack([rng.uniform(10, 454, 500), rng.uniform(10, 822, 500)], axis=1)
    d = C.distort_points(uv, cam)
    full = np.abs(C.undistort_points(d, cam, **solver) - uv).max()
    assert full < 1e-6
    with pytest.raises(C.CameraError, match="did not converge"):
        C.undistort_points(d, cam, **{**solver, "max_iter": 1})
    K, dist = cam.K(), cam.dist()
    with pytest.raises(C.CameraError, match="did not converge"):
        C.undistort_normalized(d, K, dist, **{**solver, "max_iter": 1})
    xn = C.undistort_normalized(d, K, dist, **solver)
    assert np.abs(xn * [cam.fx, cam.fy] + [cam.cx, cam.cy] - uv).max() < 1e-6
    for bad in ({"max_iter": 0, "eps_px": 1e-6}, {"max_iter": 2.5, "eps_px": 1e-6},
                {"max_iter": True, "eps_px": 1e-6}, {"max_iter": 5, "eps_px": 0.0},
                {"max_iter": 5, "eps_px": 1e-6, "roundtrip_ulps": 0}):
        with pytest.raises(C.CameraError, match="undistort_(max_iter|eps_px|roundtrip_ulps)"):
            C.undistort_points(d, cam, **{"roundtrip_ulps": 16, **bad})
    with pytest.raises(TypeError):
        C.undistort_points(d, cam)                          # no default, no global read


def test_converged_points_pass_at_the_float64_floor_of_large_frames():
    """At 4K the round trip of a CONVERGED point (200 and 1000 iterations give the same
    bits) reads 1.2–1.9e-12 px — at the float64 floor, above eps 1e-12 alone: the check
    accepts it through the ulp term and still refuses a capped solve."""
    pytest.importorskip("cv2")
    solver = _solver()
    W, H = 3840, 2160
    g = C.grid_full_frame_resize(W, H, W, H, "native")
    cam = C.CameraModel(W, H, (0.8 * W, 0.8 * W, (W - 1) / 2, (H - 1) / 2, -0.3, 0.1, 0.0, 0.0),
                        "synthetic", 0, g)
    u, v = np.meshgrid(np.linspace(0, W - 1, 97), np.linspace(0, H - 1, 55))
    uv = np.stack([u.ravel(), v.ravel()], axis=1)
    a = C.undistort_points(uv, cam, **solver)
    b = C.undistort_points(uv, cam, **{**solver, "max_iter": 5 * solver["max_iter"]})
    assert np.array_equal(a, b)                             # converged: more iterations, same bits
    assert np.linalg.norm(C.distort_points(a, cam) - uv, axis=1).max() > solver["eps_px"]
    with pytest.raises(C.CameraError, match="did not converge"):
        C.undistort_points(uv, cam, **{**solver, "max_iter": 5})
