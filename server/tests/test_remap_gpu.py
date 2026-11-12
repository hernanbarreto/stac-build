"""tools/genrecon/remap_gpu.py: GenRecon's frame undistortion on a device hands cv2.imwrite the vendor's pixels — the
remap is OpenCV's bit for bit, a frame through the twin is cv2.imread → cv2.remap → the ROI crop exactly — on the cpu
and, when one is visible, on the card."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

_SERVER = Path(__file__).resolve().parents[1]
_TOOLS = _SERVER / "tools" / "genrecon"
sys.path.insert(0, str(_TOOLS))
import remap_gpu as R  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _camera(W, H, dist):
    """The vendor's own lines (inference/get_images.py): the maps and the ROI of a camera with distortion."""
    K = np.array([[0.87 * W, 0.0, 0.503 * W], [0.0, 1.54 * H, 0.493 * H], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(dist, dtype=np.float64)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), alpha=0, newImgSize=(W, H))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (W, H), cv2.CV_16SC2)
    return mapx, mapy, tuple(int(v) for v in roi)


def _vendor(path, mapx, mapy, roi):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    return undist[y: y + h, x: x + w]


def _picture(rng, H, W):
    """A frame with structure and noise (what a JPEG keeps of both)."""
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([127 + 120 * np.sin(xx / 17.0 + c) * np.cos(yy / (11.0 + 3 * c)) for c in range(3)], -1)
    base[H // 3: H // 2, W // 4: W // 2] = (250, 20, 60)
    return np.clip(base + rng.normal(0, 12, (H, W, 3)), 0, 255).astype(np.uint8)


def _twin(img, map1, map2, device):
    t = torch.from_numpy(np.ascontiguousarray(img)).to(device)
    m1 = torch.from_numpy(map1.astype(np.int64)).to(device)
    frac = torch.from_numpy(map2.astype(np.int64) & 1023).to(device)
    return R.remap_u8(t, m1[..., 0].contiguous(), m1[..., 1].contiguous(), R.bilinear_table(device)[frac]).cpu().numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_the_remap_is_opencvs(device):
    rng = np.random.default_rng(23)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    img[:40] = 255                                                      # saturated rows: the rounding at the top
    # the undistortion's own maps
    for dist in ((0.052, -0.0495, -0.0015, -0.00038), (-0.21, 0.07, 0.003, -0.002), (0.0094, -0.014, 0.00045, 0.0012)):
        mapx, mapy, _ = _camera(320, 240, dist)
        assert np.array_equal(_twin(img, mapx, mapy, device), cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR))
    # any map: sources outside the image on every side, an output of another size, whole-pixel sources
    for shape in ((240, 320), (300, 410), (64, 48)):
        mx = rng.uniform(-4.0, 324.0, shape).astype(np.float32)
        my = rng.uniform(-4.0, 244.0, shape).astype(np.float32)
        mx[::7] = np.round(mx[::7])
        my[::5] = np.round(my[::5])
        mx[:3, :3], my[:3, :3] = [[-1.0, -0.5, 0.0]] * 3, [[-1.0], [-0.5], [0.0]]
        mx[-2:, -2:], my[-2:, -2:] = [[318.5, 319.0]] * 2, [[238.5], [239.0]]
        map1, map2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        assert map1.dtype == np.int16 and map2.dtype == np.uint16
        assert np.array_equal(_twin(img, map1, map2, device), cv2.remap(img, map1, map2, cv2.INTER_LINEAR))
    # the identity is the image
    xs, ys = np.meshgrid(np.arange(320, dtype=np.float32), np.arange(240, dtype=np.float32))
    map1, map2 = cv2.convertMaps(xs, ys, cv2.CV_16SC2)
    assert np.array_equal(_twin(img, map1, map2, device), img)


@pytest.mark.parametrize("device", DEVICES)
def test_a_frame_through_the_twin_is_the_vendors(device, tmp_path):
    rng = np.random.default_rng(29)
    for n, (W, H, dist) in enumerate(((1024, 576, (0.052, -0.0495, -0.0015, -0.00038)),
                                      (464, 832, (0.0094, -0.014, 0.00045, 0.0012)),
                                      (322, 242, (-0.12, 0.03, 0.001, -0.001)))):
        mapx, mapy, roi = _camera(W, H, dist)
        pic = _picture(rng, H, W)
        files = []
        p = tmp_path / f"cv_{n}.jpg"                                    # the intake's own writer
        assert cv2.imwrite(str(p), pic, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append(p)
        for sub in (0, 1, 2):                                           # 4:4:4, 4:2:2, 4:2:0
            p = tmp_path / f"pil_{n}_{sub}.jpg"
            Image.fromarray(pic).save(p, quality=90, subsampling=sub)
            files.append(p)
        U = R.Undistorter(mapx, mapy, roi, device)
        for p in files:
            want = _vendor(p, mapx, mapy, roi)
            got = U(p)
            assert got.dtype == np.uint8 and got.flags["C_CONTIGUOUS"] and got.shape == (roi[3], roi[2], 3)
            assert np.array_equal(got, want), p.name
            a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"               # the same pixels: the same file
            assert cv2.imwrite(str(a), want) and cv2.imwrite(str(b), got)
            assert a.read_bytes() == b.read_bytes()


def test_a_frame_with_an_exif_orientation_is_refused(tmp_path):
    rng = np.random.default_rng(31)
    W, H = 320, 240
    mapx, mapy, roi = _camera(W, H, (0.05, -0.04, 0.001, 0.0))
    pic = Image.fromarray(_picture(rng, H, W))
    plain, upright, turned = tmp_path / "plain.jpg", tmp_path / "upright.jpg", tmp_path / "turned.jpg"
    pic.save(plain, quality=92)
    for path, value in ((upright, 1), (turned, 6)):
        exif = Image.Exif()
        exif[0x0112] = value
        pic.save(path, quality=92, exif=exif)
    assert R.exif_orientation(plain.read_bytes()) == 1
    assert R.exif_orientation(upright.read_bytes()) == 1
    assert R.exif_orientation(turned.read_bytes()) == 6
    assert cv2.imread(str(turned), cv2.IMREAD_COLOR).shape[:2] == (W, H)     # why: cv2 turns it, the twin would not
    U = R.Undistorter(mapx, mapy, roi, "cpu")
    for path in (plain, upright):
        assert np.array_equal(U(path), _vendor(path, mapx, mapy, roi))
    with pytest.raises(NotImplementedError, match="EXIF orientation 6.*genrecon.undistort_device: numpy"):
        U(turned)
    with pytest.raises(FileNotFoundError, match="not readable"):
        U(tmp_path / "missing.jpg")


def test_the_device_key_is_typed(monkeypatch):
    assert [R.device_of(d) for d in ("numpy", "cpu")] == ["numpy", "cpu"]
    for bad in ("gpu", "CUDA", "cuda:0", "", None, 0):
        with pytest.raises(RuntimeError, match="genrecon.undistort_device must be one of"):
            R.device_of(bad)
    assert R.device_from_env({}) == "numpy"                               # the vendor tree run by hand: its own lines
    assert R.device_from_env({R.ENV: "cpu"}) == "cpu"
    with pytest.raises(RuntimeError, match="genrecon.undistort_device must be one of"):
        R.device_from_env({R.ENV: "card"})
    mapx, mapy, roi = _camera(64, 48, (0.05, -0.04, 0.0, 0.0))
    with pytest.raises(ValueError, match="numpy is the vendor's own"):
        R.Undistorter(mapx, mapy, roi, "numpy")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="genrecon.undistort_device is cuda and no CUDA device"):
        R.device_from_env({R.ENV: "cuda"})
    with pytest.raises(RuntimeError, match="genrecon.undistort_device is cuda and no CUDA device"):
        R.Undistorter(mapx, mapy, roi, "cuda")


def test_the_maps_and_the_roi_are_checked():
    mapx, mapy, roi = _camera(64, 48, (0.05, -0.04, 0.0, 0.0))
    R.Undistorter(mapx, mapy, roi, "cpu")
    with pytest.raises(ValueError, match="CV_16SC2"):
        R.Undistorter(mapx.astype(np.float32), mapy, roi, "cpu")
    with pytest.raises(ValueError, match="CV_16SC2"):
        R.Undistorter(mapx, mapy[:-1], roi, "cpu")
    for bad in ((0, 0, 65, 48), (0, 0, 0, 10), (-1, 0, 10, 10), (60, 40, 10, 10)):
        with pytest.raises(ValueError, match="ROI"):
            R.Undistorter(mapx, mapy, bad, "cpu")


def test_the_module_loads_in_a_process_without_the_server_tree(tmp_path):
    """GenRecon's own process: tools/genrecon alone on its path, jpeg_gpu found by its place in the tree."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import remap_gpu; J = remap_gpu._jpeg_gpu(); "
            "print(J.__file__); print(J is remap_gpu._jpeg_gpu())")
    env = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", code, str(_TOOLS)], cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == [str(_SERVER / "jpeg_gpu.py"), "True"]


def test_the_vendor_patch_and_the_runner_carry_the_wiring():
    patch = (_TOOLS / "genrecon_stac.patch").read_text()
    assert "import remap_gpu" in patch and "remap_gpu.device_from_env()" in patch and "remap_gpu.Undistorter(" in patch
    # the record's lines are still the vendor's, under the key's numpy
    assert 'if stac_device == "numpy":' in patch
    sh = (_TOOLS / "run_cloud.sh").read_text()
    assert "PYTHONPATH=$T" in sh and sh.index("PYTHONPATH=$T") < sh.index("$GR $RECON")
    # one name for the key on its way: the worker writes it, the stamp records it, GenRecon's process reads it
    import resume as S
    import workers.genrecon_worker as G
    assert R.ENV == G.UNDISTORT_ENV == S.UNDISTORT_ENV and R.DEVICES == G._UNDISTORT_DEVICES
