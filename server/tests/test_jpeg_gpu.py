"""jpeg_gpu.py — the card's JPEG decode and the device's bilinear resize are PIL's, bit for bit."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jpeg_gpu as J  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
REAL = Path("/workspace/stac-build/server/projects")


def _scene(h: int, w: int, seed: int) -> np.ndarray:
    """A natural-like RGB image: smooth waves, a saturated block (chroma edges), a little noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([127 + 100 * np.sin(xx / 17.0 + seed) * np.cos(yy / 23.0),
                    127 + 90 * np.cos(xx / 29.0) * np.sin(yy / 11.0 + seed),
                    (xx * 255 / max(1, w - 1) + yy * 128 / max(1, h - 1)) % 256], -1)
    img[h // 3: max(h // 3 + 1, h // 2), w // 4: max(w // 4 + 1, w // 2)] = [250, 20, 30]
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _jpeg(arr: np.ndarray, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", **kw)
    return buf.getvalue()


def _pil(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ── the header ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_the_header_is_read_as_libjpeg_reads_it():
    arr = _scene(77, 101, 0)
    for sub, kind in ((0, "444"), (1, "422"), (2, "420")):
        lay = J.read_layout(_jpeg(arr, quality=80, subsampling=sub))
        assert lay == J.Layout(101, 77, kind)
    assert J.Layout(101, 77, "420").chroma == (39, 51) and J.Layout(101, 77, "422").chroma == (77, 51)
    assert J.read_layout(_jpeg(arr[..., 0], quality=80)) == J.Layout(101, 77, "gray")


def test_what_the_card_does_not_decode_fails_naming_the_file_and_the_reason():
    arr = _scene(48, 64, 1)
    good = _jpeg(arr, quality=80)
    with pytest.raises(ValueError, match="a.jpg: a progressive JPEG"):
        J.read_layout(_jpeg(arr, quality=80, progressive=True), "a.jpg")
    buf = io.BytesIO()
    Image.fromarray(arr).convert("CMYK").save(buf, format="JPEG")
    with pytest.raises(ValueError, match="b.jpg: a JPEG of 4 components"):
        J.read_layout(buf.getvalue(), "b.jpg")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    with pytest.raises(ValueError, match="c.png: not a JPEG file"):
        J.read_layout(buf.getvalue(), "c.png")
    with pytest.raises(ValueError, match="d.jpg: the JPEG stream does not end in its EOI marker"):
        J.read_layout(good[:-2], "d.jpg")
    with pytest.raises(ValueError, match="decodes on cuda"):
        J.decode_rgb([good], device="cpu")
    with pytest.raises(ValueError, match="nvJPEG backend 'x'"):
        J.decode_rgb([good], backend="x")


# ── libjpeg's upsampling and colour, with no card: libjpeg's own planes ────────────────────────────────────────
@pytest.mark.parametrize("size", [(96, 128), (77, 101), (16, 16), (576, 1024)])
@pytest.mark.parametrize("quality", [35, 90])
def test_the_upsampling_and_the_colour_are_libjpegs(size, quality):
    """libjpeg hands out a 4:2:0 file's planes itself: the luma of a YCbCr decode, and — at half scale, where it
    scales the luma's DCT and leaves the chroma's whole — the stored chroma. From them the device's arithmetic
    must give libjpeg's RGB."""
    h, w = size
    data = _jpeg(_scene(h, w, quality), quality=quality, subsampling=2)
    full = Image.open(io.BytesIO(data))
    full.draft("YCbCr", full.size)
    y = np.asarray(full)[..., 0].copy()
    half = Image.open(io.BytesIO(data))
    half.draft("YCbCr", (w // 2, h // 2))               # a size the half scale reaches: the draft takes 1/2
    c = np.asarray(half)
    assert c.shape[:2] == J.Layout(w, h, "420").chroma
    rgb = J.planes_to_rgb("420", torch.from_numpy(y), torch.from_numpy(c[..., 1].copy()),
                          torch.from_numpy(c[..., 2].copy()))
    assert np.array_equal(rgb.numpy(), _pil(data))


def test_a_gray_frame_is_repeated_in_the_three_bands():
    y = torch.arange(12, dtype=torch.uint8).reshape(3, 4)
    rgb = J.planes_to_rgb("gray", y)
    assert rgb.shape == (3, 4, 3) and all(torch.equal(rgb[..., i], y) for i in range(3))


# ── Pillow's bilinear resize ───────────────────────────────────────────────────────────────────────────────────
RESIZES = [((576, 1024), (1008, 1008)), ((464, 832), (1008, 1008)), ((1080, 1920), (1008, 1008)),
           ((37, 53), (64, 64)), ((200, 300), (50, 70)), ((100, 100), (100, 37)), ((100, 100), (37, 100)),
           ((64, 64), (64, 64)), ((5, 7), (3, 2)), ((3, 2), (9, 11)), ((480, 640), (1024, 1024)),
           ((333, 517), (1, 1)), ((1, 1), (8, 8)), ((901, 1203), (300, 1100))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("src,dst", RESIZES)
def test_the_resize_is_pillows(device, src, dst):
    rng = np.random.default_rng(src[0] * 7 + dst[1])
    for arr in (rng.integers(0, 256, size=(*src, 3), dtype=np.uint8), _scene(*src, 3)):
        want = np.asarray(Image.fromarray(arr).resize((dst[1], dst[0]), Image.BILINEAR))
        got = J.resize_bilinear(torch.from_numpy(arr).to(device).permute(2, 0, 1), dst[0], dst[1])
        assert got.dtype == torch.uint8 and tuple(got.shape) == (3, *dst)
        assert np.array_equal(got.permute(1, 2, 0).cpu().numpy(), want)


def test_the_resize_takes_8_bit_bands_alone():
    with pytest.raises(TypeError, match="uint8 bands"):
        J.resize_bilinear(torch.zeros((3, 4, 4)), 2, 2)


# ── the card ───────────────────────────────────────────────────────────────────────────────────────────────────
needs_card = pytest.mark.skipif(not CUDA, reason="no card")
SIZES = [(96, 128), (77, 101), (16, 16), (17, 3), (3, 17), (8, 5), (576, 1024), (464, 832), (241, 333)]


@needs_card
@pytest.mark.parametrize("backend", [None, "gpu_hybrid"])
@pytest.mark.parametrize("sub", [0, 1, 2, "gray"])
def test_the_card_decodes_pils_pixels(backend, sub):
    for h, w in SIZES:
        for quality in (30, 75, 95):
            arr = _scene(h, w, quality + h)
            data = _jpeg(arr[..., 0], quality=quality) if sub == "gray" else \
                _jpeg(arr, quality=quality, subsampling=sub)
            got = J.decode_rgb([data], [f"{sub}-{h}x{w}-q{quality}"], backend=backend)[0]
            assert got.dtype == torch.uint8 and got.is_cuda
            assert np.array_equal(got.cpu().numpy(), _pil(data)), (sub, h, w, quality, backend)


@needs_card
def test_the_cards_jpeg_engine_alone_decodes_pils_pixels():
    if J._nvjpeg().backend("hardware") is None:
        pytest.skip("this card has no JPEG engine")
    for sub in (0, 1, 2):
        datas = [_jpeg(_scene(576, 1024, s), quality=85, subsampling=sub) for s in range(5)]
        before = J.decoded_counts()["hardware"]
        got = J.decode_rgb(datas, backend="hardware")
        assert J.decoded_counts()["hardware"] == before + 5
        assert all(np.array_equal(g.cpu().numpy(), _pil(d)) for g, d in zip(got, datas))


@needs_card
def test_a_mixed_batch_keeps_its_order_and_restart_markers_are_read():
    arrs = [_scene(96, 128, 0), _scene(77, 101, 1), _scene(96, 128, 2), _scene(64, 64, 3), _scene(96, 128, 4)]
    datas = [_jpeg(arrs[0], quality=80, subsampling=2), _jpeg(arrs[1], quality=70, subsampling=1),
             _jpeg(arrs[2], quality=90, subsampling=2), _jpeg(arrs[3][..., 0], quality=60),
             _jpeg(arrs[4], quality=80, subsampling=2, restart_marker_blocks=3)]
    got = J.decode_rgb(datas * 9)                       # 45 streams: more than one batch of a layout
    assert len(got) == 45
    for i, g in enumerate(got):
        assert np.array_equal(g.cpu().numpy(), _pil(datas[i % 5])), i


@needs_card
def test_the_files_the_stack_and_the_pil_image(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"{i:06d}.jpg"
        p.write_bytes(_jpeg(_scene(120, 160, i), quality=88, subsampling=2))
        paths.append(p)
    want = np.stack([J.record_rgb(p) for p in paths])
    assert np.array_equal(J.read_rgb_stack(paths, "cuda").cpu().numpy(), want)
    assert np.array_equal(J.read_rgb_stack(paths, "cpu").numpy(), want)
    for dev in (None, "cpu", "cuda"):
        im = J.open_rgb(paths[2], dev)
        assert im.mode == "RGB" and im.size == (160, 120) and np.array_equal(np.asarray(im), want[2])
    odd = tmp_path / "odd.jpg"
    odd.write_bytes(_jpeg(_scene(64, 64, 9), quality=88))
    with pytest.raises(ValueError, match="odd.jpg: a frame of"):
        J.read_rgb_stack(paths + [odd], "cuda")


@needs_card
def test_the_decode_and_the_resize_under_strict_deterministic_torch():
    data = _jpeg(_scene(576, 1024, 5), quality=90, subsampling=2)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        rgb = J.decode_rgb([data])[0]
        small = J.resize_bilinear(rgb.permute(2, 0, 1), 1008, 1008)
    finally:
        torch.use_deterministic_algorithms(before)
    assert np.array_equal(rgb.cpu().numpy(), _pil(data))
    want = np.asarray(Image.fromarray(_pil(data)).resize((1008, 1008), Image.BILINEAR))
    assert np.array_equal(small.permute(1, 2, 0).cpu().numpy(), want)


@needs_card
def test_a_slice_of_the_real_sessions_frames():
    frames = sorted(REAL.glob("*/scans/*/src_*/frames/*.jpg"))
    if not frames:
        pytest.skip("no real session here")
    pick = frames[:: max(1, len(frames) // 60)][:60]
    got = J.read_rgb(pick)
    for p, g in zip(pick, got):
        assert np.array_equal(g.cpu().numpy(), J.record_rgb(p)), str(p)
