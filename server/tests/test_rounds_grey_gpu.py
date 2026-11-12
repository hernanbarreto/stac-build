"""segmentation/autoprompt/rounds.py — the rounds' grey painting on the device is the record's image."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.autoprompt import rounds as R  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
SIZES = [1, 2, 3, 5, 7, 8, 13, 16, 31, 64, 100, 127, 232, 288, 416, 464, 512, 576, 720, 832, 1008, 1024, 1080, 1920]


def _scene(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([127 + 100 * np.sin(xx / 17.0 + seed) * np.cos(yy / 23.0),
                    127 + 90 * np.cos(xx / 29.0) * np.sin(yy / 11.0 + seed),
                    (xx * 255 / (w - 1) + yy * 128 / (h - 1)) % 256], -1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _blobs(h: int, w: int, seed: int) -> np.ndarray:
    """A mask of a few rectangles and thin lines (what a nearest resize keeps or drops by its own samples)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), dtype=bool)
    for _ in range(6):
        y0, x0 = int(rng.integers(0, h)), int(rng.integers(0, w))
        m[y0: y0 + int(rng.integers(1, max(2, h // 3))), x0: x0 + int(rng.integers(1, max(2, w // 3)))] = True
    m[:: max(1, h // 7), :] ^= True
    m[:, :: max(1, w // 9)] ^= True
    return m


def _frame(tmp_path, h: int, w: int, seed: int) -> Path:
    p = tmp_path / f"{seed:06d}.jpg"
    Image.fromarray(_scene(h, w, seed)).save(p, format="JPEG", quality=88, subsampling=2)
    return p


def _same_image(a: Image.Image, b: Image.Image) -> bool:
    def jpeg(im):
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    return a.mode == b.mode and a.size == b.size and a.info == b.info and \
        np.array_equal(np.asarray(a), np.asarray(b)) and jpeg(a) == jpeg(b)


def test_the_nearest_samples_are_opencvs():
    for src in SIZES:
        ramp = np.arange(src, dtype=np.int32).reshape(1, src)
        for dst in SIZES:
            want = cv2.resize(ramp, (dst, 1), interpolation=cv2.INTER_NEAREST)[0]
            assert np.array_equal(np.asarray(R.nearest_index(src, dst)), want), (src, dst)
            down = cv2.resize(ramp.T.copy(), (1, dst), interpolation=cv2.INTER_NEAREST)[:, 0]
            assert np.array_equal(np.asarray(R.nearest_index(src, dst)), down), (src, dst)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("src,dst", [((464, 832), (576, 1024)), ((576, 1024), (464, 832)), ((232, 416), (576, 1024)),
                                     ((1080, 1920), (576, 1024)), ((37, 53), (64, 64)), ((5, 7), (3, 2)),
                                     ((3, 2), (9, 11)), ((64, 64), (64, 64))])
def test_a_mask_at_another_grid_is_opencvs(device, src, dst):
    m = _blobs(*src, src[0] + dst[1])
    want = cv2.resize(m.astype(np.uint8), (dst[1], dst[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    got = R.resize_nearest(torch.from_numpy(m).to(device), dst[0], dst[1])
    assert got.dtype == torch.bool and np.array_equal(got.cpu().numpy(), want)


def test_the_record_is_the_frame_opened_and_greyed(tmp_path):
    p = _frame(tmp_path, 120, 160, 1)
    m = _blobs(120, 160, 2)
    for dev in (None, "cpu"):
        assert _same_image(R.greyed_frame(p, m, dev), R.greyed(Image.open(p), m))
    got = np.asarray(R.greyed_frame(p, m, "cpu"))
    assert (got[m] == R.GREY).all() and np.array_equal(got[~m], np.asarray(Image.open(p).convert("RGB"))[~m])


@pytest.mark.skipif(not CUDA, reason="no card")
@pytest.mark.parametrize("size,grid", [((576, 1024), (576, 1024)), ((464, 832), (464, 832)), ((576, 1024), (464, 832)),
                                       ((464, 832), (576, 1024)), ((241, 333), (120, 167)), ((120, 160), (7, 5))])
def test_the_card_gives_the_records_image(tmp_path, size, grid):
    p = _frame(tmp_path, *size, 3)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)          # the stage's own mode: no kernel here may refuse it
    try:
        for mask in (None, np.zeros(grid, dtype=bool), np.ones(grid, dtype=bool), _blobs(*grid, 4), _blobs(*grid, 5)):
            assert _same_image(R.greyed_frame(p, mask, "cuda"), R.greyed(Image.open(p), mask)), (size, grid)
    finally:
        torch.use_deterministic_algorithms(before)


@pytest.mark.skipif(not CUDA, reason="no card")
def test_a_mask_that_is_not_bool_fails_naming_the_frame(tmp_path):
    p = _frame(tmp_path, 64, 64, 6)
    with pytest.raises(TypeError, match=r"bool mask \(got uint8\).*000006.jpg"):
        R.greyed_frame(p, np.ones((64, 64), dtype=np.uint8), "cuda")
