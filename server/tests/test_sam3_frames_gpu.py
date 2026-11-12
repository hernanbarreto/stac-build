"""segmentation/sam3_frames_gpu.py — SAM3's frames prepared on the device are the vendor loader's, bit for bit."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import sam3_frames_gpu as G  # noqa: E402

CUDA = torch.cuda.is_available()
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
OTHER = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _scene(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.stack([127 + 100 * np.sin(xx / 17.0 + seed) * np.cos(yy / 23.0),
                    127 + 90 * np.cos(xx / 29.0) * np.sin(yy / 11.0 + seed),
                    (xx * 255 / (w - 1) + yy * 128 / (h - 1)) % 256], -1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _stats(mean, std):
    """The loader's own (3, 1, 1) float16 mean and std (io_utils.load_video_frames_from_image_folder)."""
    return (torch.tensor(mean, dtype=torch.float16)[:, None, None],
            torch.tensor(std, dtype=torch.float16)[:, None, None])


def _vendor_lines(pil: Image.Image, image_size: int, img_mean, img_std) -> torch.Tensor:
    """The vendor's frame preparation, operation by operation (io_utils._load_img_as_tensor and
    AsyncImageFrameLoader.__getitem__)."""
    import torchvision.transforms.functional as TF
    img = TF.resize(pil, size=(image_size, image_size))
    img = TF.to_tensor(img)
    img = img.to(dtype=torch.float16)
    img -= img_mean
    img /= img_std
    return img


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype == torch.float16 and a.shape == b.shape and \
        torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))


@pytest.mark.parametrize("stats", [(MEAN, STD), OTHER])
def test_the_table_is_the_vendors_arithmetic_on_every_level(stats):
    img_mean, img_std = _stats(*stats)
    table = G.value_table(img_mean, img_std)
    assert table.shape == (3, 256) and table.dtype == torch.float16
    for band in range(3):
        level = torch.arange(256, dtype=torch.uint8).to(torch.float32).div(255).to(torch.float16)
        level = (level - img_mean[band, 0, 0]) / img_std[band, 0, 0]
        assert _same_bits(table[band], level)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
@pytest.mark.parametrize("size,image_size", [((576, 1024), 1008), ((464, 832), 1008), ((120, 90), 64),
                                             ((64, 64), 64)])
@pytest.mark.parametrize("stats", [(MEAN, STD), OTHER])
def test_a_frame_is_the_vendors_tensor(device, size, image_size, stats):
    img_mean, img_std = _stats(*stats)
    arr = _scene(*size, image_size)
    want = _vendor_lines(Image.fromarray(arr), image_size, img_mean, img_std)
    got = G.frame_tensor(torch.from_numpy(arr).to(device), image_size, G.table_on(device, img_mean, img_std))
    assert str(got.device).startswith(device) and _same_bits(got, want)


def _vendor_loader():
    try:
        import vendor_paths  # noqa: F401 — puts the vendor's sam3 on the path
        from sam3.model import io_utils
    except Exception as e:                            # the worktree carries no vendor tree
        pytest.skip(f"the vendor's SAM3 is not importable here ({type(e).__name__})")
    return io_utils


@pytest.mark.skipif(not CUDA, reason="no card")
def test_the_installed_loader_gives_the_vendors_frames(tmp_path):
    io_utils = _vendor_loader()
    for i in range(6):
        buf = io.BytesIO()
        Image.fromarray(_scene(144, 256, i)).save(buf, format="JPEG", quality=90, subsampling=2)
        (tmp_path / f"{i}.jpg").write_bytes(buf.getvalue())
    cls = io_utils.AsyncImageFrameLoader
    vendor_method = cls.__dict__["__getitem__"]

    def load():
        frames, h, w = io_utils.load_video_frames_from_image_folder(
            image_folder=str(tmp_path), image_size=112, offload_video_to_cpu=False, img_mean=MEAN, img_std=STD,
            async_loading_frames=True)
        frames.thread.join()
        assert frames.exception is None
        return [frames[i] for i in range(len(frames))], h, w

    try:
        assert G.install(None) == "record" and cls.__dict__["__getitem__"] is vendor_method
        record, h0, w0 = load()
        assert G.install("cuda") == "cuda" and G.install("cuda") == "cuda"
        card, h1, w1 = load()
    finally:
        assert G.install("cpu") == "record"
    assert cls.__dict__["__getitem__"] is vendor_method and G._RECORD_ATTR not in cls.__dict__
    assert (h0, w0) == (h1, w1) == (144, 256) and len(record) == len(card) == 6
    for a, b in zip(record, card):
        assert a.is_cuda and b.is_cuda and _same_bits(a, b)
    with pytest.raises(ValueError, match="asked: 'mps'"):
        G.install("mps")
