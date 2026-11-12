"""SAM3's video frames prepared ON THE CARD — the vendor's own tensors, bit for bit.

THE RECORD is the vendor's lazy loader (sam3.model.io_utils.AsyncImageFrameLoader.__getitem__): each frame opened
with PIL, resized to the model's square by PIL's bilinear filter (torchvision ``TF.resize``), turned into float32
/ 255 (``TF.to_tensor``), stored as float16 and normalised by the model's mean and std — all on the CPU, a frame
at a time, in a thread beside the model — and only then sent to the card.

Here the frame is decoded on the card (:mod:`jpeg_gpu`: PIL's pixels), resized on the card by Pillow's own
fixed-point passes (``jpeg_gpu.resize_bilinear``) and normalised through a TABLE: a band holds 256 levels, and
the tensor value of each of them is computed ONCE by the vendor's own arithmetic on the CPU (the same torchvision
call and the loader's three operations on an image that holds the 256 levels). A frame's tensor is a gather of
that table — no float is ever computed on the card, so nothing depends on how the card rounds (torch's CUDA
division by a scalar is a multiplication by its reciprocal, and is not the CPU's in the last bit).

:func:`install` swaps the loader's ``__getitem__`` at run time, from our builder (segmentation.sam3_wrapper); no
vendor file is edited, and None / "cpu" puts the vendor's method back.
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image

import jpeg_gpu

_RECORD_ATTR = "_stac_record_getitem"
_TABLES: dict[tuple, torch.Tensor] = {}


def value_table(img_mean: torch.Tensor, img_std: torch.Tensor) -> torch.Tensor:
    """(3, 256) float16 on the CPU: the loader's tensor value of every 8-bit level of every band, computed by the
    vendor's own steps on an image that holds the 256 levels — torchvision's ``to_tensor``, then float16, minus
    the mean, over the std (``img_mean`` / ``img_std``: the loader's (3, 1, 1) float16 tensors)."""
    import torchvision.transforms.functional as TF
    levels = np.repeat(np.arange(256, dtype=np.uint8)[:, None, None], 3, axis=2)      # (256, 1, 3): an RGB image
    img = TF.to_tensor(Image.fromarray(levels))                                       # (3, 256, 1) float32
    img = img.to(dtype=torch.float16)
    img -= img_mean.detach().cpu()
    img /= img_std.detach().cpu()
    return img[:, :, 0].contiguous()


def table_on(device, img_mean: torch.Tensor, img_std: torch.Tensor) -> torch.Tensor:
    """:func:`value_table` on a device, kept for the session's mean and std."""
    key = (str(device), tuple(img_mean.flatten().tolist()), tuple(img_std.flatten().tolist()))
    if key not in _TABLES:
        _TABLES[key] = value_table(img_mean, img_std).to(device)
    return _TABLES[key]


def frame_tensor(rgb: torch.Tensor, image_size: int, table: torch.Tensor) -> torch.Tensor:
    """A decoded frame (H, W, 3) uint8 → the loader's tensor (3, S, S) float16, on the frame's device: Pillow's
    bilinear resize to the model's square, then every level through the table."""
    s = int(image_size)
    x = jpeg_gpu.resize_bilinear(rgb.permute(2, 0, 1), s, s)                          # (3, S, S) uint8
    return table.gather(1, x.reshape(3, -1).to(torch.int64)).reshape(3, s, s)


def _getitem_on_card(self, index):
    """AsyncImageFrameLoader.__getitem__ with the frame prepared on the card (the vendor's control flow kept)."""
    if self.exception is not None:
        raise RuntimeError("Failure in frame loading thread") from self.exception
    img = self.images[index]
    if img is not None:
        return img
    rgb = jpeg_gpu.read_rgb([self.img_paths[index]], "cuda")[0]
    self.video_height, self.video_width = int(rgb.shape[0]), int(rgb.shape[1])
    img = frame_tensor(rgb, self.image_size, table_on(rgb.device, self.img_mean, self.img_std))
    if self.offload_video_to_cpu:
        img = img.cpu()
    self.images[index] = img
    return img


def install(device: Optional[str]) -> str:
    """Make the vendor's lazy frame loader prepare its frames on ``device``: "cuda" = on the card (this module),
    None / "cpu" = the vendor's own method (THE RECORD). Returns what is installed: "cuda" | "record"."""
    if device in (None, "cpu"):
        # only this function replaces the method, and it imports the vendor's module to do it: a module never
        # imported holds the vendor's own
        cls = getattr(sys.modules.get("sam3.model.io_utils"), "AsyncImageFrameLoader", None)
        if cls is not None and _RECORD_ATTR in cls.__dict__:
            cls.__getitem__ = cls.__dict__[_RECORD_ATTR]
            delattr(cls, _RECORD_ATTR)
        return "record"
    if device != "cuda":
        raise ValueError(f"SAM3's frames are prepared on cuda or by the record (asked: {device!r})")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3's frames were asked on cuda and no CUDA device is visible — nothing is prepared "
                           "somewhere else")
    from sam3.model import io_utils
    cls = io_utils.AsyncImageFrameLoader
    if _RECORD_ATTR not in cls.__dict__:
        setattr(cls, _RECORD_ATTR, cls.__dict__["__getitem__"])
    cls.__getitem__ = _getitem_on_card
    return "cuda"
