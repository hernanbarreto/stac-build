"""GenRecon's frame undistortion on a device (USER 2026-11-11: "todo lo que me marcaste que puede pasarse a GPU hay que
hacerlo en GPU … Nada debe hacerse en cpu"; 2026-11-12: "dije ok a todo").

GenRecon (inference/get_images.py, _get_cameras) undistorts every frame of a camera with distortion on the CPU, one
after the other: cv2.imread, cv2.remap with the fixed-point maps of cv2.initUndistortRectifyMap(..., CV_16SC2) and
INTER_LINEAR, the ROI crop, cv2.imwrite. THE RECORD stays those lines (``genrecon.undistort_device: numpy`` — the
vendor's own code, as it always ran). The twin (``cpu | cuda``) hands cv2.imwrite THE SAME PIXELS, so the same files:

* the JPEG decoded by jpeg_gpu on the card (nvJPEG's planes, libjpeg-turbo's integer upsampling and colour
  conversion on the device) — PIL's pixels bit for bit, which are cv2.imread's (both decode with libjpeg-turbo); on
  ``cpu`` the decode is PIL's own;
* :func:`remap_u8` — the remap as OpenCV computes it for 8-bit images: the integer bilinear table of initInterTab2D
  (weights (32-fx)(32-fy)·32 … in units of 1/32768; the (0, 0) entry saturates to 32767 and OpenCV's correction of
  the sum puts the missing 1 on the diagonal weight), (Σ w·p + 2^14) >> 15, a neighbour outside the image 0
  (BORDER_CONSTANT) — computed for the ROI's pixels alone (a remapped pixel depends on its own map entry only);
* the frame comes back as cv2's BGR array and the vendor's own cv2.imwrite writes it.

cv2.imread turns a frame by its EXIF orientation and the decode here does not: a frame that carries one is REFUSED
naming it (the pipeline's frames carry none).

Measured 2026-11-12 in GenRecon's own environment (cv2 5.0.0, torch 2.6.0+cu126) on every keyframe of the three
sessions whose camera has distortion — bufferStop (174 frames of 1024 x 576), observatorio (195) and pccr 2026-08-31
(547 frames of 464 x 832): 0 frames, 0 of 1,282,544,748 values and 0 of the 916 files written differ from the vendor's
lines; cv2.imread's pixels are PIL's on every frame and no frame carries an EXIF block. And through the vendor's own
_get_cameras on the adapter's folders of bufferStop and observatorio: the same files and the same cameras with the key
absent, numpy, cpu and cuda (zaragoza, a camera with no distortion: nothing loaded, nothing written).

NO TIME IS GAINED, declared: in a process as GenRecon's (torchvision imported and the card's context up by its
pipeline, 8 threads) the whole step took 3.11 / 2.91 s on the vendor's lines and 3.34 / 2.85 s on the card for
bufferStop, 3.61 / 3.59 s and 3.55 / 3.26 s for observatorio — a frame's decode and remap cost 6.7 ms on the card
against 8.9–9.3 ms, nvJPEG's two backends start in ~0.3 s, and the encode that stays weighs the most. ``cpu`` is the
tests' device and slower than the record (8.6 s and 12.0 s for the same two sessions).

Stays on the CPU, declared: reading the file, the two maps (once per camera, the vendor's own cv2 calls) and the
JPEG ENCODE (cv2.imwrite: another encoder writes other bytes, and GenRecon's textures are baked from these files).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

DEVICES = ("numpy", "cpu", "cuda")
KEY = "genrecon.undistort_device"
#: how the key reaches GenRecon's own process (workers/genrecon_worker.py writes it; absent = the vendor's own lines)
ENV = "STAC_UNDISTORT_DEVICE"
_SCALE_BITS = 15
_FRACTION_MASK = 1023                                  # INTER_TAB_SIZE2 - 1: fy*32 + fx of a CV_16SC2 map's second part


def device_of(value) -> str:
    """The typed key: one of numpy | cpu | cuda, and cuda only with a card."""
    if value not in DEVICES:
        raise RuntimeError(f"config.yaml: {KEY} must be one of {' | '.join(DEVICES)} (got {value!r})")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.yaml: {KEY} is cuda and no CUDA device is visible — nothing is computed "
                           f"somewhere else")
    return value


def device_from_env(environ=None) -> str:
    """The device GenRecon's process was launched with: ``STAC_UNDISTORT_DEVICE``, written by the worker from
    config.yaml. Absent = ``numpy``, the vendor's own behaviour (the vendor tree run by hand)."""
    environ = os.environ if environ is None else environ
    return device_of(environ[ENV]) if ENV in environ else "numpy"


def _jpeg_gpu():
    """server/jpeg_gpu.py — by its path when the server tree is not on sys.path (GenRecon's own process gets this
    folder alone on its path, never the whole server tree over the vendor's modules)."""
    try:
        import jpeg_gpu
    except ImportError:
        path = Path(__file__).resolve().parents[2] / "jpeg_gpu.py"
        spec = importlib.util.spec_from_file_location("jpeg_gpu", path)
        jpeg_gpu = importlib.util.module_from_spec(spec)
        sys.modules["jpeg_gpu"] = jpeg_gpu
        try:
            spec.loader.exec_module(jpeg_gpu)
        except BaseException:
            del sys.modules["jpeg_gpu"]
            raise
    return jpeg_gpu


def exif_orientation(data: bytes) -> int:
    """The EXIF orientation a JPEG stream declares (1 when it declares none): cv2.imread turns the frame by it."""
    i, n = 2, len(data)
    while i + 4 <= n and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xFF:                             # a fill byte
            i += 1
            continue
        if marker in (0xDA, 0xD9):                     # the scan, the end: no header after it
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:   # markers with no length
            i += 2
            continue
        size = int.from_bytes(data[i + 2:i + 4], "big")
        if marker == 0xE1 and data[i + 4:i + 10] == b"Exif\x00\x00":
            t = data[i + 10:i + 2 + size]
            if len(t) >= 8 and t[:2] in (b"II", b"MM"):
                order = "little" if t[:2] == b"II" else "big"
                at = int.from_bytes(t[4:8], order)
                count = int.from_bytes(t[at:at + 2], order) if at + 2 <= len(t) else 0
                for k in range(count):
                    e = at + 2 + 12 * k
                    if e + 12 > len(t):
                        break
                    if int.from_bytes(t[e:e + 2], order) == 0x0112:
                        return int.from_bytes(t[e + 8:e + 10], order)
            return 1
        i += 2 + size
    return 1


def bilinear_table(device) -> torch.Tensor:
    """(1024, 4) int32: OpenCV's fixed-point bilinear weights per fractional index fy*32 + fx, in the order of the
    neighbours (x, y), (x+1, y), (x, y+1), (x+1, y+1)."""
    fy, fx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    fy, fx = fy.reshape(-1), fx.reshape(-1)
    w = torch.stack([(32 - fx) * (32 - fy), fx * (32 - fy), (32 - fx) * fy, fx * fy], 1) * 32
    w = w.to(torch.int32)
    w[0, 0] = 32767                       # saturate_cast<short>(32768)
    w[0, 3] += 1                          # OpenCV's correction of the sum (diff −1) on the diagonal weight
    return w.to(device)


def remap_u8(img: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``cv2.remap(img, map1, map2, cv2.INTER_LINEAR)`` of a uint8 image (H, W, C) on its device, border constant 0:
    ``sx`` / ``sy`` (Ho, Wo) the maps' integer column / row (map1's two channels), ``w`` (Ho, Wo, 4) the weights
    of the four neighbours (:func:`bilinear_table` at map2's fraction) → (Ho, Wo, C) uint8."""
    H, W, C = (int(s) for s in img.shape)
    src = img.reshape(H * W, C)
    acc = torch.zeros(tuple(sx.shape) + (C,), dtype=torch.int32, device=img.device)
    for k, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        x, y = sx + dx, sy + dy
        inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        v = src[y.clamp(0, H - 1) * W + x.clamp(0, W - 1)].to(torch.int32)                # (Ho, Wo, C)
        acc += v * (w[..., k] * inside).unsqueeze(-1)
    out = (acc + (1 << (_SCALE_BITS - 1))) >> _SCALE_BITS
    return out.clamp(0, 255).to(torch.uint8)


class Undistorter:
    """One camera's undistortion on ``device`` (cpu | cuda): its maps and OpenCV's weights there once, then a frame
    per call — the BGR array ``cv2.imread → cv2.remap(INTER_LINEAR) → [y:y+h, x:x+w]`` gives, bit for bit."""

    def __init__(self, map1: np.ndarray, map2: np.ndarray, roi, device: str):
        if device_of(device) == "numpy":
            raise ValueError(f"{KEY}: numpy is the vendor's own cv2 lines — nothing to build here")
        m1, m2 = np.asarray(map1), np.asarray(map2)
        if m1.dtype != np.int16 or m1.ndim != 3 or m1.shape[2] != 2 or m2.dtype != np.uint16 or m2.shape != m1.shape[:2]:
            raise ValueError(f"the maps are not a CV_16SC2 / CV_16UC1 pair of one size ({m1.dtype} {m1.shape}, "
                             f"{m2.dtype} {m2.shape})")
        x, y, w, h = (int(v) for v in roi)
        if x < 0 or y < 0 or w <= 0 or h <= 0 or y + h > m1.shape[0] or x + w > m1.shape[1]:
            raise ValueError(f"the ROI {(x, y, w, h)} is not inside the maps ({m1.shape[1]} x {m1.shape[0]})")
        self.device = device
        part = torch.from_numpy(np.ascontiguousarray(m1[y:y + h, x:x + w]).astype(np.int64)).to(device)
        self.sx, self.sy = part[..., 0].contiguous(), part[..., 1].contiguous()
        frac = torch.from_numpy(np.ascontiguousarray(m2[y:y + h, x:x + w]).astype(np.int64) & _FRACTION_MASK).to(device)
        self.w = bilinear_table(device)[frac]                                             # (h, w, 4) int32

    def __call__(self, src) -> np.ndarray:
        try:
            data = Path(src).read_bytes()
        except OSError as e:
            raise FileNotFoundError(f"iPhone RGB frame not readable: {src}") from e
        turn = exif_orientation(data)
        if turn != 1:
            raise NotImplementedError(f"{src} declares the EXIF orientation {turn}: cv2.imread turns the frame by it "
                                      f"and the decode on the device does not — {KEY}: numpy undistorts this session")
        J = _jpeg_gpu()
        if self.device == "cuda":
            rgb = J.decode_rgb([data], [str(src)], "cuda")[0]
        else:
            rgb = torch.from_numpy(np.array(J.record_rgb(src)))                           # a writable copy
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"{src} decoded to {tuple(rgb.shape)} — a 3-channel frame is undistorted here")
        out = remap_u8(rgb, self.sx, self.sy, self.w)
        return np.ascontiguousarray(out.flip(-1).cpu().numpy())                           # RGB → cv2's BGR
