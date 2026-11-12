"""JPEG frames decoded ON THE CARD — the record's pixels, bit for bit.

THE RECORD is PIL: ``Image.open(path).convert("RGB")`` — libjpeg-turbo's integer DCT (ISLOW), its "fancy" chroma
upsampling and its 16-bit fixed-point YCbCr → RGB. The card's path is the same arithmetic in two halves:

* nvJPEG (the library torchvision ships, called through ctypes) decodes the entropy stream and the DCT into the
  file's NATIVE planes: Y at full size, Cb and Cr as stored. Its integer IDCT is libjpeg-turbo's ISLOW.
  torchvision's own ``decode_jpeg(device="cuda")`` asks nvJPEG for RGB instead, and nvJPEG's upsampling and colour
  conversion are NOT libjpeg's — its pixels are not PIL's, so that call is not used here.
* libjpeg-turbo's upsampling (jdsample.c: h2v2 / h2v1 "fancy", the edge samples replicated) and its colour
  conversion (jdcolor.c), written in int32 on the device.

The frames go first to the card's JPEG engine (nvJPEG's HARDWARE backend, a batch a call); a batch the engine
refuses, or a card with no engine, is decoded one image at a time with the Huffman stage on the card too
(GPU_HYBRID). Both give the same planes. nvJPEG's batched call of the hybrid backends is never used: it does not.

What is decoded: baseline / extended sequential Huffman JPEG, 8 bits, gray or YCbCr 4:4:4 / 4:2:2 / 4:2:0, whole
(ending in its EOI marker). Anything else — progressive, CMYK, an RGB-coded file, another sampling, a truncated
stream — FAILS naming the file and the reason: nothing is decoded somewhere else in silence.

:func:`resize_bilinear` is Pillow's ``Image.resize(size, BILINEAR)`` for 8-bit bands (libImaging/Resample.c: two
passes of 22-bit fixed-point windows), on the device, bit for bit — what a frame needs after its decode when the
record resizes it with PIL (SAM3's loader).

MEASURED 2026-11-12 (A100, torchvision 0.24.1+cu128's libnvjpeg, PIL 12.0.0) on EVERY frame of the five sessions on
the pod, read-only: 20,279 frames (464x832 and 1024x576, all 4:2:0; 2.1 GB of JPEG), 25,651,937,280 values against
PIL's from the same bytes — 0 differ. All of them decoded by the HARDWARE backend. 0.215 ms a frame on the card
against 2.485 ms of PIL (0.18–0.56 against 2.0–4.9 by session); reading the file is ~4.6 ms either way (the
volume); 1.14 GB of torch at the peak. On 64 frames of each session: the image's info, the VLM's JPEG payload (the
whole frame and a tile) and the texture stack are the record's bytes; SAM3's loader (the vendor's own class
against the card loader, segmentation/sam3_frames_gpu.py): 0 bits differ, 30–36 ms a frame → 4.1–4.5 ms.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import glob
import math
import os
import threading
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch
from PIL import Image

#: the images of one nvJPEG call (and of one conversion on the device: ~30 bytes of int32 work a pixel)
BATCH = 32

_STATUS = {0: "SUCCESS", 1: "NOT_INITIALIZED", 2: "INVALID_PARAMETER", 3: "BAD_JPEG", 4: "JPEG_NOT_SUPPORTED",
           5: "ALLOCATOR_FAILURE", 6: "EXECUTION_FAILED", 7: "ARCH_MISMATCH", 8: "INTERNAL_ERROR",
           9: "IMPLEMENTATION_NOT_SUPPORTED", 10: "INCOMPLETE_BITSTREAM"}
_BACKENDS = {"gpu_hybrid": 2, "hardware": 3}          # nvjpegBackend_t
_OUTPUT_UNCHANGED = 0                                 # nvjpegOutputFormat_t: the file's own planes
#: the record's devices (PIL on the CPU) and the card
_RECORD_DEVICES = (None, "cpu", "numpy")


# ── the header: what libjpeg decides from it ───────────────────────────────────────────────────────────────────
class Layout(NamedTuple):
    width: int
    height: int
    kind: str                                         # "gray" | "444" | "422" | "420"

    @property
    def chroma(self) -> tuple[int, int]:
        """(rows, columns) of a stored chroma plane — libjpeg's downsampled size, rounded up."""
        if self.kind == "420":
            return (self.height + 1) // 2, (self.width + 1) // 2
        if self.kind == "422":
            return self.height, (self.width + 1) // 2
        return self.height, self.width


_SAMPLING = {(1, 1): "444", (2, 1): "422", (2, 2): "420"}


def read_layout(data: bytes, name: str = "<bytes>") -> Layout:
    """The frame of a JPEG stream as libjpeg reads it — its type, its colour space (jdapimin.c
    ``default_decompress_parms``) and its sampling — or a named failure for what this module does not decode."""
    n = len(data)
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"{name}: not a JPEG file (no SOI marker)")
    if data[-2:] != b"\xff\xd9":
        raise ValueError(f"{name}: the JPEG stream does not end in its EOI marker (truncated, or bytes after it) — "
                         f"the card decodes whole streams")
    pos, jfif, adobe, sof = 2, False, None, None
    while True:
        if pos >= n or data[pos] != 0xFF:
            raise ValueError(f"{name}: a JPEG marker was expected at byte {pos}")
        while pos < n and data[pos] == 0xFF:          # the marker's 0xFF and any fill bytes
            pos += 1
        if pos >= n:
            raise ValueError(f"{name}: the JPEG header ends before its scan")
        code = data[pos]
        pos += 1
        if code == 0x01 or 0xD0 <= code <= 0xD8:      # TEM, RSTn, SOI: no payload
            continue
        if code == 0xD9 or pos + 2 > n:
            raise ValueError(f"{name}: the JPEG header ends before its scan")
        length = (data[pos] << 8) | data[pos + 1]
        if length < 2 or pos + length > n:
            raise ValueError(f"{name}: a JPEG segment runs past the end of the file")
        body = data[pos + 2: pos + length]
        if code == 0xE0 and len(body) >= 14 and body[:5] == b"JFIF\x00":
            jfif = True
        elif code == 0xEE and len(body) >= 12 and body[:5] == b"Adobe":
            adobe = body[11]
        elif 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            sof = (code, body)
        elif code == 0xDA:
            break
        pos += length
    if sof is None:
        raise ValueError(f"{name}: no JPEG frame header before the scan")
    code, body = sof
    if code == 0xC2:
        raise ValueError(f"{name}: a progressive JPEG — the card decodes sequential Huffman frames")
    if code not in (0xC0, 0xC1):
        raise ValueError(f"{name}: JPEG frame type 0x{code:02X} (lossless / arithmetic / hierarchical) — the card "
                         f"decodes sequential Huffman frames")
    if len(body) < 6 or len(body) < 6 + 3 * body[5]:
        raise ValueError(f"{name}: a short JPEG frame header")
    if body[0] != 8:
        raise ValueError(f"{name}: {body[0]}-bit JPEG samples — the card decodes 8-bit frames")
    height, width, ncomp = (body[1] << 8) | body[2], (body[3] << 8) | body[4], body[5]
    if height == 0 or width == 0:
        raise ValueError(f"{name}: a JPEG frame of {width} x {height}")
    comps = [(body[6 + 3 * i], body[7 + 3 * i] >> 4, body[7 + 3 * i] & 15) for i in range(ncomp)]
    if ncomp == 1:
        return Layout(width, height, "gray")
    if ncomp != 3:
        raise ValueError(f"{name}: a JPEG of {ncomp} components (CMYK / YCCK) — the card decodes gray and YCbCr")
    ids = tuple(c[0] for c in comps)
    rgb_coded = (not jfif) and ((adobe == 0) if adobe is not None else ids == (82, 71, 66))
    if rgb_coded:
        raise ValueError(f"{name}: an RGB-coded JPEG (no YCbCr transform) — the card decodes gray and YCbCr")
    luma, chroma = comps[0][1:], [c[1:] for c in comps[1:]]
    if chroma != [(1, 1), (1, 1)] or luma not in _SAMPLING:
        raise ValueError(f"{name}: JPEG sampling factors {[c[1:] for c in comps]} — the card decodes 4:4:4, 4:2:2 "
                         f"and 4:2:0")
    return Layout(width, height, _SAMPLING[luma])


# ── libjpeg-turbo's upsampling and colour conversion, in int32 on the device ───────────────────────────────────
def _h2v1_fancy(c: torch.Tensor) -> torch.Tensor:
    """jdsample.c ``h2v1_fancy_upsample``: (…, h, w) → (…, h, 2w), 3/4 of the nearer sample and 1/4 of the further
    one, the row's ends replicated."""
    left = torch.cat([c[..., :1], c[..., :-1]], -1)
    right = torch.cat([c[..., 1:], c[..., -1:]], -1)
    return torch.stack([(3 * c + left + 1) >> 2, (3 * c + right + 2) >> 2], -1).flatten(-2)


def _h2v2_fancy(c: torch.Tensor) -> torch.Tensor:
    """jdsample.c ``h2v2_fancy_upsample``: (…, h, w) → (…, 2h, 2w), 9/16 3/16 3/16 1/16, the rows above the first
    and below the last and the columns beside the ends replicated."""
    up = torch.cat([c[..., :1, :], c[..., :-1, :]], -2)
    down = torch.cat([c[..., 1:, :], c[..., -1:, :]], -2)
    cs = torch.stack([3 * c + up, 3 * c + down], -2).flatten(-3, -2)          # (…, 2h, w): the column sums
    left = torch.cat([cs[..., :1], cs[..., :-1]], -1)
    right = torch.cat([cs[..., 1:], cs[..., -1:]], -1)
    return torch.stack([(3 * cs + left + 8) >> 4, (3 * cs + right + 7) >> 4], -1).flatten(-2)


def _upsample(c: torch.Tensor, kind: str) -> torch.Tensor:
    """A stored chroma plane at the luma's sampling, as libjpeg does it: the fancy filters when the plane is wider
    than two samples (jdsample.c ``jinit_upsampler``), plain replication otherwise."""
    if kind == "444":
        return c
    fancy = c.shape[-1] > 2
    if kind == "422":
        return _h2v1_fancy(c) if fancy else c.repeat_interleave(2, -1)
    if kind == "420":
        return _h2v2_fancy(c) if fancy else c.repeat_interleave(2, -1).repeat_interleave(2, -2)
    raise ValueError(f"no upsampling for a {kind!r} frame")


def _fix(x: float) -> int:
    return int(x * 65536 + 0.5)


def planes_to_rgb(kind: str, y: torch.Tensor, cb: Optional[torch.Tensor] = None,
                  cr: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The file's planes → RGB (…, H, W, 3) uint8, libjpeg-turbo's way: y (…, H, W) uint8, cb / cr the stored
    chroma planes (none for a gray frame, which PIL's ``convert("RGB")`` repeats in the three bands)."""
    if kind == "gray":
        return y.unsqueeze(-1).expand(*y.shape, 3).contiguous()
    h, w = y.shape[-2:]
    y = y.to(torch.int32)
    cb = _upsample(cb.to(torch.int32), kind)[..., :h, :w] - 128
    cr = _upsample(cr.to(torch.int32), kind)[..., :h, :w] - 128
    r = y + ((_fix(1.40200) * cr + 32768) >> 16)
    g = y + ((-_fix(0.34414) * cb - _fix(0.71414) * cr + 32768) >> 16)
    b = y + ((_fix(1.77200) * cb + 32768) >> 16)
    return torch.stack([r, g, b], -1).clamp_(0, 255).to(torch.uint8)


# ── nvJPEG through ctypes ──────────────────────────────────────────────────────────────────────────────────────
class _NvImage(ctypes.Structure):                     # nvjpegImage_t
    _fields_ = [("channel", ctypes.c_void_p * 4), ("pitch", ctypes.c_size_t * 4)]


def _find_library() -> str:
    """libnvjpeg of this environment: torchvision's own copy (its GPU decoder links it), NVIDIA's wheel, the
    system's. A missing library FAILS naming where it was looked for."""
    looked = []
    try:
        import torchvision
        pat = os.path.join(os.path.dirname(torchvision.__file__) + ".libs", "libnvjpeg*.so*")
        looked.append(pat)
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    except ImportError:
        looked.append("torchvision (not installed)")
    for base in {os.path.dirname(os.path.dirname(torch.__file__))}:
        pat = os.path.join(base, "nvidia", "nvjpeg", "lib", "libnvjpeg.so*")
        looked.append(pat)
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    name = ctypes.util.find_library("nvjpeg")
    looked.append("the system's libnvjpeg")
    if name:
        return name
    raise RuntimeError(f"libnvjpeg was not found — looked at {looked}; the card's JPEG decode needs it")


class _NvJpeg:
    """One libnvjpeg, one handle and state a backend, one lock: a state decodes one call at a time."""

    def __init__(self) -> None:
        self.path = _find_library()
        lib = ctypes.CDLL(self.path)
        V, I, S, P = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.POINTER
        lib.nvjpegCreateEx.argtypes = [I, V, V, ctypes.c_uint, P(V)]
        lib.nvjpegJpegStateCreate.argtypes = [V, P(V)]
        lib.nvjpegGetImageInfo.argtypes = [V, V, S, P(I), P(I), P(I), P(I)]
        lib.nvjpegDecode.argtypes = [V, V, V, S, I, P(_NvImage), V]
        lib.nvjpegDecodeBatchedInitialize.argtypes = [V, V, I, I, I]
        lib.nvjpegDecodeBatched.argtypes = [V, V, P(V), P(S), P(_NvImage), V]
        for f in (lib.nvjpegCreateEx, lib.nvjpegJpegStateCreate, lib.nvjpegGetImageInfo, lib.nvjpegDecode,
                  lib.nvjpegDecodeBatchedInitialize, lib.nvjpegDecodeBatched):
            f.restype = I
        self.lib = lib
        self.lock = threading.Lock()
        self.backends: dict[str, Optional[tuple]] = {}
        #: images decoded by each backend in this process (the proof reads it)
        self.decoded = {"hardware": 0, "gpu_hybrid": 0}

    def backend(self, name: str) -> Optional[tuple]:
        """(handle, state) of a backend, created at its first use; None when the card has none (no JPEG engine)."""
        if name not in self.backends:
            h, s = ctypes.c_void_p(), ctypes.c_void_p()
            st = self.lib.nvjpegCreateEx(_BACKENDS[name], None, None, 0, ctypes.byref(h))
            if st == 0:
                st = self.lib.nvjpegJpegStateCreate(h, ctypes.byref(s))
            self.backends[name] = (h, s) if st == 0 else None
        return self.backends[name]

    def planes(self, layout: Layout, datas: Sequence[bytes], names: Sequence[str],
               backend: Optional[str]) -> list[torch.Tensor]:
        """The native planes of same-layout streams on the card: Y (n, H, W) and, for a colour frame, Cb and Cr
        (n, h, w) — uint8. ``backend`` None = the engine, then the hybrid one; a name = that one alone."""
        n, lib = len(datas), self.lib
        sizes = [(layout.height, layout.width)] + ([layout.chroma] * 2 if layout.kind != "gray" else [])
        planes = [torch.empty((n, h, w), dtype=torch.uint8, device="cuda") for h, w in sizes]
        ptrs = (ctypes.c_void_p * n)(*[ctypes.cast(ctypes.c_char_p(d), ctypes.c_void_p) for d in datas])
        lens = (ctypes.c_size_t * n)(*[len(d) for d in datas])
        outs = (_NvImage * n)()
        for k in range(n):
            for i, p in enumerate(planes):
                outs[k].channel[i] = p.data_ptr() + k * p.shape[1] * p.shape[2]
                outs[k].pitch[i] = p.shape[2]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream or None)
        nc, sub = ctypes.c_int(), ctypes.c_int()
        ws, hs = (ctypes.c_int * 4)(), (ctypes.c_int * 4)()
        with self.lock:
            order = [backend] if backend is not None else ["hardware", "gpu_hybrid"]
            opened = [(b, self.backend(b)) for b in order]
            opened = [(b, hs_) for b, hs_ in opened if hs_ is not None]
            if not opened:
                raise RuntimeError(f"nvJPEG opens no {' / '.join(order)} backend on this card ({self.path})")
            handle = opened[0][1][0]
            for k in range(n):                        # nvJPEG's own reading of the header: the planes computed here
                st = lib.nvjpegGetImageInfo(handle, ptrs[k], lens[k], ctypes.byref(nc), ctypes.byref(sub), ws, hs)
                got = [(hs[i], ws[i]) for i in range(max(0, min(nc.value, 4)))]
                if st != 0 or got != sizes:
                    raise RuntimeError(f"{names[k]}: nvJPEG reads the planes {got} (status {st} "
                                       f"{_STATUS.get(st, '?')}) where the header gives {sizes}")
            last = "no backend tried"
            for b, (h, s) in opened:
                if b == "hardware":
                    st = lib.nvjpegDecodeBatchedInitialize(h, s, n, 1, _OUTPUT_UNCHANGED)
                    if st == 0:
                        st = lib.nvjpegDecodeBatched(h, s, ptrs, lens, outs, stream)
                    bad = names[0] if n == 1 else f"a batch of {n} from {names[0]}"
                else:
                    st, bad = 0, ""
                    for k in range(n):
                        st = lib.nvjpegDecode(h, s, ptrs[k], lens[k], _OUTPUT_UNCHANGED, ctypes.byref(outs[k]),
                                              stream)
                        if st != 0:
                            bad = names[k]
                            break
                if st == 0:
                    self.decoded[b] += n
                    return planes
                last = f"{bad}: nvJPEG's {b} backend refuses it — status {st} ({_STATUS.get(st, '?')})"
            raise RuntimeError(last)


_NV: Optional[_NvJpeg] = None
_NV_LOCK = threading.Lock()


def _nvjpeg() -> _NvJpeg:
    global _NV
    with _NV_LOCK:
        if _NV is None:
            if not torch.cuda.is_available():
                raise RuntimeError("the JPEG decode was asked on cuda and no CUDA device is visible — it fails "
                                   "instead of decoding somewhere else")
            torch.zeros(1, device="cuda")             # the CUDA context nvJPEG is created in
            _NV = _NvJpeg()
        return _NV


def decoded_counts() -> dict:
    """Images decoded on the card by each nvJPEG backend in this process."""
    return dict(_NV.decoded) if _NV is not None else {"hardware": 0, "gpu_hybrid": 0}


# ── the decode ─────────────────────────────────────────────────────────────────────────────────────────────────
def record_rgb(path) -> np.ndarray:
    """THE RECORD: PIL's RGB of a frame, (H, W, 3) uint8."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def decode_rgb(datas: Sequence[bytes], names: Optional[Sequence[str]] = None, device: str = "cuda",
               backend: Optional[str] = None) -> list[torch.Tensor]:
    """JPEG streams → their RGB on the card, each (H, W, 3) uint8 and PIL's bit for bit. The tensors of one batch
    of same-size frames share a buffer (clone one to keep it alone)."""
    if device != "cuda":
        raise ValueError(f"decode_rgb decodes on cuda (asked: {device!r}) — the record is record_rgb")
    if backend is not None and backend not in _BACKENDS:
        raise ValueError(f"nvJPEG backend {backend!r} — one of {sorted(_BACKENDS)} or None")
    names = [f"<stream {i}>" for i in range(len(datas))] if names is None else [str(x) for x in names]
    layouts = [read_layout(d, nm) for d, nm in zip(datas, names)]
    nv = _nvjpeg()
    out: list[Optional[torch.Tensor]] = [None] * len(datas)
    groups: dict[Layout, list[int]] = {}
    for i, lay in enumerate(layouts):
        groups.setdefault(lay, []).append(i)
    for lay, idx in groups.items():
        for s in range(0, len(idx), BATCH):
            part = idx[s:s + BATCH]
            planes = nv.planes(lay, [datas[i] for i in part], [names[i] for i in part], backend)
            rgb = planes_to_rgb(lay.kind, *planes)
            for k, i in enumerate(part):
                out[i] = rgb[k]
    return out                                        # type: ignore[return-value]


def read_rgb(paths: Sequence, device: str = "cuda", backend: Optional[str] = None) -> list[torch.Tensor]:
    """Frame files → their RGB on the card (see :func:`decode_rgb`)."""
    return decode_rgb([Path(p).read_bytes() for p in paths], [str(p) for p in paths], device, backend)


def read_rgb_stack(paths: Sequence, device: Optional[str]) -> torch.Tensor:
    """Same-size frames as ONE tensor (n, H, W, 3) uint8: decoded by the record and stacked (device None / "cpu")
    or decoded on the card into it ("cuda"), a batch at a time. Frames of another size FAIL naming the file."""
    paths = list(paths)
    if device in _RECORD_DEVICES:
        return torch.from_numpy(np.stack([record_rgb(p) for p in paths]))
    out: Optional[torch.Tensor] = None
    for s in range(0, len(paths), BATCH):
        for k, rgb in enumerate(read_rgb(paths[s:s + BATCH], device)):
            if out is None:
                out = torch.empty((len(paths), *rgb.shape), dtype=torch.uint8, device=rgb.device)
            if rgb.shape != out.shape[1:]:
                raise ValueError(f"{paths[s + k]}: a frame of {tuple(rgb.shape[:2])} among frames of "
                                 f"{tuple(out.shape[1:3])}")
            out[s + k] = rgb
    if out is None:
        raise ValueError("no frame to decode")
    return out


def open_rgb(path, device: Optional[str]) -> Image.Image:
    """A frame as the PIL image the record opens (``Image.open(path).convert("RGB")``): decoded by PIL (device
    None / "cpu") or on the card ("cuda") and brought back — the same pixels AND the same ``info`` (the file's
    header, read by PIL without decoding: a JPEG comment in it is written again by PIL's JPEG save, so an image
    without it would be encoded into other bytes)."""
    if device in _RECORD_DEVICES:
        return Image.open(path).convert("RGB")
    with Image.open(path) as im:
        info = dict(im.info)
    out = Image.fromarray(read_rgb([path], device)[0].cpu().numpy())
    out.info = info
    return out


# ── Pillow's bilinear resize ───────────────────────────────────────────────────────────────────────────────────
_PRECISION_BITS = 32 - 8 - 2                          # Resample.c: 8-bit bands in 32-bit sums


@functools.lru_cache(maxsize=64)
def _pil_bilinear_coeffs(in_size: int, out_size: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Pillow's ``precompute_coeffs`` and ``normalize_coeffs_8bpc`` (libImaging/Resample.c) for the bilinear filter
    over a whole axis: for every output sample the first input sample of its window and the window's fixed-point
    weights, padded with zeros to one length. Python's floats are C's doubles and the operations are Pillow's, in
    its order."""
    scale = filterscale = float(in_size) / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    ss = 1.0 / filterscale
    firsts, rows = [], []
    for xx in range(out_size):
        center = 0.0 + (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        k, ww = [], 0.0
        for x in range(xmax):
            w = (x + xmin - center + 0.5) * ss
            if w < 0.0:
                w = -w
            w = 1.0 - w if w < 1.0 else 0.0
            k.append(w)
            ww += w
        if ww != 0.0:
            k = [v / ww for v in k]
        one = 1 << _PRECISION_BITS
        fixed = [int(-0.5 + v * one) if v < 0 else int(0.5 + v * one) for v in k]
        rows.append(tuple(fixed + [0] * (ksize - xmax)))
        firsts.append(xmin)
    return tuple(firsts), tuple(rows)


_WINDOWS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _resample_last(x: torch.Tensor, out_size: int) -> torch.Tensor:
    """One Pillow pass along the last axis: (…, n) uint8 → (…, out_size) uint8."""
    in_size = x.shape[-1]
    key = (in_size, out_size, str(x.device))
    if key not in _WINDOWS:
        firsts, rows = _pil_bilinear_coeffs(in_size, out_size)
        kk = torch.tensor(rows, dtype=torch.int32, device=x.device)
        idx = torch.tensor(firsts, dtype=torch.int64, device=x.device)[:, None] \
            + torch.arange(kk.shape[1], dtype=torch.int64, device=x.device)[None, :]
        _WINDOWS[key] = (idx.clamp_(max=in_size - 1), kk)      # beyond its window a weight is zero
    idx, kk = _WINDOWS[key]
    acc = (x.to(torch.int32)[..., idx] * kk).sum(-1, dtype=torch.int64) + (1 << (_PRECISION_BITS - 1))
    return (acc >> _PRECISION_BITS).clamp_(0, 255).to(torch.uint8)


def resize_bilinear(img: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """(…, H, W) uint8 → (…, out_h, out_w) uint8: Pillow's ``Image.resize((out_w, out_h), BILINEAR)`` of every
    8-bit band, bit for bit — the horizontal pass, rounded to 8 bits, then the vertical one; an axis that keeps its
    size is not filtered."""
    if img.dtype != torch.uint8:
        raise TypeError(f"resize_bilinear takes uint8 bands (got {img.dtype})")
    h, w = img.shape[-2:]
    x = img
    if out_w != w:
        x = _resample_last(x, int(out_w))
    if out_h != h:
        x = _resample_last(x.transpose(-1, -2), int(out_h)).transpose(-1, -2)
    return x.contiguous()
