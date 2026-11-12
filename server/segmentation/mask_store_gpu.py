"""The SAM3 mask store inflated ON THE DEVICE (USER 2026-11-11: "todo lo que me marcaste que puede pasarse a
GPU hay que hacerlo en GPU ... Nada debe hacerse en cpu").

``seg_masks.npz`` is a zip of ``.npy`` members, each a raw DEFLATE stream (``np.savez_compressed``). Every
reader inflated them with zlib on the CPU, one member at a time: on pccr 3108's projection the audit's
mask reads were ~96 s of zlib, and pccr 2026-08-24 holds 290,418 masks. Here the file is read once, its
bytes uploaded, and the members of a request inflated together by nvCOMP's Deflate decoder on the card
(``nvidia-nvcomp-cu12`` 6.0.0.12, installed 2026-11-11 — NVIDIA's licence, the family of CUDA's own
libraries). The bytes are zlib's (an inflated stream is a function of the stream): the masks are the
CPU readers' masks bit for bit (tests/test_mask_store_gpu.py, on a real store). On a ``cpu`` device the
members are inflated by zlib — the record.

``dilate_cross_t`` / ``erode_like_cv2_t``: the readers' mask morphology on the device.
"""
from __future__ import annotations

import ast
import io
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

#: members inflated per nvCOMP batch (bounded device memory: a member is one mask of the store's grid)
BATCH = 512


def _npy_header(buf: bytes):
    """(data offset, dtype, shape, fortran) of an ``.npy`` member's bytes."""
    if buf[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy member")
    major = buf[6]
    if major == 1:
        hlen = struct.unpack("<H", buf[8:10])[0]
        start = 10
    else:
        hlen = struct.unpack("<I", buf[8:12])[0]
        start = 12
    d = ast.literal_eval(buf[start:start + hlen].decode("latin1"))
    return start + hlen, np.dtype(d["descr"]), tuple(d["shape"]), bool(d["fortran_order"])


class DeviceMaskStore:
    """``np.load(seg_masks.npz)`` for masks, on ``device``: ``files``, ``in``, and ``stack(keys)`` → the
    masks as one boolean tensor (k, H, W) on the device (nonzero = True, as every reader thresholds)."""

    def __init__(self, path, device, raw: bytes = None):
        """``raw``: the store's bytes when the caller already holds them (an npz opened from memory) —
        the file at ``path`` is then not read."""
        self.path = Path(path)
        self.device = torch.device(device)
        self.raw = self.path.read_bytes() if raw is None else raw
        zf = zipfile.ZipFile(io.BytesIO(self.raw))
        self._info: Dict[str, zipfile.ZipInfo] = {}
        for i in zf.infolist():
            self._info[i.filename[:-4] if i.filename.endswith(".npy") else i.filename] = i
        self.files = list(self._info)
        self._keys = set(self.files)
        self._raw_t = None
        self._codec = None
        self._head: Dict[str, tuple] = {}

    def __contains__(self, key) -> bool:
        return key in self._keys

    def _span(self, key: str):
        i = self._info[key]
        n_name, n_extra = struct.unpack("<HH", self.raw[i.header_offset + 26: i.header_offset + 30])
        s = i.header_offset + 30 + n_name + n_extra
        return s, s + i.compress_size, i

    def _inflate_cpu(self, key: str) -> bytes:
        s, e, i = self._span(key)
        b = self.raw[s:e]
        return zlib.decompress(b, -15) if i.compress_type == zipfile.ZIP_DEFLATED else b

    def _inflate_device(self, keys: Sequence[str]) -> List[torch.Tensor]:
        """Each member's .npy bytes as a uint8 tensor on the device (nvCOMP, one batch)."""
        from nvidia import nvcomp
        if self._raw_t is None:
            self._raw_t = torch.frombuffer(bytearray(self.raw), dtype=torch.uint8).to(self.device)
            self._codec = nvcomp.Codec(algorithm="Deflate", bitstream_kind=nvcomp.BitstreamKind.RAW,
                                       device_id=self.device.index or 0)
        spans = [self._span(k) for k in keys]
        out: List[torch.Tensor] = [None] * len(keys)                      # type: ignore[list-item]
        defl = [j for j, (_s, _e, i) in enumerate(spans) if i.compress_type == zipfile.ZIP_DEFLATED]
        for j, (s, e, i) in enumerate(spans):
            if i.compress_type != zipfile.ZIP_DEFLATED:
                out[j] = self._raw_t[s:e]                                  # a stored member: its bytes
        if defl:
            # nvCOMP reads each stream from a 4-byte aligned address and a zip member starts anywhere:
            # the batch's streams are gathered on the device into one buffer, each at a multiple of 8
            dev = self.device
            src = torch.as_tensor([spans[j][0] for j in defl], dtype=torch.int64, device=dev)
            ln = torch.as_tensor([spans[j][1] - spans[j][0] for j in defl], dtype=torch.int64, device=dev)
            pad = (ln + 7) // 8 * 8
            dst = torch.cumsum(pad, 0) - pad
            member = torch.repeat_interleave(torch.arange(len(defl), device=dev), pad)
            within = torch.arange(int(pad.sum()), device=dev) - dst[member]
            take = within < ln[member]
            idx = torch.where(take, src[member] + within, torch.zeros_like(within))
            buf = torch.where(take, self._raw_t[idx], torch.zeros_like(idx, dtype=torch.uint8))
            d_l, l_l = dst.tolist(), ln.tolist()
            srcs = nvcomp.as_arrays([buf[d_l[q]:d_l[q] + l_l[q]] for q in range(len(defl))])
            dec = self._codec.decode(srcs)
            for jj, j in enumerate(defl):
                t = torch.as_tensor(dec[jj], device=self.device)
                if int(t.numel()) != int(spans[j][2].file_size):
                    raise RuntimeError(f"nvCOMP inflated {keys[j]} to {int(t.numel())} bytes, the zip "
                                       f"declares {spans[j][2].file_size}")
                out[j] = t.clone()                                         # owned by torch, not by nvCOMP
        return out

    def stack(self, keys: Sequence[str]) -> torch.Tensor:
        """The masks of ``keys`` as one boolean tensor (k, H, W) on the device — every mask on one grid."""
        if not keys:
            raise ValueError("stack() of no mask")
        if self.device.type != "cuda":
            arrs = []
            for k in keys:
                buf = self._inflate_cpu(k)
                off, dt, shape, fortran = _npy_header(buf)
                a = np.frombuffer(buf[off:], dtype=dt).reshape(shape, order="F" if fortran else "C")
                arrs.append(a != 0)
            return torch.as_tensor(np.stack(arrs), device=self.device)
        parts = []
        for b in range(0, len(keys), BATCH):
            ks = list(keys[b:b + BATCH])
            for k, t in zip(ks, self._inflate_device(ks)):
                head = self._head.get(k)
                if head is None:
                    head = self._head[k] = _npy_header(bytes(t[:256].cpu().numpy().tobytes()))
                off, dt, shape, fortran = head
                if dt.itemsize != 1 or dt.kind not in ("b", "u") or fortran or len(shape) != 2:
                    raise RuntimeError(f"mask {k}: {dt} {shape} fortran={fortran} — the device reader takes "
                                       f"C-ordered one-byte unsigned / boolean 2-D masks (the SAM3 store's: "
                                       f"there, nonzero and > 0 are one test)")
                parts.append((t[off:off + shape[0] * shape[1]].view(shape[0], shape[1]) != 0))
        shapes = {tuple(p.shape) for p in parts}
        if len(shapes) != 1:
            raise RuntimeError(f"masks on {len(shapes)} grids in one request: {sorted(shapes)}")
        return torch.stack(parts)


def dilate_cross_t(m: torch.Tensor, iterations: int) -> torch.Tensor:
    """``scipy.ndimage.binary_dilation(m, iterations=…)`` on the device for a mask (H, W) or a stack
    (k, H, W): the 3x3 cross, the outside 0."""
    import torch.nn.functional as F
    x = m.to(torch.bool)
    if iterations <= 0:
        return x
    single = x.dim() == 2
    x = (x[None] if single else x)[:, None].to(torch.float32)
    k = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=m.device)[None, None]
    for _ in range(int(iterations)):
        x = (F.conv2d(x, k, padding=1) > 0).to(torch.float32)
    out = x[:, 0] > 0
    return out[0] if single else out


def erode_like_cv2_t(m: torch.Tensor, kernel: np.ndarray, iterations: int) -> torch.Tensor:
    """``cv2.erode(m, kernel, iterations=…)`` of a 0/1 mask (H, W) on the device — the default border
    (the outside counts as set, it erodes nothing), ``iterations`` passes."""
    import torch.nn.functional as F
    k = torch.as_tensor(np.asarray(kernel, np.float32), device=m.device)[None, None]
    kh, kw = int(k.shape[-2]), int(k.shape[-1])
    total = float(k.sum())
    x = m.to(torch.float32)[None, None]
    for _ in range(int(iterations)):
        xp = F.pad(x, (kw // 2, kw - 1 - kw // 2, kh // 2, kh - 1 - kh // 2), value=1.0)
        x = (F.conv2d(xp, k) >= total - 0.5).to(torch.float32)
    return x[0, 0] > 0
