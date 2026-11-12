"""The SAM3 mask store DEFLATED ON THE DEVICE (USER 2026-11-11/12: "todo lo que me marcaste que puede pasarse a GPU
hay que hacerlo en GPU ... Nada debe hacerse en cpu" / "dije ok a todo").

The SAM3 stage's writers of ``seg_masks.npz`` deflated every member with zlib on the CPU: on pccr 2026-08-24
(290,418 masks) that was ~750 s of zlib plus the CRC-32, in the stage's appends and again in the canonical
rewrite that closes it. Here the members of a write are laid out on the device as rows of ``.npy`` bytes (numpy's
own header, the array's bytes uploaded as they are), deflated by nvCOMP's Deflate encoder (raw streams — what a zip
member holds, and what zlib and nvCOMP both inflate), their CRC-32 computed there (segmentation/crc32_gpu.py), and
written into the zip exactly as zipfile's ``writestr`` lays an entry out
(segmentation.pipeline._write_deflated_entry). A member the canonical rewrite takes from the source store is its
inflated ``.npy`` bytes straight from the device store (segmentation/mask_store_gpu.py) — never brought to the
CPU, never re-serialised — and the CRC-32 computed on the card must be the one the source's zip declares.

EVERY STREAM IS INFLATED BACK ON THE CARD before it is written and held equal to the member's bytes: a store is
hours of SAM3, and a stream nobody can read would be found by the next reader only.

What changes against the record, declared: the STREAMS. nvCOMP's encoder is not zlib's — the members inflate to
the record's bytes (tests/test_store_deflate_gpu.py: zipfile's own CRC check and the members' bytes against the
record's, on synthetic stores and on a slice of a real one), but a store written on the card is another byte
sequence than the record's, and a LARGER one (the numbers measured on a real store: NVCOMP_DEFLATE_ALGORITHM
below). The record — zlib, byte for byte the previous writers — is what every writer runs when no device is
passed (the server's interactive edits: the server process holds no CUDA context) or the device is ``cpu``.
"""
from __future__ import annotations

import io
import zlib
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from segmentation.erase import store_on_card

#: nvCOMP's Deflate effort (its ``algorithm_type``). MEASURED 2026-11-12 (A100, nvidia-nvcomp-cu12 6.0.0.12) on 2,040
#: real masks of pccr 2026-08-31's store (832x464 uint8, 386 kB of .npy each; the encode and its read-back), bytes
#: and ms per mask — 0: 50,077 B / 0.680 ms · 1: 5,799 / 0.398 · 2: 2,514 / 0.405 · 3: 2,514 / 0.406 ·
#: 4: 6,638 / 8.636 · 5: 976 / 5.342 · the zlib record: 1,159 B / 2.363 ms. 2 is the smallest of the fast ones; the
#: one effort smaller than zlib (5) is slower than zlib.
#: THE WHOLE STORE, on a copy (the canonical rewrite, record against card):
#:   pccr 2026-08-31, 55,079 masks — record 81.1 s (1.473 ms a mask), 65.5 MB; card 19.6 s (0.356 ms, 4.1x),
#:   143.8 MB (2.2x), 1.13 GB of torch at the peak;
#:   pccr 2026-08-24, 290,418 masks — record 1280.1 s (4.408 ms a mask), 349.2 MB; card 93.3 s (0.321 ms, 13.7x),
#:   771.7 MB (2.2x), 1.42 GB of torch at the peak.
#: On both: the card's store written again from its own store is the same bytes, and every member inflates to the
#: record's (zipfile's testzip None; sizes / CRCs differing 0; bytes differing 0).
NVCOMP_DEFLATE_ALGORITHM = 2
#: the members of one nvCOMP call, and the bytes of its row buffer (its members x the longest of them) — the
#: device memory one call holds: the rows, their CRC's padded copies (up to 4x) and the streams read back (1x).
#: A member over the bytes goes alone.
BATCH = 512
BATCH_BYTES = 1 << 28


def _npy_layout(a: np.ndarray, name: str) -> Tuple[bytes, np.ndarray]:
    """What ``np.lib.format.write_array`` writes for ``a``: its header (format 1.0) and the array whose C-order
    bytes follow it — ``a``, or its transpose when numpy stores it in Fortran order (F- and not C-contiguous)."""
    if a.dtype.hasobject:
        raise ValueError(f"{name}: object arrays are not stored (no pickle in a mask store)")
    d = np.lib.format.header_data_from_array_1_0(a)
    buf = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(buf, d)
    except ValueError as e:
        raise ValueError(f"{name}: its .npy header does not fit format 1.0 ({e}) — not a mask store's member") from e
    return buf.getvalue(), (a.T if d["fortran_order"] else a)


def _record(raw) -> Tuple[int, int, bytes]:
    """(size, CRC-32, raw Deflate stream) as zipfile's ZIP_DEFLATED writes ``raw`` — zlib, the record."""
    comp = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    return len(raw), zlib.crc32(raw), comp.compress(raw) + comp.flush()


def _npy_bytes(item) -> bytes:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy().tobytes()
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asanyarray(item), allow_pickle=False)
    return buf.getvalue()


def _batches(lengths: Sequence[int]) -> Iterator[Tuple[int, int]]:
    """Consecutive members [lo, hi): at most BATCH of them and BATCH_BYTES of rows."""
    lo = 0
    while lo < len(lengths):
        hi, longest = lo + 1, lengths[lo]
        while hi < len(lengths) and hi - lo < BATCH:
            m = max(longest, lengths[hi])
            if (hi - lo + 1) * ((m + 7) // 8 * 8) > BATCH_BYTES:
                break
            hi, longest = hi + 1, m
        yield lo, hi
        lo = hi


def _u8(x, dev) -> torch.Tensor:
    """A buffer of bytes on the device as a uint8 tensor (nvCOMP hands its arrays back as int8 or uint8)."""
    t = torch.as_tensor(x, device=dev)
    if t.dtype == torch.uint8:
        return t
    if t.dtype == torch.int8:
        return t.view(torch.uint8)
    raise TypeError(f"a byte buffer of dtype {t.dtype}")


def _codec(dev: torch.device):
    from nvidia import nvcomp
    return nvcomp.Codec(algorithm="Deflate", bitstream_kind=nvcomp.BitstreamKind.RAW,
                        algorithm_type=NVCOMP_DEFLATE_ALGORITHM, device_id=dev.index or 0)


def _check_read_back(back: Sequence[torch.Tensor], srcs: Sequence[torch.Tensor], names: Sequence[str]) -> None:
    """Every stream's inflate (``back``) is its member's bytes (``srcs``) — or the write fails, named."""
    same = []
    for nm, b, s in zip(names, back, srcs):
        if int(b.numel()) != int(s.numel()):
            raise RuntimeError(f"{nm}: its nvCOMP Deflate stream inflates to {int(b.numel())} bytes, the member "
                               f"is {int(s.numel())} — nothing was written")
        same.append((b == s).all())
    bad = [i for i, ok in enumerate(torch.stack(same).tolist()) if not ok] if same else []
    if bad:
        raise RuntimeError(f"{names[bad[0]]}: its nvCOMP Deflate stream does not inflate back to the member's "
                           f"bytes ({len(bad)} of {len(same)} in its batch) — nothing was written")


def _deflate_on_card(items: Sequence, names: Sequence[str], dev: torch.device, codec) -> List[Tuple[int, int, bytes]]:
    from nvidia import nvcomp
    from segmentation.crc32_gpu import crc32_rows
    heads: List[Optional[bytes]] = []
    bodies: list = []
    lengths: List[int] = []
    for x, nm in zip(items, names):
        if isinstance(x, torch.Tensor):
            if x.dim() != 1 or x.dtype not in (torch.uint8, torch.int8) or not int(x.numel()):
                raise ValueError(f"{nm}: a member's .npy bytes come as a 1-D byte tensor, not {x.dtype} "
                                 f"{tuple(x.shape)}")
            heads.append(None)
            bodies.append(x)
            lengths.append(int(x.numel()))
        else:
            h, body = _npy_layout(np.asarray(x), nm)
            heads.append(h)
            bodies.append(body)
            lengths.append(len(h) + int(body.nbytes))
    out: List[Tuple[int, int, bytes]] = []
    for lo, hi in _batches(lengths):
        n, part = hi - lo, lengths[lo:hi]
        stride = (max(part) + 7) // 8 * 8                                  # nvCOMP reads from aligned addresses
        buf = torch.zeros((n, stride), dtype=torch.uint8, device=dev)
        for i in range(n):
            h, body = heads[lo + i], bodies[lo + i]
            if h is None:
                buf[i, :part[i]] = _u8(body, dev)
                continue
            buf[i, :len(h)] = torch.frombuffer(bytearray(h), dtype=torch.uint8).to(dev)
            if body.nbytes:
                if not body.flags.c_contiguous:
                    body = np.ascontiguousarray(body)                      # (never 0-d: a 0-d array is contiguous)
                flat = body.reshape(-1).view(np.uint8)
                buf[i, len(h):part[i]] = torch.from_numpy(flat if flat.flags.writeable else flat.copy()).to(dev)
        crcs = crc32_rows(buf, part)
        srcs = [buf[i, :part[i]] for i in range(n)]
        enc = codec.encode(nvcomp.as_arrays(srcs))
        streams = [_u8(e, dev) for e in enc]
        # read back on the card before anything is written: the inflate of each stream is its member
        _check_read_back([_u8(b, dev) for b in codec.decode(enc)], srcs, list(names[lo:hi]))
        sizes = [int(s.numel()) for s in streams]
        flat_streams = torch.cat(streams).cpu().numpy().tobytes()          # one transfer per batch
        at = 0
        for i in range(n):
            out.append((part[i], int(crcs[i]), flat_streams[at:at + sizes[i]]))
            at += sizes[i]
    return out


def deflate_members(items: Sequence, device, names: Optional[Sequence[str]] = None
                    ) -> List[Tuple[int, int, bytes]]:
    """``(size, CRC-32, raw Deflate stream)`` of every member's ``.npy`` bytes. ``items``: arrays (their bytes
    are ``np.lib.format.write_array``'s), or byte tensors already holding a member's complete ``.npy`` bytes (the
    device store's). ``names``: the members', for a failure's message. On the card: nvCOMP, each stream inflated
    back and held equal to its member; else zlib, the record."""
    names = [str(n) for n in names] if names is not None else [f"member {i}" for i in range(len(items))]
    if len(names) != len(items):
        raise ValueError(f"{len(items)} members and {len(names)} names")
    if not store_on_card(device):
        return [_record(_npy_bytes(x)) for x in items]
    dev = torch.device(device)
    return _deflate_on_card(items, names, dev, _codec(dev))


def write_members(zf, names: Sequence[str], items: Sequence, device) -> None:
    """``items`` (see :func:`deflate_members`) as the entries ``names`` of ``zf``, a ZipFile open for writing on
    a seekable stream."""
    from segmentation.pipeline import _write_deflated_entry
    for nm, (size, crc, data) in zip(names, deflate_members(items, device, names)):
        _write_deflated_entry(zf, nm, size, crc, data)


def _zip_bytes(zf) -> bytes:
    """The bytes of the file an open ZipFile reads."""
    fp = zf.fp
    if isinstance(fp, io.BytesIO):
        return fp.getvalue()
    pos = fp.tell()
    try:
        fp.seek(0)
        return fp.read()
    finally:
        fp.seek(pos)


def write_canonical_members(zf, order: Sequence[str], members: dict, source, device) -> None:
    """segmentation.pipeline.write_canonical_store's members on the card, in ``order``: a member whose value is
    None is taken from ``source`` (an open NpzFile) — its ``.npy`` bytes inflated on the card from the source's
    own bytes, their size and CRC-32 held to what the source's zip declares —, the others are the arrays given;
    all deflated there, a batch at a time."""
    from segmentation.mask_store_gpu import DeviceMaskStore
    from segmentation.pipeline import _write_deflated_entry
    if not store_on_card(device):
        raise ValueError(f"the card's canonical writer on the device {device!r}")
    order = list(order)
    dev = torch.device(device)
    st = None
    if any(members[k] is None for k in order):
        if source is None:
            raise ValueError("members are to be taken from a source store and none was given")
        st = DeviceMaskStore(getattr(source.zip, "filename", None) or "<memory>", dev, raw=_zip_bytes(source.zip))
    lengths = []
    for k in order:
        if members[k] is None:
            lengths.append(int(source.zip.getinfo(k + ".npy").file_size))
        else:
            h, body = _npy_layout(np.asarray(members[k]), k)
            lengths.append(len(h) + int(body.nbytes))
    codec = _codec(dev)
    for lo, hi in _batches(lengths):
        ks = order[lo:hi]
        src = [k for k in ks if members[k] is None]
        raw = dict(zip(src, st._inflate_device(src))) if src else {}
        done = _deflate_on_card([raw[k] if members[k] is None else members[k] for k in ks], ks, dev, codec)
        for k, (size, crc, data) in zip(ks, done):
            if members[k] is None:
                info = source.zip.getinfo(k + ".npy")
                if (size, crc) != (info.file_size, info.CRC):
                    raise RuntimeError(f"{k}: inflated on the card to {size} bytes of CRC-32 {crc:08x}; the source "
                                       f"store declares {info.file_size} bytes of {info.CRC:08x} — nothing was "
                                       f"written")
            _write_deflated_entry(zf, k + ".npy", size, crc, data)
