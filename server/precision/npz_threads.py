"""An .npz written and read on THREADS — the same arrays, a fraction of the time (USER 2026-11-20, watching the
card idle at the end of F4: "la veo ocupada pero no operativa"; and his word on speed: "todo lo que no cambie el
resultado debe programarse mas rapido").

``np.savez_compressed`` deflates its members one after another on one core: pccr 2026-08-24's tracks — 17 arrays,
12.5 GB — took 9.2 minutes, the card held and idle (measured 2026-11-20, 03:46:20 → 03:55:28 UTC). Here every
member's .npy stream is cut in pieces of ``BLOCK_BYTES`` and the pieces are deflated at the same time, one thread
a piece (zlib releases the GIL), then written out in the order given. A piece that is not a member's last ends on
a sync flush — a byte boundary and no end-of-stream mark — so the pieces put end to end ARE one deflate stream
(pigz's way, each piece with its own empty window): any reader of a ZIP inflates it, ``np.load`` among them.

WHAT IS THE SAME, AND WHAT IS NOT
  * every member is, uncompressed, the very .npy numpy writes (its header, the array's bytes in C order): the
    same CRC-32 and length as in numpy's own file; ``np.load`` gives back the same arrays, dtype, shape and bits.
  * the compressed bytes of a member no longer than one piece are numpy's own (zipfile's deflate settings); a
    longer member is the same data deflated piece by piece — a few bytes more a piece.
  * the FILE is not numpy's byte for byte, and numpy's was not reproducible either: it stamps every member with
    the clock. Here the stamp is fixed (1980-01-01) and the pieces' length is a constant: the same arrays give
    the same file at every run, whatever the number of threads.
  * the archive is always ZIP64 (numpy forces it per member too): sizes and offsets over 4 GiB are plain cases.
  * an array of Python objects is REFUSED (numpy would pickle it): nothing of that kind belongs in a product.

``load`` reads the members on threads through ``numpy.lib.format.read_array`` — what ``np.load`` calls — each
thread on its own handle of the file.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib import format as npy_format

#: a member's .npy stream is deflated in pieces of this many bytes of the array's data (its header goes with the
#: first). A CONSTANT: the file is a function of the arrays and of this number, never of the thread count.
BLOCK_BYTES = 16 << 20
#: the fixed stamp of every member: 1980-01-01 00:00:00 in DOS date / time
_DOS_TIME, _DOS_DATE = 0, (0 << 9) | (1 << 5) | 1
_VERSION = 45                    # ZIP64
_WIDE = 0xFFFFFFFF
_CRC_STEP = 256 << 20


def _npy(value) -> Tuple[bytes, np.ndarray]:
    """The .npy numpy writes for ``value`` as (header, the data's bytes) — the bytes a VIEW of the array when it
    is C-contiguous (nothing of a 2 GB member is copied), numpy's own writer into memory otherwise."""
    a = np.asanyarray(value)
    if a.dtype.hasobject:
        raise ValueError("npz_threads: an array of Python objects is not written (numpy would pickle it)")
    if a.flags.c_contiguous and a.dtype.kind in "biufcmMSUV" and a.dtype.itemsize > 0:
        head = io.BytesIO()
        try:
            # version 1.0 is what numpy's writer picks whenever the header fits it
            npy_format.write_array_header_1_0(head, npy_format.header_data_from_array_1_0(a))
        except ValueError:
            head = None                                  # a header of a later version: numpy's own writer below
        if head is not None:
            return head.getvalue(), a.reshape(-1).view(np.uint8)
    whole = io.BytesIO()
    npy_format.write_array(whole, a, allow_pickle=False)
    return b"", np.frombuffer(whole.getbuffer(), np.uint8)


def _deflate(head: bytes, block, last: bool) -> bytes:
    """One piece of a member's stream, deflated with zipfile's own settings; only the last one ends the stream."""
    z = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    out = [z.compress(head)] if head else []
    out.append(z.compress(block))
    out.append(z.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH))
    return b"".join(out)


def _crc(head: bytes, data: np.ndarray) -> int:
    c = zlib.crc32(head)
    for i in range(0, data.size, _CRC_STEP):
        c = zlib.crc32(data[i:i + _CRC_STEP], c)
    return c & 0xFFFFFFFF


def save(path, arrays: Mapping[str, object], threads: int = 8) -> None:
    """Write ``arrays`` as an .npz that ``np.load`` reads back as it reads ``np.savez_compressed``'s: the same
    members in the mapping's order, the same arrays. Their pieces are deflated ``threads`` at a time."""
    names = list(arrays)
    if len(set(names)) != len(names):
        raise ValueError("npz_threads.save: a member's name is given twice")
    directory: List[Tuple[bytes, int, int, int, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool, open(path, "wb") as fp:
        work = []
        for name in names:                               # every piece is queued in the file's order …
            head, data = _npy(arrays[name])
            cuts = list(range(0, data.size, BLOCK_BYTES)) or [0]
            pieces = [pool.submit(_deflate, head if i == 0 else b"", data[i:i + BLOCK_BYTES], i == cuts[-1])
                      for i in cuts]
            work.append((name, len(head) + data.size, pool.submit(_crc, head, data), pieces))
        for name, size, crc_f, pieces in work:           # … and written in it, as each member's pieces end
            try:
                parts = [p.result() for p in pieces]
                crc, csize = crc_f.result(), sum(len(p) for p in parts)
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)    # nothing more is deflated for a file that is lost
                raise
            pieces.clear()                               # (a written member's bytes are not kept)
            fname = (name + ".npy").encode("utf-8")
            offset = fp.tell()
            extra = struct.pack("<HHQQ", 1, 16, size, csize)
            fp.write(struct.pack("<4sHHHHHLLLHH", b"PK\x03\x04", _VERSION, 0x800, 8, _DOS_TIME, _DOS_DATE, crc,
                                 _WIDE, _WIDE, len(fname), len(extra)) + fname + extra)
            for part in parts:
                fp.write(part)
            directory.append((fname, crc, size, csize, offset))
        start = fp.tell()
        for fname, crc, size, csize, offset in directory:
            extra = struct.pack("<HHQQQ", 1, 24, size, csize, offset)
            fp.write(struct.pack("<4sHHHHHHLLLHHHHHLL", b"PK\x01\x02", _VERSION, _VERSION, 0x800, 8, _DOS_TIME,
                                 _DOS_DATE, crc, _WIDE, _WIDE, len(fname), len(extra), 0, 0, 0, 0, _WIDE)
                     + fname + extra)
        length = fp.tell() - start
        n = len(directory)
        if n == 0:                                   # an empty archive is the 22 bytes numpy knows one by
            fp.write(struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, 0, 0, 0, 0, 0))
            return
        fp.write(struct.pack("<4sQHHLLQQQQ", b"PK\x06\x06", 44, _VERSION, _VERSION, 0, 0, n, n, length, start))
        fp.write(struct.pack("<4sLQL", b"PK\x06\x07", 0, start + length, 1))
        fp.write(struct.pack("<4sHHHHLLH", b"PK\x05\x06", 0, 0, min(n, 0xFFFF), min(n, 0xFFFF), _WIDE, _WIDE, 0))


def _read(path: str, name: str) -> np.ndarray:
    with zipfile.ZipFile(path) as z, z.open(name + ".npy") as fp:
        return npy_format.read_array(fp, allow_pickle=False)


def load(path, members: Optional[Sequence[str]] = None, threads: int = 8) -> Dict[str, np.ndarray]:
    """The members of an .npz (all, or ``members``) read on threads — the arrays ``np.load`` gives."""
    path = str(Path(path))
    with zipfile.ZipFile(path) as z:
        have = [n[:-4] for n in z.namelist() if n.endswith(".npy")]
    keys = have if members is None else list(members)
    missing = [k for k in keys if k not in have]
    if missing:
        raise KeyError(f"{path} lacks {missing}")
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        return dict(zip(keys, pool.map(lambda k: _read(path, k), keys)))
