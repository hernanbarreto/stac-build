"""The SAM3 mask store decompressed ONCE for the certification's closures (USER 2026-11-01: "todo debe correr
en GPU … cualquier minuto perdido en cpu es tiempo y dinero").

MEASURED on pccr (epoch 4, 14,861 masks of 832×464 in `seg_masks.npz`, cProfile of `visit_drift_run.measure`,
analysis/2026-11-01_certify_timing): 52 of the stage's 115 s were `zlib.decompress` — `masklet_visits` reads
EVERY mask to ask whether it is empty, `points_of_masklets` reads every mask again to look its points up,
`Visibility` a third time for the refinement — 37,834 reads of the npz, one zlib inflate each, on one core.

Here every member of the raw store is inflated once, in a pool of PROCESSES (forked: zlib releases the GIL but
numpy's header parsing and the zip reader hold it — 8 threads inflated pccr's store in 18 s, 1.4× over one
thread; 8 forked workers return the packed bits through pipes, 48 KB a mask), and kept 1 bit per pixel
(48 KB per 832×464 mask, 720 MB for pccr's store); the fused objects' masks (segmentation.fuse_parent.
FusedMasks: a survivor's mask is the OR of its own and its retired parts', same shape or refused) are the OR
of the parts' bits. The readers get exactly what the npz held: ``__getitem__`` the {0,1} uint8 mask,
``any(key)`` whether it holds a pixel, ``bits(key)`` the packed bits; ``lookup`` answers "which of these masks
hold these pixels" for a keyframe's masks at once on the device (the packed bits unpacked there, one gather).
"""
from __future__ import annotations

import multiprocessing
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_MASK_KEY_RE = re.compile(r"^f(\d+)_o(\d+)$")
DEFAULT_THREADS = max(2, min(16, (__import__("os").cpu_count() or 8) >> 1))   # forked inflaters: half the cores


def _pool(workers: int):
    """Forked worker processes (the children read the npz with numpy and zlib only — never the card);
    threads when forking is not available on the platform, or when THIS process is a daemon: every
    pipeline stage runs in the manager's daemonic multiprocessing child, and a daemon may not start
    children (station 2026-11-02: the certification died on its first mask read, "daemonic processes
    are not allowed to have children"). The same members, inflated by the same function."""
    if multiprocessing.current_process().daemon:
        return ThreadPoolExecutor(max_workers=workers)
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)


def _inflate(path: Path, names: Sequence[str]) -> List[Tuple[str, np.ndarray, Tuple[int, int], bool]]:
    """(name, packed bits, shape, any) of the npz members ``names`` — one ZipFile per worker."""
    from numpy.lib import format as npformat
    out = []
    with zipfile.ZipFile(path) as zf:
        for name in names:
            with zf.open(name + ".npy") as f:
                a = np.asarray(npformat.read_array(f))
            m = a > 0
            out.append((name, np.packbits(m, axis=None), (int(m.shape[0]), int(m.shape[1])), bool(m.any())))
    return out


class MaskStore:
    """The fused masks of a session (``segmentation.fuse_parent.FusedMasks``'s view of the raw store), inflated
    once in parallel and kept packed. Drop-in for the readers that held the ``FusedMasks`` handle: ``files``,
    ``__contains__``, ``__iter__``, ``frames_for``, ``__getitem__`` (a {0,1} uint8 array for a mask key, the
    raw member for a meta key), ``close``."""

    def __init__(self, output_dir, threads: int = DEFAULT_THREADS):
        from segmentation import fuse_parent
        self.dir = Path(output_dir)
        self.fused = fuse_parent.FusedMasks(self.dir)
        self.path = self.dir / fuse_parent.MASKS
        self.files: List[str] = list(self.fused.files)
        self.threads = max(1, int(threads))
        self._bits: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        self._any: Dict[str, bool] = {}
        self._loaded = False
        self.shape: Optional[Tuple[int, int]] = None

    # ── the FusedMasks surface ────────────────────────────────────────────────────────────────────
    def __contains__(self, key) -> bool:
        return str(key) in self.fused

    def __iter__(self) -> Iterator[str]:
        return iter(self.files)

    def frames_for(self, oid: int) -> List[Tuple[int, str]]:
        return self.fused.frames_for(oid)

    def close(self) -> None:
        try:
            self.fused.close()
        except Exception:  # noqa: BLE001 — a handle already closed
            pass

    def __getitem__(self, key):
        key = str(key)
        if not _MASK_KEY_RE.match(key):
            return self.fused[key]
        bits, shape = self.bits(key)
        return np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape)

    # ── the inflated store ────────────────────────────────────────────────────────────────────────
    def preload(self) -> "MaskStore":
        """Inflate every mask member once, in ``threads`` workers, and fuse the parts' bits."""
        if self._loaded:
            return self
        raw_by_fused: Dict[str, List[str]] = {}
        for o, frames in self.fused._frames.items():
            for f, keys in frames.items():
                raw_by_fused[f"f{f}_o{o}"] = sorted(keys)
        names = sorted({k for v in raw_by_fused.values() for k in v})
        raw: Dict[str, Tuple[np.ndarray, Tuple[int, int], bool]] = {}
        if names:
            n_chunks = min(len(names), self.threads * 4)
            chunks = [names[i::n_chunks] for i in range(n_chunks)]
            with _pool(self.threads) as ex:
                for part in ex.map(_inflate, [self.path] * len(chunks), chunks):
                    for name, bits, shape, has in part:
                        raw[name] = (bits, shape, has)
        for fk, keys in raw_by_fused.items():
            bits, shape, has = raw[keys[0]]
            for k in keys[1:]:
                b2, s2, h2 = raw[k]
                if s2 != shape:
                    raise ValueError(f"mask {k} is {s2} and {keys[0]} is {shape} — they cannot be one object's mask")
                bits = np.bitwise_or(bits, b2)
                has = has or h2
            self._bits[fk] = (bits, shape)
            self._any[fk] = bool(has)
            self.shape = shape
        self._loaded = True
        return self

    def bits(self, key: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """(packed bits, (H, W)) of one fused mask."""
        if not self._loaded:
            self.preload()
        try:
            return self._bits[str(key)]
        except KeyError:
            raise KeyError(f"{key} is not a mask of the fused store") from None

    def any(self, key: str) -> bool:
        """Whether the mask holds a pixel (an EMPTY stored mask does not see the object)."""
        if not self._loaded:
            self.preload()
        return self._any[str(key)]

    def lookup(self, keys: Sequence[str], rows: np.ndarray, cols: np.ndarray, device: str) -> np.ndarray:
        """(len(keys), len(rows)) bool: whether mask ``keys[m]`` holds pixel (rows[j], cols[j]) — the masks of
        one keyframe against the points born in it, the bits unpacked on ``device`` and gathered at once
        (``numpy``: the same by unpackbits per mask)."""
        rows = np.asarray(rows, np.int64)
        cols = np.asarray(cols, np.int64)
        if not len(keys) or not len(rows):
            return np.zeros((len(keys), len(rows)), bool)
        packed = np.stack([self.bits(k)[0] for k in keys])              # (M, ceil(H·W/8)) uint8
        H, W = self.bits(keys[0])[1]
        flat = rows * W + cols                                          # the pixel in the packed order
        if device == "numpy":
            out = np.empty((len(keys), len(rows)), bool)
            for m in range(len(keys)):
                out[m] = np.unpackbits(packed[m], count=H * W)[flat] > 0
            return out
        import torch
        P = torch.as_tensor(packed, device=device)                      # uint8
        idx = torch.as_tensor(flat, device=device)
        byte = P[:, idx >> 3]                                           # (M, N) the byte of each pixel
        bit = (7 - (idx & 7)).to(torch.uint8)                           # packbits is big-endian within a byte
        return ((byte >> bit[None, :]) & 1).bool().cpu().numpy()


def shared_store(output_dir, threads: int = DEFAULT_THREADS) -> MaskStore:
    """One inflated store per session directory for this process: the closures' three readers and the
    certification's mask filter share it. Keyed by the store file's identity (size and mtime) and the
    fusion map's — a new raw store or a new fusion is a new store."""
    output_dir = Path(output_dir)
    from segmentation import fuse_parent
    ident = [str(output_dir)]
    for p in (output_dir / fuse_parent.MASKS, output_dir / "fusion_map.json"):
        try:
            st = p.stat()
            ident.append(f"{p.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            ident.append(f"{p.name}:absent")
    key = "|".join(ident)
    store = _SHARED.get(key)
    if store is None:
        _SHARED.clear()                 # one session at a time in a worker; never two stores in memory
        store = MaskStore(output_dir, threads=threads)
        _SHARED[key] = store
    return store


_SHARED: Dict[str, MaskStore] = {}
