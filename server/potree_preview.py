"""The LIGHT CLOUD the viewer draws the instant a session opens (USER 2026-10-09: *"las nubes
deben cargarse inmediatamente en versión bien liviana, para que el usuario las vea
inmediatamente"*).

It is the octree's own top levels, repacked: Potree 2 stores the coarsest levels as the
first nodes of ``octree.bin`` and indexes them in the first chunk of ``hierarchy.bin``, so
the preview reads a few tens of MB at most and never the cloud. Whole levels are taken
from the root down while they fit ``PREVIEW_POINTS``; the next level fills the rest by a
fixed stride (deterministic — the same octree gives the same bytes). Each point is
quantized to uint16 over the preview's bounding box and keeps its 8-bit colour: 9 bytes a
point against the octree's 51, so ~300 k points are ~2.7 MB.

The positions are in the OCTREE's frame (``int * scale + offset``, exactly as the viewer's
loader decodes a node): the viewer gives the preview the same floor transform it gives the
octree, so the two coincide and the octree replaces the preview in place.

Format ``STPV`` v1, little-endian::

    'STPV'  u32 version  u32 n  f32 min[3]  f32 max[3]  f32 spacing
    u16 q[n*3]   (p = min + q / 65535 * (max - min))
    u8  rgb[n*3]

Written by the stage that builds the octree (`potree_converter`, into the octree directory
before it is swapped in, so it travels with the octree through every swap and epoch store).
An octree built before this module existed has none: the viewer's endpoint builds the same
bytes in memory (opening never writes, docs/plan_determinismo.md point 111).
"""
from __future__ import annotations

import json
import struct
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PREVIEW_NAME = "preview.bin"
PREVIEW_MAGIC = b"STPV"
PREVIEW_VERSION = 1
#: points in the light cloud: enough to read the scene, light enough to arrive at once
PREVIEW_POINTS = 300_000
_ENTRY = 22   # Potree 2 hierarchy entry: u8 type, u8 childMask, u32 numPoints, i64 offset, i64 size
_PROXY = 2


def _first_chunk_nodes(potree_dir: Path, meta: dict) -> List[Tuple[int, int, int, int]]:
    """(level, numPoints, byteOffset, byteSize) of every non-proxy node of the hierarchy's
    first chunk, in its breadth-first order."""
    first = int(meta["hierarchy"]["firstChunkSize"])
    with open(potree_dir / "hierarchy.bin", "rb") as fh:
        buf = fh.read(first)
    n = len(buf) // _ENTRY
    levels = [0] * n
    nxt = 1
    out = []
    for i in range(n):
        typ, mask, npts, off, size = struct.unpack_from("<BBIqq", buf, i * _ENTRY)
        if typ != _PROXY:
            for _ in range(bin(mask).count("1")):
                if nxt < n:
                    levels[nxt] = levels[i] + 1
                    nxt += 1
            out.append((levels[i], int(npts), int(off), int(size)))
    return out


def build_preview(potree_dir, target: int = PREVIEW_POINTS) -> Optional[bytes]:
    """The ``STPV`` bytes of the octree in ``potree_dir``; None when it has no octree."""
    potree_dir = Path(potree_dir)
    meta_p = potree_dir / "metadata.json"
    if not meta_p.is_file() or not (potree_dir / "octree.bin").is_file() \
            or not (potree_dir / "hierarchy.bin").is_file():
        return None
    meta = json.loads(meta_p.read_text())
    rec = 0
    pos_off = rgb_off = None
    for a in meta["attributes"]:
        if a["name"] == "position":
            pos_off = rec
        elif a["name"] == "rgb":
            rgb_off = rec
        rec += int(a["size"])
    if pos_off is None:
        return None
    nodes = _first_chunk_nodes(potree_dir, meta)
    per_level: Dict[int, int] = {}
    for lv, npts, _, _ in nodes:
        per_level[lv] = per_level.get(lv, 0) + npts
    full, cum = -1, 0
    for lv in sorted(per_level):
        if cum + per_level[lv] > target and full >= 0:
            break
        cum += per_level[lv]
        full = lv
    fill_level = full + 1 if (full + 1) in per_level else None
    stride = (int(np.ceil(per_level[fill_level] / max(target - cum, 1)))
              if fill_level is not None and target > cum else None)

    xyz_parts, rgb_parts = [], []
    with open(potree_dir / "octree.bin", "rb") as fh:
        for lv, npts, off, size in nodes:
            if npts <= 0 or lv > full + (1 if stride else 0):
                continue
            fh.seek(off)
            raw = fh.read(npts * rec)
            if len(raw) < npts * rec:
                continue
            recs = np.frombuffer(raw, np.uint8).reshape(npts, rec)
            if lv > full:
                recs = recs[::stride]
            xyz_parts.append(np.ascontiguousarray(recs[:, pos_off:pos_off + 12]).view("<i4"))
            if rgb_off is not None:
                rgb_parts.append((np.ascontiguousarray(recs[:, rgb_off:rgb_off + 6])
                                  .view("<u2") >> 8).astype(np.uint8))
    if not xyz_parts:
        return None
    q = np.concatenate(xyz_parts).astype(np.float64)
    xyz = q * np.asarray(meta["scale"], np.float64) + np.asarray(meta["offset"], np.float64)
    rgb = (np.concatenate(rgb_parts) if rgb_parts
           else np.full((len(xyz), 3), 200, np.uint8))
    lo = xyz.min(axis=0).astype(np.float32)
    hi = xyz.max(axis=0).astype(np.float32)
    span = np.maximum(hi.astype(np.float64) - lo, 1e-9)
    qn = np.clip(np.rint((xyz - lo) / span * 65535.0), 0, 65535).astype("<u2")
    # the preview's own point spacing (the viewer sizes its points by it): a level halves the
    # octree's spacing; a stride over a surface widens it by its square root
    used = full + (1 if stride else 0)
    spacing = (float(meta.get("spacing") or 0.0) / (2.0 ** max(used, 0))
               * (float(np.sqrt(stride)) if stride else 1.0))
    head = (PREVIEW_MAGIC + struct.pack("<II", PREVIEW_VERSION, len(xyz))
            + struct.pack("<3f", *lo.tolist()) + struct.pack("<3f", *hi.tolist())
            + struct.pack("<f", spacing))
    return head + qn.tobytes() + np.ascontiguousarray(rgb, np.uint8).tobytes()


def write_preview(potree_dir) -> Optional[Path]:
    """Write ``preview.bin`` into an octree directory (the stage that built the octree)."""
    data = build_preview(potree_dir)
    if data is None:
        return None
    p = Path(potree_dir) / PREVIEW_NAME
    p.write_bytes(data)
    return p


_MEMO: Dict[str, Tuple[tuple, bytes]] = {}
_MEMO_LOCK = threading.Lock()


def preview_bytes(potree_dir) -> Optional[bytes]:
    """The preview of the octree in ``potree_dir``: its ``preview.bin`` when the stage wrote
    one, else built in memory (never written — opening never writes), memoised per octree
    directory by the octree files' (inode, size, mtime_ns)."""
    potree_dir = Path(potree_dir)
    p = potree_dir / PREVIEW_NAME
    if p.is_file():
        return p.read_bytes()
    try:
        sig = tuple((s.st_ino, s.st_size, s.st_mtime_ns) for s in
                    (( potree_dir / f).stat() for f in ("metadata.json", "hierarchy.bin", "octree.bin")))
    except OSError:
        return None
    key = str(potree_dir.resolve())
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        data = build_preview(potree_dir)
        if data is not None:
            _MEMO[key] = (sig, data)
        return data
