"""The deviation sheet's octree, made FROM THE SCAN'S OWN OCTREE — no second build.

USER 2026-11-19: "la comparacion debe hacerse en GPU en un instante". The sheet's octree was built
by PotreeConverter from the sheet's cloud: 40 of the 171 s of pccr 2026-08-31's comparison, on the
CPU. But the sheet's points ARE the scan's points. The scan's octree is UNCOMPRESSED — one record
of fixed size a point — so the sheet's is that file with ONE field changed, each point's colour
(the deviation's), and its hierarchy and metadata as they are. A point of the octree is found in
the cloud by its origin (frame_global, pixel_row, pixel_col), which both carry and no two points
share.

A point of the cloud that belongs to no compared object is painted NO_DATA (the grey the comparison
gives "no data"): the sheet built from the sheet's own cloud did not hold those points.

When the scan's octree cannot stand for the cloud (another point count, compressed, no origins, an
origin shared or not found) :func:`build` says why and writes nothing: the caller builds the sheet's
octree the old way.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

NO_DATA_RGB8 = (76, 76, 76)       # (0.3, 0.3, 0.3) as the sheet's cloud wrote it: 0.3 * 255 cut to 8 bits
ORIGIN = ("frame_global", "pixel_row", "pixel_col")
_TYPES = {"int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2", "int32": "<i4", "uint32": "<u4",
          "int64": "<i8", "uint64": "<u8", "float": "<f4", "double": "<f8"}


def record_dtype(metadata: dict) -> np.dtype:
    """One point of an UNCOMPRESSED octree.bin, as its metadata.json declares it."""
    fields = []
    for a in metadata["attributes"]:
        base = _TYPES[a["type"]]
        n = int(a["numElements"])
        fields.append((a["name"], base) if n == 1 else (a["name"], base, (n,)))
    dtype = np.dtype(fields)
    declared = sum(int(a["size"]) for a in metadata["attributes"])
    if dtype.itemsize != declared:
        raise ValueError(f"the octree's attributes declare {declared} bytes a point and read as {dtype.itemsize}")
    return dtype


def _keys(frame, row, col) -> np.ndarray:
    return (np.asarray(frame).astype(np.int64) << 32) | (np.asarray(row).astype(np.int64) << 16) | np.asarray(col).astype(np.int64)


def _join(cloud_keys: np.ndarray, octree_keys: np.ndarray, device: str) -> Tuple[Optional[np.ndarray], str]:
    """For each point of the octree, its index in the cloud — (None, why) when an origin is shared
    by two points of the cloud or a point of the octree has none of them."""
    if device == "numpy":
        order = np.argsort(cloud_keys, kind="stable")
        s = cloud_keys[order]
        if len(s) > 1 and not bool((s[1:] > s[:-1]).all()):
            return None, "two points of the cloud share an origin"
        pos = np.minimum(np.searchsorted(s, octree_keys), len(s) - 1)
        if not bool((s[pos] == octree_keys).all()):
            return None, "a point of the octree has no point of the cloud with its origin"
        return order[pos], ""
    import torch
    ck = torch.as_tensor(cloud_keys, device=device)
    ok_ = torch.as_tensor(octree_keys, device=device)
    s, order = torch.sort(ck)
    if s.numel() > 1 and not bool((s[1:] > s[:-1]).all()):
        return None, "two points of the cloud share an origin"
    pos = torch.searchsorted(s, ok_).clamp_(max=s.numel() - 1)
    if not bool((s[pos] == ok_).all()):
        return None, "a point of the octree has no point of the cloud with its origin"
    return order[pos].cpu().numpy(), ""


def build(scan_potree: Path, cloud, rgb8: np.ndarray, out_dir: Path, device: str = "numpy") -> Tuple[bool, str]:
    """Write ``out_dir`` (metadata.json, hierarchy.bin, octree.bin): the scan's octree with each
    point's colour replaced by ``rgb8`` — (N, 3) uint8, one row a point of ``cloud`` (the scan's
    cloud as exporter.cloud.open_ply gives it). (True, what was done) or (False, why not)."""
    scan_potree, out_dir = Path(scan_potree), Path(out_dir)
    meta_path = scan_potree / "metadata.json"
    if not meta_path.is_file():
        return False, f"the scan has no octree ({meta_path})"
    meta = json.loads(meta_path.read_text())
    names = [a["name"] for a in meta.get("attributes", [])]
    if meta.get("encoding") != "UNCOMPRESSED":
        return False, f"the scan's octree is {meta.get('encoding')}, not UNCOMPRESSED"
    if "rgb" not in names or any(k not in names for k in ORIGIN):
        return False, "the scan's octree carries no colour or no point origin"
    if any(k not in (cloud.dtype.names or ()) for k in ORIGIN):
        return False, "the cloud carries no point origin"
    n = len(cloud)
    if int(meta.get("points", -1)) != n or len(rgb8) != n:
        return False, f"the scan's octree holds {meta.get('points')} points and the cloud {n}"
    dtype = record_dtype(meta)
    bin_path = scan_potree / "octree.bin"
    if bin_path.stat().st_size != n * dtype.itemsize:
        return False, f"octree.bin is {bin_path.stat().st_size} bytes, not {n} records of {dtype.itemsize}"
    frame = np.asarray(cloud["frame_global"])
    row, col = np.asarray(cloud["pixel_row"]), np.asarray(cloud["pixel_col"])
    if int(frame.min()) < 0 or int(frame.max()) > 65535 or int(row.min()) < 0 or int(col.min()) < 0:
        return False, "a point's origin does not fit the octree's 16 bits"
    rec = np.fromfile(bin_path, dtype=dtype)
    index, why = _join(_keys(frame, row, col), _keys(rec["frame_global"], rec["pixel_row"], rec["pixel_col"]), device)
    if index is None:
        return False, why
    rec["rgb"] = np.asarray(rgb8, dtype=np.uint8)[index].astype(np.uint16) * 256     # 8 → 16 bits as the octree's builder does
    tmp = out_dir.with_name(out_dir.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    rec.tofile(tmp / "octree.bin")
    shutil.copyfile(scan_potree / "hierarchy.bin", tmp / "hierarchy.bin")
    shutil.copyfile(meta_path, tmp / "metadata.json")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return True, f"{n} points recoloured in the scan's own octree ({dtype.itemsize} bytes a point)"
