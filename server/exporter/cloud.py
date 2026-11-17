"""The published cloud (``cleaned_cloud.ply``) read in blocks and written as PLY, LAS and E57.

  - every point is the one on disk: its position goes through the scan's floor transform (the
    viewer's display frame — identity on a session whose orientation is baked) and the turn to
    Z-up; with an identity floor transform that is an exact permutation of its float32 values;
  - the confidence filter is THE VIEWER'S (as GenRecon's, tools/genrecon/adapter.py): the raw
    confidence normalised by the octree metadata's [min, max], a point kept when that is at
    least the slider's value; 0 keeps every point;
  - PLY keeps every property of the source (the point's origin — frame_global, pixel_row,
    pixel_col — is never broken); LAS 1.4 (point format 7: colour) carries them as extra
    dimensions; E57 carries position and colour.

Blocks of ``BLOCK`` points: the largest cloud of the project (114 M points) is never held whole —
except by the E57 writer, whose library takes a scan's arrays at once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

BLOCK = 4_000_000
# display frame (Y up) → Z up: (x, y, z) → (x, −z, y)
Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# LAS keeps a coordinate as a 32-bit integer × a scale: 10 µm (± 21 km of range) is finer than
# the float32 the cloud is stored in at any distance a scan reaches — nothing is lost in it
LAS_SCALE = 1e-5
_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
              "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
              "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}
_PLY_NAMES = {"i1": "char", "u1": "uchar", "<i2": "short", "<u2": "ushort", "<i4": "int", "<u4": "uint",
              "<f4": "float", "<f8": "double"}


class CloudError(ValueError):
    pass


def open_ply(path: Path | str) -> np.memmap:
    """The cloud's vertices as a memory map of its own records (binary little-endian PLY)."""
    path = Path(path)
    fields: List[Tuple[str, str]] = []
    count = None
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise CloudError(f"{path}: not a PLY file")
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise CloudError(f"{path}: its header does not end")
            words = line.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "format" and words[1] != "binary_little_endian":
                raise CloudError(f"{path}: format {words[1]} (binary_little_endian is the one read)")
            if words[0] == "element":
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
                elif count is not None and int(words[2]) > 0:
                    raise CloudError(f"{path}: an element '{words[1]}' after the vertices (a cloud is read)")
            elif words[0] == "property" and in_vertex:
                if words[1] == "list" or words[1] not in _PLY_TYPES:
                    raise CloudError(f"{path}: vertex property type '{words[1]}'")
                fields.append((words[2], _PLY_TYPES[words[1]]))
            elif words[0] == "end_header":
                offset = f.tell()
                break
    names = [n for n, _ in fields]
    if count is None or not {"x", "y", "z"} <= set(names):
        raise CloudError(f"{path}: no vertex element with x, y, z")
    return np.memmap(path, dtype=np.dtype(fields), mode="r", offset=offset, shape=(count,))


def floor_transform(output_dir: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """(A, t): display = A · raw + t — the scan's floor transform (identity when there is none)."""
    p = Path(output_dir) / "floor_transform.npz"
    if not p.exists():
        return np.eye(3), np.zeros(3)
    d = np.load(p)
    return float(d["s"]) * np.asarray(d["R"], np.float64), np.asarray(d["t"], np.float64).reshape(3)


def confidence_range(output_dir: Path | str, cloud: np.memmap) -> Tuple[float, float, str]:
    """[min, max] the viewer normalises the confidence by: the octree metadata's, else the cloud's own."""
    meta = Path(output_dir) / "potree" / "metadata.json"
    if meta.exists():
        a = next((a for a in json.loads(meta.read_text()).get("attributes", []) if a.get("name") == "confidence"), None)
        if a:
            return float(a["min"][0]), float(a["max"][0]), "the octree metadata"
    lo, hi = np.inf, -np.inf
    for k in range(0, len(cloud), BLOCK):
        c = cloud["confidence"][k:k + BLOCK]
        lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    return lo, hi, "the cloud itself (no octree metadata)"


class Cloud:
    """The cloud of a scan as it is exported: its points in the export frame, the ones the
    confidence filter keeps."""

    def __init__(self, output_dir: Path | str, confidence: float = 0.0):
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "cleaned_cloud.ply"
        if not self.path.exists():
            raise CloudError(f"{self.output_dir}: no published cloud (cleaned_cloud.ply)")
        if not (0.0 <= float(confidence) < 1.0):
            raise CloudError(f"the confidence filter must be in [0, 1), got {confidence!r}")
        self.data = open_ply(self.path)
        self.confidence = float(confidence)
        A, t = floor_transform(self.output_dir)
        self.identity = bool(np.array_equal(A, np.eye(3)) and not t.any())
        self.A, self.t = Z_UP @ A, Z_UP @ t
        self.range = None
        if self.confidence > 0.0:
            if "confidence" not in self.data.dtype.names:
                raise CloudError(f"{self.path}: no confidence property — it cannot be filtered by confidence")
            self.range = confidence_range(self.output_dir, self.data)
        self._stats: Optional[Dict[str, Any]] = None

    @property
    def total(self) -> int:
        return int(len(self.data))

    def blocks(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """(records kept, their positions in the export frame — float64 N×3), block by block."""
        for k in range(0, len(self.data), BLOCK):
            rec = np.asarray(self.data[k:k + BLOCK])
            if self.range is not None:
                lo, hi, _ = self.range
                rec = rec[(rec["confidence"].astype(np.float64) - lo) / max(hi - lo, 1e-6) >= self.confidence]
            if not len(rec):
                continue
            if self.identity:
                # the turn alone: (x, −z, y), exact on the stored float32
                xyz = np.column_stack([rec["x"], -rec["z"], rec["y"]]).astype(np.float64)
            else:
                xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64) @ self.A.T + self.t
            yield rec, xyz

    def stats(self) -> Dict[str, Any]:
        """How many points are exported and their bounds in the export frame (one pass)."""
        if self._stats is None:
            n, lo, hi = 0, np.full(3, np.inf), np.full(3, -np.inf)
            for _rec, xyz in self.blocks():
                n += len(xyz)
                lo, hi = np.minimum(lo, xyz.min(0)), np.maximum(hi, xyz.max(0))
            if n == 0:
                raise CloudError(f"the confidence filter {self.confidence} keeps no point of {self.path}")
            self._stats = {"points": n, "total": self.total, "min": lo.tolist(), "max": hi.tolist()}
        return self._stats


Progress = Optional[Callable[[float], Any]]


def write_ply(cloud: Cloud, out: Path | str, progress: Progress = None) -> int:
    """Binary PLY with every property of the source, the position in the export frame (float32)."""
    st = cloud.stats()
    dtype = cloud.data.dtype
    header = ["ply", "format binary_little_endian 1.0",
              "comment STAC Build export - metres, Z up, the floor at z = 0", f"element vertex {st['points']}"]
    header += [f"property {_PLY_NAMES[dtype[n].str.replace('|', '')]} {n}" for n in dtype.names]
    header.append("end_header")
    done = 0
    with open(out, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for rec, xyz in cloud.blocks():
            block = rec.copy()
            block["x"], block["y"], block["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            f.write(block.tobytes())
            done += len(block)
            if progress:
                progress(done / st["points"])
    return done


def write_las(cloud: Cloud, out: Path | str, progress: Progress = None) -> int:
    """LAS 1.4, point format 7 (position + 16-bit colour); the other properties as extra dimensions."""
    import laspy
    st = cloud.stats()
    names = cloud.data.dtype.names
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.full(3, LAS_SCALE)
    header.offsets = np.floor(np.asarray(st["min"]))
    extra = [n for n in names if n not in ("x", "y", "z", "red", "green", "blue")]
    header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=cloud.data.dtype[n].str.replace("|", "").lstrip("<"))
                           for n in extra])
    done = 0
    with laspy.open(str(out), mode="w", header=header) as writer:
        for rec, xyz in cloud.blocks():
            pts = laspy.ScaleAwarePointRecord.zeros(len(rec), header=header)
            pts.x, pts.y, pts.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            if "red" in names:
                # 8-bit colour as LAS's 16-bit: v × 257 (255 → 65535)
                pts.red, pts.green, pts.blue = (rec[c].astype(np.uint16) * 257 for c in ("red", "green", "blue"))
            for n in extra:
                pts[n] = rec[n]
            writer.write_points(pts)
            done += len(rec)
            if progress:
                progress(done / st["points"])
    return done


def write_e57(cloud: Cloud, out: Path | str, name: str, progress: Progress = None) -> int:
    """E57: one scan — position (float64) and colour. (The library takes the arrays whole.)"""
    import pye57
    st = cloud.stats()
    n = st["points"]
    has_colour = "red" in cloud.data.dtype.names
    data: Dict[str, np.ndarray] = {k: np.empty(n, np.float64) for k in ("cartesianX", "cartesianY", "cartesianZ")}
    if has_colour:
        data.update({k: np.empty(n, np.uint8) for k in ("colorRed", "colorGreen", "colorBlue")})
    at = 0
    for rec, xyz in cloud.blocks():
        m = len(rec)
        data["cartesianX"][at:at + m], data["cartesianY"][at:at + m], data["cartesianZ"][at:at + m] = xyz.T
        if has_colour:
            for k, c in (("colorRed", "red"), ("colorGreen", "green"), ("colorBlue", "blue")):
                data[k][at:at + m] = rec[c]
        at += m
        if progress:
            progress(0.5 * at / n)
    out = Path(out)
    if out.exists():
        out.unlink()
    e57 = pye57.E57(str(out), mode="w")
    try:
        e57.write_scan_raw(data, name=name)
    finally:
        e57.close()
    if progress:
        progress(1.0)
    return n
