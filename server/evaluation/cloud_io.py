"""Binary PLY access for the battery: a memory-mapped reader (a 19 M-point session cloud is
half a gigabyte; nothing here copies it) and a writer for synthetic clouds and imports."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from correction.session import _PLY_TYPE  # the repo's one PLY type table

_PLY_NAME_OF = {"<f4": "float", "<f8": "double", "|u1": "uchar", "|i1": "char", "<u2": "ushort",
                "<i2": "short", "<u4": "uint", "<i4": "int"}


class PlyError(RuntimeError):
    pass


def read_ply_header(path: Path) -> Tuple[List[Tuple[str, str]], int, int]:
    """(properties [(name, numpy dtype)], n_vertices, header_bytes) of a binary-LE PLY."""
    props: List[Tuple[str, str]] = []
    n = 0
    offset = 0
    with open(path, "rb") as f:
        first = f.readline()
        offset += len(first)
        if first.strip() != b"ply":
            raise PlyError(f"{path}: not a PLY file")
        while True:
            line = f.readline()
            if not line:
                raise PlyError(f"{path}: truncated PLY header")
            offset += len(line)
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("format") and "binary_little_endian" not in s:
                raise PlyError(f"{path}: only binary_little_endian PLY is read ({s})")
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("element") and not s.startswith("element vertex"):
                raise PlyError(f"{path}: unsupported element ({s})")
            elif s.startswith("property"):
                parts = s.split()
                if parts[1] == "list":
                    raise PlyError(f"{path}: list properties are not supported")
                props.append((parts[2], _PLY_TYPE[parts[1]]))
            elif s == "end_header":
                break
    return props, n, offset


def read_ply_memmap(path: Path) -> np.ndarray:
    """The vertex table as a read-only structured memmap (fields = the PLY properties)."""
    props, n, offset = read_ply_header(Path(path))
    dt = np.dtype(props)
    return np.memmap(path, dtype=dt, mode="r", offset=offset, shape=(n,))


def write_ply(path: Path, data: np.ndarray, comments: Sequence[str] = ()) -> Path:
    """Write a structured array as a binary-LE PLY (one property per field, in order)."""
    path = Path(path)
    if data.dtype.names is None:
        raise PlyError("write_ply needs a structured array")
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {len(data)}")
    for name in data.dtype.names:
        dt = data.dtype[name]
        key = dt.str if dt.itemsize > 1 else "|" + dt.str[1:]
        if key not in _PLY_NAME_OF:
            raise PlyError(f"field {name}: dtype {dt} has no PLY type")
        lines.append(f"property {_PLY_NAME_OF[key]} {name}")
    lines.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())
    return path


def xyz_of(data: np.ndarray) -> np.ndarray:
    """(N,3) float64 of a PLY table's x y z."""
    return np.stack([np.asarray(data["x"], np.float64), np.asarray(data["y"], np.float64),
                     np.asarray(data["z"], np.float64)], axis=1)
