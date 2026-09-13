"""Scalar witness fields inside the session's PLY clouds (cleaned_cloud.ply
and cleaned_cloud_raw.ply): added as ``uchar`` properties, point ORDER and
every other property untouched. Re-running replaces the fields in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

WITNESS_FIELDS = ("mv_votes", "mask_votes", "mask_conflicts", "status")
_PLY_NAME = {"u1": "uchar", "<u1": "uchar"}


def add_fields(header: List[bytes], data: np.ndarray,
               fields: Dict[str, np.ndarray]) -> Tuple[List[bytes], np.ndarray]:
    """(header, data) with ``fields`` (uint8 arrays of len(data)) appended or
    replaced. The header keeps its lines; new properties go before
    end_header."""
    names = list(data.dtype.names or ())
    for k, v in fields.items():
        if len(v) != len(data):
            raise ValueError(f"field {k}: {len(v)} values for {len(data)} points")
    keep = [(n, data.dtype[n]) for n in names if n not in fields]
    new_dtype = np.dtype(keep + [(k, np.uint8) for k in fields])
    out = np.empty(len(data), new_dtype)
    for n, _ in keep:
        out[n] = data[n]
    for k, v in fields.items():
        out[k] = np.asarray(v, np.uint8)
    # header: drop old lines of replaced fields, append the new ones
    new_header: List[bytes] = []
    for line in header:
        s = line.decode("ascii", "ignore").strip()
        if s.startswith("property") and s.split()[-1] in fields:
            continue
        if s == "end_header":
            for k in fields:
                new_header.append(f"property uchar {k}\n".encode("ascii"))
        new_header.append(line)
    return new_header, out


def write_fields(ply_path, fields: Dict[str, np.ndarray]) -> None:
    from correction.session import read_ply, write_ply
    header, data = read_ply(Path(ply_path))
    h2, d2 = add_fields(header, data, fields)
    write_ply(Path(ply_path), h2, d2)


def read_fields(ply_path, names=WITNESS_FIELDS) -> Dict[str, np.ndarray]:
    from correction.session import read_ply
    _, data = read_ply(Path(ply_path))
    have = data.dtype.names or ()
    missing = [n for n in names if n not in have]
    if missing:
        raise RuntimeError(f"{ply_path}: witness field(s) {missing} missing — run the "
                           f"witness stage (reconstruction.witness.run) first")
    return {n: np.asarray(data[n]) for n in names}
