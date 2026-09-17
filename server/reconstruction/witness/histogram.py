"""The vote histogram of a cloud that already carries its witnesses (§6).

The validation kit's "Votes" mode paints a point red when fewer than
``threshold`` of its neighbouring keyframes agreed with its depth. The user
asked the only question that matters in front of that view — *how many points
is that, and what share of the cloud* — and the answer must come from the same
field the shader reads, not from a second estimate.

Only the ``mv_votes`` and ``status`` columns are touched: the PLY is memory
mapped and the two uint8 columns are read with a stride, so an 850 MB cloud
costs one pass and no allocation of the geometry. The result is cached next to
the cloud and invalidated by (size, mtime), because every stage that rewrites
the cloud — the net, the geometric cleanup, an epoch swap — changes both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from reconstruction.witness.status import STATUS_CODES, STATUS_NAMES

CACHE_NAME = "witness_histogram.json"
_PLY_SIZE = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
             "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
             "int": 4, "uint": 4, "int32": 4, "uint32": 4,
             "float": 4, "float32": 4, "double": 8, "float64": 8}


def _layout(path: Path):
    """(n_points, data_offset, stride, {name: byte offset}) of a binary PLY."""
    off = 0
    n = 0
    stride = 0
    at: Dict[str, int] = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"{path}: truncated PLY header")
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("property"):
                _, ty, name = s.split()[:3]
                if ty not in _PLY_SIZE:
                    raise RuntimeError(f"{path}: unsupported PLY property type {ty!r}")
                at[name] = stride
                stride += _PLY_SIZE[ty]
            elif s == "end_header":
                off = f.tell()
                break
    return n, off, stride, at


def vote_histogram(ply_path, max_votes: int = 16) -> dict:
    """Points per mv_votes value, split by status, for one cloud.

    Every count is exact — the whole column is read, nothing is sampled.
    """
    path = Path(ply_path)
    n, off, stride, at = _layout(path)
    for name in ("mv_votes", "status"):
        if name not in at:
            raise RuntimeError(f"{path}: no {name} field — the cloud carries no witnesses")
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(n, stride))
    mv = np.ascontiguousarray(raw[:, at["mv_votes"]])
    st = np.ascontiguousarray(raw[:, at["status"]])
    top = int(mv.max()) + 1 if n else 1
    counts = np.bincount(mv, minlength=min(max(top, 1), max_votes)).tolist()
    by_status = {}
    for name, code in STATUS_CODES.items():
        c = int((st == code).sum())
        if c:
            by_status[name] = c
    # per (votes, status): what a threshold would take, and what it was labelled
    table = []
    for v in range(len(counts)):
        if not counts[v]:
            continue
        m = mv == v
        row = {"votes": v, "points": int(counts[v]),
               "status": {STATUS_NAMES[int(s)]: int(((st == s) & m).sum())
                          for s in np.unique(st[m])}}
        table.append(row)
    return {"version": 1, "provenance": "tool_measured",
            "cloud": path.name, "n_points": int(n),
            "max_votes": int(top - 1) if n else 0,
            "counts": counts, "status_counts": by_status, "table": table,
            "cumulative_below": np.cumsum(counts).tolist()}


def below(hist: dict, threshold: int) -> dict:
    """What a threshold would remove: the kit's red band, counted."""
    cum: List[int] = hist.get("cumulative_below") or []
    n = int(hist.get("n_points") or 0)
    t = max(0, int(threshold))
    pts = int(cum[t - 1]) if 0 < t <= len(cum) else (0 if t == 0 else n)
    return {"threshold": t, "points": pts, "remaining": n - pts,
            "fraction": (pts / n) if n else 0.0}


def cached(output_dir, cloud_name: str = "cleaned_cloud.ply",
           log=None) -> Optional[dict]:
    """The histogram of the session's cloud, recomputed only when it changed."""
    out = Path(output_dir)
    ply = out / cloud_name
    if not ply.exists():
        return None
    stat = ply.stat()
    stamp = {"size": int(stat.st_size), "mtime": int(stat.st_mtime)}
    cache = out / CACHE_NAME
    if cache.exists():
        try:
            prev = json.loads(cache.read_text())
            if prev.get("source") == stamp:
                return prev
        except Exception:
            pass
    try:
        hist = vote_histogram(ply)
    except RuntimeError as e:
        if log:
            log(f"[witness] {e}")
        return None
    hist["source"] = stamp
    try:
        cache.write_text(json.dumps(hist, indent=1))
    except OSError:
        pass
    return hist
