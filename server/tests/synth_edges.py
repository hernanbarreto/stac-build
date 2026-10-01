"""Synthetic scenes for the edge-definition metric (precision/edge_metric.py).

A closed box sampled on a jittered grid, sharp or with every edge and corner
rounded by a fillet of KNOWN radius (the rounded box is the Minkowski sum of a
smaller box and a sphere), out-of-surface Gaussian noise, plus the two edge
defects the metric must see: an eroded edge band (the confidence floor's
signature) and a mixed-pixel skirt hanging off every edge. Written as the
pipeline writes clouds: binary PLY with per-point provenance keys
(frame_global, pixel_row, pixel_col) and a one-instance segmentation_result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

HALF = (0.2, 0.125, 0.15)          # box half-extents (m)
SPACING = 0.005                    # grid step of the surface sampling (m)


def box(half: Sequence[float] = HALF, h: float = SPACING, r: float = 0.0,
        noise: float = 0.001, seed: int = 0) -> np.ndarray:
    """Points on a closed box surface; ``r`` > 0 rounds every edge/corner."""
    rng = np.random.default_rng(seed)
    a = np.asarray(half, np.float64)
    pts, nrm = [], []
    for ax in range(3):
        for sg in (-1.0, 1.0):
            u, v = [k for k in range(3) if k != ax]
            nu, nv = int(round(2 * a[u] / h)), int(round(2 * a[v] / h))
            gu, gv = np.meshgrid((np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv,
                                 indexing="ij")
            gu = gu.ravel() + rng.uniform(-0.35, 0.35, gu.size) / nu
            gv = gv.ravel() + rng.uniform(-0.35, 0.35, gv.size) / nv
            P = np.zeros((gu.size, 3))
            P[:, ax] = sg * a[ax]
            P[:, u] = -a[u] + 2 * a[u] * gu
            P[:, v] = -a[v] + 2 * a[v] * gv
            N = np.zeros_like(P)
            N[:, ax] = sg
            pts.append(P)
            nrm.append(N)
    P, N = np.concatenate(pts), np.concatenate(nrm)
    if r > 0:
        inner = a - r
        c = np.clip(P, -inner, inner)
        d = P - c
        N = d / np.linalg.norm(d, axis=1, keepdims=True)
        P = c + r * N
    return P + noise * rng.standard_normal(len(P))[:, None] * N


def _edges(half: Sequence[float]):
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                yield i, si, j, sj, k


def skirt(half: Sequence[float] = HALF, h: float = SPACING, t_max: float = 0.008,
          dt: float = 0.001, seed: int = 1) -> np.ndarray:
    """Mixed-pixel skirt: from every edge, a sheet of points along a 45° view
    ray past the edge — in front of one face, below the other's plane."""
    a = np.asarray(half, np.float64)
    rng = np.random.default_rng(seed)
    out = []
    for i, si, j, sj, k in _edges(half):
        tk = np.arange(-a[k] + h / 2, a[k], h)
        d = np.zeros(3)
        d[j], d[i] = sj, -si
        d /= np.linalg.norm(d)
        for t in np.arange(dt, t_max + 1e-12, dt):
            p = np.zeros((len(tk), 3))
            p[:, i], p[:, j], p[:, k] = si * a[i], sj * a[j], tk
            out.append(p + t * d + 0.0003 * rng.standard_normal(p.shape))
    return np.concatenate(out)


def eroded_keep(P: np.ndarray, half: Sequence[float] = HALF, width: float = 0.012) -> np.ndarray:
    """Mask of the points kept when every edge loses a band ``width`` wide on
    each face (what the per-view confidence floor does to occluding contours)."""
    near = (np.abs(np.abs(P) - np.asarray(half)) < width).sum(axis=1) >= 2
    return ~near


def write_ply(path: Path, P: np.ndarray, keys: Optional[np.ndarray] = None) -> Path:
    """Binary little-endian PLY as gpu_cloud_clean writes it (x y z confidence
    [frame_global pixel_row pixel_col])."""
    n = len(P)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("confidence", "<f4")]
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
           "property float x", "property float y", "property float z",
           "property float confidence"]
    if keys is not None:
        fields += [("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")]
        hdr += ["property int frame_global", "property short pixel_row",
                "property short pixel_col"]
    hdr.append("end_header")
    arr = np.empty(n, np.dtype(fields))
    arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
    arr["confidence"] = 1.0
    if keys is not None:
        keys = np.asarray(keys, np.int64)
        arr["frame_global"] = keys // 10000
        arr["pixel_row"] = (keys % 10000) // 100
        arr["pixel_col"] = keys % 100
    path = Path(path)
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        arr.tofile(f)
    return path


def write_seg(path: Path, n: int, instance_id: int = 1, label: str = "box") -> Path:
    path = Path(path)
    path.write_text(json.dumps({"instances": [
        {"instance_id": instance_id, "label": label, "globalIndices": list(range(int(n)))}]}))
    return path


def scene(tmp: Path, name: str, P: np.ndarray, keys: Optional[np.ndarray]) -> Tuple[Path, Path]:
    tmp = Path(tmp)
    return (write_ply(tmp / f"{name}.ply", P, keys), write_seg(tmp / f"{name}_seg.json", len(P)))
