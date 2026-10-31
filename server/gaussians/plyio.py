"""PLY in and out for the gaussians stage, numpy only (the training env has no Open3D):

  * ``read_ply_vertices`` — the published cloud (binary little-endian, every per-point property
    kept: x y z red green blue confidence frame_global pixel_row pixel_col …);
  * ``write_gaussians_ply`` / ``read_gaussians_ply`` — the 3DGS PLY every splat tool reads
    (x y z nx ny nz f_dc_0..2 f_rest_* opacity scale_0..2 rot_0..3, float32; f_rest channel-major as
    the reference implementation writes it; rot = (w, x, y, z); scales log; opacity logit);
  * ``write_splat`` — the compact .splat (32 bytes per gaussian: position f32×3, scale f32×3,
    colour u8×4 with the opacity, rotation u8×4), the viewer's streaming form;
  * ``write_instances`` / ``read_instances`` — one uint32 instance id per gaussian, in PLY order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_PLY_TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1", "short": "i2", "ushort": "u2",
              "int16": "i2", "uint16": "u2", "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
_PLY_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint",
              "f4": "float", "f8": "double"}
SH_C0 = 0.28209479177387814            # the DC spherical-harmonic coefficient


class PlyError(ValueError):
    pass


def read_ply_header(path: Path) -> Tuple[List[Tuple[str, str]], int, int]:
    """(vertex properties as (name, numpy dtype), vertex count, header length in bytes)."""
    props: List[Tuple[str, str]] = []
    n = 0
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise PlyError(f"{path}: not a PLY")
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise PlyError(f"{path}: header without end_header")
            s = line.decode("ascii", "replace").strip()
            if s.startswith("format"):
                if "binary_little_endian" not in s:
                    raise PlyError(f"{path}: only binary_little_endian PLY is read ({s})")
            elif s.startswith("element"):
                parts = s.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n = int(parts[2])
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                if parts[1] == "list":
                    raise PlyError(f"{path}: list properties are not vertex data")
                props.append((parts[2], "<" + _PLY_TYPES[parts[1]]))
            elif s == "end_header":
                return props, n, f.tell()


def read_ply_vertices(path: Path) -> np.ndarray:
    """The vertex table as a structured array (memory-mapped: a 19 M-point cloud is not copied)."""
    props, n, off = read_ply_header(Path(path))
    dtype = np.dtype(props)
    return np.memmap(str(path), dtype=dtype, mode="r", offset=off, shape=(n,))


def _write_ply(path: Path, data: np.ndarray, comments: Sequence[str] = ()) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in comments]
    lines.append(f"element vertex {len(data)}")
    for name in data.dtype.names:
        kind = np.dtype(data.dtype[name]).str.lstrip("<>|=")
        lines.append(f"property {_PLY_NAMES[kind]} {name}")
    lines.append("end_header")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data).astype(data.dtype.newbyteorder("<"), copy=False).tobytes())
    tmp.replace(path)
    return path


def gaussians_dtype(n_rest: int) -> np.dtype:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(n_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return np.dtype([(n, "<f4") for n in names])


def write_gaussians_ply(path: Path, means: np.ndarray, scales_log: np.ndarray, quats_wxyz: np.ndarray,
                        opacities_logit: np.ndarray, sh0: np.ndarray, shN: Optional[np.ndarray],
                        comments: Sequence[str] = ()) -> Path:
    """``means`` (N,3), ``scales_log`` (N,3), ``quats_wxyz`` (N,4), ``opacities_logit`` (N,),
    ``sh0`` (N,3) the DC coefficients, ``shN`` (N,K,3) the higher degrees or None."""
    N = len(means)
    shN_arr = np.zeros((N, 0, 3), np.float32) if shN is None else np.asarray(shN, np.float32)
    K = shN_arr.shape[1]
    data = np.zeros(N, gaussians_dtype(3 * K))
    data["x"], data["y"], data["z"] = (np.asarray(means[:, i], np.float32) for i in range(3))
    for i, c in enumerate("012"):
        data[f"f_dc_{c}"] = np.asarray(sh0[:, i], np.float32)
    rest = shN_arr.transpose(0, 2, 1).reshape(N, 3 * K)          # channel-major, as the reference writer
    for i in range(3 * K):
        data[f"f_rest_{i}"] = rest[:, i]
    data["opacity"] = np.asarray(opacities_logit, np.float32)
    for i in range(3):
        data[f"scale_{i}"] = np.asarray(scales_log[:, i], np.float32)
    for i in range(4):
        data[f"rot_{i}"] = np.asarray(quats_wxyz[:, i], np.float32)
    return _write_ply(path, data, comments)


def read_gaussians_ply(path: Path) -> Dict[str, np.ndarray]:
    v = read_ply_vertices(path)
    names = set(v.dtype.names)
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    if n_rest % 3:
        raise PlyError(f"{path}: {n_rest} f_rest properties are not three channels")
    K = n_rest // 3
    N = len(v)
    out = {
        "means": np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32),
        "sh0": np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32),
        "opacities_logit": np.asarray(v["opacity"], np.float32),
        "scales_log": np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32),
        "quats_wxyz": np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32),
    }
    rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], axis=1).astype(np.float32) if K else np.zeros((N, 0), np.float32)
    out["shN"] = rest.reshape(N, 3, K).transpose(0, 2, 1) if K else np.zeros((N, 0, 3), np.float32)
    return out


def splat_record(means, scales_log, quats_wxyz, opacities_logit, sh0) -> np.ndarray:
    """The .splat rows: position, scale (exp), colour (SH0 → RGB, opacity sigmoid), rotation
    (unit quaternion w x y z mapped to bytes)."""
    N = len(means)
    rec = np.zeros(N, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("color", "u1", 4), ("rot", "u1", 4)])
    rec["pos"] = np.asarray(means, np.float32)
    rec["scale"] = np.exp(np.asarray(scales_log, np.float32))
    rgb = np.clip(np.asarray(sh0, np.float32) * SH_C0 + 0.5, 0.0, 1.0)
    alpha = 1.0 / (1.0 + np.exp(-np.asarray(opacities_logit, np.float32)))
    rec["color"][:, :3] = np.round(rgb * 255.0).astype(np.uint8)
    rec["color"][:, 3] = np.round(alpha * 255.0).astype(np.uint8)
    q = np.asarray(quats_wxyz, np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rec["rot"] = np.clip(np.round(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
    return rec


def write_splat(path: Path, means, scales_log, quats_wxyz, opacities_logit, sh0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = splat_record(means, scales_log, quats_wxyz, opacities_logit, sh0)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(rec.tobytes())
    tmp.replace(path)
    return path


def read_splat(path: Path) -> np.ndarray:
    dtype = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("color", "u1", 4), ("rot", "u1", 4)])
    return np.fromfile(str(path), dtype=dtype)


def write_instances(path: Path, ids: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(np.asarray(ids).astype("<u4"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(arr.tobytes())
    tmp.replace(path)
    return path


def read_instances(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype="<u4")
