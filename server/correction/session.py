"""Immutable session snapshot for a correction run.

What it loads (once, read-only): ``cleaned_cloud.ply`` (+ header + full
structured array), the optional ``cleaned_cloud_raw.ply``, per-point provenance
(``frame_global``, ``pixel_row``, ``pixel_col`` — mandatory), the keyframe list
(``camera_frames.txt``, parsed through ``segmentation.session_io``), the
canonical ``camera_poses.txt`` (4x4 c2w per keyframe) and the per-frame camera
centres. It also resolves every existing pose-file COPY that a correction must
rewrite alongside the canonical file (the scale_align/orient precedent).

What it decides: nothing. It fails fast with an actionable message when a
mandatory input is missing or inconsistent, and guarantees the point→keyframe
mapping is total for the loaded cloud (integrity is a gate, but an unmappable
POSE table is a session defect reported here).
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_SERVER_DIR = str(Path(__file__).resolve().parents[1])
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from segmentation.session_io import _load_frame_index_map  # noqa: E402

# Pose copies a correction rewrites when present (scale_align.py:322 precedent:
# every copy or none). camera_poses_mapanything.json is documented-stale
# (session_io.py) and deliberately left untouched.
POSE_COPY_RELPATHS = (
    "camera_poses.txt",
    "maplong_run/camera_poses.txt",
    "omega_run/camera_poses.txt",
    "da3_run/camera_poses.txt",
)

# ── PLY I/O (binary little-endian, header preserved verbatim) ────────────
_PLY_TYPE = {
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
    'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
    'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
    'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
}


def read_ply(path: Path):
    """(header_lines, structured_array) of a binary-little-endian PLY."""
    header: List[bytes] = []
    props = []
    n = 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"{path}: truncated PLY header")
            header.append(line)
            s = line.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                n = int(s.split()[-1])
            elif s.startswith('property'):
                parts = s.split()
                props.append((parts[2], _PLY_TYPE[parts[1]]))
            elif s == 'end_header':
                break
        data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n).copy()
    return header, data


def write_ply(path: Path, header: List[bytes], data: np.ndarray) -> None:
    """Atomic write (tmp + replace) preserving the original header except the
    vertex count."""
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix=".ply")
    os.close(fd)
    with open(tmp, 'wb') as f:
        for line in header:
            s = line.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                f.write(f"element vertex {len(data)}\n".encode('ascii'))
            else:
                f.write(line)
        f.write(data.tobytes())
    Path(tmp).replace(path)


def read_poses(path: Path) -> np.ndarray:
    """(N,4,4) c2w from a camera_poses.txt (16 floats per line)."""
    poses = []
    for ln in Path(path).read_text().splitlines():
        vals = [float(x) for x in ln.split()]
        if len(vals) == 16:
            poses.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    if not poses:
        raise RuntimeError(f"{path}: no 16-value pose rows found")
    return np.stack(poses)


def write_poses(path: Path, poses: np.ndarray) -> None:
    Path(path).write_text("\n".join(
        " ".join(f"{x:.9f}" for x in P.reshape(-1)) for P in poses) + "\n")


@dataclass(frozen=True)
class CorrectionSession:
    output_dir: Path
    header: List[bytes]                 # cleaned_cloud.ply header (verbatim)
    data: np.ndarray                    # full structured cloud array
    xyz: np.ndarray                     # (N,3) float64 view copy
    fg: np.ndarray                      # (N,) frame_global int64
    ks: np.ndarray                      # (N,) keyframe index (-1 = unmappable)
    frames: List[int]                   # keyframe → real frame number
    kf_index: Dict[int, int]            # real frame number → keyframe
    poses: np.ndarray                   # (n_kf,4,4) canonical c2w
    cam_center: Dict[int, np.ndarray]   # real frame → camera centre
    raw_header: Optional[List[bytes]]   # cleaned_cloud_raw.ply (None if absent)
    raw_data: Optional[np.ndarray]
    pose_copies: List[Path] = field(default_factory=list)  # existing copies

    @property
    def n_points(self) -> int:
        return len(self.xyz)

    @property
    def n_kf(self) -> int:
        return len(self.frames)


def load_session(output_dir) -> CorrectionSession:
    output_dir = Path(output_dir)
    ply_path = output_dir / "cleaned_cloud.ply"
    if not ply_path.exists():
        raise RuntimeError(
            f"{ply_path} does not exist — run the reconstruction before "
            f"correcting")
    header, data = read_ply(ply_path)
    names = data.dtype.names or ()
    for req in ("frame_global", "pixel_row", "pixel_col"):
        if req not in names:
            raise RuntimeError(
                f"cleaned_cloud.ply lacks the '{req}' field — per-point "
                f"provenance is mandatory for correction (re-run the "
                f"CloudCompy stage, which injects the traceability fields)")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)

    frames = _load_frame_index_map(output_dir)
    if not frames:
        raise RuntimeError(
            f"{output_dir}/camera_frames.txt (or frame_list.json) is missing "
            f"— the keyframe↔frame mapping is mandatory for correction")
    kf_index = {int(f): k for k, f in enumerate(frames)}

    poses_path = output_dir / "camera_poses.txt"
    if not poses_path.exists():
        raise RuntimeError(f"{poses_path} does not exist — no camera poses "
                           f"to correct")
    poses = read_poses(poses_path)
    if len(poses) != len(frames):
        raise RuntimeError(
            f"camera_poses.txt has {len(poses)} rows but camera_frames.txt "
            f"lists {len(frames)} keyframes — the session is inconsistent; "
            f"fix the reconstruction artifacts before correcting")
    cam_center = {int(f): poses[k][:3, 3] for f, k in kf_index.items()}

    fmax = int(fg.max()) if len(fg) else -1
    kf_arr = np.full(fmax + 2, -1, dtype=np.int64)
    for f, k in kf_index.items():
        if 0 <= f <= fmax:
            kf_arr[f] = k
    ks = kf_arr[np.clip(fg, 0, fmax)]

    raw_header = raw_data = None
    raw_path = output_dir / "cleaned_cloud_raw.ply"
    if raw_path.exists():
        raw_header, raw_data = read_ply(raw_path)
        if len(raw_data) != len(data):
            raise RuntimeError(
                f"cleaned_cloud_raw.ply has {len(raw_data)} points but "
                f"cleaned_cloud.ply has {len(data)} — they must share point "
                f"order (surface_fit residuals depend on it); regenerate the "
                f"consolidate stage before correcting")

    pose_copies = [output_dir / rel for rel in POSE_COPY_RELPATHS
                   if (output_dir / rel).exists()]

    return CorrectionSession(
        output_dir=output_dir, header=header, data=data, xyz=xyz, fg=fg,
        ks=ks, frames=[int(f) for f in frames], kf_index=kf_index,
        poses=poses, cam_center=cam_center, raw_header=raw_header,
        raw_data=raw_data, pose_copies=pose_copies)
