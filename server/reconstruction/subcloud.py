"""
Confidence-aware sub-cloud helpers.
===================================

DA3 / VGGT-Long / hybrid-LiDAR clouds carry a per-point ``confidence``
channel in ``[0, 1]`` (ARKit's integer ``{0,1,2}`` is normalised ``/2.0``
upstream in ``workers/map_worker.py``). Reconstruction steps that fit
geometry — OBB, RANSAC plane/cylinder, shrink-wrap target — should run on
the *high-confidence* subset so a few noisy points don't blow up the fit.
Classification and rendering still use the full sub-cloud.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ── PLY scalar type map (mirrors pipeline.py) ──
_PLY_TYPE = {
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
    'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
    'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
    'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
}


def load_ply_xyz_conf(ply_path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Load ``xyz`` (N,3 float64), ``confidence`` (N, float32 in [0,1]) and ``rgb``
    (N,3 float32 in [0,1]) from a binary PLY.

    Returns ``(xyz, conf, rgb)``; either ``conf`` or ``rgb`` may be ``None`` if the
    PLY lacks the channel. Returns ``None`` if the file can't be parsed. Reads the
    header dynamically so extra fields (origins, normals, ...) are tolerated.
    """
    try:
        with open(ply_path, 'rb') as f:
            n_pts = 0
            props = []
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_pts = int(line.split()[-1])
                elif line.startswith('property') and n_pts > 0:
                    parts = line.split()
                    if len(parts) >= 3:
                        np_type = _PLY_TYPE.get(parts[1])
                        if np_type:
                            props.append((parts[2], np_type))
                elif line == 'end_header':
                    break
            if n_pts == 0:
                return None
            dtype = np.dtype(props)
            data = np.frombuffer(f.read(), dtype=dtype)
            names = {p[0] for p in props}
            xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float64)
            conf = data['confidence'].astype(np.float32) if 'confidence' in names else None
            rgb: Optional[np.ndarray] = None
            if {'red', 'green', 'blue'}.issubset(names):
                r = data['red']; g = data['green']; b = data['blue']
                rgb = np.column_stack([r, g, b]).astype(np.float32)
                if rgb.max() > 1.5:                      # uchar [0,255] → [0,1]
                    rgb = rgb / 255.0
            return xyz, conf, rgb
    except Exception as e:  # noqa: BLE001
        print(f"[Reconstruction] Error reading PLY {ply_path}: {e}")
        return None


def high_confidence_subcloud(
    conf: Optional[np.ndarray],
    indices: np.ndarray,
    abs_thresh: float = 0.5,
    pct: float = 60.0,
    min_pts: int = 200,
) -> np.ndarray:
    """Return the high-confidence subset of ``indices`` for one segment.

    Policy:
      1. If at least ``min_pts`` points clear the absolute bar ``abs_thresh``
         (DA3/LiDAR confidence is in [0,1], with ~0.5 meaning "decent"), use
         exactly those.
      2. Otherwise the segment is mostly low-confidence — keep the top
         ``(100 - pct)%`` of points by confidence rank, but never fewer than
         ``min_pts`` (clamped to the segment size). This guarantees RANSAC /
         OBB always have inliers without letting noise dominate.

    Used for OBB fitting, RANSAC plane/cylinder fits, and shrink-wrap targets.
    *Not* for classification (uses the full sub-cloud) or rendering.

    Args:
        conf: per-point confidence for the WHOLE cloud, or ``None`` → no
            filtering, ``indices`` returned unchanged.
        indices: int array — global point indices of this segment.
        abs_thresh: absolute confidence cutoff in [0,1].
        pct: percentile knob — when falling back to rank, keep the top
            ``(100 - pct)%`` of points.
        min_pts: minimum number of points to keep (floor).

    Returns:
        int array — a sorted subset of ``indices``.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    if conf is None or n == 0:
        return indices
    seg_conf = np.asarray(conf)[indices].astype(np.float64)
    finite = np.isfinite(seg_conf)
    if not finite.any() or float(seg_conf[finite].max()) <= 0.0:
        return indices  # no usable confidence channel → keep all
    seg_conf = np.where(finite, seg_conf, -np.inf)

    keep = seg_conf > abs_thresh
    if int(keep.sum()) >= min_pts:
        return indices[keep]

    frac_keep = max(0.0, min(1.0, 1.0 - pct / 100.0))
    n_keep = min(n, max(min_pts, int(np.ceil(n * frac_keep))))
    order = np.argsort(seg_conf)[::-1]  # high → low confidence
    return indices[np.sort(order[:n_keep])]
