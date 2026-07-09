# STAC-Builder — deterministic upright orientation, baked into the reconstruction.
#
# The VGGT/Omega world frame is the FIRST CAMERA's OpenCV frame (+X right, +Y down,
# +Z forward), so the raw cloud renders upside down in a Y-up viewer. Instead of
# betting on a floor-plane RANSAC (stochastic; failed on onion-corrupted clouds and
# its minimal-delta fallback can never produce the needed ~180° flip), gravity is
# taken from the CAMERA POSES themselves: people film roughly level, so the mean
# camera-down axis (c2w[:3,1], OpenCV +Y = down) IS the world down direction —
# measured at ~3° error on real sessions. The rotation to Y-up plus a floor offset
# (low-percentile Y → 0) is applied IN PLACE to chunk PLYs, camera poses and the
# aligned per-frame depth, exactly like scale_align applies the metric scale, with
# the same idempotency-marker pattern (.orientation_applied).
#
# PROVENANCE: ours. Same PLY/pose file handling as reconstruction/scale_align.py.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("orient")

_PLY_NP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}

MARKER_NAME = ".orientation_applied"


# ── geometry ────────────────────────────────────────────────────────

def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # 180°: rotate around any axis orthogonal to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def _read_poses(path: Path) -> List[np.ndarray]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        v = ln.split()
        if len(v) == 16:
            out.append(np.array(list(map(float, v)), np.float64).reshape(4, 4))
    return out


def estimate_gravity(output_dir: Path, log=None) -> Optional[Tuple[np.ndarray, float]]:
    """World DOWN direction = normalized mean of the cameras' +Y axis in world
    coordinates (OpenCV camera convention: +Y is down in the image). Returns
    (down_unit_vector, mean_alignment) or None when no poses are available.
    mean_alignment (mean |cos| of per-frame down vs the consensus) is the
    confidence diagnostic — near 1.0 = the operator filmed level."""
    _log = log or logger.info
    poses = []
    for base in (output_dir, output_dir / "maplong_run"):
        poses = _read_poses(base / "camera_poses.txt")
        if poses:
            break
    if not poses:
        _log("no camera_poses.txt found — cannot estimate gravity from poses")
        return None
    downs = np.stack([p[:3, 1] for p in poses])            # camera +Y axis in world
    downs = downs / (np.linalg.norm(downs, axis=1, keepdims=True) + 1e-12)
    g = downs.mean(0)
    g = g / (np.linalg.norm(g) + 1e-12)
    align = float(np.abs(downs @ g).mean())
    _log(f"gravity from {len(poses)} camera poses: down={np.round(g, 4).tolist()} "
         f"(mean alignment {align:.3f})")
    return g, align


# ── in-place appliers (mirrors scale_align's file handling) ────────

def _transform_ply_xyz(path: Path, R: np.ndarray, t: np.ndarray) -> bool:
    """x' = R·x + t on every vertex of a PLY, in place. Binary little/big endian
    and ascii. Other properties untouched. False on unexpected layout."""
    raw = path.read_bytes()
    hend = raw.find(b"end_header")
    if hend < 0:
        return False
    nl = raw.find(b"\n", hend)
    header = raw[:nl + 1].decode("ascii", "replace")
    body = raw[nl + 1:]

    fmt = None
    props: List[Tuple[str, str]] = []
    n = 0
    in_vertex = False
    for ln in header.splitlines():
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            in_vertex = (tok[1] == "vertex")
            if in_vertex:
                n = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            props.append((tok[-1], tok[1]))

    names = [p[0] for p in props]
    if not all(c in names for c in ("x", "y", "z")):
        return False

    if fmt == "ascii":
        xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
        out_lines = []
        for bl in body.decode("ascii", "replace").splitlines()[:n]:
            v = bl.split()
            if len(v) >= len(names):
                p = R @ np.array([float(v[xi]), float(v[yi]), float(v[zi])]) + t
                v[xi], v[yi], v[zi] = (f"{p[0]:.8g}", f"{p[1]:.8g}", f"{p[2]:.8g}")
            out_lines.append(" ".join(v))
        path.write_bytes(raw[:nl + 1] + ("\n".join(out_lines) + "\n").encode("ascii"))
        return True

    endian = "<" if (fmt and "little" in fmt) else ">"
    try:
        dt = np.dtype([(nm, endian + _PLY_NP[ty]) for nm, ty in props])
    except KeyError:
        return False
    nbytes = n * dt.itemsize
    arr = np.frombuffer(body[:nbytes], dtype=dt).copy()
    xyz = np.stack([arr["x"].astype(np.float64),
                    arr["y"].astype(np.float64),
                    arr["z"].astype(np.float64)], axis=1)
    xyz = xyz @ R.T + t
    for i, c in enumerate(("x", "y", "z")):
        arr[c] = xyz[:, i].astype(arr[c].dtype)
    path.write_bytes(raw[:nl + 1] + arr.tobytes() + body[nbytes:])
    return True


def _ply_y_percentile(path: Path, R: np.ndarray, pct: float) -> Optional[np.ndarray]:
    """Y values of a PLY's vertices AFTER rotation R (no translation), subsampled.
    Returns the array (may be empty) or None on unexpected layout."""
    raw = path.read_bytes()
    hend = raw.find(b"end_header")
    if hend < 0:
        return None
    nl = raw.find(b"\n", hend)
    header = raw[:nl + 1].decode("ascii", "replace")
    body = raw[nl + 1:]
    fmt, props, n, in_vertex = None, [], 0, False
    for ln in header.splitlines():
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            in_vertex = (tok[1] == "vertex")
            if in_vertex:
                n = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            props.append((tok[-1], tok[1]))
    names = [p[0] for p in props]
    if fmt == "ascii" or not all(c in names for c in ("x", "y", "z")):
        return None                       # ascii chunks are rare; caller falls back
    endian = "<" if (fmt and "little" in fmt) else ">"
    try:
        dt = np.dtype([(nm, endian + _PLY_NP[ty]) for nm, ty in props])
    except KeyError:
        return None
    arr = np.frombuffer(body[:n * dt.itemsize], dtype=dt)
    step = max(1, n // 200_000)           # ≤200k samples per chunk is plenty for a percentile
    xyz = np.stack([arr["x"][::step].astype(np.float64),
                    arr["y"][::step].astype(np.float64),
                    arr["z"][::step].astype(np.float64)], axis=1)
    return (xyz @ R.T)[:, 1]


def _transform_poses_file(path: Path, R: np.ndarray, t: np.ndarray,
                          backup_suffix: str = ".txt.preorient") -> None:
    if not path.exists():
        return
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    out = []
    for ln in path.read_text().splitlines():
        v = ln.split()
        if len(v) == 16:
            m = np.array(list(map(float, v)), np.float64).reshape(4, 4)
            m = T @ m
            ln = " ".join(f"{x:.8g}" for x in m.reshape(-1))
        out.append(ln)
    bak = path.with_suffix(backup_suffix)
    if not bak.exists():
        shutil.copy(path, bak)
    path.write_text("\n".join(out) + "\n")
    logger.info(f"  oriented {path} (backup {bak.name})")


def _transform_aligned_worldpoints(output_dir: Path, R: np.ndarray, t: np.ndarray,
                                   log=None) -> None:
    """Rotate the aligned per-frame world_points the TSDF reads. Per-frame 'depth'
    is view-axis distance — invariant under a global rigid transform (points and
    poses rotate together) — so only world_points change."""
    _log = log or logger.info
    aligned = output_dir / "maplong_run" / "_tmp_results_aligned"
    if not aligned.exists():
        return
    for p in sorted(aligned.glob("chunk_*.npy")):
        try:
            d = np.load(str(p), allow_pickle=True).item()
        except Exception as e:
            _log(f"could not load {p.name}: {e}")
            continue
        wp = d.get("world_points")
        if wp is not None:
            wp = np.asarray(wp, np.float64)
            d["world_points"] = (wp @ R.T + t).astype(np.float32)
            np.save(str(p), d)
            _log(f"oriented world_points {p.name}")


# ── entry point ─────────────────────────────────────────────────────

def run(output_dir: Path, floor_percentile: float = 1.0, log=None) -> Optional[np.ndarray]:
    """Estimate gravity from the camera poses and bake the upright orientation
    (down → -Y, floor at y≈0) into chunk PLYs + poses + aligned world_points.
    Idempotent via the .orientation_applied marker. Returns the applied 4x4
    (or the identity read back from the marker) — None only on genuine failure."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir = Path(output_dir)
    marker = output_dir / MARKER_NAME
    if marker.exists():
        _log(f"ALREADY ORIENTED: marker present ({marker.read_text().strip()}) — skipping "
             f"(Replace clears the marker to re-orient)")
        return np.eye(4)

    est = estimate_gravity(output_dir, log=_log)
    if est is None:
        return None
    g_down, align = est
    if align < 0.7:
        # Wild camera motion (e.g. inspection under a vehicle) — the consensus down
        # is unreliable; refuse rather than bake a wrong orientation.
        _log(f"REFUSED: camera-down consensus too weak (alignment {align:.3f} < 0.7) — "
             f"leaving the reconstruction frame untouched")
        return None

    R = _rotation_between(g_down, np.array([0.0, -1.0, 0.0]))

    # Floor offset: low-percentile Y after rotation → y=0
    ys = []
    for ply in sorted(output_dir.glob("chunk_*.ply")):
        y = _ply_y_percentile(ply, R, floor_percentile)
        if y is not None and len(y):
            ys.append(y)
    t = np.zeros(3)
    if ys:
        floor_y = float(np.percentile(np.concatenate(ys), floor_percentile))
        t[1] = -floor_y
        _log(f"floor at rotated y={floor_y:.3f} → offset {-floor_y:+.3f} (p{floor_percentile:g})")
    else:
        _log("no chunk PLYs sampled for the floor offset — rotation only")

    # Clouds first; a failure RAISES before poses are touched (same safety order as
    # scale_align: geometry and cameras never de-sync).
    n_ply = 0
    for ply in sorted(output_dir.glob("chunk_*.ply")):
        if not _transform_ply_xyz(ply, R, t):
            raise RuntimeError(f"could not orient {ply.name} (unexpected PLY layout) — "
                               f"aborting so cameras and cloud never de-sync")
        n_ply += 1
        _log(f"oriented cloud {ply.name}")
    _transform_aligned_worldpoints(output_dir, R, t, log=_log)
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        _transform_poses_file(base / "camera_poses.txt", R, t)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    marker.write_text(f"angle_deg={ang:.2f} ty={t[1]:.4f} align={align:.3f}\n")
    _log(f"✅ upright orientation baked: rotation {ang:.1f}°, floor offset {t[1]:+.3f} m, "
         f"{n_ply} chunk(s)")
    return T
