# STAC-Builder — Phase R: per-instance geometry primitives.
#
# Depth-lift (mask -> 3D), KNN outlier filtering, and gravity-aligned OBB fit.
#
# PROVENANCE: PORTED/ADAPTED FROM R3D
#   vendor/r3d/r3d/pipeline/scripts/build_scene.py
#     _lift_depth_to_3d (:47), _filter_outliers_knn (:82),
#     _fit_gravity_aligned_obb (:191)
#   and vendor/r3d/r3d/pipeline/eval/tool_use.py
#     _invert_rigid (:57), _closest_point_on_obb (:64)
# Our changes vs R3D:
#   - aggressive, configurable mask EROSION before lifting (spec R.2: our depth
#     is DA3, not a sensor — only mask interiors vote);
#   - GRAVITY-vector support (R3D assumes world Y-up; our BA poses are not, so we
#     rotate points into a gravity frame first — the gravity comes from R.4's
#     floor/slab plane fit crossed with ChArUco);
#   - returns points in WORLD, keyed to our session_io c2w convention.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ── depth-lift ──────────────────────────────────────────────────────

def erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Aggressive, configurable erosion (spec R.2 default 5-10 px). Only mask
    interiors feed the vote — DA3 depth is unreliable at mask borders."""
    if erosion_px <= 0 or cv2 is None:
        return mask.astype(bool)
    k = 2 * int(erosion_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (mask > 0).astype(np.uint8)
    return cv2.erode(m, kernel, iterations=1).astype(bool)


def lift_masked_depth_to_world(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    stride: int = 4,
    erosion_px: int = 8,
    max_depth: float | None = None,
) -> np.ndarray:
    """Lift the eroded mask interior to 3D world points (N,3).

    Ported from R3D `_lift_depth_to_3d` (build_scene.py:47-79); adds mask erosion
    and works at the mask/depth resolution (depth and mask must share H,W; K is
    scaled to that resolution by the caller).
    """
    h, w = depth.shape[:2]
    if mask.shape[:2] != (h, w) and cv2 is not None:
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    m = erode_mask(mask, erosion_px)

    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us = us.ravel()
    vs = vs.ravel()
    d = depth[vs, us].astype(np.float64)
    msub = m[vs, us]
    valid = msub & (d > 0) & np.isfinite(d)
    if max_depth is not None:
        valid &= d < max_depth
    if not np.any(valid):
        return np.empty((0, 3), np.float64)
    us, vs, d = us[valid], vs[valid], d[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us - cx) / fx * d
    y = (vs - cy) / fy * d
    z = d
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (N,4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world


# ── KNN outlier filter (ported R3D _filter_outliers_knn) ────────────

def filter_outliers_knn(points: np.ndarray, k: int = 6, factor: float = 3.0) -> np.ndarray:
    """Keep points whose mean distance to k nearest neighbours is < factor ×
    median NN distance. Returns a boolean keep-mask. No-op for tiny sets.
    Ported from R3D build_scene.py:82-94."""
    n = len(points)
    if n <= max(k, 20):
        return np.ones(n, bool)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)  # first col is self (0)
    mean_nn = dists[:, 1:].mean(axis=1)
    median = np.median(mean_nn)
    return mean_nn < factor * median


# ── gravity-aligned OBB (ported/adapted R3D _fit_gravity_aligned_obb) ─

def _gravity_rotation(gravity: np.ndarray | None) -> np.ndarray:
    """Rotation R mapping world -> a frame whose +Y is the up (anti-gravity)
    axis, so the R3D XZ/Y split applies. Identity when gravity is +Y already."""
    if gravity is None:
        return np.eye(3)
    up = -np.asarray(gravity, float)
    n = np.linalg.norm(up)
    if n < 1e-9:
        return np.eye(3)
    up = up / n
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    c = float(np.dot(up, y))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return R  # maps up -> +Y


def fit_gravity_aligned_obb(points: np.ndarray, gravity: np.ndarray | None = None,
                            trim_percent: float = 0.0):
    """Fit a gravity-aligned OBB. Returns (transform 4x4, aabb 6, position 3),
    aabb centered (±extent/2), placement in transform. Vertical axis is up.
    Ported/adapted from R3D build_scene.py:191-257 (+ gravity rotation).

    trim_percent: robust extents via the [p, 100-p] percentiles instead of
    min/max — depth-lifted points are noisy and raw min/max inflates the box
    (a floor's thickness read 0.47 m instead of ~0.07 m) and shifts its
    centre. 0 keeps the exact hull behaviour."""
    if len(points) < 3:
        return None
    R = _gravity_rotation(gravity)
    P = points @ R.T  # into gravity frame (Y up)

    def _rng(v: np.ndarray) -> tuple[float, float]:
        if trim_percent > 0:
            return (float(np.percentile(v, trim_percent)),
                    float(np.percentile(v, 100 - trim_percent)))
        return float(v.min()), float(v.max())

    y_min, y_max = _rng(P[:, 1])
    xz = P[:, [0, 2]]
    mean_xz = xz.mean(axis=0)
    cov = np.cov((xz - mean_xz).T)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes_2d = eigvecs[:, np.argsort(-eigvals)]  # principal axes in XZ
    proj = (xz - mean_xz) @ axes_2d
    lo0, hi0 = _rng(proj[:, 0])
    lo1, hi1 = _rng(proj[:, 1])
    p_min, p_max = np.array([lo0, lo1]), np.array([hi0, hi1])
    ex = p_max[0] - p_min[0]
    ey = y_max - y_min
    ez = p_max[1] - p_min[1]

    # in-plane center (gravity frame)
    center_xz = mean_xz + axes_2d @ ((p_min + p_max) / 2.0)
    center_y = (y_min + y_max) / 2.0
    center_g = np.array([center_xz[0], center_y, center_xz[1]])

    # frame axes in gravity frame: e0 (xz principal), e1 = +Y, e2 (xz secondary)
    e0 = np.array([axes_2d[0, 0], 0.0, axes_2d[1, 0]])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([axes_2d[0, 1], 0.0, axes_2d[1, 1]])
    Rg = np.stack([e0, e1, e2], axis=1)  # columns = axes (gravity frame)

    # back to world
    Rw = R.T @ Rg
    center_w = R.T @ center_g
    T = np.eye(4)
    T[:3, :3] = Rw
    T[:3, 3] = center_w
    aabb = np.array([-ex / 2, ex / 2, -ey / 2, ey / 2, -ez / 2, ez / 2])
    return T, aabb, center_w


# ── OBB point math (ported R3D tool_use helpers) ────────────────────

def invert_rigid(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def obb_local_coords(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World points -> OBB-local coordinates (centered frame)."""
    Tinv = invert_rigid(T)
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (Tinv @ ph.T).T[:, :3]


def signed_distances_along_axis(points: np.ndarray, T: np.ndarray, axis: int = 2) -> np.ndarray:
    """Signed coordinate of each point along an OBB axis (default the OBB's
    short/depth axis, index 2) — the 1-D distribution the onion detector uses."""
    local = obb_local_coords(points, T)
    return local[:, axis]


def unproject_pixel_to_world(
    u: float, v: float, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray,
    win: int = 3, max_depth: float | None = None,
) -> np.ndarray | None:
    """Back-project a single pixel (u,v, in the depth image's resolution) to a
    world point using a small median depth window for robustness. Returns None
    if no valid depth is available under the window. Used by Phase 3 to anchor a
    2D finding to a 3D coordinate via the R-refined pose (same math as the mask
    lift, one pixel)."""
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    u0, u1 = max(0, ui - win), min(w, ui + win + 1)
    v0, v1 = max(0, vi - win), min(h, vi + win + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    valid = (patch > 0) & np.isfinite(patch)
    if max_depth is not None:
        valid &= patch < max_depth
    if not np.any(valid):
        return None
    d = float(np.median(patch[valid]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (ui - cx) / fx * d
    y = (vi - cy) / fy * d
    p_cam = np.array([x, y, d, 1.0])
    return (c2w @ p_cam)[:3]


# ── shared camera/frame helpers (moved from build_instances when the anchoring
#    machinery was removed, 2026-07-09; consumed by phase3_findings) ─────────

def scaled_K(K, from_wh, to_wh):
    """Scale intrinsics from one resolution to another (width,height)."""
    import numpy as _np
    (fw, fh), (tw, th) = from_wh, to_wh
    sx, sy = tw / fw, th / fh
    Ks = _np.asarray(K, float).copy()
    Ks[0, 0] *= sx; Ks[0, 2] *= sx
    Ks[1, 1] *= sy; Ks[1, 2] *= sy
    return Ks


def frame_size(frames_dir, fid, mask_shape):
    """(w, h) of the real frame JPG; falls back to the mask's shape."""
    from pathlib import Path as _P
    p = _P(frames_dir) / f"{int(fid):06d}.jpg"
    if p.exists():
        try:
            from PIL import Image
            with Image.open(p) as im:
                return im.size
        except Exception:
            pass
    h, w = mask_shape[:2]
    return (w, h)
