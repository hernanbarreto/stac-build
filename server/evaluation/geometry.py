"""Pure geometry of the battery (numpy only, no session): rays, unprojection, triangulation,
chainage and the gauge scale. Pixel convention of the chain: integer coordinates are pixel
centres, u = column, v = row (precision/camera.py)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def normalized_rays(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N,3) camera-frame directions with z = 1 for undistorted pixels ``uv``."""
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    return np.stack([x, y, np.ones_like(x)], axis=1)


def unproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World points of undistorted pixels ``uv`` at camera depths ``z``:
    X = c2w · (z · K⁻¹ [u, v, 1]) — the chain's own formula (precision/depth_on_f5.py)."""
    d = normalized_rays(uv, K) * np.asarray(z, np.float64).reshape(-1, 1)
    return d @ c2w[:3, :3].T + c2w[:3, 3]


def project(X: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(uv undistorted (N,2), z (N,)) of world points in the camera of ``c2w``."""
    w2c = np.linalg.inv(c2w)
    Xc = np.atleast_2d(np.asarray(X, np.float64)) @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def camera_depth(X: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Depth (camera z) of world points in the camera of ``c2w``."""
    w2c = np.linalg.inv(c2w)
    return (np.atleast_2d(np.asarray(X, np.float64)) @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]


def world_ray(uv: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(origin (3,), unit directions (N,3)) in the world of undistorted pixels."""
    d = normalized_rays(uv, K) @ c2w[:3, :3].T
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return c2w[:3, 3].copy(), d


def triangulate_rays(origins: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, float]:
    """The point closest (least squares) to n ≥ 2 rays and the largest pairwise angle between
    the rays (degrees) — the parallax the point rests on."""
    origins = np.atleast_2d(np.asarray(origins, np.float64))
    dirs = np.atleast_2d(np.asarray(dirs, np.float64))
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            c = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            best = max(best, float(np.degrees(np.arccos(c))))
    return X, best


def chainage(c2w: np.ndarray) -> np.ndarray:
    """Cumulative distance walked by the camera, keyframe by keyframe (0 at the first)."""
    c = np.asarray(c2w, np.float64)[:, :3, 3]
    if len(c) == 0:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def median_point(P: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """(component-wise median, RMS and max distance of the samples to it)."""
    P = np.atleast_2d(np.asarray(P, np.float64))
    m = np.median(P, axis=0)
    d = np.linalg.norm(P - m, axis=1)
    return m, float(np.sqrt(np.mean(d * d))), float(d.max())


def gauge_scale(measured: Sequence[float], true: Sequence[float], sigma: Sequence[float]
                ) -> Tuple[float, str]:
    """Least-squares scale s minimising Σ w (s·m − t)², w = 1/σ² (a σ of 0 takes the smallest
    positive σ; all zero → equal weights). Returns (s, the weighting used)."""
    m = np.asarray(measured, np.float64)
    t = np.asarray(true, np.float64)
    s_ = np.asarray(sigma, np.float64)
    if m.size == 0:
        raise ValueError("gauge_scale: no gauge distance")
    pos = s_[s_ > 0]
    if pos.size == 0:
        w = np.ones_like(m)
        how = "equal (every gauge sigma is 0)"
    else:
        s_eff = np.where(s_ > 0, s_, pos.min())
        w = 1.0 / (s_eff * s_eff)
        how = "1/sigma^2" + (" (a sigma of 0 takes the smallest positive one)"
                             if (s_ <= 0).any() else "")
    denom = float(np.sum(w * m * m))
    if denom <= 0:
        raise ValueError("gauge_scale: the measured gauge distances are all 0")
    return float(np.sum(w * m * t) / denom), how


def percentile(x: Sequence[float], q: float) -> Optional[float]:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], np.float64)
    if a.size == 0:
        return None
    return float(np.percentile(a, q))
