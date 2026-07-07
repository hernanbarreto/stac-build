# STAC-Builder — Phase R.5: class-conditioned depth regularization.
#
# ONLY inside the eroded mask of a WHITELISTED structural class (wall/slab/
# column/vault/platform/floor) do we regularize DA3 depth toward the fitted
# primitive (plane/cylinder). NEVER for non-whitelisted classes (cables, pipes,
# equipment, catenary — they are not flattened). The class is proposed by the
# VLM (vlm_proposed); the whitelist decides if it applies; the deterministic
# primitive provides the value.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import numpy as np

DEFAULT_WHITELIST = {"wall", "slab", "column", "vault", "platform", "floor"}


def fit_plane(points: np.ndarray):
    """Least-squares plane (unit normal n, offset d) s.t. n·x + d = 0."""
    c = points.mean(0)
    _u, _s, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[2]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ c)
    return n, d


def _plane_depth_map(plane, K, wh):
    """Predicted depth per pixel where the camera ray meets the plane (camera
    frame plane n·X + d = 0). Returns (H,W) with NaN where the ray is parallel."""
    n, d = plane
    w, h = wh
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    dirs = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], axis=-1)
    denom = dirs @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -d / denom  # since n·(z·dir) + d = 0 -> z = -d/(n·dir)
    z[np.abs(denom) < 1e-6] = np.nan
    z[z <= 0] = np.nan
    return z


def regularize_depth_to_plane(depth: np.ndarray, mask: np.ndarray, plane,
                              K: np.ndarray, weight: float = 0.5,
                              label: str | None = None,
                              whitelist: set[str] | None = None) -> np.ndarray:
    """Blend depth toward the plane-predicted depth inside the mask. No-op if the
    class is not whitelisted. weight in [0,1] (0 = unchanged, 1 = snap to plane)."""
    wl = whitelist or DEFAULT_WHITELIST
    if label is not None and label not in wl:
        return depth  # never flatten non-structural classes
    h, w = depth.shape[:2]
    pred = _plane_depth_map(plane, K, (w, h))
    out = depth.copy().astype(float)
    m = (mask > 0) & np.isfinite(pred) & (depth > 0)
    out[m] = (1 - weight) * depth[m] + weight * pred[m]
    return out
