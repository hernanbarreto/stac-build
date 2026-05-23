"""
Visual hull & silhouette tools (CPU, no GPU / no display).
==========================================================

From the camera poses (`camera_poses.txt` c2w, floor-aligned), intrinsics, and
the per-frame SAM3 masks for one instance, this gives:

  - `project_points`     — world → image projection (OpenCV convention).
  - `render_silhouette`  — rasterise a mesh's silhouette from a camera (per-
                           triangle fill via PIL — no GPU).
  - `silhouette_iou`     — IoU between a rendered silhouette and the SAM3 mask
                           (the quality metric for the ShapeR mesh path).
  - `carve_to_visual_hull` — remove the parts of a mesh that fall *outside* the
                           back-projected mask cones (the upper bound on the
                           object's extent). Optionally uses an "occlusion mask"
                           (mask ∪ occluded-region) so it never over-carves.

Camera convention: ``c2w`` is camera→world, OpenCV (+x right, +y down,
+z forward). ``K`` = [[fx,0,cx],[0,fy,cy],[0,0,1]].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


@dataclass
class CameraView:
    c2w: np.ndarray                      # (4,4) camera→world (OpenCV)
    K: np.ndarray                        # (3,3) intrinsics
    width: int
    height: int
    mask: Optional[np.ndarray] = None    # (H,W) bool — the instance's SAM3 silhouette
    occlusion_mask: Optional[np.ndarray] = None  # (H,W) bool — mask ∪ "can't tell" region

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)

    def hull_mask(self) -> Optional[np.ndarray]:
        """The mask to use for visual-hull carving: occlusion_mask if set, else mask."""
        return self.occlusion_mask if self.occlusion_mask is not None else self.mask


def project_points(pts_world: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                   width: int, height: int):
    """world → image. Returns (uv (N,2) float, depth (N,), in_front_and_in_image (N,) bool)."""
    pts = np.asarray(pts_world, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    ph = np.column_stack([pts, np.ones(len(pts))])
    cam = (w2c @ ph.T).T[:, :3]
    z = cam[:, 2]
    front = z > 1e-6
    zz = np.where(front, z, 1.0)
    img = (np.asarray(K, dtype=np.float64) @ (cam / zz[:, None]).T).T
    uv = img[:, :2]
    inb = front & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, z, inb


def render_silhouette(verts: np.ndarray, faces: np.ndarray, view: CameraView,
                      supersample: int = 1) -> np.ndarray:
    """Rasterise the mesh's silhouette as seen from ``view`` → (H,W) bool array.
    Per-triangle polygon fill (PIL); triangles with any vertex behind the camera
    are dropped (good enough — they'd be clipped anyway)."""
    H, W = int(view.height), int(view.width)
    if not _HAS_PIL or len(faces) == 0:
        return np.zeros((H, W), dtype=bool)
    ss = max(1, int(supersample))
    uv, z, _ = project_points(verts, view.c2w, view.K, W * ss, H * ss)
    # scale uv to the supersampled image
    uv = uv * ss
    img = Image.new("1", (W * ss, H * ss), 0)
    dr = ImageDraw.Draw(img)
    behind = z <= 1e-6
    fa = np.asarray(faces, dtype=np.int64)
    bad = behind[fa].any(axis=1)
    for f in fa[~bad]:
        p = uv[f]
        # skip degenerate / huge triangles
        if not np.isfinite(p).all():
            continue
        dr.polygon([tuple(p[0]), tuple(p[1]), tuple(p[2])], fill=1)
    arr = np.array(img, dtype=bool)
    if ss > 1:
        arr = arr.reshape(H, ss, W, ss).any(axis=(1, 3))
    return arr


def silhouette_iou(verts: np.ndarray, faces: np.ndarray, view: CameraView) -> float:
    """IoU between the mesh silhouette and ``view.mask``. Returns NaN if no mask."""
    if view.mask is None:
        return float("nan")
    sil = render_silhouette(verts, faces, view)
    m = np.asarray(view.mask, dtype=bool)
    if m.shape != sil.shape:
        return float("nan")
    inter = int(np.logical_and(sil, m).sum())
    union = int(np.logical_or(sil, m).sum())
    return float(inter / union) if union else 0.0


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=int(px))
    except Exception:
        return mask


def _mask_violations(pts: np.ndarray, masks_KP) -> np.ndarray:
    """(N,3) → (N,) int: how many views see the point but place it *outside*
    their (dilated) mask. A view only counts for the points it can actually see
    (in front of it, inside the image)."""
    n = len(pts)
    viol = np.zeros(n, dtype=np.int32)
    for m, c2w, K in masks_KP:
        H, W = m.shape
        uv, z, inb = project_points(pts, c2w, K, W, H)
        vis = np.where(inb)[0]
        if len(vis) == 0:
            continue
        cols = np.clip(uv[vis, 0].astype(int), 0, W - 1)
        rows = np.clip(uv[vis, 1].astype(int), 0, H - 1)
        in_m = m[rows, cols]
        viol[vis[~in_m]] += 1
    return viol


def carve_to_visual_hull(verts: np.ndarray, faces: np.ndarray, views: List[CameraView],
                         dilate_px: int = 5, min_views: int = 3,
                         centroid: Optional[np.ndarray] = None,
                         iters: int = 18, max_violations: int = 1):
    """Pull the parts of a mesh that stick out past the back-projected mask cones
    *inward onto the visual-hull boundary* (rather than deleting them — an
    over-grown mesh has all its vertices on its too-large surface, so deletion
    would erase everything).

    A vertex is "outside the hull" only if it lands outside the (dilated) mask in
    *more than* ``max_violations`` views (so one slightly-too-small / noisy mask
    can't carve good geometry). For such a vertex, binary-search the largest
    ``s∈[0,1]`` such that ``centroid + s·(v − centroid)`` is within tolerance —
    that point sits on the visual-hull boundary along the ray from the centroid.
    The per-vertex carve factor is then Laplacian-smoothed over the mesh graph so
    neighbouring vertices carve by similar amounts — independent per-vertex carving
    spikes at the silhouette boundary otherwise. Topology is preserved.
    Returns ``(verts2, faces, carve_s)``.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    usable = [v for v in views if v.hull_mask() is not None]
    if len(usable) < min_views or len(verts) == 0:
        return verts, faces, np.ones(len(verts))
    masks_KP = [(_dilate(np.asarray(v.hull_mask(), dtype=bool), dilate_px),
                 np.asarray(v.c2w, dtype=np.float64), np.asarray(v.K, dtype=np.float64))
                for v in usable]
    thr = max(0, int(max_violations))
    if len(usable) < 6:
        thr = 0                                  # few views → be strict
    c = np.asarray(centroid, dtype=np.float64) if centroid is not None else verts.mean(0)
    rel = verts - c
    s = np.ones(len(verts))
    viol0 = _mask_violations(verts, masks_KP)
    todo = viol0 > thr
    if todo.any():
        lo = np.zeros(len(verts))
        hi = np.ones(len(verts))
        for _ in range(int(iters)):
            mid = 0.5 * (lo + hi)
            test = c + mid[:, None] * rel
            ok_mid = _mask_violations(test, masks_KP) <= thr
            adv = todo & ok_mid
            lo[adv] = mid[adv]
            ret = todo & ~ok_mid
            hi[ret] = mid[ret]
        s[todo] = lo[todo]

    # Laplacian-smooth the carve factor over the mesh graph (≈5 relaxed passes):
    # turns the spiky per-vertex carve into a smooth dent that follows the
    # (smoothed) silhouette. Edges that weren't carved (s≈1) anchor the rest.
    if len(faces) and (s < 0.999).any():
        try:
            from scipy.sparse import csr_matrix
            nv = len(verts)
            e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                                 faces[:, 1], faces[:, 2], faces[:, 0]])
            e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                                 faces[:, 0], faces[:, 1], faces[:, 2]])
            A = csr_matrix((np.ones(len(e0)), (e0, e1)), shape=(nv, nv))
            deg = np.asarray(A.sum(axis=1)).ravel()
            deg[deg == 0] = 1.0
            lam = 0.6
            for _ in range(5):
                nb_mean = np.asarray(A @ s).ravel() / deg
                s = (1.0 - lam) * s + lam * nb_mean
            s = np.clip(s, 0.0, 1.0)
        except Exception:
            pass
    new_verts = c + s[:, None] * rel
    return new_verts, faces, s
