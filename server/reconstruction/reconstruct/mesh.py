"""
ShapeR mesh path — the fallback for genuinely free-form blobs, under the
**double-bound regime**:

  upper bound  = the visual hull (intersection of back-projected SAM3 mask cones,
                 occlusion-aware) → carve away anything outside it;
  lower bound  = the high-confidence sub-cloud (+ TSDF mesh vertices, if any) →
                 shrink-wrap mesh vertices onto it where data exists;
  the middle   = ShapeR's prediction, kept untouched only where there is no data
                 (occluded regions) → those vertices are tagged ``observed=False``.

Plus a quality gate: per-camera silhouette-vs-mask IoU + a coarse ``quality_flag``.

`reconstruct_mesh_from_glb(...)` loads the GLB the ShapeR subprocess produced
and runs the above; `postprocess_shaper_mesh(...)` is the testable core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..elements import MeshElement
from ..visual_hull import CameraView, carve_to_visual_hull, silhouette_iou

try:
    import trimesh
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    trimesh = None
    cKDTree = None


def _bbox_diag(v: np.ndarray) -> float:
    return float(np.linalg.norm(v.max(0) - v.min(0))) if len(v) else 1.0


def _shrink_wrap(verts: np.ndarray, faces: np.ndarray, target: np.ndarray,
                 radius: float, alpha: float = 0.7, max_disp_frac: float = 0.6,
                 smooth_iters: int = 1):
    """Pull each mesh vertex toward the nearest target point within ``radius``
    (a fraction ``alpha`` of the way, total displacement clamped to
    ``max_disp_frac·radius``). Vertices with no target within ``radius`` keep
    ShapeR's position. Returns ``(new_verts, observed)`` (``observed`` is a bool
    per vertex)."""
    verts = np.asarray(verts, dtype=np.float64).copy()
    if target is None or len(target) == 0 or cKDTree is None:
        return verts, np.zeros(len(verts), dtype=bool)
    tree = cKDTree(np.asarray(target, dtype=np.float64))
    dist, idx = tree.query(verts, k=1)
    observed = dist <= radius
    if observed.any():
        nn = np.asarray(target)[idx[observed]]
        delta = (nn - verts[observed]) * alpha
        cap = max_disp_frac * radius
        dn = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, cap / np.maximum(dn[:, 0], 1e-12))[:, None]
        verts[observed] = verts[observed] + delta * scale
    # gentle Taubin smoothing to remove any spikes the wrap introduced
    if smooth_iters > 0 and trimesh is not None and len(faces):
        try:
            m = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, np.int64), process=False)
            trimesh.smoothing.filter_taubin(m, iterations=int(smooth_iters))
            if len(m.vertices) == len(verts):
                verts = np.asarray(m.vertices, dtype=np.float64)
        except Exception:
            pass
    return verts, observed


def _tsdf_divergence(verts: np.ndarray, observed: np.ndarray, tsdf_verts: Optional[np.ndarray],
                     radius: float) -> Optional[float]:
    """Median distance from ShapeR's observed vertices to the TSDF surface
    (sampled by its vertices). High ⇒ ShapeR disagrees with the depth-based
    reconstruction where both have data → suspicious."""
    if tsdf_verts is None or len(tsdf_verts) < 3 or cKDTree is None or not observed.any():
        return None
    tree = cKDTree(np.asarray(tsdf_verts, dtype=np.float64))
    d, _ = tree.query(verts[observed], k=1)
    return float(np.median(d))


def postprocess_shaper_mesh(verts: np.ndarray, faces: np.ndarray, *,
                            target_pts: Optional[np.ndarray] = None,
                            tsdf_verts: Optional[np.ndarray] = None,
                            views: Optional[List[CameraView]] = None,
                            radius: Optional[float] = None,
                            iou_warn: float = 0.45,
                            observed_warn: float = 0.15) -> Dict:
    """Gently correct a ShapeR mesh to the observed data **without deforming it**.

    The mesh keeps ShapeR's shape — only a translation + a per-(natural-)axis scale
    is applied so its bounding box matches the data (the observed cloud, inflated to
    estimate the true bbox; tightened per axis by the visual-hull silhouettes if
    SAM3 masks are available). Per-vertex carving / shrink-wrapping onto noisy,
    partial data wrecks the form ("los objetos están muy deformados") — a small
    anisotropic scale fixes gross size/aspect errors and leaves the shape intact.
    Returns ``vertices, faces, observed, iou_per_camera, quality_flag, stats``.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    stats: Dict = {"n_verts_in": int(len(verts)), "n_faces_in": int(len(faces)),
                   "mode": "affine_fit_to_data"}
    cloud = (np.asarray(target_pts, dtype=np.float64)
             if (target_pts is not None and len(target_pts) >= 8) else None)

    # Natural axes = PCA of the observed cloud (fall back to the mesh's own PCA).
    base_pts = cloud if cloud is not None else verts
    c_base = base_pts.mean(0)
    try:
        _, R = np.linalg.eigh(np.cov((base_pts - c_base).T))
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            R = np.eye(3)
    except Exception:
        R = np.eye(3)
    c_mesh = verts.mean(0)
    mesh_loc = (verts - c_mesh) @ R                       # mesh in the natural frame, centred on ITS centroid
    e_mesh = mesh_loc.ptp(0)

    # Deliberate per-axis correction: a lower bound (the observed cloud — the
    # object is *at least* this big along this axis) and, with masks, an upper
    # bound (the visual-hull silhouette — *at most* this big). Touch an axis ONLY
    # if ShapeR is outside [lower, upper], and by exactly the amount needed,
    # capped to ±18 %. NO translation — the mesh is already placed (by the batch
    # ICP); scaling is about its own centroid, so its position never moves.
    e_lower = ((cloud - c_base) @ R).ptp(0) if cloud is not None else None
    e_upper = None
    if views:
        try:
            cv, _cf, _cs = carve_to_visual_hull(verts, faces, views, dilate_px=3)
            if len(cv) == len(verts):
                e_carved = ((cv - c_mesh) @ R).ptp(0)
                # only trust it as an upper bound where it's a *meaningful* fraction
                # of ShapeR's extent (a misaligned mask collapses e_carved to ~0).
                e_upper = np.where(e_carved > 0.5 * np.maximum(e_mesh, 1e-9), e_carved, np.inf)
                stats["used_visual_hull_bbox"] = True
        except Exception:
            pass
    target = e_mesh.copy()
    if e_lower is not None:
        target = np.maximum(target, e_lower)              # at least cover the measured data
    if e_upper is not None:
        target = np.minimum(target, np.maximum(e_upper, e_lower if e_lower is not None else 0.0))
    scale_axes = np.clip(target / np.maximum(e_mesh, 1e-6), 0.82, 1.18)
    if np.allclose(scale_axes, 1.0, atol=0.005):
        stats["scale_axes"] = [1.0, 1.0, 1.0]             # nothing to fix
    else:
        stats["scale_axes"] = [round(float(s), 3) for s in scale_axes]
        verts = c_mesh + (mesh_loc * scale_axes[None, :]) @ R.T

    # Light Taubin pass to clean the raw GLB (low-pass + high-pass — barely changes
    # volume). Skip on a coarse mesh: there its bbox vertices are load-bearing.
    if trimesh is not None and len(faces) and len(verts) > 800:
        try:
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            trimesh.smoothing.filter_taubin(m, iterations=1)
            if len(m.vertices) == len(verts):
                verts = np.asarray(m.vertices, dtype=np.float64)
        except Exception:
            pass

    diag = _bbox_diag(verts)
    if radius is None:
        radius = float(np.clip(0.06 * diag, 0.02, 0.20))
    stats["wrap_radius_m"] = radius
    stats["wrap_target"] = "affine fit (shape preserved)"

    # observed = near a real cloud point (the actual measurement); the confidence
    # map shows green there, ShapeR's prediction (amber/red) elsewhere.
    obs_src = cloud
    if obs_src is None and tsdf_verts is not None and len(tsdf_verts) >= 4:
        obs_src = np.asarray(tsdf_verts, dtype=np.float64)
    if obs_src is not None and cKDTree is not None:
        d, _ = cKDTree(obs_src).query(verts, k=1)
        observed = d <= radius
    else:
        observed = np.zeros(len(verts), dtype=bool)
    stats["observed_fraction"] = float(observed.mean()) if len(observed) else 0.0

    # quality gate: silhouette IoU per camera + TSDF divergence
    iou_per_camera: Dict[str, float] = {}
    if views:
        for i, v in enumerate(views):
            if v.mask is None:
                continue
            try:
                iou_per_camera[str(i)] = silhouette_iou(verts, faces, v)
            except Exception:
                pass
    finite_ious = [x for x in iou_per_camera.values() if np.isfinite(x)]
    mean_iou = float(np.mean(finite_ious)) if finite_ious else None
    div = _tsdf_divergence(verts, observed, tsdf_verts, radius)
    flag = "ok"
    if stats["observed_fraction"] < observed_warn:
        flag = "sparse"
    if mean_iou is not None and mean_iou < iou_warn:
        flag = "low_iou"
    if div is not None and div > max(0.25, 3.0 * radius):
        flag = "tsdf_divergence"
    stats["mean_iou"] = mean_iou
    if div is not None:
        stats["tsdf_divergence_m"] = div
    return {"vertices": verts, "faces": faces, "observed": observed,
            "iou_per_camera": iou_per_camera, "quality_flag": flag, "stats": stats}


def make_mesh_element(post: Dict, *, instance_id: int, label: str, caption: str = "",
                      caption_fields: Optional[Dict[str, str]] = None,
                      role: str = "other", source_indices=None) -> MeshElement:
    el = MeshElement(instance_id=instance_id, label=label, geometry_class="volumetric_mesh",
                     caption=caption, caption_fields=dict(caption_fields or {}),
                     source_indices=source_indices,
                     vertices=np.asarray(post["vertices"], dtype=np.float64),
                     faces=np.asarray(post["faces"], dtype=np.int64),
                     observed=np.asarray(post["observed"], dtype=bool),
                     iou_per_camera=dict(post.get("iou_per_camera", {})),
                     quality_flag=str(post.get("quality_flag", "ok")))
    el.meta["role"] = role
    el.meta["shaper_postprocess"] = dict(post.get("stats", {}))
    el.confidence_stats = {"n_points": (int(len(source_indices)) if source_indices is not None else 0),
                           "observed_fraction": float(post["stats"].get("observed_fraction", 0.0)),
                           "mean_iou": post["stats"].get("mean_iou"),
                           "quality_flag": str(post.get("quality_flag", "ok"))}
    return el


def reconstruct_mesh_from_glb(glb_path, *, instance_id: int, label: str,
                              xyz_hc: Optional[np.ndarray] = None,
                              tsdf_mesh=None, views: Optional[List[CameraView]] = None,
                              caption: str = "", caption_fields: Optional[Dict] = None,
                              role: str = "other", source_indices=None) -> Optional[MeshElement]:
    """Load the ShapeR GLB the subprocess produced (already in floor-aligned world
    coords) and run the double-bound post-processing → `MeshElement`."""
    if trimesh is None:
        return None
    try:
        loaded = trimesh.load(str(glb_path), force="mesh")
        verts = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(loaded.faces, dtype=np.int64)
    except Exception as e:  # pragma: no cover
        print(f"[ReconstructMesh] load {glb_path}: {e}")
        return None
    if len(verts) < 4 or len(faces) < 4:
        return None
    tsdf_verts = None
    if tsdf_mesh is not None:
        try:
            tsdf_verts = np.asarray(tsdf_mesh.vertices if hasattr(tsdf_mesh, "vertices") else tsdf_mesh,
                                    dtype=np.float64)
        except Exception:
            tsdf_verts = None
    post = postprocess_shaper_mesh(verts, faces, target_pts=xyz_hc, tsdf_verts=tsdf_verts, views=views)
    return make_mesh_element(post, instance_id=instance_id, label=label, caption=caption,
                             caption_fields=caption_fields, role=role, source_indices=source_indices)
