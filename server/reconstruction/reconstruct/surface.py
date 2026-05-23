"""
Reconstruct a `SurfaceElement` (plane / cylinder / cone) from a classified
instance. The fit was already done in `classify` (cached on `cls.surface_fit`).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..elements import SurfaceElement, ProfileElement
from ..geometry import concave_hull_2d, rectangle_outline, residual_bands, clean_polygon
from ..geometry.primitives import _orthobasis, _unit, PlaneFit, CylinderFit, ConeFit


def _curved_outline(theta: np.ndarray, s_along: np.ndarray,
                    theta_extent, ratio: float = 0.2) -> np.ndarray:
    """Concave-hull outline of a curved-surface patch in (θ, s) parameter space.

    Falls back to the rectangle [θ0,θ1]×[s_min,s_max] if the patch wraps the
    seam (near-full circle) or the hull is degenerate, so the result is always
    a usable polygon.
    """
    th0, th1 = float(theta_extent[0]), float(theta_extent[1])
    s_min, s_max = float(np.min(s_along)), float(np.max(s_along))
    if (th1 - th0) >= 2 * np.pi - 0.4 or len(theta) < 8:
        return rectangle_outline(th0, th1, s_min, s_max)
    # unwrap θ so it lies inside [th0, th1] (the fit's reported arc)
    th = np.mod(theta - th0, 2 * np.pi) + th0
    pts2 = np.column_stack([th, s_along])
    try:
        hull = concave_hull_2d(pts2, ratio=ratio)
        if hull is not None and len(hull) >= 3:
            # de-zigzag in (θ,s) space — θ≈O(1) rad, s≈O(1) m, so a small uniform
            # tolerance works for both; smoothing rounds the corners a touch.
            return clean_polygon(hull, tol=0.02, smooth_iters=1)
    except Exception:
        pass
    return rectangle_outline(th0, th1, s_min, s_max)


def _confidence_stats(xyz: np.ndarray, xyz_hc: Optional[np.ndarray],
                      signed_dist: np.ndarray, tol: float, rms: float) -> Dict:
    n = int(len(xyz))
    nhc = int(len(xyz_hc)) if xyz_hc is not None else n
    obs = float(np.mean(np.abs(signed_dist) < 2.0 * tol)) if n else 0.0
    return {"n_points": n, "n_high_conf": nhc, "observed_fraction": obs,
            "fit_rms": float(rms)}


def reconstruct_surface(cls, *, instance_id: int, label: str, xyz: np.ndarray,
                        xyz_hc: Optional[np.ndarray] = None,
                        world_up: Optional[np.ndarray] = None, caption: str = "",
                        caption_fields: Optional[Dict[str, str]] = None,
                        source_indices=None, dist_thresh: float = 0.012) -> SurfaceElement:
    xyz = np.asarray(xyz, dtype=np.float64)
    world_up = _unit(world_up if world_up is not None else np.array([0.0, 0.0, 1.0]))
    # `cls.surface_fit` was computed (in classify_instance) on the high-confidence
    # sub-cloud when it had ≥12 points, else on the full sub-cloud — so any
    # `fit.inliers` mask indexes *this* array, not necessarily `xyz`.
    fit_cloud = (np.asarray(xyz_hc, dtype=np.float64)
                 if (xyz_hc is not None and len(xyz_hc) >= 12) else xyz)
    fit = cls.surface_fit.fit
    common = dict(instance_id=instance_id, label=label,
                  geometry_class=cls.geometry_class, is_structure=cls.is_structure,
                  caption=caption, caption_fields=dict(caption_fields or {}),
                  source_indices=source_indices, role=cls.role)

    if isinstance(fit, PlaneFit):
        n = fit.normal
        # in-plane basis: v ≈ world-up projected into the plane (horizontal-aware),
        # u = n × v.  If the plane is ~horizontal, fall back to the cloud's
        # principal in-plane direction.
        v_up = world_up - (world_up @ n) * n
        if np.linalg.norm(v_up) > 0.2:
            v = _unit(v_up)
            u = _unit(np.cross(n, v))
        else:
            u0, v0 = _orthobasis(n)
            proj = np.column_stack([fit_cloud @ u0, fit_cloud @ v0])
            Q = proj - proj.mean(0)
            _, evec = np.linalg.eigh(Q.T @ Q)
            uu = evec[:, 1]
            u = _unit(uu[0] * u0 + uu[1] * v0)
            v = _unit(np.cross(n, u))
        # origin: inlier centroid projected onto the plane
        ins = fit.inliers if (fit.inliers is not None and len(fit.inliers) == len(fit_cloud)) else None
        c = fit_cloud[ins].mean(0) if (ins is not None and ins.any()) else fit_cloud.mean(0)
        origin = c - (c @ n + fit.d) * n
        # outline of the (full) sub-cloud projected to (u,v): for a structural
        # surface use a *loose* hull (≈convex) — the structural-shell step snaps
        # it to the floor/ceiling/adjacent-wall rectangle, and openings (doors/
        # windows) are then found as interior holes by occlusion reasoning rather
        # than carved out of the outline. For a non-structural panel keep the
        # tight concave hull so its real irregular boundary is preserved.
        rel = xyz - origin
        uv = np.column_stack([rel @ u, rel @ v])
        outline = concave_hull_2d(uv, ratio=(0.65 if cls.is_structure else 0.18))
        # de-zigzag the raw hull of the (noisy) cloud; structural surfaces get
        # snapped to clean rectangles in assembly anyway, so a light simplify is
        # enough there — non-structural panels keep their real shape but smoothed.
        outline = clean_polygon(outline, tol=0.035,
                                smooth_iters=(0 if cls.is_structure else 2))
        # Attached profiles (skirting / cornice / pilaster / frame) from near-plane
        # residuals — ONLY on wall-like surfaces. A floor / ceiling / slab / ramp
        # doesn't carry mouldings; running the band detector there just turns
        # whatever is lying on/near the surface (object bases, recess lips, noise)
        # into phantom "pilasters" → the "floor made of layers" artifact.
        attached = []
        _wall_like = (abs(float(n @ world_up)) < 0.45 and
                      cls.role not in ("floor", "ceiling", "slab", "deck", "ramp", "platform_edge"))
        if _wall_like:
            try:
                for b in residual_bands(xyz, n, fit.d, origin, u, v, world_up,
                                        on_plane_tol=dist_thresh):
                    attached.append(ProfileElement(path=b["path"], section=b["section"],
                                                   kind=b["kind"]))
            except Exception:
                pass
        sd = xyz @ n + fit.d
        el = SurfaceElement(**common, surface_type="plane", plane_normal=n,
                            plane_d=float(fit.d), basis_origin=origin,
                            basis_u=u, basis_v=v, outline=outline, attached=attached)
        el.confidence_stats = _confidence_stats(xyz, xyz_hc, sd, dist_thresh, fit.rms)
        el.meta["curvature"] = float(getattr(fit, "curvature", 0.0))
        return el

    if isinstance(fit, CylinderFit):
        ax = _unit(fit.axis_dir)
        ap = fit.axis_point
        tref = _unit(fit.theta_ref - (fit.theta_ref @ ax) * ax)
        if np.linalg.norm(tref) < 1e-6:
            tref, _ = _orthobasis(ax)
        bnorm = np.cross(ax, tref)
        th0, th1 = fit.theta_extent
        # (θ, s) of every cloud point → concave-hull outline
        rel = xyz - ap
        s_comp = rel @ ax
        radial = rel - np.outer(s_comp, ax)
        theta = np.arctan2(radial @ bnorm, radial @ tref)
        outline = _curved_outline(theta, s_comp, (th0, th1))
        sd = np.linalg.norm(radial, axis=1) - fit.radius
        el = SurfaceElement(**common, surface_type="cylinder", axis_point=ap,
                            axis_dir=ax, radius=float(fit.radius), theta_ref=tref,
                            outline=outline)
        el.confidence_stats = _confidence_stats(xyz, xyz_hc, sd, 1.6 * dist_thresh, fit.rms)
        el.meta["theta_extent"] = [float(th0), float(th1)]
        el.meta["s_extent"] = [float(np.min(s_comp)), float(np.max(s_comp))]
        return el

    if isinstance(fit, ConeFit):
        apex = fit.apex
        ax = _unit(fit.axis_dir)
        tref = _unit(fit.theta_ref - (fit.theta_ref @ ax) * ax)
        if np.linalg.norm(tref) < 1e-6:
            tref, _ = _orthobasis(ax)
        bnorm = np.cross(ax, tref)
        th0, th1 = fit.theta_extent
        rel = xyz - apex
        s_comp = rel @ ax
        radial = rel - np.outer(s_comp, ax)
        theta = np.arctan2(radial @ bnorm, radial @ tref)
        outline = _curved_outline(theta, s_comp, (th0, th1))
        # distance to cone surface ≈ |v| sin(φ − α)
        vn = np.linalg.norm(rel, axis=1) + 1e-12
        phi = np.arccos(np.clip(s_comp / vn, -1, 1))
        sd = vn * np.sin(phi - fit.half_angle)
        el = SurfaceElement(**common, surface_type="cone", apex=apex, axis_point=apex,
                            axis_dir=ax, half_angle=float(fit.half_angle),
                            theta_ref=tref, outline=outline)
        el.confidence_stats = _confidence_stats(xyz, xyz_hc, sd, 2.0 * dist_thresh, fit.rms)
        el.meta["theta_extent"] = [float(th0), float(th1)]
        el.meta["s_extent"] = [float(np.min(s_comp)), float(np.max(s_comp))]
        return el

    raise TypeError(f"unsupported surface fit type {type(fit)}")
