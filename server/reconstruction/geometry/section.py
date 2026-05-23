"""
Cross-section classifier — label a 2-D profile polygon as one of
{circle, rect, rect_hollow, i_section, l_section, channel, rail, free} and fit
clean parametric parameters where confident.

Used by `reconstruct.swept` (pipes / HVAC ducts / structural members / rails /
kerbs / cable trays) and by `reconstruct.linear_repeat` (the continuous rails of
a railing or track). The raw `profile_polygon` is always kept and swept
faithfully; the family + params are the BIM-level semantics.

Robust features:
  - circularity  4π·A / P²            ≈1 circle, ≈0.785 square
  - convexity    A / A_hull           ≈1 circle/rect, <0.9 for I/L/channel/rail
  - notch count  components of (hull − poly)   1 ⇒ L or channel, 2 ⇒ I or rail
  - notch placement / symmetry          corner-notch ⇒ L, side-notch ⇒ channel,
                                        mirror-symmetric pair ⇒ I, else ⇒ rail
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import shapely
    from shapely.geometry import Polygon, MultiPolygon, Point
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _HAS_SHAPELY = False


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_perim(p: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)))


def _pca_align(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate the polygon so axis-0 is the longer principal direction. Returns
    (aligned_poly, R(2x2 cols=new axes in old frame), centroid)."""
    c = p.mean(0)
    Q = p - c
    _, V = np.linalg.eigh(Q.T @ Q)
    R = V[:, ::-1]                       # cols: largest→smallest variance
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    return Q @ R, R, c


def _section_width_profile(poly_xy: np.ndarray, nlevels: int = 60):
    """Width of the polygon as a function of position along its long axis.

    Returns (levels (L,), widths (L,)) where ``widths[k]`` is the total length of
    the polygon's intersection with the horizontal line at ``levels[k]`` (so a
    thin web shows up even with no vertices there). Uses shapely.
    """
    p = np.asarray(poly_xy, dtype=np.float64)
    w0 = p[:, 0].ptp()
    h0 = p[:, 1].ptp()
    if w0 > h0:                          # make axis-1 the long (depth) axis
        p = p[:, ::-1]
        w0, h0 = h0, w0
    if not _HAS_SHAPELY:
        return None, None
    from shapely.geometry import LineString
    poly = Polygon(p)
    if not poly.is_valid:
        poly = poly.buffer(0)
    ymin, ymax = p[:, 1].min(), p[:, 1].max()
    xmin, xmax = p[:, 0].min() - 1.0, p[:, 0].max() + 1.0
    levels = np.linspace(ymin + 1e-6 * h0, ymax - 1e-6 * h0, nlevels)
    widths = np.zeros(nlevels)
    for k, y in enumerate(levels):
        inter = poly.intersection(LineString([(xmin, y), (xmax, y)]))
        widths[k] = float(inter.length)
    return levels, widths


def _i_section_params(p_aligned: np.ndarray) -> Dict[str, float]:
    """(bf flange width, d depth, tf flange thickness, tw web thickness) of an
    I/H polygon (PCA-aligned). Robust to having no vertices along the web."""
    w0 = p_aligned[:, 0].ptp()
    h0 = p_aligned[:, 1].ptp()
    d = float(max(w0, h0))
    levels, widths = _section_width_profile(p_aligned)
    if widths is None or len(widths) < 5 or widths.max() <= 0:
        return {"bf": float(min(w0, h0)), "d": d, "tf": d * 0.05, "tw": d * 0.05}
    bf = float(np.max(widths))
    tw = float(np.min(widths[widths > 0.05 * bf])) if np.any(widths > 0.05 * bf) else bf * 0.1
    bin_h = float(levels[1] - levels[0]) if len(levels) > 1 else d / 60.0
    near_full = widths >= 0.7 * bf
    top = 0
    for v in near_full[::-1]:
        if v:
            top += 1
        else:
            break
    bot = 0
    for v in near_full:
        if v:
            bot += 1
        else:
            break
    tf = float(max(top, bot, 1) * bin_h)
    return {"bf": bf, "d": d, "tf": tf, "tw": tw}


def classify_section(profile_polygon: np.ndarray,
                     caption_hint: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    """Classify a 2-D cross-section. Returns ``(family, params)``.

    ``caption_hint`` (e.g. "circle", "rect", "i_section") only breaks ties when
    the geometry is ambiguous; geometry wins otherwise.
    """
    p = np.asarray(profile_polygon, dtype=np.float64)
    if p.ndim != 2 or len(p) < 4:
        return ("circle", {"r": float(np.max(np.linalg.norm(p - p.mean(0), axis=1)))}) \
            if len(p) else ("free", {})
    # drop a duplicated closing vertex
    if len(p) > 3 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    area = _poly_area(p)
    perim = _poly_perim(p)
    if area < 1e-9 or perim < 1e-6:
        return "free", {}
    circ = 4.0 * np.pi * area / (perim * perim)

    # convex hull + convexity
    hull_area = area
    notches = []
    if _HAS_SHAPELY:
        try:
            sp = Polygon(p)
            if not sp.is_valid:
                sp = sp.buffer(0)
            hull = sp.convex_hull
            hull_area = float(hull.area) if hull.area > 0 else area
            diff = hull.difference(sp)
            if diff.geom_type == "Polygon":
                cand = [diff]
            elif diff.geom_type == "MultiPolygon":
                cand = list(diff.geoms)
            else:
                cand = []
            notches = [g for g in cand if g.area > 0.015 * hull_area]
        except Exception:
            notches = []
    conv = area / max(hull_area, 1e-12)

    pa, R, c = _pca_align(p)
    w = float(pa[:, 0].ptp())
    h = float(pa[:, 1].ptp())
    long_side, short_side = (max(w, h), min(w, h))

    # — convex sections —
    if conv > 0.88 or not notches:
        if circ > 0.82:
            return "circle", {"r": float(np.sqrt(max(area, 1e-9) / np.pi))}
        # rectangle (or generic convex quad ≈ rect)
        return "rect", {"w": float(w), "h": float(h)}

    # — concave sections —
    if caption_hint in ("i_section", "l_section", "channel", "rail"):
        fam = caption_hint
        params = _i_section_params(pa) if fam in ("i_section", "rail") else {"w": w, "h": h}
        return fam, params

    nn = len(notches)
    # how many of the section's (axis-aligned) bounding-box corners lie inside the
    # material? rect → 4; channel(U) → 4 (both leg tops are material); L → 3
    # (the cut corner is in the notch); I → 4.
    if _HAS_SHAPELY:
        sp = Polygon(p)
        if not sp.is_valid:
            sp = sp.buffer(0)
        xs, ys = p[:, 0], p[:, 1]
        # corners in the ORIGINAL (un-aligned) frame using the aligned bbox →
        # transform back: pa-corner * Rᵀ + c
        pa_corners = np.array([[pa[:, 0].min(), pa[:, 1].min()],
                               [pa[:, 0].max(), pa[:, 1].min()],
                               [pa[:, 0].max(), pa[:, 1].max()],
                               [pa[:, 0].min(), pa[:, 1].max()]])
        orig_corners = pa_corners @ R.T + c
        inset = 0.06 * short_side
        ncin = 0
        for cc in orig_corners:
            # nudge the corner slightly inward toward the section centroid before testing
            t = cc + (c - cc) * (inset / max(np.linalg.norm(c - cc), 1e-9))
            if sp.buffer(1e-9).covers(Point(t[0], t[1])):
                ncin += 1
    else:
        ncin = 4

    if nn >= 2:
        # symmetric pair of notches (one on each side of the web) ⇒ I/H; an
        # asymmetric pair (a big foot, a smaller throat) ⇒ a rail profile.
        def _ac(g):
            cc = np.array([g.centroid.x, g.centroid.y])
            return (cc - c) @ R
        cs = np.array([_ac(g) for g in notches[:2]])
        depth_axis = 1 if h >= w else 0
        side_axis = 1 - depth_axis
        sym = (abs(cs[0, side_axis] + cs[1, side_axis]) < 0.30 * max(short_side, 1e-3)
               and abs(cs[0, depth_axis] - cs[1, depth_axis]) < 0.30 * long_side)
        return ("i_section" if sym else "rail"), _i_section_params(pa)

    # exactly one notch → L (3 corners in material) vs channel/U (4 corners)
    if ncin <= 3:
        return "l_section", {"a": float(w), "b": float(h)}
    return "channel", {"w": float(w), "h": float(h)}
