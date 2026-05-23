"""
Reconstruct a `LinearRepeatElement` — railing / fence / railway track (rails +
sleepers) / pipe rack / cable run with supports.

Algorithm (the robust version — separate the continuous members first):
  1. run direction = PCA principal axis of the (high-conf) sub-cloud;
  2. cluster the cloud in the cross-section plane ⊥ run direction;
  3. a cross-section cluster whose points cover ≳70 % of the run length is a
     **continuous rail** → fit a `SweptElement` to it and remove its points;
  4. the remaining points are the **periodic members** (posts / balusters /
     sleepers / supports) → robust periodicity detection (rolling-min floor
     removal + `scipy.signal.find_peaks`) → spacing, count, and one member OBB;
  5. emit a `LinearRepeatElement(path, spacing, member, rails, n_members)`.

If neither a rail nor a periodic member is found it falls back to a single
`SweptElement`; if even that fails it returns ``None`` (caller → ShapeR mesh).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..elements import LinearRepeatElement, SweptElement, BoxElement
from ..geometry import fit_swept, classify_section
from ..geometry.primitives import _unit, _orthobasis
from .box import _obb_from_points
from .swept import reconstruct_swept


# ── periodicity ─────────────────────────────────────────────────────

def _find_periodicity(s: np.ndarray, run_len: float,
                      min_spacing: float = 0.04) -> Optional[Dict]:
    """Detect regularly-spaced clumps along the 1-D coordinate ``s``.

    Removes a rolling-minimum "floor" first (so periodic spikes that sit on top
    of a continuous background still show), then `scipy.signal.find_peaks` with a
    distance derived from the autocorrelation's dominant lag. Returns
    {spacing, n, centres} or None.
    """
    s = np.asarray(s, dtype=np.float64)
    if len(s) < 50 or run_len < 0.25:
        return None
    nb = int(np.clip(run_len / max(min_spacing / 3.0, run_len / 800.0), 60, 1200))
    hist, edges = np.histogram(s, bins=nb)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_w = float(centres[1] - centres[0])
    h = hist.astype(np.float64)
    try:
        from scipy.ndimage import minimum_filter1d
        floor = minimum_filter1d(h, size=max(3, nb // 12))
    except Exception:
        floor = np.minimum.accumulate(h) * 0  # degrade gracefully
    h2 = np.maximum(h - floor, 0.0)
    if h2.max() < 3.0:
        return None
    # dominant period from autocorrelation
    hc = h2 - h2.mean()
    ac = np.correlate(hc, hc, mode="full")[len(hc) - 1:]
    if len(ac) < 3 or ac.max() <= 0:
        return None
    ac[0] = 0.0
    min_lag = max(2, int(np.ceil((min_spacing / bin_w))))
    if min_lag >= len(ac):
        return None
    lag = int(np.argmax(ac[min_lag:]) + min_lag)
    # peaks
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(h2, distance=max(2, int(lag * 0.55)),
                              prominence=max(2.0, 0.25 * h2.max()))
    except Exception:
        return None
    if len(peaks) < 3:
        # tolerate exactly 2 if the autocorrelation lag agrees with the run
        if len(peaks) == 2:
            pk2 = np.sort(centres[peaks])
            n_est = int(round(run_len / max(pk2[1] - pk2[0], 1e-6))) + 1
            if n_est < 2:
                return None
        else:
            return None
    pk = np.sort(centres[peaks])
    gaps = np.diff(pk)
    if len(gaps) == 0:
        return None
    if np.std(gaps) > 0.4 * np.mean(gaps):
        # peaks are irregular — try a regular comb seeded by the autocorr lag
        spacing = lag * bin_w
        if spacing < min_spacing:
            return None
        n_est = int(round(run_len / spacing)) + 1
        if n_est < 3:
            return None
        pk = np.linspace(s.min(), s.max(), n_est)
        gaps = np.diff(pk)
    return {"spacing": float(np.median(gaps)), "n": int(len(pk)),
            "centres": np.asarray(pk)}


def _coverage(s_sub: np.ndarray, s0: float, s1: float, nbins: int = 50) -> float:
    """Fraction of run-length bins that contain at least one of the points."""
    if len(s_sub) == 0 or s1 <= s0:
        return 0.0
    edges = np.linspace(s0, s1, nbins + 1)
    occ = np.zeros(nbins, dtype=bool)
    bi = np.clip(np.digitize(s_sub, edges) - 1, 0, nbins - 1)
    occ[np.unique(bi)] = True
    return float(occ.mean())


def _swept_from_fit(swf, instance_id: int, label: str, caption: str,
                    caption_fields, profile_hint=None) -> SweptElement:
    poly = np.asarray(swf.profile_polygon, dtype=np.float64) if swf.profile_polygon is not None else None
    if poly is not None and len(poly) >= 3:
        fam, params = classify_section(poly, caption_hint=profile_hint)
    else:
        fam, params = (swf.profile_family or "free"), dict(swf.profile_params)
    if fam == swf.profile_family and swf.profile_params:
        params = dict(swf.profile_params)
    el = SweptElement(instance_id=instance_id, label=label, geometry_class="swept",
                      caption=caption, caption_fields=dict(caption_fields or {}),
                      profile_family=fam, profile_params=dict(params),
                      profile_polygon=poly if poly is not None else np.zeros((3, 2)),
                      profile_frame=np.asarray(swf.profile_frame, dtype=np.float64),
                      path=np.asarray(swf.path, dtype=np.float64),
                      arc_segments=list(swf.arc_segments),
                      wall_thickness=float(swf.wall_thickness))
    el.confidence_stats = {"fit_rms": float(swf.rms),
                           "path_length_m": float(swf.path_length),
                           "is_straight": bool(swf.is_straight)}
    return el


# ── main ────────────────────────────────────────────────────────────

def reconstruct_linear_repeat(cls, *, instance_id: int, label: str, xyz: np.ndarray,
                              xyz_hc: Optional[np.ndarray] = None,
                              world_up: Optional[np.ndarray] = None, caption: str = "",
                              caption_fields: Optional[Dict[str, str]] = None,
                              source_indices=None, dist_thresh: float = 0.012):
    pts = np.asarray(xyz_hc if (xyz_hc is not None and len(xyz_hc) >= 30) else xyz, dtype=np.float64)
    if len(pts) < 30:
        return None
    c = pts.mean(0)
    Q = pts - c
    _, evec = np.linalg.eigh(Q.T @ Q)
    run_dir = _unit(evec[:, 2])
    perp_a = _unit(evec[:, 1])
    perp_b = _unit(evec[:, 0])
    s = Q @ run_dir
    s0, s1 = float(s.min()), float(s.max())
    run_len = s1 - s0
    if run_len < 0.25:
        return None
    path = np.array([c + s0 * run_dir, c + s1 * run_dir])

    # — separate the CONTINUOUS rails from the periodic members —
    # A rail is *long* (≈the whole run) so its points pile up at one position in
    # the cross-section plane; a post/sleeper is *short* so its few points are
    # spread thin in the cross-section "shadow". So: 2-D histogram of the cloud's
    # cross-section shadow → high-density cells = rail locations; assign their
    # neighbourhood points to candidate rails, verify each actually spans the run.
    pa_c = Q @ perp_a
    pb_c = Q @ perp_b
    rails: List[SweptElement] = []
    rail_mask = np.zeros(len(pts), dtype=bool)
    try:
        from scipy.ndimage import maximum_filter, label as cc_label
        ra = pa_c.ptp() or 1e-3
        rb = pb_c.ptp() or 1e-3
        nba = int(np.clip(ra / max(0.02, ra / 80), 6, 120))
        nbb = int(np.clip(rb / max(0.02, rb / 80), 1, 120))
        nbb = max(nbb, 1)
        H, ea, eb = np.histogram2d(pa_c, pb_c, bins=[nba, max(nbb, 2)])
        if H.max() > 0:
            thr = max(0.18 * H.max(), 4.0 * np.median(H[H > 0]) if np.any(H > 0) else 0.0)
            dense = H >= thr
            # group dense cells into connected components → one per rail location
            lab_img, ncomp = cc_label(dense)
            ai = np.clip(np.digitize(pa_c, ea) - 1, 0, H.shape[0] - 1)
            bi = np.clip(np.digitize(pb_c, eb) - 1, 0, H.shape[1] - 1)
            for comp in range(1, ncomp + 1):
                cell_mask = lab_img == comp
                # which cloud points fall in this component's cells?
                sel = cell_mask[ai, bi]
                if int(sel.sum()) < 30:
                    continue
                if _coverage(s[sel], s0, s1) < 0.65:
                    continue                          # not actually continuous
                swf = fit_swept(pts[sel], dist_thresh=max(dist_thresh, 0.02), up_hint=world_up)
                if swf is not None and swf.path_length >= 0.5 * run_len:
                    rails.append(_swept_from_fit(swf, instance_id, f"{label}_rail",
                                                 caption, caption_fields, cls.profile_family))
                    rail_mask |= sel
    except Exception:
        pass

    # — remaining points → periodic members —
    rest = ~rail_mask
    n_rest = int(rest.sum())
    member = None
    spacing = 0.0
    n_members = 0
    if n_rest >= 30:
        per = _find_periodicity(s[rest], run_len)
        if per is not None:
            spacing = float(per["spacing"])
            n_members = int(per["n"])
            cen = np.asarray(per["centres"])
            mid = float(cen[len(cen) // 2])
            half = 0.45 * max(spacing, 1e-3)
            msel = rest & (s >= mid - half) & (s <= mid + half)
            if msel.sum() >= 8:
                center_m, R_m, he_m = _obb_from_points(pts[msel])
                center_m = center_m + (s0 - mid) * run_dir          # relocate to path[0]
                member = BoxElement(instance_id=instance_id, label=f"{label}_member",
                                    geometry_class="box", center=center_m, R=R_m,
                                    half_extents=he_m)

    # — decide —
    if not rails and member is None:
        # nothing repetitive nor continuous → treat as a single swept run
        sw_el = reconstruct_swept(cls, instance_id=instance_id, label=label, xyz=xyz,
                                  xyz_hc=xyz_hc, world_up=world_up, caption=caption,
                                  caption_fields=caption_fields, source_indices=source_indices,
                                  dist_thresh=dist_thresh)
        if sw_el is not None:
            sw_el.meta["note"] = "linear_repeat: no rails / no periodicity → single swept run"
        return sw_el  # may be None → caller falls to ShapeR mesh

    el = LinearRepeatElement(instance_id=instance_id, label=label,
                             geometry_class="linear_repeat", is_structure=cls.is_structure,
                             caption=caption, caption_fields=dict(caption_fields or {}),
                             source_indices=source_indices, path=path,
                             element_spacing=spacing, member=member, rails=rails,
                             n_members=n_members)
    el.meta["role"] = cls.role
    el.confidence_stats = {"n_points": int(len(xyz)), "n_high_conf": int(len(pts)),
                           "n_rails": len(rails), "n_members": n_members,
                           "spacing_m": spacing, "run_length_m": float(run_len),
                           "members_residual_pts": n_rest}
    return el
