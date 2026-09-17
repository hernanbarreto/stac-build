"""Spatial plausibility of an identity (claude_stac.txt §4.5).

SAM3 and SALAD are not aware of space; the system is. Before a bridge is spent
on a candidate — SALAD pair, re-identified instance, duplicate cluster, manual
mark — the candidate must be compatible with the current poses, the intrinsics
and the raw cloud under a DRIFT BUDGET:

    δ(L) = max(drift_floor_m,   drift_rate_m_per_m · L)      translation
    θ(L) = max(drift_floor_deg, drift_rate_deg_per_m · L)    rotation

with L the metres WALKED between the two observations along the current
trajectory. Once the pose graph has run, δ/θ are replaced by the accumulated
odometry uncertainty of the chain between the two frames (``budget_override``)
— the measured value, not a default.

Rules (every one lands in the verdict with value / threshold / passed):
  1. cluster separation (instance & duplicate candidates): ≤ δ(L) → loop;
     > identity_reject_factor·δ(L) → NOT drift (SAM3 fused two objects) →
     instance split; in between → ``ambiguous`` (σ inflated, kit visual);
  2. reciprocal frustum: the other visit's points, transformed with the current
     poses and widened by δ/θ, must project inside ≥ min_frustum_frames frames
     of each visit AND not sit behind the frame's own measured surface
     (occlusion by the raw cloud — a wall seen from its other side is still
     the same wall: visibility, not face orientation);
  3. size and shape: the model dimensions fitted to each cluster agree within
     size_tol (relative);
  4. walk topology: a repetitive class seen twice inside a corridor needs
     ≥ min_context_instances neighbouring structural instances that also pass
     rules 1–3 — context tells one column from the next;
  5. SALAD candidates without instances: rules 2 and 4 over the frusta of
     frames i and j and the raw cloud around them (inside a corridor the
     reciprocity demanded is corridor_min_frustum_frames, or ``ambiguous``).

The view object is duck-typed (the VGGT-Long fork and the server both provide
one): ``n_frames``, ``hw`` (H, W of the depth grid), ``pose(g)`` → c2w 4x4 or
None, ``K(g)`` → 3x3, ``centres()`` → (N,3) with NaN where unposed,
``points(g, n)`` → (≤n,3) world points of frame g, and optionally
``depth(g)`` → the frame's own measured depth image (H,W) for the occlusion test.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _cfg(cfg) -> Dict[str, Any]:
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


# ── drift budget ────────────────────────────────────────────────────────────

def walked_length_m(centres: np.ndarray, i: int, j: int) -> float:
    """Metres walked along the trajectory between frames i and j (NaN rows —
    unposed frames — are skipped, the walk continues through them)."""
    lo, hi = min(int(i), int(j)), max(int(i), int(j))
    c = np.asarray(centres, np.float64)[lo:hi + 1]
    c = c[np.isfinite(c).all(1)]
    if len(c) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def drift_budget(L_m: float, cfg, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """δ(L), θ(L) — or the measured chain uncertainty when the pose graph has
    already run (override = {"delta_m", "theta_deg", "source"})."""
    c = _cfg(cfg)
    if override is not None:
        return {"L_m": float(L_m), "delta_m": float(override["delta_m"]),
                "theta_deg": float(override["theta_deg"]),
                "source": str(override.get("source", "chain_uncertainty"))}
    return {"L_m": float(L_m),
            "delta_m": float(max(c["drift_floor_m"], c["drift_rate_m_per_m"] * L_m)),
            "theta_deg": float(max(c["drift_floor_deg"], c["drift_rate_deg_per_m"] * L_m)),
            "source": "drift_rate_model"}


# ── rule 2: reciprocal frustum ───────────────────────────────────────────────

def visible_fraction(pts_world: np.ndarray, c2w: np.ndarray, K: np.ndarray, hw,
                     cfg, delta_m: float, theta_deg: float,
                     depth: Optional[np.ndarray] = None) -> float:
    """Fraction of pts_world that project inside frame (c2w, K) once the
    frustum is widened by the drift budget: a translation δ at depth z or a
    rotation θ moves a projection by up to f·δ/z + f·tan θ pixels, and the
    depth window is widened by δ. With the frame's own measured ``depth``
    (H,W) — the raw cloud, §4.5 rule 5 — a point sitting farther than
    δ + occlusion_tol_m BEHIND the measured surface on its pixel is occluded:
    inside the frustum but not visible (the block between two corridor legs)."""
    c = _cfg(cfg)
    if len(pts_world) == 0:
        return 0.0
    M = np.linalg.inv(np.asarray(c2w, np.float64))
    pc = np.asarray(pts_world, np.float64) @ M[:3, :3].T + M[:3, 3]
    z = pc[:, 2]
    H, W = int(hw[0]), int(hw[1])
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    zmin, zmax = float(c["min_depth_m"]), float(c["max_depth_m"])
    z_safe = np.maximum(z, zmin if zmin > 0 else 1e-6)
    widen = f * float(delta_m) / z_safe + f * np.tan(np.radians(float(theta_deg)))
    margin = float(c["frustum_margin_px"]) + widen
    with np.errstate(divide="ignore", invalid="ignore"):
        u = float(K[0, 0]) * pc[:, 0] / z + float(K[0, 2])
        v = float(K[1, 1]) * pc[:, 1] / z + float(K[1, 2])
    ok = ((z >= zmin - delta_m) & (z <= zmax + delta_m)
          & (u >= -margin) & (u < W + margin) & (v >= -margin) & (v < H + margin))
    if depth is not None and ok.any():
        d = np.asarray(depth, np.float64)
        dh, dw = d.shape[:2]
        ui = np.clip(np.round(u[ok] * dw / W).astype(np.int64), 0, dw - 1)
        vi = np.clip(np.round(v[ok] * dh / H).astype(np.int64), 0, dh - 1)
        zs = d[vi, ui]
        measured = np.isfinite(zs) & (zs > 0)
        occluded = measured & (z[ok] > zs + float(delta_m) + float(c["occlusion_tol_m"]))
        idx = np.flatnonzero(ok)
        ok[idx[occluded]] = False
    return float(np.mean(ok))


def _frames_around(g: int, n_frames: int, window: int) -> Sequence[int]:
    return [k for k in range(int(g) - int(window), int(g) + int(window) + 1) if 0 <= k < n_frames]


def frustum_reciprocal(i: int, j: int, view, cfg, delta_m: float, theta_deg: float,
                       pts_i: Optional[np.ndarray] = None,
                       pts_j: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Rule 2. pts_i/pts_j: the two observations' points (cluster or frame
    points); when None the frames' own raw points are used (SALAD case)."""
    c = _cfg(cfg)
    n_pts = int(c["frustum_points"])
    if pts_i is None:
        pts_i = view.points(i, n_pts)
    if pts_j is None:
        pts_j = view.points(j, n_pts)
    out = {"rule": "frustum_reciprocal", "min_frustum_frames": int(c["min_frustum_frames"]),
           "min_visible_frac": float(c["min_visible_frac"])}
    for tag, g, pts in (("j_in_frames_of_i", i, pts_j), ("i_in_frames_of_j", j, pts_i)):
        n_vis, fracs = 0, []
        depth_fn = getattr(view, "depth", None)
        for k in _frames_around(g, view.n_frames, c["frustum_window_kf"]):
            P, K = view.pose(k), view.K(k)
            if P is None or K is None:
                continue
            dep = depth_fn(k) if depth_fn is not None else None
            fr = visible_fraction(pts, P, K, view.hw, c, delta_m, theta_deg, depth=dep)
            fracs.append(fr)
            if fr >= float(c["min_visible_frac"]):
                n_vis += 1
        out[tag] = {"frames_visible": int(n_vis), "n_points": int(len(pts)),
                    "max_visible_frac": float(max(fracs)) if fracs else 0.0}
    a = out["j_in_frames_of_i"]["frames_visible"]
    b = out["i_in_frames_of_j"]["frames_visible"]
    out["frames_visible_min"] = int(min(a, b))
    out["passed"] = bool(min(a, b) >= int(c["min_frustum_frames"]))
    return out


# ── rule 4/5: walk topology (corridor) ──────────────────────────────────────

def corridor_between(i: int, j: int, view, cfg) -> Dict[str, Any]:
    """PCA of the camera centres walked between i and j: lateral extents (2nd
    and 3rd principal axes, full range) both below corridor_width_m → the
    trajectory never left a corridor."""
    c = _cfg(cfg)
    lo, hi = min(int(i), int(j)), max(int(i), int(j))
    cen = view.centres()[lo:hi + 1]
    cen = cen[np.isfinite(cen).all(1)]
    if len(cen) < 3:
        return {"rule": "corridor", "corridor": False, "n_centres": int(len(cen))}
    X = cen - cen.mean(0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt.T
    ext = proj.max(0) - proj.min(0)
    lateral = float(max(ext[1] if len(ext) > 1 else 0.0, ext[2] if len(ext) > 2 else 0.0))
    return {"rule": "corridor", "corridor": bool(lateral < float(c["corridor_width_m"])),
            "lateral_extent_m": lateral, "along_extent_m": float(ext[0]),
            "corridor_width_m": float(c["corridor_width_m"]), "n_centres": int(len(cen))}


# ── rule 1: cluster separation ───────────────────────────────────────────────

def separation_rule(dist_m: float, budget: Dict[str, float], cfg) -> Dict[str, Any]:
    """How far apart the two copies sit — and NOTHING is ever cut over it.

    USER 2026-09-16: *"no debe cortar objetos, no debe existir"*. It used to
    return "split" past `identity_reject_factor` (3.0) × the drift budget:
    two copies farther apart than that were declared "not drift, SAM3 fused two
    objects" and the instance was cut in two. Both numbers were invented, and on
    pccr the 0.30 m floor won, so the rule that cut the floor into four pieces
    and the ceiling into six was an invented multiple of an invented floor.

    A floor is one floor however far apart two of its clusters sit. Geometry in
    the wrong place is not a reason to cut anything: the mask audit says where
    an instance's mass falls and the correction moves it.
    """
    d, delta = float(dist_m), float(budget["delta_m"])
    verdict = "loop" if d <= delta else "ambiguous"
    return {"rule": "separation", "distance_m": d, "delta_m": delta,
            "verdict": verdict, "passed": True}


# ── rule 3: size and shape ───────────────────────────────────────────────────

def model_dims(points: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """Supported extents along the cluster's principal axes (percentile band,
    sorted descending) — the dimension proxy of the fusion primitives design
    without depending on a fitted surface."""
    P = np.asarray(points, np.float64)
    if len(P) < 3:
        return np.zeros(3)
    X = P - P.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt.T
    ext = np.percentile(proj, hi_pct, axis=0) - np.percentile(proj, lo_pct, axis=0)
    return np.sort(np.abs(ext))[::-1]


def size_rule(dims_a: np.ndarray, dims_b: np.ndarray, cfg) -> Dict[str, Any]:
    c = _cfg(cfg)
    a = np.asarray(dims_a, np.float64)
    b = np.asarray(dims_b, np.float64)
    den = np.maximum(np.maximum(a, b), 1e-9)
    rel = np.abs(a - b) / den
    # the smallest axis of a thin object is noise-dominated: judge the two
    # largest supported extents
    rel_judged = rel[:2] if len(rel) >= 2 else rel
    worst = float(rel_judged.max()) if len(rel_judged) else 0.0
    return {"rule": "size", "dims_a": a.tolist(), "dims_b": b.tolist(),
            "rel_diff": rel.tolist(), "worst_rel_diff": worst,
            "size_tol": float(c["size_tol"]), "passed": bool(worst <= float(c["size_tol"]))}


# ── same-surface rule: one object with an unobserved middle is not two ─────

def _pca(points: np.ndarray):
    P = np.asarray(points, np.float64)
    c = P.mean(0)
    X = P - c
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / max(len(P) - 1, 1)
    return c, ev, Vt


def same_surface_rule(pts_a: np.ndarray, pts_b: np.ndarray, cfg) -> Dict[str, Any]:
    """Geometry the two clusters share, and the OBSERVABLE separation between
    them. Two disjoint clusters of one instance that lie on the SAME plane
    (planar, normals aligned) or the SAME axis (elongated, axes aligned) can
    be one object with an unobserved middle — a wall past a doorway, a column
    behind an occluder — so the separation that matters is NOT the centroid
    distance (partial views of a long wall sit metres apart by construction)
    but the offset along the plane normal / lateral to the axis: the
    displacement the two copies can actually observe (the observability
    notion of correction/observability.py). Otherwise the centroid distance is
    the separation.

    The geometry KIND is decided by the LARGER cluster (the better sampled
    one); the smaller must not contradict it (when it is well conditioned its
    normal/axis has to align) — a sliver of a wall seen from a corner never
    turns the pair into a "compact" one. Elongated is tested first: a thin
    vertical cluster is degenerate in two directions and would pass the planar
    test with an arbitrary normal; a plane is never elongated.

    Returns kind ∈ {plane, axis, centroid}, distance_m (the observable
    separation), aligned, same_geometry (aligned plane/axis)."""
    c = _cfg(cfg)
    A, B = np.asarray(pts_a, np.float64), np.asarray(pts_b, np.float64)
    ca, cb = A.mean(0), B.mean(0)
    centroid_d = float(np.linalg.norm(cb - ca))
    if len(A) < 3 or len(B) < 3:
        return {"rule": "same_surface", "kind": "centroid", "distance_m": centroid_d,
                "same_geometry": False, "centroid_distance_m": centroid_d}
    big, small = (A, B) if len(A) >= len(B) else (B, A)
    _, eB, VB = _pca(big)
    _, eS, VS = _pca(small)
    ang_tol = np.cos(np.radians(float(c["same_surface_angle_deg"])))
    axis_ratio = float(c["same_surface_axis_ratio"])
    planar_ratio = float(c["same_surface_planar_ratio"])
    d = cb - ca

    def _elong(ev):
        return ev[1] <= axis_ratio * max(ev[0], 1e-12)

    def _planar(ev):
        return ev[2] <= planar_ratio * max(ev[0], 1e-12)

    if _elong(eB):
        u = VB[0] / (np.linalg.norm(VB[0]) + 1e-12)
        aligned = True
        if _elong(eS):
            aligned = abs(float(u @ VS[0])) >= ang_tol
        if aligned:
            lateral = float(np.linalg.norm(d - (d @ u) * u))
            return {"rule": "same_surface", "kind": "axis", "aligned": True,
                    "distance_m": lateral, "same_geometry": True,
                    "centroid_distance_m": centroid_d}
    elif _planar(eB):
        n = VB[2] / (np.linalg.norm(VB[2]) + 1e-12)
        aligned = True
        if _planar(eS) and not _elong(eS):
            aligned = abs(float(n @ VS[2])) >= ang_tol
        if aligned:
            offset = abs(float(d @ n))
            return {"rule": "same_surface", "kind": "plane", "aligned": True,
                    "distance_m": offset, "same_geometry": True,
                    "centroid_distance_m": centroid_d}
    return {"rule": "same_surface", "kind": "centroid", "distance_m": centroid_d,
            "same_geometry": False, "centroid_distance_m": centroid_d}


# ── verdicts ────────────────────────────────────────────────────────────────

def gate_frame_pair(i: int, j: int, view, cfg,
                    budget_override: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """§4.5 rule 5 — a SALAD (or manual) candidate WITHOUT instances: rules 2
    and 4 over the two frames' frusta and their raw points. Returns
    verdict ∈ {accept, ambiguous, reject} with every rule's numbers."""
    c = _cfg(cfg)
    L = walked_length_m(view.centres(), i, j)
    budget = drift_budget(L, c, budget_override)
    # Rule 0 — the walk itself. A retrieval pair whose two keyframes are only
    # metres apart ALONG THE TRAJECTORY is odometry wearing a revisit's clothes:
    # the camera never left, so the pair observes nothing the chain does not
    # already carry, and closing it stiffens a stretch that was never in doubt
    # while the real long-range closure is absorbed. Measured alongside the
    # others so every rule's numbers are in the record, whichever one fires.
    min_walk = float(c["min_walk_m"])
    walk_ok = L >= min_walk
    fr = frustum_reciprocal(i, j, view, c, budget["delta_m"], budget["theta_deg"])
    cor = corridor_between(i, j, view, c)
    out = {"candidate": [int(i), int(j)], "budget": budget,
           "rules": {"walk": {"walked_m": L, "min_walk_m": min_walk, "passed": walk_ok},
                     "frustum": fr, "corridor": cor}}
    if not walk_ok:
        out["verdict"] = "reject"
        out["reason"] = (f"only {L:.1f} m walked between the two keyframes "
                         f"(< {min_walk:.1f} m) — odometry, not a revisit")
        return out
    if not fr["passed"]:
        out["verdict"] = "reject"
        out["reason"] = (f"not co-visible even with a {budget['delta_m']*100:.0f} cm / "
                         f"{budget['theta_deg']:.1f}° budget over {L:.1f} m walked "
                         f"({fr['frames_visible_min']} < {fr['min_frustum_frames']} frames)")
        return out
    if cor["corridor"] and fr["frames_visible_min"] < int(c["corridor_min_frustum_frames"]):
        out["verdict"] = "ambiguous"
        out["reason"] = (f"corridor walk (lateral {cor['lateral_extent_m']:.1f} m) with only "
                         f"{fr['frames_visible_min']} reciprocal frames "
                         f"(< {int(c['corridor_min_frustum_frames'])}) — σ inflated")
        return out
    out["verdict"] = "accept"
    out["reason"] = "co-visible under the drift budget"
    return out


def gate_instance_pair(i: int, j: int, view, cfg, pts_a: np.ndarray, pts_b: np.ndarray,
                       label: str, context_pass: Optional[int] = None,
                       budget_override: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """§4.5 rules 1–4 for an instance / duplicate candidate: two clusters
    (pts_a seen around frame i, pts_b around frame j). ``context_pass`` = number
    of neighbouring structural instances that already passed rules 1–3 (the
    caller computes it only when the label is repetitive and the walk is a
    corridor). Returns verdict ∈ {loop, ambiguous, reject}. The
    separation judged is the OBSERVABLE one (plane offset / lateral axis
    offset for clusters sharing a plane / axis, centroid distance otherwise)."""
    c = _cfg(cfg)
    L = walked_length_m(view.centres(), i, j)
    budget = drift_budget(L, c, budget_override)
    geom = same_surface_rule(pts_a, pts_b, c)
    sep = separation_rule(float(geom["distance_m"]), budget, c)
    sep["kind"] = geom["kind"]
    # the frustum question for an instance pair is "could the other visit have
    # seen this object at all, displaced by anything we would still call
    # drift" — the reject bound of rule 1, not δ itself
    # The frustum asks "could the other visit have seen this object at all,
    # displaced by anything we would still call drift": factor × δ(L).
    #
    # This factor is what `identity_reject_factor` used to be, and the rename is
    # the point — there it decided to CUT an instance in two, here it only
    # bounds a visibility question and nothing is ever split. Using the OBSERVED
    # separation instead was tried and is worse than a chosen number: a rule
    # that tolerates exactly what it measures always passes, and an object 30 m
    # behind the camera stopped being rejected.
    fr = frustum_reciprocal(i, j, view, c,
                            float(c["frustum_tolerance_factor"]) * budget["delta_m"],
                            budget["theta_deg"],
                            pts_i=np.asarray(pts_a), pts_j=np.asarray(pts_b))
    sz = size_rule(model_dims(pts_a, c["dims_pct_lo"], c["dims_pct_hi"]),
                   model_dims(pts_b, c["dims_pct_lo"], c["dims_pct_hi"]), c)
    cor = corridor_between(i, j, view, c)
    if geom["same_geometry"]:
        # the size rule judges COMPACT objects only: partial views of one
        # plane/axis differ in extent by construction (a wall past a doorway,
        # a column cut by the field of view) — recorded as skipped
        sz = dict(sz, passed=True, skipped="same geometry (partial views of one surface)")
    out = {"candidate": [int(i), int(j)], "label": label, "budget": budget,
           "rules": {"separation": sep, "frustum": fr, "size": sz, "corridor": cor,
                     "geometry": geom}}
    # NOTHING IS EVER SPLIT (USER 2026-09-16: *"no debe cortar objetos, no debe
    # existir"*). The verdict used to be "split" — the instance was cut in two —
    # when the copies sat farther apart than an invented multiple of an invented
    # drift budget, or when their model dimensions differed. Both cut pccr's
    # floor into four pieces and its ceiling into six, and a floor is one floor
    # however far apart two of its clusters sit. Differing dimensions are
    # DECLARED and the pair is left ambiguous: the mask audit says where the
    # mass actually falls and the correction moves it.
    if not sz["passed"]:
        out["verdict"] = "ambiguous"
        out["reason"] = (f"model dimensions differ {sz['worst_rel_diff']*100:.0f}% > "
                         f"{sz['size_tol']*100:.0f}% — declared, nothing is cut")
        return out
    if not fr["passed"]:
        out["verdict"] = "reject"
        out["reason"] = (f"one copy is never inside the other visit's frusta even with the "
                         f"{budget['delta_m']*100:.0f} cm budget")
        return out
    repetitive = str(label).lower() in set(c["repetitive_labels"])
    if repetitive and cor["corridor"]:
        need = int(c["min_context_instances"])
        have = int(context_pass or 0)
        out["rules"]["context"] = {"rule": "context", "repetitive": True, "corridor": True,
                                   "context_pass": have, "min_context_instances": need,
                                   "passed": bool(have >= need)}
        if have < need:
            out["verdict"] = "ambiguous"
            out["reason"] = (f"repetitive '{label}' inside a corridor with {have} < {need} "
                             f"context instance(s) — σ inflated")
            return out
    if sep["verdict"] == "ambiguous":
        out["verdict"] = "ambiguous"
        out["reason"] = (f"copies {sep['distance_m']:.2f} m apart, beyond δ "
                         f"{sep['delta_m']:.2f} m — σ inflated, nothing is cut")
        return out
    out["verdict"] = "loop"
    out["reason"] = "same object displaced by drift within budget"
    return out
