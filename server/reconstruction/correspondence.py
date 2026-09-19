"""What licenses a correspondence — the evidence a correction is allowed to use.

A reconstruction is a field of keyframe poses. Every correction is a motion of
that field, and the only thing that authorises moving a pose is a MEASURED
disagreement between observations that ought to agree. This module is where
that measurement lives, and it answers four questions about any two point sets
proposed as copies of one thing. None of them mentions a kind of object, a
label, a scene or a threshold picked by hand:

  1. Nothing is a copy until it is measured to be one.
  2. Nothing is a witness if it could not have seen.
  3. Nothing is corrected in a direction nobody observed.
  4. Confidence is the amount of evidence, never the size of the leftover error.

The third is the one that carries the others. A point set constrains a rigid
motion only in the directions its own SURFACE constrains, and the surface says
so through its normals: the second moment ``N = Σ nᵢ nᵢᵀ`` of the matched
points' normals has one strong eigenvalue for a plane (its normal), two for a
cylinder or a beam (everything across the axis), three for a compact body, and
a near-zero one in every direction the surface slides along itself. A closure
that lies in a near-zero direction was never observed: sliding the copy there
changes no measured distance, so the number is an artifact of the fit, not a
displacement of the world.

That single test is what separates a real duplicate from a pairing of two
different PLACES on one extended element — the failure mode that cannot be
told apart by looking at the two point sets alone, because any stretch of a
duct looks like any other stretch of the same duct. It does not need to be:
the separation of such a pairing lies along the element, which is exactly the
direction its normals leave free. The same test removes a rotation fitted to a
near-symmetric body, and the in-plane sliding of a floor or a wall.

Measured on pccr 2026-09-18, where every one of these appeared at once: a duct
run demanding 3, 7, 9.6 and 10.2 m at the same time depending on which pair of
keyframes was read, a floor demanding 10.2 m, sheeting 8.9 m, beams 3.1 m, and
a monitor inventing 20.7° out of a 77 cm separation — 52 of the 69 m the pose
graph was asked to close. Over the same stretch seven bounded objects agreed,
independently, on 69 to 106 cm of pure translation.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── 1. the frame the two sets share ──────────────────────────────────────

def common_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal directions of the SHAPE, rows of a 3×3 rotation. Everything
    below is expressed here so no axis is privileged by the world frame
    (gravity, the viewer's floor transform) — the geometry picks its own axes.

    Each copy is centred on ITSELF before they are stacked. Centring the union
    instead lets the pair's own separation elongate the cloud and rotate the
    frame onto the offset, so a 2 m slide plus a 0.5 m real displacement comes
    back as neither: the axes must describe the object, never how far apart
    the two observations of it happen to be.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    P = np.vstack([a - a.mean(0), b - b.mean(0)])
    # SVD of the centred union: rows of Vt are the principal directions
    _u, _s, vt = np.linalg.svd(P, full_matrices=False)
    R = np.asarray(vt, np.float64)
    if np.linalg.det(R) < 0:                      # keep it a rotation
        R[2] = -R[2]
    return R


def spans(P: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Extent of a point set along each axis (3,)."""
    Q = np.asarray(P, np.float64) @ axes.T
    return Q.max(0) - Q.min(0)


# ── 2. surface normals: what the geometry actually constrains ────────────

def surface_normals(P: np.ndarray, k: int = 24,
                    max_points: int = 20000, seed: int = 0
                    ) -> Optional[np.ndarray]:
    """Unit normal per point from a local PCA over its k nearest neighbours.

    The sign is irrelevant here — every use below is through ``n nᵀ``, which
    is sign-free — so no orientation pass is needed.
    Returns None when the set is too small to have a local surface.
    """
    from scipy.spatial import cKDTree

    P = np.asarray(P, np.float64)
    if len(P) < max(k, 8):
        return None
    if len(P) > max_points:
        rng = np.random.default_rng(seed)
        P = P[rng.choice(len(P), max_points, replace=False)]
    tree = cKDTree(P)
    _d, idx = tree.query(P, k=min(int(k), len(P)))
    nb = P[idx]                                   # (n, k, 3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    # smallest-eigenvector of each local covariance = the local normal
    cov = np.einsum("nki,nkj->nij", nb, nb) / max(nb.shape[1] - 1, 1)
    w, v = np.linalg.eigh(cov)
    return np.ascontiguousarray(v[:, :, 0])       # (n, 3), unit by construction


def normal_energy(normals: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Per-axis share of ``N = Σ n nᵀ`` — how strongly the surface resists a
    translation along each axis, normalised so the three sum to 1.

    A plane concentrates everything on its normal; a cylinder or a beam splits
    it between the two directions across the axis and leaves the axis at ~0; a
    compact body spreads it over all three.
    """
    n = np.asarray(normals, np.float64)
    N = n.T @ n / max(len(n), 1)
    d = np.einsum("ij,jk,ik->i", axes, N, axes)   # diag of axes·N·axesᵀ
    tot = float(d.sum())
    return d / tot if tot > 1e-12 else np.zeros(3)


# ── 3. what a direction has to show to count as observed ─────────────────

def stiffness(a_aligned: np.ndarray, b: np.ndarray, axes: np.ndarray,
              delta: float, max_points: int = 20000, seed: int = 0,
              knn: int = 24) -> np.ndarray:
    """How much the MEASURED disagreement rises when the alignment is nudged
    along each axis — the principle itself, not a proxy for it.

    ``delta`` is the smallest displacement the session can tell apart (its own
    repeatability). The aligned copy is shifted by ±delta along an axis and
    the point-to-SURFACE distance is re-measured. A direction the geometry
    constrains gives the whole nudge back (stiffness ≈ 1); a direction the
    surface slides along gives nothing (≈ 0). Dimensionless, bounded, and it
    needs no constant: the scale is the nudge itself.

    Point-to-surface, never point-to-point. Against the nearest SAMPLE the
    probe measures the sampling density instead of the geometry: on a plane
    carrying a point every 6 cm, a 1 cm slide moves every point 1 cm from its
    own neighbour and the plane reports itself as stiff in a direction it
    cannot see at all. Projecting on the local normal removes the sampling
    from the question entirely.

    The normals were also tried as a shortcut for the whole test, and they are
    not one: a real scanned plane is never exactly degenerate, so its in-plane
    energy is a thousandth rather than nothing and a 10 m slide still passed
    (pccr 2026-09-18). Here they only linearise the surface locally, which is
    what they are reliable for.
    """
    from scipy.spatial import cKDTree

    a = np.asarray(a_aligned, np.float64)
    b = np.asarray(b, np.float64)
    if len(a) < 8 or len(b) < max(knn, 8):
        return np.zeros(3)
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]
    nb = surface_normals(b, k=knn, max_points=len(b))
    if nb is None:
        return np.zeros(3)
    tree = cKDTree(b)

    def cost(P):
        _d, j = tree.query(P, k=1)
        return float(np.median(np.abs(np.einsum("ij,ij->i", P - b[j], nb[j]))))

    d0 = cost(a)
    step = float(delta)
    if step <= 0:
        return np.zeros(3)
    out = np.zeros(3)
    for k in range(3):
        v = np.asarray(axes[k], np.float64) * step
        # the EASIER side: a direction is free if it is free either way
        out[k] = max(min(cost(a + v), cost(a - v)) - d0, 0.0) / step
    return np.clip(out, 0.0, 1.0)


def component_gain(pts_a: np.ndarray, pts_b: np.ndarray, axes: np.ndarray,
                   R: np.ndarray, t: np.ndarray,
                   max_points: int = 20000, seed: int = 0, knn: int = 24
                   ) -> np.ndarray:
    """What each component of the fitted translation BUYS, in metres of
    point-to-surface disagreement removed.

    For every axis, the alignment is re-measured with that component zeroed.
    If dropping it does not make the fit worse, it bought nothing and was
    never observed — whatever number the fitter put there. This is the direct
    form of the rule and the only one that survives contact with real data:
    a direction can be soft without being free, and then a large enough claim
    always clears a σ derived from softness alone. Asking what the claim buys
    cannot be fooled that way, because a free direction buys nothing at any
    magnitude.

    Negative means the component made the fit WORSE.
    """
    from scipy.spatial import cKDTree

    a = np.asarray(pts_a, np.float64)
    b = np.asarray(pts_b, np.float64)
    if len(a) < 8 or len(b) < max(knn, 8):
        return np.zeros(3)
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]
    nb = surface_normals(b, k=knn, max_points=len(b))
    if nb is None:
        return np.zeros(3)
    tree = cKDTree(b)

    def cost(P):
        _d, j = tree.query(P, k=1)
        return float(np.median(np.abs(np.einsum("ij,ij->i", P - b[j], nb[j]))))

    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3)
    aR = a @ R.T
    c = np.asarray(axes, np.float64) @ t
    full = cost(aR + t)
    out = np.zeros(3)
    for k in range(3):
        ck = c.copy()
        ck[k] = 0.0
        out[k] = cost(aR + axes.T @ ck) - full
    return out


def direction_sigma(sigma_floor_m: float, coverage_weak: float,
                    n_views: int, stiff: np.ndarray) -> np.ndarray:
    """σ per axis: the evidence, divided by how hard the surface pushes back.

    Two measured factors and nothing else. The evidence — how much of the
    weaker copy was matched, over how many independent views. The stiffness —
    what :func:`stiffness` probed. A direction that gives the nudge back keeps
    the session's floor; one that swallows it has no σ at all, so nothing can
    be claimed there however large the number the fitter produced.
    """
    floor = float(sigma_floor_m)
    cov = float(np.clip(coverage_weak, 0.0, 1.0))
    views = max(int(n_views), 1)
    support = cov * views
    if support <= 0:
        return np.full(3, np.inf)
    base = floor * np.sqrt(1.0 + 1.0 / support)
    k = np.asarray(stiff, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(k > 0, base / k, np.inf)
    return np.where(np.isfinite(sig), np.maximum(sig, floor), np.inf)


def stands_up(component: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """THE rule of this module, and the only one: a quantity is claimed when
    it exceeds its own measured uncertainty. Applied per axis to a
    translation, to a rotation against its own angular σ, and to the whole
    correspondence through the Mahalanobis norm."""
    c = np.abs(np.asarray(component, np.float64))
    s = np.asarray(sigma, np.float64)
    return np.isfinite(s) & (s > 0) & (c >= s)


def project_to_observed(t: np.ndarray, axes: np.ndarray,
                        observed: np.ndarray) -> np.ndarray:
    """Keep only the components of a translation the surface observed."""
    c = np.asarray(axes, np.float64) @ np.asarray(t, np.float64)
    c = np.where(np.asarray(observed, bool), c, 0.0)
    return axes.T @ c


def rotation_sigma_deg(sigma_t: np.ndarray, lever_m: float) -> float:
    """Angular σ of the fitted rotation: the best-pinned direction's σ over
    the lever arm the body offers. A flat panel gives its normal a small σ but
    no second direction, so the arm about that normal is whatever σ the two
    free axes carry — metres — and the angle drowns. That is where a fitted
    20.7° on a monitor comes from, and it is how it goes away without anyone
    deciding a panel cannot rotate."""
    s = np.asarray(sigma_t, np.float64)
    ok = np.isfinite(s) & (s > 0)
    if not ok.any() or lever_m <= 0:
        return float("inf")
    # the rotation is pinned by the WORST-constrained direction it must move
    worst = float(np.max(s[ok])) if ok.sum() >= 2 else float("inf")
    if not np.isfinite(worst):
        return float("inf")
    return float(np.degrees(np.arctan2(worst, lever_m)))


# ── 4. the evidence, and the confidence that follows from it ─────────────

@dataclass
class Correspondence:
    """Everything a downstream consumer needs, and nothing it has to assume."""

    n_matched: int
    coverage_a: float
    coverage_b: float
    axes: np.ndarray                      # 3×3, rows = principal directions
    span_a: np.ndarray                    # extent of each copy along the axes
    span_b: np.ndarray
    energy: np.ndarray                    # normal energy per axis (sums to 1)
    observed: np.ndarray                  # bool per axis
    t_raw: np.ndarray                     # the closure as fitted
    t: np.ndarray                         # projected onto what was observed
    R: np.ndarray                         # identity when the rotation is unobserved
    rot_deg: float
    sigma_t: np.ndarray                   # per-axis σ, in the AXES frame
    sigma_rot_deg: float
    is_duplicate: bool
    reason: str
    notes: Dict[str, float] = field(default_factory=dict)

    @property
    def dropped_m(self) -> float:
        """How much of the fitted closure fell in unobserved directions — the
        number that says an edge was an artifact rather than a displacement."""
        return float(np.linalg.norm(self.t_raw - self.t))

    def info_translation(self) -> np.ndarray:
        """3×3 information matrix in WORLD coordinates: 1/σ² on the observed
        axes, and the unobserved ones left at the caller's floor (0 here — the
        caller adds its own prior so an edge never constrains what it did not
        see)."""
        w = np.zeros(3)
        s = np.asarray(self.sigma_t, np.float64)
        ok = np.asarray(self.observed, bool) & (s > 0)
        w[ok] = 1.0 / s[ok] ** 2
        return self.axes.T @ np.diag(w) @ self.axes

    def as_dict(self) -> dict:
        return {"n_matched": int(self.n_matched),
                "coverage_a": round(float(self.coverage_a), 4),
                "coverage_b": round(float(self.coverage_b), 4),
                "energy": [round(float(x), 4) for x in self.energy],
                "observed": [bool(x) for x in self.observed],
                "t_raw_m": [round(float(x), 5) for x in self.t_raw],
                "t_m": [round(float(x), 5) for x in self.t],
                "dropped_m": round(self.dropped_m, 5),
                "rot_deg": round(float(self.rot_deg), 3),
                "sigma_t_m": [round(float(x), 5) for x in self.sigma_t],
                "sigma_rot_deg": round(float(self.sigma_rot_deg), 3),
                "is_duplicate": bool(self.is_duplicate),
                "reason": self.reason,
                "provenance": "tool_measured",
                **{k: round(float(v), 5) for k, v in self.notes.items()}}


# ── the measurement ──────────────────────────────────────────────────────

def measure(pts_a: np.ndarray, pts_b: np.ndarray,
            R: np.ndarray, t: np.ndarray,
            coverage_a: float, coverage_b: float, n_matched: int,
            sigma_floor_m: float, n_views: int = 1,
            matched_a: Optional[np.ndarray] = None,
            matched_b: Optional[np.ndarray] = None,
            knn: int = 24) -> Correspondence:
    """Turn a fitted closure into evidence: what it may claim, where, and how
    strongly. This does not refit — it decides what of the fit the geometry
    actually supports, under one rule: a quantity is claimed only when it
    exceeds its own measured uncertainty.

    ``matched_a`` / ``matched_b`` are the points the fitter actually put in
    correspondence. They are what the surface is measured on: a raw copy
    carries flyers and unmatched fringes, and a local PCA over those returns
    normals that describe the noise instead of the surface.
    """
    a = np.asarray(pts_a, np.float64)
    b = np.asarray(pts_b, np.float64)
    R = np.asarray(R, np.float64).reshape(3, 3)
    t_raw = np.asarray(t, np.float64).reshape(3)

    sa = np.asarray(matched_a, np.float64) if matched_a is not None else a
    sb = np.asarray(matched_b, np.float64) if matched_b is not None else b
    axes = common_frame(sa, sb)

    # the principle, probed on the cost the fitter itself minimised: the
    # aligned copy is nudged by what the session can just distinguish, and
    # each axis reports how much of that nudge the measurement gave back
    a_aligned = sa @ R.T + t_raw
    stiff = stiffness(a_aligned, sb, axes, float(sigma_floor_m))
    energy = stiff / stiff.sum() if stiff.sum() > 0 else np.zeros(3)
    reason = "measured" if stiff.max() > 0 else \
        "the surface constrains no direction — nothing here can be claimed"

    sig = direction_sigma(sigma_floor_m, min(coverage_a, coverage_b),
                          n_views, stiff)
    c = axes @ t_raw
    # THE decision: a component is claimed when dropping it would measurably
    # worsen the fit — by more than the session can tell apart from nothing
    gain = component_gain(sa, sb, axes, R, t_raw)
    keep = gain > float(sigma_floor_m)
    t_obs = axes.T @ np.where(keep, c, 0.0)

    rot_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    lever = float(np.max(spans(sa, axes))) / 2.0
    sig_rot = rotation_sigma_deg(sig, lever)
    rot_ok = np.isfinite(sig_rot) and rot_deg >= sig_rot
    R_out = R if rot_ok else np.eye(3)
    if not rot_ok:
        rot_deg = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.isfinite(sig) & (sig > 0), c / sig, 0.0)
    mahalanobis = float(np.sqrt(np.sum(d ** 2)))
    kept = float(np.linalg.norm(t_obs))
    raw = float(np.linalg.norm(t_raw))
    is_dup = bool(mahalanobis > 1.0 and kept > 0.0)
    if not np.isfinite(sig).any():
        reason = "the surface constrains no direction"
    elif raw > 0 and not is_dup:
        reason = "the separation bought nothing — the surface slides there"

    return Correspondence(
        n_matched=int(n_matched), coverage_a=float(coverage_a),
        coverage_b=float(coverage_b), axes=axes,
        span_a=spans(a, axes), span_b=spans(b, axes),
        energy=energy, observed=keep, t_raw=t_raw, t=t_obs,
        R=R_out, rot_deg=rot_deg, sigma_t=sig, sigma_rot_deg=sig_rot,
        is_duplicate=is_dup, reason=reason,
        notes={"kept_m": kept, "raw_m": raw, "sigmas": mahalanobis,
               "lever_m": lever,
               "stiff_x": float(stiff[0]), "stiff_y": float(stiff[1]),
               "stiff_z": float(stiff[2]),
               "gain_x": float(gain[0]), "gain_y": float(gain[1]),
               "gain_z": float(gain[2])})


# ── invariant 2: a witness must have been able to see ────────────────────

def witness_pairs(points: np.ndarray, frames: Sequence[int],
                  pose_of, K_of, depth_of, grid_wh: Tuple[int, int],
                  depth_tol_rel: float = 0.02) -> Dict[int, np.ndarray]:
    """{frame: boolean mask over ``points``} — could this frame have SEEN this
    point, on the geometry given?

    A frame testifies about a point only when the point projects inside its
    image, in front of the camera, and at the depth that frame actually
    measured: geometry closer along the same ray means the point was hidden,
    and a ray with no measurement says nothing either way.

    This is the denominator of every agreement measure. Counting a point in
    every frame where its object happens to be segmented asks it to be visible
    from places that never saw it; pccr 2026-09-18 measured 2–11% agreement
    that way, against 69% over the same keyframe when only the frames that
    could see were counted.

    It is computed on the REFERENCE geometry and never recomputed for a trial:
    the set is fixed, so a correction cannot improve its own score by pushing
    the scene out of view.
    """
    P = np.asarray(points, np.float64)
    W, H = int(grid_wh[0]), int(grid_wh[1])
    out: Dict[int, np.ndarray] = {}
    for f in frames:
        c2w = pose_of(int(f))
        K = K_of(int(f))
        if c2w is None or K is None:
            continue
        c2w4 = np.eye(4)
        c2w4[:np.asarray(c2w).shape[0], :np.asarray(c2w).shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        p = (M[:3, :3] @ P.T).T + M[:3, 3]
        z = p[:, 2]
        ok = z > 1e-3
        u = np.full(len(P), -1.0)
        v = np.full(len(P), -1.0)
        u[ok] = K[0, 0] * p[ok, 0] / z[ok] + K[0, 2]
        v[ok] = K[1, 1] * p[ok, 1] / z[ok] + K[1, 2]
        vis = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = depth_of(int(f)) if depth_of is not None else None
        if d is not None and vis.any():
            dh, dw = np.asarray(d).shape[:2]
            uu = np.clip((u * dw / W).astype(np.int64), 0, dw - 1)
            vv = np.clip((v * dh / H).astype(np.int64), 0, dh - 1)
            zs = np.asarray(d, np.float64)[vv, uu]
            measured = np.isfinite(zs) & (zs > 1e-6)
            # hidden: the frame measured something closer along the same ray
            hidden = measured & (zs < z * (1.0 - float(depth_tol_rel)))
            vis &= measured & ~hidden
        out[int(f)] = vis
    return out
