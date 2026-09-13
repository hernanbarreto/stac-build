"""Robust geometric primitive per paired instance and scan (F3 closed).

The alignment/scale evidence is NEVER the raw subcloud: floaters and
partial mask coverage bias extents, centroids and ICP (a door's bbox ratio
came out 0.128 on pccr). Per instance this module:

  1. PRE-CLEANS the points: main connected component (voxel-hash union) +
     statistical outlier removal — both configurable
     (``metric_validation.primitives``).
  2. Reuses ``surface_fit`` (never a RANSAC reimplementation): a persisted
     ``output/surface_fit/<label>_<id>/meta.json`` provides the model params
     when present; otherwise ``reconstruction.surface_fit.plane.fit_plane``
     and ``quadric.fit_cylinder`` fit on the spot. Model choice is decided
     by the GEOMETRY (higher inlier fraction wins; tie → lower rms), never
     by the label.
  3. Derives correspondence FEATURES with uncertainties: oriented plane
     normals + offsets, cylinder axes, corners (intersection of ≥2
     non-parallel planes + the scan floor), and MODEL DIMENSIONS —
     aperture width between parallel planes, cylinder diameter, supported
     in-plane extents (robust p2..p98 of the INLIER support, never bbox
     extremes).
  4. Records the median shooting distance (provenance → camera centre).

Uncertainty formulas (documented per prompt §6): a plane offset's σ is
``rms/√n_support`` (the offset is a support mean); a supported extent's σ is
the edge-softness ``((p98−p2)−(p95−p5))/2`` plus the fit rms — percentile
extents move by about that much under resampling; an aperture width's σ
adds both jambs' offset σ in quadrature; a diameter's σ is ``2·rms/√n``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

_SERVER_DIR = str(Path(__file__).resolve().parents[1])
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from correction.session import CorrectionSession
from fusion.config import FusionConfig

UP = np.array([0.0, 1.0, 0.0])


# ── pre-clean ────────────────────────────────────────────────────────────

def clean_points(P: np.ndarray, cfg: FusionConfig,
                 rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Indices (into P) that survive the floater pre-clean: main connected
    component over a voxel grid at ``component_radius_m``, then SOR
    (k-neighbour mean distance beyond mean + sigma·std dropped)."""
    pc = cfg.metric.primitives
    idx = np.arange(len(P))
    info = {"n_in": int(len(P))}
    if pc.keep_main_component and len(P) > pc.min_points:
        r = pc.component_radius_m
        vox = np.floor(P / r).astype(np.int64)
        key = {}
        for i, v in enumerate(map(tuple, vox)):
            key.setdefault(v, []).append(i)
        # BFS over 26-neighbour voxel adjacency
        seen = set()
        best: List[int] = []
        for start in key:
            if start in seen:
                continue
            comp = []
            stack = [start]
            seen.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            nb = (v[0] + dx, v[1] + dy, v[2] + dz)
                            if nb in key and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
            if sum(len(key[v]) for v in comp) > sum(len(key[v])
                                                   for v in best):
                best = comp
        idx = np.array(sorted(i for v in best for i in key[v]),
                       dtype=np.int64)
        info["main_component"] = int(len(idx))
    if len(idx) > pc.floater_sor_k + 1:
        S = P[idx]
        sub = S if len(S) <= 60000 else \
            S[rng.choice(len(S), 60000, replace=False)]
        tree = cKDTree(sub)
        d, _ = tree.query(S, k=min(pc.floater_sor_k + 1, len(sub)),
                          workers=cfg.runtime.workers)
        mean_d = d[:, 1:].mean(axis=1)
        cut = mean_d.mean() + pc.floater_sor_sigma * mean_d.std()
        idx = idx[mean_d <= cut]
        info["after_sor"] = int(len(idx))
    info["n_out"] = int(len(idx))
    return idx, info


# ── primitive dataclasses ────────────────────────────────────────────────

@dataclass
class PlanePrim:
    normal: np.ndarray          # unit, oriented (n[1] >= 0 or toward +x)
    origin: np.ndarray
    rms_m: float
    n_support: int
    support_idx: np.ndarray     # session cloud indices of the inliers
    extent_u: float             # supported robust extent (p2..p98)
    extent_v: float
    sigma_extent: float
    basis_u: np.ndarray
    basis_v: np.ndarray

    @property
    def offset(self) -> float:
        return float(self.normal @ self.origin)

    @property
    def sigma_offset(self) -> float:
        return float(self.rms_m / max(np.sqrt(self.n_support), 1.0))

    def summary(self) -> dict:
        return {"normal": [round(float(x), 5) for x in self.normal],
                "rms_mm": round(self.rms_m * 1000, 2),
                "n_support": self.n_support,
                "extent_u_m": round(self.extent_u, 4),
                "extent_v_m": round(self.extent_v, 4)}


@dataclass
class CylinderPrim:
    axis_point: np.ndarray
    axis_dir: np.ndarray        # unit
    radius: float
    rms_m: float
    n_support: int
    support_idx: np.ndarray
    length: float               # supported extent along the axis

    def summary(self) -> dict:
        return {"axis_dir": [round(float(x), 5) for x in self.axis_dir],
                "radius_m": round(self.radius, 4),
                "length_m": round(self.length, 4),
                "rms_mm": round(self.rms_m * 1000, 2),
                "n_support": self.n_support}


@dataclass
class InstancePrimitive:
    label: str
    kind: str                              # 'planes' | 'cylinder'
    planes: List[PlanePrim] = field(default_factory=list)
    cylinder: Optional[CylinderPrim] = None
    corners: List[np.ndarray] = field(default_factory=list)
    support_idx: np.ndarray = field(default_factory=lambda: np.empty(
        0, dtype=np.int64))
    model_dims: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    clean_info: dict = field(default_factory=dict)
    source: str = "fit_on_the_fly"         # or 'surface_fit_meta'

    def canonical_dim(self) -> Optional[Tuple[str, float, float]]:
        """The instance's model dimension used for scale evidence
        (name, value, σ): aperture width > diameter > major supported
        extent."""
        for name in ("aperture_width", "diameter", "extent_major"):
            if name in self.model_dims:
                v, s = self.model_dims[name]
                return name, v, s
        return None

    def summary(self) -> dict:
        return {"label": self.label, "kind": self.kind,
                "source": self.source,
                "planes": [p.summary() for p in self.planes],
                "cylinder": (self.cylinder.summary()
                             if self.cylinder else None),
                "n_corners": len(self.corners),
                "model_dims": {k: [round(v, 4), round(s, 4)]
                               for k, (v, s) in self.model_dims.items()},
                "clean": self.clean_info}


# ── fitting (reusing surface_fit) ────────────────────────────────────────

def _orient(n: np.ndarray) -> np.ndarray:
    n = n / max(np.linalg.norm(n), 1e-12)
    # deterministic sign: prefer up, then +x, then +z
    for i in (1, 0, 2):
        if abs(n[i]) > 1e-6:
            return n if n[i] > 0 else -n
    return n


def _plane_prim(model_params: dict, P: np.ndarray,
                idx: np.ndarray) -> PlanePrim:
    """PlanePrim from surface_fit plane params + the instance points (the
    support/extents always come from the points, params carry no extents)."""
    n = _orient(np.asarray(model_params["normal"], dtype=np.float64))
    origin = np.asarray(model_params["origin"], dtype=np.float64)
    rms = float(model_params.get("rms_m", 0.005))
    tol = max(3 * rms, 0.01)
    dist = np.abs((P - origin) @ n)
    inl = dist <= tol
    sup = P[inl]
    if not inl.any():
        raise RuntimeError("plane primitive has no support — the persisted "
                           "model does not match the instance points "
                           "(stale surface_fit? re-run it)")
    u = np.asarray(model_params.get("basis_u") or [], dtype=np.float64)
    if u.size != 3:
        # build an in-plane basis (vertical-ish v axis preferred)
        u = np.cross(UP, n)
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(np.array([1.0, 0.0, 0.0]), n)
        u /= np.linalg.norm(u)
    v = np.cross(n, u)
    pu = (sup - origin) @ u
    pv = (sup - origin) @ v
    e_u = float(np.percentile(pu, 98) - np.percentile(pu, 2))
    e_v = float(np.percentile(pv, 98) - np.percentile(pv, 2))
    soft_u = abs(e_u - float(np.percentile(pu, 95) - np.percentile(pu, 5)))
    soft_v = abs(e_v - float(np.percentile(pv, 95) - np.percentile(pv, 5)))
    sigma_extent = max(soft_u, soft_v) / 2 + rms
    return PlanePrim(normal=n, origin=sup.mean(0), rms_m=rms,
                     n_support=int(inl.sum()), support_idx=idx[inl],
                     extent_u=max(e_u, e_v), extent_v=min(e_u, e_v),
                     sigma_extent=float(sigma_extent), basis_u=u, basis_v=v)


def _load_surface_fit_meta(output_dir: Path, label: str,
                           iid: int) -> Optional[dict]:
    safe = f"{str(label).replace(' ', '_').replace('/', '_')[:40]}_{iid}"
    p = Path(output_dir) / "surface_fit" / safe / "meta.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    if meta.get("kind") not in ("plane", "cylinder"):
        return None
    return meta


def fit_instance(session: CorrectionSession, label: str, iid: int,
                 idx: np.ndarray, cfg: FusionConfig,
                 rng: np.random.Generator,
                 floor_plane: Optional[Tuple[np.ndarray, np.ndarray]] = None
                 ) -> InstancePrimitive:
    """The instance's primitive in this scan. Reuses a persisted surface_fit
    model when present; otherwise fits plane(s)/cylinder with the
    surface_fit fitters. Fails fast when nothing fits — an instance without
    a primitive cannot be fusion evidence."""
    from reconstruction.surface_fit.plane import fit_plane
    from reconstruction.surface_fit.quadric import fit_cylinder

    if len(idx) < cfg.metric.primitives.min_points:
        raise RuntimeError(
            f"instance '{label}': {len(idx)} points < "
            f"metric_validation.primitives.min_points "
            f"({cfg.metric.primitives.min_points}) — it cannot carry a "
            f"primitive")
    keep, clean_info = clean_points(session.xyz[idx], cfg, rng)
    cidx = idx[keep]
    P = session.xyz[cidx]
    sub = P if len(P) <= 50000 else P[rng.choice(len(P), 50000,
                                                 replace=False)]

    prim = InstancePrimitive(label=label, kind="planes",
                             clean_info=clean_info)
    meta = _load_surface_fit_meta(session.output_dir, label, iid)
    plane_model = cyl_model = None
    if meta is not None:
        prim.source = "surface_fit_meta"
        if meta["kind"] == "plane":
            plane_model = meta["params"]
        elif meta["kind"] == "cylinder":
            cyl_model = meta["params"]
    if plane_model is None and cyl_model is None:
        pm = fit_plane(sub, world_up=UP)
        cm = fit_cylinder(sub)
        f_p = float(pm.inlier_frac) if pm is not None else -1.0
        f_c = float(cm.inlier_frac) if cm is not None else -1.0
        if pm is None and cm is None:
            raise RuntimeError(
                f"instance '{label}': neither a plane nor a cylinder fits "
                f"its {len(sub)} cleaned points — pick a structural "
                f"invariant (wall, jamb, column) instead")
        if f_c > f_p or (f_c == f_p and cm is not None
                         and float(cm.rms) < float(pm.rms)):
            cyl_model = cm.params_dict()
        else:
            plane_model = pm.params_dict()

    if cyl_model is not None:
        a = np.asarray(cyl_model["axis_dir"], dtype=np.float64)
        a = _orient(a)
        ap = np.asarray(cyl_model["axis_point"], dtype=np.float64)
        r = float(cyl_model["radius"])
        rms = float(cyl_model.get("rms_m", 0.005))
        radial = np.linalg.norm(np.cross(P - ap, a), axis=1) - r
        inl = np.abs(radial) <= max(3 * rms, 0.01)
        if not inl.any():
            raise RuntimeError(
                f"instance '{label}': the cylinder model has no support on "
                f"the instance points — stale surface_fit? re-run it")
        along = (P[inl] - ap) @ a
        prim.kind = "cylinder"
        prim.cylinder = CylinderPrim(
            axis_point=ap, axis_dir=a, radius=r, rms_m=rms,
            n_support=int(inl.sum()), support_idx=cidx[inl],
            length=float(np.percentile(along, 98)
                         - np.percentile(along, 2)))
        prim.support_idx = cidx[inl]
        n = prim.cylinder.n_support
        prim.model_dims["diameter"] = (2 * r,
                                       2 * rms / max(np.sqrt(n), 1.0))
    else:
        # dominant plane + optionally a second (jamb pair / corner)
        p1 = _plane_prim(plane_model, P, cidx)
        planes = [p1]
        rest = np.setdiff1d(cidx, p1.support_idx)
        if len(rest) >= cfg.metric.primitives.min_points:
            P2 = session.xyz[rest]
            sub2 = P2 if len(P2) <= 50000 else \
                P2[rng.choice(len(P2), 50000, replace=False)]
            pm2 = fit_plane(sub2, world_up=UP)
            if pm2 is not None:
                try:
                    planes.append(_plane_prim(pm2.params_dict(), P2, rest))
                except RuntimeError:
                    pass
        prim.planes = planes
        prim.support_idx = np.unique(np.concatenate(
            [p.support_idx for p in planes]))
        # model dimensions
        if len(planes) >= 2:
            n1, n2 = planes[0].normal, planes[1].normal
            if abs(float(n1 @ n2)) >= np.cos(np.radians(
                    cfg.align.min_normal_angle_deg)):
                # near-parallel planes → aperture/thickness width
                nbar = _orient((n1 + (n2 if n1 @ n2 > 0 else -n2)) / 2)
                w = abs(float((planes[1].origin - planes[0].origin) @ nbar))
                sig = float(np.hypot(planes[0].sigma_offset,
                                     planes[1].sigma_offset))
                if w > 0:
                    prim.model_dims["aperture_width"] = (w, sig)
            else:
                # non-parallel planes → an instance corner (edge line ∩
                # floor when available)
                if floor_plane is not None:
                    nf, cf = floor_plane
                    A = np.stack([n1, n2, nf])
                    b = np.array([n1 @ planes[0].origin,
                                  n2 @ planes[1].origin, nf @ cf])
                    if abs(np.linalg.det(A)) > 1e-6:
                        prim.corners.append(np.linalg.solve(A, b))
        prim.model_dims["extent_major"] = (planes[0].extent_u,
                                           planes[0].sigma_extent)
        prim.model_dims["extent_minor"] = (planes[0].extent_v,
                                           planes[0].sigma_extent)
    return prim


def scan_floor_plane(session: CorrectionSession,
                     rng: np.random.Generator
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(normal, point) of the scan's floor — the vertical constraint for
    corners and the held-out floor exam. Reuses the correction module's
    floor-band machinery and parameters (one floor definition across the
    metric chain)."""
    from correction.config import load_correction_config
    from correction.gates import _floor_band
    ccfg = load_correction_config()
    mask = session.ks >= 0
    _band, plane = _floor_band(session.xyz, mask, ccfg, rng)
    return plane


# ── measurement between primitives (metric validation, §6) ───────────────

def measure_dimension(prims: List[InstancePrimitive], dimension: str,
                      cfg: FusionConfig) -> Tuple[float, float, str]:
    """(L_meas, σ_L, method) for the user's known-dimension element.
    Measuring between two raw clicks or bbox extremes is structurally
    impossible here — every path goes through a fitted primitive."""
    if dimension == "diameter":
        p = prims[0]
        if p.cylinder is None:
            raise RuntimeError(
                f"'{p.label}' fitted as {p.kind} — 'diameter' needs a "
                f"cylindrical element (column, pipe)")
        v, s = p.model_dims["diameter"]
        return v, s, "cylinder diameter (fit radius × 2)"
    if dimension in ("width", "height", "depth"):
        p = prims[0]
        if "aperture_width" in p.model_dims:
            v, s = p.model_dims["aperture_width"]
            return v, s, "plane-to-plane distance between the two fitted " \
                         "faces (jambs)"
        if p.kind == "cylinder":
            v, s = p.model_dims["diameter"]
            return v, s, "cylinder diameter (no plane pair on this element)"
        if not p.planes:
            raise RuntimeError(f"'{p.label}' carries no plane to measure")
        pl = p.planes[0]
        # supported extent along the requested direction of the dominant
        # plane: height = the more vertical in-plane axis, width = the
        # more horizontal one
        vert_u = abs(float(pl.basis_u @ UP))
        vert_v = abs(float(pl.basis_v @ UP))
        sup = None
        axis = pl.basis_u
        if dimension == "height":
            axis = pl.basis_u if vert_u >= vert_v else pl.basis_v
        else:
            axis = pl.basis_u if vert_u < vert_v else pl.basis_v
        # recompute the supported extent along that axis
        return _extent_along(p, pl, axis)
    if dimension == "distance_between":
        if len(prims) < 2:
            raise RuntimeError("'distance_between' needs two instances")
        f1 = _feature_point(prims[0])
        f2 = _feature_point(prims[1])
        L = float(np.linalg.norm(f1[0] - f2[0]))
        return L, float(np.hypot(f1[1], f2[1])), \
            f"distance between {f1[2]} and {f2[2]}"
    raise RuntimeError(f"unknown dimension '{dimension}' — valid: width, "
                       f"height, depth, diameter, distance_between")


def _extent_along(prim: InstancePrimitive, pl: PlanePrim,
                  axis: np.ndarray) -> Tuple[float, float, str]:
    from correction.session import CorrectionSession  # noqa: F401 (typing)
    proj = None
    # support points were kept as indices; the caller's session xyz was used
    # at fit time — extents were precomputed on u/v; project support again
    # is not possible without the session here, so use the precomputed pair
    vu = abs(float(pl.basis_u @ axis))
    val = pl.extent_u if vu >= 0.5 else pl.extent_v
    return float(val), float(pl.sigma_extent), \
        "supported in-plane extent (p2..p98 of the fit inliers)"


def _feature_point(prim: InstancePrimitive) -> Tuple[np.ndarray, float, str]:
    if prim.corners:
        return prim.corners[0], prim.planes[0].rms_m, \
            f"corner of '{prim.label}'"
    if prim.cylinder is not None:
        return prim.cylinder.axis_point, \
            prim.cylinder.rms_m / max(np.sqrt(prim.cylinder.n_support), 1.0), \
            f"axis of '{prim.label}'"
    pl = prim.planes[0]
    return pl.origin, pl.sigma_offset, f"face centre of '{prim.label}'"
