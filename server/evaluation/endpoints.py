"""An anchored END (one or more (keyframe, pixel) observations) → a 3-D point of a product.

Product WITH provenance (the chain): every observation yields a point from the product's own
depth at that pixel (window median of its surviving points); the end is the component-wise
MEDIAN of those points and its dispersion is recorded. Product WITHOUT depth maps (COLMAP, an
import): two or more observations TRIANGULATE with the variant's poses; one observation reads
the dense cloud's z-buffer at the pixel. A method that cannot answer falls through to the next
(declared in ``method``); an end no method reconstructs is "not reconstructed" — recorded,
never dropped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation import geometry as G
from evaluation.config import EndpointsConfig
from evaluation.product import Product


@dataclass
class EndResult:
    point: Optional[np.ndarray]     # world point (product units) or None
    method: str                     # provenance | zbuffer | triangulation | none
    n_obs: int                      # observations given
    n_used: int                     # observations the method used
    spread_rms_m: float             # RMS distance of the per-observation points to the median
    spread_max_m: float
    distance_m: float               # median camera distance of the used observations (product units)
    chainage_m: float               # mean chainage of the used keyframes (product units)
    frames: List[int] = field(default_factory=list)
    reason: str = ""
    per_observation: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["point"] = None if self.point is None else [float(x) for x in self.point]
        return d


def _not(n_obs: int, reason: str) -> EndResult:
    return EndResult(None, "none", n_obs, 0, float("nan"), float("nan"), float("nan"),
                     float("nan"), [], reason)


def reconstruct_end(end: Dict[str, Any], product: Product, cfg: EndpointsConfig,
                    chainage: Dict[int, float]) -> EndResult:
    obs = list(end.get("observations", []))
    usable = [o for o in obs if product.registered(int(o["frame"]))]
    if not usable:
        return _not(len(obs), "no observation on a registered keyframe"
                    if obs else "no observation")
    per: List[Dict[str, Any]] = []
    # ── by depth (provenance first, the z-buffer when the product has none or misses) ──
    pts, zs, frs = [], [], []
    for o in usable:
        f = int(o["frame"])
        uv_d = np.array([[float(o["u"]), float(o["v"])]])
        uv = product.undistort(f, uv_d)[0]
        s = product.depth_by_provenance(f, uv, cfg.depth_window_px) if product.has_provenance() else None
        rec = {"frame": f, "u": float(o["u"]), "v": float(o["v"]), "u_und": float(uv[0]),
               "v_und": float(uv[1]), "method": None, "z": None, "n": 0}
        if s is None and not product.has_provenance() and len(usable) < 2:
            s = product.depth_by_zbuffer(f, uv, cfg.zbuffer_window_px)
        if s is not None:
            X = G.unproject(uv[None], np.array([s.z]), product.K(f), product.pose(f))[0]
            pts.append(X); zs.append(s.z); frs.append(f)
            rec.update(method=s.method, z=float(s.z), n=int(s.n), spread=float(s.spread))
        per.append(rec)
    if pts and (product.has_provenance() or len(usable) < 2):
        P = np.asarray(pts)
        m, rms, mx = G.median_point(P)
        return EndResult(m, per[[r["frame"] for r in per].index(frs[0])]["method"]
                         if len(set(r["method"] for r in per if r["method"])) == 1
                         else "mixed", len(obs), len(pts), rms, mx, float(np.median(zs)),
                         float(np.mean([chainage.get(f, float("nan")) for f in frs])), frs, "", per)
    # ── triangulation (two or more observations) ──
    if len(usable) >= 2:
        origins, dirs, frs_t, uvs = [], [], [], []
        for o in usable:
            f = int(o["frame"])
            uv = product.undistort(f, np.array([[float(o["u"]), float(o["v"])]]))[0]
            o_, d_ = G.world_ray(uv[None], product.K(f), product.pose(f))
            origins.append(o_); dirs.append(d_[0]); frs_t.append(f); uvs.append(uv)
        X, parallax = G.triangulate_rays(np.asarray(origins), np.asarray(dirs))
        if parallax >= cfg.min_triangulation_deg:
            z = [float(G.camera_depth(X[None], product.pose(f))[0]) for f in frs_t]
            if all(v > 0 for v in z):
                for r in per:
                    r["method"] = "triangulation"
                return EndResult(X, "triangulation", len(obs), len(frs_t), 0.0, 0.0,
                                 float(np.median(z)),
                                 float(np.mean([chainage.get(f, float("nan")) for f in frs_t])),
                                 frs_t, f"parallax {parallax:.2f} deg", per)
            reason = "the triangulated point lies behind a camera"
        else:
            reason = (f"parallax {parallax:.2f} deg under min_triangulation_deg "
                      f"{cfg.min_triangulation_deg:g}")
        # the z-buffer as the last resort when the product has no provenance
        if not product.has_provenance():
            pts2, zs2, frs2 = [], [], []
            for o, uv in zip(usable, uvs):
                f = int(o["frame"])
                s = product.depth_by_zbuffer(f, uv, cfg.zbuffer_window_px)
                if s is not None:
                    X = G.unproject(uv[None], np.array([s.z]), product.K(f), product.pose(f))[0]
                    pts2.append(X); zs2.append(s.z); frs2.append(f)
            if pts2:
                m, rms, mx = G.median_point(np.asarray(pts2))
                return EndResult(m, "zbuffer", len(obs), len(pts2), rms, mx, float(np.median(zs2)),
                                 float(np.mean([chainage.get(f, float("nan")) for f in frs2])),
                                 frs2, reason + "; z-buffer used instead", per)
        return _not(len(obs), reason)
    if product.has_provenance():
        return _not(len(obs), f"no product point within {cfg.depth_window_px} px of the anchor "
                              f"on keyframe {int(usable[0]['frame'])}")
    return _not(len(obs), f"no dense point within {cfg.zbuffer_window_px} px of the anchor on "
                          f"keyframe {int(usable[0]['frame'])}")
