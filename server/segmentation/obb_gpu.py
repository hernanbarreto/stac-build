"""THE ORIENTED BOUNDING BOX ON THE DEVICE, end to end (USER 2026-11-11: "todo lo que me marcaste que puede
pasarse a GPU hay que hacerlo en GPU, ya lo dije mil quinientas veces. Nada debe hacerse en cpu").

``segmentation.pipeline._compute_obb_record`` — the box until 2026-11-11, kept as the reference — scored
its hypotheses on the card (segmentation/obb_torch.py, 2026-11-01) but DREW them on the CPU with numpy's
seeded Generator (three indices per RANSAC iteration: 5.2 M ``choice`` calls on pccr 3108's projection;
the bootstrap's uniforms), and computed on the CPU the stable keys, the stable sample, the flyer
components' bookkeeping, the sha256 of the sorted keys, the 3×3 eigen-decomposition, the convex hull
(scipy) and the extents. Here every step runs on the evidence device:

* EXACT — the record's results: the per-point stable keys (splitmix64 on int64 bit patterns, the
  right shifts made logical), the stable sample, the flyer components with their shares and verdicts
  (point 145).
* THE SAME CONSTRUCTION, the device's floating point (last-bit differences possible): the convex hull
  by gift wrapping (the polygon qhull returns, collinear points left out), the rotating calipers over
  its edges, the extents, the least-squares normal (torch.linalg.eigh of the 3×3 moment).
* DRAWN ON THE DEVICE (the user's decision: another generator draws other numbers, so the boxes are
  not the record's bit for bit — measured against it on pccr 3108): the RANSAC triplets (three distinct
  indices from three uniforms) and the bootstrap's keep / drop flags come from torch's generator ON THE
  CARD, seeded as before (``obb_orientation.seed``; each footprint's seed derived from the point SET and
  the yaw). The point set's digest is an order-free sum of the mixed stable keys, on the card (the
  record hashed the sorted keys with sha256 on the CPU). Deterministic on one card.

tests/test_obb_gpu.py: the exact parts equal to the record; the drawn parts deterministic and functions
of the point set.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

_C1, _C2, _C3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_TOP = 1 << 63


def _i64(u: int) -> int:
    """A uint64 constant as the int64 with the same bits."""
    return u - (1 << 64) if u >= _TOP else u


def _lshr(h: torch.Tensor, k: int) -> torch.Tensor:
    """Logical right shift of int64 bit patterns (torch's >> is arithmetic)."""
    return (h >> k) & ((1 << (64 - k)) - 1)


def _mix(h: torch.Tensor) -> torch.Tensor:
    """splitmix64's finaliser on int64 bit patterns (multiplication wraps mod 2^64)."""
    h = h ^ _lshr(h, 30)
    h = h * _i64(_C2)
    h = h ^ _lshr(h, 27)
    h = h * _i64(_C3)
    return h ^ _lshr(h, 31)


def stable_point_keys_t(P: torch.Tensor, quantum_m: float = 1e-4) -> torch.Tensor:
    """``segmentation.pipeline._stable_point_keys`` on ``P``'s device: the uint64 keys' bits as int64."""
    q = torch.floor(P / float(quantum_m)).to(torch.int64)
    h = (q[:, 0] * _i64(_C1)) ^ (q[:, 1] * _i64(_C2)) ^ (q[:, 2] * _i64(_C3))
    return _mix(h)


def unsigned_order(keys: torch.Tensor) -> torch.Tensor:
    """int64 values whose signed order is the uint64 order of ``keys`` (the top bit flipped)."""
    return keys ^ _i64(_TOP)


def stable_sample_t(P: torch.Tensor, n: int) -> torch.Tensor:
    """``segmentation.pipeline._stable_sample``: the ``n`` points with the smallest stable keys, as
    indices in ascending order."""
    m = int(P.shape[0])
    if m <= int(n):
        return torch.arange(m, device=P.device)
    order = torch.argsort(unsigned_order(stable_point_keys_t(P)), stable=True)
    return torch.sort(order[: int(n)]).values


def set_digest_t(keys: torch.Tensor) -> str:
    """An order-free digest of a point SET from its stable keys: the count and two wrapping sums of
    differently re-mixed keys (the same set in any order, one digest)."""
    a = _mix(keys ^ _i64(0xA0761D6478BD642F)).sum()
    b = _mix(keys ^ _i64(0xE7037ED1A0B428DB)).sum()
    return f"{int(keys.shape[0])}:{int(a) & ((1 << 64) - 1):016x}:{int(b) & ((1 << 64) - 1):016x}"


def _generator(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & ((1 << 63) - 1))
    return g


def voxel_components_tensor(P: torch.Tensor, voxel_m: float) -> torch.Tensor:
    """``segmentation.voxel_components_torch.voxel_components`` with the points already on the device."""
    from segmentation.voxel_components_torch import _pack, components_of_cells
    k = torch.floor(P / float(voxel_m)).to(torch.int64)
    base = k.min(0).values
    span = k.max(0).values - base + 3
    if float(span[0]) * float(span[1]) * float(span[2]) > 9.0e18:
        return torch.zeros(int(P.shape[0]), dtype=torch.int64, device=P.device)
    cells, inv = torch.unique(_pack(k, base, span), sorted=True, return_inverse=True)
    if int(cells.shape[0]) <= 1:
        return torch.zeros(int(P.shape[0]), dtype=torch.int64, device=P.device)
    return components_of_cells(cells, span)[inv]


def obb_core_points_t(P: torch.Tensor, cfg: dict, factor: float):
    """``segmentation.pipeline._obb_core_points`` on the device: (core indices or None, n dropped, record)."""
    from segmentation.pipeline import _share_verdict
    n = int(P.shape[0])
    rec = {"rule": "component share vs min_component_frac with its binomial error (point 145)",
           "n_points": n, "components": [], "kept_share": None, "kept_verdict": None, "used": "all"}
    if not cfg.get("enabled", True) or n < int(cfg.get("min_points", 200)):
        rec["used"] = "all (disabled or under min_points)"
        return None, 0, rec
    lab = voxel_components_tensor(P, float(cfg.get("voxel_m", 0.05)))
    uniq, counts = torch.unique(lab, sorted=True, return_counts=True)
    if int(uniq.shape[0]) == 1:
        rec["used"] = "all (one component)"
        return None, 0, rec
    bar = float(cfg.get("min_component_frac", 0.02))
    keep = []
    for u, c in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda t: (-t[1], t[0])):
        verdict, share, err = _share_verdict(c, n, bar, factor)
        rec["components"].append({"points": int(c), "share": share, "error": err,
                                  "margin": share - bar, "verdict": verdict,
                                  "in_extent": verdict != "below"})
        if verdict != "below":
            keep.append(u)
    if not keep:
        rec["used"] = "all (no component clears the bar)"
        return None, 0, rec
    core = torch.nonzero(torch.isin(lab, torch.as_tensor(keep, device=P.device))).flatten()
    verdict, share, err = _share_verdict(int(core.shape[0]), n, float(cfg.get("min_keep_frac", 0.5)), factor)
    rec["kept_share"], rec["kept_error"], rec["kept_verdict"] = share, err, verdict
    if verdict != "above" or int(core.shape[0]) < 4:
        rec["used"] = "all (the kept share does not clear min_keep_frac by its error)"
        return None, 0, rec
    rec["used"] = "core"
    return core, n - int(core.shape[0]), rec


def fold_yaw_t(a: torch.Tensor) -> torch.Tensor:
    """``segmentation.pipeline._fold_yaw`` elementwise: into [0, π/2)."""
    return torch.remainder(a, math.pi / 2.0)


def _triplets(n: int, iters: int, g: torch.Generator, device) -> torch.Tensor:
    """(iters, 3) distinct indices in [0, n) from three uniforms each, drawn on the device: the first
    uniform over n, the second over n − 1 skipping the first, the third over n − 2 skipping both."""
    u = torch.rand((iters, 3), generator=g, device=device, dtype=torch.float64)
    i0 = torch.floor(u[:, 0] * n).to(torch.int64)
    i1 = torch.floor(u[:, 1] * (n - 1)).to(torch.int64)
    i1 = i1 + (i1 >= i0).to(torch.int64)
    lo, hi = torch.minimum(i0, i1), torch.maximum(i0, i1)
    i2 = torch.floor(u[:, 2] * (n - 2)).to(torch.int64)
    i2 = i2 + (i2 >= lo).to(torch.int64)
    i2 = i2 + (i2 >= hi).to(torch.int64)
    return torch.stack([i0, i1, i2], 1)


def vertical_plane_yaws_g(P: torch.Tensor, vcfg: dict) -> List[Tuple[float, int]]:
    """The record's sequential RANSAC of the object's VERTICAL planes (point 143) with the hypotheses
    drawn on the device: [(folded yaw, support)]."""
    keys = ("ransac_iters", "dist_m", "vertical_tol_deg", "max_planes", "sample", "seed")
    missing = [k for k in keys if k not in vcfg]
    if missing:
        raise KeyError(f"config.yaml segmentation.obb_orientation is missing {missing}")
    if "min_plane_frac" in vcfg:
        raise KeyError("config.yaml segmentation.obb_orientation.min_plane_frac was removed "
                       "(docs/plan_determinismo.md point 143: every vertical RANSAC plane is a "
                       "candidate, its support decides among footprint ties) — delete the key")
    if int(P.shape[0]) < 10:
        return []
    dev = P.device
    g = _generator(int(vcfg["seed"]), dev)
    rest = P[stable_sample_t(P, int(vcfg["sample"]))]
    sin_tol = math.sin(math.radians(float(vcfg["vertical_tol_deg"])))
    dist = float(vcfg["dist_m"])
    iters = int(vcfg["ransac_iters"])
    yaws: List[Tuple[float, int]] = []
    for _ in range(int(vcfg["max_planes"])):
        n_rest = int(rest.shape[0])
        if n_rest < 3:
            break
        tri = rest[_triplets(n_rest, iters, g, dev)]                           # (iters, 3, 3)
        a = tri[:, 0]
        nrm = torch.cross(tri[:, 1] - a, tri[:, 2] - a, dim=1)
        ln = torch.linalg.norm(nrm, dim=1)
        ok = ln >= 1e-12
        nrm = nrm / torch.where(ok, ln, torch.ones_like(ln))[:, None]
        vertical = nrm[:, 1].abs() <= sin_tol                                  # a horizontal face: no yaw
        dd = -(nrm * a).sum(1)
        counts = torch.full((iters,), -1, dtype=torch.int64, device=dev)
        for s in range(0, iters, 64):
            e = min(s + 64, iters)
            counts[s:e] = ((rest @ nrm[s:e].T + dd[s:e]).abs() < dist).sum(0)
        counts = torch.where(ok & vertical, counts, torch.full_like(counts, -1))
        best = int(torch.argmax(counts))                                       # the first maximal count
        best_cnt = int(counts[best])
        if best_cnt < 3:
            break
        inl = (rest @ nrm[best] + dd[best]).abs() < dist
        Q = rest[inl]
        Qc = Q - Q.mean(0)
        _, vecs = torch.linalg.eigh(Qc.T @ Qc)
        nn = vecs[:, 0]                                                        # the least-squares normal
        yaws.append((float(fold_yaw_t(torch.atan2(nn[2], nn[0]))), best_cnt))
        rest = rest[~inl]
    return yaws


def footprint_error_g(r_all: torch.Tensor, keys_u: torch.Tensor, seed: int, n_boot: int, poisson_cdf0: float,
                      extremes: int) -> float:
    """σ of the footprint area of the rotated points ``r_all`` (n, 2) by the record's Poisson bootstrap
    over the ordered extremes, the keep / drop flags drawn on the device from ``seed``."""
    from segmentation.obb_torch import _extremes_t
    n = int(r_all.shape[0])
    if n < 4:
        return float("inf")
    dev = r_all.device
    T = min(int(extremes), n)
    order = {ax: (_extremes_t(-r_all[:, ax], keys_u, T), _extremes_t(r_all[:, ax], keys_u, T)) for ax in (0, 1)}
    cand = torch.unique(torch.cat([order[0][0], order[0][1], order[1][0], order[1][1]]), sorted=True)
    o = torch.argsort(cand, stable=True)
    o = o[torch.argsort(keys_u[cand][o], stable=True)]                          # by stable key, then index
    cand = cand[o]
    pos = torch.full((n,), -1, dtype=torch.int64, device=dev)
    pos[cand] = torch.arange(int(cand.shape[0]), device=dev)
    g = _generator(seed, dev)
    kept = torch.rand((int(n_boot), int(cand.shape[0])), generator=g, device=dev,
                      dtype=torch.float64) >= float(poisson_cdf0)
    full_ext = r_all.max(0).values - r_all.min(0).values
    ext = torch.empty((int(n_boot), 2), dtype=torch.float64, device=dev)
    for ax in (0, 1):
        hi_o, lo_o = order[ax]
        kh, kl = kept[:, pos[hi_o]], kept[:, pos[lo_o]]
        hi = r_all[hi_o[torch.argmax(kh.to(torch.int8), dim=1)], ax]
        lo = r_all[lo_o[torch.argmax(kl.to(torch.int8), dim=1)], ax]
        ext[:, ax] = torch.where(kh.any(1) & kl.any(1), hi - lo, full_ext[ax])
    areas = ext[:, 0] * ext[:, 1]
    return float(torch.std(areas, correction=1)) if n_boot > 1 else 0.0


def _rotate(P2: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Points (n, 2) in the frame rotated by −yaw (the record's ``P @ [[c, −s], [s, c]].T``, c = cos(−yaw))."""
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([c * P2[:, 0] - s * P2[:, 1], s * P2[:, 0] + c * P2[:, 1]], 1)


def convex_hull_t(P2: torch.Tensor) -> torch.Tensor:
    """The convex hull's vertices (h, 2), counter-clockwise, collinear points left out — gift wrapping on
    the device (each step: the smallest left turn from the last edge, the farthest point of a tie)."""
    n = int(P2.shape[0])
    if n < 3:
        return P2
    o = torch.argsort(P2[:, 1], stable=True)
    o = o[torch.argsort(P2[o, 0], stable=True)]
    start = int(o[0])                                                         # lowest x, then lowest z
    hull = [start]
    cur = start
    prev = torch.tensor([0.0, -1.0], dtype=P2.dtype, device=P2.device)
    for _ in range(n):
        d = P2 - P2[cur]
        dist2 = (d * d).sum(1)
        ang = torch.atan2(prev[0] * d[:, 1] - prev[1] * d[:, 0], prev[0] * d[:, 0] + prev[1] * d[:, 1])
        ang = torch.where(ang < 0, ang + 2.0 * math.pi, ang)
        ang = torch.where(dist2 > 0, ang, torch.full_like(ang, float("inf")))
        a_min = ang.min()
        if not bool(torch.isfinite(a_min)):
            break                                                             # every point is the same point
        nxt = int(torch.argmax(torch.where(ang == a_min, dist2, torch.full_like(dist2, -1.0))))
        if nxt == start:
            break
        hull.append(nxt)
        prev = d[nxt] / torch.sqrt(dist2[nxt])
        cur = nxt
    return P2[torch.as_tensor(hull, device=P2.device)]


def calipers_t(H: torch.Tensor) -> Tuple[float, float]:
    """The record's rotating calipers over the hull's edges: (yaw, area) of the minimum-area rectangle —
    a strictly smaller area, or the same area at a smaller yaw."""
    E = torch.roll(H, -1, 0) - H
    ok = torch.linalg.norm(E, dim=1) >= 1e-10
    if not bool(ok.any()):
        return 0.0, float("inf")
    yaw = fold_yaw_t(torch.atan2(E[:, 1], E[:, 0]))[ok]
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x = c[:, None] * H[None, :, 0] - s[:, None] * H[None, :, 1]
    z = s[:, None] * H[None, :, 0] + c[:, None] * H[None, :, 1]
    area = (x.max(1).values - x.min(1).values) * (z.max(1).values - z.min(1).values)
    o = torch.argsort(yaw, stable=True)
    o = o[torch.argsort(area[o], stable=True)]
    return float(yaw[o[0]]), float(area[o[0]])


def compute_obb_t(points_xyz: np.ndarray, face_normals=None, cfg: Optional[dict] = None,
                  device: Optional[str] = None) -> dict:
    """``segmentation.pipeline._compute_obb_record`` on the device (points 109 / 143 / 145): the same rules,
    the same record — the draws on the card."""
    from correction.floor import N_BOOT_FLOOR, _POISSON1_CDF
    from repro import exact_float, stable_id
    from reconstruction.surface_fit.hole_audit import evidence_device
    from segmentation.pipeline import FOOTPRINT_EXTREMES, _decision_params
    dev = torch.device(evidence_device() if device is None else device)
    P_all = torch.as_tensor(np.ascontiguousarray(points_xyz, np.float64), device=dev)
    if int(P_all.shape[0]) < 4:
        center = P_all.mean(0).cpu().numpy()
        return {"center": center.tolist(), "half_extents": [0.01, 0.01, 0.01],
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    if cfg is None:
        from config import cfg as _server_cfg
    else:
        _server_cfg = cfg
    factor = _decision_params(_server_cfg)[0]
    _ocfg = ((_server_cfg.get("segmentation", {}) or {}).get("obb_core", {}) or {})
    core, n_dropped, core_rec = obb_core_points_t(P_all, _ocfg, factor)
    P = P_all[core] if core is not None else P_all

    y_min, y_max = float(P[:, 1].min()), float(P[:, 1].max())
    half_y = (y_max - y_min) / 2.0
    cy = (y_min + y_max) / 2.0
    pts_xz = P[:, [0, 2]]

    _vcfg = ((_server_cfg.get("segmentation", {}) or {}).get("obb_orientation", {}) or {})
    cands = list(vertical_plane_yaws_g(P, _vcfg)) if _vcfg.get("enabled", False) else []
    for fn in (face_normals or []):
        n3 = np.asarray(fn[0], np.float64)
        nxz = np.array([n3[0], n3[2]])
        if np.linalg.norm(nxz) > 0.1:
            cands.append((float(fold_yaw_t(torch.tensor(math.atan2(nxz[1], nxz[0]), dtype=torch.float64))),
                          int(fn[1]) if len(fn) > 1 else 0))

    yaw_rec = {"rule": "dominant plane among footprints within factor x error of the minimum "
                       "(points 109 / 143); error = sigma of the footprint area by a seeded "
                       "Poisson bootstrap over the points (correction.floor's construction), "
                       "drawn on the device (USER 2026-11-11)",
               "candidates": [], "chosen": None}
    best_angle = 0.0
    if cands:
        # the point SET's identity, once per box: the stable keys of (x, 0, z) and their digest
        P3 = torch.stack([pts_xz[:, 0], torch.zeros_like(pts_xz[:, 0]), pts_xz[:, 1]], 1)
        keys = stable_point_keys_t(P3)
        keys_u = unsigned_order(keys)
        digest = set_digest_t(keys)
        n_boot = int(N_BOOT_FLOOR)
        rows = []
        for yaw, support in cands:
            r = _rotate(pts_xz, torch.tensor(float(yaw), dtype=torch.float64, device=dev))
            e = r.max(0).values - r.min(0).values
            seed = int(stable_id("obb_footprint_bootstrap_device", digest, exact_float(float(yaw)), n_hex=8), 16)
            err = footprint_error_g(r, keys_u, seed, n_boot, float(_POISSON1_CDF[0]), int(FOOTPRINT_EXTREMES))
            rows.append((float(yaw), int(support), float(e[0] * e[1]), float(err), n_boot, seed))
        a_min = min(r_[2] for r_ in rows)
        err_min = next(r_[3] for r_ in rows if r_[2] == a_min)
        tied = [r_ for r_ in rows if r_[2] - a_min <= factor * err_min]
        chosen = max(tied, key=lambda r_: (r_[1], -r_[2], -r_[0]))
        best_angle = chosen[0]
        for r_ in sorted(rows, key=lambda r_: (r_[2], -r_[1], r_[0])):
            yaw_rec["candidates"].append({"yaw_deg": float(np.degrees(r_[0])), "support": r_[1],
                                          "footprint_m2": r_[2], "footprint_error_m2": r_[3],
                                          "error_n_boot": r_[4], "error_seed": r_[5],
                                          "margin_m2": r_[2] - a_min,
                                          "tied_with_minimum": bool(r_[2] - a_min <= factor * err_min)})
        yaw_rec["chosen"] = {"yaw_deg": float(np.degrees(best_angle)), "support": chosen[1],
                             "footprint_m2": chosen[2], "footprint_error_m2": err_min,
                             "error_factor": float(factor), "n_tied": len(tied)}
    else:
        best_angle, best_area = calipers_t(convex_hull_t(pts_xz))
        yaw_rec["chosen"] = {"yaw_deg": float(np.degrees(best_angle)), "support": None,
                             "footprint_m2": float(best_area) if math.isfinite(best_area) else None,
                             "n_tied": None, "source": "rotating calipers (no vertical plane)"}

    rot = _rotate(pts_xz, torch.tensor(best_angle, dtype=torch.float64, device=dev))
    rmin, rmax = rot.min(0).values.cpu().numpy(), rot.max(0).values.cpu().numpy()
    half_x = (rmax[0] - rmin[0]) / 2.0
    half_z = (rmax[1] - rmin[1]) / 2.0
    cx_rot = (rmax[0] + rmin[0]) / 2.0
    cz_rot = (rmax[1] + rmin[1]) / 2.0
    cos_back, sin_back = math.cos(best_angle), math.sin(best_angle)
    center_xz = (cos_back * cx_rot - sin_back * cz_rot, sin_back * cx_rot + cos_back * cz_rot)
    return {
        "center": [float(center_xz[0]), float(cy), float(center_xz[1])],
        "half_extents": [float(half_x), float(half_y), float(half_z)],
        "rotation": [[float(cos_back), 0.0, float(-sin_back)], [0.0, 1.0, 0.0],
                     [float(sin_back), 0.0, float(cos_back)]],
        "n_points": int(P_all.shape[0]),
        "n_points_used": int(P.shape[0]),
        "n_flyers_excluded": int(n_dropped),
        "decisions": {"yaw": yaw_rec, "extent": core_rec},
    }
