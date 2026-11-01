"""The oriented bounding box's measurements on the device (USER 2026-11-01: "todo debe correr en GPU").

`segmentation.pipeline._compute_obb` is run for every instance of the projection and again for every
OBB the certification republishes (484 on pccr audit5, 1 min 34 s). Its three numpy kernels move here, the
numpy functions kept as the references:

* ``vertical_plane_yaws_t`` — `_vertical_plane_yaws`: the sequential RANSAC of the object's VERTICAL
  planes with the SAME hypotheses (numpy's seeded `Generator.choice`, drawn in the loop's order, one
  triplet per iteration whether or not the iteration counts), every hypothesis scored on the device
  (|(p − a)·n| < dist), the first maximal count (numpy's strict `>`), the least-squares normal of the
  inliers from their centred second moments (numpy's `eigh` of the 3×3 — the recipe's SVD of the centred
  points gives the same direction to rounding, the yaw folds the sign away);
* ``footprint_error_t`` — `_footprint_error`: the Poisson bootstrap of the footprint area at a yaw with
  the SAME uniforms (the recipe's seeded numpy stream, uploaded), the ordered extremes of each rotated
  axis by a stable sort by (value, stable key) — the recipe's "every value strictly beyond the T-th, then
  the ties at it in key order" — the resample extents by the first kept extreme, the sample standard
  deviation;
* ``extents_t`` — the rotated footprint's extents.

Equal to the numpy references to the rounding of float64 sums (tests/test_obb_torch.py: the same yaws, the
same candidate record, σ to 1e-9 relative).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def _t(a, device, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(dtype)


def vertical_plane_yaws_t(P: np.ndarray, vcfg: dict, device: str, fold_yaw, stable_sample) -> list:
    """`_vertical_plane_yaws` on ``device``: [(yaw, support)] of each vertical RANSAC plane of the stable
    sample of ``P``. ``fold_yaw`` / ``stable_sample``: the pipeline's own helpers (one definition)."""
    keys = ("ransac_iters", "dist_m", "vertical_tol_deg", "max_planes", "sample", "seed")
    missing = [k for k in keys if k not in vcfg]
    if missing:
        raise KeyError(f"config.yaml segmentation.obb_orientation is missing {missing}")
    if "min_plane_frac" in vcfg:
        raise KeyError("config.yaml segmentation.obb_orientation.min_plane_frac was removed "
                       "(docs/plan_determinismo.md point 143: every vertical RANSAC plane is a "
                       "candidate, its support decides among footprint ties) — delete the key")
    P = np.asarray(P, np.float64)
    if len(P) < 10:
        return []
    rng = np.random.default_rng(int(vcfg["seed"]))
    P = P[stable_sample(P, int(vcfg["sample"]))]
    sin_tol = float(np.sin(np.radians(float(vcfg["vertical_tol_deg"]))))
    dist = float(vcfg["dist_m"])
    iters = int(vcfg["ransac_iters"])
    rest = _t(P, device)
    yaws: list = []
    for _ in range(int(vcfg["max_planes"])):
        n_rest = int(rest.shape[0])
        if n_rest < 3:
            break
        S = np.stack([rng.choice(n_rest, 3, replace=False) for _ in range(iters)]).astype(np.int64)
        tri = rest[_t(S, device, torch.int64)]                                      # (iters, 3, 3)
        a = tri[:, 0]
        nrm = torch.cross(tri[:, 1] - a, tri[:, 2] - a, dim=1)
        ln = torch.linalg.norm(nrm, dim=1)
        ok = ln >= 1e-12
        nrm = nrm / torch.where(ok, ln, torch.ones_like(ln))[:, None]
        vertical = nrm[:, 1].abs() <= sin_tol                                       # not vertical → no yaw
        dd = -(nrm * a).sum(1)
        counts = torch.full((iters,), -1, dtype=torch.int64, device=device)
        chunk = 64
        for s in range(0, iters, chunk):
            e = min(s + chunk, iters)
            d = (rest @ nrm[s:e].T + dd[s:e]).abs()
            counts[s:e] = (d < dist).sum(0)
        counts = torch.where(ok & vertical, counts, torch.full_like(counts, -1))
        best = int(torch.argmax(counts))
        best_cnt = int(counts[best])
        if best_cnt < 3:
            break
        inl = (rest @ nrm[best] + dd[best]).abs() < dist
        Q = rest[inl]
        c0 = Q.mean(0)
        Qc = Q - c0
        C = (Qc.T @ Qc).cpu().numpy()
        _, vecs = np.linalg.eigh(C)
        n = vecs[:, 0]                                                              # least-squares normal
        yaws.append((fold_yaw(np.arctan2(n[2], n[0])), best_cnt))
        rest = rest[~inl]
    return yaws


def _extremes_t(s: torch.Tensor, keys: torch.Tensor, T: int) -> torch.Tensor:
    """`_extreme_indices` for the ascending case on the device: the indices of the T smallest ``s``,
    ordered by (value, stable key) — a stable sort by key then by value."""
    n = int(s.shape[0])
    o = torch.argsort(keys, stable=True)
    o = o[torch.argsort(s[o], stable=True)]
    return o[: min(int(T), n)]


def footprint_error_t(points_xz: np.ndarray, yaw: float, identity: Tuple[np.ndarray, str], seed_of,
                      n_boot: int, poisson_cdf0: float, extremes: int, device: str) -> Tuple[float, int, int]:
    """`_footprint_error` on ``device``: (σ of the footprint area at ``yaw`` in m², n_boot, seed), the SAME
    seeded uniforms (numpy's stream), the resample extents from the ordered extremes of each rotated axis."""
    P = np.asarray(points_xz, np.float64)
    n = len(P)
    keys_np, digest = identity
    if len(keys_np) != n:
        raise ValueError(f"_footprint_error: an identity of {len(keys_np)} keys for {n} points")
    seed = seed_of(digest, float(yaw))
    if n < 4:
        return float("inf"), int(n_boot), seed
    c, s_ = float(np.cos(-float(yaw))), float(np.sin(-float(yaw)))
    Pt = _t(P, device)
    R = torch.tensor([[c, -s_], [s_, c]], dtype=torch.float64, device=device)
    r = Pt @ R.T                                                                   # the rotated frame
    # the stable keys are uint64: the top bit flipped, their order survives in int64 (torch's sort)
    keys = torch.as_tensor(np.ascontiguousarray((np.asarray(keys_np, np.uint64) ^ np.uint64(1 << 63)).view(np.int64)),
                           device=device)
    T = min(int(extremes), n)
    order = {}
    for ax in (0, 1):
        hi_o = _extremes_t(-r[:, ax], keys, T)                                     # descending maxima
        lo_o = _extremes_t(r[:, ax], keys, T)                                      # ascending minima
        order[ax] = (hi_o, lo_o)
    cand = torch.unique(torch.cat([order[0][0], order[0][1], order[1][0], order[1][1]]), sorted=True)
    ck = keys[cand]
    o = torch.argsort(cand, stable=True)
    o = o[torch.argsort(ck[o], stable=True)]                                        # by stable key, then index
    cand = cand[o]
    pos = torch.full((n,), -1, dtype=torch.int64, device=device)
    pos[cand] = torch.arange(int(cand.shape[0]), device=device)
    rng = np.random.default_rng(seed)
    kept = _t(rng.random((int(n_boot), int(cand.shape[0]))), device) >= float(poisson_cdf0)
    full_ext = r.max(0).values - r.min(0).values
    ext = torch.empty((int(n_boot), 2), dtype=torch.float64, device=device)
    for ax in (0, 1):
        hi_o, lo_o = order[ax]
        kh, kl = kept[:, pos[hi_o]], kept[:, pos[lo_o]]
        hi = r[hi_o[torch.argmax(kh.to(torch.int8), dim=1)], ax]
        lo = r[lo_o[torch.argmax(kl.to(torch.int8), dim=1)], ax]
        both = kh.any(1) & kl.any(1)
        ext[:, ax] = torch.where(both, hi - lo, full_ext[ax])
    areas = (ext[:, 0] * ext[:, 1]).cpu().numpy()
    return (float(np.std(areas, ddof=1)) if n_boot > 1 else 0.0), int(n_boot), seed


def extents_t(points_xz: np.ndarray, yaws: List[float], device: str) -> np.ndarray:
    """The footprint extents (max − min of the rotated x and z) at every yaw: (len(yaws), 2)."""
    Pt = _t(np.asarray(points_xz, np.float64), device)
    out = np.empty((len(yaws), 2))
    for k, a in enumerate(yaws):
        c, s_ = float(np.cos(-a)), float(np.sin(-a))
        R = torch.tensor([[c, -s_], [s_, c]], dtype=torch.float64, device=device)
        r = Pt @ R.T
        out[k] = (r.max(0).values - r.min(0).values).cpu().numpy()
    return out
