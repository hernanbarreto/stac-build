"""The floor solve's kernels on the card (USER 2026-11-01: "todo debe correr en GPU … hacelo todo a GPU ya
mismo").

MEASURED on pccr (epoch 4, 19.4 M points, 295 keyframes, analysis/2026-11-01_floor_gpu/pccr_numpy_profile.json):
the session floor solve took 131 s on numpy, 109 s of them in FOUR Tukey IRLS plane fits — 850 iterations
over 3.8 M support points, each a weighted PCA that reads the points three times (`_weighted_plane`, 0.1 s a
call); the bootstraps 5.5 s, the level judge 1.2 s, the support rows 2.3 s (the segmentation's JSON). In the
certification of audit5 (23 M points, a 2.5° tilt) the same stage took 6 min 34 s.

Here the same recipes run on torch tensors in float64 on ONE device, the numpy path kept as the reference:

* ``irls_plane_t`` — `correction.floor._irls_plane` iteration for iteration: the Tukey biweight at the band,
  the weighted PCA (the 3×3 moment matrix built on the device, its eigenvectors by numpy's `eigh` — the
  reference's routine — on the CPU), the same stopping rules (IRLS_TOL on the normal and the centroid,
  IRLS_MAX_ITER per stage, the band halved to the end band);
* ``bootstrap_plane_t`` — `_bootstrap_plane` with the SAME Poisson(1) draws: the uniforms come from the
  reference's seeded numpy generator in the reference's order (one `Generator.random` stream, read in
  blocks), inverted on the device (`searchsorted` on the Poisson CDF = the reference's cascade), the ten
  moments of every resample by one matmul, the planes by the same batched `eigh`;
* ``level_judge_t`` — `_level_judge`'s per-keyframe medians (the distances on the device, the exact median:
  sorted within each keyframe, the mean of the two middle values as numpy's), the verdict by the fork's
  `decide_change` on the 295 medians as before;
* ``nearest_xz_t`` — the horizontal nearest-neighbour distance the footprint rules read from a k-d tree
  (`cKDTree.query`, k = 1): the exact minimum over the other set, blocked;
* ``chunk_candidates`` — `_chunk_candidates` with those kernels, the pool / band / rest masks as tensors,
  numpy's linear percentile reproduced operation for operation (the band's seed).

Differences from the reference are the rounding of float64 sums in another order (~1e-15 relative on a
plane's normal): tests/test_correction_floor_torch.py holds the two paths equal on the synthetic sessions —
the same levels, the same inliers, the same σ to 1e-9, the same verdicts, the same rigid motion. The one
record that may differ is the IRLS iteration count: its 1e-9 rad convergence test fires when the dot product
of two successive normals rounds to 1.0 exactly (arccos(1 − 1e-16) = 1.4e-8 rad), so which iteration that
is depends on the order of the sums; the plane it stops at agrees to 1e-9.

Selected by ``correction.floor.device``: numpy (the path of record) | cpu (the torch path on the CPU — the
tests) | cuda. A cuda with no visible card FAILS naming the key (the F6 rule): nothing runs somewhere else.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

DEVICES = ("numpy", "cpu", "cuda")
_UP = np.array([0.0, 1.0, 0.0])


def _torch():
    import torch
    return torch


def resolve_device(name: str) -> str:
    """'numpy' | 'cpu' | 'cuda', validated; cuda with no visible card FAILS naming the key."""
    name = str(name)
    if name not in DEVICES:
        raise ValueError(f"correction.floor.device must be one of {list(DEVICES)}, got {name!r}")
    if name == "cuda" and not _torch().cuda.is_available():
        raise RuntimeError("correction.floor.device is 'cuda' and torch sees no card — the floor solve does "
                           "not fall back to the CPU by itself; declare cpu or numpy")
    return name


def _t(a, device, dtype=None):
    torch = _torch()
    if dtype is None:
        dtype = torch.float64
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(dtype)


# ── the weighted plane and the Tukey IRLS ──────────────────────────────────────────────────────────────

def weighted_plane_t(P, w) -> Tuple[np.ndarray, np.ndarray]:
    """(normal with y ≥ 0, centroid) of the weighted PCA plane through the tensor ``P`` (n, 3) with
    weights ``w`` (n,) — `correction.floor._weighted_plane` on the device: the centroid and the
    centred second moments as tensors, numpy's `eigh` of the 3×3 on the CPU (one transfer)."""
    torch = _torch()
    sw = w.sum()
    cen = (P * w[:, None]).sum(0) / sw
    Q = P - cen
    C = (Q * w[:, None]).T @ Q / sw
    packed = torch.cat([C.reshape(-1), cen]).cpu().numpy()
    C_np = packed[:9].reshape(3, 3)
    vals, vecs = np.linalg.eigh(C_np)
    nrm = vecs[:, int(np.argmin(vals))]
    if nrm[1] < 0:
        nrm = -nrm
    return nrm, packed[9:12].copy()


def irls_plane_t(P, start_band_m: float, end_band_m: float, max_iter: int, tol: float
                 ) -> Tuple[np.ndarray, np.ndarray, "object", dict]:
    """`correction.floor._irls_plane` on the tensor ``P`` (n, 3): Tukey IRLS plane from the horizontal
    level of the band, the biweight scale continued from ``start_band_m`` down to ``end_band_m``.
    Returns (normal, centroid — numpy, as the reference's —, inliers within the end band as a bool
    tensor, record)."""
    torch = _torch()
    dev = P.device
    n = int(P.shape[0])
    # the start: horizontal through the mean (x, z) and the MEDIAN y (numpy's median: the mean of the
    # two middle values of the sorted column)
    ys = torch.sort(P[:, 1]).values
    med_y = (ys[(n - 1) >> 1] + ys[n >> 1]) / 2.0          # the two middle entries (one twice when n is odd)
    start = torch.stack([P[:, 0].mean(), med_y, P[:, 2].mean()]).cpu().numpy()
    nrm = _UP.copy()
    cen = np.asarray(start, np.float64)
    c = float(start_band_m)
    stages, iters, converged = 0, 0, True
    while True:
        for _ in range(int(max_iter)):
            iters += 1
            cen_t = _t(cen, dev)
            nrm_t = _t(nrm, dev)
            d = (P - cen_t) @ nrm_t
            inside = d.abs() < c
            w = torch.where(inside, (1.0 - (d / c) ** 2) ** 2, torch.zeros((), dtype=P.dtype, device=dev))
            flags = torch.stack([w.sum(), inside.sum().to(P.dtype)]).cpu().numpy()
            if flags[0] <= 0 or int(flags[1]) < 3:
                break
            nrm2, cen2 = weighted_plane_t(P, w)
            moved_n = float(np.arccos(np.clip(abs(float(nrm2 @ nrm)), -1.0, 1.0)))
            moved_c = float(np.linalg.norm(cen2 - cen))
            nrm, cen = nrm2, cen2
            if moved_n < tol and moved_c < tol:
                break
        else:
            converged = False
        stages += 1
        if c <= float(end_band_m) * (1.0 + tol):
            break
        c = max(c * 0.5, float(end_band_m))
    d = (P - _t(cen, dev)) @ _t(nrm, dev)
    inl = d.abs() < float(end_band_m)
    return nrm, cen, inl, {"stages": stages, "iterations": iters, "converged": converged}


# ── the Poisson bootstrap of a plane ──────────────────────────────────────────────────────────────────

def bootstrap_plane_t(P, seed: int, n_boot: int, poisson_cdf: np.ndarray, tol: float, block: int = 200,
                      draws: str = "numpy") -> Tuple[float, float]:
    """`correction.floor._bootstrap_plane` on the tensor ``P`` (n, 3): (σ of the tilt in degrees, σ of
    the height in metres) over ``n_boot`` Poisson(1) resamples. ``draws`` = ``numpy``: the uniforms are
    the reference's — `default_rng(seed).random`, one stream read ``block`` draws at a time (the same
    doubles in the same order as its draw-by-draw calls), the same σ to 1e-9; ``device``: torch's own
    seeded generator on the device (`correction.floor.bootstrap_draws`, 2026-11-01 — the numpy stream
    cost 15 s per chunk floor on pccr's 1 M-point inlier sets, 2000 × n doubles on the CPU; the σ then
    differs from the reference's by the bootstrap's own resampling noise, ~1/√(2·n_boot) relative, and
    the record says which draws it carries). Either way the uniforms are inverted on the device; every
    resample's weighted PCA from the ten per-point moments about the mean by one matmul; the planes by
    numpy's batched `eigh`."""
    torch = _torch()
    dev = P.device
    n = int(P.shape[0])
    if n < 4:
        return float("inf"), float("inf")
    if draws not in ("numpy", "device"):
        raise ValueError(f"bootstrap draws must be 'numpy' or 'device', got {draws!r}")
    rng = np.random.default_rng(int(seed))
    gen = None
    if draws == "device":
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
    c0 = P.mean(0)
    Q = P - c0
    one = torch.ones(n, dtype=P.dtype, device=dev)
    M = torch.stack([one, Q[:, 0], Q[:, 1], Q[:, 2],
                     Q[:, 0] * Q[:, 0], Q[:, 0] * Q[:, 1], Q[:, 0] * Q[:, 2],
                     Q[:, 1] * Q[:, 1], Q[:, 1] * Q[:, 2], Q[:, 2] * Q[:, 2]], 1)      # (n, 10)
    cdf = _t(poisson_cdf, dev)
    S = torch.empty((int(n_boot), 10), dtype=P.dtype, device=dev)
    for a in range(0, int(n_boot), int(block)):
        b = min(int(block), int(n_boot) - a)
        if gen is None:
            u = _t(rng.random((b, n)), dev)
        else:
            u = torch.rand((b, n), generator=gen, dtype=P.dtype, device=dev)
        w = torch.searchsorted(cdf, u, right=True).to(P.dtype)                        # Poisson(1) draws
        sw = w.sum(1)
        w = torch.where((sw <= 0)[:, None], one[None, :].expand(b, n), w)             # the reference's guard
        S[a:a + b] = w @ M
    S = S.cpu().numpy()
    sw = S[:, 0]
    mu = S[:, 1:4] / sw[:, None]
    C = np.empty((len(S), 3, 3))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = S[:, 4], S[:, 5], S[:, 6]
    C[:, 1, 0], C[:, 1, 1], C[:, 1, 2] = S[:, 5], S[:, 7], S[:, 8]
    C[:, 2, 0], C[:, 2, 1], C[:, 2, 2] = S[:, 6], S[:, 8], S[:, 9]
    C = C / sw[:, None, None] - mu[:, :, None] * mu[:, None, :]
    vals, vecs = np.linalg.eigh(C)
    nrm = np.take_along_axis(vecs, np.argmin(vals, axis=1)[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
    nrm = np.where((nrm[:, 1] < 0)[:, None], -nrm, nrm)
    tilts = np.degrees(np.arccos(np.clip(nrm @ _UP, -1.0, 1.0)))
    c0_np = c0.cpu().numpy()
    cen = c0_np[None, :] + mu
    ny = np.where(np.abs(nrm[:, 1]) > tol, nrm[:, 1], tol)
    heights = cen[:, 1] - (nrm[:, 0] * (c0_np[0] - cen[:, 0]) + nrm[:, 2] * (c0_np[2] - cen[:, 2])) / ny
    return float(np.std(tilts, ddof=1)), float(np.std(heights, ddof=1))


# ── the level judge ───────────────────────────────────────────────────────────────────────────────────

def segment_medians_t(values, keys) -> Tuple[np.ndarray, np.ndarray]:
    """(unique keys ascending, the exact median of ``values`` per key): `np.median` over `ks == k` for
    every `k in np.unique(ks)`, on the device — the values sorted within each key's segment (two stable
    sorts), the mean of the two middle entries (one entry read twice when the count is odd)."""
    torch = _torch()
    o1 = torch.argsort(values, stable=True)
    v = values[o1]
    k = keys[o1]
    o2 = torch.argsort(k, stable=True)
    v = v[o2]
    k = k[o2]
    uniq, counts = torch.unique_consecutive(k, return_counts=True)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    lo = starts + ((counts - 1) >> 1)
    hi = starts + (counts >> 1)
    med = (v[lo] + v[hi]) / 2.0
    return uniq.cpu().numpy(), med.cpu().numpy()


def level_judge_t(P, ks, planes: List[Tuple[np.ndarray, np.ndarray]], rep_m: float, fac: float, conf: float,
                  decide_change) -> dict:
    """`correction.floor._level_judge` with the distances and the per-keyframe medians on the device:
    per keyframe the median distance of its points to the FIRST plane (before) against the nearest of
    all the planes (after); the verdict by ``decide_change`` (the fork's) as before."""
    torch = _torch()
    dev = P.device
    d = torch.stack([((P - _t(cen, dev)) @ _t(nrm, dev)).abs() for nrm, cen in planes], 1)
    d1 = d[:, 0]
    dn = d.min(1).values
    _, before = segment_medians_t(d1, ks)
    _, after = segment_medians_t(dn, ks)
    return decide_change(np.asarray(before), np.asarray(after), error=float(rep_m),
                         error_factor=float(fac), confidence=float(conf))


# ── the footprint distance ────────────────────────────────────────────────────────────────────────────

def nearest_xz_t(A, B, block_entries: int = 50_000_000):
    """For every row of ``A`` (a, 2) the distance to its nearest row of ``B`` (b, 2) — `cKDTree(B).query(A,
    k=1)` on the device: the exact minimum of sqrt(dx² + dz²) over ``B``, blocked over ``A`` so no
    distance matrix exceeds ``block_entries`` entries. Returns a tensor (a,)."""
    torch = _torch()
    a, b = int(A.shape[0]), int(B.shape[0])
    out = torch.empty(a, dtype=A.dtype, device=A.device)
    if a == 0:
        return out
    if b == 0:
        out.fill_(float("inf"))
        return out
    step = max(1, int(block_entries) // b)
    for s in range(0, a, step):
        dx = A[s:s + step, 0][:, None] - B[None, :, 0]
        dz = A[s:s + step, 1][:, None] - B[None, :, 1]
        d2 = dx * dx + dz * dz
        out[s:s + step] = torch.sqrt(d2.min(1).values)
    return out


# ── numpy's linear percentile, operation for operation ────────────────────────────────────────────────

def percentile_t(v, pct: float) -> float:
    """`np.percentile(v, pct)` (method 'linear') of the 1-D tensor ``v``: numpy's virtual index
    `(n − 1)·q`, its neighbours, its `_lerp` (b − diff·(1 − t) when t ≥ 0.5)."""
    torch = _torch()
    n = int(v.shape[0])
    q = float(pct) / 100.0
    virtual = (n - 1) * q
    if virtual >= n - 1:
        prev = nxt = n - 1
    elif virtual < 0:
        prev = nxt = 0
    else:
        prev = int(math.floor(virtual))
        nxt = prev + 1
    t = virtual - prev
    s = torch.sort(v).values
    a, b = s[[prev, nxt]].cpu().numpy()
    diff = b - a
    r = a + diff * t
    if t >= 0.5:
        r = b - diff * (1 - t)
    return float(r)


# ── the candidates of a pool of points ─────────────────────────────────────────────────────────────────

def chunk_candidates(P: np.ndarray, ks: np.ndarray, fl, chunk: int, rep_m: float, fac: float, conf: float,
                     *, device: str, judge_scope: str = "any", refine_over_all: bool = False,
                     bootstrap_cap: Optional[int] = None, frame: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     points_frame_back: bool = False, draws: str = "numpy") -> List[dict]:
    """`correction.floor._chunk_candidates` on ``device``: the pool ``P`` (numpy, the session's frame)
    uploaded once — ``frame`` = (R, c0) brings it into the support's frame on the device (P' = (P − c0)·Rᵀ)
    — and searched level by level with the kernels above; the records as the reference's (normal,
    centroid, points in the search frame — back in the session's frame when ``points_frame_back`` —, the
    bootstrap σ, the judge). ``fl``: the floor configuration; the bounds / tolerances are the reference's
    module constants."""
    from correction import floor as FL
    torch = _torch()
    dev = device
    P_t = _t(np.asarray(P, np.float64), dev)
    ks_t = _t(np.asarray(ks), dev, dtype=torch.int64)
    if frame is not None:
        Rf, c0 = frame
        P_t = (P_t - _t(c0, dev)) @ _t(Rf, dev).T
    out: List[dict] = []
    n = int(P_t.shape[0])
    rest = torch.ones(n, dtype=torch.bool, device=dev)
    planes: List[Tuple[np.ndarray, np.ndarray]] = []
    band_m = float(fl.band_m)
    refit_m = float(fl.ransac_refit_band_m)
    for rank in range(FL.MAX_LEVELS):
        Q = P_t[rest]
        if int(Q.shape[0]) < 3:
            break
        y0 = percentile_t(Q[:, 1], float(fl.low_band_pct))
        band = Q[Q[:, 1] < y0 + band_m]
        if int(band.shape[0]) < 3:
            break
        nrm, cen, inl, fit = irls_plane_t(band, band_m, refit_m, FL.IRLS_MAX_ITER, FL.IRLS_TOL)
        if refine_over_all and int(inl.sum()) >= 3:
            near_all = ((Q - _t(cen, dev)) @ _t(nrm, dev)).abs() < band_m
            if int(near_all.sum()) >= 3:
                band = Q[near_all]
                nrm, cen, inl, fit2 = irls_plane_t(band, band_m, refit_m, FL.IRLS_MAX_ITER, FL.IRLS_TOL)
                fit = {"stages": fit["stages"] + fit2["stages"], "iterations": fit["iterations"] + fit2["iterations"],
                       "converged": bool(fit["converged"] and fit2["converged"]), "refined_over_all": True}
        n_in = int(inl.sum())
        if n_in < 3:
            break
        tilt = float(np.degrees(np.arccos(np.clip(float(nrm @ _UP), -1.0, 1.0))))
        pts_in = band[inl]
        boot = pts_in
        if bootstrap_cap is not None and int(boot.shape[0]) > int(bootstrap_cap):
            sel = np.random.default_rng(FL._chunk_seed(chunk, rank)).choice(int(boot.shape[0]), int(bootstrap_cap),
                                                                             replace=False)
            boot = boot[_t(np.sort(sel), dev, dtype=torch.int64)]
        s_tilt, s_h = bootstrap_plane_t(boot, FL._chunk_seed(chunk, rank), FL.N_BOOT_FLOOR, FL._POISSON1_CDF,
                                        FL.IRLS_TOL, draws=draws)
        planes.append((nrm, cen))
        pts_np = pts_in
        if frame is not None and points_frame_back:
            pts_np = pts_in @ _t(frame[0], dev) + _t(frame[1], dev)
        cand = {"rank": rank, "normal": nrm, "centroid": cen, "support": n_in,
                "tilt_deg": tilt, "floor_y_m": float(cen[1]), "sigma_tilt_deg": s_tilt,
                "sigma_height_m": s_h, "fit": fit, "real": True, "judge": None,
                "points": pts_np.cpu().numpy()}
        if rank >= 1:
            if judge_scope == "own":
                near = ((P_t - _t(cen, dev)) @ _t(nrm, dev)).abs() < band_m
            else:
                near = torch.stack([((P_t - _t(c_, dev)) @ _t(n_, dev)).abs() for n_, c_ in planes], 1).min(1).values < band_m
            FL._vendor_path()
            from loop_utils.metric_lock import decide_change
            dc = level_judge_t(P_t[near], ks_t[near], planes, rep_m, fac, conf, decide_change)
            cand["real"] = bool(dc["improves"])
            cand["judge"] = dc
        out.append(cand)
        if not cand["real"]:
            break
        d_all = (P_t - _t(cen, dev)) @ _t(nrm, dev)
        rest &= ~(d_all.abs() < refit_m)
    return out


def support_plane(P: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """`_weighted_plane(P, ones)` on the device: (normal with y ≥ 0, centroid) of the PCA plane of ``P``."""
    torch = _torch()
    P_t = _t(np.asarray(P, np.float64), device)
    return weighted_plane_t(P_t, torch.ones(int(P_t.shape[0]), dtype=P_t.dtype, device=device))


def rotate_back(P: np.ndarray, Rf: np.ndarray, c0: np.ndarray, device: str) -> np.ndarray:
    """``P @ Rf + c0`` on the device (the support frame's points back in the session's frame)."""
    return (_t(np.asarray(P, np.float64), device) @ _t(Rf, device) + _t(c0, device)).cpu().numpy()


def nearest_xz(A: np.ndarray, B: np.ndarray, device: str) -> np.ndarray:
    """`cKDTree(B[:, [0, 2]]).query(A[:, [0, 2]], k=1)[0]` on the device, for points given with their three
    coordinates (the x and z columns are read)."""
    A_t = _t(np.asarray(A, np.float64)[:, [0, 2]], device)
    B_t = _t(np.asarray(B, np.float64)[:, [0, 2]], device)
    return nearest_xz_t(A_t, B_t).cpu().numpy()
