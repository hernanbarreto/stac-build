"""The edge metric's two hot loops on the card (USER 2026-11-01: "todo debe correr en GPU").

MEASURED on pccr (epoch 4; analysis/2026-11-01_edge_gpu_profile/pccr_edge_profile.json, three objects,
418 s): 295 s of them in `precision.edge_metric.profile` — `fillet_radius` evaluates the filleted L's
distance field once per radius of its grid (89,164 evaluations over sections of tens of thousands of
points, 3.3 ms each on numpy) — and 60 s in `_bootstrap_normals` (2000 resamples per face: a draw of n
integers, a bincount, one matvec). In the certification of audit5 the edge metric took 19 min 42 s.

Here, the same recipes on torch tensors in float64 on ONE device, the numpy functions kept as the reference:

* ``fillet_radius_t`` — the distance field of EVERY radius of the grid at once (a (radii × points) matrix,
  in blocks of radii that bound the memory), `profile`'s formulas operation for operation; the argmin
  with numpy's tie rule (the first minimum); the paired test of each smaller radius against the minimum
  from the same matrix (zero differences dropped, mean against z standard errors, ddof 1) — the same r̂;
* ``bootstrap_normals_t`` — the SAME resamples: the indices come from the reference's seeded numpy
  generator (`Generator.integers`, one stream read in blocks — the same integers in the same order as
  its draw-by-draw calls), the per-resample counts by a deterministic accumulate on the device, the
  nine moments by one matmul, the normals by numpy's batched `eigh` (the reference's routine).

Differences from the reference are the rounding of float64 sums in another order (~1e-15 relative);
tests/test_edge_torch.py holds the two paths equal: the same r̂ on the synthetic sections, normals
equal to 1e-9, the same report on the synthetic box.

Selected by ``reconstruction.precision.cloud_metrics.edge_device``: numpy (the path of record) | cpu (the
torch path on the CPU — the tests) | cuda; cuda with no visible card FAILS naming the key (the F6 rule).
"""
from __future__ import annotations



import numpy as np

DEVICES = ("numpy", "cpu", "cuda")


def _torch():
    import torch
    return torch


def resolve_device(name: str) -> str:
    name = str(name)
    if name not in DEVICES:
        raise ValueError(f"reconstruction.precision.cloud_metrics.edge_device must be one of {list(DEVICES)}, "
                         f"got {name!r}")
    if name == "cuda" and not _torch().cuda.is_available():
        raise RuntimeError("reconstruction.precision.cloud_metrics.edge_device is 'cuda' and torch sees no card "
                           "— the edge metric does not fall back to the CPU by itself; declare cpu or numpy")
    return name


def _t(a, device, dtype=None):
    torch = _torch()
    if dtype is None:
        dtype = torch.float64
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(dtype)


def profile_dist_t(q, r, phi: float, delta: float):
    """`edge_metric.profile(q, r_k, phi, delta).dist` for every radius ``r`` (R,) of the tensor ``q`` (N, 2)
    at once: the unsigned distance to the L of interior angle ``phi`` rounded by a fillet of radius r_k,
    as a (R, N) tensor — the reference's formulas, the radius broadcast over its own row."""
    torch = _torch()
    cp, sp = float(np.cos(phi)), float(np.sin(phi))
    tan_half = float(np.tan(0.5 * phi))
    r = r.reshape(-1, 1)                                             # (R, 1)
    T = torch.where(r > 0, r / tan_half, torch.zeros_like(r))         # the tangent point per radius
    xa, ya = q[:, 0][None, :], q[:, 1][None, :]                       # (1, N)
    xb = q[:, 0] * cp + q[:, 1] * sp
    yb = q[:, 0] * sp - q[:, 1] * cp
    ta = torch.maximum(xa, T)                                         # (R, N)
    tb = torch.maximum(xb[None, :], T)
    da = torch.hypot(xa - ta, ya.expand_as(ta))
    db = torch.hypot(xb[None, :] - tb, yb[None, :].expand_as(tb))
    dist = torch.minimum(da, db)
    # the fillet: a point in the sector spanned by −w_A and −w_B from the centre (T, r) reads the arc
    vx = xa - T
    vy = ya - r
    bb = -vx / sp
    aa = bb * cp - vy
    in_sec = (aa >= 0.0) & (bb >= 0.0) & (r > 0)
    rho = torch.hypot(vx, vy)
    darc = torch.where(in_sec, (rho - r).abs(), torch.full_like(rho, float("inf")))
    return torch.where(darc < dist, darc, dist)


def fillet_radius_t(Q: np.ndarray, phi: float, delta: float, r_max: float, z: float, device: str,
                    block_entries: int = 40_000_000) -> float:
    """`edge_metric.fillet_radius` on ``device``: r̂ = the argmin over the grid {0, Δ, 2Δ, …, r_max} of the
    mean distance of the section ``Q`` (N, 2) to the filleted L, ties at the sample's own noise resolved
    to the smaller radius (the smallest k < k_min whose paired excess d_k − d_min over the points the two
    profiles tell apart has a mean within ``z`` standard errors of zero; fewer than two such points =
    that radius). The distance matrix is built in blocks of radii of at most ``block_entries`` entries."""
    torch = _torch()
    grid = delta * np.arange(int(np.floor(r_max / delta)) + 1)
    q = _t(np.asarray(Q, np.float64).reshape(-1, 2), device)
    n = int(q.shape[0])
    R = len(grid)
    g = _t(grid, device)
    step = max(1, int(block_entries) // max(n, 1))
    cost = torch.empty(R, dtype=q.dtype, device=device)
    for a in range(0, R, step):
        cost[a:a + step] = profile_dist_t(q, g[a:a + step], phi, delta).mean(1)
    k_min = int(torch.argmin(cost))                                   # the first minimum, as numpy's
    if k_min == 0:
        return float(grid[0])
    d_min = profile_dist_t(q, g[k_min:k_min + 1], phi, delta)[0]      # (N,)
    for a in range(0, k_min, step):
        b = min(a + step, k_min)
        diff = profile_dist_t(q, g[a:b], phi, delta) - d_min[None, :]            # (b − a, N)
        nz = diff != 0.0
        cnt = nz.sum(1)
        s1 = torch.where(nz, diff, torch.zeros_like(diff)).sum(1)
        mean = s1 / cnt.clamp(min=1).to(diff.dtype)
        dev = torch.where(nz, diff - mean[:, None], torch.zeros_like(diff))
        var = (dev * dev).sum(1) / (cnt - 1).clamp(min=1).to(diff.dtype)         # ddof 1
        se = torch.sqrt(var) / torch.sqrt(cnt.to(diff.dtype))
        few = cnt < 2
        ok = few | (mean <= z * se)
        hit = torch.nonzero(ok)
        if len(hit):
            return float(grid[a + int(hit[0])])
    return float(grid[k_min])


def bootstrap_normals_t(P: np.ndarray, n_boot: int, seed: int, device: str, block: int = 64) -> np.ndarray:
    """`edge_metric._bootstrap_normals` on ``device``: TLS normals of ``n_boot`` point-resamples of ``P``
    (n, 3). The resamples are the reference's — `default_rng(seed).integers(0, n, n)` per resample, read
    ``block`` resamples at a time from the one stream — counted on the device (a deterministic
    accumulate), the nine moments by one matmul, the normals by numpy's batched `eigh`."""
    torch = _torch()
    P_t = _t(np.asarray(P, np.float64), device)
    n = int(P_t.shape[0])
    Q = P_t - P_t.mean(0)
    M = torch.stack([Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 0] * Q[:, 0], Q[:, 0] * Q[:, 1], Q[:, 0] * Q[:, 2],
                     Q[:, 1] * Q[:, 1], Q[:, 1] * Q[:, 2], Q[:, 2] * Q[:, 2]], 1)          # (n, 9)
    rng = np.random.default_rng(int(seed))
    mom = torch.empty((int(n_boot), 9), dtype=P_t.dtype, device=device)
    for a in range(0, int(n_boot), int(block)):
        b = min(int(block), int(n_boot) - a)
        idx = _t(rng.integers(0, n, (b, n)), device, dtype=torch.int64)
        flat = torch.zeros(b * n, dtype=P_t.dtype, device=device)
        rows = (torch.arange(b, device=device, dtype=torch.int64)[:, None] * n + idx).reshape(-1)
        flat.index_put_((rows,), torch.ones(1, dtype=P_t.dtype, device=device).expand(b * n), accumulate=True)
        mom[a:a + b] = flat.reshape(b, n) @ M
    mom = mom.cpu().numpy() / n
    mu = mom[:, :3]
    S = np.empty((int(n_boot), 3, 3))
    S[:, 0, 0], S[:, 0, 1], S[:, 0, 2] = mom[:, 3], mom[:, 4], mom[:, 5]
    S[:, 1, 1], S[:, 1, 2], S[:, 2, 2] = mom[:, 6], mom[:, 7], mom[:, 8]
    S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]
    S -= mu[:, :, None] * mu[:, None, :]
    return np.linalg.eigh(S)[1][:, :, 0]
