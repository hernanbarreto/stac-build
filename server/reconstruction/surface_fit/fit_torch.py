"""The plane RANSAC and the support mesh of surface_fit on torch, one declared device (USER
2026-10-15: "Portar el RANSAC y los residuos a torch CUDA, sí, hacelo").

MEASURED on pccr's floor (4.58 M points) before this: the whole plane fit took 3.3 s — the
support mesh 2.0 s (``np.unique`` over the quad corners, the reductions, a cKDTree over every
point for the border snap), the RANSAC 0.6 s on a 400 k subsample, the residual statistics
0.1 s. Here:

- :func:`fit_plane_ransac_t`: the hypotheses (seeded numpy triplets, the numpy recipe's own
  sequence) are scored in BATCHES on every point (no subsample) — one matmul per chunk, the
  first maximal count wins (``torch.argmax``), the PCA refinement on the inliers' 3×3 scatter
  (eigh on the CPU), the same gates and rms. Deterministic: seeded draws, matmuls under the
  pinned cuBLAS workspace, no scatter.
- :func:`trimmed_quad_mesh_t`: the occupancy is ``isin`` over the unique cell ids (no index
  write), the disc dilation a convolution, the kept quads gathers, the node remap
  ``searchsorted`` over the unique corners; the border snap finds each vertex's exact nearest
  point on the device (fine-cell test for the surely-interior vertices, chunked ``cdist`` over
  the points of the cells around the others) — the same vertices as the numpy recipe, without
  the cKDTree over every point (2 s on pccr's floor).

The device: ``surface_fit.fit_device`` (cuda | cpu; cuda with no card FAILS) or
:func:`use_device` (the synthetic tests → cpu, tests/conftest.py). The meshing workers run
under torch's deterministic mode STRICT."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from reconstruction.geometry.primitives import PlaneFit

_DEVICE: Optional[str] = None
_DEVICES = ("cuda", "cpu")


def use_device(dev: Optional[str]) -> Optional[str]:
    global _DEVICE
    _DEVICE = None if dev is None else str(dev)
    return _DEVICE


def fit_device() -> str:
    if _DEVICE is not None:
        dev = _DEVICE
    else:
        from config import cfg
        sf = (cfg.get("surface_fit") or {}) if hasattr(cfg, "get") else {}
        if "fit_device" not in sf:
            raise RuntimeError("surface_fit.fit_device is missing (cuda | cpu) — the plane fit and the "
                               "support mesh run on one declared device, nothing is assumed")
        dev = str(sf["fit_device"])
    if dev not in _DEVICES:
        raise RuntimeError(f"surface_fit.fit_device = {dev!r} — one of {list(_DEVICES)}")
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("surface_fit.fit_device is cuda and no CUDA device is visible — the fit "
                           "fails instead of running somewhere else")
    return dev


def _t(a, dev: str, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), device=dev).to(dtype)


# ── the plane RANSAC ───────────────────────────────────────────────────────

def fit_plane_ransac_t(points: np.ndarray, dist_thresh: float, iters: int,
                       min_inlier_frac: float, seed: int = 0, chunk: int = 16
                       ) -> Optional[PlaneFit]:
    """Robust plane: ``iters`` 3-point hypotheses drawn by numpy's seeded generator (the numpy
    recipe's own sequence), scored on EVERY point in chunks of ``chunk`` hypotheses, the first
    maximal inlier count wins; PCA refinement on the inliers; None under ``min_inlier_frac``."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return None
    dev = fit_device()
    P = _t(pts, dev)
    rng = np.random.default_rng(int(seed))
    S = np.stack([rng.choice(n, 3, replace=False) for _ in range(int(iters))]).astype(np.int64)
    tri = P[torch.as_tensor(S, device=dev)]                       # (iters, 3, 3)
    p0 = tri[:, 0]
    nrm = torch.cross(tri[:, 1] - p0, tri[:, 2] - p0, dim=1)
    ln = torch.linalg.norm(nrm, dim=1)
    ok = ln > 1e-12
    nrm = nrm / torch.where(ok, ln, torch.ones_like(ln))[:, None]
    dd = -(nrm * p0).sum(dim=1)
    counts = torch.zeros(int(iters), dtype=torch.int64, device=dev)
    thr = float(dist_thresh)
    for a in range(0, int(iters), int(chunk)):
        dist = torch.abs(P @ nrm[a:a + chunk].T + dd[a:a + chunk])      # (N, c)
        counts[a:a + chunk] = (dist < thr).sum(dim=0)
    counts = torch.where(ok, counts, torch.zeros_like(counts))
    best = int(torch.argmax(counts))                              # the FIRST maximal count
    if int(counts[best]) == 0:
        return None
    normal = nrm[best]
    d = dd[best]
    inl = torch.abs(P @ normal + d) < thr
    if int(inl.sum()) >= 3:
        q = P[inl]
        ctr = q.mean(dim=0)
        Q = q - ctr
        C = (Q.T @ Q).cpu().numpy()
        _, V = np.linalg.eigh(C)
        n_np = V[:, 0] / max(float(np.linalg.norm(V[:, 0])), 1e-12)
        if float(n_np @ normal.cpu().numpy()) < 0:
            n_np = -n_np
        normal = _t(n_np, dev)
        d = -(normal @ ctr)
        inl = torch.abs(P @ normal + d) < thr
    n_in = int(inl.sum())
    if n_in < 3 or n_in / float(n) < float(min_inlier_frac):
        return None
    dist_in = torch.abs(P[inl] @ normal + d)
    rms = float(torch.sqrt(torch.mean(dist_in * dist_in)))
    return PlaneFit(normal=normal.cpu().numpy(), d=float(d), inliers=inl.cpu().numpy(),
                    rms=rms, n_points=n, curvature=0.0)


# ── the support mesh ───────────────────────────────────────────────────────

def _disc(r: int, dev: str) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(-r, r + 1, device=dev), torch.arange(-r, r + 1, device=dev),
                            indexing="ij")
    return ((xx * xx + yy * yy) <= r * r).to(torch.float64)


def _dilate(occ: torch.Tensor, r: int) -> torch.Tensor:
    """scipy binary_dilation with the disc of radius ``r`` (border value 0)."""
    if r <= 0:
        return occ
    k = _disc(r, occ.device)
    out = torch.nn.functional.conv2d(occ.to(torch.float64)[None, None], k[None, None], padding=r)
    return out[0, 0] > 0.5


def trimmed_quad_mesh_t(uv: np.ndarray, resolution: float, support_radius: float):
    """:func:`reconstruction.surface_fit.support.trimmed_quad_mesh_np` on tensors — the same
    vertices and faces."""
    from .support import TrimmedMesh
    dev = fit_device()
    uv_np = np.asarray(uv, dtype=np.float64)
    U = _t(uv_np, dev)
    occ_cell = max(min(float(resolution), float(support_radius) / 2.0), 1e-4)
    pad = max(float(support_radius), occ_cell)
    umin = float(U[:, 0].min())
    vmin = float(U[:, 1].min())
    umax = float(U[:, 0].max())
    vmax = float(U[:, 1].max())
    u0 = umin - pad
    v0 = vmin - pad
    ii = torch.floor((U[:, 0] - u0) / occ_cell).to(torch.int64)
    jj = torch.floor((U[:, 1] - v0) / occ_cell).to(torch.int64)
    w = int(ii.max()) + 1 + int(np.ceil(pad / occ_cell))
    h = int(jj.max()) + 1 + int(np.ceil(pad / occ_cell))
    lin = jj * w + ii
    lin_u = torch.unique(lin)
    occ0 = torch.isin(torch.arange(h * w, device=dev), lin_u).reshape(h, w)
    r = int(round(float(support_radius) / occ_cell))
    occ = _dilate(occ0, r)

    res = float(resolution)
    nu = max(int(np.ceil((umax - umin) / res)), 1)
    nv = max(int(np.ceil((vmax - vmin) / res)), 1)
    uc = umin + (torch.arange(nu, device=dev, dtype=torch.float64) + 0.5) * res
    vc = vmin + (torch.arange(nv, device=dev, dtype=torch.float64) + 0.5) * res
    ci = torch.clamp(torch.floor((uc - u0) / occ_cell).to(torch.int64), 0, w - 1)
    cj = torch.clamp(torch.floor((vc - v0) / occ_cell).to(torch.int64), 0, h - 1)
    keep = occ[cj][:, ci]                                         # (nv, nu)
    n_keep = int(keep.sum())
    if n_keep == 0:
        return TrimmedMesh(np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64), 0.0, 0.0)
    kj, ki = keep.nonzero(as_tuple=True)                          # row-major, as np.nonzero
    W1 = nu + 1
    corners = torch.stack([kj * W1 + ki, kj * W1 + ki + 1,
                           (kj + 1) * W1 + ki, (kj + 1) * W1 + ki + 1], dim=1)
    used = torch.unique(corners)
    c = torch.searchsorted(used, corners)
    faces = torch.cat([c[:, [0, 1, 3]], c[:, [0, 3, 2]]], dim=0)
    gj = torch.div(used, W1, rounding_mode="floor")
    gi = used - gj * W1
    verts_uv = torch.stack([umin + gi.to(torch.float64) * res, vmin + gj.to(torch.float64) * res], dim=1)

    # border snap (support.py's rule): every vertex farther than ~half a cell from ANY measured
    # point moves onto its nearest point. On the device, exact and bounded: a vertex whose own
    # FINE cell (half the tolerance) holds a point is within the tolerance for sure; the others
    # get their exact nearest point among the points of the cells around them (cdist in
    # chunks, the first minimum); the overhanging ones get their exact nearest point within
    # support_radius + cell (every such vertex has one). The cKDTree over every point (2 s on
    # pccr's floor) is gone; the vertices are the numpy recipe's.
    tol = max(0.6 * res, 1e-6)
    fine = tol / 2.0
    fw = int(torch.floor((U[:, 0] - u0) / fine).max()) + 2
    fh = int(torch.floor((U[:, 1] - v0) / fine).max()) + 2
    pfi = torch.clamp(torch.floor((U[:, 0] - u0) / fine).to(torch.int64), 0, fw - 1)
    pfj = torch.clamp(torch.floor((U[:, 1] - v0) / fine).to(torch.int64), 0, fh - 1)
    plin = pfj * fw + pfi
    occ_f = torch.isin(torch.arange(fh * fw, device=dev), torch.unique(plin)).reshape(fh, fw)
    vfi = torch.clamp(torch.floor((verts_uv[:, 0] - u0) / fine).to(torch.int64), 0, fw - 1)
    vfj = torch.clamp(torch.floor((verts_uv[:, 1] - v0) / fine).to(torch.int64), 0, fh - 1)
    unsure = (~occ_f[vfj, vfi]).nonzero()[:, 0]

    # the points sorted by fine cell (a CSR over the grid: cell → [start, start + count))
    order = torch.sort(plin, stable=True).indices
    plin_s = plin[order]
    cells_u, counts = torch.unique_consecutive(plin_s, return_counts=True)
    starts = torch.searchsorted(plin_s, cells_u)
    K = int(counts.max())
    n_cells = int(len(cells_u))

    def _nearest(vsel: torch.Tensor, radius_cells: int):
        """(distance, global point index) of the EXACT nearest point of each selected vertex
        among the points of the fine cells within ``radius_cells`` of the vertex's cell — every
        point of every such cell is tested (the cells' counts are read, the longest decides the
        gather width), chunked over the vertices; +inf and index −1 with no point there."""
        nv_ = int(len(vsel))
        if nv_ == 0:
            return (torch.zeros(0, dtype=torch.float64, device=dev),
                    torch.zeros(0, dtype=torch.int64, device=dev))
        r_ = int(radius_cells)
        off = torch.arange(-r_, r_ + 1, device=dev)
        dj, di = torch.meshgrid(off, off, indexing="ij")
        dj = dj.reshape(-1)
        di = di.reshape(-1)                                     # (C,)
        C = int(len(dj))
        kk = torch.arange(K, device=dev)
        d_all = torch.empty(nv_, dtype=torch.float64, device=dev)
        i_all = torch.empty(nv_, dtype=torch.int64, device=dev)
        step = max(1, int(2e7 // max(C * K, 1)))
        for a in range(0, nv_, step):
            vs = vsel[a:a + step]
            cj = vfj[vs][:, None] + dj[None, :]
            ci = vfi[vs][:, None] + di[None, :]
            inside = (cj >= 0) & (cj < fh) & (ci >= 0) & (ci < fw)
            cid = torch.clamp(cj, 0, fh - 1) * fw + torch.clamp(ci, 0, fw - 1)
            pos = torch.clamp(torch.searchsorted(cells_u, cid), max=n_cells - 1)
            hit = inside & (cells_u[pos] == cid)
            st = starts[pos]
            ct = torch.where(hit, counts[pos], torch.zeros_like(counts[pos]))
            idx = st[:, :, None] + kk[None, None, :]              # (v, C, K)
            valid = kk[None, None, :] < ct[:, :, None]
            pidx = order[torch.clamp(idx, max=len(order) - 1)]
            diff = U[pidx] - verts_uv[vs][:, None, None, :]
            dist = torch.sqrt((diff * diff).sum(dim=-1))
            dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
            dm, im = torch.min(dist.reshape(len(vs), -1), dim=1)    # the FIRST minimum
            d_all[a:a + step] = dm
            i_all[a:a + step] = torch.where(torch.isfinite(dm), pidx.reshape(len(vs), -1).gather(1, im[:, None])[:, 0],
                                            torch.full_like(im, -1))
        return d_all, i_all

    d_un, _ = _nearest(unsure, int(np.ceil(tol / fine)) + 1)
    over = unsure[d_un > tol]
    verts_np = verts_uv.cpu().numpy()
    faces_np = faces.cpu().numpy().astype(np.int64)
    if len(over):
        _, i_ov = _nearest(over, int(np.ceil((float(support_radius) + res) / fine)) + 1)
        found = i_ov >= 0                                       # a vertex with no point in reach stays
        verts_np = verts_np.copy()
        verts_np[over[found].cpu().numpy()] = U[i_ov[found]].cpu().numpy()
        tri = verts_np[faces_np]
        area2 = np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                       - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        faces_np = faces_np[area2 > res * res * 1e-4]
    return TrimmedMesh(vertices_uv=verts_np, faces=faces_np,
                       support_fraction=float(n_keep) / float(nu * nv),
                       area_m2=float(n_keep) * res * res)
