"""
Colored ICP in pure PyTorch on CUDA  (Park et al. 2017, joint photometric+geometric).
================================================================================
Open3D tensor ICP silently runs on CPU here and pytorch3d won't compile fast.
This reproduces Open3D's COLORED ICP (the original registration used by
dense_pose_fusion) on the GPU: a point-to-plane GEOMETRIC term + a PHOTOMETRIC
term that uses the target's per-point colour gradient — so it does NOT slide on
flat low-texture walls (the reason colored ICP is used indoors). Every heavy op
(KNN, colour-gradient solve, normal equations) runs on CUDA tensors — verifiable
with nvidia-smi. Only dependency is torch.

Intensity (grayscale) is used for the photometric term, as in Open3D.
"""
from __future__ import annotations
import numpy as np
import torch

_SIGMA = 0.968   # photometric weight (Open3D's lambda_geometric = 0.968 → colour weight)


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    """RGB(0..1) → intensity (N,)."""
    w = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
    return rgb @ w


def _voxel_down(xyz, voxel, *extra):
    """Keep one point per voxel cell (first by index). Returns xyz + same slicing of extra."""
    keys = torch.floor(xyz / voxel).to(torch.int64)
    h = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    uniq, inv = torch.unique(h, return_inverse=True)
    first = torch.full((uniq.shape[0],), xyz.shape[0], dtype=torch.int64, device=xyz.device)
    first.scatter_reduce_(0, inv, torch.arange(xyz.shape[0], device=xyz.device),
                          reduce="amin", include_self=True)
    first = first.clamp_max(xyz.shape[0] - 1)
    return (xyz[first], *(e[first] for e in extra))


def _knn(pts: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """k nearest indices for each point (incl. self), chunked on GPU. (N,k)."""
    out = torch.empty(pts.shape[0], k, dtype=torch.int64, device=pts.device)
    for i in range(0, pts.shape[0], chunk):
        d = torch.cdist(pts[i:i + chunk], pts)
        out[i:i + chunk] = torch.topk(d, k, dim=1, largest=False).indices
    return out


def _color_gradients(xyz, normals, inten, k: int = 12):
    """Per-point colour gradient d (in the tangent plane: d·n=0) s.t. for neighbours q,
    I(q) ≈ I(p) + d·(q-p). Batched least squares on GPU. Returns (N,3)."""
    nbr = _knn(xyz, k)                                  # (N,k)
    q = xyz[nbr]                                        # (N,k,3)
    a = q - xyz[:, None, :]                             # (N,k,3)
    n = normals[:, None, :]
    a = a - (a * n).sum(-1, keepdim=True) * n           # project to tangent plane
    b = inten[nbr] - inten[:, None]                     # (N,k)
    M = a.transpose(1, 2) @ a + 1e-6 * torch.eye(3, device=xyz.device)   # (N,3,3)
    v = (a * b[..., None]).sum(1)                       # (N,3)
    d = torch.linalg.solve(M, v)                        # (N,3)
    return d - (d * normals).sum(-1, keepdim=True) * normals            # keep in tangent plane


def _nn_idx(src, tgt, chunk: int = 2048):
    out = torch.empty(src.shape[0], dtype=torch.int64, device=src.device)
    for i in range(0, src.shape[0], chunk):
        out[i:i + chunk] = torch.argmin(torch.cdist(src[i:i + chunk], tgt), dim=1)
    return out


def multiscale_icp_cuda(src_xyz, src_rgb, tgt_xyz, tgt_normals, tgt_rgb,
                        init_4x4, voxels=(0.04, 0.02, 0.01), iters=(25, 18, 12),
                        max_corr_mult: float = 1.5, max_pts: int = 6000,
                        device: str = "cuda"):
    """Multiscale COLORED ICP on CUDA. src in camera frame, tgt in world,
    init = src->world (keyframe c2w). Returns (4x4 c2w float64, fitness)."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ten = lambda a, dt=np.float32: torch.as_tensor(np.ascontiguousarray(a, dt), device=dev)
    S0, Sc0 = ten(src_xyz), _gray(ten(src_rgb))
    T0, N0, Tc0 = ten(tgt_xyz), ten(tgt_normals), _gray(ten(tgt_rgb))
    Tf = ten(init_4x4)
    eye3 = torch.eye(3, device=dev)
    wG, wC = _SIGMA ** 0.5, (1 - _SIGMA) ** 0.5   # _SIGMA = lambda_geometric → GEOM dominates
    gen = torch.Generator(device=dev).manual_seed(0)
    T_full = T0                                          # full target kept for final fitness
    fitness = 0.0
    for v, nit in zip(voxels, iters):
        s, sc = _voxel_down(S0, v, Sc0)
        t, tn, tc = _voxel_down(T0, v, N0, Tc0)
        if s.shape[0] < 30 or t.shape[0] < 30:
            continue
        if s.shape[0] > max_pts:
            s, sc = (lambda i: (s[i], sc[i]))(torch.randperm(s.shape[0], generator=gen, device=dev)[:max_pts])
        if t.shape[0] > max_pts:
            i = torch.randperm(t.shape[0], generator=gen, device=dev)[:max_pts]
            t, tn, tc = t[i], tn[i], tc[i]
        dC = _color_gradients(t, tn, tc)                # target colour gradient (N,3)
        thr = v * max_corr_mult
        for _ in range(nit):
            R, tr = Tf[:3, :3], Tf[:3, 3]
            ps = s @ R.T + tr
            j = _nn_idx(ps, t)
            tj, nj, dcj, tcj = t[j], tn[j], dC[j], tc[j]
            dvec = ps - tj
            m = torch.linalg.norm(dvec, dim=1) < thr
            if int(m.sum()) < 10:
                break
            psi, tji, nji, dcji = ps[m], tj[m], nj[m], dcj[m]
            rG = (nji * (psi - tji)).sum(1)                       # geometric residual
            rC = tcj[m] + (dcji * (psi - tji)).sum(1) - sc[m]     # photometric residual
            AG = torch.cat([torch.linalg.cross(psi, nji), nji], dim=1)
            AC = torch.cat([torch.linalg.cross(psi, dcji), dcji], dim=1)
            A = torch.cat([wG * AG, wC * AC], dim=0)
            b = torch.cat([wG * (-rG), wC * (-rC)], dim=0)
            x = torch.linalg.solve(A.T @ A + 1e-6 * torch.eye(6, device=dev), A.T @ b)
            w, dt = x[:3], x[3:]
            th = torch.linalg.norm(w)
            if th > 1e-9:
                kk = w / th
                K = torch.zeros(3, 3, device=dev)
                K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -kk[2], kk[1], kk[2], -kk[0], -kk[1], kk[0]
                dR = eye3 + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
            else:
                dR = eye3
            Tn_ = Tf.clone()
            Tn_[:3, :3] = dR @ R
            Tn_[:3, 3] = dR @ tr + dt
            Tf = Tn_
    # final fitness vs the FULL target (accurate overlap for the quality gate)
    R, tr = Tf[:3, :3], Tf[:3, 3]
    sps, _ = _voxel_down(S0, voxels[-1], Sc0)
    if sps.shape[0] > max_pts:
        sps = sps[torch.randperm(sps.shape[0], generator=gen, device=dev)[:max_pts]]
    ps = sps @ R.T + tr
    jf = _nn_idx(ps, T_full)
    fitness = float((torch.linalg.norm(ps - T_full[jf], dim=1) < voxels[-1] * max_corr_mult).float().mean())
    return Tf.double().cpu().numpy(), fitness
