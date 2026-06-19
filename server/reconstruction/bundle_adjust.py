"""
Full bundle adjustment with landmarks (reprojection error) — Tier-1 pose refinement.
================================================================================
Gold-standard BA: jointly optimise camera poses (SE3) and 3D landmarks to minimise
robust reprojection error of the learned VGGSfM correspondences (tracks.npz).

Scales to tens of thousands of landmarks WITHOUT an external sparse solver by using
the SCHUR COMPLEMENT — the standard BA reduction:
  • the landmark Hessian block is block-diagonal (one 3×3 per point) → inverted trivially,
  • we form the reduced CAMERA system  S = U − W Vinv Wᵀ  (size 6N×6N, N = #poses),
    solve it for the pose updates, then back-substitute the point updates.
Everything is vectorised torch (GPU-ready); poses retract on SE3 via a closed-form
exp whose convention exactly matches the analytic Jacobians (no PyPose-convention risk).

An optional DEPTH PRIOR ties each observation's predicted camera-z to the backbone's
(MapAnything) metric depth → anchors monocular scale and fights per-segment scale drift.

This file is solver + self-test only; data loading / write-back lives in run_bundle_adjust.py.
Run the math self-test (no GPU/model needed):
    python -m reconstruction.bundle_adjust --selftest
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("BundleAdjust")


# ── SE3 helpers (convention: xi = [v(3), w(3)], update  T ← Exp(xi) @ T) ──────
def _skew(v):
    import torch
    O = torch.zeros(v.shape[:-1] + (3, 3), device=v.device, dtype=v.dtype)
    O[..., 0, 1] = -v[..., 2]; O[..., 0, 2] = v[..., 1]
    O[..., 1, 0] = v[..., 2];  O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]; O[..., 2, 1] = v[..., 0]
    return O


def _se3_exp(xi):
    """xi (...,6)=[v,w] → (...,4,4) SE3. Closed-form Rodrigues with the V matrix."""
    import torch
    v, w = xi[..., :3], xi[..., 3:]
    th = torch.linalg.norm(w, dim=-1, keepdim=True)                  # (...,1)
    K = _skew(w)
    th2 = (th * th).unsqueeze(-1)
    small = (th < 1e-8).unsqueeze(-1)
    a = torch.where(small, torch.ones_like(th2), (torch.sin(th) / th.clamp_min(1e-12)).unsqueeze(-1))
    b = torch.where(small, 0.5 * torch.ones_like(th2), ((1 - torch.cos(th)) / th.clamp_min(1e-12) ** 2).unsqueeze(-1))
    c = torch.where(small, (1.0 / 6.0) * torch.ones_like(th2),
                    ((th - torch.sin(th)) / th.clamp_min(1e-12) ** 3).unsqueeze(-1))
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(K.shape)
    R = eye + a * K + b * (K @ K)
    V = eye + b * K + c * (K @ K)
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    T = torch.zeros(xi.shape[:-1] + (4, 4), device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R; T[..., :3, 3] = t; T[..., 3, 3] = 1.0
    return T


def _huber_w(r2, delta):
    """IRLS weight sqrt for Huber loss given squared residual norm r2 (per obs)."""
    import torch
    r = torch.sqrt(r2.clamp_min(1e-12))
    return torch.where(r <= delta, torch.ones_like(r), torch.sqrt((delta / r).clamp_max(1.0)))


def bundle_adjust(
    poses_w2c: np.ndarray,      # (N,4,4) initial world→camera
    K: np.ndarray,              # (N,3,3) intrinsics at track resolution
    cam_idx: np.ndarray,        # (P,) camera index per observation
    pt_idx: np.ndarray,         # (P,) point index per observation
    uv: np.ndarray,             # (P,2) observed pixel (x=col, y=row)
    pts_w: np.ndarray,          # (M,3) initial landmark world coords
    obs_w: Optional[np.ndarray] = None,   # (P,) per-observation weight (e.g. track score)
    depth_obs: Optional[np.ndarray] = None,   # (P,) backbone metric depth at uv (scale anchor)
    depth_w: float = 0.1,       # weight of the depth prior term
    iters: int = 12,
    huber_px: float = 2.0,
    fix_first: bool = True,     # gauge: hold pose 0 fixed
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Returns (refined_poses_w2c (N,4,4), refined_pts (M,3), info)."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    f = torch.float64
    T = torch.tensor(poses_w2c, device=dev, dtype=f)               # (N,4,4)
    X = torch.tensor(pts_w, device=dev, dtype=f)                   # (M,3)
    Kt = torch.tensor(K, device=dev, dtype=f)
    ci = torch.tensor(cam_idx, device=dev, dtype=torch.long)
    pj = torch.tensor(pt_idx, device=dev, dtype=torch.long)
    obs = torch.tensor(uv, device=dev, dtype=f)
    N, M, P = T.shape[0], X.shape[0], obs.shape[0]
    w_obs = (torch.tensor(obs_w, device=dev, dtype=f) if obs_w is not None
             else torch.ones(P, device=dev, dtype=f)).clamp_min(1e-3)
    dep = torch.tensor(depth_obs, device=dev, dtype=f) if depth_obs is not None else None
    fx = Kt[ci, 0, 0]; fy = Kt[ci, 1, 1]; cx = Kt[ci, 0, 2]; cy = Kt[ci, 1, 2]

    free = torch.arange(N, device=dev)
    if fix_first:
        free = free[1:]                                            # pose 0 is the datum
    n_free = free.shape[0]
    pose_col = torch.full((N,), -1, device=dev, dtype=torch.long)
    pose_col[free] = torch.arange(n_free, device=dev)

    lm = 1e-3
    last_cost = None
    for it in range(iters):
        R = T[:, :3, :3]; t = T[:, :3, 3]
        Rci = R[ci]; tci = t[ci]
        Xc = torch.einsum("pij,pj->pi", Rci, X[pj]) + tci          # (P,3) camera-space
        z = Xc[:, 2].clamp_min(1e-6)
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        r = torch.stack([u - obs[:, 0], v - obs[:, 1]], dim=1)     # (P,2) reprojection residual
        r2 = (r * r).sum(1)
        hw = _huber_w(r2, huber_px)                                # (P,)
        sw = (w_obs.sqrt() * hw)                                   # combined sqrt weight
        # du/dXc (P,2,3)
        J_uXc = torch.zeros(P, 2, 3, device=dev, dtype=f)
        J_uXc[:, 0, 0] = fx / z; J_uXc[:, 0, 2] = -fx * Xc[:, 0] / (z * z)
        J_uXc[:, 1, 1] = fy / z; J_uXc[:, 1, 2] = -fy * Xc[:, 1] / (z * z)
        # dXc/dxi = [I | -skew(Xc)]  (camera-frame left perturbation)  (P,3,6)
        dXc = torch.zeros(P, 3, 6, device=dev, dtype=f)
        dXc[:, :, :3] = torch.eye(3, device=dev, dtype=f)
        dXc[:, :, 3:] = -_skew(Xc)
        Jc = torch.einsum("pij,pjk->pik", J_uXc, dXc)              # (P,2,6) wrt pose
        Jp = torch.einsum("pij,pjk->pik", J_uXc, Rci)             # (P,2,3) wrt point
        rw = r * sw[:, None]
        Jc = Jc * sw[:, None, None]; Jp = Jp * sw[:, None, None]

        # optional depth prior: residual z - depth_obs (1-D), Jacobian rows
        if dep is not None:
            rd = (z - dep) * np.sqrt(depth_w)
            # dz/dxi = dXc[:,2,:] ; dz/dX = Rci[2,:]
            Jdc = dXc[:, 2, :] * np.sqrt(depth_w)                  # (P,6)
            Jdp = Rci[:, 2, :] * np.sqrt(depth_w)                  # (P,3)
        else:
            rd = Jdc = Jdp = None

        # ── accumulate normal-equation blocks ──
        U = torch.zeros(N, 6, 6, device=dev, dtype=f)
        V = torch.zeros(M, 3, 3, device=dev, dtype=f)
        gc = torch.zeros(N, 6, device=dev, dtype=f)
        gp = torch.zeros(M, 3, device=dev, dtype=f)
        Wb = torch.einsum("pki,pkj->pij", Jc, Jp)                 # (P,6,3) per-obs cross block
        U.index_add_(0, ci, torch.einsum("pki,pkj->pij", Jc, Jc))
        V.index_add_(0, pj, torch.einsum("pki,pkj->pij", Jp, Jp))
        gc.index_add_(0, ci, -torch.einsum("pki,pk->pi", Jc, rw))
        gp.index_add_(0, pj, -torch.einsum("pki,pk->pi", Jp, rw))
        if dep is not None:
            U.index_add_(0, ci, torch.einsum("pi,pj->pij", Jdc, Jdc))
            V.index_add_(0, pj, torch.einsum("pi,pj->pij", Jdp, Jdp))
            gc.index_add_(0, ci, -Jdc * rd[:, None])
            gp.index_add_(0, pj, -Jdp * rd[:, None])
            Wb = Wb + torch.einsum("pi,pj->pij", Jdc, Jdp)

        # LM damping
        U = U + lm * torch.diag_embed(torch.diagonal(U, dim1=1, dim2=2))
        V = V + lm * torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2))
        Vinv = torch.linalg.inv(V + 1e-9 * torch.eye(3, device=dev, dtype=f))   # (M,3,3)

        # ── Schur complement: S (free poses), build densely (6 n_free)² ──
        S = torch.zeros(n_free * 6, n_free * 6, device=dev, dtype=f)
        b = torch.zeros(n_free * 6, device=dev, dtype=f)
        # diagonal U contribution (only free poses)
        for blk_i, cidx in enumerate(free.tolist()):
            S[blk_i * 6:blk_i * 6 + 6, blk_i * 6:blk_i * 6 + 6] += U[cidx]
            b[blk_i * 6:blk_i * 6 + 6] += gc[cidx]
        # W Vinv g_p  and  W Vinv Wᵀ : iterate observations grouped by point
        WV = torch.einsum("pij,pjk->pik", Wb, Vinv[pj])           # (P,6,3) = W_obs · Vinv_pt
        # b -= sum_obs WV · gp[pt]
        contrib_b = -torch.einsum("pik,pk->pi", WV, gp[pj])       # (P,6)
        # scatter contrib_b into free-pose rows
        cfree = pose_col[ci]
        mask = cfree >= 0
        if mask.any():
            rows = (cfree[mask, None] * 6 + torch.arange(6, device=dev)[None, :]).reshape(-1)
            b.index_add_(0, rows, contrib_b[mask].reshape(-1))
        # S -= sum over point j of  WV_a · Wb_bᵀ  for every obs pair (a,b) sharing point j
        # group observations by point; for moderate track lengths this is cheap.
        order = torch.argsort(pj)
        pj_s = pj[order]
        bounds = torch.searchsorted(pj_s, torch.arange(M + 1, device=dev))
        WV_o = WV[order]; Wb_o = Wb[order]; ci_o = ci[order]
        for j in range(M):
            a0, a1 = int(bounds[j]), int(bounds[j + 1])
            if a1 - a0 < 1:
                continue
            cs = ci_o[a0:a1]
            cols = pose_col[cs]
            sel = cols >= 0
            if not bool(sel.any()):
                continue
            wv = WV_o[a0:a1][sel]                                  # (k,6,3)
            wb = Wb_o[a0:a1][sel]                                  # (k,6,3)
            cc = cols[sel]
            block = torch.einsum("aik,bjk->abij", wv, wb)         # (k,k,6,6)
            kk = cc.shape[0]
            for a in range(kk):
                ra = cc[a] * 6
                for bb in range(kk):
                    cb = cc[bb] * 6
                    S[ra:ra + 6, cb:cb + 6] -= block[a, bb]

        # solve S dxc = b
        dxc = torch.linalg.solve(S + 1e-9 * torch.eye(n_free * 6, device=dev, dtype=f), b)
        dxc_full = torch.zeros(N, 6, device=dev, dtype=f)
        dxc_full[free] = dxc.reshape(n_free, 6)
        # back-substitute points: dX_j = Vinv (gp - sum_obs Wbᵀ dxc[ci])
        WT_dxc = torch.einsum("pij,pi->pj", Wb, dxc_full[ci])     # (P,3)
        rhs_p = gp.clone()
        rhs_p.index_add_(0, pj, -WT_dxc)
        dX = torch.einsum("mij,mj->mi", Vinv, rhs_p)              # (M,3)

        # retract
        T_new = _se3_exp(dxc_full) @ T
        X_new = X + dX
        # evaluate new cost
        Rn = T_new[:, :3, :3]; tn = T_new[:, :3, 3]
        Xc2 = torch.einsum("pij,pj->pi", Rn[ci], X_new[pj]) + tn[ci]
        z2 = Xc2[:, 2].clamp_min(1e-6)
        r2u = fx * Xc2[:, 0] / z2 + cx - obs[:, 0]
        r2v = fy * Xc2[:, 1] / z2 + cy - obs[:, 1]
        new_r2 = (r2u * r2u + r2v * r2v)
        cost = (w_obs * torch.minimum(new_r2, torch.full_like(new_r2, huber_px ** 2))).sum().item()
        if dep is not None:
            cost += depth_w * ((z2 - dep) ** 2).sum().item()

        if last_cost is None or cost < last_cost:
            T, X = T_new, X_new
            lm = max(lm * 0.5, 1e-7)
            improved = True
        else:
            lm = min(lm * 4.0, 1e3)
            improved = False
        rms = float(torch.sqrt((new_r2).mean()).item())
        if verbose:
            logger.info(f"  BA it {it+1}/{iters}  cost={cost:.3e}  reproj_rms={rms:.3f}px  "
                        f"lm={lm:.1e}  {'✓' if improved else '×rollback'}")
        if last_cost is not None and abs((last_cost - cost) / max(last_cost, 1e-9)) < 1e-5 and improved:
            last_cost = cost
            break
        if improved:
            last_cost = cost

    info = {"reproj_rms_px": rms, "cost": last_cost, "n_poses": N, "n_points": M, "n_obs": P}
    return T.detach().cpu().numpy(), X.detach().cpu().numpy(), info


# ── synthetic self-test: recover perturbed poses+points from noisy reprojections ──
def _selftest():
    import torch
    rng = np.random.RandomState(0)
    N, M = 12, 400
    # ground-truth poses along a path looking at a point cloud
    gt_T = []
    for i in range(N):
        ang = 0.15 * i
        Rwc = _se3_exp(torch.tensor([[0, 0, 0, 0, ang, 0]], dtype=torch.float64))[0, :3, :3].numpy()
        c = np.array([2.0 * np.sin(ang), 0.1 * i, -2.0 * np.cos(ang)])   # camera centre
        Rcw = Rwc.T
        t = -Rcw @ c
        T = np.eye(4); T[:3, :3] = Rcw; T[:3, 3] = t
        gt_T.append(T)
    gt_T = np.stack(gt_T)
    gt_X = rng.uniform([-3, -2, 3], [3, 2, 8], size=(M, 3))
    K = np.tile(np.array([[400, 0, 256], [0, 400, 144], [0, 0, 1]], float), (N, 1, 1))

    ci, pj, uv, dep = [], [], [], []
    for j in range(M):
        for i in range(N):
            Xc = gt_T[i, :3, :3] @ gt_X[j] + gt_T[i, :3, 3]
            if Xc[2] < 0.2:
                continue
            u = K[i, 0, 0] * Xc[0] / Xc[2] + K[i, 0, 2]
            v = K[i, 1, 1] * Xc[1] / Xc[2] + K[i, 1, 2]
            if 0 <= u < 512 and 0 <= v < 288 and rng.rand() < 0.6:
                ci.append(i); pj.append(j)
                uv.append([u + rng.randn() * 0.3, v + rng.randn() * 0.3])
                dep.append(Xc[2])
    ci = np.array(ci); pj = np.array(pj); uv = np.array(uv); dep = np.array(dep)

    # perturb poses (except #0) and points → the BA must recover them
    init_T = gt_T.copy()
    pert = _se3_exp(torch.tensor(rng.randn(N, 6) * 0.05, dtype=torch.float64)).numpy()
    pert[0] = np.eye(4)
    init_T = pert @ init_T
    init_X = gt_X + rng.randn(M, 3) * 0.1

    def reproj_err(T, X):
        e = []
        for k in range(len(ci)):
            Xc = T[ci[k], :3, :3] @ X[pj[k]] + T[ci[k], :3, 3]
            u = K[ci[k], 0, 0] * Xc[0] / Xc[2] + K[ci[k], 0, 2]
            v = K[ci[k], 1, 1] * Xc[1] / Xc[2] + K[ci[k], 1, 2]
            e.append(((u - uv[k, 0]) ** 2 + (v - uv[k, 1]) ** 2) ** 0.5)
        return float(np.mean(e))

    print(f"obs={len(ci)}  init reproj_rms={reproj_err(init_T, init_X):.3f}px")

    def centres(T):
        return np.stack([-T[i, :3, :3].T @ T[i, :3, 3] for i in range(N)])
    perr0 = np.linalg.norm(centres(init_T) - centres(gt_T), axis=1).mean()

    # (a) reprojection only — must hit the noise floor (proves the BA/Schur math)
    T_a, X_a, _ = bundle_adjust(init_T, K, ci, pj, uv, init_X, iters=20,
                                device="cpu", verbose=False)
    final_a = reproj_err(T_a, X_a)
    perr_a = np.linalg.norm(centres(T_a) - centres(gt_T), axis=1).mean()
    # (b) + strong depth prior — must additionally anchor SCALE (lower pose error)
    T_b, X_b, _ = bundle_adjust(init_T, K, ci, pj, uv, init_X, depth_obs=dep,
                                depth_w=2.0, iters=20, device="cpu", verbose=False)
    final_b = reproj_err(T_b, X_b)
    perr_b = np.linalg.norm(centres(T_b) - centres(gt_T), axis=1).mean()
    print(f"(a) reproj-only : reproj_rms={final_a:.3f}px  pose-centre err {perr0:.3f}→{perr_a:.3f} m")
    print(f"(b) +depth prior: reproj_rms={final_b:.3f}px  pose-centre err {perr0:.3f}→{perr_b:.3f} m")
    ok = (final_a < 0.5) and (perr_a < perr0) and (perr_b < perr_a)
    print("SELFTEST", "PASS" if ok else "FAIL",
          "— BA converges to noise floor; depth prior anchors scale" if ok else "")
    return ok


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
