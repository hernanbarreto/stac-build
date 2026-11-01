"""F5's bundle adjustment on the GPU (USER 2026-11-01: "todo debe correr en gpu, en modo determinista, sin
comprometer la calidad" — Ceres ran ONE thread for bit-identity, ~60 min per pccr run on a rented card).

The same problem ``precision.refine.run_rung`` hands Ceres — the reprojection residuals of the fit tracks
under the OPENCV camera (fx fy cx cy k1 k2 p1 p2) with a Huber loss at ``huber_px``, the poses, the
landmarks and the camera freedom of the rung (R0 none; R1 focal + principal point; R2 + distortion; R3
focal per block), plus EITHER the absolute position priors of the gauge (σ per keyframe) OR the relative
odometry priors between consecutive keyframes with the first pose constant — solved by Levenberg-Marquardt
with the Schur complement over the landmarks:

* residuals and analytic Jacobians batched over every observation (float64);
* the landmark blocks H_pp (3×3) and the reduced camera system S = H_cc − H_cp H_pp⁻¹ H_pc accumulated
  with SORTED segment sums (cumsum differences) — never an atomic add, so two runs are bit-identical on
  the same device; S is dense (≤ 6·N + 8·blocks unknowns) and solved by Cholesky;
* Ceres' stopping rules (function 1e-6, parameter 1e-8, gradient 1e-8, the iteration cap) and cost
  definition (½ Σ ρ(‖r‖²) + ½ Σ ‖prior residual‖²), so the rungs' reports compare like with like.

What differs from Ceres and is DECLARED: the robust step is the IRLS weighting ρ'(s) (Ceres also applies
its second-order corrector); both converge to stationary points of the same robust cost, the path
differs, so the two solvers agree to the tolerance, not to the bit. The acceptance bar of this solver on a
real session is the held-out track residual against Ceres' (beyond its own noise), never a declaration.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FUNCTION_TOLERANCE = 1e-6       # reconstruction.colmap_ba._tune_ceres
PARAMETER_TOLERANCE = 1e-8
GRADIENT_TOLERANCE = 1e-8
MIN_RELATIVE_DECREASE = 1e-3    # Ceres' default: a step with a smaller gain ratio is rejected
LM_DIAG_MIN, LM_DIAG_MAX = 1e-6, 1e32
CAM_FREE = {"R0": (), "R0s": (), "R1": (0, 1, 2, 3), "R2": (0, 1, 2, 3, 4, 5, 6, 7), "R3": (0, 1)}


class BATorchError(RuntimeError):
    pass


def _torch():
    import torch
    return torch


def device_for(name: Optional[str] = None):
    torch = _torch()
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── small algebra ─────────────────────────────────────────────────────────────────────────

def skew(v):
    torch = _torch()
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def so3_exp(w):
    """Rodrigues, batched, exact for small angles (series below 1e-8)."""
    torch = _torch()
    th = torch.linalg.norm(w, dim=-1, keepdim=True)
    small = th < 1e-8
    th_safe = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, 1.0 - th ** 2 / 6.0, torch.sin(th_safe) / th_safe)
    b = torch.where(small, 0.5 - th ** 2 / 24.0, (1.0 - torch.cos(th_safe)) / th_safe ** 2)
    K = skew(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    return I + a[..., None] * K + b[..., None] * (K @ K)


def so3_log(R):
    """Angle-axis of a rotation matrix, batched (stable through the small-angle series)."""
    torch = _torch()
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    th = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    small = th < 1e-8
    th_safe = torch.where(small, torch.ones_like(th), th)
    f = torch.where(small, 0.5 + th ** 2 / 12.0, th_safe / (2.0 * torch.sin(th_safe)))
    return v * f[..., None]


def segment_sum(values, seg_id, n_seg):
    """Sum ``values`` (M, …) by ``seg_id`` (M,) — ``seg_id`` must be SORTED ascending. Deterministic:
    a cumulative sum and its differences at the segment ends, no atomics."""
    torch = _torch()
    if values.shape[0] == 0:
        return torch.zeros((n_seg,) + tuple(values.shape[1:]), dtype=values.dtype, device=values.device)
    cs = torch.cumsum(values, dim=0)
    last = torch.ones(seg_id.shape[0], dtype=torch.bool, device=values.device)
    last[:-1] = seg_id[1:] != seg_id[:-1]
    ends = torch.nonzero(last, as_tuple=False)[:, 0]
    totals = cs[ends]
    firsts = torch.cat([torch.zeros_like(totals[:1]), totals[:-1]])
    per_present = totals - firsts
    out = torch.zeros((n_seg,) + tuple(values.shape[1:]), dtype=values.dtype, device=values.device)
    out[seg_id[ends]] = per_present
    return out


def sorted_scatter_add(target, keys, values):
    """target[keys] += values with DUPLICATE keys, deterministically: sort the keys (stable), segment-sum
    the values per unique key, add once per key."""
    torch = _torch()
    if keys.shape[0] == 0:
        return target
    order = torch.argsort(keys, stable=True)
    k = keys[order]; v = values[order]
    uniq, inverse = torch.unique_consecutive(k, return_inverse=True)
    summed = segment_sum(v, inverse, int(uniq.shape[0]))
    target[uniq] = target[uniq] + summed
    return target


def pair_indices(seg_lens):
    """All ordered pairs (a, b) of positions inside consecutive segments of lengths ``seg_lens``
    (a segment of m positions gives m² pairs), as global position indices. Deterministic order."""
    torch = _torch()
    m = seg_lens.to(torch.int64)
    starts = torch.cumsum(m, 0) - m
    n_pairs = m * m
    seg_of_pair = torch.repeat_interleave(torch.arange(m.shape[0], device=m.device), n_pairs)
    pair_start = torch.cumsum(n_pairs, 0) - n_pairs
    local = torch.arange(int(n_pairs.sum()), device=m.device) - pair_start[seg_of_pair]
    mm = m[seg_of_pair]
    a = starts[seg_of_pair] + local // mm
    b = starts[seg_of_pair] + local % mm
    return a, b


# ── the camera model: OPENCV (COLMAP's fx fy cx cy k1 k2 p1 p2), residual + analytic Jacobians ──

def project_jac(Xc, cam, cam_free: Sequence[int]):
    """Xc (M,3) camera-frame points, cam (M,8) the params of each observation's camera block.
    Returns (uv (M,2), d uv / d Xc (M,2,3), d uv / d cam_free (M,2,nf), in_front (M,))."""
    torch = _torch()
    fx, fy, cx, cy, k1, k2, p1, p2 = [cam[:, i] for i in range(8)]
    z = Xc[:, 2]
    front = z > 1e-9
    zs = torch.where(front, z, torch.ones_like(z))
    x = Xc[:, 0] / zs; y = Xc[:, 1] / zs
    r2 = x * x + y * y
    rad = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    u = fx * xd + cx; v = fy * yd + cy
    # d(xd, yd) / d(x, y)
    drad_dx = 2.0 * x * (k1 + 2.0 * k2 * r2)
    drad_dy = 2.0 * y * (k1 + 2.0 * k2 * r2)
    dxd_dx = rad + x * drad_dx + 2.0 * p1 * y + 6.0 * p2 * x
    dxd_dy = x * drad_dy + 2.0 * p1 * x + 2.0 * p2 * y
    dyd_dx = y * drad_dx + 2.0 * p1 * x + 2.0 * p2 * y
    dyd_dy = rad + y * drad_dy + 6.0 * p1 * y + 2.0 * p2 * x
    # d(x, y) / d Xc
    inv_z = 1.0 / zs
    dx_dX = torch.stack([inv_z, torch.zeros_like(z), -x * inv_z], -1)       # (M,3)
    dy_dX = torch.stack([torch.zeros_like(z), inv_z, -y * inv_z], -1)
    du_dX = fx[:, None] * (dxd_dx[:, None] * dx_dX + dxd_dy[:, None] * dy_dX)
    dv_dX = fy[:, None] * (dyd_dx[:, None] * dx_dX + dyd_dy[:, None] * dy_dX)
    J_X = torch.stack([du_dX, dv_dX], 1)                                      # (M,2,3)
    cols_u = {0: xd, 1: torch.zeros_like(xd), 2: torch.ones_like(xd), 3: torch.zeros_like(xd),
              4: fx * x * r2, 5: fx * x * r2 * r2, 6: fx * 2.0 * x * y, 7: fx * (r2 + 2.0 * x * x)}
    cols_v = {0: torch.zeros_like(yd), 1: yd, 2: torch.zeros_like(yd), 3: torch.ones_like(yd),
              4: fy * y * r2, 5: fy * y * r2 * r2, 6: fy * (r2 + 2.0 * y * y), 7: fy * 2.0 * x * y}
    if cam_free:
        J_cam = torch.stack([torch.stack([cols_u[i] for i in cam_free], -1),
                             torch.stack([cols_v[i] for i in cam_free], -1)], 1)   # (M,2,nf)
    else:
        J_cam = torch.zeros((Xc.shape[0], 2, 0), dtype=Xc.dtype, device=Xc.device)
    return torch.stack([u, v], -1), J_X, J_cam, front


# ── the problem ────────────────────────────────────────────────────────────────────────────

class _Problem:
    def __init__(self, w2c0: np.ndarray, params_by_block0: List[List[float]], block_of: np.ndarray,
                 cam_idx: np.ndarray, pt_idx: np.ndarray, uv: np.ndarray, X0: np.ndarray, huber_px: float,
                 cam_free: Sequence[int], *, device, sigmas: Optional[np.ndarray] = None,
                 prior_centres: Optional[np.ndarray] = None, rel_w2c: Optional[np.ndarray] = None,
                 rel_cov: Optional[np.ndarray] = None, first_constant: bool = False):
        torch = _torch()
        d = {"dtype": torch.float64, "device": device}
        self.torch, self.device = torch, device
        self.N = int(len(w2c0)); self.B = int(len(params_by_block0)); self.P = int(len(X0)); self.M = int(len(uv))
        self.R = torch.as_tensor(np.asarray(w2c0)[:, :3, :3], **d).clone()
        self.t = torch.as_tensor(np.asarray(w2c0)[:, :3, 3], **d).clone()
        self.cam = torch.as_tensor(np.asarray(params_by_block0, np.float64), **d).clone()      # (B,8)
        self.block_of = torch.as_tensor(np.asarray(block_of, np.int64), device=device)
        self.X = torch.as_tensor(np.asarray(X0, np.float64), **d).clone()
        # observations sorted by point (segment sums over landmarks), the camera order kept for the rest
        order = np.lexsort((cam_idx, pt_idx))
        self.cam_idx = torch.as_tensor(cam_idx[order].astype(np.int64), device=device)
        self.pt_idx = torch.as_tensor(pt_idx[order].astype(np.int64), device=device)
        self.uv = torch.as_tensor(np.asarray(uv, np.float64)[order], **d)
        self.obs_block = self.block_of[self.cam_idx]
        self.huber = float(huber_px)
        self.cam_free = tuple(int(i) for i in cam_free)
        self.nf = len(self.cam_free)
        self.first_constant = bool(first_constant)
        # per point: segment lengths (sorted by point, every point has ≥ 1 observation by construction)
        counts = torch.bincount(self.pt_idx, minlength=self.P)
        if int((counts == 0).sum()):
            raise BATorchError("a landmark without observations entered the problem")
        self.seg_len = counts
        # priors
        self.sigmas = torch.as_tensor(np.asarray(sigmas, np.float64), **d) if sigmas is not None else None
        self.prior_C = torch.as_tensor(np.asarray(prior_centres, np.float64), **d) if prior_centres is not None else None
        self.rel_w2c = torch.as_tensor(np.asarray(rel_w2c, np.float64), **d) if rel_w2c is not None else None
        if rel_cov is not None:
            cov = torch.as_tensor(np.asarray(rel_cov, np.float64), **d)                     # (N-1,6,6)
            self.rel_L = torch.linalg.cholesky(torch.linalg.inv(cov))                       # sqrt information
        else:
            self.rel_L = None
        # unknown layout: poses first (6 each), then the free camera params per block (nf each)
        self.n_pose = 6 * self.N
        self.n_c = self.n_pose + self.nf * self.B
        self.free = torch.ones(self.n_c, dtype=torch.bool, device=device)
        if self.first_constant:
            self.free[:6] = False
        if self.sigmas is None and self.rel_w2c is None and self.nf == 0 and not self.first_constant:
            raise BATorchError("a gauge is needed: priors, a constant first pose or free camera parameters")

    # ── evaluation at the current (or a candidate) state ──
    def residuals(self, R=None, t=None, cam=None, X=None, with_jac: bool = True):
        torch = self.torch
        R = self.R if R is None else R; t = self.t if t is None else t
        cam = self.cam if cam is None else cam; X = self.X if X is None else X
        Ri = R[self.cam_idx]; ti = t[self.cam_idx]
        Xw = X[self.pt_idx]
        Xc = torch.einsum("mij,mj->mi", Ri, Xw) + ti
        cam_o = cam[self.obs_block]
        uv, J_X, J_cam, front = project_jac(Xc, cam_o, self.cam_free)
        r = uv - self.uv
        if not with_jac:
            return r, front
        # d Xc / d(δω, δt) with the left perturbation Xc' = exp(δω) Xc + δt; d Xc / d X = R
        J_pose = torch.cat([-J_X @ skew(Xc), J_X], -1)                      # (M,2,6)
        J_pt = J_X @ Ri                                                     # (M,2,3)
        return r, front, J_pose, J_pt, J_cam

    def huber_weights(self, r, front):
        torch = self.torch
        s = (r * r).sum(-1)
        a2 = self.huber * self.huber
        w = torch.where(s <= a2, torch.ones_like(s), self.huber / torch.sqrt(s.clamp_min(1e-300)))
        return torch.where(front, w, torch.zeros_like(w)), s

    def robust_cost(self, s, front):
        torch = self.torch
        a2 = self.huber * self.huber
        rho = torch.where(s <= a2, s, 2.0 * self.huber * torch.sqrt(s.clamp_min(1e-300)) - a2)
        return 0.5 * float(torch.where(front, rho, torch.zeros_like(rho)).sum())

    def prior_terms(self, R, t):
        """(residual vector (K,), Jacobian (K, n_c)) of the priors at (R, t) — dense, K small."""
        torch = self.torch
        rows, jacs = [], []
        if self.sigmas is not None:
            C = -torch.einsum("nji,nj->ni", R, t)
            e = (C - self.prior_C) / self.sigmas[:, None]                               # (N,3)
            J = torch.zeros(self.N, 3, self.n_c, dtype=R.dtype, device=self.device)
            RT = R.transpose(1, 2)
            dC_dw = -RT @ skew(t)                                                        # (N,3,3)
            dC_dt = -RT
            ar = torch.arange(self.N, device=self.device)
            for a in range(3):
                for b in range(3):
                    J[ar, a, 6 * ar + b] = dC_dw[:, a, b] / self.sigmas
                    J[ar, a, 6 * ar + 3 + b] = dC_dt[:, a, b] / self.sigmas
            rows.append(e.reshape(-1)); jacs.append(J.reshape(-1, self.n_c))
        if self.rel_w2c is not None:
            Wm = self.rel_w2c
            Rm_i, tm_i = Wm[:-1, :3, :3], Wm[:-1, :3, 3]
            Rm_j, tm_j = Wm[1:, :3, :3], Wm[1:, :3, 3]
            Rij_m = Rm_i @ Rm_j.transpose(1, 2)
            tij_m = tm_i - torch.einsum("kij,kj->ki", Rij_m, tm_j)
            Ri, ti, Rj, tj = R[:-1], t[:-1], R[1:], t[1:]
            Rij = Ri @ Rj.transpose(1, 2)
            tij = ti - torch.einsum("kij,kj->ki", Rij, tj)
            e_rot = so3_log(Rij_m @ Rij.transpose(1, 2))                                 # (K,3)
            e_t = tij_m - tij
            e = torch.cat([e_rot, e_t], -1)                                              # (K,6)
            # Jacobians by finite differences in the tangent (K small — 294 pairs): d e / d (δω_i, δt_i, δω_j, δt_j)
            K_ = e.shape[0]
            J = torch.zeros(K_, 6, 12, dtype=R.dtype, device=self.device)
            h = 1e-7
            for p in range(12):
                dR_i, dt_i, dR_j, dt_j = Ri, ti, Rj, tj
                dvec = torch.zeros(K_, 3, dtype=R.dtype, device=self.device)
                if p < 3:
                    dvec[:, p] = h; dR_i = so3_exp(dvec) @ Ri
                elif p < 6:
                    dvec[:, p - 3] = h; dt_i = ti + dvec
                elif p < 9:
                    dvec[:, p - 6] = h; dR_j = so3_exp(dvec) @ Rj
                else:
                    dvec[:, p - 9] = h; dt_j = tj + dvec
                Rij2 = dR_i @ dR_j.transpose(1, 2)
                tij2 = dt_i - torch.einsum("kij,kj->ki", Rij2, dt_j)
                e2 = torch.cat([so3_log(Rij_m @ Rij2.transpose(1, 2)), tij_m - tij2], -1)
                J[:, :, p] = (e2 - e) / h
            # whiten
            e_w = torch.einsum("kab,kb->ka", self.rel_L.transpose(1, 2), e)
            J_w = torch.einsum("kab,kbc->kac", self.rel_L.transpose(1, 2), J)
            Jd = torch.zeros(K_, 6, self.n_c, dtype=R.dtype, device=self.device)
            ar = torch.arange(K_, device=self.device)
            for a in range(6):
                for b in range(6):
                    Jd[ar, a, 6 * ar + b] = J_w[:, a, b]
                    Jd[ar, a, 6 * (ar + 1) + b] = J_w[:, a, 6 + b]
            rows.append(e_w.reshape(-1)); jacs.append(Jd.reshape(-1, self.n_c))
        if not rows:
            return (torch.zeros(0, dtype=R.dtype, device=self.device),
                    torch.zeros(0, self.n_c, dtype=R.dtype, device=self.device))
        return torch.cat(rows), torch.cat(jacs)

    def total_cost(self, R=None, t=None, cam=None, X=None) -> float:
        R = self.R if R is None else R; t = self.t if t is None else t
        cam = self.cam if cam is None else cam; X = self.X if X is None else X
        r, front = self.residuals(R, t, cam, X, with_jac=False)
        s = (r * r).sum(-1)
        c = self.robust_cost(s, front)
        e, _ = self.prior_terms(R, t)
        return c + 0.5 * float((e * e).sum())

    # ── one LM step ──
    def step(self, lam: float, pair_chunk: int = 3_000_000):
        torch = self.torch
        r, front, J_pose, J_pt, J_cam = self.residuals()
        w, s = self.huber_weights(r, front)
        sw = torch.sqrt(w)[:, None]
        cost = self.robust_cost(s, front)
        r_w = r * sw
        J_pose = J_pose * sw[..., None]; J_pt = J_pt * sw[..., None]; J_cam = J_cam * sw[..., None]
        J_c = torch.cat([J_pose, J_cam], -1)                                 # (M,2,6+nf)
        nco = 6 + self.nf
        # landmark blocks
        Hpp = segment_sum(J_pt.transpose(1, 2) @ J_pt, self.pt_idx, self.P)                       # (P,3,3)
        gp = segment_sum(torch.einsum("mai,ma->mi", J_pt, r_w), self.pt_idx, self.P)              # (P,3)
        Dp = torch.diagonal(Hpp, dim1=1, dim2=2).clamp(LM_DIAG_MIN, LM_DIAG_MAX)
        Hpp_l = Hpp + lam * torch.diag_embed(Dp)
        Hpp_inv = torch.linalg.inv(Hpp_l)
        # camera side: dense H_cc, g_c
        H = torch.zeros(self.n_c, self.n_c, dtype=torch.float64, device=self.device)
        g = torch.zeros(self.n_c, dtype=torch.float64, device=self.device)
        self._accumulate_camera(H, g, J_c, r_w, nco)
        # priors
        e_p, J_p = self.prior_terms(self.R, self.t)
        if e_p.shape[0]:
            H = H + J_p.T @ J_p
            g = g + J_p.T @ e_p
            cost = cost + 0.5 * float((e_p * e_p).sum())
        # Schur: S = H − Σ W Hpp⁻¹ Wᵀ ;  ĝ = g − Σ W Hpp⁻¹ g_p
        W = J_c.transpose(1, 2) @ J_pt                                      # (M,nco,3)
        Y = W @ Hpp_inv[self.pt_idx]                                        # (M,nco,3)
        g_red = g - self._reduce_vec(torch.einsum("mij,mj->mi", Y, gp[self.pt_idx]), nco)
        S = H - self._schur_pairs(Y, W, nco, pair_chunk)
        # damping and the constant blocks
        Dc = torch.diagonal(S).clamp(LM_DIAG_MIN, LM_DIAG_MAX)
        S_l = S + lam * torch.diag(Dc)
        free = self.free
        S_f = S_l[free][:, free]
        try:
            Lc = torch.linalg.cholesky(S_f)
            dc_f = -torch.cholesky_solve(g_red[free][:, None], Lc)[:, 0]
        except Exception as ex:  # noqa: BLE001 — an indefinite reduced system: the caller raises λ
            return None, cost, float("inf"), f"cholesky: {ex}"
        dc = torch.zeros(self.n_c, dtype=torch.float64, device=self.device)
        dc[free] = dc_f
        # back-substitution: Δp = Hpp⁻¹ (−g_p − Wᵀ Δc)
        dc_obs = self._gather_cam_delta(dc, nco)                           # (M,nco)
        back = segment_sum(torch.einsum("mij,mi->mj", W, dc_obs), self.pt_idx, self.P)            # (P,3)
        dp = torch.einsum("pij,pj->pi", Hpp_inv, -gp - back)
        # predicted decrease (Ceres' model cost change): −gᵀΔ − ½ Δᵀ H Δ, computed on the full system
        dot_g = float((g_red[free] * dc_f).sum()) + float(((-gp) * dp).sum() * -1.0)
        pred = -(float((dc_f * (S_f @ dc_f)).sum()) * 0.5 + float((g_red[free] * dc_f).sum()))
        # the landmark part of the model decrease
        pred = pred - 0.5 * float(torch.einsum("pi,pij,pj->", dp, Hpp_l, dp)) - float((gp * dp).sum()) - float(
            torch.einsum("mij,mi,mj->", W, dc_obs, dp[self.pt_idx]))
        return (dc, dp), cost, pred, ""

    def _accumulate_camera(self, H, g, J_c, r_w, nco):
        torch = self.torch
        N, nf, B = self.N, self.nf, self.B
        # pose-pose (6×6 at (i,i)), sorted by camera
        order = torch.argsort(self.cam_idx, stable=True)
        Jc = J_c[order]; rw = r_w[order]; ci = self.cam_idx[order]; bi = self.obs_block[order]
        Jp = Jc[:, :, :6]
        HPP = segment_sum(Jp.transpose(1, 2) @ Jp, ci, N)                  # (N,6,6)
        gP = segment_sum(torch.einsum("mai,ma->mi", Jp, rw), ci, N)        # (N,6)
        idx = (6 * torch.arange(N, device=self.device))[:, None] + torch.arange(6, device=self.device)[None, :]
        H[idx[:, :, None], idx[:, None, :]] += HPP
        g[idx] += gP
        if nf:
            Jk = Jc[:, :, 6:]
            HPK = segment_sum(Jp.transpose(1, 2) @ Jk, ci, N)              # (N,6,nf) per camera, its block's params
            cam_cols = (self.n_pose + nf * self.block_of)[:, None] + torch.arange(nf, device=self.device)[None, :]  # (N,nf)
            H[idx[:, :, None], cam_cols[:, None, :]] += HPK
            H[cam_cols[:, :, None], idx[:, None, :]] += HPK.transpose(1, 2)
            # cam-cam per block: sort by block
            ob = torch.argsort(self.obs_block, stable=True)
            Jk_b = J_c[ob][:, :, 6:]; rw_b = r_w[ob]; bb = self.obs_block[ob]
            HKK = segment_sum(Jk_b.transpose(1, 2) @ Jk_b, bb, B)          # (B,nf,nf)
            gK = segment_sum(torch.einsum("mai,ma->mi", Jk_b, rw_b), bb, B)
            bidx = (self.n_pose + nf * torch.arange(B, device=self.device))[:, None] + torch.arange(nf, device=self.device)[None, :]
            H[bidx[:, :, None], bidx[:, None, :]] += HKK
            g[bidx] += gK

    def _cols_of_obs(self, nco):
        """(M,nco) the unknown index of every column of an observation's camera Jacobian."""
        torch = self.torch
        pose_cols = (6 * self.cam_idx)[:, None] + torch.arange(6, device=self.device)[None, :]
        if self.nf:
            cam_cols = (self.n_pose + self.nf * self.obs_block)[:, None] + torch.arange(self.nf, device=self.device)[None, :]
            return torch.cat([pose_cols, cam_cols], 1)
        return pose_cols

    def _reduce_vec(self, v_obs, nco):
        """Σ over observations of a per-observation camera vector (M,nco) into the dense (n_c,)."""
        torch = self.torch
        cols = self._cols_of_obs(nco).reshape(-1)
        out = torch.zeros(self.n_c, dtype=torch.float64, device=self.device)
        return sorted_scatter_add(out, cols, v_obs.reshape(-1))

    def _gather_cam_delta(self, dc, nco):
        return dc[self._cols_of_obs(nco)]

    def _schur_pairs(self, Y, W, nco, pair_chunk):
        """Σ_j Σ_{a,b ∈ obs(j)} Y_a W_bᵀ placed at the unknowns of a's and b's cameras — dense (n_c, n_c),
        accumulated chunk by chunk over the landmarks with sorted segment sums (deterministic)."""
        torch = self.torch
        S = torch.zeros(self.n_c, self.n_c, dtype=torch.float64, device=self.device)
        cols = self._cols_of_obs(nco)                                      # (M,nco)
        starts = torch.cumsum(self.seg_len, 0) - self.seg_len
        p0 = 0
        P = self.P
        lens = self.seg_len
        while p0 < P:
            # take landmarks until the pair budget is reached
            cum = torch.cumsum(lens[p0:] * lens[p0:], 0)
            n_take = int(torch.searchsorted(cum, torch.tensor(pair_chunk, device=cum.device)).item()) + 1
            p1 = min(P, p0 + max(1, n_take))
            o0 = int(starts[p0]); o1 = int(starts[p1 - 1] + lens[p1 - 1])
            a, b = pair_indices(lens[p0:p1])
            a = a + o0; b = b + o0
            contrib = Y[a] @ W[b].transpose(1, 2)                          # (n_pairs, nco, nco)
            rows = cols[a]; cc = cols[b]                                   # (n_pairs, nco)
            key = (rows[:, :, None] * self.n_c + cc[:, None, :]).reshape(-1)
            flat = S.reshape(-1)
            sorted_scatter_add(flat, key, contrib.reshape(-1))
            p0 = p1
        return S

    def apply(self, dc, dp):
        torch = self.torch
        dw = dc[: self.n_pose].reshape(self.N, 6)
        R = so3_exp(dw[:, :3]) @ self.R
        t = self.t + dw[:, 3:]
        cam = self.cam.clone()
        if self.nf:
            dk = dc[self.n_pose:].reshape(self.B, self.nf)
            cam[:, list(self.cam_free)] = cam[:, list(self.cam_free)] + dk
        X = self.X + dp
        return R, t, cam, X

    def norm_x(self) -> float:
        torch = self.torch
        return math.sqrt(float((self.t * self.t).sum()) + float((self.X * self.X).sum()) + float((self.cam * self.cam).sum()) + 3.0 * self.N)


def solve(problem: _Problem, max_iterations: int, log=None) -> Dict[str, object]:
    """Levenberg-Marquardt with Ceres' stopping rules. Returns the summary dict the rung report reads."""
    t0 = time.time()
    lam, nu = 1e-4, 2.0
    cost = problem.total_cost()
    initial = cost
    n_ok = n_bad = 0
    termination = "NO_CONVERGENCE"
    for it in range(int(max_iterations)):
        step, cost_now, pred, why = problem.step(lam)
        if step is None or not math.isfinite(pred):
            lam *= nu; nu *= 2.0; n_bad += 1
            if lam > 1e32:
                termination = "FAILURE"; break
            continue
        dc, dp = step
        R, t, cam, X = problem.apply(dc, dp)
        new_cost = problem.total_cost(R, t, cam, X)
        actual = cost_now - new_cost
        rho = actual / pred if pred > 0 else -1.0
        step_norm = math.sqrt(float((dc * dc).sum()) + float((dp * dp).sum()))
        if rho > MIN_RELATIVE_DECREASE and math.isfinite(new_cost):
            problem.R, problem.t, problem.cam, problem.X = R, t, cam, X
            n_ok += 1
            lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3); nu = 2.0
            rel = abs(actual) / max(cost_now, 1e-300)
            if log:
                log(f"    it {it + 1}: cost {cost_now:.6e} → {new_cost:.6e} (rel {rel:.2e}), step {step_norm:.2e}, λ {lam:.1e}")
            cost = new_cost
            if rel < FUNCTION_TOLERANCE:
                termination = "CONVERGENCE"; break
            if step_norm <= PARAMETER_TOLERANCE * (problem.norm_x() + PARAMETER_TOLERANCE):
                termination = "CONVERGENCE"; break
        else:
            lam *= nu; nu *= 2.0; n_bad += 1
            if lam > 1e32:
                termination = "FAILURE"; break
    return {"initial_cost": float(initial), "final_cost": float(cost), "num_successful_steps": n_ok,
            "num_unsuccessful_steps": n_bad, "total_time_in_seconds": time.time() - t0,
            "termination_type": termination}


# ── the rung interface (precision.refine.run_rung's contract) ──────────────────────────────

def run_rung_torch(name: str, w2c_start: np.ndarray, params_start: Sequence[float], wh: Tuple[int, int],
                   fit_groups: Dict[int, List], X_start: Dict[int, np.ndarray], sigmas: Optional[np.ndarray],
                   cfg, block_of: Optional[np.ndarray] = None, params_by_block0: Optional[List[List[float]]] = None,
                   prior_w2c: Optional[np.ndarray] = None, flags_of: Optional[str] = None, rel_prior=None,
                   device: Optional[str] = None, log=None):
    """The same inputs and the same RungResult as precision.refine.run_rung, solved on the device."""
    from precision.refine import RefineError, RungResult
    torch = _torch()
    dev = device_for(device)
    N = len(w2c_start)
    block_of = np.zeros(N, int) if block_of is None else np.asarray(block_of, int)
    blocks = sorted(set(block_of.tolist()))
    pb0 = params_by_block0 or [list(params_start) for _ in blocks]
    fl = flags_of or name
    cam_free = CAM_FREE.get(fl)
    if cam_free is None:
        raise RefineError(f"{name}: unknown rung flags {fl!r}")
    # the observations of the tracks with a start and ≥ 2 observations (run_rung's rule)
    tracks = [t for t in X_start if t in fit_groups and len(fit_groups[t]) >= 2]
    cam_idx = np.concatenate([np.fromiter((i for i, _ in fit_groups[t]), np.int64, len(fit_groups[t])) for t in tracks])
    pt_idx = np.concatenate([np.full(len(fit_groups[t]), k, np.int64) for k, t in enumerate(tracks)])
    uv = np.concatenate([np.asarray([p for _, p in fit_groups[t]], np.float64).reshape(-1, 2) for t in tracks])
    X0 = np.asarray([X_start[t] for t in tracks], np.float64).reshape(-1, 3)
    n_rep = int(len(uv))
    if rel_prior is not None:
        if len(rel_prior.w2c) != N:
            raise RefineError(f"{name}: {len(rel_prior.w2c)} prior poses for {N} keyframes")
        prob = _Problem(w2c_start, pb0, block_of, cam_idx, pt_idx, uv, X0, float(cfg.huber_px), cam_free, device=dev,
                        rel_w2c=np.asarray(rel_prior.w2c, np.float64), rel_cov=np.asarray(rel_prior.covariances(), np.float64),
                        first_constant=True)
        n_prior = N - 1
    else:
        prior_w2c = w2c_start if prior_w2c is None else np.asarray(prior_w2c, np.float64)
        if prior_w2c.shape != np.asarray(w2c_start).shape:
            raise RefineError(f"{name}: {len(prior_w2c)} prior poses for {N} keyframes")
        C = -np.einsum("nji,nj->ni", prior_w2c[:, :3, :3], prior_w2c[:, :3, 3])
        prob = _Problem(w2c_start, pb0, block_of, cam_idx, pt_idx, uv, X0, float(cfg.huber_px), cam_free, device=dev,
                        sigmas=np.asarray(sigmas, np.float64), prior_centres=C)
        n_prior = N
    summary = solve(prob, int(cfg.max_iterations), log=log)
    if summary["termination_type"] != "CONVERGENCE":
        raise RuntimeError(f"refine {name}: the torch bundle adjustment did NOT converge "
                           f"(termination={summary['termination_type']}) — BA must converge, no fallback")
    # the result in run_rung's terms
    R = prob.R.cpu().numpy(); t = prob.t.cpu().numpy(); cam = prob.cam.cpu().numpy(); X = prob.X.cpu().numpy()
    w2c = np.tile(np.eye(4), (N, 1, 1)); w2c[:, :3, :3] = R; w2c[:, :3, 3] = t
    r, front = prob.residuals(with_jac=False)
    err = torch.linalg.norm(r, dim=-1)
    err = torch.where(front, err, torch.full_like(err, float("inf")))
    # pycolmap's per-point error: the MEAN reprojection error over the track; the fit statistic is their median
    per_pt = segment_sum(err.where(torch.isfinite(err), torch.zeros_like(err)), prob.pt_idx, prob.P) / prob.seg_len.to(err.dtype)
    bad = segment_sum((~torch.isfinite(err)).to(err.dtype), prob.pt_idx, prob.P) > 0
    per_pt = torch.where(bad, torch.full_like(per_pt, float("inf")), per_pt).cpu().numpy()
    diag = float(np.hypot(wh[0], wh[1]))
    ok = np.isfinite(per_pt) & (per_pt >= 0.0) & (per_pt <= diag)
    fit_med = float(np.median(per_pt[ok])) if ok.any() else float("nan")
    X_out = {t_: X[k].copy() for k, t_ in enumerate(tracks)}
    summary["num_residual_blocks"] = n_rep + n_prior
    return RungResult(name, w2c, [list(map(float, cam[b])) for b in range(len(blocks))], block_of,
                      fit_med, int(len(tracks)), summary["termination_type"], int((~ok).sum()), X=X_out,
                      initial_cost=float(summary["initial_cost"]), final_cost=float(summary["final_cost"]),
                      iterations=int(summary["num_successful_steps"]) + int(summary["num_unsuccessful_steps"]),
                      seconds=float(summary["total_time_in_seconds"]))
