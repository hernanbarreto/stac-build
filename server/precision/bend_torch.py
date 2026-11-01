"""F6's bend on the card (USER 2026-11-01: "todo debe correr en GPU … en el menor tiempo posible").

MEASURED on pccr (the copy at F5's epoch, analysis/2026-11-01_f5torch_main/f6_bend_profile.log): the bend took 70 s
of F6 — 1,557 calls of ``depth_on_f5.irls_huber_fit`` (5 windows × 295 keyframes, plus the refits of the judged
terms), each 10 Huber IRLS steps of a numpy ``lstsq`` and a ``median`` over the window's landmark rows.

Here every keyframe of a window is fitted AT ONCE: the keyframes' landmark rows are stored in keyframe order, so the
rows of keyframe i's window are one contiguous range of that concatenation; the ranges are gathered into a padded
batch (zero rows: they change neither the least squares nor, masked, the median), and the reference's steps run on
the batch — the least squares by torch's QR solver (numpy's is an SVD; a full-rank 3-column design gives the same
solution to rounding), the MAD by sorting (the same middle element(s) and the same (a + b) / 2), the weights, the
rank test with numpy's tolerance (σ_max · max(rows, cols) · eps over each keyframe's OWN row count), the covariance
by the batched inverse (a singular matrix: σ = inf, as numpy's LinAlgError). The judged terms (point 48) are kept
and refitted per keep-pattern as ``significant_bend`` does. tests/test_bend_torch.py holds the coefficients and σ
equal to the numpy path to 1e-9 and the kept terms identical.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

def _torch():
    import torch
    return torch


def _median_masked(E, n_valid):
    """Per row of ``E`` (B, N; invalid entries +inf), the median of its first ``n_valid`` sorted values."""
    torch = _torch()
    s = torch.sort(E, dim=1).values
    lo = ((n_valid - 1) >> 1).clamp(min=0)
    hi = (n_valid >> 1).clamp(max=int(E.shape[1]) - 1)
    a = torch.gather(s, 1, lo[:, None])[:, 0]
    b = torch.gather(s, 1, hi[:, None])[:, 0]
    return torch.where(lo == hi, a, (a + b) / 2.0)


def irls_batch(A, r, valid, k: float, iterations: int):
    """``depth_on_f5.irls_huber_fit`` for a batch: ``A`` (B, N, p), ``r`` (B, N), ``valid`` (B, N) bool (the padding
    is False and zero). Returns (c (B, p), σ (B, p)) as tensors."""
    torch = _torch()
    from precision.depth_on_f5 import MAD_TO_SIGMA        # the reference's constant (1 / 0.67449…)
    B, N, p = (int(x) for x in A.shape)
    vf = valid.to(A.dtype)
    n = valid.sum(1)
    w = vf.clone()
    c = torch.zeros((B, p), dtype=A.dtype, device=A.device)
    inf = torch.full((), float("inf"), dtype=A.dtype, device=A.device)
    for _ in range(int(iterations)):
        Aw = A * w[:, :, None]
        rw = r * w
        c = torch.linalg.lstsq(Aw, rw[:, :, None]).solution[:, :, 0]
        e = r - torch.einsum("bnp,bp->bn", A, c)
        ae = torch.where(valid, e.abs(), inf)
        s = MAD_TO_SIGMA * _median_masked(ae, n) + 1e-12
        w = torch.sqrt(torch.clamp(k * s[:, None] / torch.clamp(e.abs(), min=1e-12), max=1.0)) * vf
    Aw = A * w[:, :, None]
    sv = torch.linalg.svdvals(Aw)                                              # (B, p)
    tol = sv.max(1).values * torch.clamp(n, min=p).to(A.dtype) * float(np.finfo(np.float64).eps)
    rank = (sv > tol[:, None]).sum(1)
    res = (w * (r - torch.einsum("bnp,bp->bn", A, c))) * vf
    dof = (n - p).to(A.dtype)
    s2 = (res * res).sum(1) / torch.where(dof > 0, dof, torch.ones_like(dof))
    G = torch.einsum("bnp,bnq->bpq", Aw, Aw)
    Ginv, info = torch.linalg.inv_ex(G)
    var = s2[:, None] * torch.diagonal(Ginv, dim1=1, dim2=2)
    sigma = torch.sqrt(torch.clamp(var, min=0.0))
    bad = (n <= p) | (rank < p) | (info != 0)
    sigma = torch.where(bad[:, None], inf, sigma)
    return c, sigma


def _window_batches(off: np.ndarray, n_kf: int, window: int, min_rows: int, budget: int):
    """The keyframes of a window with ≥ ``min_rows`` rows, in chunks whose padded size stays under ``budget``
    entries: [(keyframes, starts, counts)]. ``off``: prefix offsets of the keyframes' rows (n_kf + 1,)."""
    ks = np.arange(n_kf)
    lo = np.maximum(0, ks - window)
    hi = np.minimum(n_kf, ks + window + 1)
    starts, counts = off[lo], off[hi] - off[lo]
    fit = np.flatnonzero(counts >= min_rows)
    out, cur, cur_max = [], [], 0
    for i in fit:
        m = max(cur_max, int(counts[i]))
        if cur and m * (len(cur) + 1) > budget:
            out.append(np.asarray(cur)); cur, cur_max = [], 0
            m = int(counts[i])
        cur.append(int(i)); cur_max = m
    if cur:
        out.append(np.asarray(cur))
    return [(b, starts[b], counts[b]) for b in out], counts


def bend_coefficients_t(rows: Dict[int, tuple], n: int, window: int, min_rows: int, huber_k: float,
                        iterations: int, error_factor: float, device: str,
                        budget: int = 30_000_000) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """``depth_on_f5.bend_coefficients`` on ``device``: ({i: (c0, c1, c2)}, {i: info}) — every keyframe fitted on the
    rows of i−w … i+w with the judged terms of ``significant_bend``; fewer than ``min_rows`` rows → Omega's depth
    (k = 1, info['fallback'] = 'identity')."""
    torch = _torch()
    from precision.depth_on_f5 import IDENTITY
    A_all = np.concatenate([np.asarray(rows[j][0], np.float64).reshape(-1, 3) if j in rows else np.zeros((0, 3))
                            for j in range(n)])
    r_all = np.concatenate([np.asarray(rows[j][1], np.float64).ravel() if j in rows else np.zeros(0)
                            for j in range(n)])
    cnt = np.array([len(rows[j][1]) if j in rows else 0 for j in range(n)], np.int64)
    off = np.concatenate([[0], np.cumsum(cnt)])
    A_t = torch.as_tensor(A_all, device=device)
    r_t = torch.as_tensor(r_all, device=device)
    out: Dict[int, np.ndarray] = {}
    infos: Dict[int, dict] = {}
    batches, counts = _window_batches(off, n, int(window), int(min_rows), int(budget))
    fac = float(error_factor)
    for ks, st, ct in batches:
        Nmax = int(ct.max())
        t = torch.arange(Nmax, device=device)
        st_t = torch.as_tensor(st, device=device)
        ct_t = torch.as_tensor(ct, device=device)
        valid = t[None, :] < ct_t[:, None]
        idx = torch.where(valid, st_t[:, None] + t[None, :], torch.zeros_like(t)[None, :])
        A = A_t[idx] * valid[:, :, None].to(A_t.dtype)
        r = r_t[idx] * valid.to(r_t.dtype)
        c_full, s_full = irls_batch(A, r, valid, float(huber_k), int(iterations))
        keep1 = c_full[:, 1].abs() >= fac * s_full[:, 1]
        keep2 = c_full[:, 2].abs() >= fac * s_full[:, 2]
        c = c_full.clone()
        sig = s_full.clone()
        pattern = keep1.to(torch.int64) + 2 * keep2.to(torch.int64)
        for pat in (0, 1, 2):                                    # 3 = all kept: the full fit stands
            sel = torch.nonzero(pattern == pat)[:, 0]
            if int(sel.shape[0]) == 0:
                continue
            cols = [0] + ([1] if pat & 1 else []) + ([2] if pat & 2 else [])
            cr, sr = irls_batch(A[sel][:, :, cols], r[sel], valid[sel], float(huber_k), int(iterations))
            cc = torch.zeros((int(sel.shape[0]), 3), dtype=c.dtype, device=c.device)
            ss = torch.full((int(sel.shape[0]), 3), float("nan"), dtype=c.dtype, device=c.device)
            cc[:, cols] = cr
            ss[:, cols] = sr
            c[sel] = cc
            sig[sel] = ss
        c_np, sig_np = c.cpu().numpy(), sig.cpu().numpy()
        cf_np, sf_np = c_full.cpu().numpy(), s_full.cpu().numpy()
        k1, k2 = keep1.cpu().numpy(), keep2.cpu().numpy()
        for q, i in enumerate(ks.tolist()):
            out[i] = c_np[q].astype(np.float64)
            infos[i] = {"rows": int(ct[q]), "sigma_full": [float(x) for x in sf_np[q]],
                        "kept": {"c1": bool(k1[q]), "c2": bool(k2[q])},
                        "margins": {"c1": float(abs(cf_np[q, 1]) - fac * sf_np[q, 1]),
                                    "c2": float(abs(cf_np[q, 2]) - fac * sf_np[q, 2])},
                        "sigma": [float(x) for x in sig_np[q]], "fallback": None}
    for i in range(n):
        if i not in out:
            out[i] = IDENTITY.copy()
            infos[i] = {"rows": int(counts[i]), "sigma_full": None, "kept": None, "margins": None,
                        "sigma": None, "fallback": "identity"}
        infos[i]["window"] = int(window)
    return out, infos
