"""F2's per-frame gains on the card (USER 2026-11-01: "todo debe correr en GPU … en el menor tiempo posible").

MEASURED: F2 took 178 s on pccr; its cost is ``precision.gauge.huber_location`` — one Huber M-estimate of
log(DA3 / Omega) per frame of every I3 window (and of its monocular depth), each an IRLS over ~385 k pixels whose
every step takes a full median (the MAD about the current estimate) and two weighted sums.

Here the frames of a window are estimated TOGETHER: their residuals padded into one (frames × pixels) tensor, every
step the reference's arithmetic per row — the MAD by sorting (the same middle element(s) and the same (a + b) / 2 as
``np.median``), the Huber weights, the weighted mean, the reference's stopping rule (the declared tolerance, never
finer than Higham's γ_n bound on the two sums, point 39) — rows that stop are frozen with their iterate and their
step count; a row whose MAD vanishes takes the reference's own answer (the weighted median, on the CPU, exactly).
The sums are float64 in another order than numpy's: an iterate agrees to ~1e-15 relative and the step at which the
γ_n bound stops it can differ by one — the gain then differs by at most that bound (~1e-11), which the rule itself
declares unresolvable. tests/test_gauge_torch.py holds the batch equal to the reference to 1e-9 and the flags equal.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _torch():
    import torch
    return torch


def huber_location_batch(rs: Sequence[np.ndarray], w0s: Sequence[np.ndarray], k: float, tol: float,
                         max_iter: int, device: str):
    """``precision.gauge.huber_location`` for every (r, w0) of ``rs`` / ``w0s`` at once on ``device``; returns the
    list of ``precision.gauge.Gain`` in the same order."""
    torch = _torch()
    from precision import gauge as G
    B = len(rs)
    if B == 0:
        return []
    n = np.array([len(r) for r in rs], np.int64)
    N = int(n.max())
    inf = float("inf")
    R = torch.full((B, N), inf, dtype=torch.float64, device=device)
    W0 = torch.zeros((B, N), dtype=torch.float64, device=device)
    for b, (r, w) in enumerate(zip(rs, w0s)):
        R[b, :n[b]] = torch.as_tensor(np.asarray(r, np.float64), device=device)
        W0[b, :n[b]] = torch.as_tensor(np.asarray(w, np.float64), device=device)
    n_t = torch.as_tensor(n, device=device)
    valid = torch.arange(N, device=device)[None, :] < n_t[:, None]
    lo = (n_t - 1) >> 1
    hi = n_t >> 1

    def _median_rows(V):                                     # V: invalid entries +inf
        s = torch.sort(V, dim=1).values
        a = torch.gather(s, 1, lo[:, None])[:, 0]
        b = torch.gather(s, 1, hi[:, None])[:, 0]
        return torch.where(lo == hi, a, (a + b) / 2.0)
    mu = _median_rows(R)
    gamma = torch.as_tensor([G._summation_gamma(int(x)) for x in n], dtype=torch.float64, device=device)
    tol_t = float(tol)
    out: List = [None] * B
    active = torch.ones(B, dtype=torch.bool, device=device)
    Rz = torch.where(valid, R, torch.zeros((), dtype=torch.float64, device=device))
    for it in range(1, int(max_iter) + 1):
        dev_abs = torch.where(valid, (Rz - mu[:, None]).abs(), torch.full((), inf, dtype=torch.float64, device=device))
        mad = _median_rows(dev_abs)
        z = torch.where(valid, (Rz - mu[:, None]).abs() / (G.MAD_TO_SIGMA * mad[:, None]),
                        torch.zeros((), dtype=torch.float64, device=device))
        w = W0 * torch.where(z <= float(k), torch.ones_like(z), float(k) / torch.clamp(z, min=1e-12))
        w = torch.where(valid, w, torch.zeros_like(w))
        sw = w.sum(1)
        new = (w * Rz).sum(1) / sw
        resolution = 2.0 * gamma * ((w * Rz.abs()).sum(1) / sw + new.abs())
        bar = torch.maximum(tol_t * torch.clamp(mu.abs(), min=1.0), resolution)
        stop = (new - mu).abs() <= bar
        mad0 = mad == 0.0
        flags = torch.stack([active, mad0, stop]).cpu().numpy()
        mu_np, new_np = mu.cpu().numpy(), new.cpu().numpy()
        for b in np.flatnonzero(flags[0]):
            if flags[1][b]:
                out[b] = G.Gain(G._weighted_median(np.asarray(rs[b], np.float64), np.asarray(w0s[b], np.float64)),
                                True, it - 1, True)
            elif flags[2][b]:
                out[b] = G.Gain(float(new_np[b]), True, it, False)
        done = torch.as_tensor(flags[1] | flags[2], device=device)
        active = active & ~done
        mu = torch.where(active, new, mu)
        if not bool(active.any()):
            break
    mu_np = mu.cpu().numpy()
    for b in range(B):
        if out[b] is None:
            out[b] = G.Gain(float(mu_np[b]), False, int(max_iter), False)
    return out
