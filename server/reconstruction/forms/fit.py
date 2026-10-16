"""The cloud term: pose + dimensions of the VLM's form by a robust fit to the object's points.

Huber loss on the distance to the template over the session's own repeatability σ (the only
tolerance), Adam on the template's parameters, in blocks; the fit stops when a block no longer
lowers the loss by more than the optimiser's own tolerance or at ``max_blocks`` (declared). The
residual report reads EVERY point of the object (never the subsample). Deterministic: seeded
subsample, fixed schedule, no data-dependent branching on the card.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .templates import Template, _t

REL_TOL = 1e-4   # the optimiser's own stopping tolerance on a block's relative loss change


def subsample(P: np.ndarray, n: int, seed: int) -> np.ndarray:
    P = np.asarray(P, np.float64)
    if len(P) <= int(n):
        return P
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(P), int(n), replace=False))
    return P[idx]


def huber(r: torch.Tensor, k: float) -> torch.Tensor:
    a = torch.abs(r)
    return torch.where(a <= k, 0.5 * r * r, k * (a - 0.5 * k))


def residual_report(template: Template, P_all: np.ndarray, sigma: float, dev,
                    chunk: int = 2_000_000) -> Dict[str, float]:
    """Distances of EVERY point to the surface, in metres, with the shares within 1/2/3 σ."""
    ds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(P_all), chunk):
            ds.append(template.distance(_t(P_all[i:i + chunk], dev)).detach().cpu().numpy())
    d = np.concatenate(ds) if ds else np.zeros(0)
    if not len(d):
        return {"n": 0}
    s = max(float(sigma), 1e-9)
    return {"n": int(len(d)), "median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)),
            "rms_m": float(np.sqrt(np.mean(d * d))), "sigma_m": float(sigma),
            "within_1sigma": float(np.mean(d <= s)), "within_2sigma": float(np.mean(d <= 2 * s)),
            "within_3sigma": float(np.mean(d <= 3 * s))}


def fit_cloud(template: Template, P: np.ndarray, sigma: float, *, lr: float, block_iterations: int,
              max_blocks: int, huber_k: float, max_points: int, seed: int, dev,
              extra_loss: Optional[Callable[[], torch.Tensor]] = None,
              judge: Optional[Callable[[int, float], bool]] = None,
              measure_extents: bool = True,
              log: Callable[[str], None] = lambda m: None) -> dict:
    """Fit ``template`` to ``P`` (metres, the cloud's frame). ``extra_loss`` (the silhouette
    term) is added to the cloud term when given; ``judge(block, loss)`` may stop the fit (the
    consensus judge of the corrector) — without it the block's relative loss change decides.
    ``measure_extents`` (the cloud-only fit): a plate's outline is re-measured from the points
    at every block and, at the end, no extent is left beyond the last measured point (see
    ``Template.measure_extents`` / ``bound_extents``); the corrector sets it False.
    Returns the record (blocks, iterations, losses, the stop reason)."""
    t0 = time.time()
    Ps = _t(subsample(P, max_points, seed), dev)
    params = template.parameters()
    opt = torch.optim.Adam(params, lr=float(lr))
    s = max(float(sigma), 1e-6)
    losses: List[float] = []
    stop = "max_blocks"
    it = 0
    for b in range(int(max_blocks)):
        last = None
        if measure_extents:
            template.measure_extents(Ps)
        for _ in range(int(block_iterations)):
            opt.zero_grad(set_to_none=True)
            d = template.distance(Ps)
            loss = huber(d / s, float(huber_k)).mean()
            if extra_loss is not None:
                loss = loss + extra_loss()
            loss.backward()
            opt.step()
            it += 1
            last = float(loss.detach())
        losses.append(last)
        if measure_extents:
            template.bound_extents(Ps, s)  # every block: a drift into another part's points
            #                                 would hand it those points at the next assignment
        if judge is not None:
            if judge(b, last):
                stop = "judge"
                break
        elif len(losses) >= 2:
            prev = losses[-2]
            if prev - last <= REL_TOL * max(abs(prev), 1e-12):
                stop = "converged"
                break
    if measure_extents:
        template.measure_extents(Ps)       # a plate's outline: the measured one
    return {"blocks": len(losses), "iterations": it, "loss": losses, "stop": stop,
            "n_points_fit": int(len(Ps)), "seconds": round(time.time() - t0, 2)}
