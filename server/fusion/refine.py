"""Scene refinement: trimmed RIGID ICP over the overlap, scale BLOCKED.

Replaces the retired CloudComPy ``ICP(adjustScale=True)`` stages (F1/F3):
scale from an ICP over partially-overlapping clouds is exactly the failure
mode that produced the symmetric √s split — complementary coverage lets the
optimizer trade scale against sliding. Here the scale is NEVER a free
parameter of the refinement: it comes from the model-dimension ratio of the
fitted primitives (``align``) or is locked at 1 when both scans are
validated; the refinement only polishes the rigid pose on the geometry both
scans actually share.

Overlap = scan points (already coarse-aligned) whose reference NN sits
within ``overlap_radius_m``. Convergence is by step size
(``icp_convergence_m``), never a blind iteration count.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from fusion.config import FusionConfig


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2,
                                              -1, 1))))


def overlap_subset(scan_pts: np.ndarray, ref_tree: cKDTree,
                   cfg: FusionConfig,
                   rng: np.random.Generator) -> np.ndarray:
    """Indices (into scan_pts) of the overlap region, subsampled to
    icp_samples."""
    n = min(len(scan_pts), cfg.refine.icp_samples * 2)
    sub = np.arange(len(scan_pts)) if n == len(scan_pts) else \
        rng.choice(len(scan_pts), n, replace=False)
    d, _ = ref_tree.query(scan_pts[sub], workers=cfg.runtime.workers)
    inside = sub[d <= cfg.refine.overlap_radius_m]
    if len(inside) > cfg.refine.icp_samples:
        inside = rng.choice(inside, cfg.refine.icp_samples, replace=False)
    return inside


def trimmed_rigid_icp(src: np.ndarray, ref_tree: cKDTree,
                      ref_pts: np.ndarray, cfg: FusionConfig
                      ) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Trimmed point-to-point rigid ICP src→ref (full SE(3) increment; the
    tilt gate bounds pitch/roll afterwards). Returns (R, t, trimmed_rms,
    iterations)."""
    R = np.eye(3)
    t = np.zeros(3)
    S = src.copy()
    rms = float("inf")
    it = 0
    for it in range(1, cfg.refine.icp_max_iters + 1):
        d, j = ref_tree.query(S, workers=cfg.runtime.workers)
        k = max(100, int(len(S) * cfg.refine.icp_trim))
        sel = np.argsort(d)[:k]
        P, Q = S[sel], ref_pts[j[sel]]
        mp, mq = P.mean(0), Q.mean(0)
        H = (P - mp).T @ (Q - mq)
        U, _sv, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Ri = Vt.T @ D @ U.T
        ti = mq - Ri @ mp
        S = S @ Ri.T + ti
        R = Ri @ R
        t = Ri @ t + ti
        rms = float(np.sqrt((np.linalg.norm(S[sel] - Q, axis=1) ** 2).mean()))
        if np.linalg.norm(ti) < cfg.refine.icp_convergence_m \
                and rot_deg(Ri) < 0.001:
            break
    return R, t, rms, it


def refine_scene(scan_pts_aligned: np.ndarray, ref_pts: np.ndarray,
                 cfg: FusionConfig, rng: np.random.Generator,
                 log=print) -> dict:
    """Polish the coarse-aligned scan pose against the reference overlap.
    Returns {"R", "t", "rms_m", "n_overlap", "iterations"} — R,t is the
    INCREMENT to compose on top of the coarse alignment. Fails fast when
    there is no overlap (complementary scans with nothing shared cannot be
    fused)."""
    ref_tree = cKDTree(ref_pts)
    inside = overlap_subset(scan_pts_aligned, ref_tree, cfg, rng)
    if len(inside) < 100:
        raise RuntimeError(
            f"scene refinement: only {len(inside)} scan points fall within "
            f"{cfg.refine.overlap_radius_m} m of the reference after the "
            f"primitive alignment — the scans share too little geometry (or "
            f"the alignment failed); check the pairs")
    R, t, rms, it = trimmed_rigid_icp(scan_pts_aligned[inside], ref_tree,
                                      ref_pts, cfg)
    log(f"  refine: {len(inside):,} overlap pts, {it} iterations, trimmed "
        f"rms {rms*100:.2f} cm, increment rot {rot_deg(R):.3f}° |t| "
        f"{np.linalg.norm(t):.4f} m (scale LOCKED)")
    return {"R": R, "t": t, "rms_m": rms, "n_overlap": int(len(inside)),
            "iterations": int(it)}
