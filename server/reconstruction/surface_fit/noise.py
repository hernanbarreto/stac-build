"""The object's OWN measured noise — the tolerance of its planes (USER 2026-10-16: "tolerancia = el
ruido local medido del propio objeto", not the session's repeatability, not a declared bar).

The local THICKNESS of the cloud: for a sample of the object's points, the covariance of each
point's ``k`` nearest neighbours; its smallest eigenvalue is the variance across the local
surface, so its square root is the thickness of the cloud there. The object's σ is the median
over the sample — robust to the curved or cluttered minority. No threshold enters the
measurement; ``k`` is a neighbourhood size and ``n_max`` a sample cap (bounds, declared).
"""
from __future__ import annotations

import numpy as np


def local_thickness_sigma(pts: np.ndarray, k: int = 24, n_max: int = 20000, seed: int = 0) -> dict:
    """{sigma_m, n_sampled, k, p50/p90 of the local thickness} of ``pts`` (N, 3)."""
    from scipy.spatial import cKDTree
    P = np.asarray(pts, np.float64)
    n = len(P)
    k = int(max(4, min(int(k), n - 1)))
    if n < k + 1:
        return {"sigma_m": float("nan"), "n_sampled": 0, "k": k}
    rng = np.random.default_rng(int(seed))
    sample = P if n <= int(n_max) else P[np.sort(rng.choice(n, int(n_max), replace=False))]
    dist, nn = cKDTree(P).query(sample, k=k + 1, workers=1)
    Q = P[nn]                                              # (S, k+1, 3): the point and its neighbours
    Q = Q - Q.mean(axis=1, keepdims=True)
    C = np.einsum("sij,sik->sjk", Q, Q) / float(k)        # (S, 3, 3)
    ev = np.linalg.eigvalsh(C)[:, 0]                        # the smallest eigenvalue per sample
    thick = np.sqrt(np.clip(ev, 0.0, None))
    # the SAMPLING of the cloud, from the same query: the nearest-neighbour distance — its p99 is
    # the largest gap the sampling itself leaves inside a continuous surface (what tells a
    # sampling gap from a real separation when a plane's pieces are split)
    nn1 = dist[:, 1]
    return {"sigma_m": float(np.median(thick)), "n_sampled": int(len(sample)), "k": k,
            "p50_m": float(np.median(thick)), "p90_m": float(np.percentile(thick, 90)),
            "spacing_p50_m": float(np.median(nn1)), "spacing_p99_m": float(np.percentile(nn1, 99))}
