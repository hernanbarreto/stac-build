# STAC-Builder — Phase R.3: onion (duplicated-surface) detector.
#
# Per instance, pool its points from all views/windows and analyse the
# distribution ALONG the normal of its OBB. A "cebolla" (onion / doubled
# surface) from window mis-registration shows up as bimodality; the separation
# between the two modes IN METRES is the local registration error — the central
# health metric of Phase R.
#
# Metric: GMM(2) vs GMM(1) by BIC (spec R.3). Separation is |mean_a - mean_b|
# along the axis (local OBB coords are metric).
#
# PROVENANCE: ours (metric). Uses geometry.signed_distances_along_axis (OBB from
# the R3D-ported fitter).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnionResult:
    bimodal: bool
    separation_m: float          # distance between modes along the OBB normal (m)
    bic_delta: float             # BIC(1) - BIC(2); >0 favours 2 components
    weights: tuple[float, float] # mixture weights of the two modes (0,0 if 1)
    n_points: int


def _obb_normal_axis(aabb: np.ndarray) -> int:
    """Index of the OBB's shortest extent = its surface normal direction."""
    ex = aabb[1] - aabb[0]
    ey = aabb[3] - aabb[2]
    ez = aabb[5] - aabb[4]
    return int(np.argmin([ex, ey, ez]))


def detect_onion(points: np.ndarray, obb_transform: np.ndarray, obb_aabb: np.ndarray,
                 min_points: int = 40, min_separation_m: float = 0.01) -> OnionResult:
    """Fit 1- vs 2-component GMM to the point distribution along the OBB normal.
    Bimodal iff BIC(2) < BIC(1) and the modes are separated by >min_separation_m."""
    from .geometry import signed_distances_along_axis

    n = len(points)
    if n < min_points:
        return OnionResult(False, 0.0, 0.0, (0.0, 0.0), n)

    axis = _obb_normal_axis(obb_aabb)
    x = signed_distances_along_axis(points, obb_transform, axis=axis).reshape(-1, 1)

    from sklearn.mixture import GaussianMixture

    try:
        g1 = GaussianMixture(n_components=1, covariance_type="full",
                             random_state=0).fit(x)
        g2 = GaussianMixture(n_components=2, covariance_type="full",
                             random_state=0, n_init=2).fit(x)
    except Exception:
        return OnionResult(False, 0.0, 0.0, (0.0, 0.0), n)

    bic1, bic2 = g1.bic(x), g2.bic(x)
    means = g2.means_.ravel()
    sep = float(abs(means[0] - means[1]))
    weights = tuple(float(w) for w in g2.weights_.ravel())
    # require both modes to carry meaningful mass, not a tiny outlier cluster
    balanced = min(weights) > 0.1
    bimodal = bool(bic2 < bic1 and sep > min_separation_m and balanced)
    return OnionResult(bimodal, sep if bimodal else 0.0, float(bic1 - bic2),
                       weights, n)
