# STAC-Builder — Phase 4: cheap classical frame-quality metrics.
#
# The cost-saving half of ingestion QC: blur (variance of the Laplacian) and
# exposure (mean brightness + clipped-pixel fractions) are computed for EVERY
# sampled frame; the expensive VLM is only invoked on the ambiguous band (see
# prefilter.py). Pure numpy/OpenCV, fully unit-tested — no model, no I/O beyond
# an optional image path.
#
# PROVENANCE: ours (Laplacian-variance blur is the classic Pech-Pacheco metric).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class FrameMetrics:
    lap_var: float          # variance of the Laplacian (higher = sharper)
    brightness: float       # mean luma 0..255
    clip_low: float         # fraction of near-black pixels (<=5)
    clip_high: float        # fraction of near-white pixels (>=250)

    def as_dict(self) -> dict:
        return {"lap_var": self.lap_var, "brightness": self.brightness,
                "clip_low": self.clip_low, "clip_high": self.clip_high}


def _to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        if cv2 is not None:
            return cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return (a[..., :3] @ np.array([0.299, 0.587, 0.114])).astype(np.float64)
    return a.astype(np.float64)


def laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard focus/blur measure. Sharp images
    have high-frequency content and a high variance; blurred ones do not."""
    g = _to_gray(img).astype(np.float64)
    if cv2 is not None:
        lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    else:  # 4-neighbour discrete Laplacian fallback
        lap = (-4 * g + np.roll(g, 1, 0) + np.roll(g, -1, 0)
               + np.roll(g, 1, 1) + np.roll(g, -1, 1))
        lap = lap[1:-1, 1:-1]
    return float(lap.var())


def frame_metrics(img: np.ndarray) -> FrameMetrics:
    g = _to_gray(img).astype(np.float64)
    total = g.size or 1
    return FrameMetrics(
        lap_var=laplacian_variance(img),
        brightness=float(g.mean()),
        clip_low=float((g <= 5).sum()) / total,
        clip_high=float((g >= 250).sum()) / total,
    )
