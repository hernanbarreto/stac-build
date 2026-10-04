"""The floor metric as an automatic report (precision/cloud_metrics.py, 2026-10-04): a synthetic floor with
known noise and a known wave, cameras upright above it."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import cloud_metrics as CM  # noqa: E402
from precision.config import load_precision_config  # noqa: E402


def _cams(n=6, h=1.6):
    out = []
    for i in range(n):
        c = np.eye(4); c[:3, 1] = [0, -1, 0]; c[:3, 3] = [i * 1.5, h, 0]    # camera y points down → up = +y; cameras h above
        out.append(c)
    return np.array(out)


def test_floor_thickness_and_undulation_are_what_was_injected():
    rng = np.random.default_rng(0)
    n = 400_000
    xz = rng.uniform(0, 10, (n, 2))
    sigma = 0.01
    wave = 0.05 * np.sin(2 * np.pi * xz[:, 0] / 10.0)          # a 5 cm wave over the 10 m, tilt-free
    y = rng.normal(0, sigma, n) + wave
    floor = np.stack([xz[:, 0], y, xz[:, 1]], 1)
    wall = np.stack([rng.uniform(0, 10, 50_000), rng.uniform(0, 3, 50_000), np.full(50_000, 10.0)], 1)
    X = np.concatenate([floor, wall])
    cfg = load_precision_config().cloud_metrics
    rep = CM.floor_metric(X, _cams(), cfg)
    assert rep["cells"] >= 80
    assert abs(rep["thickness_m"]["median"] - 2.563 * sigma) < 0.6 * sigma      # p90 − p10 of a normal = 2.563 σ
    assert 0.05 < rep["undulation_m"] < 0.11                                   # the wave's p90 − p10 (≈ 0.08)
    assert rep["plane"]["tilt_vs_camera_up_deg"] < 1.0
    assert abs(rep["cameras_above_floor_m"]["median"] - 1.6) < 0.05


def test_no_floor_is_a_declared_error():
    rng = np.random.default_rng(1)
    X = np.stack([rng.uniform(0, 10, 20000), rng.uniform(0, 3, 20000), np.full(20000, 10.0)], 1)  # a wall only
    with pytest.raises(CM.CloudMetricsError):
        CM.floor_metric(X, _cams(), load_precision_config().cloud_metrics)
