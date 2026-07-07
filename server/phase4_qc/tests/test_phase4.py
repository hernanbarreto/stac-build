# STAC-Builder — Phase 4 unit tests (synthetic, no model).
#
# Covers the deterministic core: Laplacian-variance blur ordering, classical
# triage banding (keep / drop / ambiguous), voxel density, low-density zone
# grouping, and frame-visibility reprojection.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase4_qc.metrics import frame_metrics, laplacian_variance  # noqa: E402
from phase4_qc.prefilter import CaptureQC  # noqa: E402
from phase4_qc.coverage import (  # noqa: E402
    voxel_density, low_density_zones, frames_viewing,
)

rng = np.random.default_rng(0)


# ── blur metric ─────────────────────────────────────────────────────
def test_sharp_beats_blurred():
    sharp = (rng.random((200, 200, 3)) * 255).astype(np.uint8)  # high-freq noise
    # blur it by 4x downscale-average-upscale
    small = sharp.reshape(50, 4, 50, 4, 3).mean((1, 3))
    blurred = np.repeat(np.repeat(small, 4, 0), 4, 1).astype(np.uint8)
    assert laplacian_variance(sharp) > laplacian_variance(blurred)


def test_frame_metrics_exposure():
    dark = np.zeros((64, 64, 3), np.uint8)
    m = frame_metrics(dark)
    assert m.brightness == 0.0 and m.clip_low == 1.0 and m.clip_high == 0.0
    bright = np.full((64, 64, 3), 255, np.uint8)
    mb = frame_metrics(bright)
    assert mb.clip_high == 1.0


# ── classical triage banding ────────────────────────────────────────
def _qc():
    return CaptureQC("/nonexistent", config={"phase4": {
        "blur_keep_above": 120.0, "blur_drop_below": 25.0,
        "brightness_lo": 40.0, "brightness_hi": 220.0, "clip_frac_max": 0.25}})


def test_triage_keep():
    qc = _qc()
    d, r = qc.triage({"lap_var": 300.0, "brightness": 120.0, "clip_low": 0.0, "clip_high": 0.0})
    assert d == "keep" and r == []


def test_triage_drop_on_blur():
    qc = _qc()
    d, r = qc.triage({"lap_var": 10.0, "brightness": 120.0, "clip_low": 0.0, "clip_high": 0.0})
    assert d == "drop" and r == ["blur"]


def test_triage_ambiguous_borderline_blur():
    qc = _qc()
    d, r = qc.triage({"lap_var": 60.0, "brightness": 120.0, "clip_low": 0.0, "clip_high": 0.0})
    assert d == "ambiguous" and "borderline_blur" in r


def test_triage_ambiguous_on_exposure_even_if_sharp():
    qc = _qc()
    d, r = qc.triage({"lap_var": 500.0, "brightness": 10.0, "clip_low": 0.0, "clip_high": 0.0})
    assert d == "ambiguous" and "dark" in r


# ── coverage geometry ───────────────────────────────────────────────
def test_voxel_density_counts():
    pts = np.array([[0, 0, 0], [0.01, 0, 0], [5, 5, 5]], float)
    _keys, _pc, uniq, counts = voxel_density(pts, voxel=0.1)
    assert len(uniq) == 2 and sorted(counts.tolist()) == [1, 2]


def test_low_density_zone_isolates_sparse_region():
    # a dense blob + a lone sparse cluster far away
    dense = rng.normal([0, 0, 0], 0.05, size=(2000, 3))
    sparse = rng.normal([3, 0, 0], 0.02, size=(25, 3))
    zones = low_density_zones(np.vstack([dense, sparse]), voxel=0.1,
                              low_pct=20.0, min_zone_points=15)
    assert zones, "expected at least one low-density zone"
    # the sparsest zone centroid should be near the isolated cluster
    z = zones[0]
    assert np.linalg.norm(z.centroid - np.array([3, 0, 0])) < 0.5


def test_single_blob_reports_no_zone():
    # one connected dense body -> not an under-sampled zone (degenerate guard)
    blob = rng.normal([0, 0, 0], 0.1, size=(3000, 3))
    zones = low_density_zones(blob, voxel=0.1, low_pct=20.0, min_zone_points=20)
    assert zones == []


def test_frames_viewing_projects_in_bounds():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1.0]])
    pose_map = {
        1: np.eye(4),                                   # looks down +z, sees origin-front
        2: _translate([0, 0, -5]),                      # behind, still sees +z point
        3: _translate([100, 0, 0]),                     # far to the side, out of frame
    }
    centroid = np.array([0.0, 0.0, 2.0])
    seen = frames_viewing(centroid, pose_map, lambda f: K, (640, 480))
    assert 1 in seen and 3 not in seen


def _translate(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name); n += 1
    print(f"\n{n} tests passed")
