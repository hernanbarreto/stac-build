"""Phase B (precision task) — multi-view consistency filter, synthetic tests.

Cameras on a line looking at a fronto-parallel plane: consistent depth passes,
injected outliers are dropped, the median variant pulls survivors toward the
consensus, and mutually inconsistent frames trip the pose/scale warning."""
import json

import numpy as np

from reconstruction import mv_consistency as mvc

H = W = 64
FX = 50.0


def _K():
    return np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])


def _cam(x_off):
    T = np.eye(4)
    T[0, 3] = x_off
    return T


def _frames(depths_by_frame, spacing=0.05):
    return {10 * i: {"depth": np.full((H, W), d, np.float32),
                     "K": _K(), "T": _cam(spacing * i)}
            for i, d in enumerate(depths_by_frame)}


def test_consistent_plane_passes(tmp_path):
    frames = _frames([2.0] * 6)
    rep = mvc.run(tmp_path, _frames_override=frames, device="cpu")
    assert rep["global_discard_pct"] < 5.0            # borders may lose votes
    assert rep["pose_scale_warning"] is False
    npz = np.load(tmp_path / mvc.MV_DIRNAME / "frame_0.npz")
    inner = npz["valid"][8:-8, 8:-8]
    assert inner.all()
    assert npz["votes"][8:-8, 8:-8].min() >= 2


def test_outlier_pixels_dropped(tmp_path):
    frames = _frames([2.0] * 6)
    bad = frames[10]["depth"].copy()
    bad[20:30, 20:30] = 2.5                            # 25% off — way past tau 2%
    frames[10]["depth"] = bad
    rep = mvc.run(tmp_path, _frames_override=frames, device="cpu")
    npz = np.load(tmp_path / mvc.MV_DIRNAME / "frame_10.npz")
    assert not npz["valid"][22:28, 22:28].any()        # outlier block killed
    assert npz["valid"][40:50, 40:50].all()            # clean region untouched
    assert rep["frames"]["10"]["dropped_px"] >= 100


def test_median_variant_pulls_to_consensus(tmp_path):
    frames = _frames([2.0] * 6)
    off = frames[10]["depth"].copy()
    off[:] = 2.02                                      # +1% — inside tau, passes
    frames[10]["depth"] = off
    mvc.run(tmp_path, _frames_override=frames, replace_median=True, device="cpu")
    npz = np.load(tmp_path / mvc.MV_DIRNAME / "frame_10.npz")
    med = npz["depth"][16:-16, 16:-16]
    assert np.all(np.abs(med - 2.0) < 0.02 - 1e-6)     # closer to consensus than 2.02
    # mask-only variant stores no depth key
    mvc.run(tmp_path / "b", _frames_override=frames, device="cpu")
    npz2 = np.load(tmp_path / "b" / mvc.MV_DIRNAME / "frame_10.npz")
    assert "depth" not in npz2.files


def test_incoherent_frames_trip_warning(tmp_path):
    frames = _frames([2.0, 2.5, 3.0, 3.5, 4.0, 4.5])   # nothing agrees with anything
    rep = mvc.run(tmp_path, _frames_override=frames, device="cpu")
    assert rep["global_discard_pct"] > 90.0
    assert rep["pose_scale_warning"] is True


def test_resume_skips_with_same_params(tmp_path):
    frames = _frames([2.0] * 6)
    r1 = mvc.run(tmp_path, _frames_override=frames, device="cpu")
    # identical params → no recompute (frames override absent would fail loudly,
    # proving run() short-circuits BEFORE loading anything)
    r2 = mvc.run(tmp_path, _frames_override=None)
    assert r2["global_discard_pct"] == r1["global_discard_pct"]
    # changed params → must recompute (and with no override, loading fails loudly)
    try:
        mvc.run(tmp_path, tau_rel=0.05, _frames_override=frames, device="cpu")
    except RuntimeError:
        raise AssertionError("param change must recompute, not fail")
    rep = json.loads((tmp_path / mvc.MV_DIRNAME / mvc.REPORT_NAME).read_text())
    assert rep["params"]["tau_rel"] == 0.05


def test_neighbor_selection_is_spatial():
    centres = np.array([[0, 0, 0], [10, 0, 0], [0.1, 0, 0], [20, 0, 0]], float)
    nbrs = mvc._neighbor_indices(centres, 2)
    assert 2 in nbrs[0] and 1 not in nbrs[0][:1]       # nearest to cam0 is cam2
