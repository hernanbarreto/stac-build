"""THE RANGE of every view (USER 2026-11-09): ONE key, reconstruction.view_range_m, read by every stage — a point
farther than it from the camera that measured it is born by no keyframe and weighs in no fit, pairing, scale, closure,
co-visibility or statistic. Each piece is checked on a scene with something near and something far. Seconds."""
import json

import numpy as np
import pytest

import view_range as VR


def test_one_key_read_by_every_stage_and_the_old_key_refused():
    from precision.config import PrecisionConfigError, load_precision_config
    assert VR.view_range_m({"reconstruction": {"view_range_m": 12}}) == 12.0
    with pytest.raises(KeyError, match="reconstruction.view_range_m"):
        VR.view_range_m({"reconstruction": {}})
    with pytest.raises(ValueError):
        VR.view_range_m({"reconstruction": {"view_range_m": 0}})
    c = load_precision_config()
    assert c.view_range_m == VR.view_range_m() > 0
    raw = dict(c.raw)
    raw["reconstruction"] = dict(raw["reconstruction"], view_range_m=15.0)
    assert load_precision_config(raw).view_range_m == 15.0, "change it in the config, every stage follows"
    prec = dict(raw["reconstruction"]["precision"])
    prec["cloud"] = dict(prec["cloud"], fusion_witness_max_m=10.0)
    raw["reconstruction"] = dict(raw["reconstruction"], precision=prec)
    with pytest.raises(PrecisionConfigError, match="fusion_witness_max_m"):
        load_precision_config(raw)


def test_the_distance_is_measured_along_the_ray():
    from precision.camera import depth_map_distance, pixel_distance
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    assert pixel_distance(50.0, 40.0, 8.0, K) == pytest.approx(8.0)                      # the principal point
    assert pixel_distance(150.0, 40.0, 8.0, K) == pytest.approx(8.0 * np.sqrt(2.0))      # 45° off axis
    d = depth_map_distance(np.full((80, 100), 8.0), K)
    assert d[40, 50] == pytest.approx(8.0) and d.max() > 8.0


def test_f5_leaves_out_the_observations_of_a_far_landmark():
    """precision.refine.within_range: an observation of a landmark beyond the range of its keyframe is out."""
    from precision.refine import within_range
    from precision.camera import undistort_solver
    from precision.config import load_precision_config
    fx = 400.0
    params = [fx, fx, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]
    C = [np.array([0.0, 0, 0]), np.array([1.0, 0, 0])]
    w2c = np.stack([np.linalg.inv(np.c_[np.r_[np.eye(3), [[0, 0, 0]]], np.r_[c, 1.0]]) for c in C])
    X = {1: np.array([0.3, 0.1, 4.0]), 2: np.array([0.5, -0.2, 25.0])}        # near / far
    track, frame, uv = [], [], []
    for t, x in X.items():
        for i, f in enumerate((10, 20)):
            xc = w2c[i][:3, :3] @ x + w2c[i][:3, 3]
            track.append(t); frame.append(f); uv.append([fx * xc[0] / xc[2] + 320.0, fx * xc[1] / xc[2] + 240.0])
    keep = within_range(np.array(track), np.array(frame), np.array(uv), [10, 20], w2c, params,
                        undistort_solver(load_precision_config().camera), 0.1, 10.0)
    assert keep.tolist() == [True, True, False, False]


def test_the_amb3r_map_holds_no_point_beyond_the_range():
    """reconstruction.coverage_fill.CoverageMap (the fill, the cut, the co-visibility, the revisits): a far pixel is not
    in the map."""
    torch = pytest.importorskip("torch")
    from reconstruction.coverage_fill import CoverageMap
    D = np.full((2, 6, 8), 5.0, np.float32)
    D[:, :, 4:] = 30.0                                                    # the right half of every frame is far
    K = np.tile(np.array([[8.0, 0, 4.0], [0, 8.0, 3.0], [0, 0, 1.0]]), (2, 1, 1))
    video = {"rec_depth": D, "rec_K": K, "rec_c2w": np.tile(np.eye(4), (2, 1, 1)), "stride": 1}
    whole = CoverageMap(video, "cpu", state=False)
    near = CoverageMap(video, "cpu", state=False, range_m=10.0)
    assert whole.N == 2 * 6 * 8 and near.N == 2 * 6 * 4
    assert float(torch.linalg.norm(near.X, dim=1).max()) <= 10.0


def test_the_da3_anchor_hears_only_pixels_within_the_range():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"))
    from loop_utils.metric_lock import anchor_ratio
    om = np.full((40, 60), 2.0, np.float32)
    da = np.full((40, 60), 4.0, np.float32)                               # near: DA3 / Omega = 2
    om[:, 30:] = 7.0
    da[:, 30:] = 21.0                                                     # far (21 m): ratio 3
    K = np.array([[60.0, 0, 30.0], [0, 60.0, 20.0], [0, 0, 1.0]])
    assert anchor_ratio(om, da, near_frac=1.0, K=K, range_m=10.0) == pytest.approx(2.0)
    assert anchor_ratio(om, da, near_frac=1.0) > 2.0, "every pixel: the far ones vote too"


def test_the_floor_is_found_on_points_within_the_range(tmp_path):
    from precision.epoch0_cloud import _write_ply_xyzrgb
    from reconstruction.orient import _ply_y_percentile
    near = np.c_[np.zeros(500), np.full(500, -1.5), np.linspace(1, 5, 500)]      # the floor, 1.5 m below
    far = np.c_[np.zeros(500), np.full(500, -4.0), np.linspace(15, 25, 500)]     # a far, wrong floor
    xyz = np.concatenate([near, far]).astype(np.float32)
    _write_ply_xyzrgb(tmp_path / "chunk_000.ply", xyz, np.zeros((len(xyz), 3), np.uint8))
    np.savez(tmp_path / "chunk_000_origins.npz", frame_global=np.zeros(len(xyz), np.int32))
    centres = (np.array([0], np.int64), np.zeros((1, 3)))
    y_all = _ply_y_percentile(tmp_path / "chunk_000.ply", np.eye(3), 1.0)
    y_near = _ply_y_percentile(tmp_path / "chunk_000.ply", np.eye(3), 1.0, centres, 10.0)
    assert y_all.min() == pytest.approx(-4.0) and y_near.min() == pytest.approx(-1.5) and len(y_near) == 500


def test_amb3r_map_gets_da3_metre():
    """USER 2026-11-09 ("amber con omega escalado con da3"): AMB3R's units against DA3's metric depth on the focal
    probe's frames — a map at 1/10 of the metre comes out ×10."""
    from reconstruction.amb3r_probe import metric_scale
    n, H, W = 10, 12, 16
    K = np.array([[16.0, 0, 8.0], [0, 16.0, 6.0], [0, 0, 1.0]])
    pred = np.tile(np.eye(4), (n, 1, 1)); pred[:, 0, 3] = np.arange(n) * 0.01         # a 0.09-unit walk
    video = {"frame_no": np.arange(n), "rec_index": np.arange(n), "pred": pred, "rec_c2w": pred.copy(),
             "rec_K": np.tile(K, (n, 1, 1)), "rec_depth": np.full((n, H, W), 0.5, np.float32), "stride": 1}
    focal = {"frames": np.array([0, 3, 6, 9]), "depth": np.full((4, 2 * H, 2 * W), 5.0),
             "K": np.tile(np.array([[32.0, 0, 16.0], [0, 32.0, 12.0], [0, 0, 1.0]]), (4, 1, 1))}
    assert metric_scale(video, focal, min_px=10)["scale"] == pytest.approx(10.0)
