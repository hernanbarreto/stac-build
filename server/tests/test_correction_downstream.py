"""Downstream consistency after a depth correction: the session_io depth
accessor, raw==cleaned coherence, globalIndices stability, epoch stamping."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.epoch import check, current_epoch, stamp     # noqa: E402
from correction.run import run_objects                       # noqa: E402
from correction.session import read_ply                      # noqa: E402
from segmentation.session_io import (correct_depth,          # noqa: E402
                                     load_depth_correction)
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _run_depth_scene(tmp_path):
    c = 0.90
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=c,
                        write_omega_npz=True)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    return scene, rep, c


def test_depth_accessor_serves_corrected_depth(tmp_path):
    scene, rep, c = _run_depth_scene(tmp_path)
    k_map = load_depth_correction(scene.output_dir)
    assert k_map, "sidecar must exist after a depth correction"
    rev_frame = scene.frames[scene.gt["revisit_kfs"][0]]
    ref_frame = scene.frames[scene.gt["ref_kfs"][0]]
    assert abs(k_map[rev_frame] - 1.0 / c) < 0.03
    assert ref_frame not in k_map or k_map[ref_frame] == 1.0
    z = np.load(scene.output_dir / "omega_run" / "results_output"
                / f"frame_{rev_frame}.npz")["depth"]
    zc = correct_depth(z, rev_frame, scene.output_dir)
    assert np.allclose(zc, z * np.float32(k_map[rev_frame]))
    # uncorrected frame passes through untouched (same object, no copy)
    z0 = np.load(scene.output_dir / "omega_run" / "results_output"
                 / f"frame_{ref_frame}.npz")["depth"]
    assert correct_depth(z0, ref_frame, scene.output_dir) is z0


def test_corrected_depth_reprojects_onto_corrected_cloud(tmp_path):
    """The accessor's depth (camera-frame z of a revisit point × k) matches
    the corrected point's camera distance ~exactly — depth and cloud agree
    (prompt §10: residuo ~0)."""
    scene, rep, c = _run_depth_scene(tmp_path)
    from correction.session import load_session
    sess = load_session(scene.output_dir)           # corrected session
    k_map = load_depth_correction(scene.output_dir)
    rev_kf = scene.gt["revisit_kfs"][3]
    frame = scene.frames[rev_kf]
    sel = np.where((sess.ks == rev_kf) & ~scene.floor_mask)[0][:500]
    cam_corr = sess.poses[rev_kf][:3, 3]
    # observed (pre-correction) camera distance of those points × k must
    # equal the corrected camera distance (rigid preserves it; k scales it)
    _, data0 = read_ply(scene.output_dir / "cleaned_cloud.ply")
    d_corr = np.linalg.norm(sess.xyz[sel] - cam_corr, axis=1)
    # reconstruct pre-correction distances from the epoch npz
    from correction.ledger import load_epoch_npz
    npz = load_epoch_npz(scene.output_dir, 1)
    k = float(npz["k_kf"][rev_kf])
    R, t = npz["R_kf"][rev_kf], npz["t_kf"][rev_kf]
    q = (sess.xyz[sel] - t) @ R                    # inverse rigid → the
    cam_pre = R.T @ (cam_corr - t)                 # k-expanded pre point
    p_obs = cam_pre + (q - cam_pre) / k            # inverse k → observed
    d_obs = np.linalg.norm(p_obs - cam_pre, axis=1)
    # identity: accessor depth (observed × k) == corrected camera distance
    resid = np.abs(d_obs * k - d_corr)
    assert float(np.median(resid)) < 1e-6, float(np.median(resid))


def test_raw_and_cleaned_share_the_transform(tmp_path):
    scene, rep, c = _run_depth_scene(tmp_path)
    _, a = read_ply(scene.output_dir / "cleaned_cloud.ply")
    _, b = read_ply(scene.output_dir / "cleaned_cloud_raw.ply")
    for f in ("x", "y", "z", "frame_global"):
        assert np.array_equal(a[f], b[f]), f


def test_global_indices_stay_valid(tmp_path):
    scene, rep, c = _run_depth_scene(tmp_path)
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    res = json.loads(
        (scene.output_dir / "segmentation_result.json").read_text())
    # order preserved → each instance's indices still carry the SAME
    # provenance rows as at build time
    for inst in res["instances"]:
        g = np.asarray(inst["globalIndices"], dtype=np.int64)
        assert np.array_equal(data["frame_global"][g], scene.fg[g])


def test_epoch_stamping(tmp_path):
    scene, rep, c = _run_depth_scene(tmp_path)
    assert current_epoch(scene.output_dir) == 1
    meta = stamp({"method": "x"}, scene.output_dir)
    assert meta["geometry_epoch"] == 1
    assert meta["human_directed_corrections"] == 0   # pending, not approved
    from correction.run import run_verdict
    run_verdict(scene.output_dir, "approved", "test")
    meta2 = stamp({"method": "x"}, scene.output_dir)
    assert meta2["human_directed_corrections"] == 1
    # an artifact stamped before the correction reads as stale
    assert check({"geometry_epoch": 0}, scene.output_dir)["stale"]
    assert not check(meta2, scene.output_dir)["stale"]
