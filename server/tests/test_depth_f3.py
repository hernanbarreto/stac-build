"""claude_stac.txt F3 — §12.7 depth by tracks: a per-frame depth
compression injected into a subset of keyframes is recovered within
tolerance from exact correspondences (tracks) triangulated with the poses;
the held-out gate leaves identity when the correspondences do not make the
multi-view depth agreement better; the correction lands as an epoch along
the rays (poses and provenance untouched, sidecar affine)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops.config import load_loops_config                       # noqa: E402
from reconstruction.witness.depth_tracks import (depth_stage, load_tracks, run_depth_stage,  # noqa: E402
                                                 Tracks)
from reconstruction.witness.frames import frames_from_arrays                    # noqa: E402
from tests.synth_metric import (make_session, write_session_dir, write_aligned_chunks,  # noqa: E402
                                synthetic_tracks, raw_server_cfg, apply_depth_affine,
                                corridor_loop_scene, loop_trajectory)
from tests.synth_correction import make_correction_cfg                          # noqa: E402

N_KF = 40
COMPRESSED = {6: (0.95, 0.02), 7: (0.95, 0.02), 8: (0.96, 0.0), 21: (0.93, 0.05), 22: (0.93, 0.05)}


def _instances(sess):
    inst, iid = {}, 1
    for p in sess.scene.prims:
        if getattr(p, "label", "") in ("wall", "column", "box"):
            inst[iid] = {"label": p.label, "oids": [p.oid]}
            iid += 1
    return inst


def _cfg(**over):
    base = {"witness.tracks.min_obs_per_frame": 10, "witness.depth.pair_samples": 3000,
            "witness.contours.enabled": False}
    base.update(over)
    return load_loops_config(raw_server_cfg(**base))


@pytest.fixture(scope="module")
def truth():
    return make_session(H=40, W=56, scene=corridor_loop_scene(), poses=loop_trajectory(N_KF, extra_laps=0.0))


def test_injected_compression_recovered_from_tracks(tmp_path, truth):
    corrupted = apply_depth_affine(truth, COMPRESSED)
    root = write_session_dir(tmp_path / "s", corrupted, _instances(truth), point_stride=1)
    write_aligned_chunks(root, corrupted, chunk_size=20, overlap=10)
    synthetic_tracks(root, truth, win=12, stride=3, per_frame=60)      # the images are the truth
    cfg = _cfg()
    frames = frames_from_arrays(corrupted.depth, corrupted.K, corrupted.poses, corrupted.frame_numbers)
    rep = depth_stage(frames, cfg.witness, load_tracks(root / "output"), None, log=lambda m: None)
    assert rep["applied"], rep["reason"]
    # most keyframes carry their own correspondence row (corners of the walk
    # share too few surfaces with their window's query frame)
    assert rep["n_absolute_rows"] >= N_KF // 2
    assert rep["n_track_obs"] >= cfg.witness.tracks.min_obs_per_frame * rep["n_absolute_rows"]
    for g, (a_inj, b_inj) in COMPRESSED.items():
        f = int(truth.frame_numbers[g])
        a, b = rep["a"][f], rep["b"][f]
        # z'' = a (a_inj z + b_inj) + b must give z back
        assert abs(a * a_inj - 1.0) < 0.01, (g, a, a_inj)
        assert abs(a * b_inj + b) < 0.02, (g, a, b, b_inj)
    # clean frames stay within twice the identity prior's width (what the
    # evidence cannot resolve — a corner frame without correspondences, the
    # last keyframe with pairs on one side only — is never pushed further by
    # the pair sensor's noise)
    clean = [g for g in range(N_KF) if g not in COMPRESSED]
    tol_a = 2.0 * cfg.witness.depth.prior_sigma_rel
    tol_b = 2.0 * cfg.witness.depth.prior_sigma_rel * cfg.witness.depth.zref_m
    for g in clean:
        f = int(truth.frame_numbers[g])
        assert abs(rep["a"][f] - 1.0) < tol_a and abs(rep["b"][f]) < tol_b, (g, rep["a"][f], rep["b"][f])
        z = truth.depth[g][truth.depth[g] > 0]
        induced = np.abs((rep["a"][f] - 1.0) * z + rep["b"][f])
        assert float(np.median(induced)) < cfg.witness.depth.prior_sigma_rel * float(np.median(z)) + tol_b / 2
    v = rep["verdict"]
    assert v["improves"] and v["bounded"]
    # the correspondence residual where the defect lives (p90: the compressed
    # frames' observations) collapses; the median is the clean majority
    assert v["correspondence_p90_after"] < 0.25 * v["correspondence_p90_before"]
    assert rep["correspondence_rel_after"] < cfg.witness.depth.pair_sigma_floor_rel


def test_holdout_gate_vetoes_correspondences_that_do_not_help(tmp_path, truth):
    """Consistent depth, but tracks that contradict half the frames: the
    solve would tear even and odd frames apart — the held-out pairs say so
    and the stage stays identity."""
    root = write_session_dir(tmp_path / "s", truth, _instances(truth), point_stride=1)
    write_aligned_chunks(root, truth, chunk_size=20, overlap=10)
    synthetic_tracks(root, truth, win=12, stride=3, per_frame=60)
    tr = load_tracks(root / "output")
    even = (tr.obs_frame // 30) % 2 == 0            # frame numbers are 30·kf
    uv = tr.obs_uv.copy()
    uv[even] = (uv[even] - np.array([truth.W / 2, truth.H / 2])) * 1.06 + np.array([truth.W / 2, truth.H / 2])
    liar = Tracks(tr.obs_track, tr.obs_frame, uv.astype(np.float32), tr.obs_weight, tr.res_h, tr.res_w)
    cfg = _cfg()
    frames = frames_from_arrays(truth.depth, truth.K, truth.poses, truth.frame_numbers)
    rep = depth_stage(frames, cfg.witness, liar, None, log=lambda m: None)
    assert not rep["applied"], rep
    assert "held-out" in rep["reason"]
    assert all(v == 1.0 for v in rep["a"].values()) and all(v == 0.0 for v in rep["b"].values())


def test_depth_epoch_moves_points_along_rays_and_keeps_provenance(tmp_path, truth):
    corrupted = apply_depth_affine(truth, COMPRESSED)
    root = write_session_dir(tmp_path / "s", corrupted, _instances(truth), point_stride=2)
    write_aligned_chunks(root, corrupted, chunk_size=20, overlap=10)
    synthetic_tracks(root, truth, win=12, stride=3, per_frame=60)
    out = root / "output"
    from correction.session import read_ply, read_poses
    _, before = read_ply(out / "cleaned_cloud.ply")
    poses_before = read_poses(out / "camera_poses.txt")
    cfg = _cfg()
    frames = frames_from_arrays(corrupted.depth, corrupted.K, corrupted.poses, corrupted.frame_numbers)
    rep = run_depth_stage(out, root, cfg, frames=frames, tracks=load_tracks(out), log=lambda m: None,
                          epoch=True, operator="test", correction_cfg=make_correction_cfg())
    assert rep["applied"] and rep["epoch"] == 1
    _, after = read_ply(out / "cleaned_cloud.ply")
    for fld in ("frame_global", "pixel_row", "pixel_col"):
        assert np.array_equal(after[fld], before[fld])
    assert np.allclose(read_poses(out / "camera_poses.txt"), poses_before)
    # the corrupted frames' points are back on the true surfaces (along their rays)
    fg = after["frame_global"]
    for g in COMPRESSED:
        f = int(truth.frame_numbers[g])
        sel = np.flatnonzero(fg == f)
        P = np.stack([after["x"][sel], after["y"][sel], after["z"][sel]], 1).astype(np.float64)
        rr, cc = before["pixel_row"][sel], before["pixel_col"][sel]
        T = truth.points[g][rr, cc]
        err = np.linalg.norm(P - T, axis=1)
        assert float(np.median(err)) < 0.02, (g, float(np.median(err)))
        # the ray direction is untouched: the corrected point stays on the pixel's ray
        cam = truth.poses[g][:3, 3]
        d0 = corrupted.points[g][rr, cc] - cam
        d1 = P - cam
        cosang = np.sum(d0 * d1, 1) / (np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1) + 1e-12)
        assert float(np.min(cosang)) > 0.999999
    side = json.loads((out / "depth_correction.json").read_text())
    assert side["version"] == 2 and set(side["k"]) == set(side["b"])
    f0 = str(int(truth.frame_numbers[6]))
    assert abs(side["k"][f0] * 0.95 - 1.0) < 0.01
    # the ledger has the epoch, and SELECTING epoch 0 shows the corrupted cloud
    # exactly — including the depth sidecar, which only exists from epoch 1 on
    from correction.run import run_select
    from correction.apply import available_epochs
    run_select(out, 0, "test")
    _, restored = read_ply(out / "cleaned_cloud.ply")
    assert np.array_equal(restored["x"], before["x"]) and not (out / "depth_correction.json").exists()
    # and the depth epoch is still there, sidecar included
    run_select(out, 1, "test")
    assert (out / "depth_correction.json").exists()
    assert [e["epoch"] for e in available_epochs(out)] == [0, 1]


def test_mask_contours_give_depth_observations_where_edges_are_sharp(tmp_path, truth):
    """§6.4 last bullet: where tracks are sparse, the SAM3 mask contours of
    an instance seen in two nearby keyframes are correspondences — a
    contour sample pushed along its ray lands on the same instance's contour
    in the other frame only at the right depth. On a compressed frame the
    observation recovers the true depth; contours without image gradient
    give nothing."""
    from reconstruction.witness.depth_tracks import contour_observations, load_images
    from reconstruction.witness.mask_votes import load_mask_store
    from tests.synth_metric import write_images
    # masks and images at four times the depth-test grid: a contour is a
    # pixel curve and its depth leverage is the projection shift per depth
    # step (≈1 px per 1 % here); pairs two and four keyframes apart give the
    # baseline. At the coarse grid the nearest-contour distance is quantised
    # to ±2 % of depth — below the injected 5 % (measured)
    hi_res = make_session(H=160, W=224, scene=corridor_loop_scene(), poses=loop_trajectory(N_KF, extra_laps=0.0))
    corrupted = apply_depth_affine(hi_res, COMPRESSED)
    inst = _instances(hi_res)
    root = write_session_dir(tmp_path / "s", corrupted, inst, point_stride=1)
    write_images(root, hi_res)
    out = root / "output"
    cfg = _cfg(**{"witness.contours.enabled": True, "witness.contours.samples_per_instance": 60})
    frames = frames_from_arrays(corrupted.depth, corrupted.K, corrupted.poses, corrupted.frame_numbers)
    instances = json.loads((out / "segmentation_result.json").read_text())["instances"]
    store = load_mask_store(out)
    nums = sorted(frames)
    pairs = [(nums[i], nums[i + d]) for d in (2, 4) for i in range(len(nums) - d)]
    images = load_images(root, nums)
    assert len(images) == N_KF
    truth = hi_res
    obs = contour_observations(frames, instances, store, images, cfg.witness.contours, pairs)
    assert len(obs) > 50, len(obs)
    by = {}
    for f, zo, zt, w in obs:
        by.setdefault(int(f), []).append(zt / zo)
    # a compressed frame's contours ask for the inverse compression; clean frames for identity
    for g, (a_inj, b_inj) in COMPRESSED.items():
        f = int(truth.frame_numbers[g])
        if f in by and len(by[f]) >= 5:
            ratio = float(np.median(by[f]))
            assert abs(ratio * a_inj - 1.0) < 0.03, (g, ratio)
    clean_ratios = [r for g in range(N_KF) if g not in COMPRESSED for r in by.get(int(truth.frame_numbers[g]), [])]
    assert clean_ratios and abs(float(np.median(clean_ratios)) - 1.0) < 0.01
    # no gradient → no observation (a flat image says nothing about where the contour is)
    flat = {f: np.zeros_like(img) for f, img in images.items()}
    assert contour_observations(frames, instances, store, flat, cfg.witness.contours, pairs) == []
