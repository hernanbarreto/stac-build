"""The certify mask filter on a RENDERED scene, through the real Visibility (audit
2026-10-01, object-edge definition: items #2 and #3, USER decision (2)).

Scene: a plate (object A, "monitor") 10 cm in front of a wall (object B) — a contact
crease all around its rim — seen by seven keyframes 20 cm apart. The images (and so the
SAM3 masks) come from the SESSION camera (camera.json, F5); intrinsic.txt holds Omega's
per-keyframe record of ANOTHER camera, as on pccr (fx 363 vs 392 there).

  * the filter projects with camera.json — never intrinsic.txt — and refuses a cloud
    the session camera did not build;
  * the wall behind the plate's rim stays (it is occluded behind it, not ON it);
  * a skirt of A born in one keyframe at a mixed depth (1.48 m, on the wall's surface)
    through A's leaky mask leaves: from the keyframes that saw it, it lies on the wall,
    inside the wall's mask, in the majority of them.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from correction import visit_drift as VD

W, H = 200, 160
K_SESSION = (200.0, 198.0, 100.4, 79.6)      # camera.json — the camera of the images
K_OMEGA = (186.0, 185.0, 100.0, 80.0)        # intrinsic.txt — Omega's record, another camera
CAMS_X = (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)
FRAMES = [10 * (k + 1) for k in range(len(CAMS_X))]
Z_PLATE, Z_WALL, Z_SKIRT = 1.4, 1.5, 1.48
HALF_PLATE, WALL_X, WALL_Y = 0.2, 1.5, 1.0
SKIRT_KF = 3                                  # the keyframe whose A mask leaks one column

MIN_DEPTH = 0.05
DEPTH_TOL_M = 0.15                            # rule 1's (unchanged) occlusion tolerance
TOL_REL, MIN_VOTES, MIN_FRAC, MIN_TRI, DILATE = 0.05, 2, 0.5, 2.0, 2


def _poses():
    P = np.tile(np.eye(4), (len(CAMS_X), 1, 1))
    P[:, 0, 3] = CAMS_X
    return P


def _render(k):
    """(depth, label) on the native grid seen by the SESSION camera from keyframe k:
    label 0 = plate (A), 1 = wall (B), -1 = nothing; the skirt column in SKIRT_KF."""
    fx, fy, cx, cy = K_SESSION
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = (u - cx) / fx, (v - cy) / fy
    c = CAMS_X[k]
    plate = (np.abs(c + dx * Z_PLATE) <= HALF_PLATE) & (np.abs(dy * Z_PLATE) <= HALF_PLATE)
    wall = ~plate & (np.abs(c + dx * Z_WALL) <= WALL_X) & (np.abs(dy * Z_WALL) <= WALL_Y)
    depth = np.where(plate, Z_PLATE, np.where(wall, Z_WALL, 0.0))
    label = np.where(plate, 0, np.where(wall, 1, -1))
    skirt = np.zeros((H, W), bool)
    if k == SKIRT_KF:
        # the first wall column right of the plate, on the plate's rows: a mixed
        # pixel (depth on the wall's surface) that A's mask leaks over
        rows = np.flatnonzero(plate.any(1))
        col = int(np.flatnonzero(plate[rows[0]]).max()) + 1
        skirt[rows, col] = True
        depth = np.where(skirt, Z_SKIRT, depth)
        label = np.where(skirt, 0, label)
    return depth, label, skirt


def _session(tmp_path, build_K=K_SESSION, with_camera_json=True):
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    out = tmp_path / "output"
    out.mkdir(parents=True)
    poses = _poses()
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in FRAMES) + "\n")
    (out / "camera_poses.txt").write_text("\n".join(
        " ".join(f"{x:.10g}" for x in p.ravel()) for p in poses) + "\n")
    (out / "intrinsic.txt").write_text("\n".join(
        " ".join(str(x) for x in K_OMEGA) for _ in FRAMES) + "\n")
    if with_camera_json:
        g = grid_full_frame_resize(W, H, W, H, "omega")
        cam = CameraModel(width=W, height=H, params=(*K_SESSION, 0.0, 0.0, 0.0, 0.0),
                          source="refine", camera_epoch=1, omega_grid=g,
                          mask_grid=grid_full_frame_resize(W, H, W, H, "mask"))
        save_camera_json(out / "camera.json", cam, geometry_epoch=1)
    masks = {"mask_frame_space": np.array("keyframe_position")}
    xyz, fg, pr, pc, ks, skirt_pts, label_pts = [], [], [], [], [], [], []
    fx, fy, cx, cy = build_K
    for k in range(len(CAMS_X)):
        depth, label, skirt = _render(k)
        masks[f"f{k}_o0"] = (label == 0).astype(np.uint8)
        masks[f"f{k}_o1"] = (label == 1).astype(np.uint8)
        v, u = np.nonzero(depth > 0)
        z = depth[v, u]
        P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1) + poses[k][:3, 3]
        xyz.append(P)
        fg.append(np.full(len(u), FRAMES[k]))
        pr.append(v)
        pc.append(u)
        ks.append(np.full(len(u), k))
        skirt_pts.append(skirt[v, u])
        label_pts.append(label[v, u])
    np.savez_compressed(out / "seg_masks.npz", **masks)
    (out / "segmentation.json").write_text(json.dumps({
        "instances": [{"id": 0, "instance_id": 1, "label": "monitor"},
                      {"id": 1, "instance_id": 2, "label": "wall"}],
        "mask_file": "seg_masks.npz"}))
    cat = lambda a: np.concatenate(a)                                   # noqa: E731
    return (out, cat(xyz).astype(np.float64), cat(fg).astype(np.int64), cat(pr).astype(np.int64),
            cat(pc).astype(np.int64), cat(ks).astype(np.int64), cat(skirt_pts), cat(label_pts), poses)


def _filter(out, xyz, fg, pr, pc, ks, poses, tol_rel=TOL_REL):
    cam = VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=MIN_DEPTH,
                               log=lambda m: None)
    vis = VD.Visibility(out, xyz, ks, poses, cam, DEPTH_TOL_M, min_depth_m=MIN_DEPTH)
    masklets = VD.masklet_visits(out, log=lambda m: None)
    pm = VD.points_of_masklets(out, fg, pr, pc, log=lambda m: None,
                               mask_pixels=cam.record_to_mask(pr, pc))
    return VD.cloud_filter_masklets(
        pm, masklets, ks, xyz, vis, 1, 0.0, 8, DILATE, occlusion_tol_rel=tol_rel,
        min_votes=MIN_VOTES, min_inside_frac=MIN_FRAC, min_tri_deg=MIN_TRI,
        log=lambda m: None, group_points=None, group_roots={0: 1, 1: 2})


def test_the_crease_stays_and_the_skirt_on_the_wall_leaves(tmp_path):
    out, xyz, fg, pr, pc, ks, skirt, label, poses = _session(tmp_path)
    assert skirt.sum() > 20
    kill, rep = _filter(out, xyz, fg, pr, pc, ks, poses)
    # the wall at the contact crease: wall points within 6 cm of the plate's rim, born
    # where they are visible — from the keyframes where they hide behind the plate
    # (10 cm in front of them: beyond the relative tolerance) they are OCCLUDED, never
    # "on the plate"
    crease = (label == 1) & (np.abs(xyz[:, 2] - Z_WALL) < 1e-9) & \
        (np.abs(xyz[:, 1]) <= HALF_PLATE) & (np.abs(xyz[:, 0]) > HALF_PLATE) & \
        (np.abs(xyz[:, 0]) < HALF_PLATE + 0.06)
    assert crease.sum() > 100
    assert not kill[crease].any(), f"{int(kill[crease].sum())} crease point(s) removed"
    # the skirt: on the wall's surface, inside the wall's mask, in the majority of the
    # keyframes that saw it — and nothing else leaves
    assert kill[skirt].all(), f"{int((~kill[skirt]).sum())} skirt point(s) kept"
    assert np.array_equal(kill, skirt), f"{int((kill & ~skirt).sum())} other point(s) removed"


def test_a_tolerance_that_spans_the_gap_cuts_the_crease(tmp_path):
    """Why (b) is the measured surface at the RELATIVE tolerance: a tolerance wide
    enough to span the plate's 10-cm gap (what the absolute 15 cm of the first version
    did at this range) puts the wall behind the rim 'on the plate' and cuts the crease."""
    out, xyz, fg, pr, pc, ks, skirt, label, poses = _session(tmp_path)
    kill, _ = _filter(out, xyz, fg, pr, pc, ks, poses, tol_rel=0.1)
    crease = (label == 1) & (np.abs(xyz[:, 1]) <= HALF_PLATE) & \
        (np.abs(xyz[:, 0]) > HALF_PLATE) & (np.abs(xyz[:, 0]) < HALF_PLATE + 0.06)
    assert kill[crease].any()


def test_the_filter_projects_with_the_session_camera(tmp_path):
    out, xyz, fg, pr, pc, ks, *_rest, poses = _session(tmp_path)
    cam = VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=MIN_DEPTH,
                               log=lambda m: None)
    vis = VD.Visibility(out, xyz, ks, poses, cam, DEPTH_TOL_M, min_depth_m=MIN_DEPTH)
    # a point in front of keyframe 3, off-centre (where the two cameras disagree most)
    P = np.array([[0.35, 0.25, 1.5]])
    ok, r, c, z = vis.project(P, 3)
    q = P[0] - poses[3][:3, 3]
    fx, fy, cx, cy = K_SESSION
    assert ok[0] and (r[0], c[0]) == (round(fy * q[1] / q[2] + cy), round(fx * q[0] / q[2] + cx))
    fo, fyo, cxo, cyo = K_OMEGA
    assert (r[0], c[0]) != (round(fyo * q[1] / q[2] + cyo), round(fo * q[0] / q[2] + cxo))
    # the z-buffer is built through the same camera: the keyframe's own points land
    # on their own birth pixels (and the 5-px minimum filter only lowers it)
    own = ks == 0
    ok, r, c, z = vis.project(xyz[own], 0)
    assert ok.all() and np.array_equal(r, pr[own]) and np.array_equal(c, pc[own])
    zb = vis.zbuf(0)
    assert np.all(zb[pr[own], pc[own]] <= xyz[own, 2] + 1e-12)


def test_a_cloud_built_with_another_camera_is_refused(tmp_path):
    out, xyz, fg, pr, pc, ks, *_rest, poses = _session(tmp_path, build_K=K_OMEGA)
    with pytest.raises(VD.CameraMismatchError, match="does not reproduce") as e:
        VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=MIN_DEPTH,
                             log=lambda m: None)
    assert "intrinsic.txt" in str(e.value) and "built with Omega's camera" in str(e.value)


def test_no_session_camera_is_refused(tmp_path):
    out, xyz, fg, pr, pc, ks, *_rest, poses = _session(tmp_path, with_camera_json=False)
    with pytest.raises(VD.CameraMismatchError, match="does not exist"):
        VD.projection_camera(out, xyz, ks, poses, pr, pc, min_depth_m=MIN_DEPTH,
                             log=lambda m: None)


def test_poses_of_another_cloud_are_refused(tmp_path):
    out, xyz, fg, pr, pc, ks, *_rest, poses = _session(tmp_path)
    moved = poses.copy()
    moved[:, 0, 3] += 0.05                       # every camera 5 cm off the cloud's
    with pytest.raises(VD.CameraMismatchError, match="does not reproduce"):
        VD.projection_camera(out, xyz, ks, moved, pr, pc, min_depth_m=MIN_DEPTH,
                             log=lambda m: None)


def test_scatter_of_the_consolidation_is_not_a_camera_mismatch(tmp_path):
    """Zero-mean motion of the points (what the MLS consolidation leaves) moves the
    birth pixels by more than the rounding — and is not a systematic misprojection."""
    out, xyz, fg, pr, pc, ks, *_rest, poses = _session(tmp_path)
    rng = np.random.default_rng(0)
    jitter = xyz + rng.normal(0.0, 0.01, xyz.shape)          # 1 cm ≈ 1.4 px at 1.5 m
    VD.projection_camera(out, jitter, ks, poses, pr, pc, min_depth_m=MIN_DEPTH,
                         log=lambda m: None)


def test_visibility_refuses_a_per_keyframe_K(tmp_path):
    out, xyz, fg, pr, pc, ks, *_rest, poses = _session(tmp_path)
    with pytest.raises(TypeError, match="projection_camera"):
        VD.Visibility(out, xyz, ks, poses, np.tile(K_OMEGA, (len(FRAMES), 1)), DEPTH_TOL_M,
                      min_depth_m=MIN_DEPTH)


def test_the_certify_step_wires_the_camera_and_the_declared_votes(tmp_path):
    """visit_drift_run.filter_staged_cloud — the step itself, with the parameters
    read from config.yaml (silhouette_min_votes / _min_inside_frac, occlusion_tol_rel,
    min_tri_deg, dilate_px: the values this file's scene was drawn for) — removes the
    skirt and nothing else, and writes the staged cloud without it."""
    from types import SimpleNamespace

    from correction import visit_drift_run as VR
    from correction.session import read_ply
    out, xyz, fg, pr, pc, ks, skirt, label, poses = _session(tmp_path)
    o4 = VR._other_mask_params()
    assert (o4.min_votes, o4.min_inside_frac, o4.occlusion_tol_rel, o4.min_tri_deg,
            o4.dilate_px) == (MIN_VOTES, MIN_FRAC, TOL_REL, MIN_TRI, DILATE)
    data = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("frame_global", "<i4"), ("pixel_row", "<i2"),
                                     ("pixel_col", "<i2")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["frame_global"], data["pixel_row"], data["pixel_col"] = fg, pr, pc
    header = [b"ply\n", b"format binary_little_endian 1.0\n", b"element vertex 0\n",
              b"property float x\n", b"property float y\n", b"property float z\n",
              b"property int frame_global\n", b"property short pixel_row\n",
              b"property short pixel_col\n", b"end_header\n"]
    session = SimpleNamespace(output_dir=out, ks=ks, header=header, raw_header=None)
    cfg = SimpleNamespace(visit_drift=SimpleNamespace(min_points=1, min_visit_share=0.0))
    tx = tmp_path / "tx"
    tx.mkdir()
    xyz32 = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    res = VR.filter_staged_cloud(tx, session, data, xyz32, poses, None, cfg,
                                 log=lambda m: None)
    assert res is not None
    rep, keep = res
    assert np.array_equal(~keep, skirt), int((~keep & ~skirt).sum())
    _h, staged = read_ply(tx / "cleaned_cloud.ply")
    assert len(staged) == int(keep.sum())
