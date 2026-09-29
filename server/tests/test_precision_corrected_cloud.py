"""precision/corrected_cloud.py — the corrected Omega cloud as the product: the camera
on Omega's record grid, the chain the product selects, the publication as a new-cloud
epoch the product helper recognises (no GPU: the publication is exercised with the
octree stubbed, the recipe's GPU steps are the cloud stage's own, tested there)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from precision import corrected_cloud as CC                      # noqa: E402
from precision import product as PR                              # noqa: E402


def test_camera_on_the_record_grid_identity_and_mapped():
    from precision.camera import GridMap
    K = np.array([[400.0, 0, 232.0], [0, 402.0, 416.0], [0, 0, 1]])
    g0 = GridMap(name="omega", w=464, h=832, content_w=464, content_h=832, pad_left=0, pad_top=0,
                 crop_x=0, crop_y=0, crop_w=464, crop_h=832, native_w=464, native_h=832)
    assert np.allclose(CC.K_on_grid(K, g0), K)
    # a half-resolution grid of a crop: 2 native pixels per grid pixel, padded
    g1 = replace(g0, w=232, h=416, content_w=232, content_h=416, crop_x=10, crop_y=20, pad_left=3, pad_top=4)
    assert g1.scale_x == 2.0
    Kg = CC.K_on_grid(K, g1)
    assert Kg[0, 0] == 200.0 and Kg[1, 1] == 201.0
    assert Kg[0, 2] == (232.0 - 10) / 2 + 3 and Kg[1, 2] == (416.0 - 20) / 2 + 4


def test_the_product_selects_the_chain():
    from precision import runner as RN
    from precision.config import load_precision_config
    p = load_precision_config()
    corrected = [s.key for s in RN.chain_steps(replace(p, cloud=replace(p.cloud, source="omega_corrected")))]
    fusion = [s.key for s in RN.chain_steps(replace(p, cloud=replace(p.cloud, source="fusion")))]
    # the corrected cloud is built on F6's depth: the sweep runs first, never the COLMAP
    # reference nor the witness fusion (USER 2026-09-29)
    assert corrected == ["f0_camera", "f2_gauge", "f4_tracks", "f3_probe", "f5_refine", "f6_sweep",
                         "f7_cloud", "f6_check"], corrected
    assert "f7_cloud" not in fusion and {"f6_sweep", "f7_fuse"} <= set(fusion)
    assert fusion.index("f6_sweep") < fusion.index("f7_fuse") < fusion.index("f6_check")
    assert corrected[-1] == fusion[-1] == "f6_check"
    p_colmap = replace(p, depth=replace(p.depth, colmap=replace(p.depth.colmap, enabled=True)))
    assert "f6_colmap" not in [s.key for s in RN.chain_steps(
        replace(p_colmap, cloud=replace(p.cloud, source="omega_corrected")))]


def test_config_rejects_an_unknown_source():
    import copy
    from precision.config import PrecisionConfigError, load_precision_config
    from config import cfg as raw
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["cloud"]["source"] = "magic"
    with pytest.raises(PrecisionConfigError, match="cloud.source"):
        load_precision_config(bad)


def _ply(path: Path, n: int) -> None:
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((n, 3)).astype(np.float32)
    rgb = rng.integers(0, 255, (n, 3)).astype(np.uint8)
    CC_ = __import__("precision.epoch0_cloud", fromlist=["x"])
    CC_._write_ply_xyzrgb(path, xyz, rgb)


def test_publish_is_a_new_cloud_epoch_the_product_helper_recognises(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    _ply(out / "cleaned_cloud.ply", 20)                     # the session's previous cloud (epoch 0)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    (out / "segmentation_result.json").write_text(json.dumps({"instances": [{"id": 1}]}))
    tmp = out / CC.TX_TMP
    tmp.mkdir()
    _ply(tmp / "cleaned_cloud.ply", 30)

    def fake_potree(session_dir, force=False, ply_override=None, potree_dir_override=None):
        Path(potree_dir_override).mkdir(parents=True)
        (Path(potree_dir_override) / "metadata.json").write_text(json.dumps({"points": 30}))
        return True
    import potree_converter
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree", fake_potree)
    assert PR.product_report(out) is None
    rep = CC.publish(tmp_path, tmp, {"stage": "corrected_cloud", "per_frame": {}}, log=lambda m: None)
    assert rep["epoch_from"] == 0 and rep["epoch_to"] == 1 and rep["n_points"] == 30
    assert json.loads((out / "geometry_epoch.json").read_text())["epoch"] == 1
    from correction.epoch import epoch_kind
    assert epoch_kind(out, 1) == "new_cloud"
    assert (out / "_epoch_0" / "cleaned_cloud.ply").exists()          # the previous cloud stays selectable
    (out / CC.CLOUD_REPORT).write_text(json.dumps(rep, default=float))
    live, why = PR.product_is_live(out)
    assert live and "corrected_cloud" in why
    seg = json.loads((out / "segmentation_result.json").read_text())
    assert seg["instances"] == [] and "pending" in seg                 # the cloud stage projects on it
    # the fusion's report, if an older one were present, does not outrank the live product
    (out / "fuse_report.json").write_text(json.dumps({"epoch_to": 0}))
    assert PR.product_report(out)["product_file"] == CC.CLOUD_REPORT


# ── the corrected cloud on F6's depth: grid + metric ─────────────────────
#
# A slanted plane z = 3 + 0.1·x seen by three keyframes; the session camera has LENS
# DISTORTION and Omega's record grid is HALF the native resolution — F6's depth lives on
# the undistorted native grid, Omega's confidence and the trace grid on the record grid.

FW, FH = 96, 64
FK = (80.0, 80.0, 47.5, 31.5)
F_FRAMES = [10, 20, 30]
LOW_CONF_ROWS = 26                                  # record-grid rows ≥ this: under the gate


def _plane_depth(c2w):
    v, u = np.mgrid[0:FH, 0:FW].astype(np.float64)
    d = np.stack([(u - FK[2]) / FK[0], (v - FK[3]) / FK[1], np.ones_like(u)], -1) @ c2w[:3, :3].T
    C = c2w[:3, 3]
    return (3.0 + 0.1 * C[0] - C[2]) / (d[..., 2] - 0.1 * d[..., 0])   # camera z (ray z = 1)


def _f6_session(tmp_path, report_epochs=(5, 3), session_kind=None, f6_epoch=None):
    import cv2
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    from precision import depth_sweep as DS
    from precision.refine import REFINE_NAME, RESIDUALS_NAME
    s = tmp_path / "sess"
    out = s / "output"
    rec = out / "omega_run" / "results_output"
    ddir = out / DS.DEPTH_DIRNAME
    for d in (s / "frames", rec, ddir, out / "precision"):
        d.mkdir(parents=True)
    g = grid_full_frame_resize(FW, FH, FW // 2, FH // 2, "omega")
    cam = CameraModel(FW, FH, (*FK, -0.06, 0.01, 0.0005, -0.0005), "refine", 3, g)
    save_camera_json(out / "camera.json", cam, 5)
    ep = {"epoch": 5}
    if session_kind:
        ep["kind"] = session_kind
    (out / "geometry_epoch.json").write_text(json.dumps(ep))
    c2w = []
    for i, f in enumerate(F_FRAMES):
        T = np.eye(4)
        T[0, 3] = 0.3 * (i - 1)
        c2w.append(T)
        cv2.imwrite(str(s / "frames" / f"{f:06d}.png"),
                    np.random.default_rng(f).integers(0, 255, (FH, FW, 3)).astype(np.uint8))
        conf = np.full((FH // 2, FW // 2), 5.0, np.float32)
        conf[LOW_CONF_ROWS:] = 0.5
        np.savez(rec / f"frame_{f}.npz", depth=np.ones((FH // 2, FW // 2), np.float32), conf=conf,
                 chunk=np.int64(0 if i < 2 else 1))
        z = _plane_depth(T).astype(np.float32)
        src = np.full((FH, FW), DS.SOURCE_SWEEP, np.uint8)
        src[:, FW // 2:] = DS.SOURCE_PRIOR_FILL
        src[:4] = DS.DISCARD_CONTRADICTED
        src[:, :2] = DS.DISCARD_NO_PRIOR
        np.savez_compressed(ddir / f"frame_{f}.npz", depth=np.where(src <= 1, z, 0).astype(np.float32),
                            ncc=np.full((FH, FW), 0.9, np.float32),
                            n_consistent=np.where(src == 0, 3, 1).astype(np.uint8),
                            n_contradict=np.zeros((FH, FW), np.uint8), source=src,
                            normal=np.zeros((FH, FW, 3), np.float32),
                            residual_rel=np.full((FH, FW), 0.01, np.float32))
    np.savetxt(out / "camera_poses.txt", np.stack([T.ravel() for T in c2w]))
    (out / "camera_frames.txt").write_text("\n".join(map(str, F_FRAMES)) + "\n")
    (out / "precision" / REFINE_NAME).write_text(json.dumps(
        {"applied": True, "epoch_to": 5 if f6_epoch is None else f6_epoch}))
    np.savez(out / "precision" / RESIDUALS_NAME, heldout_rms_px=np.full(20, 0.5, np.float32))
    ge, ce = report_epochs
    (ddir / DS.REPORT_NAME).write_text(json.dumps({
        "geometry_epoch": ge, "camera_epoch": ce, "tier1_rule": "max_coverage", "contradiction": True,
        "per_frame": {str(f): {"s_k": 1.1} for f in F_FRAMES}}))
    return s, cam, np.stack(c2w)


@pytest.fixture
def f6_inputs(tmp_path):
    from precision import depth_sweep as DS
    from precision.config import load_precision_config
    s, cam, c2w = _f6_session(tmp_path)
    inp = DS.load_inputs(s, load_precision_config())
    return s, cam, c2w, inp


GATE = {"reconstruction": {"simple": {"conf_percentile": 20, "conf_min_norm": 0.1}}}


def _chunk_points(tmp):
    from correction.session import read_ply
    xyz, fg, pr, pc = [], [], [], []
    for p in sorted(tmp.glob("chunk_*.ply")):
        _, d = read_ply(p)
        with np.load(p.with_name(p.stem + "_origins.npz")) as z:
            fg.append(z["frame_global"]); pr.append(z["pixel_row"]); pc.append(z["pixel_col"])
        xyz.append(np.stack([d["x"], d["y"], d["z"]], 1))
    return (np.concatenate(xyz).astype(np.float64), np.concatenate(fg).astype(np.int64),
            np.concatenate(pr).astype(np.int64), np.concatenate(pc).astype(np.int64))


def test_f7_cloud_from_f6_reproduces_the_geometry_on_the_right_grid(f6_inputs, tmp_path):
    """Every point = c2w · (z_F6(u, v) · K⁻¹ [u, v, 1]) with (u, v) F6's pixel and K the
    session camera (NOT Omega's half-resolution record grid): it lies on the plane; only
    tier 0 / tier 1 enter; the record's confidence gates it through the lens + grid map."""
    from precision.camera import distort_points, native_to_grid
    from precision import depth_sweep as DS
    s, cam, c2w, inp = f6_inputs
    assert np.allclose(inp.K, cam.K()) and tuple(inp.wh) == (FW, FH)
    f6 = CC.load_f6(s / "output", inp)
    assert f6["frames"] == F_FRAMES and np.all(f6["s_k"] == 1.1)
    tmp = tmp_path / "chunks"
    chunks = CC.write_chunks(inp, f6, GATE, tmp, log=lambda m: None)
    assert [c["chunk"] for c in chunks] == [0, 1]
    xyz, fg, pr, pc = _chunk_points(tmp)
    K = cam.K()
    for i, f in enumerate(F_FRAMES):
        m = fg == f
        z = _plane_depth(c2w[i])[pr[m], pc[m]]
        Xc = np.stack([(pc[m] - K[0, 2]) / K[0, 0] * z, (pr[m] - K[1, 2]) / K[1, 1] * z, z], 1)
        assert np.allclose(xyz[m], Xc @ c2w[i][:3, :3].T + c2w[i][:3, 3], atol=1e-5)
    assert np.allclose(xyz[:, 2], 3.0 + 0.1 * xyz[:, 0], atol=1e-4)            # the metric plane
    assert not np.allclose(CC.K_on_grid(K, cam.omega_grid), K)                  # the grids do differ here
    assert pr.min() >= 4 and pc.min() >= 2                                      # contradicted / no prior: out
    # the confidence gate on the record grid, reached through the lens (independently)
    rows = native_to_grid(distort_points(np.stack([pc, pr], 1).astype(np.float64), cam), cam.omega_grid)[:, 1]
    assert rows.max() < LOW_CONF_ROWS - 0.5 + 1e-6
    for f in F_FRAMES:
        with np.load(s / "output" / DS.DEPTH_DIRNAME / f"frame_{f}.npz") as zf:
            v, u = np.nonzero(zf["source"] <= 1)
        rr = native_to_grid(distort_points(np.stack([u, v], 1).astype(np.float64), cam), cam.omega_grid)[:, 1]
        assert int((fg == f).sum()) >= int((rr < LOW_CONF_ROWS - 1).sum())    # everything clear of it entered
    assert sum(c["tier0"] + c["tier1"] for c in chunks) == len(xyz)
    assert all(c["tier0"] > 0 and c["tier1"] > 0 for c in chunks)


def test_f6_frame_off_the_native_grid_fails(f6_inputs):
    from precision import depth_sweep as DS
    s, cam, c2w, inp = f6_inputs
    ddir = s / "output" / DS.DEPTH_DIRNAME
    np.savez(ddir / "frame_20.npz", depth=np.ones((FH // 2, FW // 2), np.float32),
             source=np.zeros((FH // 2, FW // 2), np.uint8))
    with pytest.raises(CC.CorrectedCloudError, match="not on the grid"):
        CC.f6_frame(ddir, 20, inp.wh)


def test_f6_must_have_measured_this_camera_and_geometry(tmp_path):
    from precision import depth_sweep as DS
    from precision.config import load_precision_config
    pc = load_precision_config()
    s, _, _ = _f6_session(tmp_path / "a", report_epochs=(5, 2))
    with pytest.raises(CC.CorrectedCloudError, match="camera epoch 2"):
        CC.load_f6(s / "output", DS.load_inputs(s, pc))
    s, _, _ = _f6_session(tmp_path / "b", report_epochs=(4, 3))
    with pytest.raises(CC.CorrectedCloudError, match="re-run F6"):
        CC.load_f6(s / "output", DS.load_inputs(s, pc))
    # a NEW-CLOUD epoch after F6 moves no camera: F6's depth still holds
    s, _, _ = _f6_session(tmp_path / "c", report_epochs=(4, 3), f6_epoch=4, session_kind="new_cloud")
    assert CC.load_f6(s / "output", DS.load_inputs(s, pc))["frames"] == F_FRAMES
    s, _, _ = _f6_session(tmp_path / "d")
    (s / "output" / DS.DEPTH_DIRNAME / DS.REPORT_NAME).unlink()
    with pytest.raises(CC.CorrectedCloudError, match="run F6 first"):
        CC.load_f6(s / "output", DS.load_inputs(s, pc))


def test_provenance_carries_f6_and_moves_the_pixels_to_the_record_grid(f6_inputs, tmp_path):
    """The final rows get F6's tier / evidence and exact pixel in origins, and pixel_row/col
    of BOTH PLYs move to Omega's record grid through the lens."""
    from correction.session import read_ply, write_ply
    from precision.camera import distort_points, native_to_grid
    from precision import depth_sweep as DS
    s, cam, c2w, inp = f6_inputs
    f6 = CC.load_f6(s / "output", inp)
    tmp = tmp_path / "tx"
    CC.write_chunks(inp, f6, GATE, tmp / "chunks", log=lambda m: None)
    xyz, fg, pr, pc = _chunk_points(tmp / "chunks")
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"),
          ("pixel_col", "<i2")]
    a = np.empty(len(xyz), dt)
    a["x"], a["y"], a["z"] = xyz.T
    a["frame_global"], a["pixel_row"], a["pixel_col"] = fg, pr, pc
    hdr = [b"ply\n", b"format binary_little_endian 1.0\n", f"element vertex {len(a)}\n".encode(),
           b"property float x\n", b"property float y\n", b"property float z\n", b"property int frame_global\n",
           b"property short pixel_row\n", b"property short pixel_col\n", b"end_header\n"]
    write_ply(tmp / "cleaned_cloud.ply", hdr, a)
    write_ply(tmp / "cleaned_cloud_raw.ply", hdr, a)
    cols = CC.provenance_columns(tmp, inp, f6, log=lambda m: None)
    assert np.array_equal(cols["pixel_u_und"], pc) and np.array_equal(cols["pixel_v_und"], pr)
    assert np.array_equal(cols["source"] == DS.SOURCE_SWEEP, pc < FW // 2)
    assert np.array_equal(cols["n_consistent"], np.where(pc < FW // 2, 3, 1))
    assert np.all(cols["camera_epoch"] == 3)
    uv = native_to_grid(distort_points(np.stack([pc, pr], 1).astype(np.float64), cam), cam.omega_grid)
    for name in ("cleaned_cloud.ply", "cleaned_cloud_raw.ply"):
        _, b = read_ply(tmp / name)
        assert b["pixel_col"].max() < FW // 2 and b["pixel_row"].max() < FH // 2
        dc = np.abs(b["pixel_col"] - np.rint(uv[:, 0]))
        dr = np.abs(b["pixel_row"] - np.rint(uv[:, 1]))
        assert (dc + dr == 0).mean() > 0.99 and max(dc.max(), dr.max()) <= 1
