"""depth_on_f5.compute end to end on a synthetic session (audit 2026-10-23): Omega's depth of an
oblique plane carries monocular depth's scale-and-shift error; F5's landmarks are exact. Under
``bend.model: judged`` the held-out chooses the plane-preserving family, the bent maps are planes
again, the vote keeps them and the report names what it did. No GPU, no PointDiT."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import bend_model as BM                            # noqa: E402

FW, FH = 80, 60
FK = (70.0, 70.0, 40.0, 30.0)
FRAMES = [10, 20, 30, 40, 50]


def _plane_depth(c2w):
    """Camera z of the plane x·0.1 − z = −3 (a wall slanting away to the right), per pixel."""
    v, u = np.mgrid[0:FH, 0:FW].astype(np.float64)
    d = np.stack([(u - FK[2]) / FK[0], (v - FK[3]) / FK[1], np.ones_like(u)], -1) @ c2w[:3, :3].T
    C = c2w[:3, 3]
    return (3.0 + 0.1 * C[0] - C[2]) / (d[..., 2] - 0.1 * d[..., 0])


def _session(tmp_path, a: float = 1.08, b0: float = -0.03):
    """A session the way F0-F5 leave it: camera.json (no lens, grid = native), F5's poses, Omega's
    records at the camera grid with 1/z_omega = (1/z − b0) / a, the tracks' observations of 300
    plane points split fit / held-out, F5's sidecars."""
    import cv2
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    from precision.refine import REFINE_NAME, RESIDUALS_NAME
    from precision.tracks import TRACKS_NAME
    s = tmp_path / "sess"
    out = s / "output"
    rec = out / "omega_run" / "results_output"
    for d in (s / "frames", rec, out / "precision"):
        d.mkdir(parents=True)
    g = grid_full_frame_resize(FW, FH, FW, FH, "omega")
    cam = CameraModel(FW, FH, (*FK, 0.0, 0.0, 0.0, 0.0), "refine", 3, g)
    save_camera_json(out / "camera.json", cam, 5)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 5}))
    c2w = []
    for i, f in enumerate(FRAMES):
        T = np.eye(4); T[0, 3] = 0.15 * (i - 2)
        c2w.append(T)
        cv2.imwrite(str(s / "frames" / f"{f:06d}.jpg"),
                    np.random.default_rng(f).integers(0, 255, (FH, FW, 3)).astype(np.uint8))
        z = _plane_depth(T)
        z_om = 1.0 / ((1.0 / z - b0) / a)                             # Omega's error, in inverse depth
        np.savez(rec / f"frame_{f}.npz", depth=z_om.astype(np.float32), conf=np.full((FH, FW), 5.0, np.float32),
                 chunk=np.int64(0))
    np.savetxt(out / "camera_poses.txt", np.stack([T.ravel() for T in c2w]))
    (out / "camera_frames.txt").write_text("\n".join(map(str, FRAMES)) + "\n")
    (out / "precision" / REFINE_NAME).write_text(json.dumps({"applied": True, "epoch_to": 5}))
    np.savez(out / "precision" / RESIDUALS_NAME, heldout_rms_px=np.full(20, 0.5, np.float32))
    # the tracks: 300 points of the plane, observed in every keyframe that sees them
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(-2.5, 2.5, 300), rng.uniform(-1.5, 1.5, 300), np.zeros(300)], 1)
    X[:, 2] = 3.0 + 0.1 * X[:, 0]                                      # on the plane
    obs_t, obs_f, obs_uv = [], [], []
    for t in range(300):
        for i, f in enumerate(FRAMES):
            Xc = np.linalg.inv(c2w[i])[:3, :3] @ X[t] + np.linalg.inv(c2w[i])[:3, 3]
            u = FK[0] * Xc[0] / Xc[2] + FK[2]; v = FK[1] * Xc[1] / Xc[2] + FK[3]
            if 1 <= u <= FW - 2 and 1 <= v <= FH - 2:
                obs_t.append(t); obs_f.append(f); obs_uv.append((u, v))
    obs_t = np.asarray(obs_t, np.int64)
    np.savez(out / "precision" / TRACKS_NAME, obs_track=obs_t, obs_frame=np.asarray(obs_f, np.int64),
             obs_uv_native=np.asarray(obs_uv, np.float32), track_query_id=np.arange(300, dtype=np.int64),
             track_split=(np.arange(300) % 2).astype(np.int8))
    return s, cam, np.stack(c2w)


def _pcfg(model: str, colmap_tier: bool = False):
    from precision.config import load_precision_config
    pcfg = load_precision_config()
    return replace(pcfg, mono_detail=replace(pcfg.mono_detail, enabled=False),
                   bend=replace(pcfg.bend, model=model, windows=(0, 1), neighbors=(-1, 1)),
                   fusion=replace(pcfg.fusion, colmap_tier=colmap_tier, colmap_max_sources=4, colmap_rows_stride=2))


def _plane_rms(dep: np.ndarray, K: np.ndarray) -> float:
    return BM.plane_rms_m(dep.astype(np.float64), K, dep > 0)[0]


@pytest.mark.parametrize("model", [BM.JUDGED, BM.INVERSE_DEPTH, BM.AFFINE_PIXEL])
def test_compute_bends_omega_to_the_landmarks_and_reports_the_model(tmp_path, model):
    from precision.depth_on_f5 import compute
    s, cam, c2w = _session(tmp_path)
    C = compute(s, _pcfg(model), log=lambda *a: None)
    K = np.array([[FK[0], 0, FK[2]], [0, FK[1], FK[3]], [0, 0, 1]])
    assert C.N == len(FRAMES) and C.W == FW and C.H == FH
    # the raw record is NOT a plane (the inverse-depth shift curves it); the bent maps are judged on
    # the half-A held-out rows the fit never saw
    expect = BM.INVERSE_DEPTH if model == BM.JUDGED else model
    assert C.model == expect and C.window_rule.get("model") == expect
    if model == BM.JUDGED:
        assert C.window_rule["best_model"] == BM.INVERSE_DEPTH and set(C.score) == set(BM.MODELS)
        assert C.score[BM.INVERSE_DEPTH][C.wb] < C.score[BM.AFFINE_PIXEL][C.wb]
    assert C.score_B < (0.004 if expect == BM.INVERSE_DEPTH else 0.03)
    for i, f in enumerate(C.frames):
        coef = C.coefs[i]
        assert len(coef) == BM.n_coef(expect) and C.provenance[i]["model"] == expect
        if expect == BM.INVERSE_DEPTH and C.provenance[i]["status"] != "identity":
            assert abs(coef[0] - 1.08) < 0.02 and abs(coef[1] + 0.03) < 0.01
    # the voted maps: the middle keyframe is a plane again under the inverse family (exact synthetic
    # planes make the measured tau vanishingly small, so the vote drops the pixels a half-pixel of
    # splat resolution contradicts — a property of the data, not of the rule)
    z, agree, contra = C.voted[C.frames[2]]
    assert (z > 0).mean() > 0.7 and (contra[z > 0] <= agree[z > 0]).all()
    rms = _plane_rms(z, K)
    if expect == BM.INVERSE_DEPTH:
        assert rms < 0.002, f"the inverse family must give the plane back: rms {rms * 100:.2f} cm"
    assert C.vst["mixed_not_judging"] == 0 and 0 <= C.tau < 0.05
    # the owner fusion ran over the voted maps (audit 2026-10-23): the five keyframes publish the plane
    # ONCE between them — the owners are a fraction of the observations, every consumed pixel names an
    # owner, and the fused points of the confirmed owners are the plane
    F = C.fusion
    assert F is not None and 0 < F.report["owners"] < F.report["observations"]
    assert F.report["owners"] + F.report["consumed"] == F.report["observations"]
    rep_ = F.report["representation"]
    assert rep_["resolved_share"] > 0.99 and rep_["distance_footprints"]["p95"] < 3.0, rep_
    if expect == BM.INVERSE_DEPTH:
        P = []
        for i, f in enumerate(C.frames):
            m = F.owner[f] & (F.members[f] >= 2) & (F.depth[f] > 0)
            zz = F.depth[f][m].astype(np.float64)
            uu, vv = np.meshgrid(np.arange(FW), np.arange(FH))
            X = np.stack([(uu[m] - FK[2]) / FK[0] * zz, (vv[m] - FK[3]) / FK[1] * zz, zz], 1)
            P.append(X @ c2w[i][:3, :3].T + c2w[i][:3, 3])
        P = np.concatenate(P)
        n = np.array([0.1, 0.0, -1.0]); n /= np.linalg.norm(n)              # the plane x·0.1 − z = −3
        assert np.sqrt(np.mean((P @ n + 3.0 * n[2] * -1) ** 2)) < 0.01 or np.sqrt(np.mean((P @ n - 3.0 / np.linalg.norm([0.1, 0, -1])) ** 2)) < 0.01


def _fake_patch_match(stripe_rows):
    """A stand-in for the COLMAP binary: writes, for every reference image of patch-match.cfg, the TRUE
    depth of the plane on a horizontal stripe (the textured band PatchMatch measures) and 0 elsewhere."""
    from precision.depth_colmap import write_colmap_array

    def run(cmd, logf):
        ws = Path(cmd[cmd.index("--workspace_path") + 1])
        cfg = (ws / "stereo" / "patch-match.cfg").read_text().splitlines()
        names = [cfg[k] for k in range(0, len(cfg), 2)]
        # images.txt: image id → frame (from the name) and its pose; cameras.txt: K
        poses = {}
        lines = (ws / "sparse" / "images.txt").read_text().splitlines()
        for k in range(0, len(lines), 2):
            parts = lines[k].split()
            poses[parts[-1]] = parts
        from precision.depth_colmap import qvec_to_rotmat
        for name in names:
            parts = poses[name]
            q = [float(x) for x in parts[1:5]]; t = np.array([float(x) for x in parts[5:8]])
            R = qvec_to_rotmat(q)
            w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
            z = _plane_depth(np.linalg.inv(w2c))
            dm = np.where(stripe_rows, z, 0.0).astype(np.float32)
            write_colmap_array(ws / "stereo" / "depth_maps" / f"{name}.geometric.bin", dm)
        Path(logf).write_text("fake patch_match_stereo\n")
        return 0
    return run


def test_the_photometric_reference_calibrates_the_bend_and_owns_the_textured_band(tmp_path):
    """Phase 3 end to end on the synthetic session: the COLMAP tier step (its binary replaced by a
    stand-in that measures the plane exactly on a stripe) writes the tier-0 maps and the report;
    compute() takes COLMAP's dense rows for the bend (judged on the landmarks), fuses the stripe at
    tier 0 and marks those points `colmap` in the source column."""
    from precision import colmap_tier as CT
    from precision.depth_on_f5 import SRC_COLMAP, compute
    s, cam, c2w = _session(tmp_path)
    pcfg = _pcfg(BM.INVERSE_DEPTH, colmap_tier=True)
    stripe = (np.arange(FH)[:, None] > FH * 0.35) & (np.arange(FH)[:, None] < FH * 0.65) & np.ones((1, FW), bool)
    rep = CT.run_colmap_tier(s, pcfg, log=lambda *a: None, runner=_fake_patch_match(stripe))
    assert rep["n_frames"] == len(FRAMES) and rep["tier0_pixels"] > 0 and 0 <= rep["tau_rel"] < 0.01
    from correction.epoch import reconstruction_id
    rid = reconstruction_id(s / "output")
    maps, rep2 = CT.load_colmap_tier(s / "output", FRAMES, {"geometry_epoch": 5, "camera_epoch": 3}, rid, 3, (FW, FH))
    assert set(maps) == set(FRAMES) and all(m[1][stripe].mean() > 0.5 for m in maps.values())
    C = compute(s, pcfg, log=lambda *a: None)
    assert C.colmap_rep is not None and C.bend_rows in ("landmarks", "landmarks+colmap")
    assert any("|landmarks+colmap" in k for k in C.cand_keys)
    F = C.fusion
    assert F.report["tiers"]["owners_by_fused_tier"]["0"] > 0.5 * int(stripe.sum())
    # the stripe's owners carry the photometric depth: the plane exactly, whatever Omega's bias
    for i, f in enumerate(C.frames):
        own0 = F.owner[f] & (F.fused_tier[f] == 0)
        if own0.any():
            zt = _plane_depth(c2w[i])
            assert np.abs(F.depth[f][own0] / zt[own0] - 1).max() < 0.005
    assert SRC_COLMAP == 5
    # another geometry's report is refused by name
    import json
    rp = CT.report_path(s / "output"); d = json.loads(rp.read_text()); d["geometry_epoch"] = 4; rp.write_text(json.dumps(d))
    with pytest.raises(CT.ColmapError, match="re-run the COLMAP tier"):
        CT.load_colmap_tier(s / "output", FRAMES, {"geometry_epoch": 5, "camera_epoch": 3}, rid, 3, (FW, FH))
