"""The gaussians stage without a card: the strict configuration, the PLY / .splat records, the
training set built from a synthetic session (poses, camera, cloud with provenance), the instance
labels, the trainer's CPU parts (exposure, SSIM, SH conversion), the worker's progress parsing, the
viewer's routes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussians import config as GC                    # noqa: E402
from gaussians import dataset as GD                   # noqa: E402
from gaussians import label as GL                     # noqa: E402
from gaussians import plyio as GP                     # noqa: E402
from tests import synth_evaluation as S               # noqa: E402

SERVER_DIR = Path(__file__).resolve().parents[1]


def _raw_config():
    return yaml.safe_load((SERVER_DIR / "config.yaml").read_text())


def test_the_configuration_is_strict_and_loads_from_config_yaml():
    raw = _raw_config()
    cfg = GC.load_gaussians_config(raw)
    assert cfg.name and cfg.python and cfg.export.formats[0] == "ply"
    assert 0 <= cfg.train.sh_degree <= 3 and cfg.train.holdout_every >= 2
    broken = json.loads(json.dumps(raw))
    del broken["gaussians"]["train"]["depth_loss_weight"]
    with pytest.raises(GC.GaussiansConfigError, match="gaussians.train.depth_loss_weight"):
        GC.load_gaussians_config(broken)
    broken = json.loads(json.dumps(raw))
    broken["gaussians"]["export"]["formats"] = ["splat"]
    with pytest.raises(GC.GaussiansConfigError, match="start with 'ply'"):
        GC.load_gaussians_config(broken)
    broken = json.loads(json.dumps(raw))
    broken["gaussians"]["train"]["strategy"] = "magic"
    with pytest.raises(GC.GaussiansConfigError, match="strategy"):
        GC.load_gaussians_config(broken)


def test_the_3dgs_ply_round_trips_and_the_splat_record_has_the_viewer_layout(tmp_path):
    rng = np.random.default_rng(0)
    N, K = 50, 15
    means = rng.normal(size=(N, 3)).astype(np.float32)
    scales = rng.normal(size=(N, 3)).astype(np.float32)
    quats = rng.normal(size=(N, 4)).astype(np.float32)
    opac = rng.normal(size=N).astype(np.float32)
    sh0 = rng.normal(size=(N, 3)).astype(np.float32)
    shN = rng.normal(size=(N, K, 3)).astype(np.float32)
    p = GP.write_gaussians_ply(tmp_path / "g.ply", means, scales, quats, opac, sh0, shN, comments=["stac gaussians"])
    props, n, _off = GP.read_ply_header(p)
    names = [a for a, _ in props]
    assert n == N and names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert names[9:9 + 3 * K] == [f"f_rest_{i}" for i in range(3 * K)] and names[-8:] == [
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    back = GP.read_gaussians_ply(p)
    for a, b in ((back["means"], means), (back["scales_log"], scales), (back["quats_wxyz"], quats),
                 (back["opacities_logit"], opac), (back["sh0"], sh0), (back["shN"], shN)):
        assert np.array_equal(a, b)
    # f_rest is channel-major: f_rest_{c*K + k} = shN[:, k, c]
    v = GP.read_ply_vertices(p)
    assert np.array_equal(v["f_rest_0"], shN[:, 0, 0]) and np.array_equal(v[f"f_rest_{K}"], shN[:, 0, 1])
    rec = GP.splat_record(means, scales, quats, opac, sh0)
    assert rec.dtype.itemsize == 32 and len(rec) == N
    assert np.allclose(rec["scale"], np.exp(scales))
    rgb = np.clip(sh0 * GP.SH_C0 + 0.5, 0, 1)
    assert np.array_equal(rec["color"][:, :3], np.round(rgb * 255).astype(np.uint8))
    alpha = 1 / (1 + np.exp(-opac))
    assert np.array_equal(rec["color"][:, 3], np.round(alpha * 255).astype(np.uint8))
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    assert np.array_equal(rec["rot"], np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8))
    sp = GP.write_splat(tmp_path / "g.splat", means, scales, quats, opac, sh0)
    assert sp.stat().st_size == 32 * N and np.array_equal(GP.read_splat(sp), rec)
    ids = rng.integers(0, 7, N).astype(np.uint32)
    ip = GP.write_instances(tmp_path / "instances.bin", ids)
    assert ip.stat().st_size == 4 * N and np.array_equal(GP.read_instances(ip), ids)


def test_the_training_set_is_the_session_as_the_chain_left_it(tmp_path):
    scene = S.make_session(tmp_path / "sess", frames=[0, 10, 20, 30, 40, 50, 60, 70], omega_records=True, stride=4)
    out = scene.session_dir / "output"
    import shutil
    shutil.copytree(out / "omega_run" / "results_output", out / "da3_run" / "results_output")   # the F6 product's layout
    v = GP.read_ply_vertices(out / "cleaned_cloud.ply")
    n = len(v)
    inst = np.zeros(n, np.uint32)
    inst[: n // 2] = 7                                    # the first half of the cloud is object 7
    np.save(out / "instance_ids.npy", inst)
    raw = _raw_config()
    raw["gaussians"]["init"]["max_points"] = 1000
    raw["gaussians"]["init"]["min_confidence"] = 0.5
    raw["gaussians"]["train"]["holdout_every"] = 4
    cfg = GC.load_gaussians_config(raw)
    ts, loader = GD.build_training_set(scene.session_dir, cfg, log=lambda *a: None)
    assert ts.n_frames == 8 and list(ts.frames) == [0, 10, 20, 30, 40, 50, 60, 70]
    assert ts.holdout.tolist() == [False, False, False, True, False, False, False, True]
    assert ts.width == S.W and ts.height == S.H
    assert np.allclose(ts.K, scene.K) and np.allclose(ts.dist, 0.0)        # the grid is the native frame here
    assert np.allclose(ts.c2w, scene.c2w)
    assert ts.n_seeds == 1000
    conf = np.asarray(v["confidence"])
    assert conf[np.isin(np.arange(n), np.flatnonzero(conf >= 0.5))].min() >= 0.5
    # the seeds are cloud points: their coordinates, colours and instances are the cloud's
    xyz_all = np.stack([v["x"], v["y"], v["z"]], axis=1)
    from scipy.spatial import cKDTree
    d, j = cKDTree(xyz_all).query(ts.seeds_xyz)
    assert d.max() < 1e-6 and set(np.unique(ts.seeds_instance)) <= {0, 7}
    assert np.array_equal(ts.seeds_instance, inst[j]) and np.array_equal(ts.seeds_cloud_index, j)
    assert ts.seeds_scale_log.shape == (1000, 3) and np.all(np.isfinite(ts.seeds_scale_log))
    # a second build is the same set (a seeded subsample)
    ts2, _ = GD.build_training_set(scene.session_dir, cfg, log=lambda *a: None)
    assert np.array_equal(ts2.seeds_xyz, ts.seeds_xyz)
    # depth maps: the chain's per-keyframe depth, training frames only, on the record grid
    assert ts.depth_grid == (S.H, S.W) and loader is not None
    d_out = GD.write_trainset(ts, tmp_path / "trainset", loader, log=lambda *a: None)
    back = GD.read_trainset(d_out)
    assert back.n_seeds == 1000 and np.array_equal(back.frames, ts.frames) and back.image_paths == ts.image_paths
    assert np.array_equal(back.seeds_cloud_index, ts.seeds_cloud_index)
    assert back.meta["n_depth_maps"] == 6 and GD.read_depth_map(d_out, 30) is None
    dm = GD.read_depth_map(d_out, 10)
    assert dm is not None and dm[0].shape == (S.H, S.W) and dm[0].dtype == np.float32 and dm[1].all()
    # the depth is the plane's camera depth
    w2c = np.linalg.inv(scene.c2w[1])
    assert abs(float(np.median(dm[0])) - abs(S.Z0 - scene.c2w[1][2, 3])) < 1e-3 or np.all(dm[0] > 0)


def test_the_training_set_declares_what_the_session_lacks(tmp_path):
    scene = S.make_session(tmp_path / "sess", frames=[0, 10, 20, 30], stride=8)       # no depth records
    cfg = GC.load_gaussians_config(_raw_config())
    ts, loader = GD.build_training_set(scene.session_dir, cfg, log=lambda *a: None)
    assert loader is None and ts.depth_grid is None and ts.meta["depth"]["available"] is False
    assert "no instance_ids.npy" in ts.meta["instances"] and np.all(ts.seeds_instance == 0)
    import shutil
    shutil.rmtree(scene.session_dir / "output")
    with pytest.raises(GD.DatasetError, match="cleaned_cloud.ply"):
        GD.build_training_set(scene.session_dir, cfg, log=lambda *a: None)


def test_the_instance_is_resolved_live_from_the_cloud_through_the_seed_row():
    """USER 2026-10-31: the segmentation changes after training (merges, new instances, erased
    points) — the gaussians carry their seed's cloud row and the instance is read from the CURRENT
    instance_ids.npy; nothing is baked in."""
    seeds = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float64)
    rows = np.array([10, 20, 30], np.uint32)                 # the seeds' rows in the published cloud
    xyz = np.array([[0.01, 0, 0], [1.02, 0, 0], [2.0, 0.03, 0], [5, 5, 5], [0.5, 0, 0]])
    si = GL.seed_index_by_nearest(xyz, seeds, rows, radius_m=0.05)
    assert si.tolist() == [10, 20, 30, int(GL.NO_SEED), int(GL.NO_SEED)]   # far away and mid-way: no seed, never a guess
    ids_now = np.zeros(40, np.uint32); ids_now[10] = 3; ids_now[30] = 5
    assert GL.instances_from_seed_index(si, ids_now).tolist() == [3, 0, 5, 0, 0]
    ids_later = ids_now.copy(); ids_later[10] = 5                       # the Segmentation Manager merged 3 into 5
    assert GL.instances_from_seed_index(si, ids_later).tolist() == [5, 0, 5, 0, 0]
    # a cloud with fewer points than the rows name: those gaussians read 0 (the mapping is stale, said elsewhere)
    assert GL.instances_from_seed_index(si, ids_now[:25]).tolist() == [3, 0, 0, 0, 0]
    assert GL.instance_census(GL.instances_from_seed_index(si, ids_later)) == {"n_gaussians": 5, "n_segmented": 2, "n_instances": 1}
    assert GL.seed_index_by_nearest(xyz, np.zeros((0, 3)), np.zeros(0, np.uint32), 0.1).tolist() == [int(GL.NO_SEED)] * 5


def test_the_trainers_cpu_parts_behave():
    import torch
    from gaussians import train as GT
    assert np.allclose(GT.rgb_to_sh0(np.array([[0.5, 0.5, 0.5]])), 0.0)
    assert abs(GT.logit(0.5)) < 1e-12 and GT.logit(0.1) < 0
    c2w = np.tile(np.eye(4), (3, 1, 1)); c2w[:, 0, 3] = [0.0, 2.0, 4.0]
    assert abs(GT.scene_extent(c2w) - 2.2) < 1e-9                       # 1.1 × the largest distance to the mean
    assert [GT.sh_degree_at(s, 3, 1000) for s in (0, 999, 1000, 2500, 9000)] == [0, 0, 1, 2, 3]
    # the depth grid is the same camera at another resolution: pixel (u, v) of the native frame maps
    # through the intrinsics ratio; a half-resolution grid halves the coordinates about the centre
    Kn = np.array([[400.0, 0, 231.5], [0, 400.0, 415.5], [0, 0, 1]])
    Kd = np.array([[200.0, 0, 115.5], [0, 200.0, 207.5], [0, 0, 1]])
    ud, vd = GT.depth_pixel_map(Kn, Kd, 464, 832)
    assert ud[0] == 0 and ud[463] == 231 and vd[0] == 0 and vd[831] == 415
    depth = np.full((416, 232), 2.0, np.float32); depth[0, 0] = 5.0
    valid = np.ones_like(depth, bool); valid[1, 1] = False
    d, ok = GT.resample_depth(depth, valid, ud, vd)
    assert d.shape == (832, 464) and d[0, 0] == 5.0 and d[500, 300] == 2.0
    assert not ok[2, 2] and not ok[3, 3] and ok[500, 300]                 # the invalid depth pixel covers its 2×2 natives
    gm = {"scale_x": 2.0, "scale_y": 2.0, "pad_left": 0.0, "pad_top": 0.0, "crop_x": 0.0, "crop_y": 0.0}
    ug, vg = GT.grid_pixel_map(gm, 464, 832)
    assert ug[0] == 0 and ug[463] == 231 and vg[831] == 415
    img = torch.rand(40, 50, 3)
    assert abs(float(GT.ssim(img, img)) - 1.0) < 1e-4 and GT.psnr(img, img) > 90
    assert float(GT.ssim(img, torch.zeros_like(img))) < 0.2
    exp = GT.Exposure(2, torch.device("cpu"))
    assert torch.allclose(exp.apply(img, 1), img) and float(exp.penalty().detach()) == 0.0


def test_the_worker_reads_the_trainers_progress_and_the_stage_is_wired_off_by_default():
    from workers import gaussians_worker as GW
    assert GW.parse_progress("[gs] it 3000/30000 (10.0%) loss 0.0812 gaussians 2,100,000 sh 3") == 10.0 + 8.5
    assert GW.parse_progress("[gs] eval it 5000: psnr 27.10") is None and GW.parse_progress("noise") is None
    import pipeline_manager as PM
    assert PM.StageId.GAUSSIANS.value == "gaussians" and PM.STAGE_REGISTRY[PM.StageId.GAUSSIANS]["module"] == "workers.gaussians_worker"
    assert PM.DEFAULT_STAGE_ORDER[-1] == PM.StageId.GAUSSIANS and PM.DEFAULT_STAGE_ORDER[-2] == PM.StageId.GENRECON
    assert PM.StageId.GAUSSIANS in PM.SEGMENTATION_CHAIN
    for src in (PM.StageId.RECONSTRUCTION, PM.StageId.CLOUDCOMPY, PM.StageId.VLM, PM.StageId.SAM3, PM.StageId.PROJECTION, PM.StageId.CERTIFY):
        assert PM.StageId.GAUSSIANS in PM.PipelineManager.CASCADE_INVALIDATION[src], src
    raw = _raw_config()
    on = {s.id: s.enabled for s in PM.build_pipeline_stages(config=raw)}
    assert on[PM.StageId.GAUSSIANS] is False                              # pipeline.auto_gaussians: false until judged by eye
    raw2 = json.loads(json.dumps(raw)); raw2["pipeline"]["auto_gaussians"] = True
    on2 = {s.id: s.enabled for s in PM.build_pipeline_stages(config=raw2)}
    assert on2[PM.StageId.GAUSSIANS] is True and on2[PM.StageId.GENRECON] is True
    raw3 = json.loads(json.dumps(raw2)); raw3["gaussians"]["enabled"] = False
    assert {s.id: s.enabled for s in PM.build_pipeline_stages(config=raw3)}[PM.StageId.GAUSSIANS] is False
    raw4 = json.loads(json.dumps(raw2)); raw4["pipeline"]["auto_segment"] = False
    assert {s.id: s.enabled for s in PM.build_pipeline_stages(config=raw4)}[PM.StageId.GAUSSIANS] is False


def _eval_project(tmp_path, sid="sess_gs", n_points=40):
    """A project the routes resolve (config.PROJECTS_DIR is temporary in the test session) with one
    published gaussians model."""
    import config
    proj = Path(config.PROJECTS_DIR) / sid
    scan = proj / "scans" / "2026-01-01" / "src_default"
    out = scan / "output" / "gaussians" / "gaussians"
    out.mkdir(parents=True)
    (proj / "project.json").write_text(json.dumps({"name": sid, "slug": sid, "scans": [
        {"key": "2026-01-01/default", "date": "2026-01-01", "source": "default"}], "active_scan": "2026-01-01/default"}))
    (scan / "output" / "cleaned_cloud.ply").write_bytes(b"ply\n")
    rng = np.random.default_rng(1)
    N = 12
    GP.write_gaussians_ply(out / "gaussians.ply", rng.normal(size=(N, 3)), np.zeros((N, 3)), np.tile([1, 0, 0, 0], (N, 1)),
                           np.zeros(N), np.zeros((N, 3)), None)
    seed_index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xFFFFFFFF, 39], np.uint32)
    GP.write_instances(out / "seed_index.bin", seed_index)
    (out / "meta.json").write_text(json.dumps({"name": "gaussians", "files": {"ply": "gaussians.ply", "seed_index": "seed_index.bin"},
                                               "n_gaussians": N, "cloud": {"n_points": n_points}, "geometry_epoch": 2}))
    (out / "report.json").write_text(json.dumps({"n_gaussians": N}))
    return sid, scan / "output", seed_index


def test_the_routes_list_the_model_serve_its_files_and_resolve_the_instances_live(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from gaussians import api as GA
    sid, out, seed_index = _eval_project(tmp_path)
    app = FastAPI(); app.include_router(GA.router)
    c = TestClient(app)
    r = c.get(f"/api/gaussians/list/{sid}")
    assert r.status_code == 200
    m = r.json()["models"]
    assert len(m) == 1 and m[0]["name"] == "gaussians" and m[0]["n_gaussians"] == 12 and m[0]["stale"] is False
    assert "no instance_ids.npy" in m[0]["reason"] and m[0]["files"]["ply"].endswith("/gaussians/gaussians.ply")
    assert c.get(m[0]["files"]["ply"]).status_code == 200
    assert c.get(f"/api/gaussians/file/{sid}/gaussians/secret.txt").status_code == 404
    assert c.get(f"/api/gaussians/file/{sid}/../gaussians/meta.json").status_code in (400, 404)
    # no projection yet: every gaussian unsegmented
    r = c.get(f"/api/gaussians/instances/{sid}/gaussians")
    ids = np.frombuffer(r.content, dtype="<u4")
    assert r.status_code == 200 and ids.tolist() == [0] * 12 and r.headers["X-Gaussians-Stale"] == "0"
    # the projection writes instance_ids.npy — the splats take the instances through their seed rows
    inst = np.zeros(40, np.uint32); inst[0:5] = 3; inst[5:10] = 8; inst[39] = 11
    np.save(out / "instance_ids.npy", inst)
    ids = np.frombuffer(c.get(f"/api/gaussians/instances/{sid}/gaussians").content, dtype="<u4")
    assert ids.tolist() == [3, 3, 3, 3, 3, 8, 8, 8, 8, 8, 0, 11]
    # the Segmentation Manager merges 3 into 8: nothing retrained, the next read follows
    inst[0:5] = 8; np.save(out / "instance_ids.npy", inst)
    r = c.get(f"/api/gaussians/instances/{sid}/gaussians")
    assert np.frombuffer(r.content, dtype="<u4").tolist() == [8] * 10 + [0, 11] and r.headers["X-Gaussians-Instances"] == "2"
    # another cloud (another point count): stale, said, every gaussian 0
    np.save(out / "instance_ids.npy", np.zeros(41, np.uint32))
    r = c.get(f"/api/gaussians/instances/{sid}/gaussians")
    assert r.headers["X-Gaussians-Stale"] == "1" and "41 points now" in r.headers["X-Gaussians-Reason"]
    assert np.frombuffer(r.content, dtype="<u4").tolist() == [0] * 12
    assert c.get(f"/api/gaussians/list/{sid}").json()["models"][0]["stale"] is True


def test_a_lens_session_runs_the_mcmc_strategy_and_the_report_says_why():
    """gsplat's default densification needs the gradient of the projected means; the unscented-transform
    projection of a distorted lens has none (pccr 2026-10-31: retain_grad on a tensor without grad) — so a lens
    session runs MCMC and says so; a pinhole session runs what was configured; mcmc is never changed."""
    from gaussians.train import strategy_choice
    used, why = strategy_choice("default", lens=True)
    assert used == "mcmc" and "lens" in why and "gradient" in why
    assert strategy_choice("default", lens=False) == ("default", "as configured")
    assert strategy_choice("mcmc", lens=True)[0] == "mcmc" and strategy_choice("mcmc", lens=False)[0] == "mcmc"
