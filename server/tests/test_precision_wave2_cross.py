"""Wave-2 cross items of the precision package (docs/plan_determinismo.md, 2026-10-08):

- point 69 (DECIDIDO USER 2026-10-08): the precision chain takes the FROZEN configuration of
  the run (output/run_config.yaml), never config.yaml from the disk — ONE accessor
  (precision.config.raw_config_of), the loaded config carries the dict it came from
  (PrecisionConfig.raw / raw_of), the runner's step stamps read their ``raw:`` sections from it
  and every step record carries the frozen file's sha256; the worker verifies the dict it was
  handed against the frozen file (frozen_config).
- point 135: a poses epoch (F2 / F5) derives depth_correction.json from the epoch chain, never
  from the live sidecar.
- point 158: the certification's re-level stages its floor transform SEALED for the staged
  cloud, in the fields segmentation.pipeline._floor_seal_status reads.
- point 108: the reprojection verifier maps a record pixel to a mask pixel through the exact
  grid maps (hole_audit._Evidence.mask_pixels), never a scale alone.
- product_epoch: a session without a product report is epoch 0, DECLARED, never refused.

CPU, synthetic.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as server_config                                               # noqa: E402
import repro                                                                 # noqa: E402
from intake.run_config import RunConfigError, freeze_run_config, load_run_config   # noqa: E402
from precision import runner as RN                                           # noqa: E402
from precision.config import (PrecisionConfigError, frozen_config, load_precision_config,  # noqa: E402
                              raw_config_of, raw_of)

QUIET = lambda *a: None  # noqa: E731


def _session(tmp_path: Path) -> Path:
    sd = tmp_path / "sess"
    (sd / "output").mkdir(parents=True)
    return sd


def _server_raw_with(conf_min_norm: float) -> dict:
    raw = json.loads(json.dumps(dict(server_config.cfg)))
    raw["reconstruction"]["simple"]["conf_min_norm"] = conf_min_norm
    return raw


# ── point 69: the one accessor ─────────────────────────────────────────────

def test_raw_config_of_reads_the_frozen_copy_and_declares_the_servers_without_one(tmp_path):
    sd = _session(tmp_path)
    logs = []
    raw, sha = raw_config_of(sd, log=logs.append)
    assert raw is server_config.cfg and sha is None
    assert any("DECLARED" in m and "run_config.yaml" in m for m in logs)
    # a job froze its configuration: THAT one, verified, with its sha256
    rec = freeze_run_config(sd, _server_raw_with(0.37), log=QUIET)
    logs.clear()
    raw, sha = raw_config_of(sd, log=logs.append)
    assert raw["reconstruction"]["simple"]["conf_min_norm"] == 0.37 and sha == rec["sha256"]
    assert raw is not server_config.cfg and not any("DECLARED" in m for m in logs)
    # an edited frozen file is refused, never read
    p = sd / "output" / "run_config.yaml"
    p.write_text(p.read_text() + "# edited after it was frozen\n")
    with pytest.raises(RunConfigError, match="edited after it was frozen"):
        raw_config_of(sd, log=QUIET)


def test_the_loaded_config_carries_its_dict_and_the_steps_read_from_it(tmp_path):
    sd = _session(tmp_path)
    freeze_run_config(sd, _server_raw_with(0.37), log=QUIET)
    pcfg = load_precision_config(session_dir=sd, log=QUIET)
    assert raw_of(pcfg)["reconstruction"]["simple"]["conf_min_norm"] == 0.37
    # the runner's step stamp of a ``raw:`` section is the FROZEN section, not the disk's
    f6 = next(s for s in RN.STEPS if s.key == "f6_bend")
    sec = RN.step_config(f6, pcfg)
    assert sec["reconstruction.simple"]["conf_min_norm"] == 0.37
    # the whole section is the FROZEN section (the server's copy was built from the same
    # template, so only the frozen value tells them apart)
    assert sec["reconstruction.simple"] == raw_of(pcfg)["reconstruction"]["simple"]
    # no session, no dict: the server's — declared as the tests' path, never the pipeline's
    plain = load_precision_config()
    assert raw_of(plain) is server_config.cfg
    # a config built by hand without a dict completes itself from the server's (one place) and
    # SAYS so, once per process: on the pipeline path that line never appears
    from precision import config as PC
    PC._RAW_FALLBACK_DECLARED = False
    logs = []
    bare = replace(plain, raw=None)
    assert raw_of(bare, log=logs.append) is server_config.cfg
    assert len(logs) == 1 and "DECLARED" in logs[0] and "built by hand" in logs[0]
    assert raw_of(bare, log=logs.append) is server_config.cfg and len(logs) == 1
    logs.clear()
    assert raw_of(pcfg, log=logs.append) is not server_config.cfg and logs == []


def test_frozen_config_verifies_the_dict_the_worker_was_handed(tmp_path):
    sd = _session(tmp_path)
    mine = _server_raw_with(0.37)
    freeze_run_config(sd, mine, log=QUIET)
    raw, sha = frozen_config(sd, mine, log=QUIET)
    assert raw["reconstruction"]["simple"]["conf_min_norm"] == 0.37 and len(sha) == 64
    other = _server_raw_with(0.41)
    with pytest.raises(PrecisionConfigError, match="'reconstruction'"):
        frozen_config(sd, other, log=QUIET)
    # the manager's private keys are not configuration
    raw2, sha2 = frozen_config(sd, {**mine, "_stages": {"x": 1}}, log=QUIET)
    assert sha2 == sha


_FAKE = ("import sys, pathlib; s = pathlib.Path(sys.argv[sys.argv.index('--session') + 1]);"
         "(s / 'out_a.txt').write_text('a'); sys.exit(0)")


def test_every_step_record_carries_the_frozen_configurations_sha256(tmp_path, monkeypatch):
    (tmp_path / "fakestep.py").write_text(_FAKE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(repro, "require_exclusive_gpu", lambda log=print: {"cards": [], "compute_processes": 0})
    p = load_precision_config()
    pcfg = replace(p, runner=replace(p.runner, python_da3=sys.executable,
                                     python_mapanything=sys.executable, threads=1))
    steps = [RN.Step("a", "a", "da3", "fakestep", ("a",), reads=(), writes=("out_a.txt",))]
    # an unfrozen session (a chain by hand): None, declared
    sd = _session(tmp_path)
    logs = []
    rep = RN.run_chain(sd, pcfg, log=logs.append, steps=steps)
    assert rep["steps"][0]["run_config_sha256"] is None
    assert any("DECLARED" in m and "run_config.yaml" in m for m in logs)
    # a frozen one: its sha256 on the record
    sd2 = tmp_path / "sess2"
    (sd2 / "output").mkdir(parents=True)
    rec = freeze_run_config(sd2, _server_raw_with(0.37), log=QUIET)
    rep = RN.run_chain(sd2, pcfg, log=QUIET, steps=steps)
    assert rep["steps"][0]["run_config_sha256"] == rec["sha256"]
    state = json.loads((sd2 / "output" / "precision" / RN.STATE_NAME).read_text())
    assert state["done"][0]["run_config_sha256"] == rec["sha256"]
    # an edited frozen file fails the chain naming it, before any step runs
    q = sd2 / "output" / "run_config.yaml"
    q.write_text(q.read_text() + "# edited\n")
    with pytest.raises(RN.ChainError, match="edited after it was frozen"):
        RN.run_chain(sd2, pcfg, log=QUIET, steps=steps)


# ── point 135: the poses epoch's sidecar comes from the chain ─────────────

def _poses_session(tmp_path: Path, n: int = 4) -> Path:
    out = tmp_path / "ps" / "output"
    out.mkdir(parents=True)
    frames = [10 * i for i in range(n)]
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n, dtype=float)
    (out / "camera_poses.txt").write_text(
        "\n".join(" ".join(f"{v:.10g}" for v in p.ravel()) for p in poses) + "\n")
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    return out


def test_a_poses_epoch_never_compounds_a_sidecar_left_live_by_another_run(tmp_path):
    from precision.poses_epoch import apply_pose_epoch
    out = _poses_session(tmp_path)
    # a previous certification's factors, left live under this geometry (what point 135 ends)
    (out / "depth_correction.json").write_text(json.dumps(
        {"version": 2, "epoch": 0, "k": {"0": 3.0, "10": 3.0}, "b": {"0": 0.5, "10": 0.0}}))
    n = 4
    R = np.tile(np.eye(3), (n, 1, 1))
    apply_pose_epoch(out, R, np.zeros((n, 3)), np.full(n, 1.02), "gauge", [], log=QUIET)
    dc = json.loads((out / "depth_correction.json").read_text())
    assert dc["epoch"] == 1 and dc["base_epoch"] == 0 and dc["chain"] == [1]
    assert all(v == pytest.approx(1.02) for v in dc["k"].values()) and len(dc["k"]) == n
    assert all(v == 0.0 for v in dc["b"].values())
    assert dc["chain_stamp"] == repro.sha256_json([])          # the product's empty chain
    # the second epoch composes the FIRST epoch's persisted warp with its own, from the npz
    apply_pose_epoch(out, R, np.zeros((n, 3)), np.full(n, 1.1), "refine", [], log=QUIET)
    dc = json.loads((out / "depth_correction.json").read_text())
    assert dc["epoch"] == 2 and dc["chain"] == [1, 2]
    assert all(v == pytest.approx(1.02 * 1.1) for v in dc["k"].values())
    from correction.chain import chain_stamp
    assert dc["chain_stamp"] == chain_stamp(out, [1])
    # an identity epoch still writes the sidecar (the reader's convention: no frame listed)
    apply_pose_epoch(out, R, np.zeros((n, 3)), np.ones(n), "gauge", [], log=QUIET)
    dc = json.loads((out / "depth_correction.json").read_text())
    assert dc["epoch"] == 3 and all(v == pytest.approx(1.02 * 1.1) for v in dc["k"].values())


# ── point 158: the re-level is sealed for the staged cloud ─────────────────

def test_the_floor_epoch_stages_its_transform_sealed_for_its_own_cloud(tmp_path):
    from correction.apply import stage_transaction, swap_transaction
    from correction.epoch import reconstruction_id_or_none
    from correction.session import load_session
    from segmentation.pipeline import _floor_seal_status
    from tests.synth_correction import build_scene, make_correction_cfg
    scene = build_scene(tmp_path, write_omega_npz=True)
    out = scene.output_dir
    sess = load_session(out)
    n_kf = sess.n_kf
    R = np.tile(np.eye(3), (n_kf, 1, 1))
    t = np.zeros((n_kf, 3))
    k = np.ones(n_kf)
    floor_npz = {"s": np.float64(1.0), "R": np.eye(3), "t": np.array([0.0, 0.1, 0.0])}
    cfg = make_correction_cfg()
    info = stage_transaction(sess, cfg, R, t, k, correction_id="floor-test", scale_diag_new=None,
                             floor_npz=floor_npz, log=QUIET)
    tx = Path(info["tx_dir"])
    rels = [a["rel"] for a in info["artifacts"]]
    assert "floor_transform.npz" in rels and "floor_level.json" in rels
    seal = json.loads((tx / "floor_level.json").read_text())
    st = seal["stamp"]
    assert st["reconstruction_id"] == reconstruction_id_or_none(out) and st["reconstruction_id"]
    assert st["cloud_sha256"] == repro.sha256_file(tx / "cleaned_cloud.ply")
    assert st["floor_transform_sha256"] == repro.sha256_file(tx / "floor_transform.npz")
    assert seal["selected_instance_id"] is None and "floor-test" in seal["leveled_by"]
    z = np.load(tx / "floor_transform.npz")
    assert float(z["s"]) == 1.0 and np.array_equal(z["t"], floor_npz["t"])
    # once live, the stage's reader accepts the transform for THIS cloud and refuses it for another
    swap_transaction(out, info, log=QUIET)
    ok, why = _floor_seal_status(out, out / "cleaned_cloud.ply")
    assert ok, why
    ok, why = _floor_seal_status(out, out / "cleaned_cloud_raw.ply")
    assert not ok and "names cloud" in why
    # the npz bytes are canonical: the same transform staged twice is the same file
    info2 = stage_transaction(load_session(out), cfg, R, t, k, correction_id="floor-test-2",
                              scale_diag_new=None, floor_npz=floor_npz, log=QUIET)
    assert repro.sha256_file(Path(info2["tx_dir"]) / "floor_transform.npz") == st["floor_transform_sha256"]
    # the correction's own log SAYS when the live transform is left unsealed for the live cloud
    # (a depth epoch moved the cloud and no re-level followed): one DECLARED line, none when sealed
    from correction.visit_drift_run import declare_unsealed_floor
    logs = []
    assert declare_unsealed_floor(out, logs.append) is None and logs == []
    import shutil
    shutil.copyfile(out / "cleaned_cloud_raw.ply", out / "cleaned_cloud.ply")
    why = declare_unsealed_floor(out, logs.append)
    assert why and "names cloud" in why
    assert len(logs) == 1 and "DECLARED" in logs[0] and "NOT sealed" in logs[0] and "point 158" in logs[0]
    (out / "floor_transform.npz").unlink()
    logs.clear()
    assert declare_unsealed_floor(out, logs.append) is None and logs == []   # no transform, nothing to seal


# ── point 108: the reprojection verifier's footprint ───────────────────────

def test_the_footprint_takes_the_mask_pixel_from_the_exact_maps_not_a_scale():
    from reconstruction.loops.reprojection import _footprint
    K = np.array([[100.0, 0, 40.0], [0, 100.0, 30.0], [0, 0, 1]])
    seen = []

    def mask_pixels(u, v, mh, mw):
        seen.append((np.asarray(u).copy(), np.asarray(v).copy()))
        # a crop of 7 / 3 px that a scale alone would miss
        return (np.rint(u / 2 + 7).astype(np.int64), np.rint(v / 2 + 3).astype(np.int64))
    ev = SimpleNamespace(cam=SimpleNamespace(pose_map={0: np.eye(4)}, K_for=lambda f: K),
                         kw=80, kh=60, mask_pixels=mask_pixels)
    # ten points on the optical axis at different depths: record pixel (40, 30) every time
    pts = np.stack([np.zeros(10), np.zeros(10), np.linspace(1.0, 3.0, 10)], 1)
    fp = _footprint(ev, 0, pts, 30, 40, 0, occlusion=False)
    assert fp is not None and fp.shape == (30, 40) and seen
    assert fp[int(30 / 2 + 3), int(40 / 2 + 7)] and fp.sum() == 1
    assert not fp[int(30 * 30 / 60), int(40 * 40 / 80)]          # the old scale-only pixel


# ── product_epoch: a legacy session is epoch 0, declared ──────────────────

def test_a_session_without_a_product_report_is_epoch_0_declared_not_refused(tmp_path):
    from correction.chain import product_epoch
    out = tmp_path / "legacy" / "output"
    out.mkdir(parents=True)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    logs = []
    assert product_epoch(out, log=logs.append) == (0, None)
    assert any("DECLARED" in m and "epoch 0" in m and "not refused" in m for m in logs)
    assert product_epoch(out) == (0, None)                        # quiet without a log
