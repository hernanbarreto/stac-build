"""The FLOOR stage (USER 2026-11-05: "el piso lo acomodaba la certificación? hay que agregar al pipeline eso … lo del
piso sí, porque está inclinada la nube"; "sí, implementalo, adelante"): the certification's floor correction alone,
after the projection — its place in the pipeline and the worker's three paths."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline_manager as PM
from pipeline_manager import StageId
from workers import floor_worker as FW


def _cfg(auto_floor=True, auto_segment=True):
    return {"pipeline": {"auto_floor": auto_floor, "auto_segment": auto_segment, "auto_genrecon": False},
            "certify": {"auto_after_segmentation": False}}


def test_the_floor_runs_after_the_projection_and_before_the_certification():
    order = PM.DEFAULT_STAGE_ORDER
    assert order.index(StageId.PROJECTION) < order.index(StageId.FLOOR) < order.index(StageId.CERTIFY)
    assert StageId.FLOOR in PM.SEGMENTATION_CHAIN and StageId.FLOOR in PM.SEGMENT_WHEN_DONE
    assert PM.STAGE_REGISTRY[StageId.FLOOR]["module"] == "workers.floor_worker"
    on = {s.id: s.enabled for s in PM.build_pipeline_stages(config=_cfg())}
    assert on[StageId.FLOOR] and not on[StageId.CERTIFY]
    assert not {s.id: s.enabled for s in PM.build_pipeline_stages(config=_cfg(auto_floor=False))}[StageId.FLOOR]
    assert not {s.id: s.enabled for s in PM.build_pipeline_stages(config=_cfg(auto_segment=False))}[StageId.FLOOR]
    # "Reconstruir" without segmentation ends at the cloud: no floor (nothing to find it on)
    off = {s.id: s.enabled for s in PM.select_stages(PM.build_pipeline_stages(config=_cfg()), segment=False)}
    assert not off[StageId.FLOOR]
    # the projection before it builds no octree: the floor publishes the final one
    job = SimpleNamespace(stages=[SimpleNamespace(stage=s) for s in PM.build_pipeline_stages(config=_cfg())])
    i_proj = next(i for i, st in enumerate(job.stages) if st.stage.id == StageId.PROJECTION)
    assert PM.certify_follows(job, i_proj)
    assert PM.floor_after_segmentation(_cfg()) and not PM.floor_after_segmentation(_cfg(auto_floor=False))
    live = __import__("yaml").safe_load((Path(PM.__file__).parent / "config.yaml").read_text())
    assert live["pipeline"]["auto_floor"] is True


def test_the_floor_record_is_current_only_for_the_geometry_and_segmentation_it_left(tmp_path):
    out = tmp_path
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 4, "correction_id": "abc"}))
    (out / "segmentation_result.json").write_text('{"instances": []}')
    assert FW.floor_is_current(out) == (False, "no floor record")
    FW._write_record(out, status="applied")
    assert FW.floor_is_current(out)[0]
    (out / "segmentation_result.json").write_text('{"instances": [1]}')          # projected again
    assert not FW.floor_is_current(out)[0]
    FW._write_record(out, status="applied")
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 5, "correction_id": "def"}))   # another geometry
    assert not FW.floor_is_current(out)[0]


class _Pipe:
    def __init__(self):
        self.logs, self.progress = [], []

    def send_log(self, m, level="info"):
        self.logs.append(m)

    def send_progress(self, pct, msg, stage=None):
        self.progress.append((pct, msg))


@pytest.fixture
def session(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    (out / "cleaned_cloud.ply").write_text("ply")
    (out / "camera_poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
    (out / "segmentation_result.json").write_text('{"instances": []}')
    calls = {"octree": 0, "run_floor": None, "discard": 0}
    monkeypatch.setattr(FW, "_frozen_config", lambda p, c, log: ({"correction": {}}, "f" * 64))
    import correction.config as CC
    import correction.floor as CF
    import correction.session as CS
    import correction.run as CR
    import segmentation.pipeline as SP
    import workers.map_worker as MW
    apply = SimpleNamespace(mask_filter=True, reconsolidate=True, potree_rebuild=True, depth_correction_mode="sidecar")
    monkeypatch.setattr(CC, "load_correction_config", lambda raw=None: _FakeCfg(apply))
    monkeypatch.setattr(CS, "load_session", lambda o: object())
    monkeypatch.setattr(SP, "ensure_live_octree", lambda o, log=print: calls.__setitem__("octree", calls["octree"] + 1))

    def _run_floor(o, model, kfs, operator, log, progress, cfg, record_extra):
        calls["run_floor"] = {"model": model, "mask_filter": cfg.apply.mask_filter,
                              "reconsolidate": cfg.apply.reconsolidate, "extra": record_extra}
        (Path(o) / "geometry_epoch.json").write_text(json.dumps({"epoch": 5, "correction_id": "c1"}))
        return {"status": "applied", "correction_id": "c1", "epoch_to": 5}
    monkeypatch.setattr(CR, "run_floor", _run_floor)
    monkeypatch.setattr(MW, "_discard_previous_epochs", lambda o: calls.__setitem__("discard", calls["discard"] + 1) or 0)
    return tmp_path, out, calls, CF, monkeypatch


class _FakeCfg:
    def __init__(self, apply):
        self.apply = apply


def _replace_ok(monkeypatch):
    # dataclasses.replace on the fakes: a plain copy with the fields changed
    import dataclasses
    real = dataclasses.replace

    def fake(obj, **kw):
        if isinstance(obj, (_FakeCfg, SimpleNamespace)):
            d = dict(vars(obj)); d.update(kw)
            return _FakeCfg(d["apply"]) if isinstance(obj, _FakeCfg) else SimpleNamespace(**d)
        return real(obj, **kw)
    monkeypatch.setattr(FW, "replace", fake)


def test_a_tilted_floor_is_one_rigid_motion_with_nothing_else_and_one_cloud_stays(session):
    root, out, calls, CF, mp = session
    _replace_ok(mp)
    R = np.array([[1, 0, 0], [0, np.cos(0.05), -np.sin(0.05)], [0, np.sin(0.05), np.cos(0.05)]])
    mp.setattr(CF, "solve_floor_session", lambda s, cfg, log=print: {
        "R_kf": np.stack([R] * 3), "t_kf": np.tile([0.0, -0.1, 0.0], (3, 1)),
        "model_params": {"tilt_deg": 2.86, "tilt_applied": True, "offset_m": -0.1, "height_applied": True}})
    pipe = _Pipe()
    FW._floor_work(pipe, str(root), {})
    assert calls["run_floor"] == {"model": "session", "mask_filter": False, "reconsolidate": False,
                                  "extra": {"run_config_sha256": "f" * 64, "stage": "floor"}}
    assert calls["discard"] == 1                       # the replaced epoch goes: one cloud
    rec = json.loads((out / FW.RECORD).read_text())
    assert rec["status"] == "applied" and rec["live_epoch"] == {"epoch": 5, "correction_id": "c1"}
    assert FW.floor_is_current(out)[0]
    assert any("ONE rigid motion" in m for m in pipe.logs)


def test_a_level_floor_publishes_no_epoch_and_the_octree_is_built_here(session):
    root, out, calls, CF, mp = session
    _replace_ok(mp)
    mp.setattr(CF, "solve_floor_session", lambda s, cfg, log=print: {
        "R_kf": np.stack([np.eye(3)] * 3), "t_kf": np.zeros((3, 3)), "model_params": {"tilt_applied": False}})
    FW._floor_work(_Pipe(), str(root), {})
    assert calls["run_floor"] is None and calls["octree"] == 1 and calls["discard"] == 0
    assert json.loads((out / FW.RECORD).read_text())["status"] == "level"


def test_without_a_segmentation_nothing_moves_declared(session):
    root, out, calls, CF, mp = session
    (out / "segmentation_result.json").unlink()
    mp.setattr(CF, "solve_floor_session", lambda *a, **k: pytest.fail("no floor without a segmentation"))
    pipe = _Pipe()
    FW._floor_work(pipe, str(root), {})
    assert calls["run_floor"] is None and calls["octree"] == 1
    assert json.loads((out / FW.RECORD).read_text())["status"] == "no segmentation"
    assert any("nothing is moved" in m for m in pipe.logs)
