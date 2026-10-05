"""The precision core as a pipeline stage: the one step list, in order, resumed after
the last finished step while the session is still in the epoch it left."""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from precision import runner as RN


def _steps(tmp):
    # each fake step appends its key to a log and (for "pub") publishes an epoch
    code = ("import sys, json, pathlib; s = pathlib.Path(sys.argv[sys.argv.index('--session') + 1]);"
            "(s / 'ran.txt').open('a').write(sys.argv[-1] + '\\n');"
            "p = s / 'output' / 'geometry_epoch.json';"
            "e = json.loads(p.read_text())['epoch'] if p.exists() else 0;"
            "p.write_text(json.dumps({'epoch': e + 1})) if sys.argv[-1] == 'pub' else None;"
            "sys.exit(3 if (s / ('fail_' + sys.argv[-1])).exists() else 0)")
    mod = tmp / "fakestep.py"
    mod.write_text(code)
    return [RN.Step(k, k, "da3", "fakestep", (k,)) for k in ("a", "pub", "c")]


@pytest.fixture
def pcfg(monkeypatch, tmp_path):
    from precision.config import load_precision_config
    p = load_precision_config()
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return replace(p, runner=replace(p.runner, python_da3=sys.executable,
                                     python_mapanything=sys.executable, threads=1))


def test_runs_in_order_and_resumes_after_a_failure(tmp_path, pcfg, monkeypatch):
    s = tmp_path / "sess"
    (s / "output").mkdir(parents=True)
    steps = _steps(tmp_path)
    monkeypatch.setattr(RN, "Path", Path)
    (s / "fail_c").write_text("")
    with pytest.raises(RN.ChainError, match="resumes from this step"):
        RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    assert (s / "ran.txt").read_text().split() == ["a", "pub", "c"]
    (s / "fail_c").unlink()
    rep = RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    # a and pub are not repeated: the session is still in the epoch pub left
    assert (s / "ran.txt").read_text().split() == ["a", "pub", "c", "c"]
    assert [r["key"] for r in rep["steps"]] == ["a", "pub", "c"] and rep["epoch"] == 1


def test_refuses_to_resume_on_another_epoch(tmp_path, pcfg):
    s = tmp_path / "sess"
    (s / "output").mkdir(parents=True)
    steps = _steps(tmp_path)
    (s / "fail_c").write_text("")
    with pytest.raises(RN.ChainError):
        RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    (s / "output" / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    with pytest.raises(RN.ChainError, match="now in epoch 0"):
        RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)


def test_the_core_runs_inside_the_reconstruction_stage():
    """USER 2026-09-28: F0-F7 is part of the reconstruction, before the cloud stage;
    there is no separate PRECISION stage and no visit_drift step in the chain."""
    from pipeline_manager import DEFAULT_STAGE_ORDER, STAGE_REGISTRY, StageId
    assert not hasattr(StageId, "PRECISION")
    assert all("precision" not in v["module"] for v in STAGE_REGISTRY.values())
    assert DEFAULT_STAGE_ORDER.index(StageId.RECONSTRUCTION) < DEFAULT_STAGE_ORDER.index(StageId.CLOUDCOMPY)
    import workers.precision_worker as W
    assert callable(W.run)
    import inspect
    import workers.map_worker as M
    assert callable(M._run_precision_core) and not hasattr(M, "_run_semantics_2d")
    # no working cloud: the core merges / filters nothing before F7 (USER 2026-09-29)
    src = inspect.getsource(M._run_precision_core)
    assert "cloudcompy_worker" not in src and "merge" not in src.lower().replace("merged", "")
    for mod in ("precision.gauge", "precision.refine"):
        text = pathlib.Path(mod.replace(".", "/") + ".py").read_text()
        assert "apply_pose_epoch" in text and "apply_transform_epoch(" not in text, mod
    keys = [s.key for s in RN.STEPS]
    assert keys[0] == "f0_camera" and keys[-1] == "f6_check" and "f7_cloud" in keys, keys
    assert not any("measure" in k or "visit_drift" in s.module for k, s in zip(keys, RN.STEPS)), keys


def test_rerun_from_a_step_tolerates_the_clouds_it_published(monkeypatch, tmp_path):
    """USER 2026-10-04: a changed depth stage re-runs over the same F5 — the new-cloud epochs the
    previous bend published move no camera, so the finished prefix still counts."""
    from precision import runner as R
    steps = [R.Step("f5_refine", "F5", "da3", "m"), R.Step("f6_bend", "F6", "da3", "m"), R.Step("f6_check", "chk", "da3", "m")]
    state = {"done": [{"key": "f5_refine", "epoch_after": 2}, {"key": "f6_bend", "epoch_after": 3}, {"key": "f6_check", "epoch_after": 3}]}
    import correction.epoch as CE
    monkeypatch.setattr(CE, "epoch_kind", lambda out, e: "new_cloud" if e == 3 else "transform")
    # forgetting from f6_bend: the prefix ends at epoch 2, the session is at 3 (a published cloud) → resumes at f6_bend
    state["done"] = state["done"][:1]
    assert R.resume_point(state, steps, 3, out=tmp_path) == 1
    # a TRANSFORM epoch in between is not tolerated
    monkeypatch.setattr(CE, "epoch_kind", lambda out, e: "transform")
    import pytest
    with pytest.raises(R.ChainError):
        R.resume_point(state, steps, 3, out=tmp_path)
    # without `out` the old rule holds
    with pytest.raises(R.ChainError):
        R.resume_point(state, steps, 3)
