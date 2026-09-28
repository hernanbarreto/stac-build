"""The precision core as a pipeline stage: the one step list, in order, resumed after
the last finished step while the session is still in the epoch it left."""

from __future__ import annotations

import json
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


def test_the_stage_is_wired_after_sam3():
    from pipeline_manager import DEFAULT_STAGE_ORDER, STAGE_REGISTRY, StageId
    o = DEFAULT_STAGE_ORDER
    assert o.index(StageId.SAM3) < o.index(StageId.PRECISION) < o.index(StageId.CERTIFY)
    assert STAGE_REGISTRY[StageId.PRECISION]["module"] == "workers.precision_worker"
    import workers.precision_worker as W
    assert callable(W.run)
    assert [s.module for s in RN.STEPS][-1] == "precision.fuse"
