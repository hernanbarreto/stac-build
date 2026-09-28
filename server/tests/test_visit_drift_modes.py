"""visit_drift modes (claude_stac.txt §4-F3): `measure` publishes the evidence of the
epoch it runs on — scale_loop_rows.json for the gauge (F2), instance_loops.json for
the correspondence stage (F4) — and `verify` the residual closures; neither writes
an epoch."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import correction.visit_drift_run as V                          # noqa: E402
from tests.synth_correction import make_correction_cfg           # noqa: E402


def _kept():
    c = SimpleNamespace(instance_id=7, label="desk")
    dr = SimpleNamespace(visit_a=(10, 20), visit_b=(150, 161), t=np.array([0.3, 0.0, 0.4]),
                         worst_disagreement=0.012)
    return [(c, dr)]


def test_instance_loops_are_keyframe_pairs_with_the_silhouette_sigma():
    (loop,) = V.instance_loops_of(_kept())
    assert loop["instance_id"] == 7 and loop["label"] == "desk"
    assert (loop["i"], loop["j"]) == (15, 156)
    assert loop["visit_a"] == [10, 20] and loop["visit_b"] == [150, 161]
    assert np.allclose(loop["t_m"], [0.3, 0.0, 0.4]) and loop["sigma_m"] == 0.012
    json.dumps(loop)                                        # JSON-clean


def _fake(monkeypatch, tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    rows = [{"i": 15, "j": 156, "k_b": 1.1, "s_ab": 1 / 1.1, "residual_m": 0.01,
             "D_b_m": 3.0, "instance_id": 7}]
    monkeypatch.setattr(V, "measure_epoch",
                        lambda o, c, r, log=print: {"scale_rows": rows,
                                                    "instance_loops": V.instance_loops_of(_kept()),
                                                    "rejected": []})
    monkeypatch.setattr(V, "_repeatability_m", lambda o, d, log: 0.05)
    applied = []
    monkeypatch.setattr(V, "apply_transform_epoch", lambda *a, **k: applied.append(1))
    return out, applied


def test_measure_publishes_the_evidence_and_writes_no_epoch(tmp_path, monkeypatch):
    out, applied = _fake(monkeypatch, tmp_path)
    before = sorted(p.name for p in out.iterdir())
    rep = V.measure(tmp_path, log=lambda *a: None, cfg=make_correction_cfg())
    assert rep["n_scale_rows"] == 1 and rep["n_instance_loops"] == 1
    loops = json.loads((out / "instance_loops.json").read_text())
    rows = json.loads((out / "scale_loop_rows.json").read_text())
    assert loops["measured_on_epoch"] == 0 == rows["measured_on_epoch"]
    assert loops["loops"][0]["sigma_m"] == 0.012
    assert not applied
    after = sorted(p.name for p in out.iterdir())
    assert set(after) - set(before) == {"instance_loops.json", "scale_loop_rows.json"}
    assert not (out / "geometry_epoch.json").exists()


def test_verify_reports_the_residual_closures_and_writes_no_epoch(tmp_path, monkeypatch):
    out, applied = _fake(monkeypatch, tmp_path)
    rep = V.verify(tmp_path, log=lambda *a: None, cfg=make_correction_cfg())
    assert rep["n_closures"] == 1 and abs(rep["median_residual_m"] - 0.5) < 1e-12
    assert rep["n_within_repeatability"] == 0
    doc = json.loads((out / "visit_drift_verify.json").read_text())
    assert doc["closures"][0]["residual_m"] == rep["median_residual_m"]
    assert not applied and not (out / "geometry_epoch.json").exists()
