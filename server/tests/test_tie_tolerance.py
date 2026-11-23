"""No F2 (USER 2026-11-22: "parala, y lanzala sin F2"): the chain without f2_gauge only when the switch is off, and
F5's anchor tolerance from the tie's seams (the median of residual / lever) written as the gauge.json F5 reads."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from precision.config import load_precision_config
from precision.runner import chain_steps
from precision import tie_tolerance as TT

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_the_chain_drops_f2_only_when_it_is_switched_off():
    raw = yaml.safe_load(CONFIG.read_text())
    assert raw["reconstruction"]["precision"]["gauge"]["enabled"] is False     # the main pipeline (USER 2026-11-23)
    on_raw, off_raw, absent_raw = copy.deepcopy(raw), copy.deepcopy(raw), copy.deepcopy(raw)
    on_raw["reconstruction"]["precision"]["gauge"]["enabled"] = True
    off_raw["reconstruction"]["precision"]["gauge"]["enabled"] = False
    del absent_raw["reconstruction"]["precision"]["gauge"]["enabled"]          # a session frozen before the key
    on = [s.key for s in chain_steps(load_precision_config(on_raw))]
    off = [s.key for s in chain_steps(load_precision_config(off_raw))]
    absent = [s.key for s in chain_steps(load_precision_config(absent_raw))]
    assert "f2_gauge" in on and absent == on
    assert off == [k for k in on if k != "f2_gauge"]


def test_the_tolerance_is_the_median_of_the_seams_relative_errors(tmp_path):
    (tmp_path / "maplong_run").mkdir()
    seams = [{"a": 1, "b": 0, "sigma_m": 0.03, "lever_m": 2.0}, {"a": 2, "b": 1, "sigma_m": 0.04, "lever_m": 1.0},
             {"a": 3, "b": 2, "sigma_m": 0.02, "lever_m": 1.0}]
    (tmp_path / "maplong_run" / "chunk_graph.json").write_text(json.dumps({"seams": seams}))
    sigma = TT.write_tolerance_from_seams(tmp_path, log=lambda m: None)
    assert sigma == pytest.approx(0.02)                 # relative 0.015, 0.04, 0.02 → median 0.02
    doc = json.loads((tmp_path / "gauge.json").read_text())
    assert doc["applied_instrument"] == TT.INSTRUMENT
    assert doc["sigma_by_instrument"][doc["applied_instrument"]] == pytest.approx(0.02)
    mtime = (tmp_path / "gauge.json").stat().st_mtime_ns
    TT.write_tolerance_from_seams(tmp_path, log=lambda m: None)        # unchanged content: not rewritten
    assert (tmp_path / "gauge.json").stat().st_mtime_ns == mtime


def test_one_cube_has_no_seam_and_no_tolerance(tmp_path):
    (tmp_path / "maplong_run").mkdir()
    (tmp_path / "maplong_run" / "chunk_graph.json").write_text(json.dumps({"seams": []}))
    with pytest.raises(TT.TieToleranceError):
        TT.write_tolerance_from_seams(tmp_path, log=lambda m: None)
