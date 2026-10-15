"""point2cad for objects (USER 2026-10-15 "dale, probemos"): the CAD chooser decides by GEOMETRY
(the share of the object's points its analytic primitives explain), the worker tolerates a missing
VLM proposal, the keys are configured. CPU only."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from segmentation.p2c_object import cad_candidate  # noqa: E402


def _region(kind, n):
    return {"kind": kind, "model": None, "f_idx": np.arange(n), "v_idx": np.arange(n), "notes": []}


def test_the_chooser_counts_primitives_only_and_records_its_margin():
    cfg = {"p2c_min_explained_frac": 0.5}
    ok, rec = cad_candidate([_region("plane", 600), _region("cylinder", 100), _region("freeform", 250)], 1000, cfg)
    assert ok is True and rec["explained_frac"] == 0.7 and rec["freeform_frac"] == 0.25
    assert rec["unlabeled_frac"] == 0.05 and rec["n_primitives"] == 2 and rec["n_freeform"] == 1
    assert rec["margin"] == 0.2 and rec["kinds"] == ["cylinder", "plane"] and "never by label" in rec["rule"]
    ok, rec = cad_candidate([_region("plane", 300), _region("freeform", 650)], 1000, cfg)
    assert ok is False and rec["explained_frac"] == 0.3 and rec["margin"] == -0.2
    assert cad_candidate([_region("plane", 500)], 1000, cfg)[0] is True          # at the bar: a candidate
    assert cad_candidate([], 1000, cfg) == (False, cad_candidate([], 1000, cfg)[1])


def test_the_keys_are_configured_and_the_mesh_button_runs_p2c():
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())["surface_fit"]
    assert raw["mesh_runs_p2c"] is True and 0.0 < raw["p2c_min_explained_frac"] <= 1.0
    assert raw["p2c_propose"] is True and raw["p2c_from_cloud"] is True


def test_the_worker_runs_point2cad_without_a_vlm_proposal(tmp_path, monkeypatch):
    """The Mesh button must not depend on the chat's vLLM: a failing proposal is declared and
    point2cad still runs."""
    import run_perfect_objects as W
    out = tmp_path / "output"
    out.mkdir()
    (out / "segmentation_result.json").write_text(json.dumps({"instances": [{"instance_id": 7, "id": 7, "label": "desk"}]}))
    calls = []
    import segmentation.shape_proposer as SP
    import segmentation.p2c_object as P2C

    def boom(*a, **k):
        raise ConnectionError("semantic service down")

    monkeypatch.setattr(SP, "propose_object", boom)
    monkeypatch.setattr(P2C, "build_p2c_object", lambda od, iid, cfg=None, source=None, log=print: calls.append(iid) or (out / "x.glb"))
    monkeypatch.setattr(sys, "argv", ["run_perfect_objects.py", "--output-dir", str(out), "--mode", "p2c", "--instance-id", "7"])
    rc = W.main()
    assert rc == 0 and calls == [7]
