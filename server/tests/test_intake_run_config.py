"""The run's configuration frozen at job start (docs/plan_determinismo.md point 69):
output/run_config.yaml + its sha256, canonical (the same bytes for the same configuration,
whatever the key order), verified on every read (an edited copy is refused), the stage
overrides recorded beside the job's configuration, the intake refusing a configuration that
is not the frozen one, the CLI freezing its own, and the pipeline manager freezing after the
replace wipe and before the first stage."""

import ast
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import run_config as RC                             # noqa: E402
from intake.config import load_intake_config                    # noqa: E402

SERVER = Path(__file__).resolve().parents[1]


def _raw():
    with open(SERVER / "config.yaml") as f:
        return yaml.safe_load(f)


def test_freeze_writes_canonical_yaml_with_its_digest(tmp_path):
    raw = _raw()
    logs = []
    rec = RC.freeze_run_config(tmp_path, raw, log=logs.append)
    p = RC.run_config_path(tmp_path)
    assert rec["path"] == "output/run_config.yaml" and p.exists()
    data = p.read_bytes()
    import hashlib
    assert rec["sha256"] == hashlib.sha256(data).hexdigest()
    assert RC.run_config_sha_path(tmp_path).read_text().strip() == rec["sha256"]
    assert any("frozen" in m for m in logs)
    # the same configuration in another key order: the same bytes
    shuffled = {k: raw[k] for k in sorted(raw, reverse=True)}
    rec2 = RC.freeze_run_config(tmp_path, shuffled, log=lambda m: None)
    assert rec2["sha256"] == rec["sha256"] and p.read_bytes() == data
    # round trip: what is read is what was frozen
    doc, sha = RC.load_run_config(tmp_path)
    assert sha == rec["sha256"] and doc == json.loads(json.dumps(raw))
    assert yaml.safe_load(data.decode()) == doc


def test_load_refuses_an_edited_or_unverifiable_copy(tmp_path):
    RC.freeze_run_config(tmp_path, {"a": 1}, log=lambda m: None)
    p = RC.run_config_path(tmp_path)
    p.write_text(p.read_text() + "b: 2\n")
    with pytest.raises(RC.RunConfigError, match="edited after it was frozen"):
        RC.load_run_config(tmp_path)
    RC.freeze_run_config(tmp_path, {"a": 1}, log=lambda m: None)
    RC.run_config_sha_path(tmp_path).unlink()
    with pytest.raises(RC.RunConfigError, match="cannot be verified"):
        RC.load_run_config(tmp_path)
    with pytest.raises(RC.RunConfigError, match="does not exist"):
        RC.load_run_config(tmp_path / "nowhere")
    assert not RC.has_run_config(tmp_path / "nowhere")
    with pytest.raises(RC.RunConfigError, match="not a plain YAML"):
        RC.freeze_run_config(tmp_path, {"a": object()}, log=lambda m: None)


def test_stage_overrides_are_recorded_and_applied(tmp_path):
    raw = {"intake": {"x": 1}, "other": 2}
    RC.freeze_run_config(tmp_path, raw, stages={"vlm": {"other": 3}, "sam3": {}},
                         log=lambda m: None)
    doc, _ = RC.load_run_config(tmp_path)
    assert doc[RC.STAGES_KEY] == {"vlm": {"other": 3}}
    assert RC.effective_config(doc) == raw
    assert RC.effective_config(doc, "vlm") == {"intake": {"x": 1}, "other": 3}
    assert RC.effective_config(doc, "sam3") == raw


def test_verify_intake_config_names_the_field_that_differs(tmp_path):
    raw = _raw()
    icfg = load_intake_config(raw)
    rec = RC.freeze_run_config(tmp_path, raw, log=lambda m: None)
    assert RC.verify_intake_config(tmp_path, icfg) == rec["sha256"]
    other = replace(icfg, parallax=replace(icfg.parallax, grid_side=icfg.parallax.grid_side + 1))
    with pytest.raises(RC.RunConfigError, match=r"parallax\.grid_side: frozen"):
        RC.verify_intake_config(tmp_path, other)
    d = RC.intake_config_differences(icfg, other)
    assert list(d) == ["parallax.grid_side"]
    # a typed configuration back to a raw one whose load IS it
    back = RC.raw_with_intake(raw, other)
    assert load_intake_config(back) == other
    assert back["models"]["segmentation"]["batch_size"] == other.content.sam3_batch
    RC.freeze_run_config(tmp_path, back, log=lambda m: None)
    assert RC.verify_intake_config(tmp_path, other)


def test_cli_intake_config_freezes_the_servers_or_reads_the_frozen(tmp_path, monkeypatch):
    import config as server_config
    raw = _raw()
    monkeypatch.setattr(server_config, "cfg", raw)
    logs = []
    icfg, sha = RC.cli_intake_config(tmp_path, log=logs.append)
    assert icfg == load_intake_config(raw) and RC.has_run_config(tmp_path)
    assert sha == RC.load_run_config(tmp_path)[1] and any("frozen" in m for m in logs)
    # a second CLI reads the frozen copy even if the server's configuration moved on
    raw2 = json.loads(json.dumps(raw))
    raw2["intake"]["parallax"]["grid_side"] += 1
    monkeypatch.setattr(server_config, "cfg", raw2)
    icfg2, sha2 = RC.cli_intake_config(tmp_path, log=logs.append)
    assert icfg2 == icfg and sha2 == sha and any("reading the frozen" in m for m in logs)


def test_pipeline_manager_freezes_after_the_wipe_and_before_the_first_stage():
    """The manager's _run_pipeline: freeze_run_config runs once, after the replace wipe
    (_wipe_outputs_for_replace) and before the stage loop (_run_stage)."""
    src = (SERVER / "pipeline_manager.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_pipeline")
    calls = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in ("_wipe_outputs_for_replace", "freeze_run_config", "_run_stage"):
                calls.append((node.lineno, name))
    calls.sort()
    names = [n for _, n in calls]
    assert names.count("freeze_run_config") == 1
    assert names.index("_wipe_outputs_for_replace") < names.index("freeze_run_config") \
        < names.index("_run_stage")
    body = src[src.index("async def _run_pipeline"):src.index("async def _run_stage")]
    assert "freeze_run_config(" in body and "session_dir, config" in body
