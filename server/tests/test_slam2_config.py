"""claude_stac_slam2.txt, phase 0 — the two OPTIONAL sections of the VGGT-SLAM 2.0 work,
``correction_graph.seams`` and ``loops.attention_gate``, on the server side.

Rule 5 of the prompt: ABSENT = off = the fork's f576cbf products bit for bit; PRESENT = every
subkey mandatory (no default in code); the map worker writes a section into the fork config
ONLY when the server config turns it on, so the reference's vggt_omega_config.yaml does not
change by a byte. CPU only."""
import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SERVER = Path(__file__).resolve().parents[1]
for p in (SERVER, SERVER / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reconstruction.loops.config import (LoopsConfigError, fork_loop_salad,            # noqa: E402
                                         fork_loops_attention_gate, fork_model_seams,
                                         gate_salad_similarity_threshold, load_loops_config)
from synth_metric import raw_server_cfg                                                # noqa: E402

SEAMS_ON = {"focal_model": "common_K", "common_K_source": "session_median", "judge_samples": 20000}
CALIBRATION = {"select": "max_auc_lower_ci", "target_precision": 0.95, "min_labels": 5,
               "n_negatives": 50, "n_boot": 2000, "confidence": 0.95,
               "salad_grid": [0.55, 0.60, 0.65, 0.70]}
GATE_ON = {"mode": "gate", "layer": 4, "min_score": 0.5, "top_fraction": 0.25,
           "salad_similarity_threshold": None, "calibration": CALIBRATION}


def _raw(seams=None, gate=None):
    raw = raw_server_cfg()
    if seams is not None:
        raw["correction_graph"]["seams"] = copy.deepcopy(seams)
    if gate is not None:
        raw["loops"]["attention_gate"] = copy.deepcopy(gate)
    return raw


def _fork_cfg():
    """A representative fork config, as the map worker holds it right after the SALAD block."""
    return {"Model": {"chunk_ranges": [[0, 60], [30, 90]], "loop_enable": True,
                      "metric_lock": {"enable": True, "anchor_dir": "/x"},
                      "loops": {"anchors_per_bridge": 3, "stac_server_dir": "/srv"},
                      "scale": {"sigma_loop": 0.005}},
            "Loop": {"SALAD": {"similarity_threshold": 0.65}}}


def _helper():
    from workers.map_worker import _apply_slam2_sections
    return _apply_slam2_sections


def _bytes(cfg):
    return yaml.dump(cfg, default_flow_style=False).encode()


def test_absent_sections_are_off_and_write_nothing():
    mg = load_loops_config(_raw())
    assert mg.seams is None and mg.loops.attention_gate is None
    assert fork_model_seams(mg) is None and fork_loops_attention_gate(mg) is None
    assert gate_salad_similarity_threshold(mg) is None
    before = _bytes(_fork_cfg())
    cfg = _fork_cfg()
    assert _helper()(cfg, mg) is cfg
    assert _bytes(cfg) == before, "an absent section changed the fork config"
    assert fork_loop_salad(mg)["similarity_threshold"] == 0.65


def test_present_but_off_sections_write_nothing_either():
    seams_off = dict(SEAMS_ON, focal_model="none")
    gate_off = dict(GATE_ON, mode="off", salad_similarity_threshold=0.60)
    mg = load_loops_config(_raw(seams=seams_off, gate=gate_off))
    assert mg.seams is not None and mg.seams.focal_model == "none"
    assert mg.loops.attention_gate is not None and mg.loops.attention_gate.mode == "off"
    assert fork_model_seams(mg) is None and fork_loops_attention_gate(mg) is None
    # the SALAD bar is replaced only with the gate ON
    assert gate_salad_similarity_threshold(mg) is None
    assert fork_loop_salad(mg)["similarity_threshold"] == 0.65
    before = _bytes(_fork_cfg())
    cfg = _fork_cfg()
    _helper()(cfg, mg)
    assert _bytes(cfg) == before


def test_on_sections_reach_the_fork_config():
    gate_on = dict(GATE_ON, salad_similarity_threshold=0.60)
    mg = load_loops_config(_raw(seams=SEAMS_ON, gate=gate_on))
    assert fork_model_seams(mg) == SEAMS_ON
    g = fork_loops_attention_gate(mg)
    assert g["mode"] == "gate" and g["layer"] == 4 and g["min_score"] == 0.5
    assert g["top_fraction"] == 0.25 and g["salad_similarity_threshold"] == 0.60
    assert g["calibration"]["salad_grid"] == [0.55, 0.60, 0.65, 0.70]
    assert isinstance(g["calibration"]["salad_grid"], list)       # yaml-safe: no tuples
    assert gate_salad_similarity_threshold(mg) == 0.60
    assert fork_loop_salad(mg)["similarity_threshold"] == 0.60
    cfg = _fork_cfg()
    _helper()(cfg, mg)
    assert cfg["Model"]["seams"] == SEAMS_ON
    assert cfg["Model"]["loops"]["attention_gate"] == g
    # gate on, bar null: the SALAD bar stays
    mg2 = load_loops_config(_raw(gate=GATE_ON))
    assert fork_loop_salad(mg2)["similarity_threshold"] == 0.65
    assert fork_loops_attention_gate(mg2)["salad_similarity_threshold"] is None


@pytest.mark.parametrize("seams, gate, named", [
    ({"focal_model": "common_K", "judge_samples": 20000}, None, "correction_graph.seams.common_K_source"),
    ({"common_K_source": "session_median", "judge_samples": 20000}, None, "correction_graph.seams.focal_model"),
    ({"focal_model": "common_K", "common_K_source": "session_median"}, None, "correction_graph.seams.judge_samples"),
    ({"focal_model": "affine", "common_K_source": "session_median", "judge_samples": 20000}, None,
     "correction_graph.seams.focal_model"),
    (None, {k: v for k, v in GATE_ON.items() if k != "min_score"}, "loops.attention_gate.min_score"),
    (None, {k: v for k, v in GATE_ON.items() if k != "salad_similarity_threshold"},
     "loops.attention_gate.salad_similarity_threshold"),
    (None, {k: v for k, v in GATE_ON.items() if k != "calibration"}, "loops.attention_gate.calibration"),
    (None, dict(GATE_ON, calibration={k: v for k, v in CALIBRATION.items() if k != "target_precision"}),
     "loops.attention_gate.calibration.target_precision"),
    (None, dict(GATE_ON, mode="verify"), "loops.attention_gate.mode"),
    (None, dict(GATE_ON, salad_similarity_threshold="0.6"), "loops.attention_gate.salad_similarity_threshold"),
    (None, dict(GATE_ON, calibration=dict(CALIBRATION, salad_grid=[])),
     "loops.attention_gate.calibration.salad_grid"),
])
def test_a_present_section_with_a_missing_or_bad_subkey_fails_by_name(seams, gate, named):
    with pytest.raises(LoopsConfigError) as e:
        load_loops_config(_raw(seams=seams, gate=gate))
    assert named in str(e.value), str(e.value)


def test_a_section_that_is_not_a_mapping_is_refused():
    with pytest.raises(LoopsConfigError, match="correction_graph.seams must be a mapping"):
        load_loops_config(_raw(seams="common_K"))
    with pytest.raises(LoopsConfigError, match="loops.attention_gate must be a mapping"):
        load_loops_config(_raw(gate=True))


def test_the_real_config_has_both_sections_absent_and_documented():
    text = (SERVER / "config.yaml").read_text()
    raw = yaml.safe_load(text)
    assert "seams" not in raw["correction_graph"] and "attention_gate" not in raw["loops"]
    assert "# seams:" in text and "# attention_gate:" in text
    mg = load_loops_config(raw)
    assert mg.seams is None and mg.loops.attention_gate is None


def test_the_omega_builder_writes_the_sections_through_the_one_helper():
    src = (SERVER / "workers" / "map_worker.py").read_text()
    tree = ast.parse(src)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "_apply_slam2_sections"
               for n in ast.walk(tree))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "_apply_slam2_sections"]
    assert len(calls) == 1, "one writer of the slam2 sections (the same discipline as _apply_stac_model_keys)"
    body = src[src.index("def _run_vggtomega("):]
    i_salad = body.index('["SALAD"] = fork_loop_salad(_mg)')
    i_call = body.index("_apply_slam2_sections(cfg_v, _mg,")
    assert i_salad < i_call, "the sections are written after the SALAD block, inside the chunked-metric path"


# ── phase 1: the unaligned chunk npys survive for the offline seam instrument ──────────────

def test_keep_unaligned_chunks_reaches_the_fork_only_when_on():
    import copy as _copy
    raw = _raw()
    assert raw["certify"]["keep_unaligned_chunks"] is False
    mg = load_loops_config(raw)
    before = _bytes(_fork_cfg())
    cfg = _fork_cfg()
    _helper()(cfg, mg)
    assert _bytes(cfg) == before and "keep_unaligned_chunks" not in cfg["Model"]
    raw_on = _copy.deepcopy(raw)
    raw_on["certify"]["keep_unaligned_chunks"] = True
    cfg = _fork_cfg()
    _helper()(cfg, load_loops_config(raw_on))
    assert cfg["Model"]["keep_unaligned_chunks"] is True
    with pytest.raises(LoopsConfigError, match="certify.keep_unaligned_chunks"):
        load_loops_config(_raw(seams=None, gate=None) | {"certify": {k: v for k, v in raw["certify"].items()
                                                                      if k != "keep_unaligned_chunks"}})


def test_the_real_config_keeps_the_unaligned_chunks_off_by_default():
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())
    assert raw["certify"]["keep_unaligned_chunks"] is False


@pytest.mark.parametrize("keep", [True, False])
def test_the_cleanup_spares_the_unaligned_chunks_when_asked(tmp_path, keep):
    from workers import map_worker as MW
    save, out = tmp_path / "maplong_run", tmp_path / "output"
    for d in ("_tmp_results_unaligned", "_tmp_results_loop", "pcd", "uncert", "_tmp_results_aligned"):
        (save / d).mkdir(parents=True)
        (save / d / "x.bin").write_bytes(b"x" * 8)
    out.mkdir()
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())
    raw["certify"]["keep_unaligned_chunks"] = keep
    logs = []

    class _Pipe:
        def send_log(self, m, level="info"):
            logs.append(str(m))

        def send_progress(self, *a, **k):
            pass

    MW._cleanup_recon_temps(save, out, "mapanything", _Pipe(), raw)
    assert (save / "_tmp_results_unaligned").exists() is keep
    assert not (save / "pcd").exists() and (save / "_tmp_results_aligned").exists()
    assert any("_tmp_results_unaligned KEPT" in m for m in logs) is keep


# ── phase 2: the intake probe's K reaches the fork on the Ω grid, only when the session has it ──

def _probe_session(tmp_path, K_native, native_wh):
    sess = tmp_path / "session"
    (sess / "intake").mkdir(parents=True)
    (sess / "intake" / "focal_probe.json").write_text(json.dumps(
        {"version": 2, "K": K_native.tolist(), "fx": K_native[0, 0], "fy": K_native[1, 1],
         "cx": K_native[0, 2], "cy": K_native[1, 2], "native_wh": list(native_wh),
         "frames": ["000000.jpg", "000100.jpg"], "spread_pct": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}}))
    return sess


def test_the_probe_K_is_carried_to_the_omega_grid_or_declared_absent(tmp_path):
    import json as _json
    from precision.camera import K_native_to_grid, omega_grid_for
    from reconstruction.loops.config import intake_probe_K_for_omega
    K = np.array([[1500.0, 0.0, 959.5], [0.0, 1500.0, 539.5], [0.0, 0.0, 1.0]])
    grid = omega_grid_for(1920, 1080, "max_size", 832)
    sess = _probe_session(tmp_path, K, (1920, 1080))
    out = intake_probe_K_for_omega(sess, grid)
    assert out["K"] is not None and np.allclose(out["K"], K_native_to_grid(K, grid))
    assert out["record"]["source"] == "intake/focal_probe.json" and out["record"]["grid_wh"] == [grid.w, grid.h]
    assert intake_probe_K_for_omega(tmp_path / "nowhere", grid)["K"] is None
    other = intake_probe_K_for_omega(sess, omega_grid_for(464, 832, "max_size", 832))
    assert other["K"] is None and "1920x1080" in other["reason"]


def test_the_section_carries_the_probe_K_only_for_the_intake_source(tmp_path):
    from precision.camera import omega_grid_for
    K = np.array([[1500.0, 0.0, 959.5], [0.0, 1500.0, 539.5], [0.0, 0.0, 1.0]])
    grid = omega_grid_for(1920, 1080, "max_size", 832)
    sess = _probe_session(tmp_path, K, (1920, 1080))
    seams_probe = dict(SEAMS_ON, common_K_source="intake_probe")
    logs = []
    cfg = _fork_cfg()
    _helper()(cfg, load_loops_config(_raw(seams=seams_probe)), session_dir=sess, omega_grid=grid, log=logs.append)
    se = cfg["Model"]["seams"]
    assert se["common_K_source"] == "intake_probe" and se["judge_samples"] == 20000
    assert np.asarray(se["intake_probe_K"]).shape == (3, 3) and se["intake_probe"]["native_wh"] == [1920, 1080]
    # no probe for the session: the section goes without the K and the worker says so
    cfg2 = _fork_cfg()
    _helper()(cfg2, load_loops_config(_raw(seams=seams_probe)), session_dir=tmp_path / "none", omega_grid=grid,
              log=logs.append)
    assert "intake_probe_K" not in cfg2["Model"]["seams"]
    assert any("no probe K for this session" in m for m in logs)
    # the session median source never carries it
    cfg3 = _fork_cfg()
    _helper()(cfg3, load_loops_config(_raw(seams=SEAMS_ON)), session_dir=sess, omega_grid=grid, log=logs.append)
    assert "intake_probe_K" not in cfg3["Model"]["seams"]
