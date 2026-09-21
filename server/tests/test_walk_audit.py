"""The walk that decides the structure of the reconstruction, and the session
that has only one unit to correct with.

Two findings of 2026-09-21, both about a number that DECIDES and was not said
out loud:

  * FINDING 25 — three readings of ONE 216-keyframe walk of one scan: 44.1 m
    and <=15 m from the phase-1 probe on two runs, 19.3 m from `chainage()` on
    the geometry delivered. The probe's reading is what `map_worker` weighs
    against `reconstruction.simple.max_walk_single_pass_m`, and on the run that
    read under the limit NOTHING was logged at all — the run whose structure
    differed from the other two is the one the log could not explain.
  * FINDING 24b — a single-pass session has no `chunk_plan.json`, so the depth
    graph has ONE unit and solves ONE factor: a global scale change, not a
    drift correction. It used to publish that factor as the correction.

These tests pin the properties, not the wording: a line on every decision path,
the two walk measurements made by the same function, an alert whenever the two
readings could have decided differently, and the declared limit present in the
report that reaches the acta.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── FINDING 25: the walk is ONE measurement ──────────────────────────────

def _poses_txt(path: Path, centres, with_index: bool = False) -> Path:
    """camera_poses.txt in the layout the worker writes: one flattened 4x4
    c2w per line, optionally prefixed by the frame index."""
    lines = []
    for i, c in enumerate(centres):
        T = np.eye(4)
        T[:3, 3] = c
        flat = " ".join(f"{v!r}" for v in T.reshape(-1))
        lines.append(f"{i} {flat}" if with_index else flat)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("with_index", [False, True])
def test_walk_is_the_correction_s_own_chainage(tmp_path, with_index):
    """The probe and the correction measure the walk with the SAME function.

    `chainage()` is the definition of record (`reconstruction/loops/drift.py`);
    the worker's reading has to BE its last entry, not another implementation
    of the same sum, or the confrontation below compares two codebases instead
    of two geometries.
    """
    from reconstruction.loops.drift import chainage
    from workers.map_worker import _walk_chainage_m

    rng = np.random.default_rng(25)
    centres = np.cumsum(rng.normal(0, 0.4, size=(40, 3)), axis=0)
    p = _poses_txt(tmp_path / "camera_poses.txt", centres, with_index)
    assert _walk_chainage_m(p) == pytest.approx(float(chainage(centres)[-1]), rel=1e-12)


def test_walk_of_a_known_path(tmp_path):
    """An L of 3 m + 4 m walks 7 m — the number is not an opinion."""
    from workers.map_worker import _walk_chainage_m
    centres = np.array([[0.0, 0, 0], [3.0, 0, 0], [3.0, 0, 4.0]])
    p = _poses_txt(tmp_path / "camera_poses.txt", centres)
    assert _walk_chainage_m(p) == pytest.approx(7.0, abs=1e-9)


def test_a_single_keyframe_walks_nothing(tmp_path):
    from workers.map_worker import _walk_chainage_m
    p = _poses_txt(tmp_path / "camera_poses.txt", np.zeros((1, 3)))
    assert _walk_chainage_m(p) == 0.0


# ── FINDING 25: every decision path leaves a line ────────────────────────

_PATHS = [
    # (walk, comfort, n_kf, chunked_already, expected re-run)
    (19.3, 15.0, 216, False, True),    # pccr: over the limit, phase 2
    (12.7, 15.0, 216, False, False),   # the 10:00 run: under the limit, silent until now
    (44.1, 15.0, 10, False, False),    # over the limit, fewer kf than one chunk
    (44.1, 15.0, 216, True, False),    # the frame count already chunked it
]


@pytest.mark.parametrize("walk,comfort,n_kf,chunked,expected", _PATHS)
def test_every_phase2_path_says_what_it_decided(walk, comfort, n_kf, chunked, expected):
    from workers.map_worker import _phase2_decision
    go, msg, level = _phase2_decision(walk, comfort, n_kf, chunked)
    assert go is expected
    assert level in ("info", "warning")
    # the line carries BOTH numbers the decision was taken on
    assert f"{walk:.1f}" in msg and f"{comfort:g}" in msg
    assert msg.startswith("[chunk-plan] ")


def test_the_single_pass_path_is_no_longer_silent():
    """The run that kept one chunk used to leave no line at all."""
    from workers.map_worker import _phase2_decision
    go, msg, _ = _phase2_decision(12.7, 15.0, 216, False)
    assert go is False
    assert "SINGLE PASS" in msg and "within comfort" in msg


def test_too_few_keyframes_to_chunk_is_a_warning():
    """Chunking below the planner's own `min_size` returns ONE chunk: the walk
    is over the limit and nothing can be done about it — that is a warning, not
    a quiet pass."""
    from workers.map_worker import _MIN_CHUNK_KEYFRAMES, _phase2_decision
    go, msg, level = _phase2_decision(44.1, 15.0, _MIN_CHUNK_KEYFRAMES - 1, False)
    assert go is False and level == "warning"
    assert str(_MIN_CHUNK_KEYFRAMES) in msg


# ── FINDING 25: the two measurements are confronted ──────────────────────

def test_pccr_probe_against_the_geometry_it_produced():
    """44.1 m of probe over geometry that measures 19.3 m: the decision came
    out right (19.3 also exceeds 15) BY LUCK, and the log has to say so."""
    from workers.map_worker import _walk_confrontation
    msg, level = _walk_confrontation(44.1, 19.3, 15.0, chunked=True)
    assert level == "warning"
    assert "ALERT" in msg and "luck" in msg
    assert "44.1" in msg and "19.3" in msg and "2.28x" in msg


def test_a_disagreement_that_would_have_flipped_the_decision():
    """Probe over the limit, delivered geometry under it: the two readings do
    not even agree on which side of the limit the session is."""
    from workers.map_worker import _walk_confrontation
    msg, level = _walk_confrontation(16.0, 14.0, 15.0, chunked=True)
    assert level == "warning"
    assert "OPPOSITE sides" in msg


def test_two_readings_that_agree_are_not_an_alert():
    from workers.map_worker import _walk_confrontation
    msg, level = _walk_confrontation(19.3, 19.8, 15.0, chunked=True)
    assert level == "info"
    assert "ALERT" not in msg and "agree" in msg


def test_a_single_pass_session_still_gets_its_audit_line():
    from workers.map_worker import _walk_confrontation
    msg, level = _walk_confrontation(12.7, 12.7, 15.0, chunked=False)
    assert level == "info" and "single pass" in msg


# ── FINDING 25: the comfort limit comes from config.yaml, from nowhere else ──

def test_the_comfort_limit_has_no_second_opinion_in_the_code():
    """The `.get(..., 25.0)` fallback disagreed with the 15.0 the file carries,
    and the in-code comment beside it still said "25 m default" while the log
    line printed the config value."""
    src = (ROOT / "workers" / "map_worker.py").read_text()
    assert 'max_walk_single_pass_m", 25.0' not in src
    assert "25 m default" not in src
    assert "reconstruction.simple.max_walk_single_pass_m" in src


def test_config_declares_the_limit_the_worker_now_demands():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["reconstruction"]["simple"].get("max_walk_single_pass_m") is not None


# ── FINDING 24b: one unit cannot carry drift, and says so ────────────────

def _scale_cfg():
    """The session's own σ and caps — not numbers invented for a test."""
    from reconstruction.loops.config import load_loops_config
    return load_loops_config().certify.scale


def test_single_unit_limit_fires_only_on_one_chunk(tmp_path):
    from reconstruction.certify.scale_stage import single_unit_limit
    d = single_unit_limit(tmp_path, 1)
    assert d is not None and d["limit"] == "single_unit" and d["n_chunks"] == 1
    assert "no chunk_plan.json" in d["source"]
    assert "OUT OF REACH" in d["detail"]
    assert single_unit_limit(tmp_path, 7) is None


def test_single_unit_limit_names_a_one_range_plan(tmp_path):
    (tmp_path / "chunk_plan.json").write_text(json.dumps(
        {"version": 1, "phase": "direct-chunked", "n_keyframes": 40,
         "chunk_size": 40, "overlap": 0, "chunk_ranges": [[0, 40]]}))
    from reconstruction.certify.scale_stage import single_unit_limit
    d = single_unit_limit(tmp_path, 1)
    assert d is not None and "one range" in d["source"]


def _session(n_kf: int):
    return SimpleNamespace(n_kf=n_kf, frames=list(range(n_kf)))


def test_a_single_pass_session_declares_the_limit_in_its_report(tmp_path):
    """No rows at all: the stage returns identity, and the declaration is in
    the report anyway — the limit is a property of the session, not of whether
    something was applied."""
    from reconstruction.certify.scale_stage import solve_scale_stage
    lines = []
    rep = solve_scale_stage(tmp_path, _session(30), [], _scale_cfg(), log=lines.append)
    assert rep["n_chunks"] == 1
    assert [d["limit"] for d in rep["declared_limits"]] == ["single_unit"]
    assert any("DECLARED LIMIT" in ln for ln in lines)


def test_the_global_factor_is_applied_but_never_called_a_drift_correction(tmp_path):
    """An absolute row gives the one-unit graph something to solve. The factor
    is applied — nothing is skipped and nothing is substituted — and the report
    still carries the declaration beside it."""
    (tmp_path / "scale_absolute_rows.json").write_text(json.dumps(
        {"rows": [{"chunk": 0, "log_s": 0.05, "sigma": 0.01, "source": "regulated"}]}))
    from reconstruction.certify.scale_stage import solve_scale_stage
    lines = []
    rep = solve_scale_stage(tmp_path, _session(30), [], _scale_cfg(), log=lines.append)
    assert rep["applied"] is True
    assert len(rep["r"]) == 1 and rep["r"][0] == pytest.approx(np.exp(0.05), rel=1e-3)
    assert [d["limit"] for d in rep["declared_limits"]] == ["single_unit"]


def test_a_chunked_session_declares_nothing(tmp_path):
    (tmp_path / "chunk_plan.json").write_text(json.dumps(
        {"version": 1, "phase": "chunked-metric", "n_keyframes": 216,
         "chunk_size": 59, "overlap": 29,
         "chunk_ranges": [[0, 59], [30, 89], [60, 119], [90, 149],
                          [120, 179], [150, 209], [180, 216]]}))
    from reconstruction.certify.scale_stage import solve_scale_stage
    rep = solve_scale_stage(tmp_path, _session(216), [], _scale_cfg(), log=lambda _m: None)
    assert rep["n_chunks"] == 7
    assert rep["declared_limits"] == []


def test_the_declaration_reaches_the_stage_record_the_acta_reads(tmp_path, monkeypatch):
    """`certify/run.py` copies `vd["stages"]` into `acta["correction"]`, so a
    limit that stops at the log stops before the acta."""
    import correction.run as crun
    import correction.visit_drift_run as vdr

    limit = {"limit": "single_unit", "n_chunks": 1, "source": "no chunk_plan.json",
             "detail": "one factor is an average, not a drift correction"}
    n = 12
    monkeypatch.setattr(vdr, "solve_depth", lambda *a, **k: (
        np.full(n, 1.02), np.zeros((n, 3)),
        {"r": [1.02], "applied": True, "declared_limits": [limit]}))
    monkeypatch.setattr(crun, "run_floor",
                        lambda *a, **k: {"status": "applied", "correction_id": "c1"})

    out = vdr.run(tmp_path, log=lambda _m: None, cfg=object())
    depth = [s for s in out["stages"] if s["stage"] == "depth"]
    assert len(depth) == 1
    assert depth[0]["declared_limits"] == [limit]
