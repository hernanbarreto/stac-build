"""A fit has to be POSSIBLE, not only good (finding 27).

Measured on the pccr run of 2026-09-21, epoch 1, written into the pose graph:

    desk#198 kf 204<->2: copies 74.1 -> 11.2 cm, closure |t| 876 cm / 107.7
                         deg, full, sigma 67.0 cm

`measure_copy` chose among the rotated, the translated and the centroid fit by
QUALITY OF FIT alone — the smallest max(nearest-neighbour median, centroid
distance) — and the rotated one won outright; there was no tie to break. But
107.7 deg and 8.76 m over a 19.3 m walk is not drift: `desk#198` holds TWO
DIFFERENT DESKS under one label and the rigid fit "closes" them by putting one
on top of the other. The same pathology is recorded twice in the module before
this (desk#201 at 63.9 deg / 5.44 m, floor#44 at 77.9 deg), each time caught by
a better fit happening to exist.

The session already knows what motion is plausible: the drift budget
delta(L) / theta(L) of `spatial_gate`, over the metres WALKED between the two
visits. These tests pin what it buys: a fit beyond that budget does not win by
fitting better, the pair is NEVER vetoed (USER 2026-09-16: "no debe cortar
objetos"), the refusal travels with both numbers, and the bound is the WALK —
the same fit over a walk long enough to produce it is kept.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify import loops_posthoc as lp                  # noqa: E402
from reconstruction.certify.loops_posthoc import measure_copy           # noqa: E402
from reconstruction.loops import spatial_gate as sg                     # noqa: E402
from reconstruction.loops.config import load_loops_config               # noqa: E402

CFG = load_loops_config()
SCFG, SPATIAL = CFG.certify.scale, CFG.loops.spatial

WALK_M = 19.3                    # pccr's own walk, the one that produced the finding
BUDGET = sg.drift_budget(WALK_M, SPATIAL)


def _slab(n=3000, size=(1.40, 0.70, 0.04), seed=1):
    """A desk top: wide, deep, thin."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) - 0.5) * np.asarray(size)


def _two_desks(angle_deg=107.7, centre=(9.0, 2.0, 1.5)):
    """The shape of `desk#198`: two different desks filed under one label, one
    turned against the other, and BOTH far from the world origin — which is
    where the metres of translation come from, since the fit's t is taken
    about that origin (a rotation about a distant centre moves the object
    centimetres and the frame metres).

    The ICP lands on 32.1° / 5.11 m here — 17.0× the budget of a 19.3 m walk
    — rather than pccr's own 107.7° / 8.76 m at 53.9×: the same shape of
    impossible motion, at the depth of the basin a rectangle offers."""
    c = np.asarray(centre, np.float64)
    b = _slab() + c
    R = Rotation.from_rotvec([0.0, 0.0, np.deg2rad(angle_deg)]).as_matrix()
    a = _slab() @ R.T + c
    return a, b


# ── the state the finding describes ────────────────────────────────────────

def test_the_rotated_fit_wins_the_quality_comparison():
    """The premise, measured: judged on fit alone the rotation wins, and it
    demands a rotation and metres of translation no walk could produce."""
    a, b = _two_desks()
    m = measure_copy(a, b, SCFG)
    assert m["fit_chosen"] == "rotated", m["fit_scores_m"]
    assert m["fit_scores_m"]["rotated"] < m["fit_scores_m"]["translated"]
    assert m["rot_deg_fitted"] > 30.0, m["rot_deg_fitted"]
    assert np.linalg.norm(m["t"]) > 5.0, m["t"]


# ── what the budget buys ───────────────────────────────────────────────────

def test_a_motion_no_drift_could_produce_does_not_win():
    a, b = _two_desks()
    m = measure_copy(a, b, SCFG, budget=BUDGET)
    assert m["fit_chosen"] != "rotated", m["fit_motion"]
    assert not m["rotation_earned"]
    assert np.allclose(np.asarray(m["R"]), np.eye(3)), m["R"]
    # it was refused DESPITE fitting better — the comparison is not quality
    assert m["fit_scores_m"]["rotated"] < m["fit_scores_m"][m["fit_chosen"]]


def test_the_demotion_records_both_numbers():
    """Nothing is dropped silently: the rotation, the translation, the budget
    they were judged against and what each fit scored."""
    a, b = _two_desks()
    d = measure_copy(a, b, SCFG, budget=BUDGET)["fit_demoted"]
    assert d is not None
    assert d["refused"] == "rotated" and d["chosen"] in ("translated", "centroid")
    assert d["refused_rot_deg"] > 30.0 and d["refused_t_norm_m"] > 5.0
    assert d["walk_m"] == pytest.approx(WALK_M)
    assert d["delta_m"] == pytest.approx(BUDGET["delta_m"])
    assert d["theta_deg"] == pytest.approx(BUDGET["theta_deg"])
    # the motion in units of the drift the walk can produce, both sides of it
    assert d["refused_over_budget"] > 1.0 > d["chosen_over_budget"]
    assert d["chosen_over_budget"] < d["refused_over_budget"]
    assert f"{d['refused_rot_deg']:.1f}" in d["why"] and "19.3 m" in d["why"]
    assert "refused" in d["why"] and d["chosen"] in d["why"]


def test_every_candidate_declares_what_it_would_have_moved():
    a, b = _two_desks()
    mo = measure_copy(a, b, SCFG, budget=BUDGET)["fit_motion"]
    assert set(mo) == {"rotated", "translated", "centroid"}
    for r in mo.values():
        assert {"rot_deg", "t_norm_m", "over_budget"} == set(r)
        assert r["over_budget"] is not None
    assert mo["translated"]["rot_deg"] == 0.0 and mo["centroid"]["rot_deg"] == 0.0


def test_the_pair_is_never_vetoed():
    """USER 2026-09-16 ("no debe cortar objetos"): the pair still produces a
    closure, from a fit that is physically possible, and it still reduces the
    separation the eye can see."""
    a, b = _two_desks()
    m = measure_copy(a, b, SCFG, budget=BUDGET)
    assert m is not None
    assert m["offset_after_m"] < m["offset_before_m"]
    assert m["offset_after_m"] == pytest.approx(m["fit_scores_m"][m["fit_chosen"]], abs=1e-3)


def test_a_refused_rotation_writes_no_scale_row():
    """The Sim3 scale belongs to the rotated fit; refused, it has no scale to
    give and must not leave 1.0 dressed as a measurement."""
    a, b = _two_desks()
    m = measure_copy(a, b, SCFG, budget=BUDGET)
    assert m["s_ab"] == 1.0 and m["scale_trusted"] is False


def test_the_bound_is_the_walk_not_the_rotation():
    """Same points, same fit, same rotation — only the walk between the two
    visits changes. Over 600 m the drift budget is 60 deg / 7.8 m and the very
    fit refused above is kept: the rule measures the session, it does not hold
    an opinion about rotations."""
    a, b = _two_desks()
    long_walk = sg.drift_budget(600.0, SPATIAL)
    assert long_walk["theta_deg"] > 50.0 and long_walk["delta_m"] > 7.0
    m = measure_copy(a, b, SCFG, budget=long_walk)
    assert m["fit_chosen"] == "rotated" and m["fit_demoted"] is None
    assert m["fit_motion"]["rotated"]["over_budget"] < 1.0


def test_without_a_budget_nothing_is_judged_on_motion():
    """Backward compatible by construction: the callers that state no drift
    rate get exactly the pre-2026-09-21 behaviour, and the record says the
    budget was unavailable instead of pretending one was applied."""
    a, b = _two_desks()
    m = measure_copy(a, b, SCFG)
    assert m["fit_chosen"] == "rotated" and m["fit_demoted"] is None
    assert m["drift_budget"] is None
    assert all(r["over_budget"] is None for r in m["fit_motion"].values())


def test_a_tie_goes_to_the_fewer_degrees_of_freedom():
    """The promise the comparison has carried since 2026-09-17 and did not
    keep: a stable sort handed every tie to `rotated` for being first in the
    list. Two identical copies leave all three fits at the same identity."""
    b = _slab()
    m = measure_copy(b.copy(), b, SCFG, budget=BUDGET)
    assert m["fit_scores_m"]["rotated"] == m["fit_scores_m"]["translated"]
    assert m["fit_chosen"] != "rotated", m["fit_scores_m"]
    assert lp._FIT_DOF["rotated"] > lp._FIT_DOF["translated"]


# ── the walk reaches the measurement (the callers) ─────────────────────────

class _Session:
    """`desk#198`: two desks under one label, seen at keyframe 2 and again at
    keyframe 204, with a straight walk of ~19.3 m between the two visits."""

    def __init__(self, tmp_path, n_kf=216, step_m=WALK_M / 202.0):
        import json
        self.output_dir = tmp_path
        a, b = _two_desks()
        self.xyz = np.vstack([a, b])
        self.ks = np.concatenate([np.full(len(a), 204), np.full(len(b), 2)])
        self.n_points = len(self.xyz)
        self.n_kf = n_kf
        self.poses = np.tile(np.eye(4), (n_kf, 1, 1))
        self.poses[:, 0, 3] = np.arange(n_kf) * step_m
        (tmp_path / "segmentation_result.json").write_text(json.dumps(
            {"instances": [{"instance_id": 198, "id": 198, "label": "desk",
                            "globalIndices": list(range(self.n_points))}]}))


def _cand():
    return [{"instance_id": 198, "label": "desk", "i": 204, "j": 2,
             "verdict": "loop", "kind": "instance", "class": "structural"}]


def _capture(monkeypatch):
    seen = {}

    def _fake(pa, pb, scfg, seed=0, budget=None, **kw):
        seen["budget"] = budget
        seen["called"] = seen.get("called", 0) + 1
        return None
    monkeypatch.setattr(lp, "measure_copy", _fake)
    return seen


def test_the_graph_hands_the_measurement_the_walk(tmp_path, monkeypatch):
    seen = _capture(monkeypatch)
    lp.instance_edges(_Session(tmp_path), _cand(), ccfg=None, cfg=CFG,
                      log=lambda m: None)
    assert seen["called"] == 1
    bud = seen["budget"]
    assert bud is not None and bud["source"] == "drift_rate_model"
    assert bud["L_m"] == pytest.approx(WALK_M, abs=0.05)
    assert bud["delta_m"] == pytest.approx(
        max(SPATIAL.drift_floor_m, SPATIAL.drift_rate_m_per_m * WALK_M), abs=1e-6)
    assert bud["theta_deg"] == pytest.approx(
        max(SPATIAL.drift_floor_deg, SPATIAL.drift_rate_deg_per_m * WALK_M), abs=1e-6)


def test_the_edge_is_written_and_says_the_rotated_fit_was_refused(tmp_path):
    """The whole point: the pair keeps its edge — from a possible fit — and
    the record carries the refusal."""
    from tests.synth_correction import make_correction_cfg
    edges = lp.instance_edges(_Session(tmp_path), _cand(),
                              ccfg=make_correction_cfg(), cfg=CFG,
                              log=lambda m: None)
    assert len(edges) == 1
    e = edges[0]
    assert e["accepted"] is True, e.get("reason")
    assert e["rot_deg"] == pytest.approx(0.0, abs=1e-6), e["rot_deg"]
    assert e["fit_chosen"] != "rotated"
    assert e["fit_demoted"] is not None and e["fit_demoted"]["refused"] == "rotated"
    assert e["drift_budget"]["L_m"] == pytest.approx(WALK_M, abs=0.05)
    # and the translation it carries is one the walk could have produced,
    # where the refused fit demanded metres
    assert e["t_norm_m"] < e["fit_demoted"]["refused_t_norm_m"]


def test_the_scale_stage_judges_the_same_motion(tmp_path):
    rows = lp.copy_scale_rows(_Session(tmp_path), _cand(), SCFG,
                              window_kf=CFG.certify.visit_loops.window_kf,
                              max_pairs_per_instance=3, log=lambda m: None,
                              spatial_cfg=SPATIAL)
    assert len(rows) == 1
    r = rows[0]
    assert r["fit_demoted"] is not None and r["fit_demoted"]["refused"] == "rotated"
    assert r["s_ab"] == 1.0 and r["scale_trusted"] is False
    assert r["drift_budget"]["delta_m"] == pytest.approx(BUDGET["delta_m"])


def test_a_spatial_block_without_a_drift_rate_says_so(tmp_path, monkeypatch):
    """`copy_scale_rows` takes its spatial config optionally and a caller may
    hand only the same-surface keys the overlap refusal needs. A bound nobody
    stated is not invented: the measurement is made on quality alone and the
    record says why."""
    seen = _capture(monkeypatch)
    partial = {"dims_pct_lo": 5.0, "dims_pct_hi": 95.0,
               "same_surface_angle_deg": 10.0, "same_surface_axis_ratio": 0.15,
               "same_surface_planar_ratio": 0.05}
    rows = lp.copy_scale_rows(_Session(tmp_path), _cand(), SCFG,
                              window_kf=CFG.certify.visit_loops.window_kf,
                              max_pairs_per_instance=3, log=lambda m: None,
                              spatial_cfg=partial)
    assert seen["called"] == 1 and seen["budget"] is None
    assert "no drift rate" in rows[0]["drift_budget_why"]


def test_the_bound_is_configured_not_hardcoded():
    """Every number the rule decides with comes from `loops.spatial` through
    `spatial_gate.drift_budget` — there is no rotation or translation literal
    in the module to retune."""
    src = (Path(__file__).resolve().parents[1]
           / "reconstruction" / "certify" / "loops_posthoc.py").read_text()
    for forbidden in ("_MAX_ROT_DEG", "_MAX_T_M", "max_fit_rot_deg", "max_fit_t_m"):
        assert forbidden not in src, forbidden
    assert "drift_budget" in src and "walked_length_m" in src
