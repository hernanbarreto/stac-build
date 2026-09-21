"""Two findings of the pccr relaunch of 2026-09-21, both about what the
console and the acta SAY about a run.

FINDING 9.b — the longest stage of the certification printed nothing. USER:
*"no se esta imprimiendo nada en consola, anota que todos estos pasos deben
estar en consola"*. Eleven minutes between

    [certify] loaded 216 keyframes (depth+K from chunk npy, poses from camera_poses.txt)
    [certify] before the correction: objective 4.3000 | seams 0.0242 | closure 1.0647

with the process alive the whole time (444 % CPU, 336 threads, 3.5 GB RSS,
seg_masks.npz open). The instrumentation existed — `instance_loops` already
emitted "N/M instance(s), K candidate(s) so far" — and the caller threw it
away: `_measure_now` passes `log=lambda m: None` because it wants no
per-candidate detail in the acta's BEFORE, and the advance went out on that
same channel. The advance now has its own.

FINDING 28 — the acta declared a 71.7 % REGRESSION over a cloud that improved
41 %. From the run's own certify_acta.json:

    metrics_initial objective 4.3000   closure median 1.065 m  max 10.43 m  (21 edges)
    metrics_final   objective 7.3823   closure median 2.351 m  max 12.99 m  (22 edges)

Split into the two populations those edges actually hold:

    BEFORE  21 edges = 10 refused (one surface against itself, 3.8-10.43 m)
                     + 11 real: 106 106 97 97 97 80 77 77 73 69 33 cm  → median 80.5
    AFTER   22 edges = 11 refused (3.78-12.99 m)
                     + 11 real:  92  74 61 61 61 48 40 16 16 13 12 cm  → median 47.7

The real closures improved 41 %, and so did everything independent of the
edge set (seam residual median 2.417 → 2.358 cm, residual scale per chunk
7.8 % → 0.6 %, the user's desk 67.1 → 16.1 cm). What inverted the verdict was
the MIX of refused pairs — pairs that close nothing, whose magnitudes are
metres BY CONSTRUCTION (`loops.spatial_gate.surface_overlap` is what refuses
them). They stay in the record; they do not vote.

No GPU, no pipeline: the synthetic session of tests/synth_metric.py for the
console, the acta's own numbers for the populations.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify.run import (refused_closures,  # noqa: E402
                                        split_closure_population)
from reconstruction.loops.config import load_loops_config  # noqa: E402
from reconstruction.quality.report import closure_errors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN_SRC = (ROOT / "reconstruction" / "certify" / "run.py").read_text()
DET_SRC = (ROOT / "reconstruction" / "loops" / "instance_loops.py").read_text()


# ── FINDING 28: the two populations of the acta ────────────────────────────

# the acta's own edges, in metres (see the module docstring)
REAL_BEFORE = [1.06, 1.06, 0.97, 0.97, 0.97, 0.80, 0.77, 0.77, 0.73, 0.69, 0.33]
REAL_AFTER = [0.92, 0.74, 0.61, 0.61, 0.61, 0.48, 0.40, 0.16, 0.16, 0.13, 0.12]
# one surface against itself: the two ends of a floor / a wall / a ceiling
# duct, "closed" by sliding one end onto the other
REFUSED_BEFORE = [3.90, 4.60, 5.30, 6.20, 7.40, 8.50, 9.70, 10.36, 10.40, 10.43]
REFUSED_AFTER = [3.78, 4.50, 5.40, 6.30, 7.50, 8.60, 9.80, 10.40, 11.50, 12.50, 12.99]


def _edges(real, refused):
    """A loop list as the measurement returns it: the pairs it kept
    (``accepted``) followed by the ones it refused, each with its reason."""
    out = [{"i": 100 + k, "j": k, "label": f"desk#{k}", "instance_id": k,
            "source": "instance", "accepted": True, "trusted": True,
            "offset_before_m": v, "offset_after_m": v * 0.1, "t_norm_m": v,
            "icp_rms_m": 0.05} for k, v in enumerate(real)]
    # the refused ones carry the separation they measured — that is the very
    # number the acta averaged into its median (10.36 m for
    # `white_tiled_floor#45`, 12.99 m for `white_wall#98`) before the
    # supports were ever asked whether they overlap
    out += [{"i": 200 + k, "j": k, "label": f"white_tiled_floor#{k}",
             "instance_id": 100 + k, "source": "instance", "accepted": False,
             "kind": "duplicate", "offset_before_m": v, "offset_after_m": 0.3,
             "reason": "two parts of one plane — the supports do not overlap on it",
             "surface_overlap": {"kind": "plane", "overlap_frac": 0.0, "gap_m": v - 0.5}}
            for k, v in enumerate(refused)]
    return out


def _closure_median(loops):
    return closure_errors(None, loops)["median_m"]


def test_the_refused_pairs_carry_the_median_when_they_vote():
    """The bug, reproduced from the acta: with every pair voting the median
    closure goes the wrong way and the objective follows it."""
    before = _edges(REAL_BEFORE, REFUSED_BEFORE)
    after = _edges(REAL_AFTER, REFUSED_AFTER)
    assert len(before) == 21 and len(after) == 22
    assert _closure_median(before) == pytest.approx(1.06, abs=0.01)
    assert _closure_median(after) == pytest.approx(2.35, abs=0.01)
    assert _closure_median(after) > _closure_median(before)      # the inversion


def test_the_plausible_population_says_the_cloud_improved():
    before, _ = split_closure_population(_edges(REAL_BEFORE, REFUSED_BEFORE))
    after, _ = split_closure_population(_edges(REAL_AFTER, REFUSED_AFTER))
    assert len(before) == len(after) == 11
    assert _closure_median(before) == pytest.approx(0.80, abs=0.01)
    assert _closure_median(after) == pytest.approx(0.48, abs=0.01)
    gain = 1.0 - _closure_median(after) / _closure_median(before)
    assert gain == pytest.approx(0.41, abs=0.02), "the 41 % the user saw"


def test_the_objective_follows_the_plausible_population():
    """The term the acta prints as a verdict. With the production weights, the
    closure contribution must fall over these two states, not grow 71.7 %."""
    w = load_loops_config().certify.objective.closure_m
    assert w > 0, "the closure has no weight in the objective any more"
    all_b, all_a = (_edges(REAL_BEFORE, REFUSED_BEFORE), _edges(REAL_AFTER, REFUSED_AFTER))
    plaus_b, _ = split_closure_population(all_b)
    plaus_a, _ = split_closure_population(all_a)
    assert w * _closure_median(all_a) > w * _closure_median(all_b)        # inverted
    assert w * _closure_median(plaus_a) < w * _closure_median(plaus_b)    # measured


def test_a_pair_is_refused_by_the_measurement_never_by_its_magnitude():
    """There is no ">2 m is false" anywhere: a real duplicate of a hall-sized
    object votes, and a refused pair votes never, whatever it measures."""
    loops = [{"accepted": True, "offset_before_m": 9.9, "label": "big_real"},
             {"accepted": False, "offset_before_m": 0.02, "label": "refused_tiny",
              "reason": "the rigid closure did not reduce the copies' offset"}]
    plausible, refused = split_closure_population(loops)
    assert [p["label"] for p in plausible] == ["big_real"]
    assert [r["label"] for r in refused] == ["refused_tiny"]
    src = ast.get_source_segment(RUN_SRC, next(
        n for n in ast.parse(RUN_SRC).body
        if isinstance(n, ast.FunctionDef) and n.name == "split_closure_population"))
    body = src.split('"""')[-1]
    assert "offset_before_m" not in body and ">" not in body, \
        f"the split invented a magnitude threshold:\n{body}"


def test_the_refused_population_is_published_whole():
    """Nothing is hidden: the count, what they would have contributed, and one
    record per pair with the reason and the numbers that refused it."""
    _, refused = split_closure_population(_edges(REAL_AFTER, REFUSED_AFTER))
    rep = refused_closures(refused)
    assert rep["n"] == 11 and len(rep["per_pair"]) == 11
    assert rep["n_with_offset"] == 11
    # what they would have contributed had they voted, declared as its own
    # number instead of mixed into the one the user reads
    assert rep["median_m"] == pytest.approx(8.60)
    assert rep["max_m"] == pytest.approx(12.99)
    assert rep["median_surface_gap_m"] == pytest.approx(8.60 - 0.5, abs=1e-6)
    assert rep["provenance"] == "tool_measured"
    for r in rep["per_pair"]:
        assert r["reason"], r
        assert r["surface_overlap"]["gap_m"] > 1.0
        assert {"i", "j", "instance_id", "label", "source"} <= set(r)


def test_a_refused_pair_that_was_measured_reports_what_it_measured():
    """`the rigid closure did not reduce the copies' offset` is refused AFTER
    the measurement, so its number exists and must be declared — it is the
    one refusal that used to vote in the closure median."""
    rep = refused_closures([
        {"accepted": False, "offset_before_m": 2.4, "offset_after_m": 2.6,
         "label": "wall#98", "i": 115, "j": 26, "source": "instance",
         "reason": "the rigid closure did not reduce the copies' offset"},
        {"accepted": False, "offset_before_m": 1.0, "label": "duct#116",
         "reason": "copies starved (3 / 8 points)"}])
    assert rep["n"] == 2 and rep["n_with_offset"] == 2
    assert rep["median_m"] == pytest.approx(1.7)
    assert rep["max_m"] == pytest.approx(2.4)


def test_a_pair_refused_before_it_was_measured_has_no_number_to_declare():
    """`surface_overlap` refuses two ends of one floor BEFORE the rigid fit
    runs, so there is no offset to report — and none is invented."""
    rep = refused_closures([
        {"accepted": False, "label": "white_tiled_floor#45", "i": 119, "j": 25,
         "source": "instance",
         "reason": "two parts of one plane — the supports do not overlap on it",
         "surface_overlap": {"kind": "plane", "overlap_frac": 0.0, "gap_m": 9.9}}])
    assert rep["n"] == 1 and rep["n_with_offset"] == 0
    assert rep["median_m"] is None and rep["max_m"] is None
    assert rep["median_surface_gap_m"] == pytest.approx(9.9)


def test_the_certification_measures_over_the_plausible_population():
    """The wiring: `_state_metrics` splits, hands compute_metrics the
    plausible pairs and publishes the refused ones beside them."""
    body = RUN_SRC.split("def _state_metrics")[1].split("def ")[0]
    assert "split_closure_population(loops)" in body
    assert "compute_metrics(session, fields, plausible," in body, body
    assert 'm["closure_refused"] = refused_closures(refused)' in body
    # and the verdict the user reads says both populations
    regression_line = RUN_SRC.split("⚠ REGRESSION")[1].split(")\n")[0]
    assert "closure_refused" in regression_line, regression_line
    assert "refused and not voting" in regression_line


# ── FINDING 9.b: the stage says where it is ────────────────────────────────

@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    """A written session the detector can run over end to end (no GPU)."""
    from tests.synth_metric import make_session, write_session_dir
    sess = make_session(n_kf=40, H=20, W=28)
    cols = [p for p in sess.scene.prims if getattr(p, "label", "") == "column"]
    return write_session_dir(tmp_path_factory.mktemp("adv") / "s", sess,
                             {1: {"label": "column", "oids": [cols[5].oid]}})


def _cfg():
    from tests.synth_metric import raw_server_cfg
    return load_loops_config(raw_server_cfg(**{"loops.cluster_min_points": 50,
                                               "loops.dbscan_min_samples": 5,
                                               "loops.min_gap_keyframes": 10}))


def test_the_advance_survives_a_silenced_log(session_dir):
    """What `_measure_now` does: no per-candidate detail, and the advance
    still reaches the console."""
    from reconstruction.loops.instance_loops import detect_instance_loops
    detail, advance = [], []
    detect_instance_loops(session_dir / "output", session_dir, cfg=_cfg(),
                          log=detail.append, progress=advance.append,
                          apply_splits=False)
    assert advance, "the stage ran silently — this is finding 9.b again"
    joined = " | ".join(advance)
    assert "session loaded" in joined and "classified" in joined
    assert "instance(s)," in joined, joined          # the per-instance advance
    assert "candidate(s):" in joined                 # and how it ended
    # the DETAIL stayed on the other channel: the advance is not a copy of it
    assert not [d for d in detail if "instance(s), " in d and "so far" in d]


def test_without_a_progress_channel_the_advance_still_goes_to_the_log(session_dir):
    """Every existing verbose call site keeps printing exactly what it did."""
    from reconstruction.loops.instance_loops import detect_instance_loops
    lines = []
    detect_instance_loops(session_dir / "output", session_dir, cfg=_cfg(),
                          log=lines.append, apply_splits=False)
    joined = " | ".join(lines)
    assert "so far" in joined and "candidate(s):" in joined, joined


def test_the_detector_advances_at_least_every_tenth_of_the_instances():
    body = DET_SRC.split("def detect_instance_loops")[1]
    assert "_step = max(1, _total // 10)" in body, \
        "the cadence of the advance is gone — 82 instances may print once"
    loop = body.split("while queue:")[1]
    assert "_advance(f\"[instance-loops] {_done}/{_total} instance(s), " in loop


def test_the_two_channels_are_separate_in_the_detector():
    assert "progress: Optional[Callable[[str], None]] = None" in DET_SRC
    assert "_advance = progress if progress is not None else log" in DET_SRC


def test_the_metrics_stage_prefixes_its_advance_and_names_its_phase():
    body = RUN_SRC.split("def _measure_now")[1].split("\n    # The acta's BEFORE")[0]
    assert 'log(f"[certify] [metrics] {phase}: {msg}")' in body
    assert "progress=_pg" in body, "the detector is silenced again"
    assert "_all_edges(sess, cn, quiet=True, progress=_pg)" in body
    # the phase the user could not see
    assert '"measuring the state before the correction"' in RUN_SRC


def test_the_edge_measurement_reports_which_of_its_two_halves_is_running():
    body = RUN_SRC.split("def _all_edges")[1].split("\n    base = ")[0]
    assert "progress=None" in body
    assert "_pg(f\"measuring the copies of" in body
    assert "now the geometric revisits" in body
