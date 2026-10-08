"""THE USER'S RULE (2026-10-07, docs/plan_determinismo.md point 1): a correction applies only when
it is significant (whole bootstrap CI on the improving side), at least min_judge_closures(confidence)
judges testify (5 at 0.95), and the median improvement is >= the user's factor (2, from
correction_graph.graph.improvement_error_factor) x the measured error. metric_lock.decide_change."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "vendor" / "VGGT-Long"))

import loop_utils.metric_lock as ml  # noqa: E402
from loop_utils.loop_judge import min_judge_closures  # noqa: E402
from loop_utils.metric_lock import _pooled_median, decide_change, heldout_change  # noqa: E402

FACTOR = 2.0          # the user's factor; the production value is asserted below


def _rule(before, after, error, **kw):
    kw.setdefault("confidence", 0.95)
    return decide_change(before, after, error=error, error_factor=FACTOR, **kw)


def test_all_three_conditions_hold_then_it_improves_with_every_margin_reported():
    rng = np.random.default_rng(1)
    before = 0.5 + 0.1 * rng.random(8)
    after = before - 0.30 + 0.01 * rng.standard_normal(8)
    v = _rule(before, after, error=0.05)
    assert v["improves"] and v["significant"] and v["enough_judges"] and v["beyond_error"]
    assert v["failed"] == [] and v["reason"].startswith("IMPROVES")
    assert v["n_judges"] == 8 and v["min_judges"] == 5 and v["judges_margin"] == 3
    assert v["required_delta"] == pytest.approx(0.10)
    assert v["error_margin"] == pytest.approx(v["median_delta"] - 0.10)
    assert v["ci_margin"] == v["ci_low"] > 0
    json.dumps(v)                                     # goes to the logs / reports as is


def test_too_few_judges_never_apply_however_large_the_improvement():
    before = [1.0, 1.1, 0.9, 1.05]
    after = [0.1, 0.1, 0.1, 0.1]
    v = _rule(before, after, error=0.01)
    assert v["significant"] and v["beyond_error"] and not v["enough_judges"]
    assert not v["improves"] and v["failed"] == ["judges"] and v["judges_margin"] == -1


def test_a_consistent_improvement_smaller_than_twice_the_error_does_not_apply():
    """The case the user's factor exists for (pccr 2026-08-31: 34.5 -> 34.5 cm read as IMPROVE)."""
    before = np.array([0.3455, 0.20, 0.31, 0.27, 0.40, 0.22])
    after = before - 0.0001                         # every judge improves by 0.1 mm
    v = _rule(before, after, error=0.03)            # the largest bridge sigma: 3 cm
    assert v["significant"] and v["enough_judges"] and not v["beyond_error"]
    assert not v["improves"] and v["failed"] == ["error"]


def test_an_improvement_inside_the_noise_is_not_significant():
    rng = np.random.default_rng(7)
    before = 0.5 + 0.1 * rng.random(12)
    after = before + 0.2 * rng.standard_normal(12)
    v = _rule(before, after, error=0.0)
    assert not v["significant"] and "significance" in v["failed"] and not v["improves"]


def test_exactly_factor_times_error_passes_condition_c():
    before = np.full(6, 1.0)
    after = np.full(6, 0.75)                        # 0.25 = 2 x 0.125 (exact in binary)
    v = _rule(before, after, error=0.125)
    assert v["median_delta"] == 0.25 and v["required_delta"] == 0.25 and v["beyond_error"]
    assert v["improves"]


def test_worsening_is_reported_and_never_applies():
    before = np.full(7, 0.1)
    after = np.full(7, 0.4)
    v = _rule(before, after, error=0.0)
    assert v["worsens"] and not v["improves"] and not v["significant"]


def test_min_judges_follow_the_declared_confidence():
    assert min_judge_closures(0.95) == 5
    v = _rule(np.ones(6), np.zeros(6), error=0.0, confidence=0.99)
    assert v["min_judges"] == min_judge_closures(0.99) == 7 and not v["enough_judges"]
    v = _rule(np.ones(3), np.zeros(3), error=0.0, min_judges=3)
    assert v["min_judges"] == 3 and v["improves"]


def test_unclustered_interval_is_bit_identical_to_heldout_change():
    rng = np.random.default_rng(3)
    before = rng.random(23)
    after = before - 0.05 + 0.1 * rng.standard_normal(23)
    h = heldout_change(before, after, confidence=0.95)
    v = _rule(before, after, error=0.0)
    assert (v["ci_low"], v["ci_high"], v["median_delta"]) == (h["ci_low"], h["ci_high"],
                                                              h["median_delta"])
    assert v["significant"] == h["improves"] and v["worsens"] == h["worsens"]


def test_same_call_same_answer_and_blocks_draw_what_one_matrix_draws(monkeypatch):
    rng = np.random.default_rng(11)
    before = rng.random(300)
    after = before - 0.02 + 0.05 * rng.standard_normal(300)
    clusters = np.repeat(np.arange(30), 10)
    ref_u = _rule(before, after, error=0.001)
    ref_c = _rule(before, after, error=0.001, clusters=clusters)
    assert _rule(before, after, error=0.001) == ref_u
    assert _rule(before, after, error=0.001, clusters=clusters) == ref_c
    monkeypatch.setattr(ml, "_BOOT_BLOCK_ELEMS", 1000)          # many small blocks
    assert _rule(before, after, error=0.001) == ref_u
    assert _rule(before, after, error=0.001, clusters=clusters) == ref_c


def test_pooled_median_is_np_median_of_the_expanded_sample():
    rng = np.random.default_rng(5)
    for trial in range(200):
        n = int(rng.integers(1, 12))
        vals = np.sort(rng.standard_normal(n))
        w = rng.integers(0, 4, size=n)
        if w.sum() == 0:
            w[int(rng.integers(0, n))] = 1
        assert _pooled_median(vals, w) == float(np.median(np.repeat(vals, w))), trial


def test_clusters_are_the_judges_resampled_whole():
    """Point 46: resampling by keyframe, not by track — 4 keyframes with 1000 observations each are
    4 judges, never 4000."""
    rng = np.random.default_rng(2)
    clusters = np.repeat([10, 20, 30, 40], 1000)
    before = 1.0 + 0.1 * rng.random(4000)
    after = before - 0.5
    v = _rule(before, after, error=0.01, clusters=clusters)
    assert v["clustered"] and v["n_obs"] == 4000 and v["n_judges"] == 4
    assert not v["enough_judges"] and v["failed"] == ["judges"]
    # with enough keyframes it passes; label values and order do not matter
    clusters = np.repeat(np.arange(6)[::-1] * 7, 100)
    before = 1.0 + 0.1 * rng.random(600)
    after = before - 0.5 + 0.01 * rng.standard_normal(600)
    v = _rule(before, after, error=0.01, clusters=clusters)
    assert v["n_judges"] == 6 and v["improves"]


def test_a_cluster_whose_observations_disagree_widens_the_interval():
    """One keyframe that improves massively (many observations) cannot carry the verdict alone."""
    rng = np.random.default_rng(9)
    clusters = np.concatenate([np.zeros(2000, int), np.repeat(np.arange(1, 7), 20)])
    before = np.ones(clusters.size)
    after = np.where(clusters == 0, 0.0, 1.0 + 0.05 * rng.standard_normal(clusters.size))
    v_obs = _rule(before, after, error=0.0)                     # per observation: looks decisive
    v_kf = _rule(before, after, error=0.0, clusters=clusters)   # per keyframe: it is one judge
    assert v_obs["significant"]
    assert v_kf["ci_high"] - v_kf["ci_low"] > v_obs["ci_high"] - v_obs["ci_low"]


def test_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError, match="paired"):
        _rule([1.0, 2.0], [1.0], error=0.0)
    with pytest.raises(ValueError, match="non-finite"):
        _rule([1.0, np.nan], [1.0, 1.0], error=0.0)
    with pytest.raises(ValueError, match="error"):
        _rule([1.0], [1.0], error=-1.0)
    with pytest.raises(ValueError, match="error"):
        _rule([1.0], [1.0], error=float("nan"))
    with pytest.raises(ValueError, match="error_factor"):
        decide_change([1.0], [0.0], error=0.0, error_factor=0.0, confidence=0.95)
    with pytest.raises(ValueError, match="confidence"):
        _rule([1.0], [0.0], error=0.0, confidence=1.0)
    with pytest.raises(ValueError, match="cluster"):
        _rule([1.0, 2.0], [0.0, 0.0], error=0.0, clusters=[1])
    with pytest.raises(TypeError):
        decide_change([1.0], [0.0], 0.1, 2.0, 0.95)               # keywords only


def test_no_judge_at_all_is_declared_not_applied():
    v = _rule([], [], error=0.0)
    assert not v["improves"] and v["n_judges"] == 0
    assert v["failed"] == ["judges", "significance", "error"]


# ── the user's factor lives in the config, one key ───────────────────────────────────────────

def _raw():
    return yaml.safe_load((ROOT / "server" / "config.yaml").read_text())


def test_production_config_declares_the_users_factor_and_the_fork_receives_it():
    from reconstruction.loops.config import (fork_model_graph, improvement_error_factor,
                                             load_loops_config)
    raw = _raw()
    cfg = load_loops_config(raw)
    assert cfg.graph.improvement_error_factor == 2.0 == improvement_error_factor(raw)
    assert fork_model_graph(cfg)["improvement_error_factor"] == 2.0
    txt = (ROOT / "server" / "config.yaml").read_text()
    assert "improvement_error_factor: 2.0   # USER 2026-10-07" in txt


@pytest.mark.parametrize("bad", ["missing", 0.0, -2.0, "two", True])
def test_the_factor_fails_the_load_when_missing_or_not_positive(bad):
    from reconstruction.loops.config import (LoopsConfigError, improvement_error_factor,
                                             load_loops_config)
    raw = copy.deepcopy(_raw())
    if bad == "missing":
        del raw["correction_graph"]["graph"]["improvement_error_factor"]
    else:
        raw["correction_graph"]["graph"]["improvement_error_factor"] = bad
    with pytest.raises(LoopsConfigError, match="improvement_error_factor"):
        improvement_error_factor(raw)
    with pytest.raises(LoopsConfigError, match="improvement_error_factor"):
        load_loops_config(raw)


def test_the_test_mirror_of_the_graph_config_carries_the_factor():
    from tests.synth_metric import fork_graph_cfg
    assert fork_graph_cfg()["improvement_error_factor"] == 2.0
