"""The σ floor is what the session can REPEAT, not a constant over four.

USER 2026-09-16: `sigma_floor_m = max_residual_m / 4` gave 2.5 cm while pccr's
own two-copy disagreement measured 4.77 cm — the edges claimed more certainty
than the pipeline could reproduce. And it had to work for a SINGLE-CHUNK
session too, where no frame is ever reconstructed twice by two chunks.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify.repeatability import session_repeatability  # noqa: E402


@pytest.fixture()
def out(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d


def _uncertainty(out, median_m, n=174):
    (out / "uncertainty.json").write_text(json.dumps(
        {"version": 1, "session_median_m": median_m, "n_shared_frames": n}))


def _elastic(out, residuals):
    (out / "elastic_seams.json").write_text(json.dumps(
        {"seams": {"0": {str(g): {"residual_m": r, "before_m": r * 2}
                         for g, r in enumerate(residuals)}}}))


def _intra(out, held_after_cm):
    (out / "intra_chunk.json").write_text(json.dumps(
        {"chunk_indices": [[0, 216]],
         "chunks": {str(k): {"verdict": "APPLY", "held_before_cm": cm * 2,
                             "held_after_cm": cm}
                    for k, cm in enumerate(held_after_cm)}}))


def test_the_two_copy_disagreement_wins(out):
    _uncertainty(out, 0.0477)
    _elastic(out, [0.02, 0.03])
    got = session_repeatability(out, fallback_m=0.01)
    assert got["sigma_floor_m"] == pytest.approx(0.0477)
    assert got["source"] == "uncertainty.json"
    assert got["detail"]["n_shared_frames"] == 174


def test_the_seams_answer_when_uncertainty_is_missing(out):
    _elastic(out, [0.04, 0.06, 0.05])
    got = session_repeatability(out, fallback_m=0.01)
    assert got["sigma_floor_m"] == pytest.approx(0.05)
    assert got["source"] == "elastic_seams.json"


def test_a_single_chunk_session_still_measures_itself(out):
    """No shared frames between chunks: the frames of the one chunk agreeing
    with each other are the repeatability (this is the case the old formula
    silently covered with a constant)."""
    _intra(out, [3.4])
    got = session_repeatability(out, fallback_m=0.01)
    assert got["sigma_floor_m"] == pytest.approx(0.034)
    assert got["source"] == "intra_chunk.json"
    assert "same chunk" in got["detail"]["what"]


def test_nothing_measured_is_declared_as_such(out):
    got = session_repeatability(out, fallback_m=0.01)
    assert got["sigma_floor_m"] == pytest.approx(0.01)
    assert got["source"] == "config_fallback"
    assert "nothing measured" in got["detail"]["what"]


def test_garbage_never_becomes_a_floor(out):
    (out / "uncertainty.json").write_text("{not json")
    (out / "elastic_seams.json").write_text(json.dumps({"seams": {"0": {"1": {}}}}))
    (out / "intra_chunk.json").write_text(json.dumps(
        {"chunks": {"0": {"held_after_cm": None}, "1": {"verdict": "SKIP"}}}))
    got = session_repeatability(out, fallback_m=0.01)
    assert got["source"] == "config_fallback"


def test_zero_and_negative_are_not_measurements(out):
    _uncertainty(out, 0.0)
    _intra(out, [2.5])
    got = session_repeatability(out, fallback_m=0.01)
    assert got["source"] == "intra_chunk.json"      # the zero was not believed


def test_the_floor_is_higher_than_the_old_formula_on_pccr(out):
    """The regression this fixes: 0.10/4 = 2.5 cm against a measured 4.77."""
    _uncertainty(out, 0.0477)
    assert session_repeatability(out)["sigma_floor_m"] > 0.10 / 4.0
