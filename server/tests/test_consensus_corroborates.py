"""USER 2026-09-17, after looking at epochs 1 and 2 in the viewer: the desk's
copies came together along one direction but stayed shifted sideways, and the
floor duplicate at that spot never moved at all.

Both symptoms have one cause. Seven independent objects — two desks, a chair, a
monitor, two glass doors and the floor — each measured the SAME 57-63 cm drift
over the same stretch of the walk. Every one of them was inflated x3 as
`ambiguous`, so the pose graph, weighing sigma against a measurement of the same
size, closed 0% of the loop and the only correction that ever landed came from
the greedy loop, one edge at a time, each cut down to the DOF that single object
observes. A direction no accepted object observes alone can never be reached
that way.

The consensus test that could settle it already existed and only ever PUNISHED:
deviation above `outlier_mad_k` inflates sigma, agreement earned nothing. These
tests pin the other half — agreement corroborates the identity and lifts the
inflation, while the class factor stays (a chair that was pushed is still the
same chair).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SRC = (Path(__file__).resolve().parents[1] / "reconstruction" / "loops" / "kf_graph.py").read_text()


def test_agreement_lifts_the_ambiguous_inflation():
    """The rule, read off the source: a loop inside the MAD band divides its
    sigma by the recorded ×ambiguous factor instead of being left alone."""
    assert 'amb = float((e.get("sigma_factors") or {}).get("ambiguous", 1.0))' in _SRC
    assert 'e["sigma_m"] = float(e.get("sigma_m", sigma_odo_m)) / amb' in _SRC


def test_the_information_matrix_follows_the_sigma():
    """sigma and info_t are the same statement twice; moving one without the
    other would make the edge's weight disagree with its own sigma."""
    assert 'e["info_t"] = np.asarray(e["info_t"], np.float64) * (amb * amb)' in _SRC


def test_the_class_factor_is_never_lifted_by_the_consensus():
    """CLAUDE.md, USER 2026-09-14: 'is it the same object?' and 'did the object
    move between the visits?' are different questions. Only ×ambiguous goes."""
    head = _SRC[:_SRC.index('amb = float(')]
    tail = _SRC[_SRC.index('amb = float('):]
    lifted = tail[:tail.index("log(f\"[kf-graph] loop {int(e['i'])}<->{int(e['j'])}: drift \"\n                        f\"{r * 100:.1f} cm/m AGREES")]
    assert "class" not in lifted, "the consensus must not touch the class factor"
    assert "nonstructural" not in lifted


def test_the_outlier_branch_still_inflates_and_no_longer_falls_through():
    """The punishing branch must keep its behaviour and must not also run the
    corroboration on the same edge."""
    assert "if dev > gc.outlier_mad_k:" in _SRC
    i = _SRC.index("if dev > gc.outlier_mad_k:")
    j = _SRC.index("amb = float(")
    assert "continue" in _SRC[i:j], "an outlier must not fall through to corroboration"


def test_no_new_threshold_is_introduced():
    """Corroboration is the other side of ONE measurement, not a second rule
    with its own numbers: it inherits the band (`outlier_mad_k`) and the peer
    count from the test that already ran, and reads no config of its own."""
    i = _SRC.index("peers = [m[4] for m in meas")
    j = _SRC.index("drift_consensus = consensus")
    block = _SRC[i:j]
    assert "gc.outlier_mad_k" in block and "len(peers) < 3" in block
    lift = block[block.index("amb = float("):block.index("consensus.append", block.index("amb = float("))]
    assert "gc." not in lift, f"the corroboration branch must read no config: {lift!r}"
    # the only numbers it may carry are the neutral factor and the square
    assert sorted(set(re.findall(r"\d+(?:\.\d+)?", lift))) == ["1.0", "64"], \
        sorted(set(re.findall(r"\d+(?:\.\d+)?", lift)))


# ── the arithmetic of the lift ───────────────────────────────────────────

def test_dividing_out_the_factor_restores_the_measured_sigma():
    """desk#201: the copies are 60.5 cm apart and the fit residual gives
    sigma 9.5 cm; ×ambiguous 3 made it 28.6 cm — the size of the very thing it
    was supposed to weigh."""
    measured, amb = 0.0953, 3.0
    inflated = measured * amb
    assert inflated == pytest.approx(0.2859, abs=1e-3)
    assert inflated / amb == pytest.approx(measured)
    info_inflated = 1.0 / inflated ** 2
    info_restored = info_inflated * (amb * amb)
    assert info_restored == pytest.approx(1.0 / measured ** 2, rel=1e-9)


def test_an_edge_with_no_recorded_factor_is_untouched():
    """Geometric revisit edges carry no sigma_factors; the lift must be a no-op
    for them rather than an exception."""
    e = {"sigma_m": 0.14}
    amb = float((e.get("sigma_factors") or {}).get("ambiguous", 1.0))
    assert amb == 1.0
    assert not amb > 1.0
