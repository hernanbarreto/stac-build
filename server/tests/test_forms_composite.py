"""FORMS P4 (USER 2026-10-16): composites decomposed GUIDED by the VLM's part list — a table
(a top, four equal legs), the joint fit of the union. CPU, synthetic."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms.catalog import FORMS, parse_answer  # noqa: E402
from reconstruction.forms.composite import Composite, build_composite, shares_section  # noqa: E402
from reconstruction.forms.fit import fit_cloud, residual_report  # noqa: E402
from reconstruction.forms.templates import Extrusion, Frame, Pose, Section  # noqa: E402

DEV = "cpu"
CAT = list(FORMS)
UP = np.array([0.0, 1.0, 0.0])
FRAME = Frame(UP, np.array([1.0, 0.0, 0.0]))
SIGMA = 0.004
FIT = dict(lr=0.02, block_iterations=20, max_blocks=30, huber_k=1.345, max_points=40000, seed=0)
BUILD = dict(FIT, ransac_iters=400, error_factor=2.0)


def _table_answer(legs=4, relation="four equal legs under the top"):
    return parse_answer({"form": "composite", "orientation": "unknown", "axis_aligned": True, "hollow": None,
                         "outline": "unknown", "profile": "unknown",
                         "parts": [{"name": "top", "form": "box", "count": 1, "orientation": "horizontal",
                                    "relation": "the top plate"},
                                   {"name": "leg", "form": "cylinder", "count": legs, "orientation": "vertical",
                                    "relation": relation}],
                         "proportions": "", "alternatives": [], "confidence": "high",
                         "what_would_help": "", "notes": ""}, CAT)


def _table_cloud(n_legs=4, seed=0):
    rng = np.random.default_rng(seed)
    top = Extrusion("box", Section("rect", {"bx": 0.6, "by": 0.4}, DEV),
                    Pose(FRAME, "fixed", "snapped", DEV, np.array([0.0, 0.75, 0.0]), UP), 0.015, False, DEV)
    pts = [top.samples(6000).detach().numpy()]
    corners = [(0.55, 0.35), (-0.55, 0.35), (-0.55, -0.35), (0.55, -0.35)][:n_legs]
    for cx, cz in corners:
        leg = Extrusion("cylinder", Section("circle", {"r": 0.025}, DEV),
                        Pose(FRAME, "fixed", "snapped", DEV, np.array([cx, 0.36, cz]), UP), 0.36, False, DEV)
        pts.append(leg.samples(1500).detach().numpy())
    P = np.vstack(pts)
    return P + rng.normal(0, SIGMA, P.shape)


def test_relation_words_decide_the_sharing():
    a = _table_answer()
    assert shares_section(a.parts[1]) and not shares_section(a.parts[0])
    b = _table_answer(relation="four legs of different thickness")
    assert not shares_section(b.parts[1])


def test_a_table_is_decomposed_by_the_list_and_fitted_jointly():
    P = _table_cloud()
    comp, rec = build_composite(_table_answer(), P, FRAME, SIGMA, DEV, **BUILD)
    assert isinstance(comp, Composite) and len(comp.parts) == 5
    assert [n for n, _ in comp.parts] == ["top", "leg_1", "leg_2", "leg_3", "leg_4"]
    assert rec["missing"] == [] and rec["shared"][0]["spec"] == "leg"
    # the equal legs share ONE section tensor
    secs = {id(p.section) for n, p in comp.parts if n.startswith("leg")}
    assert len(secs) == 1
    fit_cloud(comp, P, SIGMA, dev=DEV, **FIT)
    d = comp.dims()
    # the top: 1.2 × 0.8 in the room's axes, thin along up (its underside joined it in the joint fit)
    assert d["top"]["extent_X_m"] == pytest.approx(1.2, abs=0.04) and d["top"]["extent_Z_m"] == pytest.approx(0.8, abs=0.04)
    assert 0.02 < d["top"]["extent_up_m"] < 0.07
    for k in range(1, 5):
        assert d[f"leg_{k}"]["radius_m"] == pytest.approx(0.025, abs=0.003)
        assert d[f"leg_{k}"]["length_m"] == pytest.approx(0.72, abs=0.04)
    centres = np.array(sorted((float(p.pose.c[0]), float(p.pose.c[2]))
                              for n, p in comp.parts if n.startswith("leg")))
    assert sorted(map(tuple, np.round(centres, 2).tolist())) == [(-0.55, -0.35), (-0.55, 0.35), (0.55, -0.35), (0.55, 0.35)]
    rep = residual_report(comp, P, SIGMA, DEV)
    assert rep["within_2sigma"] > 0.9
    V, F = comp.mesh()
    assert len(V) > 100 and F.max() < len(V)
    assert comp.record()["parts"][1]["name"] == "leg_1"


def test_a_missing_leg_is_declared_not_invented():
    P = _table_cloud(n_legs=3)
    comp, rec = build_composite(_table_answer(legs=4), P, FRAME, SIGMA, DEV, **BUILD)
    assert len(comp.parts) == 4 and rec["missing"] == [{"spec": "leg", "wanted": 4, "found": 3,
                                                        "reason": "the cloud has no separate piece for it (declared, not invented)"}]


def test_the_composite_is_deterministic():
    outs = []
    for _ in range(2):
        P = _table_cloud()
        comp, _rec = build_composite(_table_answer(), P, FRAME, SIGMA, DEV, **BUILD)
        fit_cloud(comp, P, SIGMA, dev=DEV, **FIT)
        outs.append(comp.state())
    assert all(np.array_equal(a, b) for a, b in zip(outs[0], outs[1]))
