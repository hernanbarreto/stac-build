"""Stamps and margins around the plan (docs/plan_determinismo.md points 14, 16, 22): the session's
Omega resolution persisted and reused for the same plan, every SALAD candidate's margin to every
bar, the absolute scale rows taken only for the plan and epoch they were measured on."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction import chunk_plan as CP                     # noqa: E402

A100 = {"card": "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0", "total_gb": 80.0,
        "footprint_factor": 1.2139, "footprint_provenance": "zaragoza 2026-10-06"}
ZARA = [(0, 183)]


def test_the_session_resolution_is_decided_once_and_reused_for_the_same_plan(tmp_path):
    logs = []
    r = CP.session_omega_resolution(tmp_path, ZARA, (1920, 1080), "max_size", 1920, 0.15, A100,
                                    log=logs.append)
    assert r["resolution"] == 1520 and r["persisted"] == "decided"           # zaragoza's last runs
    p = tmp_path / "intake" / CP.OMEGA_RESOLUTION_NAME
    doc = json.loads(p.read_text())
    assert doc["key"]["chunk_ranges"] == [[0, 183]] and doc["card"]["footprint_factor"] == 1.2139
    # the same plan on a card whose table entry changed (or another card): the session keeps
    # its resolution, declared — never silently another grid
    other = dict(A100, footprint_factor=1.0)
    r2 = CP.session_omega_resolution(tmp_path, ZARA, (1920, 1080), "max_size", 1920, 0.15, other,
                                     log=logs.append)
    assert r2["resolution"] == 1520 and r2["persisted"] == "reused"
    assert any("reused" in m for m in logs) and any("⚠ decided on" in m and "keeps its resolution" in m
                                                     for m in logs)
    # a fresh decision on that entry would have been higher: the reuse is what makes it stable
    assert CP.omega_resolution_for(183, 80.0, (1920, 1080), "max_size", 1920, margin_frac=0.15,
                                   footprint_factor=1.0)["resolution"] > 1520
    # another plan (one keyframe more) decides again and replaces the record
    r3 = CP.session_omega_resolution(tmp_path, [(0, 184)], (1920, 1080), "max_size", 1920, 0.15, other,
                                     log=logs.append)
    assert r3["persisted"] == "decided" and r3["resolution"] > 1520
    assert json.loads(p.read_text())["key"]["n_keyframes"] == 184
    assert any("another plan" in m for m in logs)
    p.write_text("not json")
    with pytest.raises(CP.ChunkLayoutError, match="unreadable"):
        CP.session_omega_resolution(tmp_path, ZARA, (1920, 1080), "max_size", 1920, 0.15, A100)


def test_omega_card_comes_from_the_table_and_an_unknown_card_fails():
    import card_table
    c = CP.omega_card(A100["card"])
    assert c["total_gb"] == 80.0 and c["footprint_factor"] == 1.2139 and "zaragoza" in c["footprint_provenance"]
    assert CP.omega_resolution_for(139, c["total_gb"], (464, 832), "max_size", 832, margin_frac=0.15,
                                   footprint_factor=c["footprint_factor"])["resolution"] == 832   # pccr
    with pytest.raises(card_table.CardTableError, match="--omega-footprint"):
        CP.omega_card("NVIDIA RTX A6000 | 49140 MiB | sm_8.6")


def test_every_salad_candidate_records_its_margin_to_every_bar():
    from reconstruction.loops import spatial_gate as sg
    m = sg.salad_margin(0.431, 0.416)
    assert m == {"rule": "salad", "similarity": 0.431, "threshold": 0.416,
                 "margin": pytest.approx(0.015), "passed": True}
    assert sg.salad_margin(0.416, 0.416)["passed"] is False            # AT the bar: rejected
    assert sg.salad_margin(None, 0.416)["margin"] is None
    src = Path(sg.__file__).read_text()
    blk = src[src.index("def gate_frame_pair"):src.index("def gate_instance") if "def gate_instance" in src else len(src)]
    for key in ('"margin_m": L - min_walk', 'fr["margin_frames"]', 'cor["margin_m"]', 'out["margins"]'):
        assert key in blk, key
    assert "min_walk_m" in blk and 'walk_ok = L >= min_walk' in blk, "rule 0 (1 m, the user's) stays"


def test_gate_frame_pair_reports_margins_on_a_synthetic_walk():
    from tests.test_graph_f1 import _View, _cfg, sess as _sess_fixture        # noqa: F401
    from tests.synth_metric import make_session
    from reconstruction.loops import spatial_gate as sg
    s = make_session(n_kf=150, H=40, W=56)
    view = _View(s)
    c = _cfg().loops.spatial
    near = sg.gate_frame_pair(8, 9, view, c, salad={"similarity": 0.5, "threshold": 0.416})
    assert near["verdict"] == "reject" and near["margins"]["walk_m"] < 0
    assert "margin" in near["reason"] and near["rules"]["salad"]["margin"] == pytest.approx(0.084)
    far = sg.gate_frame_pair(140, 8, view, c)
    assert far["margins"]["walk_m"] > 0 and "frustum_frames" in far["margins"]
    assert far["margins"]["salad"] is None and "walk margin" in far["reason"]


def test_absolute_rows_enter_only_for_their_plan_and_epoch(tmp_path):
    from reconstruction.loops import structural as st
    out = tmp_path / "output"
    out.mkdir()
    rows = [{"chunk": 1, "log_s": 0.01, "sigma": 0.02, "source": "regulated"},
            {"chunk": None, "log_s": -0.01, "sigma": 0.02, "source": "regulated"}]
    ranges = [[0, 60], [30, 90], [60, 120]]
    p = st.write_absolute_rows(out, rows, ranges)
    doc = json.loads(p.read_text())
    assert doc["version"] == st.ABSOLUTE_ROWS_VERSION == 2
    assert doc["chunk_ranges"] == ranges and doc["n_keyframes"] == 120 and doc["measured_on_epoch"] == 0
    got, why = st.absolute_rows_for_plan(p, [(0, 60), (30, 90), (60, 120)], 120)
    assert why is None and [r["chunk"] for r in got] == [1, 0, 1, 2]
    assert st.absolute_rows_for_plan(p, [(0, 60), (30, 90), (60, 121)], 121)[0] == []
    assert "this run plans" in st.absolute_rows_for_plan(p, [(0, 60), (30, 90), (60, 121)], 121)[1]
    doc["measured_on_epoch"] = 2
    p.write_text(json.dumps(doc))
    got, why = st.absolute_rows_for_plan(p, ranges, 120)
    assert got == [] and "epoch 2" in why
    p.write_text(json.dumps({"version": 1, "rows": rows}))
    got, why = st.absolute_rows_for_plan(p, ranges, 120)
    assert got == [] and "no plan / epoch stamp" in why
    assert st.absolute_rows_for_plan(tmp_path / "none.json", ranges, 120) == ([], None)
