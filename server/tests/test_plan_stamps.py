"""Stamps and margins around the plan (docs/plan_determinismo.md points 14, 16, 22): the session's
Omega resolution persisted and reused for the same plan, every SALAD candidate's margin to every
bar, the absolute scale rows taken only for the plan and epoch they were measured on."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "vendor" / "VGGT-Long"))          # the fork's loop_utils (point 16)

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


# ── point 16, DECIDIDO USER 2026-10-08 ("que quede anotado por candidato"): the fork holds every
# candidate to the SALAD bar it applied ─────────────────────────────────────────────────────

def _stamped_loop_file(tmp_path, stamp):
    """A loop_closures.txt with one SALAD candidate and the given stamp (the fork writes the
    bar it applied into that stamp right after the detector runs)."""
    from loop_utils import loop_bridges as LB
    p = tmp_path / "loop_closures.txt"
    p.write_text("# Loop candidates (index1, index2, similarity, source)\n\n# Loop pairs:\n"
                 "3, 80, 0.5, salad\n")
    LB.write_loop_stamp(str(p), stamp)
    return p


def test_the_salad_bar_is_the_stamps_and_the_detector_that_ran_must_agree_with_it(tmp_path):
    """ONE path (plan point 8's principle, kept for point 16): fresh or reused, the bar comes
    from the stamp of loop_closures.txt, so loop_edges.json holds the same record either way —
    no 'origin' that would tell a fresh run from a resumed one. The detector of a fresh pass
    is a cross-check that fails the pass when the stamp it was written to reads back another
    bar (a resumed run would then hold the candidates to a bar the run never applied)."""
    from loop_utils import loop_bridges as LB

    class Det:                                  # the LoopDetector of THIS pass, after run()
        applied_threshold = 0.416
        threshold_source = "calibrated"

    p = _stamped_loop_file(tmp_path, {"sha256": "abc", "salad_threshold": 0.416,
                                      "salad_threshold_source": "calibrated"})
    bar = {"threshold": 0.416, "source": "calibrated", "reason": None}
    assert LB.salad_applied_threshold(str(p)) == bar                  # reused on its stamp
    assert LB.salad_applied_threshold(str(p), detector=Det()) == bar  # fresh: the detector agrees
    assert LB.load_loop_candidates(str(p)) == [{"i": 3, "j": 80, "sim": 0.5, "source": "salad"}]

    class OtherBar(Det):
        applied_threshold = 0.42

    class OtherSource(Det):
        threshold_source = "configured"

    with pytest.raises(RuntimeError, match=r"applied the bar 0\.42 .*reads back 0\.416"):
        LB.salad_applied_threshold(str(p), detector=OtherBar())
    with pytest.raises(RuntimeError, match="'configured'.*'calibrated'"):
        LB.salad_applied_threshold(str(p), detector=OtherSource())
    with pytest.raises(RuntimeError, match="no loop candidate file"):
        LB.salad_applied_threshold(None, detector=Det())               # ran, nothing stamped: a bug
    # no readable bar: None WITH the reason, never a crash, never an invented bar
    LB.write_loop_stamp(str(p), {"sha256": "abc"})            # stamped before the bar was recorded
    b = LB.salad_applied_threshold(str(p))
    assert b == {"threshold": None, "source": None, "reason": b["reason"]}
    assert "records no salad_threshold" in b["reason"] and "origin" not in b
    for bad in (float("nan"), float("inf"), float("-inf"), "x", "nan"):   # non-finite: refused
        LB.write_loop_stamp(str(p), {"sha256": "abc", "salad_threshold": bad})
        assert LB.salad_applied_threshold(str(p))["threshold"] is None, bad
    assert "does not exist" in LB.salad_applied_threshold(str(tmp_path / "none.txt"))["reason"]
    assert "no loop candidate file" in LB.salad_applied_threshold(None)["reason"]
    assert LB.salad_applied_threshold(str(p), use_dbow=True)["reason"].startswith("DBoW")
    assert "loop_enable false" in LB.salad_applied_threshold(str(p), loop_enable=False)["reason"]
    p.write_text("3, 80, 0.5, salad\n")                       # no stamp line at all
    assert "no readable stamp" in LB.salad_applied_threshold(str(p))["reason"]

    class Bad:                                  # a detector that ran but recorded no bar
        applied_threshold = None
        threshold_source = None

    # ... and stamped exactly that: the cross-check agrees, the record says why there is none
    LB.write_loop_stamp(str(p), {"sha256": "abc", "salad_threshold": None,
                                 "salad_threshold_source": None})
    b = LB.salad_applied_threshold(str(p), detector=Bad())
    assert b["threshold"] is None and "records no salad_threshold (None)" in b["reason"]
    with pytest.raises(RuntimeError, match="reads back None"):   # it did NOT record what it applied
        LB.salad_applied_threshold(str(p), detector=Det())


def test_every_candidate_gets_its_similarity_and_the_bar_or_the_reason_it_has_no_margin():
    from loop_utils import loop_bridges as LB
    bar = {"threshold": 0.416, "source": "calibrated", "origin": "detector", "reason": None}
    salad = {"i": 3, "j": 80, "sim": 0.5, "source": "salad"}
    assert LB.salad_gate_input(salad, bar) == ({"similarity": 0.5, "threshold": 0.416}, None)
    # the server's post-hoc candidates (instance / manual) carry no SALAD similarity
    s, why = LB.salad_gate_input({"i": 3, "j": 80, "sim": None, "source": "instance"}, bar)
    assert s == {"similarity": None, "threshold": None} and "not a SALAD candidate" in why
    s, why = LB.salad_gate_input({"i": 3, "j": 80, "sim": None, "source": "salad"}, bar)
    assert s == {"similarity": None, "threshold": 0.416} and "no SALAD similarity" in why
    nobar = LB.salad_applied_threshold(None, use_dbow=True)
    s, why = LB.salad_gate_input(salad, nobar)
    assert s == {"similarity": 0.5, "threshold": None} and why == nobar["reason"]
    s, why = LB.salad_gate_input(salad, None)
    assert s["threshold"] is None and "no SALAD bar recorded" in why
    s, why = LB.salad_gate_input(dict(salad, sim=float("inf")), bar)      # non-finite: no similarity
    assert s["similarity"] is None and "no SALAD similarity" in why


def test_the_gate_records_the_margin_the_fork_hands_it_on_a_synthetic_walk(tmp_path):
    from tests.test_graph_f1 import _View, _cfg
    from tests.synth_metric import make_session
    from reconstruction.loops import spatial_gate as sg
    from loop_utils import loop_bridges as LB
    s = make_session(n_kf=150, H=40, W=56)
    view = _View(s)
    c = _cfg().loops.spatial
    p = _stamped_loop_file(tmp_path, {"sha256": "abc", "salad_threshold": 0.416,
                                      "salad_threshold_source": "calibrated"})
    bar = LB.salad_applied_threshold(str(p), detector=type("D", (), {"applied_threshold": 0.416,
                                                                     "threshold_source": "calibrated"})())
    salad_in, why = LB.salad_gate_input({"i": 140, "j": 8, "sim": 0.5, "source": "salad"}, bar)
    assert why is None
    g = sg.gate_frame_pair(140, 8, view, c, salad=salad_in)
    assert g["margins"]["salad"] == pytest.approx(0.084) and g["rules"]["salad"]["passed"] is True
    # a post-hoc candidate: the rule block exists, margin None — and the fork writes why
    salad_in, why = LB.salad_gate_input({"i": 140, "j": 8, "sim": None, "source": "instance"}, bar)
    g = sg.gate_frame_pair(140, 8, view, c, salad=salad_in)
    assert g["margins"]["salad"] is None and g["rules"]["salad"]["margin"] is None and why


def test_the_forks_production_call_passes_the_bar_and_loop_edges_json_records_it():
    """The wiring is the part that failed three times in this repo (CLAUDE.md, the confidence
    floor): the bar must reach the ONE production call of gate_frame_pair, be captured before
    the detector is deleted, and land in loop_edges.json beside every margin."""
    src = (ROOT / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()
    pairs = src[src.index("    def get_loop_pairs(self):"):src.index("        if self.useDBoW: # DBoW2")]
    # the bar is the STAMP's (loop_txt), fresh or reused — one path; the detector of a fresh
    # pass only cross-checks it
    assert ("self._stac_salad_bar = salad_applied_threshold(\n"
            "                loop_txt, detector=self.loop_detector if fresh else None)") in pairs
    # captured BEFORE `del self.loop_detector` (run order: get_loop_pairs, then the delete)
    assert src.index("self.get_loop_pairs()") < src.index("del self.loop_detector")
    # the record is set in ONE place (the helper): __init__ for a pass without a SALAD
    # detector, get_loop_pairs from the stamp; no hand-written {threshold: None, ...} anywhere
    init = src[src.index("    def __init__(self"):src.index("    def get_loop_pairs(self):")]
    assert ("self._stac_salad_bar = salad_applied_threshold(None, use_dbow=self.useDBoW,\n"
            "                                                       loop_enable=self.loop_enable)") in init
    plan = src[src.index("    def _stac_plan_bridges(self"):src.index("    def _stac_extra_frames(self")]
    assert plan.count("gate_mod.gate_frame_pair(") == 1 and "salad=salad_in" in plan
    assert "salad_in, salad_why = salad_gate_input(cand, salad_bar)" in plan
    assert "salad_bar = self._stac_salad_bar_record()" in plan
    assert 'gate["rules"]["salad"]["reason"] = salad_why' in plan
    verify = src[src.index("    def _stac_verify_loops(self"):src.index("    def _stac_graph_cfg(self)")]
    assert '"salad_threshold": self._stac_salad_bar_record()' in verify and "save_loop_report(" in verify
    assert src.count('"threshold": None') == 0 and "'origin'" not in src and '"origin"' not in src


def test_a_runner_without_init_reads_the_same_record_through_the_helper():
    """The fork's own stage tests build VGGT_Long with object.__new__: the accessor hands them
    the helper's 'no file read' record, by the same function, not a dict of its own."""
    src = (ROOT / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()
    acc = src[src.index("    def _stac_salad_bar_record(self):"):src.index("    def _stac_plan_bridges(self")]
    assert "bar = self._stac_salad_bar = salad_applied_threshold(None)" in acc
    assert "{" not in acc.split('"""')[-1], "no inline record in the accessor"


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
