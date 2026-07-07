# STAC-Builder — Phase 7 unit tests (scoring logic + coexistence, no model).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.instance_store import InstanceStore  # noqa: E402
from phase7_validation.qa_runner import QASuiteRunner  # noqa: E402
from phase7_validation import coexistence  # noqa: E402


def _store(path):
    st = InstanceStore(path)
    st.upsert_instance(1, "wall")
    st.upsert_instance(2, "wall")
    st.upsert_instance(3, "door")
    st.close()


def _runner(path):
    _store(path)
    r = QASuiteRunner(path)
    return r


def _trace(*tools):
    return [{"iteration": i, "tool": t, "arguments": {}, "result": {}} for i, t in enumerate(tools)]


def test_suite_loads_and_has_traps():
    with tempfile.TemporaryDirectory() as d:
        r = _runner(os.path.join(d, "s.db"))
        cats = [q["category"] for q in r.suite]
        assert cats.count("trap") >= 5
        assert cats.count("health") >= 3
        assert len(r.suite) >= 25


def test_count_all_and_label():
    with tempfile.TemporaryDirectory() as d:
        r = _runner(os.path.join(d, "s.db"))
        ok, _ = r._score({"check": {"type": "count_all"}}, "There are 3 objects.", [])
        assert ok
        ok2, _ = r._score({"check": {"type": "count_label", "label": "wall"}},
                          "I count 2 walls.", [])
        assert ok2
        bad, _ = r._score({"check": {"type": "count_label", "label": "wall"}},
                          "There are 5 walls.", [])
        assert not bad


def test_insufficient_detection():
    with tempfile.TemporaryDirectory() as d:
        r = _runner(os.path.join(d, "s.db"))
        for ans in ["Data is insufficient.", "I cannot determine that.",
                    "Object 9999 does not exist.", "The tools do not provide temperature."]:
            ok, _ = r._score({"check": {"type": "insufficient"}}, ans, [])
            assert ok, ans
        ok, _ = r._score({"check": {"type": "insufficient"}}, "It is 3.2 metres.", [])
        assert not ok


def test_measured_needs_tool_and_number():
    with tempfile.TemporaryDirectory() as d:
        r = _runner(os.path.join(d, "s.db"))
        chk = {"check": {"type": "measured", "tools": ["get_object_size"]}}
        ok, _ = r._score(chk, "The size is 3.45 m wide.", _trace("get_object_size"))
        assert ok
        no_tool, _ = r._score(chk, "The size is 3.45 m.", _trace("list_objects"))
        assert not no_tool
        no_num, _ = r._score(chk, "It is quite large.", _trace("get_object_size"))
        assert not no_num


def test_health_needs_tool_and_keyword():
    with tempfile.TemporaryDirectory() as d:
        r = _runner(os.path.join(d, "s.db"))
        chk = {"check": {"type": "health_bimodal", "tools": ["get_alignment_health"]}}
        ok, _ = r._score(chk, "It is bimodal with separation 0.2 m.",
                         _trace("get_alignment_health"))
        assert ok
        bad, _ = r._score(chk, "It looks fine.", _trace("get_alignment_health"))
        assert not bad


def test_coexistence_recommendation():
    r = coexistence.probe(recon_headroom_needed_mib=15000)
    assert "gpu" in r
    if "free_mib" in r["gpu"]:
        assert "coexists" in r and "recommendation" in r


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name); n += 1
    print(f"\n{n} tests passed")
