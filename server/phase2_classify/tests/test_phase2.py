# STAC-Builder — Phase 2 classification unit tests (no VLM required).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase2_classify.classify import (  # noqa: E402
    _best_masklet, _parse, is_structural, route_for,
)
from phase_r.instance_store import InstanceStore  # noqa: E402


def test_is_structural_multiword():
    # multi-word structural classes must be eligible (was exact-match before)
    assert is_structural("tiled floor")
    assert is_structural("concrete wall")
    assert is_structural("column")
    assert not is_structural("cable tray")
    assert not is_structural("scaffolding")


def test_route_consistency_with_eligibility():
    # routing and R.5 whitelist eligibility use the SAME matcher
    for cls in ["tiled floor", "brick wall", "vault", "pipe", "person"]:
        assert (route_for(cls) == "surface_fitting") == is_structural(cls)


def test_parse_tolerates_wrapping():
    assert _parse('noise {"class": "wall", "confidence": 0.8} trailing')["class"] == "wall"
    assert _parse("no json here") is None


def test_best_masklet_prefers_area_and_score():
    with tempfile.TemporaryDirectory() as d:
        st = InstanceStore(os.path.join(d, "s.db"))
        st.upsert_instance(1, "wall")
        st.add_masklet_ref(1, 0, "f0_o1", box_xywh=np.array([0.1, 0.1, 0.1, 0.1]), score=0.9)
        st.add_masklet_ref(1, 5, "f5_o1", box_xywh=np.array([0.1, 0.1, 0.5, 0.5]), score=0.8)
        best = _best_masklet(st, 1)
        assert best[1] == 5  # bigger area wins despite slightly lower score
        st.close()


def test_classification_roundtrip_updates_r5_eligibility():
    """Phase 2 -> R.5 feedback loop: whitelist_eligible persisted and read back
    exactly as the depth-regularization writeback consumes it."""
    with tempfile.TemporaryDirectory() as d:
        st = InstanceStore(os.path.join(d, "s.db"))
        st.upsert_instance(1, "obj12")  # raw prompting label is NOT structural
        st.set_classification(1, class_final="tiled floor", material="ceramic",
                              state="good", confidence=0.9,
                              whitelist_eligible=is_structural("tiled floor"))
        c = st.get_classification(1)
        assert c["whitelist_eligible"] is True
        assert c["origin"] == "vlm_proposed"
        # the route table entry Phase 2 persists
        st.set_meta("route_1", route_for(c["class_final"]))
        assert st.get_meta("route_1") == "surface_fitting"
        st.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all phase-2 tests passed")
