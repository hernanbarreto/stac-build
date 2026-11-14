"""The spatial chat around its tools (USER 2026-11-13: "el chat no supo encontrar … sign#797,
estaba bien escrito … la cantidad de elementos que encontro eran menores a la cantidad de la
lista … muchas cosas no las entiende, no las encuentra"): the scan it answers about, the whole
inventory, the objects a question names by id, the bounded listings, the conversation's turns
and the provenance check of every figure."""
from types import SimpleNamespace

import numpy as np
import pytest

from phase5_qa import provenance
from phase5_qa import tools as chat_tools
from phase5_qa.orchestrator import SpatialQA, _inventory
from phase5_qa.tools import SpatialTools, id_ranges, norm_label
from phase_r.geometry import fit_gravity_aligned_obb
from phase_r.instance_store import InstanceStore


def _store(tmp_path):
    rng = np.random.default_rng(0)
    store = InstanceStore(tmp_path / "scene_r.db")
    store.set_meta("built_from", "test_display_frame")
    labels = {1: "floor", 2: "wall", 3: "wall", 5: "wall", 6: "fire_extinguisher",
              7: "sign", 8: "sign", 20: "door", 21: "door", 22: "door", 40: "door"}
    for k, (iid, label) in enumerate(labels.items()):
        pts = rng.normal(size=(200, 3)) * [0.3, 0.3, 0.05] + [2.0 * k, 1.0, 0.0]
        store.upsert_instance(iid, label, source="test")
        store.set_points(iid, pts)
        obb = fit_gravity_aligned_obb(np.asarray(pts, np.float32).astype(np.float64))
        store.set_obb(iid, obb[0], obb[1], obb[2], n_points=len(pts))
    store.close()
    return str(tmp_path / "scene_r.db")


def _limits(monkeypatch, **kw):
    cfg = dict(chat_tools.chat_config())
    cfg.update(kw)
    monkeypatch.setattr(chat_tools, "chat_config", lambda: dict(cfg))


# ── names, ranges, listings ─────────────────────────────────────────────────────────────────
def test_labels_are_compared_as_people_write_them():
    assert norm_label("Fire_Extinguisher") == norm_label("fire extinguisher") == "fire extinguisher"
    assert norm_label("  metal-frame ") == "metal frame"
    assert id_ranges([4, 2, 3, 9, 11, 12, 2]) == "2-4, 9, 11-12"


def test_list_and_count_take_the_label_asked(tmp_path, monkeypatch):
    sp = _store(tmp_path)
    store = InstanceStore(sp)
    try:
        t = SpatialTools(store, device="numpy")
        assert [(o["id"], o["points"]) for o in t.list_objects("fire extinguisher")["objects"]] == [(6, 200)]
        assert t.count_objects("signs") == {"count": 2, "unit": "count", "label": "signs",
                                            "matched_labels": ["sign"]}
        total = t.count_objects()
        assert total["count"] == 11 and total["by_label"]["door"] == 4
        assert t.count_objects("crane")["count"] == 0
        # beyond the bound a listing is by label, with every id as ranges
        _limits(monkeypatch, list_limit=3)
        out = t.list_objects()
        assert out["listed"] == "by_label" and out["count"] == 11
        # … and the label's largest object (most points; the lowest id among equals)
        assert {"label": "door", "count": 4, "ids": "20-22, 40", "largest_id": 20} in out["labels"]
        assert {"label": "floor", "count": 1, "ids": "1"} in out["labels"]
        assert t.get_object(7)["label"] == "sign" and t.get_object(99)["insufficient_data"]
    finally:
        store.close()


def test_the_inventory_names_every_object(tmp_path, monkeypatch):
    sp = _store(tmp_path)
    store = InstanceStore(sp)
    try:
        whole = _inventory(store)                       # few objects: one line each
        assert "id=7 sign" in whole and "id=40 door" in whole
        _limits(monkeypatch, list_limit=3)
        by_label = _inventory(store)
        assert "11 objects" in by_label
        assert "sign (2): ids 7-8 — largest: 7" in by_label
        assert "door (4): ids 20-22, 40 — largest: 20" in by_label
        assert "floor (1): ids 1\n" in by_label + "\n"
        _limits(monkeypatch, list_limit=1)              # a label scattered over more runs
        assert "door (4): ids between 20 and 40, not contiguous — list_objects('door')" in _inventory(store)
    finally:
        store.close()


def test_objects_named_by_id_are_resolved_on_the_store(tmp_path):
    sp = _store(tmp_path)
    qa = SpatialQA(sp)
    try:
        asked, refs = qa.question_with_references(
            "distancia entre fire_extinguisher#6 y el objeto sign#7, y el sign#99, y el id 21, y #2")
        by_id = {r["id"]: r for r in refs}
        assert by_id[6]["exists"] and by_id[6]["label"] == "fire_extinguisher" and "label_asked" not in by_id[6]
        assert by_id[7]["label"] == "sign" and len(by_id[7]["position_m"]) == 3
        assert not by_id[99]["exists"] and by_id[99]["ids_with_that_label"] == "7-8"
        assert by_id[21]["label"] == "door" and by_id[2]["label"] == "wall"
        assert "[REFERENCED OBJECTS" in asked and "id 99: NO object with this id" in asked
        # a label that is not the id's own is said, never silently accepted
        _a, refs = qa.question_with_references("mostrame el sign#20")
        assert refs[0]["label"] == "door" and refs[0]["label_asked"] == "sign"
        assert qa.question_with_references("cuantas puertas hay?") == ("cuantas puertas hay?", [])
    finally:
        qa.close()


def test_the_chat_answers_about_the_scan_asked(tmp_path, monkeypatch):
    from phase5_qa import api
    import project_paths
    seen = []

    def resolve(server_dir, session_id, scan_key=None):
        seen.append((session_id, scan_key))
        d = tmp_path / (scan_key or "active").replace("/", "_")
        d.mkdir(exist_ok=True)
        (d / "scene_r.db").write_bytes(b"")
        return SimpleNamespace(output_dir=d)
    monkeypatch.setattr(project_paths, "resolve_session", resolve)
    assert api._resolve_store({"session_id": "pccr", "scan": "2026-08-31/default"}).endswith(
        "2026-08-31_default/scene_r.db")
    assert api._resolve_store({"session_id": "pccr"}).endswith("active/scene_r.db")
    assert seen == [("pccr", "2026-08-31/default"), ("pccr", None)]
    assert api._resolve_store({}) is None


def test_a_label_or_a_written_reference_stands_for_an_id(tmp_path):
    sp = _store(tmp_path)
    store = InstanceStore(sp)
    try:
        t = SpatialTools(store, device="numpy")
        assert t.resolve_id(7) == (7, None) and t.resolve_id("7") == (7, None)
        assert t.resolve_id("sign#8") == (8, None) and t.resolve_id("#21") == (21, None)
        # a label: its largest object (all have 200 points here: the lowest id among equals)
        assert t.resolve_id("wall") == (2, {"label": "wall", "id": 2})
        assert t.resolve_id("fire extinguisher") == (6, {"label": "fire_extinguisher", "id": 6})
        assert t.resolve_id("crane") == ("crane", None)
        out = t.impls()["get_distance"](id1="floor", id2="sign#7")
        assert out["objects_taken_by_label"] == {"id1": {"label": "floor", "id": 1}}
        assert "distance_m" in out
        assert t.impls()["small_talk"]()["ok"] is True
    finally:
        store.close()


# ── the provenance check ────────────────────────────────────────────────────────────────────
NAMES = ["measure_between", "get_distance", "count_objects", "get_object", "travel"]


def test_a_figure_no_tool_returned_is_found():
    trace = [{"tool": "measure_between", "arguments": {"id1": 72, "id2": 141},
              "result": {"distance_m": 0.1462, "point_a_m": [1.4895, 1.7878, -0.0893]}}]
    ok = provenance.check("La altura libre es de **0.1462 metros** (unos **14.6 cm**), con "
                          "measure_between.", trace, tool_names=NAMES)
    assert ok == {"ok": True, "figures": [], "tools": []}
    bad = provenance.check("La distancia es de **1.14 metros**, medida con `get_distance`.", [],
                           tool_names=NAMES, context_numbers=[-3.903, 1.489, -0.809])
    assert bad == {"ok": False, "figures": ["1.14 metros"], "tools": ["get_distance"]}
    # positions and sizes written as vectors are checked number by number; id lists are not
    vec = "Está en [-3.903, 1.489, -0.809] metros. Hay 56 puertas (IDs: 2, 5, 7)."
    assert provenance.check(vec, [], tool_names=NAMES, context_numbers=[-3.903, 1.489, -0.809])["ok"]
    assert provenance.check(vec, [], tool_names=NAMES)["figures"] == ["-3.903", "1.489", "-0.809"]


def test_rounding_units_and_the_users_own_numbers_are_allowed():
    trace = [{"tool": "get_distance", "arguments": {}, "result": {"distance_m": 1.3002}}]
    for text in ("1.30 meters", "1.3 m", "130 cm", "1300 mm", "1,30 metros", "about 1 m"):
        assert provenance.check(f"The gap is {text}.", trace, tool_names=NAMES)["ok"], text
    for text in ("1.35 m", "13 cm", "2 m"):
        assert not provenance.check(f"The gap is {text}.", trace, tool_names=NAMES)["ok"], text
    deg = [{"tool": "get_plumb", "arguments": {}, "result": {"plumb_deviation_deg": 0.42,
                                                              "deviation_mm_per_m": 7.33}}]
    assert provenance.check("Desplome de 0.42° (7.33 mm/m).", deg, tool_names=NAMES)["ok"]
    assert provenance.check("Un cubo de 2 m³.", [], tool_names=NAMES,
                            context_numbers=provenance.numbers_in("definí un cubo de 2 m3"))["ok"]
    assert provenance.figures_in("faltan 3 más y 5 metal_frame") == []
    assert "1.14 metros" in provenance.retry_message({"figures": ["1.14 metros"], "tools": []})


class _FakeClient:
    """run_tool_loop scripted: each step names the tools it calls and the answer it gives."""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), []

    def run_tool_loop(self, messages, tools, tool_impls, max_iterations=None, consumer=None,
                      first_tool_choice=None, **_kw):
        self.calls.append({"messages": list(messages), "first_tool_choice": first_tool_choice,
                           "consumer": consumer})
        step = self.steps.pop(0)
        trace = [{"iteration": k, "tool": name, "arguments": args, "result": tool_impls[name](**args)}
                 for k, (name, args) in enumerate(step.get("tools", []))]
        answer = step["answer"](trace) if callable(step["answer"]) else step["answer"]
        return {"answer": answer, "iterations": len(trace) + 1, "tool_trace": trace,
                "messages": list(messages), "stopped": "final_answer"}


def _qa(tmp_path, monkeypatch, steps):
    import semantic.client as client_mod
    qa = SpatialQA(_store(tmp_path), session_log_dir=tmp_path / "logs")
    fake = _FakeClient(steps)
    monkeypatch.setattr(client_mod, "get_semantic_client", lambda **_kw: fake)
    return qa, fake


def test_an_invented_figure_sends_the_answer_back_with_a_tool_required(tmp_path, monkeypatch):
    qa, fake = _qa(tmp_path, monkeypatch, [
        {"answer": "La distancia es de 1.14 metros, medida con get_distance."},
        {"tools": [("get_distance", {"id1": 6, "id2": 7})],
         "answer": lambda tr: f"La distancia es de {tr[0]['result']['distance_m']} m (get_distance)."}])
    try:
        res = qa.ask("a que distancia esta el fire_extinguisher#6 del sign#7?")
    finally:
        qa.close()
    assert res["provenance"] == {"ok": True, "figures": [], "tools": [], "retried": True}
    assert [c["first_tool_choice"] for c in fake.calls] == ["required", "required"]
    assert [e["tool"] for e in res["tool_trace"]] == ["get_distance"]
    assert "PROVENANCE CHECK" in fake.calls[1]["messages"][-1].text
    assert [r["id"] for r in res["references"]] == [6, 7]


def test_a_measured_answer_is_not_asked_again(tmp_path, monkeypatch):
    qa, fake = _qa(tmp_path, monkeypatch, [
        {"tools": [("get_distance", {"id1": 6, "id2": 7})],
         "answer": lambda tr: f"{tr[0]['result']['distance_m']} m"}])
    try:
        res = qa.ask("distancia entre #6 y #7")
    finally:
        qa.close()
    assert res["provenance"]["ok"] and not res["provenance"]["retried"] and len(fake.calls) == 1


def test_what_still_fails_is_declared_with_the_answer(tmp_path, monkeypatch):
    qa, _fake = _qa(tmp_path, monkeypatch, [{"answer": "Mide 3.21 m."}, {"answer": "Mide 3.21 m."}])
    try:
        res = qa.ask("cuanto mide el muro?")
    finally:
        qa.close()
    assert res["provenance"] == {"ok": False, "figures": ["3.21 m"], "tools": [], "retried": True}


def test_the_last_turns_of_the_conversation_go_with_the_question(tmp_path, monkeypatch):
    _limits(monkeypatch, history_turns=2)
    qa, fake = _qa(tmp_path, monkeypatch, [{"answer": "ok"}])
    history = [{"role": "user", "text": "q1"}, {"role": "assistant", "text": "a1"},
               {"role": "user", "text": "q2"}, {"role": "assistant", "text": "a2"},
               {"role": "tool", "text": "ignored"}, {"role": "user", "text": "  "}]
    try:
        qa.ask("y el otro?", history=history)
    finally:
        qa.close()
    msgs = fake.calls[0]["messages"]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert [m.text for m in msgs[1:]] == ["q2", "a2", "y el otro?"]
