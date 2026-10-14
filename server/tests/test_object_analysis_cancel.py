"""segmentation.object_analysis.ensure_session_analyses polls the caller's gate (``cancelled``)
before EVERY per-object dossier and before the scene description, and stops at the first True
('intel deferred') — docs/plan_determinismo.md point 157 (2026-10-08): an intel never runs
co-batched with a job's stage calls or keeps writing scene_r.db on a scan a job rewrites. CPU
only: the VLM calls are faked."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import object_analysis as OA                  # noqa: E402


def _session(tmp_path, n=4):
    out = tmp_path / "output"
    out.mkdir()
    (out / "segmentation_result.json").write_text(json.dumps(
        {"instances": [{"instance_id": i + 1, "label": f"obj{i + 1}"} for i in range(n)]}))
    return out


class _Store:
    def __init__(self):
        self.meta = {}

    def get_meta(self, k):
        return self.meta.get(k)


def _fake(monkeypatch, out, store):
    done, scene = [], []
    monkeypatch.setattr(OA, "get_cached_analysis", lambda o, iid: None)
    monkeypatch.setattr(OA, "analyze_object", lambda o, iid, log=print: done.append(iid) or {"que_es": "x"})
    monkeypatch.setattr(OA, "_store", lambda o: store)

    class _Tools:
        def __init__(self, st):
            pass

        def describe_scene(self):
            scene.append(1)
            store.meta["scene_description"] = "a room"
    import types
    mod = types.ModuleType("phase5_qa.tools")
    mod.SpatialTools = _Tools
    monkeypatch.setitem(sys.modules, "phase5_qa.tools", mod)
    return done, scene


def test_the_gate_is_polled_before_every_object_and_before_the_scene(tmp_path, monkeypatch):
    out = _session(tmp_path)
    store = _Store()
    done, scene = _fake(monkeypatch, out, store)
    polls = []

    def gate():
        polls.append(len(done))
        return len(done) >= 2               # a job takes the engine after the second dossier

    logs = []
    res = OA.ensure_session_analyses(out, objects=True, log=logs.append, cancelled=gate)
    assert done == [1, 2] and scene == []                      # stopped at the first True
    assert res == {"generated": 2, "cached": 0, "failed": 0, "deferred": True}
    assert polls == [0, 1, 2]                                  # asked before each call, no more
    assert any("intel deferred" in m and "instance 3" in m for m in logs)
    assert any("intel deferred" in m for m in logs if m.startswith("[analysis] session dossiers"))
    # the gate closes only before the scene description: every dossier done, no description
    done.clear(); scene.clear(); logs.clear()
    res = OA.ensure_session_analyses(out, objects=True, log=logs.append,
                                     cancelled=lambda: len(done) == 4)
    assert done == [1, 2, 3, 4] and scene == [] and res["deferred"] is True
    assert any("intel deferred before describe_scene" in m for m in logs)


def test_without_a_gate_or_an_open_one_everything_runs(tmp_path, monkeypatch):
    out = _session(tmp_path, n=3)
    store = _Store()
    done, scene = _fake(monkeypatch, out, store)
    res = OA.ensure_session_analyses(out, objects=True, log=lambda m: None)
    assert done == [1, 2, 3] and scene == [1]
    assert res == {"generated": 3, "cached": 0, "failed": 0, "deferred": False}
    done.clear(); scene.clear(); store.meta.clear()
    res = OA.ensure_session_analyses(out, objects=True, log=lambda m: None, cancelled=lambda: False)
    assert done == [1, 2, 3] and scene == [1] and res["deferred"] is False
    # main.py hands the gate: the parameter exists under that name
    import inspect
    assert "cancelled" in inspect.signature(OA.ensure_session_analyses).parameters
