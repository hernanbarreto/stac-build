"""What keeps a chat question at ~1 s outside the engine (USER 2026-11-13: "Determinista pero en 1
o 2 segundos ninguna es negociable"): config.yaml parsed once per file on disk, and the engine's
first call paid by the chat's warm-up, never by the user's first question."""
from __future__ import annotations

import time

import pytest


def test_the_semantic_config_is_parsed_once_per_file_and_never_shared(tmp_path, monkeypatch):
    from semantic import semantic_config as sc
    cfg = tmp_path / "config.yaml"
    cfg.write_text("semantic:\n  chat:\n    list_limit: 7\n")
    monkeypatch.setattr(sc, "_CONFIG_PATH", cfg)
    monkeypatch.setattr(sc, "_CACHE", {})
    parses = []
    real = sc.yaml.safe_load
    monkeypatch.setattr(sc.yaml, "safe_load", lambda f: parses.append(1) or real(f))

    a = sc.load_semantic_config()
    assert a["chat"]["list_limit"] == 7 and a["service"]["port"] == 8799     # defaults merged
    a["chat"]["list_limit"] = 99                      # a caller's own copy: nobody else sees it
    a["service"]["port"] = 1
    b = sc.load_semantic_config()
    assert b["chat"]["list_limit"] == 7 and b["service"]["port"] == 8799
    assert len(parses) == 1                           # the second call did not parse the file

    # an edit of the file is read at the next call (another size, another stamp)
    cfg.write_text("semantic:\n  chat:\n    list_limit: 12345\n")
    assert sc.load_semantic_config()["chat"]["list_limit"] == 12345
    assert len(parses) == 2

    # no file: the defaults, and nothing cached under a stamp that does not exist
    cfg.unlink()
    assert sc.load_semantic_config()["default_backend"] == "qwen_local"


def test_the_engines_first_call_is_the_warm_ups(monkeypatch):
    from phase5_qa import api

    calls = []

    class _Client:
        def chat(self, messages, tools=None, tool_choice=None, max_tokens=None, consumer=None):
            calls.append({"tools": len(tools or []), "tool_choice": tool_choice, "consumer": consumer})

    import semantic.client as client_mod
    monkeypatch.setattr(client_mod, "get_semantic_client", lambda **kw: _Client())
    monkeypatch.setattr(api, "_WARM", {"engine": None, "state": "cold", "seconds": None})
    engine = {"key": (111, 1.0)}
    monkeypatch.setattr(api, "_engine_key", lambda: engine["key"])

    assert api.engine_warm_state() == "cold"
    api.warm_engine()
    api.warm_engine()                                  # asked twice: ONE call
    deadline = time.time() + 20
    while api.engine_warm_state() != "warm" and time.time() < deadline:
        time.sleep(0.02)
    assert api.engine_warm_state() == "warm"
    assert len(calls) == 1
    # the chat's own shape: its tools' schemas and a required call (the grammar compiled then)
    assert calls[0]["tools"] > 20 and calls[0]["tool_choice"] == "required"
    api.warm_engine()                                  # warm: nothing is sent
    assert len(calls) == 1

    # another engine (restarted) is cold again; a question answered by it marks it warm
    engine["key"] = (222, 2.0)
    assert api.engine_warm_state() == "cold"
    api.mark_engine_warm()
    assert api.engine_warm_state() == "warm"
    assert len(calls) == 1

    # an engine with no identity record cannot be told apart: treated as warm, nothing sent
    engine["key"] = None
    assert api.engine_warm_state() == "warm"
    api.warm_engine()
    assert len(calls) == 1


def test_a_failed_warm_up_leaves_the_engine_cold(monkeypatch):
    from phase5_qa import api

    class _Client:
        def chat(self, *a, **k):
            raise RuntimeError("engine gone")

    import semantic.client as client_mod
    monkeypatch.setattr(client_mod, "get_semantic_client", lambda **kw: _Client())
    monkeypatch.setattr(api, "_WARM", {"engine": None, "state": "cold", "seconds": None})
    monkeypatch.setattr(api, "_engine_key", lambda: (333, 3.0))
    api.warm_engine()
    deadline = time.time() + 20
    while api._WARM["seconds"] is None and time.time() < deadline:
        time.sleep(0.02)
    assert api.engine_warm_state() == "cold"           # declared in the log; the question pays it


def test_the_tools_schemas_need_no_store():
    from phase5_qa.tools import SpatialTools
    names = [t["function"]["name"] for t in SpatialTools.openai_schemas()]
    assert len(names) == len(set(names)) and "get_clearance" in names and "small_talk" in names


def test_the_backend_warms_the_engine_it_started_when_it_answers(monkeypatch):
    """semantic.chat_on_demand: false (USER 2026-11-14: the chat's VLM "debe iniciar cuando levante
    el backend"): the backend asks for the engine at its start and ONE waiter sends the warm-up
    call when the engine answers — or gives up when a job takes the card meanwhile."""
    from phase5_qa import api
    import semantic.service as service

    looks = {"n": 0}

    def _alive(*a, **k):
        looks["n"] += 1
        return looks["n"] >= 3                         # loading, loading, up

    warmed = []
    monkeypatch.setattr(service, "is_alive", _alive)
    monkeypatch.setattr(api, "_engine_busy", lambda: None)
    monkeypatch.setattr(api, "warm_engine", lambda *a, **k: warmed.append(1))
    monkeypatch.setattr(api, "_WARM_POLL_S", 0.01)
    monkeypatch.setattr(api, "_WARM_WAITER", {"thread": None})

    api.warm_engine_when_up(10.0)
    api.warm_engine_when_up(10.0)                      # asked twice while waiting: one waiter
    api._WARM_WAITER["thread"].join(10)
    assert warmed == [1] and looks["n"] == 3

    # a job took the engine while it loaded: nothing is sent (that job's end asks again)
    looks["n"] = 0
    monkeypatch.setattr(api, "_engine_busy", lambda: "a pipeline is using the GPU")
    api.warm_engine_when_up(10.0)
    api._WARM_WAITER["thread"].join(10)
    assert warmed == [1] and looks["n"] == 0
