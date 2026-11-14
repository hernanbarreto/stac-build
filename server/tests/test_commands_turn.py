"""A turn of STAC with the commands of the interface (USER 2026-11-14: "le pedi que conecte con el
servidor y no conecto" / "le pedi que abra pccr y no abrio nada" / "debe poder discernir entre un
comando y una consulta de chat" / "debe preguntarnos primero el nombre, y hablarnos usando nuestro
nombre"): what the model chose is told apart by its row (action / query / talk), the scene's tools
run here, the interface's commands are handed to the interface with their texts in the
conversation's language — with no scan open too."""
from __future__ import annotations

import numpy as np
import pytest

from phase5_qa import answers
from phase5_qa.orchestrator import SpatialQA
from phase_r.geometry import fit_gravity_aligned_obb
from phase_r.instance_store import InstanceStore

UI = {"connected": True, "session": None, "sessions": ["pccr", "zaragoza"], "lang": "en"}


class _Call:
    def __init__(self, name, arguments, k):
        self.name, self.arguments, self.id = name, arguments, f"call_{k}"


class _Reply:
    def __init__(self, calls=()):
        self.content = ""
        self.tool_calls = [_Call(n, a, k) for k, (n, a) in enumerate(calls)]
        self.has_tool_calls = bool(self.tool_calls)


class _Engine:
    """The engine, scripted: it answers the one call of a turn with the given tool calls."""

    def __init__(self, calls):
        self.reply, self.chats = _Reply(calls), []

    def chat(self, messages, tools=None, tool_choice=None, max_tokens=None, consumer=None, **_kw):
        self.chats.append({"system": messages[0].text, "user": messages[-1].text, "n_messages": len(messages),
                           "tools": [t["function"]["name"] for t in (tools or [])], "tool_choice": tool_choice})
        return self.reply


def _store(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "scene_r.db"
    store = InstanceStore(path)
    store.set_meta("built_from", "test_display_frame")
    for iid, label, centre in ((1, "floor", [0, 0, 0]), (2, "wall", [0, 1.3, 4]), (3, "door", [1, 1, 4])):
        pts = rng.normal(size=(400, 3)) * [1.0, 0.02, 1.0] + centre
        store.upsert_instance(iid, label, source="test")
        store.set_points(iid, pts)
        obb = fit_gravity_aligned_obb(np.asarray(pts, np.float32).astype(np.float64))
        store.set_obb(iid, obb[0], obb[1], obb[2], n_points=len(pts))
    store.close()
    return str(path)


def _ask(monkeypatch, tmp_path, calls, question, store=None, **kw):
    import semantic.client as client_mod
    engine = _Engine(calls)
    monkeypatch.setattr(client_mod, "get_semantic_client", lambda **_kw: engine)
    qa = SpatialQA(store, session_log_dir=tmp_path / "logs")
    try:
        return qa.ask(question, **kw), engine
    finally:
        qa.close()


def test_a_command_of_the_interface_with_no_scan_open(tmp_path, monkeypatch):
    res, engine = _ask(monkeypatch, tmp_path, [("sessions__open", {"session": "pccr"})], "abrí pccr",
                       ui=UI, user_name="Ana")
    chat = engine.chats[0]
    # no scan: none of the scene's tools is offered, the model is told it cannot measure
    assert "get_distance" not in chat["tools"] and "count_objects" not in chat["tools"]
    assert chat["tools"][:2] == ["small_talk", "help__commands"] and "sessions__open" in chat["tools"]
    assert "NO SCAN WITH OBJECTS IS OPEN" in chat["system"] and chat["tool_choice"] == "required"
    assert "sessions on the server: pccr, zaragoza" in chat["system"]
    assert "their name is Ana" in chat["system"]
    # the command goes to the interface with what to say when it is done, in the conversation's language
    assert res["lang"] == "es" and res["answer"] == "" and res["stopped"] == "composed_by_tools"
    assert res["ui_actions"] == [{"id": "sessions.open", "tool": "sessions__open", "args": {"session": "pccr"},
                                  "kind": "action", "confirm": False, "title": "Abrir una sesión",
                                  "done": "Abriendo la sesión {detail}.", "done_off": ""}]
    assert res["acted_only"] is True and res["texts"]["done_prefix"] == "Listo, Ana."
    assert res["texts"]["reasons"]["not_found"] == "No encontré «{detail}»."
    assert res["commands"] == [{"id": "sessions.open", "kind": "action", "runs_on": "ui"}]


def test_what_tells_a_command_from_a_question_is_the_row_it_picks(tmp_path, monkeypatch):
    sp = _store(tmp_path)
    ui = {**UI, "session": "pccr", "scan": "2026-08-31/default"}
    # an ACTION of the interface and a QUERY of the scene in one sentence
    res, engine = _ask(monkeypatch, tmp_path, [("app__fullscreen", {"on": True}), ("count_objects", {"label": "door"})],
                       "ampliá la pantalla y decime cuántas puertas hay", store=sp, ui=ui, user_name="Ana")
    tools = engine.chats[0]["tools"]
    assert "count_objects" in tools and "app__fullscreen" in tools            # the scene's and the candidates
    assert "THE APPLICATION" in engine.chats[0]["system"] and "session open: pccr" in engine.chats[0]["system"]
    assert [c["kind"] for c in res["commands"]] == ["action", "query"]
    assert [c["runs_on"] for c in res["commands"]] == ["ui", "scene"]
    assert res["answer"] == "Hay 1 objetos «door»."                           # the scene's part, worded here
    assert [a["id"] for a in res["ui_actions"]] == ["app.fullscreen"]
    assert res["ui_actions"][0]["done"] == "Pantalla completa." and res["ui_actions"][0]["done_off"] == "Salí de pantalla completa."
    assert res["acted_only"] is False                                         # it also answered a question
    # a QUERY alone: the first answer of a conversation starts with the user's name
    res, _ = _ask(monkeypatch, tmp_path, [("count_objects", {"label": "door"})], "cuántas puertas hay?",
                  store=sp, ui=ui, user_name="Ana")
    assert res["answer"] == "Ana, hay 1 objetos «door»." and res["ui_actions"] == []
    res, _ = _ask(monkeypatch, tmp_path, [("count_objects", {"label": "door"})], "y ahora?", store=sp, ui=ui,
                  user_name="Ana", history=[{"role": "user", "text": "hola"}, {"role": "assistant", "text": "Hola, Ana."}])
    assert res["answer"] == "Hay 1 objetos «door»."                           # … and only the first
    # an ACTION of the scene alone is confirmed by name
    res, _ = _ask(monkeypatch, tmp_path, [("show_only", {"label": "door"})], "mostrame solo las puertas",
                  store=sp, ui=ui, user_name="Ana", history=[{"role": "user", "text": "hola"}, {"role": "assistant", "text": "Hola."}])
    assert res["answer"] == "Listo, Ana. Se muestran solo 1 objetos (door)." and res["acted_only"] is True
    # TALK: the model's own reply — unless it carries a figure with a unit
    res, _ = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "¡Hola, Ana! ¿Qué necesitas?"})], "hola stac",
                  store=sp, ui=ui, user_name="Ana")
    assert res["answer"] == "¡Hola, Ana! ¿Qué necesitas?" and res["commands"][0]["kind"] == "talk"
    res, _ = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "La pared mide 3.2 m."})], "hola", store=sp, ui=ui)
    assert res["answer"] == answers.small_talk_default("es")


def test_the_three_languages(tmp_path, monkeypatch):
    sp = _store(tmp_path)
    for lang, question, want, done in (
            ("es", "ocultá la grilla", "Hay 3 objetos. Los más numerosos: door 1, floor 1, wall 1.", ("Grilla visible.", "Grilla oculta.")),
            ("en", "hide the grid", "There are 3 objects. The most numerous: door 1, floor 1, wall 1.", ("Grid shown.", "Grid hidden.")),
            ("fr", "masque la grille", "Il y a 3 objets. Les plus nombreux : door 1, floor 1, wall 1.", ("Grille affichée.", "Grille masquée."))):
        res, _ = _ask(monkeypatch, tmp_path, [("count_objects", {}), ("display__grid", {"on": False})], question,
                      store=sp, lang=lang)
        assert res["lang"] == lang and res["answer"] == want
        assert (res["ui_actions"][0]["done"], res["ui_actions"][0]["done_off"]) == done
        assert set(res["texts"]["reasons"]) >= {"no_session", "not_found", "ambiguous", "needs_gesture", "server_down", "not_wired"}
    # a written sentence's language is its own; the interface's when it cannot be told
    assert answers.language_of("ouvre la session et affiche les axes") == "fr"
    assert answers.language_of("open the session and show the axes") == "en"
    assert answers.language_of("abrí la sesión y mostrá los ejes") == "es"
    assert answers.language_of("no te voy a decir") == "es"
    assert answers.language_of("pccr", default="fr") == "fr" and answers.language_of("pccr", default="xx") == "es"


def test_stac_asks_the_name_and_never_stores_one_that_was_not_said(tmp_path, monkeypatch):
    # the answer gives a name: the name's command is offered first, beside the sentence's own
    res, engine = _ask(monkeypatch, tmp_path, [("talk__set_user_name", {"name": "Hernán"})], "me llamo Hernán Barreto",
                       ui=UI, expecting="user_name", lang="es")
    assert {"talk__set_user_name", "small_talk", "help__commands"} <= set(engine.chats[0]["tools"])
    assert "STAC HAS JUST ASKED THE USER THEIR NAME" in engine.chats[0]["system"]
    assert res["ui_actions"][0]["args"] == {"name": "Hernán"} and res["expecting"] is None
    assert res["ui_actions"][0]["done"] == "Encantado, {name}. Dime qué necesitas." and res["answer"] == ""
    # the name AND a request in one sentence (he said "My name is Hernan. Could you connect the server"):
    # both are done
    res, _ = _ask(monkeypatch, tmp_path, [("talk__set_user_name", {"name": "Hernan"}), ("app__connect", {})],
                  "Hello, STAC Build. My name is Hernan. Could you connect the server, please?", ui=UI, expecting="user_name")
    assert res["lang"] == "en" and [a["id"] for a in res["ui_actions"]] == ["talk.set_user_name", "app.connect"]
    assert res["expecting"] is None and res["texts"]["tail"] == ""
    # the model "extracts" a name the user did not say: nothing is stored and STAC asks again itself
    for calls in ([("talk__set_user_name", {"name": "Hernán"})],
                  [("small_talk", {"reply": "¿Cómo te llamas?"}), ("talk__set_user_name", {"name": "Hernán"})],
                  [("small_talk", {"reply": "cualquier cosa"})]):
        res, _ = _ask(monkeypatch, tmp_path, calls, "no te voy a decir", ui=UI, expecting="user_name", lang="es")
        assert res["ui_actions"] == [] and res["expecting"] == "user_name"
        assert res["answer"] == answers.ask_name("es") == "Hola, soy STAC. ¿Cómo te llamas?"
    # no name, but a request: it is done, and the name is asked once more after it
    res, _ = _ask(monkeypatch, tmp_path, [("sessions__open", {"session": "pccr"})], "abrí pccr", ui=UI, expecting="user_name")
    assert [a["id"] for a in res["ui_actions"]] == ["sessions.open"] and res["expecting"] == "user_name"
    assert res["texts"]["tail"] == "¿Y cómo te llamas?"
    res, _ = _ask(monkeypatch, tmp_path, [("help__commands", {})], "qué puedes hacer?", ui=UI, expecting="user_name")
    assert res["expecting"] == "user_name" and res["answer"].endswith("¿Y cómo te llamas?") and res["texts"]["tail"] == ""
    assert answers.name_in_sentence("Hernán", "soy hernan, mucho gusto") and answers.name_in_sentence("Ana María", "me llamo ana maría")
    assert not answers.name_in_sentence("Hernán", "no te voy a decir") and not answers.name_in_sentence("", "ana")
    # outside the question too: a name is stored only when it was said
    res, engine = _ask(monkeypatch, tmp_path, [("talk__set_user_name", {"name": "Pedro"})], "mi nombre no importa",
                       ui=UI, lang="es")
    assert "talk__set_user_name" in engine.chats[0]["tools"]
    # (the refused call leaves the turn: it is not an error to tell the user about)
    assert res["ui_actions"] == [] and res["tool_trace"] == [] and res["answer"] == answers.small_talk_default("es")
    # another person's name said in the sentence is still the model's to leave alone — when it
    # does call the command with it, the name WAS said and is stored: the description tells the
    # model the command is for the user's OWN name
    res, engine = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "Hello, Virginia!"})],
                       "the name of my wife is Virginia. Say hello, please.", ui=UI, user_name="Hernan")
    assert res["lang"] == "en" and res["answer"] == "Hello, Virginia!" and res["ui_actions"] == []
    assert "THEIR OWN name" in next(t for t in __import__("commands").database().tool_schemas(["talk.set_user_name"]))["function"]["description"]
    # … and a command the sentence was not offered does not run at all
    res, engine = _ask(monkeypatch, tmp_path, [("talk__set_user_name", {"name": "Pedro"})], "gracias", ui=UI, lang="es")
    assert "talk__set_user_name" not in engine.chats[0]["tools"]
    assert res["ui_actions"] == [] and "unknown tool" in res["answer"]


def test_the_answers_language_is_the_sentences_then_the_conversations(tmp_path, monkeypatch):
    """USER 2026-11-14: "a veces le hablo en ingles y contesta en español, debe mantener el idioma de
    la conversacion". The recogniser's language token is a hint, not the verdict (it said Spanish
    for his English while writing the English words)."""
    sp = _store(tmp_path)
    for question, hint, conversation, want in (
            ("Connect the server, please.", "es", None, "en"),            # the words tell: English
            ("Please open observatory.", "es", "es", "en"),
            ("Describe what I are seeing in this scene.", "es", None, "en"),
            ("abrí pccr", "en", "en", "es"),
            ("ouvre la session", "es", "es", "fr"),
            ("PCCR", "es", "en", "en"),                                   # no telling: the conversation's
            ("PCCR", "fr", None, "fr"),                                   # … then the recogniser's
            ("PCCR", None, None, "en")):                                  # … then the interface's (UI: en)
        res, _ = _ask(monkeypatch, tmp_path, [("count_objects", {})], question, store=sp, ui=UI, lang=hint,
                      conversation_lang=conversation)
        assert res["lang"] == want, (question, hint, conversation)
    # a description the VLM wrote in Spanish is said in the conversation's language: translated
    # once by the model, kept in the store
    from phase_r.instance_store import InstanceStore
    from phase5_qa.tools import SpatialTools
    import semantic.client as client_mod
    store = InstanceStore(sp)
    store.set_meta("scene_description", "Sala técnica con 3 racks.")
    asked = []

    class _Translator:
        def chat(self, messages, **_kw):
            asked.append(messages[0].text)
            return type("R", (), {"content": "Technical room with 3 racks."})()
    monkeypatch.setattr(client_mod, "get_semantic_client", lambda **_kw: _Translator())
    try:
        t = SpatialTools(store, device="numpy")
        assert t.describe_scene()["description"] == "Sala técnica con 3 racks." and asked == []     # Spanish: as written
        t.lang = "en"
        assert t.describe_scene()["description"] == "Technical room with 3 racks."
        assert t.describe_scene()["description"] == "Technical room with 3 racks." and len(asked) == 1   # kept
        assert "Spanish to English" in asked[0] and store.get_meta("scene_description@en") == "Technical room with 3 racks."
    finally:
        store.close()


def test_a_call_that_is_not_the_commands_is_refused_by_the_tool_layer(tmp_path, monkeypatch):
    res, _ = _ask(monkeypatch, tmp_path, [("display__point_opacity", {"opacity": 7})], "opacidad 7", ui=UI, lang="es")
    assert res["ui_actions"] == [] and "display.point_opacity: 'opacity' is over 1" in res["answer"]
    assert res["acted_only"] is False
    res, _ = _ask(monkeypatch, tmp_path, [("no_such_tool", {})], "algo", ui=UI, lang="en")
    assert res["ui_actions"] == [] and "unknown tool" in res["answer"]


def test_how_the_user_is_addressed():
    assert answers.done_prefix("Ana", "es") == "Listo, Ana." and answers.done_prefix("Ana", "fr") == "C'est fait, Ana."
    assert answers.done_prefix(None, "en") == "" and answers.done_prefix("  ", "en") == ""
    say = answers.address
    assert say("Vista cambiada.", "Ana", "es", acted=True, first=False) == "Listo, Ana. Vista cambiada."
    assert say("La holgura es 1 m.", "Ana", "es", acted=False, first=True) == "Ana, la holgura es 1 m."
    assert say("wall#110: desplome 0.1°.", "Ana", "es", acted=False, first=True) == "Ana, wall#110: desplome 0.1°."
    assert say("BIM cargado.", "Ana", "es", acted=False, first=True) == "Ana, BIM cargado."          # an acronym stays
    assert say("La holgura es 1 m.", "Ana", "es", acted=False, first=False) == "La holgura es 1 m."
    assert say("Hola, Ana.", "Ana", "es", acted=True, first=True) == "Hola, Ana."                    # already said
    assert say("Vista cambiada.", None, "es", acted=True, first=True) == "Vista cambiada."


def test_the_conversation_is_in_the_turns_language(tmp_path, monkeypatch):
    """USER 2026-11-14: "a veces le hablo en ingles y contesta en español". The model is TOLD the
    answer's language, and a reply whose own words are another language's is not said."""
    res, engine = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "¡Hola, Hernan! Aquí estoy, listo para ayudarte con el escenario."})],
                       "this place is right for kids.", ui=UI, user_name="Hernan", conversation_lang="es")
    assert res["lang"] == "en"                                           # the sentence's own words
    assert "THE ANSWER'S LANGUAGE: English" in engine.chats[0]["system"]
    assert res["answer"] == answers.small_talk_default("en")              # the Spanish reply is not said
    res, _ = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "I am fine, thank you. What do you need?"})],
                  "how are you today?", ui=UI)
    assert res["answer"] == "I am fine, thank you. What do you need?"
    res, engine = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "Très bien, merci."})], "comment ça va ?", ui=UI)
    assert res["lang"] == "fr" and res["answer"] == "Très bien, merci." and "THE ANSWER'S LANGUAGE: French" in engine.chats[0]["system"]
