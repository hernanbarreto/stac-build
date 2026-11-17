"""THE SPOKEN CONFIRMATION (USER 2026-11-17: "confirmacion hablada con lo que stac nos dice que va
a hacer, STAC, SI CONFIRMO"): a command declared ``confirm: true`` goes to the interface with what
STAC says it is going to do and is HELD there; the next sentence releases it only when it IS the
confirmation — decided on its words, with no call of the model."""
from __future__ import annotations

import pytest

import commands
from commands import confirm
from phase5_qa import answers
from phase5_qa.orchestrator import SpatialQA

from test_commands_turn import UI, _ask

HELD = {"id": "volumes.delete", "args": {"volume_id": 2}, "detail": "zona de paso"}
OPEN = {**UI, "session": "pccr", "lang": "es"}


@pytest.mark.parametrize("sentence, said", [
    ("STAC, sí confirmo", "yes"), ("SI CONFIRMO", "yes"), ("Sí, confirmo.", "yes"), ("confirmo", "yes"),
    ("stack, yes, I confirm", "yes"), ("I confirm", "yes"), ("Oui, je confirme !", "yes"), ("je confirme", "yes"),
    ("no", "no"), ("STAC, cancelá", "no"), ("STAC, cancela", "no"), ("cancel", "no"), ("non", "no"),
    ("annule", "no"), ("no confirmo", "no"),
    # a bare yes answers too many things; a confirmation inside another sentence is not one
    ("sí", None), ("yes", None), ("oui", None), ("ok", None), ("dale", None),
    ("sí confirmo y borrá también el 3", None), ("no sé si confirmo", None), ("", None),
    ("confirmame cuántas puertas hay", None),
])
def test_what_confirms(sentence, said):
    assert confirm.verdict(sentence) == said


def test_only_a_declared_confirm_command_can_be_held():
    db = commands.database()
    row = confirm.held(db, HELD, "es")
    assert row["id"] == "volumes.delete" and row["call_args"] == {"volume_id": 2} and row["detail"] == "zona de paso"
    assert row["ask"] == "Voy a borrar el volumen {detail}."
    # not a command with confirm · one STAC does not run · not a call of it · not a command at all
    assert confirm.held(db, {"id": "app.fullscreen", "args": {}}, "es") is None
    assert confirm.held(db, {"id": "pipeline.run", "args": {}}, "es") is None or \
        db.get("pipeline.run")["status"] == "available"
    assert confirm.held(db, {"id": "volumes.delete", "args": {}}, "es") is None
    assert confirm.held(db, {"id": "volumes.delete", "args": {"volume_id": 2, "x": 1}}, "es") is None
    assert confirm.held(db, {"id": "nothing.here", "args": {}}, "es") is None
    assert confirm.held(db, "volumes.delete", "es") is None


def test_the_request_is_handed_over_with_what_stac_will_say(tmp_path, monkeypatch):
    res, _engine = _ask(monkeypatch, tmp_path, [("volumes__delete", {"volume_id": 2})], "borrá el volumen 2",
                        ui=OPEN, user_name="Ana")
    (action,) = res["ui_actions"]
    assert action["id"] == "volumes.delete" and action["confirm"] is True and "confirmed" not in action
    assert action["ask"] == "Voy a borrar el volumen {detail}." and action["done"] == "Volumen {detail} borrado."
    assert res["texts"]["confirm_how"] == "Para hacerlo, di: «STAC, sí confirmo»."
    assert "one_confirmation" in res["texts"]["reasons"]


def test_the_confirmation_releases_it_without_the_model(tmp_path, monkeypatch):
    res, engine = _ask(monkeypatch, tmp_path, [], "STAC, sí confirmo", ui=OPEN, user_name="Ana", pending=HELD,
                       conversation_lang="es")
    assert engine.chats == []                                   # decided on the words alone
    (action,) = res["ui_actions"]
    assert action["id"] == "volumes.delete" and action["args"] == {"volume_id": 2}
    assert action["confirmed"] is True and action["done"] == "Volumen {detail} borrado."
    assert res["answer"] == "" and res["acted_only"] is True and res["stopped"] == "confirmed"
    assert res["confirmation"] == {"id": "volumes.delete", "args": {"volume_id": 2}, "detail": "zona de paso",
                                   "verdict": "yes"}
    assert res["texts"]["done_prefix"] == "Listo, Ana."


def test_a_refusal_cancels_it_without_the_model(tmp_path, monkeypatch):
    res, engine = _ask(monkeypatch, tmp_path, [], "no", ui=OPEN, pending=HELD, conversation_lang="en")
    assert engine.chats == [] and res["ui_actions"] == [] and res["stopped"] == "cancelled"
    assert res["answer"] == "Cancelled: nothing was done." and res["confirmation"]["verdict"] == "no"


def test_any_other_sentence_drops_it_and_is_answered(tmp_path, monkeypatch):
    res, engine = _ask(monkeypatch, tmp_path, [("app__fullscreen", {"on": True})], "poné pantalla completa",
                       ui=OPEN, pending=HELD)
    assert len(engine.chats) == 1 and [a["id"] for a in res["ui_actions"]] == ["app.fullscreen"]
    assert all(not a.get("confirmed") for a in res["ui_actions"])
    assert res["answer"].startswith("No lo confirmaste: «Borrar un volumen» queda sin hacer.")
    assert res["confirmation"] == {"id": "volumes.delete", "verdict": "dropped"}


def test_a_confirmation_with_nothing_held_confirms_nothing(tmp_path, monkeypatch):
    # … nor does one for a command that cannot be held (the interface's word is checked)
    for pending in (None, {"id": "app.fullscreen", "args": {"on": True}}, {"id": "volumes.delete", "args": {}}):
        res, engine = _ask(monkeypatch, tmp_path, [("small_talk", {"reply": "¿Qué confirmo?"})], "sí confirmo",
                           ui=OPEN, pending=pending)
        assert len(engine.chats) == 1 and res["ui_actions"] == [] and "confirmation" not in res


def test_the_texts_exist_in_the_three_languages():
    for lang in answers.LANGS:
        assert answers.confirm_how(lang) and answers.cancelled(lang) and answers.not_confirmed("x", lang)
        assert {"one_confirmation", "needs_confirmation"} <= set(answers.ui_reasons(lang))
