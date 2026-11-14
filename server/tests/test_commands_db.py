"""The database of commands (USER 2026-11-14: "debes armar una base de datos con absolutamente
todos los comandos a los que luego se iran incorporando los nuevos comandos"): the declarations
are validated, the database is their deterministic image, and the chat's tools are its rows."""
from __future__ import annotations

import sqlite3
import textwrap

import pytest

import commands
from commands import catalog as cat
from commands import db as cdb


def _area(tmp_path, body: str, name: str = "10_demo.yaml"):
    (tmp_path / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


_GOOD = """
    area: demo
    position: 10
    title: {es: Demo, en: Demo, fr: Démo}
    summary: {es: Un área de prueba., en: A test area., fr: Une zone de test.}
    commands:
      - id: demo.full_screen
        kind: action
        runs_on: ui
        model: Full screen on or off.
        args:
          "on": {type: boolean, required: true}
        confirm: false
        text:
          es: {title: Pantalla completa, say: [ampliá la pantalla, pantalla completa], done: Pantalla completa.}
          en: {title: Full screen, say: [full screen, enlarge the screen], done: Full screen.}
          fr: {title: Plein écran, say: [plein écran, agrandis l'écran], done: Plein écran.}
      - id: demo.count
        tool: count_things
        kind: query
        runs_on: scene
        model: Count things.
        text:
          es: {title: Contar, say: [cuántas cosas hay]}
          en: {title: Count, say: [how many things]}
          fr: {title: Compter, say: [combien de choses]}
"""


def test_the_real_catalogue_is_complete_in_three_languages():
    c = cat.load_catalog()
    assert c.sha256 == cat.load_catalog().sha256                       # no clock, no order at random
    assert len({x["id"] for x in c.commands}) == len(c.commands)
    assert len({x["tool"] for x in c.commands}) == len(c.commands)
    for x in c.commands:
        assert x["kind"] in cat.KINDS and x["runs_on"] in cat.RUNS_ON
        for lang in cat.LANGS:
            t = x["text"][lang]
            assert t["title"] and t["say"], (x["id"], lang)
            if x["runs_on"] == "ui":
                assert t["done"], (x["id"], lang)                       # what STAC says when done
    for a in c.areas:
        assert all(a["text"][lang]["title"] and a["text"][lang]["summary"] for lang in cat.LANGS)
        assert any(x["area"] == a["id"] for x in c.commands), a["id"]


def test_every_command_has_its_executor():
    """A command is never half built: every row that runs on the scene has its handler in the
    tool layer (with the row's argument names), every system row its handler, and the interface's
    are typed against the generated file — which must be the catalogue's."""
    import inspect
    from commands import export_ui
    from phase5_qa.tools import SpatialTools
    db = commands.database()
    scene = db.list(runs_on="scene")
    assert len(scene) >= 44
    for r in scene:
        fn = getattr(SpatialTools, r["handler"], None)
        assert callable(fn), r["id"]
        assert set(r["args"]) <= set(inspect.signature(fn).parameters), r["id"]
    # … and the model is offered exactly the rows (the tool layer reads its tools from the database)
    assert [t["function"]["name"] for t in SpatialTools.openai_schemas()] == [r["tool"] for r in scene]
    system = commands.SystemTools(db, "es").impls()
    assert set(system) == {r["tool"] for r in db.list(runs_on="system")} == {"small_talk", "help__commands"}
    assert db.core() == ["talk.small_talk", "help.commands"]
    assert export_ui.check(), "ui/src/commands/catalog.generated.ts is stale: python -m commands.export_ui"
    ui = db.list(runs_on="ui")
    text = export_ui.OUT.read_text()
    assert all(f"'{r['id']}'" in text for r in ui) and len(ui) >= 40
    for r in ui:
        assert r["tool"] == r["id"].replace(".", "__") and r["done"], r["id"]


def test_the_systems_own_commands():
    db = commands.database()
    es, fr = commands.SystemTools(db, "es"), commands.SystemTools(db, "fr")
    assert es.small_talk("Hola")["reply"] == "Hola"
    areas = es.help_commands()
    assert areas["total"] == db.count() and areas["areas"][0]["title"] == "Conversación"
    assert fr.help_commands()["areas"][0]["title"] == "Conversation"
    one = es.help_commands("paneles")
    assert one["area"] == "layout" and "Mostrar el inspector" in one["commands"]
    assert es.help_commands("Panels")["area"] == "layout" and es.help_commands("layout")["area"] == "layout"
    assert es.help_commands("zzz")["insufficient_data"] is True


def test_the_candidates_a_sentence_is_offered():
    from commands import router
    db = commands.database()
    for text, want in (("conectá con el servidor", "app.connect"), ("abrí pccr", "sessions.open"),
                       ("ampliá la pantalla", "app.fullscreen"), ("ocultá la grilla", "display.grid"),
                       ("mostrame la malla", "meshes.show"), ("open the console", "layout.show_dock"),
                       ("affiche les axes", "display.axes"), ("me llamo Ana", "talk.set_user_name"),
                       ("quiero calibrar", "calibration.measure_distance")):
        got = router.candidates(db, text, limit=10)
        assert got[:2] == ["talk.small_talk", "help.commands"], text      # always offered
        assert got[2] == want, (text, got)
    # with no session open, opening one is offered whatever is said
    got = router.candidates(db, "pccr", limit=10, ui={"session": None})
    assert got[:5] == ["talk.small_talk", "help.commands", "sessions.open", "sessions.list", "app.connect"]
    assert "sessions.open" not in router.candidates(db, "zzz", limit=10, ui={"session": "pccr"})
    assert len(router.candidates(db, "abrí la sesión y mostrá la consola y los ejes", limit=3)) <= 2 + 3
    # STAC asked the name: only its answer's commands
    got = router.candidates(db, "soy Ana, abrí pccr", limit=10, expecting="user_name")
    assert got[:3] == ["talk.set_user_name", "talk.small_talk", "help.commands"] and "sessions.open" in got
    row = db.get("display.point_opacity")
    assert router.validate_args(row, {"opacity": 0.5}) is None
    assert "over 1" in router.validate_args(row, {"opacity": 2})
    assert "missing" in router.validate_args(row, {}) and "unknown" in router.validate_args(row, {"x": 1})
    assert "must be one of" in router.validate_args(db.get("layout.show_dock"), {"tab": "nope"})
    assert "must be boolean" in router.validate_args(db.get("display.grid"), {"on": "yes"})


def test_the_database_is_the_declarations_image_and_is_rebuilt_when_they_change(tmp_path):
    d = _area(tmp_path, _GOOD)
    path = tmp_path / "commands.db"
    db = cdb.database(d, path)
    assert db.count() == 2 and path.exists()
    on_disk = sqlite3.connect(path)
    assert on_disk.execute("SELECT value FROM meta WHERE key = 'catalog_sha256'").fetchone()[0] == db.sha256
    assert [r[0] for r in on_disk.execute("SELECT id FROM commands ORDER BY position")] == \
        ["demo.full_screen", "demo.count"]
    on_disk.close()
    # the same declarations: the same rows, byte for byte in a dump
    a = "\n".join(cdb.build(cat.load_catalog(d), None).iterdump())
    b = "\n".join(cdb.build(cat.load_catalog(d), None).iterdump())
    assert a == b
    row = db.get("demo.full_screen", "fr")
    assert row["title"] == "Plein écran" and row["say"] == ["plein écran", "agrandis l'écran"]
    assert row["args"] == {"on": {"type": "boolean", "required": True}} and row["tool"] == "demo__full_screen"
    assert db.by_tool("count_things", "es")["title"] == "Contar"
    assert db.tool_schemas(["demo.full_screen"])[0] == {"type": "function", "function": {
        "name": "demo__full_screen", "description": "Full screen on or off.",
        "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}}
    assert [a["id"] for a in db.areas("es")] == ["demo"] and db.areas("es")[0]["commands"] == 2

    # a NEW command is one more declaration: the database follows at the next call
    _area(tmp_path, _GOOD + """
      - id: demo.new_one
        kind: talk
        runs_on: scene
        model: Something new.
        text:
          es: {title: Nuevo, say: [algo nuevo]}
          en: {title: New, say: [something new]}
          fr: {title: Nouveau, say: [quelque chose de nouveau]}
    """)
    again = cdb.database(d, path)
    assert again.count() == 3 and again.sha256 != db.sha256
    assert cdb._stamp_of(path) == again.sha256


def test_the_search_finds_a_command_by_how_people_ask_for_it(tmp_path):
    db = cdb.CommandsDB(cdb.build(cat.load_catalog(_area(tmp_path, _GOOD)), None), "x")
    assert db.search("amplia la pantalla por favor", "es")[0]["id"] == "demo.full_screen"   # no accent
    assert db.search("AMPLIÁ LA PANTALLA", "es")[0]["id"] == "demo.full_screen"
    assert db.search("agrandis l'écran", "fr")[0]["id"] == "demo.full_screen"
    assert db.search("how many things are there", "en")[0]["id"] == "demo.count"
    assert db.search("how many things", "es") == []                    # another language's words
    assert db.search("how many things", None)[0]["id"] == "demo.count"  # every language
    assert db.search("pantalla", "es", runs_on="scene") == []
    assert db.search("pantallas", "es") == []                           # another word, exactly
    assert db.search("pantallas", "es", stems=True)[0]["id"] == "demo.full_screen"    # … by its start
    assert db.search("", "es") == [] and db.search("¿?", "es") == []

    real = commands.database()
    for text, lang, want in (("mostrame solo las puertas", "es", "viewer.show_only"),
                             ("a qué distancia está el cartel del extintor", "es", "measure.get_distance"),
                             ("is the wall plumb", "en", "measure.get_plumb"),
                             ("quelle est la hauteur libre sous le plafond", "fr", "measure.get_height_profile"),
                             ("cuántas puertas hay", "es", "scene.count_objects")):
        assert real.search(text, lang, limit=3)[0]["id"] == want, text


@pytest.mark.parametrize("change, said", [
    (("fr: {title: Plein écran, say: [plein écran, agrandis l'écran], done: Plein écran.}", ""), "'fr' text is missing"),
    (("kind: action", "kind: order"), "'kind' is missing or not one of"),
    (("id: demo.count", "id: demo.full_screen"), "id declared twice"),
    (("id: demo.count", "id: other.count"), "'id' must be 'demo.<name>'"),
    (("tool: count_things", "tool: demo__full_screen"), "is already demo.full_screen's"),
    (('"on": {type: boolean, required: true}', '"on": {type: bool}'), "argument 'on': type must be"),
    (('"on": {type: boolean, required: true}', 'on: {type: boolean}'), "quote it"),
    (("done: Pantalla completa.}", "}"), "'es.done' is missing"),
    (("model: Count things.", "model: ''"), "'model'"),
    (("say: [cuántas cosas hay]", "say: []"), "'es.say' must be a list of phrases"),
])
def test_a_half_declared_command_fails_the_load_naming_it(tmp_path, change, said):
    old, new = change
    assert old in _GOOD
    with pytest.raises(cat.CatalogError) as e:
        cat.load_catalog(_area(tmp_path, _GOOD.replace(old, new, 1)))
    assert said in str(e.value) and "10_demo.yaml" in str(e.value)
