"""SHOW ME / SELECT <an object of this scan>: the same words, the same answer (USER 2026-11-24: "parece que no
responde siempre igual a los mismos comandos" — phase5_qa/selection.py). The sentences are his, as the
recogniser wrote them that day in grupo_electrogeno's chat."""
from __future__ import annotations

import pytest

from phase5_qa import selection

# grupo_electrogeno's kinds of object, with ids of the test's own
SCAN = [(1, "floor"), (131, "column"), (132, "column"), (195, "window"), (7, "concrete_slab"), (8, "concrete_slab"),
        (20, "door"), (21, "glass_door"), (30, "grating"), (40, "manhole_cover")]


@pytest.mark.parametrize("said, ids", [
    # his, that day — each of these once reached another command
    ("show me the floor.", [1]), ("Show me floor.", [1]), ("select floor.", [1]), ("Select floor.", [1]),
    ("Select and show me floor.", [1]), ("select and show me floor.", [1]),
    ("Show me column.", [131, 132]), ("Show me the column.", [131, 132]), ("select column.", [131, 132]),
    ("STAC, show me the window.", [195]), ("STAC, STAC, show me the window.", [195]),
    # the three languages, ids, plurals, a request
    ("muéstrame floor", [1]), ("mostrame el floor", [1]), ("seleccioná column", [131, 132]),
    ("montre-moi column 131", [131]), ("sélectionne la window", [195]),
    ("show me column#132", [132]), ("show me #131", [131]), ("show me 131", [131]), ("select column number 132", [132]),
    ("show me the concrete slabs", [7, 8]), ("show me the manhole covers please", [40]),
    ("can you show me the floor?", [1]), ("Hey STAC, select the glass doors", [21]),
    ("show me the door", [20]),                     # a WHOLE label: "door" is not "glass_door"
])
def test_a_selection_of_an_object_of_the_scan_is_decided_by_the_sentence_alone(said, ids):
    hit = selection.match(said, SCAN)
    assert hit is not None and hit["ids"] == ids, (said, hit)


@pytest.mark.parametrize("said", [
    # no object of this scan: the model's, as before
    "show me the grid.", "select grid.", "show me grid.", "show me the scene.", "show me camera poses, please.",
    "SELECT FLOR", "show me the fan.", "show me the distance between floor and column",
    "show me the floor and the column",
    # showing ONLY them, or seen again: other commands
    "show me only the doors", "mostrame solo las puertas", "montre-moi seulement les portes", "show the doors again",
    # an id that is not the label said, an id that is not there
    "select wall 131", "show me column 999",
    # not a selection at all
    "describe the scene.", "what is your name?", "Thank you.", "", "STAC",
    # Spanish / French nouns are not this scan's labels: the model translates them, as before
    "seleccioná la columna", "montre-moi la porte",
])
def test_what_names_no_object_or_asks_something_else_stays_with_the_model(said):
    assert selection.match(said, SCAN) is None, said


def test_the_answer_does_not_depend_on_anything_but_the_sentence_and_the_scan():
    first = selection.match("Show me column.", SCAN)
    for _ in range(3):
        assert selection.match("Show me column.", list(reversed(SCAN))) == first
    assert first == {"ids": [131, 132], "label": "column", "by": "label"}
