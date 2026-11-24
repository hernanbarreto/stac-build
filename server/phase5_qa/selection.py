"""SHOW ME / SELECT <an object of this scan>: THE SAME WORDS, THE SAME ANSWER (USER 2026-11-24: "lo más importante
es que la lista de comandos sea clara, le digo show me X elemento y lo selecciona de la lista de segmentos pero no
me los muestra como si yo lo tocara en la pantalla, lo filtra, parece que no responde siempre igual a los mismos
comandos"; to the rule proposed to him: "así como lo propusiste está OK").

Traced on his turns of that day (grupo_electrogeno's spatial_qa_logs): the same sentence went to different
commands — "Show me column." once selected the column (viewer.focus_object: what a touch does) and once showed
every column (viewer.show_objects); "Select floor." once selected the floor and three times reached
instances.set_floor_level (held by its spoken confirmation). The model chooses among the candidates that share
the sentence's words, with the conversation's last turns beside it: another conversation, another command.

THE RULE, asked BEFORE the model: a selection verb — show me / show / select · muéstrame / mostrame / muestra /
seleccioná / selecciona · montre-moi / montre / sélectionne — followed by AN OBJECT OF THIS SCAN: an id
(column#131, #131, "column 131", 131) or a WHOLE label (floor, concrete slab; a plural names its singular). The
object is selected as a touch on the model selects it: its box and dimensions, the camera travelling to it, no
list filtered, no panel opened. Several objects of that label: the one nearest the camera is selected and all of
them are framed (the interface chooses: it holds the camera). "Only / just / solo / sólo / solamente /
únicamente / seulement / uniquement" is showing ONLY them — the model's, as before — and so is every sentence
whose words name no object of this scan (the grid, the scene, the camera poses …).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

from .tools import norm_label


def _fold(text: str) -> str:
    """Lower case, accents off (muéstrame = muestrame, sélectionne = selectionne), apostrophes, hyphens and
    punctuation as spaces (l'escalier = l escalier, montre-moi = montre moi, as norm_label writes a label),
    '#' kept (column#131), single spaces."""
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^\w#\s]", " ", t)
    return " ".join(t.replace("_", " ").split())


# the selection verbs as _fold writes them, the longest first
_VERBS = sorted({
    "show me", "show", "select",                                                        # en
    "muestrame", "mostrame", "muestra", "mostra", "ensename", "selecciona", "seleccioname",
    "seleccionar",                                                                      # es (mostrá, seleccioná)
    "montre moi", "montrez moi", "montre", "montrez", "selectionne", "selectionnez", "selectionner",   # fr
}, key=lambda v: (-len(v), v))
# what may come before the verb: STAC's name as a recogniser writes it, a greeting, a request
_LEAD = re.compile(r"^(?:(?:hey|ok|okay|oye|dis)\s+)?(?:(?:e?s[td]a+(?:c|ck|k|q)|s t a c)(?:\s+build)?\s+)*"
                   r"(?:(?:please|por favor|can you|could you|podes|podrias|puedes|peux tu|pouvez vous|"
                   r"pourrais tu|tu peux)\s+)?")
_AND = ("and", "y", "e", "et")
# showing ONLY them is another command (the model's)
_ONLY = {"only", "just", "solo", "solamente", "unicamente", "seulement", "uniquement"}
# words before the object that name nothing
_DET = {"me", "the", "a", "an", "this", "that", "all", "el", "la", "los", "las", "lo", "un", "una", "unos", "unas",
        "este", "esta", "ese", "esa", "todos", "todas", "le", "les", "l", "une", "des", "du", "de", "ce", "cette",
        "tous", "toutes"}
_TAIL = re.compile(r"\s+(?:please|por favor|s il te plait|s il vous plait)$")
# "show the pipes again": seen again, the visibility command (the model's)
_AGAIN = re.compile(r"\b(?:again|otra vez|de nuevo|nuevamente|de nouveau|a nouveau)$")
_ID = re.compile(r"(?:(?P<label>[a-z][a-z0-9 ]*?)\s*)?(?:(?:number|numero|num|nro)\s*)?#?\s*(?P<id>\d{1,6})")


def _object_phrase(question: str) -> Optional[str]:
    """What follows the selection verb(s), or None when the sentence is not a selection."""
    t = _LEAD.sub("", _fold(question), count=1)
    said = False
    while True:
        verb = next((v for v in _VERBS if t == v or t.startswith(v + " ")), None)
        if verb is None:
            break
        said, t = True, t[len(verb):].strip()
        nxt = t.split(" ", 1)
        if nxt[0] in _AND and len(nxt) > 1:                  # "select and show me the floor"
            t = nxt[1]
    if not said or not t:
        return None
    t = _TAIL.sub("", t).strip()
    words = t.split()
    if any(w in _ONLY for w in words) or _AGAIN.search(t):
        return None
    while words and words[0] in _DET:
        words = words[1:]
    return " ".join(words) or None


def _singular(word_phrase: str) -> List[str]:
    out = []
    for suffix in ("es", "s"):
        if word_phrase.endswith(suffix) and len(word_phrase) > len(suffix) + 1:
            out.append(word_phrase[: -len(suffix)])
    return out


def match(question: str, instances: Iterable[Tuple[int, str]]) -> Optional[Dict[str, object]]:
    """``instances``: (id, label) of this scan's store. Returns ``{"ids", "label", "by"}`` — the objects the
    sentence selects (ids ascending), their label, and whether an id or a label named them — or None."""
    phrase = _object_phrase(question)
    if not phrase:
        return None
    by_label: Dict[str, List[int]] = {}
    label_of: Dict[int, str] = {}
    for iid, label in instances:
        key = _fold(norm_label(label))
        by_label.setdefault(key, []).append(int(iid))
        label_of[int(iid)] = str(label)
    m = _ID.fullmatch(phrase)
    if m:
        iid = int(m.group("id"))
        if iid not in label_of:
            return None
        said = (m.group("label") or "").strip()
        if said and _fold(norm_label(label_of[iid])) not in [said] + _singular(said):
            return None                                       # "wall 7" when 7 is a door: not this rule's
        return {"ids": [iid], "label": label_of[iid], "by": "id"}
    for key in [phrase] + _singular(phrase):
        if key in by_label:
            ids = sorted(by_label[key])
            return {"ids": ids, "label": label_of[ids[0]], "by": "label"}
    return None
