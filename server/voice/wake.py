"""IS STAC BEING SPOKEN TO? — the system's name at the start (or the end) of what was heard.

USER 2026-11-14: "el sistema se llama stac, no hace falta llamarlo por el nombre completo stac
build, aunque tambien se acepta pero no es obligatorio". In hands-free listening everything said
in the room is written by the recogniser; only what names STAC is taken as said TO it — the rest
is dropped by the interface, unsent to the model.

Deterministic, on the recogniser's text: the name as it is written ("STAC") and as a recogniser
may spell it when it does not know the word ("Stack", "Stak", the Spanish "Estac"), alone or
followed by "Build", after an optional greeting ("hola STAC", "hey STAC", "dis STAC").
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

# stac · stack · stak · staq · estac · estack … and, as a recogniser wrote the English voice's
# "STAC Build" on 2026-11-14, sdac; spelled out too ("S.T.A.C.", "s t a c")
_NAME = r"(?:e?s[td]a+(?:c|ck|k|q)|s[\s.\-]t[\s.\-]a[\s.\-]c\.?)"
_BUILD = r"(?:[\s\-]*(?:build|bild|builder))?"
_LEAD = r"(?:hey|ok|okay|oye|oiga|hola|eh|che|hello|hi|dis|dites|salut|bonjour|buenas|buen dia|buenos dias)"
_SEP = r"[\s,.:;!¡?¿\-–—]*"
_START = re.compile(rf"^{_SEP}(?:(?P<lead>{_LEAD}){_SEP})?{_NAME}{_BUILD}(?![a-z0-9]){_SEP}", re.IGNORECASE)
_END = re.compile(rf"{_SEP}(?<![a-z0-9]){_NAME}{_BUILD}{_SEP}$", re.IGNORECASE)

# what STAC says when only its name was said: it is listening
ACK: Dict[str, str] = {"es": "Dime.", "en": "Yes?", "fr": "Oui ?"}


def _plain(text: str) -> str:
    """The text with its accents removed, SAME LENGTH (so a match's span is the original's)."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFKD", ch)
        stripped = "".join(c for c in base if not unicodedata.combining(c))
        out.append(stripped if len(stripped) == 1 else ch)
    return "".join(out)


def split(text: str) -> Tuple[bool, str]:
    """(addressed, command): whether what was heard names STAC, and what was said to it — the
    text without the name (the greeting kept when nothing else was said: "hola STAC" → "hola";
    "" when only the name was said). Not addressed: (False, the text as it is)."""
    original = " ".join(str(text or "").split())
    plain = _plain(original)
    m = _START.match(plain)
    if m:
        rest = original[m.end():].strip()
        if not rest and m.group("lead"):
            rest = original[m.start("lead"):m.end("lead")]
        return True, rest
    m = _END.search(plain)
    if m and m.start() > 0:
        return True, original[:m.start()].strip(" ,;:")
    return False, original
