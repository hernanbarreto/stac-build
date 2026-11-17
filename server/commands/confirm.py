"""THE SPOKEN CONFIRMATION of a command that launches GPU work or deletes data.

USER 2026-11-17: "confirmacion hablada con lo que stac nos dice que va a hacer, STAC, SI CONFIRMO".
A command declared ``confirm: true`` is never run on the sentence that asks for it: STAC says
WHAT IT IS GOING TO DO (the command's ``ask`` text, with what the interface resolved — which
scan, which object) and holds it; it runs only when the user's NEXT sentence is the
confirmation. Anything else drops it.

Whether a sentence confirms is decided HERE, on its words — never by the model: the whole
sentence must be the confirmation ("sí, confirmo" / "yes, I confirm" / "oui, je confirme" and
their short forms). A bare "sí" / "yes" / "oui" does not confirm: it answers too many things.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .db import CommandsDB, fold

# the whole sentence, folded (lower case, no accents, no punctuation)
_YES = frozenset({
    # es
    "si confirmo", "confirmo", "si lo confirmo", "lo confirmo", "confirmado", "si confirmado",
    # en
    "yes i confirm", "i confirm", "yes confirm", "confirm", "confirmed", "yes confirmed", "i do confirm",
    # fr
    "oui je confirme", "je confirme", "oui confirme", "confirme", "oui c est confirme", "c est confirme",
})
_NO = frozenset({
    # es
    "no", "no confirmo", "no lo confirmo", "cancela", "cancelar", "cancelalo", "cancelado", "no lo hagas",
    "dejalo", "mejor no",
    # en
    "cancel", "cancel it", "cancel that", "i do not confirm", "i don t confirm", "don t", "do not",
    "don t do it", "do not do it", "never mind",
    # fr
    "non", "annule", "annuler", "annulez", "je ne confirme pas", "ne le fais pas", "laisse tomber",
})


def verdict(sentence: str) -> Optional[str]:
    """'yes' when the sentence IS the confirmation, 'no' when it IS a refusal, None for anything
    else (which confirms nothing). STAC's name in it is not part of what was said."""
    from voice import wake
    _addressed, said = wake.split(sentence)
    words = fold(said)
    if words in _YES:
        return "yes"
    if words in _NO:
        return "no"
    return None


def held(db: CommandsDB, pending: Any, lang: str) -> Optional[Mapping[str, Any]]:
    """The command the interface says STAC is holding for confirmation, checked against the
    database — None unless it is a command of the interface STAC runs, declared with
    ``confirm: true``, with arguments that are a call of it. Returns the row (in ``lang``) with
    the call's ``args`` and the ``detail`` the interface resolved."""
    from .router import validate_args
    if not isinstance(pending, dict) or not isinstance(pending.get("id"), str):
        return None
    row = next((r for r in db.list(lang, runs_on="ui", status="available") if r["id"] == pending["id"]), None)
    args = pending.get("args") if isinstance(pending.get("args"), dict) else {}
    if row is None or not row["confirm"] or validate_args(row, args):
        return None
    return {**row, "call_args": args, "detail": " ".join(str(pending.get("detail") or "").split())[:200]}
