"""WHICH COMMANDS A SENTENCE IS OFFERED. The model chooses the command; it cannot be shown every
command of STAC at each sentence (a tool costs ~100 tokens of prompt and the commands will be
"muchísimos", USER 2026-11-14) — so each sentence gets a SHORT list of candidates and the model
picks among them. A list that changes costs little: on the chat's engine a new set of tools added
0.03-0.2 s the first time (measured 2026-11-14).

Today the candidates of the interface and the system are found by WORDS: the sentence against how
each command is named and asked for, in the three languages at once (commands.db, FTS5, words of
four letters or more also matching the words that share their start), best first, plus the
commands every sentence is offered (conversation, help) and the ones the interface's state calls
for (opening a session when none is open). The scene's own tools are all offered while a scan is
open, as they were validated.

THE SCALING PATH, when the catalogue outgrows the words (declared, not built): a first call of
the model on the list of AREAS alone (a tiny prompt: 0.3 s measured) choosing the areas whose
commands are then offered. It goes inside :func:`candidates`; nothing else changes.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

from .db import CommandsDB

# when STAC has just asked something, the answer's commands are always offered
EXPECTING = {"user_name": ["talk.set_user_name"]}
# what the interface's state calls for whatever the sentence says
WITHOUT_SESSION = ["sessions.open", "sessions.list", "app.connect"]


def candidates(db: CommandsDB, text: str, *, limit: int, ui: Optional[Mapping[str, Any]] = None,
               expecting: Optional[str] = None) -> List[str]:
    """The ids of the interface's and the system's commands this sentence is offered, the ones
    always offered first, then the sentence's own by rank; at most ``limit`` of the latter."""
    known = {c["id"] for c in db.list(runs_on=("ui", "system"), status=("available", "manual"))}
    # STAC has just asked something: the answer's commands are offered first — beside the
  # sentence's own (the user may answer AND ask: "soy Ana, abrí pccr")
    ids: List[str] = [c for c in EXPECTING.get(expecting or "", []) if c in known]
    ids += [c for c in db.core() if c in known and c not in ids]
    if ui is not None and not ui.get("session"):
        ids += [c for c in WITHOUT_SESSION if c in known and c not in ids]
    # (a command the interface only does by hand is a candidate too: STAC then says so)
    for hit in db.search(text, None, limit=int(limit), runs_on=("ui", "system"), stems=True,
                         status=("available", "manual")):
        if hit["id"] not in ids:
            ids.append(hit["id"])
    return ids


def validate_args(row: Mapping[str, Any], args: Any) -> Optional[str]:
    """Why ``args`` is not a call of the command ``row`` (None when it is): unknown or missing
    names, a wrong type, a value outside its enum or its bounds. The engine's grammar already
    holds the model to the schema; this is the tool layer's own check."""
    if not isinstance(args, dict):
        return "the arguments are not an object"
    spec = row["args"]
    unknown = sorted(set(args) - set(spec))
    if unknown:
        return f"unknown argument(s) {unknown}"
    for name, s in spec.items():
        if name not in args or args[name] is None:
            if s.get("required"):
                return f"'{name}' is missing"
            continue
        v = args[name]
        types = s["type"] if isinstance(s["type"], list) else [s["type"]]
        ok = any((t == "string" and isinstance(v, str)) or (t == "boolean" and isinstance(v, bool))
                 or (t == "integer" and isinstance(v, int) and not isinstance(v, bool))
                 or (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool))
                 or (t == "array" and isinstance(v, list)) or (t == "object" and isinstance(v, dict))
                 for t in types)
        if not ok:
            return f"'{name}' must be {' or '.join(types)}"
        if "enum" in s and v not in s["enum"]:
            return f"'{name}' must be one of {s['enum']}"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "minimum" in s and v < s["minimum"]:
                return f"'{name}' is under {s['minimum']}"
            if "maximum" in s and v > s["maximum"]:
                return f"'{name}' is over {s['maximum']}"
    return None
