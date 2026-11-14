# STAC-Builder — Phase 5: the chat's provenance check.
#
# PROVENANCE (inviolable): the VLM proposes / describes / orchestrates — it NEVER measures.
# The system prompt says so; this module CHECKS it, deterministically, on every answer:
#
#   · every figure with a unit (m, cm, mm, m², m³, degrees) and every numeric vector
#     ((x, y, z), [w, h, d]) of the answer must be a number a tool returned IN THIS TURN, a
#     number the tool layer wrote into the prompt (the referenced objects, the inventory) or a
#     number the user wrote in the question — up to the rounding of the figure as written and
#     a change of unit (m ↔ cm ↔ mm);
#   · every tool the answer cites must have been called in this turn.
#
# Found 2026-11-13 on pccr 2026-08-31: asked for the distance between two objects the model
# answered "1.14 metres, measured with get_distance" having called no tool at all.
#
# Pure functions, no model and no store: the orchestrator asks, and re-asks the model with a
# tool call required when the check fails.

from __future__ import annotations

import re
from typing import Any, Iterable

_NUM = r"[-−]?\d+(?:[.,]\d+)?"
# the unit right after a number; longest first, never the first letters of a word ("3 más")
_UNIT = (r"(?:mm|cm|km|m[²³23]|metros?|metres?|meters?|mil[ií]metros?|cent[ií]metros?|"
         r"millimet(?:er|re)s?|centimet(?:er|re)s?|m|°|º|grados?|degrees?|deg)")
_FIGURE = re.compile(rf"(?<![\w.,#])({_NUM})\s*({_UNIT})(?![A-Za-zÀ-ÿ0-9])")
_VECTOR = re.compile(rf"[\(\[]\s*({_NUM}(?:\s*[,;]\s*{_NUM}){{1,3}})\s*[\)\]]")
_DECIMAL = re.compile(r"(?<![\w.])[-−]?\d+\.\d+")

# the factors a tool's number may be multiplied by to become the figure, by the figure's unit:
# the tools answer in metres (and in millimetres for flatness / plumb per metre)
_LENGTH_M = (1.0, 0.001, 0.01)
_FACTORS = {"m": _LENGTH_M, "cm": (100.0, 1.0, 0.1), "mm": (1000.0, 1.0, 10.0),
            "km": (0.001, 1.0)}


def _unit_family(unit: str) -> str:
    u = unit.lower()
    if u in ("mm",) or u.startswith("mil"):
        return "mm"
    if u in ("cm",) or u.startswith("cent"):
        return "cm"
    if u == "km":
        return "km"
    if u in ("m", "metro", "metros", "metre", "metres", "meter", "meters"):
        return "m"
    return "other"                      # m², m³, degrees: no change of unit


def _parse(raw: str) -> tuple[list[float], int]:
    """The values a written number may stand for, and its decimals: '1,30' is 1.30 (decimal
    comma) and '8,777' may also be 8777 (thousands)."""
    txt = raw.replace("−", "-")
    if "," in txt and "." not in txt:
        whole, frac = txt.split(",", 1)
        vals = [float(f"{whole}.{frac}")]
        if len(frac) == 3:
            vals.append(float(whole + frac))
        return vals, len(frac)
    dec = len(txt.split(".", 1)[1]) if "." in txt else 0
    return [float(txt.replace(",", ""))], dec


def numbers_of(obj: Any) -> list[float]:
    """Every number inside a tool result (dicts and lists walked; booleans are not numbers)."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(numbers_of(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(numbers_of(v))
    return out


def decimals_in(text: str) -> list[float]:
    """The decimal numbers written in a text of the tool layer (the inventory's y extents, the
    referenced objects' positions) or in the user's question."""
    return [float(m.group(0).replace("−", "-")) for m in _DECIMAL.finditer(text or "")]


def numbers_in(text: str) -> list[float]:
    """Every number written in a text — the user's own question: what the user wrote may be
    said back."""
    out: list[float] = []
    for m in re.finditer(rf"(?<![\w.]){_NUM}", text or ""):
        out.extend(_parse(m.group(0))[0])
    return out


def is_grounded(raw: str, unit: str, pool: Iterable[float]) -> bool:
    """A written figure is grounded when a pool number, in the figure's unit, rounds to it at
    the decimals it was written with."""
    vals, dec = _parse(raw)
    tol = 0.5 * 10.0 ** (-dec) + 1e-9
    factors = _FACTORS.get(_unit_family(unit), (1.0,))
    for t in pool:
        for k in factors:
            for v in vals:
                if abs(t * k - v) <= tol or abs(abs(t) * k - abs(v)) <= tol:
                    return True
    return False


def figures_in(answer: str) -> list[dict]:
    """The figures an answer states: numbers with a unit, and the numbers of every bracketed
    vector that holds a decimal (coordinates, sizes — a list of ids holds none)."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for m in _FIGURE.finditer(answer or ""):
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            out.append({"raw": m.group(1), "unit": m.group(2), "text": m.group(0).strip()})
    for m in _VECTOR.finditer(answer or ""):
        parts = [p.strip() for p in re.split(r"[,;]", m.group(1))]
        if not any("." in p for p in parts):
            continue
        for p in parts:
            key = (p, "m")
            if key not in seen:
                seen.add(key)
                out.append({"raw": p, "unit": "m", "text": p})
    return out


def cited_tools(answer: str, tool_names: Iterable[str]) -> list[str]:
    """The tools an answer names. Only names with an underscore are looked for: a one-word
    name ("travel") is also a word of the language."""
    return sorted(n for n in tool_names
                  if "_" in n and re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", answer or ""))


def check(answer: str, tool_trace: list[dict], *, tool_names: Iterable[str],
          context_numbers: Iterable[float] = ()) -> dict:
    """{"ok", "figures": the figures no tool returned, "tools": the tools cited and not
    called}. ``context_numbers``: the numbers of the tool layer's own text and of the user's
    question."""
    pool = list(context_numbers)
    called = set()
    for e in tool_trace or []:
        called.add(str(e.get("tool")))
        pool.extend(numbers_of(e.get("result")))
        pool.extend(numbers_of(e.get("arguments")))
    bad = [f["text"] for f in figures_in(answer) if not is_grounded(f["raw"], f["unit"], pool)]
    lied = [n for n in cited_tools(answer, tool_names) if n not in called]
    return {"ok": not bad and not lied, "figures": bad, "tools": lied}


def retry_message(result: dict) -> str:
    """What the model is told when its answer failed the check (it is then asked again, a tool
    call required)."""
    parts = ["[PROVENANCE CHECK — by the tool layer, not by the user]"]
    if result.get("figures"):
        parts.append("These figures of your answer were not returned by any tool in this turn: "
                     + "; ".join(result["figures"]) + ".")
    if result.get("tools"):
        parts.append("You cited these tools without calling them in this turn: "
                     + ", ".join(result["tools"]) + ".")
    parts.append("You never compute, convert from memory, recall or estimate a figure. Call "
                 "the tool(s) that measure what was asked NOW, then answer the user's question "
                 "again using only figures the tools returned, in the user's language.")
    return " ".join(parts)
