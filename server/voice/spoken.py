"""A written answer as it is SAID. The chat writes for the eye ("sign#797", "0.0018 m (1.8 mm)",
"1.421° (24.8 mm/m)", "2.5 × 1.2 m"); a voice reading that aloud spells symbols. This turns the
text into what a person would say in each language — deterministic rules, no model.

Only the FORM changes: every figure said is the figure written (the tool's), never rounded here.
"""
from __future__ import annotations

import re
from typing import Dict

LANGS = ("es", "en", "fr")

# unit symbol → (singular, plural) per language; longest symbols first when matching
_UNITS: Dict[str, Dict[str, tuple]] = {
    "mm/m": {"es": ("milímetro por metro", "milímetros por metro"), "en": ("millimetre per metre", "millimetres per metre"),
             "fr": ("millimètre par mètre", "millimètres par mètre")},
    "m³": {"es": ("metro cúbico", "metros cúbicos"), "en": ("cubic metre", "cubic metres"), "fr": ("mètre cube", "mètres cubes")},
    "m3": {"es": ("metro cúbico", "metros cúbicos"), "en": ("cubic metre", "cubic metres"), "fr": ("mètre cube", "mètres cubes")},
    "m²": {"es": ("metro cuadrado", "metros cuadrados"), "en": ("square metre", "square metres"), "fr": ("mètre carré", "mètres carrés")},
    "m2": {"es": ("metro cuadrado", "metros cuadrados"), "en": ("square metre", "square metres"), "fr": ("mètre carré", "mètres carrés")},
    "mm": {"es": ("milímetro", "milímetros"), "en": ("millimetre", "millimetres"), "fr": ("millimètre", "millimètres")},
    "cm": {"es": ("centímetro", "centímetros"), "en": ("centimetre", "centimetres"), "fr": ("centimètre", "centimètres")},
    "km": {"es": ("kilómetro", "kilómetros"), "en": ("kilometre", "kilometres"), "fr": ("kilomètre", "kilomètres")},
    "m": {"es": ("metro", "metros"), "en": ("metre", "metres"), "fr": ("mètre", "mètres")},
    "°": {"es": ("grado", "grados"), "en": ("degree", "degrees"), "fr": ("degré", "degrés")},
    "%": {"es": ("por ciento", "por ciento"), "en": ("percent", "percent"), "fr": ("pour cent", "pour cent")},
    "s": {"es": ("segundo", "segundos"), "en": ("second", "seconds"), "fr": ("seconde", "secondes")},
}
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_NUM = r"-?\d+(?:[.,]\d+)?"
_NUM_UNIT = re.compile(rf"(?<![\w.,])({_NUM})\s?({_UNIT_ALT})(?![\w³²/])")
_BY = {"es": " por ", "en": " by ", "fr": " par "}
_MINUS = {"es": "menos ", "en": "minus ", "fr": "moins "}
_POINT = {"es": ",", "en": ".", "fr": ","}          # the decimal mark each voice reads as a number


def _number(text: str, lang: str) -> str:
    neg = text.startswith("-")
    body = text.lstrip("-").replace(",", ".")
    body = body.replace(".", _POINT[lang])
    return (_MINUS[lang] if neg else "") + body


def _is_one(text: str) -> bool:
    try:
        return abs(float(text.replace(",", "."))) == 1.0
    except ValueError:
        return False


def say(text: str, lang: str) -> str:
    """``text`` as it is said in ``lang`` (es | en | fr; anything else is read as English)."""
    lang = lang if lang in LANGS else "en"
    t = str(text or "")
    # markdown and typography the voice would spell
    t = re.sub(r"[*_`]{1,3}([^*_`]+)[*_`]{1,3}", r"\1", t)
    t = t.replace("«", "").replace("»", "").replace("“", "").replace("”", "").replace('"', "")
    # an object reference: fire_extinguisher#425 → fire extinguisher 425 ; #425 → 425
    t = re.sub(r"\b([A-Za-z][A-Za-z0-9_]*)#(\d+)\b", lambda m: f"{m.group(1).replace('_', ' ')} {m.group(2)}", t)
    t = re.sub(r"#(\d+)\b", r"\1", t)
    t = re.sub(r"\b([a-z]+(?:_[a-z0-9]+)+)\b", lambda m: m.group(1).replace("_", " "), t)   # labels
    # a number with its unit
    t = _NUM_UNIT.sub(lambda m: f"{_number(m.group(1), lang)} "
                                f"{_UNITS[m.group(2)][lang][0 if _is_one(m.group(1)) else 1]}", t)
    # dimensions: 2.5 × 1.2 × 0.6 → "2,5 por 1,2 por 0,6"
    t = re.sub(r"\s?×\s?", _BY[lang], t)
    t = re.sub(r"(?<=\d)\s?x\s?(?=-?\d)", _BY[lang], t)
    # the numbers left (counts, ids, bare decimals): the language's decimal mark, a spoken minus
    t = re.sub(rf"(?<![\w.,])({_NUM})(?![\w.,]*\d)", lambda m: _number(m.group(1), lang), t)
    # brackets and separators become pauses
    t = t.replace("(", ", ").replace(")", "").replace("[", "").replace("]", "")
    t = t.replace(" — ", ", ").replace(" – ", ", ").replace("·", ",").replace(";", ",")
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r",\s*\.", ".", t)
    return re.sub(r"\s{2,}", " ", t).strip()
