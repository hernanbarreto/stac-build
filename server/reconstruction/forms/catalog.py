"""The catalogue of forms the VLM can answer, and the typed configuration of the stage.

A closed list, declared in ``config.yaml`` ``forms.catalog``: the query's JSON schema takes its
enums from here, the fitter maps every name to a template, and an answer outside the list cannot
be produced (vLLM's structured output constrains it) — a form the catalogue does not know is
declared, never guessed. Every key of ``forms:`` is required (a missing one fails naming itself).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── the forms ──────────────────────────────────────────────────────────────────
# name → how the template is built: `section` (a 2-D section extruded along an axis — a panel is
# a section with ~zero thickness, a box a rectangle extruded, a cylinder a disc extruded, a tube a
# ring, a profile an I/L/C/T/rect/rect_hollow/rail section), `sphere`, `cone` (a frustum),
# `revolution` (a radius profile about an axis), `composite` (parts of the above), `freeform`
# (no template: meshed as measured).
FORMS: Dict[str, Dict[str, Any]] = {
    "panel": {"family": "section", "what": "a flat plate: wall, floor, ceiling, door leaf, sign, "
                                           "table top, window pane — a surface with an outline"},
    "box": {"family": "section", "what": "a rectangular solid: cabinet, crate, block, step, desk body"},
    "cylinder": {"family": "section", "what": "a round bar, pipe, column, duct, tank, post "
                                              "(solid, or a hollow tube)"},
    "sphere": {"family": "sphere", "what": "a ball or dome"},
    "cone": {"family": "cone", "what": "a cone or truncated cone: funnel, bollard top, hopper"},
    "profile": {"family": "section", "what": "a straight structural section: I/H beam, L angle, "
                                             "C channel, T, rectangular tube, rail"},
    "revolution": {"family": "revolution", "what": "round about one axis with a varying radius: "
                                                   "lamp, bottle, barrel, bollard, vase, wheel"},
    "composite": {"family": "composite", "what": "several parts of the above: table, chair, ladder, "
                                                 "railing, shelf, desk with legs, stair"},
    "freeform": {"family": "freeform", "what": "organic or irregular: plant, person, rock, gravel "
                                               "pile, cable, debris, cloth"},
}
ORIENTATIONS = ("vertical", "horizontal", "inclined", "unknown")
OUTLINES = ("rectangle", "circle", "arch", "polygon", "irregular", "unknown")
PROFILES = ("I", "L", "C", "T", "rect", "rect_hollow", "round_tube", "rail", "unknown")
CONFIDENCES = ("high", "medium", "low")
PART_FORMS = ("panel", "box", "cylinder", "sphere", "cone", "profile", "revolution", "freeform")


def family_of(form: str) -> str:
    try:
        return str(FORMS[str(form)]["family"])
    except KeyError:
        raise ValueError(f"'{form}' is not a form of the catalogue {sorted(FORMS)}") from None


# ── the configuration ──────────────────────────────────────────────────────────

class FormsConfigError(KeyError):
    """A missing / invalid ``forms`` key — always named."""


@dataclass(frozen=True)
class FormsConfig:
    enabled: bool
    catalog: List[str]            # the forms the VLM may answer (a subset of FORMS)
    views: int                    # BOUND (cost): mask keyframes shown per object in the query
    max_rounds: int               # BOUND: VLM rounds per object (the query + the reviews)
    max_tokens: int               # BOUND: tokens of the structured answer
    device: str                   # cuda | cpu — where the templates, the fit and the silhouettes live
    max_points: int               # BOUND (cost): object points in the fit (deterministic subsample)
    surface_samples: int          # BOUND (cost): surface samples of the form per silhouette render
    boundary_px_per_view: int     # BOUND (cost): mask boundary pixels per view in the recall term
    max_views: int                # BOUND (cost): views per object in the corrector, largest masks first
    block_iterations: int         # the optimiser runs in blocks; the consensus is judged after each
    max_blocks: int               # BOUND: blocks per fit
    lr: float                     # the optimiser's step
    review_views: int             # overlays shown to the VLM in a review round (the worst views)
    ransac_iters: int             # BOUND (cost): hypotheses per part extraction of a composite
    seed: int                     # every subsample / bootstrap of the stage
    raw: Dict[str, Any] = field(default_factory=dict, compare=False)


def _req(sec: Dict[str, Any], key: str):
    if key not in sec:
        raise FormsConfigError(f"forms.{key} is missing")
    return sec[key]


def load_forms_config(config: Dict[str, Any]) -> FormsConfig:
    sec = config.get("forms")
    if not isinstance(sec, dict):
        raise FormsConfigError("forms section is missing")
    catalog = [str(x) for x in (_req(sec, "catalog") or [])]
    unknown = [c for c in catalog if c not in FORMS]
    if unknown or not catalog:
        raise FormsConfigError(f"forms.catalog must list forms of {sorted(FORMS)}; got {catalog}")
    out = FormsConfig(
        enabled=bool(_req(sec, "enabled")), catalog=catalog,
        views=int(_req(sec, "views")), max_rounds=int(_req(sec, "max_rounds")),
        max_tokens=int(_req(sec, "max_tokens")), device=str(_req(sec, "device")),
        max_points=int(_req(sec, "max_points")), surface_samples=int(_req(sec, "surface_samples")),
        boundary_px_per_view=int(_req(sec, "boundary_px_per_view")),
        max_views=int(_req(sec, "max_views")), block_iterations=int(_req(sec, "block_iterations")),
        max_blocks=int(_req(sec, "max_blocks")), lr=float(_req(sec, "lr")),
        review_views=int(_req(sec, "review_views")), ransac_iters=int(_req(sec, "ransac_iters")),
        seed=int(_req(sec, "seed")), raw=dict(sec))
    for k in ("views", "max_rounds", "max_tokens", "max_points", "surface_samples",
              "boundary_px_per_view", "max_views", "block_iterations", "max_blocks", "review_views",
              "ransac_iters"):
        if int(getattr(out, k)) < 1:
            raise FormsConfigError(f"forms.{k} must be >= 1, got {getattr(out, k)}")
    if out.lr <= 0:
        raise FormsConfigError(f"forms.lr must be > 0, got {out.lr}")
    if out.device not in ("cuda", "cpu"):
        raise FormsConfigError(f"forms.device must be cuda or cpu, got {out.device!r}")
    return out


# ── the VLM's answer ───────────────────────────────────────────────────────────

@dataclass
class Part:
    name: str
    form: str
    count: int
    orientation: str
    relation: str

    def to_dict(self) -> dict:
        return {"name": self.name, "form": self.form, "count": int(self.count),
                "orientation": self.orientation, "relation": self.relation}


@dataclass
class FormAnswer:
    """One VLM answer, validated against the catalogue (``vlm_proposed``)."""
    form: str
    orientation: str
    axis_aligned: bool
    hollow: Optional[bool]
    outline: str
    profile: str
    parts: List[Part]
    proportions: str
    alternatives: List[str]
    confidence: str
    what_would_help: str
    notes: str

    @property
    def family(self) -> str:
        return family_of(self.form)

    def to_dict(self) -> dict:
        return {"form": self.form, "family": self.family, "orientation": self.orientation,
                "axis_aligned": bool(self.axis_aligned), "hollow": self.hollow,
                "outline": self.outline, "profile": self.profile,
                "parts": [p.to_dict() for p in self.parts], "proportions": self.proportions,
                "alternatives": list(self.alternatives), "confidence": self.confidence,
                "what_would_help": self.what_would_help, "notes": self.notes,
                "provenance": "vlm_proposed"}


def answer_schema(catalog: List[str]) -> dict:
    """The JSON schema of the query (vLLM structured output — enums from the catalogue)."""
    forms = [f for f in catalog if f in FORMS]
    part_forms = [f for f in PART_FORMS if f in forms] or list(PART_FORMS)
    return {
        "type": "object",
        "properties": {
            "form": {"type": "string", "enum": forms},
            "orientation": {"type": "string", "enum": list(ORIENTATIONS)},
            "axis_aligned": {"type": "boolean"},
            "hollow": {"type": ["boolean", "null"]},
            "outline": {"type": "string", "enum": list(OUTLINES)},
            "profile": {"type": "string", "enum": list(PROFILES)},
            "parts": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "form": {"type": "string", "enum": part_forms},
                               "count": {"type": "integer"},
                               "orientation": {"type": "string", "enum": list(ORIENTATIONS)},
                               "relation": {"type": "string"}},
                "required": ["name", "form", "count", "orientation", "relation"]}},
            "proportions": {"type": "string"},
            "alternatives": {"type": "array", "items": {"type": "string", "enum": forms}},
            "confidence": {"type": "string", "enum": list(CONFIDENCES)},
            "what_would_help": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["form", "orientation", "axis_aligned", "hollow", "outline", "profile",
                     "parts", "proportions", "alternatives", "confidence", "what_would_help",
                     "notes"],
    }


class AnswerError(ValueError):
    """The VLM's answer does not fit the catalogue — declared, never coerced."""


def _enum(value, allowed, what: str, default: Optional[str] = None) -> str:
    v = str(value).strip() if value is not None else ""
    if v in allowed:
        return v
    if default is not None and v == "":
        return default
    raise AnswerError(f"{what} {value!r} is not one of {list(allowed)}")


def parse_answer(doc: Any, catalog: List[str]) -> FormAnswer:
    """A parsed JSON document → :class:`FormAnswer`; raises :class:`AnswerError`."""
    if not isinstance(doc, dict):
        raise AnswerError(f"the answer is not a JSON object: {type(doc).__name__}")
    forms = [f for f in catalog if f in FORMS]
    form = _enum(doc.get("form"), forms, "form")
    parts: List[Part] = []
    for i, p in enumerate(doc.get("parts") or []):
        if not isinstance(p, dict):
            raise AnswerError(f"parts[{i}] is not an object")
        try:
            count = int(p.get("count", 1))
        except (TypeError, ValueError):
            raise AnswerError(f"parts[{i}].count {p.get('count')!r} is not an integer") from None
        parts.append(Part(name=str(p.get("name") or f"part{i + 1}")[:60],
                          form=_enum(p.get("form"), PART_FORMS, f"parts[{i}].form"),
                          count=max(1, count),
                          orientation=_enum(p.get("orientation"), ORIENTATIONS,
                                            f"parts[{i}].orientation", "unknown"),
                          relation=str(p.get("relation") or "")[:200]))
    if form == "composite" and not parts:
        raise AnswerError("a composite answer carries no parts")
    hollow = doc.get("hollow")
    if hollow is not None and not isinstance(hollow, bool):
        raise AnswerError(f"hollow {hollow!r} is not a boolean or null")
    alts = []
    for a in doc.get("alternatives") or []:
        a = str(a).strip()
        if a in forms and a != form and a not in alts:
            alts.append(a)
    return FormAnswer(
        form=form,
        orientation=_enum(doc.get("orientation"), ORIENTATIONS, "orientation", "unknown"),
        axis_aligned=bool(doc.get("axis_aligned", False)),
        hollow=hollow,
        outline=_enum(doc.get("outline"), OUTLINES, "outline", "unknown"),
        profile=_enum(doc.get("profile"), PROFILES, "profile", "unknown"),
        parts=parts, proportions=str(doc.get("proportions") or "")[:300],
        alternatives=alts,
        confidence=_enum(doc.get("confidence"), CONFIDENCES, "confidence", "medium"),
        what_would_help=str(doc.get("what_would_help") or "")[:300],
        notes=str(doc.get("notes") or "")[:500])


def answer_from_dict(d: dict) -> FormAnswer:
    """A stored ``to_dict`` record back to :class:`FormAnswer` (every form of the catalogue)."""
    return parse_answer(d, list(FORMS))
