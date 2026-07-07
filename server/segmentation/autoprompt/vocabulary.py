# STAC-Builder — Auto-prompter vocabulary loader (Phase 1).
#
# Loads the construction vocabulary (vocabulary.yaml) into typed records and
# exposes the derived sets the pipeline needs (structural / dynamic / small,
# per-class confidence overrides, grounding phrases). Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_VOCAB = Path(__file__).resolve().parent / "vocabulary.yaml"


@dataclass(frozen=True)
class VocabClass:
    label: str
    es: str = ""
    phrase: str = ""
    structural: bool = False
    dynamic: bool = False
    small: bool = False
    min_confidence: float | None = None

    @property
    def grounding_phrase(self) -> str:
        return self.phrase or self.label


@dataclass
class Vocabulary:
    classes: list[VocabClass] = field(default_factory=list)

    def __post_init__(self):
        self._by_label = {c.label: c for c in self.classes}

    # ── lookups ─────────────────────────────────────────────────────
    def labels(self) -> list[str]:
        return [c.label for c in self.classes]

    def get(self, label: str) -> VocabClass | None:
        return self._by_label.get(label)

    def structural_labels(self) -> set[str]:
        return {c.label for c in self.classes if c.structural}

    def dynamic_labels(self) -> set[str]:
        return {c.label for c in self.classes if c.dynamic}

    def small_labels(self) -> set[str]:
        return {c.label for c in self.classes if c.small}

    def min_confidence_for(self, label: str, default: float) -> float:
        c = self._by_label.get(label)
        if c is not None and c.min_confidence is not None:
            return c.min_confidence
        return default

    def resolve_label(self, raw: str) -> str | None:
        """Map a VLM-returned string to a canonical label (exact, alias, or
        substring), else None. Keeps detection robust to label drift."""
        if not raw:
            return None
        s = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if s in self._by_label:
            return s
        for c in self.classes:
            if s == c.es.lower().replace(" ", "_"):
                return c.label
        # loose contains match (e.g. "concrete_column" -> column)
        for c in self.classes:
            if c.label in s or (c.es and c.es.lower().replace(" ", "_") in s):
                return c.label
        return None


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    p = Path(path) if path else _DEFAULT_VOCAB
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    classes = [VocabClass(**{k: v for k, v in c.items()}) for c in data.get("classes", [])]
    return Vocabulary(classes=classes)
