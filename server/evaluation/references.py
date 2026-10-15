"""Phase 0 — reference measures of a scan: hand-measured distances and planar regions anchored
in IMAGE space (keyframe, pixel), never in 3D, so every variant of the chain can rebuild them.

One JSON per scan (``output/<evaluation.references.file>``), ``schema_version`` 1, validated
against ``reference_schema.json`` (JSON Schema 2020-12) plus the semantic checks the schema
cannot express (unique ids, pixels inside the image when its size is known).

Not the same artefact as ``output/known_dimensions.json`` (F8: per-OBJECT dimensions with a
measured value, read by the gauge as a scale instrument): a reference distance is two physical
points, and it is measured by the battery, never by the chain.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from atomic_io import atomic_write_json

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("reference_schema.json")
ROLES = ("gauge", "test")
TOOLS = ("tape", "laser", "total_station")
PROVENANCE = "human_validated"


class ReferencesError(ValueError):
    """The reference document is invalid or incomplete for what was asked of it."""


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def empty_document() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "provenance": PROVENANCE, "image_size": None,
            "distances": [], "planes": []}


def validate_document(doc: Any) -> List[str]:
    """Every problem of ``doc`` (schema and semantic), [] when it is valid."""
    import jsonschema
    problems: List[str] = []
    validator = jsonschema.Draft202012Validator(load_schema())
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        problems.append(f"{where}: {err.message}")
    if problems or not isinstance(doc, dict):
        return problems
    size = doc.get("image_size")
    seen: Dict[str, str] = {}
    for kind in ("distances", "planes"):
        for item in doc.get(kind, []):
            iid = item["id"]
            if iid in seen:
                problems.append(f"{kind}/{iid}: id already used by {seen[iid]}")
            seen[iid] = kind
    if size is not None:
        w, h = int(size[0]), int(size[1])

        def _check(u, v, where):
            if not (0 <= u <= w and 0 <= v <= h):
                problems.append(f"{where}: pixel ({u}, {v}) outside the {w}x{h} image")

        for d in doc.get("distances", []):
            for e_i, end in enumerate(d["ends"]):
                for o_i, ob in enumerate(end["observations"]):
                    _check(ob["u"], ob["v"], f"distances/{d['id']}/ends/{e_i}/observations/{o_i}")
        for p in doc.get("planes", []):
            for g_i, poly in enumerate(p["polygons"]):
                for pt_i, (u, v) in enumerate(poly["points"]):
                    _check(u, v, f"planes/{p['id']}/polygons/{g_i}/points/{pt_i}")
    return problems


def counts(doc: Dict[str, Any]) -> Dict[str, int]:
    dists = doc.get("distances", [])
    return {"distances": len(dists),
            "gauge": sum(1 for d in dists if d.get("role") == "gauge"),
            "test": sum(1 for d in dists if d.get("role") == "test"),
            "planes": len(doc.get("planes", []))}


def role_warnings(doc: Dict[str, Any]) -> List[str]:
    c = counts(doc)
    out = []
    if c["gauge"] == 0:
        out.append("no gauge distance: nothing calibrates the scale")
    if c["test"] == 0:
        out.append("no test distance: nothing measures the error")
    return out


def check_roles(doc: Dict[str, Any]) -> None:
    """The evaluation needs at least one gauge (calibrates) and one test (measures) distance."""
    w = role_warnings(doc)
    if w:
        raise ReferencesError("the reference document cannot be evaluated: " + "; ".join(w))


def references_path(output_dir: Path, file_name: str) -> Path:
    return Path(output_dir) / file_name


def load_references(output_dir: Path, file_name: str, *, require_roles: bool = False
                    ) -> Dict[str, Any]:
    """The scan's document (an empty one when the file does not exist). An unreadable or invalid
    file RAISES naming every problem — never a default in its place."""
    p = references_path(output_dir, file_name)
    if not p.exists():
        doc = empty_document()
    else:
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            raise ReferencesError(f"{p} is unreadable ({e})") from e
        problems = validate_document(doc)
        if problems:
            raise ReferencesError(f"{p} is invalid: " + "; ".join(problems))
    if require_roles:
        check_roles(doc)
    return doc


def save_references(output_dir: Path, file_name: str, doc: Dict[str, Any]) -> Path:
    """Validate and write atomically (sorted keys, no clock, no absolute path inside)."""
    doc = dict(doc)
    doc.setdefault("schema_version", SCHEMA_VERSION)
    doc.setdefault("provenance", PROVENANCE)
    problems = validate_document(doc)
    if problems:
        raise ReferencesError("the reference document is invalid: " + "; ".join(problems))
    p = references_path(output_dir, file_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, doc, indent=1, sort_keys=True, ensure_ascii=False)
    return p


def new_distance(iid: str, value_m: float, sigma_m: float, role: str, tool: str,
                 end_a: List[Dict[str, Any]], end_b: List[Dict[str, Any]], note: str = ""
                 ) -> Dict[str, Any]:
    return {"id": iid, "value_m": float(value_m), "sigma_m": float(sigma_m), "role": role,
            "tool": tool, "note": note,
            "ends": [{"observations": list(end_a)}, {"observations": list(end_b)}]}


def new_plane(iid: str, label: str, polygons: List[Dict[str, Any]], note: str = ""
              ) -> Dict[str, Any]:
    return {"id": iid, "label": label, "note": note, "polygons": list(polygons)}


def observation(frame: int, u: float, v: float, file: Optional[str] = None) -> Dict[str, Any]:
    ob: Dict[str, Any] = {"frame": int(frame), "u": float(u), "v": float(v)}
    if file:
        ob["file"] = file
    return ob
