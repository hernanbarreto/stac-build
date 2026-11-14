"""Calibration measurements of a scan, taken ON THE 3D MODEL (USER 2026-11-13: "para calibrar debe
hacerse tomando mediciones sobre el modelo 3d no sobre las imagenes porque es dificil, se debe
poder seleccionar que una medicion es de calibracion y deben quedar almacenadas").

One JSON per scan, ``output/calibration_3d.json`` (schema_version 1):

* a MEASUREMENT is two points the user picked on the model in the viewer (the scan's display
  frame — the frame of the cloud on screen, of the boxes and of the chat's measurements), the
  REAL length between them as the user states it (``value_m``, human_validated, with its
  instrument and its sigma) and its ROLE: ``gauge`` calibrates the scale, ``test`` measures the
  error that remains after it — the two roles of the evaluation battery's references;
* the length the MODEL gives (``measured_m``) is computed HERE from the two stored points
  (tool_measured) — never taken from the client — and carries the geometry epoch it was taken
  on: a measurement of another epoch is STALE (the cloud under its points is no longer the one
  it was picked on), listed and never used;
* ``tool: other_scan`` (with ``reference_scan``) is a length taken from ANOTHER SCAN of the same
  place instead of an instrument: it brings this scan to that scan's scale (USER 2026-11-13:
  "emparejar las escalas de pccr2408 y de 3108 … debería hacerse con la calibración").

THE SCALE FACTOR of the scan is ``evaluation.geometry.gauge_scale`` over its gauge measurements
— the evaluation battery's estimator: the least-squares s of Σ w (s · measured − real)²,
w = 1/σ². It is REPORTED with the residuals of the fit and the error on the test measurements.
Nothing here applies it to a cloud, a pose or a mesh: what the factor is applied to is the
user's decision.

Not the same artefact as ``output/<evaluation.references.file>`` (reference distances anchored on
keyframe PIXELS, rebuilt by every variant of the chain): these are anchored on the model, on one
geometry epoch.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from atomic_io import atomic_write_json

SCHEMA_VERSION = 1
FILE_NAME = "calibration_3d.json"
ROLES = ("gauge", "test")
TOOLS = ("tape", "laser", "total_station", "other_scan")
PROVENANCE = "human_validated"
FRAME = "display"


class CalibrationError(ValueError):
    """The calibration document or a measurement is invalid."""


def path_of(output_dir: Path) -> Path:
    return Path(output_dir) / FILE_NAME


def empty_document() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "provenance": PROVENANCE, "frame": FRAME,
            "measurements": []}


def _point(v: Any, where: str) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise CalibrationError(f"{where}: three coordinates [x, y, z] are expected")
    out = []
    for c in v:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(float(c)):
            raise CalibrationError(f"{where}: {c!r} is not a finite number")
        out.append(float(c))
    return out


def _number(v: Any, where: str, *, positive: bool) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise CalibrationError(f"{where}: {v!r} is not a finite number")
    v = float(v)
    if (positive and v <= 0) or (not positive and v < 0):
        raise CalibrationError(f"{where}: {v} must be {'> 0' if positive else '>= 0'}")
    return v


def length_of(a: List[float], b: List[float]) -> float:
    """The model's length between two picked points, metres of the model."""
    return float(np.linalg.norm(np.asarray(b, np.float64) - np.asarray(a, np.float64)))


def make_measurement(body: Dict[str, Any], *, mid: str, geometry_epoch: int,
                     created: Optional[str] = None) -> Dict[str, Any]:
    """A validated measurement from what the viewer sends: ``a``, ``b`` (picked points),
    ``value_m`` (the real length), and optionally ``sigma_m``, ``role``, ``tool``, ``note``,
    ``reference_scan``. Unknown keys are refused."""
    known = {"a", "b", "value_m", "sigma_m", "role", "tool", "note", "reference_scan", "scan",
             "scan_key"}
    extra = sorted(set(body) - known)
    if extra:
        raise CalibrationError(f"unknown field(s): {', '.join(extra)}")
    a, b = _point(body.get("a"), "a"), _point(body.get("b"), "b")
    measured = length_of(a, b)
    if measured <= 0:
        raise CalibrationError("the two points are the same: the model's length is 0")
    role = body.get("role", "gauge")
    if role not in ROLES:
        raise CalibrationError(f"role must be one of {list(ROLES)}, got {role!r}")
    tool = body.get("tool", "tape")
    if tool not in TOOLS:
        raise CalibrationError(f"tool must be one of {list(TOOLS)}, got {tool!r}")
    ref_scan = body.get("reference_scan")
    if ref_scan is not None and not isinstance(ref_scan, str):
        raise CalibrationError("reference_scan must be a scan key ('date/source')")
    if tool == "other_scan" and not ref_scan:
        raise CalibrationError("tool 'other_scan' needs reference_scan: the scan the length "
                               "was measured on")
    note = body.get("note", "")
    if not isinstance(note, str):
        raise CalibrationError("note must be text")
    out = {"id": mid, "a": a, "b": b, "measured_m": measured,
           "value_m": _number(body.get("value_m"), "value_m", positive=True),
           "sigma_m": _number(body.get("sigma_m", 0.0), "sigma_m", positive=False),
           "role": role, "tool": tool, "note": note,
           "geometry_epoch": int(geometry_epoch),
           "created": created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if ref_scan:
        out["reference_scan"] = ref_scan
    return out


def validate_document(doc: Any) -> List[str]:
    """Every problem of a stored document, [] when it is valid."""
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["<root>: an object is expected"]
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version: {doc.get('schema_version')!r} is not {SCHEMA_VERSION}")
    if doc.get("frame") != FRAME:
        problems.append(f"frame: {doc.get('frame')!r} is not {FRAME!r}")
    ms = doc.get("measurements")
    if not isinstance(ms, list):
        return problems + ["measurements: a list is expected"]
    seen = set()
    for k, m in enumerate(ms):
        where = f"measurements/{k}"
        if not isinstance(m, dict):
            problems.append(f"{where}: an object is expected")
            continue
        mid = m.get("id")
        if not isinstance(mid, str) or not mid:
            problems.append(f"{where}: id is missing")
        elif mid in seen:
            problems.append(f"{where}: id {mid!r} is used twice")
        seen.add(mid)
        try:
            body = {key: m[key] for key in ("a", "b", "value_m", "sigma_m", "role", "tool", "note",
                                           "reference_scan") if key in m}
            chk = make_measurement(body, mid=str(mid), geometry_epoch=int(m.get("geometry_epoch", -1)),
                                   created=str(m.get("created", "")))
            if abs(chk["measured_m"] - float(m.get("measured_m", float("nan")))) > 1e-9:
                problems.append(f"{where}: measured_m is not the length between its two points")
            if isinstance(m.get("geometry_epoch"), bool) or not isinstance(m.get("geometry_epoch"), int):
                problems.append(f"{where}: geometry_epoch is missing")
        except (CalibrationError, TypeError, ValueError) as e:
            problems.append(f"{where}: {e}")
    return problems


def load(output_dir: Path) -> Dict[str, Any]:
    p = path_of(output_dir)
    if not p.exists():
        return empty_document()
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CalibrationError(f"{p} cannot be read: {e}") from e
    problems = validate_document(doc)
    if problems:
        raise CalibrationError(f"{p} is invalid: " + "; ".join(problems[:5])
                               + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else ""))
    return doc


def save(output_dir: Path, doc: Dict[str, Any]) -> Path:
    problems = validate_document(doc)
    if problems:
        raise CalibrationError("invalid calibration document: " + "; ".join(problems[:5]))
    p = path_of(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, doc, indent=2)
    return p


def next_id(doc: Dict[str, Any]) -> str:
    used = {m["id"] for m in doc.get("measurements", [])}
    k = 1
    while f"c{k}" in used:
        k += 1
    return f"c{k}"


def add(doc: Dict[str, Any], body: Dict[str, Any], geometry_epoch: int) -> Dict[str, Any]:
    m = make_measurement(body, mid=next_id(doc), geometry_epoch=geometry_epoch)
    doc.setdefault("measurements", []).append(m)
    return m


def update(doc: Dict[str, Any], mid: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Change what the USER states about a measurement (value, sigma, role, tool, note,
    reference scan). Its points, its model length and its epoch are what was measured: they
    never change — a new measurement is taken instead."""
    editable = {"value_m", "sigma_m", "role", "tool", "note", "reference_scan"}
    extra = sorted(set(body) - editable - {"scan", "scan_key"})
    if extra:
        raise CalibrationError(f"not editable: {', '.join(extra)}")
    for k, m in enumerate(doc.get("measurements", [])):
        if m["id"] == mid:
            merged = {key: m[key] for key in ("a", "b", "value_m", "sigma_m", "role", "tool", "note",
                                              "reference_scan") if key in m}
            merged.update({key: body[key] for key in editable if key in body})
            if merged.get("reference_scan") in ("", None):
                merged.pop("reference_scan", None)
            new = make_measurement(merged, mid=mid, geometry_epoch=m["geometry_epoch"],
                                   created=m["created"])
            doc["measurements"][k] = new
            return new
    raise KeyError(mid)


def remove(doc: Dict[str, Any], mid: str) -> None:
    before = len(doc.get("measurements", []))
    doc["measurements"] = [m for m in doc.get("measurements", []) if m["id"] != mid]
    if len(doc["measurements"]) == before:
        raise KeyError(mid)


def summary(doc: Dict[str, Any], geometry_epoch: int) -> Dict[str, Any]:
    """The scan's scale factor from its gauge measurements of the CURRENT geometry epoch, the
    residuals of that fit and the error on the test measurements — a report, tool_measured."""
    from evaluation import geometry as G
    rows = []
    for m in doc.get("measurements", []):
        rows.append({"id": m["id"], "role": m["role"], "tool": m["tool"],
                     "measured_m": m["measured_m"], "value_m": m["value_m"],
                     "sigma_m": m["sigma_m"], "ratio": m["value_m"] / m["measured_m"],
                     "geometry_epoch": m["geometry_epoch"],
                     "stale": int(m["geometry_epoch"]) != int(geometry_epoch)})
    live = [r for r in rows if not r["stale"]]
    gauge = [r for r in live if r["role"] == "gauge"]
    out: Dict[str, Any] = {
        "geometry_epoch": int(geometry_epoch), "n": len(rows), "n_gauge": len(gauge),
        "n_test": sum(1 for r in live if r["role"] == "test"),
        "n_stale": sum(1 for r in rows if r["stale"]),
        "scale_factor": None, "weighting": None, "rows": rows, "provenance": "tool_measured"}
    if not gauge:
        out["reason"] = ("no gauge measurement on the current geometry epoch: nothing "
                         "calibrates the scale")
        for r in rows:
            r["calibrated_m"] = None
            r["residual_m"] = None
        return out
    s, how = G.gauge_scale([r["measured_m"] for r in gauge], [r["value_m"] for r in gauge],
                           [r["sigma_m"] for r in gauge])
    out["scale_factor"], out["weighting"] = s, how
    for r in rows:
        r["calibrated_m"] = None if r["stale"] else s * r["measured_m"]
        r["residual_m"] = None if r["stale"] else s * r["measured_m"] - r["value_m"]
    g_res = [abs(r["residual_m"]) for r in gauge]
    out["gauge_fit_residual_rms_m"] = float(np.sqrt(np.mean(np.square(g_res))))
    out["gauge_fit_residual_max_m"] = float(max(g_res))
    out["gauge_ratio_min"] = float(min(r["ratio"] for r in gauge))
    out["gauge_ratio_max"] = float(max(r["ratio"] for r in gauge))
    test = [r for r in live if r["role"] == "test"]
    if test:
        abs_e = [abs(r["residual_m"]) for r in test]
        rel_e = [abs(r["residual_m"]) / r["value_m"] for r in test]
        out["test_abs_error_m"] = {"median": G.percentile(abs_e, 50), "max": float(max(abs_e)),
                                   "n": len(abs_e)}
        out["test_rel_error"] = {"median": G.percentile(rel_e, 50), "max": float(max(rel_e)),
                                 "n": len(rel_e)}
    out["note"] = ("the gauge residuals describe the fit, they are not the error (the error is "
                   "measured on the test measurements only); the factor is reported, not applied")
    return out
