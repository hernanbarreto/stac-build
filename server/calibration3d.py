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
* a PLANE (USER 2026-11-13: "la calibracion por plano tambien debe quedar sobre la nube … y debe
  dibujar el plano sobre la nube 3d") is three or more points picked on a flat surface of the
  model — a floor, a wall. The server fits the plane to them (least squares) and stores what it
  measures: its normal, its inclination from the horizontal (0° a floor, 90° a wall), the height
  of its centroid, how far the picked points lie from it (rms) and the area they enclose. Like a
  length it carries its geometry epoch, and nothing is levelled or moved by it;
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
# THE TYPE OF CALIBRATION (USER 2026-11-13: "que tipo de calibracion, si es por ejes, debe indicarse y
# no debe dejarse calibrar si no estan las 3 medidas, si solo hay una o dos se hace global y la otra
# medida es de verificacion"): global — ONE factor for the whole cloud; axes — one factor per axis of
# the display frame (X and Z horizontal, Y vertical), allowed only with a gauge measurement along
# each of the three
MODES = ("global", "axes")
AXES = ("X", "Y", "Z")
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
            "measurements": [], "planes": []}


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


def axis_of(a: List[float], b: List[float]) -> str:
    """The axis a measurement runs along: the one with the largest component of b − a (the first
    among equals)."""
    d = np.abs(np.asarray(b, np.float64) - np.asarray(a, np.float64))
    return AXES[int(np.argmax(d))]


def axis_factors(gauge: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The per-axis calibration the gauge measurements allow. Each measurement gives
    real² = Σ_k (s_k · d_k)² with d = b − a in the model: linear in s_k². It is READY only when a
    gauge measurement runs along each of the three axes (its largest component) and the least
    squares gives three positive s_k²; otherwise ``reason`` says what is missing."""
    covered = sorted({g["axis"] for g in gauge}, key=AXES.index)
    missing = [ax for ax in AXES if ax not in covered]
    out: Dict[str, Any] = {"covered": covered, "missing": missing, "ready": False,
                           "factors": None, "reason": None}
    if missing:
        out["reason"] = ("per-axis calibration needs a gauge measurement along each of the three "
                         "axes — missing: " + ", ".join(missing) + "; with one or two the "
                         "calibration is global and the other measurements verify")
        return out
    d2 = np.square(np.asarray([g["d"] for g in gauge], np.float64))
    t = np.asarray([g["value_m"] for g in gauge], np.float64)
    sig = np.asarray([g["sigma_m"] for g in gauge], np.float64)
    pos = sig[sig > 0]
    # the uncertainty of real² is 2·real·sigma (a sigma of 0 takes the smallest positive one;
    # all zero: equal weights) — the weighting of evaluation.geometry.gauge_scale, on squares
    w = np.ones_like(t) if pos.size == 0 else 1.0 / (2.0 * t * np.where(sig > 0, sig, pos.min()))
    u, _res, rank, _sv = np.linalg.lstsq(d2 * w[:, None], t * t * w, rcond=None)
    if int(rank) < 3 or not np.all(u > 0):
        out["reason"] = ("the gauge measurements do not determine a factor for each axis (they "
                         "run too close to one another) — the calibration is global")
        return out
    f = np.sqrt(u)
    out.update(ready=True, factors={"x": float(f[0]), "y": float(f[1]), "z": float(f[2])})
    return out


def set_mode(doc: Dict[str, Any], mode: str, geometry_epoch: int) -> str:
    """Choose the type of calibration. 'axes' is REFUSED unless the three axes are covered."""
    if mode not in MODES:
        raise CalibrationError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if mode == "axes":
        ax = summary(doc, geometry_epoch)["axes"]
        if not ax["ready"]:
            raise CalibrationError(ax["reason"])
    doc["mode"] = mode
    return mode


def fit_plane(points: List[List[float]]) -> Dict[str, Any]:
    """The least-squares plane of the picked points and what it measures (tool_measured): unit
    normal (pointing up, +Y), centroid, inclination from the horizontal in degrees (the angle
    between the normal and the vertical: 0 a floor, 90 a wall), the rms and the largest distance
    of the points to the plane, and the area of the polygon they draw, in the picked order."""
    P = np.asarray(points, np.float64)
    c = P.mean(0)
    _u, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] / (np.linalg.norm(vt[2]) or 1.0)
    if n[1] < 0 or (n[1] == 0 and (n[0] < 0 or (n[0] == 0 and n[2] < 0))):
        n = -n
    d = (P - c) @ n
    u = vt[0] / (np.linalg.norm(vt[0]) or 1.0)
    v = np.cross(n, u)
    q = np.stack([(P - c) @ u, (P - c) @ v], 1)
    area = 0.5 * abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])))
    return {"normal": [float(x) for x in n], "centroid": [float(x) for x in c],
            "inclination_deg": float(np.degrees(np.arccos(min(1.0, abs(float(n[1])))))),
            "height_m": float(c[1]),
            "rms_m": float(np.sqrt(np.mean(d * d))), "max_m": float(np.abs(d).max()),
            "area_m2": area, "spread_m": float(sv[1] / np.sqrt(len(P)))}


def make_plane(body: Dict[str, Any], *, pid: str, geometry_epoch: int,
               created: Optional[str] = None) -> Dict[str, Any]:
    """A validated plane from what the viewer sends: ``points`` (three or more picked points, in
    the order they were marked), and optionally ``label`` and ``note``. Unknown keys are refused."""
    extra = sorted(set(body) - {"points", "label", "note", "scan", "scan_key"})
    if extra:
        raise CalibrationError(f"unknown field(s): {', '.join(extra)}")
    pts = body.get("points")
    if not isinstance(pts, (list, tuple)) or len(pts) < 3:
        raise CalibrationError("points: three or more picked points are expected")
    pts = [_point(p, f"points/{k}") for k, p in enumerate(pts)]
    fit = fit_plane(pts)
    if not fit["area_m2"] > 0 or not fit["spread_m"] > 0:
        raise CalibrationError("the points lie on one line: they define no plane")
    label = body.get("label", "")
    note = body.get("note", "")
    if not isinstance(label, str) or not isinstance(note, str):
        raise CalibrationError("label and note must be text")
    fit.pop("spread_m")
    return {"id": pid, "points": pts, "label": label, "note": note, **fit,
            "geometry_epoch": int(geometry_epoch),
            "created": created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def validate_document(doc: Any) -> List[str]:
    """Every problem of a stored document, [] when it is valid."""
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["<root>: an object is expected"]
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version: {doc.get('schema_version')!r} is not {SCHEMA_VERSION}")
    if doc.get("frame") != FRAME:
        problems.append(f"frame: {doc.get('frame')!r} is not {FRAME!r}")
    if doc.get("mode", "global") not in MODES:
        problems.append(f"mode: {doc.get('mode')!r} is not one of {list(MODES)}")
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
    planes = doc.get("planes", [])                     # absent in a document of before the planes
    if not isinstance(planes, list):
        return problems + ["planes: a list is expected"]
    for k, pl in enumerate(planes):
        where = f"planes/{k}"
        if not isinstance(pl, dict):
            problems.append(f"{where}: an object is expected")
            continue
        pid = pl.get("id")
        if not isinstance(pid, str) or not pid:
            problems.append(f"{where}: id is missing")
        elif pid in seen:
            problems.append(f"{where}: id {pid!r} is used twice")
        seen.add(pid)
        try:
            chk = make_plane({key: pl[key] for key in ("points", "label", "note") if key in pl},
                             pid=str(pid), geometry_epoch=int(pl.get("geometry_epoch", -1)),
                             created=str(pl.get("created", "")))
            for key in ("inclination_deg", "rms_m", "area_m2"):
                if abs(chk[key] - float(pl.get(key, float("nan")))) > 1e-9:
                    problems.append(f"{where}: {key} is not what its points measure")
            if isinstance(pl.get("geometry_epoch"), bool) or not isinstance(pl.get("geometry_epoch"), int):
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


def next_id(doc: Dict[str, Any], prefix: str = "c") -> str:
    used = {m["id"] for m in doc.get("measurements", [])} | {p["id"] for p in doc.get("planes", [])}
    k = 1
    while f"{prefix}{k}" in used:
        k += 1
    return f"{prefix}{k}"


def add_plane(doc: Dict[str, Any], body: Dict[str, Any], geometry_epoch: int) -> Dict[str, Any]:
    pl = make_plane(body, pid=next_id(doc, "p"), geometry_epoch=geometry_epoch)
    doc.setdefault("planes", []).append(pl)
    return pl


def update_plane(doc: Dict[str, Any], pid: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Change a plane's label or note. Its points and what they measure never change."""
    extra = sorted(set(body) - {"label", "note", "scan", "scan_key"})
    if extra:
        raise CalibrationError(f"not editable: {', '.join(extra)}")
    for k, pl in enumerate(doc.get("planes", [])):
        if pl["id"] == pid:
            merged = {"points": pl["points"], "label": body.get("label", pl["label"]),
                      "note": body.get("note", pl["note"])}
            doc["planes"][k] = make_plane(merged, pid=pid, geometry_epoch=pl["geometry_epoch"],
                                          created=pl["created"])
            return doc["planes"][k]
    raise KeyError(pid)


def remove_plane(doc: Dict[str, Any], pid: str) -> None:
    before = len(doc.get("planes", []))
    doc["planes"] = [p for p in doc.get("planes", []) if p["id"] != pid]
    if len(doc["planes"]) == before:
        raise KeyError(pid)


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
        dy = abs(float(m["b"][1]) - float(m["a"][1]))
        rows.append({"id": m["id"], "role": m["role"], "tool": m["tool"],
                     "axis": axis_of(m["a"], m["b"]),
                     "d": [float(m["b"][k]) - float(m["a"][k]) for k in range(3)],
                     "measured_m": m["measured_m"], "value_m": m["value_m"],
                     "sigma_m": m["sigma_m"], "ratio": m["value_m"] / m["measured_m"],
                     # the direction it was measured in: 0 = horizontal, 90 = vertical (a factor
                     # is ONE number for the whole cloud; measures in other directions and zones
                     # are what tells whether one number is enough)
                     "elevation_deg": float(np.degrees(np.arcsin(min(1.0, dy / m["measured_m"])))),
                     "geometry_epoch": m["geometry_epoch"],
                     "stale": int(m["geometry_epoch"]) != int(geometry_epoch)})
    live = [r for r in rows if not r["stale"]]
    gauge = [r for r in live if r["role"] == "gauge"]
    planes = [{"id": p["id"], "label": p["label"], "inclination_deg": p["inclination_deg"],
               "height_m": p["height_m"], "rms_m": p["rms_m"], "max_m": p["max_m"],
               "area_m2": p["area_m2"], "n_points": len(p["points"]),
               "geometry_epoch": p["geometry_epoch"],
               "stale": int(p["geometry_epoch"]) != int(geometry_epoch)}
              for p in doc.get("planes", [])]
    mode = doc.get("mode", "global")
    axes = axis_factors(gauge)
    # what is used: per axis only when it was chosen AND the three axes are covered; else global
    mode_used = "axes" if (mode == "axes" and axes["ready"]) else "global"
    out: Dict[str, Any] = {
        "mode": mode, "mode_used": mode_used, "axes": axes,
        "planes": planes,
        "geometry_epoch": int(geometry_epoch), "n": len(rows), "n_gauge": len(gauge),
        "n_test": sum(1 for r in live if r["role"] == "test"),
        "n_stale": sum(1 for r in rows if r["stale"]),
        "scale_factor": None, "weighting": None, "rows": rows, "provenance": "tool_measured"}
    # THE STATE OF THE SCAN (USER 2026-11-13: "debe indicarse si la nube esta calibrada o no, eso lo
    # debe indicar al abrir la nube"): uncalibrated — no gauge measurement on this geometry;
    # calibrated — a factor, not checked by any test measurement; verified — checked
    out["state"] = "uncalibrated"
    out["verification"] = None
    if not gauge:
        out["reason"] = ("no gauge measurement on the current geometry epoch: nothing "
                         "calibrates the scale")
        for r in rows:
            r["calibrated_m"] = None
            r["residual_m"] = None
            r["within_sigma"] = None
            r.pop("d")
        return out
    s, how = G.gauge_scale([r["measured_m"] for r in gauge], [r["value_m"] for r in gauge],
                           [r["sigma_m"] for r in gauge])
    out["scale_factor"], out["weighting"] = s, how
    fx = axes["factors"] if mode_used == "axes" else None
    for r in rows:
        d = r.pop("d")
        if r["stale"]:
            r["calibrated_m"] = r["residual_m"] = None
        else:
            # the length after the calibration IN USE: one factor, or one per axis
            cal = (s * r["measured_m"] if fx is None else
                   float(np.sqrt((fx["x"] * d[0]) ** 2 + (fx["y"] * d[1]) ** 2 + (fx["z"] * d[2]) ** 2)))
            r["calibrated_m"] = cal
            r["residual_m"] = cal - r["value_m"]
        # a check against the user's own uncertainty, when he gave one; no other bar is invented
        r["within_sigma"] = (None if (r["stale"] or not r["sigma_m"] > 0)
                             else bool(abs(r["residual_m"]) <= r["sigma_m"]))
    g_res = [abs(r["residual_m"]) for r in gauge]
    out["gauge_fit_residual_rms_m"] = float(np.sqrt(np.mean(np.square(g_res))))
    out["gauge_fit_residual_max_m"] = float(max(g_res))
    out["gauge_ratio_min"] = float(min(r["ratio"] for r in gauge))
    out["gauge_ratio_max"] = float(max(r["ratio"] for r in gauge))
    test = [r for r in live if r["role"] == "test"]
    out["state"] = "verified" if test else "calibrated"
    if test:
        abs_e = [abs(r["residual_m"]) for r in test]
        rel_e = [abs(r["residual_m"]) / r["value_m"] for r in test]
        # THE VERIFICATION (USER 2026-11-13: "una vez que se ajusta la calibracion debe verificarse con
        # la medida de verificacion que de correctamente, sino debe indicarse la diferencia"): the
        # difference of every test measurement after the factor, the largest one, and whether each
        # lies within the uncertainty the user stated for it (None when he stated none)
        judged = [r["within_sigma"] for r in test if r["within_sigma"] is not None]
        worst = max(test, key=lambda r: abs(r["residual_m"]))
        out["verification"] = {"n": len(test), "max_abs_m": float(max(abs_e)),
                               "max_rel": float(max(rel_e)), "worst_id": worst["id"],
                               "worst_difference_m": float(worst["residual_m"]),
                               "n_with_sigma": len(judged),
                               "all_within_sigma": (bool(all(judged)) if judged else None)}
        out["test_abs_error_m"] = {"median": G.percentile(abs_e, 50), "max": float(max(abs_e)),
                                   "n": len(abs_e)}
        out["test_rel_error"] = {"median": G.percentile(rel_e, 50), "max": float(max(rel_e)),
                                 "n": len(rel_e)}
    out["note"] = ("the gauge residuals describe the fit, they are not the error (the error is "
                   "measured on the test measurements only); the factor is reported, not applied")
    return out
