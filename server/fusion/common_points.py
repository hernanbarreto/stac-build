"""COMMON POINTS — the same physical point marked on two (or more) scans of a project, and the
similarity they give between two scans.

USER 2026-11-17, on comparing the epochs of a project (pccr 2026-08-24 against 2026-08-31): "para
comparar dos o más épocas hay que emparejar sus escalas … deberían marcarse dos puntos y decir estos
son los mismos en ambas … y que se ajusten a uno de ellos, o a una medida externa … para que se
puedan posar uno sobre otro y comparar".

WHAT IS HERE: the points (``<project>/common_points.json``: a name and, per scan it was marked on,
the picked position in that scan's display frame with the scan's geometry epoch), and
:func:`solve` — the similarity that brings one scan onto another from the points marked on both,
REPORTED with what each point is left off by. NOTHING IS APPLIED here: where the factor goes (the
composition's transform, the lengths shown, the scan's geometry) is the user's decision
(CLAUDE.md §7, B-5); this gives him the number and its residuals to decide with.

THE MODEL — a scan's display frame is levelled (its floor step: Y up), so two scans of one place
differ by a SCALE, a turn about the vertical (YAW) and a shift:

    reference ≈ s · R_y(θ) · scan + t

Least squares over the n pairs (n ≥ 2), in closed form: with both sets centred, the horizontal
parts as complex numbers u (scan) and v (reference) and the heights a_y, b_y,

    A = Σ v · conj(u)            θ = arg A
    s = (|A| + Σ a_y b_y) / Σ |a|²       (or the scale GIVEN — "o a una medida externa")
    t = mean(reference) − s · R_y(θ) · mean(scan)

Two points fix the horizontal scale, the yaw and the shift exactly (their residual is then the
height the two scans disagree by); three or more leave a residual per point, the measure of how
well the scans can be laid one on the other. A mark made on an earlier geometry of its scan (the
cloud was corrected since) is STALE: listed, not used.
"""
from __future__ import annotations

import json
import math
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

VERSION = 1
FILE = "common_points.json"
NAME_MAX = 80
_lock = threading.Lock()


class CommonPointsError(ValueError):
    """What was asked cannot be done with these points — the message says why."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(project_dir: Path | str) -> Dict[str, Any]:
    path = Path(project_dir) / FILE
    if not path.exists():
        return {"version": VERSION, "next_number": 1, "points": []}
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("version") != VERSION or not isinstance(doc.get("points"), list):
        raise CommonPointsError(f"{path}: not a common-points document of version {VERSION}")
    return doc


def _save(project_dir: Path | str, doc: Dict[str, Any]) -> None:
    path = Path(project_dir) / FILE
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    os.replace(part, path)


def _position(value: Any) -> List[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise CommonPointsError("position must be three finite numbers (x, y, z in metres, the scan's display frame)")
    return [float(v) for v in value]


def _find(doc: Dict[str, Any], point_id: str) -> Dict[str, Any]:
    for p in doc["points"]:
        if p["id"] == point_id or p["name"] == point_id:
            return p
    raise KeyError(point_id)


def add(project_dir: Path | str, name: Optional[str] = None) -> Dict[str, Any]:
    """A new common point (named P<n> unless a name is given), marked on no scan yet."""
    with _lock:
        doc = load(project_dir)
        number = int(doc["next_number"])
        label = (name or "").strip() or f"P{number}"
        if len(label) > NAME_MAX:
            raise CommonPointsError(f"name is longer than {NAME_MAX} characters")
        if any(p["name"] == label for p in doc["points"]):
            raise CommonPointsError(f"a common point named '{label}' exists already")
        point = {"id": secrets.token_hex(6), "name": label, "marks": {}}
        doc["points"].append(point)
        doc["next_number"] = number + 1
        _save(project_dir, doc)
    return point


def mark(project_dir: Path | str, point_id: str, scan: str, position: Any, geometry_epoch: Optional[int] = None,
         author: str = "") -> Dict[str, Any]:
    """Mark (or mark again) a common point on one scan: the picked position in that scan's display frame."""
    if not isinstance(scan, str) or not scan.strip():
        raise CommonPointsError("the scan the point is marked on is missing")
    pos = _position(position)
    with _lock:
        doc = load(project_dir)
        point = _find(doc, point_id)
        point["marks"][scan] = {"position": pos, "geometry_epoch": None if geometry_epoch is None else int(geometry_epoch),
                                "marked_by": str(author or ""), "marked_at": _now()}
        _save(project_dir, doc)
    return point


def unmark(project_dir: Path | str, point_id: str, scan: str) -> Dict[str, Any]:
    with _lock:
        doc = load(project_dir)
        point = _find(doc, point_id)
        if scan not in point["marks"]:
            raise KeyError(f"{point_id} on {scan}")
        del point["marks"][scan]
        _save(project_dir, doc)
    return point


def delete(project_dir: Path | str, point_id: str) -> Dict[str, Any]:
    with _lock:
        doc = load(project_dir)
        point = _find(doc, point_id)
        doc["points"].remove(point)
        _save(project_dir, doc)
    return point


def listing(project_dir: Path | str, epochs: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """The points, each mark with ``stale`` (its scan's geometry is no longer the one it was marked
    on — ``epochs``: {scan key: current epoch}; a scan not in it is not judged)."""
    out = []
    for p in load(project_dir)["points"]:
        marks = {}
        for scan, m in p["marks"].items():
            now = (epochs or {}).get(scan)
            marks[scan] = {**m, "stale": now is not None and m.get("geometry_epoch") is not None and int(now) != int(m["geometry_epoch"])}
        out.append({"id": p["id"], "name": p["name"], "marks": marks})
    return out


# ── the similarity ─────────────────────────────────────────────────────────────────────────────
def similarity(scan_points: np.ndarray, reference_points: np.ndarray, scale: Optional[float] = None) -> Dict[str, Any]:
    """reference ≈ s · R_y(θ) · scan + t from n ≥ 2 pairs (rows), least squares; ``scale`` given
    fixes s. Returns {scale, yaw_deg, translation, matrix (4×4, row-major, scan → reference),
    residuals (n × 3, reference − fitted), rms_m, max_m, baseline_m}."""
    a = np.asarray(scan_points, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    if len(a) != len(b) or len(a) < 2:
        raise CommonPointsError("two scans are laid one on the other from at least 2 points marked on both")
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb
    # horizontal parts as complex numbers (x + i z); the vertical is y
    u = a0[:, 0] + 1j * a0[:, 2]
    v = b0[:, 0] + 1j * b0[:, 2]
    A = np.sum(v * np.conj(u))
    spread = float(np.sum(a0 * a0))
    if spread <= 0.0 or float(np.sum(np.abs(u) ** 2)) <= 0.0:
        raise CommonPointsError("the points marked on the scan coincide (or lie on one vertical): they fix no scale and no turn")
    theta = float(np.angle(A)) if abs(A) > 0.0 else 0.0
    if scale is None:
        s = (abs(A) + float(np.sum(a0[:, 1] * b0[:, 1]))) / spread
        source = "points"
    else:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise CommonPointsError("the given scale must be a positive number")
        s, source = float(scale), "given"
    if s <= 0.0:
        raise CommonPointsError("the points give no positive scale (are the same points marked on both scans?)")
    c, n = math.cos(theta), math.sin(theta)
    # x' + i z' = e^{iθ} (x + i z):  x' = c x − n z,  z' = n x + c z ; y' = y
    R = np.array([[c, 0.0, -n], [0.0, 1.0, 0.0], [n, 0.0, c]])
    t = cb - s * (R @ ca)
    fitted = s * (a @ R.T) + t
    residuals = b - fitted
    norms = np.linalg.norm(residuals, axis=1)
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    baseline = float(max(np.linalg.norm(b[i] - b[j]) for i in range(len(b)) for j in range(i + 1, len(b))))
    return {"scale": float(s), "scale_source": source, "yaw_deg": float(math.degrees(theta)),
            "translation": [float(x) for x in t], "matrix": [float(x) for x in M.reshape(-1)],
            "residuals": residuals.tolist(), "residual_m": [float(x) for x in norms],
            "rms_m": float(math.sqrt(float(np.mean(norms ** 2)))), "max_m": float(norms.max()),
            "baseline_m": baseline, "pairs": int(len(a))}


def solve(project_dir: Path | str, scan: str, reference: str, epochs: Optional[Dict[str, int]] = None,
          scale: Optional[float] = None) -> Dict[str, Any]:
    """The similarity that brings ``scan`` onto ``reference`` from the points marked on both (stale
    marks left out and listed), with each point's residual. Raises CommonPointsError when fewer
    than two usable points are marked on both."""
    if scan == reference:
        raise CommonPointsError("a scan is laid on ANOTHER scan")
    used, stale, only_one = [], [], []
    for p in listing(project_dir, epochs):
        ma, mb = p["marks"].get(scan), p["marks"].get(reference)
        if ma is None or mb is None:
            if ma is not None or mb is not None:
                only_one.append(p["name"])
            continue
        if ma["stale"] or mb["stale"]:
            stale.append(p["name"])
            continue
        used.append((p, ma["position"], mb["position"]))
    if len(used) < 2:
        raise CommonPointsError(f"{len(used)} point(s) marked on both {scan} and {reference}: at least 2 are needed"
                                + (f" ({len(stale)} more were marked on an earlier geometry)" if stale else ""))
    fit = similarity(np.array([u[1] for u in used]), np.array([u[2] for u in used]), scale)
    points = [{"id": p["id"], "name": p["name"], "residual_m": fit["residual_m"][k], "residual": fit["residuals"][k]}
              for k, (p, _, _) in enumerate(used)]
    return {"scan": scan, "reference": reference, **{k: fit[k] for k in ("scale", "scale_source", "yaw_deg", "translation",
                                                                         "matrix", "rms_m", "max_m", "baseline_m", "pairs")},
            "points": points, "stale": stale, "marked_on_one_only": only_one,
            # two points are fitted exactly in plan: what is left is the height the scans disagree by
            "determined": "exactly (2 points: no residual in plan)" if len(used) == 2 else "least squares",
            "applied": False}
