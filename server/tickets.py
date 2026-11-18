"""TICKETS — a point of the model marked as something to look at.

USER 2026-11-09 (what follows the reconstruction, for the December presentation): "TICKETS — una
herramienta simple para marcar tickets en cualquier parte del proyecto, como un punto de revisión o
una no conformidad: es para mostrar el concepto nomás".

A ticket is a POINT picked on a scan's cloud (the scan's display frame, metres), a kind — a REVIEW
item or a NON-CONFORMITY —, a title, a few words, who marked it and when, and whether it is still
OPEN or was RESOLVED (by whom, when). When the point lies on a segmented object the ticket names it
(instance id and label). It carries the scan's GEOMETRY EPOCH at the moment it was marked: after a
correction that moved the cloud its point is the old geometry's, and it is listed as such (``stale``)
instead of being silently shown where it no longer is.

They are the PROJECT's (``<project>/tickets.json``, one numbering for all its scans — "ticket 7" is
one thing whichever scan is open), and travel with it (the session's ZIP). The file is rewritten
whole, atomically; one writer at a time in the server's process.

PROVENANCE: a ticket is a human's statement — ``human_stated`` — never a measurement; nothing
here computes anything about the scene.
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

VERSION = 1
FILE = "tickets.json"
KINDS = ("review", "non_conformity")
STATUSES = ("open", "resolved")
TITLE_MAX, TEXT_MAX = 200, 4000
_lock = threading.Lock()


class TicketError(ValueError):
    """What was asked cannot be a ticket — the message says why."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(project_dir: Path | str) -> Dict[str, Any]:
    """The project's ticket document ({version, next_number, tickets}); an empty one when it has none."""
    path = Path(project_dir) / FILE
    if not path.exists():
        return {"version": VERSION, "next_number": 1, "tickets": []}
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict) or doc.get("version") != VERSION or not isinstance(doc.get("tickets"), list):
        raise TicketError(f"{path}: not a ticket document of version {VERSION}")
    return doc


def _save(project_dir: Path | str, doc: Dict[str, Any]) -> None:
    path = Path(project_dir) / FILE
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    os.replace(part, path)


def _text(value: Any, name: str, limit: int, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise TicketError(f"{name} must be text")
    value = value.strip()
    if required and not value:
        raise TicketError(f"a ticket needs a {name}")
    if len(value) > limit:
        raise TicketError(f"{name} is longer than {limit} characters")
    return value


def _position(value: Any) -> List[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise TicketError("position must be three finite numbers (x, y, z in metres, the scan's display frame)")
    return [float(v) for v in value]


def _kind(value: Any) -> str:
    if value not in KINDS:
        raise TicketError(f"kind must be one of {list(KINDS)}")
    return value


def create(project_dir: Path | str, *, kind: str, title: str, position: Any, author: str, description: str = "",
           scan: Optional[str] = None, geometry_epoch: Optional[int] = None,
           instance_id: Optional[int] = None, label: Optional[str] = None) -> Dict[str, Any]:
    """Mark a ticket. Returns it (with its number: the project's next)."""
    ticket = {
        "id": secrets.token_hex(6),
        "number": 0,
        "kind": _kind(kind),
        "status": "open",
        "title": _text(title, "title", TITLE_MAX, required=True),
        "description": _text(description, "description", TEXT_MAX),
        "position": _position(position),
        "scan": scan or None,
        "geometry_epoch": None if geometry_epoch is None else int(geometry_epoch),
        "object": None,
        "provenance": "human_stated",
        "created_by": _text(author, "author", TITLE_MAX, required=True),
        "created_at": _now(),
        "updated_at": None,
        "resolved_by": None,
        "resolved_at": None,
    }
    if instance_id is not None:
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
            raise TicketError("instance_id must be the id of a segmented object")
        ticket["object"] = {"instance_id": instance_id, "label": _text(label, "label", TITLE_MAX)}
    with _lock:
        doc = load(project_dir)
        ticket["number"] = int(doc["next_number"])
        doc["next_number"] = ticket["number"] + 1
        doc["tickets"].append(ticket)
        _save(project_dir, doc)
    return ticket


def _find(doc: Dict[str, Any], ticket_id: str) -> Dict[str, Any]:
    for t in doc["tickets"]:
        if t["id"] == ticket_id or str(t["number"]) == str(ticket_id):
            return t
    raise KeyError(ticket_id)


def update(project_dir: Path | str, ticket_id: str, changes: Dict[str, Any], author: str) -> Dict[str, Any]:
    """Change what a ticket states — title, description, kind — or its status (resolving it records
    who and when; reopening clears that). Anything else in ``changes`` is refused."""
    unknown = set(changes) - {"title", "description", "kind", "status"}
    if unknown:
        raise TicketError(f"a ticket's {sorted(unknown)} cannot be changed")
    with _lock:
        doc = load(project_dir)
        t = _find(doc, ticket_id)
        if "title" in changes:
            t["title"] = _text(changes["title"], "title", TITLE_MAX, required=True)
        if "description" in changes:
            t["description"] = _text(changes["description"], "description", TEXT_MAX)
        if "kind" in changes:
            t["kind"] = _kind(changes["kind"])
        if "status" in changes:
            if changes["status"] not in STATUSES:
                raise TicketError(f"status must be one of {list(STATUSES)}")
            if changes["status"] != t["status"]:
                t["status"] = changes["status"]
                resolved = t["status"] == "resolved"
                t["resolved_by"] = _text(author, "author", TITLE_MAX, required=True) if resolved else None
                t["resolved_at"] = _now() if resolved else None
        t["updated_at"] = _now()
        _save(project_dir, doc)
    return t


def delete(project_dir: Path | str, ticket_id: str) -> Dict[str, Any]:
    """Remove a ticket (its number is not given again). Returns what was removed."""
    with _lock:
        doc = load(project_dir)
        t = _find(doc, ticket_id)
        doc["tickets"].remove(t)
        _save(project_dir, doc)
    return t


def listing(project_dir: Path | str, epochs: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """The project's tickets, each with ``stale`` — true when its scan's geometry is no longer the
    one it was marked on (``epochs``: {scan key: current epoch}; a scan not in it is not judged) —
    and the counts."""
    doc = load(project_dir)
    out = []
    for t in doc["tickets"]:
        now = (epochs or {}).get(t.get("scan") or "")
        out.append({**t, "stale": now is not None and t.get("geometry_epoch") is not None and int(now) != int(t["geometry_epoch"])})
    counts = {"open": sum(1 for t in out if t["status"] == "open"), "resolved": sum(1 for t in out if t["status"] == "resolved"),
              **{k: sum(1 for t in out if t["kind"] == k and t["status"] == "open") for k in KINDS}}
    return {"tickets": out, "counts": counts}
