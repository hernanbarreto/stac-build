"""THE GLOBE'S ROUTES (geo.py: one georeference per project — USER 2026-11-24).

    GET    /api/geo                    every project: its name, its scans, its reference scan, its georeference (or None)
    GET    /api/geo/{session_id}       the project's georeference (or None)
    PUT    /api/geo/{session_id}       {lat, lon, height_m, heading_deg?, source, accuracy_m?, note?} — signed in
    DELETE /api/geo/{session_id}       signed in

Anyone who can open the session reads it; setting or removing it needs a signed-in user (it says who placed the
work). Wired by main.py through :func:`configure` (the session resolver, the projects' folder and the audit log are
the server's; this module never imports main).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import geo as G

router = APIRouter(prefix="/api/geo", tags=["geo"])
_security = HTTPBearer(auto_error=False)
_resolve_ctx: Optional[Callable] = None
_projects_dir: Optional[Callable[[], Path]] = None
_audit_log: Optional[Callable] = None


def configure(resolve_ctx: Callable, projects_dir: Callable[[], Path], audit_log: Optional[Callable] = None) -> None:
    global _resolve_ctx, _projects_dir, _audit_log
    _resolve_ctx, _projects_dir, _audit_log = resolve_ctx, projects_dir, audit_log


def _user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> Optional[Dict[str, Any]]:
    if not credentials:
        return None
    from auth import decode_token
    return decode_token(credentials.credentials)


def _name(user: Optional[Dict[str, Any]]) -> str:
    if user is None:
        raise HTTPException(401, "Sign in to place a work on the globe: the place records who set it")
    return str(user.get("username") or user.get("sub") or "unknown")


def _project_dir(session_id: str) -> Path:
    if _resolve_ctx is None or _projects_dir is None:
        raise HTTPException(500, "geo router not configured")
    d = Path(_projects_dir()) / session_id
    if "/" in session_id or "\\" in session_id or session_id in (".", "..") or not (d / "project.json").is_file():
        raise HTTPException(404, f"project {session_id} does not exist")
    return d


def _project_row(d: Path) -> Dict[str, Any]:
    try:
        meta = json.loads((d / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    scans: List[str] = []
    scans_dir = d / "scans"
    if scans_dir.is_dir():
        for day in sorted(p for p in scans_dir.iterdir() if p.is_dir()):
            for src in sorted(p for p in day.iterdir() if p.is_dir() and p.name.startswith("src_")):
                scans.append(f"{day.name}/{src.name[4:]}")
    reference = (meta.get("composition") or {}).get("reference") or (scans[0] if scans else None)
    try:
        georef = G.load(d)
    except G.GeoError:
        georef = None
    # its name is the session's, the folder's, as everywhere in the app (a project.json's own "name" can be an old
    # one: zaragoza's says "test5")
    return {"session_id": d.name, "name": d.name, "scans": scans, "reference": reference, "georef": georef}


@router.get("")
def geo_projects():
    if _projects_dir is None:
        raise HTTPException(500, "geo router not configured")
    root = Path(_projects_dir())
    rows = [_project_row(d) for d in sorted(root.iterdir()) if (d / "project.json").is_file()] if root.is_dir() else []
    return {"ok": True, "projects": rows}


@router.get("/{session_id}")
def geo_get(session_id: str):
    return {"ok": True, "georef": G.load(_project_dir(session_id))}


@router.put("/{session_id}")
def geo_put(session_id: str, body: Dict[str, Any], user: Optional[Dict[str, Any]] = Depends(_user)):
    who = _name(user)
    d = _project_dir(session_id)
    scan = body.get("scan") or _project_row(d)["reference"]
    try:
        georef = G.save(d, G.validate(body, set_by=who, scan=scan))
    except G.GeoError as e:
        raise HTTPException(400, str(e)) from e
    if _audit_log is not None:
        _audit_log("geo.set", session_id, user=who, role=(user or {}).get("role", ""),
                   detail=f"{georef['lat']:.6f},{georef['lon']:.6f} heading={georef['heading_deg']} "
                          f"source={georef['source']}")
    return {"ok": True, "georef": georef}


@router.delete("/{session_id}")
def geo_delete(session_id: str, user: Optional[Dict[str, Any]] = Depends(_user)):
    who = _name(user)
    removed = G.remove(_project_dir(session_id))
    if removed and _audit_log is not None:
        _audit_log("geo.remove", session_id, user=who, role=(user or {}).get("role", ""), detail="")
    return {"ok": True, "removed": removed}
