"""THE TICKETS' ROUTES (tickets.py: a point of the model marked as a review item or a
non-conformity — USER 2026-11-09: "una herramienta simple … para mostrar el concepto nomás").

    GET    /api/tickets/{session_id}                 the project's tickets (each with ``stale``) and their counts
    POST   /api/tickets/{session_id}                 {kind, title, description?, position, scan?, instance_id?, label?}
    PATCH  /api/tickets/{session_id}/{ticket}        {title?, description?, kind?, status?} — ticket: its id or its number
    DELETE /api/tickets/{session_id}/{ticket}

Anyone who can open the session reads them. Marking, changing and resolving one needs a signed-in
user (a ticket says who marked it and who resolved it); removing one is its author's, a manager's
or an administrator's. Wired by main.py through :func:`configure` (the session resolver and the
audit log are the server's; this module never imports main).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import tickets as T

router = APIRouter(prefix="/api/tickets", tags=["tickets"])
_security = HTTPBearer(auto_error=False)
_resolve_ctx: Optional[Callable] = None
_audit_log: Optional[Callable] = None


def configure(resolve_ctx: Callable, audit_log: Optional[Callable] = None) -> None:
    global _resolve_ctx, _audit_log
    _resolve_ctx, _audit_log = resolve_ctx, audit_log


def _user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)) -> Optional[Dict[str, Any]]:
    if not credentials:
        return None
    from auth import decode_token
    return decode_token(credentials.credentials)


def _signed_in(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if user is None:
        raise HTTPException(401, "Sign in to mark or change a ticket: it records who did")
    return user


def _name(user: Dict[str, Any]) -> str:
    return str(user.get("username") or user.get("sub") or "unknown")


def _ctx(session_id: str, scan: Optional[str] = None):
    if _resolve_ctx is None:
        raise HTTPException(500, "tickets router not configured")
    try:
        ctx = _resolve_ctx(session_id, scan) if scan else _resolve_ctx(session_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    if not Path(ctx.session_dir).is_dir():
        # (the resolver returns a to-be-created context for an unknown id: nothing is created here)
        raise HTTPException(404, f"session {session_id}{' (' + scan + ')' if scan else ''} does not exist")
    return ctx


def _project_dir(ctx) -> Path:
    """Where the tickets live: the project's folder (a session of the old layout: its own)."""
    return Path(getattr(ctx, "project_dir", None) or ctx.session_dir)


def _epoch(session_id: str, scan: Optional[str]) -> Optional[int]:
    """The geometry epoch of a scan now — None when it cannot be told (no such scan any more)."""
    from correction.epoch import current_epoch
    try:
        return current_epoch(_ctx(session_id, scan).output_dir)
    except Exception:  # noqa: BLE001 — a ticket of a scan that is gone is listed, not judged
        return None


def _listing(session_id: str) -> Dict[str, Any]:
    project_dir = _project_dir(_ctx(session_id))
    try:
        scans = {t.get("scan") for t in T.load(project_dir)["tickets"]}
        epochs = {s or "": e for s in scans for e in [_epoch(session_id, s)] if e is not None}
        return T.listing(project_dir, epochs)
    except T.TicketError as e:
        raise HTTPException(500, str(e)) from e


@router.get("/{session_id}")
def tickets_list(session_id: str):
    return {"ok": True, **_listing(session_id)}


@router.post("/{session_id}")
def tickets_create(session_id: str, body: Dict[str, Any], user: Optional[Dict[str, Any]] = Depends(_user)):
    user = _signed_in(user)
    scan = body.get("scan") or None
    ctx = _ctx(session_id, scan)
    try:
        ticket = T.create(_project_dir(ctx), kind=body.get("kind"), title=body.get("title"), position=body.get("position"),
                          author=_name(user), description=body.get("description") or "", scan=scan,
                          geometry_epoch=_epoch(session_id, scan), instance_id=body.get("instance_id"), label=body.get("label"))
    except T.TicketError as e:
        raise HTTPException(400, str(e)) from e
    if _audit_log is not None:
        _audit_log("ticket.create", session_id, user=_name(user), role=user.get("role", ""),
                   detail=f"#{ticket['number']} {ticket['kind']} scan={scan or ''}")
    return {"ok": True, "ticket": {**ticket, "stale": False}, "counts": _listing(session_id)["counts"]}


@router.patch("/{session_id}/{ticket}")
def tickets_update(session_id: str, ticket: str, body: Dict[str, Any], user: Optional[Dict[str, Any]] = Depends(_user)):
    user = _signed_in(user)
    try:
        changed = T.update(_project_dir(_ctx(session_id)), ticket, body, _name(user))
    except KeyError:
        raise HTTPException(404, f"no ticket {ticket} in {session_id}") from None
    except T.TicketError as e:
        raise HTTPException(400, str(e)) from e
    if _audit_log is not None:
        _audit_log("ticket.update", session_id, user=_name(user), role=user.get("role", ""),
                   detail=f"#{changed['number']} {sorted(body)} status={changed['status']}")
    listing = _listing(session_id)
    return {"ok": True, "ticket": next(t for t in listing["tickets"] if t["id"] == changed["id"]), "counts": listing["counts"]}


@router.delete("/{session_id}/{ticket}")
def tickets_delete(session_id: str, ticket: str, user: Optional[Dict[str, Any]] = Depends(_user)):
    user = _signed_in(user)
    project_dir = _project_dir(_ctx(session_id))
    try:
        found = T._find(T.load(project_dir), ticket)
        if user.get("role") not in ("admin", "manager") and found["created_by"] != _name(user):
            raise HTTPException(403, "A ticket is removed by who marked it, a manager or an administrator")
        removed = T.delete(project_dir, found["id"])
    except KeyError:
        raise HTTPException(404, f"no ticket {ticket} in {session_id}") from None
    except T.TicketError as e:
        raise HTTPException(400, str(e)) from e
    if _audit_log is not None:
        _audit_log("ticket.delete", session_id, user=_name(user), role=user.get("role", ""), detail=f"#{removed['number']}")
    return {"ok": True, "removed": removed["id"], "counts": _listing(session_id)["counts"]}
