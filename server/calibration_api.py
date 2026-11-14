"""REST of a scan's calibration measurements taken on the 3D model (calibration3d.py).

    GET    /api/calibration/{session_id}?scan_key=                  the document + its summary
    POST   /api/calibration/{session_id}/measurements               add one (two picked points
                                                                    + the real length)
    PATCH  /api/calibration/{session_id}/measurements/{mid}         what the user states about it
    DELETE /api/calibration/{session_id}/measurements/{mid}?scan_key=

Wired by ``main.py`` through :func:`configure` (the session resolver and the audit log are the
server's; this module never imports main), as the evaluation router is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

import calibration3d as C

router = APIRouter(prefix="/api/calibration", tags=["calibration"])
_resolve_ctx: Optional[Callable] = None
_audit_log: Optional[Callable] = None


def configure(resolve_ctx: Callable, audit_log: Optional[Callable] = None) -> None:
    global _resolve_ctx, _audit_log
    _resolve_ctx = resolve_ctx
    _audit_log = audit_log


def _output_dir(session_id: str, scan_key: Optional[str]) -> Path:
    if _resolve_ctx is None:
        raise HTTPException(500, "calibration router not configured")
    try:
        ctx = _resolve_ctx(session_id, scan_key) if scan_key else _resolve_ctx(session_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    if not Path(ctx.session_dir).is_dir():
        # the resolver returns a to-be-created context for an unknown id: never create a scan here
        raise HTTPException(404, f"scan {session_id} ({scan_key or 'active'}) does not exist")
    return Path(ctx.output_dir)


def _epoch(out: Path) -> int:
    from correction.epoch import current_epoch
    return int(current_epoch(out))


def _answer(out: Path, doc: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    return {"ok": True, "path": str(C.path_of(out)), "document": doc,
            "summary": C.summary(doc, _epoch(out)), **extra}


def _load(out: Path) -> Dict[str, Any]:
    try:
        return C.load(out)
    except C.CalibrationError as e:
        raise HTTPException(409, detail={"error": str(e)}) from e


def _log(action: str, session_id: str, detail: Dict[str, Any]) -> None:
    if _audit_log is not None:
        _audit_log(action, session_id, detail=json.dumps(detail))


@router.get("/{session_id}")
def get_calibration(session_id: str, scan_key: Optional[str] = Query(default=None)):
    out = _output_dir(session_id, scan_key)
    return _answer(out, _load(out))


@router.post("/{session_id}/measurements")
def add_measurement(session_id: str, body: Dict[str, Any]):
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        m = C.add(doc, body, _epoch(out))
        C.save(out, doc)
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.add", session_id, {"id": m["id"], "role": m["role"], "value_m": m["value_m"],
                                         "measured_m": m["measured_m"]})
    return _answer(out, doc, measurement=m)


@router.patch("/{session_id}/measurements/{mid}")
def update_measurement(session_id: str, mid: str, body: Dict[str, Any]):
    scan_key = body.get("scan_key") or body.get("scan") or None
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        m = C.update(doc, mid, body)
        C.save(out, doc)
    except KeyError:
        raise HTTPException(404, f"no calibration measurement {mid!r}")
    except C.CalibrationError as e:
        raise HTTPException(400, detail={"error": str(e)}) from e
    _log("calibration.update", session_id, {"id": mid})
    return _answer(out, doc, measurement=m)


@router.delete("/{session_id}/measurements/{mid}")
def delete_measurement(session_id: str, mid: str, scan_key: Optional[str] = Query(default=None)):
    out = _output_dir(session_id, scan_key)
    doc = _load(out)
    try:
        C.remove(doc, mid)
        C.save(out, doc)
    except KeyError:
        raise HTTPException(404, f"no calibration measurement {mid!r}")
    _log("calibration.delete", session_id, {"id": mid})
    return _answer(out, doc)
