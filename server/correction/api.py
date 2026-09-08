"""FastAPI router of the correction module.

``main.py`` contains no correction logic: it calls ``configure()`` with its
session resolver and viewer-notify helper, then ``app.include_router(router)``.
Every mutating endpoint:
  * decodes the operator from the JWT the UI attaches (a missing token is
    recorded as the explicit literal "unauthenticated" — visible, never
    guessed);
  * takes the per-session correction LOCK — a second correction/floor/verdict
    while one runs gets 409 with the blocking task id (prompt §7);
  * runs in the default executor with task_manager progress
    (``task_type="correction"``) so the UI can poll ``/api/tasks/{sid}``.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter(prefix="/api/correction", tags=["correction"])
_security = HTTPBearer(auto_error=False)

_resolve_ctx: Optional[Callable] = None
_notify_viewer: Optional[Callable] = None

# per-session lock registry: session_id → task_id (in-memory, like the Potree
# build registry — dies with the process, can never go stale on disk)
_locks: Dict[str, str] = {}
_locks_guard = threading.Lock()


def configure(resolve_ctx: Callable, notify_viewer: Callable) -> None:
    global _resolve_ctx, _notify_viewer
    _resolve_ctx = resolve_ctx
    _notify_viewer = notify_viewer


def _ctx(session_id: str):
    if _resolve_ctx is None:
        raise HTTPException(500, "correction router not configured")
    return _resolve_ctx(session_id)


def _operator(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    if not credentials:
        return "unauthenticated"
    try:
        from auth import decode_token
        payload = decode_token(credentials.credentials)
        return str(payload.get("username") or payload.get("sub")
                   or "unauthenticated")
    except Exception as e:  # noqa: BLE001 — an invalid token must not 500 a run; it IS recorded
        return f"invalid-token({e.__class__.__name__})"


def _acquire(session_id: str, task_id: str) -> None:
    with _locks_guard:
        holder = _locks.get(session_id)
        if holder is not None:
            raise HTTPException(
                409, detail={"error": "another correction operation is "
                                      "running on this session",
                             "blocking_task_id": holder})
        _locks[session_id] = task_id


def _release(session_id: str) -> None:
    with _locks_guard:
        _locks.pop(session_id, None)


async def _run_locked(session_id: str, label: str, fn: Callable):
    """Common wrapper: task + lock + executor + viewer notify."""
    from task_manager import task_manager
    tid = task_manager.start(session_id, "correction", label)
    _acquire(session_id, tid)
    loop = asyncio.get_event_loop()

    def _progress(pct, msg):
        task_manager.update(tid, pct=pct, detail=msg)

    try:
        result = await loop.run_in_executor(None, lambda: fn(_progress))
        task_manager.finish(tid)
    except HTTPException:
        task_manager.fail(tid, "conflict")
        raise
    except Exception as e:
        task_manager.fail(tid, str(e))
        raise HTTPException(500, f"{label} failed: {e}")
    finally:
        _release(session_id)
    return result


def _log(msg: str) -> None:
    print(f"[Correction] {msg}", flush=True)


@router.post("/run")
async def correction_run(request: Request,
                         credentials: Optional[HTTPAuthorizationCredentials]
                         = Depends(_security)):
    body = await request.json()
    session_id = body.get("session_id")
    instance_ids = body.get("instance_ids")
    if not session_id or not instance_ids:
        raise HTTPException(400, "session_id and instance_ids required")
    override = bool(body.get("override_scale_check", False))
    operator = _operator(credentials)
    ctx = _ctx(session_id)
    from correction.run import run_objects

    report = await _run_locked(
        session_id, "Correction analysis",
        lambda progress: run_objects(
            ctx.output_dir, [int(i) for i in instance_ids], operator,
            override_scale_check=override, log=_log, progress=progress))
    if report.get("status") == "pending" and _notify_viewer:
        await _notify_viewer(session_id, ctx.output_dir)
    return {"ok": True, "status": report.get("status"), "report": report}


@router.post("/floor")
async def correction_floor(request: Request,
                           credentials: Optional[
                               HTTPAuthorizationCredentials]
                           = Depends(_security)):
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    model = body.get("model")           # None → config default
    keyframes = body.get("keyframes")   # None/"auto" → all qualifying
    if keyframes in ("auto", []):
        keyframes = None
    operator = _operator(credentials)
    ctx = _ctx(session_id)
    from correction.run import run_floor

    report = await _run_locked(
        session_id, f"Floor alignment ({model or 'default model'})",
        lambda progress: run_floor(
            ctx.output_dir, model,
            [int(k) for k in keyframes] if keyframes else None,
            operator, log=_log, progress=progress))
    if report.get("status") == "pending" and _notify_viewer:
        await _notify_viewer(session_id, ctx.output_dir)
    return {"ok": True, "status": report.get("status"), "report": report}


@router.post("/approve")
async def correction_approve(request: Request,
                             credentials: Optional[
                                 HTTPAuthorizationCredentials]
                             = Depends(_security)):
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    operator = _operator(credentials)
    ctx = _ctx(session_id)
    from correction.run import run_verdict

    res = await _run_locked(
        session_id, "Approve correction",
        lambda progress: run_verdict(ctx.output_dir, "approved", operator,
                                     log=_log))
    return {**res, "status": "approved"}


@router.post("/undo")
async def correction_undo(request: Request,
                          credentials: Optional[
                              HTTPAuthorizationCredentials]
                          = Depends(_security)):
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    operator = _operator(credentials)
    ctx = _ctx(session_id)
    from correction.run import run_verdict

    res = await _run_locked(
        session_id, "Undo correction",
        lambda progress: run_verdict(ctx.output_dir, "undone", operator,
                                     log=_log))
    if _notify_viewer:
        await _notify_viewer(session_id, ctx.output_dir)
    return {**res, "status": "none"}


@router.get("/state/{session_id}")
async def correction_state(session_id: str):
    ctx = _ctx(session_id)
    from correction.run import state
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None,
                                      lambda: state(ctx.output_dir))


@router.get("/ledger/{session_id}")
async def correction_ledger(session_id: str):
    ctx = _ctx(session_id)
    from correction.epoch import current_epoch
    from correction.ledger import ledger_view
    loop = asyncio.get_event_loop()
    entries = await loop.run_in_executor(
        None, lambda: ledger_view(ctx.output_dir))
    return {"epoch": current_epoch(ctx.output_dir), "entries": entries}


@router.get("/artifacts/{session_id}")
async def correction_artifacts(session_id: str):
    ctx = _ctx(session_id)
    from correction.epoch import current_epoch
    from correction.invalidate import derived_artifacts_status
    loop = asyncio.get_event_loop()
    rows = await loop.run_in_executor(
        None, lambda: derived_artifacts_status(ctx.output_dir))
    return {"epoch": current_epoch(ctx.output_dir), "artifacts": rows}
