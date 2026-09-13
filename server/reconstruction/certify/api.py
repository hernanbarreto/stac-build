"""HTTP surface of the certification loop (§9) and its acta (§10) — what the
visual validation kit (§11) calls. Mirrors correction/api.py: a per-session
lock (409 with the blocking task id), the work in an executor thread, the
operator from the bearer token, Approve/Undo through the correction
package (a certification leaves a CHAIN of pending epochs: Undo pops the
last one, Approve accepts them all).

    POST /api/certify/run        {session_id, max_iters?}      → acta
    GET  /api/certify/state/{session_id}                        → epoch, pending chain, acta summary
    GET  /api/certify/acta/{session_id}                         → the acta
    GET  /api/certify/report/{session_id}[?epoch=N]             → quality report (§10)
    GET  /api/certify/attention/{session_id}                    → §11 attention list
    POST /api/certify/approve|undo  {session_id}                → verdict (correction.run)
    POST /api/certify/witnesses  {session_id}                   → witness epoch only
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter(prefix="/api/certify", tags=["certify"])
_security = HTTPBearer(auto_error=False)
_resolve_ctx: Optional[Callable] = None
_notify_viewer: Optional[Callable] = None
_locks: Dict[str, str] = {}
_locks_guard = threading.Lock()


def configure(resolve_ctx: Callable, notify_viewer: Callable) -> None:
    global _resolve_ctx, _notify_viewer
    _resolve_ctx = resolve_ctx
    _notify_viewer = notify_viewer


def _ctx(session_id: str):
    if _resolve_ctx is None:
        raise HTTPException(500, "certify router not configured")
    return _resolve_ctx(session_id)


def _operator(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    if not credentials:
        return "unauthenticated"
    try:
        from auth import decode_token
        payload = decode_token(credentials.credentials)
        return str(payload.get("username") or payload.get("sub") or "unauthenticated")
    except Exception as e:  # noqa: BLE001 — an invalid token must not 500 a run; it IS recorded
        return f"invalid-token({e.__class__.__name__})"


def _acquire(session_id: str, task_id: str) -> None:
    with _locks_guard:
        holder = _locks.get(session_id)
        if holder is not None:
            raise HTTPException(409, detail={"error": "a certification is running on this session",
                                             "blocking_task_id": holder})
        _locks[session_id] = task_id


def _release(session_id: str) -> None:
    with _locks_guard:
        _locks.pop(session_id, None)


def _log(msg: str) -> None:
    print(f"[Certify] {msg}", flush=True)


def _session_dir(ctx) -> Path:
    return Path(ctx.output_dir).parent


async def _run_locked(session_id: str, label: str, fn: Callable, output_dir=None):
    from task_manager import task_manager
    tid = task_manager.start(session_id, "certify", label)
    _acquire(session_id, tid)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, fn)
        task_manager.finish(tid)
    except HTTPException:
        task_manager.fail(tid, "conflict")
        raise
    except Exception as e:
        task_manager.fail(tid, str(e))
        raise HTTPException(500, f"{label} failed: {e}")
    finally:
        _release(session_id)
    if _notify_viewer is not None and output_dir is not None:
        # the same potree_ready broadcast the correction module sends (async)
        await _notify_viewer(session_id, Path(output_dir))
    return result


@router.post("/run")
async def run(body: dict, credentials: HTTPAuthorizationCredentials = Depends(_security)):
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    ctx = _ctx(session_id)
    operator = _operator(credentials)
    max_iters = body.get("max_iters")
    from reconstruction.certify.run import certify_session

    def _work():
        return certify_session(_session_dir(ctx), operator=operator, log=_log,
                               max_iters=(int(max_iters) if max_iters else None))

    acta = await _run_locked(session_id, "certification loop", _work, ctx.output_dir)
    return {"ok": True, "acta": _acta_summary(acta)}


@router.post("/witnesses")
async def witnesses(body: dict, credentials: HTTPAuthorizationCredentials = Depends(_security)):
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    ctx = _ctx(session_id)
    operator = _operator(credentials)
    from reconstruction.witness.run import run_witnesses

    def _work():
        return run_witnesses(ctx.output_dir, operator=operator, log=_log)

    rep = await _run_locked(session_id, "witness epoch", _work, ctx.output_dir)
    return {"ok": True, "report": rep}


def auto_run(session_id: str, loop=None) -> None:
    """The post-segmentation hook (certify.auto_after_segmentation): runs the
    loop as operator ``auto`` under the session lock (in an executor thread;
    ``loop`` is the event loop the viewer broadcast must run on); failures
    are logged, never swallowed into silence."""
    from task_manager import task_manager
    try:
        ctx = _ctx(session_id)
    except HTTPException as e:
        _log(f"auto certification skipped for {session_id}: {e.detail}")
        return
    tid = task_manager.start(session_id, "certify", "certification loop (auto)")
    try:
        _acquire(session_id, tid)
    except HTTPException:
        task_manager.fail(tid, "conflict")
        _log(f"auto certification skipped for {session_id}: another certification is running")
        return
    try:
        from reconstruction.certify.run import certify_session
        acta = certify_session(_session_dir(ctx), operator="auto", log=_log)
        task_manager.finish(tid)
        _log(f"auto certification of {session_id}: {acta.get('stop_reason')} (epoch {acta.get('epoch_final')})")
        if _notify_viewer is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(_notify_viewer(session_id, Path(ctx.output_dir)), loop)
    except Exception as e:  # noqa: BLE001 — declared in the task + log
        task_manager.fail(tid, str(e))
        _log(f"auto certification of {session_id} FAILED: {e}")
    finally:
        _release(session_id)


def _acta_summary(acta: dict) -> dict:
    return {k: acta.get(k) for k in ("started_at", "operator", "max_iters", "stopped_at", "stop_reason",
                                     "epoch_initial", "epoch_final", "elapsed_s")} | {
        "iterations": [{k: it.get(k) for k in ("iteration", "verdict", "reason", "epoch_from", "epoch_to",
                                                "objective", "objective_prev", "improvement", "gates")}
                       for it in acta.get("iterations", [])],
        "metrics_initial": acta.get("metrics_initial"), "metrics_final": acta.get("metrics_final")}


@router.get("/state/{session_id}")
async def state(session_id: str):
    ctx = _ctx(session_id)
    out = Path(ctx.output_dir)
    from correction.epoch import current_epoch
    from correction.apply import pending_prev_dirs
    from correction import ledger
    from reconstruction.certify.run import ACTA_JSON
    from reconstruction.witness.fields import WITNESS_FIELDS
    acta_p = out / ACTA_JSON
    acta = json.loads(acta_p.read_text()) if acta_p.exists() else None
    has_fields = False
    ply = out / "cleaned_cloud.ply"
    if ply.exists():
        with open(ply, "rb") as f:
            head = f.read(4096).decode("ascii", "ignore")
        has_fields = all(f"property uchar {n}" in head for n in WITNESS_FIELDS)
    with _locks_guard:
        running = _locks.get(session_id)
    return {"epoch": current_epoch(out), "pending_epochs": len(pending_prev_dirs(out)),
            "pending_runs": [{"correction_id": r["correction_id"], "kind": r["kind"],
                              "epoch_to": r["epoch_to"], "operator": r["operator"]}
                             for r in ledger.pending_runs(out)],
            "witness_fields": has_fields, "running_task": running,
            "acta": _acta_summary(acta) if acta else None}


@router.get("/acta/{session_id}")
async def acta(session_id: str):
    ctx = _ctx(session_id)
    from reconstruction.certify.run import ACTA_JSON
    p = Path(ctx.output_dir) / ACTA_JSON
    if not p.exists():
        raise HTTPException(404, "no certification acta for this session yet")
    return json.loads(p.read_text())


@router.get("/report/{session_id}")
async def report(session_id: str, epoch: Optional[int] = None):
    ctx = _ctx(session_id)
    from reconstruction.quality.report import QUALITY_DIR, REPORT_JSON
    qdir = Path(ctx.output_dir) / QUALITY_DIR
    p = qdir / (f"report_epoch_{int(epoch)}.json" if epoch is not None else REPORT_JSON)
    if not p.exists():
        raise HTTPException(404, f"no quality report ({p.name}) for this session")
    rep = json.loads(p.read_text())
    rep["available_epochs"] = sorted(int(q.stem.split("_")[-1]) for q in qdir.glob("report_epoch_*.json"))
    return rep


@router.get("/attention/{session_id}")
async def attention(session_id: str):
    ctx = _ctx(session_id)
    from reconstruction.certify.attention import attention_list
    return attention_list(Path(ctx.output_dir))


@router.get("/edges/{session_id}")
async def edges(session_id: str):
    """§11 trajectory with edges + duplicates list (from the session's records)."""
    ctx = _ctx(session_id)
    from reconstruction.certify.kit import kit_edges
    return kit_edges(Path(ctx.output_dir))


@router.get("/epochs/{session_id}")
async def epochs(session_id: str):
    """§11 before/after: the epoch chain and which epochs carry an octree."""
    ctx = _ctx(session_id)
    from reconstruction.certify.kit import epoch_layers
    return epoch_layers(Path(ctx.output_dir))


@router.post("/approve")
async def approve(body: dict, credentials: HTTPAuthorizationCredentials = Depends(_security)):
    return await _verdict(body, "approved", credentials)


@router.post("/undo")
async def undo(body: dict, credentials: HTTPAuthorizationCredentials = Depends(_security)):
    return await _verdict(body, "undone", credentials)


async def _verdict(body: dict, verdict: str, credentials):
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    ctx = _ctx(session_id)
    operator = _operator(credentials)
    from correction.run import run_verdict

    def _work():
        return run_verdict(ctx.output_dir, verdict, operator, log=_log)

    return await _run_locked(session_id, f"epoch {verdict}", _work, ctx.output_dir)
