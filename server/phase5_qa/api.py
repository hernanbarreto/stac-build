# STAC-Builder — Phase 5: spatial Q&A + scene-geometry backend endpoints.
#
# Isolated router (auth-router precedent) included from main.py. Serves both the
# AI spatial-Q&A chat and the geometry the immersive viewer needs to render, pick
# and animate measurements / user-defined evaluation volumes.
#
#   POST /api/spatial_qa            {store_path?|session_id?, question, backend?}
#   POST /api/scene/objects         -> instances + OBB geometry (render/pick)
#   POST /api/scene/volumes/list    -> user-defined volumes
#   POST /api/scene/volumes/add     {name, center, size, yaw_deg?}
#   POST /api/scene/volumes/delete  {volume_id}
#   POST /api/scene/volumes/evaluate{volume_id?|center,size,yaw_deg?}
#   POST /api/scene/volumes/fits    {item_size, volume_id?|center,size}
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["spatial_qa"])


def _resolve_store(body: dict) -> str | None:
    """The instance store the question is about: the scan the VIEWER shows (``scan`` =
    "date/source"), else the project's active scan — through project_paths, as every other
    route. (Before 2026-11-13 this was the FIRST ``scene_r.db`` a glob found under the
    project: with pccr 2026-08-31 open the chat answered from pccr 2026-08-24's store —
    sign#797 "did not exist" and the counts were another scan's.)"""
    sp = body.get("store_path")
    if sp and Path(sp).exists():
        return sp
    session_id = body.get("session_id")
    if not session_id:
        return None
    try:
        import project_paths
        server_dir = Path(__file__).resolve().parent.parent          # main.SERVER_DIR
        ctx = project_paths.resolve_session(str(server_dir), session_id,
                                            scan_key=body.get("scan") or None)
        out_dir = Path(ctx.output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[SpatialQA] session {session_id!r} (scan {body.get('scan')!r}) not resolved: {e}")
        return None
    store = out_dir / "scene_r.db"
    if store.exists():
        return str(store)
    # Segmented scan but no store (segmented before scene_r.db existed, or the store was
    # wiped by a recon re-run while the result survived) → rebuild it from
    # segmentation_result.json so the chat gets its measurement tools back (2026-08-28:
    # text-only answers with no 3D interaction traced back to exactly this).
    res = out_dir / "segmentation_result.json"
    if res.exists():
        try:
            print(f"[SpatialQA] no scene_r.db for {session_id} — rebuilding from {res}")
            from segmentation.pipeline import rebuild_instance_store
            if rebuild_instance_store(out_dir):
                return str(store)
        except Exception as e:  # noqa: BLE001
            print(f"[SpatialQA] store rebuild failed: {e}")
    return None


def _resolve_images(body: dict, store_path: str) -> list:
    """Multimodal disambiguation over HTTP: accepts `images` (data-URLs /
    base64 strings from the UI) and/or `frame_ids` (session keyframes resolved
    to files next to the store: <session>/frames/<fid>.jpg)."""
    images: list = list(body.get("images") or [])
    fids = body.get("frame_ids") or []
    if fids:
        session_dir = Path(store_path).parent.parent  # output/scene_r.db -> session
        for fid in fids:
            for cand in (session_dir / "frames" / f"{int(fid):06d}.jpg",
                         session_dir / "frames" / f"{int(fid)}.jpg"):
                if cand.exists():
                    images.append(str(cand))
                    break
    return images


CHAT_OWNER = "chat"      # the chat's name at the engine lease: it never holds one


def _engine_busy() -> str | None:
    """Why the chat may not start or query vLLM now (docs/plan_determinismo.md points 81 /
    154): a pipeline job is running or queued (its stages take the GPU and the engine for
    themselves — vLLM rests at ~40 GB of VRAM, and a chat question mid-SAM3 used to start
    it on the card the stage holds), or SOMEONE holds the engine lease — asked by the chat's
    OWN owner, pid or no pid: the manager's job permit lives under the backend's own pid
    (a pid-only check let the chat through mid-stage, point 81), a stage worker's own vLLM
    (its answers are co-batched with nobody), a server GPU task (meshing, the Segmentation
    Manager's SAM3 — the chat never starts vLLM beside them, point 154).
    None when the chat may go ahead."""
    try:
        from main import pipeline_manager           # late import: avoids a cycle
        if any(j.get("status") in ("running", "queued")
               for j in pipeline_manager.get_all_jobs().values()):
            return "a pipeline is using the GPU — the chat waits for it to end"
    except Exception:  # noqa: BLE001 — no manager (a test, a CLI): nothing to wait for
        pass
    try:
        from semantic.service import engine_available_to
        ok, why = engine_available_to(owner=CHAT_OWNER)
    except Exception:  # noqa: BLE001
        return None
    return None if ok else why


def _engine_busy_response() -> JSONResponse | None:
    """The 503 'busy' answer when the engine is not the chat's right now — asked right
    BEFORE a call (point 81: a job can start while a question is being typed)."""
    busy = _engine_busy()
    if busy:
        return JSONResponse({"status": "busy", "error": busy}, status_code=503)
    return None


# ── the engine's first call ──────────────────────────────────────────────────────────────────
# A started engine answers its FIRST request of the chat's shape in ~9.5 s (the decode graphs are
# captured and the grammar of the tools' required call is compiled then) and every later one in
# ~0.8 s (measured 2026-11-13/14, semantic.chat.engine: single_sequence). The chat pays that once,
# while its panel says "loading" — never on the user's first question (USER 2026-11-13:
# "Determinista pero en 1 o 2 segundos ninguna es negociable").
_WARM: dict = {"engine": None, "state": "cold", "seconds": None}
_WARM_LOCK = threading.Lock()


def _engine_key():
    """Which engine is answering now (its launcher's pid and start): a restarted one is cold."""
    try:
        from semantic.service import service_identity
        proc = (service_identity() or {}).get("process") or {}
    except Exception:  # noqa: BLE001 — no record: an engine nobody can tell apart
        return None
    return (proc.get("pid"), proc.get("started_unix")) if proc.get("pid") is not None else None


def engine_warm_state() -> str:
    """'warm' | 'warming' | 'cold' for the engine answering now ('warm' too for an engine with
    no identity record: it cannot be told apart from a warmed one, and a question warms it)."""
    key = _engine_key()
    if key is None:
        return "warm"
    with _WARM_LOCK:
        if _WARM["engine"] != key:
            _WARM.update(engine=key, state="cold", seconds=None)
        return _WARM["state"]


def mark_engine_warm() -> None:
    """A question was answered by the engine: it is warm, whoever sent the first call."""
    key = _engine_key()
    if key is None:
        return
    with _WARM_LOCK:
        if _WARM["engine"] != key or _WARM["state"] != "warm":
            _WARM.update(engine=key, state="warm")


def warm_engine(backend: str = "qwen_local") -> None:
    """Send the engine ONE call of the chat's own shape (the tools' schemas, a required call) in
    a thread, unless it is warm or being warmed. Never beside a job (the caller checked)."""
    key = _engine_key()
    if key is None:
        return
    with _WARM_LOCK:
        if _WARM["engine"] == key and _WARM["state"] in ("warm", "warming"):
            return
        _WARM.update(engine=key, state="warming", seconds=None)

    def _call():
        import time as _time
        t0 = _time.time()
        state = "cold"
        try:
            from phase5_qa.tools import SpatialTools, chat_config
            from semantic.client import get_semantic_client
            from semantic.types import system, user
            import commands
            db = commands.database()
            client = get_semantic_client(backend=backend, consumer="phase5.qa.warmup")
            talk = db.tool_schemas(db.core())
            # the chat's two shapes: a scan open (the scene's tools, a long prompt) and none (a
            # short prompt, a few commands — its first call took 2.2 s when not warmed, 0.3 after)
            for tools in (SpatialTools.openai_schemas() + talk, talk):
                client.chat([system("Warm-up of the engine: call small_talk with the reply 'ok'."), user("hola")],
                            tools=tools, tool_choice="required",
                            max_tokens=chat_config()["max_tokens"], consumer="phase5.qa.warmup")
            state = "warm"
        except Exception as e:  # noqa: BLE001 — declared; the first question then pays the call
            print(f"[Chat] the engine's warm-up call failed: {e}")
        with _WARM_LOCK:
            if _WARM["engine"] == key:
                _WARM.update(state=state, seconds=round(_time.time() - t0, 2))
        if state == "warm":
            print(f"[Chat] the engine is warm (its first call took {_time.time() - t0:.1f} s)")

    threading.Thread(target=_call, name="chat-engine-warmup", daemon=True).start()


_WARM_WAITER: dict = {"thread": None}
_WARM_POLL_S = 3.0                       # between two looks at an engine that is loading


def warm_engine_when_up(timeout_s: float) -> None:
    """The backend just asked for the chat's engine (its start, the end of a pipeline): wait for it
    in ONE thread, bounded, and send its warm-up call when it answers. Ends without it when a job
    takes the engine meanwhile (that job's end asks again)."""
    with _WARM_LOCK:
        t = _WARM_WAITER["thread"]
        if t is not None and t.is_alive():
            return

        def _wait():
            import time as _time
            from semantic.service import is_alive
            deadline = _time.time() + float(timeout_s)
            while _time.time() < deadline:
                if _engine_busy():
                    return
                if is_alive():
                    warm_engine()
                    return
                _time.sleep(_WARM_POLL_S)
            print(f"[Chat] the engine did not answer within {timeout_s:.0f} s — not warmed")

        t = threading.Thread(target=_wait, name="chat-engine-warm-when-up", daemon=True)
        _WARM_WAITER["thread"] = t
        t.start()


@router.get("/semantic/status")
async def semantic_status(warmup: bool = False):
    """State of the Qwen3-VL service, so the chat can say "loading…" the moment it
    opens instead of only after the user has waited on a question.

    `warmup=true` also starts it — but never while a pipeline is running or a job holds
    the engine lease: vLLM rests at ~40 GB of VRAM and the reconstruction stages stop it
    precisely to get the GPU back. Warming up mid-reconstruction would OOM the very run
    the user is waiting on.
    """
    from semantic.service import is_alive, is_starting, ensure_service

    busy = _engine_busy()
    if busy:
        return JSONResponse({"status": "busy", "detail": busy})
    if is_alive():
        # up for the chat = alive AND past its first call (asked for when the chat warms it up)
        # (an engine that is alive is the chat's to use — no job holds it: the call is sent
        # whether or not this poll asked to START an engine)
        if engine_warm_state() == "warm":
            return JSONResponse({"status": "up"})
        warm_engine()
        return JSONResponse({"status": "loading", "detail": "warming the engine (its first call)"})
    if is_starting():
        return JSONResponse({"status": "loading"})

    if warmup:
        ensure_service(log=print, timeout_s=0.0, owner=CHAT_OWNER)   # fire and forget
        return JSONResponse({"status": "loading"})

    return JSONResponse({"status": "down"})


@router.post("/spatial_qa")
async def spatial_qa(body: dict):
    question = (body or {}).get("question")
    if not question:
        return JSONResponse({"error": "missing 'question'"}, status_code=400)
    # ONE permit to the engine per job (point 81): while a pipeline job runs, or a job's
    # process holds the engine lease, the chat neither starts vLLM nor queries it — it
    # gets 'busy' and the client retries when the stage ends.
    busy = _engine_busy_response()
    if busy is not None:
        return busy
    # The reconstruction / SAM3 stages stop vLLM to get the whole GPU, so by the
    # time the user opens the chat the service is usually down. Kick the launcher
    # and return immediately with status=loading: the weights take minutes, and
    # holding the request open just turns into a timeout the user reads as an
    # error. The client keeps its pending bubble and retries.
    from semantic.service import ensure_service, is_alive
    if not is_alive():
        ensure_service(log=print, timeout_s=0.0, owner=CHAT_OWNER)   # fire and forget
        return JSONResponse(
            {"status": "loading",
             "error": "Loading the model (Qwen3-VL). It was unloaded to free the "
                      "GPU for reconstruction; this takes a couple of minutes."},
            status_code=503)

    backend = body.get("backend", "qwen_local")
    # No instance store (no session open, or one never segmented): STAC still converses and
    # OPERATES THE APPLICATION (USER 2026-11-14: "le pedi que conecte con el servidor y no
    # conecto" / "le pedi que abra pccr y no abrio nada") — it only cannot measure, and is told so
    store_path = _resolve_store(body) or None

    # Run the WHOLE tool loop in a thread: it blocks for the full multi-call
    # vLLM conversation (30–120 s) and running it inline froze the event loop —
    # /health timed out and the entire UI hung while the chat "thought"
    # (verified live 2026-08-29: backend /health 000 with a question in flight).
    import asyncio as _aio

    class _Busy(Exception):
        pass

    import time as _time
    t_request = _time.time()

    def _run():
        # the semantic client is asked for right BEFORE the call (point 81): the store was
        # resolved meanwhile and a job may have taken the engine
        t_run = _time.time()
        why = _engine_busy()
        if why:
            raise _Busy(why)
        from phase5_qa.orchestrator import SpatialQA
        t_open = _time.time()
        qa = SpatialQA(store_path, backend=backend)
        try:
            t_ask = _time.time()
            out = qa.ask(question,
                         images=(_resolve_images(body, store_path) or None) if store_path else None,
                         max_iterations=body.get("max_iterations", 8),
                         history=body.get("history") if isinstance(body.get("history"), list) else None,
                         lang=body.get("lang") if isinstance(body.get("lang"), str) else None,
                         user_name=body.get("user_name") if isinstance(body.get("user_name"), str) else None,
                         ui=body.get("ui") if isinstance(body.get("ui"), dict) else None,
                         expecting=body.get("expecting") if isinstance(body.get("expecting"), str) else None)
            # where the request's time went outside the model (the whole answer is asked in
            # 1-2 s: every tenth of a second is accounted for)
            out["timing_s"] = {"before_thread": round(t_run - t_request, 3),
                               "engine_check": round(t_open - t_run, 3),
                               "open_store": round(t_ask - t_open, 3),
                               "ask": round(_time.time() - t_ask, 3)}
            return out
        finally:
            qa.close()

    try:
        res = await _aio.get_running_loop().run_in_executor(None, _run)
    except _Busy as e:
        return JSONResponse({"status": "busy", "error": str(e)}, status_code=503)
    mark_engine_warm()
    if isinstance(res.get("timing_s"), dict):
        res["timing_s"]["total"] = round(_time.time() - t_request, 3)
    return JSONResponse(res)


# ── scene geometry + user volumes (immersive viewer) ────────────────
def _store_or_error(body: dict):
    sp = _resolve_store(body or {})
    if not sp:
        return None, JSONResponse(
            {"error": "no instance store (pass store_path or session_id)"}, status_code=404)
    from phase_r.instance_store import InstanceStore
    return InstanceStore(sp), None


@router.post("/scene/objects")
async def scene_objects(body: dict):
    """Instances + OBB geometry so the viewer can render, pick and reference
    objects by id in the chat. Every field is tool_measured geometry, served in
    the DISPLAY frame (floor-aligned; the exact frame the viewer renders) via
    SpatialTools — the same frame every chat measurement uses."""
    store, err = _store_or_error(body)
    if err:
        return err
    try:
        from phase5_qa.tools import SpatialTools
        tools = SpatialTools(store)
        obbs = tools.all_obbs()                 # ONE query each (2 per object before:
        cls = store.list_classifications()      # seconds on pccr 2408's 8,777 objects)
        out = []
        for i in store.list_instances():
            iid = i["instance_id"]
            obb = obbs.get(int(iid))
            entry = {"id": iid, "label": i["label"], "status": i["status"],
                     "n_views": i.get("n_views")}
            c = cls.get(int(iid))
            if c:
                entry["class"] = c["class_final"]
                entry["material"] = c["material"]
                entry["state"] = c["state"]
            if obb is not None:
                T, aabb, pos = obb
                half = [float((aabb[1] - aabb[0]) / 2), float((aabb[3] - aabb[2]) / 2),
                        float((aabb[5] - aabb[4]) / 2)]
                entry["obb"] = {"transform": [float(x) for x in np.asarray(T).ravel()],
                                "center": [float(x) for x in pos], "half_extents": half}
            out.append(entry)
        return JSONResponse({"objects": out, "count": len(out), "frame": "display"})
    finally:
        store.close()


@router.post("/scene/volumes/list")
async def volumes_list(body: dict):
    store, err = _store_or_error(body)
    if err:
        return err
    try:
        return JSONResponse({"volumes": store.list_user_volumes()})
    finally:
        store.close()


@router.post("/scene/volumes/add")
async def volumes_add(body: dict):
    store, err = _store_or_error(body)
    if err:
        return err
    try:
        c, s = body.get("center"), body.get("size")
        if not (isinstance(c, list) and isinstance(s, list) and len(c) == 3 and len(s) == 3):
            return JSONResponse({"error": "center and size must be [x,y,z]"}, status_code=400)
        vid = store.add_user_volume(body.get("name") or "volume", c, s,
                                    float(body.get("yaw_deg", 0.0)))
        return JSONResponse(store.get_user_volume(vid))
    finally:
        store.close()


@router.post("/scene/volumes/update")
async def volumes_update(body: dict):
    """Gizmo edits from the viewer: move / rotate / resize a saved volume."""
    store, err = _store_or_error(body)
    if err:
        return err
    try:
        v = store.update_user_volume(
            int(body["volume_id"]),
            center=body.get("center"), size=body.get("size"),
            yaw_deg=body.get("yaw_deg"), name=body.get("name"))
        if v is None:
            return JSONResponse({"error": "unknown volume_id"}, status_code=404)
        return JSONResponse(v)
    finally:
        store.close()


@router.post("/scene/volumes/delete")
async def volumes_delete(body: dict):
    store, err = _store_or_error(body)
    if err:
        return err
    try:
        store.delete_user_volume(int(body["volume_id"]))
        return JSONResponse({"ok": True})
    finally:
        store.close()


def _volume_tool(body: dict, method: str):
    """One volume measurement for the viewer (a box dragged, the chat opened). With
    ``semantic.chat.tools_device: cuda`` it runs in the chat's tools process, which keeps the
    scene's points on the card — and answers 'busy' while a job has the GPU (the backend never
    measures beside a stage, and never falls back to the CPU by itself)."""
    sp = _resolve_store(body or {})
    if not sp:
        return JSONResponse(
            {"error": "no instance store (pass store_path or session_id)"}, status_code=404)
    from phase5_qa.tools import SpatialTools, effective_device
    kwargs = {k: body[k] for k in ("volume_id", "center", "size", "yaw_deg", "voxel_m",
                                   "item_size") if k in body}
    dev = effective_device()
    if dev == "cuda":
        busy = _engine_busy_response()
        if busy is not None:
            return busy
        from phase5_qa import tools_worker
        try:
            return JSONResponse(tools_worker.call(sp, method, kwargs, device="cuda"))
        except tools_worker.ToolsWorkerBusy as e:
            return JSONResponse({"status": "busy", "error": str(e)}, status_code=503)
        except tools_worker.ToolsWorkerError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    from phase_r.instance_store import InstanceStore
    store = InstanceStore(sp)
    try:
        t = SpatialTools(store, device=dev)
        try:
            return JSONResponse(getattr(t, method)(**kwargs))
        finally:
            t.release()
    finally:
        store.close()


@router.post("/scene/volumes/evaluate")
async def volumes_evaluate(body: dict):
    # in a thread: the measurement (and the tools process's start) must not hold the event loop
    import asyncio as _aio
    return await _aio.get_running_loop().run_in_executor(
        None, lambda: _volume_tool(body, "evaluate_volume"))


@router.post("/scene/volumes/fits")
async def volumes_fits(body: dict):
    import asyncio as _aio
    return await _aio.get_running_loop().run_in_executor(
        None, lambda: _volume_tool(body, "fits_in_volume"))


@router.get("/spatial_qa/tools")
async def spatial_qa_tools_status():
    """Is the chat's tools process up, on which device and store."""
    from phase5_qa import tools_worker
    from phase5_qa.tools import effective_device
    return JSONResponse({"device": effective_device(), **tools_worker.status()})


@router.post("/spatial_qa/tools/stop")
async def spatial_qa_tools_stop():
    """End the chat's tools process (its CUDA context leaves the card). The pipeline manager
    and the server's GPU tasks do this by themselves when they take the engine lease; this is
    for GPU work launched OUTSIDE them (a script in a shell), whose exclusive-GPU check would
    find the process on the card. The next measurement of the chat starts it again."""
    import asyncio as _aio
    from phase5_qa import tools_worker
    gone = await _aio.get_running_loop().run_in_executor(
        None, lambda: tools_worker.stop("asked through the API"))
    return JSONResponse({"stopped": bool(gone), **tools_worker.status()})


@router.post("/spatial_qa/warmup")
async def spatial_qa_warmup(body: dict):
    """The chat was opened on a scan: start its tools process (semantic.chat.tools_device:
    cuda) so the first measurement does not wait for it. Never beside a job: 'busy' then."""
    from phase5_qa.tools import effective_device
    dev = effective_device()
    if dev != "cuda":
        return JSONResponse({"status": "not_needed", "device": dev})
    if _engine_busy():
        return JSONResponse({"status": "busy"})
    sp = _resolve_store(body or {})
    if not sp:
        return JSONResponse({"status": "no_store"})
    import asyncio as _aio
    from phase5_qa import tools_worker

    def _start():
        try:
            return tools_worker.warm(sp, device="cuda")
        except Exception as e:  # noqa: BLE001 — declared; the first question will say why
            print(f"[ChatTools] warm-up failed: {e}")
            return False
    _aio.get_running_loop().run_in_executor(None, _start)          # fire and forget
    return JSONResponse({"status": "starting", "device": dev})
