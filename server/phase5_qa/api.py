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

from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["spatial_qa"])


def _resolve_store(body: dict) -> str | None:
    sp = body.get("store_path")
    if sp and Path(sp).exists():
        return sp
    # session convention: <PROJECTS_DIR>/<session>/.../output/scene_r.db
    session_id = body.get("session_id")
    if session_id:
        try:
            from config import PROJECTS_DIR
            for cand in Path(PROJECTS_DIR).glob(f"**/{session_id}/**/scene_r.db"):
                return str(cand)
            for cand in Path(PROJECTS_DIR).glob("**/output/scene_r.db"):
                if session_id in str(cand):
                    return str(cand)
            # Segmented session but no store (segmented before scene_r.db
            # existed, or the store was wiped by a recon re-run while the
            # result survived) → rebuild it from segmentation_result.json so
            # the chat gets its measurement tools back (2026-08-28: text-only
            # answers with no 3D interaction traced back to exactly this).
            for res in Path(PROJECTS_DIR).glob(
                    f"**/{session_id}/**/segmentation_result.json"):
                print(f"[SpatialQA] no scene_r.db for {session_id} — rebuilding "
                      f"from {res}")
                from segmentation.pipeline import rebuild_instance_store
                if rebuild_instance_store(res.parent):
                    return str(res.parent / "scene_r.db")
                break
        except Exception as e:  # noqa: BLE001
            print(f"[SpatialQA] store resolve/rebuild failed: {e}")
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


@router.get("/semantic/status")
async def semantic_status(warmup: bool = False):
    """State of the Qwen3-VL service, so the chat can say "loading…" the moment it
    opens instead of only after the user has waited on a question.

    `warmup=true` also starts it — but never while a pipeline is running: vLLM rests
    at ~40 GB of VRAM and the reconstruction stages stop it precisely to get the GPU
    back. Warming up mid-reconstruction would OOM the very run the user is waiting on.
    """
    from semantic.service import is_alive, is_starting, ensure_service

    if is_alive():
        return JSONResponse({"status": "up"})
    if is_starting():
        return JSONResponse({"status": "loading"})

    if warmup:
        try:
            from main import pipeline_manager           # late import: avoids a cycle
            busy = any(j.get("status") in ("running", "queued")
                       for j in pipeline_manager.get_all_jobs().values())
        except Exception:  # noqa: BLE001
            busy = False
        if busy:
            return JSONResponse({"status": "busy",
                                 "detail": "a pipeline is using the GPU"})
        ensure_service(log=print, timeout_s=0.0)        # fire and forget
        return JSONResponse({"status": "loading"})

    return JSONResponse({"status": "down"})


@router.post("/spatial_qa")
async def spatial_qa(body: dict):
    question = (body or {}).get("question")
    if not question:
        return JSONResponse({"error": "missing 'question'"}, status_code=400)
    # The reconstruction / SAM3 stages stop vLLM to get the whole GPU, so by the
    # time the user opens the chat the service is usually down. Kick the launcher
    # and return immediately with status=loading: the weights take minutes, and
    # holding the request open just turns into a timeout the user reads as an
    # error. The client keeps its pending bubble and retries.
    from semantic.service import ensure_service, is_alive
    if not is_alive():
        ensure_service(log=print, timeout_s=0.0)   # fire and forget
        return JSONResponse(
            {"status": "loading",
             "error": "Loading the model (Qwen3-VL). It was unloaded to free the "
                      "GPU for reconstruction; this takes a couple of minutes."},
            status_code=503)

    backend = body.get("backend", "qwen_local")
    store_path = _resolve_store(body)
    if not store_path:
        # No instance store — the session was never segmented (or no session is
        # loaded). Still answer (user 2026-08-28): general chat, no tools. The
        # provenance rule holds because the model is told it CANNOT measure and
        # must not invent figures.
        return await _general_chat(question, body, backend)

    # Run the WHOLE tool loop in a thread: it blocks for the full multi-call
    # vLLM conversation (30–120 s) and running it inline froze the event loop —
    # /health timed out and the entire UI hung while the chat "thought"
    # (verified live 2026-08-29: backend /health 000 with a question in flight).
    import asyncio as _aio

    def _run():
        from phase5_qa.orchestrator import SpatialQA
        qa = SpatialQA(store_path, backend=backend)
        try:
            return qa.ask(question,
                          images=_resolve_images(body, store_path) or None,
                          max_iterations=body.get("max_iterations", 8))
        finally:
            qa.close()

    res = await _aio.get_running_loop().run_in_executor(None, _run)
    return JSONResponse(res)


_GENERAL_SYSTEM = (
    "You are the assistant of STAC-Build, a construction-site 3D reconstruction "
    "and supervision tool (video scan → point cloud → segmentation → measured "
    "spatial Q&A). The CURRENT session has NO segmentation / instance store yet, "
    "so the deterministic measurement tools are unavailable. Answer general "
    "questions helpfully (construction, reconstruction workflow, how to use the "
    "tool, anything you know). If asked for scene-specific measurements or "
    "object counts, explain that the session must be segmented first (Segment "
    "button / Segmentation Manager) so every figure can be tool-measured — "
    "NEVER estimate or invent measurements. Answer in the SAME LANGUAGE as the "
    "question."
)


async def _general_chat(question: str, body: dict, backend: str) -> JSONResponse:
    """Tool-less fallback chat for sessions without a Phase R store."""
    import time

    from semantic.client import get_semantic_client
    from semantic.types import system, user

    t0 = time.time()
    try:
        client = get_semantic_client(backend=backend, consumer="phase5.qa.general")
        images = list(body.get("images") or [])
        import asyncio as _aio
        # thread, not inline: a vLLM completion takes tens of seconds and must
        # not freeze the event loop (same fix as the orchestrated path)
        resp = await _aio.get_running_loop().run_in_executor(
            None, lambda: client.chat([system(_GENERAL_SYSTEM),
                                       user(question, images=images or None)]))
        return JSONResponse({
            "question": question,
            "answer": resp.content or "(no answer)",
            "tool_trace": [],
            "iterations": 1,
            "stopped": "general_chat_no_store",
            "latency_s": round(time.time() - t0, 2),
        })
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"general chat failed: {e}"},
                            status_code=500)


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
        out = []
        for i in store.list_instances():
            iid = i["instance_id"]
            obb = tools._obb(iid)
            entry = {"id": iid, "label": i["label"], "status": i["status"],
                     "n_views": i.get("n_views")}
            c = store.get_classification(iid)
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
    store, err = _store_or_error(body)
    if err:
        return err
    try:
        from phase5_qa.tools import SpatialTools
        t = SpatialTools(store)
        kwargs = {k: body[k] for k in ("volume_id", "center", "size", "yaw_deg", "voxel_m",
                                       "item_size") if k in body}
        return JSONResponse(getattr(t, method)(**kwargs))
    finally:
        store.close()


@router.post("/scene/volumes/evaluate")
async def volumes_evaluate(body: dict):
    return _volume_tool(body, "evaluate_volume")


@router.post("/scene/volumes/fits")
async def volumes_fits(body: dict):
    return _volume_tool(body, "fits_in_volume")
