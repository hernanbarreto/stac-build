# STAC-Builder — Phase 5: spatial Q&A backend endpoint (APIRouter).
#
# Isolated router (auth-router precedent) included from main.py. UI is out of
# scope for this task; this is the backend endpoint per repo convention.
#
#   POST /api/spatial_qa  {store_path? | session_id?, question, backend?}
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from pathlib import Path

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
        except Exception:
            pass
    return None


@router.post("/spatial_qa")
async def spatial_qa(body: dict):
    question = (body or {}).get("question")
    if not question:
        return JSONResponse({"error": "missing 'question'"}, status_code=400)
    store_path = _resolve_store(body)
    if not store_path:
        return JSONResponse(
            {"error": "no instance store found (pass store_path, or build the "
                      "Phase R store for this session first)"}, status_code=404)
    backend = body.get("backend", "qwen_local")
    from phase5_qa.orchestrator import SpatialQA
    qa = SpatialQA(store_path, backend=backend)
    try:
        res = qa.ask(question, max_iterations=body.get("max_iterations", 8))
    finally:
        qa.close()
    return JSONResponse(res)
