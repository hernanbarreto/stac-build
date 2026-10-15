"""REST of the evaluation battery.

Phase 0: the reference measures of a scan (read / write, validated).
Phase 8: read-only listing of the evaluation runs and their files.

Wired by ``main.py`` through :func:`configure` (the session resolver and the audit log are the
server's; this module never imports main). Prefix ``/api/evaluation``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from evaluation import references as R
from evaluation.config import EvaluationConfig, load_evaluation_config, runs_root

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
_resolve_ctx: Optional[Callable] = None
_audit_log: Optional[Callable] = None
_cfg: Optional[EvaluationConfig] = None

# file name prefix → phase label of the battery (the panel groups by it)
PHASE_OF_PREFIX = (
    ("dimensions", "error"), ("curve_", "curves"), ("plane_noise", "curves"),
    ("resolution", "curves"), ("growth", "curves"), ("confidence_distance", "curves"),
    ("coverage", "variants"), ("comparison", "variants"), ("cost", "cost"),
    ("confidence", "confidence"), ("reliability", "confidence"),
    ("repeatability", "repeatability"), ("manifest", "repeatability"),
    ("pareto", "frontier"), ("profiles", "frontier"), ("variant", "run"), ("log", "run"),
)
KIND_OF_SUFFIX = {".csv": "csv", ".json": "json", ".png": "png", ".txt": "log", ".log": "log",
                  ".yaml": "log"}
MEDIA_OF_KIND = {"csv": "text/csv", "json": "application/json", "png": "image/png",
                 "log": "text/plain; charset=utf-8", "other": "application/octet-stream"}


def configure(resolve_ctx: Callable, audit_log: Optional[Callable] = None,
              cfg: Optional[EvaluationConfig] = None) -> None:
    global _resolve_ctx, _audit_log, _cfg
    _resolve_ctx = resolve_ctx
    _audit_log = audit_log
    _cfg = cfg


def _config() -> EvaluationConfig:
    global _cfg
    if _cfg is None:
        _cfg = load_evaluation_config()
    return _cfg


def _ctx(session_id: str, scan_key: Optional[str]):
    if _resolve_ctx is None:
        raise HTTPException(500, "evaluation router not configured")
    try:
        return _resolve_ctx(session_id, scan_key) if scan_key else _resolve_ctx(session_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


# ── phase 0: references ──────────────────────────────────────────────────

class ReferencesBody(BaseModel):
    scan_key: Optional[str] = None
    document: Dict[str, Any]


@router.get("/references/{session_id}")
def get_references(session_id: str, scan_key: Optional[str] = Query(default=None)):
    ctx = _ctx(session_id, scan_key)
    cfg = _config()
    try:
        doc = R.load_references(Path(ctx.output_dir), cfg.references.file)
    except R.ReferencesError as e:
        raise HTTPException(409, detail={"error": str(e)}) from e
    return {"session_id": session_id, "scan_key": scan_key,
            "path": str(R.references_path(Path(ctx.output_dir), cfg.references.file)),
            "document": doc, "counts": R.counts(doc), "warnings": R.role_warnings(doc)}


@router.post("/references/{session_id}")
def post_references(session_id: str, body: ReferencesBody):
    ctx = _ctx(session_id, body.scan_key)
    if not Path(ctx.session_dir).is_dir():
        # the resolver returns a to-be-created context for an unknown id: never create a scan here
        raise HTTPException(404, f"scan {session_id} ({body.scan_key or 'active'}) does not exist")
    cfg = _config()
    problems = R.validate_document(body.document)
    if problems:
        raise HTTPException(400, detail={"error": "invalid reference document",
                                         "problems": problems})
    p = R.save_references(Path(ctx.output_dir), cfg.references.file, body.document)
    c = R.counts(body.document)
    if _audit_log is not None:
        _audit_log("evaluation.references", session_id, detail=json.dumps(c))
    return {"ok": True, "path": str(p), "counts": c, "warnings": R.role_warnings(body.document)}


# ── phase 8: runs (read-only) ────────────────────────────────────────────

def _phase_of(name: str) -> Optional[str]:
    for prefix, phase in PHASE_OF_PREFIX:
        if name.startswith(prefix):
            return phase
    return None


def _run_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def list_run_files(run_dir: Path, manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scenes = set((manifest or {}).get("scenes") or [])
    files = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir)
        parts = rel.parts
        scene = parts[0] if len(parts) >= 2 and (not scenes or parts[0] in scenes) else None
        variant = parts[1] if scene is not None and len(parts) >= 3 else None
        files.append({"path": rel.as_posix(), "kind": KIND_OF_SUFFIX.get(p.suffix.lower(), "other"),
                      "scene": scene, "variant": variant, "phase": _phase_of(p.name),
                      "size": p.stat().st_size})
    return files


def list_runs(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    out = []
    for d in sorted((x for x in root.iterdir() if x.is_dir()), key=lambda x: x.name, reverse=True):
        m = _run_manifest(d) or {}
        n_files = sum(1 for p in d.rglob("*") if p.is_file())
        out.append({"run_id": d.name, "created": m.get("created"), "scenes": m.get("scenes") or [],
                    "variants": m.get("variants") or [], "status": m.get("status") or "unknown",
                    "n_files": n_files})
    return out


def _run_dir(run_id: str) -> Path:
    if "/" in run_id or ".." in run_id or not run_id:
        raise HTTPException(400, "invalid run id")
    d = runs_root(_config()) / run_id
    if not d.is_dir():
        raise HTTPException(404, f"run {run_id} not found")
    return d


@router.get("/runs")
def get_runs():
    return {"runs": list_runs(runs_root(_config()))}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    d = _run_dir(run_id)
    m = _run_manifest(d)
    return {"run_id": run_id, "manifest": m, "files": list_run_files(d, m)}


@router.get("/runs/{run_id}/file")
def get_run_file(run_id: str, path: str = Query(...)):
    d = _run_dir(run_id)
    target = (d / path).resolve()
    if d.resolve() not in target.parents or not target.is_file():
        raise HTTPException(404, "file not found in this run")
    kind = KIND_OF_SUFFIX.get(target.suffix.lower(), "other")
    media = MEDIA_OF_KIND[kind]
    if kind == "png" or target.stat().st_size > 8_000_000:
        return FileResponse(str(target), media_type=media,
                            headers={"Cache-Control": "no-cache"})
    return Response(content=target.read_bytes(), media_type=media,
                    headers={"Cache-Control": "no-cache"})
