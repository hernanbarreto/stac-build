"""The viewer's routes for the Gaussian-splat layer (mounted by main.py):

  GET /api/gaussians/list/{session_id}                       the published models with their files and meta
  GET /api/gaussians/file/{session_id}/{name}/{filename}     gaussians.ply | gaussians.splat | seed_index.bin |
                                                             meta.json | report.json
  GET /api/gaussians/instances/{session_id}/{name}           the CURRENT instance of every gaussian (uint32 LE),
                                                             resolved live from the cloud's instance_ids.npy
                                                             through seed_index.bin — never baked in; stale
                                                             (another cloud) is said in the header and the list
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from gaussians import label as GL
from gaussians.plyio import read_instances

router = APIRouter()
FILES = {"gaussians.ply": "application/octet-stream", "gaussians.splat": "application/octet-stream",
         "seed_index.bin": "application/octet-stream", "meta.json": "application/json", "report.json": "application/json"}
SERVER_DIR = Path(__file__).resolve().parent.parent


def _output_dir(session_id: str) -> Path:
    from project_paths import resolve_session
    ctx = resolve_session(str(SERVER_DIR), session_id)
    return Path(ctx.output_dir)


def _safe(part: str) -> str:
    if not part or "/" in part or "\\" in part or ".." in part:
        raise HTTPException(status_code=400, detail="Invalid path component")
    return part


def staleness(meta: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Whether the cloud the splats were seeded from is still the cloud on disk (by point count and
    sha), so the live instance lookup is meaningful."""
    cloud = output_dir / "cleaned_cloud.ply"
    inst = output_dir / "instance_ids.npy"
    rec: Dict[str, Any] = {"stale": False, "reason": ""}
    if not cloud.exists():
        return {"stale": True, "reason": "no cleaned_cloud.ply on disk"}
    n_meta = (meta.get("cloud") or {}).get("n_points")
    n_now: Optional[int] = None
    if inst.exists():
        try:
            n_now = int(np.load(inst, mmap_mode="r").shape[0])
        except Exception as e:  # noqa: BLE001
            return {"stale": True, "reason": f"instance_ids.npy unreadable: {e}"}
    if n_meta is not None and n_now is not None and int(n_meta) != n_now:
        return {"stale": True, "reason": f"the cloud changed: {n_now} points now, {n_meta} when the splats were trained"}
    if n_now is None:
        rec["reason"] = "no instance_ids.npy — every gaussian unsegmented until the projection runs"
    return rec


def list_models(output_dir: Path, session_id: str) -> Dict[str, Any]:
    gdir = output_dir / "gaussians"
    models = []
    if gdir.exists():
        for d in sorted(p for p in gdir.iterdir() if p.is_dir()):
            mp = d / "meta.json"
            if not mp.exists():
                continue
            try:
                meta = json.loads(mp.read_text())
            except ValueError:
                continue
            files = {k: f"/api/gaussians/file/{session_id}/{d.name}/{v}" for k, v in (meta.get("files") or {}).items()
                     if v in FILES}
            models.append({"name": d.name, "files": files,
                           "instances_url": f"/api/gaussians/instances/{session_id}/{d.name}",
                           "n_gaussians": meta.get("n_gaussians"), "heldout": meta.get("heldout"),
                           "geometry_epoch": meta.get("geometry_epoch"), **staleness(meta, output_dir), "meta": meta})
    return {"ok": True, "models": models}


@router.get("/api/gaussians/list/{session_id}")
async def gaussians_list(session_id: str):
    return list_models(_output_dir(session_id), session_id)


@router.get("/api/gaussians/file/{session_id}/{name}/{filename}")
async def gaussians_file(session_id: str, name: str, filename: str):
    name, filename = _safe(name), _safe(filename)
    if filename not in FILES:
        raise HTTPException(status_code=404, detail=f"{filename} is not a file of the gaussians stage")
    p = _output_dir(session_id) / "gaussians" / name / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(str(p), media_type=FILES[filename], headers={"Cache-Control": "no-cache"})


def current_instances(output_dir: Path, name: str) -> tuple:
    """(uint32 array per gaussian, staleness record)."""
    d = output_dir / "gaussians" / name
    si_p = d / "seed_index.bin"
    if not si_p.exists():
        raise HTTPException(status_code=404, detail="seed_index.bin not found")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    st = staleness(meta, output_dir)
    si = read_instances(si_p)
    inst_p = output_dir / "instance_ids.npy"
    if st["stale"] or not inst_p.exists():
        return np.zeros(len(si), np.uint32), st
    ids = np.load(inst_p, mmap_mode="r")
    return GL.instances_from_seed_index(si, np.asarray(ids)), st


@router.get("/api/gaussians/instances/{session_id}/{name}")
async def gaussians_instances(session_id: str, name: str):
    name = _safe(name)
    ids, st = current_instances(_output_dir(session_id), name)
    census = GL.instance_census(ids)
    reason = str(st.get("reason", ""))[:200].encode("ascii", "replace").decode("ascii")   # a header is latin-1
    headers = {"Cache-Control": "no-cache", "X-Gaussians-Stale": "1" if st["stale"] else "0",
               "X-Gaussians-Reason": reason,
               "X-Gaussians-Segmented": str(census["n_segmented"]), "X-Gaussians-Instances": str(census["n_instances"])}
    return Response(content=np.ascontiguousarray(ids.astype("<u4")).tobytes(), media_type="application/octet-stream",
                    headers=headers)
