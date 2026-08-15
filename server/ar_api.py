# STAC-Builder — AR/XR delivery endpoints (phone viewer over the tailnet).
#
# Serves the assets the WebXR app (static/ar/) needs: the session catalog, the
# textured TSDF mesh (scene.glb — meshopt+WebP, three.js decodes it on-device)
# and a phone-sized binary point cloud decimated from the session's best cloud
# (pgsr_cloud.ply in precision mode, else cleaned_cloud.ply).
#
# ACCESS MODEL: these endpoints carry no JWT — the AR surface is reached ONLY
# through the private tailnet (scripts/tailscale_up.sh); nothing here is
# exposed on a public port. If that ever changes, add auth first.
#
# Cloud wire format "ARC1" (little-endian):
#   bytes 0-3   magic b"ARC1"
#   bytes 4-7   uint32 point count N
#   bytes 8-19  float32[3] bbox min (model space, meters)
#   bytes 20-31 float32[3] bbox max
#   then        N * float32[3] xyz
#   then        N * uint8[3]  rgb
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from config import PROJECTS_DIR

logger = logging.getLogger("ARApi")

router = APIRouter(prefix="/api/ar", tags=["ar"])

_AR_CLOUD_NAME = "ar_cloud_{n}.bin"


def _latest_output_dir(session_id: str) -> Optional[Path]:
    """The session's most recent scans/*/src_*/output dir (mtime order), or a
    legacy <session>/output. None when the session doesn't exist."""
    base = Path(PROJECTS_DIR) / session_id
    if not base.is_dir():
        return None
    outs = sorted(base.glob("scans/*/src_*/output"),
                  key=lambda p: p.stat().st_mtime)
    if outs:
        return outs[-1]
    legacy = base / "output"
    return legacy if legacy.is_dir() else None


def _floor_transform(out: Path) -> Optional[list]:
    """Row-major 4x4 upright transform for sessions whose in-pipeline
    orientation was refused (weak camera-down consensus): CloudCompy's RANSAC
    floor leveler persists s/R/t in floor_transform.npz. None when the model
    is already upright (.orientation_applied) or no transform exists."""
    p = out / "floor_transform.npz"
    if not p.exists():
        return None
    try:
        d = np.load(p)
        s, R, t = float(d["s"]), np.asarray(d["R"], np.float64), \
            np.asarray(d["t"], np.float64)
        M = np.eye(4)
        M[:3, :3] = s * R
        M[:3, 3] = t
        return [round(float(x), 8) for x in M.reshape(-1)]
    except Exception:  # noqa: BLE001
        logger.warning(f"[ar] unreadable {p}")
        return None


def _session_assets(out: Path) -> dict:
    mesh = out / "tsdf" / "scene" / "scene.glb"
    cloud = next((c for c in (out / "pgsr_cloud.ply",
                              out / "cleaned_cloud.ply") if c.exists()), None)
    store = out / "scene_r.db"
    return {
        "has_mesh": mesh.exists(),
        "mesh_bytes": mesh.stat().st_size if mesh.exists() else 0,
        "has_cloud": cloud is not None,
        "cloud_source": cloud.name if cloud is not None else None,
        "has_ai": store.exists(),      # spatial QA needs the instance store
        "floor_transform": _floor_transform(out),   # row-major 4x4 or null
    }


@router.get("/sessions")
async def ar_sessions():
    """Catalog for the AR session picker: every project with at least a mesh
    or a cloud, newest first."""
    items = []
    root = Path(PROJECTS_DIR)
    if root.is_dir():
        for proj in sorted(root.iterdir()):
            if not proj.is_dir():
                continue
            out = _latest_output_dir(proj.name)
            if out is None:
                continue
            a = _session_assets(out)
            if not (a["has_mesh"] or a["has_cloud"]):
                continue
            items.append({"id": proj.name, "mtime": out.stat().st_mtime, **a})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"sessions": items}


@router.get("/mesh/{session_id}")
async def ar_mesh(session_id: str):
    """The textured TSDF mesh, as shipped (meshopt + WebP — three.js's
    GLTFLoader with the meshopt decoder consumes it directly). FileResponse
    honours HTTP Range so the phone can resume."""
    out = _latest_output_dir(session_id)
    mesh = (out / "tsdf" / "scene" / "scene.glb") if out else None
    if mesh is None or not mesh.exists():
        return JSONResponse({"error": f"no mesh for session '{session_id}'"},
                            status_code=404)
    return FileResponse(str(mesh), media_type="model/gltf-binary",
                        filename=f"{session_id}.glb")


def _build_ar_cloud(src: Path, dst: Path, target_points: int) -> dict:
    """Voxel-decimate `src` to ≤ target_points and write the ARC1 blob. Uses
    open3d (already in the server env). Returns the meta dict."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(src))
    n0 = len(pcd.points)
    if n0 == 0:
        raise RuntimeError(f"{src.name} is empty")
    # The cloud is a SURFACE, not a filled volume — a bbox-volume voxel guess
    # lands orders of magnitude too coarse. Grow a fine voxel geometrically
    # until the count fits (each pass over a few-M-point cloud is ~1 s).
    dec = pcd
    if n0 > target_points:
        voxel = 0.004
        for _ in range(24):
            dec = pcd.voxel_down_sample(voxel)
            if len(dec.points) <= target_points:
                break
            voxel *= 1.3
    pts = np.asarray(dec.points, np.float32)
    if dec.has_colors():
        rgb = (np.asarray(dec.colors) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(pts), 3), 200, np.uint8)
    bmin = pts.min(axis=0) if len(pts) else np.zeros(3, np.float32)
    bmax = pts.max(axis=0) if len(pts) else np.zeros(3, np.float32)
    tmp = dst.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(b"ARC1")
        f.write(struct.pack("<I", len(pts)))
        f.write(np.asarray(bmin, np.float32).tobytes())
        f.write(np.asarray(bmax, np.float32).tobytes())
        f.write(np.ascontiguousarray(pts).tobytes())
        f.write(np.ascontiguousarray(rgb).tobytes())
    tmp.replace(dst)
    meta = {"source": src.name, "source_mtime": src.stat().st_mtime,
            "source_points": n0, "points": int(len(pts)),
            "bytes": dst.stat().st_size}
    dst.with_suffix(".json").write_text(json.dumps(meta))
    logger.info(f"[ar] cloud {src.name}: {n0:,} → {len(pts):,} pts "
                f"({dst.stat().st_size / 1e6:.1f} MB) → {dst.name}")
    return meta


@router.get("/cloud/{session_id}")
async def ar_cloud(session_id: str, points: int = 1_500_000):
    """Phone-sized point cloud (ARC1 binary). Decimated once per
    (session, points) and cached next to the source; the cache invalidates
    when the source cloud is newer (e.g. a re-run rebuilt pgsr_cloud.ply)."""
    points = max(50_000, min(int(points), 4_000_000))
    out = _latest_output_dir(session_id)
    if out is None:
        return JSONResponse({"error": f"unknown session '{session_id}'"},
                            status_code=404)
    src = next((c for c in (out / "pgsr_cloud.ply",
                            out / "cleaned_cloud.ply") if c.exists()), None)
    if src is None:
        return JSONResponse({"error": f"no cloud for session '{session_id}'"},
                            status_code=404)
    dst = out / _AR_CLOUD_NAME.format(n=points)
    meta_p = dst.with_suffix(".json")
    fresh = False
    if dst.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            fresh = meta.get("source_mtime") == src.stat().st_mtime
        except Exception:
            fresh = False
    if not fresh:
        try:
            _build_ar_cloud(src, dst, points)
        except Exception as e:  # noqa: BLE001
            logger.exception("[ar] cloud build failed")
            return JSONResponse({"error": f"cloud build failed: {e}"},
                                status_code=500)
    return FileResponse(str(dst), media_type="application/octet-stream",
                        filename=f"{session_id}_cloud.bin")


# ── server-side USDZ for AR Quick Look ───────────────────────────────────────
# Building on the phone OOM-crashed the tab on multi-M-tri meshes; here the
# textured GLB is decimated (UV-preserving) and packaged ARKit-compliant by
# tools/glb_to_usdz.py under the isolated /workspace/usdtools venv (~2-3 min,
# cached). The app calls ?prepare=1 and polls until {status: ready}, then
# navigates an <a rel="ar"> to the bare URL → native Quick Look.

_USDTOOLS_PY = "/workspace/usdtools/bin/python"
_usdz_jobs: dict = {}


def _build_usdz(glb: Path, dst: Path, floor_tf: Optional[list],
                tris: int) -> dict:
    import subprocess
    script = Path(__file__).resolve().parent / "tools" / "glb_to_usdz.py"
    cmd = [_USDTOOLS_PY, str(script), "--glb", str(glb), "--out", str(dst),
           "--target-tris", str(tris)]
    if floor_tf:
        cmd += ["--floor-transform", json.dumps(floor_tf)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:]
        raise RuntimeError(f"usdz build failed: {' '.join(tail)[:300]}")
    stats = json.loads(r.stdout.strip().splitlines()[-1])
    meta = {"source_mtime": glb.stat().st_mtime, "target_tris": tris, **stats}
    dst.with_suffix(".json").write_text(json.dumps(meta))
    logger.info(f"[ar] usdz built: {dst.name} {meta}")
    return meta


@router.get("/usdz/{session_id}")
async def ar_usdz(session_id: str, prepare: int = 0, tris: int = 700_000):
    import asyncio
    out = _latest_output_dir(session_id)
    glb = (out / "tsdf" / "scene" / "scene.glb.orig") if out else None
    if glb is None or not glb.exists():
        return JSONResponse({"error": f"no mesh (scene.glb.orig) for "
                             f"'{session_id}'"}, status_code=404)
    dst = out / "ar_scene.usdz"
    meta_p = dst.with_suffix(".json")
    fresh = False
    if dst.exists() and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            fresh = (meta.get("source_mtime") == glb.stat().st_mtime
                     and meta.get("target_tris") == tris)
        except Exception:
            fresh = False
    if fresh:
        if prepare:
            return {"status": "ready", **json.loads(meta_p.read_text())}
        return FileResponse(str(dst), media_type="model/vnd.usdz+zip",
                            filename=f"{session_id}.usdz")
    floor_tf = _floor_transform(out)
    if prepare:
        job = _usdz_jobs.get(session_id)
        if job is not None:
            if not job.done():
                return {"status": "building"}
            exc = job.exception()
            _usdz_jobs.pop(session_id, None)
            if exc is not None:
                return JSONResponse({"status": "error", "error": str(exc)},
                                    status_code=500)
            return {"status": "ready"}
        _usdz_jobs[session_id] = asyncio.create_task(
            asyncio.to_thread(_build_usdz, glb, dst, floor_tf, tris))
        return {"status": "building"}
    # direct GET while stale: build synchronously (CLI/testing path)
    await asyncio.to_thread(_build_usdz, glb, dst, floor_tf, tris)
    return FileResponse(str(dst), media_type="model/vnd.usdz+zip",
                        filename=f"{session_id}.usdz")


@router.post("/log")
async def ar_client_log(body: dict):
    """Client-side telemetry from the phone app (XR capabilities, errors) —
    remote debugging is impossible otherwise: the XR browsers have no
    devtools. Appended to logs/ar_client.jsonl and echoed to the server log."""
    import time
    line = {"ts": time.time(), **(body or {})}
    logger.info(f"[ar-client] {json.dumps(line)[:800]}")
    try:
        logdir = Path(__file__).resolve().parent.parent / "logs"
        logdir.mkdir(exist_ok=True)
        with open(logdir / "ar_client.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
    except OSError:
        pass
    return {"ok": True}


# The phone XR viewer lives in the built React app: /app/xr.html.
