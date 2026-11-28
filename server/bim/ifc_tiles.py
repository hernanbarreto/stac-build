"""THE VIEWER'S TILES OF A PROJECT'S IFC (USER 2026-11-28: "se pone densa la pagina al mostrar el ifc y ademas aun
luego de apagar el ifc queda lento, como lo podemos manejar?, tipo los tiles del glb?" / "si, mejora todo" / "las
soluciones siempre deben ser completas y de fondo, no parches").

The page built every triangle of every IFC of the session at its open (grupo_electrogeno's: 7.54 M) and kept them, hidden
or not. Each IFC of ``<project>/ifcs/`` now has its tiles beside it, ``ifcs/.tiles/<file name>/`` (tileset.json, the
tiles, elements.json — the parts and the spatial tree), made by tools/glb/ifc_tiles.mjs in a process of its own (CPU,
low priority). They belong to the IFC as it was read — its size and date, recorded in tiles_report.json: an IFC
uploaded again, regenerated or rescaled in place is tiled again the next time its tiles are asked for, and a viewer
asks under the IFC's version, so a browser never mixes two.

States: ``ready`` (with ``version``), ``building``, ``failed`` (with ``error``: not retried for the same file — a new
version is), ``missing`` (no such IFC).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "glb" / "ifc_tiles.mjs"
TILES = ".tiles"
JOBS = 4


NODE_ENV_BIN = "/workspace/miniforge3/envs/nodejs/bin/node"


def _node() -> str:
    """The pod's node (env nodejs — the one the GenRecon tiles run with); the PATH's only on a machine without it
    (the backend's PATH found the system's node 18 first)."""
    return NODE_ENV_BIN if os.path.exists(NODE_ENV_BIN) else (shutil.which("node") or "node")


def tiles_dir(ifcs_dir: Path, name: str) -> Path:
    return Path(ifcs_dir) / TILES / name


def _identity(ifc: Path) -> tuple[int, int]:
    st = ifc.stat()
    return st.st_size, st.st_mtime_ns


def version_of(ifc: Path) -> str:
    """The IFC as it is now, as the viewer's address carries it (size and date: what the tiles record)."""
    size, mtime = _identity(ifc)
    return f"{size:x}-{mtime:x}"


def _ready(ifc: Path, out: Path) -> bool:
    try:
        rep = json.loads((out / "tiles_report.json").read_text())
    except (OSError, ValueError):
        return False
    if not (out / "tileset.json").is_file() or not (out / "elements.json").is_file():
        return False
    size, mtime = _identity(ifc)
    return rep.get("source_bytes") == size and str(rep.get("source_mtime_ns")) == str(mtime)


def _alive(pid: int) -> bool:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return SCRIPT.name.encode() in cmd


def state(ifcs_dir: Path, name: str) -> dict:
    """Where the tiles of ``ifcs/<name>`` stand, without starting anything."""
    ifc = Path(ifcs_dir) / name
    if not ifc.is_file():
        return {"state": "missing"}
    out = tiles_dir(ifcs_dir, name)
    run = out.parent / f"{name}.building.json"
    if _ready(ifc, out):
        if run.is_file():                                # the job that made them has ended: its record goes
            try:
                pid = int(json.loads(run.read_text()).get("pid") or 0)
            except (OSError, ValueError):
                pid = 0
            if not (pid and _alive(pid)):
                run.unlink(missing_ok=True)
        return {"state": "ready", "version": version_of(ifc)}
    failed = out.parent / f"{name}.failed.json"
    me = version_of(ifc)
    if run.is_file():
        try:
            job = json.loads(run.read_text())
        except (OSError, ValueError):
            job = {}
        if job.get("pid") and _alive(int(job["pid"])):
            return {"state": "building", "since": job.get("started")}
        # the job ended without tiles of this file: what it said is kept as the failure of that version
        log = out.parent / f"{name}.log"
        tail = log.read_text(errors="replace")[-1500:] if log.is_file() else ""
        run.unlink(missing_ok=True)
        if job.get("version") == me:
            failed.write_text(json.dumps({"version": me, "error": tail.strip() or "the conversion ended with no tiles"}))
    if failed.is_file():
        try:
            f = json.loads(failed.read_text())
        except (OSError, ValueError):
            f = {}
        if f.get("version") == me:
            return {"state": "failed", "error": f.get("error", "")}
    return {"state": "absent"}


def ensure(ifcs_dir: Path, name: str) -> dict:
    """The tiles' state, the conversion started when they are absent or of another version of the IFC."""
    st = state(ifcs_dir, name)
    if st["state"] != "absent":
        return st
    ifcs_dir = Path(ifcs_dir).resolve()                  # (the converter runs in its own directory)
    ifc = ifcs_dir / name
    out = tiles_dir(ifcs_dir, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / f"{name}.log"
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "UV_THREADPOOL_SIZE": "4"})
    with open(log, "w") as fh:
        p = subprocess.Popen(
            ["nice", "-n", "15", _node(), "--max-old-space-size=24000", str(SCRIPT), str(ifc), str(out), "--jobs", str(JOBS)],
            cwd=str(SCRIPT.parent), stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env)
    (out.parent / f"{name}.building.json").write_text(json.dumps(
        {"pid": p.pid, "version": version_of(ifc), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    (out.parent / f"{name}.failed.json").unlink(missing_ok=True)
    return {"state": "building", "since": None}


def remove(ifcs_dir: Path, name: str) -> None:
    """An IFC deleted: its tiles and their records go with it."""
    base = Path(ifcs_dir) / TILES
    shutil.rmtree(base / name, ignore_errors=True)
    for suffix in (".part", ".work"):
        shutil.rmtree(base / f"{name}{suffix}", ignore_errors=True)
    for suffix in (".building.json", ".failed.json", ".log"):
        (base / f"{name}{suffix}").unlink(missing_ok=True)


def file_path(ifcs_dir: Path, name: str, version: str, rel: str) -> Path | None:
    """A file of the tiles of ``name`` at ``version`` — None when that version is not the one on disk."""
    ifc = Path(ifcs_dir) / name
    if not ifc.is_file() or version_of(ifc) != version:
        return None
    out = tiles_dir(ifcs_dir, name)
    if not _ready(ifc, out):
        return None
    p = (out / rel).resolve()
    if out.resolve() not in p.parents or not p.is_file():
        return None
    return p
