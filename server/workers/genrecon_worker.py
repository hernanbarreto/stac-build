# STAC-Builder: GenRecon mesh worker (subprocess) — the mesh of the scene as the LAST stage of the
# segmentation chain (USER 2026-10-17: "el botón meshing se va, porque este pipeline de mesh se hace al
# finalizar el autosegment … lo cableamos ahí").
#
# What it runs: server/tools/genrecon/run_cloud.sh in mode `cloud` — the session's keyframes, certified
# poses and PUBLISHED cloud handed to GenRecon (Trellis.2 scene generation, vendor/GenRecon + the STAC
# patch) with the active voxels of every chunk taken FROM THE CLOUD (presence is the cloud's, never the
# model's; the cube anchored on the segmentation's floor/ceiling levels), baked and published into
# output/tsdf/<mesh_name>/ for the viewer. Measured on pccr 2026-10-16: 95 % of the mesh area within 5 cm
# of the cloud, floor 0 cm — the user's "está perfecta".
#
# The adapter and the publish run on python_da3, the generation and the bake on the `genrecon` env (the
# script owns the interpreters). vLLM is NOT stopped: the generation peaks at ~12 GB beside it.
# Generative: presence and shape are declared in the mesh's meta (provenance vlm_generated).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
import json
import os
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

from workers.base import WorkerPipe, run_worker_safe

STAGE = "genrecon"
_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "genrecon" / "run_cloud.sh"
# progress by the script's own milestones (a look, not a measurement)
_MARKS = [
    ("=== SESSION", 2, "handing the session to GenRecon"),
    ("levels:", 8, "floor and ceiling levels from the segmentation"),
    ("images, ", 12, "keyframes, poses and cloud written"),
    ("undistorted", 18, "keyframes undistorted"),
    ("coords from the cloud", 25, "active voxels from the cloud"),
    ("Sampling shape", 35, "generating the surface inside the cloud's voxels"),
    ("Sampling texture", 55, "generating the texture"),
    ("joint_decode_shape", 62, "decoding the scene mesh"),
    ("joint_decode_tex", 68, "decoding the texture"),
    ("[chunked_to_glb] baking chunk", 72, "baking chunks"),
    ("writing", 95, "writing scene.glb"),
    ("published", 99, "published"),
]
_FAILS = ("adapter failed", "reconstruct failed", "bake failed", "publish failed", "hull failed")


def _genrecon_work(pipe: WorkerPipe, session_dir: str, config: dict):
    session_path = Path(session_dir).resolve()
    output_dir = session_path / "output"
    if not (output_dir / "cleaned_cloud.ply").exists():
        raise FileNotFoundError("cleaned_cloud.ply not found — no published cloud; the GenRecon mesh needs it")
    parts = session_path.parts
    try:
        sid, scan = parts[-4], parts[-2]            # projects/<sid>/scans/<scan>/src_default
    except IndexError as e:  # noqa: BLE001
        raise RuntimeError(f"session path {session_path} is not projects/<sid>/scans/<scan>/src_default") from e
    gcfg = dict(config.get("genrecon") or {})
    name = str(gcfg.get("mesh_name", "genrecon_photo"))
    mode = str(gcfg.get("mode", "cloud"))
    min_pts = int(gcfg.get("min_points_per_voxel", 10))
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "GENRECON_SIMPLIFY": str(int(gcfg.get("simplify_faces_per_chunk", 150000))),
           "GENRECON_TEX": str(int(gcfg.get("texture_size", 2048))),
           "GENRECON_TOPK": str(int(gcfg.get("bake_views", 3))),
           "GENRECON_PRESENCE": str(gcfg.get("presence_source", "cloud"))}
    if not _SCRIPT.exists():
        raise FileNotFoundError(f"GenRecon runner missing: {_SCRIPT}")
    pipe.send_progress(1, f"GenRecon mesh ({mode}, ≥{min_pts} pts/voxel) for {sid}:{scan}", stage=STAGE)
    pipe.send_log(f"[genrecon] {_SCRIPT.name} {sid}:{scan} {name} {min_pts} {mode} "
                  f"(simplify {env['GENRECON_SIMPLIFY']} faces/chunk, texture {env['GENRECON_TEX']})")
    proc = subprocess.Popen(["bash", str(_SCRIPT), f"{sid}:{scan}", name, str(min_pts), mode],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(_SCRIPT.parent.parent.parent), env=env)
    assert proc.stdout is not None
    pct = 1
    failed = None
    published = None
    bake_n = 0
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("+ "):          # the script's own `set -x` echo
            continue
        for mark, p, msg in _MARKS:
            if mark in line and p > pct:
                pct = p
                pipe.send_progress(pct, msg, stage=STAGE)
                break
        if "[chunked_to_glb] baking chunk" in line:
            bake_n += 1
            pipe.send_progress(min(94, 72 + bake_n), f"baking chunk {bake_n}", stage=STAGE)
        if line.startswith("image convention") or line.startswith("mesh depth rendered"):
            pct = max(pct, 95); pipe.send_progress(pct, "baking the keyframes into the generated atlas", stage=STAGE)
        if line.lstrip().startswith("part ") and "photographed" in line:
            pct = max(pct, min(98, 95 + bake_n // 8)); bake_n += 1
            pipe.send_progress(pct, f"photo albedo: {line.strip()[:60]}", stage=STAGE)
        if line.startswith("published "):
            published = line
        for f in _FAILS:
            if line.startswith(f):
                failed = line
        if "Traceback" in line or "Error" in line or "[STAC]" in line or "[VRAM]" in line or line.startswith("published"):
            pipe.send_log(f"[genrecon] {line[:300]}")
    code = proc.wait()
    if code != 0 or failed is not None:
        raise RuntimeError(f"GenRecon stage failed ({failed or f'exit {code}'}) — see the stage log")
    glb = output_dir / "tsdf" / name / f"{name}.glb"
    if not glb.exists():
        raise RuntimeError(f"GenRecon ended without {glb}")
    meta_p = output_dir / "tsdf" / name / f"{name}.meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    pipe.send_progress(100, f"GenRecon mesh published: {name} ({meta.get('n_faces', '?')} faces)", stage=STAGE)
    pipe.send_log(f"[genrecon] {published or glb}")


def run(conn: Connection, session_dir: str, config: dict):
    run_worker_safe(_genrecon_work, conn, session_dir, config)


if __name__ == "__main__":
    print("genrecon_worker is spawned by the pipeline manager", file=sys.stderr)
    sys.exit(2)
