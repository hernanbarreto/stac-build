# STAC-Builder: GenRecon mesh worker (subprocess) — the mesh of the scene as the LAST stage of the
# segmentation chain (USER 2026-10-17: "el botón meshing se va, porque este pipeline de mesh se hace al
# finalizar el autosegment … lo cableamos ahí").
#
# What it runs: server/tools/genrecon/run_cloud.sh, TWICE (USER 2026-11-02: "veamos cómo sale generativa y
# no generativa, que aparezcan las dos variantes"), the session's keyframes, certified poses and PUBLISHED
# cloud handed to GenRecon (Trellis.2 scene generation, vendor/GenRecon + the STAC patch), the chunk cubes
# on the scene's levels (tools/genrecon/genrecon_levels.py):
#   1. genrecon.mesh_name, mode genrecon.mode (`cloud`) — the active voxels of every chunk taken FROM THE
#      CLOUD (presence is the cloud's, never the model's), the keyframes baked into its atlas (the real
#      textures). Measured on pccr 2026-10-16: 95 % of the mesh area within 5 cm of the cloud, floor 0 cm.
#   2. genrecon.generative_mesh_name, mode `generative` — the model's own presence from the keyframes (the
#      cloud only lays out the chunks) and the texture it generated: no photo bake ("en la generativa no,
#      porque no va a ser exactamente igual"). An empty name runs only the first.
# Both published into output/tsdf/<name>/ for the viewer.
#
# The adapter and the publish run on python_da3, the generation and the bake on the `genrecon` env (the
# script owns the interpreters). vLLM is STOPPED (verified) before the generation: pccr's 3.7 m cubes peaked
# at ~12 GB beside it, station's 7 m cubes ran out of memory with its 38 GB on the card (2026-11-02, a GenRecon
# job ordered right after another, the backend having brought vLLM back between them); the chat comes back
# when the job ends.
# Generative: presence and shape are declared in the mesh's meta (provenance vlm_generated).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
import json
import os
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

from workers.base import WorkerPipe, run_worker_safe, stop_semantic_service_verified

STAGE = "genrecon"
_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "genrecon" / "run_cloud.sh"
# progress by the script's own milestones (a look, not a measurement)
_MARKS = [
    ("=== SESSION", 2, "handing the session to GenRecon"),
    ("levels:", 8, "floor and ceiling levels from the segmentation"),
    ("images, ", 12, "keyframes, poses and cloud written"),
    ("undistorted", 18, "keyframes undistorted"),
    ("coords from the cloud", 25, "active voxels from the cloud"),
    ("Sampling sparse structure", 25, "the model generating the scene's structure"),
    ("Sampling shape", 35, "generating the surface"),
    ("Sampling texture", 55, "generating the texture"),
    ("joint_decode_shape", 62, "decoding the scene mesh"),
    ("joint_decode_tex", 68, "decoding the texture"),
    ("[chunked_to_glb] baking chunk", 72, "baking chunks"),
    ("writing", 95, "writing scene.glb"),
    ("published", 99, "published"),
]
_FAILS = ("adapter failed", "reconstruct failed", "bake failed", "publish failed", "hull failed")


def _run_variant(pipe: WorkerPipe, output_dir: Path, sid: str, scan: str, name: str, min_pts: int, mode: str,
                 env: dict, lo: int, hi: int, tag: str) -> dict:
    """One run of the runner: its milestones mapped into [lo, hi] % of the stage, its published mesh's
    meta returned (a failure raises, naming the variant)."""
    def prog(p: int, msg: str):
        pipe.send_progress(lo + (hi - lo) * p // 100, f"{tag}: {msg}", stage=STAGE)

    pipe.send_log(f"[genrecon] {tag}: {_SCRIPT.name} {sid}:{scan} {name} {min_pts} {mode} "
                  f"(simplify {env['GENRECON_SIMPLIFY']} faces/chunk, texture {env['GENRECON_TEX']}, "
                  f"cubes {env['GENRECON_CUBE_M']} m)")
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
                prog(pct, msg)
                break
        if "[chunked_to_glb] baking chunk" in line:
            bake_n += 1
            prog(min(94, 72 + bake_n), f"baking chunk {bake_n}")
        if line.startswith("image convention") or line.startswith("mesh depth rendered"):
            pct = max(pct, 95); prog(pct, "baking the keyframes into the generated atlas")
        if line.lstrip().startswith("part ") and "photographed" in line:
            pct = max(pct, min(98, 95 + bake_n // 8)); bake_n += 1
            prog(pct, f"photo albedo: {line.strip()[:60]}")
        if line.startswith("published "):
            published = line
        for f in _FAILS:
            if line.startswith(f):
                failed = line
        if ("Traceback" in line or "Error" in line or "[STAC]" in line or "[VRAM]" in line or line.startswith("published")
                or line.startswith("levels:")):
            pipe.send_log(f"[genrecon] {line[:300]}")
    code = proc.wait()
    if code != 0 or failed is not None:
        raise RuntimeError(f"GenRecon stage failed — {tag} ({failed or f'exit {code}'}) — see the stage log")
    glb = output_dir / "tsdf" / name / f"{name}.glb"
    if not glb.exists():
        raise RuntimeError(f"GenRecon ended without {glb} ({tag})")
    meta_p = glb.parent / f"{name}.meta.json"
    pipe.send_log(f"[genrecon] {tag}: {published or glb}")
    return json.loads(meta_p.read_text()) if meta_p.exists() else {}


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
    gen_name = str(gcfg.get("generative_mesh_name", "genrecon_generative") or "")
    min_pts = int(gcfg.get("min_points_per_voxel", 10))
    conf = float(gcfg.get("confidence", 0.0))
    if not 0.0 <= conf < 1.0:
        raise ValueError(f"genrecon.confidence must be in [0, 1) — the viewer's slider as a fraction, got {conf}")
    if gen_name and gen_name == name:
        raise ValueError(f"genrecon.generative_mesh_name equals genrecon.mesh_name ({name}): one would overwrite the other")
    env = {**os.environ, "PYTHONUNBUFFERED": "1",
           "GENRECON_SIMPLIFY": str(int(gcfg.get("simplify_faces_per_chunk", 150000))),
           "GENRECON_TEX": str(int(gcfg.get("texture_size", 2048))),
           "GENRECON_TOPK": str(int(gcfg.get("bake_views", 3))),
           "GENRECON_CUBE_M": str(float(gcfg.get("cube_size_m", 0.0))),
           "GENRECON_TILES": "1" if bool(gcfg.get("tiles", False)) else "0",
           # USER 2026-11-05: the viewer's confidence slider chosen for THIS order (the Autosegment window), the
           # chunk bakes in parallel processes, the photo atlas bounds
           "GENRECON_CONF": str(conf),
           "GENRECON_BAKE_WORKERS": str(int(gcfg.get("bake_workers", 1))),
           "GENRECON_TEX_MIN": str(int(gcfg.get("photo_atlas_min", 512))),
           "GENRECON_TEX_MAX": str(int(gcfg.get("photo_atlas_max", 4096)))}
    if not _SCRIPT.exists():
        raise FileNotFoundError(f"GenRecon runner missing: {_SCRIPT}")
    stop_semantic_service_verified(pipe, stage="GenRecon mesh")
    variants = [(name, mode, f"1/{2 if gen_name else 1} from the cloud, real textures")]
    if gen_name:
        variants.append((gen_name, "generative", "2/2 generative, its own texture"))
    pipe.send_progress(1, f"GenRecon mesh ({len(variants)} variant(s)) for {sid}:{scan}", stage=STAGE)
    done = []
    for k, (vname, vmode, tag) in enumerate(variants):
        if k and pipe.check_cancel():              # between the variants: the first one stays published
            pipe.send_log(f"[genrecon] cancelled before {tag}")
            return
        lo, hi = 100 * k // len(variants), 100 * (k + 1) // len(variants)
        meta = _run_variant(pipe, output_dir, sid, scan, vname, min_pts, vmode, env, lo, hi, tag)
        done.append(f"{vname} ({meta.get('n_faces', '?')} faces)")
    pipe.send_progress(100, f"GenRecon mesh published: {', '.join(done)}", stage=STAGE)


def run(conn: Connection, session_dir: str, config: dict):
    run_worker_safe(_genrecon_work, conn, session_dir, config)


if __name__ == "__main__":
    print("genrecon_worker is spawned by the pipeline manager", file=sys.stderr)
    sys.exit(2)
