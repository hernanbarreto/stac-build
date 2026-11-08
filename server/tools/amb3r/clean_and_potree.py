"""The cloud stage's clean and octree for a session built outside the pipeline (tools/vidmap). For a VALIDATION look only (USER 2026-11-02: "era solo limpiar la nube, sin normales, potree y afuera"): the
pipeline's own GPU clean (reconstruction.gpu_cloud_clean, the cloud stage's arguments from config.yaml
postprocessing, the witnesses at merge as the stage runs them) and the pipeline's own octree (potree_converter) —
no consolidation, no normals, no floor levelling.

  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 python clean_and_potree.py <session src_default> [--no-witness]
--no-witness (USER 2026-11-02: "pasala por cloudcompy sin filtro de vecinos"): the witness
filter off — no point leaves for lacking a second keyframe that confirms it; the rest of the clean as configured.
"""
import subprocess
import sys
import time
from pathlib import Path

import yaml

SERVER = Path("/workspace/stac-build/server")
sys.path.insert(0, str(SERVER))
ses = Path(sys.argv[1]).resolve()
no_witness = "--no-witness" in sys.argv[2:]
out = ses / "output"
cfg = yaml.safe_load(open(SERVER / "config.yaml"))
pp = cfg.get("postprocessing", {})
from reconstruction.loops.config import load_loops_config  # noqa: E402
cmd = ["/workspace/miniforge3/envs/da3/bin/python", "-m", "reconstruction.gpu_cloud_clean",
       "--input-dir", str(out), "--output", str(out / "cleaned_cloud.ply"),
       "--voxel-size", str(pp.get("voxel_size", 0.001)), "--sor-knn", str(pp.get("sor_knn", 6)),
       "--sor-sigma", str(pp.get("sor_sigma", 1.0)), "--noise-radius", str(pp.get("noise_radius", 0.01)),
       "--noise-sigma", str(pp.get("noise_sigma", 1.0)), "--conf-min-norm", str(pp.get("conf_min_norm", 0.0))]
if load_loops_config(cfg).witness.at_merge and not no_witness:
    cmd.append("--witness")
for flag in ("skip_duplicates", "skip_sor", "skip_noise", "skip_normals"):
    if pp.get(flag, False):
        cmd.append(f"--{flag.replace('_', '-')}")
t0 = time.time()
rc = subprocess.call(cmd, cwd=str(SERVER))
if rc != 0:
    raise SystemExit(f"gpu_cloud_clean exited {rc}")
print(f"[clean] done in {time.time() - t0:.0f} s", flush=True)
from potree_converter import convert_ply_to_potree  # noqa: E402
t0 = time.time()
ok = convert_ply_to_potree(ses, force=True)
print(f"[potree] {'ok' if ok else 'FAILED'} in {time.time() - t0:.0f} s", flush=True)
