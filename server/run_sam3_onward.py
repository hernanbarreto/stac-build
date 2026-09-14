"""SAM3 → mask→cloud mapping, from the VLM analysis already on disk.

Everything downstream of SAM3 is rebuilt from scratch: segmentation.json,
seg_masks.npz and segmentation_result.json. Certification is NOT run here —
run_certify.py does that, once this result is on disk and inspected.
"""
import sys, json, shutil
sys.path.insert(0, "/workspace/stac-build/server")
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

SESS = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
OUT = SESS / "output"
FRAMES = SESS / "frames"

vlm = json.loads((OUT / "vlm_analysis.json").read_text())
prompt = vlm.get("prompt", "")
if not prompt:
    print("!! vlm_analysis.json has no prompt — aborting"); sys.exit(1)
print("=" * 70)
print("1/2  SAM3")
print("=" * 70)
print(f"categorias: {len(prompt.split(';'))}")

# SIMPLE pipeline: text concepts only, all frames per concept, no box seeds
for stale in ("segmentation.json", "seg_masks.npz", "segmentation_result.json",
              "scene_r.db", "classification.npy", "seg_broadcast.json"):
    p = OUT / stale
    if p.exists():
        p.unlink(); print(f"  borrado previo: {stale}")

# SAM3 wants the whole GPU: vLLM's ~40 GB resident starve a long session.
# Any later consumer restarts it (semantic.service.ensure_service).
import subprocess, time
if subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True).returncode == 0:
    subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
    for _ in range(30):
        time.sleep(2)
        if subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True).returncode != 0:
            break
    print("  vLLM detenido — GPU exclusiva para SAM3")

from segmentation_pipeline import run_segmentation
res = run_segmentation(frames_dir=str(FRAMES), output_dir=str(OUT),
                       prompt=prompt, frame_map={}, boxes_map=None,
                       on_progress=lambda pct, msg: None)
if res.get("error"):
    print(f"!! SAM3 fallo: {res['error']}"); sys.exit(1)
print(f"SAM3 ok: {len(res.get('instances', []))} instancias")

print()
print("=" * 70)
print("2/2  mapeo mascara -> nube")
print("=" * 70)
from segmentation.pipeline import map_segmentation_to_cloud
r = map_segmentation_to_cloud(OUT)
print()
print("=" * 70)
print("ERROR      :", r.get("error"))
print("instancias :", len(r.get("instances", [])))
print("cobertura  :", r.get("coverage"))
print("=" * 70)
