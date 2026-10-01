"""ShapeR on the backpack (instance 276) of the live corrected epoch — the backend's own recipe
(main.py _export_shaper + run_shaper.sh), config.yaml `shaper:` values, outside the backend."""
import json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
IID = 276
t0 = time.time()
from config import cfg
sc = cfg["shaper"]
from segmentation.shaper_export import export_shaper_pkls
seg = json.loads((O / "segmentation_result.json").read_text())
pkls = export_shaper_pkls(output_dir=O, frames_dir=S / "frames", segments_result=seg, session_dir=S, obj_ids=[IID],
                          caption_fn=None, captions={}, image_format=str(sc["image_format"]), grayscale=bool(sc["grayscale"]),
                          max_views=int(sc["max_views"]), min_view_points=int(sc["min_view_points"]))
print(f"[shaper] PKL(s): {[str(p) for p in pkls]} ({time.time()-t0:.0f} s)", flush=True)
if not pkls:
    sys.exit("no PKL exported")
from workers.base import stop_semantic_service_verified
stop_semantic_service_verified(None, stage="shaper", log=print)
cmd = ["bash", "/workspace/stac-build/server/run_shaper.sh", "--pkls", *map(str, pkls),
       "--output_dir", str(O / "shape"), "--config", str(sc["preset"])]
if not bool(sc["fit_to_cloud"]):
    cmd.append("--no_fit_to_cloud")
print("[shaper] " + " ".join(cmd), flush=True)
r = subprocess.run(cmd)
print(f"[shaper] exit {r.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
for g in sorted((O / "shape").rglob("*.glb")):
    if str(IID) in g.name or str(IID) in g.parent.name:
        print(f"[shaper] GLB {g} ({g.stat().st_size/1e6:.1f} MB)", flush=True)
