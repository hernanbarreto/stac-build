"""Epoch 6 (already built, live): SAM3 masks projected with today's code (co-visible split with the free-space
test, class byte + class_map) and the certify stage; the corrected epoch renumbered 6 and its segmentation
stored with it. Run with the backend restarted on the new code and pccr NOT open in the viewer (the backend
re-projects on its own when it loads a session whose segmentation is stale)."""
import json, shutil, sys, time
from pathlib import Path
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"


def log(m, level="info"):
    print(f"[e6 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


t0 = time.time()
assert live() == 6, live()
from config import cfg
for r in ("segmentation_result.json", "classification.npy", "class_map.json"):
    if (O / r).exists():
        (O / r).rename(O / "precision" / f"stale_{r}")
from segmentation.pipeline import map_segmentation_to_cloud
d = map_segmentation_to_cloud(O)
if d.get("error"):
    log(f"❌ projection: {d['error']}"); sys.exit(1)
n = int(open(O / "cleaned_cloud.ply", "rb").read(400).decode("ascii", "ignore").split("element vertex")[1].split()[0])
log(f"{len(d.get('instances', []))} instances, coverage {d.get('coverage')}, indexes {d.get('total_points')} of {n} pts "
    f"({(time.time() - t0) / 60:.1f} min)")
assert d.get("total_points") == n
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
e_fin = live()
if e_fin != 6:
    shutil.rmtree(O / "_epoch_6", ignore_errors=True)
    recp = O / "geometry_epoch.json"; r_ = json.loads(recp.read_text()); r_.update(epoch=6, renumbered_from=e_fin)
    recp.write_text(json.dumps(r_, indent=2))
log(f"DONE in {(time.time() - t0) / 60:.1f} min — live {live()}, stored {sorted(x.name for x in O.glob('_epoch_*'))}")
