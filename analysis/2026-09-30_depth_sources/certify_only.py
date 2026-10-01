"""Stage E of final_epoch1.py alone (the certify stage's correction on the live epoch), no compaction."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
def log(m, level="info"): print(f"[certify {time.strftime('%H:%M:%S')}] {m}", flush=True)
t0 = time.time()
from config import cfg
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
log(f"live epoch {json.loads((O / 'geometry_epoch.json').read_text())['epoch']}")
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
log(f"{acta.get('stop_reason')} — epoch {acta.get('epoch_initial')} -> {acta.get('epoch_final')}")
log(f"DONE in {(time.time()-t0)/60:.1f} min — live epoch {json.loads((O / 'geometry_epoch.json').read_text())['epoch']}, "
    f"stored {sorted(p.name for p in O.glob('_epoch_*'))}")
