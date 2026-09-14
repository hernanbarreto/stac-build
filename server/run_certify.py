"""Certification of pccr / 2026-08-31 over the cloud and segmentation on disk.

Brings the semantic service up first: the instance classification (structural |
movable | dynamic) needs Qwen, and without it every instance falls to the
default class, which costs the run its instance loops.
"""
import sys

sys.path.insert(0, "/workspace/stac-build/server")
sys.stdout.reconfigure(line_buffering=True)

SESSION = "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"

print("=" * 70)
print("1/2  semantic service (vLLM / Qwen3-VL)")
print("=" * 70)
from config import cfg
from semantic.service import ensure_service

if not ensure_service(cfg, log=print, timeout_s=900):
    print("\n!! the semantic service did not come up — ABORTING.")
    print("   Without it every instance falls to the default class and the run")
    print("   ends with 0 instance loops, which is what we are trying to fix.")
    sys.exit(1)
print("semantic service ready")

print()
print("=" * 70)
print("2/2  certification loop")
print("=" * 70)
from reconstruction.certify.run import certify_session

acta = certify_session(SESSION, operator="manual", log=print)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print("epoch      :", acta.get("epoch_initial"), "->", acta.get("epoch_final"))
print("stop reason:", acta.get("stop_reason"))
print("regressed  :", acta.get("regressed"))
for it in acta.get("iterations", []):
    print(f"  iter {it.get('iteration')}: verdict={it.get('verdict')} "
          f"objective={it.get('objective')} improvement={it.get('improvement')}")
    for w in it.get("gate_warnings", []):
        print(f"     warning: {w}")
m = acta.get("metrics_final") or {}
print("duplicates :", (m.get("duplicates") or {}).get("n"))
print("closure    :", (m.get("closure") or {}).get("median_m"))
print("loop edges :", (m.get("loop_residual") or {}).get("n_edges"))
print("=" * 70)
