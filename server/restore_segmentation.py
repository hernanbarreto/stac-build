"""Restore the pre-certification segmentation snapshot (see snapshot_segmentation.py)."""
import shutil, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output")
SNAP = OUT / "_presplit"
if not SNAP.is_dir():
    print(f"no hay snapshot en {SNAP}"); sys.exit(1)
for src in SNAP.iterdir():
    shutil.copy2(src, OUT / src.name)
    print(f"  restaurado {src.name}")
# los productos de la certificacion quedan invalidos
for name in ("certify_acta.json", "loop_candidates.json", "duplicates.json",
             "loop_semantics.json", "keyframe_graph.json"):
    p = OUT / name
    if p.exists():
        p.unlink(); print(f"  borrado {name}")
print("segmentacion restaurada al estado pre-split")
