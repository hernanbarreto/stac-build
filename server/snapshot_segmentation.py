"""Snapshot of the segmentation BEFORE the certification is allowed near it.

An `instance-split` rewrites segmentation.json and seg_masks.npz in place —
points moved to a new instance, mask pixels repainted — and there is no undo.
Every certification run with a wrong verdict therefore costs a full SAM3 pass
to get back to a clean state (45 min, paid twice on 2026-09-14). One copy
before the run makes that a file restore.
"""
import shutil, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output")
SNAP = OUT / "_presplit"
SNAP.mkdir(exist_ok=True)
for name in ("segmentation.json", "seg_masks.npz", "segmentation_result.json",
             "scene_r.db", "classification.npy"):
    src = OUT / name
    if src.exists():
        shutil.copy2(src, SNAP / name)
        print(f"  {name}  ({src.stat().st_size / 1048576:.1f} MB)")
print(f"snapshot pre-split en {SNAP}")
