"""Orient object GLBs outward in place (reconstruction/orient_outward.py) — for meshes generated before
2026-10-01.   python server/tools/fix_object_normals.py <object.glb> [...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.orient_outward import orient_glb  # noqa: E402

if __name__ == "__main__":
    for a in sys.argv[1:]:
        n, tot = orient_glb(Path(a))
        print(f"{Path(a).name}: {n:,} of {tot:,} faces flipped outward")
