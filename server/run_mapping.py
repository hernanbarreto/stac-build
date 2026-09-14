import sys
sys.path.insert(0, "/workspace/stac-build/server")
from segmentation.pipeline import map_segmentation_to_cloud

OUT = "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output"
r = map_segmentation_to_cloud(OUT)
print("=" * 60)
print("ERROR      :", r.get("error"))
print("instancias :", len(r.get("instances", [])))
print("cobertura  :", r.get("coverage"))
print("=" * 60)
