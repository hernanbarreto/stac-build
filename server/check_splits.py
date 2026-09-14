"""Re-judge this session's splits with the reprojection: same object or not."""
import sys, json
sys.path.insert(0, "/workspace/stac-build/server")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path

SESS = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
OUT = SESS / "output"

from segmentation.pipeline import _load_ply_origins, _mask_frame_lookup
from reconstruction.loops.reprojection import copy_evidence
from reconstruction.loops.config import load_loops_config as _llc
_RC = _llc().loops.reprojection
_RKW = dict(dilate_px=_RC.dilate_px, max_frames=_RC.max_frames,
            min_self_recall=_RC.min_self_recall, min_cross_recall=_RC.min_cross_recall,
            min_agreeing_frac=_RC.min_agreeing_frac)
from reconstruction.loops.instance_loops import disjoint_clusters
from segmentation.erase import _mask_obj_by_iid

xyz, fg, _pr, _pc = _load_ply_origins(OUT / "cleaned_cloud.ply")
res = json.loads((OUT / "segmentation_result.json").read_text())
inst_by_id = {int(i.get("instance_id", i["id"])): i for i in res["instances"]}
oid_of = _mask_obj_by_iid(OUT)
z = np.load(OUT / "seg_masks.npz", allow_pickle=True)
c2m = _mask_frame_lookup(OUT, z["frames"].tolist(), sorted({int(f) for f in np.unique(fg)}))

class C:
    dbscan_eps_m = 0.15; dbscan_min_samples = 20; cluster_min_points = 300

targets = [int(a) for a in sys.argv[1:]] or sorted(inst_by_id)
print(f"instancias a revisar: {len(targets)}\n")
for iid in targets:
    inst = inst_by_id.get(iid)
    if inst is None:
        continue
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    if len(gi) < 600:
        continue
    cl = disjoint_clusters(xyz[gi], C)
    if len(cl) < 2:
        continue
    a, b = gi[cl[0]], gi[cl[1]]
    sep = float(np.linalg.norm(xyz[a].mean(0) - xyz[b].mean(0)))
    fa = sorted({int(f) for f in np.unique(fg[a])})
    fb = sorted({int(f) for f in np.unique(fg[b])})
    print(f"--- instancia {iid} '{inst['label']}' | {len(a):,} + {len(b):,} pts | separacion {sep:.2f} m")
    ev = copy_evidence(OUT, SESS, iid, oid_of.get(iid), xyz[a], xyz[b], fa, fb,
                       cloud_to_mask=c2m, **_RKW)
    print(f"    self IoU A/B : {ev.get('self_iou_a', 0):.2f} / {ev.get('self_iou_b', 0):.2f}")
    if "cross_iou" in ev:
        print(f"    cross IoU    : {ev['cross_iou']:.2f} -> alineado {ev['cross_iou_aligned']:.2f}"
              f" | shift {ev.get('shift_px_median', 0):.0f} px"
              f" | dispersion {ev.get('shift_dispersion_px', 0):.0f} px"
              f" | {ev.get('n_cross_frames', 0)} frames")
    print(f"    VEREDICTO    : {ev['verdict']}")
    print(f"    {ev.get('reason','')}\n")
