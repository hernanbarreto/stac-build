"""Re-judge the splits the run already made: parent vs child, by reprojection."""
import sys, json
sys.path.insert(0, "/workspace/stac-build/server")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path

SESS = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
OUT = SESS / "output"
from segmentation.pipeline import _load_ply_origins, _mask_frame_lookup
from segmentation.erase import _mask_obj_by_iid
from reconstruction.loops.reprojection import copy_evidence
from reconstruction.loops.config import load_loops_config as _llc
_RC = _llc().loops.reprojection
_RKW = dict(dilate_px=_RC.dilate_px, max_frames=_RC.max_frames,
            min_self_recall=_RC.min_self_recall, min_cross_recall=_RC.min_cross_recall,
            min_agreeing_frac=_RC.min_agreeing_frac)

xyz, fg, _pr, _pc = _load_ply_origins(OUT / "cleaned_cloud.ply")
res = json.loads((OUT / "segmentation_result.json").read_text())
by = {int(i.get("instance_id", i["id"])): i for i in res["instances"]}
oid_of = _mask_obj_by_iid(OUT)
z = np.load(OUT / "seg_masks.npz", allow_pickle=True)
c2m = _mask_frame_lookup(OUT, z["frames"].tolist(), sorted({int(f) for f in np.unique(fg)}))

PAIRS = [(111,214,1.24,"silla"), (157,222,1.04,"escritorio"), (158,223,1.19,"escritorio"),
         (159,224,1.29,"escritorio"), (122,216,8.67,"ducto"), (169,225,8.72,"ducto"),
         (150,220,6.14,"viga"), (125,219,4.40,"pared")]

def pts_frames(iid):
    i = by.get(iid)
    if i is None: return None, None
    gi = np.asarray(i["globalIndices"], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    return xyz[gi], sorted({int(f) for f in np.unique(fg[gi])})

print(f"{'par':<12}{'sep':>7}  {'objeto':<12}{'selfA':>6}{'selfB':>6}{'cross':>7}{'align':>7} frames  veredicto")
print("-" * 88)
for a, b, sep, what in PAIRS:
    PA, FA = pts_frames(a); PB, FB = pts_frames(b)
    if PA is None or PB is None:
        print(f"{a}->{b:<8}{sep:>6.2f}m  {what:<12}  (falta una instancia)"); continue
    ev = copy_evidence(OUT, SESS, a, oid_of.get(a), PA, PB, FA, FB,
                       oid_b=oid_of.get(b), cloud_to_mask=c2m, **_RKW)
    print(f"{a}->{b:<8}{sep:>6.2f}m  {what:<12}"
          f"{ev.get('self_recall_a',0):>6.2f}{ev.get('self_recall_b',0):>6.2f}"
          f"{ev.get('cross_recall',0):>7.2f}{ev.get('cross_recall_aligned',0):>7.2f}"
          f"{ev.get('agreeing_frames',0):>3}/{ev.get('n_cross_frames',0):<3} {ev['verdict']}")
