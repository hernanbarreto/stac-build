"""Apply the co-visible split (camera-return rule + free-space test) to the LIVE, already-certified
segmentation, without re-projecting or re-certifying: split instances, recompute their OBBs, rewrite the
class bytes + class_map, rebuild the instance store and the octree (the class is baked into it)."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
t0 = time.time()
from config import cfg
from segmentation.pipeline import _split_covisible_components, _compute_obb, rebuild_instance_store
from segmentation.republish import write_classification
from reconstruction.gpu_cloud_clean import _read_ply
seg = json.loads((O / "segmentation_result.json").read_text())
v = _read_ply(str(O / "cleaned_cloud.ply"))
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
assert int(seg["total_points"]) == len(xyz), (seg["total_points"], len(xyz))
s, R, t = 1.0, np.eye(3), np.zeros(3)
ft = O / "floor_transform.npz"
if ft.exists():
    d = np.load(ft); s, R, t = float(d["s"]), d["R"], d["t"]
xyz_d = s * (xyz @ R.T) + t
P = np.loadtxt(O / "camera_poses.txt").reshape(-1, 4, 4)
F = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
C = s * (P[:, :3, 3] @ R.T) + t
cams = {f: C[k] for k, f in enumerate(F)}
sd = cfg["segmentation"]; vd = cfg["correction"]["visit_drift"]
inst = seg["instances"]
n0 = len(inst)
added = _split_covisible_components(inst, xyz_d, v["frame_global"].astype(np.int32), gap_m=float(sd["fragment_gap_m"]),
                                    min_points=int(vd["min_points"]), covis_share=float(sd["dedupe_overlap"]),
                                    cam_centre=cams, min_walk_m=float(vd["min_walk_m"]))
print(f"[split] {n0} -> {len(inst)} instances (+{added})", flush=True)
if added:
    for i in inst:
        m = np.asarray(i["globalIndices"], np.int64)
        if len(m) >= 4:
            i["obb"] = _compute_obb(xyz_d[m])
    seg["instances"] = inst
    (O / "segmentation_result.json").write_text(json.dumps(seg))
    write_classification(O, inst, len(xyz))
    print("[split] store:", rebuild_instance_store(O), flush=True)
    from potree_converter import convert_ply_to_potree
    tmp = O / "potree_split"
    shutil.rmtree(tmp, ignore_errors=True)
    if convert_ply_to_potree(S, force=True, ply_override=O / "cleaned_cloud.ply", potree_dir_override=tmp):
        old = O / "potree_old"; shutil.rmtree(old, ignore_errors=True)
        (O / "potree").rename(old); tmp.rename(O / "potree"); shutil.rmtree(old)
        print("[split] octree rebuilt with the new class bytes", flush=True)
for i in inst:
    if i.get("label") == "desk":
        print("[split] desk", i.get("instance_id"), len(i["globalIndices"]), [round(2 * x, 2) for x in i["obb"]["half_extents"]], flush=True)
print(f"[split] DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
