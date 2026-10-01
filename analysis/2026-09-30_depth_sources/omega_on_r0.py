"""Omega's own geometry on F5-R0's poses, WITHOUT touching it inside a chunk -> epoch 3.
USER 2026-09-30: "un omega con poses de F5 sin romper su geometría".

Per Omega chunk (maplong_run/_tmp_results_aligned/chunk_c.npy: world_points, conf, the chunk's
c2w, images — exactly what Omega produced), ONE Sim(3) = Umeyama from the chunk's camera centres
to R0's (output/precision/f5_r0_camera_poses.txt, Omega's camera fixed). The whole chunk moves
with it: nothing inside a chunk is re-projected, re-scaled or re-shot. Each keyframe comes from
the chunk where it sits most central. Per frame the residual (R0 pose vs the Sim(3)-moved Omega
pose) is REPORTED against Omega's own intra-chunk held-out.
Confidence: the reconstruction's own PLY gate per chunk (conf_percentile and the min-max
conf_min_norm floor, the stricter wins). Then the cloud stage's voxel + SOR, octree, _epoch_3.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
EPOCH = 3
DST, TMP = O / f"_epoch_{EPOCH}", O / "_tx_omega_r0"
CH = O / "maplong_run" / "_tmp_results_aligned"


def log(m):
    print(f"[omega-r0 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def sim3_from_poses(Psrc, Pdst):
    """ONE Sim(3) taking the source cameras onto the destination ones, using ORIENTATIONS
    (chordal mean of R_dst R_src^T) and then scale + translation of the centres by least
    squares — a walk is nearly a line, so centres alone leave the roll about it free."""
    M = sum(Pdst[i, :3, :3] @ Psrc[i, :3, :3].T for i in range(len(Psrc)))
    U, _, Vt = np.linalg.svd(M); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt)); R = U @ D @ Vt
    A, B = Psrc[:, :3, 3], Pdst[:, :3, 3]; ma, mb = A.mean(0), B.mean(0)
    RA = (R @ (A - ma).T).T; s = float((RA * (B - mb)).sum() / (RA * RA).sum())
    return s, R, mb - s * R @ ma


t0 = time.time()
from config import cfg
simple = cfg["reconstruction"]["simple"]
pct, floor_norm = float(simple["conf_percentile"]), float(simple["conf_min_norm"])
plan = json.loads((O / "chunk_plan.json").read_text())["chunk_ranges"]
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
R0 = np.loadtxt(O / "precision" / "f5_r0_camera_poses.txt").reshape(-1, 4, 4)
FIN = np.loadtxt(O / "maplong_run" / "camera_poses.txt").reshape(-1, 4, 4)   # epoch 0 final poses
N = len(frames)
owner = {i: min((c for c, (a, b) in enumerate(plan) if a <= i < b), key=lambda c: abs(i - (plan[c][0] + plan[c][1] - 1) / 2))
         for i in range(N)}
log(f"{len(plan)} Omega chunks {plan}; confidence gate p{pct:g} + min-max {floor_norm:g}")
for p in (TMP, DST):
    if p.exists(): shutil.rmtree(p)
(TMP / "chunks").mkdir(parents=True)
from precision.epoch0_cloud import _write_ply_xyzrgb
poses_out = np.zeros((N, 4, 4)); res_t, res_r, rows = [], [], []
for c, (a, b) in enumerate(plan):
    d = np.load(CH / f"chunk_{c}.npy", allow_pickle=True).item()
    k = np.arange(a, b)
    s, R, t = sim3_from_poses(FIN[k], R0[k])
    moved_C = (s * (R @ FIN[k, :3, 3].T)).T + t
    Rm = np.einsum("ij,njk->nik", R, FIN[k, :3, :3])
    dt = np.linalg.norm(moved_C - R0[k, :3, 3], axis=1)
    dr = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Rm, R0[k, :3, :3]) - 1) / 2, -1, 1)))
    res_t.append(dt); res_r.append(dr)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    log(f"chunk {c} [{a},{b}): Sim(3) s {s:.4f} rot {ang:.2f} deg |t| {np.linalg.norm(t)*100:.1f} cm; residual vs R0: "
        f"centres median {np.median(dt)*100:.1f} cm max {dt.max()*100:.1f} cm, rotation median {np.median(dr):.2f} deg max {dr.max():.2f}")
    conf = d["world_points_conf"]
    v = conf[np.isfinite(conf) & (conf > 0)]
    thr = max(np.percentile(v, pct), v.min() + floor_norm * (v.max() - v.min()))
    xyz_l, rgb_l, fg_l, pr_l, pc_l, cf_l = [], [], [], [], [], []
    for jj, i in enumerate(k):
        if owner[i] != c:
            continue
        P = np.eye(4); P[:3, :3] = Rm[jj]; P[:3, 3] = moved_C[jj]; poses_out[i] = P
        m = np.isfinite(conf[jj]) & (conf[jj] >= thr) & (d["depth"][jj] > 0)
        r_, c_ = np.nonzero(m)
        K = d["intrinsic"][jj].astype(np.float64); z = d["depth"][jj][r_, c_].astype(np.float64)
        Xc = np.stack([(c_ - K[0, 2]) / K[0, 0] * z, (r_ - K[1, 2]) / K[1, 1] * z, z], 1)
        X = Xc @ FIN[i][:3, :3].T + FIN[i][:3, 3]                     # epoch 0's own placement
        xyz_l.append(((s * (R @ X.T)).T + t).astype(np.float32))
        img = d["images"][jj]
        rgb_l.append((np.clip(np.moveaxis(img, 0, -1)[r_, c_], 0, 1) * 255).astype(np.uint8))
        fg_l.append(np.full(len(r_), frames[i], np.int32)); pr_l.append(r_.astype(np.int16)); pc_l.append(c_.astype(np.int16))
        cf_l.append(conf[jj][r_, c_].astype(np.float32))
    xyz = np.concatenate(xyz_l); rows.append(len(xyz))
    _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{c:03d}.ply", xyz, np.concatenate(rgb_l))
    np.savez(TMP / "chunks" / f"chunk_{c:03d}_origins.npz", frame_global=np.concatenate(fg_l),
             pixel_row=np.concatenate(pr_l), pixel_col=np.concatenate(pc_l), confidence=np.concatenate(cf_l))
    del d
res_t, res_r = np.concatenate(res_t), np.concatenate(res_r)
log(f"ALL chunks: residual centres median {np.median(res_t)*100:.1f} cm p90 {np.percentile(res_t,90)*100:.1f} cm; "
    f"rotation median {np.median(res_r):.2f} deg p90 {np.percentile(res_r,90):.2f}; {sum(rows):,} raw points")
unc = O / "maplong_run" / "uncertainty.json"
if unc.exists():
    log(f"Omega's own intra-chunk evidence (uncertainty.json keys): {list(json.loads(unc.read_text()).keys())[:8]}")

from precision.epoch0_cloud import clean_cmd
cleaned = TMP / "cleaned_cloud.ply"
r = subprocess.run(clean_cmd(cfg, TMP / "chunks", cleaned), cwd="/workspace/stac-build/server", capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "✅" in ln and ("→" in ln or "Merged" in ln): log(ln.strip())
if r.returncode or not cleaned.exists():
    log("cleaner failed " + r.stderr[-800:]); sys.exit(1)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
np.savetxt(DST / "camera_poses.txt", poses_out.reshape(N, -1))
(DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
    "note": "Omega's own chunks, each moved by ONE Sim(3) onto F5-R0 poses (Omega camera fixed); outside the pipeline 2026-09-30",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"DONE {json.loads((DST / 'potree' / 'metadata.json').read_text())['points']:,} pts -> epoch {EPOCH} in {(time.time()-t0)/60:.1f} min")
