"""The pure classical baseline's model (COLMAP SfM, no prior) → the session's RAW frame by ONE Sim(3) over the camera
centres (Umeyama, matched by keyframe name; cameras beyond 3× the median residual dropped once and the fit redone),
so the OpenMVS mesh built on it lands where the viewer draws meshes and measure_mesh.py compares it with the
certified cloud on equal terms. The residual after alignment IS the two trajectories' disagreement, recorded.
    python align_colmap.py <session src_default> <colmap TXT model dir> <undistorted images dir> <out work dir>
"""
import sys, json, shutil, numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from scipy.spatial.transform import Rotation
from correction.session import load_session

SES, MOD, IMG, OUT = (Path(a) for a in sys.argv[1:5])
s = load_session(SES / "output")
ours = {f"{int(f):06d}.jpg": np.asarray(p, np.float64) for f, p in zip(s.frames, s.poses)}

def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); Xs, Xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Xd.T @ Xs / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    R = U @ S @ Vt; sc = float(np.trace(np.diag(D) @ S) / ((Xs ** 2).sum() / len(src)))
    return sc, R, mu_d - sc * R @ mu_s

# images.txt: a pose line (10 fields, name last) followed by its POINTS2D line, kept verbatim
hdr, recs, pts_lines = [], [], []
lines = (MOD / "images.txt").read_text().splitlines(); it = iter(lines)
for ln in it:
    if ln.startswith("#"): hdr.append(ln); continue
    if not ln.strip(): continue
    tok = ln.split(); recs.append(tok); pts_lines.append(next(it, ""))
names = [r[9] for r in recs]
w2c = []
for r in recs:
    q = np.array([float(v) for v in r[1:5]]); t = np.array([float(v) for v in r[5:8]])
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; w2c.append(M)
w2c = np.array(w2c); c2w = np.linalg.inv(w2c)
common = [i for i, n in enumerate(names) if n in ours]
src = c2w[common, :3, 3]; dst = np.array([ours[names[i]][:3, 3] for i in common])
sc, R, t = umeyama(src, dst); res = np.linalg.norm((sc * (R @ src.T)).T + t - dst, axis=1)
keep = res <= 3 * np.median(res)
if keep.sum() >= 3 and keep.sum() < len(res):
    sc, R, t = umeyama(src[keep], dst[keep]); res = np.linalg.norm((sc * (R @ src.T)).T + t - dst, axis=1)
rot_err = []
for j, i in enumerate(common):
    Rc = R @ c2w[i, :3, :3]; Ro = ours[names[i]][:3, :3]
    rot_err.append(np.degrees(np.arccos(np.clip((np.trace(Ro.T @ Rc) - 1) / 2, -1, 1))))
# the transformed model: x' = s R x + t;  C' = sRC + t;  R_w2c' = R_w2c Rᵀ;  t' = −R_w2c' C'
sp = OUT / "sparse"; im = OUT / "images"
if OUT.exists(): shutil.rmtree(OUT)
sp.mkdir(parents=True); im.mkdir()
shutil.copy(MOD / "cameras.txt", sp / "cameras.txt")
out = hdr[:]
for r, p2 in zip(recs, pts_lines):
    i = recs.index(r); C = sc * R @ c2w[i, :3, 3] + t; Rw = w2c[i, :3, :3] @ R.T; tw = -Rw @ C
    q = Rotation.from_matrix(Rw).as_quat()
    out.append(f"{r[0]} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f} {r[8]} {r[9]}"); out.append(p2)
    (im / r[9]).symlink_to((IMG / r[9]).resolve())
(sp / "images.txt").write_text("\n".join(out) + "\n")
n_pts = 0
with (MOD / "points3D.txt").open() as fi, (sp / "points3D.txt").open("w") as fo:
    for ln in fi:
        if ln.startswith("#") or not ln.strip(): fo.write(ln); continue
        tok = ln.split(); p = sc * R @ np.array([float(v) for v in tok[1:4]]) + t; n_pts += 1
        fo.write(f"{tok[0]} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} " + " ".join(tok[4:]) + "\n")
rec = {"session": str(SES), "model": str(MOD), "registered_images": len(recs), "session_keyframes": len(ours), "matched": len(common),
       "used_in_fit": int(keep.sum()), "scale": sc, "points": n_pts,
       "trajectory_residual_m": {"rms": float(np.sqrt((res ** 2).mean())), "median": float(np.median(res)), "p90": float(np.percentile(res, 90)), "max": float(res.max())},
       "rotation_residual_deg": {"median": float(np.median(rot_err)), "p90": float(np.percentile(rot_err, 90))}}
json.dump(rec, open(OUT / "align.json", "w"), indent=1)
print(f"{len(recs)} registered of {len(ours)} keyframes, {n_pts:,} points; Sim(3) scale {sc:.4f}; trajectory residual rms {rec['trajectory_residual_m']['rms']*100:.1f} cm, "
      f"median {rec['trajectory_residual_m']['median']*100:.1f} cm, max {res.max()*100:.1f} cm; rotation median {np.median(rot_err):.2f}° → {sp}")
