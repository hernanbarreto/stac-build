"""One record of a scene mesh against the session: area within bands of the cloud, the cloud against the
mesh surface, the cameras' height over the floor under them (cloud vs mesh), the ceiling, named objects.
    python measure_mesh.py <glb> [session output dir]"""
import sys, json, time, numpy as np, trimesh
sys.path.insert(0, "/workspace/stac-build/server")
from segmentation.perfect_object import _read_ply_fields
from correction.session import load_session
from scipy.spatial import cKDTree
GLB = sys.argv[1]; SES = sys.argv[2] if len(sys.argv) > 2 else "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output"
t0 = time.time()
sc = trimesh.load(GLB, force="scene", process=False); parts = sc.dump(concatenate=False)
V = np.concatenate([p.vertices for p in parts]); A = np.zeros(len(V)); off = 0; P, Nn = [], []
for p in parts:
    np.add.at(A, p.faces.ravel() + off, np.repeat(p.area_faces / 3.0, 3)); off += len(p.vertices)
    s_, fi = trimesh.sample.sample_surface(p, max(1000, int(p.area / 0.02 ** 2))); P.append(s_); Nn.append(p.face_normals[fi])
P = np.concatenate(P); Nn = np.concatenate(Nn)
print(f"{GLB.split('/')[-1]}: {len(parts)} parts, {sum(len(p.faces) for p in parts):,} faces, area {A.sum():.0f} m², bounds y {V[:,1].min():+.2f}..{V[:,1].max():+.2f}")
f = _read_ply_fields(SES + "/cleaned_cloud.ply"); xyz = np.column_stack([f["x"], f["y"], f["z"]]).astype(np.float64)
Q = xyz[np.linspace(0, len(xyz) - 1, min(len(xyz), 6_000_000)).astype(np.int64)]; tq = cKDTree(Q)
d, _ = tq.query(V, k=1, workers=6); tot = A.sum()
print("mesh area within |d| of the cloud: " + "  ".join(f"<{hi*100:.0f}cm {A[d<hi].sum()/tot*100:.0f}%" for hi in (0.02, 0.05, 0.10, 0.20, 0.50)))
ts = cKDTree(P); dc, _ = ts.query(xyz[::4], k=1, workers=6)
print(f"cloud → mesh surface: p50 {np.median(dc)*100:.1f} cm p90 {np.percentile(dc,90)*100:.1f} cm; cloud within " + "  ".join(f"{hi*100:.0f}cm {np.mean(dc<hi)*100:.0f}%" for hi in (0.02, 0.05, 0.10, 0.20)))
s = load_session(SES); cams = np.asarray(s.poses, np.float64)[:, :3, 3]
tc = cKDTree(xyz[:, [0, 2]]); tm = cKDTree(P[:, [0, 2]]); up = Nn[:, 1] > 0.8; down = Nn[:, 1] < -0.8
hc, hm, cc_, cm_ = [], [], [], []
for k in range(0, len(cams), 4):
    c = cams[k]; ic = tc.query_ball_point(c[[0, 2]], 0.8); im = np.array(tm.query_ball_point(c[[0, 2]], 0.8))
    if len(ic) < 200 or len(im) < 50: continue
    yc = xyz[ic, 1]; ym = P[im, 1]; um = up[im] & (ym < c[1]); dm = down[im] & (ym > c[1])
    hc.append(c[1] - np.percentile(yc, 3)); hm.append(c[1] - np.percentile(ym[um], 5) if um.sum() > 50 else np.nan)
    cc_.append(np.percentile(yc, 99) - c[1]); cm_.append(np.percentile(ym[dm], 95) - c[1] if dm.sum() > 50 else np.nan)
hc, hm, cc_, cm_ = map(np.array, (hc, hm, cc_, cm_)); ok = np.isfinite(hm); ok2 = np.isfinite(cm_)
print(f"camera over the floor under it ({ok.sum()} kf): cloud {np.median(hc[ok]):.2f} m | mesh {np.median(hm[ok]):.2f} m; mesh floor lower by {np.median(hm[ok]-hc[ok])*100:+.0f} cm (p10/p90 {np.percentile(hm[ok]-hc[ok],10)*100:+.0f}/{np.percentile(hm[ok]-hc[ok],90)*100:+.0f})")
print(f"ceiling over the camera ({ok2.sum()} kf): cloud {np.median(cc_[ok2]):.2f} m | mesh {np.median(cm_[ok2]):.2f} m; mesh higher by {np.median(cm_[ok2]-cc_[ok2])*100:+.0f} cm")
seg = json.loads(open(SES + "/segmentation_result.json").read())
want = {25, 109, 141, 273, 338, 411, 599, 1108}
for inst in seg["instances"]:
    if inst.get("id") in want:
        gi = np.asarray(inst["globalIndices"], np.int64); pts = xyz[gi[::max(1, len(gi) // 20000)]]
        dd, j = ts.query(pts, k=1, workers=6)
        print(f"  {inst['label']:>18s}#{inst['id']:<5d} to mesh p50 {np.median(dd)*100:5.1f} cm; within 5 cm {np.mean(dd<0.05)*100:3.0f}%; mean shift {np.round((P[j]-pts).mean(0)*100,0).astype(int).tolist()} cm")
print(f"{time.time()-t0:.0f}s")
