"""Point 3 (USER 2026-10-16: "debe sí o sí estar solidario con la nube"): every vertex of the scene mesh within
the band of the cloud moves onto the cloud's LOCAL surface (the plane of its nearest points, when that plane
faces the way the vertex does); vertices without support follow their neighbours (the displacement field
diffused over the mesh graph), so the completion keeps its shape but sits on the measured surface.
    python snap_mesh.py <glb_in> <glb_out> <band_m> [session output dir]
The band is the run's own voxel (chunk size / 32), never a chosen number."""
import sys, json, time, numpy as np, trimesh
sys.path.insert(0, "/workspace/stac-build/server")
from segmentation.perfect_object import _read_ply_fields
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
GLB, OUT, BAND = sys.argv[1], sys.argv[2], float(sys.argv[3])
SES = sys.argv[4] if len(sys.argv) > 4 else "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output"
K_NB, MIN_NB, DIFF_ITERS = 24, 8, 30
t0 = time.time()
sc = trimesh.load(GLB, force="scene", process=False)
f = _read_ply_fields(SES + "/cleaned_cloud.ply"); xyz = np.column_stack([f["x"], f["y"], f["z"]]).astype(np.float64)
Q = xyz[np.linspace(0, len(xyz) - 1, min(len(xyz), 8_000_000)).astype(np.int64)]; tq = cKDTree(Q)
rec = {"band_m": BAND, "k": K_NB, "min_neighbours": MIN_NB, "diffusion_iters": DIFF_ITERS, "parts": []}
for name, g in sc.geometry.items():
    V = np.asarray(g.vertices, np.float64); N = np.asarray(g.vertex_normals, np.float64); F = np.asarray(g.faces)
    d, idx = tq.query(V, k=K_NB, distance_upper_bound=BAND, workers=6)
    ok = np.isfinite(d); n_ok = ok.sum(1)
    disp = np.zeros_like(V); supported = np.zeros(len(V), bool)
    cand = np.flatnonzero(n_ok >= MIN_NB)
    for i in cand:
        pts = Q[idx[i, ok[i]]]; c = pts.mean(0); U, S, Vt = np.linalg.svd(pts - c, full_matrices=False); nrm = Vt[2]
        if S[2] > 0.5 * S[1]:                                   # not a surface patch (a volume of flyers)
            continue
        if abs(nrm @ N[i]) < 0.5:                               # a wall's vertex at the floor: not its surface
            continue
        h = (c - V[i]) @ nrm                                     # signed distance of the vertex to the local plane
        if abs(h) <= BAND:
            disp[i] = h * nrm; supported[i] = True
    # the unsupported vertices follow their neighbours: diffuse the displacement over the mesh graph
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]])
    W = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    deg = np.asarray(W.sum(1)).ravel(); deg[deg == 0] = 1
    D = disp.copy()
    for _ in range(DIFF_ITERS):
        avg = (W @ D) / deg[:, None]
        D = np.where(supported[:, None], disp, avg)
    g.vertices = V + D
    rec["parts"].append({"part": name, "vertices": int(len(V)), "supported": int(supported.sum()),
                         "moved_p50_cm": float(np.median(np.linalg.norm(disp[supported], axis=1)) * 100) if supported.any() else None,
                         "moved_p90_cm": float(np.percentile(np.linalg.norm(disp[supported], axis=1), 90) * 100) if supported.any() else None})
    print(f"  {name}: {supported.sum():,}/{len(V):,} supported; moved p50 {rec['parts'][-1]['moved_p50_cm']} cm", flush=True)
sc.export(OUT)
json.dump(rec, open(OUT + ".snap.json", "w"), indent=2)
print(f"snapped → {OUT} ({time.time()-t0:.0f}s)")
