#!/usr/bin/env python3
"""Scan ↔ scan registration on shared invariant segments — CloudComPy side.

Runs INSIDE the CloudComPy310 environment (see run_cloudcompy_script.sh).
Input: a JSON spec written by server/fuse_scans.py:

    {"reference": {"pairs": {label: "path.npy", ...}, "cloud": "sub.npy"},
     "scans": [{"key": "...", "pairs": {label: "path.npy", ...},
                "cloud": "sub.npy"}],
     "params": {"overlap": 0.8, "ransac_epsilon": 0.02, "size_tol": 0.05}}

Per scan (generic — nothing here knows what the objects are):
  1. dominant PLANES of every paired object copy (RANSAC_SD, planes only):
     normal + center + support size — robust to how far the mask reached and
     to floaters (a bbox would not be);
  2. per-pair SIZE measure from the object's own geometry (plane-to-plane
     spreads, else robust radius) → size ratio scan/ref; the global scale is
     the ROBUST MEDIAN over pairs; pairs whose ratio deviates > size_tol are
     flagged `suspect` (segmentation/depth artefact) and excluded from the
     scale (kept for the pose, down-weighted by subsampling);
  3. coarse alignment (floors are already at Y=0 in both scans → XZ + yaw):
     yaw from the direction between pair centroids, translation from the
     centroid means; then ICP with adjustScale on the union of the paired
     points → similarity scan→reference;
  4. conditioning: the spread of plane normals across the pairs (a single
     direction leaves the lateral translation weak — reported, decided by
     the orchestrator's guards).

Output JSON: per scan the similarity (row-major 4x4, scale, rms) plus the
per-pair diagnostics. The orchestrator applies the guards, the symmetric
scale split and the held-out validation (it has scipy + the full clouds).
"""

import json
import sys
import numpy as np

import cloudComPy as cc
cc.initCC()
import cloudComPy.RANSAC_SD as rsd


def _cloud(P):
    c = cc.ccPointCloud("c")
    c.coordsFromNPArray_copy(np.ascontiguousarray(P, dtype=np.float64))
    return c


def dominant_planes(P, eps, max_planes=3):
    """[(normal, center, n_support)] largest first, planes only."""
    if len(P) < 200:
        return []
    c = _cloud(P)
    p = rsd.RansacParams()
    p.epsilon = float(eps)
    p.bitmapEpsilon = float(eps) * 2.5
    p.supportPoints = max(100, int(len(P) * 0.02))
    p.probability = 0.01
    for t in (rsd.RPT_CYLINDER, rsd.RPT_SPHERE, rsd.RPT_CONE, rsd.RPT_TORUS):
        p.setPrimEnabled(t, False)
    p.setPrimEnabled(rsd.RPT_PLANE, True)
    try:
        prims, clouds = rsd.computeRANSAC_SD(c, p)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for m, cl in zip(prims, clouds):
        if m is None or type(m).__name__ != "ccPlane":
            continue
        n = np.array(m.getNormal(), dtype=np.float64)
        n /= max(np.linalg.norm(n), 1e-12)
        out.append((n, np.array(m.getCenter(), dtype=np.float64), int(cl.size())))
    out.sort(key=lambda t: -t[2])
    return out[:max_planes]


def size_measure(P, _planes=None):
    """Object size from its own geometry: robust radius (median distance to
    the median point) — the SAME measure on both copies (mixing a plane
    spread on one side with a radius on the other gave door 0.128 on pccr,
    2026-09-06). Insensitive to mask extremes/floaters (median), only
    mildly to how far the mask reached."""
    c = np.median(P, axis=0)
    return float(np.median(np.linalg.norm(P - c, axis=1)))


def coarse_align(src_cents, dst_cents):
    """Floors at Y=0 → yaw + XZ translation from ≥2 centroid pairs."""
    S = np.asarray(src_cents); D = np.asarray(dst_cents)
    if len(S) >= 2:
        vs = S[1] - S[0]; vd = D[1] - D[0]
        if len(S) > 2:                       # use principal direction
            vs = S[-1] - S[0]; vd = D[-1] - D[0]
        yaw = np.arctan2(vd[2], vd[0]) - np.arctan2(vs[2], vs[0])
    else:
        yaw = 0.0
    ca, sa = np.cos(yaw), np.sin(yaw)
    R = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    t = D.mean(0) - R @ S.mean(0)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T, float(np.degrees(yaw))


def register_scan(ref_pairs, scan_pairs, params, log,
                  scan_cloud=None, ref_cloud=None, init=None):
    overlap = float(params.get("overlap", 0.8))
    eps = float(params.get("ransac_epsilon", 0.02))
    size_tol = float(params.get("size_tol", 0.05))
    labels = [l for l in scan_pairs if l in ref_pairs]
    pairs = []
    src_cents, dst_cents, normals = [], [], []
    for l in labels:
        A = np.load(ref_pairs[l]); B = np.load(scan_pairs[l])
        pa = dominant_planes(A, eps); pb = dominant_planes(B, eps)
        sa_ = size_measure(A, pa); sb_ = size_measure(B, pb)
        ratio = (sb_ / sa_) if sa_ > 1e-6 else float("nan")
        pairs.append({"label": l, "ref_points": int(len(A)),
                      "scan_points": int(len(B)),
                      "planes_ref": [[n.tolist(), c.tolist(), k] for n, c, k in pa],
                      "planes_scan": [[n.tolist(), c.tolist(), k] for n, c, k in pb],
                      "size_ref": sa_, "size_scan": sb_, "size_ratio": ratio})
        src_cents.append(np.median(B, axis=0)); dst_cents.append(np.median(A, axis=0))
        normals += [n for n, _c, _k in pa]
        log(f"  pair {l}: ref {len(A):,} pts / scan {len(B):,} pts, "
            f"planes {len(pa)}/{len(pb)}, size ratio {ratio:.4f}")
    if len(pairs) < (1 if init is not None else 2):
        return {"error": "fewer than 2 pairs", "pairs": pairs}
    # robust global size ratio + suspects
    ratios = np.array([p["size_ratio"] for p in pairs if np.isfinite(p["size_ratio"])])
    med = float(np.median(ratios)) if len(ratios) else 1.0
    for p in pairs:
        p["suspect"] = bool(not np.isfinite(p["size_ratio"])
                            or abs(p["size_ratio"] / med - 1.0) > size_tol)
    # conditioning: max angle between any two plane normals across pairs
    cond = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cos = abs(float(normals[i] @ normals[j]))
            cond = max(cond, float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    # coarse (all pairs) → ICP with scale on the union (suspects subsampled);
    # with an `init` (re-solve after culling a bad witness) the previous
    # solution replaces the coarse step — one good pair is enough then
    if init is not None:
        T0 = np.array(init, dtype=np.float64).reshape(4, 4)
        yaw = float(np.degrees(np.arctan2(T0[0, 2], T0[0, 0])))
        log(f"  initialised from the previous solution (culled re-solve)")
    else:
        T0, yaw = coarse_align(src_cents, dst_cents)
    rng = np.random.default_rng(0)
    srcs, dsts = [], []
    for p in pairs:
        A = np.load(ref_pairs[p["label"]]); B = np.load(scan_pairs[p["label"]])
        if p["suspect"]:
            B = B[rng.choice(len(B), max(1, len(B) // 4), replace=False)]
        srcs.append(B); dsts.append(A)
    S = np.vstack(srcs); D = np.vstack(dsts)
    S_h = np.c_[S, np.ones(len(S))] @ T0.T
    data = _cloud(S_h[:, :3]); model = _cloud(D)
    # scale from the pairs only when ≥2 (their separation fixes it); a
    # single compact object under ICP-with-scale COLLAPSES (box1 alone:
    # scale 0.70 at rms 0.8 cm, 2026-09-06) — then the pairs fix the pose
    # and the scene stage fixes the scale
    pair_scale = len(pairs) >= 2 and init is None
    res = cc.ICP(data, model, adjustScale=pair_scale, finalOverlapRatio=overlap,
                 maxIterationCount=int(params.get("icp_iters", 40)),
                 randomSamplingLimit=int(params.get("icp_samples", 100000)))
    if not pair_scale:
        log(f"  pairs ICP rigid (no scale — {len(pairs)} pair(s), "
            f"{'re-solve' if init is not None else 'single'}); scale from the scene")
    T1 = np.array(res.transMat.data(), dtype=np.float64).reshape(4, 4).T  # column-major → standard
    T = T1 @ T0
    # total scale so far = init's scale (a similarity when re-solving) × ICP's
    init_scale = float(np.cbrt(abs(np.linalg.det(T0[:3, :3])))) if init is not None else 1.0
    sc = float(res.finalScale) * init_scale if pair_scale else init_scale
    log(f"  coarse yaw {yaw:.2f}°, ICP scale {sc:.5f}, rms {res.finalRMS*100:.2f} cm, "
        f"{res.finalPointCount:,} pts, size-ratio median {med:.4f}, conditioning {cond:.1f}°")
    out = {"T_pairs": T.reshape(-1).tolist(), "T_coarse": T0.reshape(-1).tolist(),
           "scale_pairs": sc, "rms_pairs_cm": float(res.finalRMS * 100),
           "coarse_yaw_deg": yaw, "size_ratio_median": med,
           "conditioning_deg": cond, "pairs": pairs}
    # ── stage 2: SCENE refinement (USER 2026-09-06: "los pares calzan pero
    # el resto quedó expandido") — two neighbouring objects cannot fix the
    # global scale; the whole shared geometry (walls, floor, everything
    # both scans saw) can. ICP with scale on the full clouds, initialised
    # from the pair solution, partial overlap (complementary scans). The
    # orchestrator re-checks that the pairs still land (guard).
    if scan_cloud is not None and ref_cloud is not None:
        Sfull = np.load(scan_cloud); Dfull = np.load(ref_cloud)
        Sh = np.c_[Sfull, np.ones(len(Sfull))] @ T.T
        data2 = _cloud(Sh[:, :3]); model2 = _cloud(Dfull)
        res2 = cc.ICP(data2, model2, adjustScale=True,
                      finalOverlapRatio=float(params.get("scene_overlap", 0.5)),
                      maxIterationCount=int(params.get("scene_icp_iters", 60)),
                      randomSamplingLimit=int(params.get("scene_icp_samples", 300000)))
        T2 = np.array(res2.transMat.data(), dtype=np.float64).reshape(4, 4).T
        Tf = T2 @ T
        s2 = float(res2.finalScale)
        log(f"  scene refinement on {len(Sfull):,} vs {len(Dfull):,} pts: "
            f"extra scale {s2:.5f} → total {sc*s2:.5f}, rms {res2.finalRMS*100:.2f} cm, "
            f"{res2.finalPointCount:,} pts")
        out.update({"T": Tf.reshape(-1).tolist(), "scale": sc * s2,
                    "scale_scene_extra": s2,
                    "rms_cm": float(res2.finalRMS * 100)})
    else:
        out.update({"T": out["T_pairs"], "scale": sc, "rms_cm": out["rms_pairs_cm"]})
    return out


def main():
    spec = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    log = lambda m: print(f"[register] {m}", flush=True)
    result = {"scans": []}
    for sc in spec["scans"]:
        log(f"scan {sc['key']}:")
        r = register_scan(spec["reference"]["pairs"], sc["pairs"],
                          spec.get("params", {}), log,
                          scan_cloud=sc.get("cloud"),
                          ref_cloud=spec["reference"].get("cloud"),
                          init=sc.get("init"))
        r["key"] = sc["key"]
        result["scans"].append(r)
    json.dump(result, open(out_path, "w"), indent=1)
    log(f"done → {out_path}")


if __name__ == "__main__":
    main()
