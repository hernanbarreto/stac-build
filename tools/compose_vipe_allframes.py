#!/usr/bin/env python3
"""
Stage-1 all-frames validation: compose the cloud from ViPE poses + per-frame
calibrated depth (vipe_run/depth_calibrated/*.npz), keeping only well-calibrated
frames (robust outlier rejection). Reports coverage vs the DA3 keyframe cloud and
a coherence metric. Runs in the da3 env (calibrated depth is plain npz; no EXR).

    python tools/compose_vipe_allframes.py --session <src_default> [--subsample 8]
"""
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--subsample", type=int, default=8)
    ap.add_argument("--scale-lo", type=float, default=0.7)
    ap.add_argument("--scale-hi", type=float, default=1.7)
    ap.add_argument("--res-max", type=float, default=0.10)  # max affine residual (m)
    args = ap.parse_args()
    S = args.session
    cal_dir = S / "output" / "vipe_run" / "depth_calibrated"

    pose = np.load(S / "output" / "vipe_run" / "pose" / "frames.npz")
    vmats = pose["data"]; vrow = {int(f): r for r, f in enumerate(pose["inds"])}
    manifest = json.load(open(cal_dir / "calibration_manifest.json"))

    # robust outlier rejection on the per-frame affine fit
    good = {int(f): m for f, m in manifest.items()
            if args.scale_lo <= m["scale"] <= args.scale_hi and m["residual_m"] <= args.res_max}
    scales = np.array([m["scale"] for m in good.values()])
    g = float(np.median(scales)) if len(scales) else 1.0
    print(f"[compose] frames calibrados={len(manifest)}  bien-calibrados={len(good)} "
          f"(scale∈[{args.scale_lo},{args.scale_hi}], res≤{args.res_max}m)")
    print(f"[compose] escala global de poses g={g:.3f}")

    allX, allC, allF = [], [], []
    for fr in sorted(good):
        if fr not in vrow:
            continue
        z = np.load(cal_dir / f"{fr:06d}.npz")
        d = z["depth"].astype(np.float32); K = z["intrinsics"]
        c2w = vmats[vrow[fr]].copy(); c2w[:3, 3] *= g
        H, W = d.shape
        m = (d > 0.3) & (d < 8) & np.isfinite(d)
        sub = np.zeros_like(m); sub[::args.subsample, ::args.subsample] = True; m &= sub
        vv, uu = np.nonzero(m); zz = d[vv, uu].astype(np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Xc = np.stack([(uu - cx) / fx * zz, (vv - cy) / fy * zz, zz], 1)
        Xw = (c2w[:3, :3] @ Xc.T).T + c2w[:3, 3]
        allX.append(Xw.astype(np.float32))
        jpg = S / "frames" / f"{fr:06d}.jpg"
        if jpg.exists():
            import cv2
            im = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (W, H))
            allC.append((im[vv, uu].astype(np.float32) / 255.0))
        allF.append(np.full(len(Xw), fr, np.int32))
    X = np.concatenate(allX); F = np.concatenate(allF)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X.astype(np.float64)))
    if allC:
        pc.colors = o3d.utility.Vector3dVector(np.concatenate(allC))
    out = S / "output" / "tsdf" / "vipe_cloud_allframes.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pc)
    print(f"\n[compose] wrote {out}")
    print(f"[compose] all-frames cloud: {len(X):,} pts (subsample={args.subsample}), "
          f"frames usados={len(set(F.tolist()))}")
    print(f"[compose] bbox size = {np.round(X.max(0)-X.min(0),2)}  centroid={np.round(X.mean(0),2)}")

    # coverage vs DA3 keyframe cloud (downsample both to a common voxel, compare occupied voxels)
    cc = S / "output" / "cleaned_cloud.ply"
    if cc.exists():
        da3 = o3d.io.read_point_cloud(str(cc))
        vox = 0.05
        def occ(p):
            q = np.floor(np.asarray(p.points) / vox).astype(np.int64)
            return set(map(tuple, q))
        ov, od = occ(pc), occ(da3)
        print(f"\n[coverage] voxeles 5cm ocupados — ViPE-allframes={len(ov):,}  DA3-kf={len(od):,}")
        print(f"[coverage] solo-en-ViPE={len(ov-od):,}  solo-en-DA3={len(od-ov):,}  comunes={len(ov&od):,}")
        print(f"[coverage] ViPE/DA3 ratio = {len(ov)/max(len(od),1):.2f}  "
              f"(>1 ⇒ más cobertura espacial)")


if __name__ == "__main__":
    main()
