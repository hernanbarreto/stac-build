#!/usr/bin/env python3
"""READ-ONLY mesh A/B metrics (precision task, Phases B/C — reused by F).

Compares TSDF mesh variants of the SAME scene on the SAME zones:

  1. planar-patch RMS (mm)   — deterministic seed points are picked ONCE on the
     reference mesh (farthest-point sampling); every variant is measured on the
     SAME world-space patches: local RANSAC plane → inlier RMS. Lower = less
     surface noise.
  2. double-surface metric   — the onion statistic (phase_r): per patch, the
     1-D distribution of signed distances along the local plane normal is fit
     with 1 vs 2 Gaussians; ΔBIC = BIC1 − BIC2 > 0 ⇒ bimodal (a doubled
     surface), with the mode separation in metres. Reported as the fraction of
     bimodal patches + median separation among them.
  3. crop pairs              — each patch neighbourhood is exported as a small
     PLY per variant (same zone, same radius) for the visual verdict.
  4. context                 — vertex/triangle count, surface area.

Usage (da3 env — needs open3d):
    python tools/mesh_ab_metrics.py <ref_mesh> <variant_mesh> [...] \
        [--patches 12] [--radius 0.6] [--out DIR] [--labels a b c]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SERVER = Path(__file__).resolve().parent.parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

SEED = 7
RANSAC_THRESH = 0.02          # 2 cm inlier band for the local plane
BAND = 0.10                   # ±10 cm slab along the normal for the onion stat
MIN_PATCH_PTS = 400


def _load_trimesh(path: Path):
    """open3d first; trimesh fallback (draco-free .glb/.glb.orig, odd extensions)."""
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    if len(m.vertices):
        return np.asarray(m.vertices), np.asarray(m.triangles)
    pc = o3d.io.read_point_cloud(str(path))
    if len(pc.points):
        return np.asarray(pc.points), np.zeros((0, 3), np.int64)
    import trimesh
    t = trimesh.load(str(path), file_type="glb" if "glb" in path.name else None,
                     force="mesh")
    return np.asarray(t.vertices), np.asarray(t.faces)


def _load_pts(path: Path) -> np.ndarray:
    return _load_trimesh(Path(path))[0]


def _mesh_context(path: Path) -> dict:
    v, f = _load_trimesh(Path(path))
    area = None
    if len(f):
        tri = v[f]
        area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                             tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
    return {"vertices": int(len(v)), "triangles": int(len(f)), "area_m2": area}


def _fps_seeds(pts: np.ndarray, k: int, rng) -> np.ndarray:
    """Farthest-point sampling (deterministic) on a subsample."""
    sub = pts[:: max(1, len(pts) // 200000)]
    seeds = [sub[rng.integers(len(sub))]]
    d = np.linalg.norm(sub - seeds[0], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        seeds.append(sub[i])
        d = np.minimum(d, np.linalg.norm(sub - sub[i], axis=1))
    return np.stack(seeds)


def _patch(pts: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    d2 = np.sum((pts - centre) ** 2, axis=1)
    nb = pts[d2 < radius * radius]
    if len(nb) > 40000:
        nb = nb[:: len(nb) // 40000]
    return nb


def _ransac_plane(nb: np.ndarray, rng, iters: int = 200):
    """(normal, point, inlier_mask) or None."""
    best = None
    for _ in range(iters):
        tri = nb[rng.choice(len(nb), 3, replace=False)]
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        dist = np.abs((nb - tri[0]) @ nrm)
        inl = dist < RANSAC_THRESH
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), nrm, tri[0], inl)
    if best is None or best[0] < MIN_PATCH_PTS // 2:
        return None
    _, nrm, p0, inl = best
    # refine on inliers (least-squares plane)
    q = nb[inl]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    nrm = vt[2]
    return nrm, c, np.abs((nb - c) @ nrm) < RANSAC_THRESH


def _gauss_bic(x: np.ndarray) -> dict:
    """1- vs 2-component Gaussian fit on a 1-D sample (tiny EM, deterministic
    init at the ±1σ quantiles). Returns the onion statistic: ΔBIC = BIC1 − BIC2
    (>0 ⇒ two surfaces win) and the fitted mode separation."""
    n = x.size
    mu, sd = float(x.mean()), float(x.std() + 1e-9)
    ll1 = float(np.sum(-0.5 * np.log(2 * np.pi * sd * sd)
                       - 0.5 * ((x - mu) / sd) ** 2))
    bic1 = -2 * ll1 + 2 * np.log(n)
    m1, m2 = np.percentile(x, [25, 75])
    s1 = s2 = sd / 2 + 1e-9
    w = 0.5
    for _ in range(50):
        p1 = w * np.exp(-0.5 * ((x - m1) / s1) ** 2) / (s1 + 1e-12)
        p2 = (1 - w) * np.exp(-0.5 * ((x - m2) / s2) ** 2) / (s2 + 1e-12)
        r = p1 / (p1 + p2 + 1e-300)
        w = float(r.mean())
        if w < 1e-6 or w > 1 - 1e-6:
            break
        m1 = float(np.sum(r * x) / (r.sum() + 1e-12))
        m2 = float(np.sum((1 - r) * x) / ((1 - r).sum() + 1e-12))
        s1 = float(np.sqrt(np.sum(r * (x - m1) ** 2) / (r.sum() + 1e-12)) + 1e-9)
        s2 = float(np.sqrt(np.sum((1 - r) * (x - m2) ** 2) / ((1 - r).sum() + 1e-12)) + 1e-9)
    pmix = (w * np.exp(-0.5 * ((x - m1) / s1) ** 2) / (np.sqrt(2 * np.pi) * s1)
            + (1 - w) * np.exp(-0.5 * ((x - m2) / s2) ** 2) / (np.sqrt(2 * np.pi) * s2))
    ll2 = float(np.sum(np.log(pmix + 1e-300)))
    bic2 = -2 * ll2 + 5 * np.log(n)
    sep = abs(m2 - m1)
    minw = min(w, 1 - w)
    # a "second surface" needs real mass and real separation, not EM dust
    bimodal = (bic1 - bic2) > 10.0 and sep > 0.015 and minw > 0.10
    return {"bic_delta": float(bic1 - bic2), "separation_m": float(sep),
            "min_weight": float(minw), "bimodal": bool(bimodal)}


def measure_mesh(pts: np.ndarray, seeds: np.ndarray, radius: float, rng) -> list:
    rows = []
    for si, c in enumerate(seeds):
        nb = _patch(pts, c, radius)
        if len(nb) < MIN_PATCH_PTS:
            rows.append({"patch": si, "skipped": "too_few_points", "n": int(len(nb))})
            continue
        fit = _ransac_plane(nb, rng)
        if fit is None:
            rows.append({"patch": si, "skipped": "no_plane", "n": int(len(nb))})
            continue
        nrm, c0, inl = fit
        d = (nb - c0) @ nrm
        rms = float(np.sqrt(np.mean(d[inl] ** 2)) * 1000.0)
        slab = d[np.abs(d) < BAND]
        onion = _gauss_bic(slab) if slab.size >= MIN_PATCH_PTS else None
        rows.append({"patch": si, "n": int(len(nb)), "plane_rms_mm": rms,
                     "onion": onion})
    return rows


def _summ(rows: list) -> dict:
    rms = [r["plane_rms_mm"] for r in rows if "plane_rms_mm" in r]
    on = [r["onion"] for r in rows if r.get("onion")]
    bi = [o for o in on if o["bimodal"]]
    return {"patches_measured": len(rms),
            "plane_rms_mm_median": float(np.median(rms)) if rms else None,
            "plane_rms_mm_p90": float(np.percentile(rms, 90)) if rms else None,
            "bimodal_patches": len(bi),
            "bimodal_frac": (len(bi) / len(on)) if on else None,
            "bimodal_sep_m_median": (float(np.median([o["separation_m"] for o in bi]))
                                     if bi else None)}


def run(mesh_paths, labels, patches, radius, out_dir: Path, crops: bool = True) -> dict:
    rng = np.random.default_rng(SEED)
    ref_pts = _load_pts(Path(mesh_paths[0]))
    if len(ref_pts) < 10000:
        raise RuntimeError(f"reference mesh too small ({len(ref_pts)} vertices)")
    seeds = _fps_seeds(ref_pts, patches, rng)
    out = {"seeds": seeds.tolist(), "radius": radius, "variants": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, lab in zip(mesh_paths, labels):
        pts = _load_pts(Path(path))
        rng_v = np.random.default_rng(SEED)      # same RANSAC draws per variant
        rows = measure_mesh(pts, seeds, radius, rng_v)
        out["variants"][lab] = {"mesh": str(path), "context": _mesh_context(Path(path)),
                                "patches": rows, "summary": _summ(rows)}
        if crops:
            import open3d as o3d
            for si, c in enumerate(seeds):
                nb = _patch(pts, np.asarray(c), radius)
                if len(nb) < 50:
                    continue
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(nb)
                o3d.io.write_point_cloud(str(out_dir / f"crop_p{si:02d}_{lab}.ply"), pc)
    (out_dir / "mesh_ab.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--patches", type=int, default=12)
    ap.add_argument("--radius", type=float, default=0.6)
    ap.add_argument("--out", default="mesh_ab_out")
    ap.add_argument("--no-crops", action="store_true")
    a = ap.parse_args()
    labels = a.labels or [f"v{i}" for i in range(len(a.meshes))]
    if len(labels) != len(a.meshes):
        sys.exit("FAIL: --labels count must match meshes")
    res = run(a.meshes, labels, a.patches, a.radius, Path(a.out),
              crops=not a.no_crops)
    print(f"{'variant':<12} {'patches':>7} {'RMS med mm':>10} {'RMS p90':>8} "
          f"{'bimodal':>8} {'sep med m':>9} {'verts':>10}")
    for lab, v in res["variants"].items():
        s = v["summary"]
        print(f"{lab:<12} {s['patches_measured']:>7} "
              f"{s['plane_rms_mm_median'] if s['plane_rms_mm_median'] else float('nan'):>10.2f} "
              f"{s['plane_rms_mm_p90'] if s['plane_rms_mm_p90'] else float('nan'):>8.2f} "
              f"{(s['bimodal_frac'] if s['bimodal_frac'] is not None else float('nan')):>8.2f} "
              f"{(s['bimodal_sep_m_median'] or float('nan')):>9.3f} "
              f"{v['context']['vertices']:>10}")
    print(f"\nwritten: {Path(a.out) / 'mesh_ab.json'} (+ crop PLYs)")


if __name__ == "__main__":
    main()
