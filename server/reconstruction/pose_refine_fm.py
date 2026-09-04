"""
FASE 3 of the DINOv3 plan (USER 2026-09-04): FEATURE-METRIC pose refinement —
the third family after point-to-plane (in production, self-gated) and
photometric (LOST: RMS +11%, PGSR variant).

DINOv3 patch features are matched between keyframe pairs (sequential
neighbours + measured REVISITS by camera proximity — the loop that SALAD only
detects, here it is measured); each match links two cells whose 3D positions
come from each frame's OWN measured points (per-point pixel provenance — the
fused cloud would give zero signal). The matched blocks feed the SAME
point-to-plane pose-graph solver, odometry prior and held-out gate as
``pose_refine`` — applied only when the held-out disagreement improves by
``min_gain``, else identity. Points move with their frames, poses follow
(backup ``camera_poses.txt.prefm``).

Runs at the same pipeline stage as pose_refine (chunk PLYs still on disk).

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from reconstruction.pose_refine import (edge_errors, normal_grid,
                                        robust_rigid, solve_pose_graph)
from reconstruction.dino_features import (FeatureCache,
                                          extract_session_features,
                                          load_session_cameras)

logger = logging.getLogger(__name__)


def _mutual_matches(fa: np.ndarray, fb: np.ndarray, min_cos: float,
                    max_matches: int = 800):
    """Mutual nearest neighbours between two (N,D)/(M,D) unit feature sets.
    Returns (ia, ib, cos) arrays. Torch/GPU when available."""
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        A = torch.from_numpy(fa).to(dev)
        B = torch.from_numpy(fb).to(dev)
        S = A @ B.T
        ab = S.argmax(dim=1)
        ba = S.argmax(dim=0)
        ia = torch.arange(len(fa), device=dev)
        mutual = ba[ab] == ia
        cos = S[ia, ab]
        keep = mutual & (cos >= min_cos)
        ia = ia[keep].cpu().numpy()
        ib = ab[keep].cpu().numpy()
        cos = cos[keep].cpu().numpy()
    except Exception:  # noqa: BLE001 — numpy fallback
        S = fa @ fb.T
        ab = S.argmax(axis=1)
        ba = S.argmax(axis=0)
        ia_all = np.arange(len(fa))
        mutual = ba[ab] == ia_all
        cos = S[ia_all, ab]
        keep = mutual & (cos >= min_cos)
        ia, ib, cos = ia_all[keep], ab[keep], cos[keep]
    if len(ia) > max_matches:
        top = np.argsort(-cos)[:max_matches]
        ia, ib, cos = ia[top], ib[top], cos[top]
    return ia, ib, cos


def run(output_dir: Path, frames_dir: Optional[Path] = None,
        cfg: Optional[dict] = None, log=logger.info) -> int:
    """Feature-metric refinement of all keyframe poses. Returns #frames moved
    (0 = gate rejected or inputs missing)."""
    from plyfile import PlyData, PlyElement

    cfg = cfg or {}
    min_cos = float(cfg.get("match_min_cos", 0.60))
    min_matches = int(cfg.get("min_matches", 40))
    pair_max_med_m = float(cfg.get("pair_max_med_m", 0.30))
    seq_offsets = tuple(cfg.get("seq_offsets", (1, 2, 4, 8, 16)))
    revisit_radius = float(cfg.get("revisit_radius_m", 2.5))
    revisit_min_gap = int(cfg.get("revisit_min_gap", 30))
    max_revisits = int(cfg.get("max_revisit_pairs", 400))
    odo_weight = float(cfg.get("odo_weight", 2.0))
    leash_weight = float(cfg.get("leash_weight", 0.1))
    min_gain = float(cfg.get("min_gain", 0.10))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))

    output_dir = Path(output_dir)
    if frames_dir is None:
        frames_dir = output_dir.parent / "frames"
    pp = output_dir / "camera_poses.txt"
    chunk_files = [p for p in sorted(output_dir.glob("chunk_*.ply"))
                   if not p.stem.startswith(("chunk_997", "chunk_998",
                                             "chunk_999"))]
    if not pp.exists() or not chunk_files:
        raise RuntimeError(
            "[pose-fm] poses or chunk PLYs missing — this stage runs before "
            "CloudCompy merges; missing inputs mean a broken pipeline "
            "(USER 2026-09-04: nothing fails silently)")

    poses, frames, Ks = load_session_cameras(output_dir)
    n = len(frames)
    fidx = {f: k for k, f in enumerate(frames)}
    extract_session_features(output_dir, frames_dir, cfg, log=log)
    fc = FeatureCache(output_dir)
    hp, wp = fc.hp, fc.wp

    # ── per-frame world grids on the FEATURE patch lattice, from each
    # frame's OWN points (pixel provenance — no projection, no pose bias) ──
    plys = []
    acc: Dict[int, np.ndarray] = {}
    cnt: Dict[int, np.ndarray] = {}
    Hg = Wg = None
    for p in chunk_files:
        org = output_dir / f"{p.stem}_origins.npz"
        if not org.exists():
            log(f"[pose-fm] {org.name} missing — skipped")
            return 0
        z = np.load(org)
        pd = PlyData.read(str(p))
        v = np.array(pd["vertex"].data)
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        fg = z["frame_global"].astype(np.int64)
        if Hg is None and "scaled_resolution" in z.files:
            Hg, Wg = (int(x) for x in z["scaled_resolution"])
        pr = z["pixel_row"].astype(np.int64)
        pc_ = z["pixel_col"].astype(np.int64)
        plys.append((p, v, fg))
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None or not fc.has(int(f)):
                continue
            m = fg == f
            r = np.clip(pr[m] * hp // max(Hg or (pr.max() + 1), 1), 0, hp - 1)
            c = np.clip(pc_[m] * wp // max(Wg or (pc_.max() + 1), 1),
                        0, wp - 1)
            cell = r * wp + c
            a = acc.setdefault(k, np.zeros((hp * wp, 3)))
            q = cnt.setdefault(k, np.zeros(hp * wp))
            np.add.at(a, cell, xyz[m])
            np.add.at(q, cell, 1)
    if Hg is None:
        log("[pose-fm] no scaled_resolution in origins — skipped")
        return 0
    world = {}
    valid = {}
    for k, a in acc.items():
        q = cnt[k]
        ok = q > 0
        g = np.zeros((hp * wp, 3))
        g[ok] = a[ok] / q[ok, None]
        world[k] = g.reshape(hp, wp, 3)
        valid[k] = ok.reshape(hp, wp)
    log(f"[pose-fm] {len(world)} frames gridded on the {hp}x{wp} "
        f"feature lattice")

    # ── pairs: sequential + measured revisits by camera proximity ──
    ctr = poses[:, :3, 3]
    pairs = [(i, i + o) for o in seq_offsets for i in range(n - o)
             if i in world and (i + o) in world]
    rev = []
    from scipy.spatial import cKDTree
    kd = cKDTree(ctr)
    for i, js in enumerate(kd.query_ball_point(ctr, revisit_radius)):
        for j in js:
            if j - i >= revisit_min_gap and i in world and j in world:
                rev.append((i, j))
    if len(rev) > max_revisits:
        rng = np.random.default_rng(0)
        rev = [rev[t] for t in rng.choice(len(rev), max_revisits,
                                          replace=False)]
    pairs += rev
    log(f"[pose-fm] {len(pairs)} pairs ({len(rev)} measured revisits)")

    # ── feature matching → point-to-plane edges ──
    feat_flat: Dict[int, np.ndarray] = {}

    def _feats(k):
        if k not in feat_flat:
            if len(feat_flat) > 64:
                feat_flat.pop(next(iter(feat_flat)))
            feat_flat[k] = fc.grid(frames[k]).reshape(hp * wp, fc.dim)
        return feat_flat[k]

    normals = {k: normal_grid(world[k], valid[k], step=1)
               for k in world}
    edges: List[tuple] = []
    n_dropped = 0
    for (a, b) in pairs:
        va = valid[a].reshape(-1)
        vb = valid[b].reshape(-1)
        ia_v = np.flatnonzero(va)
        ib_v = np.flatnonzero(vb)
        if len(ia_v) < min_matches or len(ib_v) < min_matches:
            continue
        ia, ib, _cos = _mutual_matches(_feats(a)[ia_v], _feats(b)[ib_v],
                                       min_cos)
        if len(ia) < min_matches:
            continue
        P = world[a].reshape(-1, 3)[ia_v[ia]]
        Q = world[b].reshape(-1, 3)[ib_v[ib]]
        N = normals[b].reshape(-1, 3)[ib_v[ib]]
        nok = (np.linalg.norm(N, axis=1) > 0.5) \
            & np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=1) \
            & np.isfinite(N).all(axis=1)
        if nok.sum() < min_matches:
            continue
        P, Q, N = P[nok], Q[nok], N[nok]
        rr = robust_rigid(P, Q)
        if rr is None or rr[1] > pair_max_med_m:
            n_dropped += 1
            continue
        edges.append((a, b, P, Q, N))
    log(f"[pose-fm] {len(edges)} edges accepted ({n_dropped} pairs dropped "
        f"by the rigid-fit gate)")
    if len(edges) < 10:
        log("[pose-fm] too few edges — NOTHING APPLIED")
        return 0

    # ── held-out gated solve (same discipline as pose_refine) ──
    rng = np.random.default_rng(1)
    hold = rng.random(len(edges)) < holdout_frac
    train = [e for e, h in zip(edges, hold) if not h]
    held = [e for e, h in zip(edges, hold) if h] or edges
    ident = [np.eye(4)] * n
    Cs = solve_pose_graph(n, train, odo_weight=odo_weight,
                          leash_weight=leash_weight)
    h0 = float(np.median(edge_errors(held, ident)))
    h1 = float(np.median(edge_errors(held, Cs)))
    shifts = np.array([np.linalg.norm(C[:3, 3]) for C in Cs])
    accepted = h1 <= h0 * (1.0 - min_gain)
    report = {
        "n_frames": n, "n_pairs": len(pairs), "n_revisit_pairs": len(rev),
        "n_edges": len(edges), "feature_lattice": [hp, wp],
        "holdout_before_m": h0, "holdout_after_m": h1,
        "min_gain": min_gain, "accepted": bool(accepted),
        "correction_m": {"median": float(np.median(shifts)),
                         "max": float(shifts.max())},
        "provenance": "tool_measured",
    }
    (output_dir / "pose_refine_fm_report.json").write_text(
        json.dumps(report, indent=1))
    log(f"[pose-fm] held-out {h0 * 1000:.1f}mm → {h1 * 1000:.1f}mm | "
        f"corrections median {np.median(shifts) * 1000:.1f}mm "
        f"max {shifts.max() * 1000:.1f}mm")
    if not accepted:
        log(f"[pose-fm] gain {(1 - h1 / max(h0, 1e-12)) * 100:.1f}% < "
            f"{min_gain * 100:.0f}% required — NOTHING APPLIED (identity)")
        return 0

    # ── apply: points move with their frame, poses follow ──
    moved = 0
    for p, v, fg in plys:
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        changed = False
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None or np.allclose(Cs[k], np.eye(4), atol=1e-12):
                continue
            m = fg == f
            xyz[m] = xyz[m] @ Cs[k][:3, :3].T + Cs[k][:3, 3]
            changed = True
        if changed:
            v["x"] = xyz[:, 0].astype(v["x"].dtype)
            v["y"] = xyz[:, 1].astype(v["y"].dtype)
            v["z"] = xyz[:, 2].astype(v["z"].dtype)
            PlyData([PlyElement.describe(v, "vertex")], text=False,
                    byte_order="<").write(str(p))
    bak = output_dir / "camera_poses.txt.prefm"
    if not bak.exists():
        bak.write_text(pp.read_text())
    out_lines = []
    for k, M in enumerate(poses):
        if not np.allclose(Cs[k], np.eye(4), atol=1e-12):
            moved += 1
        out_lines.append(" ".join(f"{x:.9g}"
                                  for x in (Cs[k] @ M).reshape(-1)))
    pp.write_text("\n".join(out_lines) + "\n")
    log(f"[pose-fm] ✅ {moved}/{n} poses refined (backup {bak.name})")
    return moved


def main():
    # ENVIRONMENT LESSON (Poisson/PGSR, this 252-core box): unpinned BLAS/TBB
    # thrashes — the CPU parts (rigid fits, pose-graph solve) run pinned to 8
    # cores; the heavy matching runs on GPU regardless.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    try:
        os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
    except Exception:  # noqa: BLE001 — mitigation, not a requirement
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Feature-metric (DINOv3) pose refinement — fase 3")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", default=None)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import cfg as _cfg
    dcfg = _cfg.get("dino_features") or {}
    run(Path(args.output_dir),
        Path(args.frames_dir) if args.frames_dir else None,
        cfg={**dcfg, **(dcfg.get("pose_refine_fm") or {})},
        log=lambda m: logging.getLogger(__name__).info(m))


if __name__ == "__main__":
    main()
