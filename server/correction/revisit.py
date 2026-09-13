"""Geometric revisit detection — loop closure from OUR geometry, no image
descriptors, no segmentation (USER 2026-09-09: "ver los mismos lugares desde
diferentes posiciones, eso nos debe servir para comparar").

What it measures (everything tool_measured, nothing is modified):
  1. CO-VISIBILITY: every keyframe's own written points (provenance) are
     projected into every other keyframe's camera (poses + per-keyframe
     intrinsics + the depth window); two keyframes far apart in the walk
     (≥ min_gap_kf) that see ≥ min_covis of each other's points saw the same
     place from different positions — a revisit, whether or not the walker
     passed through the same spot.
  2. REGIONS: co-visible keyframe pairs are clustered (2-D grid of pair
     indices, cluster_gap_kf) into revisit regions: the EARLIER keyframe run,
     the LATER run, their chunks, the 3-D volume both saw.
  3. DUPLICATION per region: the points the earlier run wrote inside the
     volume vs the points the later run wrote there — median NN distance
     (≥ offset_min_m ⇒ duplicated), the rigid closure (trimmed yaw+t ICP,
     later → earlier) with its residual, and the region's observable DOF
     (a flat floor observes only its normal).
  4. 2-D PREVIEWS: the other visit's points reprojected onto the real frame
     of a representative keyframe of each side, so the user can verify with
     his eyes which KF saw what from where.

Output: ``output/corrections/revisits.json`` (+ PNGs under
``output/corrections/revisits/``). Consumed by the UI and — next step — by
the chunk pose graph that places the closures on the seams that produced
them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from correction import observability as obs_mod, solve
from correction.config import CorrectionConfig
from correction.session import CorrectionSession
from correction.units import chunks_of_keyframe, load_chunk_plan

REPORT_NAME = "revisits.json"
IMG_DIR = "revisits"


# ── intrinsics / grid ────────────────────────────────────────────────────

def load_intrinsics(output_dir: Path, n_kf: int) -> np.ndarray:
    """(n_kf, 4) fx fy cx cy per keyframe on the PROVENANCE grid
    (intrinsic.txt: one row per keyframe, DA3/omega processed resolution);
    a single row is broadcast."""
    for p in (output_dir / "intrinsic.txt",
              output_dir / "maplong_run" / "intrinsic.txt",
              output_dir / "da3_run" / "intrinsic.txt"):
        if p.exists():
            arr = np.loadtxt(str(p), ndmin=2)
            if arr.shape[1] < 4:
                raise RuntimeError(f"{p}: expected fx fy cx cy rows")
            arr = arr[:, :4]
            if len(arr) == n_kf:
                return arr
            if len(arr) == 1:
                return np.repeat(arr, n_kf, axis=0)
            raise RuntimeError(
                f"{p} has {len(arr)} rows for {n_kf} keyframes — the "
                f"intrinsics are not row-aligned with camera_frames.txt")
    raise RuntimeError(
        f"no intrinsic.txt under {output_dir} (nor maplong_run/, da3_run/) "
        f"— the revisit detector needs per-keyframe intrinsics")


def provenance_grid(session: CorrectionSession) -> Tuple[int, int]:
    """(H, W) of the pixel grid the provenance lives on: the depth map
    shape when a per-frame npz exists, else the max pixel indices + 1."""
    out = session.output_dir
    for d in (out / "omega_run" / "results_output",
              out / "da3_run" / "results_output"):
        if d.is_dir():
            for f in d.glob("frame_*.npz"):
                z = np.load(f)
                if "depth" in z:
                    h, w = z["depth"].shape
                    return int(h), int(w)
                break
    return (int(session.data["pixel_row"].max()) + 1,
            int(session.data["pixel_col"].max()) + 1)


# ── co-visibility ────────────────────────────────────────────────────────

def _project(P: np.ndarray, w2c: np.ndarray, k: np.ndarray):
    X = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0] * X[:, 0] / z + k[2]
        v = k[1] * X[:, 1] / z + k[3]
    return u, v, z


def covisibility(session: CorrectionSession, K: np.ndarray, grid: Tuple[int, int],
                 cfg: CorrectionConfig, rng: np.random.Generator,
                 log=print):
    """Occlusion-aware co-visibility. For every keyframe i, its OWN points
    build a coarse depth image (z-buffer, ``zbuffer_cell_px``); the sampled
    points of every other keyframe j are projected into i and count as
    co-visible only when they land on a cell where i measured a surface AT
    THE SAME DEPTH (|Δz| ≤ max(same_surface_tol_m, rel·z)). Points behind
    that surface are occluded (frustum-only co-visibility looked through
    walls), points in front were not seen by i either.

    Returns (covis[i, j] = fraction of j's samples on i's surface,
    samples, hits) where hits[(i, j)] are the sample indices (into the
    session cloud) of j that matched i's surface — the 3-D evidence the
    region builder groups."""
    n = session.n_kf
    H, W = grid
    rc = cfg.revisit
    cs = rc.zbuffer_cell_px
    Hc, Wc = H // cs + 1, W // cs + 1
    samples: Dict[int, np.ndarray] = {}
    order = np.argsort(session.ks, kind="stable")
    bounds = np.searchsorted(session.ks[order], np.arange(-1, n + 1))
    own: Dict[int, np.ndarray] = {}
    for k in range(n):
        sel = order[bounds[k + 1]:bounds[k + 2]]
        own[k] = sel
        samples[k] = (sel if len(sel) <= rc.sample_per_kf
                      else rng.choice(sel, rc.sample_per_kf, replace=False))
    all_idx = np.concatenate([samples[k] for k in range(n)])
    owner = np.concatenate([np.full(len(samples[k]), k) for k in range(n)])
    P = session.xyz[all_idx]
    counts = np.bincount(owner, minlength=n).astype(np.float64)
    covis = np.zeros((n, n), dtype=np.float64)
    hits: Dict[Tuple[int, int], np.ndarray] = {}
    w2cs = np.linalg.inv(session.poses)
    for i in range(n):
        # z-buffer of i from ALL its own points (median depth per cell)
        u0, v0, z0 = _project(session.xyz[own[i]], w2cs[i], K[i])
        ok0 = (z0 > rc.depth_min_m) & (u0 >= 0) & (u0 < W) & (v0 >= 0) & (v0 < H)
        cell0 = (v0[ok0] // cs).astype(np.int64) * Wc + (u0[ok0] // cs).astype(np.int64)
        zbuf = np.full(Hc * Wc, np.nan)
        if len(cell0):
            srt = np.argsort(cell0, kind="stable")
            cs_sorted, zs_sorted = cell0[srt], z0[ok0][srt]
            uniq, start = np.unique(cs_sorted, return_index=True)
            end = np.append(start[1:], len(cs_sorted))
            for c_, s_, e_ in zip(uniq, start, end):
                zbuf[c_] = np.median(zs_sorted[s_:e_])
        # project every other keyframe's samples
        u, v, z = _project(P, w2cs[i], K[i])
        vis = ((z > rc.depth_min_m) & (z < rc.depth_max_m)
               & (u >= 0) & (u < W) & (v >= 0) & (v < H))
        cell = np.full(len(z), -1, dtype=np.int64)
        cell[vis] = (v[vis] // cs).astype(np.int64) * Wc + (u[vis] // cs).astype(np.int64)
        zc = np.full(len(z), np.nan)
        zc[vis] = zbuf[cell[vis]]
        tol = np.maximum(rc.same_surface_tol_m, rc.same_surface_tol_rel * z)
        same = vis & np.isfinite(zc) & (np.abs(z - zc) <= tol)
        same &= owner != i
        hit = np.bincount(owner[same], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            covis[i] = np.where(counts > 0, hit / counts, 0.0)
        # keep the matched samples of the keyframes that pass min_covis
        for j in np.where(covis[i] >= rc.min_covis)[0]:
            if abs(int(j) - i) >= rc.min_gap_kf:
                m = same & (owner == j)
                hits[(i, int(j))] = all_idx[m]
        if (i + 1) % 50 == 0:
            log(f"  co-visibility: {i + 1}/{n} keyframes")
    return covis, samples, hits


def build_regions(session: CorrectionSession, hits: Dict[Tuple[int, int], np.ndarray],
                  cfg: CorrectionConfig) -> List[dict]:
    """Revisit regions in 3-D: every occlusion-verified co-visible point is
    binned into a FIXED block of ``region_cell_m`` (one block = one place;
    connected cells chained whole corridors into one region). The observers
    of a block (the keyframe that wrote the point and the keyframe that saw
    it) are split into VISITS by temporal gap (≥ min_gap_kf keyframes
    without seeing the block); one region per (block, visit a, visit b)."""
    from correction.units import visits_from_keyframes
    if not hits:
        return []
    cell = cfg.revisit.region_cell_m
    pts_idx, i_of, j_of = [], [], []
    for (i, j), idx in hits.items():
        if not len(idx):
            continue
        pts_idx.append(idx)
        i_of.append(np.full(len(idx), i))
        j_of.append(np.full(len(idx), j))
    pts_idx = np.concatenate(pts_idx)
    i_of = np.concatenate(i_of)
    j_of = np.concatenate(j_of)
    P = session.xyz[pts_idx]
    lo = P.min(0)
    vox = np.floor((P - lo) / cell).astype(np.int64)
    key = vox[:, 0] * 1_000_003 + vox[:, 1] * 1_009 + vox[:, 2]
    regions = []
    for k in np.unique(key):
        m = key == k
        if m.sum() < cfg.evidence.min_object_points_solve:
            continue
        observers = np.concatenate([i_of[m], j_of[m]])
        visits = visits_from_keyframes(observers, cfg.revisit.min_gap_kf)
        if len(visits) < 2:
            continue
        v0 = vox[m][0]
        vol_lo = lo + v0 * cell - cfg.evidence.obb_margin_m
        vol_hi = lo + (v0 + 1) * cell + cfg.evidence.obb_margin_m
        for a_ in range(len(visits)):
            for b_ in range(a_ + 1, len(visits)):
                va, vb = visits[a_], visits[b_]
                mm = m & ((np.isin(i_of, va) & np.isin(j_of, vb))
                          | (np.isin(i_of, vb) & np.isin(j_of, va)))
                if mm.sum() < cfg.evidence.min_object_points_solve:
                    continue
                regions.append({"early": va, "late": vb,
                                "n_hits": int(mm.sum()),
                                "vol_lo": vol_lo, "vol_hi": vol_hi})
    return regions


# ── region measurement ───────────────────────────────────────────────────

def _region_points(session, kfs: List[int], vol_lo, vol_hi,
                   rng, n_max: int) -> np.ndarray:
    mask = np.isin(session.ks, kfs)
    idx = np.where(mask)[0]
    P = session.xyz[idx]
    inside = np.all((P >= vol_lo) & (P <= vol_hi), axis=1)
    idx = idx[inside]
    if len(idx) > n_max:
        idx = rng.choice(idx, n_max, replace=False)
    return idx


def measure_region(session: CorrectionSession, early: List[int],
                   late: List[int], vol_lo: np.ndarray, vol_hi: np.ndarray,
                   cfg: CorrectionConfig, rng: np.random.Generator) -> dict:
    """Duplication + closure of one revisit region (later → earlier)."""
    rc = cfg.revisit
    ia = _region_points(session, early, vol_lo, vol_hi, rng, rc.region_sample)
    ib = _region_points(session, late, vol_lo, vol_hi, rng, rc.region_sample)
    out = {"n_early_pts": int(len(ia)), "n_late_pts": int(len(ib))}
    if len(ia) < cfg.evidence.min_object_points_solve \
            or len(ib) < cfg.evidence.min_object_points_solve:
        out.update({"measured": False,
                    "why": "too few points on one side of the region"})
        return out
    A, B = session.xyz[ia], session.xyz[ib]
    tree = cKDTree(A)
    d0, _ = tree.query(B, workers=cfg.runtime.workers)
    before = float(np.median(d0))
    shape = obs_mod.classify_object(A, 0, "region", cfg)
    init_t = np.median(A, axis=0) - np.median(B, axis=0)
    sub = B[rng.choice(len(B), min(cfg.solve.icp_sample, len(B)),
                       replace=False)] + init_t
    full = shape.shape == obs_mod.SHAPE_COMPACT or (
        shape.shape == obs_mod.SHAPE_PLANAR
        and shape.eig_ratios[0] <= cfg.observability.yaw_anisotropy_max)
    R, t, rms = solve.trimmed_icp(sub, tree, A, cfg, rotation=full)
    t_full = R @ init_t + t
    if not full:
        R, t_full = solve.project_solution(
            R, t_full,
            {"mode": "normal", "normal": shape.normal.tolist()}
            if shape.shape == obs_mod.SHAPE_PLANAR else
            {"mode": "perp_axis", "axis": shape.axis.tolist()})
    d1, _ = tree.query(B @ R.T + t_full, workers=cfg.runtime.workers)
    after = float(np.median(d1))
    closure_ok = after < before
    out.update({
        "measured": True,
        "duplicated": before >= rc.offset_min_m,
        "offset_before_cm": round(before * 100, 1),
        "offset_after_cm": round(after * 100, 1),
        "closure_found": closure_ok,
        "closure": ({"rot_deg": round(solve.rot_deg(R), 3),
                     "t_m": [round(float(x), 4) for x in t_full],
                     "t_norm_m": round(float(np.linalg.norm(t_full)), 4),
                     "icp_rms_cm": round(rms * 100, 2)} if closure_ok else
                    {"why": "ICP did not reduce the offset — closure not "
                            "trusted (partial coverage / structureless)"}),
        "shape": shape.shape,
        "eig_ratios": [round(float(x), 4) for x in shape.eig_ratios],
        "observes": ("yaw+t" if full else
                     "t along normal only" if shape.shape == obs_mod.SHAPE_PLANAR
                     else "t across the axis only"),
    })
    return out


# ── joint closure per visit pair ─────────────────────────────────────────

def consolidate_closures(session: CorrectionSession, regions: List[dict],
                         hits: Dict[Tuple[int, int], np.ndarray],
                         cfg: CorrectionConfig, rng: np.random.Generator,
                         log=print) -> List[dict]:
    """ONE rigid closure per pair of visits (USER 2026-09-09): the blocks of
    the same (earlier visit, later visit) are the same physical revisit, so
    they are solved TOGETHER — trimmed yaw+t ICP. A block-wise ICP on 3 m of
    floor + desk rotated 24°–69° and still "improved" its own median; the
    joint solve cannot rotate one block away from the others.

    The evidence is ONLY what both visits actually saw: the occlusion-
    verified hits — the later visit's points that landed on the earlier
    visit's measured surface (source) and the earlier visit's points that
    landed on the later visit's surface (target). Taking every point of the
    block instead mixed in what only one visit covered (the far wall one
    pass saw, the floor the other pass walked on) and the ICP slid along
    that asymmetry (pccr 31/08: |t| 0.91 m, one block 11 → 62 cm).

    Acceptance by CONSISTENCY, not by a threshold: the joint transform must
    reduce the median offset in ≥ min_blocks_improved of the pair's blocks
    (default: all of them)."""
    pairs: Dict[Tuple[int, int, int, int], List[dict]] = {}
    for rg in regions:
        if not rg.get("measured"):
            continue
        key = (rg["earlier_kfs"][0], rg["earlier_kfs"][1],
               rg["later_kfs"][0], rg["later_kfs"][1])
        pairs.setdefault(key, []).append(rg)
    # blocks of the same physical revisit may differ by a keyframe or two at
    # the visit edges: merge pairs whose early AND late spans overlap
    merged: List[Tuple[list, list, List[dict]]] = []
    for (ea, eb, la, lb), rgs in sorted(pairs.items()):
        for m in merged:
            if not (eb < m[0][0] or ea > m[0][1]) and \
                    not (lb < m[1][0] or la > m[1][1]):
                m[0][0], m[0][1] = min(m[0][0], ea), max(m[0][1], eb)
                m[1][0], m[1][1] = min(m[1][0], la), max(m[1][1], lb)
                m[2].extend(rgs)
                break
        else:
            merged.append(([ea, eb], [la, lb], list(rgs)))
    closures = []
    for early_span, late_span, rgs in merged:
        early = set(range(early_span[0], early_span[1] + 1))
        late = set(range(late_span[0], late_span[1] + 1))
        src_parts = [idx for (i, j), idx in hits.items()
                     if i in early and j in late and len(idx)]
        tgt_parts = [idx for (i, j), idx in hits.items()
                     if i in late and j in early and len(idx)]
        if not src_parts or not tgt_parts:
            continue
        src_all = np.unique(np.concatenate(src_parts))
        tgt_all = np.unique(np.concatenate(tgt_parts))
        S_all, T_all = session.xyz[src_all], session.xyz[tgt_all]
        # per block: the co-visible points of each side inside the block
        blocks, S_parts, T_parts = [], [], []
        for rg in rgs:
            lo = np.asarray(rg["volume_m"]["lo"]); hi = np.asarray(rg["volume_m"]["hi"])
            sb = S_all[np.all((S_all >= lo) & (S_all <= hi), axis=1)]
            tb = T_all[np.all((T_all >= lo) & (T_all <= hi), axis=1)]
            if len(sb) < cfg.solve.icp_min_corr or len(tb) < cfg.solve.icp_min_corr:
                continue
            blocks.append(rg["region"]); S_parts.append(sb); T_parts.append(tb)
        if not blocks:
            continue
        S = np.concatenate(S_parts); T = np.concatenate(T_parts)
        tree = cKDTree(T)
        init_t = np.median(np.stack([np.median(t_, 0) - np.median(s_, 0)
                                     for s_, t_ in zip(S_parts, T_parts)]), 0)
        sub = S[rng.choice(len(S), min(cfg.solve.icp_sample, len(S)),
                           replace=False)] + init_t
        R, t, rms = solve.trimmed_icp(sub, tree, T, cfg, rotation=True)
        t_full = R @ init_t + t
        per_block = []
        n_imp = 0
        for rid, s_, t_ in zip(blocks, S_parts, T_parts):
            tb = cKDTree(t_)
            d0, _ = tb.query(s_, workers=cfg.runtime.workers)
            d1, _ = tb.query(s_ @ R.T + t_full, workers=cfg.runtime.workers)
            before, after = float(np.median(d0)), float(np.median(d1))
            improved = after < before
            n_imp += int(improved)
            per_block.append({"region": rid, "n_src": int(len(s_)),
                              "n_tgt": int(len(t_)),
                              "before_cm": round(before * 100, 1),
                              "after_cm": round(after * 100, 1),
                              "improved": improved})
        frac = n_imp / len(blocks)
        accepted = frac >= cfg.posegraph.min_blocks_improved
        cl = {"earlier_kfs": early_span, "later_kfs": late_span,
              "n_blocks": len(blocks), "blocks_improved": n_imp,
              "n_covisible_src": int(len(S)), "n_covisible_tgt": int(len(T)),
              "accepted": accepted,
              "rot_deg": round(solve.rot_deg(R), 3),
              "t_m": [round(float(x), 4) for x in t_full],
              "t_norm_m": round(float(np.linalg.norm(t_full)), 4),
              "icp_rms_cm": round(rms * 100, 2),
              "per_block": per_block,
              "R": R.tolist(), "t": t_full.tolist()}
        closures.append(cl)
        log(f"  joint closure kf {early_span} ↔ {late_span} on "
            f"{len(S)}/{len(T)} co-visible pts: yaw {cl['rot_deg']}°, |t| "
            f"{cl['t_norm_m']} m, blocks improved {n_imp}/{len(blocks)} → "
            f"{'ACCEPTED' if accepted else 'rejected'} — "
            + ", ".join(f"R{b['region']} {b['before_cm']}→{b['after_cm']}"
                        for b in per_block))
    return closures


# ── previews ─────────────────────────────────────────────────────────────

def _frame_path(session: CorrectionSession, kf: int) -> Optional[Path]:
    frames_dir = session.output_dir.parent / "frames"
    f = session.frames[kf]
    for name in (f"{f:06d}.jpg", f"{f:05d}.jpg", f"{f}.jpg"):
        p = frames_dir / name
        if p.exists():
            return p
    return None


def render_preview(session: CorrectionSession, kf: int, other_idx: np.ndarray,
                   own_idx: np.ndarray, K: np.ndarray, grid: Tuple[int, int],
                   out_png: Path, cfg: CorrectionConfig) -> Optional[str]:
    """The real frame of keyframe kf with the OTHER visit's points (red) and
    its own (green) reprojected. Returns the png name or None (no frame)."""
    from PIL import Image, ImageDraw
    fp = _frame_path(session, kf)
    if fp is None:
        return None
    im = Image.open(fp).convert("RGB")
    W_img, H_img = im.size
    H, W = grid
    sx, sy = W_img / W, H_img / H
    w2c = np.linalg.inv(session.poses[kf])
    draw = ImageDraw.Draw(im)
    for idx, color in ((own_idx, (40, 220, 40)), (other_idx, (230, 40, 40))):
        if not len(idx):
            continue
        u, v, z = _project(session.xyz[idx], w2c, K)
        ok = (z > cfg.revisit.depth_min_m) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for uu, vv in zip(u[ok] * sx, v[ok] * sy):
            draw.point((float(uu), float(vv)), fill=color)
    scale = cfg.revisit.image_max_px / max(W_img, H_img)
    if scale < 1:
        im = im.resize((int(W_img * scale), int(H_img * scale)))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return out_png.name


# ── entry ────────────────────────────────────────────────────────────────

def detect_revisits(session: CorrectionSession, cfg: CorrectionConfig,
                    log=print, progress=None, previews: bool = True) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.solve.seed)

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    K = load_intrinsics(session.output_dir, session.n_kf)
    grid = provenance_grid(session)
    plan = load_chunk_plan(session.output_dir)
    _p(5, f"co-visibility over {session.n_kf} keyframes (grid {grid[1]}x{grid[0]})...")
    covis, samples, hits = covisibility(session, K, grid, cfg, rng, log=log)
    rc = cfg.revisit
    n = session.n_kf
    n_pairs = len(hits)
    _p(55, f"{n_pairs} occlusion-verified co-visible keyframe pairs beyond "
           f"{rc.min_gap_kf} keyframes apart")
    raw_regions = build_regions(session, hits, cfg)
    regions = []
    img_dir = session.output_dir / "corrections" / IMG_DIR
    for gi, rg in enumerate(sorted(raw_regions, key=lambda r: (r["early"][0], r["late"][0]))):
        early, late = rg["early"], rg["late"]
        vol_lo, vol_hi = rg["vol_lo"], rg["vol_hi"]
        # representative pair: the (i, j) of the region with most hits
        pair_counts: Dict[Tuple[int, int], int] = {}
        for (i, j), idx in hits.items():
            if (i in early and j in late) or (i in late and j in early):
                Pi = session.xyz[idx]
                inside = np.all((Pi >= vol_lo) & (Pi <= vol_hi), axis=1)
                pair_counts[(i, j)] = int(inside.sum())
        if not pair_counts:
            continue
        best = max(pair_counts, key=pair_counts.get)
        meas = measure_region(session, early, late, vol_lo, vol_hi, cfg, rng)
        region = {
            "region": gi,
            "earlier_kfs": [early[0], early[-1]],
            "later_kfs": [late[0], late[-1]],
            "earlier_chunks": sorted({c for k in early for c in chunks_of_keyframe(plan, k)}),
            "later_chunks": sorted({c for k in late for c in chunks_of_keyframe(plan, k)}),
            "n_pairs": len(pair_counts), "n_hits": rg["n_hits"],
            "best_pair": {"earlier_kf": int(best[0]), "later_kf": int(best[1]),
                          "covis": round(float(max(covis[best[0], best[1]],
                                                    covis[best[1], best[0]])), 3)},
            "volume_m": {"lo": [round(float(x), 3) for x in vol_lo],
                         "hi": [round(float(x), 3) for x in vol_hi]},
            **meas,
        }
        if previews and meas.get("measured"):
            ia = _region_points(session, early, vol_lo, vol_hi, rng, rc.region_sample)
            ib = _region_points(session, late, vol_lo, vol_hi, rng, rc.region_sample)
            region["preview_earlier"] = render_preview(
                session, best[0], ib, ia, K[best[0]], grid,
                img_dir / f"region_{gi}_kf{best[0]}_seen_by_earlier.png", cfg)
            region["preview_later"] = render_preview(
                session, best[1], ia, ib, K[best[1]], grid,
                img_dir / f"region_{gi}_kf{best[1]}_seen_by_later.png", cfg)
        regions.append(region)
        log(f"  region {gi}: kf {early[0]}..{early[-1]} ↔ kf {late[0]}..{late[-1]}"
            f" ({len(pair_counts)} pairs, {rg['n_hits']} hits) — "
            + (f"offset {meas['offset_before_cm']} → {meas['offset_after_cm']} cm, "
               f"{'DUPLICATED' if meas['duplicated'] else 'consistent'}, "
               f"{meas['observes']}"
               f"{'' if meas['closure_found'] else ' (no closure)'}"
               if meas.get("measured") else meas.get("why", "")))
    _p(85, "joint closure per pair of visits...")
    closures = consolidate_closures(session, regions, hits, cfg, rng, log=log)
    report = {"n_keyframes": n, "grid_hw": list(grid),
              "min_gap_kf": rc.min_gap_kf, "min_covis": rc.min_covis,
              "n_pairs": n_pairs, "regions": regions,
              "closures": [{k: v for k, v in c.items() if k not in ("R", "t")}
                           for c in closures],
              "elapsed_s": round(time.time() - t0, 1),
              "provenance": "tool_measured"}
    (session.output_dir / "corrections").mkdir(exist_ok=True)
    (session.output_dir / "corrections" / REPORT_NAME).write_text(
        json.dumps(report, indent=1))
    _p(100, f"revisit detection done: {len(regions)} region(s), "
            f"{sum(1 for r in regions if r.get('duplicated'))} duplicated, "
            f"{sum(1 for c in closures if c['accepted'])} closure(s) accepted")
    report["_closures_full"] = closures
    return report
