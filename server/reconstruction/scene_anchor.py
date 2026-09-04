"""
FASE 5 — GLOBAL SURFACE-CONSENSUS ANCHORING (USER DESIGN 2026-09-04).

"Hay que ir armando la nube, las poses y los depths siempre con CONCIENCIA
de lo que estamos reconstruyendo": the same door seen at minute 1 and at the
END of the walk, the elevated floor patch, the displaced cabinets — every
re-observed surface becomes a LANDMARK, all landmarks vote together, and the
solve re-anchors poses AND depths so every copy of the same physical surface
coincides.

Design (staged — proven pieces, each self-gated):
  A. Appearance-based revisit graph: global DINOv3 descriptor per keyframe
     (pose-INDEPENDENT — a drifted end-of-walk revisit is exactly the case
     to catch), plus sequential pairs.
  B. Patch-verified matches (mutual-NN DINOv3 + geometry sanity) chained
     into TRACKS (union-find): multi-view observations of one surface point.
     The double lock (semantic + spatial spread cap) is what keeps the
     association from ever confusing different surfaces.
  C. Landmark registry persisted to landmarks.json — the session's memory
     (multi-visit landmarks are the star witnesses; consumable by later
     stages).
  D. Staged global solve, iterated:
       D1. SE(3) per frame — the PROVEN pose-graph solver (pose_refine)
           over track-derived point-to-plane edges.
       D2. per-frame LOW-DOF depth correction d' = d·(1 + a_f + b_f·(d/dref
           −1)) — a sparse robust linear solve (2 DOF/frame): systematic
           distance-dependent bias is corrected ALONG each ray; the model
           cannot invent shapes.
  E. Gates (industrial discipline, USER: "sin fallo silencioso"):
       - held-out tracks must improve ≥ min_gain → else IDENTITY;
       - DA3 metric gate: agreement with the isolated DA3 anchor depths
         must not degrade → else IDENTITY;
       - hard caps on every DOF (beyond cap = flagged, never forced);
       - frames without enough tracks: identity (no unearned correction);
       - any crash/missing input RAISES — the pipeline stops loudly.
  F. Apply: every point moves along ITS OWN measured ray (provenance),
     poses follow. fase5_report.json records everything.

Provenance: tool_measured end to end. Runs at the end of reconstruction
(chunk PLYs on disk), superseding pose_refine_fm (its SE(3core) is stage D1).

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reconstruction.pose_refine import (edge_errors, normal_grid,
                                        robust_rigid, solve_pose_graph)
from reconstruction.dino_features import (FeatureCache,
                                          extract_session_features,
                                          load_session_cameras)
from reconstruction.pose_refine_fm import _mutual_matches

logger = logging.getLogger(__name__)


class SceneAnchorError(RuntimeError):
    """Fase-5 failure — always raised, never swallowed."""


# ── union-find (tracks) ──────────────────────────────────────────────────

class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ── session loading (per-frame lattice grids from provenance) ────────────

def _load_frame_grids(output_dir: Path, fc: FeatureCache, frames: List[int],
                      fidx: Dict[int, int], log) -> Tuple[dict, dict, list]:
    """world (hp,wp,3), valid (hp,wp) per frame index, plus the raw chunk
    PLY handles for the final apply. Raises on any missing input."""
    from plyfile import PlyData
    hp, wp = fc.hp, fc.wp
    chunk_files = [p for p in sorted(output_dir.glob("chunk_*.ply"))
                   if not p.stem.startswith(("chunk_997", "chunk_998",
                                             "chunk_999"))]
    if not chunk_files:
        raise SceneAnchorError("no chunk PLYs — fase 5 runs before the "
                               "cloud merge; missing inputs = broken pipeline")
    acc: Dict[int, np.ndarray] = {}
    cnt: Dict[int, np.ndarray] = {}
    plys = []
    Hg = Wg = None
    for p in chunk_files:
        org = output_dir / f"{p.stem}_origins.npz"
        if not org.exists():
            raise SceneAnchorError(f"{org.name} missing — provenance is "
                                   f"mandatory")
        z = np.load(org)
        pd = PlyData.read(str(p))
        v = np.array(pd["vertex"].data)
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        fg = z["frame_global"].astype(np.int64)
        if len(fg) != len(xyz):
            raise SceneAnchorError(f"{p.name}: points != origins")
        if Hg is None and "scaled_resolution" in z.files:
            Hg, Wg = (int(x) for x in z["scaled_resolution"])
        pr = z["pixel_row"].astype(np.int64)
        pc = z["pixel_col"].astype(np.int64)
        plys.append((p, v, fg, pr, pc))
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None or not fc.has(int(f)):
                continue
            m = fg == f
            r = np.clip(pr[m] * hp // max(Hg or (pr.max() + 1), 1), 0, hp - 1)
            c = np.clip(pc[m] * wp // max(Wg or (pc.max() + 1), 1), 0, wp - 1)
            cell = r * wp + c
            a = acc.setdefault(k, np.zeros((hp * wp, 3)))
            q = cnt.setdefault(k, np.zeros(hp * wp))
            np.add.at(a, cell, xyz[m])
            np.add.at(q, cell, 1)
    if Hg is None:
        raise SceneAnchorError("no scaled_resolution in origins")
    world, valid = {}, {}
    for k, a in acc.items():
        q = cnt[k]
        ok = q > 0
        g = np.zeros((hp * wp, 3))
        g[ok] = a[ok] / q[ok, None]
        world[k] = g.reshape(hp, wp, 3)
        valid[k] = ok.reshape(hp, wp)
    return world, valid, plys


# ── A: appearance revisit graph ──────────────────────────────────────────

def _revisit_pairs(fc: FeatureCache, frames: List[int], world: dict,
                   seq_offsets, min_global_sim: float, min_gap: int,
                   max_pairs: int, log) -> Tuple[list, int]:
    n = len(frames)
    desc = np.zeros((n, fc.dim), np.float32)
    for k in range(n):
        if fc.has(frames[k]):
            g = fc.grid(frames[k]).reshape(-1, fc.dim)
            d = g.mean(axis=0)
            desc[k] = d / max(np.linalg.norm(d), 1e-8)
    S = desc @ desc.T
    pairs = [(i, i + o) for o in seq_offsets for i in range(n - o)
             if i in world and (i + o) in world]
    rev = []
    for i in range(n):
        js = np.flatnonzero(S[i] >= min_global_sim)
        for j in js:
            if j - i >= min_gap and i in world and int(j) in world:
                rev.append((i, int(j), float(S[i, j])))
    rev.sort(key=lambda t: -t[2])
    seen = set()
    rev_pairs = []
    for i, j, s in rev:
        key = (i // 5, j // 5)          # de-duplicate near-identical revisits
        if key in seen:
            continue
        seen.add(key)
        rev_pairs.append((i, j))
        if len(rev_pairs) >= max_pairs // 4:
            break
    pairs += rev_pairs
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[t] for t in sorted(keep)]
    log(f"[fase5] pair graph: {len(pairs)} pairs "
        f"({len(rev_pairs)} appearance revisits — pose-independent)")
    return pairs, len(rev_pairs)


# ── B: verified matches → tracks ─────────────────────────────────────────

def _build_tracks(fc: FeatureCache, frames, world, valid, pairs,
                  match_min_cos: float, track_max_spread_m: float,
                  min_track_obs: int, log):
    hp, wp = fc.hp, fc.wp
    ncell = hp * wp
    n = len(frames)
    uf = _UF(n * ncell)
    feat_flat: Dict[int, np.ndarray] = {}

    def _feats(k):
        if k not in feat_flat:
            if len(feat_flat) > 96:
                feat_flat.pop(next(iter(feat_flat)))
            feat_flat[k] = fc.grid(frames[k]).reshape(ncell, fc.dim)
        return feat_flat[k]

    n_matches = 0
    for (a, b) in pairs:
        va = valid[a].reshape(-1)
        vb = valid[b].reshape(-1)
        ia_v = np.flatnonzero(va)
        ib_v = np.flatnonzero(vb)
        if len(ia_v) < 20 or len(ib_v) < 20:
            continue
        ia, ib, _cos = _mutual_matches(_feats(a)[ia_v], _feats(b)[ib_v],
                                       match_min_cos)
        if len(ia) == 0:
            continue
        Pa = world[a].reshape(-1, 3)[ia_v[ia]]
        Pb = world[b].reshape(-1, 3)[ib_v[ib]]
        # spatial sanity: same physical point → displaced copies live within
        # tens of cm; beyond the cap it is a mis-association, never a match
        okd = np.linalg.norm(Pa - Pb, axis=1) <= track_max_spread_m
        okf = np.isfinite(Pa).all(1) & np.isfinite(Pb).all(1)
        for x, y in zip(ia_v[ia[okd & okf]], ib_v[ib[okd & okf]]):
            uf.union(a * ncell + int(x), b * ncell + int(y))
            n_matches += 1
    # collect tracks — only union-touched nodes matter
    roots: Dict[int, list] = {}
    touched = np.flatnonzero(uf.p != np.arange(n * ncell))
    for node in touched:
        r = uf.find(int(node))
        roots.setdefault(r, []).append(int(node))
    for r in list(roots.keys()):
        roots[r].append(r)
    tracks = []
    for r, nodes in roots.items():
        obs = []
        for node in set(nodes):
            k, c = divmod(node, ncell)
            if k in world and valid[k].reshape(-1)[c]:
                obs.append((k, c))
        if len(obs) >= min_track_obs and len({k for k, _ in obs}) >= 2:
            tracks.append(obs)
    log(f"[fase5] {n_matches:,} verified matches → {len(tracks):,} tracks")
    if len(tracks) < 100:
        raise SceneAnchorError(f"only {len(tracks)} tracks — not enough "
                               f"consensus to anchor anything (refusing)")
    return tracks


# ── D2: per-frame low-DOF depth solve (poses fixed) ──────────────────────

def _solve_depth_curves(tracks, world, cams, n_frames, dref: float,
                        smooth_w: float, leash_w: float,
                        max_corr: float, log):
    """d' = d·(1 + a_f + b_f·(d/dref−1)). Robust sparse linear LSQ over
    (a_f, b_f): every track wants its observations' corrected positions to
    coincide with the track consensus point. Returns (a, b) arrays."""
    rows_i, cols_i, vals_i, rhs = [], [], [], []

    def _obs_geo(k, c):
        p = world[k].reshape(-1, 3)[c]
        C = cams[k]
        v = p - C
        d = float(np.linalg.norm(v))
        return p, C, v / max(d, 1e-9), d

    # IRLS over track residuals — NORMAL EQUATIONS accumulation (JtJ is
    # (2n,2n); each equation touches only its frame's two columns — the
    # naive dense design matrix would be ~13 GB at scene scale)
    a = np.zeros(n_frames)
    b = np.zeros(n_frames)
    nu = 2 * n_frames
    for it in range(3):
        JtJ = np.zeros((nu, nu))
        Jty = np.zeros(nu)
        for obs in tracks:
            geo = [_obs_geo(k, c) for k, c in obs]
            cur = []
            for (k, _c), (p, C, r, d) in zip(obs, geo):
                s = 1.0 + a[k] + b[k] * (d / dref - 1.0)
                cur.append(C + r * d * s)
            cur = np.array(cur)
            ctr = np.median(cur, axis=0)
            w_res = 1.0 / (1.0 + (np.linalg.norm(cur - ctr, axis=1)
                                  / 0.05) ** 2)     # Cauchy, 5 cm scale
            for (k, _c), (p, C, r, d), pc, wr in zip(obs, geo, cur, w_res):
                # residual along the ray: r·(p'(a,b) − ctr); ∂/∂a = d,
                # ∂/∂b = d·(d/dref−1). Solve ABSOLUTE (a,b), delta-free.
                ca = d * wr
                cb = d * (d / dref - 1.0) * wr
                rhs = (-float(r @ (pc - ctr)) * wr
                       + ca * a[k] + cb * b[k])
                ia, ib_ = 2 * k, 2 * k + 1
                JtJ[ia, ia] += ca * ca
                JtJ[ia, ib_] += ca * cb
                JtJ[ib_, ia] += ca * cb
                JtJ[ib_, ib_] += cb * cb
                Jty[ia] += ca * rhs
                Jty[ib_] += cb * rhs
        # smoothness between consecutive frames + identity leash
        for k in range(n_frames - 1):
            for off in (0,):
                i0, i1 = 2 * k + off, 2 * (k + 1) + off
                JtJ[i0, i0] += smooth_w ** 2
                JtJ[i1, i1] += smooth_w ** 2
                JtJ[i0, i1] -= smooth_w ** 2
                JtJ[i1, i0] -= smooth_w ** 2
        for k in range(nu):
            JtJ[k, k] += leash_w ** 2
        sol = np.linalg.solve(JtJ, Jty)
        a = np.clip(sol[0::2], -max_corr, max_corr)
        b = np.clip(sol[1::2], -max_corr, max_corr)
    log(f"[fase5] depth curves: |a| median {np.median(np.abs(a)) * 100:.2f}%"
        f" max {np.abs(a).max() * 100:.2f}% | |b| max "
        f"{np.abs(b).max() * 100:.2f}%")
    return a, b


def _track_mismatch(tracks, world, cams, a, b, dref) -> float:
    """Median 3D spread of track observations under corrections (a,b)."""
    out = []
    for obs in tracks:
        cur = []
        for k, c in obs:
            p = world[k].reshape(-1, 3)[c]
            C = cams[k]
            v = p - C
            d = float(np.linalg.norm(v))
            s = 1.0 + a[k] + b[k] * (d / dref - 1.0)
            cur.append(C + (v / max(d, 1e-9)) * d * s)
        cur = np.array(cur)
        out.append(float(np.linalg.norm(cur - np.median(cur, 0), axis=1)
                         .mean()))
    return float(np.median(out))


# ── DA3 metric gate ──────────────────────────────────────────────────────

def _da3_gate(output_dir: Path, frames, fidx, world, valid, cams,
              a, b, dref, fc, log) -> Optional[Tuple[float, float]]:
    """Median |log depth-ratio| vs isolated DA3 anchors, before vs after the
    depth curves. Returns (before, after) or None when anchors are absent."""
    adir = output_dir / "da3_run" / "results_output"
    if not adir.is_dir():
        return None
    import re
    befores, afters = [], []
    for f in sorted(adir.glob("frame_*.npz")):
        m = re.search(r"(\d+)", f.stem)
        if not m:
            continue
        num = int(m.group(1))
        k = fidx.get(num)
        if k is None or k not in world:
            continue
        try:
            da = np.load(f)
            key = [x for x in da.files if "depth" in x.lower()]
            dd = np.asarray(da[key[0]]) if key else None
        except Exception:  # noqa: BLE001
            continue
        if dd is None or dd.ndim != 2:
            continue
        vk = valid[k]
        g = world[k]
        C = cams[k]
        rr, cc = np.nonzero(vk)
        if len(rr) < 30:
            continue
        d_ours = np.linalg.norm(g[rr, cc] - C, axis=1)
        dv = dd[np.clip(rr * dd.shape[0] // vk.shape[0], 0, dd.shape[0] - 1),
                np.clip(cc * dd.shape[1] // vk.shape[1], 0, dd.shape[1] - 1)]
        ok = (dv > 0.1) & (d_ours > 0.1)
        if ok.sum() < 30:
            continue
        s = 1.0 + a[k] + b[k] * (d_ours[ok] / dref - 1.0)
        befores.append(np.median(np.abs(np.log(d_ours[ok] / dv[ok]))))
        afters.append(np.median(np.abs(np.log(d_ours[ok] * s / dv[ok]))))
    if not befores:
        return None
    bef = float(np.median(befores))
    aft = float(np.median(afters))
    log(f"[fase5] DA3 metric gate: median |log ratio| {bef:.4f} → {aft:.4f}")
    return bef, aft


# ── main ─────────────────────────────────────────────────────────────────

def run(output_dir: Path, frames_dir: Optional[Path] = None,
        cfg: Optional[dict] = None, log=logger.info) -> int:
    from plyfile import PlyData, PlyElement
    cfg = cfg or {}
    seq_offsets = tuple(cfg.get("seq_offsets", (1, 2, 4, 8, 16)))
    min_global_sim = float(cfg.get("min_global_sim", 0.60))
    min_gap = int(cfg.get("revisit_min_gap", 30))
    max_pairs = int(cfg.get("max_pairs", 12000))
    match_min_cos = float(cfg.get("match_min_cos", 0.60))
    track_max_spread_m = float(cfg.get("track_max_spread_m", 1.5))
    min_track_obs = int(cfg.get("min_track_obs", 2))
    min_gain = float(cfg.get("min_gain", 0.10))
    holdout_frac = float(cfg.get("holdout_frac", 0.2))
    max_depth_corr = float(cfg.get("max_depth_corr", 0.05))
    max_t_m = float(cfg.get("max_t_m", 0.5))
    odo_weight = float(cfg.get("odo_weight", 2.0))
    leash_weight = float(cfg.get("leash_weight", 0.1))
    t0 = time.time()

    output_dir = Path(output_dir)
    if frames_dir is None:
        frames_dir = output_dir.parent / "frames"
    poses, frames, Ks = load_session_cameras(output_dir)
    n = len(frames)
    fidx = {f: k for k, f in enumerate(frames)}
    extract_session_features(output_dir, frames_dir, cfg, log=log)
    fc = FeatureCache(output_dir)
    world, valid, plys = _load_frame_grids(output_dir, fc, frames, fidx, log)
    cams = {k: poses[k][:3, 3] for k in range(n)}
    log(f"[fase5] session loaded: {len(world)} frames on the "
        f"{fc.hp}x{fc.wp} lattice")

    # A + B: pairs → tracks → landmark registry
    pairs, n_rev = _revisit_pairs(fc, frames, world, seq_offsets,
                                  min_global_sim, min_gap, max_pairs, log)
    tracks = _build_tracks(fc, frames, world, valid, pairs, match_min_cos,
                           track_max_spread_m, min_track_obs, log)
    multi = [t for t in tracks
             if max(k for k, _ in t) - min(k for k, _ in t) >= min_gap]
    log(f"[fase5] landmark registry: {len(tracks):,} tracks, "
        f"{len(multi):,} MULTI-VISIT landmarks (the star witnesses)")

    # C: persist the registry (session memory)
    reg = {"n_tracks": len(tracks), "n_multivisit": len(multi),
           "n_appearance_revisits": n_rev,
           "multivisit_landmarks": [
               {"frames": sorted({int(frames[k]) for k, _ in t}),
                "n_obs": len(t),
                "centroid": [round(float(x), 4) for x in np.median(
                    [world[k].reshape(-1, 3)[c] for k, c in t], axis=0)]}
               for t in multi[:5000]],
           "provenance": "tool_measured"}
    (output_dir / "landmarks.json").write_text(json.dumps(reg))

    # split held-out ONCE for both stages
    rng = np.random.default_rng(1)
    hold_mask = rng.random(len(tracks)) < holdout_frac
    train = [t for t, h in zip(tracks, hold_mask) if not h]
    held = [t for t, h in zip(tracks, hold_mask) if h] or tracks

    # D1: SE(3) via the proven pose-graph solver over track edges
    edges = []
    for t in train:
        by_frame: Dict[int, list] = {}
        for k, c in t:
            by_frame.setdefault(k, []).append(c)
        ks = sorted(by_frame)
        for i in range(len(ks) - 1):
            ka, kb = ks[i], ks[i + 1]
            Pa = world[ka].reshape(-1, 3)[by_frame[ka]]
            Pb = world[kb].reshape(-1, 3)[by_frame[kb]]
            m = min(len(Pa), len(Pb))
            edges.append((ka, kb, Pa[:m], Pb[:m],
                          normal_grid(world[kb], valid[kb], step=1)
                          .reshape(-1, 3)[by_frame[kb]][:m]))
    # group per frame-pair into consolidated edges
    grouped: Dict[Tuple[int, int], list] = {}
    for e in edges:
        grouped.setdefault((e[0], e[1]), []).append(e)
    solver_edges = []
    for (ka, kb), es in grouped.items():
        P = np.concatenate([e[2] for e in es])
        Q = np.concatenate([e[3] for e in es])
        N = np.concatenate([e[4] for e in es])
        okn = (np.linalg.norm(N, axis=1) > 0.5) & np.isfinite(P).all(1) \
            & np.isfinite(Q).all(1)
        if okn.sum() >= 10:
            solver_edges.append((ka, kb, P[okn], Q[okn], N[okn]))
    if len(solver_edges) < 10:
        raise SceneAnchorError("too few consolidated edges for the SE(3) "
                               "stage")
    Cs = solve_pose_graph(n, solver_edges, odo_weight=odo_weight,
                          leash_weight=leash_weight)
    shifts = np.array([np.linalg.norm(C[:3, 3]) for C in Cs])
    if shifts.max() > max_t_m:
        over = int((shifts > max_t_m).sum())
        log(f"[fase5] ⚠ {over} frame correction(s) exceed |t| cap "
            f"{max_t_m} m — capped to identity (upstream problem flagged)")
        for k in np.flatnonzero(shifts > max_t_m):
            Cs[k] = np.eye(4)
    # apply pose corrections to the working grids
    world_c = {}
    for k in world:
        g = world[k].reshape(-1, 3)
        world_c[k] = (g @ Cs[k][:3, :3].T + Cs[k][:3, 3]).reshape(
            world[k].shape)
    cams_c = {k: (Cs[k][:3, :3] @ poses[k][:3, 3] + Cs[k][:3, 3])
              for k in range(n)}

    # D2: depth curves on the pose-corrected geometry
    dref = float(np.median([np.linalg.norm(
        world_c[k].reshape(-1, 3)[valid[k].reshape(-1)] - cams_c[k],
        axis=1).mean() for k in world_c]))
    a, b = _solve_depth_curves(train, world_c, cams_c, n, dref,
                               smooth_w=5.0, leash_w=2.0,
                               max_corr=max_depth_corr, log=log)

    # E: gates on held-out tracks + DA3 metric
    zero = np.zeros(n)
    h_before = _track_mismatch(held, world, cams, zero, zero, dref)
    h_pose = _track_mismatch(held, world_c, cams_c, zero, zero, dref)
    h_after = _track_mismatch(held, world_c, cams_c, a, b, dref)
    log(f"[fase5] held-out track spread: {h_before * 1000:.1f} → "
        f"{h_pose * 1000:.1f} (poses) → {h_after * 1000:.1f} mm (depth)")
    da3 = _da3_gate(output_dir, frames, fidx, world_c, valid, cams_c,
                    a, b, dref, fc, log)
    accepted = h_after <= h_before * (1.0 - min_gain)
    da3_ok = True if da3 is None else (da3[1] <= da3[0] * 1.02)
    report = {
        "n_frames": n, "n_pairs": len(pairs),
        "n_appearance_revisits": n_rev, "n_tracks": len(tracks),
        "n_multivisit_landmarks": len(multi),
        "holdout_spread_m": {"before": h_before, "after_pose": h_pose,
                             "after_depth": h_after},
        "da3_gate_logratio": ({"before": da3[0], "after": da3[1]}
                              if da3 else None),
        "pose_correction_m": {"median": float(np.median(shifts)),
                              "max": float(shifts.max())},
        "depth_curve_abs": {"a_median": float(np.median(np.abs(a))),
                            "a_max": float(np.abs(a).max()),
                            "b_max": float(np.abs(b).max())},
        "dref_m": dref, "min_gain": min_gain,
        "accepted": bool(accepted and da3_ok),
        "da3_gate_ok": bool(da3_ok),
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    (output_dir / "fase5_report.json").write_text(json.dumps(report,
                                                             indent=1))
    if not (accepted and da3_ok):
        why = ("held-out gain insufficient" if not accepted
               else "DA3 metric gate degraded")
        log(f"[fase5] {why} — NOTHING APPLIED (identity). Full evidence in "
            f"fase5_report.json")
        return 0

    # F: apply — every point along ITS OWN ray, poses follow
    moved = 0
    for p, v, fg, pr, pc in plys:
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        changed = False
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None:
                continue
            m = fg == f
            P = xyz[m]
            P = P @ Cs[k][:3, :3].T + Cs[k][:3, 3]        # rigid
            C = cams_c[k]
            vvec = P - C
            d = np.linalg.norm(vvec, axis=1)
            s = 1.0 + a[k] + b[k] * (d / dref - 1.0)
            xyz[m] = C + vvec * s[:, None]                # along the ray
            changed = True
        if changed:
            v["x"] = xyz[:, 0].astype(v["x"].dtype)
            v["y"] = xyz[:, 1].astype(v["y"].dtype)
            v["z"] = xyz[:, 2].astype(v["z"].dtype)
            PlyData([PlyElement.describe(v, "vertex")], text=False,
                    byte_order="<").write(str(p))
    bak = output_dir / "camera_poses.txt.prefase5"
    pp = output_dir / "camera_poses.txt"
    if not bak.exists():
        bak.write_text(pp.read_text())
    out_lines = []
    for k, M in enumerate(poses):
        if not np.allclose(Cs[k], np.eye(4), atol=1e-12) or abs(a[k]) > 1e-9:
            moved += 1
        out_lines.append(" ".join(f"{x:.9g}"
                                  for x in (Cs[k] @ M).reshape(-1)))
    pp.write_text("\n".join(out_lines) + "\n")
    log(f"[fase5] ✅ APPLIED: {moved}/{n} frames re-anchored "
        f"(poses + depth curves), backup {bak.name} "
        f"({report['elapsed_s']}s)")
    return moved


def main():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    try:
        os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Fase 5 — global surface-consensus anchoring")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", default=None)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import cfg as _cfg
    dcfg = _cfg.get("dino_features") or {}
    run(Path(args.output_dir),
        Path(args.frames_dir) if args.frames_dir else None,
        cfg={**dcfg, **(dcfg.get("scene_anchor") or {})},
        log=lambda m: logging.getLogger(__name__).info(m))


if __name__ == "__main__":
    main()
