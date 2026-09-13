"""Per-KEYFRAME pose graph from the pairwise consistency evidence.

USER 2026-09-09: the only good thing so far is the matrix — every pair of
keyframes that wrote the same blocks disagrees by a measured amount. Scale
is coherent inside one scan (not a variable here); what can be wrong is the
pose: rotation + translation per keyframe. This module:

  1. PAIR TRANSFORMS. For every pair (i, j) of the consistency matrix,
     over the blocks BOTH wrote: trimmed 3-D ICP of i's points onto j's
     → T_ij (p_j ≈ T_ij p_i) plus the point-to-plane information matrix
     H_ij = Σ [n; p×n][n; p×n]ᵀ over the inlier correspondences. H carries
     the observability: a pair that shares only a floor has near-zero
     information along the floor and about its normal — that DOF simply
     does not constrain the graph (no classification, no threshold).
  2. GRAPH. Unknown X_k ∈ SE(3) per keyframe, X_0 = I (the start is exact).
     Each pair: X_i = X_j ∘ T_ij  (after the correction both write the
     same surface). Weighted Gauss-Newton, small-perturbation Jacobians
     (world frame), sparse normal equations. Consecutive pairs (1.5 cm)
     keep the walk continuous; distant pairs close the loops. No chunks,
     no distribution rule: the split falls out of the constraints.
  3. WHERE. The increment δX_k = X_k ∘ X_{k-1}⁻¹ is the error BORN at that
     step of the walk (the user's δE_k). Reported per keyframe in degrees
     and centimetres.
  4. CHECK. Every pair's point-level deviation re-measured with the solved
     poses applied to the same samples: if rotation + translation explain
     the matrix, the residuals fall to the neighbour floor everywhere; where
     they do not, something non-rigid lives inside the keyframe — reported,
     not invented.

Read-only: writes ``corrections/kfgraph.json`` + ``kfgraph.png`` +
``kfgraph_poses.npz`` (R_kf, t_kf) for a later, separate apply.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from correction.config import CorrectionConfig
from correction.session import CorrectionSession
from correction.units import chunks_of_keyframe, load_chunk_plan

REPORT_NAME = "kfgraph.json"
PLOT_NAME = "kfgraph.png"
POSES_NAME = "kfgraph_poses.npz"


# ── blocks (same binning as consistency.py) ──────────────────────────────

def bin_blocks(xyz: np.ndarray, cell_m: float, min_points: int):
    lo = xyz.min(0)
    vox = np.floor((xyz - lo) / cell_m).astype(np.int64)
    dims = vox.max(0) + 1
    key = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
    order = np.argsort(key, kind="stable")
    ks_sorted = key[order]
    uniq, starts = np.unique(ks_sorted, return_index=True)
    ends = np.append(starts[1:], len(ks_sorted))
    return [order[s:e] for s, e in zip(starts, ends) if e - s >= min_points]


# ── pair measurement ─────────────────────────────────────────────────────

def _normals(P: np.ndarray, k: int, workers: int) -> np.ndarray:
    tree = cKDTree(P)
    _, nn = tree.query(P, k=min(k, len(P)), workers=workers)
    Q = P[nn]                              # (n, k, 3)
    Q = Q - Q.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', Q, Q)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]                      # smallest eigenvector


def _icp_p2pl(src: np.ndarray, tgt: np.ndarray, tree, Nt: np.ndarray,
              cfg: CorrectionConfig) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Trimmed point-to-PLANE ICP, Levenberg-Marquardt steps linearised
    about the source centroid. Returns (R, t, inlier src idx, target idx)
    with p_t ≈ R p_s + t (world frame). A step is accepted only if the
    trimmed point-to-plane cost drops on the same correspondences, else the
    damping grows: a DOF the shared geometry does not observe (sliding along
    a floor, turning about its normal) has no gradient and never moves, and
    a bad linearisation cannot throw the pair away (the undamped version
    returned 160° turns and 10¹³ m on pccr 31/08)."""
    kc = cfg.kfgraph
    R = np.eye(3); t = np.zeros(3)
    S = src.copy()
    c = S.mean(0)
    workers = cfg.runtime.workers
    lam = kc.damping
    sel = np.arange(len(S)); j = np.zeros(len(S), dtype=np.int64)
    for _ in range(cfg.solve.icp_iters):
        d, j = tree.query(S, workers=workers)
        k = max(cfg.solve.icp_min_corr, int(len(S) * cfg.solve.icp_trim))
        sel = np.argsort(d)[:k]
        P, Q, n = S[sel], tgt[j[sel]], Nt[j[sel]]
        r = np.einsum('ij,ij->i', P - Q, n)
        cost = float(r @ r)
        J = np.hstack([n, np.cross(P - c, n)])
        A = J.T @ J
        g = J.T @ r
        accepted = False
        for _try in range(kc.lm_tries):
            # relative damping on the observed DOF + an absolute floor scaled
            # by the strongest DOF: a direction with less than lam × that
            # information does not move on noise
            dx = np.linalg.solve(A + np.diag(np.diag(A)) * lam
                                 + np.eye(6) * lam * np.diag(A).max(), -g)
            dt, dth = dx[:3], dx[3:]
            Rd = Rotation.from_rotvec(dth).as_matrix()
            P2 = (P - c) @ Rd.T + c + dt
            r2 = np.einsum('ij,ij->i', P2 - Q, n)
            if float(r2 @ r2) < cost:
                accepted = True
                lam = max(lam / kc.lm_factor, kc.damping_min)
                break
            lam *= kc.lm_factor
        if not accepted:
            break
        S = (S - c) @ Rd.T + c + dt
        R = Rd @ R
        t = Rd @ t + (c + dt - Rd @ c)
        if np.linalg.norm(dth) < np.radians(cfg.solve.icp_converge_deg) \
                and np.linalg.norm(dt) < cfg.solve.icp_converge_m:
            break
    return R, t, sel, j[sel]


def measure_pairs(session: CorrectionSession, cfg: CorrectionConfig,
                  log=print, progress=None) -> Tuple[List[dict], dict]:
    """T_ij + H_ij for every pair of keyframes that wrote common blocks."""
    kc = cfg.kfgraph
    cc = cfg.consistency
    rng = np.random.default_rng(cfg.solve.seed)
    workers = cfg.runtime.workers
    ks = session.ks
    # centred frame: rotations about the scene centre keep the pair
    # information and the graph well conditioned (|p| ~ 10 m otherwise)
    xyz = session.xyz - session.xyz.mean(0)
    blocks = bin_blocks(xyz, cc.cell_m, cc.block_min_points)
    # writer → blocks, and per (block, writer) the point indices
    per_block_writers: List[Dict[int, np.ndarray]] = []
    pair_blocks: Dict[Tuple[int, int], List[int]] = {}
    for bi, b in enumerate(blocks):
        kb = ks[b]
        wk, wc = np.unique(kb, return_counts=True)
        ws = {int(k): b[kb == k] for k, c in zip(wk, wc)
              if k >= 0 and c >= cc.writer_min_points}
        per_block_writers.append(ws)
        wl = sorted(ws)
        for a in range(len(wl)):
            for c in range(a + 1, len(wl)):
                pair_blocks.setdefault((wl[a], wl[c]), []).append(bi)
    log(f"  {len(blocks)} blocks, {len(pair_blocks)} keyframe pairs share blocks")
    pairs = []
    samples: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    n_done = 0
    for (i, j), bl in sorted(pair_blocks.items()):
        si = np.concatenate([per_block_writers[b][i] for b in bl])
        sj = np.concatenate([per_block_writers[b][j] for b in bl])
        if len(si) > kc.pair_src_sample:
            si = rng.choice(si, kc.pair_src_sample, replace=False)
        if len(sj) > kc.pair_tgt_sample:
            sj = rng.choice(sj, kc.pair_tgt_sample, replace=False)
        Pi, Pj = xyz[si], xyz[sj]
        tree = cKDTree(Pj)
        # from identity: point-to-plane moves along the normals it sees; a
        # centroid init would plant a value in the directions it cannot see
        d0, _ = tree.query(Pi, workers=workers)
        Nj = _normals(Pj, kc.normal_k, workers)
        R, t, sel, jj = _icp_p2pl(Pi, Pj, tree, Nj, cfg)
        # point-to-plane information over the inlier correspondences,
        # world frame, ordering (t, θ): row = [n, p × n]
        n = Nj[jj]
        p = Pj[jj]
        rows = np.hstack([n, np.cross(p, n)])          # (m, 6)
        H = rows.T @ rows
        before = float(np.median(d0))
        d1, _ = tree.query(Pi @ R.T + t, workers=workers)
        after = float(np.median(d1))
        ev = np.linalg.eigvalsh(H)
        pairs.append({"i": i, "j": j, "n_blocks": len(bl),
                      "n_src": int(len(Pi)), "n_tgt": int(len(Pj)),
                      "R": R, "t": t, "H": H,
                      "dev_before_m": before, "dev_icp_m": after,
                      "rot_deg": float(np.degrees(np.linalg.norm(
                          Rotation.from_matrix(R).as_rotvec()))),
                      "t_norm_m": float(np.linalg.norm(t)),
                      "info_eig": ev.tolist()})
        samples[(i, j)] = (si, sj)
        n_done += 1
        if n_done % 500 == 0:
            msg = f"  pairs: {n_done}/{len(pair_blocks)}"
            log(msg)
            if progress:
                progress(5 + 60 * n_done / len(pair_blocks), msg)
    return pairs, samples


# ── graph solve ──────────────────────────────────────────────────────────

def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _residuals(R, t, pairs):
    """e_ij = log(X_j T_ij X_i⁻¹) as world-frame (t, θ) per pair, and the
    weighted cost Σ eᵀ H e."""
    E = np.zeros((len(pairs), 6))
    cost = 0.0
    for n, pr in enumerate(pairs):
        i, j = pr["i"], pr["j"]
        R_ji = R[j] @ pr["R"]
        t_ji = R[j] @ pr["t"] + t[j]
        RY = R_ji @ R[i].T
        tY = t_ji - RY @ t[i]
        e = np.concatenate([tY, Rotation.from_matrix(RY).as_rotvec()])
        E[n] = e
        cost += float(e @ pr["H"] @ e)
    return E, cost


def solve_graph(n_kf: int, pairs: List[dict], cfg: CorrectionConfig,
                log=print) -> Tuple[np.ndarray, np.ndarray, dict]:
    """X_k = (R_k, t_k) per keyframe, X_0 = I. Constraint per pair:
    X_i = X_j ∘ T_ij, residual e = log(X_j T_ij X_i⁻¹) as a world-frame
    perturbation (t, θ), weighted by the pair's information H_ij.
    Levenberg-Marquardt: a step is accepted only if the exact weighted cost
    drops; otherwise the damping grows (a step along a DOF no pair observes
    would otherwise run away — the unobserved directions stay put)."""
    kc = cfg.kfgraph
    R = np.tile(np.eye(3), (n_kf, 1, 1))
    t = np.zeros((n_kf, 3))
    free = np.arange(1, n_kf)               # kf 0 fixed
    col = {k: 6 * (k - 1) for k in free}
    N = 6 * (n_kf - 1)
    lam = kc.damping
    hist = []
    E, cost = _residuals(R, t, pairs)
    for it in range(kc.gn_iters):
        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)
        for n, pr in enumerate(pairs):
            i, j = pr["i"], pr["j"]
            H = pr["H"]; e = E[n]
            for k_, sgn in ((j, 1.0), (i, -1.0)):
                if k_ == 0:
                    continue
                c0 = col[k_]
                b[c0:c0 + 6] += -sgn * (H @ e)
                for k2, sgn2 in ((j, 1.0), (i, -1.0)):
                    if k2 == 0:
                        continue
                    c1 = col[k2]
                    A[c0:c0 + 6, c1:c1 + 6] = A[c0:c0 + 6, c1:c1 + 6] + sgn * sgn2 * H
        A = A.tocsr()
        diag = A.diagonal()
        accepted = False
        for _try in range(kc.lm_tries):
            dx = spsolve(A + sparse.diags(lam * diag + lam * diag.max()), b)
            R2, t2 = R.copy(), t.copy()
            step = 0.0
            for k in free:
                c0 = col[k]
                dt, dth = dx[c0:c0 + 3], dx[c0 + 3:c0 + 6]
                Rd = Rotation.from_rotvec(dth).as_matrix()
                R2[k] = Rd @ R[k]
                t2[k] = Rd @ t[k] + dt
                step = max(step, float(np.linalg.norm(dt)), float(np.linalg.norm(dth)))
            E2, cost2 = _residuals(R2, t2, pairs)
            if cost2 < cost:
                R, t, E, cost_prev, cost = R2, t2, E2, cost, cost2
                lam = max(lam / kc.lm_factor, kc.damping_min)
                accepted = True
                break
            lam *= kc.lm_factor
        hist.append({"iter": it, "cost": cost, "max_step": step, "lambda": lam,
                     "accepted": accepted})
        log(f"  LM iter {it}: cost {cost:.1f}, max step {step:.4f}, λ {lam:.2e}"
            f"{'' if accepted else ' (no step accepted)'}")
        if not accepted or step < kc.gn_converge:
            break
    return R, t, {"iterations": hist, "final_cost": cost}


# ── verification with the solved poses ───────────────────────────────────

def recheck_pairs(session: CorrectionSession, pairs: List[dict], samples,
                  R: np.ndarray, t: np.ndarray, cfg: CorrectionConfig
                  ) -> np.ndarray:
    workers = cfg.runtime.workers
    xyz = session.xyz - session.xyz.mean(0)
    after = np.zeros(len(pairs))
    for n, pr in enumerate(pairs):
        i, j = pr["i"], pr["j"]
        si, sj = samples[(i, j)]
        Pi = xyz[si] @ R[i].T + t[i]
        Pj = xyz[sj] @ R[j].T + t[j]
        d, _ = cKDTree(Pj).query(Pi, workers=workers)
        after[n] = float(np.median(d))
    return after


# ── entry ────────────────────────────────────────────────────────────────

def run_kfgraph(session: CorrectionSession, cfg: CorrectionConfig, log=print,
                progress=None, plot: bool = True) -> dict:
    t0 = time.time()
    n_kf = session.n_kf

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    cache = session.output_dir / "corrections" / "kfgraph_pairs.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if int(z["n_points"]) == session.n_points and int(z["n_kf"]) == n_kf:
            _p(3, f"loading measured pairs from {cache.name}...")
            pairs = list(z["pairs"])
            samples = dict(z["samples"].item())
        else:
            pairs = None
    else:
        pairs = None
    if pairs is None:
        _p(3, "measuring T_ij + information for every pair sharing blocks...")
        pairs, samples = measure_pairs(session, cfg, log=log, progress=progress)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, pairs=np.array(pairs, dtype=object),
                 samples=np.array(samples, dtype=object),
                 n_points=session.n_points, n_kf=n_kf)
    _p(66, f"solving the keyframe graph ({n_kf} poses, {len(pairs)} pairs)...")
    R, t, sol = solve_graph(n_kf, pairs, cfg, log=log)
    _p(85, "re-measuring every pair with the solved poses...")
    after = recheck_pairs(session, pairs, samples, R, t, cfg)
    before = np.array([pr["dev_before_m"] for pr in pairs])
    icp = np.array([pr["dev_icp_m"] for pr in pairs])

    plan = load_chunk_plan(session.output_dir)
    per_kf = []
    for k in range(n_kf):
        if k == 0:
            dR, dt = np.eye(3), np.zeros(3)
        else:
            dR = R[k] @ R[k - 1].T
            dt = t[k] - dR @ t[k - 1]
        per_kf.append({
            "kf": k, "chunks": chunks_of_keyframe(plan, k),
            "rot_deg": round(float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(R[k]).as_rotvec()))), 3),
            "t_cm": round(float(np.linalg.norm(t[k])) * 100, 1),
            "step_rot_deg": round(float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(dR).as_rotvec()))), 3),
            "step_t_cm": round(float(np.linalg.norm(dt)) * 100, 1),
        })
    gap = np.array([pr["j"] - pr["i"] for pr in pairs])
    bands = []
    for lo_, hi_ in ((1, 3), (4, 10), (11, 30), (31, 100), (101, n_kf)):
        m = (gap >= lo_) & (gap <= hi_)
        if m.any():
            bands.append({"gap_kf": [lo_, hi_], "n_pairs": int(m.sum()),
                          "before_cm": round(float(np.median(before[m])) * 100, 1),
                          "after_cm": round(float(np.median(after[m])) * 100, 1),
                          "after_p90_cm": round(float(np.percentile(after[m], 90)) * 100, 1),
                          "after_max_cm": round(float(after[m].max()) * 100, 1)})
    unexplained = [{"kf": [pr["i"], pr["j"]], "before_cm": round(b_ * 100, 1),
                    "after_cm": round(a_ * 100, 1), "n_blocks": pr["n_blocks"]}
                   for pr, b_, a_ in zip(pairs, before, after)
                   if a_ > cfg.kfgraph.floor_m]
    unexplained.sort(key=lambda x: -x["after_cm"])
    report = {
        "n_keyframes": n_kf, "n_pairs": len(pairs),
        "solver": sol, "bands": bands, "per_keyframe": per_kf,
        "pairs": [{"kf": [pr["i"], pr["j"]], "n_blocks": pr["n_blocks"],
                   "rot_deg": round(pr["rot_deg"], 3),
                   "t_cm": round(pr["t_norm_m"] * 100, 1),
                   "before_cm": round(b_ * 100, 1),
                   "icp_cm": round(c_ * 100, 1),
                   "after_cm": round(a_ * 100, 1)}
                  for pr, b_, c_, a_ in zip(pairs, before, icp, after)],
        "unexplained_pairs": unexplained,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    out = session.output_dir / "corrections"
    out.mkdir(exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=1))
    # back to the world frame: p' = R (p − c) + c + t  →  t_world = t + c − R c
    c = session.xyz.mean(0)
    t_world = t + c - np.einsum('kij,j->ki', R, c)
    np.savez(out / POSES_NAME, R_kf=R, t_kf=t_world, centre=c)
    if plot:
        report["plot"] = _plot(report, pairs, before, after, n_kf, plan, out / PLOT_NAME)
    _p(100, f"keyframe graph done: {len(pairs)} pairs, "
            f"{len(unexplained)} above the floor after, {report['elapsed_s']} s")
    return report


def _plot(report, pairs, before, after, n_kf, plan, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Mb = np.full((n_kf, n_kf), np.nan); Ma = np.full((n_kf, n_kf), np.nan)
    for pr, b_, a_ in zip(pairs, before, after):
        Mb[pr["i"], pr["j"]] = Mb[pr["j"], pr["i"]] = b_ * 100
        Ma[pr["i"], pr["j"]] = Ma[pr["j"], pr["i"]] = a_ * 100
    vmax = np.nanpercentile(Mb, 98)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, M, title in ((axs[0], Mb, "before (cm)"), (axs[1], Ma, "after solved poses (cm)")):
        im = ax.imshow(M, origin="lower", cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(f"pairwise deviation {title}"); ax.set_xlabel("keyframe")
        fig.colorbar(im, ax=ax, fraction=1 / 20)
    rows = report["per_keyframe"]
    ax = axs[2]
    ax.plot([r["kf"] for r in rows], [r["step_t_cm"] for r in rows], "-",
            color="#2266cc", label="δt per step (cm)")
    ax2 = ax.twinx()
    ax2.plot([r["kf"] for r in rows], [r["step_rot_deg"] for r in rows], "-",
             color="#cc4422", label="δθ per step (°)")
    if plan:
        for i, (a, b) in enumerate(plan["chunk_ranges"]):
            ax.axvspan(a, b, color="#888888", alpha=(6 if i % 2 else 12) / 100)
    ax.set_xlabel("keyframe"); ax.set_ylabel("cm", color="#2266cc")
    ax2.set_ylabel("deg", color="#cc4422")
    ax.set_title("error born at each step of the walk (δX_k)")
    ax.grid(alpha=3 / 10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name
