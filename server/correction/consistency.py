"""Multi-view consistency of the reconstruction — region-centric, ALL
observers (USER 2026-09-09: "me paro en una región de la nube y veo quién
me ve, qué KF me crearon; la verificación se debe hacer con TODOS").

Nothing here decides which two keyframes to compare and nothing requires a
temporal gap: every region of the cloud is seen by many keyframes — the
consecutive ones while the camera moves and the later ones when it comes
back — and every keyframe that wrote points in the region is checked
against the CONSENSUS of all the others that wrote there. A keyframe whose
surface lands away from that consensus carries an error; where it happens
(which regions), who (which keyframes) and how much (the deviation vector)
is the output. This module VERIFIES only — it never modifies geometry.

Per region (fixed block of ``cell_m``):
  * writers: keyframes that created points inside the block (they saw it,
    by construction — the occlusion question is answered by the points);
  * viewers: keyframes whose camera sees the block unoccluded (frustum +
    their own z-buffer) — reported so the user knows who looked at it;
  * per writer: median NN distance and median displacement vector of its
    points against the points of all OTHER writers of the block.
Per keyframe: its point-weighted deviation over every block it wrote.

Output: ``output/corrections/consistency.json`` + ``consistency.png``
(deviation along the walk).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from correction.config import CorrectionConfig
from correction.revisit import _project, load_intrinsics, provenance_grid
from correction.session import CorrectionSession
from correction.units import chunks_of_keyframe, load_chunk_plan

REPORT_NAME = "consistency.json"
PLOT_NAME = "consistency.png"


def _zbuffer(P_own: np.ndarray, w2c: np.ndarray, K: np.ndarray,
             grid: Tuple[int, int], cs: int, depth_min: float) -> np.ndarray:
    """Coarse depth image (median depth per cs×cs cell) of a keyframe's own
    points; NaN where it measured nothing."""
    H, W = grid
    Hc, Wc = H // cs + 1, W // cs + 1
    u, v, z = _project(P_own, w2c, K)
    ok = (z > depth_min) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    cell = (v[ok] // cs).astype(np.int64) * Wc + (u[ok] // cs).astype(np.int64)
    zbuf = np.full(Hc * Wc, np.nan)
    if len(cell):
        srt = np.argsort(cell, kind="stable")
        cs_sorted, zs_sorted = cell[srt], z[ok][srt]
        uniq, start = np.unique(cs_sorted, return_index=True)
        end = np.append(start[1:], len(cs_sorted))
        for c_, s_, e_ in zip(uniq, start, end):
            zbuf[c_] = np.median(zs_sorted[s_:e_])
    return zbuf


def check_consistency(session: CorrectionSession, cfg: CorrectionConfig,
                      log=print, progress=None, plot: bool = True) -> dict:
    t0 = time.time()
    cc = cfg.consistency
    rc = cfg.revisit
    rng = np.random.default_rng(cfg.solve.seed)
    workers = cfg.runtime.workers
    n_kf = session.n_kf
    xyz, ks = session.xyz, session.ks

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    # ── blocks ────────────────────────────────────────────────────────────
    _p(3, f"binning {len(xyz):,} points into {cc.cell_m} m blocks...")
    lo = xyz.min(0)
    vox = np.floor((xyz - lo) / cc.cell_m).astype(np.int64)
    dims = vox.max(0) + 1
    key = (vox[:, 0] * dims[1] + vox[:, 1]) * dims[2] + vox[:, 2]
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    uniq_keys, starts = np.unique(key_sorted, return_index=True)
    ends = np.append(starts[1:], len(key_sorted))
    blocks_idx = [order[s:e] for s, e in zip(starts, ends)
                  if e - s >= cc.block_min_points]
    log(f"  {len(blocks_idx)} blocks with ≥ {cc.block_min_points} points")

    # ── viewers (frustum + own z-buffer, unoccluded) ──────────────────────
    _p(10, "viewers per block (frustum + z-buffer of every keyframe)...")
    K = load_intrinsics(session.output_dir, n_kf)
    grid = provenance_grid(session)
    H, W = grid
    cs = rc.zbuffer_cell_px
    Wc = W // cs + 1
    w2cs = np.linalg.inv(session.poses)
    own_order = np.argsort(ks, kind="stable")
    own_bounds = np.searchsorted(ks[own_order], np.arange(-1, n_kf + 1))
    probe_idx = [b[rng.choice(len(b), min(cc.viewer_probe, len(b)),
                              replace=False)] for b in blocks_idx]
    probe_owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(probe_idx)])
    probes = xyz[np.concatenate(probe_idx)]
    probe_count = np.bincount(probe_owner, minlength=len(blocks_idx)).astype(float)
    viewers: List[List[int]] = [[] for _ in blocks_idx]
    for k in range(n_kf):
        own = own_order[own_bounds[k + 1]:own_bounds[k + 2]]
        if not len(own):
            continue
        zbuf = _zbuffer(xyz[own], w2cs[k], K[k], grid, cs, rc.depth_min_m)
        u, v, z = _project(probes, w2cs[k], K[k])
        vis = ((z > rc.depth_min_m) & (z < rc.depth_max_m)
               & (u >= 0) & (u < W) & (v >= 0) & (v < H))
        cell = np.full(len(z), -1, dtype=np.int64)
        cell[vis] = (v[vis] // cs).astype(np.int64) * Wc + (u[vis] // cs).astype(np.int64)
        zc = np.full(len(z), np.nan)
        zc[vis] = zbuf[cell[vis]]
        tol = np.maximum(rc.same_surface_tol_m, rc.same_surface_tol_rel * z)
        # unoccluded: the keyframe measured nothing in FRONT of the probe
        # (its surface there is at the probe's depth or beyond / unmeasured)
        seen = vis & (np.isnan(zc) | (z <= zc + tol))
        frac = np.bincount(probe_owner[seen], minlength=len(blocks_idx)) / probe_count
        for b in np.where(frac >= cc.viewer_min_frac)[0]:
            viewers[b].append(k)
        if (k + 1) % 50 == 0:
            log(f"  viewers: {k + 1}/{n_kf} keyframes")

    # ── writers, pairwise ─────────────────────────────────────────────────
    # Every writer against EVERY other writer of the block, not against the
    # pooled others: the pool is dominated by the consecutive keyframes of
    # the same pass, and a keyframe always agrees with its own pass — the
    # disagreement between passes (the duplicate) only shows PAIRWISE.
    _p(45, "every writer of every block against every other writer...")
    blocks_out = []
    pair_rows: Dict[Tuple[int, int], List[Tuple[float, float, np.ndarray]]] = {}
    for bi, b in enumerate(blocks_idx):
        kb = ks[b]
        wk, wc = np.unique(kb, return_counts=True)
        writers = [(int(k), int(c)) for k, c in zip(wk, wc)
                   if k >= 0 and c >= cc.writer_min_points]
        centre = lo + (vox[b[0]] + 0.5) * cc.cell_m
        entry = {"block": bi, "centre_m": [round(float(x), 2) for x in centre],
                 "n_points": int(len(b)), "n_viewers": len(viewers[bi]),
                 "viewers_kf": viewers[bi], "n_writers": len(writers),
                 "writers_kf": [k for k, _ in writers], "pairs": []}
        if len(writers) >= 2:
            pts, trees = {}, {}
            for k, c in writers:
                mine = b[kb == k]
                if len(mine) > cc.writer_query_sample:
                    mine = rng.choice(mine, cc.writer_query_sample, replace=False)
                pts[k] = xyz[mine]
                trees[k] = cKDTree(pts[k])
            wl = [k for k, _ in writers]
            devs = []
            for a_ in range(len(wl)):
                for b_ in range(a_ + 1, len(wl)):
                    i, j = wl[a_], wl[b_]
                    d_ij, nn_ij = trees[j].query(pts[i], workers=workers)
                    d_ji, nn_ji = trees[i].query(pts[j], workers=workers)
                    m_ij, m_ji = float(np.median(d_ij)), float(np.median(d_ji))
                    # the smaller direction is the one from the less-covered
                    # side onto the better-covered one (coverage asymmetry)
                    if m_ij <= m_ji:
                        dev, vec = m_ij, np.median(pts[j][nn_ij] - pts[i], axis=0)
                    else:
                        dev, vec = m_ji, -np.median(pts[i][nn_ji] - pts[j], axis=0)
                    w = float(min(len(pts[i]), len(pts[j])))
                    entry["pairs"].append({"kf": [i, j], "dev_cm": round(dev * 100, 1),
                                           "vec_cm": [round(float(x) * 100, 1) for x in vec]})
                    pair_rows.setdefault((i, j), []).append((dev, w, vec))
                    devs.append((dev, i, j))
            if devs:
                worst = max(devs)
                entry["dev_max_cm"] = round(worst[0] * 100, 1)
                entry["dev_max_pair"] = [worst[1], worst[2]]
                entry["writer_span_kf"] = [wl[0], wl[-1]]
        blocks_out.append(entry)
        if (bi + 1) % 100 == 0:
            _p(45 + 45 * (bi + 1) / len(blocks_idx),
               f"  consistency: {bi + 1}/{len(blocks_idx)} blocks")

    # ── keyframe × keyframe deviation (point-weighted median over blocks) ──
    M = np.full((n_kf, n_kf), np.nan)
    for (i, j), rows in pair_rows.items():
        d = np.array([r[0] for r in rows]); w = np.array([r[1] for r in rows])
        srt = np.argsort(d)
        cw = np.cumsum(w[srt]) / w.sum()
        M[i, j] = M[j, i] = d[srt][int(np.searchsorted(cw, 0.5))]
    plan = load_chunk_plan(session.output_dir)
    per_kf = []
    for k in range(n_kf):
        row = M[k]
        idx = np.where(np.isfinite(row))[0]
        near = idx[np.abs(idx - k) <= cc.near_kf]
        far = idx[np.abs(idx - k) > cc.near_kf]
        per_kf.append({
            "kf": k, "chunks": chunks_of_keyframe(plan, k),
            "n_partners": int(len(idx)),
            "dev_near_cm": (round(float(np.median(row[near])) * 100, 1)
                            if len(near) else None),
            "dev_far_cm": (round(float(np.median(row[far])) * 100, 1)
                           if len(far) else None),
            "far_partners": [int(x) for x in far],
            "dev_far_max_cm": (round(float(row[far].max()) * 100, 1)
                               if len(far) else None),
        })
    np.save(session.output_dir / "corrections" / "consistency_matrix.npy", M)

    report = {"n_keyframes": n_kf, "cell_m": cc.cell_m,
              "n_blocks": len(blocks_out),
              "n_blocks_verifiable": sum(1 for b in blocks_out if b["pairs"]),
              "near_kf": cc.near_kf,
              "per_keyframe": per_kf, "blocks": blocks_out,
              "elapsed_s": round(time.time() - t0, 1),
              "provenance": "tool_measured"}
    out = session.output_dir / "corrections"
    out.mkdir(exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=1))
    if plot:
        report["plot"] = _plot(report, M, plan, out / PLOT_NAME)
    _p(100, f"consistency check done: {report['n_blocks_verifiable']}/"
            f"{len(blocks_out)} blocks verifiable, {report['elapsed_s']} s")
    return report


def _plot(report: dict, M: np.ndarray, plan: Optional[dict], path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["per_keyframe"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                  gridspec_kw={"width_ratios": [1, 1]})
    kf = [r["kf"] for r in rows if r["dev_near_cm"] is not None]
    ax.plot(kf, [r["dev_near_cm"] for r in rows if r["dev_near_cm"] is not None],
            "-", color="#2266cc", label=f"vs neighbours (≤{report['near_kf']} kf)")
    kf2 = [r["kf"] for r in rows if r["dev_far_cm"] is not None]
    ax.plot(kf2, [r["dev_far_cm"] for r in rows if r["dev_far_cm"] is not None],
            ".", color="#cc4422", label=f"vs distant writers (>{report['near_kf']} kf)")
    if plan:
        for i, (a, b) in enumerate(plan["chunk_ranges"]):
            ax.axvspan(a, b, color="#888888", alpha=(6 if i % 2 else 12) / 100)
    ax.set_xlabel("keyframe"); ax.set_ylabel("cm")
    ax.set_title("pairwise deviation of a keyframe's points vs other writers")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=3 / 10)
    im = ax2.imshow(M * 100, origin="lower", cmap="magma_r",
                    vmin=0, vmax=np.nanpercentile(M * 100, 98))
    ax2.set_xlabel("keyframe"); ax2.set_ylabel("keyframe")
    ax2.set_title("keyframe × keyframe deviation (cm, blocks both wrote)")
    fig.colorbar(im, ax=ax2, fraction=1 / 20)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name
