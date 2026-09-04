"""
FASE 1/2 of the DINOv3 plan (USER 2026-09-04): multi-view feature-consistency
score per cloud point, wired after CloudCompy and before Potree.

FASE 1 (``dino_features.score.enabled``): every point of cleaned_cloud.ply is
scored by the coherence of its DINOv3 feature across the views that saw it —
own feature (source frame + source pixel, both already stored per point) vs
the feature sampled at its reprojection into neighbouring keyframes
(z-buffer occlusion-checked). The score lands as extra PLY channels
(``feature_score`` float, ``feature_views`` uchar) — the cloud's geometry is
NEVER touched; filtering is a separate, reversible decision.

FASE 2 (``dino_features.filter.enabled``): with the Ω write gate opened
(low/zero conf_percentile) the score does the noise/surface separation:
keep = enough views AND (semantic coherence OR high Ω confidence). Two
independent witnesses instead of one global percentile. The pre-filter
cloud is kept as ``cleaned_cloud.prefeature.ply``.

Provenance: tool_measured — reprojection + cosine over measured pixels.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from reconstruction.dino_features import (FeatureCache,
                                          calibrate_provenance_grid,
                                          extract_session_features,
                                          load_session_cameras)

logger = logging.getLogger(__name__)

_VIEW_OFFSETS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)


def _read_cloud(path: Path):
    from segmentation.perfect_object import _read_ply_fields
    return _read_ply_fields(path)


def _write_cloud(path: Path, fields: dict, log):
    """Binary-little-endian PLY writer preserving field order/types."""
    order = list(fields.keys())
    _T = {"float32": "float", "float64": "double", "uint8": "uchar",
          "int8": "char", "int16": "short", "uint16": "ushort",
          "int32": "int", "uint32": "uint"}
    n = len(fields[order[0]])
    dt = np.dtype([(k, fields[k].dtype.str) for k in order])
    packed = np.empty(n, dt)
    for k in order:
        packed[k] = fields[k]
    tmp = path.with_suffix(".tmp.ply")
    with open(tmp, "wb") as f:
        head = ["ply", "format binary_little_endian 1.0",
                f"element vertex {n}"]
        head += [f"property {_T[fields[k].dtype.name]} {k}" for k in order]
        head.append("end_header")
        f.write(("\n".join(head) + "\n").encode("ascii"))
        packed.tofile(f)
    tmp.replace(path)
    log(f"[dino-score] wrote {path.name} ({n:,} pts, "
        f"{path.stat().st_size / 1e6:.0f} MB)")


def _frame_zbuffer(xyz_sub: np.ndarray, w2c: np.ndarray, K: np.ndarray,
                   Hg: int, Wg: int, scale: float) -> np.ndarray:
    """Min-depth raster of the (subsampled) cloud in one view, at
    (Hg*scale, Wg*scale) — the occlusion witness for that view."""
    h = max(2, int(Hg * scale))
    w = max(2, int(Wg * scale))
    pc = xyz_sub @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    m = z > 1e-6
    u = (K[0, 0] * pc[m, 0] / z[m] + K[0, 2]) * (w / (2 * K[0, 2]))
    v = (K[1, 1] * pc[m, 1] / z[m] + K[1, 2]) * (h / (2 * K[1, 2]))
    ui = u.astype(np.int64)
    vi = v.astype(np.int64)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    buf = np.full(h * w, np.inf, np.float32)
    np.minimum.at(buf, vi[ok] * w + ui[ok], z[m][ok].astype(np.float32))
    return buf.reshape(h, w)


def run(output_dir: Path, frames_dir: Optional[Path] = None,
        cfg: Optional[dict] = None, log=logger.info,
        allow_filter: bool = True) -> Optional[dict]:
    """Score cleaned_cloud.ply (fase 1); optionally filter it (fase 2).
    Returns the report dict, or None when inputs are missing."""
    cfg = cfg or {}
    scfg = cfg.get("score") or {}
    fcfg = cfg.get("filter") or {}
    n_views_cfg = int(scfg.get("views", 6))
    min_views = int(scfg.get("min_views", 2))
    zbuf_scale = float(scfg.get("zbuf_scale", 0.25))
    occl_tol_m = float(scfg.get("occlusion_tol_m", 0.15))
    block = int(scfg.get("block_pts", 2_000_000))

    output_dir = Path(output_dir)
    ply = output_dir / "cleaned_cloud.ply"
    if not ply.exists():
        raise RuntimeError("[dino-score] cleaned_cloud.ply missing — "
                           "nothing fails silently (USER 2026-09-04)")
    if frames_dir is None:
        frames_dir = output_dir.parent / "frames"

    t0 = time.time()
    extract_session_features(output_dir, frames_dir, cfg, log=log)
    fc = FeatureCache(output_dir)
    poses, frames, Ks = load_session_cameras(output_dir)
    fields = _read_cloud(ply)
    need = ("x", "y", "z", "confidence", "frame_global",
            "pixel_row", "pixel_col")
    if any(k not in fields for k in need):
        raise RuntimeError(
            f"[dino-score] cloud lacks provenance fields "
            f"({[k for k in need if k not in fields]}) — nothing fails "
            f"silently (USER 2026-09-04)")
    xyz = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(
        np.float64)
    fg = np.asarray(fields["frame_global"], np.int64)
    pr = np.asarray(fields["pixel_row"], np.int64)
    pc_ = np.asarray(fields["pixel_col"], np.int64)
    n = len(xyz)

    Hg, Wg, cal_err = calibrate_provenance_grid(
        xyz, fg, pr, pc_, poses, frames, Ks, log=log)

    fidx = {f: k for k, f in enumerate(frames)}
    src_k = np.array([fidx.get(int(f), -1) for f in fg], np.int64)
    valid_src = (src_k >= 0) & np.array([fc.has(int(f)) for f in
                                         np.unique(fg)])[np.searchsorted(
        np.unique(fg), fg)]

    # per-view geometry, lazily built
    w2c_all = np.linalg.inv(poses)
    zbufs: dict = {}
    grids: dict = {}
    rng = np.random.default_rng(0)
    zsub = xyz[rng.choice(n, min(n, 3_000_000), replace=False)]

    def _zbuf(k):
        if k not in zbufs:
            zbufs[k] = _frame_zbuffer(zsub, w2c_all[k], Ks[k], Hg, Wg,
                                      zbuf_scale)
        return zbufs[k]

    def _grid(k):
        if k not in grids:
            if len(grids) > 48:      # bounded feature-grid cache
                grids.pop(next(iter(grids)))
            grids[k] = fc.grid(frames[k])
        return grids[k]

    score_sum = np.zeros(n, np.float32)
    score_cnt = np.zeros(n, np.uint8)
    offsets = [o for o in _VIEW_OFFSETS][:max(n_views_cfg * 2, 4)]

    order = np.argsort(src_k, kind="stable")   # frame-coherent processing
    for b0 in range(0, n, block):
        idx = order[b0:b0 + block]
        idx = idx[valid_src[idx]]
        if len(idx) == 0:
            continue
        # own feature at the SOURCE pixel
        own = np.empty((len(idx), fc.dim), np.float32)
        for k in np.unique(src_k[idx]):
            m = src_k[idx] == k
            g = _grid(int(k))
            own[m] = fc.sample(g, (pr[idx[m]] + 0.5) / Hg,
                               (pc_[idx[m]] + 0.5) / Wg)
        P = xyz[idx]
        used = np.zeros(len(idx), np.uint8)
        ssum = np.zeros(len(idx), np.float32)
        for off in offsets:
            tgt = src_k[idx] + off
            mtgt = (tgt >= 0) & (tgt < len(frames)) & \
                   (used < n_views_cfg)
            if not mtgt.any():
                continue
            for k in np.unique(tgt[mtgt]):
                m = mtgt & (tgt == k)
                w2c = w2c_all[k]
                pcam = P[m] @ w2c[:3, :3].T + w2c[:3, 3]
                z = pcam[:, 2]
                front = z > 1e-6
                if not front.any():
                    continue
                K = Ks[k]
                u = K[0, 0] * pcam[front, 0] / z[front] + K[0, 2]
                v = K[1, 1] * pcam[front, 1] / z[front] + K[1, 2]
                un = u / (2 * K[0, 2])
                vn = v / (2 * K[1, 2])
                inb = (un >= 0) & (un < 1) & (vn >= 0) & (vn < 1)
                if not inb.any():
                    continue
                zb = _zbuf(int(k))
                zh, zw = zb.shape
                zi = zb[np.clip((vn[inb] * zh).astype(np.int64), 0, zh - 1),
                        np.clip((un[inb] * zw).astype(np.int64), 0, zw - 1)]
                vis = z[front][inb] <= zi * 1.02 + occl_tol_m
                if not vis.any():
                    continue
                sel_local = np.flatnonzero(m)[front][inb][vis]
                fv = fc.sample(_grid(int(k)), vn[inb][vis], un[inb][vis])
                cosv = np.einsum("nd,nd->n", own[sel_local], fv)
                ssum[sel_local] += cosv.astype(np.float32)
                used[sel_local] += 1
        score_sum[idx] = ssum
        score_cnt[idx] = used
        if (b0 // block) % 5 == 0:
            log(f"[dino-score] {min(b0 + block, n):,}/{n:,} pts scored")

    score = np.where(score_cnt > 0, score_sum / np.maximum(score_cnt, 1),
                     -1.0).astype(np.float32)
    fields["feature_score"] = score
    fields["feature_views"] = score_cnt
    _write_cloud(ply, fields, log)

    scored = score_cnt >= min_views
    report = {
        "n_points": int(n),
        "grid": [int(Hg), int(Wg)],
        "calibration_median_px": round(float(cal_err), 2),
        "scored_frac": round(float(scored.mean()), 4),
        "score_median": round(float(np.median(score[scored])), 4)
        if scored.any() else None,
        "score_p10": round(float(np.percentile(score[scored], 10)), 4)
        if scored.any() else None,
        "views_mean": round(float(score_cnt[scored].mean()), 2)
        if scored.any() else None,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }

    # ── FASE 2: two-witness filter (opt-in) ──────────────────────────────
    if allow_filter and bool(fcfg.get("enabled", False)):
        s_keep = float(fcfg.get("score_keep", 0.55))
        c_pct = float(fcfg.get("conf_keep_pct", 60.0))
        conf = np.asarray(fields["confidence"], np.float64)
        c_thr = float(np.percentile(conf[conf > 1e-5], c_pct))
        keep = (~scored) | (score >= s_keep) | (conf >= c_thr)
        backup = ply.with_name("cleaned_cloud.prefeature.ply")
        if not backup.exists():
            ply.rename(backup)
        else:
            ply.unlink()
        _write_cloud(ply, {k: v[keep] for k, v in fields.items()}, log)
        report["filter"] = {
            "score_keep": s_keep, "conf_keep_pct": c_pct,
            "conf_thr": round(c_thr, 4),
            "dropped": int((~keep).sum()),
            "dropped_frac": round(float((~keep).mean()), 4),
            "backup": backup.name,
        }
        log(f"[dino-score] FILTER: dropped {int((~keep).sum()):,} pts "
            f"({float((~keep).mean()) * 100:.1f}%) — backup {backup.name}")

    (output_dir / "feature_score_report.json").write_text(
        json.dumps(report, indent=1))
    log(f"[dino-score] ✅ scored {float(scored.mean()) * 100:.0f}% of "
        f"{n:,} pts, median {report['score_median']} "
        f"({report['elapsed_s']}s)")
    return report
