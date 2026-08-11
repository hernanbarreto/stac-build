#!/usr/bin/env python3
"""READ-ONLY A/B harness for scale_align v2 (precision task, Phase A.6).

Compares estimator modes and anchor counts on FINISHED session outputs, without
touching production artifacts. Internal-consistency metrics only (no external
reference — that is Phase E's job):

  1. held-out anchor depth error  — k-fold BY FRAME: fit the mode on train
     anchors, predict DA3 depth on held-out anchors, median |d̂−d|/d (%).
     The primary generalization metric.
  2. jackknife stability          — leave-one-anchor-out spread of s
     (max relative deviation + MAD). The stability metric.
  3. eval-pair depth reprojection — DA3 metric depth of keyframe i projected
     into neighbour keyframe j with poses whose translations are scaled by the
     candidate s: median |z_proj − d_j|/d_j (%). DA3 depths are FIXED metric
     values, so this discriminates the ABSOLUTE s (a wrong s misplaces the
     baseline between cameras and the surfaces misalign along it). Needs DA3
     npz for consecutive-keyframe pairs (tools/extract_da3_anchors.py --pairs).
  4. plane-patch RMS (context)    — RANSAC planes on the scaled cloud. Reported
     for the record; it scales ~linearly with s, so between pure-scalar
     candidates it only reflects the s ratio (said explicitly in the output —
     shape metrics are Phase B/E territory).

Acceptance criteria are DEFINED BEFORE running: docs/scale_ab_criteria.md.

Usage (mapanything or da3 env — numpy only):
    python tools/scale_ab.py <output_dir> [<output_dir> ...] \
        [--modes global_median affine_robust depth_dependent] \
        [--anchor-counts 12 24 32] [--kfold 6] [--out DIR]
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

from reconstruction import scale_model as sm                      # noqa: E402
from reconstruction.scale_align import (_read_poses, _PLY_NP)     # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Estimators per mode (train-frames → s), shared by held-out fold and jackknife
# ──────────────────────────────────────────────────────────────────────────────

AB_MAX_PX = 4000     # per-frame pixel budget for the A/B fits: a 1–2 parameter
                     # robust fit saturates long before 20k px/frame; the full
                     # budget made the (folds × jackknife × modes) sweep crawl


def _decimate(frames, max_px=AB_MAX_PX):
    """Deterministic per-frame pixel decimation for the A/B fit sweep."""
    out = []
    for f in frames:
        if f["z"].size <= max_px:
            out.append(f)
            continue
        step = int(np.ceil(f["z"].size / max_px))
        g = dict(f)
        g["z"], g["d"] = f["z"][::step], f["d"][::step]
        g["z_near_mask"] = f["z_near_mask"][::step]
        g["n_px"] = int(g["z"].size)
        out.append(g)
    return out


def _select_once(frames, mode):
    """Model-kind selection runs ONCE per (mode, K) cell on all frames; held-out
    and jackknife then refit THAT kind on subsets. Re-running the CV/BIC
    selection inside every fold/jackknife subset would be nested CV — quadratic
    cost for no statistical gain (the stability metrics measure the CHOSEN
    estimator, not the selector)."""
    if mode == "global_median":
        return None, {"mode_used": "global_median"}
    sel = sm.select_model(frames, mode)
    return (sel["model"]["kind"] if sel["model"] else None), sel


def _fit_kind(frames, kind):
    """(s, model|None) fitting a fixed model kind (None → production median)."""
    if kind is None:
        return sm.trimmed_median_s(frames), None
    z, d, w = sm._pooled(frames)
    if z.size < 100:
        return sm.trimmed_median_s(frames), None
    model = sm._fit_model(kind, z, d, w)
    if model["params"] is None:
        return sm.trimmed_median_s(frames), None
    return sm.applied_gain(model, frames), model


def _heldout_error(frames, kind, k):
    """k-fold by frame: median |d̂−d|/d (%) on held-out anchors, fixed kind."""
    n = len(frames)
    if n < 3:
        return None
    k = max(2, min(k, n))
    errs = []
    for fold in range(k):
        test_idx = set(range(fold, n, k))
        train = [f for i, f in enumerate(frames) if i not in test_idx]
        test = [f for i, f in enumerate(frames) if i in test_idx]
        if len(train) < 2 or not test:
            continue
        s, model = _fit_kind(train, kind)
        if s is None:
            continue
        for f in test:
            m = f["z_near_mask"]
            z, d = (f["z"][m], f["d"][m]) if m.sum() >= 20 else (f["z"], f["d"])
            pred = sm._predict(model, z) if model is not None else s * z
            errs.append(float(np.median(np.abs(pred - d) / np.maximum(d, 1e-6))))
    return float(100.0 * np.median(errs)) if errs else None


def _jackknife(frames, kind):
    return sm.jackknife_s(frames, estimator=lambda fr: _fit_kind(fr, kind)[0])


def _subset_anchors(frames, k):
    """K frames evenly spread in TIME over the available anchor set (mirrors the
    production evenly-spread pick)."""
    if k >= len(frames):
        return list(frames), len(frames)
    idx = sorted({round(i * (len(frames) - 1) / (k - 1)) for i in range(k)})
    return [frames[int(i)] for i in idx], len(idx)


# ──────────────────────────────────────────────────────────────────────────────
# Eval-pair depth reprojection
# ──────────────────────────────────────────────────────────────────────────────

def _load_calib(output_dir: Path):
    """poses (UNSCALED: .prescale preferred) + per-row intrinsics, keyed by frame
    number. Returns dict num -> (T_c2w 4x4, K 3x3, (H, W)) or None on failure."""
    pp = None
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        cand = base / "camera_poses.txt.prescale"
        if cand.exists():
            pp = cand
            break
    lines, nums, path = _read_poses(output_dir)
    if pp is not None:
        lines = [l for l in pp.read_text().splitlines() if len(l.split()) == 16]
        scaled = False
    else:
        lines = [l for l in lines if len(l.split()) == 16]
        scaled = True                      # no prescale backup → poses already metric
    intr = output_dir / "intrinsic.txt"
    if not intr.exists():
        return None, scaled
    K_rows = [[float(x) for x in ln.split()] for ln in intr.read_text().splitlines()
              if ln.strip()]
    if len(K_rows) != len(nums) or len(lines) != len(nums):
        return None, scaled
    from reconstruction.scale_align import _omega_depth
    shapes = {n: d.shape for n, d in _omega_depth(output_dir).items()}
    out = {}
    for ln, n, kr in zip(lines, nums, K_rows):
        T = np.array([float(x) for x in ln.split()], np.float64).reshape(4, 4)
        fx, fy, cx, cy = kr[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        out[n] = (T, K, shapes.get(n))
    return out, scaled


def _find_pairs(output_dir: Path, calib):
    """Consecutive-KEYFRAME pairs with DA3 npz on both sides."""
    from reconstruction.scale_align import _da3_npz_dir
    da3_dir = _da3_npz_dir(output_dir)
    if da3_dir is None or calib is None:
        return []
    have = sorted(int(p.stem.split("_")[1]) for p in da3_dir.glob("frame_*.npz"))
    _, nums, _ = _read_poses(output_dir)
    kf = sorted(nums)
    pos = {n: i for i, n in enumerate(kf)}
    pairs = []
    hs = set(have)
    for n in have:
        i = pos.get(n)
        if i is None or i + 1 >= len(kf):
            continue
        nxt = kf[i + 1]
        if nxt in hs and n in calib and nxt in calib and (n, nxt) not in pairs:
            pairs.append((n, nxt))
    return pairs


def _load_da3(output_dir: Path, num, shape):
    from reconstruction.scale_align import _da3_npz_dir
    p = Path(_da3_npz_dir(output_dir)) / f"frame_{num}.npz"
    d = np.load(p)["depth"].astype(np.float64)
    if shape is not None and d.shape != shape:
        H, W = shape
        yi = (np.arange(H) * d.shape[0] / H).astype(int).clip(0, d.shape[0] - 1)
        xi = (np.arange(W) * d.shape[1] / W).astype(int).clip(0, d.shape[1] - 1)
        d = d[yi][:, xi]
    return d


def _pair_reproj_err(output_dir: Path, calib, pair, s, max_px=30000):
    """median |z_proj − d_j| / d_j (%) for DA3 depth i → camera j at candidate s."""
    ni, nj = pair
    Ti, Ki, shp_i = calib[ni]
    Tj, Kj, shp_j = calib[nj]
    if shp_i is None or shp_j is None:
        return None
    di = _load_da3(output_dir, ni, shp_i)
    dj = _load_da3(output_dir, nj, shp_j)
    Ti = Ti.copy(); Tj = Tj.copy()
    Ti[:3, 3] *= s
    Tj[:3, 3] *= s
    H, W = di.shape
    v, u = np.mgrid[0:H, 0:W]
    m = np.isfinite(di) & (di > 1e-3)
    u, v, z = u[m].astype(np.float64), v[m].astype(np.float64), di[m]
    if z.size > max_px:
        step = int(np.ceil(z.size / max_px))
        u, v, z = u[::step], v[::step], z[::step]
    x = (u - Ki[0, 2]) / Ki[0, 0] * z
    y = (v - Ki[1, 2]) / Ki[1, 1] * z
    Pw = (Ti @ np.stack([x, y, z, np.ones_like(z)]))[:3]
    Pc = np.linalg.inv(Tj) @ np.vstack([Pw, np.ones(Pw.shape[1])])
    zc = Pc[2]
    ok = zc > 1e-3
    if ok.sum() < 200:
        return None
    uu = (Kj[0, 0] * Pc[0, ok] / zc[ok] + Kj[0, 2]).round().astype(int)
    vv = (Kj[1, 1] * Pc[1, ok] / zc[ok] + Kj[1, 2]).round().astype(int)
    zc = zc[ok]
    Hj, Wj = dj.shape
    inb = (uu >= 0) & (uu < Wj) & (vv >= 0) & (vv < Hj)
    if inb.sum() < 200:
        return None
    samp = dj[vv[inb], uu[inb]]
    zin = zc[inb]
    val = np.isfinite(samp) & (samp > 1e-3)
    if val.sum() < 200:
        return None
    err = np.abs(zin[val] - samp[val]) / samp[val]
    # occlusion guard: pixels seeing a different surface produce gross relative
    # errors; the median over the survivors of a loose 25% gate is the statistic
    keep = err < 0.25
    if keep.sum() < 100:
        return float(100.0 * np.median(err))
    return float(100.0 * np.median(err[keep]))


# ──────────────────────────────────────────────────────────────────────────────
# Plane-patch RMS (context only — see module docstring)
# ──────────────────────────────────────────────────────────────────────────────

def _read_ply_xyz(path: Path, max_pts=400000):
    raw = path.read_bytes()
    hend = raw.find(b"end_header")
    if hend < 0:
        return None
    nl = raw.find(b"\n", hend)
    header = raw[:nl + 1].decode("ascii", "replace")
    fmt, props, n, in_v = None, [], 0, False
    for ln in header.splitlines():
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_v = t[1] == "vertex"
            if in_v:
                n = int(t[2])
        elif t[0] == "property" and in_v and t[1] != "list":
            props.append((t[-1], t[1]))
    if fmt == "ascii" or not all(c in [p[0] for p in props] for c in "xyz"):
        return None
    endian = "<" if "little" in fmt else ">"
    try:
        dt = np.dtype([(nm, endian + _PLY_NP[ty]) for nm, ty in props])
    except KeyError:
        return None
    arr = np.frombuffer(raw[nl + 1:nl + 1 + n * dt.itemsize], dtype=dt)
    step = max(1, n // max_pts)
    return np.stack([arr["x"][::step], arr["y"][::step], arr["z"][::step]], 1).astype(np.float64)


def _plane_rms_mm(pts, n_patches=8, radius=1.0, iters=120, seed=7):
    """RANSAC plane per local patch → inlier RMS (mm). Deterministic seed."""
    if pts is None or len(pts) < 5000:
        return None
    rng = np.random.default_rng(seed)
    rms = []
    for _ in range(n_patches * 3):                       # tries; keep n_patches successes
        c = pts[rng.integers(len(pts))]
        d2 = np.sum((pts - c) ** 2, 1)
        nb = pts[d2 < radius * radius]
        if len(nb) < 800:
            continue
        if len(nb) > 20000:
            nb = nb[:: len(nb) // 20000]
        best = None
        for _i in range(iters):
            tri = nb[rng.choice(len(nb), 3, replace=False)]
            nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            nn = np.linalg.norm(nrm)
            if nn < 1e-9:
                continue
            nrm = nrm / nn
            dist = np.abs((nb - tri[0]) @ nrm)
            inl = dist < 0.02
            if best is None or inl.sum() > best[0]:
                best = (inl.sum(), nrm, tri[0], inl)
        if best is None or best[0] < 400:
            continue
        _, nrm, p0, inl = best
        d = (nb[inl] - p0) @ nrm
        rms.append(float(np.sqrt(np.mean(d * d)) * 1000.0))
        if len(rms) >= n_patches:
            break
    return float(np.median(rms)) if rms else None


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def run_session(output_dir: Path, modes, anchor_counts, kfold):
    out = {"output_dir": str(output_dir)}
    frames_all = sm.collect_frame_samples(output_dir)
    if not frames_all:
        out["error"] = "no paired DA3/omega anchors on disk (extract_da3_anchors.py first)"
        return out
    out["anchors_available"] = len(frames_all)

    calib, calib_scaled = _load_calib(output_dir)
    pairs = _find_pairs(output_dir, calib)
    out["eval_pairs"] = [list(p) for p in pairs]
    if calib is None:
        out["eval_pairs_note"] = "no calib (intrinsic.txt/poses mismatch) — reprojection skipped"
    elif calib_scaled:
        out["eval_pairs_note"] = ("no .prescale pose backup — poses on disk are already "
                                  "metric; reprojection uses them with s_rel = s/s_applied")

    # applied s of the finished run (for s_rel when poses are already scaled)
    marker = output_dir / ".metric_scale_applied"
    s_applied_run = None
    if marker.exists():
        try:
            s_applied_run = float(marker.read_text().strip().split("=")[-1])
        except Exception:
            pass
    out["s_applied_run"] = s_applied_run

    cloud = None
    for name in ("cleaned_cloud.ply", "cleaned_cloud_raw.ply"):
        p = output_dir / name
        if p.exists():
            cloud = _read_ply_xyz(p)
            break
    base_plane_rms = _plane_rms_mm(cloud)
    out["plane_rms_mm_at_applied_s"] = base_plane_rms
    out["plane_rms_note"] = ("context only: scales ~linearly with s, discriminates "
                             "pure-scalar candidates only via the s ratio")

    cells = []
    for K in anchor_counts:
        frames_full, k_eff = _subset_anchors(frames_all, K)
        frames = _decimate(frames_full)
        for mode in modes:
            kind, info = _select_once(frames, mode)
            s, model = _fit_kind(frames, kind)
            cell = {"mode": mode, "anchors_requested": K, "anchors_used": k_eff,
                    "s": s, "mode_used": info.get("mode_used"),
                    "degraded": info.get("degraded"),
                    "degrade_reason": info.get("degrade_reason"),
                    "model": ({"kind": model["kind"], "params": model["params"]}
                              if model else None)}
            cell["heldout_err_pct"] = _heldout_error(frames, kind, kfold)
            cell["jackknife"] = _jackknife(frames, kind)
            if s is not None and calib is not None and pairs:
                s_use = s if not calib_scaled else (
                    s / s_applied_run if s_applied_run else None)
                if s_use is not None:
                    errs = [
                        _pair_reproj_err(output_dir, calib, p, s_use) for p in pairs]
                    errs = [e for e in errs if e is not None]
                    cell["pair_reproj_err_pct"] = (float(np.median(errs))
                                                   if errs else None)
                    cell["pair_reproj_n"] = len(errs)
            if s is not None and base_plane_rms is not None and s_applied_run:
                cell["plane_rms_mm_scaled"] = float(base_plane_rms * s / s_applied_run)
            cells.append(cell)
    out["cells"] = cells
    return out


def _md_table(res):
    lines = [f"### {res['output_dir']}",
             f"anchors available: {res.get('anchors_available')} — eval pairs: "
             f"{len(res.get('eval_pairs', []))} — plane RMS @applied s: "
             f"{res.get('plane_rms_mm_at_applied_s')}",
             "",
             "| mode | K | s | used | held-out err % | jack max dev % | jack MAD % "
             "| pair reproj % | plane RMS mm |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in res.get("cells", []):
        jk = c.get("jackknife") or {}
        lines.append(
            "| {mode} | {K} | {s} | {used} | {ho} | {jmax} | {jmad} | {pr} | {pl} |".format(
                mode=c["mode"], K=c["anchors_used"],
                s=f"{c['s']:.4f}" if c.get("s") else "—",
                used=c.get("mode_used") or "—",
                ho=f"{c['heldout_err_pct']:.2f}" if c.get("heldout_err_pct") is not None else "—",
                jmax=f"{100 * jk['max_dev_rel']:.2f}" if jk.get("max_dev_rel") is not None else "—",
                jmad=f"{100 * jk['mad_rel']:.2f}" if jk.get("mad_rel") is not None else "—",
                pr=f"{c['pair_reproj_err_pct']:.2f}" if c.get("pair_reproj_err_pct") is not None else "—",
                pl=f"{c['plane_rms_mm_scaled']:.1f}" if c.get("plane_rms_mm_scaled") is not None else "—"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dirs", nargs="+")
    ap.add_argument("--modes", nargs="+",
                    default=["global_median", "affine_robust", "depth_dependent"])
    ap.add_argument("--anchor-counts", nargs="+", type=int, default=[12, 24, 32])
    ap.add_argument("--kfold", type=int, default=6)
    ap.add_argument("--out", default=None, help="dir for scale_ab.json / scale_ab.md "
                                                "(default: first output_dir)")
    a = ap.parse_args()

    results = [run_session(Path(d), a.modes, a.anchor_counts, a.kfold)
               for d in a.output_dirs]
    out_dir = Path(a.out) if a.out else Path(a.output_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_ab.json").write_text(json.dumps(results, indent=2))
    md = "# scale_align v2 — A/B (internal consistency)\n\n" + \
         "\n\n".join(_md_table(r) for r in results) + "\n"
    (out_dir / "scale_ab.md").write_text(md)
    print(md)
    print(f"\nwritten: {out_dir / 'scale_ab.json'}  {out_dir / 'scale_ab.md'}")


if __name__ == "__main__":
    main()
