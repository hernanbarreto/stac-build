# STAC-Builder — Phase R: post-anchoring metric-scale re-check (REPORT-ONLY).
#
# The global scale S (reconstruction.scale_align: s = median(DA3/omega depth))
# is estimated at reconstruction time, before any segmentation exists, over
# near-band (+ confidence-gated) pixels of EVERY kind of surface. Once Phase R
# has instances, the same ratio can be re-estimated restricted to the eroded
# masks of STRUCTURAL classes (wall/slab/column/vault/platform/floor): planar,
# static, well-observed surfaces where both depth sources are at their best.
#
# INDEPENDENCE RULE: the re-check uses the RAW depths — the .prephaser backups
# written by the R.5 regularization when it edited a store, else the file as-is.
# Re-estimating S from REGULARIZED depth would be circular: the R.5 planes are
# fit on S-scaled store points, so the blend drags DA3 toward the current S and
# the re-check would just confirm it.
#
# APPLY IS GATED (R.6 metric hierarchy): refined_scale_report() only measures;
# apply_refined_scale() applies the delta as a global similarity ONLY when the
# config gates pass (scale_recheck_apply + min structural frames + min/max
# delta). Outside the gates the delta is logged + persisted in
# phase_r_report.json with the refusal reason — never a silent side effect.
#
# PROVENANCE: ours (reuses reconstruction.scale_align._ratio).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_WHITELIST = ["wall", "slab", "column", "vault", "platform", "floor"]

logger = logging.getLogger("PhaseR")


def _load_depth(path: Path, key: str = "depth", backup_suffix: str = ".prephaser"):
    """Load a per-frame npz array, preferring the pre-regularization backup."""
    bak = path.with_suffix(path.suffix + backup_suffix)
    p = bak if bak.exists() else path
    if not p.exists():
        return None
    try:
        d = np.load(p)
        return d[key].astype(np.float32) if key in d.files else None
    except Exception:
        return None


def refined_scale_report(output_dir, session_dir, config: dict | None = None) -> Optional[dict]:
    """Re-estimate the DA3↔omega metric scale over eroded structural-instance
    masks using RAW depths, and compare it to the applied marker scale.

    Returns None when the session has no basis for the check (not a vggtomega
    run, DA3 depth freed, no segmentation, no structural pixels). Never raises
    for data reasons — the caller treats None/exception as 'skipped'."""
    from reconstruction.scale_align import _ratio          # single ratio implementation
    from .writeback import _erode_mask

    output_dir = Path(output_dir)
    cfg = (config or {}).get("phase_r", {}) if config else {}
    va = ((config or {}).get("reconstruction", {}) or {}).get("vggtomega", {}) or {}
    whitelist = set(cfg.get("depth_reg_whitelist", DEFAULT_WHITELIST))
    erosion_px = int(cfg.get("depth_reg_erosion_px", cfg.get("mask_erosion_px", 8)))
    near_frac = float(va.get("scale_near_frac", 0.25))
    conf_top_frac = float(va.get("scale_conf_top_frac", 0.10))

    da3_dir = output_dir / "da3_run" / "results_output"
    omega_dir = output_dir / "omega_run" / "results_output"
    marker = output_dir / ".metric_scale_applied"
    seg_path = output_dir / "segmentation.json"
    if not (da3_dir.is_dir() and omega_dir.is_dir() and marker.exists() and seg_path.exists()):
        return None
    try:
        s_marker = float(marker.read_text().strip().split("=")[-1])
    except Exception:
        return None

    seg = json.loads(seg_path.read_text())
    npz = np.load(output_dir / seg.get("mask_file", "seg_masks.npz"))
    structural_ids = {i["id"] for i in seg.get("instances", [])
                      if any(w in str(i.get("label", "")) for w in whitelist)}
    if not structural_ids:
        return None
    # mask keys are f{frame}_o{objid} → frame → structural keys
    by_frame: dict[int, list[str]] = {}
    for key in npz.files:
        m = re.match(r"f(\d+)_o(\d+)$", key)
        if m and int(m.group(2)) in structural_ids:
            by_frame.setdefault(int(m.group(1)), []).append(key)

    ratios = []
    n_seen = 0
    for op in sorted(omega_dir.glob("frame_*.npz")):
        m = re.match(r"frame_(\d+)\.npz$", op.name)
        if not m:
            continue
        fid = int(m.group(1))
        keys = by_frame.get(fid)
        if not keys:
            continue
        od = _load_depth(op)
        if od is None:
            continue
        dp = da3_dir / f"frame_{fid}.npz"
        dd = _load_depth(dp)
        if dd is None:
            continue
        cc = _load_depth(dp, key="conf")
        union = None
        for k in keys:
            mk = _erode_mask(npz[k], erosion_px)
            union = mk if union is None else (union | mk)
        if union is None or not union.any():
            continue
        n_seen += 1
        r = _ratio(dd, od, conf=cc, near_frac=near_frac,
                   conf_top_frac=conf_top_frac, restrict=union)
        if r is not None and np.isfinite(r) and r > 0:
            ratios.append(r)

    if len(ratios) < 3:
        return {"s_marker": s_marker, "s_refined": None, "delta_pct": None,
                "n_frames": len(ratios), "n_frames_with_masks": n_seen,
                "verdict": "insufficient structural pixels (need ≥3 usable frames)"}
    arr = np.asarray(ratios, np.float64)
    lo, hi = np.percentile(arr, [10, 90])
    s_ref = float(np.median(arr[(arr >= lo) & (arr <= hi)]))
    return {
        "s_marker": s_marker,
        "s_refined": s_ref,
        "delta_pct": (s_ref / s_marker - 1.0) * 100.0,
        "n_frames": int(len(ratios)),
        "n_frames_with_masks": int(n_seen),
        "spread": [float(arr.min()), float(arr.max())],
        "filters": {"structural_whitelist": sorted(whitelist),
                    "erosion_px": erosion_px, "near_frac": near_frac,
                    "conf_top_frac": conf_top_frac, "raw_depths": True},
        "applied": False,   # report-only: R.6 — a rescale is never a side effect
    }


def per_window_scale_priors(output_dir, window_map: dict[int, str],
                            config: dict | None = None) -> dict[str, float]:
    """R.6 — DA3 top-confidence scale S per WINDOW, as a PRIOR for the Sim(3)
    pose graph. Per frame: median(raw DA3 / raw omega) over near-band ∩
    conf-top pixels (same filters as scale_align); per window: median of its
    frames' ratios. The prior handed to the optimizer is RELATIVE to the
    applied global scale (S_w / s_marker): 1.0 means "this window is already
    at the global metric — do not rescale it"; a deviation tells the solver
    how much this window's metric disagrees with the global S.

    Returns {} when the session has no basis (no da3_run / omega_run / marker),
    so the caller falls back to the neutral 1.0 priors."""
    from reconstruction.scale_align import _ratio

    output_dir = Path(output_dir)
    va = ((config or {}).get("reconstruction", {}) or {}).get("vggtomega", {}) or {}
    near_frac = float(va.get("scale_near_frac", 0.25))
    conf_top_frac = float(va.get("scale_conf_top_frac", 0.10))
    da3_dir = output_dir / "da3_run" / "results_output"
    omega_dir = output_dir / "omega_run" / "results_output"
    marker = output_dir / ".metric_scale_applied"
    if not (da3_dir.is_dir() and omega_dir.is_dir() and marker.exists() and window_map):
        return {}
    try:
        s_marker = float(marker.read_text().strip().split("=")[-1])
    except Exception:
        return {}
    if not np.isfinite(s_marker) or s_marker <= 0:
        return {}

    by_window: dict[str, list[float]] = {}
    for op in sorted(omega_dir.glob("frame_*.npz")):
        m = re.match(r"frame_(\d+)\.npz$", op.name)
        if not m:
            continue
        fid = int(m.group(1))
        w = window_map.get(fid)
        if w is None:
            continue
        od = _load_depth(op)
        dd = _load_depth(da3_dir / f"frame_{fid}.npz")
        if od is None or dd is None:
            continue
        cc = _load_depth(da3_dir / f"frame_{fid}.npz", key="conf")
        r = _ratio(dd, od, conf=cc, near_frac=near_frac, conf_top_frac=conf_top_frac)
        if r is not None and np.isfinite(r) and r > 0:
            by_window.setdefault(w, []).append(r)

    priors: dict[str, float] = {}
    for w, rs in by_window.items():
        if len(rs) >= 2:
            priors[w] = float(np.median(rs) / s_marker)
    if priors:
        vals = np.array(list(priors.values()))
        logger.info(f"[R.6] per-window S priors (S_w/s_global) over {len(priors)} windows: "
                    f"median {np.median(vals):.4f}, spread {vals.min():.4f}–{vals.max():.4f}"
                    f" — {dict((k, round(v, 4)) for k, v in sorted(priors.items()))}")
    else:
        logger.info("[R.6] per-window S priors: no usable frames — neutral 1.0 priors")
    return priors


def apply_refined_scale(output_dir, store_path, recheck: dict,
                        config: dict | None = None) -> dict:
    """Apply the re-checked metric scale as a GLOBAL similarity δ = s_refined /
    s_marker, when the R.6 gates pass. Rescales everything the fusion and the
    store read — chunk PLYs, aligned chunk depth, camera poses (via
    reconstruction.scale_align.apply_scale), the .metric_scale_applied marker,
    and the instance store (points, OBBs) — so every consumer stays coherent.

    Gates (config phase_r.*): scale_recheck_apply (master switch),
    scale_recheck_min_frames, scale_recheck_min_delta_pct (don't churn files
    for noise), scale_recheck_max_delta_pct (a huge delta means something is
    wrong — refuse and report instead of 'correcting').
    Idempotent: after application the marker holds s_refined, so a re-run
    measures delta ≈ 0 and passes below min_delta."""
    cfg = (config or {}).get("phase_r", {}) if config else {}
    out = {"applied": False, "delta_pct": recheck.get("delta_pct")}
    if not cfg.get("scale_recheck_apply", False):
        out["reason"] = "scale_recheck_apply disabled"
        return out
    s_ref, s_marker = recheck.get("s_refined"), recheck.get("s_marker")
    n_frames = int(recheck.get("n_frames") or 0)
    min_frames = int(cfg.get("scale_recheck_min_frames", 10))
    min_delta = float(cfg.get("scale_recheck_min_delta_pct", 0.2))
    max_delta = float(cfg.get("scale_recheck_max_delta_pct", 10.0))
    if not s_ref or not s_marker:
        out["reason"] = "no refined scale"
        return out
    delta_pct = (s_ref / s_marker - 1.0) * 100.0
    if n_frames < min_frames:
        out["reason"] = f"only {n_frames} structural frames (< {min_frames})"
        return out
    if abs(delta_pct) < min_delta:
        out["reason"] = f"delta {delta_pct:+.2f}% below min {min_delta}% — nothing to fix"
        return out
    if abs(delta_pct) > max_delta:
        out["reason"] = (f"delta {delta_pct:+.2f}% exceeds max {max_delta}% — refusing "
                         f"(a jump this size means bad inputs, not a scale fix)")
        logger.warning(f"[R.6] refined-S NOT applied: {out['reason']}")
        return out

    delta = float(s_ref / s_marker)
    from reconstruction.scale_align import _scale_ply_xyz, apply_scale
    output_dir = Path(output_dir)
    logger.info(f"[R.6] applying refined S: {s_marker:.4f} → {s_ref:.4f} "
                f"(δ ×{delta:.5f}, {delta_pct:+.2f}%, {n_frames} structural frames) "
                f"— rescaling chunks + aligned depth + poses + marker + store")
    apply_scale(output_dir, delta)
    # Sessions whose chunks were already merged+deleted (delete_chunks_after_merge):
    # apply_scale found no chunk PLYs → scale the MERGED cloud in place instead,
    # and drop the artifacts derived from its old scale (octree, floor transform)
    # so the cloudcompy light-resume regenerates them coherently.
    if not list(output_dir.glob("chunk_*.ply")):
        for name in ("cleaned_cloud.ply", "cleaned_cloud_raw.ply", "corrected_cloud.ply"):
            ply = output_dir / name
            if ply.exists():
                if _scale_ply_xyz(ply, delta):
                    logger.info(f"[R.6] scaled merged {name} XYZ ×{delta:.5f}")
                else:
                    logger.warning(f"[R.6] could not scale {name} (unexpected layout)")
        import shutil as _sh
        for stale in ("potree", "floor_transform.npz"):
            p = output_dir / stale
            if p.exists():
                (_sh.rmtree(p, ignore_errors=True) if p.is_dir()
                 else p.unlink(missing_ok=True))
                logger.info(f"[R.6] dropped stale {stale} (rebuilt from the rescaled cloud)")
    (output_dir / ".metric_scale_applied").write_text(f"s={s_ref:.6f}\n")

    # keep the instance store coherent (points, OBBs, positions)
    n_pts, n_obbs = 0, 0
    try:
        from .instance_store import InstanceStore
        st = InstanceStore(store_path)
        for inst in st.list_instances():
            iid = inst["instance_id"]
            pts = st.get_points(iid)
            if pts is not None:
                st.set_points(iid, pts * delta, frame_ids=st.get_point_frames(iid))
                n_pts += 1
            for w in st.list_obb_windows(iid):
                obb = st.get_obb(iid, w)
                if obb is None:
                    continue
                T, aabb, pos = obb
                T = np.asarray(T, float).copy()
                T[:3, 3] *= delta
                st.set_obb(iid, T, np.asarray(aabb, float) * delta,
                           np.asarray(pos, float) * delta, window_id=w)
                n_obbs += 1
        st.close()
    except Exception as e:  # store rescale is best-effort; geometry is already coherent
        out["store_error"] = str(e)
        logger.warning(f"[R.6] store rescale failed (geometry already rescaled): {e}")
    out.update({"applied": True, "delta": delta, "s_from": s_marker, "s_to": s_ref,
                "store_instances_rescaled": n_pts, "store_obbs_rescaled": n_obbs})
    logger.info(f"[R.6] refined S APPLIED: marker now s={s_ref:.4f}; "
                f"store rescaled ({n_pts} point sets, {n_obbs} OBBs)")
    return out
