"""§10 metrics harness — evidence, never a verdict. One report per epoch
(``output/quality/report_epoch_<N>.json``), the latest as
``quality/report.json`` with the comparison against the previous epoch.

Items (numbering of claude_stac.txt §10): 1 loop residual (m) per edge
before/after the graph; 2 seam residual (held-out surface pairs, median and
p95); 3 closure error of the walk (distance between the two observations of
a revisited instance); 4 automatic duplicates (target 0); 5 scale (s, DA3
agreement, corrections, scale_break); 6 witnesses (% per status, mean
mv_votes, total conflicts); 7 multi-view depth disagreement before/after;
8 known dimensions (p50/p95 error) when the session registers any
(``output/known_dimensions.json``: [{label, dimension, value_m}]);
9 loop coverage and authority saturation per stage. 10–13 (known answer,
envelope, adversarial, determinism) live in their own modules and land in
the acta.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

QUALITY_DIR = "quality"
REPORT_JSON = "report.json"


def _median(vals) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def _p95(vals) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.percentile(vals, 95)) if vals else None


def seam_residuals(session, gcfg) -> dict:
    """Held-out surface pairs of the cloud (the same judge the keyframe
    graph uses): median and p95 of the NN distance between the points of
    keyframes f and f+d."""
    from reconstruction.loops.kf_graph import _per_kf_points_fn, _holdout_pairs
    per_kf = _per_kf_points_fn(session)
    pairs = _holdout_pairs(session, per_kf, gcfg.holdout_offsets, gcfg.holdout_stride,
                           gcfg.holdout_samples, gcfg.holdout_max_nn_m)
    meds = [float(np.median(np.linalg.norm(p - q, axis=1))) for _f, _g, p, q in pairs]
    return {"median_m": _median(meds), "p95_m": _p95(meds), "n_pairs": len(pairs)}


def closure_errors(session, loops: List[dict]) -> dict:
    """Per revisited place: the offset between what the two visits wrote
    there (median NN distance of the co-visible points, measured by the
    revisit detection on the CURRENT geometry) — the closure the walk left."""
    vals, per = [], []
    for m in loops:
        d = m.get("offset_before_m")
        if d is None:
            continue
        vals.append(float(d))
        per.append({"i": m.get("i"), "j": m.get("j"), "earlier_kfs": m.get("earlier_kfs"),
                    "later_kfs": m.get("later_kfs"), "closure_m": float(d), "accepted": m.get("accepted")})
    return {"median_m": _median(vals), "max_m": (max(vals) if vals else None), "per_loop": per}


def known_dimension_errors(session, instances: List[dict], up: np.ndarray, dims_pct,
                           output_dir) -> Optional[dict]:
    """p50/p95 of |measured − registered| over output/known_dimensions.json
    (None when the session registers no dimension)."""
    p = Path(output_dir) / "known_dimensions.json"
    if not p.exists():
        return None
    from reconstruction.loops.structural import _instance_dimension
    regs = json.loads(p.read_text())
    regs = regs.get("dimensions", regs) if isinstance(regs, dict) else regs
    errs, rows = [], []
    for reg in regs:
        label = str(reg["label"]).lower()
        for inst in instances:
            if str(inst.get("label", "")).lower() != label:
                continue
            gi = np.asarray(inst.get("globalIndices") or [], np.int64)
            gi = gi[(gi >= 0) & (gi < session.n_points)]
            if len(gi) < 50:
                continue
            meas = _instance_dimension(session.xyz[gi], str(reg["dimension"]), up, dims_pct[0], dims_pct[1])
            if meas is None:
                continue
            e = abs(float(meas) - float(reg["value_m"]))
            errs.append(e)
            rows.append({"label": label, "instance_id": int(inst.get("instance_id", inst.get("id"))),
                         "dimension": reg["dimension"], "registered_m": float(reg["value_m"]),
                         "measured_m": float(meas), "error_m": e})
    return {"p50_m": _median(errs), "p95_m": _p95(errs), "n": len(errs), "rows": rows}


def compute_metrics(session, fields: Optional[Dict[str, np.ndarray]], loops: List[dict],
                    stages: dict, cfg, output_dir, instances: Optional[List[dict]] = None,
                    duplicates_sep_m: Optional[float] = None, frames: Optional[dict] = None) -> dict:
    """The §10 items for one state of the session (in memory or on disk).
    ``stages``: {"scale": ..., "poses": ..., "depth": ...} stage reports of
    the iteration that produced this state (empty for the initial state —
    then the multi-view depth disagreement is measured here from
    ``frames``, so every state carries the same items)."""
    from reconstruction.witness.run import summarize
    output_dir = Path(output_dir)
    gcfg = cfg.graph
    poses_rep = stages.get("poses") or {}
    depth_rep = stages.get("depth") or {}
    scale_rep = stages.get("scale") or {}
    if not depth_rep and frames:
        from reconstruction.witness.depth_tracks import pairwise_rows
        nums = sorted(int(k) for k in frames)
        rows = pairwise_rows(frames, nums, cfg.witness.depth.pair_offsets, cfg.witness.depth.pair_samples,
                             clip_sigma=cfg.witness.depth.pair_scatter_clip_sigma,
                             edge_tol_rel=cfg.witness.tracks.depth_edge_tol_rel)
        depth_rep = {"pair_rel_median": (float(np.median([r[4] for r in rows])) if rows else None),
                     "n_pair_rows": len(rows), "applied": False}
    seams = seam_residuals(session, gcfg)
    closure = closure_errors(session, loops)
    sep = float(duplicates_sep_m if duplicates_sep_m is not None else cfg.loops.duplicate_min_sep_m)
    n_dup = sum(1 for r in closure["per_loop"] if r["closure_m"] > sep)
    loop_gate = (poses_rep.get("gates") or {}).get("loop_gain") or {}
    # the STATE's loop residual: what the accepted region closures still
    # demand at this geometry (median |t| over edges) — comparable between
    # states; the pose graph's summed before/after is the solver's diagnostic
    demands = [float(m["t_norm_m"]) for m in loops if m.get("accepted") and m.get("t_norm_m") is not None]
    loop_res = {"median_demand_m": _median(demands), "p95_demand_m": _p95(demands),
                "pose_graph_sum_before_m": loop_gate.get("loop_residual_before_m"),
                "pose_graph_sum_after_m": loop_gate.get("loop_residual_after_m"),
                "per_edge_rms_m": [float(m["icp_rms_m"]) for m in loops if m.get("icp_rms_m") is not None],
                "n_edges": len(loops), "n_accepted": len(demands),
                "n_active": poses_rep.get("n_loop_edges_active")}
    v = depth_rep.get("verdict") or {}
    applied = bool(depth_rep.get("applied", False))
    # the STATE's disagreement: after the correction when it was applied,
    # the current one otherwise (a would-be "after" of an identity stage is
    # not a measurement of this state)
    depth = {"pair_rel_median_before": depth_rep.get("pair_rel_median"),
             "holdout_rel_before": v.get("med_before"),
             "holdout_rel_after": (v.get("med_after") if applied else v.get("med_before")),
             "applied": applied, "n_track_obs": depth_rep.get("n_track_obs"),
             "n_contour_obs": depth_rep.get("n_contour_obs")}
    scale = {"r_per_chunk": scale_rep.get("r"), "max_abs_log_r": scale_rep.get("max_abs_log_r"),
             "applied": bool(scale_rep.get("applied", False)),
             "n_loop_rows": len(scale_rep.get("loop_rows") or []),
             "n_absolute_rows": len(scale_rep.get("absolute_rows") or [])}
    sd = output_dir / "scale_diagnostics.json"
    if sd.exists():
        diag = json.loads(sd.read_text())
        scale["s_applied"] = diag.get("s_applied")
        scale["scale_confidence"] = diag.get("scale_confidence")
    sg = output_dir / "maplong_run" / "scale_graph.json"
    if sg.exists():
        g = json.loads(sg.read_text())
        scale["scale_break"] = g.get("scale_break")
    witnesses = summarize(fields) if fields is not None else None
    downs = session.poses[:, :3, 1]
    g_down = downs.mean(0); g_down = g_down / (np.linalg.norm(g_down) + 1e-12)
    known = known_dimension_errors(session, instances or [], -g_down,
                                   (cfg.loops.spatial.dims_pct_lo, cfg.loops.spatial.dims_pct_hi),
                                   output_dir)
    coverage = poses_rep.get("loop_coverage")
    authority = {"pose_graph": (poses_rep.get("gates") or {}).get("authority")}
    ap = output_dir / "maplong_run" / "authority.json"
    if ap.exists():
        authority["reconstruction"] = json.loads(ap.read_text())
    depth_frac = depth["holdout_rel_after"] if depth["holdout_rel_after"] is not None else \
        depth["pair_rel_median_before"]
    w = cfg.certify.objective
    parts = {"loop_residual_m": loop_res["median_demand_m"],
             "seam_residual_m": seams["median_m"], "closure_m": closure["median_m"],
             "depth_disagreement_frac": depth_frac, "duplicates": float(n_dup)}
    weights = {"loop_residual_m": w.loop_residual_m, "seam_residual_m": w.seam_residual_m,
               "closure_m": w.closure_m, "depth_disagreement_frac": w.depth_disagreement_frac,
               "duplicates": w.duplicates}
    objective = float(sum(weights[k] * float(v_) for k, v_ in parts.items() if v_ is not None))
    return {"loop_residual": loop_res, "seam_residual": seams, "closure": closure,
            "duplicates": {"n": n_dup, "target": 0, "separation_m": sep},
            "scale": scale, "witnesses": witnesses, "depth_disagreement": depth,
            "known_dimensions": known, "loop_coverage": coverage, "authority": authority,
            "objective": objective, "objective_parts": parts, "objective_weights": weights,
            "provenance": "tool_measured"}


def compare(current: dict, previous: Optional[dict]) -> dict:
    """Deltas of the scalar items (current − previous)."""
    if previous is None:
        return {}
    out = {}
    pairs = {"seam_residual_m": ("seam_residual", "median_m"),
             "seam_p95_m": ("seam_residual", "p95_m"),
             "closure_m": ("closure", "median_m"),
             "duplicates": ("duplicates", "n"),
             "objective": ("objective", None)}
    for k, (a, b) in pairs.items():
        cv = current.get(a); pv = previous.get(a)
        if b is not None:
            cv = (cv or {}).get(b); pv = (pv or {}).get(b)
        if cv is not None and pv is not None:
            out[k] = float(cv) - float(pv)
    cw, pw = current.get("witnesses"), previous.get("witnesses")
    if cw and pw:
        out["verified_fraction"] = float(cw["status_fraction"]["verified"]) - float(pw["status_fraction"]["verified"])
    return out


def write_epoch_report(output_dir, epoch: int, metrics: dict, previous: Optional[dict] = None,
                       extra: Optional[dict] = None) -> Path:
    qdir = Path(output_dir) / QUALITY_DIR
    qdir.mkdir(parents=True, exist_ok=True)
    rep = {"version": 1, "epoch": int(epoch), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "metrics": metrics, "comparison_vs_previous": compare(metrics, previous),
           "previous_epoch_objective": (previous or {}).get("objective"), **(extra or {})}
    p = qdir / f"report_epoch_{int(epoch)}.json"
    p.write_text(json.dumps(rep, indent=1, default=float))
    (qdir / REPORT_JSON).write_text(json.dumps(rep, indent=1, default=float))
    return p
