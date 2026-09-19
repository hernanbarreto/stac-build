"""Orchestration of a correction run — the flow of prompt §5.3:

mark → evidence → observability → diagnose → solve → gates → distribute →
apply(tx) → invalidate/regenerate → report → the epoch is selectable.

Every stage's structured output lands in the report, ALSO when the run is
rejected: the user always receives the numbers and the exact reason. A gate
failure returns a ``rejected`` report (and a ledger record) without touching
one byte of the session; only a fully-gated solution reaches the
transactional apply.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from correction import diagnose, distribute, floor as floor_mod, gates, \
    ledger, observability as obs_mod, solve
from correction.apply import (assert_no_interrupted_swap, available_epochs,
                              stage_transaction, swap_transaction)
from correction.config import CorrectionConfig, load_correction_config
from correction.epoch import current_epoch, epoch_path
from correction.evidence import extract_evidence
from correction.invalidate import update_instance_store
from correction.report import build_report, save_report
from correction.session import CorrectionSession, load_session


def _noop_progress(pct: float, msg: str) -> None:
    pass


def _check_ready(output_dir: Path) -> None:
    """A half-finished swap is the only thing that blocks a new correction.

    It used to refuse while a previous epoch was "pending approval". There is
    no approval any more (USER 2026-09-16: *"todas viven, solo se seleccionan y
    la que se selecciona se muestra"*): every epoch stays on disk and a new
    correction simply runs on top of whichever one is being shown.
    """
    assert_no_interrupted_swap(output_dir)


def _group_displaced(evidence) -> List[dict]:
    """Displaced copies grouped by overlapping keyframe spans: one group =
    one revisit = one solved unit (several marked objects share it)."""
    disp = evidence.displaced
    spans = [(cp.kfs[0], cp.kfs[-1]) for cp in disp]
    groups: List[set] = []
    for i, (a, b) in enumerate(spans):
        placed = None
        for g in groups:
            if any(not (b < spans[j][0] or a > spans[j][1]) for j in g):
                g.add(i)
                placed = g
                break
        if placed is None:
            groups.append({i})
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                si = {k for m in groups[i]
                      for k in range(spans[m][0], spans[m][1] + 1)}
                sj = {k for m in groups[j]
                      for k in range(spans[m][0], spans[m][1] + 1)}
                if si & sj:
                    groups[i] |= groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for g in sorted(groups, key=lambda g: min(spans[m][0] for m in g)):
        members = [disp[m] for m in sorted(g)]
        kfs = sorted({k for cp in members for k in cp.kfs})
        out.append({"members": members, "kfs": kfs,
                    "kf_span": [kfs[0], kfs[-1]]})
    return out


def run_objects(output_dir, instance_ids: List[int], operator: str,
                override_scale_check: bool = False,
                log: Callable = print,
                progress: Callable = _noop_progress,
                cfg: Optional[CorrectionConfig] = None) -> dict:
    """The object-marked correction. Returns the full report
    (status: applied | rejected)."""
    t0 = time.time()
    output_dir = Path(output_dir)
    if cfg is None:
        cfg = load_correction_config()
    _check_ready(output_dir)
    correction_id = ledger.new_correction_id()
    epoch_from = current_epoch(output_dir)
    rng = np.random.default_rng(cfg.solve.seed)

    def _p(pct, msg):
        log(msg)
        progress(pct, msg)

    _p(2, "loading cloud + provenance...")
    session = load_session(output_dir)

    def _reject(reason: str, *, gates_list=None, visits=None,
                observability=None, diagnosis=None, solutions=None,
                distribution=None, suggestion=None) -> dict:
        report = build_report(
            correction_id=correction_id, kind="objects", operator=operator,
            status="rejected", instance_ids=instance_ids, visits=visits,
            observability=observability, diagnosis=diagnosis,
            solutions=solutions, distribution=distribution,
            gates=gates_list or [], overrides=None, epoch_from=epoch_from,
            epoch_to=None, rejection_reason=reason, suggestion=suggestion,
            elapsed_s=time.time() - t0)
        path = save_report(output_dir, report)
        ledger.record_run(
            output_dir, correction_id=correction_id, epoch_from=epoch_from,
            epoch_to=epoch_from, kind="objects", operator=operator,
            instance_ids=instance_ids, visits=visits or [],
            observability=observability or [], diagnosis=diagnosis or [],
            anchors=[], gates=gates_list or [], overrides=None,
            report_path=str(path.relative_to(output_dir)),
            verdict="rejected")
        _p(100, f"❌ correction REJECTED: {reason}")
        return report

    # integrity first — an unresolvable point would silently stay behind
    g_int = gates.gate_integrity(session)
    if not g_int["passed"]:
        return _reject(g_int["detail"], gates_list=[g_int])

    _p(8, f"extracting copies of {len(instance_ids)} marked segment(s)...")
    ev = extract_evidence(session, instance_ids, cfg, log=log)
    target = session.xyz[ev.target_idx]
    tree = cKDTree(target)

    groups = _group_displaced(ev)
    _p(20, f"solving {len(groups)} displaced visit group(s)...")

    solutions: List[dict] = []
    deferred: List[dict] = []
    obs_reports: List[dict] = []
    diag_reports: List[dict] = []
    visit_summaries = [cp.summary() for cp in ev.copies]
    gate_results: List[dict] = [g_int]
    advisory = cfg.gates.mode == "advisory"

    # ── pass 1: evidence, observability, diagnosis (k) per group ─────────
    prepared: List[dict] = []
    skipped_objects: List[str] = []
    for gi, grp in enumerate(groups):
        members = grp["members"]
        label = f"visit kf {grp['kf_span'][0]}..{grp['kf_span'][1]}"
        shapes = []
        centroids: Dict[int, np.ndarray] = {}
        seg_parts = []
        # an object with no reference copy carries no closure: it must not
        # enter the evidence (its points would pull the ICP toward nothing)
        no_ref = sorted({cp.label for cp in members if cp.iid not in ev.ref})
        if no_ref:
            skipped_objects.append(
                f"{label}: {no_ref} seen only once (no reference copy) — "
                f"not a duplicate, excluded from the evidence")
            log(f"  {skipped_objects[-1]}")
        members = [cp for cp in members if cp.iid in ev.ref]
        if not members:
            continue
        for cp in members:
            if len(cp.seg_idx) >= cfg.evidence.min_object_points_fingerprint:
                centroids[cp.iid] = np.median(session.xyz[cp.seg_idx], axis=0)
                shapes.append(obs_mod.classify_object(
                    session.xyz[cp.seg_idx], cp.iid, cp.label, cfg))
            if len(cp.seg_idx):
                seg_parts.append(cp.seg_idx)
        src_idx = (np.unique(np.concatenate(seg_parts)) if seg_parts
                   else np.empty(0, dtype=np.int64))
        if len(src_idx) < cfg.evidence.min_object_points_solve:
            deferred.append({"group": gi, "kf_span": grp["kf_span"],
                             "n_points": int(len(src_idx))})
            log(f"  {label}: only {len(src_idx)} curated pts — will inherit "
                f"the nearest solved visit's transform")
            continue
        if not shapes:
            # enough points to solve but no single object passes the
            # fingerprint minimum: classify the pooled evidence
            shapes = [obs_mod.classify_object(
                session.xyz[src_idx], members[0].iid,
                members[0].label + " (pooled)", cfg)]
            centroids[members[0].iid] = np.median(session.xyz[src_idx],
                                                  axis=0)
        bounded = {}
        for cp in members:
            if cp.iid in ev.ref and len(cp.seg_idx) \
                    and len(ev.ref[cp.iid].seg_idx):
                bounded[cp.iid] = obs_mod.bounded_copies(
                    session.xyz[ev.ref[cp.iid].seg_idx],
                    session.xyz[cp.seg_idx], cfg)
        visit_obs = obs_mod.analyze_visit(shapes, centroids, cfg, bounded)
        obs_reports.append({"group": gi, "kf_span": grp["kf_span"],
                            **visit_obs.summary()})
        if not visit_obs.ok:
            return _reject(
                f"{label}: the evidence observes no degree of freedom",
                gates_list=gate_results, visits=visit_summaries,
                observability=obs_reports, suggestion=visit_obs.suggestion)
        k, ratio, n_pairs = diagnose.fingerprint_k(
            centroids, ev.ref_fingerprint, cfg)
        diag = diagnose.diagnose_visit(k, ratio, n_pairs,
                                       visit_obs.depth_allowed, cfg)
        diag_reports.append({"group": gi, "kf_span": grp["kf_span"], **diag})
        log(f"  {label}: {diag['diagnosis']}"
            + (f" (k={diag['k']})" if diag["depth_needed"] else ""))
        prepared.append({"gi": gi, "grp": grp, "label": label,
                         "members": members, "src_idx": src_idx,
                         "visit_obs": visit_obs, "diag": diag})

    # ── depth k FIRST: a compressed visit also lifts its own floor, so the
    #    floor constraint must be measured on the k-expanded geometry ─────
    k_pre = np.ones(session.n_kf)
    for pr in prepared:
        if pr["diag"]["depth_needed"]:
            a_, b_ = pr["grp"]["kf_span"]
            k_pre[int(a_):int(b_) + 1] = pr["diag"]["k"]
    xyz_k = session.xyz
    if (k_pre != 1.0).any():
        xyz_k = session.xyz.copy()
        for kf in np.where(k_pre != 1.0)[0]:
            sel = np.where(session.ks == kf)[0]
            if len(sel):
                cam = session.cam_center[session.frames[int(kf)]]
                xyz_k[sel] = cam + (xyz_k[sel] - cam) * k_pre[kf]
    import dataclasses as _dc
    session_k = _dc.replace(session, xyz=xyz_k)

    floor_report: dict = {"used": False,
                          "why": "USER 2026-09-09: the per-keyframe low band "
                                 "is not a validated floor curve — the "
                                 "closure follows the drift-rate model only"}

    # ── pass 2: solve each group on the k-expanded evidence ──────────────
    _p(30, "solving the object evidence (trimmed ICP per visit)...")
    for pr in prepared:
        gi, grp, label = pr["gi"], pr["grp"], pr["label"]
        members, src_idx = pr["members"], pr["src_idx"]
        visit_obs, diag = pr["visit_obs"], pr["diag"]
        S = xyz_k[src_idx].copy()
        # ICP init from the EXPANDED evidence: depth expansion can move a
        # copy by metres (compression pulls objects toward far revisit
        # cameras) — an offset computed on the raw centroids lands the ICP
        # in a local minimum where a wall slides along its own plane.
        exp_centroids: Dict[int, np.ndarray] = {}
        for cp in members:
            m = np.isin(src_idx, cp.seg_idx)
            if m.any():
                exp_centroids[cp.iid] = np.median(S[m], axis=0)
        offs = [ev.ref[cp.iid].centroid - exp_centroids[cp.iid]
                for cp in members
                if cp.iid in ev.ref and cp.iid in exp_centroids]
        if offs:
            init_t = np.mean(offs, axis=0)
        else:
            big = max((cp for cp in members if cp.iid in ev.ref),
                      key=lambda cp: len(cp.idx), default=None)
            n_pl, c_pl = solve.fit_plane(
                session.xyz[ev.ref[big.iid].seg_idx], rng, cfg)
            off = float(np.median((S - c_pl) @ n_pl))
            init_t = -off * n_pl
        sub = S[rng.choice(len(S), min(cfg.solve.icp_sample, len(S)),
                           replace=False)] + init_t
        R, t, rms = solve.trimmed_icp(
            sub, tree, target, cfg,
            rotation=(visit_obs.projection.get("mode") == "full"))
        t_full = R @ init_t + t
        R, t_full = solve.project_solution(R, t_full, visit_obs.projection)

        # residuals after PROJECTION (what will actually be applied)
        res_before = solve.eval_residual(session.xyz[src_idx], tree, rng, cfg)
        corr = S @ R.T + t_full
        res_after = solve.eval_residual(corr, tree, rng, cfg)
        centroid_offsets = {}
        proj = visit_obs.projection
        for cp in members:
            if cp.iid in ev.ref and cp.iid in exp_centroids:
                c_corr = R @ exp_centroids[cp.iid] + t_full
                d = c_corr - ev.ref[cp.iid].centroid
                # measure only along the OBSERVED directions: a plane cannot
                # be corrected (nor judged) inside its own plane
                if proj.get("mode") == "normal":
                    n = np.asarray(proj["normal"])
                    d = float(d @ (n / np.linalg.norm(n)))
                    centroid_offsets[cp.label] = abs(d)
                elif proj.get("mode") == "perp_axis":
                    a = np.asarray(proj["axis"])
                    a = a / np.linalg.norm(a)
                    centroid_offsets[cp.label] = float(np.linalg.norm(
                        d - (d @ a) * a))
                else:
                    centroid_offsets[cp.label] = float(np.linalg.norm(d))
        g_col = gates.gate_object_collapse(res_before, res_after, cfg,
                                           centroid_offsets)
        g_col["group"] = gi
        gate_results.append(g_col)
        log(f"  {label}: rot {solve.rot_deg(R):.2f}° |t| "
            f"{np.linalg.norm(t_full):.3f} m, rms {rms*100:.1f} cm, "
            f"residual {res_before*100:.1f} → {res_after*100:.1f} cm")
        if not g_col["passed"] and not advisory:
            return _reject(
                f"{label}: the copies did NOT collapse — {g_col['detail']}",
                gates_list=gate_results, visits=visit_summaries,
                observability=obs_reports, diagnosis=diag_reports)
        solutions.append({
            "group": gi, "anchor_kf": int(np.median(session.ks[src_idx])),
            "kf_span": grp["kf_span"], "R": R, "t": t_full,
            "k": diag["k"], "rot_deg": round(solve.rot_deg(R), 3),
            "t_m": round(float(np.linalg.norm(t_full)), 4),
            "icp_rms_cm": round(rms * 100, 2),
            "residual_cm": {"before": round(res_before * 100, 1),
                            "after": round(res_after * 100, 1)},
            "dof": visit_obs.dof, "unrestrained": visit_obs.unrestrained})

    if not solutions:
        return _reject("no marked object is seen TWICE (no duplicate to "
                       "close): " + "; ".join(skipped_objects),
                       gates_list=gate_results,
                       visits=visit_summaries, observability=obs_reports,
                       diagnosis=diag_reports,
                       suggestion="mark objects with more curated points in "
                                  "the displaced visit")
    # deferred groups inherit the nearest solved group's transform (nothing
    # inside a marked bbox is ever skipped)
    for d in deferred:
        nearest = min(solutions,
                      key=lambda s: abs(s["kf_span"][0] - d["kf_span"][0]))
        solutions.append({**{k: nearest[k] for k in
                             ("R", "t", "k", "rot_deg", "t_m")},
                          "group": d["group"], "anchor_kf": d["kf_span"][0],
                          "kf_span": d["kf_span"],
                          "inherited_from_group": nearest["group"],
                          "n_own_points": d["n_points"]})
        log(f"  visit kf {d['kf_span']}: INHERITS group "
            f"{nearest['group']}'s transform ({d['n_points']} own pts)")

    # distribute ----------------------------------------------------------
    _p(40, "distributing the correction over keyframes (slerp+lerp)...")
    # reference keyframe = middle of the reference visit; closures measured
    # against it; the drift-rate line through the walk start distributes
    # each closure is attributed to the chainage where it was MEASURED: the
    # median keyframe of the evidence points on each side (a copy spans
    # several keyframes and the drift grows across them)
    ref_pts = np.concatenate([rr.seg_idx for rr in ev.ref.values()
                              if len(rr.seg_idx)])
    ref_kf_mid = int(np.median(session.ks[ref_pts]))
    d_kf = distribute.chainage(session.poses)
    R_o, t_o, k_kf, dist_report = distribute.distribute(
        session.n_kf, d_kf, ref_kf_mid, solutions)
    log(f"  drift model: {dist_report['drift_rate_mm_per_m']} mm/m over a "
        f"{dist_report['walk_m']} m walk; reference copy moves "
        f"{dist_report['reference_correction_m']} m; knots "
        f"{dist_report['knots']}")
    R_kf, t_kf = R_o, t_o
    dist_report["floor_constraint"] = floor_report
    # claude_stac.txt §4.4 ("manual = misma arista"): every solved visit pair
    # is also a loop candidate (source `manual`) for the exact-bridge machinery
    # — the reference visit's middle keyframe and the displaced visit's anchor.
    from reconstruction.loops.instance_loops import add_manual_candidate
    manual_candidates = []
    for sol in solutions:
        add_manual_candidate(output_dir, int(sol["anchor_kf"]), ref_kf_mid,
                             [int(i) for i in instance_ids], log=log)
        manual_candidates.append([int(max(sol["anchor_kf"], ref_kf_mid)),
                                  int(min(sol["anchor_kf"], ref_kf_mid))])
    dist_report["manual_loop_candidates"] = manual_candidates

    # global gates --------------------------------------------------------
    g_plaus = gates.gate_plausibility(solutions, cfg)
    g_cont = gates.gate_continuity(dist_report, cfg)
    gate_results += [g_plaus, g_cont]
    affected_kfs = list(range(ev.ref_kf_end + 1, session.n_kf))
    _p(45, "scene exam: floor + unmarked witness instances...")
    g_scene = gates.gate_scene_exam(
        session, [int(i) for i in instance_ids], ev.ref_kf_end,
        affected_kfs, R_kf, t_kf, k_kf, cfg, rng)
    gate_results.append(g_scene)
    k_by_frame = {session.frames[i]: float(k_kf[i])
                  for i in range(session.n_kf) if k_kf[i] != 1.0}
    g_scale = diagnose.scale_check(output_dir, k_by_frame, cfg)
    overrides = None
    if not g_scale["passed"] and override_scale_check:
        overrides = {"scale_check": {"overridden_by": operator,
                                     "at": time.strftime(
                                         "%Y-%m-%d %H:%M:%S"),
                                     "numbers": g_scale}}
        g_scale = {**g_scale, "passed": True,
                   "detail": g_scale["detail"] + " — OVERRIDDEN by the "
                                                 "operator (recorded in the "
                                                 "ledger)"}
    gate_results.append(g_scale)

    failed = [g for g in gate_results if not g["passed"]]
    warnings: List[str] = list(skipped_objects)
    if failed:
        names = [g["name"] for g in failed]
        if not advisory:
            return _reject(
                f"gate(s) failed: {names} — {failed[0]['detail']}",
                gates_list=gate_results, visits=visit_summaries,
                observability=obs_reports, diagnosis=diag_reports,
                solutions=[{k: v for k, v in s.items()
                            if k not in ("R", "t")} for s in solutions],
                distribution=dist_report,
                suggestion=(g_scene.get("floor", {}).get("detail")
                            if "scene_exam" in names else None))
        # USER 2026-09-09: "no debes rechazar correcciones por umbrales
        # arbitrarios ... siempre debe aplicarse" — gates are ADVISORY: the
        # numbers go to the report as warnings, the correction is applied
        # and the epoch selector is where the user compares them.
        for g in failed:
            g["advisory"] = True
            warnings.append(f"{g['name']}: {g['detail']}")
        log(f"  ⚠ advisory gate(s) failed (applied anyway, USER 2026-09-09): "
            f"{names}")

    # apply (transaction) -------------------------------------------------
    _p(50, "all gates passed — staging the transaction...")
    scale_diag_new = diagnose.regenerate_scale_diagnostics(
        output_dir, k_by_frame, epoch_from + 1, correction_id)
    tx_info = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=correction_id,
        scale_diag_new=scale_diag_new, log=log, progress=progress)
    _p(90, "atomic swap...")
    swap_transaction(output_dir, tx_info, log=log)

    _p(93, "updating the instance store in place...")
    store_summary = update_instance_store(
        output_dir, R_kf, t_kf, k_kf, session.frames, log=log)

    solutions_clean = [{k: v for k, v in s.items() if k not in ("R", "t")}
                       for s in solutions]
    report = build_report(
        correction_id=correction_id, kind="objects", operator=operator,
        status="applied", instance_ids=instance_ids, visits=visit_summaries,
        observability=obs_reports, diagnosis=diag_reports,
        solutions=solutions_clean, distribution=dist_report,
        gates=gate_results, overrides=overrides, epoch_from=epoch_from,
        epoch_to=tx_info["epoch_to"],
        extra={"points_moved": tx_info["points_moved"],
               "pose_copies_skipped": tx_info["pose_copies_skipped"],
               "instance_store": store_summary,
               "floor_constraint": floor_report,
               "warnings": warnings},
        elapsed_s=time.time() - t0)
    path = save_report(output_dir, report)
    ledger.record_run(
        output_dir, correction_id=correction_id, epoch_from=epoch_from,
        epoch_to=tx_info["epoch_to"], kind="objects", operator=operator,
        instance_ids=instance_ids, visits=visit_summaries,
        observability=obs_reports, diagnosis=diag_reports,
        anchors=[{"kf": s["anchor_kf"], "rot_deg": s["rot_deg"],
                  "t_m": s["t_m"], "k": s.get("k", 1.0)}
                 for s in solutions_clean],
        gates=gate_results, overrides=overrides,
        report_path=str(path.relative_to(output_dir)))
    _p(100, f"✅ correction applied (epoch {tx_info['epoch_to']}) — select "
            f"any epoch to compare")
    return report


def run_floor(output_dir, model: Optional[str], keyframes: Optional[List[int]],
              operator: str, log: Callable = print,
              progress: Callable = _noop_progress,
              cfg: Optional[CorrectionConfig] = None) -> dict:
    """Floor alignment (kind=floor): same gates, same transactional apply,
    same ledger."""
    t0 = time.time()
    output_dir = Path(output_dir)
    if cfg is None:
        cfg = load_correction_config()
    _check_ready(output_dir)
    correction_id = ledger.new_correction_id()
    epoch_from = current_epoch(output_dir)
    model = model or cfg.floor.model_default
    rng = np.random.default_rng(cfg.solve.seed)

    def _p(pct, msg):
        log(msg)
        progress(pct, msg)

    _p(2, "loading cloud + provenance...")
    session = load_session(output_dir)

    def _reject(reason, gates_list, sol=None):
        report = build_report(
            correction_id=correction_id, kind="floor", operator=operator,
            status="rejected", instance_ids=None, visits=None,
            observability=None, diagnosis=None,
            solutions=([{k: v for k, v in sol.items()
                         if k not in ("R_kf", "t_kf", "k_kf", "floor_npz")}]
                       if sol else None),
            distribution=None, gates=gates_list, overrides=None,
            epoch_from=epoch_from, epoch_to=None, rejection_reason=reason,
            elapsed_s=time.time() - t0)
        path = save_report(output_dir, report)
        ledger.record_run(
            output_dir, correction_id=correction_id, epoch_from=epoch_from,
            epoch_to=epoch_from, kind="floor", operator=operator,
            instance_ids=[], visits=[], observability=[],
            diagnosis=[{"model": model}], anchors=[], gates=gates_list,
            overrides=None,
            report_path=str(path.relative_to(output_dir)),
            verdict="rejected")
        _p(100, f"❌ floor alignment REJECTED: {reason}")
        return report

    g_int = gates.gate_integrity(session)
    if not g_int["passed"]:
        return _reject(g_int["detail"], [g_int])

    _p(15, f"solving floor alignment (model: {model})...")
    sol = floor_mod.solve_floor(session, cfg, model, keyframes, rng, log=log)
    R_kf, t_kf, k_kf = sol["R_kf"], sol["t_kf"], sol["k_kf"]

    anchors_as_solutions = [
        {"R": R_kf[a["kf"]], "t": t_kf[a["kf"]]} for a in sol["anchors"]]
    g_plaus = gates.gate_plausibility(anchors_as_solutions, cfg)
    steps_t = np.linalg.norm(np.diff(t_kf, axis=0), axis=1)
    steps_r = [solve.rot_deg(R_kf[i + 1] @ R_kf[i].T)
               for i in range(session.n_kf - 1)]
    dist_report = {
        "identity_until_kf": -1,
        "anchors": sol["anchors"],
        "keyframes_warped": session.n_kf,
        "max_step_between_keyframes_mm":
            round(float(steps_t.max()) * 1000, 2) if len(steps_t) else 0.0,
        "max_step_between_keyframes_deg":
            round(float(max(steps_r)), 4) if steps_r else 0.0,
        "depth_keyframes": 0,
    }
    g_cont = gates.gate_continuity(dist_report, cfg)
    worst_mm = sol["exam"]["worst_residual_mm"]
    g_exam = {"name": "floor_model_exam",
              "passed": worst_mm <= cfg.gates.heldout_floor_abs_m * 1000,
              "worst_residual_mm": worst_mm,
              "detail": f"worst anchor-floor residual vs the {model} model: "
                        f"{worst_mm:.1f} mm (limit "
                        f"{cfg.gates.heldout_floor_abs_m*1000:.0f} mm)"}
    gate_results = [g_int, g_plaus, g_cont, g_exam]
    failed = [g for g in gate_results if not g["passed"]]
    warnings: List[str] = []
    if failed:
        if cfg.gates.mode != "advisory":
            return _reject(f"gate(s) failed: {[g['name'] for g in failed]} — "
                           f"{failed[0]['detail']}", gate_results, sol)
        for g in failed:
            g["advisory"] = True
            warnings.append(f"{g['name']}: {g['detail']}")
        log(f"  ⚠ advisory gate(s) failed (applied anyway, USER 2026-09-09): "
            f"{[g['name'] for g in failed]}")

    _p(50, "all gates passed — staging the transaction...")
    tx_info = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=correction_id,
        scale_diag_new=diagnose.regenerate_scale_diagnostics(
            output_dir, {}, epoch_from + 1, correction_id),
        floor_npz=sol["floor_npz"], log=log, progress=progress)
    _p(90, "atomic swap...")
    swap_transaction(output_dir, tx_info, log=log)
    _p(93, "updating the instance store in place...")
    store_summary = update_instance_store(
        output_dir, R_kf, t_kf, k_kf, session.frames, log=log)

    report = build_report(
        correction_id=correction_id, kind="floor", operator=operator,
        status="applied", instance_ids=None, visits=None,
        observability=None,
        diagnosis=[{"model": model, "model_params": sol["model_params"]}],
        solutions=[{"anchors": sol["anchors"],
                    "per_kf_report": sol["per_kf_report"],
                    "exam": sol["exam"]}],
        distribution=dist_report, gates=gate_results, overrides=None,
        epoch_from=epoch_from, epoch_to=tx_info["epoch_to"],
        extra={"points_moved": tx_info["points_moved"],
               "pose_copies_skipped": tx_info["pose_copies_skipped"],
               "instance_store": store_summary,
               "warnings": warnings},
        elapsed_s=time.time() - t0)
    path = save_report(output_dir, report)
    ledger.record_run(
        output_dir, correction_id=correction_id, epoch_from=epoch_from,
        epoch_to=tx_info["epoch_to"], kind="floor", operator=operator,
        instance_ids=[], visits=[], observability=[],
        diagnosis=[{"model": model, "model_params": sol["model_params"]}],
        anchors=sol["anchors"], gates=gate_results, overrides=None,
        report_path=str(path.relative_to(output_dir)))
    _p(100, f"✅ floor alignment applied (epoch {tx_info['epoch_to']}, "
            f"model {model}) — select any epoch to compare")
    return report


def run_revisit(output_dir, operator: str, log: Callable = print,
                progress: Callable = _noop_progress,
                cfg: Optional[CorrectionConfig] = None,
                previews: bool = True) -> dict:
    """Geometric loop closure (kind=revisit, USER 2026-09-09): detect the
    revisits from poses + intrinsics + provenance (no segmentation, no
    descriptors), solve ONE joint closure per pair of visits, distribute it
    over the chunk pose graph, apply transactionally — same gates (advisory),
    same ledger, same epoch selector."""
    from correction import posegraph, revisit as revisit_mod
    from correction.units import load_chunk_plan
    t0 = time.time()
    output_dir = Path(output_dir)
    if cfg is None:
        cfg = load_correction_config()
    _check_ready(output_dir)
    correction_id = ledger.new_correction_id()
    epoch_from = current_epoch(output_dir)
    advisory = cfg.gates.mode == "advisory"

    def _p(pct, msg):
        log(msg)
        progress(pct, msg)

    def _reject(reason, gates_list, revisits=None, dist=None):
        report = build_report(
            correction_id=correction_id, kind="revisit", operator=operator,
            status="rejected", instance_ids=None, visits=None,
            observability=None, diagnosis=None,
            solutions=(revisits or {}).get("closures"),
            distribution=dist, gates=gates_list, overrides=None,
            epoch_from=epoch_from, epoch_to=None, rejection_reason=reason,
            extra={"revisits": {k: v for k, v in (revisits or {}).items()
                                if k != "_closures_full"}},
            elapsed_s=time.time() - t0)
        path = save_report(output_dir, report)
        ledger.record_run(
            output_dir, correction_id=correction_id, epoch_from=epoch_from,
            epoch_to=epoch_from, kind="revisit", operator=operator,
            instance_ids=[], visits=[], observability=[], diagnosis=[],
            anchors=[], gates=gates_list, overrides=None,
            report_path=str(path.relative_to(output_dir)),
            verdict="rejected")
        _p(100, f"❌ revisit closure REJECTED: {reason}")
        return report

    _p(2, "loading cloud + provenance...")
    session = load_session(output_dir)
    g_int = gates.gate_integrity(session)
    if not g_int["passed"]:
        return _reject(g_int["detail"], [g_int])

    def _sub_progress(pct, msg):
        progress(5 + pct * 35 / 100, msg)   # detector spans 5..40 %

    rev = revisit_mod.detect_revisits(session, cfg, log=log,
                                      progress=_sub_progress,
                                      previews=previews)
    closures = [c for c in rev["_closures_full"] if c["accepted"]]
    if not closures:
        why = ("no revisit found — the walk never saw the same place twice "
               "beyond the temporal gap" if not rev["regions"] else
               "no joint closure improved every block of its visit pair — "
               "nothing consistent to apply")
        return _reject(why, [g_int], rev)

    _p(42, "distributing the closures over the chunk pose graph...")
    plan = load_chunk_plan(output_dir)
    R_kf, t_kf, dist_report = posegraph.distribute_closures(
        plan, session.poses, closures, cfg, log=log)
    k_kf = np.ones(session.n_kf)

    g_plaus = gates.gate_plausibility(
        [{"R": np.asarray(c["R"]), "t": np.asarray(c["t"])} for c in closures], cfg)
    g_cont = gates.gate_continuity(dist_report, cfg)
    gate_results = [g_int, g_plaus, g_cont]
    failed = [g for g in gate_results if not g["passed"]]
    warnings: List[str] = []
    if failed:
        if not advisory:
            return _reject(f"gate(s) failed: {[g['name'] for g in failed]} — "
                           f"{failed[0]['detail']}", gate_results, rev,
                           dist_report)
        for g in failed:
            g["advisory"] = True
            warnings.append(f"{g['name']}: {g['detail']}")
        log(f"  ⚠ advisory gate(s) failed (applied anyway, USER 2026-09-09): "
            f"{[g['name'] for g in failed]}")

    _p(50, "staging the transaction...")
    tx_info = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=correction_id,
        scale_diag_new=diagnose.regenerate_scale_diagnostics(
            output_dir, {}, epoch_from + 1, correction_id),
        log=log, progress=progress)
    _p(90, "atomic swap...")
    swap_transaction(output_dir, tx_info, log=log)
    _p(93, "updating the instance store in place...")
    store_summary = update_instance_store(
        output_dir, R_kf, t_kf, k_kf, session.frames, log=log)

    closures_clean = [{k: v for k, v in c.items() if k not in ("R", "t")}
                      for c in closures]
    report = build_report(
        correction_id=correction_id, kind="revisit", operator=operator,
        status="applied", instance_ids=None, visits=None,
        observability=None, diagnosis=None, solutions=closures_clean,
        distribution=dist_report, gates=gate_results, overrides=None,
        epoch_from=epoch_from, epoch_to=tx_info["epoch_to"],
        extra={"points_moved": tx_info["points_moved"],
               "pose_copies_skipped": tx_info["pose_copies_skipped"],
               "instance_store": store_summary,
               "revisits": {k: v for k, v in rev.items()
                            if k != "_closures_full"},
               "warnings": warnings},
        elapsed_s=time.time() - t0)
    path = save_report(output_dir, report)
    ledger.record_run(
        output_dir, correction_id=correction_id, epoch_from=epoch_from,
        epoch_to=tx_info["epoch_to"], kind="revisit", operator=operator,
        instance_ids=[], visits=[], observability=[], diagnosis=[],
        anchors=[{"kf": c["later_kfs"][0], "rot_deg": c["rot_deg"],
                  "t_m": c["t_norm_m"], "k": 1.0} for c in closures_clean],
        gates=gate_results, overrides=None,
        report_path=str(path.relative_to(output_dir)))
    _p(100, f"✅ revisit closure applied (epoch {tx_info['epoch_to']}, "
            f"{len(closures)} closure(s)) — select any epoch to compare")
    return report


def run_select(output_dir, epoch: int, operator: str = "user",
               log: Callable = print) -> dict:
    """Show the session in one of its epochs. Nothing is approved or undone.

    USER 2026-09-16: *"todas viven, solo se seleccionan y la que se selecciona
    se muestra"*. Approve used to delete every previous epoch and Undo the
    current one, so a session could only hold two states and choosing wrong
    destroyed the other. Every epoch now stays on disk and this only decides
    which one is on screen.

    The geometry is swapped by `select_epoch`; the instance store has to follow
    it, which means composing the transforms of the epochs BETWEEN the two —
    inverted while walking UP to their common ancestor, forward while walking
    DOWN to the chosen one. Ancestry, not arithmetic: a correction run on top
    of an older epoch branches the history, so cur and epoch are not always on
    the same line (`epoch_path`).
    """
    from correction.apply import select_epoch
    output_dir = Path(output_dir)
    epoch = int(epoch)
    cur = current_epoch(output_dir)
    if epoch == cur:
        return {"ok": True, "epoch": cur, "changed": False,
                "available": [e["epoch"] for e in available_epochs(output_dir)]}

    # (epoch, inverse) for every edge to travel — each transform is stored and
    # exact, so the store lands on the geometry, never near it
    moves = []
    for e, inverse in epoch_path(output_dir, cur, epoch):
        try:
            moves.append((ledger.load_epoch_npz(output_dir, e), inverse))
        except RuntimeError as err:
            raise RuntimeError(
                f"epoch {e} has no persisted transform, so the instance store "
                f"cannot follow the geometry to epoch {epoch}: {err}")

    res = select_epoch(output_dir, epoch, log=log)

    # The record has to name the epoch actually on screen. It travels with the
    # geometry whenever the epoch that wrote it listed it as an artifact; an
    # epoch published before that was the rule leaves the previous record live,
    # and then `current_epoch` lies — the session showed epoch 0 while the file
    # still said 3, and the next correction numbered itself from the lie
    # (pccr 2026-09-18). Repairing it here costs the correction_id of those old
    # epochs and nothing else.
    if current_epoch(output_dir) != epoch:
        from correction.epoch import make_epoch_record, EPOCH_FILE
        (output_dir / EPOCH_FILE).write_text(json.dumps(make_epoch_record(
            epoch, f"select/epoch_{epoch}", max(epoch - 1, 0)), indent=2))
        log(f"  epoch record did not travel with the geometry — rewritten to "
            f"epoch {epoch}")

    for mv, inverse in moves:
        R, t, k, b, frames = (mv["R_kf"], mv["t_kf"], mv["k_kf"],
                              np.asarray(mv["b_kf"]), mv["frames"])
        if inverse:
            R = np.transpose(R, (0, 2, 1))
            t = -np.einsum('nij,nj->ni', R, mv["t_kf"])
            k = 1.0 / mv["k_kf"]
            b = -b / mv["k_kf"]        # inverse of z' = k z + b is z = z'/k − b/k
        try:
            update_instance_store(output_dir, R, t, k, frames, log=log, b_kf=b)
        except RuntimeError as e:
            log(f"  instance-store refresh failed (the geometry IS at epoch "
                f"{epoch}; the store stays stale until the next rebuild): {e}")
            break
    res["ok"] = True
    res["available"] = [e["epoch"] for e in available_epochs(output_dir)]
    log(f"[correction] session shown at epoch {epoch} "
        f"(available: {res['available']})")
    return res


def state(output_dir) -> dict:
    """Current correction state for the UI.

    There is no verdict to wait for any more (USER 2026-09-16): the state is
    simply which epoch is on screen and which ones the session holds, so the
    UI can offer them. ``status`` is "applied" while the session has more than
    the original reconstruction — that is when there is something to compare.
    """
    output_dir = Path(output_dir)
    epochs = available_epochs(output_dir)
    last = ledger.last_run(output_dir)
    st = {"epoch": current_epoch(output_dir),
          "epochs": epochs,
          "status": "applied" if len(epochs) > 1 else "none"}
    if last is not None:
        st["correction_id"] = last["correction_id"]
        st["kind"] = last["kind"]
        rp = output_dir / last.get("report", "")
        if rp.is_file():
            st["report"] = json.loads(rp.read_text())
    return st
