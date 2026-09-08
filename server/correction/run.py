"""Orchestration of a correction run — the flow of prompt §5.3:

mark → evidence → observability → diagnose → solve → gates → distribute →
apply(tx) → invalidate/regenerate → report → pending → approve | undo.

Every stage's structured output lands in the report, ALSO when the run is
rejected: the user always receives the numbers and the exact reason. A gate
failure returns a ``rejected`` report (and a ledger record) without touching
one byte of the session; only a fully-gated solution reaches the
transactional apply.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from correction import diagnose, distribute, floor as floor_mod, gates, \
    ledger, observability as obs_mod, solve
from correction.apply import (approve_swap, assert_no_interrupted_swap,
                              prev_dir_for, stage_transaction,
                              swap_transaction, undo_swap)
from correction.config import CorrectionConfig, load_correction_config
from correction.epoch import current_epoch
from correction.evidence import extract_evidence
from correction.invalidate import update_instance_store
from correction.report import build_report, save_report
from correction.session import CorrectionSession, load_session


def _noop_progress(pct: float, msg: str) -> None:
    pass


def _check_ready(output_dir: Path) -> None:
    assert_no_interrupted_swap(output_dir)
    if prev_dir_for(output_dir) is not None:
        raise RuntimeError(
            "a correction is pending approval — approve or undo it before "
            "running a new one (the next correction runs ON TOP of the "
            "approved cloud, never beside a pending one)")


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
    (status: pending | rejected)."""
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

    for gi, grp in enumerate(groups):
        members = grp["members"]
        label = f"visit kf {grp['kf_span'][0]}..{grp['kf_span'][1]}"
        # per-object curated evidence inside this group
        shapes = []
        centroids: Dict[int, np.ndarray] = {}
        seg_parts = []
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

        visit_obs = obs_mod.analyze_visit(shapes, centroids, cfg)
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

        # solve --------------------------------------------------------------
        S = session.xyz[src_idx].copy()
        if diag["depth_needed"]:
            cams = np.stack([session.cam_center[int(f)]
                             for f in session.fg[src_idx]])
            S = solve.expand_depth(S, cams, diag["k"])
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
            if big is None:
                return _reject(
                    f"{label}: none of its objects has a reference copy — "
                    f"there is nothing to align it against",
                    gates_list=gate_results, visits=visit_summaries,
                    observability=obs_reports, diagnosis=diag_reports,
                    suggestion="mark also an object that appears in the "
                               "first visit")
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
        if not g_col["passed"]:
            return _reject(
                f"{label}: the copies did NOT collapse — {g_col['detail']}",
                gates_list=gate_results, visits=visit_summaries,
                observability=obs_reports, diagnosis=diag_reports)
        solutions.append({
            "group": gi, "anchor_kf": grp["kfs"][0],
            "kf_span": grp["kf_span"], "R": R, "t": t_full,
            "k": diag["k"], "rot_deg": round(solve.rot_deg(R), 3),
            "t_m": round(float(np.linalg.norm(t_full)), 4),
            "icp_rms_cm": round(rms * 100, 2),
            "residual_cm": {"before": round(res_before * 100, 1),
                            "after": round(res_after * 100, 1)},
            "dof": visit_obs.dof, "unrestrained": visit_obs.unrestrained})

    if not solutions:
        return _reject("no displaced visit gathered enough evidence to "
                       "solve", gates_list=gate_results,
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
    R_kf, t_kf, k_kf, dist_report = distribute.distribute(
        session.n_kf, ev.ref_kf_end, solutions)

    # global gates --------------------------------------------------------
    g_plaus = gates.gate_plausibility(solutions, cfg)
    g_cont = gates.gate_continuity(dist_report, cfg)
    gate_results += [g_plaus, g_cont]
    affected_kfs = list(range(dist_report["identity_until_kf"] + 1,
                              session.n_kf))
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
    if failed:
        names = [g["name"] for g in failed]
        return _reject(
            f"gate(s) failed: {names} — {failed[0]['detail']}",
            gates_list=gate_results, visits=visit_summaries,
            observability=obs_reports, diagnosis=diag_reports,
            solutions=[{k: v for k, v in s.items()
                        if k not in ("R", "t")} for s in solutions],
            distribution=dist_report,
            suggestion=(g_scene.get("floor", {}).get("detail")
                        if "scene_exam" in names else None))

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
        status="pending", instance_ids=instance_ids, visits=visit_summaries,
        observability=obs_reports, diagnosis=diag_reports,
        solutions=solutions_clean, distribution=dist_report,
        gates=gate_results, overrides=overrides, epoch_from=epoch_from,
        epoch_to=tx_info["epoch_to"],
        extra={"points_moved": tx_info["points_moved"],
               "pose_copies_skipped": tx_info["pose_copies_skipped"],
               "instance_store": store_summary},
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
    _p(100, f"✅ correction applied (epoch {tx_info['epoch_to']}) — awaiting "
            f"your verdict: Approve or Undo")
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
    if failed:
        return _reject(f"gate(s) failed: {[g['name'] for g in failed]} — "
                       f"{failed[0]['detail']}", gate_results, sol)

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
        status="pending", instance_ids=None, visits=None,
        observability=None,
        diagnosis=[{"model": model, "model_params": sol["model_params"]}],
        solutions=[{"anchors": sol["anchors"],
                    "per_kf_report": sol["per_kf_report"],
                    "exam": sol["exam"]}],
        distribution=dist_report, gates=gate_results, overrides=None,
        epoch_from=epoch_from, epoch_to=tx_info["epoch_to"],
        extra={"points_moved": tx_info["points_moved"],
               "pose_copies_skipped": tx_info["pose_copies_skipped"],
               "instance_store": store_summary},
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
            f"model {model}) — awaiting your verdict: Approve or Undo")
    return report


def run_verdict(output_dir, verdict: str, operator: str,
                log: Callable = print) -> dict:
    """Approve or undo the pending correction. Undo restores the previous
    epoch exactly; approve removes it. Both land in the ledger."""
    output_dir = Path(output_dir)
    pend = ledger.pending_run(output_dir)
    if pend is None:
        raise RuntimeError(f"no pending correction to {verdict}")
    if verdict == "approved":
        manifest = approve_swap(output_dir, log=log)
    elif verdict == "undone":
        # capture the undone epoch's exact transform BEFORE the swap discards
        # it — the store's findings must be inverse-warped back
        undone = ledger.load_epoch_npz(output_dir, current_epoch(output_dir))
        manifest = undo_swap(output_dir, log=log)
        R_inv = np.transpose(undone["R_kf"], (0, 2, 1))
        t_inv = -np.einsum('nij,nj->ni', R_inv, undone["t_kf"])
        k_inv = 1.0 / undone["k_kf"]
        try:
            update_instance_store(output_dir, R_inv, t_inv, k_inv,
                                  undone["frames"], log=log)
        except RuntimeError as e:
            log(f"  instance-store refresh after undo failed (geometry is "
                f"restored; store stays stale until the next rebuild): {e}")
    else:
        raise RuntimeError(f"invalid verdict {verdict!r}")
    entry = ledger.record_verdict(output_dir, pend["correction_id"],
                                  verdict, operator)
    return {"ok": True, "correction_id": pend["correction_id"],
            "verdict": verdict, "epoch": current_epoch(output_dir),
            "manifest": manifest, "ledger": entry}


def state(output_dir) -> dict:
    """Current correction state for the UI."""
    output_dir = Path(output_dir)
    pend = ledger.pending_run(output_dir)
    st = {"epoch": current_epoch(output_dir),
          "status": "pending" if (pend is not None
                                  and prev_dir_for(output_dir) is not None)
          else "none"}
    if pend is not None:
        st["correction_id"] = pend["correction_id"]
        st["kind"] = pend["kind"]
        rp = output_dir / pend.get("report", "")
        if rp.is_file():
            import json as _json
            st["report"] = _json.loads(rp.read_text())
    return st
