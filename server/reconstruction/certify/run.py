"""§9 certification loop on a finished session — a STAGE of the reconstruction
pipeline (workers/certify_worker.py; USER 2026-09-13: every loop closure —
SALAD/exact bridges inside the fork, SAM3 instance copies, geometric
revisits — is applied inside "Reconstruir", the cloud the UI receives is the
corrected one).

    for it in range(certify.max_iters):
        cands   = SAM3 instance candidates (§4.4/§4.5, the spatial gate decides)
        loops   = the copies of every candidate measured on the CURRENT cloud
                  (rigid closure → SE(3) edge; Sim3 → scale row s_ab)
                + the geometric revisit regions (correction.revisit) → SE(3) edges
        scales  = scale graph (§5): loop rows + absolute rows → r_k per chunk
        apply(scales)                        in memory: depth × r_k, chain-continuous shift
        loops   = re-measured on the scaled geometry → exact SE(3) edges
        poses   = keyframe graph (§4.3 + §4.6, structural constraints) → X_g
        depth   = depth by correspondences (§6.4): tracks + contours + pairwise sensor
        cloud   = the composed state + witnesses (§6.1–6.3)
        m       = metrics (§10) on that state
        gates(m, prev): MEASURED against the previous state. certify.gates.mode
                  advisory (production) → a failed gate is a ⚠ warning in the
                  acta / the kit, the iteration is APPLIED and the visual
                  the epoch selector is the verdict; veto (evaluation) → the
                  iteration is rejected, the previous epoch stays bit-for-bit
        → ONE epoch (transaction + swap + ledger, kind "certify") holding the
          composed per-keyframe transform (depth k·z+b along the ray, then
          rigid) and the witness fields; report per epoch
        improvement(m, prev) < eps → stop

Order inside an iteration: scale → poses → depth (poses on an open scale do
not close). Every epoch stays on disk and is SELECTABLE in the kit (USER
2026-09-16: nothing is approved and nothing is undone). The acta
(output/certify_acta.json) lists the iterations, their metrics, gates and
where and why the loop stopped.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from reconstruction.certify.loops_posthoc import copy_scale_rows, instance_edges, visit_edges
from reconstruction.certify.scale_stage import chunk_of_keyframes, scale_transforms, solve_scale_stage
from reconstruction.quality.report import compute_metrics, write_epoch_report
from reconstruction.witness.fields import WITNESS_FIELDS, add_fields
from reconstruction.witness.frames import load_session_frames
from reconstruction.witness.mask_votes import load_mask_store
from reconstruction.witness.run import dynamic_instance_ids, witness_fields

ACTA_JSON = "certify_acta.json"


def _vendor_on_path():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def transformed_session(session, R_kf, t_kf, k_kf, b_kf=None):
    """The session after a per-keyframe (depth k·z+b, then rigid) transform,
    in memory (cloud, poses, camera centres)."""
    from correction.apply import transform_poses, warp_full_cloud
    xyz, _ = warp_full_cloud(session, R_kf, t_kf, k_kf, log=lambda m: None, b_kf=b_kf)
    poses = transform_poses(session.poses, R_kf, t_kf)
    cam = {int(f): poses[k][:3, 3] for f, k in session.kf_index.items()}
    return dataclasses.replace(session, xyz=xyz, poses=poses, cam_center=cam)


def frames_with_state(base_frames: Dict[int, dict], session, k_by_frame=None, b_by_frame=None
                      ) -> Dict[int, dict]:
    """The frames dict re-posed from the session's CURRENT poses, depth
    corrected by the iteration's affine (k, b) where given."""
    out = {}
    for f, rec in base_frames.items():
        k = session.kf_index.get(int(f))
        if k is None:
            continue
        d = np.asarray(rec["depth"], np.float32)
        kv = float(k_by_frame.get(int(f), 1.0)) if k_by_frame else 1.0
        bv = float(b_by_frame.get(int(f), 0.0)) if b_by_frame else 0.0
        if kv != 1.0 or bv != 0.0:
            valid = np.isfinite(d) & (d > 0)
            d = np.where(valid, d * kv + bv, d).astype(np.float32)
        out[int(f)] = {"depth": d, "K": rec["K"], "T": session.poses[k]}
    return out


def _subsample(items: list, density: float) -> list:
    if density >= 1.0 or not items:
        return items
    n = max(1, int(round(len(items) * float(density))))
    order = sorted(range(len(items)), key=lambda q: (items[q].get("i", 0), items[q].get("j", 0)))
    step = len(items) / float(n)
    keep = sorted(set(int(q * step) for q in range(n)))
    return [items[order[q]] for q in keep]


def _gates(m: dict, prev: Optional[dict], gcfg) -> List[dict]:
    out = []
    if prev is None:
        return out

    def _g(name, value, thr, passed, detail=""):
        out.append({"name": name, "value": value, "threshold": thr, "passed": bool(passed),
                    "detail": detail})

    sm, sp = m["seam_residual"]["median_m"], prev["seam_residual"]["median_m"]
    if sm is not None and sp is not None:
        _g("seam_residual_not_worse", sm, sp + gcfg.max_seam_degradation_m,
           sm <= sp + gcfg.max_seam_degradation_m, "held-out surface pairs, median")
    lm = m["loop_residual"]["median_demand_m"]; lp = prev["loop_residual"]["median_demand_m"]
    if lm is not None and lp is not None:
        _g("loop_residual_not_worse", lm, lp + gcfg.max_loop_residual_increase_m,
           lm <= lp + gcfg.max_loop_residual_increase_m, "median correction the revisits still demand")
    dm = m["depth_disagreement"]; dp = prev["depth_disagreement"]
    cur = dm["holdout_rel_after"] if dm["holdout_rel_after"] is not None else dm["pair_rel_median_before"]
    old = dp["holdout_rel_after"] if dp["holdout_rel_after"] is not None else dp["pair_rel_median_before"]
    if cur is not None and old is not None:
        _g("depth_disagreement_not_worse", cur, old + gcfg.max_depth_disagreement_increase,
           cur <= old + gcfg.max_depth_disagreement_increase, "multi-view depth disagreement")
    wm, wp = m.get("witnesses"), prev.get("witnesses")
    if wm and wp:
        v0 = wp["status_fraction"]["verified"]; v1 = wm["status_fraction"]["verified"]
        _g("verified_share_kept", v1, v0 - gcfg.max_verified_drop_frac,
           v1 >= v0 - gcfg.max_verified_drop_frac, "fraction of verified points")
    if gcfg.duplicates_must_not_increase:
        _g("duplicates_not_more", m["duplicates"]["n"], prev["duplicates"]["n"],
           m["duplicates"]["n"] <= prev["duplicates"]["n"], "instances still split in two copies")
    return out


def certify_session(session_dir, *args, **kwargs) -> dict:
    """Run the certification holding the session lock.

    The run rewrites segmentation_result.json, seg_masks.npz,
    classification.npy and scene_r.db many times (every split does), and it
    swaps whole epochs. Nothing else may write those files while it does.
    pccr 2026-09-14: the UI's floor levelling rewrote the 235 MB result in the
    middle of a split's own write and the run died on the corrupt document —
    the correction module's lock never saw it because that lock is a dict
    inside the backend process and the certification is not in it.
    """
    from pathlib import Path as _Path
    from session_lock import session_lock
    output_dir = _Path(session_dir) / "output"
    with session_lock(output_dir, "certification",
                      owner=str(kwargs.get("operator", "auto"))):
        return _certify_session(session_dir, *args, **kwargs)


def _certify_session(session_dir, cfg=None, operator: str = "auto", log: Callable[[str], None] = print,
                    correction_cfg=None, device=None, base_frames: Optional[Dict[int, dict]] = None,
                    tracks=None, loop_density: float = 1.0, detect_loops: bool = True,
                    extra_loop_edges: Optional[Sequence[dict]] = None, use_fork_edges: bool = True,
                    max_iters: Optional[int] = None, apply: bool = True,
                    progress: Optional[Callable[[int, str], None]] = None) -> dict:
    """Run the loop; returns the acta (also output/certify_acta.json).

    `progress(pct, message)` reports the REAL advance of each stage. It used
    to report nothing: the worker sent 5 % on entry and 100 % on exit, so the
    user watched a bar frozen at 5 % for the 40 minutes in between, and the
    correction's own progress was discarded at the `stage_transaction` call
    (`progress=None`). The percentages below are the measured weights of a
    full pccr run, not guesses: loops ~35 %, the correction ~30 % (it reports
    its own inside that band), the post-hoc measurement ~25 %, the acta ~10 %.
    """
    def _pc(pct: int, msg: str) -> None:
        if progress is not None:
            try:
                progress(int(pct), str(msg))
            except Exception:  # noqa: BLE001 — reporting never breaks the run
                pass
    from correction.session import load_session
    from correction.apply import stage_transaction, swap_transaction, assert_no_interrupted_swap
    from correction.config import load_correction_config
    from correction.invalidate import update_instance_store
    from correction import ledger
    from reconstruction.loops.config import load_loops_config
    from reconstruction.loops.kf_graph import run_keyframe_graph
    from reconstruction.loops.instance_loops import detect_instance_loops
    from reconstruction.witness.depth_tracks import depth_stage, contour_observations, load_images
    _vendor_on_path()
    cfg = cfg or load_loops_config()
    ccfg = correction_cfg or load_correction_config()
    ccert = cfg.certify
    session_dir = Path(session_dir)
    output_dir = session_dir / "output"
    t_start = time.time()
    assert_no_interrupted_swap(output_dir)
    n_iters = int(max_iters if max_iters is not None else ccert.max_iters)
    # σ FLOOR — measured, not derived from another constant (USER 2026-09-16).
    # No edge may claim more precision than this session can repeat; the session
    # measures exactly that (two copies of a shared frame, the seam residual, or
    # — in a single-chunk run — the frames of one chunk agreeing with each
    # other). The old `max_residual_m / 4` gave 2.5 cm while pccr's own
    # repeatability was 4.77 cm: the edges were claiming certainty the pipeline
    # could not reproduce.
    from reconstruction.certify.repeatability import session_repeatability
    rep_floor = session_repeatability(output_dir,
                                      fallback_m=ccert.visit_loops.sigma_floor_m,
                                      log=log)
    sigma_floor_m = float(rep_floor["sigma_floor_m"])
    acta = {"version": 1, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "operator": operator,
            "max_iters": n_iters, "eps": ccert.eps, "loop_density": float(loop_density),
            # what this session can repeat — every σ in the graph is floored by
            # it, and the acta says whether it was MEASURED or declared
            "sigma_floor": {"m": sigma_floor_m, **{k: v for k, v in rep_floor.items()
                                                   if k != "sigma_floor_m"}},
            "iterations": [], "stopped_at": None, "stop_reason": None, "regressed": False,
            "provenance": "tool_measured"}

    def _instances():
        p = output_dir / "segmentation_result.json"
        return json.loads(p.read_text()).get("instances") or [] if p.exists() else []

    def _state_metrics(session, frames, loops, stages, instances, fields):
        return compute_metrics(session, fields, loops, stages, cfg, output_dir, instances, frames=frames)

    def _all_edges(sess, cands_now, quiet=False):
        """Every loop edge measurable on a state: the SAM3 instance copies
        (§4.4 — the detector's loop|ambiguous candidates, any class but
        dynamic) and the geometric revisit regions, subsampled alike."""
        _log = (lambda m: None) if quiet else log
        inst = (instance_edges(sess, cands_now, ccfg, cfg, log=_log,
                               sigma_floor_m=sigma_floor_m) if cands_now else [])
        vis = visit_edges(sess, ccfg, ccert.visit_loops, cfg.graph.loop_sigma_rot_deg,
                          log=_log, sigma_floor_m=sigma_floor_m)
        return _subsample(inst, loop_density) + _subsample(vis, loop_density)

    base = base_frames if base_frames is not None else load_session_frames(output_dir, log)
    from correction.epoch import current_epoch
    acta["epoch_initial"] = current_epoch(output_dir)

    # ── THE CORRECTION ───────────────────────────────────────────────────
    # USER 2026-09-18: *"esto agregalo completo como algoritmo de correccion
    # en lugar del que tenemos actualmente porque el piso quedo perfecto,
    # hasta las lineas perfectamente alineadas … y que se generen tantas
    # epocas automaticamente hasta que no mejore mas, ademas antes aplicale a
    # la nube el filtrado que te dije en cada etapa … ademas se debe aplicar
    # floor transform en cada ajuste tambien"*.
    #
    # One epoch of the loop: FILTER the cloud for real (the visits that
    # contribute almost nothing and the objects too small to be measured stop
    # existing), MEASURE the drift of every object two separated visits saw,
    # SPREAD what the determined closures agree on over the walk by the
    # drift-rate model, PUBLISH it as a selectable epoch and RE-LEVEL the
    # floor on the new geometry. It repeats until the correction the closures
    # agree on falls under what the session can repeat.
    #
    # Everything below this point MEASURES the result: the loops, the
    # witnesses, the metrics, the gates and the acta. The scale and depth
    # stages still solve their own degrees of freedom, which the visit-drift
    # loop does not touch.
    def _measure_now(sess, insts, st, dy):
        """The state of the session, measured: loops, witnesses, metrics.
        Used for the acta's BEFORE and reused as iteration 0's baseline."""
        fr = frames_with_state(base, sess)
        fl = witness_fields(sess.xyz, sess.fg, sess.data["pixel_row"],
                            sess.data["pixel_col"], fr, cfg.witness, insts, st, dy,
                            device=device)
        cn = []
        if detect_loops and insts:
            cn = [c for c in detect_instance_loops(
                output_dir, session_dir, cfg, log=lambda m: None,
                apply_splits=False)["candidates"]
                  if c["verdict"] in ("loop", "ambiguous")]
        return _state_metrics(sess, fr, _all_edges(sess, cn, quiet=True), {}, insts, fl)

    # The acta's BEFORE is measured BEFORE the correction — it used to be
    # iteration 0's own starting state, which was the same thing only because
    # the correction happened inside iteration 0. It does not any more, so a
    # baseline taken after it would compare the corrected session with itself
    # and report that nothing improved.
    prev = None
    last_m = None
    last_fields = None
    # THE DELIVERABLE IS ONE CORRECTED EPOCH (USER 2026-09-22: *"el entregable es
    # una sola epoch1 ademas del epoch0 que es la original ... no hace falta
    # ahora correr el acta etc, que lleva muchisimo tiempo"*). With
    # `certify.deliverable_only` the stage runs the CORRECTION and nothing else:
    # no §9 before/after measurement, no iteration loop, no acta metrics. The
    # epoch is still published transactionally and epoch 0 is still selectable,
    # so the two states the user compares are both there. The measurement is
    # what costs the time (≈25 % of the stage each pass) and it buys a verdict
    # the user gives by eye at this stage of development.
    _deliverable_only = bool(getattr(ccert, "deliverable_only", False))
    if _deliverable_only:
        n_iters = 0
        log("[certify] deliverable-only: the correction runs and publishes its "
            "epoch; the §9 measurement, the iteration loop and the acta metrics "
            "are SKIPPED (certify.deliverable_only: true)")
    if apply and not _deliverable_only:
        s0 = load_session(output_dir)
        i0 = _instances()
        prev = _measure_now(s0, i0, load_mask_store(output_dir) if i0 else None,
                            dynamic_instance_ids(output_dir))
        acta["metrics_initial"] = prev
        _pc(35, "certification: correcting (depth, then the floor)")
        log(f"[certify] before the correction: objective {prev['objective']:.4f} | "
            f"seams {prev['seam_residual']['median_m']} | closure {prev['closure']['median_m']}")
        del s0, i0
    if apply:

        # THE CORRECTION (USER-VALIDATED, pccr 2026-09-19): depth first, then
        # the floor plane. No translation stage and no loop — see
        # `correction.visit_drift_run.run`.
        from correction.visit_drift_run import run as run_correction
        # the config THIS function already resolved — dropping it made the
        # correction re-read production config.yaml behind the caller's back
        # (found 2026-09-21: on a synthetic session `floor.min_inliers: 5000`
        # demoted every keyframe and no epoch was ever published)
        vd = run_correction(session_dir, log=log, cfg=ccfg,
                            progress=lambda p, m: _pc(35 + int(p * 0.27), m))
        acta["correction"] = {"stages": vd["stages"],
                              "elapsed_s": vd["elapsed_s"],
                              # the correction's own epoch and provenance tag:
                              # the acta used to drop both
                              "epoch": vd.get("epoch"),
                              "provenance": vd.get("provenance")}
        acta["epoch_after_correction"] = current_epoch(output_dir)
        base = load_session_frames(output_dir, log)   # poses moved under us

    for it in range(n_iters):
        t_it = time.time()
        rec = {"iteration": it, "epoch_from": current_epoch(output_dir)}
        session = load_session(output_dir)
        instances = _instances()
        store = load_mask_store(output_dir) if instances else None
        dyn = dynamic_instance_ids(output_dir)
        N = session.n_kf
        # 1) loops on the CURRENT session: the instance candidates (§4.4/§4.5,
        #    splits applied) give the scale rows; the geometric revisits give
        #    the pose edges; the same measurement is this iteration's "before"
        cands = []
        if detect_loops and instances:
            det = detect_instance_loops(output_dir, session_dir, cfg, log=log, apply_splits=True)
            cands = det.get("candidates", [])
            if det.get("splits"):
                session = load_session(output_dir)
                instances = _instances()
                store = load_mask_store(output_dir) if instances else None
            rec["candidates"] = {"n": len(cands), "n_splits": len(det.get("splits", [])),
                                 "n_duplicates": len(det.get("duplicates", []))}
        cands = _subsample([c for c in cands if c.get("verdict") in ("loop", "ambiguous")], loop_density)
        scale_meas = copy_scale_rows(
            session, cands, ccert.scale, ccert.visit_loops.window_kf,
            ccert.visit_loops.max_pairs_per_instance, log=log)
        edges_now = _all_edges(session, cands)
        extra = [dict(e) for e in (extra_loop_edges or [])]
        rec["loops"] = [{k: v for k, v in m.items() if k not in ("Z", "X", "info_t", "info_rot")}
                        for m in edges_now]
        rec["scale_measurements"] = scale_meas
        if prev is None:          # apply=False: the acta measures, nothing moved
            frames0 = frames_with_state(base, session)
            fields0 = witness_fields(session.xyz, session.fg, session.data["pixel_row"],
                                     session.data["pixel_col"], frames0, cfg.witness, instances, store, dyn,
                                     device=device)
            prev = _state_metrics(session, frames0, edges_now, {}, instances, fields0)
            acta["metrics_initial"] = prev
            _pc(62, "certification: measuring the corrected session")
            log(f"[certify] initial: objective {prev['objective']:.4f} | seams "
                f"{prev['seam_residual']['median_m']} | closure {prev['closure']['median_m']} | "
                f"verified {prev['witnesses']['status_fraction']['verified']:.3f}")
        # 2) scale FIRST
        srep = solve_scale_stage(output_dir, session, scale_meas, ccert.scale, log=log)
        ranges, owner = chunk_of_keyframes(output_dir, N)
        I3 = np.tile(np.eye(3), (N, 1, 1))
        if srep.get("applied"):
            k1, t1 = scale_transforms(session, ranges, owner, np.asarray(srep["r"]))
        else:
            k1, t1 = np.ones(N), np.zeros((N, 3))
        session_s = transformed_session(session, I3, t1, k1) if srep.get("applied") else session
        k1_by_frame = {int(f): float(k1[k]) for f, k in session.kf_index.items()}
        # 3) the loops re-measured on the closed scale → SE(3) edges → pose graph
        edges_s = _all_edges(session_s, cands, quiet=True) if srep.get("applied") else edges_now
        edges = [m for m in edges_s if "Z" in m and m.get("trusted")] + extra
        # 3a) THE POSE CORRECTION IS NOT SOLVED HERE ANY MORE (USER 2026-09-18).
        #     It is the VISIT-DRIFT loop, which ran before this iteration and
        #     published its own epochs. It measures the drift where the drift
        #     is actually visible — the SILHOUETTES of the two copies of one
        #     object, aligned in the three orthogonal views of that object's
        #     own OBB, each component measured twice so its determination is
        #     measured too — instead of a 3-D fit that slides freely along
        #     every flat or symmetric surface. What used to decide here, the
        #     greedy chain, regressed this very session's objective by 4.5 %.
        #     The keyframe pose graph still RUNS: its loop residual is a
        #     measurement, and measuring is what the certification is for. It
        #     no longer moves a point.
        prep = run_keyframe_graph(output_dir, session_dir, cfg, operator=operator, apply=False,
                                  extra_loop_edges=edges, use_structural=True, log=log,
                                  session=session_s, use_fork_edges=use_fork_edges)
        pose_moved = False
        R2, t2 = I3.copy(), np.zeros((N, 3))
        if prep.get("verdict") == "APPLY":
            prep = dict(prep, verdict="MEASURED", source="visit_drift",
                        not_applied_reason="the pose correction of this session is the "
                                           "visit-drift loop (correction.visit_drift_run); "
                                           "the keyframe graph is measured, not applied")
            _pc(78, "certification: pose graph measured")
            log("[certify] pose graph measured and NOT applied — the pose "
                "correction of this session is the visit-drift loop")
        session_p = transformed_session(session_s, R2, t2, np.ones(N)) if pose_moved else session_s
        # 4) depth by correspondences on the closed poses
        frames_p = frames_with_state(base, session_p, k1_by_frame)
        contour_obs = None
        if cfg.witness.contours.enabled and instances and store is not None:
            nums = sorted(frames_p)
            pairs = [(nums[i], nums[i + d]) for d in cfg.witness.depth.pair_offsets
                     for i in range(len(nums) - d)]
            pairs += [(int(session.frames[m["i"]]), int(session.frames[m["j"]])) for m in edges_s
                      if "Z" in m]
            images = load_images(session_dir, nums)
            contour_obs = contour_observations(frames_p, instances, store, images, cfg.witness.contours,
                                               pairs) if images else None
        drep = depth_stage(frames_p, cfg.witness, tracks, contour_obs, log=log)
        if drep.get("applied"):
            a_kf = np.array([float(drep["a"][int(f)]) for f in session.frames])
            b_kf = np.array([float(drep["b"][int(f)]) for f in session.frames])
        else:
            a_kf, b_kf = np.ones(N), np.zeros(N)
        session_d = transformed_session(session_p, I3, np.zeros((N, 3)), a_kf, b_kf) if drep.get("applied") \
            else session_p
        # composed per-keyframe transform of this iteration (depth k·z+b about the
        # ORIGINAL camera, then rigid): k = r·a, b = b, R = R2, t = R2·t1 + t2
        k_tot = k1 * a_kf
        b_tot = b_kf
        R_tot = R2
        t_tot = np.einsum("nij,nj->ni", R2, t1) + t2
        moved = bool(srep.get("applied") or pose_moved or drep.get("applied"))
        # 5) witnesses + the loops re-measured on the composed state
        k_by_frame = {int(f): float(k_tot[k]) for f, k in session.kf_index.items()}
        b_by_frame = {int(f): float(b_tot[k]) for f, k in session.kf_index.items()}
        frames_d = frames_with_state(base, session_d, k_by_frame, b_by_frame)
        fields = witness_fields(session_d.xyz, session_d.fg, session_d.data["pixel_row"],
                                session_d.data["pixel_col"], frames_d, cfg.witness, instances, store, dyn,
                                device=device)
        edges_d = _all_edges(session_d, cands, quiet=True) if moved else edges_now
        # 6) metrics + gates
        stages = {"scale": srep, "poses": {k: v for k, v in prep.items() if k not in ("xi",)}, "depth": drep}
        m = _state_metrics(session_d, frames_d, edges_d, stages, instances, fields)
        # the acta's AFTER is the last state MEASURED, not the last one
        # applied: since the correction moved out of the iterations, an
        # iteration that applies nothing is the normal ending, and reporting
        # the pre-correction baseline as "final" said the session had not
        # improved when it had
        last_m, last_fields = m, fields
        gates = _gates(m, prev, ccert.gates)
        failed = [g for g in gates if not g["passed"]]
        advisory = ccert.gates.mode == "advisory"
        gate_warnings = [f"{g['name']}: {g['value']} vs {g['threshold']} ({g['detail']})" for g in failed]
        if failed and advisory:
            for g in failed:
                g["advisory"] = True
            log(f"[certify] ⚠ advisory gate(s) failed — declared, iteration applied (USER 2026-09-13): "
                f"{'; '.join(gate_warnings)}")
        improvement = ((prev["objective"] - m["objective"]) / prev["objective"]
                       if prev["objective"] > 0 else 0.0)
        # An objective that GREW is a regression, never convergence: the iteration
        # left the session worse than it found it. Declared here so the acta, the
        # attention list and the kit all carry it; advisory mode still applies the
        # epoch and leaves the epoch selector as the verdict (USER 2026-09-13).
        regressed = improvement < -ccert.regression_eps
        if regressed:
            gate_warnings.append(
                f"objective_regressed: {prev['objective']:.4f} → {m['objective']:.4f} "
                f"({improvement * 100:+.1f}%, tolerance {ccert.regression_eps * 100:.1f}%)")
            log(f"[certify] ⚠ REGRESSION: objective {prev['objective']:.4f} → "
                f"{m['objective']:.4f} ({improvement * 100:+.1f}%)")
        rec.update({"stages": {"scale": srep,
                               "poses": {k: v for k, v in prep.items() if k not in ("xi", "solver")},
                               "depth": {k: v for k, v in drep.items() if k not in ("a", "b")}},
                    "metrics": m, "gates": gates, "gate_mode": ccert.gates.mode,
                    "gate_warnings": gate_warnings, "geometry_moved": moved,
                    "objective": m["objective"], "objective_prev": prev["objective"],
                    "improvement": improvement, "regressed": regressed,
                    "elapsed_s": round(time.time() - t_it, 1)})
        if failed and not advisory:
            rec["verdict"] = "rejected"
            rec["reason"] = "; ".join(f"{g['name']}: {g['value']} vs {g['threshold']}" for g in failed)
            cid = ledger.new_correction_id()
            rep_path = output_dir / "corrections" / f"report_{cid}.json"
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(json.dumps(rec, indent=1, default=float))
            ledger.record_run(output_dir, correction_id=cid, epoch_from=rec["epoch_from"],
                              epoch_to=rec["epoch_from"], kind="certify", operator=operator,
                              instance_ids=[int(mm["instance_id"]) for mm in scale_meas],
                              visits=[], observability=[], diagnosis=[], anchors=[], gates=gates,
                              overrides={}, report_path=str(rep_path.relative_to(output_dir)),
                              verdict="rejected")
            rec["correction_id"] = cid
            acta["iterations"].append(rec)
            acta["stopped_at"] = it
            acta["stop_reason"] = f"iteration {it} rejected by gate(s): {rec['reason']} — previous epoch kept"
            log(f"[certify] iteration {it} REJECTED: {rec['reason']}")
            break
        if not moved:
            rec["verdict"] = "identity"
            rec["reason"] = "no stage moved geometry (nothing to close)"
            acta["iterations"].append(rec)
            acta["stopped_at"] = it
            acta["stop_reason"] = rec["reason"]
            log(f"[certify] iteration {it}: identity — {rec['reason']}")
            break
        # 7) the epoch
        if apply:
            h2, d2 = add_fields(session.header, session.data, fields)
            rh, rd = session.raw_header, session.raw_data
            if rd is not None and len(rd) == len(d2):
                rh, rd = add_fields(rh, rd, fields)
            sess_tx = dataclasses.replace(session, header=h2, data=d2, raw_header=rh, raw_data=rd)
            cid = ledger.new_correction_id()
            from correction.diagnose import regenerate_scale_diagnostics
            scale_diag_new = regenerate_scale_diagnostics(
                output_dir, {int(f): float(k_tot[k]) for f, k in session.kf_index.items()},
                current_epoch(output_dir) + 1, cid)
            tx = stage_transaction(sess_tx, ccfg, R_tot, t_tot, k_tot, correction_id=cid,
                                   scale_diag_new=scale_diag_new, floor_npz=None, log=log,
                                   progress=lambda p, m: _pc(78 + int(p * 0.18), m),
                                   b_kf=b_tot)
            swap_transaction(output_dir, tx, log=log)
            update_instance_store(output_dir, R_tot, t_tot, k_tot, session.frames, log=log, b_kf=b_tot)
            rep_path = output_dir / "corrections" / f"report_{cid}.json"
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rec["correction_id"] = cid
            rec["epoch_to"] = tx["epoch_to"]
            rep_path.write_text(json.dumps(rec, indent=1, default=float))
            ledger.record_run(output_dir, correction_id=cid, epoch_from=tx["epoch_from"],
                              epoch_to=tx["epoch_to"], kind="certify", operator=operator,
                              instance_ids=[int(mm["instance_id"]) for mm in scale_meas], visits=[],
                              observability=[], diagnosis=[],
                              anchors=[{"kf": int(mm["i"]), "kf2": int(mm["j"]), "sigma_m": float(mm["sigma_m"]),
                                        "observes": mm.get("observability")} for mm in edges_s if "Z" in mm],
                              gates=gates + [g for g in [srep.get("gate")] if g], overrides={},
                              report_path=str(rep_path.relative_to(output_dir)))
            write_epoch_report(output_dir, tx["epoch_to"], m, prev,
                               extra={"iteration": it, "correction_id": cid, "stages": rec["stages"]})
        rec["verdict"] = "applied" if apply else "measured"
        acta["iterations"].append(rec)
        log(f"[certify] iteration {it}: objective {prev['objective']:.4f} → {m['objective']:.4f} "
            f"({rec['improvement'] * 100:+.1f}%) | seams {m['seam_residual']['median_m']} | closure "
            f"{m['closure']['median_m']} | duplicates {m['duplicates']['n']} | verified "
            f"{m['witnesses']['status_fraction']['verified']:.3f} → epoch {rec.get('epoch_to')}"
            + (f" | ⚠ {len(gate_warnings)} advisory gate warning(s)" if gate_warnings else ""))
        prev = m
        if regressed:
            acta["stopped_at"] = it
            acta["regressed"] = True
            acta["stop_reason"] = (
                f"iteration {it} REGRESSED: objective {rec['objective_prev']:.4f} → "
                f"{rec['objective']:.4f} ({improvement * 100:+.1f}%) — the epoch was applied "
                f"under gates.mode advisory; the epoch selector is the verdict")
            break
        if improvement < ccert.eps:
            acta["stopped_at"] = it
            acta["stop_reason"] = f"improvement {improvement:.4f} below eps {ccert.eps} — converged"
            break
    else:
        acta["stopped_at"] = n_iters - 1
        acta["stop_reason"] = f"max_iters {n_iters} reached"
    # ── what the certification MEASURED reaches the cloud ─────────────────
    # The per-point witness columns are a measurement of the geometry the
    # session is showing, and this is the stage that measures them. They used
    # to ride along on the epoch the iteration applied — so they only ever
    # landed when a stage moved geometry, and the correction moved out of the
    # iterations in 2026-09-18. Written only when they are ABSENT: production
    # clouds already carry them from the merge (`witness.at_merge`), and
    # rewriting values a second time would make a rejected iteration look like
    # it had touched the session.
    if apply and last_fields is not None:
        try:
            from correction.session import read_ply, write_ply
            from reconstruction.witness.fields import add_fields as _add
            cp = output_dir / "cleaned_cloud.ply"
            hdr, dat = read_ply(cp)
            missing = [f for f in WITNESS_FIELDS if f not in (dat.dtype.names or ())]
            n_ok = all(len(v) == len(dat) for v in last_fields.values())
            if missing and n_ok:
                h2, d2 = _add(hdr, dat, last_fields)
                write_ply(cp, h2, d2)
                log(f"[certify] witness columns written to the cloud: "
                    f"{', '.join(missing)}")
            elif missing:
                log(f"[certify] ⚠ witness columns not written: measured on "
                    f"{len(next(iter(last_fields.values())))} points, the cloud "
                    f"has {len(dat)}")
        except Exception as e:  # noqa: BLE001 — declared, never silent
            log(f"[certify] ⚠ witness columns could not be written ({e})")

    # ── one quality report per epoch the correction published ─────────────
    # The kit's acta panel reads `quality/report_epoch_<N>.json`. The epochs
    # now come from the visit-drift loop, so each one gets its report here,
    # carrying its OWN measurement (the closures, the filter, the
    # distribution) against the session measured before the correction.
    # It used to read `acta["visit_drift"]["epochs"]`, a key that stopped
    # existing when the multi-epoch loop became the ONE composed epoch
    # (db52016). The loop was dead, so `output/quality/` stayed empty and both
    # `GET /api/certify/report/{id}` and `?epoch=N` answered 404 after every
    # real certification — the acta panel got nothing (found 2026-09-21).
    for e in range(int(acta.get("epoch_initial", 0)) + 1,
                   int(acta.get("epoch_after_correction",
                                acta.get("epoch_initial", 0))) + 1):
        write_epoch_report(output_dir, int(e),
                           last_m if last_m is not None else acta.get("metrics_initial", {}),
                           acta.get("metrics_initial"),
                           extra={"correction": acta.get("correction")})

    # ── the second moment of the geometric cleanup cycle ──────────────────
    # The mask audit MARKED, at match time, every point landing off its own
    # mask in a view that saw it, and removed nothing: the certification still
    # had a chance to move it where it belongs. That chance is now spent. What
    # lands inside today was a drift orphan and is cured; what still lands
    # outside has no correction left to wait for, and leaves the cloud (USER
    # 2026-09-15: "si no lo puedo corregir, lamentablemente lo voy a tener que
    # sacar"). No gate of its own: the marks only exist when the audit ran.
    # SUPERSEDED BY THE MASK FILTER (USER 2026-09-19). Both delete the points
    # that still fall outside their own mask once the pose is corrected, but
    # this one judges against the FUSED INSTANCES and runs AFTER the epoch,
    # while `visit_drift.cloud_filter_masklets` judges against the 270
    # MASKLETS of segmentation.json — the unit the user corrected us to
    # ("no eran 82 instancias, está mal") — and runs INSIDE the transaction,
    # before the single consolidation and the single octree, carrying
    # `min_points` and `min_visit_share` as well. Running both deletes twice,
    # the second time by the weaker criterion, and on a run where the
    # correction did NOT apply it would judge every point with the pose
    # UNcorrected, which is exactly what step 12 exists to avoid.
    # The module stays, selectable, OFF by default — the repo's pattern.
    if not getattr(ccert, "geometric_cleanup", False):
        acta["geometric_cleanup"] = {
            "applied": False,
            "reason": "superseded by the mask filter inside the epoch "
                      "transaction (certify.geometric_cleanup: false)"}
        log("[certify] geometric cleanup OFF — the mask filter already ran "
            "inside the epoch, on the masklets and with the pose corrected")
    else:
      try:
        from segmentation.geometric_cleanup import geometric_cleanup
        acta["geometric_cleanup"] = geometric_cleanup(
            output_dir, session_dir, apply=apply, log=log)
      except Exception as e:  # noqa: BLE001 — declared, never silent
        log(f"[certify] ⚠ geometric cleanup failed ({e}) — the cloud keeps the "
            f"points the audit marked out of place")
        acta["geometric_cleanup"] = {"applied": False, "reason": str(e)}

    if _deliverable_only and not acta.get("stop_reason"):
        acta["stop_reason"] = ("deliverable-only: correction applied and its epoch "
                               "published; §9 measurement, iterations and acta "
                               "metrics skipped by certify.deliverable_only")
    acta["metrics_final"] = (last_m if last_m is not None else
                             (prev if prev is not None else acta.get("metrics_initial")))
    acta["epoch_final"] = current_epoch(output_dir)
    acta["elapsed_s"] = round(time.time() - t_start, 1)
    (output_dir / ACTA_JSON).write_text(json.dumps(acta, indent=1, default=float))
    _pc(97, "certification: writing the acta")
    log(f"[certify] stopped: {acta['stop_reason']} (epoch {acta['epoch_final']}, {acta['elapsed_s']}s)")
    return acta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="certification loop (§9) on a session")
    ap.add_argument("--session", required=True)
    ap.add_argument("--operator", default="cli")
    ap.add_argument("--max-iters", type=int, default=None)
    args = ap.parse_args(argv)
    certify_session(args.session, operator=args.operator, max_iters=args.max_iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
