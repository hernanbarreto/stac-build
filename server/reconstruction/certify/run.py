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
                  Approve/Undo is the verdict; veto (evaluation) → the
                  iteration is rejected, the previous epoch stays bit-for-bit
        → ONE epoch (transaction + swap + ledger, kind "certify") holding the
          composed per-keyframe transform (depth k·z+b along the ray, then
          rigid) and the witness fields; report per epoch
        improvement(m, prev) < eps → stop

Order inside an iteration: scale → poses → depth (poses on an open scale do
not close). Every epoch stays pending (Approve/Undo in the kit — several
pending epochs form a chain; Undo pops the last one). The acta
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
                    max_iters: Optional[int] = None, apply: bool = True) -> dict:
    """Run the loop; returns the acta (also output/certify_acta.json)."""
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
    from loop_utils.lie import se3_exp
    cfg = cfg or load_loops_config()
    ccfg = correction_cfg or load_correction_config()
    ccert = cfg.certify
    session_dir = Path(session_dir)
    output_dir = session_dir / "output"
    t_start = time.time()
    assert_no_interrupted_swap(output_dir)
    n_iters = int(max_iters if max_iters is not None else ccert.max_iters)
    gdict = {"window_kf": cfg.loops.min_gap_keyframes // 2,
             "sigma_floor_m": cfg.loop.max_residual_m / 4.0,
             "ambiguous_sigma_factor": cfg.loop.ambiguous_sigma_factor}
    acta = {"version": 1, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "operator": operator,
            "max_iters": n_iters, "eps": ccert.eps, "loop_density": float(loop_density),
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
        inst = instance_edges(sess, cands_now, ccfg, cfg, log=_log) if cands_now else []
        vis = visit_edges(sess, ccfg, ccert.visit_loops, cfg.graph.loop_sigma_rot_deg, log=_log)
        return _subsample(inst, loop_density) + _subsample(vis, loop_density)

    base = base_frames if base_frames is not None else load_session_frames(output_dir, log)
    from correction.epoch import current_epoch
    acta["epoch_initial"] = current_epoch(output_dir)
    prev = None

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
        scale_meas = copy_scale_rows(session, cands, ccert.scale, ccert.visit_loops.window_kf, log=log)
        edges_now = _all_edges(session, cands)
        extra = [dict(e) for e in (extra_loop_edges or [])]
        rec["loops"] = [{k: v for k, v in m.items() if k not in ("Z", "X", "info_t", "info_rot")}
                        for m in edges_now]
        rec["scale_measurements"] = scale_meas
        if prev is None:
            frames0 = frames_with_state(base, session)
            fields0 = witness_fields(session.xyz, session.fg, session.data["pixel_row"],
                                     session.data["pixel_col"], frames0, cfg.witness, instances, store, dyn,
                                     device=device)
            prev = _state_metrics(session, frames0, edges_now, {}, instances, fields0)
            acta["metrics_initial"] = prev
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
        # 3a) the GREEDY loop first (USER 2026-09-14): one duplicate at a time,
        #     and the measurement — points landing inside their masks — decides.
        #     A batch fit over closures that contradict each other by a factor
        #     of five lands on a compromise satisfying none of them; applying
        #     one and looking is what separates the right ones from the wrong.
        #     It falls through to the graph when it accepts nothing.
        greedy_rep = None
        R_g = t_g = None
        if ccert.greedy.enabled and edges:
            from reconstruction.certify.iterate import GreedyLoop
            gl = GreedyLoop(session_s, edges, ccfg, cfg, output_dir, session_dir,
                            instances, ccert.greedy, log=log)
            greedy_rep = gl.run()
            comp = gl.composed()
            if comp is not None:
                R_g, t_g, _k_g = comp
            rec["greedy"] = greedy_rep
        prep = run_keyframe_graph(output_dir, session_dir, cfg, operator=operator, apply=False,
                                  extra_loop_edges=edges, use_structural=True, log=log,
                                  session=session_s, use_fork_edges=use_fork_edges)
        pose_moved = False
        if prep.get("verdict") == "APPLY" and prep.get("xi"):
            X = np.stack([se3_exp(np.asarray(x)) for x in prep["xi"]])
            # a correction inside the closures' own σ floor is noise, not a move
            pose_moved = bool(np.max(np.linalg.norm(X[:, :3, 3], axis=1)) > float(ccert.visit_loops.sigma_floor_m))
        if R_g is not None:
            # the greedy chain is the pose correction: every step of it was
            # accepted because MORE points landed in their masks, which the
            # graph's own judges cannot see
            R2, t2 = R_g, t_g
            pose_moved = bool(np.max(np.linalg.norm(t2, axis=1))
                              > float(ccert.visit_loops.sigma_floor_m))
            prep = dict(prep, verdict="APPLY" if pose_moved else "IDENTITY",
                        source="greedy", greedy=greedy_rep)
            log(f"[certify] pose correction from the greedy chain: "
                f"{greedy_rep['epochs']} step(s) over {greedy_rep['trials']} trial(s), "
                f"points in mask {greedy_rep['points_in_mask_before']:,} → "
                f"{greedy_rep['points_in_mask_after']:,}")
        elif pose_moved:
            R2, t2 = X[:, :3, :3].copy(), X[:, :3, 3].copy()
        else:
            R2, t2 = I3.copy(), np.zeros((N, 3))
            if prep.get("verdict") == "APPLY":
                prep = dict(prep, verdict="IDENTITY",
                            identity_reason=f"pose correction within the closure σ floor "
                                            f"({ccert.visit_loops.sigma_floor_m} m)")
                log(f"[certify] pose graph: correction within the σ floor — identity")
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
        # epoch and leaves Approve/Undo as the verdict (USER 2026-09-13).
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
                                   scale_diag_new=scale_diag_new, floor_npz=None, log=log, progress=None,
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
                f"under gates.mode advisory; Approve/Undo is the verdict")
            break
        if improvement < ccert.eps:
            acta["stopped_at"] = it
            acta["stop_reason"] = f"improvement {improvement:.4f} below eps {ccert.eps} — converged"
            break
    else:
        acta["stopped_at"] = n_iters - 1
        acta["stop_reason"] = f"max_iters {n_iters} reached"
    acta["metrics_final"] = prev if prev is not None else acta.get("metrics_initial")
    acta["epoch_final"] = current_epoch(output_dir)
    acta["elapsed_s"] = round(time.time() - t_start, 1)
    (output_dir / ACTA_JSON).write_text(json.dumps(acta, indent=1, default=float))
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
