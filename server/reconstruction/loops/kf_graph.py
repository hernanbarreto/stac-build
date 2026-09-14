"""Post-hoc keyframe SE(3) graph over a reconstructed SESSION (§4.3 + §4.6 + §8).

The fork runs the graph once inside the reconstruction (loops from SALAD);
this runner re-solves it on the finished session with everything the
reconstruction could not know yet — the SAM3 instance loops (§4.4), the
geometric revisit closures, the structural constraints (§4.6) — and applies
the result through the correction package's transactional machinery: an
EPOCH (geometry_epoch.json), the ledger, ``corrections/epoch_<N>.npz``,
Approve/Undo in the UI. Nothing here is silent: every gate lands in the
report with value / threshold / verdict; identity is a verdict too.

Inputs: cleaned_cloud.ply (+ provenance), camera_poses.txt, camera_frames.txt,
segmentation_result.json (structural edges), maplong_run/loop_edges.json
(keyframe loop edges, translations rescaled by the residual metric s),
loop_semantics.json / the instance store (classes). Solver:
vendor/VGGT-Long/loop_utils/pose_graph.py.

Priors (pccr 2026-09-13 21:00 lesson): the graph holds ODOMETRY (σ from
config — Omega's adjacent keyframe poses are precise, the two-copy point
disagreement is NOT a pose uncertainty), the measured LOOP closures, and the
structural constraints that come from segmented, VLM-classified structural
instances (walls, columns, parallel members). No per-pose gravity prior (a
handheld camera pitches; "camera down = consensus down" bent the chain) and
no floor datum from the per-keyframe low band (USER 2026-09-09: that band is
not a validated floor). With those two and σ_odo inflated to the point
disagreement, 0 loops still moved poses 3.2 m / 15.6°.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from reconstruction.loops import structural as st
from reconstruction.loops.config import MetricGraphConfig, load_loops_config

REPORT_JSON = "keyframe_graph.json"


def _vendor_on_path():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def _metric_scale_applied(output_dir: Path) -> float:
    p = output_dir / ".metric_scale_applied"
    if not p.exists():
        return 1.0
    try:
        return float(p.read_text().strip().split("=")[-1])
    except ValueError as e:
        raise RuntimeError(f"{p}: unreadable scale marker ({e})") from e


def _loop_edges(output_dir: Path, s_metric: float) -> List[dict]:
    """Keyframe loop edges measured by the fork (loop_edges.json), translations
    scaled by the residual metric factor scale_align applied afterwards."""
    p = output_dir / "maplong_run" / "loop_edges.json"
    if not p.exists():
        return []
    rep = json.loads(p.read_text())
    out = []
    for e in rep.get("edges", []):
        ke = e.get("keyframe_edge")
        if not ke or e.get("status") not in ("accepted", "scale_break"):
            continue
        Z = np.asarray(ke["Z"], np.float64).copy()
        Z[:3, 3] *= float(s_metric)
        out.append({"i": int(ke["i"]), "j": int(ke["j"]), "Z": Z,
                    "sigma_m": float(ke["sigma_m"]) * float(s_metric),
                    "sigma_deg": float(ke["sigma_deg"]), "bridge": int(ke["bridge"]),
                    "status": e["status"], "source": e.get("candidate", {}).get("source", "salad")})
    return out


def _classes(output_dir: Path, instances: List[dict], default_class: str) -> Dict[int, str]:
    p = output_dir / "loop_semantics.json"
    out = {}
    if p.exists():
        rep = json.loads(p.read_text())
        for k, v in rep.get("classes", {}).items():
            out[int(k)] = str(v.get("class", default_class))
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        out.setdefault(iid, default_class)
    return out


def _per_kf_points_fn(session):
    order = np.argsort(session.ks, kind="stable")
    ks_sorted = session.ks[order]

    def fn(k, n_max=20000, seed=0):
        lo = np.searchsorted(ks_sorted, int(k), side="left")
        hi = np.searchsorted(ks_sorted, int(k), side="right")
        idx = order[lo:hi]
        if len(idx) > n_max:
            idx = np.random.default_rng(seed).choice(idx, n_max, replace=False)
        return session.xyz[idx]
    return fn


def _holdout_pairs(session, per_kf_points, offsets, stride, n_samp, max_nn_m, seed=0):
    """Held-out judge on the cloud: NN distance between the points of
    keyframes f and f+d (exact correspondences do not exist across frames of
    the cloud, the nearest measured point is the surface witness). Only NN
    pairs within ``max_nn_m`` are correspondences — beyond it the two frames
    do not see the same surface."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    pairs = []
    for f in range(0, session.n_kf, max(int(stride), 1)):
        Pf = per_kf_points(f, n_samp, seed)
        if len(Pf) < 50:
            continue
        for d in offsets:
            g = f + int(d)
            if g >= session.n_kf:
                continue
            Pg = per_kf_points(g, 4 * n_samp, seed)
            if len(Pg) < 50:
                continue
            tree = cKDTree(Pg)
            dist, nn = tree.query(Pf, k=1)
            keep = dist <= float(max_nn_m)
            if keep.sum() < 20:
                continue
            pairs.append((f, g, Pf[keep], Pg[nn[keep]]))
    return pairs


def _held_median(pairs, X):
    vals = []
    for f, g, p, q in pairs:
        p2 = p @ X[f][:3, :3].T + X[f][:3, 3]
        q2 = q @ X[g][:3, :3].T + X[g][:3, 3]
        vals.append(float(np.median(np.linalg.norm(p2 - q2, axis=1))))
    return float(np.median(vals)) if vals else float("nan")


def run_keyframe_graph(output_dir, session_dir, cfg: Optional[MetricGraphConfig] = None,
                       operator: str = "auto", apply: bool = True,
                       extra_loop_edges: Optional[List[dict]] = None,
                       use_structural: bool = True,
                       log: Callable[[str], None] = print, session=None,
                       use_fork_edges: bool = True) -> dict:
    """Solve the keyframe graph on the session; apply as an epoch when the
    gates pass (or record identity). Returns the report (also written to
    output/keyframe_graph.json). ``session``: an in-memory CorrectionSession
    (the certification loop hands the state of the current iteration);
    ``use_fork_edges``: include the reconstruction's own keyframe loop edges
    (maplong_run/loop_edges.json)."""
    _vendor_on_path()
    from loop_utils.pose_graph import PoseGraph
    from loop_utils.lie import se3_inv, se3_log
    from correction.session import load_session
    cfg = cfg or load_loops_config()
    gc = cfg.graph
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    t0 = time.time()
    session = session if session is not None else load_session(output_dir)
    N = session.n_kf
    per_kf = _per_kf_points_fn(session)
    s_metric = _metric_scale_applied(output_dir)
    loops = (_loop_edges(output_dir, s_metric) if use_fork_edges else []) + list(extra_loop_edges or [])
    res_path = output_dir / "segmentation_result.json"
    instances = json.loads(res_path.read_text()).get("instances", []) if res_path.exists() else []
    classes = _classes(output_dir, instances, cfg.loops.semantic.default_class)

    # the world vertical for the axis constraints: the session is Y-up after
    # orient (floor at y=0); the consensus camera-down of the poses says which
    # way "down" is in THIS frame (only the structural axis edges use it)
    downs = session.poses[:, :3, 1]
    g_down = downs.mean(0); g_down = g_down / (np.linalg.norm(g_down) + 1e-12)
    up = -g_down

    T0 = session.poses.copy()
    gcfg = dict(sigma_odo_intra_m=gc.sigma_odo_intra_m, sigma_odo_intra_deg=gc.sigma_odo_intra_deg,
                loop_sigma_rot_deg=gc.loop_sigma_rot_deg,
                huber_delta_m=gc.huber_delta_m, huber_delta_deg=gc.huber_delta_deg,
                dense_max_unknowns=gc.dense_max_unknowns, lambda_init=gc.lambda_init,
                lambda_max=gc.lambda_max, lm_diag_floor=gc.lm_diag_floor, tol=gc.tol,
                rel_tol=gc.rel_tol, max_iters=gc.max_iters, pcg_tol=gc.pcg_tol,
                pcg_max_iters=gc.pcg_max_iters)

    def build():
        """Poses + PLANE nodes (walls) solved jointly."""
        pg = PoseGraph(T0, gcfg)
        for g in range(N - 1):
            Z = se3_inv(T0[g]) @ T0[g + 1]
            pg.add_relative(g, g + 1, Z, gc.sigma_odo_intra_deg, gc.sigma_odo_intra_m,
                            huber=False, tag="odo")
        loop_ids = []
        for e in loops:
            eid = pg.add_relative(int(e["i"]), int(e["j"]), np.asarray(e["Z"]),
                                  float(e["sigma_deg"]), float(e["sigma_m"]), huber=True,
                                  tag=f"loop:{e.get('bridge', -1)}",
                                  info_t=e.get("info_t"), info_rot=e.get("info_rot"))
            loop_ids.append((eid, e))
        srep = {}
        plane_nodes = {}
        if use_structural:
            sc = cfg.structural
            if sc.wall_planarity.enabled:
                we, wrep = st.wall_edges(session, instances, classes, sc.wall_planarity)
                by_wall = {}
                for k, n_l, d_l, n_t, d_t, wid in we:
                    by_wall.setdefault(wid, {"target": (n_t, d_t), "edges": []})["edges"].append((k, n_l, d_l))
                for wid, rec in by_wall.items():
                    n_t, d_t = rec["target"]
                    node = pg.add_plane_node(n_t, np.asarray(n_t) * float(d_t))
                    plane_nodes[f"wall:{wid}"] = node
                    for k, n_l, d_l in rec["edges"]:
                        # Huber width = the wall's own tolerance (a pilaster or
                        # a door frame in the patch stops pulling beyond it)
                        pg.add_plane_edge(k, node, n_l, d_l, sc.wall_planarity.sigma_angle_deg,
                                          sc.wall_planarity.sigma_offset_m, huber=True,
                                          tag=f"wall:{wid}",
                                          huber_delta_m=sc.wall_planarity.wall_tol_m)
                srep["wall_planarity"] = wrep
            if sc.column_vertical.enabled:
                ce, crep = st.axis_edges(session, instances, classes, sc.column_vertical, up, True)
                for k, a_l, tgt, iid in ce:
                    pg.add_axis_vertical(k, a_l, tgt, sc.column_vertical.sigma_deg, huber=True,
                                         tag=f"column:{iid}")
                srep["column_vertical"] = crep
            if sc.repeated_parallel.enabled:
                pe, prep = st.axis_edges(session, instances, classes, sc.repeated_parallel, up, False)
                for k, a_l, tgt, iid in pe:
                    pg.add_axis_vertical(k, a_l, tgt, sc.repeated_parallel.sigma_deg, huber=True,
                                         tag=f"parallel:{iid}")
                srep["repeated_parallel"] = prep
        return pg, loop_ids, srep, plane_nodes

    held = _holdout_pairs(session, per_kf, gc.holdout_offsets, gc.holdout_stride,
                          gc.holdout_samples, gc.holdout_max_nn_m)
    I = np.tile(np.eye(4), (N, 1, 1))
    held_before = _held_median(held, I)

    budget = cfg.loops.spatial
    centres = T0[:, :3, 3]
    pg, loop_ids, srep, plane_nodes = build()
    # residuals in the directions the edges OBSERVE (a floor-only closure has
    # no in-plane claim): the gain judges those, not fake zeros
    loop_before = float(np.sum([r["t_obs_m"] for r in pg.edge_residuals("loop").values()]))
    pg.solve(log=log)
    Xc = pg.corrections()[:N]
    # Drift budget δ(L) = max(floor, rate·L) per loop: ADVISORY. USER
    # 2026-09-09 ("siempre debe aplicarse la corrección de duplicados, no
    # importa lo mucho que haya que corregir") and pccr 2026-09-13 21:00: the
    # four measured start↔end closures (36–57 cm over an 18 m walk, Omega
    # drifting 3 cm/m against the 1.3 cm/m the budget assumed) were vetoed
    # one by one and the duplicates stayed. A closure is a MEASUREMENT (joint
    # ICP of the two copies, per-DOF information); the budget is a prior on
    # how much Omega usually drifts. When they disagree the measurement
    # stands and the acta/attention list says by how much.
    edge_res = pg.edge_residuals("loop")
    over_budget = []
    for eid, e in loop_ids:
        i, j = int(e["i"]), int(e["j"])
        lo, hi = min(i, j), max(i, j)
        L = float(np.linalg.norm(np.diff(centres[lo:hi + 1], axis=0), axis=1).sum())
        delta = max(budget.drift_floor_m, budget.drift_rate_m_per_m * L)
        need = max(float(np.linalg.norm(Xc[i][:3, 3])), float(np.linalg.norm(Xc[j][:3, 3])),
                   float(edge_res.get(eid, {}).get("t_obs_m", 0.0)))
        if need > delta:
            over_budget.append({"i": i, "j": j, "correction_m": need, "budget_m": delta,
                                "walk_m": L, "bridge": e.get("bridge", -1),
                                "reason": "closure beyond the drift budget (§4.7) — applied, declared"})
            log(f"[kf-graph] loop {i}<->{j}: {need * 100:.0f} cm > drift budget {delta * 100:.0f} cm "
                f"(walk {L:.1f} m) — the measured closure stands, declared in the acta")
    planes_solved = {name: {"normal": pg.plane_of(node)[0].tolist(), "offset_m": pg.plane_of(node)[1]}
                     for name, node in plane_nodes.items()}
    after = pg.edge_residuals("loop")
    loop_after = float(np.sum([r["t_obs_m"] for r in after.values()]))
    n_active = sum(1 for eid, _ in loop_ids if pg._edges[eid]["active"])
    held_after = _held_median(held, Xc)
    # the gain against what the edges can deliver: no edge closes better than
    # its own σ (a post-hoc region closure carries its ICP residual), so the
    # residual floor Σσ is subtracted from both sides — a 30 cm drift closed
    # to the 5 cm the closures resolve is a full gain, not a half one
    sigma_floor = float(np.sum([float(e["sigma_m"]) for eid, e in loop_ids if pg._edges[eid]["active"]]))
    if loop_before <= sigma_floor:
        gain = 1.0 if loop_after <= loop_before else 0.0     # already within the edges' σ
    else:
        gain = 1.0 - max(loop_after - sigma_floor, 0.0) / (loop_before - sigma_floor)
    ok_gain = (gain >= gc.min_loop_gain) if n_active > 0 else (use_structural and bool(srep))
    ok_held = (not np.isfinite(held_before)) or (held_after <= held_before + gc.max_seam_degradation_m)
    t_mag = np.linalg.norm(Xc[:, :3, 3], axis=1)
    r_mag = np.array([np.degrees(np.linalg.norm(se3_log(M)[:3])) for M in Xc])
    frac = max(float(t_mag.max()) / cfg.authority.pose_graph_max_m,
               float(r_mag.max()) / cfg.authority.pose_graph_max_deg)
    saturated = frac > cfg.authority.saturation_warn
    # The verdict (USER 2026-09-13, pccr: "nunca debe descartarse un
    # duplicado detectado ... aunque haya que corregir 1000 km"): a measured
    # closure IS applied. The gates below (loop gain, held-out pairs,
    # authority) are MEASURED and declared — in ``advisory`` mode they are
    # warnings in the report/acta and the visual Approve/Undo is the verdict;
    # in ``veto`` mode (evaluation only) any failed gate keeps identity. The
    # pccr run of 2026-09-13 19:25 vetoed four real start↔end closures
    # (36–57 cm) and then went IDENTITY on held-out (2.7→8.6 cm) + authority
    # (319 %): the duplicates stayed. IDENTITY now only when there is
    # nothing to close: no active loop edge and no structural constraint.
    has_evidence = n_active > 0 or (use_structural and bool(srep))
    gate_warnings = []
    if not ok_gain:
        gate_warnings.append(f"loop gain {gain * 100:.0f}% < {gc.min_loop_gain * 100:.0f}%")
    if not ok_held:
        gate_warnings.append(f"held-out {held_before * 100:.2f}→{held_after * 100:.2f} cm "
                             f"(> +{gc.max_seam_degradation_m * 100:.1f} cm)")
    if frac > 1.0:
        gate_warnings.append(f"authority {frac * 100:.0f}% (max {cfg.authority.pose_graph_max_m} m / "
                             f"{cfg.authority.pose_graph_max_deg}°)")
    if gc.gate_mode == "veto":
        verdict = "APPLY" if (has_evidence and not gate_warnings) else "IDENTITY"
    else:
        verdict = "APPLY" if has_evidence else "IDENTITY"
    for w in gate_warnings:
        log(f"[kf-graph] ⚠ gate: {w} — {'declared, correction applied' if gc.gate_mode == 'advisory' else 'veto'}")
    struct_res = {tag: v for tag, v in pg.edge_residuals().items()
                  if not str(v.get("tag", "")).startswith(("odo", "loop"))}
    cov = st.loop_coverage(centres, [(e["i"], e["j"]) for eid, e in loop_ids
                                     if pg._edges[eid]["active"]],
                           cfg.loops.coverage_radius_m)
    report = {"version": 1, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "verdict": verdict, "operator": operator, "n_kf": N,
              "n_loop_edges": len(loop_ids), "n_loop_edges_active": n_active,
              "s_metric_applied": s_metric,
              "gate_mode": gc.gate_mode, "gate_warnings": gate_warnings,
              "identity_reason": (None if verdict == "APPLY" else
                                  ("no active loop edge and no structural constraint — nothing to close"
                                   if not has_evidence else "; ".join(gate_warnings))),
              "gates": {"loop_gain": {"value": gain, "min": gc.min_loop_gain, "passed": ok_gain,
                                      "loop_residual_before_m": loop_before,
                                      "loop_residual_after_m": loop_after,
                                      "sigma_floor_m": sigma_floor},
                        "holdout_pairs": {"n_pairs": len(held), "median_before_m": held_before,
                                          "median_after_m": held_after,
                                          "max_degradation_m": gc.max_seam_degradation_m,
                                          "passed": ok_held},
                        "authority": {"fraction_used": frac, "saturated": bool(saturated),
                                      "exceeded": bool(frac > 1.0),
                                      "max_m": cfg.authority.pose_graph_max_m,
                                      "max_deg": cfg.authority.pose_graph_max_deg,
                                      "used_max_m": float(t_mag.max()),
                                      "used_max_deg": float(r_mag.max())}},
              "over_budget": over_budget,
              "structural": srep, "structural_residuals_n": len(struct_res),
              "planes_solved": planes_solved,
              "loop_coverage": cov,
              "coverage_ok": cov["coverage"] >= cfg.loops.min_coverage,
              "solver": {k: v for k, v in pg.report.items() if k != "per_edge"},
              "xi": [se3_log(M).tolist() for M in Xc] if verdict == "APPLY" else None,
              "elapsed_s": round(time.time() - t0, 1),
              "provenance": "tool_measured"}
    log(f"[kf-graph] loop {loop_before * 100:.1f}→{loop_after * 100:.1f} cm (gain {gain * 100:.0f}%) "
        f"| held-out {held_before * 100:.2f}→{held_after * 100:.2f} cm | authority {frac * 100:.0f}% "
        f"| coverage {cov['coverage'] * 100:.0f}% → {verdict}")
    if verdict == "APPLY" and apply:
        from correction.apply import stage_transaction, swap_transaction, assert_no_interrupted_swap
        from correction.config import load_correction_config
        from correction import ledger, epoch as epoch_mod
        from correction.invalidate import update_instance_store
        assert_no_interrupted_swap(output_dir)
        if ledger.pending_run(output_dir):
            raise RuntimeError("a previous correction is still pending Approve/Undo — resolve it "
                               "before applying the keyframe graph")
        ccfg = load_correction_config()
        cid = ledger.new_correction_id()
        R_kf = Xc[:, :3, :3].copy()
        t_kf = Xc[:, :3, 3].copy()
        k_kf = np.ones(N)
        tx = stage_transaction(session, ccfg, R_kf, t_kf, k_kf, correction_id=cid,
                               scale_diag_new=None, floor_npz=None, log=log, progress=None)
        swap_transaction(output_dir, tx, log=log)
        update_instance_store(output_dir, R_kf, t_kf, k_kf, session.frames, log=log)
        rep_path = output_dir / "corrections" / f"report_{cid}.json"
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=1, default=float))
        ledger.record_run(output_dir, correction_id=cid, epoch_from=tx["epoch_from"],
                          epoch_to=tx["epoch_to"], kind="keyframe_graph", operator=operator,
                          instance_ids=[], visits=[], observability=[], diagnosis=[],
                          anchors=[{"kf": int(e["i"]), "kf2": int(e["j"]),
                                    "sigma_m": float(e["sigma_m"])} for _, e in loop_ids],
                          gates=[{"name": k, **v} for k, v in report["gates"].items()],
                          overrides={}, report_path=str(rep_path.relative_to(output_dir)),
                          verdict="pending")
        report["correction_id"] = cid
        report["epoch"] = tx["epoch_to"]
    (output_dir / REPORT_JSON).write_text(json.dumps(report, indent=1, default=float))
    return report
