"""Post-hoc keyframe SE(3) graph over a reconstructed SESSION (§4.3 + §4.6 + §8).

The fork runs the graph once inside the reconstruction (loops from SALAD);
this runner re-solves it on the finished session with everything the
reconstruction could not know yet — the SAM3 instance loops (§4.4), the
structural constraints (§4.6), the measured chain uncertainty — and applies
the result through the correction package's transactional machinery: an
EPOCH (geometry_epoch.json), the ledger, ``corrections/epoch_<N>.npz``,
Approve/Undo in the UI. Nothing here is silent: every gate lands in the
report with value / threshold / verdict; identity is a verdict too.

Inputs: cleaned_cloud.ply (+ provenance), camera_poses.txt, camera_frames.txt,
segmentation_result.json (structural edges), maplong_run/loop_edges.json
(keyframe loop edges, translations rescaled by the residual metric s),
uncertainty.json (σ per frame), loop_semantics.json / the instance store
(classes). Solver: vendor/VGGT-Long/loop_utils/pose_graph.py.
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


def _uncertainty(output_dir: Path, frames: List[int]):
    p = output_dir / "maplong_run" / "uncertainty.json"
    if not p.exists():
        return {}, 0.0
    rep = json.loads(p.read_text())
    by_frame = {int(v["frame"]): float(v["median_m"]) for v in rep.get("frames", {}).values()
                if v.get("median_m") is not None}
    per_kf = {k: by_frame[f] for k, f in enumerate(frames) if f in by_frame}
    return per_kf, float(rep.get("session_median_m", 0.0) or 0.0)


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
                       log: Callable[[str], None] = print) -> dict:
    """Solve the keyframe graph on the session; apply as an epoch when the
    gates pass (or record identity). Returns the report (also written to
    output/keyframe_graph.json)."""
    _vendor_on_path()
    from loop_utils.pose_graph import PoseGraph
    from loop_utils.lie import se3_inv, se3_log
    from correction.session import load_session
    cfg = cfg or load_loops_config()
    gc = cfg.graph
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    t0 = time.time()
    session = load_session(output_dir)
    N = session.n_kf
    per_kf = _per_kf_points_fn(session)
    s_metric = _metric_scale_applied(output_dir)
    loops = _loop_edges(output_dir, s_metric) + list(extra_loop_edges or [])
    unc, unc_med = _uncertainty(output_dir, session.frames)
    res_path = output_dir / "segmentation_result.json"
    instances = json.loads(res_path.read_text()).get("instances", []) if res_path.exists() else []
    classes = _classes(output_dir, instances, cfg.loops.semantic.default_class)

    # gravity: the session is Y-up after orient (floor at y=0); the consensus
    # camera-down of the poses says which way "down" is in THIS frame
    downs = session.poses[:, :3, 1]
    g_down = downs.mean(0); g_down = g_down / (np.linalg.norm(g_down) + 1e-12)
    up = -g_down

    T0 = session.poses.copy()
    gcfg = dict(sigma_odo_intra_m=gc.sigma_odo_intra_m, sigma_odo_intra_deg=gc.sigma_odo_intra_deg,
                loop_sigma_rot_deg=gc.loop_sigma_rot_deg, sigma_gravity_deg=gc.sigma_gravity_deg,
                huber_delta_m=gc.huber_delta_m, huber_delta_deg=gc.huber_delta_deg,
                dense_max_unknowns=gc.dense_max_unknowns, lambda_init=gc.lambda_init,
                lambda_max=gc.lambda_max, lm_diag_floor=gc.lm_diag_floor, tol=gc.tol,
                rel_tol=gc.rel_tol, max_iters=gc.max_iters, pcg_tol=gc.pcg_tol,
                pcg_max_iters=gc.pcg_max_iters)

    def build():
        """Poses + PLANE nodes (walls, floor datum) solved jointly."""
        pg = PoseGraph(T0, gcfg)
        for g in range(N - 1):
            Z = se3_inv(T0[g]) @ T0[g + 1]
            s_t = float(np.sqrt(gc.sigma_odo_intra_m ** 2 + unc.get(g, unc_med) ** 2
                                + unc.get(g + 1, unc_med) ** 2))
            pg.add_relative(g, g + 1, Z, gc.sigma_odo_intra_deg, s_t, huber=False, tag="odo")
        loop_ids = []
        for e in loops:
            eid = pg.add_relative(int(e["i"]), int(e["j"]), np.asarray(e["Z"]),
                                  float(e["sigma_deg"]), float(e["sigma_m"]), huber=True,
                                  tag=f"loop:{e.get('bridge', -1)}")
            loop_ids.append((eid, e))
        for g in range(N):
            pg.add_gravity(g, g_down, gc.sigma_gravity_deg)
        srep = {}
        plane_nodes = {}
        if use_structural:
            sc = cfg.structural
            if sc.floor_datum.enabled:
                fe, frep = st.floor_datum_edges(session, per_kf, up, sc.floor_datum)
                if fe:
                    n_star, d_star = np.asarray(frep["datum_normal"]), float(frep["datum_offset_m"])
                    node = pg.add_plane_node(n_star, n_star * d_star)
                    plane_nodes["floor"] = node
                    for k, n_l, d_l, _n_t, _d_t in fe:
                        pg.add_plane_edge(k, node, n_l, d_l, sc.floor_datum.sigma_angle_deg,
                                          sc.floor_datum.sigma_offset_m, huber=True, tag="floor")
                srep["floor_datum"] = frep
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

    vetoed = []
    budget = cfg.loops.spatial
    centres = T0[:, :3, 3]
    pg, loop_ids, srep, plane_nodes = build()
    loop_before = float(np.sum([r["t_m"] for r in pg.edge_residuals("loop").values()]))
    while True:
        pg.solve(log=log)
        Xc_all = pg.corrections()
        Xc = Xc_all[:N]
        edge_res = pg.edge_residuals("loop")
        offenders = []
        for eid, e in loop_ids:
            if not pg._edges[eid]["active"]:
                continue
            i, j = int(e["i"]), int(e["j"])
            lo, hi = min(i, j), max(i, j)
            L = float(np.linalg.norm(np.diff(centres[lo:hi + 1], axis=0), axis=1).sum())
            delta = max(budget.drift_floor_m, budget.drift_rate_m_per_m * L)
            # what the edge DEMANDS: obtained correction + what it still asks for
            need = max(float(np.linalg.norm(Xc[i][:3, 3])), float(np.linalg.norm(Xc[j][:3, 3])),
                       float(edge_res.get(eid, {}).get("t_m", 0.0)))
            if need > delta:
                offenders.append((need - delta, eid, e, need, delta))
        if not offenders:
            break
        offenders.sort(key=lambda x: x[0], reverse=True)
        _, eid, e, need, delta = offenders[0]
        pg.deactivate(eid)
        vetoed.append({"edge": e, "i": e["i"], "j": e["j"], "correction_m": need,
                       "budget_m": delta, "bridge": e.get("bridge", -1),
                       "reason": "loop edge demands a correction beyond the drift budget (§4.7)"})
        log(f"[kf-graph] VETO loop {e['i']}<->{e['j']}: {need * 100:.0f} cm > budget "
            f"{delta * 100:.0f} cm — re-solving")
    Xc = pg.corrections()[:N]
    planes_solved = {name: {"normal": pg.plane_of(node)[0].tolist(), "offset_m": pg.plane_of(node)[1]}
                     for name, node in plane_nodes.items()}
    after = pg.edge_residuals("loop")
    loop_after = float(np.sum([r["t_m"] for r in after.values()]))
    n_active = sum(1 for eid, _ in loop_ids if pg._edges[eid]["active"])
    held_after = _held_median(held, Xc)
    gain = (1.0 - loop_after / loop_before) if loop_before > 0 else 0.0
    ok_gain = (gain >= gc.min_loop_gain) if n_active > 0 else (use_structural and bool(srep))
    ok_held = (not np.isfinite(held_before)) or (held_after <= held_before + gc.max_seam_degradation_m)
    t_mag = np.linalg.norm(Xc[:, :3, 3], axis=1)
    r_mag = np.array([np.degrees(np.linalg.norm(se3_log(M)[:3])) for M in Xc])
    frac = max(float(t_mag.max()) / cfg.authority.pose_graph_max_m,
               float(r_mag.max()) / cfg.authority.pose_graph_max_deg)
    saturated = frac > cfg.authority.saturation_warn
    verdict = "APPLY" if (ok_gain and ok_held and frac <= 1.0) else "IDENTITY"
    struct_res = {tag: v for tag, v in pg.edge_residuals().items()
                  if not str(v.get("tag", "")).startswith(("odo", "loop", "gravity"))}
    cov = st.loop_coverage(centres, [(e["i"], e["j"]) for eid, e in loop_ids
                                     if pg._edges[eid]["active"]],
                           cfg.loops.coverage_radius_m)
    report = {"version": 1, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "verdict": verdict, "operator": operator, "n_kf": N,
              "n_loop_edges": len(loop_ids), "n_loop_edges_active": n_active,
              "s_metric_applied": s_metric,
              "gates": {"loop_gain": {"value": gain, "min": gc.min_loop_gain, "passed": ok_gain,
                                      "loop_residual_before_m": loop_before,
                                      "loop_residual_after_m": loop_after},
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
              "vetoed": [{k: v for k, v in e.items() if k != "edge"} for e in vetoed],
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
        try:
            update_instance_store(output_dir, session, R_kf, t_kf, k_kf, log=log)
        except TypeError:
            update_instance_store(output_dir, R_kf, t_kf, k_kf)
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
