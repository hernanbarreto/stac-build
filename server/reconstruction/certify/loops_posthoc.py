"""Loop measurement on the finished cloud (§4.4 doctrine, post-hoc).

Two sources, two products:

  * VISIT LOOPS → pose-graph edges. The correction package's geometric
    revisit detection (``correction.revisit``, USER 2026-09-09: "ver los
    mismos lugares desde diferentes posiciones") finds every place two
    visits both wrote — occlusion-verified co-visibility → REGIONS, each
    with its own rigid closure (trimmed yaw+t ICP on what both visits saw
    there) and its own observability. Every region becomes an edge
    Z_ij = T_i⁻¹ X⁻¹ T_j between the central keyframes of the two visits
    that saw it (X = later → earlier, world), with a per-DOF information
    matrix from the region's shape (a floor observes its normal, a column
    the plane across its axis, a compact block everything; yaw only when
    the block is anisotropic): unobserved directions enter with a huge σ,
    never as a fake zero. The module's single joint closure per visit pair
    is a rigid compromise meant for a manual correction — drift within a
    visit is not rigid, and the graph wants the local evidence. Measured on
    partial pieces one object at a time this was wrong too (certify smoke:
    half a column aligned onto the other half demanded 1 m).

  * INSTANCE COPIES → scale rows (§5.1). A SAM3 instance seen in two visits
    (the F1 detector's temporal / duplicate candidates) is the same object
    twice; a Sim3 fit between its copies (rigid ICP → nearest neighbours →
    weighted Umeyama with Cauchy IRLS) measures ``s_ab``, trusted only when
    the object is compact and the copies cover each other (a wall patch has
    no scale to give).

Nothing here decides: every record carries its residual, its observability
and whether it is trusted; the scale graph, the pose graph and the gates
weigh it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _vendor_on_path():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


# ── copies of one instance → scale rows ─────────────────────────────────────

def _kabsch(P: np.ndarray, Q: np.ndarray):
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def rigid_icp(src: np.ndarray, dst: np.ndarray, iters: int, trim: float):
    """Trimmed point-to-point ICP src → dst (full SE(3)). Returns (R, t,
    trimmed rms)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    R = np.eye(3); t = np.zeros(3)
    S = src.copy()
    rms = float("inf")
    for _ in range(int(iters)):
        d, j = tree.query(S, k=1)
        k = max(10, int(len(S) * float(trim)))
        sel = np.argsort(d)[:k]
        Ri, ti = _kabsch(S[sel], dst[j[sel]])
        S = S @ Ri.T + ti
        R = Ri @ R
        t = Ri @ t + ti
        rms = float(np.sqrt((np.linalg.norm(S[sel] - dst[j[sel]], axis=1) ** 2).mean()))
        if np.degrees(np.arccos(np.clip((np.trace(Ri) - 1) / 2, -1, 1))) < 1e-3 and np.linalg.norm(ti) < 1e-5:
            break
    return R, t, rms


# the degrees of freedom each candidate fit spends: the rigid Sim3 fits a
# rotation as well, the other two only move the copy. The docstring of the
# comparison below has promised since 2026-09-17 that "a tie goes to the fewer
# degrees of freedom"; this is the number that makes it true.
_FIT_DOF = {"rotated": 6, "translated": 3, "centroid": 3}


def _rot_deg(R) -> float:
    """Rotation angle of R in degrees."""
    A = np.asarray(R, np.float64)
    return float(np.degrees(np.arccos(np.clip((np.trace(A) - 1.0) / 2.0, -1.0, 1.0))))


def _over_budget(R, t, budget: Optional[Dict[str, float]]) -> Optional[float]:
    """The motion a fit demands, in units of the drift the WALK between the two
    visits can produce: max(|t| / δ(L), rot / θ(L)). 1.0 is the budget itself.
    pccr 2026-09-21, epoch 1: the rigid fit of `desk#198` (107.7°, 8.76 m over
    19.3 m walked, δ 0.30 m / θ 2.0°) reads 53.9, while the 74.1 cm its two
    copies actually stood apart reads 2.5 — the two sit on opposite sides of
    everything the walk could explain. None when no budget was handed in:
    then nothing is judged on motion."""
    if budget is None:
        return None
    d = max(float(budget["delta_m"]), 1e-9)
    th = max(float(budget["theta_deg"]), 1e-9)
    return float(max(float(np.linalg.norm(np.asarray(t, np.float64))) / d,
                     _rot_deg(R) / th))


def _pair_budget(session, i: int, j: int, spatial_cfg):
    """δ(L), θ(L) for one candidate pair — ``spatial_gate.drift_budget`` over
    the metres WALKED between keyframes i and j along the current camera
    centres. Returns (budget, why): the budget is None, and the reason travels
    into the record, when the spatial block handed in states no drift rate
    (``copy_scale_rows`` takes its spatial config optionally, and a caller may
    hand only the same-surface keys the overlap refusal needs). A bound nobody
    stated is not a bound to invent here."""
    from reconstruction.loops import spatial_gate as sg
    if spatial_cfg is None:
        return None, "no spatial config: the drift budget was never stated"
    c = sg._cfg(spatial_cfg)
    missing = [k for k in ("drift_floor_m", "drift_rate_m_per_m",
                           "drift_floor_deg", "drift_rate_deg_per_m") if k not in c]
    if missing:
        return None, ("the spatial config states no drift rate "
                      f"({', '.join(missing)}) — the fit is judged on quality alone")
    L = sg.walked_length_m(np.asarray(session.poses, np.float64)[:, :3, 3], int(i), int(j))
    return sg.drift_budget(L, c), None


def measure_copy(pts_a: np.ndarray, pts_b: np.ndarray, scfg, seed: int = 0,
                 max_points: int = 20000,
                 budget: Optional[Dict[str, float]] = None) -> Optional[dict]:
    """Sim3 of copy A onto copy B. None when starved; else {"s_ab",
    "residual_m", "n", "coverage_a", "coverage_b", "trusted", "scale_trusted",
    "R", "t", "centroid_a", "offset_before_m", "offset_after_m"} — (R, t) is
    the RIGID closure of A onto B (world, p_b = R·p_a + t), exact at A's
    centroid: the Sim3 fit p_b = s·R·p_a + t_s and the rigid map agree there
    when t = t_s + (s − 1)·R·c_a. The scale goes to the scale graph as its
    own row; the pose graph takes the rigid part (``instance_edges``).

    ``budget`` — δ(L), θ(L) from ``spatial_gate.drift_budget`` over the metres
    WALKED between the two visits, built by ``_pair_budget`` — is what the
    chosen fit's MOTION is judged against. Without it the fit is chosen on
    quality alone, the way it was before 2026-09-21, and the record says the
    budget was unavailable instead of pretending one was applied."""
    _vendor_on_path()
    from loop_utils.loop_bridges import robust_sim3
    from scipy.spatial import cKDTree
    a = np.asarray(pts_a, np.float64); b = np.asarray(pts_b, np.float64)
    if len(a) < int(scfg.min_copy_points) or len(b) < int(scfg.min_copy_points):
        return None
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]
    R0, t0, rms = rigid_icp(a, b, scfg.icp_iters, scfg.icp_trim)
    a2 = a @ R0.T + t0
    tree_b = cKDTree(b)
    d_ab, j = tree_b.query(a2, k=1)
    thr = max(2.0 * float(np.median(d_ab)), 1e-3)
    ok = d_ab <= thr
    cov_a = float(ok.mean())
    d_ba, _ = cKDTree(a2).query(b, k=1)
    cov_b = float((d_ba <= thr).mean())
    if ok.sum() < int(scfg.min_copy_points):
        return None
    fit = robust_sim3(a[ok], b[j[ok]], min_points=int(scfg.min_copy_points))
    if fit is None:
        return None
    s, R_s, t_s, med, n = fit
    c_a = a.mean(0)
    R_s = np.asarray(R_s, np.float64)
    t_r = np.asarray(t_s, np.float64) + (float(s) - 1.0) * (R_s @ c_a)
    d_before, _ = tree_b.query(a, k=1)

    # ── the rotation has to EARN its degrees of freedom ──────────────────
    # Six DOF fitted to two copies of a near-symmetric object land on a
    # rotation that buys nothing and costs everything. pccr 2026-09-17,
    # measured on the copies themselves: desk#201 closes to 2.6 cm with a
    # 63.9 deg rotation and |t| = 5.44 m, and to THE SAME 2.6 cm with a pure
    # translation of 66.8 cm — the size of the duplicate the user can see and
    # of which he says "no parece haber rotación". The floor is worse: 77.9
    # deg leaves 30.0 cm where a pure translation leaves 19.0.
    #
    # That invented rotation is where the rest of the day's trouble came
    # from: |t| of metres, drift rates of 100 cm/m, a discrepancy of 30 m for
    # the graph to absorb, and the observability projection throwing away part
    # of the REAL translation to compensate for it — the sideways residual the
    # user sees after the desks close.
    #
    # No threshold: both fits are measured against the same points and the
    # rotation is kept only when it is strictly better. A tie goes to the
    # fewer degrees of freedom.
    t_only = _translation_fit(a, b, tree_b, scfg)
    t_cen = b.mean(0) - c_a            # the displacement the two BODIES demand
    rot_deg = _rot_deg(R_s)

    # ── the separation is measured where it can be SEEN ──────────────────
    # Nearest-neighbour distance cannot see a copy slid ALONG its own
    # surface: two desks side by side overlapping 50% have near neighbours
    # everywhere they overlap. pccr 2026-09-17, desk#201: the loop reported
    # "copies 60.4 -> 2.7 cm" and declared the duplicate closed while the two
    # centroids stood 52.6 cm apart — exactly the "duplicado al costado con
    # superposición del 50%" the user was looking at. Between two of its own
    # epochs the desk slid 15 cm FURTHER apart and the measure read that as
    # 2.7 -> 4.2 cm, i.e. nothing.
    #
    # So the separation of two copies is the WORSE of what the surfaces say
    # and what the bodies say. The surface term catches a gap, the centroid
    # term catches a slide, and neither can hide the other. The centroid term
    # carries the bias of unequal coverage (pccr desk#201: extents 1.30 vs
    # 1.47 m, so up to ~17 cm of it) — declared here, and small against the
    # 52 cm it exposes.
    def _sep(P):
        d, _ = tree_b.query(P, k=1)
        return float(np.median(d)), float(np.linalg.norm(P.mean(0) - b.mean(0)))

    nn_before, cen_before = _sep(a)
    cands = [("rotated", R_s, t_r), ("translated", np.eye(3), t_only),
             ("centroid", np.eye(3), t_cen)]
    scored = []
    for name, R_c, t_c in cands:
        R_c = np.asarray(R_c, np.float64); t_c = np.asarray(t_c, np.float64)
        nn, cen = _sep(a @ R_c.T + t_c)
        scored.append({"name": name, "R": R_c, "t": t_c, "dof": _FIT_DOF[name],
                       "score": float(max(nn, cen)), "nn": nn, "cen": cen,
                       "rot_deg": _rot_deg(R_c),
                       "t_norm_m": float(np.linalg.norm(t_c)),
                       "over_budget": _over_budget(R_c, t_c, budget)})
    # the tie the paragraph above promises is now actually broken that way: the
    # DOF is IN the sort key, where a stable sort used to hand every tie to
    # `rotated` for being first in the list
    scored.sort(key=lambda r: (r["score"], r["dof"]))
    best = scored[0]

    # ── and the motion has to be DRIFT-SHAPED ────────────────────────────
    # Everything above compares QUALITY OF FIT. Nothing in it compares
    # PLAUSIBILITY OF MOTION, and a rigid fit between two DIFFERENT objects
    # filed under one label "closes" by putting one on top of the other. pccr
    # 2026-09-21, epoch 1: `desk#198` kf 204<->2 — TWO desks in one masklet —
    # fitted 107.7° and 8.76 m of translation over a 19.3 m walk, took the
    # copies 74.1 → 11.2 cm and won outright; there was no tie to break. It
    # entered the pose graph at σ 67.0 cm. The paragraph above records the
    # same pathology twice over (desk#201 at 63.9° / 5.44 m, floor#44 at
    # 77.9°) and both times it was caught by a BETTER fit happening to exist,
    # never by the motion being impossible.
    #
    # The session already knows what motion is plausible: δ(L), θ(L) of
    # `spatial_gate.drift_budget` over the metres WALKED between the two
    # visits — the same "what we would still call drift" bound the identity
    # gate spends. A fit that exceeds it does not get to win by fitting
    # better: it is DEMOTED to the best-scoring fit that is BOTH more
    # parsimonious and smaller in motion, and the refusal travels with both
    # numbers (rotation and translation, against θ and δ).
    #
    # Nothing is vetoed (USER 2026-09-16: *"no debe cortar objetos"*, and a
    # duplicate SAM3 detected is never discarded): the pair still produces its
    # edge, now from a fit that is physically possible. And because the
    # demotion demands a candidate that is simpler AND moves less, a copy that
    # really did turn — the one case where the rotation buys motion a
    # translation cannot — keeps its rotation.
    demoted = None
    if best["over_budget"] is not None and best["over_budget"] > 1.0:
        alt = [r for r in scored if r["dof"] < best["dof"]
               and r["over_budget"] < best["over_budget"]]
        if alt:
            demoted = {
                "refused": best["name"], "chosen": alt[0]["name"],
                "refused_rot_deg": best["rot_deg"], "refused_t_norm_m": best["t_norm_m"],
                "refused_over_budget": best["over_budget"], "refused_score_m": best["score"],
                "chosen_rot_deg": alt[0]["rot_deg"], "chosen_t_norm_m": alt[0]["t_norm_m"],
                "chosen_over_budget": alt[0]["over_budget"], "chosen_score_m": alt[0]["score"],
                "walk_m": float(budget["L_m"]), "delta_m": float(budget["delta_m"]),
                "theta_deg": float(budget["theta_deg"]), "source": budget.get("source"),
                "why": (f"the {best['name']} fit demands {best['rot_deg']:.1f}° / "
                        f"{best['t_norm_m']:.2f} m where drift over {float(budget['L_m']):.1f} m "
                        f"walked produces at most {float(budget['theta_deg']):.1f}° / "
                        f"{float(budget['delta_m']):.2f} m ({best['over_budget']:.1f}× the "
                        f"budget) — refused although it fits better "
                        f"({best['score'] * 100:.1f} vs {alt[0]['score'] * 100:.1f} cm); "
                        f"the {alt[0]['name']} fit ({alt[0]['rot_deg']:.1f}° / "
                        f"{alt[0]['t_norm_m']:.2f} m, {alt[0]['over_budget']:.1f}×) is "
                        f"applied instead")}
            best = alt[0]
    sep_after, nn_after, cen_after = best["score"], best["nn"], best["cen"]
    chosen, R_s, t_r = best["name"], best["R"], best["t"]
    rotation_earned = chosen == "rotated"
    med = nn_after
    trusted = bool(sep_after <= float(scfg.max_copy_residual_m))
    return {"s_ab": float(s) if rotation_earned else 1.0,
            "residual_m": float(med), "n": int(n), "coverage_a": cov_a,
            "coverage_b": cov_b, "rigid_rms": float(rms), "trusted": trusted,
            "scale_trusted": bool(rotation_earned and trusted and cov_a >= 0.8 and cov_b >= 0.8),
            "R": R_s.tolist(), "t": t_r.tolist(), "centroid_a": c_a.tolist(),
            "rotation_earned": rotation_earned, "rot_deg_fitted": rot_deg,
            "fit_chosen": chosen,
            "fit_scores_m": {r["name"]: round(r["score"], 4) for r in scored},
            # what each fit would have MOVED, and by how much it exceeds the
            # drift of the walk — the numbers a demotion is argued from
            "fit_motion": {r["name"]: {"rot_deg": round(r["rot_deg"], 2),
                                       "t_norm_m": round(r["t_norm_m"], 4),
                                       "over_budget": (None if r["over_budget"] is None
                                                       else round(r["over_budget"], 2))}
                           for r in scored},
            "drift_budget": (dict(budget) if budget is not None else None),
            "fit_demoted": demoted,
            "nn_before_m": nn_before, "nn_after_m": nn_after,
            "centroid_before_m": cen_before, "centroid_after_m": cen_after,
            # what everything downstream decides on: the separation that can be
            # SEEN, not the one that hides a slide
            "offset_before_m": float(max(nn_before, cen_before)),
            "offset_after_m": float(sep_after)}


def _translation_fit(a: np.ndarray, b: np.ndarray, tree_b, scfg) -> np.ndarray:
    """The best PURE translation taking copy A onto copy B — the same trimmed
    nearest-neighbour loop the rigid fit uses, with the rotation held out."""
    t = np.zeros(3, np.float64)
    trim = float(scfg.icp_trim)
    for _ in range(int(scfg.icp_iters)):
        p = a + t
        d, j = tree_b.query(p, k=1)
        k = max(int(trim * len(d)), 3)
        idx = np.argpartition(d, k - 1)[:k]
        t = t + (b[j[idx]] - p[idx]).mean(0)
    return t


def _copy_indices(session, inst: dict, i: int, j: int, window_kf: int):
    """Point indices of the two copies of an instance: around keyframe i
    (copy A, the later visit) and around keyframe j (copy B, the earlier)."""
    gi = np.asarray(inst.get("globalIndices") or [], np.int64)
    gi = gi[(gi >= 0) & (gi < session.n_points)]
    ks = session.ks[gi]
    return gi[np.abs(ks - i) <= int(window_kf)], gi[np.abs(ks - j) <= int(window_kf)], ks


def copy_scale_rows(session, candidates: List[dict], scfg, window_kf: int,
                    max_pairs_per_instance: int, log=print,
                    spatial_cfg=None) -> List[dict]:
    """Scale measurements from the copies of every revisited instance
    (candidates with verdict loop|ambiguous and a temporal gap).

    ``spatial_cfg`` (the loops' spatial block) lets the stage refuse a pair
    that is two PARTS of one surface BEFORE paying for its ICP — see
    ``surface_overlap`` — and states the DRIFT BUDGET the chosen fit's motion
    is judged against (``_pair_budget``): a Sim3 that closes two different
    objects by turning one 107.7° writes a scale row out of a rotation that
    never happened. Without it the pair is still measured and still priced out
    by ``scale_trusted``; the refusal only saves the work."""
    res_path = Path(session.output_dir) / "segmentation_result.json"
    instances = {int(i.get("instance_id", i.get("id"))): i
                 for i in (json.loads(res_path.read_text()).get("instances") or [])} if res_path.exists() else {}
    out = []
    # a pair is (instance, i, j) regardless of order; an object may not spend
    # the budget of the whole stage on itself
    _seen_pairs: set = set()
    _n_of: Dict[int, int] = {}

    for cand in candidates:
        if cand.get("verdict") not in ("loop", "ambiguous"):
            continue
        iid = int(cand["instance_id"])
        i, j = int(cand["i"]), int(cand["j"])
        # THE SAME PAIR IS NEVER MEASURED TWICE. The candidate list carries
        # repeats (pccr 2026-09-21: 7 of 56 evaluations were a pair already
        # done, with an identical residual to four decimals), and each one is
        # an ICP over the whole cloud.
        if (iid, min(i, j), max(i, j)) in _seen_pairs:
            continue
        _seen_pairs.add((iid, min(i, j), max(i, j)))
        # AND NO OBJECT IS MEASURED MORE THAN `max_pairs_per_instance` TIMES.
        # One extended surface generated 42 of 56 evaluations by pairing its
        # own clusters across every keyframe that saw it — n² growth that
        # yields the same (non-)information every time.
        _n_of[iid] = _n_of.get(iid, 0) + 1
        if _n_of[iid] > int(max_pairs_per_instance):
            continue
        rec = {"instance_id": iid, "label": cand.get("label"), "i": i, "j": j, "verdict": cand["verdict"],
               "kind": cand.get("kind")}
        inst = instances.get(iid)
        if inst is None:
            rec["reason"] = "instance no longer in the segmentation"
            out.append(rec); continue
        if abs(i - j) <= 2 * int(window_kf):
            rec["reason"] = "not a revisit (the two copies come from overlapping keyframe windows)"
            out.append(rec); continue
        idx_a, idx_b, _ks = _copy_indices(session, inst, i, j, window_kf)
        # Both sides must have something to align. It used to run the ICP with
        # one side EMPTY and discard the answer afterwards (pccr 2026-09-21:
        # four pairs at coverage 0.00) — the work is skipped, not undone.
        if len(idx_a) == 0 or len(idx_b) == 0:
            rec["reason"] = (f"one copy has no points ({len(idx_a)} / "
                             f"{len(idx_b)}) — nothing to align")
            out.append(rec); continue
        if spatial_cfg is not None:
            from reconstruction.loops import spatial_gate as _sg
            _g = _sg.same_surface_rule(session.xyz[idx_a], session.xyz[idx_b],
                                       spatial_cfg)
            _o = _sg.surface_overlap(session.xyz[idx_a], session.xyz[idx_b],
                                     _g, spatial_cfg)
            if _o.get("applies") and _o.get("disjoint"):
                rec["reason"] = (
                    f"two parts of one {_g.get('kind')} — the supports do not "
                    f"overlap on it (gap {_o['gap_m']:.2f} m); a surface "
                    f"measures no scale against itself")
                rec["surface_overlap"] = _o
                out.append(rec); continue
        # the drift the walk between the two visits can produce — handed in
        # only when the spatial block actually states a rate, so a caller on
        # the pre-2026-09-21 signature is left exactly as it was
        budget, budget_why = _pair_budget(session, i, j, spatial_cfg)
        # recorded the moment it is known, so a pair that never reaches the
        # measurement still says on what its fit was (not) judged
        if budget_why:
            rec["drift_budget_why"] = budget_why
        m = measure_copy(session.xyz[idx_a], session.xyz[idx_b], scfg, seed=iid,
                         **({} if budget is None else {"budget": budget}))
        if m is None:
            rec["reason"] = f"copies starved ({len(idx_a)} / {len(idx_b)} points)"
            out.append(rec); continue
        rec.update({k: v for k, v in m.items() if k not in ("R", "t", "centroid_a")})
        if m.get("fit_demoted"):
            log(f"[loops-posthoc] copies {cand.get('label')}#{iid} kf {i}<->{j}: "
                f"{m['fit_demoted']['why']}")
        rec["extent_m"] = float(np.linalg.norm(np.ptp(session.xyz[idx_a], axis=0)))
        out.append(rec)
        log(f"[loops-posthoc] copies {cand.get('label')}#{iid} kf {i}<->{j}: s_ab {m['s_ab']:.4f}, residual "
            f"{m['residual_m'] * 100:.1f} cm, coverage {m['coverage_a']:.2f}/{m['coverage_b']:.2f} → "
            f"{'scale row' if m['scale_trusted'] else 'no scale row'}")
    return out


# ── instance copies → pose-graph edges (§4.4: SAM3 as loop detector) ────────

def instance_edges(session, candidates: List[dict], ccfg, cfg, log=print,
                   sigma_floor_m: Optional[float] = None) -> List[dict]:
    """Pose-graph loop edges from the two copies of a SAM3 instance
    (USER 2026-09-13: the loop closures SAM3 detects are applied inside the
    pipeline, not only proposed). Every candidate the spatial gate judged
    loop | ambiguous — whatever the VLM class, a movable instance may have
    moved so its σ is inflated, never dropped — is measured on the CURRENT
    geometry: rigid closure of copy A (later visit, keyframe i) onto copy B
    (earlier, keyframe j) from ``measure_copy``; the edge Z_ij = T_i⁻¹ X⁻¹ T_j
    between the two visits' keyframes with a per-DOF information matrix from
    the copy's PCA shape (a wall patch observes its normal, a column the
    plane across its axis, a compact object everything). An edge is
    ``trusted`` when the closure REDUCED the copies' offset; its σ is the
    residual after the closure × the ambiguity / class factors. The same
    record shape as ``visit_edges`` so the graph, the metrics (closure,
    duplicates) and the kit treat both sources alike; unmeasurable
    candidates are returned ``accepted: False`` with the reason."""
    from correction import observability as obs_mod, solve
    from reconstruction.loops import spatial_gate as sg
    res_path = Path(session.output_dir) / "segmentation_result.json"
    instances = {int(x.get("instance_id", x.get("id"))): x
                 for x in (json.loads(res_path.read_text()).get("instances") or [])} if res_path.exists() else {}
    scfg, vcfg = cfg.certify.scale, cfg.certify.visit_loops
    # the floor this SESSION measures (certify.repeatability), falling back to
    # the configured one when nothing was measured — USER 2026-09-16
    floor_m = float(sigma_floor_m) if sigma_floor_m is not None else float(vcfg.sigma_floor_m)
    sem = cfg.loops.semantic
    up = -session.poses[:, :3, 1].mean(0); up = up / (np.linalg.norm(up) + 1e-12)
    Pu = np.outer(up, up)
    wb_rot = 1.0 / np.radians(vcfg.unobserved_sigma_deg) ** 2
    wy_rot = 1.0 / np.radians(cfg.graph.loop_sigma_rot_deg) ** 2
    out, skipped = [], []
    for cand in candidates:
        if cand.get("verdict") not in ("loop", "ambiguous"):
            continue
        iid = int(cand["instance_id"])
        i, j = int(cand["i"]), int(cand["j"])
        label = str(cand.get("label", "segment"))
        cls = str(cand.get("class") or sem.default_class)
        base = {"i": i, "j": j, "instance_id": iid, "label": label, "class": cls, "kind": cand.get("kind"),
                "gate_verdict": cand.get("verdict"), "source": "instance", "bridge": -1}
        if cls == "dynamic":
            skipped.append(dict(base, reason="dynamic instance (never a loop)"))
            continue
        inst = instances.get(iid)
        if inst is None:
            skipped.append(dict(base, reason="instance no longer in the segmentation"))
            continue
        if abs(i - j) <= 2 * int(vcfg.window_kf):
            skipped.append(dict(base, reason="not a revisit (overlapping keyframe windows)"))
            continue
        idx_a, idx_b, ks = _copy_indices(session, inst, i, j, vcfg.window_kf)
        # ── is this pair two COPIES, or two PARTS of one surface? ────────
        # The gate accepted on the separation it can OBSERVE (the offset
        # across the shared plane / axis — `same_surface_rule`), which for
        # two pieces of one ceiling is ~0. The measurement below takes the
        # copies by KEYFRAME WINDOW instead, so it gets the two ENDS of that
        # ceiling and the rigid fit "closes" them by sliding one onto the
        # other: pccr 2026-09-21 epoch 1 wrote `white_tiled_floor#45` at
        # 10.36 m / 0.0 deg, `white_wall#98` at 12.99 m and EIGHT edges for
        # one ceiling duct, 3.8 to 10.7 m each. Those are not observations of
        # drift — along its own surface a plane determines nothing — and they
        # are not harmless: they outnumbered the real closures in the acta's
        # median and turned a 41 % improvement into a reported 71.7 %
        # regression.
        #
        # Nothing is discarded that could speak: the pair is refused only
        # when the two supports do not OVERLAP on the surface they share, in
        # which case there is no common piece to close, and the refusal is
        # recorded with its numbers like every other one.
        # The shape is asked of the OBJECT, not of the two windows. Asked of
        # the windows, the same ceiling duct answered "axis" on one pair and
        # "neither" on the next, and the pair it could not classify went on to
        # close 10.30 m at 0.0 deg (pccr 2026-09-21, epoch 1: seven of the
        # acta's thirteen voting pairs were one duct and one floor against
        # themselves). `same_surface_rule` still speaks first — when the two
        # copies AGREE on a shared plane or axis that is the strongest
        # statement available, and it carries the offset across it that the
        # refusal reports — and the instance answers when they do not.
        _geom = sg.same_surface_rule(session.xyz[idx_a], session.xyz[idx_b],
                                     cfg.loops.spatial)
        if not _geom.get("same_geometry"):
            _whole = np.asarray(inst.get("globalIndices") or [], np.int64)
            _whole = _whole[(_whole >= 0) & (_whole < session.n_points)]
            if len(_whole) >= 3:
                _shape = sg.surface_of_instance(session.xyz[_whole],
                                                cfg.loops.spatial)
                if _shape.get("same_geometry"):
                    _shape["centroid_distance_m"] = _geom.get("centroid_distance_m")
                    _shape["distance_m"] = _geom.get("distance_m")
                    _geom = _shape
        _ov = sg.surface_overlap(session.xyz[idx_a], session.xyz[idx_b],
                                 _geom, cfg.loops.spatial)
        if _ov.get("applies") and _ov.get("disjoint"):
            skipped.append(dict(
                base, geometry=_geom.get("kind"), surface_overlap=_ov,
                reason=(f"two parts of one {_geom.get('kind')} "
                        f"({'the object' if _geom.get('source') == 'instance' else 'both copies'} "
                        f"says so) — the supports do not overlap on it "
                        f"(gap {_ov['gap_m']:.2f} m, centroids "
                        f"{(_geom.get('centroid_distance_m') or 0.0):.2f} m apart, "
                        f"offset across the surface "
                        f"{(_geom.get('distance_m') or 0.0) * 100:.1f} cm): one "
                        f"surface against itself observes nothing along it")))
            continue
        # what drift over the walk between these two visits could produce —
        # the bound the chosen fit's MOTION is held to. Measured here because
        # `measure_copy` sees two point clouds and not the trajectory that
        # separates them; it is the walk, not the number of keyframes, that
        # says how much error could have accumulated.
        budget, budget_why = _pair_budget(session, i, j, cfg.loops.spatial)
        m = measure_copy(session.xyz[idx_a], session.xyz[idx_b], scfg, seed=iid,
                         budget=budget)
        if m is None:
            skipped.append(dict(base, reason=f"copies starved ({len(idx_a)} / {len(idx_b)} points)"))
            continue
        if m.get("fit_demoted"):
            log(f"[loops-posthoc] instance {label}#{iid} kf {i}<->{j}: "
                f"{m['fit_demoted']['why']}")
        if not (m["offset_after_m"] < m["offset_before_m"]):
            skipped.append(dict(base, reason="the rigid closure did not reduce the copies' offset",
                                offset_before_m=m["offset_before_m"], offset_after_m=m["offset_after_m"]))
            continue
        shape = obs_mod.classify_object(session.xyz[idx_a], iid, label, ccfg)
        full = shape.shape == obs_mod.SHAPE_COMPACT or (
            shape.shape == obs_mod.SHAPE_PLANAR and shape.eig_ratios[0] <= ccfg.observability.yaw_anisotropy_max)
        if full:
            mode, axis = "full", None
        elif shape.shape == obs_mod.SHAPE_PLANAR and shape.normal is not None:
            mode, axis = "normal", np.asarray(shape.normal, np.float64)
        elif shape.shape == obs_mod.SHAPE_LINEAR and shape.axis is not None:
            mode, axis = "perp_axis", np.asarray(shape.axis, np.float64)
        else:
            mode, axis = "full", None
        sigma_t = max(float(m["offset_after_m"]), floor_m)
        factors = {}
        if cand.get("verdict") == "ambiguous":
            # The ×ambiguous inflation answers ONE question: "are these two
            # copies the same object?". When the reprojection answered exactly
            # that question with evidence — one rigid shift landing copy A on
            # copy B's mask across the frames that see both — the identity is
            # MEASURED and the inflation is no longer a statement about this
            # pair. pccr 2026-09-14 left the chair at σ 25.8 cm although its
            # copies closed to 4.3 cm, and 14 of 14 frames agreed: evidence
            # gathered and then discounted.
            #
            # The class factor is NOT touched by it. "Is it the same object?"
            # and "did the object move between the visits?" are different
            # questions, and a chair that was pushed is still the same chair —
            # the frames cannot see the difference, so a non-structural
            # proposer keeps its inflation.
            rep = cand.get("reprojection") or {}
            if rep.get("verdict") == "same_object":
                factors["reprojection:same_object"] = 1.0
                log(f"[loops-posthoc] instance {label}#{iid}: the frames measured the "
                    f"identity ({rep.get('agreeing_frames', '?')}/{rep.get('n_cross_frames', '?')} "
                    f"frame(s), agreement {rep.get('cross_recall', 0):.2f}→"
                    f"{rep.get('cross_recall_aligned', 0):.2f}) — no ×ambiguous inflation")
            else:
                factors["ambiguous"] = float(cfg.loop.ambiguous_sigma_factor)
        if cls != "structural":
            factors[f"class:{cls}"] = float(sem.nonstructural_sigma_factor)
        for f in factors.values():
            sigma_t *= f
        R = np.asarray(m["R"], np.float64); t = np.asarray(m["t"], np.float64)
        # The edge may only CARRY the DOF it observes. info_t below is already
        # built from `mode`/`axis`, so the graph is told the edge observes one
        # direction — and until 2026-09-17 it was then handed a Z built from the
        # unprojected fit, which is a different statement. A Sim3 between two
        # copies of a near-symmetric object lands on rotations drift cannot
        # produce (pccr: desk#201 165.3 deg, floor#44 40.9 deg, over a 19.3 m
        # walk) and a rotation about a distant centre moves the object 60 cm
        # while moving a camera 10 m away by TEN METRES. That is where the
        # "drift rates" of 4 to 110 cm/m came from — kf_graph reads them as
        # |(inv(Z) @ Zc).t| / walked, so a 10 m Z reads as a metre-per-metre
        # drift, the consensus compares nonsense and the graph closes 0%.
        #
        # Projecting about the copy's own centroid is what the greedy loop
        # already applies before it tries the same closure; the edge now states
        # the same thing the loop applies.
        if mode != "full":
            spec = {"mode": mode, ("normal" if mode == "normal" else "axis"): axis}
            try:
                R, t = solve.project_solution(R, t, spec,
                                              about=np.asarray(m["centroid_a"], np.float64))
            except RuntimeError:
                pass
        X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t
        Ti, Tj = session.poses[i], session.poses[j]
        Z = np.linalg.inv(Ti) @ np.linalg.inv(X) @ Tj
        R_i = Ti[:3, :3]
        info_t = _info_from_projection(sigma_t, float(vcfg.unobserved_sigma_m), mode, axis, R_i)
        wy = wy_rot if full else wb_rot
        info_rot = R_i.T @ (wy * Pu + wb_rot * (np.eye(3) - Pu)) @ R_i
        rot_deg = _rot_deg(R)
        ks_a, ks_b = ks[np.abs(ks - i) <= int(vcfg.window_kf)], ks[np.abs(ks - j) <= int(vcfg.window_kf)]
        out.append(dict(base, earlier_kfs=[int(ks_b.min()), int(ks_b.max())],
                        later_kfs=[int(ks_a.min()), int(ks_a.max())], accepted=True, trusted=True,
                        rot_deg=rot_deg, t_norm_m=float(np.linalg.norm(t)), icp_rms_m=sigma_t,
                        n_hits=int(m["n"]), offset_before_m=float(m["offset_before_m"]),
                        offset_after_m=float(m["offset_after_m"]),
                        duplicated=bool(m["offset_before_m"] > float(cfg.loops.duplicate_min_sep_m)),
                        shape=shape.shape, observability=mode,
                        observed_axis=(axis.tolist() if axis is not None else None), yaw_observed=bool(full),
                        s_ab=float(m["s_ab"]), copy_residual_m=float(m["residual_m"]), sigma_factors=factors,
                        # the edge says which fit it came from, what the
                        # alternatives would have moved, and — when the
                        # best-fitting one was refused as impossible over this
                        # walk — why (finding 27, pccr 2026-09-21)
                        fit_chosen=m.get("fit_chosen"), fit_motion=m.get("fit_motion"),
                        fit_demoted=m.get("fit_demoted"), drift_budget=m.get("drift_budget"),
                        drift_budget_why=budget_why,
                        Z=Z, X=X, sigma_m=float(sigma_t), sigma_deg=float(cfg.graph.loop_sigma_rot_deg),
                        info_t=info_t, info_rot=info_rot))
        log(f"[loops-posthoc] instance {label}#{iid} ({cls}, {cand.get('verdict')}) kf {i}<->{j}: copies "
            f"{m['offset_before_m'] * 100:.1f} → {m['offset_after_m'] * 100:.1f} cm, closure |t| "
            f"{np.linalg.norm(t) * 100:.0f} cm / {rot_deg:.1f}°, {mode}, σ {sigma_t * 100:.1f} cm"
            + (f" (×{', '.join(f'{k} {v:g}' for k, v in factors.items())})" if factors else ""))
    for s in skipped:
        s.update({"accepted": False})
    if out or skipped:
        log(f"[loops-posthoc] {len(out)} instance loop edge(s) from {len(out) + len(skipped)} "
            f"candidate(s) ({sum(1 for e in out if e['duplicated'])} duplicated), {len(skipped)} not measurable")
    return out + skipped


# ── visits → pose-graph edges ────────────────────────────────────────────────

def _info_from_projection(sigma_obs: float, sigma_big: float, mode: str, axis: Optional[np.ndarray],
                          R_i: np.ndarray) -> np.ndarray:
    """3×3 translation information in node i's frame for an observability
    mode: full | normal (only along ``axis``) | perp_axis (all but ``axis``)."""
    I3 = np.eye(3)
    wo, wb = 1.0 / sigma_obs ** 2, 1.0 / sigma_big ** 2
    if mode == "full" or axis is None:
        Sw = wo * I3
    else:
        a = np.asarray(axis, np.float64); a = a / (np.linalg.norm(a) + 1e-12)
        P = np.outer(a, a)
        Sw = (wo * P + wb * (I3 - P)) if mode == "normal" else (wo * (I3 - P) + wb * P)
    # the residual's translation lives in node i's frame: Σ_i = R_iᵀ Σ_w R_i → info_i = R_iᵀ info_w R_i
    return R_i.T @ Sw @ R_i


def _pair_key(rg: dict):
    return (int(rg["earlier_kfs"][0]), int(rg["earlier_kfs"][1]), int(rg["later_kfs"][0]), int(rg["later_kfs"][1]))


def _joint_closures_by_pair(rep: dict) -> Dict[tuple, dict]:
    """The joint closure (R, t; later → earlier) of every visit pair, keyed
    by the regions' (earlier span, later span): a region belongs to the
    merged pair whose spans contain its own."""
    out = {}
    closures = rep.get("_closures_full") or []
    for rg in rep.get("regions", []):
        if not rg.get("measured"):
            continue
        ea, eb = rg["earlier_kfs"]; la, lb = rg["later_kfs"]
        for cl in closures:
            if cl["earlier_kfs"][0] <= ea and eb <= cl["earlier_kfs"][1] and \
                    cl["later_kfs"][0] <= la and lb <= cl["later_kfs"][1]:
                out[_pair_key(rg)] = cl
                break
    return out


def _refine_region(session, rg: dict, joint: dict, ccfg, rng) -> dict:
    """Re-measure a region's closure (later → earlier) starting from the visit
    pair's joint closure: local trimmed ICP with the region's own
    observability (yaw only for compact / anisotropic blocks; translation
    projected to the observed direction otherwise). Returns the region
    record with its closure replaced (closure_found by improvement)."""
    from correction import observability as obs_mod, solve
    from correction.revisit import _region_points
    from scipy.spatial import cKDTree
    rc = ccfg.revisit
    lo = np.asarray(rg["volume_m"]["lo"]); hi = np.asarray(rg["volume_m"]["hi"])
    early = list(range(int(rg["earlier_kfs"][0]), int(rg["earlier_kfs"][1]) + 1))
    late = list(range(int(rg["later_kfs"][0]), int(rg["later_kfs"][1]) + 1))
    ia = _region_points(session, early, lo, hi, rng, rc.region_sample)
    ib = _region_points(session, late, lo, hi, rng, rc.region_sample)
    if len(ia) < ccfg.evidence.min_object_points_solve or len(ib) < ccfg.evidence.min_object_points_solve:
        return dict(rg, closure_found=False, closure={"why": "too few points on one side"})
    A, B = session.xyz[ia], session.xyz[ib]
    Rj, tj = np.asarray(joint["R"], np.float64), np.asarray(joint["t"], np.float64)
    B0 = B @ Rj.T + tj
    tree = cKDTree(A)
    d0, _ = tree.query(B, workers=ccfg.runtime.workers)
    before = float(np.median(d0))
    shape = obs_mod.classify_object(A, 0, "region", ccfg)
    full = shape.shape == obs_mod.SHAPE_COMPACT or (
        shape.shape == obs_mod.SHAPE_PLANAR and shape.eig_ratios[0] <= ccfg.observability.yaw_anisotropy_max)
    sub = B0[rng.choice(len(B0), min(ccfg.solve.icp_sample, len(B0)), replace=False)]
    R, t, rms = solve.trimmed_icp(sub, tree, A, ccfg, rotation=full)
    # total closure: p_a = R (Rj p_b + tj) + t
    R_tot = R @ Rj
    t_tot = R @ tj + t
    if not full:
        # the unobserved components come from the joint closure (a real
        # measurement of the whole pair); only the observed one is refined
        spec = ({"mode": "normal", "normal": shape.normal.tolist()} if shape.shape == obs_mod.SHAPE_PLANAR
                else {"mode": "perp_axis", "axis": shape.axis.tolist()})
        _R_l, t_l = solve.project_solution(R, t, spec)
        R_tot = Rj
        t_tot = tj + t_l
    d1, _ = tree.query(B @ R_tot.T + t_tot, workers=ccfg.runtime.workers)
    after = float(np.median(d1))
    ok = after < before
    out = dict(rg)
    out["offset_before_cm"] = round(before * 100, 1)
    out["offset_after_cm"] = round(after * 100, 1)
    out["closure_found"] = bool(ok)
    out["closure"] = ({"rot_deg": round(solve.rot_deg(R_tot), 3), "t_m": [float(x) for x in t_tot],
                       "t_norm_m": float(np.linalg.norm(t_tot)), "icp_rms_cm": round(rms * 100, 2),
                       "R": R_tot.tolist(), "yaw_observed": bool(full), "init": "joint"} if ok else
                      {"why": "ICP from the joint closure did not reduce the offset"})
    out["shape"] = shape.shape
    out["normal"] = shape.normal.tolist() if shape.normal is not None else None
    out["axis"] = shape.axis.tolist() if shape.axis is not None else None
    return out


def visit_edges(session, ccfg, vcfg, loop_sigma_rot_deg: float, log=print,
                sigma_floor_m: Optional[float] = None,
                revisits: Optional[dict] = None) -> List[dict]:
    """Pose-graph loop edges from the revisited places: one edge per
    REGION (a block both visits wrote, with its own closure and
    observability), between the central keyframes of the two visits that
    saw it. The correction module's joint closure per visit pair is a
    rigid compromise for a manual correction; the graph wants the local
    evidence — drift within a visit is not rigid. Regions sharing the same
    keyframe pair keep the best-supported one (the most hits, compact
    before partial shapes) so correlated blocks do not over-count."""
    from correction import observability as obs_mod
    from correction.revisit import detect_revisits
    rep = revisits if revisits is not None else detect_revisits(session, ccfg, log=lambda m: None,
                                                                  previews=False)
    up = -session.poses[:, :3, 1].mean(0); up = up / (np.linalg.norm(up) + 1e-12)
    rank = {obs_mod.SHAPE_COMPACT: 2, obs_mod.SHAPE_PLANAR: 1, obs_mod.SHAPE_LINEAR: 1}
    joints = _joint_closures_by_pair(rep)
    rng = np.random.default_rng(ccfg.solve.seed)
    best: Dict[tuple, dict] = {}
    skipped = []
    for rg in rep.get("regions", []):
        i = int(round((rg["later_kfs"][0] + rg["later_kfs"][1]) / 2.0))
        j = int(round((rg["earlier_kfs"][0] + rg["earlier_kfs"][1]) / 2.0))
        if not rg.get("measured"):
            skipped.append({"region": rg.get("region"), "i": i, "j": j,
                            "reason": rg.get("why") or "not measured"})
            continue
        # the region's closure re-measured from the visit pair's JOINT closure
        # as the start (a block's centroid difference starts half the ICPs in
        # a wrong basin — 0.13 vs 0.33 m for neighbouring keyframes, measured)
        joint = joints.get(_pair_key(rg))
        rg = _refine_region(session, rg, joint, ccfg, rng) if joint is not None else rg
        if not rg.get("closure_found"):
            skipped.append({"region": rg.get("region"), "i": i, "j": j,
                            "reason": (rg.get("closure") or {}).get("why") or "no closure"})
            continue
        key = (i, j)
        score = (rank.get(rg.get("shape"), 0), int(rg.get("n_hits", 0)))
        if key not in best or score > best[key][0]:
            best[key] = (score, rg)
    out = []
    for (i, j), (_score, rg) in sorted(best.items()):
        cl = rg["closure"]
        shape = rg.get("shape")
        if shape == obs_mod.SHAPE_COMPACT or cl.get("yaw_observed"):
            mode, axis = "full", None
        elif shape == obs_mod.SHAPE_PLANAR and rg.get("normal") is not None:
            mode, axis = "normal", np.asarray(rg["normal"], np.float64)
        elif shape == obs_mod.SHAPE_LINEAR and rg.get("axis") is not None:
            mode, axis = "perp_axis", np.asarray(rg["axis"], np.float64)
        else:
            mode, axis = "full", None
        X = np.eye(4); X[:3, :3] = np.asarray(cl["R"], np.float64); X[:3, 3] = np.asarray(cl["t_m"], np.float64)
        Ti, Tj = session.poses[i], session.poses[j]
        Z = np.linalg.inv(Ti) @ np.linalg.inv(X) @ Tj
        sigma_t = max(float(cl.get("icp_rms_cm", 0.0)) / 100.0,
                      float(sigma_floor_m) if sigma_floor_m is not None
                      else float(vcfg.sigma_floor_m))
        R_i = Ti[:3, :3]
        info_t = _info_from_projection(sigma_t, float(vcfg.unobserved_sigma_m), mode, axis, R_i)
        Pu = np.outer(up, up)
        wb = 1.0 / np.radians(vcfg.unobserved_sigma_deg) ** 2
        wy = (1.0 / np.radians(loop_sigma_rot_deg) ** 2) if cl.get("yaw_observed") else wb
        info_rot = R_i.T @ (wy * Pu + wb * (np.eye(3) - Pu)) @ R_i
        out.append({"i": i, "j": j, "region": rg.get("region"), "earlier_kfs": rg["earlier_kfs"],
                    "later_kfs": rg["later_kfs"], "accepted": True, "trusted": True,
                    # the block this region occupies, so the greedy loop can take
                    # the points of each visit and TRY the closure the way it
                    # tries an instance's (USER 2026-09-17)
                    "volume_m": rg.get("volume_m"),
                    "label": f"revisit_region_{rg.get('region')}",
                    "rot_deg": cl.get("rot_deg"), "t_norm_m": cl.get("t_norm_m"),
                    "icp_rms_m": sigma_t, "n_hits": rg.get("n_hits"),
                    "offset_before_m": float(rg.get("offset_before_cm", 0.0)) / 100.0,
                    "offset_after_m": float(rg.get("offset_after_cm", 0.0)) / 100.0,
                    "duplicated": bool(rg.get("duplicated")), "shape": shape, "observability": mode,
                    "observed_axis": (axis.tolist() if axis is not None else None),
                    "yaw_observed": bool(cl.get("yaw_observed")), "source": "revisit", "bridge": -1,
                    "Z": Z, "X": X, "sigma_m": float(sigma_t), "sigma_deg": float(loop_sigma_rot_deg),
                    "info_t": info_t, "info_rot": info_rot})
    for s in skipped:
        s.update({"accepted": False, "source": "revisit"})
    log(f"[loops-posthoc] {len(rep.get('regions', []))} revisit region(s) → {len(out)} loop edge(s) "
        f"({sum(1 for e in out if e['observability'] == 'full')} full, "
        f"{sum(1 for e in out if e['duplicated'])} duplicated), {len(skipped)} without closure")
    return out + skipped
