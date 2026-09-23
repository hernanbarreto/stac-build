"""The epoch loop: filter, measure, correct, repeat until it stops improving.

One pass is one epoch of the session:

  1. FILTER the cloud for real — the visits that contribute almost nothing to
     an object and the objects too small to be measured are deleted, and the
     instance index is rebuilt over what remains. Every pass tightens the
     cloud a little more, which is the point: the next measurement is made on
     cleaner evidence.
  2. MEASURE every object that has two separated visits, by aligning the
     silhouettes of its two copies in the three orthogonal views of its own
     OBB (``visit_drift.measure``).
  3. CORRECT by the translation of the best-determined object — the one whose
     two independent measurements of each component agree — spread over the
     keyframes by the drift-rate model, start pinned.
  4. PUBLISH the result as a new epoch: cloud, poses, segmentation, octree and
     the per-keyframe transform, all of it selectable and none of it
     overwriting what came before.

The loop stops when the measurement no longer improves: the best determined
drift is not smaller than the previous pass by more than what the session can
repeat, or nothing measurable is left. Nothing here decides by a constant —
the bar is the session's own repeatability.

    python -m correction.visit_drift_run --session <dir> [--max-epochs N]

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from correction import visit_drift as vd


def _repeatability_m(output_dir: Path, default_m: float,
                     log: Callable[[str], None] = lambda _m: None) -> float:
    """What the session can repeat, measured by itself.

    The bar every agreement in this module is judged against — a disagreement
    below it is the reconstruction repeating itself, not a measurement. It is
    the SAME bar the certification floors every loop σ with, so it comes from
    the same cascade (`session_repeatability`: uncertainty.json → elastic seam
    residual → intra-chunk agreement), never from a second opinion of its own.
    The config value is used only when the session measured nothing.
    """
    from reconstruction.certify.repeatability import session_repeatability
    rep = session_repeatability(output_dir, fallback_m=float(default_m), log=log)
    return float(rep["sigma_floor_m"])


def _depth_tol() -> float:
    """How far in front of a surface measured geometry has to sit to count as
    occluding it. Lives in config.yaml with the rest of the mask filter's
    parameters; a missing key fails here naming itself rather than falling
    back to a number nobody chose."""
    from config import get_param
    v = get_param("segmentation.mask_filter.depth_tol_m")
    if v is None:
        raise RuntimeError(
            "config.yaml is missing 'segmentation.mask_filter.depth_tol_m' — "
            "the masks cannot be read against the geometry without it")
    return float(v)


def _condemn(record: Optional[dict], iid, inst: dict, points: int,
             reason: str, min_points: int) -> None:
    """Write down why an instance stopped existing, in the result file's own
    ``absorbed`` format (`segmentation.mask_fates`) — the list endpoint reads
    exactly this to tell a mask that IS an object from one that is part of the
    provenance of another."""
    if record is None or iid is None:
        return
    try:
        key = str(int(iid))
    except (TypeError, ValueError):
        return
    record[key] = {"into": None, "into_label": None, "reason": reason,
                   "label": inst.get("label"), "points": int(points),
                   "min_points": int(min_points),
                   "detail": "dropped by the correction epoch"}


def _reindex(instances: List[dict], keep: np.ndarray,
             min_points: int = 0,
             record: Optional[dict] = None,
             log: Callable[[str], None] = lambda m: None) -> List[dict]:
    """globalIndices over the filtered cloud. An instance left without points
    disappears — it no longer exists in the geometry.

    `min_points` extends that to the SIZE the user asked of a segmented object
    (USER 2026-09-19: *"que no queden segmentados nada más"*). It is the same
    `visit_drift.min_points` that already governs the MASKLETS, now applied to
    the fused INSTANCES too — they are what the viewer, the OBBs, the findings
    and the chat consume, and a three-point "light fixture" is not an object.

    The points are NOT deleted: they stay in the cloud as UNSEGMENTED, which is
    the same treatment every point with no masklet gets. Only the claim that
    they form an object goes away.

    `record` is the result file's ``absorbed`` map and every drop is written
    into it. Removing the instance is not enough: the list endpoint takes its
    entries from the MASK file and hides only what the record condemns, so an
    instance dropped in silence walks straight back into the list with no
    points at all (pccr 2026-09-20: the 22 dropped here reappeared as 22
    zero-point entries the moment the panel refreshed).
    """
    new_of = np.full(len(keep), -1, np.int64)
    new_of[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    out, empty, small = [], 0, []
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < len(keep))]
        gi = new_of[gi]
        gi = gi[gi >= 0]
        iid = inst.get("instance_id", inst.get("id"))
        if not len(gi):
            empty += 1
            _condemn(record, iid, inst, 0, "unmatched", min_points)
            continue
        if min_points and len(gi) < int(min_points):
            small.append((int(len(gi)), inst.get("label", "?"),
                          inst.get("id", inst.get("instance_id", "?"))))
            _condemn(record, iid, inst, int(len(gi)), "too_small", min_points)
            continue
        d = dict(inst)
        d["globalIndices"] = gi.tolist()
        d["total_points"] = int(len(gi))
        out.append(d)
    if empty or small:
        log(f"  instances: {len(out)} kept, {empty} left with no points, "
            f"{len(small)} under {min_points} points (their points stay in the "
            f"cloud, unsegmented)")
        for n, lbl, iid in sorted(small)[:10]:
            log(f"      {lbl}#{iid}: {n} points")
    return out



def measure_epoch(output_dir: Path, cfg, rep_m: float,
                  log: Callable[[str], None] = print) -> dict:
    """The whole chain, steps 1 to 10, on the session as it stands.

    Returns the REPORT, whose `scale_rows` are the deliverable — empty when no
    object can testify. Nothing here writes anything.

    (It used to return `(t_kf, report)`, the translation solver deleted
    2026-09-19. Three early exits still returned the pair while the last
    returned the report alone, and the only caller handed it straight to
    `_write_scale_rows`, which calls `.get()` on it: a session where nothing
    could testify aborted the whole certification with `AttributeError:
    'tuple' object has no attribute 'get'`. Found 2026-09-21.)
    """
    from correction.config import load_correction_config
    from correction.distribute import chainage
    from correction.session import read_ply
    from segmentation import mask_space
    from config import get_param

    output_dir = Path(output_dir)
    _, data = read_ply(output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    poses = np.loadtxt(output_dir / "camera_poses.txt").reshape(-1, 4, 4)
    K_all = np.loadtxt(output_dir / "intrinsic.txt").reshape(-1, 4)
    up = -poses[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    chain = chainage(poses)
    kfs = mask_space.keyframe_numbers(output_dir) or []
    kf_of = np.full(int(max(kfs)) + 2, -1, np.int64)
    for k, f in enumerate(kfs):
        kf_of[int(f)] = k
    ks = kf_of[np.clip(data["frame_global"].astype(np.int64), 0, len(kf_of) - 1)]
    tol = _depth_tol()

    rep = {"chain": {}, "objects": [], "rejected": [], "provenance": "tool_measured"}

    # 1) the objects are SAM3's masklets; their visits come from the masks
    masklets = vd.masklet_visits(output_dir, log=log)
    # 2) their points, and the nested filters
    pm = vd.points_of_masklets(output_dir, data["frame_global"], data["pixel_row"],
                               data["pixel_col"], log=log)
    label_of = {m.oid: m.label for m in masklets}
    cands, steps = vd.filter_chain(masklets, pm, ks, chain, cfg.min_points,
                                   cfg.min_walk_m, cfg.min_visit_share,
                                   xyz=xyz, log=log,
                                   group_points=vd.fused_object_points(output_dir))
    rep["chain"] = steps
    if not cands:
        rep["scale_rows"] = []
        return rep

    vis = vd.Visibility(output_dir, xyz, ks, poses, K_all, tol)
    det: List[Tuple] = []
    for c in cands:
        A, B = c.copies[0], c.copies[1]
        axes = vd.obb_axes(A if len(A) >= len(B) else B, up)
        # 3) the seed: the drift over the three views, unrestricted
        t0, _pv0, _d0 = vd.drift_by_views(
            A, B, axes, float(cfg.silhouette_cell_m), float(cfg.search_margin_m),
            int(cfg.silhouette_close_px), float(cfg.silhouette_blur_px))
        # 4-5-6) the common region, and the measurement inside it
        t, pv, dis, vrep = vd.refine_drift(vis, c, axes, t0, cfg, log=lambda m: None)
        if t is None:
            rep["rejected"].append({"instance_id": c.instance_id, "label": c.label,
                                    "step": "common_region", **vrep})
            continue
        dr = vd.Drift(c.instance_id, c.label, c.visits[0], c.visits[1],
                      c.walked[0], axes, t, pv, dis, len(A), len(B))
        # 7) determination: the two views that measure each component agree
        if dr.worst_disagreement > 2.0 * rep_m:
            rep["rejected"].append({"instance_id": c.instance_id, "label": c.label,
                                    "step": "determination",
                                    "worst_disagreement_m": round(dr.worst_disagreement, 5),
                                    "bar_m": round(2.0 * rep_m, 5)})
            continue
        det.append((c, dr, vrep))
    log(f"[visit-drift] step 4-6: {len(cands)} -> "
        f"{len(cands) - sum(1 for r in rep['rejected'] if r['step'] == 'common_region')}"
        f" with a common region; step 7: {len(det)} determined "
        f"(bar {2.0 * rep_m * 100:.1f} cm)")
    if not det:
        rep["scale_rows"] = []
        return rep

    # 8) the identity has to be the only candidate
    kept, arep = vd.drop_ambiguous([(c, dr) for c, dr, _ in det], pm, label_of,
                                   xyz, cfg.max_ambiguity, log=log)
    rep["ambiguity"] = arep
    if not kept:
        rep["scale_rows"] = []
        return rep

    # 9) every closure, read as the DEPTH RATIO it measures — the scale
    # graph's §5.1 loop rows. USER-VALIDATED 2026-09-19: that is the whole
    # deliverable of this function now. The per-keyframe TRANSLATION solver
    # that used to live here was DELETED the same day: its epoch fixed
    # `glass_door#117` and tore the floor, because the floor's equations are
    # VERTICAL ONLY and the closure was horizontal, so the solve was free to
    # build a horizontal accordion the floor could not see — 131.6 mm of
    # correction between keyframes 9.0 cm apart, a local drift rate of 146 %/m
    # against the 37.2 mm/m the session actually measured.
    # the rivals each object has inside its own displacement PRICE its closure
    # (they no longer veto it) — see `visit_drift.rival_sigma_factor`
    _rivals = {int(o["oid"]): int(o.get("ambiguity", 0))
               for o in (arep.get("objects") or [])}
    rep["scale_rows"] = vd.scale_rows(kept, poses, ks, log=log,
                                      rivals_of=_rivals)
    rep["_masklets"] = masklets          # for the cloud filter, same pass
    rep["_points_by_oid"] = pm
    rep["_ks"] = ks
    rep["_vis"] = vis
    return rep


# ── one pass ─────────────────────────────────────────────────────────────

def _write_scale_rows(output_dir: Path, mrep: dict,
                      log: Callable[[str], None] = print) -> None:
    """Publish this pass's closures as scale loop rows, stamped with the epoch
    they were MEASURED on.

    The stamp is not decoration. This loop applies its translation immediately
    after measuring, so rows written by pass N describe the geometry pass N
    STARTED FROM, and a translation that closes a duplicate also hides the
    radial signal these rows are made of. `scale_stage` declares the stamp
    rather than assuming it is current.
    """
    from correction.epoch import current_epoch

    rows = (mrep or {}).get("scale_rows") or []
    if not rows:
        return
    doc = {"version": 1, "source": "correction.visit_drift",
           "measured_on_epoch": int(current_epoch(output_dir)),
           "provenance": "tool_measured", "rows": rows}
    (Path(output_dir) / "scale_loop_rows.json").write_text(json.dumps(doc, indent=1))
    log(f"[visit-drift] {len(rows)} scale loop row(s) written for the scale "
        f"graph (measured on epoch {doc['measured_on_epoch']})")


def solve_depth(output_dir: Path, log: Callable[[str], None] = print,
                cfg=None) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """THE DEPTH CORRECTION, measured and solved — nothing applied.

    Returns ``(k_kf, t_kf, report)``: the depth factor of each keyframe and the
    translation that keeps the walk continuous while the chunks change size.

    The drift is a DEPTH error, not a pose error: pccr's duplicates are
    separated ALONG THE LINE OF SIGHT (five of six closures 97-99 % radial), so
    it grows with distance and no translation can represent it — which is why
    the desk closed at 3.6 m while the tile lines 8 m away stayed 18 cm off.
    Read as depth ratios the same closures agree (1.099-1.251) and agree with
    the DA3 anchors (1.140), which never see a silhouette.

    One factor per reconstruction CHUNK, because each chunk carries its own
    gauge; a per-FRAME depth change would break the multi-view consistency the
    reconstruction still has inside a chunk. The solving is
    `certify/scale_stage`'s, unchanged.

    DECLARED LIMIT: the anchors show a CONTINUOUS drift and seven chunks can
    only spell a STAIRCASE — with `sigma_seam_log` 0.02 over six seams the
    model tops out near 12 % and pccr needs ~14 %.
    """
    from correction.config import load_correction_config
    from correction.epoch import current_epoch
    from correction.session import load_session
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.scale_stage import (
        chunk_of_keyframes, scale_transforms, solve_scale_stage)

    output_dir = Path(output_dir)
    # the rows carry the epoch they were measured on and MUST be current: this
    # correction changes the very depths they are made of, so rows from an
    # older epoch ask for a correction already in the geometry and compound it
    rows_file = output_dir / "scale_loop_rows.json"
    stamp = (json.loads(rows_file.read_text()).get("measured_on_epoch")
             if rows_file.exists() else None)
    now = int(current_epoch(output_dir))
    if stamp != now:
        vcfg = (cfg or load_correction_config()).visit_drift
        rep_m = _repeatability_m(output_dir, vcfg.default_repeatability_m, log)
        _write_scale_rows(output_dir, measure_epoch(output_dir, vcfg, rep_m,
                                                    log=log), log=log)
    if not rows_file.exists():
        log("[depth] no closure could be measured — no depth row to solve")
        return None

    session = load_session(output_dir)
    srep = solve_scale_stage(output_dir, session,
                             [], load_loops_config().certify.scale, log=log)
    if not srep.get("applied"):
        log(f"[depth] nothing to apply: {srep.get('reason')}")
        return None
    ranges, owner = chunk_of_keyframes(output_dir, session.n_kf)
    k_kf, t_kf = scale_transforms(session, ranges, owner,
                                  np.asarray(srep["r"], np.float64))
    log(f"[depth] r per chunk {np.round(srep['r'], 4).tolist()} | depth factor "
        f"{k_kf.min():.4f}-{k_kf.max():.4f} | camera shift up to "
        f"{np.linalg.norm(t_kf, axis=1).max() * 100:.1f} cm")
    return k_kf, t_kf, srep


def filter_staged_cloud(tx: Path, session, data_new, xyz_new: np.ndarray,
                        poses_new: np.ndarray, raw_data_new, cfg,
                        log: Callable[[str], None] = print
                        ) -> Optional[Tuple[dict, np.ndarray]]:
    """STEP 12 — the mask filter, INSIDE the last transaction.

    USER 2026-09-19: *"al final de todo como último paso antes de la
    consolidación y del octree"*, and *"no vamos a hacer un octree atrás de
    otro"*. So it runs here, on the geometry already staged, and the ONE
    consolidation and the ONE octree that follow carry it.

    Three rules, all on the MASKLETS of `segmentation.json` — not on the fused
    instances (USER 2026-09-18: *"no eran 82 instancias, está mal"*):
    a point that still falls outside its own object's mask in every view that
    saw it unoccluded, a masklet under `min_points`, and a visit contributing
    at or under `min_visit_share`.

    Judged HERE and not earlier because the pose is already corrected: a point
    judged before is deleted for being where the correction was about to move
    it away from (USER 2026-09-18).

    **UNSEGMENTED POINTS ARE NEVER TOUCHED** — a point that belongs to no
    masklet has no mask to judge it, and silence is not a verdict (on pccr
    that is 5.56 M points, a quarter of the cloud).

    Both clouds are rewritten with the SAME mask. `cleaned_cloud.ply` and
    `cleaned_cloud_raw.ply` must keep the same rows in the same order or
    `load_session` refuses the epoch outright — measured 2026-09-19, when an
    epoch that filtered only one of them could not be loaded again.
    """
    from config import get_param
    from correction.session import write_ply

    output_dir = Path(session.output_dir)
    masklets = vd.masklet_visits(output_dir, log=lambda m: None)
    if not masklets:
        log("  mask filter: the session has no masklet — nothing to judge")
        return None
    pm = vd.points_of_masklets(output_dir, data_new["frame_global"],
                               data_new["pixel_row"], data_new["pixel_col"],
                               log=lambda m: None)
    K_all = np.loadtxt(output_dir / "intrinsic.txt").reshape(-1, 4)
    tol = _depth_tol()
    vis = vd.Visibility(output_dir, xyz_new, session.ks, poses_new, K_all, tol)

    # the "too small to be worth anything" test is about the OBJECT, not the
    # mask: small masklets fuse into big objects (USER 2026-09-22)
    _grp = vd.fused_object_points(output_dir)
    if _grp:
        log(f"  mask filter: the {cfg.visit_drift.min_points}-point minimum is "
            f"judged on the FUSED object ({len(_grp)} masklet(s) mapped)")
    kill, frep = vd.cloud_filter_masklets(
        pm, masklets, session.ks, xyz_new, vis,
        cfg.visit_drift.min_points, cfg.visit_drift.min_visit_share,
        int(get_param("segmentation.mask_filter.max_frames_per_visit", 8)),
        int(get_param("segmentation.mask_filter.dilate_px", 2)), log=log,
        group_points=_grp)
    if not kill.any():
        log("  mask filter: every point is where its own mask says — nothing to do")
        return None
    keep = ~kill

    write_ply(tx / "cleaned_cloud.ply", session.header, data_new[keep])
    if raw_data_new is not None:
        if len(raw_data_new) != len(keep):
            raise RuntimeError(
                f"the raw cloud has {len(raw_data_new):,} rows and the cleaned "
                f"one {len(keep):,} — the mask filter cannot keep them in step; "
                f"the epoch would not load again")
        write_ply(tx / "cleaned_cloud_raw.ply", session.raw_header,
                  raw_data_new[keep])

    seg = tx / "segmentation_result.json"
    if seg.exists():
        doc = json.loads(seg.read_text())
        absorbed = dict(doc.get("absorbed") or {})
        doc["instances"] = _reindex(doc.get("instances") or [], keep,
                                    int(cfg.visit_drift.min_points),
                                    absorbed, log)
        doc["absorbed"] = absorbed
        # the census (total_points / segmented_points / coverage) is written by
        # `segmentation.republish` at step 9c, over the FINAL staged geometry —
        # one writer, so the three numbers cannot disagree with each other
        seg.write_text(json.dumps(doc))

    rep = {"dropped_points": int(kill.sum()), "kept": int(keep.sum()),
           "detail": getattr(frep, "detail", None),
           "provenance": "tool_measured"}
    log(f"  mask filter: {int(kill.sum()):,} of {len(keep):,} points leave, "
        f"{int(keep.sum()):,} stay")
    # the caller needs the mask: the transaction verifies the staged cloud
    # against the source, and after this step the expectation is the FILTERED
    # one — count and provenance both (pccr 2026-09-19: the first run with the
    # filter inside the transaction was discarded by that very check)
    return rep, keep


def apply_transform_epoch(output_dir: Path, R_kf: np.ndarray, t_kf: np.ndarray,
                          k_kf: np.ndarray, kind: str, diagnosis: List[dict],
                          log: Callable[[str], None] = print,
                          cfg=None) -> Optional[dict]:
    """Publish a per-keyframe transform as its own epoch, through the same
    transactional path as every other correction — so the mask filter at
    `apply` step 9a, the single consolidation and the single octree all run.

    USER 2026-09-19: *"si alguna de las etapas fallara o queda rechazada debe
    aplicar la época hasta donde llegó … si profundidad aplica, piso rechaza,
    salta a lo que sigue y aplica"*. A stage that solved something correctly
    must not lose it because a LATER stage was rejected: the stages contribute
    to one epoch, and a rejection skips that contribution, it does not throw
    away the ones already earned.
    """
    from correction.apply import stage_transaction, swap_transaction
    from correction.config import load_correction_config
    from correction.epoch import current_epoch
    from correction.invalidate import update_instance_store
    from correction.session import load_session
    from correction import diagnose as diag_mod, ledger

    output_dir = Path(output_dir)
    cfg = cfg or load_correction_config()
    session = load_session(output_dir)
    cid = ledger.new_correction_id()
    epoch_from = int(current_epoch(output_dir))
    tx = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=cid,
        scale_diag_new=diag_mod.regenerate_scale_diagnostics(
            output_dir, {}, epoch_from + 1, cid),
        log=log)
    swap_transaction(output_dir, tx, log=log)
    store = update_instance_store(output_dir, R_kf, t_kf, k_kf,
                                  session.frames, log=log)
    rec = {"epoch": int(tx["epoch_to"]), "corrected_by": kind,
           "points_moved": int(tx["points_moved"]), "diagnosis": diagnosis,
           "instance_store": store, "provenance": "tool_measured"}
    rp = output_dir / "corrections" / f"report_{cid}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1, default=float))
    ledger.record_run(
        output_dir, correction_id=cid, epoch_from=epoch_from,
        epoch_to=tx["epoch_to"], kind=kind, operator="auto",
        instance_ids=[], visits=[], observability=[], anchors=[],
        diagnosis=diagnosis, gates=[], overrides={}, verdict="applied",
        report_path=str(rp.relative_to(output_dir)))
    log(f"[correction] {kind} applied on its own → epoch {tx['epoch_to']}")
    return {"correction_id": cid, "epoch_to": tx["epoch_to"],
            "points_moved": tx["points_moved"], "instance_store": store}


# ── THE CORRECTION ───────────────────────────────────────────────────────

def run(session_dir, log: Callable[[str], None] = print, cfg=None,
        progress: Optional[Callable[[int, str], None]] = None) -> dict:
    """The session's correction: ONE epoch, depth and floor composed.

    USER-VALIDATED on pccr 2026-09-19 — *"el pipeline de corrección es este de
    escala y luego el aplanado de suelo respetando rampas escalones etc"* — and
    then, the same day: *"podría generarse una sola época que tenga la
    profundidad y el piso, es decir, la cero y la corregida, nada más"*.

    So the session ends with exactly two states: epoch 0 as reconstructed, and
    one corrected epoch. One transaction, one consolidation, one octree.

      1. DEPTH is measured and SOLVED (`solve_depth`) — applied to nothing.
      2. The FLOOR is solved on the geometry that depth produces, in memory,
         and the two are composed exactly before a single apply.
      3. The MASK FILTER runs last inside that same transaction
         (`correction.apply`, step 9a), before the consolidation and the octree.

    The floor MUST be measured after the depth: measuring both on the raw cloud
    gives the wrong floor. And there is no TRANSLATION stage — it was deleted
    on 2026-09-19 after its epoch closed one object and tore the floor, the
    floor's equations being vertical-only while the closure was horizontal.
    """
    t0 = time.time()
    session_dir = Path(session_dir)
    output_dir = (session_dir / "output" if (session_dir / "output").is_dir()
                  else session_dir)
    stages: List[dict] = []
    pre = None

    def _pc(pct: int, msg: str) -> None:
        if progress is not None:
            try:
                progress(int(pct), str(msg))
            except Exception:  # noqa: BLE001 — reporting never breaks the run
                pass

    # the two milestones of this stage; the floor reports inside its band
    _DEPTH_PCT, _FLOOR_PCT, _DONE_PCT = 5, 45, 100
    _pc(_DEPTH_PCT, "correction: measuring the depth on the object closures")
    log("[correction] 1/2 — DEPTH: measuring and solving (applied to nothing yet)")
    # the CALLER's config, not a second opinion. The certification resolves a
    # CorrectionConfig and used to drop it here, so the correction silently
    # re-read production config.yaml — on a synthetic session that means
    # `floor.min_inliers: 5000` against ~1,120 points per keyframe, every
    # keyframe demoted, no epoch published at all (found 2026-09-21).
    if cfg is None:
        from correction.config import load_correction_config
        cfg = cfg or load_correction_config()
    dep = solve_depth(output_dir, log=log, cfg=cfg)
    _pc(_FLOOR_PCT, "correction: solving the floor on that geometry")
    if dep is None:
        log("[correction] no depth correction — the floor runs on its own")
    else:
        k_kf, t_kf, srep = dep
        pre = {"R_kf": np.tile(np.eye(3), (len(k_kf), 1, 1)),
               "t_kf": t_kf, "k_kf": k_kf}
        stages.append({"stage": "depth", "r_per_chunk": srep.get("r"),
                       "k_min": float(k_kf.min()), "k_max": float(k_kf.max())})

    log("[correction] 2/2 — FLOOR on that geometry, composed and applied ONCE")
    from correction.run import run_floor
    try:
        frec = run_floor(output_dir, None, None, "auto", log=log, pre=pre,
                         cfg=cfg,
                         progress=lambda p, m: _pc(
                             _FLOOR_PCT + int(p * (_DONE_PCT - _FLOOR_PCT) / 100),
                             m))
        stages.append({"stage": "floor_plane+depth" if pre else "floor_plane",
                       "status": frec.get("status"),
                       "correction_id": frec.get("correction_id")})
        floor_applied = frec.get("status") == "applied"
    except Exception as e:  # noqa: BLE001 — declared, never silent
        log(f"[correction] the floor stage failed ({e})")
        stages.append({"stage": "floor_plane", "status": "failed",
                       "reason": str(e)})
        floor_applied = False

    # the floor was rejected or failed, but the DEPTH solved something real:
    # it does not go down with it (USER 2026-09-19). Applied on its own, it
    # still carries the mask filter, the consolidation and the octree.
    if not floor_applied and pre is not None:
        log("[correction] the floor did not apply — publishing the DEPTH "
            "correction on its own so what it solved is not lost")
        try:
            rec2 = apply_transform_epoch(
                output_dir, pre["R_kf"], pre["t_kf"], pre["k_kf"],
                "scale_depth",
                [{"kind": "depth",
                  "evidence": "the closures are radial: a per-chunk depth "
                              "factor, cross-checked against the DA3 anchors"}],
                log=log, cfg=cfg)
            stages.append({"stage": "depth_alone", "status": "applied",
                           **{k: v for k, v in (rec2 or {}).items()
                              if k != "instance_store"}})
        except Exception as e2:  # noqa: BLE001 — declared, never silent
            log(f"[correction] the depth-only apply failed too ({e2}) — the "
                f"session is untouched")
            stages.append({"stage": "depth_alone", "status": "failed",
                           "reason": str(e2)})

    from correction.epoch import current_epoch
    out = {"stages": stages, "epoch": int(current_epoch(output_dir)),
           "elapsed_s": round(time.time() - t0, 1),
           "provenance": "tool_measured"}
    log(f"[correction] done in {out['elapsed_s']:.0f} s — session at epoch "
        f"{out['epoch']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    a = ap.parse_args(argv)
    run(a.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
