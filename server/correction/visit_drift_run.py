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

DETERMINISM (docs/plan_determinismo.md, 2026-10-08): the evidence files this
module writes — ``scale_loop_rows.json``, ``instance_loops.json``,
``visit_drift_verify.json`` — carry a ``repro.stamp`` over the cloud, the
segmentation, the masks, the poses, the camera, the configuration and the code
they were measured with, and every reader takes them only on a matching stamp
(point 134; never the equality of an epoch number). Every parameter the
measurement reads outside ``correction.visit_drift`` comes from the configuration
the job was handed (``CorrectionConfig.raw``, point 139), never from config.yaml
re-read by the process. Every visit after the earliest is its own closure
(point 144); every closure is attributed to chunk pairs by the chunk shares of
its copies' birth keyframes (point 132); the correction ids are derived from
what they apply (point 137).

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from correction import visit_drift as vd

# the evidence files this module writes, by name; every reader verifies the stamp
EVIDENCE_FILES = ("scale_loop_rows.json", "instance_loops.json", "visit_drift_verify.json")


def measurement_record(mrep: Optional[dict]) -> dict:
    """The RECORD of a measurement (points 130-133 / 144): every candidate's margin to the
    user's bars, every rejected visit pair and why, the axes rule, the views' two peaks and the
    rivals' margins — the report without the rows it produced. It travels inside the rows file
    (stamped with them) and in the correction's report the acta carries."""
    # the in-memory handles the cloud filter reuses (``_masklets``, ``_points_by_oid``, ``_ks``,
    # ``_vis``: per-point arrays, a visibility cache) are not part of the record
    return {k: v for k, v in (mrep or {}).items() if k != "scale_rows" and not str(k).startswith("_")}
# the session files the evidence is measured on (those present enter the stamp; the absent
# ones are listed in it — a file appearing later is a difference too)
_EVIDENCE_INPUTS = ("cleaned_cloud.ply", "camera_poses.txt", "camera_frames.txt", "camera.json",
                    "segmentation.json", "segmentation_result.json", "chunk_plan.json",
                    # the fused objects the measurement reads (segmentation.fuse_parent, point 100)
                    # and the repeatability it is judged with (repeatability.session_repeatability)
                    "fusion_map.json", "uncertainty.json", "elastic_seams.json", "intra_chunk.json",
                    "maplong_run/uncertainty.json", "maplong_run/elastic_seams.json",
                    "maplong_run/intra_chunk.json")
EVIDENCE_STAMP_KEY = "stamp"


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


def _depth_tol(cfg=None) -> float:
    """How far in front of a surface measured geometry has to sit to count as
    occluding it. Lives in config.yaml with the rest of the mask filter's
    parameters, read from the configuration the correction was handed
    (``CorrectionConfig.raw``, point 139); a missing key fails here naming
    itself rather than falling back to a number nobody chose."""
    from correction.config import raw_param
    return float(raw_param(cfg, "segmentation.mask_filter.depth_tol_m"))


def _mask_filter_cost_cap(cfg=None) -> int:
    """``segmentation.mask_filter.max_frames_per_visit`` — mask keyframes measured
    per visit, the ones showing most of the object first: a COST cap, declared in
    config.yaml next to the rest of the mask filter. A missing key fails here
    naming itself (it used to fall back to a literal 8 nobody could see)."""
    from correction.config import raw_param
    return int(raw_param(cfg, "segmentation.mask_filter.max_frames_per_visit"))


def _other_mask_params(cfg=None):
    """The parameters of the mask filter's fourth rule, from their ONE declared
    home: the same reader as the silhouette flyer filter
    (``precision.silhouette_filter.params_from``) — ``silhouette_min_votes`` /
    ``silhouette_min_inside_frac`` (``reconstruction.precision.cloud``, the
    majority semantics, USER 2026-10-01: rule 4 decides by majority of the
    views like the silhouette criterion), ``loops.witness.occlusion_tol_rel``
    (occlusion AND "on that surface", relative to the measured depth — replaces
    the invented ``depth_tol_m`` for this rule), ``precision.refine.min_tri_deg``
    (a view along the birth ray cannot place the point) and
    ``segmentation.mask_filter.dilate_px`` (the point's OWN rim tolerance). The
    typed loaders fail on a missing key, naming it. Read from the configuration
    the correction was handed (point 139)."""
    from correction.config import precision_raw
    from precision.config import load_precision_config
    from precision.silhouette_filter import params_from
    raw = precision_raw(cfg)
    return params_from(load_precision_config(raw), raw)


def _vcfg_of(cfg):
    """The visit_drift block of a CorrectionConfig (or the block itself)."""
    return cfg.visit_drift if hasattr(cfg, "visit_drift") else cfg


def _load_cfg(cfg):
    from correction.config import load_correction_config
    return cfg if cfg is not None else load_correction_config()


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
        # the instance's margin to the user's bar, recorded (point 133)
        d["min_points_margin"] = int(len(gi) - int(min_points)) if min_points else None
        out.append(d)
    if empty or small:
        log(f"  instances: {len(out)} kept, {empty} left with no points, "
            f"{len(small)} under {min_points} points (their points stay in the "
            f"cloud, unsegmented)")
        for n, lbl, iid in sorted(small)[:10]:
            log(f"      {lbl}#{iid}: {n} points")
    return out


def instance_loops_of(kept) -> List[dict]:
    """(candidate, drift) pairs → the keyframe-pair records of ``instance_loops.json``."""
    return [
        {"instance_id": int(c.instance_id), "label": c.label,
         "visit_a": [int(dr.visit_a[0]), int(dr.visit_a[1])],
         "visit_b": [int(dr.visit_b[0]), int(dr.visit_b[1])],
         "i": int(round((dr.visit_a[0] + dr.visit_a[1]) / 2.0)),
         "j": int(round((dr.visit_b[0] + dr.visit_b[1]) / 2.0)),
         "t_m": [float(v) for v in np.asarray(dr.t, np.float64)],
         "sigma_m": float(dr.worst_disagreement),
         "ambiguity_m": float(getattr(dr, "ambiguity_m", 0.0) or 0.0),
         "provenance": "tool_measured"}
        for c, dr in kept]


def _owner_of(output_dir: Path, n_kf: int) -> Optional[np.ndarray]:
    """The chunk that owns each keyframe (the reconstruction's plan), None without a plan —
    a single-pass session is one chunk and the shares carry nothing."""
    from correction.units import load_chunk_plan
    if load_chunk_plan(output_dir) is None:
        return None
    from reconstruction.certify.scale_stage import chunk_of_keyframes
    return chunk_of_keyframes(output_dir, int(n_kf))[1]


def measure_epoch(output_dir: Path, cfg, rep_m: float,
                  log: Callable[[str], None] = print) -> dict:
    """The whole chain, steps 1 to 10, on the session as it stands.

    Returns the REPORT, whose `scale_rows` are the deliverable — empty when no
    object can testify. Nothing here writes anything.

    EVERY SURVIVING VISIT IS MEASURED AGAINST THE EARLIEST (docs/plan_determinismo.md
    point 144, 2026-10-08), each pair its own closure with its own σ — the start↔end pair,
    the longest lever arm, used to be skipped whenever a SAM3 gap split the first pass.
    The reference copy is the earliest visit and the OBB axes come from it, by a stable
    rule (point 130); the determination test keeps the user's bar (disagreement ≤ 2 × the
    repeatability) with its margin recorded.

    (It used to return `(t_kf, report)`, the translation solver deleted
    2026-09-19. Three early exits still returned the pair while the last
    returned the report alone, and the only caller handed it straight to
    `_write_scale_rows`, which calls `.get()` on it: a session where nothing
    could testify aborted the whole certification with `AttributeError:
    'tuple' object has no attribute 'get'`. Found 2026-09-21.)
    """
    from correction.config import judge_of
    from correction.distribute import chainage
    from correction.session import read_ply
    from segmentation import mask_space

    output_dir = Path(output_dir)
    vcfg = _vcfg_of(cfg)
    fac, _conf = judge_of(vcfg)
    _, data = read_ply(output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    poses = np.loadtxt(output_dir / "camera_poses.txt").reshape(-1, 4, 4)
    up = -poses[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    chain = chainage(poses)
    kfs = mask_space.keyframe_numbers(output_dir) or []
    kf_of = np.full(int(max(kfs)) + 2, -1, np.int64)
    for k, f in enumerate(kfs):
        kf_of[int(f)] = k
    ks = kf_of[np.clip(data["frame_global"].astype(np.int64), 0, len(kf_of) - 1)]
    tol = _depth_tol(vcfg)

    rep = {"chain": {}, "objects": [], "rejected": [], "provenance": "tool_measured",
           "determination_bar_m": 2.0 * float(rep_m), "error_factor": fac}

    # 1) the objects are SAM3's masklets; their visits come from the masks
    masklets = vd.masklet_visits(output_dir, log=log)
    # 2) their points, and the nested filters
    pm = vd.points_of_masklets(output_dir, data["frame_global"], data["pixel_row"],
                               data["pixel_col"], log=log, aspect_tol=float(vcfg.grid_aspect_tol))
    label_of = {m.oid: m.label for m in masklets}
    cands, steps = vd.filter_chain(masklets, pm, ks, chain, vcfg.min_points,
                                   vcfg.min_walk_m, vcfg.min_visit_share,
                                   xyz=xyz, log=log,
                                   group_points=vd.fused_object_points(output_dir))
    rep["chain"] = steps
    if not cands:
        rep["scale_rows"] = []
        return rep

    # the camera the cloud was BUILT with, verified on its birth pixels (audit
    # 2026-10-01: Omega's intrinsic.txt is another camera once F5 refined it)
    cam = vd.projection_camera(output_dir, xyz, ks, poses, data["pixel_row"],
                               data["pixel_col"], min_depth_m=float(vcfg.min_depth_m), log=log)
    vis = vd.Visibility(output_dir, xyz, ks, poses, cam, tol, min_depth_m=float(vcfg.min_depth_m))
    det: List[Tuple] = []
    n_pairs = 0
    for c in cands:
        # the reference copy is the EARLIEST visit, always; its axes by the stable rule
        axes, axes_info = vd.obb_axes_info(c.copies[0], up, fac)
        for ib_ in range(1, len(c.visits)):
            n_pairs += 1
            A, B = c.copies[0], c.copies[ib_]
            vb = c.visits[ib_]
            # 3) the seed: the drift over the three views, unrestricted
            t0, _pv0, _d0, _amb0 = vd.drift_by_views(
                A, B, axes, float(vcfg.silhouette_cell_m), float(vcfg.search_margin_m),
                int(vcfg.silhouette_close_px), float(vcfg.silhouette_blur_px), fac)
            # 4-5-6) the common region, and the measurement inside it
            t, pv, dis, vrep = vd.refine_drift(vis, c, axes, t0, vcfg, log=lambda m: None,
                                               visit_b=vb)
            base = {"instance_id": c.instance_id, "label": c.label,
                    "visit_a": [int(c.visits[0][0]), int(c.visits[0][1])],
                    "visit_b": [int(vb[0]), int(vb[1])], "axes": axes_info}
            if t is None:
                rep["rejected"].append({**base, "step": "common_region", **vrep})
                continue
            walked = float(chain[vb[0]] - chain[c.visits[0][1]])
            amb = vrep.get("ambiguity") or {}
            dr = vd.Drift(c.instance_id, c.label, c.visits[0], vb, walked, axes, t, pv, dis,
                          len(A), len(B), ambiguity_m=float(amb.get("ambiguity_m", 0.0) or 0.0),
                          converged=bool(vrep.get("converged", True)),
                          cycle_period=int(vrep.get("cycle_period", 0) or 0))
            # 7) determination: the two views that measure each component agree — the
            #    user's bar (2 × the repeatability), its margin recorded either way
            margin = 2.0 * rep_m - dr.worst_disagreement
            if margin < 0.0:
                rep["rejected"].append({**base, "step": "determination",
                                        "worst_disagreement_m": round(dr.worst_disagreement, 5),
                                        "bar_m": round(2.0 * rep_m, 5),
                                        "margin_m": round(margin, 5)})
                continue
            det.append((c, dr, {**vrep, "determination_margin_m": round(margin, 5),
                                "axes": axes_info}))
    n_common = n_pairs - sum(1 for r in rep['rejected'] if r['step'] == 'common_region')
    log(f"[visit-drift] step 4-6: {len(cands)} objects, {n_pairs} visit pair(s) -> "
        f"{n_common} with a common region; step 7: {len(det)} determined "
        f"(bar {2.0 * rep_m * 100:.1f} cm)")
    rep["objects"] = [{"instance_id": c.instance_id, "label": c.label, **dr.as_dict(),
                       "determination_margin_m": v["determination_margin_m"],
                       "common_voxels": v.get("common_voxels"), "axes": v["axes"]}
                      for c, dr, v in det]
    if not det:
        rep["scale_rows"] = []
        return rep

    # 8) the identity has to be the only candidate
    kept, arep = vd.drop_ambiguous([(c, dr) for c, dr, _ in det], pm, label_of,
                                   xyz, vcfg.max_ambiguity, log=log)
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
    owner = _owner_of(output_dir, len(poses))
    rep["scale_rows"] = vd.scale_rows(kept, poses, ks, log=log,
                                      rivals_of=_rivals, owner=owner)
    # the re-identified objects as KEYFRAME PAIRS for the correspondence stage
    # (claude_stac.txt §4-F3 → F4): each object's two visits, the drift measured
    # between them and its σ — the disagreement of the two silhouette views that
    # measure the same component (the determination test of step 7)
    rep["instance_loops"] = instance_loops_of(kept)
    rep["_masklets"] = masklets          # for the cloud filter, same pass
    rep["_points_by_oid"] = pm
    rep["_ks"] = ks
    rep["_vis"] = vis
    return rep


# ── EVIDENCE: measure, publish, apply nothing (claude_stac.txt §4-F3) ────

MODES = ("measure", "apply", "verify")


def evidence_stamp(output_dir: Path, cfg=None) -> dict:
    """The identity of this module's evidence (point 134): ``repro.stamp`` over the session
    files the closures are measured on — the cloud, the poses, the keyframe list, the camera,
    the masklets (segmentation.json and the mask store it names), the fused objects, the chunk
    plan — those present, with the absent ones listed; the code that measures; the parameters
    (``correction.visit_drift``). Never an epoch number."""
    from repro import stamp
    out = Path(output_dir)
    vcfg = _vcfg_of(_load_cfg(cfg))
    names = list(_EVIDENCE_INPUTS)
    seg = out / "segmentation.json"
    if seg.exists():
        try:
            mask_file = str(json.loads(seg.read_text()).get("mask_file") or "seg_masks.npz")
        except (OSError, ValueError):
            mask_file = "seg_masks.npz"
        names.append(mask_file)
    present = {n: out / n for n in names if (out / n).exists()}
    absent = sorted(set(names) - set(present))
    params = vcfg.as_params() if hasattr(vcfg, "as_params") else dict(vcfg.__dict__)
    from correction.config import judge_of
    import sys as _sys
    return stamp(inputs=present, code=[vd, _sys.modules[__name__]],
                 config={"correction.visit_drift": params, "absent": absent,
                         # what the measurement reads outside its own block: the user's rule
                         # (factor, confidence — the axes, the peaks) and the occlusion bar
                         "judge": list(judge_of(vcfg)),
                         "segmentation.mask_filter.depth_tol_m": _depth_tol(vcfg)})


def evidence_fresh(output_dir: Path, doc: Optional[dict], cfg=None) -> Tuple[bool, List[str]]:
    """(fresh, why): is an evidence file measured on the session AS IT IS NOW — its stamp
    equals the one computed now (point 134). No stamp (a file written before 2026-10-08, or
    by hand) is stale, named."""
    from repro import check_stamp
    if not isinstance(doc, dict):
        return False, ["no evidence document"]
    diffs = check_stamp(doc.get(EVIDENCE_STAMP_KEY), evidence_stamp(output_dir, cfg))
    return (not diffs), diffs


def read_evidence(output_dir: Path, name: str, cfg=None) -> Tuple[Optional[dict], bool, List[str]]:
    """(doc, fresh, why) of one evidence file; (None, False, [reason]) when absent."""
    p = Path(output_dir) / name
    if not p.exists():
        return None, False, [f"{name} does not exist"]
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return None, False, [f"{name} is unreadable ({e})"]
    fresh, why = evidence_fresh(output_dir, doc, cfg)
    return doc, fresh, why


def _stamp_out(output_dir: Path, name: str, doc: dict, cfg=None) -> Path:
    # stamped with the RECONSTRUCTION too (docs/plan_determinismo.md point 34): epoch numbers
    # restart with every reconstruction; F2 / F4 take these files only with THIS id — and with
    # the evidence stamp (point 134) every reader verifies
    from correction.epoch import RECONSTRUCTION_ID_KEY, current_epoch, reconstruction_id_or_none
    doc = {"version": 2, "source": "correction.visit_drift",
           "measured_on_epoch": int(current_epoch(output_dir)),
           RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(output_dir),
           EVIDENCE_STAMP_KEY: evidence_stamp(output_dir, cfg),
           "provenance": "tool_measured", **doc}
    p = Path(output_dir) / name
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=float, sort_keys=True))
    tmp.replace(p)
    return p


def measure(session_dir, log: Callable[[str], None] = print, cfg=None) -> dict:
    """``mode: measure`` — the closures on the session AS IT STANDS (epoch 0 in the
    EVIDENCE stage), published for the stages that consume them and nothing
    applied: ``scale_loop_rows.json`` (the gauge's relative rows, F2) and
    ``instance_loops.json`` (keyframe pairs per re-identified object with the σ of
    its silhouette match, F4). Writes no epoch."""
    session_dir = Path(session_dir)
    output_dir = (session_dir / "output" if (session_dir / "output").is_dir()
                  else session_dir)
    cfg = _load_cfg(cfg)
    vcfg = cfg.visit_drift
    rep_m = _repeatability_m(output_dir, vcfg.default_repeatability_m, log)
    mrep = measure_epoch(output_dir, vcfg, rep_m, log=log)
    _write_scale_rows(output_dir, mrep, log=log, cfg=cfg)
    loops = mrep.get("instance_loops") or []
    p = _stamp_out(output_dir, "instance_loops.json",
                   {"repeatability_m": rep_m, "loops": loops}, cfg=cfg)
    log(f"[visit-drift] MEASURE: {len(mrep.get('scale_rows') or [])} scale row(s), "
        f"{len(loops)} instance loop(s) → {p.name} (nothing applied)")
    return {"mode": "measure", "n_scale_rows": len(mrep.get("scale_rows") or []),
            "n_instance_loops": len(loops), "repeatability_m": rep_m,
            "rejected": mrep.get("rejected", []), "provenance": "tool_measured"}


def verify(session_dir, log: Callable[[str], None] = print, cfg=None) -> dict:
    """``mode: verify`` — the closures re-measured on the CURRENT epoch (the
    precision core's epoch N): what each duplicated object still shows, against
    the session's own repeatability — the core's acceptance measurement
    (``visit_drift_verify.json``). Writes no epoch."""
    session_dir = Path(session_dir)
    output_dir = (session_dir / "output" if (session_dir / "output").is_dir()
                  else session_dir)
    cfg = _load_cfg(cfg)
    vcfg = cfg.visit_drift
    rep_m = _repeatability_m(output_dir, vcfg.default_repeatability_m, log)
    mrep = measure_epoch(output_dir, vcfg, rep_m, log=log)
    loops = mrep.get("instance_loops") or []
    res = [float(np.linalg.norm(l["t_m"])) for l in loops]
    doc = {"repeatability_m": rep_m,
           "closures": [{**l, "residual_m": r} for l, r in zip(loops, res)],
           "median_residual_m": float(np.median(res)) if res else None,
           "n_within_repeatability": int(sum(1 for r in res if r <= rep_m)),
           "n_closures": len(res)}
    p = _stamp_out(output_dir, "visit_drift_verify.json", doc, cfg=cfg)
    log(f"[visit-drift] VERIFY: {len(res)} closure(s), median residual "
        + (f"{doc['median_residual_m'] * 100:.1f} cm" if res else "n/a")
        + f" vs repeatability {rep_m * 100:.1f} cm → {p.name} (nothing applied)")
    return {"mode": "verify", **doc, "provenance": "tool_measured"}


# ── one pass ─────────────────────────────────────────────────────────────

def _write_scale_rows(output_dir: Path, mrep: dict,
                      log: Callable[[str], None] = print, cfg=None) -> bool:
    """Publish this pass's closures as scale loop rows, stamped with what they were
    MEASURED on (point 134) and the epoch they were measured on (a record, not the
    freshness test).

    The stamp is not decoration. This loop applies its correction immediately
    after measuring, so rows written by pass N describe the geometry pass N
    STARTED FROM, and a correction that closes a duplicate also hides the
    radial signal these rows are made of. Every reader verifies the stamp
    rather than assuming the rows are current. A pass that measured NO row
    deletes the file: a stale file left behind used to be the one solved.
    Returns whether rows were written."""
    rows = (mrep or {}).get("scale_rows") or []
    p = Path(output_dir) / "scale_loop_rows.json"
    if not rows:
        if p.exists():
            p.unlink()
            log("[visit-drift] no scale loop row measured — the stale scale_loop_rows.json "
                "on disk was deleted (nothing solves it)")
        return False
    doc = _stamp_out(output_dir, "scale_loop_rows.json",
                     {"rows": rows, "measurement": measurement_record(mrep)}, cfg=cfg)
    ep = json.loads(doc.read_text()).get("measured_on_epoch")
    log(f"[visit-drift] {len(rows)} scale loop row(s) written for the scale "
        f"graph (measured on epoch {ep}, stamped)")
    return True


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
    `certify/scale_stage`'s — judged by THE USER'S RULE on the closures themselves
    (point 127: leave-one-out, ≥ 5 judges, significant, ≥ the factor × their error).

    DECLARED LIMIT: the anchors show a CONTINUOUS drift and seven chunks can
    only spell a STAIRCASE — with `sigma_seam_log` 0.02 over six seams the
    model tops out near 12 % and pccr needs ~14 %.

    AFTER THE PRECISION GAUGE (USER 2026-09-29) this runs on the RESIDUAL: the
    gauge set the scale continuously along the walk, the closures re-measured
    on the corrected geometry say what is left, and the scale stage lets only
    them drive the graph (`scale_stage.stand_down_for_gauge`) — the DA3 trend,
    anchor and absolute rows would apply the gauge's drift a second time. The
    unit stays the Omega chunk (`chunk_plan.json` → `frame_owner`, the same
    nearest-centre rule that writes the records' ``chunk`` field). DECLARED: a
    single-pass session is ONE chunk — one degree of freedom, its size, which
    the gauge holds — so there the closures cannot move anything.

    Returns ``(k_kf, t_kf, report)``; ``k_kf`` and ``t_kf`` are None when nothing
    is applied, and the report says why (the acta records it either way).
    """
    from correction.config import loops_raw
    from correction.epoch import current_epoch
    from correction.session import load_session
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.scale_stage import (
        chunk_of_keyframes, scale_transforms, solve_scale_stage)

    output_dir = Path(output_dir)
    cfg = _load_cfg(cfg)
    vcfg = cfg.visit_drift
    # the rows carry the stamp of what they were measured on and MUST match the session
    # as it is now (point 134): this correction changes the very depths they are made of,
    # so rows measured on another geometry ask for a correction already in it and compound it
    now = int(current_epoch(output_dir))
    doc, fresh, why = read_evidence(output_dir, "scale_loop_rows.json", cfg)
    old_epoch = doc.get("measured_on_epoch") if isinstance(doc, dict) else None
    record = doc.get("measurement") if (fresh and isinstance(doc, dict)) else None
    if not fresh:
        if doc is not None:
            log(f"[depth] scale_loop_rows.json is not this geometry's — {'; '.join(why[:4])} "
                f"— re-measuring")
        rep_m = _repeatability_m(output_dir, vcfg.default_repeatability_m, log)
        mrep = measure_epoch(output_dir, vcfg, rep_m, log=log)
        record = measurement_record(mrep)
        _write_scale_rows(output_dir, mrep, log=log, cfg=cfg)
        doc, fresh, why = read_evidence(output_dir, "scale_loop_rows.json", cfg)
    if not fresh:
        reason = (f"no closure could be measured on epoch {now}"
                  + (f" (the rows on file were measured on epoch {old_epoch} and are not this "
                     f"geometry's: {'; '.join(why[:2])} — not used)" if doc is not None or
                     old_epoch is not None else "")
                  + " — no depth row to solve")
        log(f"[depth] {reason}")
        return None, None, {"applied": False, "reason": reason, "measurement": record}

    session = load_session(output_dir)
    lcfg = load_loops_config(loops_raw(cfg))
    srep = solve_scale_stage(output_dir, session, [], lcfg.certify.scale, log=log,
                             graph=lcfg.graph, ccfg=cfg)
    srep["measurement"] = record                 # the margins of the measurement (130-133 / 144)
    if not srep.get("applied"):
        log(f"[depth] nothing to apply: {srep.get('reason')}")
        return None, None, srep
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

    Four rules, all on the MASKLETS of `segmentation.json` — not on the fused
    instances (USER 2026-09-18: *"no eran 82 instancias, está mal"*):
    a point that still falls outside its own object's mask in every view that
    saw it unoccluded, a masklet under `min_points`, a visit contributing
    at or under `min_visit_share`, and a point that lies ON another object's
    surface inside its mask in the majority of the views that saw it
    (`visit_drift.cloud_filter_masklets`, rewritten 2026-10-01 for edge
    definition; its parameters: `_other_mask_params`). Every projection uses
    the camera the cloud was built with (`visit_drift.projection_camera`).
    The rules and their bars are the USER'S (points 106 / 133); what this adds
    is the RECORD: every deleted point's ballot and the distance of each judged
    pixel to the mask edge (``mask_filter_votes.npz``, an epoch artifact).

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
    from correction.session import write_ply

    output_dir = Path(session.output_dir)
    vcfg = cfg.visit_drift
    masklets = vd.masklet_visits(output_dir, log=lambda m: None)
    if not masklets:
        log("  mask filter: the session has no masklet — nothing to judge")
        return None
    tol = _depth_tol(cfg)
    # the camera the cloud was BUILT with — the mask lookup AND the z-buffer —
    # verified on the staged cloud's own birth pixels with the staged poses (the
    # warp moves a point with its birth camera, so its birth pixel is invariant);
    # a camera that does not reproduce them fails the step instead of being mixed
    # with another (audit 2026-10-01: intrinsic.txt misplaced pccr's rims by
    # 3-23 px)
    # a partial configuration (a CLI's, a test's) completes itself from the server's — the
    # job's frozen one carries every key (correction.config.raw_param's rule)
    _min_depth = getattr(vcfg, "min_depth_m", None)
    _aspect = getattr(vcfg, "grid_aspect_tol", None)
    cam = vd.projection_camera(output_dir, xyz_new, session.ks, poses_new,
                               data_new["pixel_row"], data_new["pixel_col"],
                               min_depth_m=(float(_min_depth) if _min_depth is not None else None),
                               log=log)
    # which masklet each point belongs to: its birth pixel carried onto the mask
    # grid through the same camera's exact grid maps
    pm = vd.points_of_masklets(output_dir, data_new["frame_global"],
                               data_new["pixel_row"], data_new["pixel_col"],
                               log=lambda m: None,
                               aspect_tol=(float(_aspect) if _aspect is not None else None),
                               mask_pixels=cam.record_to_mask(data_new["pixel_row"],
                                                              data_new["pixel_col"]))
    vis = vd.Visibility(output_dir, xyz_new, session.ks, poses_new, cam, tol,
                        min_depth_m=(float(_min_depth) if _min_depth is not None else None))
    o4 = _other_mask_params(cfg)

    # the "too small to be worth anything" test is about the OBJECT, not the
    # mask: small masklets fuse into big objects (USER 2026-09-22)
    _grp = vd.fused_object_points(output_dir)
    if _grp:
        log(f"  mask filter: the {vcfg.min_points}-point minimum is "
            f"judged on the FUSED object ({len(_grp)} masklet(s) mapped)")
    kill, frep = vd.cloud_filter_masklets(
        pm, masklets, session.ks, xyz_new, vis,
        vcfg.min_points, vcfg.min_visit_share,
        _mask_filter_cost_cap(cfg), int(o4.dilate_px),
        occlusion_tol_rel=float(o4.occlusion_tol_rel), min_votes=int(o4.min_votes),
        min_inside_frac=float(o4.min_inside_frac), min_tri_deg=float(o4.min_tri_deg),
        log=log, group_points=_grp, group_roots=vd.fused_object_roots(output_dir))
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
                                    int(vcfg.min_points),
                                    absorbed, log)
        doc["absorbed"] = absorbed
        # the census (total_points / segmented_points / coverage) is written by
        # `segmentation.republish` at step 9c, over the FINAL staged geometry —
        # one writer, so the three numbers cannot disagree with each other
        seg.write_text(json.dumps(doc))

    rep = {"dropped_points": int(kill.sum()), "kept": int(keep.sum()),
           "detail": getattr(frep, "detail", None),
           "margins": getattr(frep, "margins", None),
           "rules": {"min_points": int(vcfg.min_points), "min_visit_share": float(vcfg.min_visit_share),
                     "dilate_px": int(o4.dilate_px), "min_votes": int(o4.min_votes),
                     "min_inside_frac": float(o4.min_inside_frac), "min_tri_deg": float(o4.min_tri_deg),
                     "occlusion_tol_rel": float(o4.occlusion_tol_rel), "depth_tol_m": float(tol),
                     "max_frames_per_visit": int(_mask_filter_cost_cap(cfg))},
           "votes_file": "mask_filter_votes.npz",
           "provenance": "tool_measured",
           "_votes": getattr(frep, "votes", None)}
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
                          cfg=None, record_extra: Optional[dict] = None) -> Optional[dict]:
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
    from correction.chain import applied_depth_factor
    from correction.epoch import current_epoch
    from correction.invalidate import update_instance_store
    from correction.session import load_session
    from correction import diagnose as diag_mod, ledger

    output_dir = Path(output_dir)
    cfg = _load_cfg(cfg)
    session = load_session(output_dir)
    epoch_from = int(current_epoch(output_dir))
    # the id is what the epoch applies (point 137), never a draw
    cid = ledger.new_correction_id(*ledger.transform_parts(kind, epoch_from, session.frames,
                                                           R_kf, t_kf, k_kf))
    k_applied = {int(f): float(k) for f, k in
                 zip(session.frames, applied_depth_factor(output_dir, session.frames))}
    tx = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=cid,
        scale_diag_new=diag_mod.regenerate_scale_diagnostics(
            output_dir, {int(f): float(k_kf[i]) for i, f in enumerate(session.frames)},
            epoch_from + 1, cid, k_applied_by_frame=k_applied),
        log=log, record_extra=record_extra)
    swap_transaction(output_dir, tx, log=log)
    store = update_instance_store(output_dir, R_kf, t_kf, k_kf,
                                  session.frames, log=log)
    rec = {"epoch": int(tx["epoch_to"]), "corrected_by": kind,
           "points_moved": int(tx["points_moved"]), "diagnosis": diagnosis,
           "instance_store": store, "provenance": "tool_measured",
           # the run's configuration digest and input stamp (point 139)
           **(record_extra or {})}
    rp = output_dir / "corrections" / f"report_{cid}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1, default=float, sort_keys=True))
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
        progress: Optional[Callable[[int, str], None]] = None,
        record_extra: Optional[dict] = None) -> dict:
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

    ``record_extra`` is sealed into the epoch's record (the certification's
    run-configuration sha256 and input stamp). The elapsed time is returned
    in memory only; no compared artifact carries it (point 166).
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
    cfg = _load_cfg(cfg)
    from precision.gauge import gauge_applied
    after_gauge = gauge_applied(output_dir)
    k_kf = t_kf = None
    srep: dict = {}
    if cfg.visit_drift.skip_when_gauge_applied and after_gauge:
        srep = {"applied": False,
                "reason": "correction.visit_drift.skip_when_gauge_applied: the precision "
                          "gauge already applied the continuous scale along the walk "
                          "(gauge.json) — the depth stage stands down entirely"}
        log(f"[correction] DEPTH stands down: {srep['reason']}")
    else:
        if after_gauge:
            # USER 2026-09-29: the closures ALSO correct after the gauge, on the
            # RESIDUAL — measured on this epoch, the gauge's own rows stood down
            log("[correction] DEPTH on the RESIDUAL after the precision gauge: only the "
                "closures measured on the current epoch drive it (the DA3 trend, anchor "
                "and absolute rows stand down — the gauge already spent them)")
        k_kf, t_kf, srep = solve_depth(output_dir, log=log, cfg=cfg)
    _pc(_FLOOR_PCT, "correction: solving the floor on that geometry")
    # the acta records the depth stage either way: stood down by the switch,
    # identity, one chunk, did not earn the right — they are not the same thing
    depth_stage = {"stage": "depth", "applied": k_kf is not None,
                   "after_gauge": bool(after_gauge),
                   "stood_down_for_gauge": srep.get("stood_down_for_gauge"),
                   "judge": srep.get("judge"),
                   # the user's rule on the closures (point 127) and the margins of the
                   # measurement that produced them (points 130-133 / 144), applied or not
                   "earned": srep.get("earned"), "bound": srep.get("gate"),
                   "anchor_margins": srep.get("anchor_margins"),
                   "measurement": srep.get("measurement")}
    if k_kf is None:
        log("[correction] no depth correction — the floor runs on its own")
        depth_stage.update({"status": "not_applied", "reason": srep.get("reason"),
                            "n_chunks": srep.get("n_chunks")})
    else:
        pre = {"R_kf": np.tile(np.eye(3), (len(k_kf), 1, 1)),
               "t_kf": t_kf, "k_kf": k_kf}
        depth_stage.update({"r_per_chunk": srep.get("r"),
                            "k_min": float(k_kf.min()), "k_max": float(k_kf.max())})
    stages.append(depth_stage)

    log("[correction] 2/2 — FLOOR on that geometry, composed and applied ONCE")
    from correction.run import run_floor
    try:
        frec = run_floor(output_dir, None, None, "auto", log=log, pre=pre,
                         cfg=cfg, record_extra=record_extra,
                         progress=lambda p, m: _pc(
                             _FLOOR_PCT + int(p * (_DONE_PCT - _FLOOR_PCT) / 100),
                             m))
        stages.append({"stage": "floor_plane+depth" if pre else "floor_plane",
                       "status": frec.get("status"),
                       "correction_id": frec.get("correction_id"),
                       "floor": (frec.get("diagnosis") or [{}])[0].get("model_params")})
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
        _sd = srep.get("stood_down_for_gauge")
        evidence = ("the closures are radial: a per-chunk depth factor from the "
                    f"closures measured on epoch {_sd.get('current_epoch')} only — the "
                    "DA3 anchor, trend and absolute rows stood down after the "
                    "precision gauge" if _sd else
                    "the closures are radial: a per-chunk depth factor, "
                    "cross-checked against the DA3 anchors")
        try:
            rec2 = apply_transform_epoch(
                output_dir, pre["R_kf"], pre["t_kf"], pre["k_kf"],
                "scale_depth", [{"kind": "depth", "evidence": evidence}],
                log=log, cfg=cfg, record_extra=record_extra)
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
    ap.add_argument("--mode", choices=MODES, default="apply",
                    help="measure: publish scale_loop_rows.json + instance_loops.json on "
                         "the current epoch, apply nothing; apply: the correction epoch; "
                         "verify: the residual closures on the current epoch")
    a = ap.parse_args(argv)
    {"measure": measure, "apply": run, "verify": verify}[a.mode](a.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
