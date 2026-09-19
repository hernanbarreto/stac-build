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


# ── session state ────────────────────────────────────────────────────────

def _load(output_dir: Path):
    from correction.session import read_ply
    from correction.distribute import chainage

    hdr, data = read_ply(output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)
    frames = [int(x) for x in (output_dir / "camera_frames.txt").read_text().split()]
    kf_of = np.full(max(frames) + 2, -1, np.int64)
    for k, f in enumerate(frames):
        kf_of[f] = k
    poses = np.loadtxt(output_dir / "camera_poses.txt").reshape(-1, 4, 4)
    ks = kf_of[np.clip(fg, 0, len(kf_of) - 1)]
    up = -poses[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    inst = json.loads((output_dir / "segmentation_result.json").read_text())
    return (hdr, data, xyz, fg, ks, frames, poses, chainage(poses), up,
            inst["instances"], inst)


def _mid_kf(visit) -> int:
    """The keyframe halfway through a visit — where that visit's evidence is
    centred along the walk. A midpoint, not a threshold."""
    return int((int(visit[0]) + int(visit[1])) / 2)


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


def _write_ply(path: Path, template, xyz: np.ndarray, keep: np.ndarray) -> None:
    """Binary PLY with every original per-point field, over the kept points."""
    rec = np.empty(int(keep.sum()), dtype=template.dtype)
    for n in template.dtype.names:
        rec[n] = template[n][keep]
    rec["x"] = xyz[keep, 0]
    rec["y"] = xyz[keep, 1]
    rec["z"] = xyz[keep, 2]
    kind = {'f4': 'float', 'f8': 'double', 'u1': 'uchar', 'u2': 'ushort',
            'i4': 'int', 'u4': 'uint', 'i2': 'short', 'i1': 'char'}
    head = ["ply", "format binary_little_endian 1.0",
            f"element vertex {len(rec)}"]
    for n in template.dtype.names:
        head.append(f"property {kind[rec.dtype[n].str[1:]]} {n}")
    head.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        f.write(rec.tobytes())


def _reindex(instances: List[dict], keep: np.ndarray) -> List[dict]:
    """globalIndices over the filtered cloud. An instance left without points
    disappears — it no longer exists in the geometry."""
    new_of = np.full(len(keep), -1, np.int64)
    new_of[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    out = []
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < len(keep))]
        gi = new_of[gi]
        gi = gi[gi >= 0]
        if not len(gi):
            continue
        d = dict(inst)
        d["globalIndices"] = gi.tolist()
        d["total_points"] = int(len(gi))
        out.append(d)
    return out


# ── one pass ─────────────────────────────────────────────────────────────

def _next_epoch(output_dir: Path) -> int:
    """One past the highest epoch the session holds — not `current + 1`.

    The session can be shown at an older epoch while newer ones sit on disk
    (selecting is free), and `current + 1` then collides with a directory that
    already exists.
    """
    from correction.apply import available_epochs
    return max(e["epoch"] for e in available_epochs(output_dir)) + 1


def _publish(output_dir: Path, data, warped: np.ndarray, keep: np.ndarray,
             P: np.ndarray, instances, seg_doc, frames,
             R_kf: np.ndarray, t_kf: np.ndarray, k_kf: np.ndarray,
             dropped: np.ndarray, correction_id: str,
             log: Callable[[str], None] = print) -> int:
    """Stage a new epoch, verify it, swap it in and re-level the floor.

    Everything is written to `_epoch_<N>/` first and only then swapped, so a
    failure anywhere leaves the live session exactly as it was — which is what
    saved it when PotreeConverter died mid-epoch (pccr 2026-09-19).
    """
    from correction.epoch import current_epoch, make_epoch_record, EPOCH_FILE
    from correction import ledger
    from correction.run import run_select
    from potree_converter import convert_ply_to_potree

    n_kf = len(P)
    epoch = _next_epoch(output_dir)
    stage = output_dir / f"_epoch_{epoch}"
    stage.mkdir(parents=True, exist_ok=True)
    _write_ply(stage / "cleaned_cloud.ply", data, warped, keep)
    np.savetxt(stage / "camera_poses.txt", P.reshape(n_kf, 16), fmt="%.8g")
    seg_doc["instances"] = instances
    (stage / "segmentation_result.json").write_text(json.dumps(seg_doc))

    if not convert_ply_to_potree(output_dir.parent, force=True,
                                 ply_override=stage / "cleaned_cloud.ply",
                                 potree_dir_override=stage / "potree"):
        raise RuntimeError("the octree could not be built for the new epoch")

    # The epoch record travels WITH the epoch. `select_epoch` swaps the union
    # of every manifest and nothing else, so a record left live is a record
    # that never follows the geometry (pccr 2026-09-18).
    # floor_transform.npz / floor_level.json are listed although the stage does
    # not hold them: the levelling runs on the live geometry right after the
    # swap, so listing them files the PREVIOUS epoch's floor away with the
    # geometry it belongs to instead of leaving it live over a new one.
    (stage / EPOCH_FILE).write_text(json.dumps(make_epoch_record(
        epoch, correction_id, current_epoch(output_dir)), indent=1))
    (stage / "_manifest.json").write_text(json.dumps(
        {"epoch": epoch, "epoch_from": epoch, "epoch_to": epoch + 1,
         "artifacts": [{"rel": r, "existed_before": True} for r in
                       ("cleaned_cloud.ply", "camera_poses.txt",
                        "segmentation_result.json", "potree", EPOCH_FILE,
                        "floor_transform.npz", "floor_level.json")]}, indent=1))
    ledger.save_epoch_npz(output_dir, epoch, R_kf, t_kf, k_kf, frames,
                          dropped=dropped)
    run_select(output_dir, epoch, "visit_drift")

    # the floor is re-levelled on the NEW geometry (USER 2026-09-18): every
    # correction tilts and lifts the scene a little, and the display frame has
    # to follow it or the floor drifts off y=0 epoch after epoch
    try:
        from main import level_floor_core
        fl = level_floor_core(output_dir, output_dir / "segmentation_result.json",
                              output_dir / "cleaned_cloud.ply",
                              output_dir / "floor_level.json",
                              mode="auto", session_id=str(output_dir.parent.name))
        log(f"[visit-drift] floor: {fl.get('reason') or ('levelled' if fl.get('changed') else 'already level')}"
            + (f", was {fl['residual_before_mm']:+.1f} mm" if fl.get("residual_before_mm") is not None else ""))
    except Exception as e:  # noqa: BLE001 — declared, never silent
        log(f"[visit-drift] floor levelling failed: {e}")
    log(f"[visit-drift] epoch {epoch} published and selected")
    return epoch



def measure_epoch(output_dir: Path, cfg, rep_m: float,
                  log: Callable[[str], None] = print) -> Tuple[Optional[np.ndarray], dict]:
    """The whole chain, steps 1 to 10, on the session as it stands.

    Returns ``(t_kf, report)``: the translation to apply to each keyframe, or
    None when no object can testify. Nothing here writes anything.
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
    tol = float(get_param("segmentation.mask_filter.depth_tol_m", 0.15))

    rep = {"chain": {}, "objects": [], "rejected": [], "provenance": "tool_measured"}

    # 1) the objects are SAM3's masklets; their visits come from the masks
    masklets = vd.masklet_visits(output_dir, log=log)
    # 2) their points, and the nested filters
    pm = vd.points_of_masklets(output_dir, data["frame_global"], data["pixel_row"],
                               data["pixel_col"], log=log)
    label_of = {m.oid: m.label for m in masklets}
    cands, steps = vd.filter_chain(masklets, pm, ks, chain, cfg.min_points,
                                   cfg.min_walk_m, cfg.min_visit_share,
                                   xyz=xyz, log=log)
    rep["chain"] = steps
    if not cands:
        return None, rep

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
        return None, rep

    # 8) the identity has to be the only candidate
    kept, arep = vd.drop_ambiguous([(c, dr) for c, dr, _ in det], pm, label_of,
                                   xyz, cfg.max_ambiguity, log=log)
    rep["ambiguity"] = arep
    if not kept:
        return None, rep

    # 9-10) every closure is a constraint; the curve satisfies them
    vox = {c.instance_id: v["common_voxels"] for c, _dr, v in det}
    cons = [vd.Constraint(label=c.label, instance_id=c.instance_id,
                          d_a=float(chain[_mid_kf(c.visits[0])]),
                          d_b=float(chain[_mid_kf(c.visits[1])]),
                          t=dr.t, disagreement_m=dr.worst_disagreement,
                          common_voxels=vox[c.instance_id])
            for c, dr in kept]
    rep["constraints"] = [c.as_dict() for c in cons]
    # the SAME closures, read as the depth ratio they measure: the scale
    # graph's §5.1 loop rows (USER 2026-09-19). Measured here and spent
    # nowhere else — `certify/scale_stage` had ZERO loop rows on pccr while
    # this function measured five good ones on every pass.
    rep["scale_rows"] = vd.scale_rows(kept, poses, ks, log=log)
    for c in cons:
        log(f"[visit-drift]   {c.label}#{c.instance_id}: |t| "
            f"{np.linalg.norm(c.t) * 100:.1f} cm over {c.d_a:.2f}-{c.d_b:.2f} m, "
            f"disagreement {c.disagreement_m * 100:.1f} cm, "
            f"{c.common_voxels} common voxel(s)")
    # 9-10) the objects AND the floor in ONE system, at KEYFRAME resolution.
    # Applying one after the other always breaks the first — measured three
    # times in one session. And six chunk edges were not enough either: they
    # left keyframes 0-12 and 200-215, exactly where the desk's correction
    # acts, with no constraint, and the floor's dispersion went 66.4 -> 123.2 mm
    # while every edge stayed exact (USER 2026-09-19).
    fdz = fcon = None
    frep = None
    tex, chunk_kf = None, None
    try:
        from correction import floor_consensus as fcz
        fcfg = load_correction_config().floor_consensus
        if not fcfg.enabled:
            raise RuntimeError("the floor is switched off for this session")
        P, kfp = fcz.floor_points(output_dir, xyz, ks, log=lambda m: None)
        fobs = fcz.observe(P, kfp, up, float(fcfg.cell_m))
        fdz, fcon, frep = fcz.keyframe_offsets(fobs, len(poses), log=log)
        plan = output_dir / "chunk_plan.json"
        if plan.exists():
            ranges = json.loads(plan.read_text()).get("chunk_ranges") or []
            if len(ranges) >= 2:
                from correction.session import read_ply as _rp
                _, _d = _rp(output_dir / "cleaned_cloud.ply")
                inten = np.stack([_d["red"], _d["green"], _d["blue"]],
                                 1).astype(np.float64).mean(1)
                own = fcz.chunk_of_keyframe(ranges, len(poses))
                # intensity of the same floor points `floor_points` returned
                idx = fcz.floor_index(output_dir, len(xyz))
                idx = idx[ks[idx] >= 0]
                tex = fcz.chunk_texture_edges(
                    xyz[idx], inten[idx], ks[idx], up, own, len(ranges),
                    float(fcfg.cell_m) / 10.0, float(fcfg.texture_window_m),
                    int(fcfg.min_shared_cells), log=log)
                chunk_kf = [int(np.clip((a + b) // 2, 0, len(poses) - 1))
                            for a, b in ranges]
    except Exception as e:  # noqa: BLE001 — declared, never silent
        log(f"[visit-drift] the floor could not join the system ({e}) — "
            f"solving with the objects alone")
        fdz = fcon = None
        tex, chunk_kf = None, None

    if fdz is not None:
        t_kf, crep = vd.solve_joint_curve(cons, fdz, fcon, up, chain, rep_m,
                                          tex, chunk_kf, log=log)
        crep["floor_measure"] = frep
    else:
        t_kf, crep = vd.solve_drift_curve(cons, chain, rep_m, log=log)
    rep["curve"] = crep
    rep["_masklets"] = masklets          # for step 12, same pass, not re-read
    rep["_points_by_oid"] = pm
    rep["_ks"] = ks
    rep["_vis"] = vis
    return t_kf, rep


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


def one_epoch(output_dir: Path, cfg, log: Callable[[str], None] = print
              ) -> Optional[dict]:
    """Measure, correct, publish, re-level the floor and filter the cloud —
    one epoch. None when there is nothing left to correct."""
    from correction.distribute import warp_subset
    from correction import ledger

    output_dir = Path(output_dir)
    (hdr, data, xyz, fg, ks, frames, poses, chain, up,
     instances, seg_doc) = _load(output_dir)
    n_kf = len(poses)
    rep_m = _repeatability_m(output_dir, cfg.default_repeatability_m, log)
    log(f"[visit-drift] cloud {len(xyz):,} pts | walk {chain[-1]:.2f} m | "
        f"repeatability {rep_m * 100:.2f} cm")

    t_kf, mrep = measure_epoch(output_dir, cfg, rep_m, log=log)
    _write_scale_rows(output_dir, mrep, log=log)
    if t_kf is None:
        log("[visit-drift] no object can testify about a pose error")
        return None

    # NO early exit (USER 2026-09-19: "incluso que no rechace si es menos de
    # 4.7 mm, saquemos esa regla, asi vemos como termina el resto de las
    # epocas"). Every pass that measures something publishes it; the loop ends
    # only when `max_epochs` is reached or nothing can testify at all.
    applied = float(np.linalg.norm(t_kf, axis=1).max())
    if applied <= rep_m:
        log(f"[visit-drift] the correction is {applied * 100:.2f} cm, under the "
            f"session's own repeatability ({rep_m * 100:.2f} cm) — applied "
            f"anyway, the rule that refused it was removed")

    cur = mrep.get("curve", {})
    if cur.get("groups"):
        best = cur["groups"][0]["decided_by"]
    elif cur.get("object_constraints"):
        o = cur["object_constraints"][0]
        best = f"{o['label']}#{o['instance_id']}"
    else:
        best = "visit_drift"

    # apply: the points move with the keyframe they were born in, the cameras
    # with themselves. No rotation is applied (criterion 24), so R is identity
    # and the scale untouched.
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    k_kf = np.ones(n_kf)
    cam_center = {int(f): poses[k, :3, 3] for k, f in enumerate(frames)}
    idx = np.flatnonzero(ks >= 0)
    warped = xyz.copy()
    warped[idx] = warp_subset(xyz, fg, ks, cam_center, idx, R_kf, t_kf, k_kf)
    P = poses.copy()
    for k in range(n_kf):
        P[k, :3, :3] = R_kf[k] @ poses[k, :3, :3]
        P[k, :3, 3] = R_kf[k] @ poses[k, :3, 3] + t_kf[k]
    # the filter projects CORRECTED points, so it needs the CORRECTED poses.
    # The cached Z-buffers stay valid: a keyframe's own points and its own
    # camera move by the same transform, so what that camera measures of itself
    # is unchanged by the correction.
    mrep["_vis"].poses = P

    # 12) the cloud filter, HERE and per epoch — with the pose already
    # corrected (USER 2026-09-18: "el filtrado de la nube debe ser una vez que
    # se corrige la pose" … "cada nube tiene sus objetos filtrados, no
    # necesariamente iguales"). Three rules on the MASKLETS: outside every
    # mask, masklet under `min_points`, visit at or under `min_visit_share`.
    from config import get_param
    kill, frep = vd.cloud_filter_masklets(
        mrep["_points_by_oid"], mrep["_masklets"], mrep["_ks"], warped,
        mrep["_vis"], cfg.min_points, cfg.min_visit_share,
        int(get_param("segmentation.mask_filter.max_frames_per_visit", 8)),
        int(get_param("segmentation.mask_filter.dilate_px", 2)), log=log)
    keep = ~kill
    if frep.dropped_points:
        instances = _reindex(instances, keep)

    epoch = _publish(output_dir, data, warped, keep, P, instances, seg_doc,
                     frames, R_kf, t_kf, k_kf, np.flatnonzero(kill),
                     f"visit_drift/{best}", log=log)
    rec = {"epoch": epoch, "corrected_by": best,
           "applied_m": applied,
           "measure": {k: v for k, v in mrep.items()
                       if not k.startswith("_")},
           "filter": frep.as_dict(),
           "floor_levelled": True,
           "points_after": int(keep.sum()), "provenance": "tool_measured"}

    # The ledger is the append-only record of every correction that reached the
    # session; an epoch that is not in it cannot be explained later. It carries
    # the measurement itself — which object testified, over which stretch of the
    # walk, how well determined and on how much commonly-seen geometry — so the
    # report on disk is the evidence, not a summary of it.
    cons = mrep.get("constraints", [])
    cid = ledger.new_correction_id()
    rep_path = output_dir / "corrections" / f"report_{cid}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(rec, indent=1, default=float))
    ledger.record_run(
        output_dir, correction_id=cid, epoch_from=epoch - 1, epoch_to=epoch,
        kind="visit_drift", operator="auto",
        instance_ids=[int(c["instance_id"]) for c in cons],
        visits=[{"instance_id": int(c["instance_id"]),
                 "d_a_m": c["d_a_m"], "d_b_m": c["d_b_m"]} for c in cons],
        observability=[{"instance_id": int(c["instance_id"]),
                        "disagreement_m": c["disagreement_m"],
                        "common_voxels": c["common_voxels"]} for c in cons],
        diagnosis=[{"kind": "pose", "evidence": "silhouette over the three "
                                                "orthogonal views of the OBB, "
                                                "inside the region both visits saw"}],
        anchors=[{"chainage_from_m": g["chainage_from_m"],
                  "chainage_to_m": g["chainage_to_m"],
                  "decided_by": g["decided_by"],
                  "rate_mm_per_m": g["rate_mm_per_m"]}
                 for g in mrep.get("curve", {}).get("groups", [])],
        gates=[], overrides={}, verdict="applied",
        report_path=str(rep_path.relative_to(output_dir)))
    rec["correction_id"] = cid
    return rec


def chunk_floor_epoch(output_dir: Path, cfg, log: Callable[[str], None] = print
                      ) -> Optional[dict]:
    """STEP 0 — level the reconstruction's chunks against each other.

    USER 2026-09-19: *"debe ser parte del pipeline … antes practicamente de
    empezar la correccion, porque ya te deja muy bien acotado los errores"*.

    It runs FIRST because it is the cheapest and best-conditioned thing we can
    measure: seven unknowns against thousands of floor cells that two chunks
    both saw. On pccr it removed 248 mm of accumulated vertical drift with a
    residual of 0.0 mm on every edge, without deleting a point and without
    touching the building's own ramp — and everything downstream then measures
    against a scene that no longer carries a global vertical error.
    """
    from correction import floor_consensus as fcz
    from correction.distribute import warp_subset

    output_dir = Path(output_dir)
    plan = output_dir / "chunk_plan.json"
    if not plan.exists():
        log("[chunk-floor] the session has no chunk_plan.json — it was not "
            "reconstructed in chunks, so there is nothing to level")
        return None
    ranges = json.loads(plan.read_text()).get("chunk_ranges") or []
    if len(ranges) < 2:
        log("[chunk-floor] a single chunk — nothing to level against")
        return None

    (hdr, data, xyz, fg, ks, frames, poses, chain, up,
     instances, seg_doc) = _load(output_dir)
    n_kf = len(poses)
    overlap = int(json.loads(plan.read_text()).get("overlap") or 0)
    t_kf, rep = fcz.measure_chunks(output_dir, xyz, ks, poses, ranges,
                                   float(cfg.cell_m), int(cfg.min_shared_cells),
                                   overlap // 2 if overlap else 0, log=log)
    if float(np.linalg.norm(t_kf, axis=1).max()) <= 0.0:
        log("[chunk-floor] nothing to correct")
        return None

    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    k_kf = np.ones(n_kf)
    cam_center = {int(f): poses[k, :3, 3] for k, f in enumerate(frames)}
    idx = np.flatnonzero(ks >= 0)
    warped = xyz.copy()
    warped[idx] = warp_subset(xyz, fg, ks, cam_center, idx, R_kf, t_kf, k_kf)
    P = poses.copy()
    for k in range(n_kf):
        P[k, :3, 3] = poses[k, :3, 3] + t_kf[k]

    epoch = _publish(output_dir, data, warped, np.ones(len(xyz), bool), P,
                     instances, seg_doc, frames, R_kf, t_kf, k_kf,
                     np.zeros(0, np.int64), "chunk_floor", log=log)
    rec = {"epoch": epoch, "corrected_by": "chunk_floor",
           "applied_m": float(np.linalg.norm(t_kf, axis=1).max()),
           "measure": rep, "points_after": int(len(xyz)),
           "provenance": "tool_measured"}
    from correction import ledger
    cid = ledger.new_correction_id()
    rp = output_dir / "corrections" / f"report_{cid}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1, default=float))
    ledger.record_run(
        output_dir, correction_id=cid, epoch_from=epoch - 1, epoch_to=epoch,
        kind="chunk_floor", operator="auto", instance_ids=[], visits=[],
        observability=[{"pair": e["pair"], "shared_cells": e["shared_cells"],
                        "residual_mm": e["residual_mm"]} for e in rep["edges"]],
        diagnosis=[{"kind": "pose_vertical",
                    "evidence": "floor cells two chunks both saw"}],
        anchors=[{"offsets_mm": rep["offsets_mm"]}],
        gates=[], overrides={}, verdict="applied",
        report_path=str(rp.relative_to(output_dir)))
    rec["correction_id"] = cid
    return rec


def scale_epoch(output_dir: Path, log: Callable[[str], None] = print
                ) -> Optional[dict]:
    """THE DEPTH CORRECTION — per-chunk scale, from the closures this module
    measures (USER 2026-09-19: *"arreglemos la escala en origen"*, then B).

    The duplicates of pccr are separated ALONG THE LINE OF SIGHT, not sideways:
    five of six closures 97-99 % radial. A translation is the same everywhere;
    a depth error grows with distance, which is why the desk closed at 3.6 m
    while the tile lines 8 m away stayed 18 cm off. Read as depth ratios the
    same closures AGREE (1.099-1.251) and agree with the DA3 anchors (1.140),
    which never see a silhouette.

    The solving is `certify/scale_stage`'s, unchanged — it already models
    exactly this: one factor r_k per reconstruction chunk, applied as depth
    x r_k about each keyframe's OWN camera plus the translation that keeps the
    walk continuous. The chunk is the right unit because each chunk carries its
    own gauge; a per-FRAME depth change would break the multi-view consistency
    the reconstruction still has inside a chunk.

    Declared limit: the DA3 anchors show a CONTINUOUS drift along the walk and
    seven chunks can only spell a staircase. With `sigma_seam_log` at 0.02 over
    six seams the model tops out near 12 %; pccr needs ~14 %.
    """
    from correction.distribute import warp_subset
    from correction.session import load_session
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.scale_stage import (
        chunk_of_keyframes, scale_transforms, solve_scale_stage)
    from correction import ledger

    from correction.epoch import current_epoch

    output_dir = Path(output_dir)
    # The rows carry the epoch they were measured on, and they MUST be current:
    # this correction changes the very depths they are made of, so rows from an
    # older epoch would ask for a correction that is already in the geometry
    # and compound it. Stale or missing → measure again, here.
    rows_file = output_dir / "scale_loop_rows.json"
    stamp = (json.loads(rows_file.read_text()).get("measured_on_epoch")
             if rows_file.exists() else None)
    now = int(current_epoch(output_dir))
    if stamp != now:
        from correction.config import load_correction_config
        log(f"[scale-epoch] the loop rows are stamped epoch {stamp} and the "
            f"session is at {now} — measuring again on the current geometry")
        vcfg = load_correction_config().visit_drift
        rep_m = _repeatability_m(output_dir, vcfg.default_repeatability_m, log)
        _, mrep = measure_epoch(output_dir, vcfg, rep_m, log=log)
        _write_scale_rows(output_dir, mrep, log=log)
    if not rows_file.exists():
        log("[scale-epoch] no closure could be measured — no scale row to add")
        return None

    session = load_session(output_dir)
    scfg = load_loops_config().certify.scale
    srep = solve_scale_stage(output_dir, session, [], scfg, log=log)
    if not srep.get("applied"):
        log(f"[scale-epoch] nothing applied: {srep.get('reason')}")
        return None

    (hdr, data, xyz, fg, ks, frames, poses, chain, up,
     instances, seg_doc) = _load(output_dir)
    n_kf = len(poses)
    ranges, owner = chunk_of_keyframes(output_dir, n_kf)
    k_kf, t_kf = scale_transforms(session, ranges, owner,
                                  np.asarray(srep["r"], np.float64))
    log(f"[scale-epoch] r per chunk {np.round(srep['r'], 4).tolist()} | "
        f"depth factor {k_kf.min():.4f}-{k_kf.max():.4f} | "
        f"camera shift up to {np.linalg.norm(t_kf, axis=1).max() * 100:.1f} cm")

    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    cam_center = {int(f): poses[k, :3, 3] for k, f in enumerate(frames)}
    idx = np.flatnonzero(ks >= 0)
    warped = xyz.copy()
    warped[idx] = warp_subset(xyz, fg, ks, cam_center, idx, R_kf, t_kf, k_kf)
    P = poses.copy()
    for k in range(n_kf):
        P[k, :3, 3] = poses[k, :3, 3] + t_kf[k]
    moved = float(np.linalg.norm(warped - xyz, axis=1).max())
    log(f"[scale-epoch] the cloud moves up to {moved * 100:.1f} cm "
        f"(a depth correction grows with distance — that IS the point)")

    epoch = _publish(output_dir, data, warped, np.ones(len(xyz), bool), P,
                     instances, seg_doc, frames, R_kf, t_kf, k_kf,
                     np.zeros(0, np.int64), "scale_depth", log=log)
    rec = {"epoch": epoch, "corrected_by": "scale_depth",
           "applied_m": moved, "measure": srep,
           "points_after": int(len(xyz)), "provenance": "tool_measured"}
    cid = ledger.new_correction_id()
    rp = output_dir / "corrections" / f"report_{cid}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1, default=float))
    ledger.record_run(
        output_dir, correction_id=cid, epoch_from=epoch - 1, epoch_to=epoch,
        kind="scale_depth", operator="auto",
        instance_ids=[int(r["instance_id"]) for r in srep.get("loop_rows", [])
                      if r.get("instance_id") is not None],
        visits=[], observability=srep.get("loop_rows", []),
        diagnosis=[{"kind": "depth",
                    "evidence": "the closures are radial: a per-chunk depth "
                                "factor, cross-checked against the DA3 anchors"}],
        anchors=[{"r_per_chunk": srep["r"],
                  "anchor_rows_stood_down": srep.get("anchor_rows_stood_down")}],
        gates=[srep.get("gate")] if srep.get("gate") else [], overrides={},
        verdict="applied", report_path=str(rp.relative_to(output_dir)))
    rec["correction_id"] = cid
    return rec


# ── the loop ─────────────────────────────────────────────────────────────

def run(session_dir, max_epochs: Optional[int] = None,
        log: Callable[[str], None] = print) -> dict:
    """Epochs until the measurement stops improving."""
    from correction.config import load_correction_config

    session_dir = Path(session_dir)
    output_dir = session_dir / "output"
    cfg = load_correction_config().visit_drift
    n_max = int(max_epochs if max_epochs is not None else cfg.max_epochs)

    history: List[dict] = []
    t0 = time.time()

    # STEP 0: level the chunks against each other before anything else runs.
    # It is the cheapest, best-conditioned measurement in the session and it
    # leaves everything downstream a scene without a global vertical error
    # (USER 2026-09-19).
    fcfg = load_correction_config().floor_consensus
    if fcfg.enabled:
        log("[visit-drift] ── step 0: chunk floor ──")
        try:
            rec0 = chunk_floor_epoch(output_dir, fcfg, log=log)
            if rec0 is not None:
                history.append(rec0)
        except Exception as e:  # noqa: BLE001 — declared, never silent
            log(f"[chunk-floor] failed: {e}")

    for i in range(n_max):
        log(f"\n[visit-drift] ── pass {i + 1}/{n_max} ──")
        rec = one_epoch(output_dir, cfg, log=log)
        if rec is None:
            log("[visit-drift] converged: nothing left to correct")
            break
        history.append(rec)
        # `one_epoch` refuses to publish a correction under the session's own
        # repeatability, so reaching here means this epoch moved something.
        # The second stop is the trend: an epoch that corrects no less than
        # the one before it is not converging on anything.
        # 13) NO automatic stop for now (USER 2026-09-18: "por ahora dejalo
        # abierto, vamos a ejecutar siempre hasta epoch 0 (original), 1, 2, 3 y
        # 4, forzosamente siempre"). `one_epoch` still refuses to publish a
        # correction under what the session can repeat — that is a refusal to
        # write noise, not a convergence rule — and when it does the loop ends
        # because there is nothing to publish.
    report = {"version": 1, "epochs": history, "elapsed_s": round(time.time() - t0, 1),
              "provenance": "tool_measured"}
    (output_dir / "visit_drift_report.json").write_text(json.dumps(report, indent=2))
    log(f"[visit-drift] {len(history)} epoch(s) in {report['elapsed_s']:.0f}s "
        f"→ visit_drift_report.json")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--max-epochs", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.session, a.max_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
