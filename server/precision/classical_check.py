"""The classical cross-check — COLMAP 4.x / GLOMAP as an INDEPENDENT photogrammetry of the same
keyframes (audit 2026-10-23, phase 3d; claude_stac.txt plan h2; USER: "lo mismo con colmap …
usa lo mejor que existe").

Two things, two moments:
  1. BEFORE F4 (the chain's step f3_classical): SIFT features, sequential matching with the
     vocabulary-tree loop detection, the global mapper — ``evaluation.colmap_baseline``'s own
     commands (the battery's h2), in its own env, no learned model anywhere. Every VERIFIED
     two-view geometry between keyframes far apart along the walk is a LOOP CANDIDATE
     (``loop_candidates.json``, stamped with the reconstruction id and the keyframe list): F4 opens
     a loop window on each, so the revisits SALAD missed still constrain the drift.
  2. AFTER F5 (``cross_check`` in refine.json): the classical trajectory Sim3-aligned to F5's over
     the registered keyframes (Umeyama with scale — COLMAP's metre is its own — one Huber
     reweighting at the median), the discrepancy per keyframe in metres and degrees, the registered
     share. Two photogrammetries that share nothing but the pixels agreeing within their errors
     CERTIFY the poses; where they part, the keyframes are named. An instrument, never a correction.

Outputs ``output/precision/classical/`` (the COLMAP workspace: database, sparse_<mapper>) and
``classical.json``; a report of another reconstruction or keyframe list is refused by name.

2026-10-31 (pccr_audit): COLMAP had been starting from its DEFAULT focal (1.2 × the long side: 998 px
against 364 measured) because the keyframe JPEGs carry no EXIF — GLOMAP's rotation averaging marked 156
pairs invalid and the walk came out in segments turned 177° / 90° / 66° (75° median against F5). The
session's own camera (camera.json, F0's at this step) is now handed to COLMAP as the prior in native
pixels (``camera_prior_params``), and BOTH mappers run on the one database (``classical.mappers``):
GLOMAP with the prior registered 295/295 keyframes in one model (5.3° / 11 cm median against F5, no
broken step), the incremental mapper 184/295 in 2–3 models. The cross-check reports every mapper and the
consecutive-keyframe rotation STEPS of each against F5's — a step of 90–180° in one trajectory where
the other steps a few degrees is a broken segment of that model, not drift.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DIRNAME = "classical"
REPORT_NAME = "classical.json"
LOOPS_NAME = "loop_candidates.json"
LOG_TAG = "[classical]"
VERSION = 2                      # 2: mapping per mapper, camera_prior, cross-check per mapper + steps
MAX_IMAGE_ID = 2147483647        # COLMAP's pair_id = image_id1 * MAX_IMAGE_ID + image_id2


class ClassicalError(RuntimeError):
    pass


def keyframe_list_sha(kf: Sequence[int]) -> str:
    return hashlib.sha256(" ".join(str(int(f)) for f in kf).encode()).hexdigest()


def pair_ids(pair_id: int) -> Tuple[int, int]:
    return int(pair_id // MAX_IMAGE_ID), int(pair_id % MAX_IMAGE_ID)


def loop_pairs_from_database(db_path: Path, kf: Sequence[int], min_inliers: int, min_gap: int
                             ) -> List[Tuple[int, int, int]]:
    """(i, j, inliers) over the verified two-view geometries of ``database.db`` between keyframes
    at least ``min_gap`` apart in the walk, with at least ``min_inliers`` inliers."""
    from evaluation.colmap_io import frame_of_name
    kidx = {int(f): i for i, f in enumerate(kf)}
    con = sqlite3.connect(str(db_path))
    try:
        names = {int(r[0]): str(r[1]) for r in con.execute("SELECT image_id, name FROM images")}
        rows = con.execute("SELECT pair_id, rows FROM two_view_geometries").fetchall()
    finally:
        con.close()
    out: List[Tuple[int, int, int]] = []
    for pid, n in rows:
        a, b = pair_ids(int(pid))
        fa, fb = frame_of_name(names.get(a, "")), frame_of_name(names.get(b, ""))
        if fa is None or fb is None or fa not in kidx or fb not in kidx:
            continue
        i, j = kidx[fa], kidx[fb]
        if abs(i - j) < int(min_gap) or int(n or 0) < int(min_inliers):
            continue
        out.append((min(i, j), max(i, j), int(n)))
    out = sorted(set(out))
    return out


def camera_prior_params(out_dir: Path) -> Tuple[List[float], Dict[str, Any]]:
    """The session's camera (``output/camera.json`` — F0's at the time of this step) as COLMAP's
    single-camera prior in NATIVE pixels: fx, fy, cx, cy taken from the record grid through the
    grid map (``K_grid_to_native``), the distortion as recorded (OPENCV's k1 k2 p1 p2 live in
    normalised coordinates: grid-free). The classical images are the native keyframes."""
    from precision.camera import K_grid_to_native, load_camera_json
    cam = load_camera_json(Path(out_dir) / "camera.json")
    if str(cam.model) != "OPENCV" or len(cam.params) != 8:
        raise ClassicalError(f"camera.json: model {cam.model!r} with {len(cam.params)} parameter(s) — the classical "
                             f"prior expects OPENCV (fx fy cx cy k1 k2 p1 p2)")
    fx, fy, cx, cy, k1, k2, p1, p2 = [float(v) for v in cam.params]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)
    Kn = K_grid_to_native(K, cam.omega_grid)
    params = [float(Kn[0, 0]), float(Kn[1, 1]), float(Kn[0, 2]), float(Kn[1, 2]), k1, k2, p1, p2]
    rec = {"used": True, "source": str(cam.source), "camera_epoch": int(cam.camera_epoch), "model": "OPENCV",
           "native_size": [int(cam.width), int(cam.height)], "grid": str(getattr(cam.omega_grid, "name", "")),
           "params_native": params, "params_grid": [fx, fy, cx, cy, k1, k2, p1, p2]}
    return params, rec


def run_classical(session_dir: Path, pcfg, log: Callable = print, runner: Optional[Callable] = None) -> Dict[str, Any]:
    """The step: features + matches (with the session camera as COLMAP's prior, ``classical.camera_prior``)
    + every configured mapper (``classical.mappers``) on the keyframes; the loop candidates and the
    report, stamped. ``runner(ws, files, ccfg, log, camera_params=, mappers=)``: replaces
    ``evaluation.colmap_baseline``'s stages (tests) and returns (features record, {mapper: mapping record})."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    from intake.quality import read_session_epochs
    from precision.config import raw_of
    session_dir = Path(session_dir)
    out = session_dir / "output"
    cc = pcfg.classical
    raw = raw_of(pcfg)
    try:
        ccfg = dict(raw["evaluation"]["colmap"])
    except (KeyError, TypeError) as e:
        raise ClassicalError("config.yaml: evaluation.colmap (the classical baseline's configuration) is missing") from e
    sel = session_dir / "frames" / "selected_frames.json"
    if not sel.exists():
        raise ClassicalError(f"{sel} is missing — the keyframes are not selected")
    files = sorted(json.loads(sel.read_text()).get("selected_files", []))
    from evaluation.colmap_io import frame_of_name
    kf = [frame_of_name(n) for n in files]
    if any(f is None for f in kf) or len(kf) < 3:
        raise ClassicalError(f"{sel}: {len(kf)} keyframe file(s), need 3 with a frame number each")
    ws = out / "precision" / DIRNAME
    ws.mkdir(parents=True, exist_ok=True)
    mappers = [str(m) for m in cc.mappers]
    if not mappers:
        raise ClassicalError("precision.classical.mappers is empty — name at least one mapper")
    prior, prior_rec = (camera_prior_params(out) if bool(cc.camera_prior)
                        else (None, {"used": False, "reason": "classical.camera_prior: false — COLMAP's default focal"}))
    if prior is not None:
        log(f"{LOG_TAG} camera prior for COLMAP (native px, {prior_rec['source']} @ camera epoch {prior_rec['camera_epoch']}): "
            f"fx {prior[0]:.2f} fy {prior[1]:.2f} cx {prior[2]:.2f} cy {prior[3]:.2f}")
    else:
        log(f"{LOG_TAG} no camera prior for COLMAP (classical.camera_prior: false) — its default focal")
    t0 = time.time()
    if runner is None:
        from evaluation import colmap_baseline as CB
        CB.prepare_images(session_dir / "frames", files, ws)
        feat = CB.ensure_features_and_matches(ws, "SIFT", "SIFT_BRUTEFORCE", ccfg, log, camera_params=prior)
        mapped = {m: CB.map_sparse(ws, m, ccfg, log) for m in mappers}
    else:
        feat, mapped = runner(ws, files, ccfg, log, camera_params=prior, mappers=mappers)
    if not isinstance(mapped, dict) or set(mapped) != set(mappers):
        raise ClassicalError(f"the mapping records do not match the configured mappers {mappers}: "
                             f"{sorted(mapped) if isinstance(mapped, dict) else type(mapped).__name__}")
    largest: Dict[str, Dict[str, Any]] = {}
    for m in mappers:
        models = sorted(mapped[m].get("models", []), key=lambda d: -int(d["n_images"]))
        if not models:
            raise ClassicalError(f"the {m} mapper registered no model")
        largest[m] = models[0]
    pairs = loop_pairs_from_database(ws / "database.db", kf, int(cc.loop_min_inliers), int(cc.loop_min_gap_keyframes))
    epochs = read_session_epochs(session_dir)
    rid = reconstruction_id_or_none(out)
    stamp = {RECONSTRUCTION_ID_KEY: rid, "keyframes_sha256": keyframe_list_sha(kf), "n_keyframes": len(kf), **epochs}
    loops = {"version": VERSION, "stage": "classical", "provenance": "tool_measured", **stamp,
             "rule": f"verified two-view geometries with >= {int(cc.loop_min_inliers)} inliers between keyframes "
                     f">= {int(cc.loop_min_gap_keyframes)} apart (COLMAP sequential + vocabulary-tree matching)",
             "pairs": [[int(kf[i]), int(kf[j]), int(n)] for i, j, n in pairs]}
    (out / "precision" / LOOPS_NAME).write_text(json.dumps(loops, indent=1))
    rep = {"version": VERSION, "stage": "classical", "provenance": "tool_measured", **stamp,
           "camera_prior": prior_rec, "features": feat, "mappers": mappers, "mapping": mapped,
           "largest_model": largest,
           "registered_share": {m: float(largest[m]["n_images"]) / float(len(kf)) for m in mappers},
           "loop_candidates": len(pairs), "seconds": round(time.time() - t0, 1)}
    (ws / REPORT_NAME).write_text(json.dumps(rep, indent=1, default=float))
    log(f"{LOG_TAG} " + "; ".join(f"{m} mapper: {len(mapped[m].get('models', []))} model(s), the largest "
                                  f"{largest[m]['n_images']}/{len(kf)} keyframes" for m in mappers)
        + f"; {len(pairs)} loop candidate(s) >= {cc.loop_min_gap_keyframes} keyframes apart → {out / 'precision' / LOOPS_NAME}")
    return rep


def classical_loops(out_dir: Path, kf: Sequence[int], rid: Optional[str]) -> Tuple[List[Tuple[int, int]], dict]:
    """F4's reader: the loop candidates as keyframe-INDEX pairs when the file belongs to THIS
    reconstruction and THIS keyframe list; otherwise none, with the reason."""
    from correction.epoch import same_reconstruction
    p = Path(out_dir) / "precision" / LOOPS_NAME
    if not p.exists():
        return [], {"present": False, "taken": False, "reason": "no loop_candidates.json (f3_classical did not run)"}
    doc = json.loads(p.read_text())
    taken, why = same_reconstruction(doc, rid)
    if not taken:
        return [], {"present": True, "taken": False, "reason": why}
    if doc.get("keyframes_sha256") != keyframe_list_sha(kf):
        return [], {"present": True, "taken": False, "reason": "measured on another keyframe list"}
    kidx = {int(f): i for i, f in enumerate(kf)}
    pairs = [(kidx[int(a)], kidx[int(b)]) for a, b, _n in doc.get("pairs", []) if int(a) in kidx and int(b) in kidx]
    return pairs, {"present": True, "taken": True, "n_pairs": len(pairs), "rule": doc.get("rule")}


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) with dst ≈ s R src + t, weighted."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if w is None else np.asarray(w, np.float64)
    w = w / w.sum()
    ms = (w[:, None] * src).sum(0); md = (w[:, None] * dst).sum(0)
    S = src - ms; D = dst - md
    H = (S * w[:, None]).T @ D
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ Dm @ U.T
    var_s = float(np.sum(w[:, None] * S ** 2))
    s = float(np.sum(sig * np.diag(Dm)) / var_s) if var_s > 0 else 1.0
    t = md - s * (R @ ms)
    return s, R, t


def _check_model(model, kf: Sequence[int], w2c_f5: np.ndarray, n_models: int) -> Dict[str, Any]:
    """One classical model against F5: Sim3 on the registered keyframe centres (one Huber reweighting
    at the median), the per-keyframe rotation discrepancy after the chordal-mean frame rotation, the
    worst keyframes — and the consecutive-keyframe rotation STEPS of both trajectories."""
    from evaluation.colmap_io import frame_of_name
    from precision.scale_lock import centres_of
    kidx = {int(f): i for i, f in enumerate(kf)}
    rows = []
    for im in model.images.values():
        f = frame_of_name(im.name)
        if f is None or f not in kidx:
            continue
        W = np.asarray(im.w2c(), np.float64)
        rows.append((kidx[f], W[:3, :3], W[:3, 3]))
    rows.sort(key=lambda r: r[0])
    if len(rows) < 3:
        return {"measured": False, "reason": f"{len(rows)} registered keyframe(s), need 3", "n_models": int(n_models)}
    idx = np.array([r[0] for r in rows])
    Rc = np.stack([r[1] for r in rows]); tc = np.stack([r[2] for r in rows])
    C_cl = -np.einsum("nji,nj->ni", Rc, tc)                     # classical centres (its own metre)
    W5 = np.asarray(w2c_f5, np.float64)[idx]
    C_f5 = centres_of(W5)
    s, R, t = umeyama_sim3(C_cl, C_f5)
    res = np.linalg.norm((s * (C_cl @ R.T) + t) - C_f5, axis=1)
    med = float(np.median(res))
    if med > 0.0:
        w = np.minimum(1.0, med / np.maximum(res, np.finfo(float).tiny))
        s, R, t = umeyama_sim3(C_cl, C_f5, w)
        res = np.linalg.norm((s * (C_cl @ R.T) + t) - C_f5, axis=1)
    # rotation: the frame rotation between the two worlds from the ROTATIONS themselves — the
    # chordal mean of classical_kᵀ · f5_k over the keyframes (one Huber reweighting) — then the
    # per-keyframe discrepancy angle(classical_k · A vs f5_k), independent of the centres' fit
    def chordal_mean(ws):
        M = sum(w * (Rc[k].T @ W5[k, :3, :3]) for k, w in enumerate(ws))
        U, _s, Vt = np.linalg.svd(M)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt

    def angle(Rd) -> float:
        return float(np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)))

    def angles(A):
        return np.asarray([angle((Rc[k] @ A) @ W5[k, :3, :3].T) for k in range(len(rows))])
    A = chordal_mean(np.ones(len(rows)))
    a0 = angles(A)
    ma = float(np.median(a0))
    if ma > 0.0:
        A = chordal_mean(np.minimum(1.0, ma / np.maximum(a0, np.finfo(float).tiny)))
    ang = np.degrees(angles(A))
    worst = np.argsort(-res)[:5]
    # the consecutive steps: registered keyframes that follow each other in the keyframe list
    cons = np.flatnonzero(idx[1:] - idx[:-1] == 1)
    step_cl = np.degrees([angle(Rc[k + 1] @ Rc[k].T) for k in cons]) if len(cons) else np.zeros(0)
    step_f5 = np.degrees([angle(W5[k + 1, :3, :3] @ W5[k, :3, :3].T) for k in cons]) if len(cons) else np.zeros(0)
    dstep = np.abs(step_cl - step_f5)
    steps: Dict[str, Any] = {"n_steps": int(len(cons))}
    if len(cons):
        order = np.argsort(-dstep)[:5]
        steps.update({"classical_median_deg": float(np.median(step_cl)), "f5_median_deg": float(np.median(step_f5)),
                      "abs_diff_deg": {"median": float(np.median(dstep)), "p90": float(np.percentile(dstep, 90)),
                                       "max": float(dstep.max())},
                      "worst": [{"from_frame": int(kf[idx[cons[k]]]), "to_frame": int(kf[idx[cons[k] + 1]]),
                                 "classical_step_deg": float(step_cl[k]), "f5_step_deg": float(step_f5[k])}
                                for k in order],
                      "reading": "a step of 90-180 deg in ONE trajectory where the other steps a few degrees is a "
                                 "broken segment of that model, not drift"})
    cams = list(model.cameras.values()) if getattr(model, "cameras", None) else []
    cam_rec = None
    if cams:
        pr = [float(v) for v in np.asarray(cams[0].params, np.float64).ravel()]
        cam_rec = {"model": str(getattr(cams[0], "model", "")), "params": pr}
    return {"measured": True, "n_models": int(n_models), "n_registered": int(len(rows)), "n_keyframes": int(len(kf)),
            "registered_share": float(len(rows)) / float(len(kf)),
            "alignment": {"scale_f5_over_classical": float(s), "method": "Umeyama Sim3 on the registered keyframe "
                                                                       "centres, one Huber reweighting at the median"},
            "position_m": {"median": float(np.median(res)), "p90": float(np.percentile(res, 90)),
                           "max": float(res.max())},
            "rotation_deg": {"median": float(np.median(ang)), "p90": float(np.percentile(ang, 90)),
                             "max": float(ang.max())},
            "worst_keyframes": [{"frame": int(kf[idx[k]]), "position_m": float(res[k]), "rotation_deg": float(ang[k])}
                                for k in worst],
            "steps": steps, "camera_refined": cam_rec,
            "reading": "two independent photogrammetries of the same pixels: where they agree within their errors "
                       "the poses are certified by both; the worst keyframes name where they part"}


def _load_classical(out_dir: Path, kf: Sequence[int], rid: Optional[str]):
    """(report, mapper names, workspace) of a classical run that belongs to THIS reconstruction and
    keyframe list, or (None, reason, workspace)."""
    from correction.epoch import same_reconstruction
    ws = Path(out_dir) / "precision" / DIRNAME
    rp = ws / REPORT_NAME
    if not rp.exists():
        return None, "no classical model (f3_classical did not run)", ws
    rep = json.loads(rp.read_text())
    taken, why = same_reconstruction(rep, rid)
    if not taken:
        return None, why, ws
    if rep.get("keyframes_sha256") != keyframe_list_sha(kf):
        return None, "the classical model was built on another keyframe list", ws
    return rep, [str(m) for m in (rep.get("mappers") or ["global"])], ws


def classical_init(out_dir: Path, kf: Sequence[int], w2c_gauge: np.ndarray, rid: Optional[str],
                   log: Callable = print) -> Tuple[np.ndarray, Dict[str, Any]]:
    """F5's STARTING poses from the classical trajectory (``refine.init_poses: classical``): the first
    configured mapper's largest model, taken only when it registers EVERY keyframe, Sim3-aligned to the
    gauge's trajectory (Umeyama on the camera centres, one Huber reweighting at the median — the
    gauge's metre and world frame, the classical SHAPE). Returns (w2c per keyframe, the record); a
    missing, foreign or partial model returns the gauge's poses with the reason (``taken: False``).

    Why (pccr_audit3, 2026-10-31): a bundle adjustment started at the drifted walk never closed the
    start-end loop although the MASt3R bridges held it (their two-view geometry agrees with GLOMAP
    under 1 deg; F5's final poses disagree with both by 5-6 deg) — it bent the weakest structure
    instead. GLOMAP's global solve has the loop consistent by construction."""
    from evaluation.colmap_io import frame_of_name, list_models, read_model
    from precision.scale_lock import centres_of
    W_g = np.asarray(w2c_gauge, np.float64)
    rep, mappers, ws = _load_classical(out_dir, kf, rid)
    if rep is None:
        log(f"{LOG_TAG} init_poses classical: {mappers} — the gauge's poses start the ladder")
        return W_g, {"source": "omega", "taken": False, "reason": str(mappers)}
    mapper = mappers[0]
    sparse = ws / f"sparse_{mapper}"
    dirs = list_models(sparse) if sparse.exists() else []
    if not dirs:
        why = f"{sparse}: no model"
        log(f"{LOG_TAG} init_poses classical: {why} — the gauge's poses start the ladder")
        return W_g, {"source": "omega", "taken": False, "mapper": mapper, "reason": why}
    model = max((read_model(d) for d in dirs), key=lambda mm: mm.n_images)
    kidx = {int(f): i for i, f in enumerate(kf)}
    rows = []
    for im in model.images.values():
        f = frame_of_name(im.name)
        if f is not None and f in kidx:
            W = np.asarray(im.w2c(), np.float64)
            rows.append((kidx[f], W[:3, :3], W[:3, 3]))
    rows.sort(key=lambda r: r[0])
    n_reg = len(rows)
    base = {"source": "classical", "mapper": mapper, "n_models": int(len(dirs)), "n_registered": int(n_reg),
            "n_keyframes": int(len(kf))}
    if n_reg != len(kf) or len({r[0] for r in rows}) != len(kf):
        why = (f"the {mapper} mapper's largest model registers {n_reg} of {len(kf)} keyframes "
               f"({len(dirs)} model(s)) — a partial trajectory starts nothing; the gauge's poses start the ladder")
        log(f"{LOG_TAG} init_poses classical: {why}")
        return W_g, {**base, "source": "omega", "taken": False, "reason": why}
    Rc = np.stack([r[1] for r in rows]); tc = np.stack([r[2] for r in rows])
    C_cl = -np.einsum("nji,nj->ni", Rc, tc)
    C_g = centres_of(W_g)
    s, R, t = umeyama_sim3(C_cl, C_g)
    res = np.linalg.norm((s * (C_cl @ R.T) + t) - C_g, axis=1)
    med = float(np.median(res))
    if med > 0.0:
        w = np.minimum(1.0, med / np.maximum(res, np.finfo(float).tiny))
        s, R, t = umeyama_sim3(C_cl, C_g, w)
        res = np.linalg.norm((s * (C_cl @ R.T) + t) - C_g, axis=1)
    # the Sim3 acting on the cameras: world' = s R world + t  →  R'_k = R_k Rᵀ, C'_k = s R C_k + t
    C_new = s * (C_cl @ R.T) + t
    R_new = np.einsum("nij,kj->nik", Rc, R)                      # R_k @ R.T
    W_new = np.tile(np.eye(4), (len(kf), 1, 1))
    W_new[:, :3, :3] = R_new
    W_new[:, :3, 3] = -np.einsum("nij,nj->ni", R_new, C_new)
    rot = np.degrees([float(np.arccos(np.clip((np.trace(R_new[k] @ W_g[k, :3, :3].T) - 1.0) / 2.0, -1.0, 1.0)))
                      for k in range(len(kf))])
    rec = {**base, "taken": True, "scale_gauge_over_classical": float(s),
           "centres_vs_gauge_m": {"median": float(np.median(res)), "p90": float(np.percentile(res, 90)),
                                  "max": float(res.max())},
           "rotation_vs_gauge_deg": {"median": float(np.median(rot)), "p90": float(np.percentile(rot, 90)),
                                     "max": float(rot.max())},
           "reason": (f"the {mapper} mapper registers every keyframe in one model: its trajectory, Sim3-aligned "
                      f"to the gauge's, starts the ladder (the gauge keeps the metre and the world frame)")}
    log(f"{LOG_TAG} init_poses classical: {mapper} {n_reg}/{len(kf)} keyframes — the ladder starts at its "
        f"trajectory (centres vs the gauge median {100 * np.median(res):.1f} cm, p90 {100 * np.percentile(res, 90):.1f} cm; "
        f"rotation median {np.median(rot):.2f} deg)")
    return W_new, rec


def cross_check(out_dir: Path, kf: Sequence[int], w2c_f5: np.ndarray, rid: Optional[str]) -> Dict[str, Any]:
    """refine.json["classical_cross_check"]: every classical model (one per configured mapper,
    ``models``) against F5's poses; the FIRST configured mapper's numbers at the top level
    (``top_level_from``), the camera prior COLMAP was given (``camera_prior``)."""
    from evaluation.colmap_io import list_models, read_model
    rep, mappers, ws = _load_classical(out_dir, kf, rid)
    if rep is None:
        return {"measured": False, "reason": str(mappers)}
    found: Dict[str, Dict[str, Any]] = {}
    for m in mappers:
        sparse = ws / f"sparse_{m}"
        dirs = list_models(sparse) if sparse.exists() else []
        if not dirs:
            found[m] = {"measured": False, "reason": f"{sparse}: no model"}
            continue
        model = max((read_model(d) for d in dirs), key=lambda mm: mm.n_images)
        found[m] = _check_model(model, kf, w2c_f5, n_models=len(dirs))
    first = found[mappers[0]]
    out: Dict[str, Any] = {"measured": bool(first.get("measured")), "mappers": mappers, "models": found,
                           "camera_prior": rep.get("camera_prior"), "top_level_from": mappers[0]}
    if first.get("measured"):
        out.update({k: v for k, v in first.items() if k != "measured"})
    else:
        out["reason"] = first.get("reason")
    return out

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.classical_check",
                                 description="COLMAP / GLOMAP on the keyframes: loop candidates for F4, the pose cross-check for F5.")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from repro import require_exclusive_gpu
    require_exclusive_gpu(log=print)
    run_classical(Path(args.session), load_precision_config(session_dir=Path(args.session)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
