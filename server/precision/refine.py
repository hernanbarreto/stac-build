"""P4 — joint refinement of the keyframe poses, the session camera and the
landmarks, and the localisation of the witness frames (claude_stac.txt §4-F5).

Unknowns: an SE(3) per keyframe, ONE session camera (OPENCV: fx, fy, cx, cy, k1,
k2, p1, p2 — F0's ``camera.json``), the landmarks. Residuals: reprojection in
NATIVE pixels under a Huber loss (``huber_px``). Engine: pycolmap / COLMAP's Ceres
(``reconstruction.colmap_ba``: ``_tune_ceres``, ``_solve_checked`` — a rung that
does not converge fails, no fallback), Ceres on ONE thread (``ceres_threads`` must be
1: its multi-threaded accumulation is not bit-reproducible) and every COLMAP RANSAC
seeded (``seed``) — identical inputs give bit-identical outputs.

Initialisation: the current keyframe poses (Omega × F2's gauge when it was
applied), F0's camera, landmarks triangulated from the FIT tracks (F4
``track_split`` 0) seen from at least ``min_tri_deg`` apart; the prior's depth
(Omega) is the cross-check (reported).

The gauge: no fixed prior. Each keyframe's centre is held by a soft prior whose σ
grows with the walk the way consecutive-distance priors of relative σ_gauge
(F2's measured scatter of its applied instrument) accumulate: σ_k =
σ_gauge · sqrt(Σ d_i²) over the steps up to k.

The ladder, each rung judged ONLY on the held-out tracks (split 1: never
optimised — triangulated with the rung's own poses and camera and reprojected):
  R0 camera fixed → R1 + fx, fy, cx, cy → R2 + k1, k2, p1, p2 → R3 focal per
  temporal block (``focal_block_frames``; tried only when R2's held-out residual
  varies systematically between blocks — a permutation test — the phone's EIS).
A rung replaces the best so far only when the paired bootstrap over the tracks
(``metric_lock.heldout_change``) says the held-out reprojection IMPROVES. Every
rung is reported in ``refine.json``.

Witness frames (the second pass of ``refine_poses_ba_twopass``): PnP + refinement
of every witness against the FIXED landmarks with its witness→keyframe tracks,
the session camera fixed; a witness whose reprojection RMS exceeds the bootstrap
upper bound of the keyframes' held-out quantile stays ``unlocalized`` with the
reason. Rolling shutter: DECLARED, not corrected — the held-out residual vs image
row × angular speed, its correlation reported.

Outputs (``output/precision``): ``refine.json``, ``refine_residuals.npz``,
``witness_poses.txt`` (+ ``witness_frames.txt``); ``output/camera.json`` at
camera_epoch + 1; the keyframes' rigid motion as an epoch through
``correction.visit_drift_run.apply_transform_epoch``.

CLI (the mapanything env: pycolmap 4): ``python -m precision.refine --session <dir>``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REFINE_NAME = "refine.json"
RESIDUALS_NAME = "refine_residuals.npz"
WITNESS_POSES_NAME = "witness_poses.txt"
WITNESS_FRAMES_NAME = "witness_frames.txt"
REFINE_VERSION = 1
PROVENANCE = "tool_measured"
LOG_TAG = "[refine]"
RUNGS = ("R0", "R1", "R2", "R3")


class RefineError(RuntimeError):
    """A structural impossibility of the stage — always with the exact reason."""


def _vendor_path() -> None:
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)


# ── geometry (numpy / OpenCV) ────────────────────────────────────────────

def K_of(params: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = (float(v) for v in params[:4])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def normalized(uv: np.ndarray, params: Sequence[float], solver: dict) -> np.ndarray:
    """Distorted native pixels → normalised image coordinates (x/z, y/z) under the
    OPENCV model, the undistortion iterated to convergence (F0's solver bounds) and
    verified by re-distortion (``precision.camera.undistort_normalized``: a point that
    did not converge fails the call)."""
    from precision.camera import undistort_normalized
    return undistort_normalized(np.asarray(uv, np.float64).reshape(-1, 2), K_of(params),
                                np.asarray(params[4:8], np.float64), **solver)


def project(X: np.ndarray, w2c: np.ndarray, params: Sequence[float]) -> np.ndarray:
    import cv2
    R, t = w2c[:3, :3], w2c[:3, 3]
    rv, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rv, t,
                              K_of(params), np.asarray(params[4:8], np.float64))
    return uv.reshape(-1, 2)


def triangulate(xn: np.ndarray, w2c: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Multi-view DLT of one track from normalised coordinates ``xn`` (k, 2) seen by
    cameras ``w2c`` (k, 4, 4). Returns (X, max pairwise ray angle in degrees, in
    front of every camera)."""
    A = []
    for (x, y), T in zip(xn, w2c):
        P = T[:3, :4]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _u, _s, Vt = np.linalg.svd(np.asarray(A))
    Xh = Vt[-1]
    if abs(Xh[3]) < np.finfo(float).eps:
        return np.full(3, np.nan), 0.0, False
    X = Xh[:3] / Xh[3]
    z = np.array([(T[:3, :3] @ X + T[:3, 3])[2] for T in w2c])
    C = np.array([-T[:3, :3].T @ T[:3, 3] for T in w2c])
    rays = X[None] - C
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    cosm = np.clip(rays @ rays.T, -1.0, 1.0)
    return X, float(np.degrees(np.arccos(cosm.min()))), bool((z > 0).all())


def group_tracks(track: np.ndarray, frame: np.ndarray, uv: np.ndarray,
                 frames: Sequence[int]) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """{track: [(index of the frame in ``frames``, uv)]} over the given frames."""
    idx = {int(f): i for i, f in enumerate(frames)}
    out: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for t, f, p in zip(track, frame, uv):
        i = idx.get(int(f))
        if i is not None:
            out.setdefault(int(t), []).append((i, p))
    return out


def triangulate_tracks(groups: Dict[int, List], w2c: np.ndarray, params: Sequence[float],
                       solver: dict, min_tri_deg: float) -> Dict[int, np.ndarray]:
    out = {}
    for t, obs in groups.items():
        if len(obs) < 2:
            continue
        ii = np.array([i for i, _ in obs])
        xn = normalized(np.array([p for _, p in obs]), params, solver)
        X, ang, front = triangulate(xn, w2c[ii])
        if front and ang >= min_tri_deg and np.all(np.isfinite(X)):
            out[t] = X
    return out


def heldout_rms(groups: Dict[int, List], w2c: np.ndarray, params: Sequence[float],
                solver: dict, min_tri_deg: float) -> Dict[int, float]:
    """Per held-out track: triangulated with these poses and this camera, the RMS
    of its reprojection over its own observations (native px)."""
    X = triangulate_tracks(groups, w2c, params, solver, min_tri_deg)
    out = {}
    for t, P in X.items():
        e = [np.linalg.norm(project(P[None], w2c[i], params)[0] - np.asarray(p)) for i, p in groups[t]]
        out[t] = float(np.sqrt(np.mean(np.square(e))))
    return out


def prior_sigmas(centres: np.ndarray, sigma_rel: float) -> np.ndarray:
    """σ_k = σ_rel · sqrt(Σ_{i≤k} d_i²): consecutive-distance priors of relative σ
    accumulated along the walk (the first keyframe carries its first step's)."""
    d = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    acc = np.sqrt(np.concatenate([[d[0] ** 2 if len(d) else 0.0], np.cumsum(d ** 2)]))
    return float(sigma_rel) * np.maximum(acc, acc[acc > 0].min() if (acc > 0).any() else 1.0)


# ── one rung (pycolmap) ──────────────────────────────────────────────────

@dataclass
class RungResult:
    name: str
    w2c: np.ndarray
    params_by_block: List[List[float]]      # one camera, or one per focal block (R3)
    block_of: np.ndarray                    # keyframe → camera block
    fit_rms_px: float
    n_points: int
    termination: str


def run_rung(name: str, w2c0: np.ndarray, params0: Sequence[float], wh: Tuple[int, int],
             fit_groups: Dict[int, List], X0: Dict[int, np.ndarray], sigmas: np.ndarray,
             cfg, block_of: Optional[np.ndarray] = None,
             params_by_block0: Optional[List[List[float]]] = None) -> RungResult:
    import pycolmap
    _vendor_path()
    from reconstruction.colmap_ba import _rigid, _tune_ceres, _solve_checked
    N = len(w2c0)
    block_of = np.zeros(N, int) if block_of is None else np.asarray(block_of, int)
    blocks = sorted(set(block_of.tolist()))
    pb0 = params_by_block0 or [list(params0) for _ in blocks]
    rec = pycolmap.Reconstruction()
    for b in blocks:
        cam = pycolmap.Camera.create_from_model_name(b + 1, "OPENCV", float(pb0[b][0]),
                                                     int(wh[0]), int(wh[1]))
        cam.params = [float(v) for v in pb0[b]]
        cam.camera_id = b + 1
        rec.add_camera_with_trivial_rig(cam)
    per_img: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in range(N)}
    for t, obs in fit_groups.items():
        if t in X0:
            for i, p in obs:
                per_img[i].append((t, p))
    p2d: Dict[Tuple[int, int], int] = {}
    for i in range(N):
        pts = [pycolmap.Point2D(np.asarray(p, float)) for _, p in per_img[i]]
        im = pycolmap.Image(name=f"{i}.jpg", camera_id=int(block_of[i]) + 1, points2D=pts)
        im.image_id = i + 1
        rec.add_image_with_trivial_frame(im, _rigid(w2c0[i], pycolmap))
        for k, (t, _p) in enumerate(per_img[i]):
            p2d[(i, t)] = k
    n_pts = 0
    for t, X in X0.items():
        tr = pycolmap.Track()
        for i, _p in fit_groups[t]:
            if (i, t) in p2d:
                tr.add_element(pycolmap.TrackElement(i + 1, p2d[(i, t)]))
        if tr.length() >= 2:
            rec.add_point3D(np.asarray(X, float), tr)
            n_pts += 1
    priors = []
    for i in range(N):
        pp = pycolmap.PosePrior()
        pp.position = -w2c0[i, :3, :3].T @ w2c0[i, :3, 3]
        pp.position_covariance = np.eye(3) * float(sigmas[i]) ** 2
        pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        pp.corr_data_id = rec.image(i + 1).data_id
        priors.append(pp)
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = name in ("R1", "R2", "R3")
    opts.refine_principal_point = name in ("R1", "R2")
    opts.refine_extra_params = name == "R2"
    opts.refine_points3D = True
    opts.refine_rig_from_world = True
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    opts.ceres.loss_function_scale = float(cfg.huber_px)
    _tune_ceres(opts, int(cfg.max_iterations))
    # ONE thread (the config refuses any other value): Ceres' multi-threaded evaluation
    # and Schur accumulation sum in a run-dependent order — measured 1.8e-11 in the
    # poses between two identical solves at 8 threads, bit for bit the same at 1
    opts.ceres.solver_options.num_threads = int(cfg.ceres_threads)
    bcfg = pycolmap.BundleAdjustmentConfig()
    for i in range(N):
        bcfg.add_image(i + 1)
    popts = pycolmap.PosePriorBundleAdjustmentOptions()
    popts.prior_position_fallback_stddev = float(np.max(sigmas))
    # the adjuster first aligns the reconstruction to the priors by RANSAC; with COLMAP's
    # default seed (-1) and its PRNG unseeded, every solve of identical inputs differed
    # (measured) — both are fixed, so a rung does not depend on what ran before it
    popts.alignment_ransac.random_seed = int(cfg.seed)
    popts.alignment_ransac.num_threads = 1
    pycolmap.set_random_seed(int(cfg.seed))
    summary = _solve_checked(pycolmap.create_pose_prior_bundle_adjuster(
        opts, popts, bcfg, priors, rec), f"refine {name}")
    w2c = np.tile(np.eye(4), (N, 1, 1))
    for i in range(N):
        w2c[i, :3, :4] = np.asarray(rec.image(i + 1).cam_from_world().matrix())
    # pycolmap 4 reports the STORED point errors: refresh them, or the mean reads 0
    rec.update_point_3d_errors()
    return RungResult(name, w2c, [list(rec.camera(b + 1).params) for b in blocks], block_of,
                      float(rec.compute_mean_reprojection_error()), n_pts,
                      str(getattr(summary, "termination_type", "")))


# ── the refinement ───────────────────────────────────────────────────────

def _heldout_of(r: RungResult, groups, solver, cfg) -> Dict[int, float]:
    if len(r.params_by_block) == 1:
        return heldout_rms(groups, r.w2c, r.params_by_block[0], solver, cfg.min_tri_deg)
    # per-block cameras: a track is judged with the camera of each observation's block —
    # triangulated in normalised coordinates of its own block
    out = {}
    for t, obs in groups.items():
        if len(obs) < 2:
            continue
        ii = np.array([i for i, _ in obs])
        xn = np.array([normalized(np.asarray(p)[None], r.params_by_block[r.block_of[i]], solver)[0]
                       for i, p in obs])
        X, ang, front = triangulate(xn, r.w2c[ii])
        if not (front and ang >= cfg.min_tri_deg and np.all(np.isfinite(X))):
            continue
        e = [np.linalg.norm(project(X[None], r.w2c[i], r.params_by_block[r.block_of[i]])[0]
                            - np.asarray(p)) for i, p in obs]
        out[t] = float(np.sqrt(np.mean(np.square(e))))
    return out


def _judge(best: Dict[int, float], cand: Dict[int, float], confidence: float) -> dict:
    _vendor_path()
    from loop_utils.metric_lock import heldout_change
    common = sorted(set(best) & set(cand))
    chg = heldout_change([best[t] for t in common], [cand[t] for t in common],
                         confidence=float(confidence))
    chg["n_tracks"] = len(common)
    return chg


def _systematic_in_time(per_frame: Dict[int, float], block_of: np.ndarray, cfg) -> dict:
    """Does the held-out residual differ between temporal blocks beyond chance?
    Permutation test (fixed seed) on the variance of the block medians."""
    fr = sorted(per_frame)
    if len(set(block_of[fr].tolist())) < 2:
        return {"systematic": False, "p": None, "reason": "one block"}
    v = np.array([per_frame[i] for i in fr])
    b = block_of[fr]

    def stat(labels):
        return float(np.var([np.median(v[labels == k]) for k in np.unique(labels)]))
    obs = stat(b)
    rng = np.random.default_rng(int(cfg.seed))
    perm = np.array([stat(rng.permutation(b)) for _ in range(int(cfg.permutations))])
    p = float((1 + np.sum(perm >= obs)) / (1 + len(perm)))
    return {"systematic": bool(p < 1.0 - float(cfg.heldout_confidence)), "p": p,
            "stat": obs}


def refine_core(w2c0: np.ndarray, params0: Sequence[float], wh: Tuple[int, int],
                track: np.ndarray, frame: np.ndarray, uv: np.ndarray, split: np.ndarray,
                kf_frames: Sequence[int], sigma_rel: float, cfg, solver: dict,
                log: Callable = print) -> Dict[str, Any]:
    """The ladder over the keyframes. ``split`` per OBSERVATION (its track's split)."""
    import pycolmap
    pycolmap.set_random_seed(int(cfg.seed))           # COLMAP's PRNG: identical runs, identical bits
    fit_g = group_tracks(track[split == 0], frame[split == 0], uv[split == 0], kf_frames)
    held_g = group_tracks(track[split == 1], frame[split == 1], uv[split == 1], kf_frames)
    X0 = triangulate_tracks(fit_g, w2c0, params0, solver, cfg.min_tri_deg)
    if len(X0) < 3:
        raise RefineError(f"{len(X0)} fit track(s) triangulate at ≥ {cfg.min_tri_deg:g}° — "
                          f"nothing to refine against")
    sig = prior_sigmas(np.array([-T[:3, :3].T @ T[:3, 3] for T in w2c0]), sigma_rel)
    base = RungResult("init", w2c0, [list(params0)], np.zeros(len(w2c0), int), float("nan"), 0, "")
    held = {"init": _heldout_of(base, held_g, solver, cfg)}
    rungs, best = {}, None
    from precision.camera import CameraError
    for name in ("R0", "R1", "R2"):
        r = run_rung(name, w2c0, params0, wh, fit_g, X0, sig, cfg)
        try:
            held[name] = _heldout_of(r, held_g, solver, cfg)
        except CameraError as e:
            # a rung whose estimated distortion cannot be inverted over the held-out
            # observations is not a camera: rejected with the reason, the ladder goes on
            rungs[name] = {"params": r.params_by_block[0], "fit_rms_px": r.fit_rms_px,
                           "taken": False, "rejected": f"distortion not invertible over the "
                                                       f"held-out observations ({e})"}
            log(f"{LOG_TAG} {name}: rejected — {rungs[name]['rejected']}")
            continue
        verdict = None if best is None else _judge(held[best.name], held[name],
                                                   cfg.heldout_confidence)
        take = best is None or bool(verdict["improves"])
        rungs[name] = {"params": r.params_by_block[0], "fit_rms_px": r.fit_rms_px,
                       "heldout_median_px": float(np.median(list(held[name].values())))
                       if held[name] else None,
                       "n_heldout_tracks": len(held[name]), "n_points": r.n_points,
                       "termination": r.termination,
                       "verdict_vs_best": verdict, "taken": take}
        if take:
            best = r
        log(f"{LOG_TAG} {name}: fit {r.fit_rms_px:.3f} px, held-out median "
            f"{rungs[name]['heldout_median_px']} px"
            + ("" if verdict is None else
               f" — vs {best.name if not take else 'previous'}: "
               f"{'improves' if verdict['improves'] else 'worsens' if verdict['worsens'] else 'within noise'}"))
    if best is None:
        raise RefineError("every rung was rejected (its distortion is not invertible over the "
                          "held-out observations) — the session camera cannot be refined; see "
                          "the rung reasons in the log")
    # R3: focal per temporal block, only when the best rung leaves a temporal pattern
    blocks = np.arange(len(w2c0)) // int(cfg.focal_block_frames)
    per_frame: Dict[int, List[float]] = {}
    Xb = triangulate_tracks(held_g, best.w2c, best.params_by_block[0], solver, cfg.min_tri_deg)
    for t, P in Xb.items():
        for i, p in held_g[t]:
            per_frame.setdefault(i, []).append(
                float(np.linalg.norm(project(P[None], best.w2c[i], best.params_by_block[0])[0] - p)))
    # per keyframe: the RMS of its held-out reprojections — the statistic a witness is
    # compared with (its own reprojection RMS)
    pf = {i: float(np.sqrt(np.mean(np.square(v)))) for i, v in per_frame.items()}
    sysrep = _systematic_in_time(pf, blocks, cfg)
    rungs["R3"] = {"tried": False, "temporal_test": sysrep}
    if sysrep["systematic"]:
        r = run_rung("R3", best.w2c, best.params_by_block[0], wh, fit_g, X0, sig, cfg,
                     block_of=blocks,
                     params_by_block0=[list(best.params_by_block[0])
                                       for _ in range(int(blocks.max()) + 1)])
        try:
            held["R3"] = _heldout_of(r, held_g, solver, cfg)
        except CameraError as e:
            held["R3"] = None
            rungs["R3"].update({"tried": True, "taken": False,
                                "rejected": f"distortion not invertible over the held-out "
                                            f"observations ({e})"})
        if held["R3"] is not None:
            verdict = _judge(held[best.name], held["R3"], cfg.heldout_confidence)
            rungs["R3"].update({"tried": True, "params_by_block": r.params_by_block,
                                "fit_rms_px": r.fit_rms_px,
                                "heldout_median_px": float(np.median(list(held["R3"].values()))),
                                "verdict_vs_best": verdict, "taken": bool(verdict["improves"])})
            if verdict["improves"]:
                best = r
    return {"best": best, "rungs": rungs, "held": held, "X": X0, "fit_groups": fit_g,
            "held_groups": held_g, "prior_sigma_m": sig, "per_frame_heldout": pf}


def localize_witnesses(best: RungResult, fit_track_X: Dict[int, np.ndarray],
                       track: np.ndarray, frame: np.ndarray, uv: np.ndarray,
                       witness_frames: Sequence[int], wh: Tuple[int, int], bound_px: float,
                       cfg) -> Dict[int, dict]:
    """PnP + refinement of every witness against the FIXED landmarks, the session
    camera fixed (the best rung's; R3 → the block of the nearest keyframe is not
    known for a witness: its first block's camera)."""
    import pycolmap
    pycolmap.set_random_seed(int(cfg.seed))
    params = best.params_by_block[0]
    cam = pycolmap.Camera.create_from_model_name(1, "OPENCV", float(params[0]), int(wh[0]), int(wh[1]))
    cam.params = [float(v) for v in params]
    # explicit estimation options: COLMAP's defaults, but the LO-RANSAC seeded explicitly
    # (its default seed is -1, the one that left the pose-prior alignment run-dependent)
    # and single-threaded
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.ransac.random_seed = int(cfg.seed)
    est.ransac.num_threads = 1
    ref = pycolmap.AbsolutePoseRefinementOptions()
    out = {}
    wset = set(int(w) for w in witness_frames)
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for t, f, p in zip(track, frame, uv):
        if int(f) in wset and int(t) in fit_track_X:
            by_frame.setdefault(int(f), []).append((int(t), p))
    for f in sorted(wset):
        obs = by_frame.get(f, [])
        if len(obs) < int(cfg.min_witness_corr):
            out[f] = {"localized": False, "reason": "too_few_correspondences", "n": len(obs)}
            continue
        p2 = np.array([p for _, p in obs], np.float64)
        p3 = np.array([fit_track_X[t] for t, _ in obs], np.float64)
        res = pycolmap.estimate_and_refine_absolute_pose(p2, p3, cam, est, ref)
        if not res:
            out[f] = {"localized": False, "reason": "pnp_failed", "n": len(obs)}
            continue
        T = np.eye(4)
        T[:3, :4] = np.asarray(res["cam_from_world"].matrix())
        inl = np.asarray(res["inlier_mask"], bool)
        e = np.linalg.norm(project(p3[inl], T, params) - p2[inl], axis=1)
        rms = float(np.sqrt(np.mean(e ** 2))) if inl.any() else float("inf")
        ok = rms <= bound_px
        out[f] = {"localized": bool(ok), "reason": None if ok else "rms_above_keyframe_heldout",
                  "rms_px": rms, "n": len(obs), "n_inliers": int(inl.sum()),
                  "c2w": np.linalg.inv(T).tolist()}
    return out


def heldout_leave_one_view_out(held_g: Dict[int, List], best: RungResult, solver: dict,
                               cfg) -> Dict[int, float]:
    """Per keyframe: the RMS of its held-out observations reprojected from their track
    triangulated WITHOUT that view — the situation of a witness frame, measured
    against landmarks it never shaped. (A track triangulated with its own view
    absorbs part of that view's noise and reads low: 0.38 vs 0.46 px on the
    synthetic walk.)"""
    params = best.params_by_block[0]
    per: Dict[int, List[float]] = {}
    for t, obs in held_g.items():
        if len(obs) < 3:
            continue
        for k, (i, p) in enumerate(obs):
            rest = [o for kk, o in enumerate(obs) if kk != k]
            ii = np.array([j for j, _ in rest])
            xn = normalized(np.array([q for _, q in rest]), params, solver)
            X, ang, front = triangulate(xn, best.w2c[ii])
            if not (front and ang >= cfg.min_tri_deg and np.all(np.isfinite(X))):
                continue
            per.setdefault(i, []).append(
                float(np.linalg.norm(project(X[None], best.w2c[i], params)[0] - np.asarray(p))))
    return {i: float(np.sqrt(np.mean(np.square(v)))) for i, v in per.items()}


def heldout_bound(per_frame: Dict[int, float], cfg) -> float:
    """The bootstrap upper bound of the keyframes' held-out quantile
    (``heldout_confidence``) — what a frame localised as well as a keyframe may show."""
    v = np.array(list(per_frame.values()))
    rng = np.random.default_rng(int(cfg.seed))
    q = 100.0 * float(cfg.heldout_confidence)
    boots = [np.percentile(v[rng.integers(0, len(v), len(v))], q) for _ in range(int(cfg.permutations))]
    return float(np.percentile(boots, q))


def rolling_shutter_diag(held_g, best: RungResult, solver, cfg) -> dict:
    """DECLARED, not corrected: the held-out reprojection error vs (image row ×
    angular speed of the keyframe) — Spearman correlation."""
    from scipy.stats import spearmanr
    N = len(best.w2c)
    ang = np.zeros(N)
    for i in range(N):
        a, b = max(0, i - 1), min(N - 1, i + 1)
        if a == b:
            continue
        R = best.w2c[b, :3, :3] @ best.w2c[a, :3, :3].T
        ang[i] = math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))) / (b - a)
    params = best.params_by_block[0]
    X = triangulate_tracks(held_g, best.w2c, params, solver, cfg.min_tri_deg)
    xs, ys = [], []
    cy = float(params[3])
    for t, P in X.items():
        for i, p in held_g[t]:
            e = float(np.linalg.norm(project(P[None], best.w2c[i], params)[0] - p))
            xs.append(abs(float(p[1]) - cy) * ang[i])
            ys.append(e)
    if len(xs) < 3:
        return {"n": len(xs), "spearman": None}
    rho, pval = spearmanr(xs, ys)
    return {"n": len(xs), "spearman": float(rho), "p": float(pval), "corrected": False}


# ── session I/O ──────────────────────────────────────────────────────────

def _gauge_sigma(output_dir: Path) -> Tuple[float, str]:
    p = output_dir / "gauge.json"
    if not p.exists():
        raise RefineError(f"{p} is missing — the refinement's gauge priors come from F2 "
                          f"(python -m precision.gauge --session <dir>)")
    g = json.loads(p.read_text())
    inst = g["applied_instrument"]
    return float(g["sigma_by_instrument"][inst]), inst


def run_refine(session_dir: Path, pcfg, *, apply: bool = True, log: Callable = print) -> Dict[str, Any]:
    from precision.camera import load_camera_json, save_camera_json, undistort_solver
    from precision.tracks import load_tracks_v2, KIND_WITNESS
    session_dir = Path(session_dir)
    out = session_dir / "output"
    cfg = pcfg.refine
    cam = load_camera_json(out / "camera.json")
    solver = undistort_solver(pcfg.camera)
    poses = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
    if len(kf) != len(poses):
        raise RefineError(f"camera_poses.txt ({len(poses)}) and camera_frames.txt ({len(kf)}) "
                          f"disagree")
    w2c0 = np.linalg.inv(poses)
    tr = load_tracks_v2(session_dir)
    tid = tr["track_query_id"]
    split_of_track = dict(zip(tid.tolist(), tr["track_split"].tolist()))
    split_obs = np.array([split_of_track[int(t)] for t in tr["obs_track"]], np.int8)
    sigma_rel, inst = _gauge_sigma(out)
    log(f"{LOG_TAG} {len(kf)} keyframes, {len(tid):,} tracks, gauge σ {sigma_rel:.4f} ({inst})")
    core = refine_core(w2c0, cam.params, (cam.width, cam.height), tr["obs_track"], tr["obs_frame"],
                       tr["obs_uv_native"], split_obs, kf, sigma_rel, cfg, solver, log=log)
    best = core["best"]
    bound = heldout_bound(heldout_leave_one_view_out(core["held_groups"], best, solver, cfg), cfg)
    wit = sorted(set(tr["obs_frame"][tr["frame_kind"] == KIND_WITNESS].tolist()) - set(kf))
    wloc = localize_witnesses(best, core["X"], tr["obs_track"], tr["obs_frame"],
                              tr["obs_uv_native"], wit, (cam.width, cam.height), bound, cfg)
    rs = rolling_shutter_diag(core["held_groups"], best, solver, cfg)
    # prior-depth cross-check: triangulated depth vs Omega's depth at the query pixel
    cross = _prior_cross_check(out, kf, core["X"], core["fit_groups"], best, cam)
    pdir = out / "precision"
    pdir.mkdir(parents=True, exist_ok=True)
    loc = [f for f, r in wloc.items() if r["localized"]]
    with open(pdir / WITNESS_POSES_NAME, "w") as fh:
        for f in loc:
            fh.write(" ".join(f"{v:.10g}" for v in np.asarray(wloc[f]["c2w"]).ravel()) + "\n")
    (pdir / WITNESS_FRAMES_NAME).write_text(" ".join(str(f) for f in loc) + "\n")
    from intake.quality import read_session_epochs
    epochs = read_session_epochs(session_dir)
    doc = {"version": REFINE_VERSION, "provenance": PROVENANCE, **epochs,
           "params": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
           "gauge": {"sigma_rel": sigma_rel, "instrument": inst,
                     "prior_sigma_m": {"first": float(core["prior_sigma_m"][0]),
                                       "last": float(core["prior_sigma_m"][-1])}},
           "rungs": core["rungs"], "applied_rung": best.name,
           "camera": {"before": list(cam.params), "after": best.params_by_block},
           "n_landmarks": len(core["X"]),
           "keyframe_heldout_bound_px": bound,
           "witnesses": {"n": len(wit), "localized": len(loc),
                         "unlocalized": {str(f): r["reason"] for f, r in wloc.items()
                                         if not r["localized"]},
                         "rms_px": {str(f): r.get("rms_px") for f, r in wloc.items()}},
           "rolling_shutter": rs, "prior_depth_cross_check": cross, "applied": False}
    np.savez_compressed(pdir / RESIDUALS_NAME,
                        heldout_track=np.array(list(core["held"][best.name].keys()), np.int64),
                        heldout_rms_px=np.array(list(core["held"][best.name].values()), np.float32),
                        init_track=np.array(list(core["held"]["init"].keys()), np.int64),
                        init_rms_px=np.array(list(core["held"]["init"].values()), np.float32))
    if apply:
        new_cam = cam.with_params(best.params_by_block[0], "refine", cam.camera_epoch + 1,
                                  report={"rung": best.name, "blocks": len(best.params_by_block)})
        save_camera_json(out / "camera.json", new_cam, epochs.get("geometry_epoch"))
        T = np.linalg.inv(best.w2c) @ w2c0              # c2w_new · w2c_old: world motion
        from correction.visit_drift_run import apply_transform_epoch
        res = apply_transform_epoch(out, T[:, :3, :3], T[:, :3, 3], np.ones(len(T)), "refine",
                                    [{"stage": "refine", "rung": best.name}], log=log)
        doc["applied"] = bool(res)
        doc["epoch_to"] = (res or {}).get("epoch_to")
        doc["camera_epoch_to"] = new_cam.camera_epoch
    (pdir / REFINE_NAME).write_text(json.dumps(doc, indent=1, default=float))
    log(f"{LOG_TAG} applied rung {best.name}; witnesses {len(loc)}/{len(wit)} localized "
        f"(bound {bound:.2f} px) → {pdir / REFINE_NAME}")
    return doc


def _prior_cross_check(out: Path, kf, X, fit_g, best: RungResult, cam) -> dict:
    """Triangulated landmark depth vs Omega's depth at the observing keyframe's pixel
    (Omega grid via F0) — reported, the landmarks are not moved by it. Omega's record
    carries the units of the reconstruction BEFORE the metric lock, so the two are
    compared as a ratio: its median is the scale between them, and the spread of the
    ratios around it is the prior's relative depth error."""
    from precision.camera import native_to_grid, grid_like
    ratio = []
    for t, P in list(X.items())[:5000]:
        i, p = fit_g[t][0]
        npz = out / "omega_run" / "results_output" / f"frame_{kf[i]}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as z:
            d = z["depth"]
        g = cam.omega_grid if (cam.omega_grid.w, cam.omega_grid.h) == (d.shape[1], d.shape[0]) \
            else grid_like(cam.omega_grid, d.shape[1], d.shape[0], "omega_npz")
        q = native_to_grid(np.asarray(p, np.float64)[None], g)[0]
        u, v = int(round(q[0])), int(round(q[1]))
        if not (0 <= v < d.shape[0] and 0 <= u < d.shape[1]) or d[v, u] <= 0:
            continue
        z_tri = float((best.w2c[i, :3, :3] @ P + best.w2c[i, :3, 3])[2])
        ratio.append(z_tri / float(d[v, u]))
    if not ratio:
        return {"n": 0, "scale": None, "median_rel": None}
    r = np.asarray(ratio)
    s = float(np.median(r))
    return {"n": int(r.size), "scale": s, "median_rel": float(np.median(np.abs(r / s - 1)))}


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.refine",
                                 description="Joint pose + session camera refinement (F5).")
    ap.add_argument("--session", required=True)
    ap.add_argument("--no-apply", action="store_true")
    args = ap.parse_args(argv)
    run_refine(Path(args.session), load_precision_config(), apply=not args.no_apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
