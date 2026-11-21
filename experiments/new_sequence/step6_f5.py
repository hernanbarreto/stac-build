"""THE NEW ORDER, step 6 — "F5 por cubo, con una sola cámara para todos."

The pipeline's F5 adjustment (precision.refine.refine_core: the rungs R0, R1, R2, each with its continuation and the
held-out judge), unchanged, given the five cubes at once:

  * a keyframe as a cube holds it is a camera of its own (64 + 64 + 64 + 64 + 46 = 302), starting at the pose its cube
    has after F2 (the cube session's camera_poses.txt, in the cube's own frame);
  * a keyframe's observations (F4's tracks) go to each cube that holds it; the landmarks are separate per cube, so the
    cubes share NO geometry — each is adjusted on its own;
  * the camera model is ONE for every cube (F5 solves one camera block);
  * each cube's anchors are F5's own position priors along that cube's walk, with the tolerance F2 gave THAT cube
    (precision.refine._gauge_sigma on the cube's gauge.json).

The result enters each cube's session through F5's own door (camera.json, the poses as the cube's next epoch, F5's
report and residual files), so that the pipeline's F6 reads it there. Not done: the localisation of the witness frames
(F6 does not read their poses).

    python step6_f5.py <session scan dir> <root of the cube sessions> <out npz>"""
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)
FRAME_STRIDE = 1_000_000                 # a version's frame id = cube x this + the keyframe's video frame


def main() -> int:
    scan, root, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from intake.quality import read_session_epochs
    from precision import refine as R
    from precision.camera import load_camera_json, save_camera_json, undistort_solver
    from precision.config import load_precision_config, raw_of
    from precision.poses_epoch import apply_pose_epoch
    from precision.tracks import load_tracks_v2
    from reconstruction.loops.config import improvement_error_factor
    pcfg = load_precision_config(session_dir=scan)
    log = lambda m: print(m, flush=True)   # noqa: E731

    cam = load_camera_json(scan / "output" / "camera.json")
    dirs = sorted(root.glob("cube_*"), key=lambda p: int(p.name.split("_")[1]))
    cubes = []
    for d in dirs:
        out = d / "output"
        ep = read_session_epochs(d)
        if int(ep["geometry_epoch"]) != 1:
            raise SystemExit(f"{d.name} is at geometry epoch {ep['geometry_epoch']}: F5 starts from the cube after F2 (epoch 1)")
        poses = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
        kfs = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
        sigma, inst = R._gauge_sigma(out)
        cubes.append({"dir": d, "poses": poses, "frames": kfs, "sigma": sigma, "inst": inst, "epochs": ep})
        log(f"[step 6] {d.name}: {len(kfs)} keyframes, anchors' tolerance {sigma:.5f} ({inst})")
    C = len(cubes)
    cube_of = np.concatenate([np.full(len(cb["frames"]), c) for c, cb in enumerate(cubes)])
    c2w0 = np.concatenate([cb["poses"] for cb in cubes])
    w2c0 = np.linalg.inv(c2w0)
    kf_frames = [c * FRAME_STRIDE + f for c, cb in enumerate(cubes) for f in cb["frames"]]

    R.use_device(pcfg.refine.triangulation_device)
    tr = load_tracks_v2(scan)
    split = R.split_of_observations(tr["track_query_id"], tr["track_split"], tr["obs_track"])
    stride_t = int(tr["obs_track"].max()) + 1
    T, F, UV, SP = [], [], [], []
    for c, cb in enumerate(cubes):
        sel = np.isin(tr["obs_frame"], np.asarray(cb["frames"]))
        T.append(tr["obs_track"][sel].astype(np.int64) + c * stride_t)
        F.append(tr["obs_frame"][sel].astype(np.int64) + c * FRAME_STRIDE)
        UV.append(np.asarray(tr["obs_uv_native"][sel], np.float64)); SP.append(split[sel])
        log(f"[step 6]   cube {c}: {int(sel.sum()):,} observations on its keyframes")
    T, F, UV, SP = np.concatenate(T), np.concatenate(F), np.concatenate(UV), np.concatenate(SP)

    # F5's own position priors, per cube: each cube is a walk of its own, with ITS tolerance from ITS F2
    walk_sigmas = R.prior_sigmas

    def per_cube_sigmas(centres, _rel):
        sig = np.empty(len(centres))
        for c, cb in enumerate(cubes):
            sig[cube_of == c] = walk_sigmas(centres[cube_of == c], cb["sigma"])
        return sig
    R.prior_sigmas = per_cube_sigmas
    sig0 = per_cube_sigmas(c2w0[:, :3, 3], None)
    log("[step 6] anchors per cube (first / last keyframe, cm): "
        + ", ".join(f"cube {c}: {sig0[cube_of == c][0] * 100:.2f} / {sig0[cube_of == c][-1] * 100:.2f}" for c in range(C)))

    t0 = time.time()
    fac = float(improvement_error_factor(raw_of(pcfg)))
    core = R.refine_core(w2c0, cam.params, (cam.width, cam.height), T, F, UV, SP, kf_frames,
                         float(np.median([cb["sigma"] for cb in cubes])), pcfg.refine, undistort_solver(pcfg.camera),
                         error_factor=fac, log=log, solve=bool(pcfg.refine.enabled), range_m=float(pcfg.view_range_m))
    best = core["best"]
    params = [float(v) for v in best.params_by_block[0]]
    c2w = np.linalg.inv(best.w2c)
    log(f"[step 6] the ladder took {time.time() - t0:.0f} s; rung applied: {best.name}")
    log(f"[step 6] ONE CAMERA for the 5 cubes: fx {params[0]:.2f} fy {params[1]:.2f} cx {params[2]:.2f} cy {params[3]:.2f} "
        f"k1 {params[4]:+.4f} k2 {params[5]:+.4f} p1 {params[6]:+.5f} p2 {params[7]:+.5f} (F0: fx {cam.params[0]:.2f} "
        f"fy {cam.params[1]:.2f})")
    for name, rep in core["rungs"].items():
        log(f"[step 6]   {name}: fit {rep.get('fit_median_px', rep.get('fit_rms_px'))} px, held-out median "
            f"{rep.get('heldout_median_px')} px, taken {rep.get('taken')}")

    # each cube's result through F5's own door, into the cube's session
    new_cam = cam.with_params(params, "refine", cam.camera_epoch + 1, report={"rung": best.name, "blocks": 1})
    held = core["held"][best.name]
    held_init = core["held"]["init"]
    for c, cb in enumerate(cubes):
        out, vs = cb["dir"] / "output", np.flatnonzero(cube_of == c)
        pdir = out / "precision"
        pdir.mkdir(parents=True, exist_ok=True)
        for name in ("tracks.npz", "tracks.json", "tracks.timing.json"):
            src = scan / "output" / "precision" / name
            if src.exists() and not os.path.lexists(pdir / name):
                os.symlink(src, pdir / name)
        (pdir / R.WITNESS_POSES_NAME).write_text("")
        (pdir / R.WITNESS_FRAMES_NAME).write_text("\n")
        mine = {int(t) % stride_t: float(v) for t, v in held.items() if int(t) // stride_t == c}
        mine0 = {int(t) % stride_t: float(v) for t, v in held_init.items() if int(t) // stride_t == c}
        np.savez_compressed(pdir / R.RESIDUALS_NAME,
                            heldout_track=np.array(list(mine.keys()), np.int64),
                            heldout_rms_px=np.array(list(mine.values()), np.float32),
                            init_track=np.array(list(mine0.keys()), np.int64),
                            init_rms_px=np.array(list(mine0.values()), np.float32))
        save_camera_json(out / "camera.json", new_cam, cb["epochs"].get("geometry_epoch"))
        M = c2w[vs] @ np.linalg.inv(c2w0[vs])              # c2w_new · w2c_old: each keyframe's motion in the cube's frame
        res = apply_pose_epoch(out, M[:, :3, :3], M[:, :3, 3], np.ones(len(vs)), "refine",
                               [{"stage": "refine", "rung": best.name, "new_sequence": "F5 per cube, one camera"}], log=log)
        moved = np.linalg.norm(c2w[vs, :3, 3] - c2w0[vs, :3, 3], axis=1)
        hv = np.asarray([v for v in mine.values() if np.isfinite(v)])
        doc = {"version": R.REFINE_VERSION, "provenance": R.PROVENANCE, **cb["epochs"],
               "params": dataclasses.asdict(pcfg.refine),
               "new_sequence": "step 6 of the new order: F5 per cube with one camera for the five cubes "
                               "(experiments/new_sequence/step6_f5.py), written in F5's format for the pipeline's F6",
               "gauge": {"sigma_rel": cb["sigma"], "instrument": cb["inst"]},
               "ladder": core["ladder"], "applied_rung": best.name,
               "camera": {"before": list(cam.params), "after": [params]},
               "heldout_median_px": float(np.median(hv)) if len(hv) else None,
               "witnesses": {"n": 0, "localized": 0, "unlocalized": {}, "rms_px": {}},
               "applied": bool(res), "epoch_to": (res or {}).get("epoch_to"), "camera_epoch_to": new_cam.camera_epoch}
        (pdir / R.REFINE_NAME).write_text(json.dumps(doc, indent=1, default=float))
        log(f"[step 6] cube {c}: held-out median {doc['heldout_median_px']:.4f} px over {len(hv):,} tracks; its cameras "
            f"moved {np.median(moved) * 100:.2f} cm (median) / {moved.max() * 100:.2f} cm (most); epoch {doc['epoch_to']}")
    ids = np.fromiter(best.X.keys(), np.int64, len(best.X))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, params=np.asarray(params), wh=np.asarray([cam.width, cam.height]), rung=best.name, c2w=c2w,
             c2w_start=c2w0, cube_of=cube_of, frame=np.asarray([f % FRAME_STRIDE for f in kf_frames]),
             landmark_cube=ids // stride_t, landmark_track=ids % stride_t,
             landmark_xyz=np.asarray([best.X[int(i)] for i in ids], np.float64),
             heldout_px=json.dumps({n: r.get("heldout_median_px") for n, r in core["rungs"].items()}))
    log(f"[step 6] written: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
