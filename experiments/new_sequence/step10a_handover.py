"""THE NEW ORDER, step 10 (preparation): what the pipeline's F6 asks of a session before it starts.

F6 reads the session's camera and cameras as "F5's" and refuses to start without F5's report at the session's epoch
(precision.depth_sweep.load_inputs). In the new order the session's cameras after the tie ARE F5's — adjusted cube by
cube with one camera (step 6), then tied (step 8) and put in metres (step 9) — so the files are written here, in F5's
format and through F5's own writers, saying exactly that:

  * camera.json            the ONE camera of step 6;
  * precision/refine.json  F5's report, with the held-out error MEASURED on the final cameras with F5's own function
                           (refine_core, solve off): the session's held-out tracks triangulated and reprojected with
                           the cameras the vote is about to use;
  * precision/refine_residuals.npz, witness files (no witness frame is localised: F6 does not read their poses).

    python step10a_handover.py <session scan dir> <root of the cube sessions>"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)


def main() -> int:
    scan, root = Path(sys.argv[1]), Path(sys.argv[2])
    out = scan / "output"
    from intake.quality import read_session_epochs
    from precision import refine as R
    from precision.camera import load_camera_json, save_camera_json, undistort_solver
    from precision.config import load_precision_config, raw_of
    from precision.tracks import load_tracks_v2
    from reconstruction.loops.config import improvement_error_factor
    pcfg = load_precision_config(session_dir=scan)
    log = lambda m: print(m, flush=True)   # noqa: E731
    cam0 = load_camera_json(out / "camera.json")
    dirs = sorted(root.glob("cube_*"), key=lambda p: int(p.name.split("_")[1]))
    cams = [load_camera_json(d / "output" / "camera.json") for d in dirs]
    if any(tuple(c.params) != tuple(cams[0].params) for c in cams):
        raise SystemExit("the cubes do not hold one and the same camera")
    params = [float(v) for v in cams[0].params]
    epochs = read_session_epochs(scan)
    poses = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]

    R.use_device(pcfg.refine.triangulation_device)
    tr = load_tracks_v2(scan)
    split = R.split_of_observations(tr["track_query_id"], tr["track_split"], tr["obs_track"])
    sigma = float(np.median([R._gauge_sigma(d / "output")[0] for d in dirs]))
    core = R.refine_core(np.linalg.inv(poses), params, (cam0.width, cam0.height), tr["obs_track"], tr["obs_frame"],
                         np.asarray(tr["obs_uv_native"], np.float64), split, kf, sigma, pcfg.refine,
                         undistort_solver(pcfg.camera), error_factor=float(improvement_error_factor(raw_of(pcfg))),
                         log=lambda m: None, solve=False, range_m=float(pcfg.view_range_m))
    held = core["held"]["init"]
    hv = np.asarray([v for v in held.values() if np.isfinite(v)], np.float64)
    log(f"[step 10a] the session's final cameras (F5 per cube, tied, in metres) with the one camera: held-out median "
        f"{np.median(hv):.4f} px over {len(hv):,} held-out tracks; {len(core['X']):,} landmarks from the fit tracks")

    pdir = out / "precision"
    (pdir / R.WITNESS_POSES_NAME).write_text("")
    (pdir / R.WITNESS_FRAMES_NAME).write_text("\n")
    np.savez_compressed(pdir / R.RESIDUALS_NAME,
                        heldout_track=np.array(list(held.keys()), np.int64),
                        heldout_rms_px=np.array(list(held.values()), np.float32),
                        init_track=np.array(list(held.keys()), np.int64),
                        init_rms_px=np.array(list(held.values()), np.float32))
    rung = json.loads((dirs[0] / "output" / "precision" / R.REFINE_NAME).read_text()).get("applied_rung")
    new_cam = cam0.with_params(params, "refine", cam0.camera_epoch + 1, report={"rung": rung, "blocks": 1})
    save_camera_json(out / "camera.json", new_cam, epochs.get("geometry_epoch"))
    doc = {"version": R.REFINE_VERSION, "provenance": R.PROVENANCE, **epochs,
           "params": dataclasses.asdict(pcfg.refine),
           "new_sequence": "the new order: these cameras are F5's — adjusted cube by cube with one camera (step 6), "
                           "then tied by the pipeline's graph (step 8) and put in metres (step 9). No session-level "
                           "F5 ran; this report is written in F5's format so that the pipeline's F6 starts "
                           "(experiments/new_sequence/step10a_handover.py)",
           "applied_rung": rung, "camera": {"before": list(cam0.params), "after": [params]},
           "n_landmarks": len(core["X"]), "heldout_median_px": float(np.median(hv)),
           "witnesses": {"n": 0, "localized": 0, "unlocalized": {}, "rms_px": {}},
           "applied": True, "epoch_to": int(epochs["geometry_epoch"]), "camera_epoch_to": new_cam.camera_epoch}
    (pdir / R.REFINE_NAME).write_text(json.dumps(doc, indent=1, default=float))
    log(f"[step 10a] camera.json = the one camera (fx {params[0]:.2f} fy {params[1]:.2f}), camera epoch "
        f"{new_cam.camera_epoch}; F5's report written for geometry epoch {doc['epoch_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
