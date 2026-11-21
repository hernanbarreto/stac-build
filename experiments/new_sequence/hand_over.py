"""THE NEW ORDER, steps 6 and 7: METRES, and the cameras handed to the test session through F5's own door.

Step 6 — metres: the tied cameras (one pose per keyframe, the first cube fixed) are carried onto the session's own frame
— the metres and the upright frame the cube stage's metre step left (epoch 0) — with ONE similarity for the whole
session, fitted on the 174 cameras. One number of scale for everything; no shape is taken from the old cameras.

Step 7 — the hand-over: exactly what F5 writes when it applies its result (precision.refine.run_refine): the ONE
camera into camera.json, the poses as the next geometry epoch (precision.poses_epoch.apply_pose_epoch, kind "refine"),
F5's report and residual files — so the pipeline's F6, unchanged, starts from it. The held-out error written there is
MEASURED on the final state with F5's own function (refine_core, solve off): the session's held-out tracks, triangulated
and reprojected with the cameras handed over. No witness frame is localised (F6 does not read their poses).

    python hand_over.py <scan dir> <tied npz> <f5_per_cube npz> [--measure-only]"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from f5_per_cube import SERVER, similarity  # noqa: E402,F401


def main() -> int:
    scan, tied_npz, f5_npz = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    measure_only = "--measure-only" in sys.argv[4:]
    out = scan / "output"
    from config import cfg as raw_cfg
    from intake.quality import read_session_epochs
    from precision import refine as R
    from precision.camera import load_camera_json, save_camera_json, undistort_solver
    from precision.config import load_precision_config
    from precision.tracks import load_tracks_v2
    from reconstruction.loops.config import improvement_error_factor
    pcfg = load_precision_config()
    log = lambda m: print(m, flush=True)   # noqa: E731

    z = np.load(tied_npz, allow_pickle=False)
    kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
    if kf != [int(x) for x in z["kf_frames"]]:
        raise SystemExit("the tied cameras are not this session's keyframes")
    epochs = read_session_epochs(scan)
    if int(epochs["geometry_epoch"]) != 0:
        raise SystemExit(f"the session is at geometry epoch {epochs['geometry_epoch']}: the metre step's cameras are "
                         f"epoch 0's — nothing is handed over")
    cam = load_camera_json(out / "camera.json")
    sess = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    tied, params, rung = z["c2w"], [float(v) for v in z["params"]], str(z["rung"])

    # step 6: metres
    s, Q, t = similarity(tied[:, :3, 3], sess[:, :3, 3], tied[:, :3, :3], sess[:, :3, :3])
    c2w = np.tile(np.eye(4), (len(kf), 1, 1))
    c2w[:, :3, :3] = np.einsum("ij,njk->nik", Q, tied[:, :3, :3])
    c2w[:, :3, 3] = s * tied[:, :3, 3] @ Q.T + t
    d = np.linalg.norm(c2w[:, :3, 3] - sess[:, :3, 3], axis=1)
    path = lambda P: float(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1).sum())   # noqa: E731
    log(f"[metres] ONE similarity onto the session's frame: scale x{s:.5f}; the new cameras are {np.median(d) * 100:.2f} cm "
        f"(median) / {d.max() * 100:.2f} cm (most) from the cube stage's; the walk measures {path(c2w):.3f} m "
        f"(the cube stage's: {path(sess):.3f} m)")

    # the held-out, measured with F5's own function on both states
    tr = load_tracks_v2(scan)
    split = R.split_of_observations(tr["track_query_id"], tr["track_split"], tr["obs_track"])
    g = json.loads(Path(SERVER, "projects/bufferStop_prueba/scans/2026-08-11/src_default/output/gauge.json").read_text())
    sigma_rel = float(g["sigma_by_instrument"][g["applied_instrument"]])
    R.use_device("cuda")
    fac = float(improvement_error_factor(dict(raw_cfg)))

    def state(w2c, p, name):
        core = R.refine_core(w2c, list(p), (cam.width, cam.height), tr["obs_track"], tr["obs_frame"],
                             np.asarray(tr["obs_uv_native"], np.float64), split, kf, sigma_rel, pcfg.refine,
                             undistort_solver(pcfg.camera), error_factor=fac, log=lambda m: None, solve=False,
                             range_m=float(pcfg.view_range_m))
        h = np.asarray(list(core["held"]["init"].values()), np.float64)
        ok = np.isfinite(h)
        log(f"[held-out] {name}: median {np.median(h[ok]):.4f} px over {int(ok.sum()):,} held-out tracks "
            f"({len(core['X']):,} landmarks triangulate from the fit tracks)")
        return core
    state(np.linalg.inv(sess), cam.params, "the cube stage's cameras + F0's camera (where today's F5 starts)")
    core = state(np.linalg.inv(c2w), params, "THE NEW ORDER's cameras + its one camera (what F6 receives)")
    if measure_only:
        return 0

    # step 7: F5's own door
    pdir = out / "precision"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / R.WITNESS_POSES_NAME).write_text("")
    (pdir / R.WITNESS_FRAMES_NAME).write_text("\n")
    held = core["held"]["init"]
    np.savez_compressed(pdir / R.RESIDUALS_NAME,
                        heldout_track=np.array(list(held.keys()), np.int64),
                        heldout_rms_px=np.array(list(held.values()), np.float32),
                        init_track=np.array(list(held.keys()), np.int64),
                        init_rms_px=np.array(list(held.values()), np.float32))
    f5 = np.load(f5_npz, allow_pickle=False)
    tie_report = json.loads(str(z["report"]))
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    doc = {"version": R.REFINE_VERSION, "provenance": R.PROVENANCE, **epochs,
           RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out),
           "params": dataclasses.asdict(pcfg.refine),
           "new_sequence": {"what": "the experiment of 2026-11-20 — F5 per cube with one camera, then the tie of the "
                                    "cubes (the first fixed), then metres; written by experiments/new_sequence/hand_over.py "
                                    "in F5's format so that the pipeline's F6 starts from it",
                            "f5_per_cube_heldout_px": json.loads(str(f5["heldout_px"])),
                            "tie": tie_report, "metres_scale": s,
                            "moved_from_cube_stage_cm": {"median": float(np.median(d) * 100), "max": float(d.max() * 100)}},
           "applied_rung": rung, "camera": {"before": list(cam.params), "after": [params]},
           "n_landmarks": len(core["X"]),
           "heldout_median_px": float(np.median([v for v in held.values() if np.isfinite(v)])),
           "witnesses": {"n": 0, "localized": 0, "unlocalized": {}, "rms_px": {}}, "applied": False}
    new_cam = cam.with_params(params, "refine", cam.camera_epoch + 1, report={"rung": rung, "blocks": 1})
    save_camera_json(out / "camera.json", new_cam, epochs.get("geometry_epoch"))
    T = c2w @ np.linalg.inv(sess)                      # c2w_new · w2c_old: each keyframe's world motion
    from precision.poses_epoch import apply_pose_epoch
    res = apply_pose_epoch(out, T[:, :3, :3], T[:, :3, 3], np.ones(len(T)), "refine",
                           [{"stage": "refine", "rung": rung, "new_sequence": True}], log=log)
    doc["applied"] = bool(res)
    doc["epoch_to"] = (res or {}).get("epoch_to")
    doc["camera_epoch_to"] = new_cam.camera_epoch
    (pdir / R.REFINE_NAME).write_text(json.dumps(doc, indent=1, default=float))
    back = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    log(f"[hand-over] camera.json (camera epoch {new_cam.camera_epoch}) and the poses as geometry epoch {doc['epoch_to']}; "
        f"the session's cameras now equal the new order's within {np.abs(back - c2w).max():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
