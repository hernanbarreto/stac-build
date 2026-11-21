"""THE NEW ORDER, step 8 (first part): the cubes the tie reads, already corrected.

The pipeline's tie (CubeRun.seams / bridges / closures / graph / apply) reads each cube file as Omega left it: every
keyframe's points, their confidence and its camera. Here each step-2 cube file is written again with what steps 4, 6
and 7 corrected IN THAT CUBE, and nothing else changed:

  * the camera of every keyframe  = the cube's camera after F2 and F5 (step 6), in the cube's own frame;
  * the depth of every keyframe   = the corrected depth of step 7 (the factor fitted on the correspondences, then
    PointDiT at the edges);
  * the points                    = that depth seen from that camera (as the fork itself builds a cube's points from
    its depth and camera);
  * the confidence                = Omega's, and 0 where step 7 left no depth (the tie skips a pixel of confidence 0).

Step 7's depth lives on the undistorted pixel grid of the one camera (the pipeline's F6 convention); a cube file is on
the original pixel grid, the grid of Omega's bridges, which the tie matches pixel to pixel. Each original pixel
therefore takes the depth of its own ray: the lens is removed from the pixel (precision.camera.undistort_points, the
pipeline's function) and the corrected depth is read at the nearest pixel of the undistorted grid.

    python step8_cubes.py <step-2 run dir> <root of the cube sessions> <step-7 dir> <tie save dir (maplong_run)>"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)


def main() -> int:
    run, root, d7, tie = (Path(a) for a in sys.argv[1:5])
    from precision.camera import load_camera_json, undistort_points, undistort_solver
    from precision.config import load_precision_config
    log = lambda m: print(m, flush=True)   # noqa: E731
    for name in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd"):
        (tie / name).mkdir(parents=True, exist_ok=True)
    for name in ("fork_stamp.json", "metric_lock.json", "frame_list.json", "omega_environment.json", "omega_load.json",
                 "vggt_omega_config.yaml"):
        shutil.copyfile(run / name, tie / name)
    for name in ("sky_masks", "context_views"):
        if not os.path.lexists(tie / name):
            os.symlink(run / name, tie / name)

    dirs = sorted(root.glob("cube_*"), key=lambda p: int(p.name.split("_")[1]))
    pcfg = load_precision_config(session_dir=dirs[0])
    solver = undistort_solver(pcfg.camera)
    cam = load_camera_json(dirs[0] / "output" / "camera.json")           # the ONE camera (step 6)
    fx, fy, cx, cy = (float(v) for v in cam.params[:4])
    W, H = int(cam.width), int(cam.height)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel()], 1)
    und = undistort_points(pts, cam, **solver) if np.any(cam.dist()) else pts
    iu, iv = np.rint(und[:, 0]).astype(np.int64), np.rint(und[:, 1]).astype(np.int64)
    inside = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H) & np.isfinite(und).all(1)
    xn, yn = (und[:, 0] - cx) / fx, (und[:, 1] - cy) / fy                 # each original pixel's ray, lens removed
    log(f"[step 8] the one camera: fx {fx:.2f} fy {fy:.2f}, lens {[round(float(v), 5) for v in cam.params[4:8]]}; an "
        f"original pixel moves up to {np.abs(und - pts)[inside].max():.2f} px when the lens is removed; "
        f"{int((~inside).sum()):,} of {len(pts):,} original pixels fall outside the undistorted grid")
    Kf = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)

    for d in dirs:
        k = int(d.name.split("_")[1])
        cube = np.load(run / "_tmp_results_unaligned" / f"chunk_{k}.npy", allow_pickle=True).item()
        z = np.load(d7 / f"depth_{d.name}.npz")
        frames = [int(float(x)) for x in (d / "output" / "camera_frames.txt").read_text().split()]
        if [int(f) for f in z["frames"]] != frames or z["dep"].shape[1:] != (H, W) or len(frames) != len(cube["depth"]):
            raise SystemExit(f"{d.name}: step 7's maps are not this cube's keyframes on the camera's grid")
        n_ok = n_was = 0
        for i in range(len(frames)):
            zz = np.zeros(H * W, np.float64)
            zz[inside] = z["dep"][i][iv[inside], iu[inside]]
            ok = zz > 0
            c2w = np.asarray(z["c2w"][i], np.float64)
            Xw = np.stack([xn * zz, yn * zz, zz], 1) @ c2w[:3, :3].T + c2w[:3, 3]
            was = np.asarray(cube["world_points_conf"][i]).reshape(-1) > 1e-5
            n_ok += int((ok & was).sum()); n_was += int(was.sum())
            cube["world_points"][i] = Xw.reshape(H, W, 3).astype(np.float32)
            cube["depth"][i] = zz.reshape(H, W).astype(np.float32)
            cube["world_points_conf"][i] = np.where(ok.reshape(H, W), cube["world_points_conf"][i], 0).astype(np.float32)
            cube["depth_conf"][i] = np.where(ok.reshape(H, W), cube["depth_conf"][i], 0).astype(np.float32)
            cube["extrinsic"][i] = c2w.astype(np.float32)
            cube["intrinsic"][i] = Kf
        cube["_stac_new_sequence"] = "corrected cube: cameras after F2 + F5, depth after the factor + PointDiT (steps 4, 6, 7)"
        np.save(tie / "_tmp_results_unaligned" / f"chunk_{k}.npy", cube)
        log(f"[step 8] cube {k}: {len(frames)} keyframes written corrected; {n_ok:,} of the {n_was:,} pixels Omega "
            f"gave a confidence keep a depth ({100.0 * n_ok / max(n_was, 1):.1f} %)")
        del cube
    (tie / "corrected_cubes.json").write_text(json.dumps({"cubes": len(dirs), "camera": [fx, fy, cx, cy]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
