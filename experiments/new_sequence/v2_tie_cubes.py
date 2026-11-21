"""VERSION 2, step 11 (first part): the cubes the second tie reads — the session as it stands after F5 and after F6's
depth stage.

The pipeline's tie (CubeRun.bridges / seams / closures / graph) reads each cube file: every keyframe's points, their
confidence and its camera. Each cube the first tie placed (kept by the pipeline's own switch,
certify.keep_aligned_chunks) is written again with the session's CURRENT state of each of its keyframes, and nothing
else changed:

  * the camera   = the keyframe's camera after F2 and F5 (one per keyframe: the session's);
  * the depth    = the keyframe's depth after F6's factor (step 10, v2_f6.py depth);
  * the points   = that depth seen from that camera;
  * confidence   = Omega's in that cube, and 0 where F6 left no depth (the tie skips a pixel of confidence 0).

So every cube is written in the session's frame and metres, and two cubes that share a keyframe hold the SAME points
for it: the seams of the second tie measure nothing (the user's order: "la segunda atada es solo con los puentes omega").

F6's depth lives on the undistorted pixel grid of the session's camera; a cube file is on the original pixel grid, the
grid of Omega's bridges, which the tie matches pixel to pixel: each original pixel takes the depth of its own ray (the
lens removed with the pipeline's precision.camera.undistort_points, the depth read at the nearest undistorted pixel).

    python v2_tie_cubes.py <scan dir> <state.npz of step 10> <tie dir>"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    scan, state, tie = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    out = scan / "output"
    ml = out / "maplong_run"
    from precision.camera import load_camera_json, undistort_points, undistort_solver
    from precision.config import load_precision_config
    from v2_common import config_path, layout_of
    log = lambda m: print(m, flush=True)   # noqa: E731
    sets, owner, names = layout_of(out)
    num_of = [int(Path(n).stem) for n in names]
    z = np.load(state)
    frames = [int(f) for f in z["frames"]]
    idx = {f: i for i, f in enumerate(frames)}
    if sorted(frames) != sorted(num_of):
        raise SystemExit("step 10's keyframes are not the session's keyframes")
    if tie.exists():
        shutil.rmtree(tie)
    for name in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd"):
        (tie / name).mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("fork_stamp.json", "metric_lock.json", "frame_list.json", "omega_environment.json", "omega_load.json"):
        if (ml / name).exists():
            shutil.copyfile(ml / name, tie / name)
            copied.append(name)
    shutil.copyfile(config_path(out), tie / "vggt_omega_config.yaml")
    for name in ("sky_masks", "context_views"):
        if (ml / name).exists() and not os.path.lexists(tie / name):
            os.symlink((ml / name).resolve(), tie / name)
    log(f"[step 11] the tie's directory {tie}: {copied} + the fork's configuration copied from the session's run")

    pcfg = load_precision_config(session_dir=scan)
    solver = undistort_solver(pcfg.camera)
    cam = load_camera_json(out / "camera.json")                 # the session's ONE camera (F5)
    fx, fy, cx, cy = (float(v) for v in cam.params[:4])
    W, H = int(cam.width), int(cam.height)
    if tuple(z["dep"].shape[1:]) != (H, W):
        raise SystemExit(f"step 10's maps {z['dep'].shape[1:]} are not on the camera's grid {(H, W)}")
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel()], 1)
    und = undistort_points(pts, cam, **solver) if np.any(cam.dist()) else pts
    iu, iv = np.rint(und[:, 0]).astype(np.int64), np.rint(und[:, 1]).astype(np.int64)
    inside = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H) & np.isfinite(und).all(1)
    xn, yn = (und[:, 0] - cx) / fx, (und[:, 1] - cy) / fy       # each original pixel's ray, lens removed
    log(f"[step 11] the session's camera: fx {fx:.2f} fy {fy:.2f}, lens {[round(float(v), 5) for v in cam.params[4:8]]}; "
        f"an original pixel moves up to {np.abs(und - pts)[inside].max():.2f} px when the lens is removed; "
        f"{int((~inside).sum()):,} of {len(pts):,} original pixels fall outside the undistorted grid")
    Kf = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)

    src_dir = ml / "_tmp_results_aligned"
    for k, F in enumerate(sets):
        p = src_dir / f"chunk_{k}.npy"
        if not p.exists():
            raise SystemExit(f"{p} is not on disk — the placed cubes were not kept (certify.keep_aligned_chunks)")
        cube = np.load(p, allow_pickle=True).item()
        fr = [num_of[g] for g in F]
        if len(fr) != len(cube["depth"]) or tuple(np.asarray(cube["depth"][0]).shape[-2:]) != (H, W):
            raise SystemExit(f"cube {k}: {len(cube['depth'])} views of {np.asarray(cube['depth'][0]).shape} on disk, "
                             f"the layout says {len(fr)} keyframes on {(H, W)}")
        n_ok = n_was = 0
        for i, f in enumerate(fr):
            j = idx[f]
            zz = np.zeros(H * W, np.float64)
            zz[inside] = z["dep"][j][iv[inside], iu[inside]]
            ok = zz > 0
            c2w = np.asarray(z["c2w"][j], np.float64)
            Xw = np.stack([xn * zz, yn * zz, zz], 1) @ c2w[:3, :3].T + c2w[:3, 3]
            was = np.asarray(cube["world_points_conf"][i]).reshape(-1) > 1e-5
            n_ok += int((ok & was).sum())
            n_was += int(was.sum())
            cube["world_points"][i] = Xw.reshape(H, W, 3).astype(np.float32)
            cube["depth"][i] = zz.reshape(np.asarray(cube["depth"][i]).shape).astype(np.float32)
            cube["world_points_conf"][i] = np.where(ok.reshape(H, W), cube["world_points_conf"][i], 0).astype(np.float32)
            cube["depth_conf"][i] = np.where(ok.reshape(H, W), cube["depth_conf"][i], 0).astype(np.float32)
            cube["extrinsic"][i] = c2w.astype(np.float32)[:np.asarray(cube["extrinsic"][i]).shape[0]]
            cube["intrinsic"][i] = Kf
        cube.pop("_stac_cube_pose", None)
        cube["_stac_new_sequence"] = "version 2, second tie: the session's cameras after F5 and depth after F6's factor"
        np.save(tie / "_tmp_results_unaligned" / f"chunk_{k}.npy", cube)
        log(f"[step 11] cube {k}: {len(fr)} keyframes written in the session's state; {n_ok:,} of the {n_was:,} pixels "
            f"Omega gave a confidence keep a depth ({100.0 * n_ok / max(n_was, 1):.1f} %)")
        del cube
    (tie / "second_tie_cubes.json").write_text(json.dumps({"cubes": len(sets), "camera": [fx, fy, cx, cy]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
