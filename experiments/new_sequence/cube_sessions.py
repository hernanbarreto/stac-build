"""A cube as a session of its own — the view the pipeline's per-session steps (F2, F5's door, F6) are run on, cube by cube.

For cube k of the step-2 run a directory in the pipeline's own session layout is made:

    frames              -> the test session's frames (link)
    intake/walk.json    the walk of the CUBE: the session's walk restricted to the cube's keyframes, its chainage
                        starting at 0 at the cube's first keyframe; every other intake file is the session's (link)
    output/camera_poses.txt, intrinsic.txt   the cube's own cameras (its metric-locked extrinsics, in the cube's own
                        frame), written with the pipeline's writer (loop_utils.camera_files.write_camera_files)
    output/camera_frames.txt                 the cube's keyframes (video frame numbers)
    output/omega_run/results_output/         each keyframe's Omega record AS THIS CUBE HOLDS IT, written by the
                        pipeline's own function (workers.map_worker._emit_omega_depth) from the cube
    output/da3_windows  -> the session's I3 window depth (link)
    output/run_config.yaml (+ sha256), camera.json           the session's

Nothing is computed here but by the pipeline's functions; this module only lays the cube's files out."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

SERVER = "/workspace/stac-build/server"
FORK = "/workspace/stac-build/vendor/VGGT-Long"
for p in (SERVER, FORK):
    if p not in sys.path:
        sys.path.insert(0, p)


class _Pipe:
    """The worker pipe's log, to this process's output."""

    def send_log(self, msg, level="info"):
        print(msg, flush=True)

    def send_progress(self, *a, **k):
        pass

    def check_cancel(self):
        return False


def layout(run: Path):
    cfg = yaml.safe_load((run / "vggt_omega_config.yaml").read_text())
    names = json.loads((run / "frame_list.json").read_text())
    return [[int(g) for g in F] for F in cfg["Model"]["chunk_sets"]], [int(x) for x in cfg["Model"]["chunk_owner"]], names


def cube_walk(walk: dict, nums) -> dict:
    """The session's walk restricted to one cube: its keyframes' chainage from 0, the windows that reach it."""
    keep = set(int(n) for n in nums)
    ch = [dict(c) for c in walk["chainage"] if int(c["frame"]) in keep]
    if len(ch) != len(keep):
        raise RuntimeError("a keyframe of the cube has no chainage in the session's walk")
    c0 = float(ch[0]["chainage_m"])
    for c in ch:
        c["chainage_m"] = float(c["chainage_m"]) - c0
    lo, hi = min(keep), max(keep)
    wins = [dict(w) for w in walk["windows"] if int(w["frames"][0]) <= hi and int(w["frames"][-1]) >= lo]
    out = dict(walk)
    out.update({"chainage": ch, "windows": wins, "n_keyframes": len(ch), "n_windows": len(wins),
                "walk_length_m": float(ch[-1]["chainage_m"]),
                "cube_of_session_walk": {"first_frame": lo, "last_frame": hi, "chainage_offset_m": c0}})
    out.pop("seams", None)
    return out


def build(run: Path, scan: Path, k: int, dst: Path, log=print) -> Path:
    from loop_utils.camera_files import write_camera_files
    from workers.map_worker import _emit_omega_depth
    sets, _owner, names = layout(run)
    F = sets[k]
    files = [names[g] for g in F]
    nums = [int(Path(f).stem) for f in files]
    if nums != sorted(nums):
        raise RuntimeError(f"cube {k}: its keyframes are not in the order of the video")
    out, ml = dst / "output", dst / "output" / "maplong_run"
    (ml / "_tmp_results_aligned").mkdir(parents=True, exist_ok=True)
    (dst / "intake").mkdir(exist_ok=True)
    cube = run / "_tmp_results_unaligned" / f"chunk_{k}.npy"

    def link(src, to):
        if not os.path.lexists(to):
            os.symlink(src, to)
    link(scan / "frames", dst / "frames")
    for p in (scan / "intake").iterdir():
        if p.name != "walk.json":
            link(p, dst / "intake" / p.name)
    (dst / "intake" / "walk.json").write_text(json.dumps(cube_walk(json.loads((scan / "intake" / "walk.json").read_text()), nums)))
    link(scan / "output" / "da3_windows", out / "da3_windows")
    link(cube, ml / "_tmp_results_aligned" / "chunk_0.npy")
    for name in ("run_config.yaml", "run_config.sha256", "camera.json", "vggt_omega_config.yaml"):
        shutil.copyfile(scan / "output" / name, out / name)

    data = np.load(cube, allow_pickle=True).item()
    ext = np.asarray(data["extrinsic"], np.float64)
    ext = ext[0] if ext.ndim == 4 else ext
    K = np.asarray(data["intrinsic"], np.float64)
    K = K[0] if K.ndim == 4 else K
    if len(ext) != len(F):
        raise RuntimeError(f"cube {k} holds {len(ext)} views, its plan says {len(F)} keyframes")
    del data
    write_camera_files(str(out), [ext[i] for i in range(len(F))], [K[i] for i in range(len(F))])
    (out / "camera_frames.txt").write_text("\n".join(str(n) for n in nums) + "\n")
    sel = dst / "selected_frames.json"
    sel.write_text(json.dumps({"selected_files": files}))
    _emit_omega_depth(ml, out, None, str(sel), _Pipe(), cubes=([list(range(len(F)))], [0] * len(F)))
    log(f"[cube session] cube {k}: {len(F)} keyframes (video {nums[0]}..{nums[-1]}) laid out in {dst}")
    return dst
