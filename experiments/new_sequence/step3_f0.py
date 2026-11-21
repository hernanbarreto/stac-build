"""THE NEW ORDER, step 3 — "F0: como hoy."

The pipeline's F0 (python -m precision.camera --session <scan>), unchanged. It reads Omega's focal lengths per keyframe
from output/maplong_run/intrinsic.txt. In today's order that file is written by the last method of the tie
(CubeRun.camera_poses: each keyframe's K from the cube that owns it); in the new order the tie comes later, so the file
is written HERE from the cubes of step 2 — the same numbers (each keyframe's K in its owner cube), with the pipeline's
own writer (repro.write_intrinsics_exact) — together with the keyframe list (camera_frames.txt, as the map worker
writes it), Omega's configuration and the job's frozen configuration.

    python step3_f0.py <new session scan dir> <step-2 run dir (maplong_run)> <session the frozen config is taken from>"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)


def main() -> int:
    scan, run, cfg_from = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out = scan / "output"
    (out / "maplong_run").mkdir(parents=True, exist_ok=True)
    import repro
    cfg = yaml.safe_load((run / "vggt_omega_config.yaml").read_text())
    sets, owner = cfg["Model"]["chunk_sets"], [int(x) for x in cfg["Model"]["chunk_owner"]]
    rows = np.full((len(owner), 4), np.nan)
    for k, F in enumerate(sets):
        K = np.asarray(np.load(run / "_tmp_results_unaligned" / f"chunk_{k}.npy", allow_pickle=True).item()["intrinsic"],
                       np.float64)
        K = K[0] if K.ndim == 4 else K
        for i, g in enumerate(F):
            if owner[int(g)] == k:
                rows[int(g)] = [K[i][0, 0], K[i][1, 1], K[i][0, 2], K[i][1, 2]]
        print(f"[step 3] cube {k}: {sum(1 for g in F if owner[int(g)] == k)} of its {len(F)} keyframes are its own")
    if not np.isfinite(rows).all():
        raise SystemExit("a keyframe got no focal from its owner cube")
    repro.write_intrinsics_exact(out / "maplong_run" / "intrinsic.txt", rows)
    names = json.loads((run / "frame_list.json").read_text())
    nums = [str(int(re.search(r"(\d+)", str(n)).group(1))) for n in names]
    (out / "camera_frames.txt").write_text("\n".join(nums) + "\n")
    shutil.copyfile(run / "frame_list.json", out / "frame_list.json")
    shutil.copyfile(run / "vggt_omega_config.yaml", out / "vggt_omega_config.yaml")
    for name in ("run_config.yaml", "run_config.sha256"):
        shutil.copyfile(cfg_from / "output" / name, out / name)
    print(f"[step 3] Omega's focals of the {len(rows)} keyframes: fx median {np.median(rows[:, 0]):.2f} "
          f"({rows[:, 0].min():.2f} .. {rows[:, 0].max():.2f}), fy median {np.median(rows[:, 1]):.2f} "
          f"({rows[:, 1].min():.2f} .. {rows[:, 1].max():.2f})")

    # the pipeline's F0, launched as the pipeline's runner launches a step
    from precision import runner as RN
    from precision.config import load_precision_config
    pcfg = load_precision_config(session_dir=scan)
    step = next(s for s in RN.STEPS if s.key == "f0_camera")
    env = RN.step_env(RN.step_threads(step, pcfg.runner))
    cmd = [pcfg.runner.python_da3, "-u", "-m", step.module, "--session", str(scan), *step.args]
    print("[step 3] run: " + " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, cwd=SERVER, env=env)
    if r.returncode:
        return r.returncode
    cam = json.loads((out / "camera.json").read_text())
    print(f"[step 3] camera.json: {json.dumps({k: cam[k] for k in cam if k in ('model', 'params', 'width', 'height', 'source', 'camera_epoch')})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
