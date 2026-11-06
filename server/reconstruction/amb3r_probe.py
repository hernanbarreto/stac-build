"""The structure probe by AMB3R-SLAM (USER 2026-11-06: "para recorridos de cientos de metros y loops … lo que importa es
que las poses sean lo más reales posibles en la primer pasada … incorporemos amb3r-slam, con todo su slam, para la
primer pasada, luego de eso se hace el corte").

The first pass only has to say WHERE every view is and that a place the walk came back to is the same place — the cut
into cubes (reconstruction/cubes.py) is made on it, Omega then reconstructs each cube. One Omega pass over the whole walk
has no loop closure and no room on the card for hundreds of metres; AMB3R-SLAM (vendor/AMB3R-SLAM, branch `stac`: a
DA3-Small front-end tracking, the heavy model building submaps, long-range Sim(3) edges, verified loops, an offline
pose graph) gives every view a pose on one closed trajectory.

It reads EXACTLY the probe's views — the session's keyframes and turn frames, in frame order (hard links in one
folder: its `demo` dataset reads a folder sorted by name) — every one of them in its submaps (frame_stride 1: they are
already the intake's selection, not a video to subsample), at the session's Omega grid, and dumps one record per view
(its --stac_dump: depth along the world camera's axis, the pinhole K its points follow, camera-to-world through the
submap's final Sim(3), confidence, sky). The probe npz is the cube plan's input as before: c2w, K, depth, conf, paths.

Run as the trials ran it (2026-11-03): its own env, torch on 8 threads and 16 cores (its default threads on this
252-core box ran it 7x slower), Hugging Face offline; under repro's deterministic numerics (reconstruction/
amb3r_launch.py). A sequence AMB3R records as failed, or a view without a record, FAILS the probe."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

AMB3R = Path(__file__).resolve().parents[2] / "vendor" / "AMB3R-SLAM"
LAUNCHER = Path(__file__).resolve().with_name("amb3r_launch.py")
CONDA_SH = "/workspace/miniforge3/etc/profile.d/conda.sh"
ENV_NAME = "amb3r"
CORES = "0-15"                 # the trials (2026-11-03): its default threads on this box ran it 7x slower
THREADS = 8
OVERRIDES = ("backend.frame_stride=1",)   # every probe view in the submaps: they are the intake's own selection


class Amb3rProbeError(RuntimeError):
    pass


def identity(model: str, grid_wh: Tuple[int, int]) -> Dict:
    """What the probe's views are reconstructed by: AMB3R's commit (and whether its tree is dirty), the launcher, the
    model, the grid and the overrides — part of the probe's stamp."""
    def _git(*a):
        r = subprocess.run(["git", "-C", str(AMB3R), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        raise Amb3rProbeError(f"{AMB3R} is not a git checkout — the probe cannot say which AMB3R-SLAM ran")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"probe": "amb3r", "amb3r_commit": commit, "amb3r_dirty": dirty, "model": str(model),
            "grid_wh": [int(grid_wh[0]), int(grid_wh[1])], "overrides": list(OVERRIDES),
            "launcher": hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()}


def run_amb3r_probe(view_paths: Sequence[str], out_npz: Path, *, model: str, grid_wh: Tuple[int, int],
                    log: Callable[[str], None]) -> Path:
    """AMB3R-SLAM over ``view_paths`` (frame order) → ``out_npz`` (module docstring)."""
    import da3_weights
    import repro
    out_npz = Path(out_npz)
    work = out_npz.with_name("amb3r_probe")
    shutil.rmtree(work, ignore_errors=True)
    fdir, run_dir, dump = work / "frames", work / "run", work / "dump"
    fdir.mkdir(parents=True)
    names = [Path(p).name for p in view_paths]
    if names != sorted(names) or len(set(names)) != len(names):
        raise Amb3rProbeError("the probe's views are not distinct and in name (frame) order — AMB3R's folder "
                              "reader would read them in another order")
    for p in view_paths:
        try:
            os.link(p, fdir / Path(p).name)
        except OSError:
            os.symlink(os.path.abspath(p), fdir / Path(p).name)
    # AMB3R's model reads a grid on its own patch (VGGT-Ω 16, DA3 14): the Omega grid as it is for Ω, the nearest
    # multiples of 14 for DA3
    patch = 16 if str(model) == "omega" else 14
    W, H = (max(2 * patch, int(round(int(v) / patch)) * patch) for v in grid_wh)
    args = ["--dataset", "demo", "--data_path", str(fdir), "--stride", "1", "--model_name", str(model),
            "--res_hw", f"{W}x{H}", "--save_map", "--save_every", "1", "--stac_dump", str(dump),
            "--out", str(run_dir), "--override", *OVERRIDES]
    cmd = ["bash", "-c", f"source {CONDA_SH} && conda activate {ENV_NAME} && exec taskset -c {CORES} python -u "
           f"{shlex.quote(str(LAUNCHER))} " + " ".join(shlex.quote(a) for a in args)]
    env = repro.deterministic_env()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(THREADS)
    env.update({"HF_HUB_OFFLINE": "1", "HF_HOME": da3_weights.HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    log(f"[amb3r] AMB3R-SLAM over {len(view_paths)} views ({model} at {W}x{H}, every view in its submaps)")
    p = subprocess.Popen(cmd, cwd=str(AMB3R), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-40:]
        if any(t in line for t in ("===", "-->", "loops", "stac dump", "map:", "config:", "Error", "Traceback",
                                   "[amb3r-launch]", "input resolution")):
            log(f"[amb3r] {line}")
    if p.wait() != 0:
        raise Amb3rProbeError("AMB3R-SLAM failed:\n" + "\n".join(tail[-20:]))
    res_p = run_dir / "results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    seq = fdir.name
    rec = res.get(seq) or (next(iter(res.values())) if len(res) == 1 else None)
    if rec is None or "error" in rec:
        raise Amb3rProbeError(f"AMB3R-SLAM recorded the sequence as failed: {(rec or {}).get('error')}\n"
                              + "\n".join(tail[-20:]))
    traj = sorted(run_dir.glob("traj_*.npz"))
    if len(traj) != 1:
        raise Amb3rProbeError(f"{run_dir}: {len(traj)} trajectories (one expected)")
    S = len(view_paths)
    pred = np.asarray(np.load(traj[0])["pred"], np.float64)
    if len(pred) != S:
        raise Amb3rProbeError(f"AMB3R-SLAM gave {len(pred)} cameras for {S} views")
    files = {int(np.load(q)["frame"]): q for q in sorted(dump.glob("frame_*.npz"))}
    missing = [i for i in range(S) if i not in files]
    if missing:
        raise Amb3rProbeError(f"{len(missing)} view(s) without an AMB3R record (first {view_paths[missing[0]]}) — "
                              f"every probe view must be in a submap")
    c2w = np.zeros((S, 4, 4), np.float32)
    K = np.zeros((S, 3, 3), np.float32)
    depth = conf = None
    kres = []
    for i in range(S):
        z = np.load(files[i])
        d = np.asarray(z["depth"], np.float32)
        if depth is None:
            depth = np.zeros((S,) + d.shape, np.float32)
            conf = np.zeros((S,) + d.shape, np.float32)
        if d.shape != depth.shape[1:]:
            raise Amb3rProbeError(f"view {i}: record grid {d.shape}, the others {depth.shape[1:]}")
        depth[i], conf[i] = d, np.asarray(z["conf"], np.float32)
        c2w[i] = np.asarray(z["c2w"], np.float64)
        K[i] = np.asarray(z["K"], np.float64)
        kres.append(float(z["k_residual_px"]))
    meta = {"identity": identity(model, grid_wh), "frames": int(rec.get("frames", S)),
            "keyframes": int(rec.get("keyframes", 0)), "loops": int(rec.get("loops", 0)),
            "time_s": float(rec.get("time_s", 0.0)), "peak_mem_gb": float(rec.get("peak_mem_gb", 0.0)),
            "k_residual_px_median": float(np.median(kres)),
            "traj_vs_record_m_max": float(np.max(np.linalg.norm(pred[:, :3, 3] - c2w[:, :3, 3], axis=1)))}
    tmp = out_npz.with_name(out_npz.name + ".part.npz")
    np.savez(tmp, c2w=c2w, K=K, depth=depth, conf=conf, paths=np.array([str(q) for q in view_paths]),
             resolution=np.int32(max(W, H)), source=np.array("amb3r"), meta=np.array(json.dumps(meta)))
    os.replace(tmp, out_npz)
    # the folder of links, the map and the dump are consumed; the run's small records stay beside the probe
    shutil.rmtree(fdir, ignore_errors=True)
    shutil.rmtree(dump, ignore_errors=True)
    for q in run_dir.glob("map_*.ply"):
        q.unlink()
    log(f"[amb3r] probe → {out_npz.name}: {S} views on a {depth.shape[2]}x{depth.shape[1]} grid, "
        f"{meta['keyframes']} AMB3R keyframes, {meta['loops']} loop closure(s), {meta['time_s']:.0f} s, peak "
        f"{meta['peak_mem_gb']:.1f} GB")
    return out_npz
