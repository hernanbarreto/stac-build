"""The structure probe by AMB3R-SLAM over the WHOLE VIDEO (USER 2026-11-06: "a amber le tenés que dar el video completo, no
los kf del intake, nosotros además luego calculamos el intake con nuestros rellenos etc, y los colocamos en los cubos en
lugar de los kf que generó amber"; "amber a paso 1, no importa el tiempo, importa la precisión y la completitud").

AMB3R-SLAM (vendor/AMB3R-SLAM, branch `stac`: a DA3-Small front-end tracking every frame, the heavy model building
submaps, long-range Sim(3) edges, verified loops, an offline pose graph — it needs the video, not a selection, to find
its loops) reads every frame of the video (frame stride 1). Its backend reconstructs one frame in ``backend.frame_stride``
(4 at a video's rate: its own design) and its final trajectory gives EVERY frame a pose — a frame it did not reconstruct
gets the pose Sim(3)-interpolated between its two reconstructed neighbours (AMB3R's ``_frame_poses``). For every frame
it reconstructed, the light map (--stac_light_map) records what that frame SAW: its depth on the pixel-stride grid, its
camera-to-world and the pinhole K its points follow, placed by its submap's final Sim(3) — every point of the probe's
cloud with the frame it came from (USER 2026-11-06: "cada punto de la nube amber tiene los frames de procedencia … no
necesitás tampoco proyectar ningún rayo"). A view of ours (a keyframe, a turn frame) sees what its own frame saw when
AMB3R reconstructed it, else what the nearest reconstructed frame saw (ties → the earlier): half a backend step away at
most.

intake/amb3r_video.npz — ``names`` / ``frame_no`` (the video's frames in AMB3R's order), ``pred`` (every frame's
camera-to-world), ``kf`` (AMB3R's own keyframes, recorded), and per reconstructed frame ``rec_index`` (its position in
the video), ``rec_c2w``, ``rec_K`` (on the model grid), ``rec_depth`` (on the pixel-stride grid); ``grid_wh``,
``stride``, ``meta``. Stamped (intake/amb3r_video.stamp.json) by the video (the frames manifest's inventory sha256, or
the names and sizes), AMB3R's commit, the launcher, the model, the grid and the strides; reused while all are the same.
Run as the trials ran it (2026-11-03): its own env, torch on 8 threads and 16 cores, Hugging Face offline; under
repro's deterministic numerics (reconstruction/amb3r_launch.py). A sequence AMB3R records as failed FAILS the probe."""
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
FRAME_STRIDE = 1               # USER 2026-11-06: "amber a paso 1"
PIXEL_STRIDE = 4               # what a frame saw: its depth on one pixel in 4 x 4 of the model grid
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")   # AMB3R's demo reader
VIDEO_NAME = "amb3r_video.npz"
SOURCE = "amb3r-video"


class Amb3rProbeError(RuntimeError):
    pass


def video_frames(frames_dir: Path) -> list:
    """The frames AMB3R reads, in its order (its demo dataset: every image of the folder sorted by name)."""
    return sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(IMAGE_EXTS))


def identity(model: str, grid_wh: Tuple[int, int]) -> Dict:
    """What the probe is made by: AMB3R's commit (and whether its tree is dirty), the launcher, the model, the grid,
    the frame and pixel strides — part of the probe's stamp."""
    def _git(*a):
        r = subprocess.run(["git", "-C", str(AMB3R), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        raise Amb3rProbeError(f"{AMB3R} is not a git checkout — the probe cannot say which AMB3R-SLAM ran")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"probe": SOURCE, "amb3r_commit": commit, "amb3r_dirty": dirty, "model": str(model),
            "grid_wh": [int(grid_wh[0]), int(grid_wh[1])], "frame_stride": FRAME_STRIDE,
            "pixel_stride": PIXEL_STRIDE, "launcher": hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()}


def video_identity(frames_dir: Path) -> Dict:
    """The video AMB3R reads: the frames manifest's inventory sha256 (frames/manifest.json, every frame's bytes) when
    the intake wrote it, else the frames' names and sizes."""
    frames_dir = Path(frames_dir)
    names = video_frames(frames_dir)
    man = frames_dir / "manifest.json"
    inv = None
    if man.exists():
        try:
            inv = json.loads(man.read_text()).get("frames_sha256")
        except ValueError:
            inv = None
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        if inv is None:
            h.update(str((frames_dir / n).stat().st_size).encode())
    return {"n_frames": len(names), "names_sha256": h.hexdigest(), "frames_sha256": inv}


def model_grid(model: str, grid_wh: Tuple[int, int]) -> Tuple[int, int]:
    """The grid AMB3R's model reads: the session's Omega grid on the model's own patch (VGGT-Ω 16, DA3 14)."""
    patch = 16 if str(model) == "omega" else 14
    W, H = (max(2 * patch, int(round(int(v) / patch)) * patch) for v in grid_wh)
    return int(W), int(H)


def run_amb3r_video(frames_dir: Path, out_npz: Path, *, model: str, grid_wh: Tuple[int, int],
                    log: Callable[[str], None], check_cancel: Optional[Callable[[], bool]] = None) -> Path:
    """AMB3R-SLAM over every frame of ``frames_dir`` → ``out_npz`` (module docstring). ``check_cancel`` true stops it
    (intake.quality.IntakeCancelled)."""
    import da3_weights
    import repro
    out_npz = Path(out_npz)
    work = out_npz.with_name("amb3r_probe")
    shutil.rmtree(work, ignore_errors=True)
    run_dir = work / "run"
    run_dir.mkdir(parents=True)
    frames = video_frames(frames_dir)
    if len(frames) < 2:
        raise Amb3rProbeError(f"{frames_dir} holds {len(frames)} frame(s) — no video to run AMB3R-SLAM on")
    W, H = model_grid(model, grid_wh)
    args = ["--dataset", "demo", "--data_path", str(frames_dir), "--stride", str(FRAME_STRIDE),
            "--model_name", str(model), "--res_hw", f"{W}x{H}", "--save_map", "--stac_light_map",
            "--save_every", "1", "--pixel_stride", str(PIXEL_STRIDE), "--out", str(run_dir)]
    cmd = ["bash", "-c", f"source {CONDA_SH} && conda activate {ENV_NAME} && exec taskset -c {CORES} python -u "
           f"{shlex.quote(str(LAUNCHER))} " + " ".join(shlex.quote(a) for a in args)]
    env = repro.deterministic_env()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(THREADS)
    env.update({"HF_HUB_OFFLINE": "1", "HF_HOME": da3_weights.HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    log(f"[amb3r] AMB3R-SLAM over the whole video: {len(frames)} frames at stride {FRAME_STRIDE} ({model} at {W}x{H})")
    p = subprocess.Popen(cmd, cwd=str(AMB3R), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1)
    tail = []
    for line in p.stdout:
        if check_cancel is not None and check_cancel():
            p.terminate()
            p.wait()
            from intake.quality import IntakeCancelled
            raise IntakeCancelled("cancelled during AMB3R-SLAM's whole-video pass")
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-40:]
        if any(t in line for t in ("===", "-->", "loops", "map:", "config:", "Error", "Traceback", "[amb3r-launch]",
                                   "input resolution", "no ground truth")):
            log(f"[amb3r] {line}")
    if p.wait() != 0:
        raise Amb3rProbeError("AMB3R-SLAM failed:\n" + "\n".join(tail[-20:]))
    seq = Path(os.path.normpath(str(frames_dir))).name
    res_p = run_dir / "results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    rec = res.get(seq) or (next(iter(res.values())) if len(res) == 1 else None)
    if rec is None or "error" in rec:
        raise Amb3rProbeError(f"AMB3R-SLAM recorded the sequence as failed: {(rec or {}).get('error')}\n"
                              + "\n".join(tail[-20:]))
    traj = sorted(run_dir.glob("traj_*.npz"))
    light = sorted(run_dir.glob("map_*_light.npz"))
    if len(traj) != 1 or len(light) != 1:
        raise Amb3rProbeError(f"{run_dir}: {len(traj)} trajectories and {len(light)} light maps (one each expected)")
    with np.load(traj[0]) as t:
        pred = np.asarray(t["pred"], np.float64)
        kf = np.asarray(t["kf"], np.int64)
    if len(pred) != len(frames) or not np.isfinite(pred).all():
        raise Amb3rProbeError(f"AMB3R-SLAM gave {len(pred)} cameras for the video's {len(frames)} frames "
                              f"(finite: {bool(np.isfinite(pred).all())})")
    with np.load(light[0]) as L:
        r_idx = np.asarray(L["frame"], np.int64)
        r_c2w = np.asarray(L["c2w"], np.float64)
        r_K = np.asarray(L["K"], np.float64)
        r_depth = np.asarray(L["depth"], np.float32)
        k_res = np.asarray(L["k_residual_px"], np.float64)
        stride = int(L["stride"])
    order = np.argsort(r_idx, kind="stable")
    r_idx, r_c2w, r_K, r_depth, k_res = r_idx[order], r_c2w[order], r_K[order], r_depth[order], k_res[order]
    good = np.isfinite(r_c2w).all((1, 2)) & np.isfinite(r_K).all((1, 2)) & ((r_depth > 0).sum((1, 2)) > 0)
    if len(r_idx) == 0 or not good.any():
        raise Amb3rProbeError("AMB3R-SLAM's light map holds no frame with a depth")
    if len(np.unique(r_idx)) != len(r_idx) or r_idx.min() < 0 or r_idx.max() >= len(frames):
        raise Amb3rProbeError("AMB3R-SLAM's light map: frame indices repeat or fall outside the video")
    r_idx, r_c2w, r_K, r_depth, k_res = r_idx[good], r_c2w[good], r_K[good], r_depth[good], k_res[good]
    meta = {"identity": identity(model, grid_wh), "video": video_identity(frames_dir), "video_frames": len(frames),
            "reconstructed_frames": int(len(r_idx)), "dropped_records": int((~good).sum()),
            "keyframes_amb3r": int(rec.get("keyframes", len(kf))), "loops": int(rec.get("loops", 0)),
            "loop_reject_reasons": rec.get("loop_reject_reasons", {}),
            "time_s": float(rec.get("time_s", 0.0)), "peak_mem_gb": float(rec.get("peak_mem_gb", 0.0)),
            "k_residual_px_median": float(np.median(k_res)), "model_grid_wh": [W, H], "stride": stride}
    tmp = out_npz.with_name(out_npz.name + ".part.npz")
    np.savez(tmp, names=np.array(frames), frame_no=np.array([int(Path(f).stem) for f in frames], np.int64),
             pred=pred, kf=kf, rec_index=r_idx, rec_c2w=r_c2w, rec_K=r_K, rec_depth=r_depth,
             grid_wh=np.array([W, H], np.int64), stride=np.int64(stride), source=np.array(SOURCE),
             meta=np.array(json.dumps(meta)))
    os.replace(tmp, out_npz)
    for q in light:
        q.unlink()                                     # in the npz now
    log(f"[amb3r] → {out_npz.name}: {len(frames)} video frames, every one with a pose; {len(r_idx)} reconstructed by "
        f"its backend with what each saw (depth every {stride} px of {W}x{H}, its K fit to {meta['k_residual_px_median']:.2f}"
        f" px); {meta['keyframes_amb3r']} AMB3R keyframes, {meta['loops']} loop closure(s); {meta['time_s']:.0f} s, peak "
        f"{meta['peak_mem_gb']:.1f} GB")
    return out_npz


def session_amb3r(scan_dir: Path, frames_dir: Path, *, model: str, grid_wh: Tuple[int, int],
                  log: Callable[[str], None], check_cancel: Optional[Callable[[], bool]] = None
                  ) -> Tuple[Path, Dict]:
    """intake/amb3r_video.npz of the session — run, or reused while its stamp (the video, what ran it, how) is the
    same. Returns (path, stamp)."""
    intake = Path(scan_dir) / "intake"
    intake.mkdir(parents=True, exist_ok=True)
    npz, stamp_p = intake / VIDEO_NAME, intake / "amb3r_video.stamp.json"
    stamp = {"video": video_identity(frames_dir), **identity(model, grid_wh),
             "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if npz.exists() and stamp_p.exists():
        try:
            if json.loads(stamp_p.read_text()) == stamp:
                log(f"[amb3r] the whole-video pass reused ({npz.name}: the same video, AMB3R, model and grid)")
                return npz, stamp
        except ValueError:
            pass
    stamp_p.unlink(missing_ok=True)
    run_amb3r_video(Path(frames_dir), npz, model=model, grid_wh=grid_wh, log=log, check_cancel=check_cancel)
    tmp = stamp_p.with_name(stamp_p.name + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=1))
    os.replace(tmp, stamp_p)
    return npz, stamp


def load_video(npz: Path) -> Dict[str, np.ndarray]:
    """The whole-video pass as arrays (module docstring)."""
    with np.load(npz) as z:
        out = {k: z[k] for k in z.files}
    if str(out.get("source")) != SOURCE:
        raise Amb3rProbeError(f"{npz} is not AMB3R's whole-video pass (source {out.get('source')!r})")
    return out


def video_index(video: Dict[str, np.ndarray], frame_numbers: Sequence[int]) -> np.ndarray:
    """The position in the video of every frame number (a frame AMB3R did not read FAILS)."""
    pos = {int(n): i for i, n in enumerate(video["frame_no"])}
    miss = [int(f) for f in frame_numbers if int(f) not in pos]
    if miss:
        raise Amb3rProbeError(f"{len(miss)} frame(s) are not frames of the video AMB3R read (first {miss[0]})")
    return np.array([pos[int(f)] for f in frame_numbers], np.int64)


def source_records(video: Dict[str, np.ndarray], frame_numbers: Sequence[int]) -> np.ndarray:
    """For every frame number, the reconstructed frame whose record says what it saw: itself when AMB3R reconstructed
    it, else the nearest reconstructed frame along the video (ties → the earlier). Indices into the records."""
    vi = video_index(video, frame_numbers)
    rec = np.asarray(video["rec_index"], np.int64)
    j = np.searchsorted(rec, vi)
    lo = np.clip(j - 1, 0, len(rec) - 1)
    hi = np.clip(j, 0, len(rec) - 1)
    return np.where(np.abs(rec[hi] - vi) < np.abs(vi - rec[lo]), hi, lo).astype(np.int64)
