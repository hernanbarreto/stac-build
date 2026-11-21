"""THE ARCHITECTURE OF 2026-11-21 (CHECKLIST_arquitectura.md) on a test session — the pipeline's own reconstruction
worker (workers.map_worker._map_work) run in this process, with the substitutions its nine steps ask for. No file of
the pipeline is edited: the substitutions are made on the loaded modules of this process.

  1. AMB3R over every frame: the session's whole-video pass (intake/amb3r_video.npz — its poses and depth).
  2. The selection (arq_select.py, on the card): the keyframes at 92 – 96 % of overlap with the last chosen one;
     frames/selected_frames.json, and frames/witness_frames.json = every frame (F4's witnesses are the frames between
     keyframes). The revisits come from what each keyframe sees of every other on AMB3R's poses and depth, measured
     once with the cube plan (intake/cubes.json) and read from there by the tie, F4 and F6's vote.
  3. The cubes of Omega: the pipeline's planner, 64 keyframes sharing 32 (reconstruction/cubes.py).
  4. DA3 per cube at 504: one DA3 window per cube, the cube's keyframes (the pipeline's extractor,
     extract_da3_depth.py --windows_json); output/da3_cubes/window_<cube>.npz.
  5. Omega per cube: the fork's cube stage, one pass per cube (arq_omega.py).
  6. The tie: seams, Omega bridges on the revisits, the Sim(3) graph (the fork's); each cube sized with the DA3 depth
     of its own window, so that the range is 15 m.
  7. The scale: the pipeline's scale step (reconstruction.scale_align) — one number, DA3 against the tied Omega
     depth; DA3's depth of a keyframe is the one of the window of the cube that owns it.
  8. F0, F4, F5 (precision.runner's own steps); F5's anchor tolerance is the median over the tie's seams of
     (residual / the distance it was measured at).
  9. F6: the factor per keyframe, the vote between keyframes, the cleaner, the publication.

    python arq_front.py <scan dir>"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

TAG = "[arq]"
OMEGA_DRIVER = str(HERE / "arq_omega.py")
SELECT = str(HERE / "arq_select.py")
KEYFRAMES_NAME = "keyframes_arq.json"
DA3_CUBES_DIRNAME = "da3_cubes"
DA3_RES = 504
STEPS = ("f0_camera", "f4_tracks", "f5_refine", "f6_bend")
# lines of the worker that describe steps this architecture does not run (their calls are replaced below)
NOT_RUN = ("frame quality comes from intake I0", "DA3-dense set:", "extracting DA3 metric depth (per-frame)",
           "DA3 metric anchor on ALL", "DA3 anchor: all per-frame depths", "the probe's metre: I3's walk",
           "against the I3 walk", "[scale-verify]", "CHUNKED-METRIC:", "[loops] SALAD candidates over keyframes")
CTX = {}


class Pipe:
    """The worker's pipe, to this process's log: no manager is on the other end."""

    def send_log(self, msg, level="info"):
        s = str(msg)
        if any(k in s for k in NOT_RUN):
            return
        if "DA3 per-cube scales recorded" in s:
            s = s.split("; the ")[0]
        print(s if level == "info" else f"[{level}] {s}", flush=True)

    def send_progress(self, pct, msg, stage=""):
        if any(k in str(msg) for k in NOT_RUN):
            return
        print(f"[progress {float(pct):5.1f} %] {msg}", flush=True)

    def check_cancel(self):
        return False

    def send_done(self, *a, **k):
        pass

    def send_error(self, msg, tb=None):
        print(f"[error] {msg}\n{tb or ''}", flush=True)

    def close(self):
        pass


def job_config() -> dict:
    cfg = yaml.safe_load((Path(SERVER) / "config.yaml").read_text())
    r = cfg["reconstruction"]
    r["precision"]["mono_detail"]["enabled"] = False
    r["vggtomega"]["turn_frames"]["enabled"] = False
    r["vggtomega"]["depth_graph"] = False
    return cfg


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


# ── steps 1 and 2 ─────────────────────────────────────────────────────────────────────────────────────────────────

def intake(pipe, session_path, frames_dir, config, replace):
    """map_worker._run_intake_selection for this architecture: AMB3R's whole-video pass, the selection on it, the
    keyframe list and the witness list."""
    from reconstruction import amb3r_probe as A3
    scan = Path(session_path)
    npz = scan / "intake" / A3.VIDEO_NAME
    if not npz.exists():
        raise RuntimeError(f"{npz} is missing — AMB3R's pass over every frame of this video")
    video = A3.load_video(npz)
    names = [str(n) for n in video["names"]]
    if names != A3.video_frames(frames_dir):
        raise RuntimeError(f"{npz} is not of the frames in {frames_dir}")
    numbers = [int(f) for f in video["frame_no"]]
    name_of = dict(zip(numbers, names))
    pipe.send_log(f"{TAG} step 1 — AMB3R over every frame: {len(numbers)} frames with a pose, "
                  f"{len(video['rec_index'])} with depth ({npz.name})")
    out = scan / "intake" / KEYFRAMES_NAME
    if not out.exists():
        t0 = time.time()
        r = subprocess.run([sys.executable, SELECT, str(scan), str(npz), str(out)], cwd=SERVER, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in r.stdout.splitlines():
            if line.strip() and not line.startswith("[Config]"):
                pipe.send_log(line)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError("the selection failed")
        pipe.send_log(f"{TAG} step 2 — the selection ran in {time.time() - t0:.0f} s")
    kfs = [int(f) for f in json.loads(out.read_text())["frames"]]
    if kfs != sorted(set(kfs)) or kfs[0] != numbers[0] or any(k not in name_of for k in kfs):
        raise RuntimeError("the selection is not an ordered list of the video's frames starting at the first one")
    kfset = set(kfs)
    stamps = {"provenance": "tool_measured", "geometry_epoch": 0, "camera_epoch": 0}
    _write_json(Path(frames_dir) / "selected_frames.json", {
        "version": "2.0", "method": "amb3r_overlap_92_96", "total_frames": len(numbers),
        "selected_count": len(kfs), "selected_files": [name_of[k] for k in kfs], **stamps,
        "keyframes": [{"frame": k, "file": name_of[k], "anchor": (kfs[i - 1] if i else None),
                       "closed_by": ("overlap" if i else "first_frame")} for i, k in enumerate(kfs)],
        "windows": [], "n_witness": len(numbers) - len(kfs)})
    _write_json(Path(frames_dir) / "witness_frames.json", {
        "version": 1, "method": "every_frame", **stamps, "total_frames": len(numbers),
        "selected_count": len(numbers), "selected_files": names,
        "frames": [{"frame": f, "file": name_of[f], "reason": ("keyframe" if f in kfset else "between_keyframes")}
                   for f in numbers]})
    # the walk document the worker reads (the cube planner's plumbing): the keyframes' path on AMB3R's poses
    c2w = np.asarray(video["pred"], np.float64)[A3.video_index(video, kfs)]
    C = c2w[:, :3, 3]
    ch = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    _write_json(scan / "intake" / "walk.json", {
        "version": 3, **stamps, "method": "the keyframes' path on AMB3R's poses", "params": {},
        "n_keyframes": len(kfs), "n_windows": 0, "walk_length_m": float(ch[-1]),
        "chainage": [{"frame": int(f), "chainage_m": float(c), "c2w": T[:3].tolist()} for f, c, T in zip(kfs, ch, c2w)],
        "windows": [], "seams": [], "n_anchors_written": 0})
    gaps = np.diff(kfs)
    pipe.send_log(f"{TAG} step 2 — {len(kfs)} keyframes of {len(numbers)} frames (frames between keyframes: median "
                  f"{np.median(gaps):g}, max {int(gaps.max())}); F4's witnesses: the {len(numbers) - len(kfs)} "
                  f"frames between keyframes")


def no_turn_frames(scan_dir, cfg, log=print):
    """intake.turn_frames.session_turn_frames: Omega sees the keyframes of a cube and no other frame."""
    return {"frames": [], "files": [], "witness_files": [], "windows": [], "n_turn_frames": 0, "n_witness_frames": 0}


def walk_is_current(*a, **k):
    return True


def no_per_frame_da3(*a, **k):
    return None


# ── step 4 ────────────────────────────────────────────────────────────────────────────────────────────────────────

def da3_per_cube(pipe, scan: Path, config_path: Path, selected_path: Path) -> Path:
    """One DA3 window per cube (the cube's keyframes) at 504, by the pipeline's extractor; then every keyframe's DA3
    depth from the window of the cube that owns it, where the scale step reads it (output/da3_run/results_output)."""
    import da3_weights
    import repro
    import intake.walk as W
    from intake.vram import IDENTITY_EXIT, OOM_EXIT
    cfg_v = yaml.safe_load(Path(config_path).read_text())
    M = cfg_v["Model"]
    sel = list(json.loads(Path(selected_path).read_text())["selected_files"])
    if M.get("chunk_sets") is not None:
        sets = [[int(g) for g in F] for F in M["chunk_sets"]]
        owner = [int(x) for x in M["chunk_owner"]]
    else:                                                    # the session is one cube
        sets, owner = [list(range(len(sel)))], [0] * len(sel)
    job = CTX["config"]
    model_id = str(((job.get("reconstruction") or {}).get("da3") or {}).get(
        "model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"))
    out = scan / "output"
    wdir = out / DA3_CUBES_DIRNAME
    wdir.mkdir(parents=True, exist_ok=True)
    spec = {"windows": [[f"frames/{sel[g]}" for g in F] for F in sets], "process_res": DA3_RES, "model_id": model_id,
            "da3_environment": W.da3_identity(sys.executable, model_id, pipe.send_log)}
    spec_path = wdir / "windows.json"
    old = json.loads(spec_path.read_text()) if spec_path.exists() else None
    if old != spec:
        for p in wdir.glob("window_*.npz"):
            p.unlink()
        (wdir / W.REFERENCE_VIEWS_NAME).unlink(missing_ok=True)
        spec_path.write_text(json.dumps(spec))
    cmd = [sys.executable, str(Path(SERVER) / "extract_da3_depth.py"), "--image_dir", str(scan / "frames"),
           "--output_dir", str(wdir), "--model", model_id, "--process_res", str(DA3_RES),
           "--windows_json", str(spec_path), "--windows_root", str(scan)]
    pipe.send_log(f"{TAG} step 4 — DA3 per cube at {DA3_RES}: {len(sets)} window(s) of "
                  f"{min(len(F) for F in sets)}–{max(len(F) for F in sets)} keyframes ({model_id})")
    repro.require_exclusive_gpu(log=pipe.send_log)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=da3_weights.hf_env(repro.deterministic_env()))
    for line in proc.stdout:
        line = line.strip()
        if line:
            pipe.send_log(line)
    proc.wait()
    if proc.returncode == OOM_EXIT:
        raise RuntimeError(f"a DA3 window of a cube's keyframes at {DA3_RES} did not fit the card")
    if proc.returncode == IDENTITY_EXIT:
        raise RuntimeError("the extracting interpreter is not the DA3 environment the windows were planned for")
    if proc.returncode != 0:
        raise RuntimeError(f"the DA3 windows exited with code {proc.returncode}")
    anchors = out / "da3_run" / "results_output"
    anchors.mkdir(parents=True, exist_ok=True)
    local = [{g: i for i, g in enumerate(F)} for F in sets]
    num = [int(Path(f).stem) for f in sel]
    for k in range(len(sets)):
        items = [(num[g], local[k][g]) for g in range(len(sel)) if owner[g] == k]
        if not items:
            continue
        with np.load(wdir / f"window_{k:04d}.npz") as z:
            if [int(x) for x in z["frames"]] != [num[g] for g in sets[k]]:
                raise RuntimeError(f"window_{k:04d}.npz does not hold cube {k}'s keyframes")
            W._write_window_anchors(anchors, items, z["depth"], z["conf"], z["intrinsics"])
    pipe.send_log(f"{TAG} step 4 — {len(sets)} DA3 window(s) in {time.time() - t0:.0f} s → {wdir.name}/; every "
                  f"keyframe's DA3 depth from its owner cube's window → da3_run/results_output/")
    return wdir


class Launch:
    """The worker's `subprocess`: its launch of Omega (bash run_mapanything.sh …) is preceded by DA3 per cube on the
    plan it carries, and started on arq_omega.py — the same activation, directory, interpreter and arguments as
    server/run_mapanything.sh."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Popen(self, cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and len(cmd) >= 2 and cmd[0] == "bash" \
                and str(cmd[1]).endswith("run_mapanything.sh"):
            args = [str(x) for x in cmd[2:]]
            opt = dict(zip(args[::2], args[1::2]))
            wdir = da3_per_cube(CTX["pipe"], CTX["scan"], Path(opt["--config"]), Path(opt["--selected_frames"]))
            os.environ["STAC_ARQ_DA3_CUBES"] = str(wdir)
            if isinstance(k.get("env"), dict):
                k["env"]["STAC_ARQ_DA3_CUBES"] = str(wdir)
            py = os.environ.get("STAC_PYTHON_MAPANYTHING", "/workspace/miniforge3/envs/mapanything/bin/python")
            shell = ('source "/workspace/miniforge3/etc/profile.d/conda.sh" && conda activate mapanything && '
                     'export PYTHONUNBUFFERED=1 && cd "/workspace/stac-build/vendor/VGGT-Long" && '
                     f'exec "{py}" -u "$0" "$@"')
            cmd = ["bash", "-c", shell, OMEGA_DRIVER, *args]
        return self._real.Popen(cmd, *a, **k)


# ── steps 8 and 9 ─────────────────────────────────────────────────────────────────────────────────────────────────

def tolerance_from_seams(pipe, output_dir: Path) -> float:
    """output/gauge.json as F5 reads it (applied_instrument, sigma_by_instrument), from the tie's seams."""
    cg = json.loads((output_dir / "maplong_run" / "chunk_graph.json").read_text())
    rows = [{"seam": [int(e["b"]), int(e["a"])], "residual_m": float(e["sigma_m"]), "lever_m": float(e["lever_m"]),
             "relative": float(e["sigma_m"]) / float(e["lever_m"])} for e in cg["seams"]]
    if not rows:
        raise RuntimeError("the session is ONE cube: it has no seam, so the tie measured no tolerance for F5's "
                           "anchors — this architecture has no rule for that case")
    sigma = float(np.median([r["relative"] for r in rows]))
    _write_json(output_dir / "gauge.json", {
        "version": "arq", "provenance": "tool_measured",
        "note": "F5's anchor tolerance: the median over the tie's seams of (residual / the distance it was measured at)",
        "applied_instrument": "tie_seams", "sigma_by_instrument": {"tie_seams": sigma}, "seams": rows})
    rel = np.asarray([r["relative"] for r in rows]) * 100.0
    pipe.send_log(f"{TAG} step 8 — F5's anchor tolerance from the tie's {len(rows)} seam(s): median "
                  f"{sigma * 100:.2f} % (min {rel.min():.2f} %, max {rel.max():.2f} %) → output/gauge.json")
    return sigma


def chain(pipe, session_path, output_dir, config):
    """map_worker._run_precision_core for this architecture: F0, F4, F5, F6, then the worker's own end — the published
    cloud is the session's, the pose-only epochs and Omega's chunk files go."""
    from precision.config import frozen_config, load_precision_config
    from precision.product import product_is_live
    from precision.runner import chain_steps, run_chain
    import workers.map_worker as MW
    tolerance_from_seams(pipe, Path(output_dir))
    cfg, _sha = frozen_config(Path(session_path), config, log=pipe.send_log)
    pcfg = load_precision_config(cfg)
    steps = [s for s in chain_steps(pcfg) if s.key in STEPS]
    if [s.key for s in steps] != list(STEPS):
        raise RuntimeError(f"the chain's steps are not {STEPS}: {[s.key for s in steps]}")
    rep = run_chain(Path(session_path), pcfg, log=lambda m: pipe.send_log(str(m)),
                    progress=lambda pct, msg: pipe.send_progress(float(pct), str(msg), stage="precision"),
                    cancelled=pipe.check_cancel,
                    before_gpu=lambda label: MW.stop_semantic_service_verified(pipe, stage=f"precision {label}"),
                    steps=steps)
    live, why = product_is_live(Path(output_dir))
    if not live:
        raise RuntimeError(f"the chain ended without a published cloud — {why}")
    freed = MW._discard_previous_epochs(Path(output_dir))
    pipe.send_log(f"{TAG} steps 8 and 9 — F0, F4, F5, F6 in {rep['seconds'] / 60:.1f} min; the session is at epoch "
                  f"{rep['epoch']}; {why}; {freed / 1048576:.0f} MB of pose-only epochs and Omega chunk files removed")


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    import intake.turn_frames as TF
    import intake.walk as W
    import workers.map_worker as MW
    from card import card
    from intake import run_config as RC
    pipe = Pipe()
    config = job_config()
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"{TAG} the job's configuration frozen: {frozen['path']} (sha256 {frozen['sha256'][:12]})")
    job = {**config, "_pipeline_replace": False}
    CTX.update(pipe=pipe, scan=scan, frames_dir=(scan / "frames").resolve(), config=job)
    MW._run_intake_selection = intake
    TF.session_turn_frames = no_turn_frames
    W.walk_is_current = walk_is_current
    MW._run_da3_anchor = no_per_frame_da3
    MW._run_precision_core = chain
    MW.subprocess = Launch(subprocess)
    t0 = time.time()
    with card("the architecture of 2026-11-21", session=scan.parents[2].name, log=pipe.send_log):
        MW._map_work(pipe, str(scan), job)
    pipe.send_log(f"{TAG} the cloud is published; the whole run took {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
