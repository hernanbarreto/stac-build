"""AMB3R-SLAM → the session the viewer shows WITHOUT corrections (USER 2026-11-03: "si no funciona este slam probamos con el
otro que usaba Ω … instalá AMB3R"; VGGT-SLAM and VidMap were dropped the same day).

  1. AMB3R-SLAM (env amb3r, vendor/AMB3R-SLAM) on the session's frames at --stride (2 = 15 Hz from a 30 fps video, as
     its authors run TUM), the heavy model --model (da3 = DA3NESTED-GIANT-LARGE-1.1, its default; omega = VGGT-Ω),
     torch on 8 threads and 16 cores (on this 252-core box its default threads ran it 7x slower, CPU-bound)
  2. the session (amb3r_session.py): its map as chunk_0.ply, the cameras of the frames the map was saved from
  3. the upright orientation (reconstruction.orient's appliers, the vertical from the cameras' right axes)
  4. the cloud stage's voxel + SOR (no witness filter: the map carries no per-point frame) and the octree; the session
     moved into server/projects/ whole
  5. --corrections (the run made with --stac_dump: one depth record per saved frame): F0 the camera, F4 the tracks, F5
     the refinement, F6 the bend (PointDiT at the edges with --pointdit) on the published session — F5/F6 publish their
     own epochs, the cloud without corrections stays selectable as epoch 0 (USER 2026-11-03: "corré ambos completos,
     con F0 a F6"). MEASURED on station the same night: the corrections made every AMB3R variant WORSE (a standing pan
     gave F5 zero-baseline cameras and F6 bent 19 keyframes to ~0); AMB3R is kept for long walks, uncorrected
     (USER 2026-11-04: "amber lo dejamos … porque puede servir en recorridos largos más adelante")

  (env da3, from server/) python tools/amb3r/chain.py <source src_default> <new project> --model da3|omega
        [--stride 2] [--reuse <AMB3R run dir>] [--align-to <metric run>] [--corrections] [--pointdit]
        [--from-step 5] [--replace]
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER))
HERE = Path(__file__).resolve().parent
PROJECTS = Path("/workspace/stac-build/server/projects")
STAGING = Path("/workspace/tmp/amb3r_staging")
AMB3R = Path("/workspace/stac-build/vendor/AMB3R-SLAM")
CONDA = "/workspace/miniforge3/etc/profile.d/conda.sh"
DA3 = "/workspace/miniforge3/envs/da3/bin/python"
HF_HOME = "/workspace/hf_cache"
N_BOOT = 2000
SAVE_EVERY = 2                        # AMB3R's --save_every (its default): the map keeps every 2 x frame_stride-th frame


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [chain] {msg}", flush=True)


def run(cmd, env=None, cwd=None, tag=""):
    t0 = time.time()
    p = subprocess.run(cmd, env=env, cwd=cwd)
    if p.returncode != 0:
        raise SystemExit(f"{tag} failed (exit {p.returncode})")
    log(f"{tag} done ({time.time() - t0:.0f} s)")


def deterministic_env(**extra):
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _unit_rows(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def vertical(X, Y):
    """The down direction: the direction every camera's right axis leaves out (people keep the phone level sideways;
    the smallest eigenvector of Σ x xᵀ), pointed down by the cameras' own down axes Y."""
    w, V = np.linalg.eigh(X.T @ X)
    g = V[:, 0] / np.linalg.norm(V[:, 0])
    return g if float((Y @ g).mean()) >= 0 else -g


def gravity_estimator(output_dir, log=None):
    """The drop-in for reconstruction.orient.estimate_gravity: (down unit vector, mean |sin| of each camera's right
    axis to it — what faces the module's 0.7 bar)."""
    from reconstruction.orient import _read_poses
    _log = log or print
    poses = _read_poses(Path(output_dir) / "camera_poses.txt")
    if not poses:
        _log("no camera_poses.txt found — cannot estimate gravity from the cameras")
        return None
    X = _unit_rows(np.stack([p[:3, 0] for p in poses]))
    Y = _unit_rows(np.stack([p[:3, 1] for p in poses]))
    g = vertical(X, Y)
    align = float(np.sqrt(np.clip(1.0 - (X @ g) ** 2, 0.0, 1.0)).mean())
    deg = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
    rng = np.random.default_rng(0)
    spread = [deg(vertical(X[i], Y[i]), g) for i in (rng.integers(0, len(X), len(X)) for _ in range(N_BOOT))]
    w = np.linalg.eigvalsh(X.T @ X) / len(X)
    _log(f"vertical (right axes) from {len(poses)} cameras: down={np.round(g, 4).tolist()}; right axes ⟂ it: mean "
         f"|sin| {align:.3f}; the right axes' spread {np.round(w, 4).tolist()}; bootstrap p95 "
         f"{np.percentile(spread, 95):.1f}° ({N_BOOT} draws, seed 0); {deg(g, _unit_rows(Y.mean(0)[None])[0]):.1f}° "
         f"from the cameras' mean down")
    return g, align




def amb3r_run(src, run_dir, model, stride, loops_json=None):
    shutil.rmtree(run_dir, ignore_errors=True)
    extra = ""
    if loops_json:
        # USER 2026-11-08 ("decirle dónde tiene que ajustar"): the exporter's VIDEO FRAME NUMBER pairs mapped onto
        # AMB3R's processed sequence (its runner reads the sorted frames at --stride): index = round(frame/stride),
        # clamped — declared approximation for keyframes that fall between strided frames.
        files = sorted(p for p in (src / "frames").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        n = len(files[::stride])
        doc = json.loads(Path(loops_json).read_text())
        idx = sorted({(min(ia, ib), max(ia, ib))
                      for a, b in doc["pairs"]
                      for ia, ib in [(min(round(int(a) / stride), n - 1), min(round(int(b) / stride), n - 1))]
                      if ia != ib})
        run_dir.mkdir(parents=True, exist_ok=True)
        lp = run_dir / "stac_loops.json"
        lp.write_text(json.dumps({"pairs": [[a, b] for a, b in idx], "from": str(loops_json)}, indent=1))
        extra = f"--stac_loops {lp} "
        log(f"1b. {len(idx)} external loop pair(s) (from {len(doc['pairs'])} frame pairs) → {lp}")
    log(f"1. AMB3R-SLAM on {src / 'frames'} at stride {stride}, model {model}")
    run(["bash", "-c", f"source {CONDA} && conda activate amb3r && exec taskset -c 0-15 python -u run.py --dataset demo "
         f"--data_path {src / 'frames'} --stride {stride} --model_name {model} --save_map --save_every {SAVE_EVERY} "
         f"--stac_dump {run_dir / 'frames'} "       # the per-frame depth records the precision chain corrects on
         f"{extra}"
         f"--out {run_dir}"], cwd=str(AMB3R),
        env=deterministic_env(OMP_NUM_THREADS=8, MKL_NUM_THREADS=8, OPENBLAS_NUM_THREADS=8, HF_HUB_OFFLINE=1,
                              HF_HOME=HF_HOME, HF_HUB_ENABLE_HF_TRANSFER=0), tag="1. AMB3R-SLAM")


PRECISION_STEPS = ("f0_camera", "f4_tracks", "f5_refine", "f6_bend")


def corrections(ses, pointdit):
    """F0, F4, F5, F6 on the session in place, this session's frozen configuration (the validated recipe)."""
    from config import cfg as raw_cfg
    from intake.run_config import freeze_run_config
    from precision.config import load_precision_config
    from precision.runner import STEPS, run_chain
    from workers.base import stop_semantic_service_verified
    c = copy.deepcopy(raw_cfg)
    c["reconstruction"]["precision"]["mono_detail"]["enabled"] = bool(pointdit)
    c["reconstruction"]["simple"]["conf_min_norm"] = 0.0
    freeze_run_config(ses, c, log=lambda m: log(f"   {m}"))
    steps = [s for s in STEPS if s.key in PRECISION_STEPS]
    log(f"5. corrections: {[s.key for s in steps]} (PointDiT {'at the edges' if pointdit else 'off'})")
    run_chain(ses, load_precision_config(session_dir=ses), log=lambda m: log(f"   {m}"), steps=steps,
              before_gpu=lambda label: stop_semantic_service_verified(None, stage=f"precision {label}", log=print))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("name")
    ap.add_argument("--model", choices=("da3", "omega"), required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--reuse", default=None, help="an AMB3R run directory already made with the same model and stride")
    ap.add_argument("--loops", default=None, help="export_loops.py json: OUR revisit pairs injected as loop "
                                                  "candidates, verified by AMB3R's own gates (USER 2026-11-08)")
    ap.add_argument("--align-to", default=None, help="another AMB3R run (metric) to bring this one's frame and size to")
    ap.add_argument("--corrections", action="store_true")
    ap.add_argument("--pointdit", action="store_true")
    ap.add_argument("--from-step", type=int, default=1, choices=(1, 5))
    ap.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    src = Path(a.source).resolve()
    final = PROJECTS / a.name
    fses = final / "scans" / src.parent.name / "src_default"
    if a.from_step == 5:
        corrections(fses, a.pointdit)
        log(f"→ {final} corrected")
        return
    if final.exists() and not a.replace:
        raise SystemExit(f"{final} exists (--replace to overwrite it)")
    t0 = time.time()
    run_dir = Path(a.reuse) if a.reuse else STAGING / f"{a.name}.run"
    if not a.reuse:
        amb3r_run(src, run_dir, a.model, a.stride, loops_json=a.loops)
    cfg = json.loads((run_dir / "config.json").read_text())
    every = SAVE_EVERY * int(cfg["resolved"]["backend"]["frame_stride"])
    shutil.rmtree(STAGING / a.name, ignore_errors=True)
    label = {"da3": "AMB3R-SLAM (DA3 Giant)", "omega": "AMB3R-SLAM (Ω)"}[a.model]
    run([DA3, str(HERE / "amb3r_session.py"), str(run_dir), str(src / "frames"), str(src), a.name, "--stride",
         str(a.stride), "--every", str(every), "--label", label, "--replace"]
        + (["--align-to", a.align_to] if a.align_to else []),
        env=deterministic_env(STAC_PROJECTS_DIR=STAGING), cwd=str(SERVER), tag="2. session")
    ses = STAGING / a.name / "scans" / src.parent.name / "src_default"
    out = ses / "output"

    import reconstruction.orient as OR
    OR.estimate_gravity = gravity_estimator
    if OR.run(out, log=lambda m: log(f"   [orient] {m}"), aligned_chunks=False) is None:
        raise SystemExit("3. orientation refused — see the log above")
    log("3. orientation applied")

    run([DA3, str(HERE / "clean_and_potree.py"), str(ses), "--no-witness"], env=deterministic_env(),
        cwd=str(SERVER), tag="4. cloud without corrections (voxel + SOR, octree)")
    if not (out / "cleaned_cloud.ply").exists() or not (out / "potree").exists():
        raise SystemExit("4. no cloud with its octree")
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(STAGING / a.name), str(final))
    log(f"→ {final} without corrections ({(time.time() - t0) / 60:.1f} min); AMB3R's run kept in {run_dir}")
    if a.corrections:
        corrections(fses, a.pointdit)
        log(f"→ {final} corrected ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
