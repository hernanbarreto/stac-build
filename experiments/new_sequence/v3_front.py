"""VERSION 3 — the pipeline's own reconstruction worker (workers.map_worker._map_work) run in this process on the test
session, as version 2 ran it (AMB3R first, the new keyframe selection, no 8-frame rule, no coverage fill), and
WITHOUT what the user took out on 2026-11-21:

  * NO I3. The worker's DA3 windows and their walk are not run (intake.walk.walk_is_current answers yes); the walk
    the cube plan reads is AMB3R's own, in metres since step 1 (intake/walk.json written from AMB3R's poses of the
    chosen keyframes). DA3 then runs ONCE on the keyframes, frame by frame (the worker's own anchor pass), for the
    session's metre and nothing else.
  * NO DA3 LOCK PER CUBE. The worker's launch of Omega is redirected to v3_omega.py: the fork's cube stage with
    CubeRun.metric_lock replaced — each cube at AMB3R's metre, so that the 15 m range means 15 m.
  * NO F2. The chain runs F0, F4, F5, F6. F5's one input from F2 — the relative tolerance of its position anchors —
    comes from the tie: the median over the seams of (the seam's residual / the distance it was measured at), written
    where F5 reads it (output/gauge.json, instrument "tie_seams"). F5's formula is unchanged.
  * NO SECOND TIE, and nothing after F6: the chain's report-only check is not run.
  * PointDiT off in this run (cloud A); v3_pointdit.py runs F6 again with it on (cloud B).

No file of the pipeline is edited: the substitutions are made on the loaded modules of this process.

    python v3_front.py <scan dir>"""
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
import v2_front as V2  # noqa: E402

OMEGA_DRIVER = str(HERE / "v3_omega.py")
STEPS = ("f0_camera", "f4_tracks", "f5_refine", "f6_bend")


def job_config(mono: bool = False) -> dict:
    cfg = yaml.safe_load((Path(SERVER) / "config.yaml").read_text())
    q = cfg["intake"]["quality"]
    q["luma_lo"], q["luma_hi"], q["clip_frac_max"] = 0.0, 255.0, 1.0
    cfg["reconstruction"]["precision"]["mono_detail"]["enabled"] = bool(mono)
    return cfg


def write_amb3r_walk(pipe, scan: Path) -> None:
    """intake/walk.json from AMB3R's poses of the chosen keyframes (no I3), and the keyframes' centres for the cubes."""
    from intake.quality import _write_json_atomic
    from reconstruction import amb3r_probe as A3
    sel = json.loads((scan / "frames" / "selected_frames.json").read_text())
    kfs = sorted(int(Path(f).stem) for f in sel["selected_files"])
    video = A3.load_video(scan / "intake" / A3.VIDEO_NAME)
    if "metric_scale" not in video:
        raise RuntimeError("AMB3R's whole-video pass is not in metres")
    c2w = np.asarray(video["pred"], np.float64)[A3.video_index(video, kfs)]
    C = c2w[:, :3, 3]
    ch = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    doc = {"version": 3, "provenance": "tool_measured", "geometry_epoch": 0, "camera_epoch": 0,
           "method": "amb3r_walk — version 3: no I3; the walk of the chosen keyframes in AMB3R's whole-video pass "
                     "(metres)", "params": {}, "n_keyframes": len(kfs), "n_windows": 0,
           "walk_length_m": float(ch[-1]),
           "chainage": [{"frame": int(f), "chainage_m": float(c), "c2w": T[:3].tolist()}
                        for f, c, T in zip(kfs, ch, c2w)],
           "windows": [], "seams": [], "n_anchors_written": 0}
    _write_json_atomic(scan / "intake" / "walk.json", doc)
    cen = scan / "intake" / "amb3r_keyframes_v3.json"
    _write_json_atomic(cen, {str(int(f)): c.tolist() for f, c in zip(kfs, C)})
    os.environ["STAC_V3_AMB3R_CENTRES"] = str(cen)
    pipe.send_log(f"[v3] the walk from AMB3R's poses: {ch[-1]:.2f} m over {len(kfs)} keyframes → intake/walk.json; "
                  f"the keyframes' centres for the cubes → {cen.name}")


def amb3r_then_walk(pipe, session_path, frames_dir, config):
    """map_worker._amb3r_coverage_fill for version 3: AMB3R (run or reused on its stamp), no fill, then AMB3R's walk."""
    V2.amb3r_only(pipe, session_path, frames_dir, config)
    write_amb3r_walk(pipe, Path(session_path))


def walk_is_current(*a, **k):
    return True


class OmegaRedirect:
    """The worker's `subprocess`, with its launch of Omega (bash run_mapanything.sh …) started on v3_omega.py instead of
    vggt_long.py — the same activation, directory, interpreter and arguments as server/run_mapanything.sh."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Popen(self, cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and len(cmd) >= 2 and cmd[0] == "bash" \
                and str(cmd[1]).endswith("run_mapanything.sh"):
            py = os.environ.get("STAC_PYTHON_MAPANYTHING", "/workspace/miniforge3/envs/mapanything/bin/python")
            shell = ('source "/workspace/miniforge3/etc/profile.d/conda.sh" && conda activate mapanything && '
                     'export PYTHONUNBUFFERED=1 && cd "/workspace/stac-build/vendor/VGGT-Long" && '
                     f'exec "{py}" -u "$0" "$@"')
            cmd = ["bash", "-c", shell, OMEGA_DRIVER, *cmd[2:]]
        return self._real.Popen(cmd, *a, **k)


def write_tolerance_from_seams(pipe, output_dir: Path) -> float:
    """output/gauge.json as F5 reads it (applied_instrument, sigma_by_instrument), from the tie's seams."""
    from intake.quality import _write_json_atomic
    cg = json.loads((output_dir / "maplong_run" / "chunk_graph.json").read_text())
    rows = [{"seam": [int(e["b"]), int(e["a"])], "residual_m": float(e["sigma_m"]), "lever_m": float(e["lever_m"]),
             "relative": float(e["sigma_m"]) / float(e["lever_m"])} for e in cg["seams"]]
    if not rows:
        raise RuntimeError("the session is ONE cube: it has no seam, so the tie measured no tolerance for F5's "
                           "anchors — version 3 has no rule for that case")
    sigma = float(np.median([r["relative"] for r in rows]))
    _write_json_atomic(output_dir / "gauge.json", {
        "version": "v3", "provenance": "tool_measured",
        "note": "version 3: F2 did not run. F5's anchor tolerance is the tie's own error: the median over the seams "
                "of (residual / the distance it was measured at).",
        "applied_instrument": "tie_seams", "sigma_by_instrument": {"tie_seams": sigma}, "seams": rows})
    pipe.send_log(f"[v3] F5's anchor tolerance from the tie's {len(rows)} seam(s): "
                  + ", ".join(f"{r['seam'][0]}<->{r['seam'][1]} {r['residual_m'] * 100:.1f} cm at {r['lever_m']:.2f} m "
                              f"= {r['relative'] * 100:.1f} %" for r in rows)
                  + f" → median {sigma * 100:.2f} % → output/gauge.json")
    return sigma


def run_steps(pipe, session_path, config, from_step=None):
    from precision.config import frozen_config, load_precision_config
    from precision.runner import chain_steps, run_chain
    import workers.map_worker as MW
    config, _sha = frozen_config(Path(session_path), config, log=pipe.send_log)
    pcfg = load_precision_config(config)
    steps = [s for s in chain_steps(pcfg) if s.key in STEPS]
    if [s.key for s in steps] != list(STEPS):
        raise RuntimeError(f"the chain's steps are not {STEPS}: {[s.key for s in steps]}")
    return run_chain(Path(session_path), pcfg, log=lambda m: pipe.send_log(str(m)),
                     progress=lambda pct, msg: pipe.send_progress(float(pct), str(msg), stage="precision"),
                     cancelled=pipe.check_cancel,
                     before_gpu=lambda label: MW.stop_semantic_service_verified(pipe, stage=f"precision {label}"),
                     steps=steps, from_step=from_step)


def chain_v3(pipe, session_path, output_dir, config):
    """map_worker._run_precision_core for version 3: F0, F4, F5, F6 — no F2, no report-only check, no discard."""
    write_tolerance_from_seams(pipe, Path(output_dir))
    rep = run_steps(pipe, session_path, config)
    pipe.send_log(f"[v3] the chain F0, F4, F5, F6 done in {rep['seconds'] / 60:.1f} min, the session is at epoch "
                  f"{rep['epoch']}")


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    import intake.parallax as P
    import intake.walk as W
    import workers.map_worker as MW
    from card import card
    from intake import run_config as RC
    pipe = V2.Pipe()
    config = job_config(mono=False)
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"[v3] the job's configuration frozen: {frozen['path']} (sha256 {frozen['sha256'][:12]}) — "
                  f"config.yaml with intake.quality open (luma 0..255, clip 1.0) and mono_detail off")
    job = {**config, "_pipeline_replace": False}
    V2.CTX.update(pipe=pipe, scan=scan, frames_dir=(scan / "frames").resolve(), config=job)
    P.select_keyframes = V2.imposed_select
    P.stride_fill = V2.no_stride_fill
    W.walk_is_current = walk_is_current
    MW._amb3r_coverage_fill = amb3r_then_walk
    MW._run_precision_core = chain_v3
    MW.subprocess = OmegaRedirect(subprocess)
    t0 = time.time()
    with card("version 3: intake to the cloud (no I3, no F2, no DA3 lock)", session=scan.parents[2].name,
              log=pipe.send_log):
        MW._map_work(pipe, str(scan), job)
    pipe.send_log(f"[v3] cloud A (PointDiT off) published; the whole run took {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
