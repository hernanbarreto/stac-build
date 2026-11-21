"""VERSION 2, steps 1–9 — the pipeline's own reconstruction worker (workers.map_worker._map_work), run in this process
on the test session, with the substitutions the twelve steps ask for and nothing else. No file of the pipeline is
edited: the substitutions are made on the loaded modules of this process.

  1. AMB3R FIRST, THEN THE NEW SELECTION. intake.parallax.select_keyframes is replaced: it runs AMB3R-SLAM over the
     whole video (the pipeline's own call, reconstruction.amb3r_probe.session_amb3r, as map_worker makes it), then
     the new selection on its map (v2_select.py, a process of its own on the card), and returns the keyframes it
     chose with the windows and per-frame readings the turn frames need — measured by the pipeline's own tracker
     (intake.parallax._Window) from each chosen keyframe to the next one.
  2. NO 8-FRAME RULE: intake.parallax.stride_fill returns the keyframes as they are.
  3. NO COVERAGE FILL: map_worker._amb3r_coverage_fill runs its AMB3R half (reused on its stamp) and not the fill.
  4. THE CHAIN STOPS AFTER F5: map_worker._run_precision_core runs the chain's own steps F0, F2, F4, F5
     (precision.runner.run_chain) and not F6 — the depth, the second tie and the vote are steps 10–12.
  5. THE CUBES ARE KEPT by the pipeline's own switch (certify.keep_aligned_chunks: true in the job's configuration):
     the placed cubes and the bridges stay on disk after the cube stage — the second tie of step 11 reads them.

THE JOB'S CONFIGURATION is server/config.yaml as it stands, with: intake.quality.luma_lo 0, luma_hi 255,
clip_frac_max 1 (no frame is rejected by its exposure), reconstruction.precision.mono_detail.enabled false (PointDiT
off) and certify.keep_aligned_chunks true — frozen into the session before the worker starts, as the pipeline manager does.

    python v2_front.py <scan dir>"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
CTX = {}


class Pipe:
    """The worker's pipe, to this process's log: no manager is on the other end."""

    def send_log(self, msg, level="info"):
        print(str(msg) if level == "info" else f"[{level}] {msg}", flush=True)

    def send_progress(self, pct, msg, stage=""):
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
    q = cfg["intake"]["quality"]
    q["luma_lo"], q["luma_hi"], q["clip_frac_max"] = 0.0, 255.0, 1.0
    cfg["reconstruction"]["precision"]["mono_detail"]["enabled"] = False
    # the pipeline's own switch that keeps the placed cubes and the bridges after the cube stage (it deletes them
    # otherwise — CubeRun.apply removes each cube as inferred once it is placed): the second tie of step 11 reads them
    cfg["certify"]["keep_aligned_chunks"] = True
    return cfg


def amb3r_only(pipe, session_path, frames_dir, config):
    """map_worker._amb3r_coverage_fill without the fill: AMB3R-SLAM over the whole video, run or reused on its stamp."""
    import repro
    import workers.map_worker as MW
    from precision.camera import omega_grid_for
    from reconstruction import amb3r_probe as A3
    va = ((config.get("reconstruction") or {}).get("vggtomega") or {})
    st = MW._structure_config(va)
    if not st["enabled"] or st["probe"] != "amb3r":
        raise RuntimeError("the structure probe of this configuration is not AMB3R — version 2 selects on its map")
    vcfg = MW._build_vggtomega_config(config, frames_dir)
    nw, nh = MW._native_frame_wh(frames_dir)
    g = omega_grid_for(int(nw), int(nh), str(vcfg["Model"]["omega_mode"]), int(vcfg["Model"]["omega_resolution"]))
    pipe.send_progress(4, f"AMB3R-SLAM over the whole video ({st['amb3r_model']}, {g.w}x{g.h}) — no coverage fill",
                       stage="reconstruction")
    MW.stop_semantic_service_verified(pipe, stage="AMB3R-SLAM whole-video pass")
    repro.require_exclusive_gpu(log=pipe.send_log)
    npz, _stamp = A3.session_amb3r(session_path, frames_dir, model=st["amb3r_model"], grid_wh=(int(g.w), int(g.h)),
                                   log=pipe.send_log, check_cancel=pipe.check_cancel)
    return npz


def imposed_select(chain, frames, by_frame, cfg, T, Kn, prog, rot):
    """intake.parallax.select_keyframes for version 2 (module docstring, 1): (keyframes, windows, records)."""
    import intake.parallax as P
    pipe, scan, frames_dir, config = CTX["pipe"], CTX["scan"], CTX["frames_dir"], CTX["config"]
    t0 = time.time()
    amb3r_only(pipe, scan, frames_dir, config)
    pipe.send_log(f"[v2] step 1 — AMB3R over the whole video: {time.time() - t0:.0f} s")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "v2_select.py"), str(scan)], cwd=SERVER, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in r.stdout.splitlines():
        if line.strip() and not line.startswith("[Config]"):
            pipe.send_log(line)
    if r.returncode != 0:
        raise RuntimeError("the new selection failed")
    doc = json.loads((scan / "intake" / "keyframes_v2.json").read_text())
    kfs = [int(f) for f in doc["frames"]]
    pipe.send_log(f"[v2] step 2 — the new selection: {len(kfs)} keyframes in {time.time() - t0:.0f} s")
    usable = set(chain)
    if kfs[0] != chain[0] or any(f not in usable for f in kfs) or kfs != sorted(set(kfs)):
        raise RuntimeError("the new selection is not an ordered list of usable frames starting at the first one")
    q = float(cfg.parallax_quantum_px)
    lo, hi = q * (1.0 - cfg.keyframe_band_frac), q * (1.0 + cfg.keyframe_band_frac)
    rq = float(rot["theta_q_deg"])
    lo_r, hi_r = rq * (1.0 - cfg.keyframe_band_frac), rq * (1.0 + cfg.keyframe_band_frac)
    pos = {f: i for i, f in enumerate(chain)}
    keyframes = [{"frame": chain[0], "file": by_frame[chain[0]]["file"], "anchor": None, "parallax_px": 0.0,
                  "rotation_deg": 0.0, "closed_by": "first_usable_frame"}]
    windows, records = [], {}
    ends = kfs[1:] + [None]
    for a, k in zip(kfs, ends):
        win = P._Window(a, frames, cfg, Kn)
        stretch = chain[pos[a] + 1:(pos[k] + 1 if k is not None else len(chain))]
        allm = []
        for f in stretch:
            prog.tick(f)
            m = win.measure(f, T)
            records[f] = m
            allm.append(m)
        seen = [m for m in allm if not m.lost]
        mk = records.get(k) if k is not None else None
        chosen = mk if (mk is not None and not mk.lost) else None
        windows.append({"anchor": a, "anchor_is_keyframe": True, "closed_by": "overlap" if k is not None else "end",
                        "n_frames": len(allm), "keyframe": k,
                        "max_parallax_px": (max(s.parallax_px for s in seen) if seen else None),
                        "margins": P.window_margins(allm, [], None, by_frame, cfg, lo, hi),
                        "rotation": P.rotation_record(seen, [], None, chosen, by_frame, rot, lo_r, hi_r)})
        if k is not None:
            keyframes.append({"frame": k, "file": by_frame[k]["file"], "anchor": a,
                              "parallax_px": (None if mk.lost else mk.parallax_px),
                              "rotation_deg": (None if mk.lost else mk.rot_deg), "closed_by": "overlap"})
    lost = sum(1 for k in kfs[1:] if records[k].lost)
    pipe.send_log(f"[v2] the windows of the {len(kfs)} chosen keyframes measured by I1's own tracker ({len(records)} "
                  f"readings; {lost} keyframe(s) without a reading from the one before). THE NEXT LINE OF I1 COUNTS ITS "
                  f"OWN CLOSERS (parallax / rotation / stride): none applies — every keyframe was closed by 'overlap'.")
    return keyframes, windows, records


def no_stride_fill(chain, keyframes, by_frame, gap):
    return keyframes


def chain_to_f5(pipe, session_path, output_dir, config):
    """map_worker._run_precision_core for version 2: the chain's own steps F0, F2, F4, F5 — as
    workers.precision_worker runs them — and nothing after."""
    from precision.config import frozen_config, load_precision_config
    from precision.runner import chain_steps, run_chain
    import workers.map_worker as MW
    config, _sha = frozen_config(Path(session_path), config, log=pipe.send_log)
    pcfg = load_precision_config(config)
    steps = [s for s in chain_steps(pcfg) if s.key in ("f0_camera", "f2_gauge", "f4_tracks", "f5_refine")]
    if [s.key for s in steps] != ["f0_camera", "f2_gauge", "f4_tracks", "f5_refine"]:
        raise RuntimeError(f"the chain's steps are not F0, F2, F4, F5 in order: {[s.key for s in steps]}")
    rep = run_chain(Path(session_path), pcfg, log=lambda m: pipe.send_log(str(m)),
                    progress=lambda pct, msg: pipe.send_progress(float(pct), str(msg), stage="precision"),
                    cancelled=pipe.check_cancel,
                    before_gpu=lambda label: MW.stop_semantic_service_verified(pipe, stage=f"precision {label}"),
                    steps=steps)
    pipe.send_log(f"[v2] the chain F0 → F5 done in {rep['seconds'] / 60:.1f} min, the session is at epoch {rep['epoch']}")


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    import intake.parallax as P
    import workers.map_worker as MW
    from card import card
    from intake import run_config as RC
    pipe = Pipe()
    config = job_config()
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"[v2] the job's configuration frozen: {frozen['path']} (sha256 {frozen['sha256'][:12]}) — "
                  f"config.yaml with intake.quality open (luma 0..255, clip 1.0), mono_detail off, keep_aligned_chunks on")
    job = {**config, "_pipeline_replace": False}
    CTX.update(pipe=pipe, scan=scan, frames_dir=(scan / "frames").resolve(), config=job)
    P.select_keyframes = imposed_select
    P.stride_fill = no_stride_fill
    MW._amb3r_coverage_fill = amb3r_only
    MW._run_precision_core = chain_to_f5
    t0 = time.time()
    with card("version 2: intake to F5", session=scan.parents[2].name, log=pipe.send_log):
        MW._map_work(pipe, str(scan), job)
    pipe.send_log(f"[v2] steps 1-9 done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
