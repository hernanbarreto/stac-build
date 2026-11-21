"""VERSION 4 — version 3 with ONE more rule of the user (2026-11-21): THE ONLY FRAMES OF THE SESSION ARE THE KEYFRAMES.

"solo los frames que votan los que entran en todo el circuito son los KF, nada mas … hace de cuenta que los unicos
frames que tiene la sesion son los kf, nada mas, ni para f6 ni para relleno de nada" / "son 64 con solape de 32, no hay
vuelta con eso". AMB3R reads the whole video to choose them; from there on no other frame enters any stage:

  * no turn frame in Omega's pass of a cube and no interval witness (the pipeline's own switch,
    reconstruction.vggtomega.turn_frames.enabled: false) — a cube is its keyframes, 64 sharing 32;
  * no witness frame for F4 and F5: the witness list I1 writes is the keyframes themselves
    (intake.parallax.select_witnesses replaced), so F4 builds no witness window and F5 localises none;
  * I1's tracker is not run at all: its readings between keyframes fed the turn frames and I1's warnings, and neither
    exists here (intake.parallax.select_keyframes replaced by the selection alone).

Everything else is version 3 (v3_front.py): AMB3R first, the new selection, the walk from AMB3R, each cube at AMB3R's
metre, DA3 once for the session's metre, the chain F0, F4, F5, F6 with F5's anchor tolerance from the tie's seams,
PointDiT off.

    python v4_front.py <scan dir>"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import v2_front as V2  # noqa: E402
import v3_front as V3  # noqa: E402


def job_config() -> dict:
    cfg = V3.job_config(mono=False)
    cfg["reconstruction"]["vggtomega"]["turn_frames"]["enabled"] = False
    return cfg


def select_only(chain, frames, by_frame, cfg, T, Kn, prog, rot):
    """intake.parallax.select_keyframes for version 4: AMB3R's whole-video pass (run or reused), the selection on
    it, and nothing measured between the keyframes. (keyframes, windows, records)."""
    pipe, scan, frames_dir, config = (V2.CTX[k] for k in ("pipe", "scan", "frames_dir", "config"))
    V2.amb3r_only(pipe, scan, frames_dir, config)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "v2_select.py"), str(scan)], cwd=SERVER, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in r.stdout.splitlines():
        if line.strip() and not line.startswith("[Config]"):
            pipe.send_log(line)
    if r.returncode != 0:
        raise RuntimeError("the selection failed")
    kfs = [int(f) for f in json.loads((scan / "intake" / "keyframes_v2.json").read_text())["frames"]]
    usable = set(chain)
    if kfs[0] != chain[0] or any(f not in usable for f in kfs) or kfs != sorted(set(kfs)):
        raise RuntimeError("the selection is not an ordered list of usable frames starting at the first one")
    keyframes = [{"frame": chain[0], "file": by_frame[chain[0]]["file"], "anchor": None, "parallax_px": 0.0,
                  "rotation_deg": 0.0, "closed_by": "first_usable_frame"}]
    keyframes += [{"frame": k, "file": by_frame[k]["file"], "anchor": a, "parallax_px": None, "rotation_deg": None,
                   "closed_by": "overlap"} for a, k in zip(kfs, kfs[1:])]
    pipe.send_log(f"[v4] {len(kfs)} keyframes selected on AMB3R's map in {time.time() - t0:.0f} s")
    return keyframes, [], {}


def keyframes_as_witnesses(chain, kf_frames, frames, by_frame, cfg, T, Kn, prog):
    """intake.parallax.select_witnesses for version 4: the keyframes, and no other frame."""
    return [{"frame": int(f), "file": by_frame[int(f)]["file"],
             "reason": ("first_usable_frame" if int(f) == chain[0] else "keyframe"), "parallax_px": None}
            for f in sorted(int(k) for k in kf_frames)]


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    import intake.parallax as P
    import intake.walk as W
    import workers.map_worker as MW
    from card import card
    from intake import run_config as RC
    pipe = V2.Pipe()
    config = job_config()
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"[v4] the job's configuration frozen: {frozen['path']} (sha256 {frozen['sha256'][:12]})")
    job = {**config, "_pipeline_replace": False}
    V2.CTX.update(pipe=pipe, scan=scan, frames_dir=(scan / "frames").resolve(), config=job)
    P.select_keyframes = select_only
    P.stride_fill = V2.no_stride_fill
    P.select_witnesses = keyframes_as_witnesses
    W.walk_is_current = V3.walk_is_current
    MW._amb3r_coverage_fill = V3.amb3r_then_walk
    MW._run_precision_core = V3.chain_v3
    MW.subprocess = V3.OmegaRedirect(subprocess)
    t0 = time.time()
    with card("version 4: the keyframes alone", session=scan.parents[2].name, log=pipe.send_log):
        MW._map_work(pipe, str(scan), job)
    pipe.send_log(f"[v4] the cloud is published; the whole run took {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
