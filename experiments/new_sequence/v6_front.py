"""VERSION 6 — EVERY FRAME OF THE VIDEO IS A KEYFRAME.

USER 2026-11-21: "quiero probar ahora bufferstop v6 armando los cubos de omega con todos los frames con 64 frames y
solape del 32, o sea igual, pero le metemos todos los frames a lo bruto no hace falta el relleno ni nada" (earlier:
"y luego el resto f0 a f6 igual que ahora … dejamos que lo resuelvan el resto de las etapas").

The pipeline's worker as version 5 runs it (the walk from AMB3R's poses, each cube at AMB3R's metre, DA3 once for the
session's metre, the chain F0, F4, F5, F6 with F5's anchor tolerance from the tie's seams), with:

  * the keyframes = every frame of the video (intake.parallax.select_keyframes replaced; no frame is rejected by its
    exposure) — so there is no other frame: no turn frame (the pipeline's own switch off), no witness frame
    (the witness list is the keyframes), no coverage fill;
  * the cubes by the pipeline's planner: 64 keyframes sharing 32;
  * the vote asks for 10 confirming views instead of 5 (his value for this version);
  * PointDiT off (said to the user at the launch).

    python v6_front.py <scan dir>"""
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
import v4_front as V4  # noqa: E402


def every_frame(chain, frames, by_frame, cfg, T, Kn, prog, rot):
    """intake.parallax.select_keyframes for version 6: every usable frame, in order. (keyframes, windows, records)."""
    keyframes = [{"frame": chain[0], "file": by_frame[chain[0]]["file"], "anchor": None, "parallax_px": 0.0,
                  "rotation_deg": 0.0, "closed_by": "first_usable_frame"}]
    keyframes += [{"frame": f, "file": by_frame[f]["file"], "anchor": a, "parallax_px": None, "rotation_deg": None,
                   "closed_by": "every_frame"} for a, f in zip(chain, chain[1:])]
    V2.CTX["pipe"].send_log(f"[v6] the keyframes are the video's {len(keyframes)} frames")
    return keyframes, [], {}


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    import intake.parallax as P
    import intake.walk as W
    import workers.map_worker as MW
    from card import card
    from intake import run_config as RC
    pipe = V2.Pipe()
    config = V4.job_config()
    # USER 2026-11-21, for this version: "lo que hay que cambiar es el voto multivista porque ahora votan 5 no?,
    # deberian votar 10 por lo menos" — every frame is a view now
    config["reconstruction"]["precision"]["cloud"]["min_support_views"] = 10
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"[v6] the job's configuration frozen: {frozen['path']} (sha256 {frozen['sha256'][:12]})")
    job = {**config, "_pipeline_replace": False}
    V2.CTX.update(pipe=pipe, scan=scan, frames_dir=(scan / "frames").resolve(), config=job)
    P.select_keyframes = every_frame
    P.stride_fill = V2.no_stride_fill
    P.select_witnesses = V4.keyframes_as_witnesses
    W.walk_is_current = V3.walk_is_current
    MW._amb3r_coverage_fill = V3.amb3r_then_walk
    MW._run_precision_core = V3.chain_v3
    MW.subprocess = V3.OmegaRedirect(subprocess)
    t0 = time.time()
    with card("version 6: every frame a keyframe", session=scan.parents[2].name, log=pipe.send_log):
        MW._map_work(pipe, str(scan), job)
    pipe.send_log(f"[v6] the cloud is published; the whole run took {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
