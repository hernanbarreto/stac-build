"""VERSION 5 — the pipeline as it is today (its own keyframe selection with the 8-frame rule and the coverage fill, its
turn frames and witness frames, PointDiT on) with ONLY what the user took out on 2026-11-21 and versions 3 and 4
showed neutral: no I3 (the walk from AMB3R's poses), no DA3 lock per cube (each cube at AMB3R's metre), DA3 once for
the session's metre, no F2 (F5's anchor tolerance from the tie's seams), nothing after F6.

USER 2026-11-21, to "los keyframes de hoy (los 174) con el pipeline ya limpio, sin I3, sin F2 y sin candado, dejando
todo lo demás como en la original": "si, dale, veamos como sale".

The job's configuration is server/config.yaml as it stands. No file of the pipeline is edited.

    python v5_front.py <scan dir>"""
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import v2_front as V2  # noqa: E402
import v3_front as V3  # noqa: E402

REAL = {}


def fill_then_walk(pipe, session_path, frames_dir, config):
    """map_worker._amb3r_coverage_fill as it is (AMB3R's whole-video pass and the coverage fill), then the walk of
    the final keyframes from AMB3R's poses."""
    REAL["fill"](pipe, session_path, frames_dir, config)
    V3.write_amb3r_walk(pipe, Path(session_path))


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    import intake.walk as W
    import workers.map_worker as MW
    from card import card
    from intake import run_config as RC
    pipe = V2.Pipe()
    config = yaml.safe_load((Path(SERVER) / "config.yaml").read_text())
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"[v5] the job's configuration frozen: {frozen['path']} (sha256 {frozen['sha256'][:12]}) — "
                  f"config.yaml as it stands")
    job = {**config, "_pipeline_replace": False}
    REAL["fill"] = MW._amb3r_coverage_fill
    W.walk_is_current = V3.walk_is_current
    MW._amb3r_coverage_fill = fill_then_walk
    MW._run_precision_core = V3.chain_v3
    MW.subprocess = V3.OmegaRedirect(subprocess)
    t0 = time.time()
    with card("version 5: today's keyframes, the pipeline without I3, F2 and the DA3 lock", session=scan.parents[2].name,
              log=pipe.send_log):
        MW._map_work(pipe, str(scan), job)
    pipe.send_log(f"[v5] the cloud is published; the whole run took {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
