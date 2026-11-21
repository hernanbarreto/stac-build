"""VERSION 3, cloud B — "el segundo rehacemos f6 con pointdit prendido" (USER 2026-11-21).

The session's frozen configuration gets reconstruction.precision.mono_detail.enabled: true and nothing else; the
chain's own F6 runs again over the SAME F0, F4 and F5 (precision.runner.run_chain with from_step f6_bend — the
pipeline's own way of redoing its depth stage) and publishes a new epoch. Cloud A (PointDiT off) is copied aside first.

    python v3_pointdit.py <scan dir> <where cloud A is kept>"""
import shutil
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


def main() -> int:
    scan, keep = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    from card import card
    from intake import run_config as RC
    pipe = V2.Pipe()
    out = scan / "output"
    keep.parent.mkdir(parents=True, exist_ok=True)
    if not keep.exists():
        shutil.copy2(out / "cleaned_cloud.ply", keep)
        shutil.copy2(out / "precision" / "depth_on_f5.json", keep.with_suffix(".depth_on_f5.json"))
        pipe.send_log(f"[v3] cloud A (PointDiT off) kept aside → {keep}")
    config = V3.job_config(mono=True)
    frozen = RC.freeze_run_config(scan, config, stages=None, log=pipe.send_log)
    pipe.send_log(f"[v3] the job's configuration frozen again with mono_detail ON (sha256 {frozen['sha256'][:12]})")
    job = {**config, "_pipeline_replace": False}
    t0 = time.time()
    with card("version 3: F6 again with PointDiT on", session=scan.parents[2].name, log=pipe.send_log):
        rep = V3.run_steps(pipe, scan, job, from_step="f6_bend")
    pipe.send_log(f"[v3] cloud B (PointDiT on) published as epoch {rep['epoch']} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
