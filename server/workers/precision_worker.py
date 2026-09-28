# STAC-Builder: Precision Worker (Subprocess)
# claude_stac.txt §3 EVIDENCE + CORE as the pipeline stage after the semantics of
# epoch 0 (USER 2026-09-28: "Reconstruir" builds the cloud WITH F0-F7, in place of
# the old correction): visit_drift MEASURE → F0 camera → F2 gauge → F4 tracks →
# F3 probe → F5 refinement → F6 native depth (+ COLMAP reference) → F7 witness
# fusion → the published epoch IS the deliverable. The steps and their resume live
# in precision/runner.py (the one list, also `python -m precision.runner`).

import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _precision_work(pipe: WorkerPipe, session_dir: str, config: dict):
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from precision.config import load_precision_config
    from precision.runner import ChainError, run_chain
    from workers.base import stop_semantic_service_verified

    pcfg = load_precision_config(config)
    out = Path(session_dir) / "output"
    for need in ("cleaned_cloud.ply", "camera_poses.txt"):
        if not (out / need).exists():
            raise RuntimeError(f"No {need} in {out} — the reconstruction and the cloud cleaning "
                               f"must run before the precision core")

    def _progress(pct, msg):
        pipe.send_progress(float(pct), str(msg), stage="precision")

    def _before_gpu(label):
        stop_semantic_service_verified(pipe, stage=f"precision {label}")

    try:
        rep = run_chain(Path(session_dir), pcfg, log=lambda m: pipe.send_log(str(m)),
                        progress=_progress, cancelled=pipe.check_cancel,
                        before_gpu=_before_gpu)
    except ChainError as e:
        if pipe.check_cancel():
            return
        raise RuntimeError(str(e)) from e
    pipe.send_log(f"[precision] F0-F7 done in {rep['seconds'] / 60:.1f} min — the fused cloud is "
                  f"epoch {rep['epoch']} (earlier epochs stay selectable)")
    pipe.send_progress(100, f"Precision F0-F7: epoch {rep['epoch']}", stage="precision")


def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_precision_work, conn, session_dir, config)
