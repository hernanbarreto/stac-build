# STAC-Builder: Certification Worker (Subprocess)
# claude_stac.txt §9 as a PIPELINE stage (USER 2026-09-13: "todo debe
# ejecutarse automáticamente en el pipeline de reconstrucción" — no button,
# no manual trigger). Runs after the cleaned cloud + the automatic
# segmentation: instance + revisit loops → closed scale → keyframe SE(3)
# graph → depth by correspondences → witnesses; one pending epoch per
# iteration (selectable in the kit), acta in output/certify_acta.json.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _certify_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """Certification loop — runs inside a dedicated subprocess."""
    session_path = Path(session_dir)
    output_dir = (session_path / "output").resolve()

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    from reconstruction.loops.config import load_loops_config
    cfg = load_loops_config(config)

    cloud = output_dir / "cleaned_cloud.ply"
    if not cloud.exists():
        raise RuntimeError(f"No cleaned_cloud.ply in {output_dir} — the cloud cleaning stage "
                           f"must run before the certification")
    if not (output_dir / "camera_poses.txt").exists():
        raise RuntimeError(f"No camera_poses.txt in {output_dir} — the reconstruction left no poses")

    seg = output_dir / "segmentation_result.json"
    if seg.exists():
        pipe.send_log("[certify] instances on disk — instance loops (§4.4) + geometric revisits")
    else:
        pipe.send_log("[certify] no segmentation_result.json — geometric revisits only "
                      "(no instance loops, no structural constraints)", level="warning")

    if pipe.check_cancel():
        return

    # The semantic service stays UP here: the instance classification
    # (§4.4 structural|movable|dynamic) needs Qwen — SAM3 stopped vLLM for
    # its exclusive window, so the classifier brings it back
    # (semantic.service.ensure_service). pccr 2026-09-13 21:03: with vLLM
    # down every instance fell to the default class → 0 instance loops,
    # 0 structural constraints, and the epoch was rejected.
    pipe.send_progress(2, "Certification: instance loops and geometric revisits",
                       stage="certify")

    from reconstruction.certify.run import certify_session

    def _log(msg: str):
        pipe.send_log(str(msg))

    # The bar used to jump from 5 % to 100 % with 40 minutes of silence in
    # between (pccr 2026-09-21: the user watched it frozen at 5 % while the
    # correction ran, applied and published an epoch). Certify now reports the
    # real advance of every stage and the correction reports its own inside it.
    def _progress(pct: int, msg: str):
        pipe.send_progress(int(pct), str(msg), stage="certify")

    acta = certify_session(session_path, cfg=cfg, operator="pipeline", log=_log,
                           progress=_progress)

    if pipe.check_cancel():
        return

    its = acta.get("iterations", [])
    applied = sum(1 for it in its if it.get("verdict") == "applied")
    mi, mf = acta.get("metrics_initial") or {}, acta.get("metrics_final") or {}

    def _m(m, *keys):
        v = m
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        return v

    pipe.send_log(f"[certify] {acta.get('stop_reason')} — {len(its)} iteration(s), {applied} applied, "
                  f"epoch {acta.get('epoch_initial')} → {acta.get('epoch_final')}; "
                  f"objective {_m(mi, 'objective')} → {_m(mf, 'objective')}; "
                  f"duplicates {_m(mi, 'duplicates', 'n')} → {_m(mf, 'duplicates', 'n')}; "
                  f"closure median {_m(mi, 'closure', 'median_m')} → {_m(mf, 'closure', 'median_m')} m")
    pipe.send_progress(100, f"Certification: {acta.get('stop_reason')} "
                            f"(epoch {acta.get('epoch_final')})", stage="certify")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_certify_work, conn, session_dir, config)
