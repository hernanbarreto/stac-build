# STAC-Builder: Certification Worker (Subprocess)
# claude_stac.txt §9 as a PIPELINE stage (USER 2026-09-13: "todo debe
# ejecutarse automáticamente en el pipeline de reconstrucción" — no button,
# no manual trigger). Runs after the cleaned cloud + the automatic
# segmentation: instance + revisit loops → closed scale → keyframe SE(3)
# graph → depth by correspondences → witnesses; one pending epoch per
# iteration (selectable in the kit), acta in output/certify_acta.json.
#
# After the certification — the last mutation of the instances and the last
# GPU-exclusive step of the pipeline — every object gets its ShapeR description
# from the session's Qwen3-VL (USER 2026-10-01; segmentation/object_captioner.py).
#
# DETERMINISM (docs/plan_determinismo.md, 2026-10-08):
#   point 139 — the stage reads ONE configuration: the job's frozen
#     output/run_config.yaml (verified against its digest, and against the dict
#     the manager handed this process), passed explicitly to every loader; its
#     sha256 is sealed into the epoch, the acta and the reports. Nothing here
#     re-reads config.yaml.
#   point 148 — the object descriptions are ALL OR NOTHING per run: a service
#     that does not come up or a call that fails FAILS the stage, declared; no
#     later run fills gaps.
#   point 160 — this worker is the ONE launcher of the certification; the
#     server never runs it in a thread (reconstruction/certify/api.auto_run
#     enqueues this stage).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import json
import sys
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe

STAGE = "certify"


class CaptionsFailed(RuntimeError):
    """The per-object description pass did not describe every object — the stage fails
    (point 148); the reason names what did not happen."""


def _caption_objects(pipe: WorkerPipe, session_path: Path, config: dict, ccfg):
    """One ShapeR description per object (USER 2026-10-01: "la descripción de
    los objetos la que armó el VLM como descripción para ShapeR"): Qwen3-VL
    sees every instance isolated in its best SAM3-mask views, ONE call per
    object, written back on the instance as `shape_caption` (source 'object',
    vlm_proposed). Runs HERE because the certification is the last stage that
    rewrites the instances and the last one that needs the GPU to itself: the
    service is brought up if SAM3 / the reconstruction stopped it (cold start
    ~4.5 min, declared) and LEFT UP — the pipeline's end reloads it anyway.

    ALL OR NOTHING (point 148): a service that cannot be brought up, a call
    that fails or an object left without a view FAILS THE STAGE with the
    reason — the delivered segmentation_result.json never mixes descriptions
    of different runs. Disabled in the configuration: nothing runs, declared."""
    if not ccfg.enabled:
        pipe.send_log("[captions] segmentation.object_captions.enabled is false — the "
                      "instances keep the concept descriptions of the prompt pass")
        return None
    from segmentation.object_captioner import caption_session_objects
    pipe.send_progress(96, "ShapeR descriptions: one Qwen3-VL call per object "
                           f"({ccfg.views} view(s) each)", stage=STAGE)
    try:
        res = caption_session_objects(session_path / "output", session_path,
                                      views=ccfg.views, log=pipe.send_log,
                                      cancelled=pipe.check_cancel, config=config)
    except Exception as e:  # noqa: BLE001 — declared, and it FAILS the stage
        raise CaptionsFailed(f"per-object descriptions failed ({type(e).__name__}: {e}) — the "
                             f"stage fails: a run describes every object or none (point 148)") from e
    pipe.send_log(f"[captions] object descriptions: {res.get('generated')} generated, "
                  f"{res.get('kept')} kept, {res.get('failed')} failed, "
                  f"{res.get('skipped')} skipped of {res.get('n_instances')} instance(s)"
                  + (f" — {res['reason']}" if res.get("reason") else ""),
                  level="warning" if (res.get("skipped") or res.get("failed")) else "info")
    if res.get("skipped") or res.get("failed"):
        raise CaptionsFailed(f"{res.get('failed')} object(s) failed and {res.get('skipped')} were "
                             f"skipped of {res.get('n_instances')}"
                             + (f" — {res['reason']}" if res.get("reason") else "")
                             + " — the stage fails: a run describes every object or none "
                               "(point 148); nothing is left for a later run to fill")
    return res


def _frozen_config(session_path: Path, handed: dict, log) -> tuple:
    """(the configuration this stage runs with, its sha256): the job's frozen
    output/run_config.yaml with the stage's overrides, verified against the dict the manager
    handed this process (the same job's) — a difference FAILS naming the first key (two
    configurations in one run is what point 139 ends)."""
    from intake.run_config import canonical_yaml, effective_config, load_run_config
    frozen, sha = load_run_config(session_path)
    raw = effective_config(frozen, STAGE)
    mine = {k: v for k, v in handed.items() if not str(k).startswith("_")}
    for key in sorted(set(mine) | set(raw)):
        a, b = mine.get(key), raw.get(key)
        if canonical_yaml({key: a}) != canonical_yaml({key: b}):
            raise RuntimeError(f"the configuration handed to the certification differs from the "
                               f"job's frozen output/run_config.yaml at '{key}' — a stage never runs "
                               f"on a configuration nobody froze (point 139)")
    log(f"[certify] configuration: the job's frozen run_config.yaml (sha256 {sha[:12]})")
    return raw, sha


def _certify_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """Certification loop — runs inside a dedicated subprocess."""
    session_path = Path(session_dir)
    output_dir = (session_path / "output").resolve()

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    # from here on `config` IS the job's frozen configuration (point 139)
    config, config_sha = _frozen_config(session_path, config, pipe.send_log)
    from reconstruction.loops.config import load_loops_config
    cfg = load_loops_config(config)
    from correction.config import load_correction_config
    ccfg_corr = load_correction_config(config)
    # the per-object description pass reads its keys HERE, before the hour of
    # certification: a missing / bad key fails the stage naming it, now
    from segmentation.object_captioner import load_object_captions
    captions_cfg = load_object_captions(config)

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
                       stage=STAGE)

    from reconstruction.certify.run import certify_session

    def _log(msg: str):
        pipe.send_log(str(msg))

    # The bar used to jump from 5 % to 100 % with 40 minutes of silence in
    # between (pccr 2026-09-21: the user watched it frozen at 5 % while the
    # correction ran, applied and published an epoch). Certify now reports the
    # real advance of every stage and the correction reports its own inside it.
    def _progress(pct: int, msg: str):
        pipe.send_progress(int(pct), str(msg), stage=STAGE)

    acta = certify_session(session_path, cfg=cfg, operator="pipeline", log=_log,
                           progress=_progress, correction_cfg=ccfg_corr,
                           raw_config=config, config_sha256=config_sha)

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
                  f"epoch {acta.get('epoch_initial')} → {acta.get('epoch_final')}"
                  f"{' (REUSED)' if acta.get('reused') else ''}; "
                  f"objective {_m(mi, 'objective')} → {_m(mf, 'objective')}; "
                  f"duplicates {_m(mi, 'duplicates', 'n')} → {_m(mf, 'duplicates', 'n')}; "
                  f"closure median {_m(mi, 'closure', 'median_m')} → {_m(mf, 'closure', 'median_m')} m")

    # the instances are final now: describe each one for ShapeR — all or nothing (point 148)
    if seg.exists():
        _caption_objects(pipe, session_path, config, captions_cfg)

    pipe.send_progress(100, f"Certification: {acta.get('stop_reason')} "
                            f"(epoch {acta.get('epoch_final')})", stage=STAGE)


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_certify_work, conn, session_dir, config)
