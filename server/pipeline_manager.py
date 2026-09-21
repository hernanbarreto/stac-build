# STAC-Builder: Pipeline Manager
# Orchestrates reconstruction stages as independent subprocesses.
# Each stage runs in its own multiprocessing.Process with Pipe IPC.
# The server process never loads GPU models — it only spawns workers.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process, get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


# ── Stage Definitions ────────────────────────────────────────

class StageId(str, Enum):
    RECONSTRUCTION = "reconstruction"
    CLOUDCOMPY = "cloudcompy"
    PGSR = "pgsr"
    TSDF = "tsdf"
    VLM = "vlm"
    SAM3 = "sam3"
    CERTIFY = "certify"
    INSTANCE_CLEANER = "instance_cleaner"


STAGE_REGISTRY = {
    StageId.RECONSTRUCTION:   {"label": "3D Reconstruction", "icon": "🔨", "module": "workers.map_worker"},
    StageId.CLOUDCOMPY:       {"label": "Cloud Cleaning",    "icon": "🧹", "module": "workers.cloudcompy_worker"},
    StageId.PGSR:             {"label": "Precision (PGSR)",  "icon": "💎", "module": "workers.pgsr_worker"},
    StageId.TSDF:             {"label": "TSDF Mesh",         "icon": "🧊", "module": "workers.tsdf_worker"},
    StageId.VLM:              {"label": "Scene Analysis",    "icon": "🔍", "module": "workers.vlm_worker"},
    StageId.SAM3:             {"label": "Segmentation",      "icon": "🏷️", "module": "workers.sam3_worker"},
    StageId.CERTIFY:          {"label": "Certification",     "icon": "📐", "module": "workers.certify_worker"},
    StageId.INSTANCE_CLEANER: {"label": "Instance Cleaning", "icon": "✨", "module": "workers.instance_cleaner_worker"},
}

# THE reconstruction pipeline — the default run is end to end; a caller that
# needs a partial relaunch names the stages (build_pipeline_stages stages= /
# from_stage=), which is the only way to get one.
#   Reconstruction (SIMPLE: sparse frames → ONE Omega pass → metric scale →
#                   upright orientation baked into cloud + poses)
#   → VLM   (Qwen3-VL understands the scene → RICH concept phrases, no boxes)
#   → SAM3  (one concept session per phrase — SAM3's tracking IS the identity;
#            instance store scene_r.db rebuilt from the clean instances)
#   → CloudCompy (merge/clean → cleaned cloud + witnesses + Potree + mask→cloud)
#   → Certify (claude_stac.txt §9, USER 2026-09-13 "todo automático": instance +
#            revisit loops → closed scale → keyframe SE(3) graph → depth by
#            correspondences → witnesses; one pending epoch per iteration,
#            the epoch selector in the kit — the duplicates of a drifted revisit
#            are closed HERE, so the stage is part of "Reconstruir")
#   → scene TSDF (fusion + texrecon photo texture).
# The former Phase R (semantic anchoring) was REMOVED 2026-07-09: the one-pass
# reconstruction has no window seams to anchor, and the anchoring never beat
# its own A/B gate. Its shared primitives (instance store, geometry) live on
# under phase_r/ as the data layer for phases 2-6.
DEFAULT_STAGE_ORDER: List[StageId] = [
    StageId.RECONSTRUCTION,
    StageId.CLOUDCOMPY,    # BEFORE the semantic stages: SAM3 has to project its
                           # masks onto the scene cloud the moment it finishes,
                           # and with CloudCompy downstream that cloud did not
                           # exist yet — the match was structurally impossible
                           # and degraded to instances with no points. It also
                           # means a SAM3 failure no longer costs the user the
                           # cloud: it is on screen before segmentation starts.
    StageId.VLM,
    StageId.SAM3,
    StageId.CERTIFY,
    StageId.PGSR,          # precision mode only: no-ops unless backend is
                           # vggtomega_pgsr (seeds from cleaned_cloud, so it runs
                           # after CloudCompy; the TSDF then integrates its
                           # rendered depths via depth_source "pgsr_render")
    StageId.TSDF,
]


class PipelineSelectionError(ValueError):
    """An explicit stage selection that cannot be honoured: an unknown name, a
    stage this backend does not have, or a selection that comes out empty.
    It is RAISED, never quietly narrowed — the caller asked for a precise set
    of stages (USER 2026-09-21: "relances desde segmentacion con todas sus
    etapas, ademas de la certificacion") and a relaunch that silently runs a
    different set than the one asked for is exactly how 3.5 h of
    reconstruction got re-run."""


@dataclass
class PipelineStage:
    id: StageId
    enabled: bool = True
    config: dict = field(default_factory=dict)
    # True when the CALLER named this stage (stages=/from_stage=), as opposed to
    # the automatic full chain. An explicitly named stage RUNS: the resume probe
    # is not consulted for it, because the selection IS the resume decision the
    # user already made (finding 21 — before this there was no way to say "from
    # the VLM down", and every worker decided for itself whether it had run).
    explicit: bool = False

    @property
    def label(self) -> str:
        return STAGE_REGISTRY[self.id]["label"]

    @property
    def icon(self) -> str:
        return STAGE_REGISTRY[self.id]["icon"]


# ── Job State ────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageState:
    stage: PipelineStage
    status: JobStatus = JobStatus.QUEUED
    pct: float = 0
    message: str = ""
    elapsed: float = 0


@dataclass
class PipelineJob:
    session_id: str
    stages: List[StageState]
    status: JobStatus = JobStatus.QUEUED
    current_stage_idx: int = -1
    # The scan this job resolved to. A project has several scans and the ACTIVE
    # one is not necessarily the one being reconstructed: on pccr 2026-09-14 the
    # pipeline rebuilt 2026-08-31 while the active scan was 2026-08-24, and the
    # completion handler — which asked the session for its active scan instead
    # of asking the job — pointed the Potree build at an output/ with no cloud
    # in it, declared "Potree conversion failed" over a perfectly good octree
    # and fell through to the raw-cloud broadcast.
    session_dir: Optional[str] = None
    _process: Optional[Process] = field(default=None, repr=False)
    _server_conn: Optional[Connection] = field(default=None, repr=False)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "status": self.status.value,
            "current_stage_idx": self.current_stage_idx,
            "stages": [
                {
                    "id": ss.stage.id.value,
                    "label": ss.stage.label,
                    "icon": ss.stage.icon,
                    "enabled": ss.stage.enabled,
                    "explicit": ss.stage.explicit,
                    "status": ss.status.value,
                    "pct": ss.pct,
                    "message": ss.message,
                    "elapsed": ss.elapsed,
                }
                for ss in self.stages
            ],
        }


# ── Pipeline Manager ─────────────────────────────────────────

# Type alias for progress callback: async fn(session_id, job_dict)
ProgressCallback = Callable[[str, dict], Awaitable[None]]


class PipelineManager:
    """Orchestrates reconstruction pipelines for sessions.
    
    Usage:
        pm = PipelineManager()
        await pm.start_pipeline(session_id, stages, config, on_progress)
        await pm.cancel_pipeline(session_id)
        status = pm.get_status(session_id)
    """

    def __init__(self):
        self._jobs: Dict[str, PipelineJob] = {}

    # ── Public API ───────────────────────────────────────────

    async def start_pipeline(
        self,
        session_id: str,
        stages: List[PipelineStage],
        config: dict,
        on_progress: Optional[ProgressCallback] = None,
        on_complete: Optional[Callable[[str, bool], Awaitable[None]]] = None,
        replace: bool = False,
        scan_key: Optional[str] = None,
    ) -> PipelineJob:
        """Start a pipeline for the given session.
        
        Args:
            session_id: Session directory name (e.g. "2026-01-31-22-23-52")
            stages: Ordered list of stages to run (only enabled ones execute)
            config: Server config dict (from config.yaml)
            on_progress: Async callback(session_id, job_dict) for progress updates
            on_complete: Async callback(session_id, success) when pipeline finishes
            replace: If True, WIPE output/ (and its derived caches) and re-run every
                     stage from scratch. Defaults to False: resume is the safe default,
                     an omitted flag must never destroy a reconstruction — and since
                     2026-09-21 it does not, replace=False deletes NOTHING. A stage
                     that cannot re-run without rewriting a finished predecessor
                     refuses instead (_destructive_conflict).
            scan_key: Optional "date/source" key (e.g. "2026-03-07/legacy") to target
                      a specific scan. If None, resolves to latest scan/first source.
        """
        # ONE AND ONLY ONE (USER ORDER 2026-09-05): a reconstruction command
        # never queues and never cancel-and-replaces a running one — if a job
        # for this session is active, the new request is REFUSED loudly.
        _existing = self._jobs.get(session_id)
        if _existing and _existing.status in (JobStatus.QUEUED,
                                              JobStatus.RUNNING):
            raise RuntimeError(
                f"[Pipeline] a pipeline for {session_id} is already "
                f"{_existing.status.value} — one and only one; command refused")

        # Build job
        stage_states = [StageState(stage=s) for s in stages]
        job = PipelineJob(session_id=session_id, stages=stage_states)
        self._jobs[session_id] = job

        # Resolve session directory (supports both new-style projects/ and legacy scans/)
        from project_paths import resolve_session, ProjectPaths
        server_dir = str(Path(__file__).parent)

        if scan_key:
            # Explicit scan target: "date/source"
            parts = scan_key.split("/", 1)
            date = parts[0]
            source = parts[1] if len(parts) > 1 else "default"
            from config import PROJECTS_DIR
            projects_dir = PROJECTS_DIR
            if (projects_dir / session_id / "project.json").exists():
                paths = ProjectPaths(str(projects_dir), session_id)
                ctx = paths.for_source(date, source)
                session_dir = str(ctx.session_dir)
                # The scan being reconstructed becomes the project's ACTIVE scan.
                # Work is per scan (USER 2026-09-06) and everything downstream
                # that resolves a session without an explicit key — the viewer,
                # /api/segmentation/refresh, the certify state, the post-pipeline
                # Potree build — asks for the active one. On pccr 2026-09-14 the
                # pipeline rebuilt 2026-08-31 while 2026-08-24 stayed active, so
                # those callers kept looking into a scan with no reconstruction
                # in it.
                try:
                    from project_scans import set_active
                    set_active(paths, f"{date}/{source}")
                except Exception as e:  # noqa: BLE001 — never block a run over this
                    logger.warning(f"[Pipeline] could not set {date}/{source} "
                                   f"as the active scan: {e}")
            else:
                # Legacy: scan_key is ignored, use normal resolution
                ctx = resolve_session(server_dir, session_id)
                session_dir = str(ctx.session_dir)
        else:
            ctx = resolve_session(server_dir, session_id)
            session_dir = str(ctx.session_dir)

        job.session_dir = session_dir

        # Start the orchestration loop as an asyncio task
        job._task = asyncio.create_task(
            self._run_pipeline(job, session_dir, config, on_progress, on_complete, replace)
        )

        logger.info(f"[Pipeline] Started for {session_id} (scan={scan_key or 'auto'}): {[s.stage.id.value for s in stage_states if s.stage.enabled]}, replace={replace}")
        return job

    async def cancel_pipeline(self, session_id: str):
        """Cancel a running pipeline."""
        job = self._jobs.get(session_id)
        if not job or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return

        logger.info(f"[Pipeline] Cancelling {session_id}")

        # Send cancel via pipe
        if job._server_conn:
            try:
                job._server_conn.send({"type": "cancel"})
            except Exception:
                pass

        # Terminate the WHOLE stage process group (workers call os.setsid(),
        # so bash/DA3/CloudCompy children die with them — a bare terminate()
        # left GPU-holding orphans behind), then the worker itself as fallback.
        if job._process and job._process.is_alive():
            self._kill_stage_tree(job._process)

        # Cancel the asyncio task
        if job._task and not job._task.done():
            job._task.cancel()

        job.status = JobStatus.CANCELLED
        if 0 <= job.current_stage_idx < len(job.stages):
            job.stages[job.current_stage_idx].status = JobStatus.CANCELLED

    @staticmethod
    def _kill_stage_tree(process, timeout: float = 5.0) -> None:
        """Kill a stage worker AND every child it spawned. Workers call
        os.setsid() at startup (workers/base.py), so the whole stage subtree
        (bash launchers, DA3/CloudCompy/SAM3 children) shares one process
        group: SIGTERM the group, wait, SIGKILL survivors. Without this, a
        cancel orphaned the children, which kept running and holding GPU
        memory (the 24 GB zombie). Falls back to plain terminate/kill when
        the group is gone already."""
        import os
        import signal

        pid = process.pid
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
        try:
            if pgid is not None and pgid != os.getpgid(0):
                os.killpg(pgid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError):
            process.terminate()
        process.join(timeout=timeout)
        if process.is_alive():
            try:
                if pgid is not None and pgid != os.getpgid(0):
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.join(timeout=2)
        # sweep any survivors of the group (children that outlived the worker)
        if pgid is not None and pgid != os.getpgid(0):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def get_status(self, session_id: str) -> Optional[dict]:
        """Get current status of a pipeline job."""
        job = self._jobs.get(session_id)
        return job.to_dict() if job else None

    def job_session_dir(self, session_id: str) -> Optional[str]:
        """The scan directory the running/last job resolved to — the one whose
        output/ it wrote. Callers must prefer this over the session's ACTIVE
        scan, which can be a different scan entirely."""
        job = self._jobs.get(session_id)
        return job.session_dir if job else None

    def get_all_jobs(self) -> Dict[str, dict]:
        """Get status of all pipeline jobs."""
        return {sid: job.to_dict() for sid, job in self._jobs.items()}

    # ── Internal: Pipeline Orchestration Loop ─────────────────

    # Files each stage produces (in output/) — used by replace mode to clean before re-running
    STAGE_OUTPUT_FILES: Dict[StageId, List[str]] = {
        StageId.RECONSTRUCTION: ["chunk_*.ply", "chunk_*_origins.npz", "chunk_*_meta.json",
                      "slam_reconstruction.ply", "maplong_run", "da3_run", "gs_ply",
                      "gaus_slam_run", "da3_depth",
                      "vggt_long_config.yaml", "vggt_omega_config.yaml", "da3_streaming_config.yaml",
                      "camera_poses_mapanything.json", "camera_poses.txt", "camera_frames.txt",
                      "intrinsic.txt", "lidar_complement.ply", "omega_run",
                      ".metric_scale_applied", ".orientation_applied",
                      # a NEW reconstruction is geometry epoch 0 again: the
                      # epoch marker, the depth-correction sidecar, the exact
                      # per-epoch transforms and any pending tx/prev dirs are
                      # cleared. corrections.jsonl (the ledger) is NEVER
                      # deleted — history survives; replay re-keys by
                      # frame_global (USER 2026-09-08).
                      "geometry_epoch.json", "depth_correction.json",
                      "corrections/epoch_*.npz", "chunk_plan.json",
                      "_tx_epoch_*", "_epoch_*", "scale_diagnostics.json"],
                      # ↑ idempotency markers: MUST be cleared on Replace, else scale_align /
                      # orient skip on every re-run → cloud not metric, scene upside down.
        StageId.CLOUDCOMPY: ["cleaned_cloud.ply", "cleaned_cloud_raw.ply",
                             "floor_transform.npz", "scene_consolidate_report.json",
                             "corrected_cloud.ply"],
        StageId.PGSR: ["pgsr_scene", "pgsr_model", "pgsr_render"],
        StageId.TSDF: ["tsdf/scene"],
        StageId.VLM: ["scene_analysis.json", "vlm_analysis.json",
                      "scene_understanding.json", "autoprompt_instances.json",
                      "autoprompt_review_queue.json"],
        # the mask→cloud mapping now runs inside SAM3 (the cloud already
        # exists when it finishes) → its products are SAM3's outputs
        StageId.SAM3: ["segmentation.json", "segmentation_result.json",
                       "seg_masks.npz", "seg_broadcast.json", "scene_r.db",
                       "classification.npy", "class_map.json",
                       # the fusion folds the parent and archives the raw SAM3
                       # output beside it; both belong to the stage that
                       # produced them, so a re-run starts from masklets again
                       "fusion_map.json", "_sam3_raw"],
        # the certification's records (the acta, the per-epoch quality reports,
        # the post-hoc graph, the candidates/duplicates lists). Its epochs are
        # correction artifacts: a NEW reconstruction wipes output/ (epoch 0
        # again) — the ledger corrections.jsonl is never deleted.
        StageId.CERTIFY: ["certify_acta.json", "visit_drift_report.json", "quality",
                          "keyframe_graph.json", "loop_candidates.json", "duplicates.json",
                          "loop_semantics.json"],
        StageId.INSTANCE_CLEANER: ["instance_*.ply", "inst_cleaned_cloud.ply"],
    }

    # Files in frames/ dir that should be regenerated on reconstruction.
    #
    # frames/selected_frames.json is NOT here and must not come back: it is the
    # INPUT of the segmentation, not an output of the reconstruction. On
    # 2026-09-21 at 10:00 a resume run deleted it on the way in, and SAM3 —
    # which falls back to "all frames" when the file is missing (main.py,
    # _sam3_init) — started segmenting 3,044 video frames in 12 batches
    # instead of the 216 keyframes: 14× the work, with masks in video-frame
    # space instead of keyframe space. map_worker WRITES the file on every
    # reconstruction that is allowed to re-select frames (replace=on), so
    # deleting it first buys nothing and costs the segmentation its keyframe
    # space whenever the run is cancelled in between.
    FRAMES_DIR_FILES: List[str] = [
        "selected_frames_seg*.json", "frame_quality.json",
        "da3_frames.json",   # dense-set list (legacy da3/mapanything backends)
    ]

    # BIM comparison / sábana artifacts (may live in output/, session root, or bim_comparison/)
    BIM_COMPARISON_FILES: List[str] = [
        "sabana.npz", "sabana_cloud.ply", "sabana_meta.json", "sabana_potree",
    ]

    # Cascade: when a stage re-runs, these DOWNSTREAM stages' outputs are ALSO
    # invalidated. MUST mirror DEFAULT_STAGE_ORDER:
    #   RECONSTRUCTION → CLOUDCOMPY → VLM → SAM3 → CERTIFY → TSDF
    # A cascade edge pointing UPSTREAM deletes freshly produced artifacts
    # mid-pipeline — that exact bug (CLOUDCOMPY → SAM3, a relic of the old
    # cloudcompy-before-sam3 order) silently erased segmentation.json +
    # seg_masks.npz right after SAM3 wrote them (test2, 2026-07-08).
    CASCADE_INVALIDATION: Dict[StageId, List[StageId]] = {
        StageId.RECONSTRUCTION: [
            StageId.CLOUDCOMPY,       # cleaned_cloud depends on chunks
            StageId.VLM,              # scene analysis ran on old keyframes
            StageId.SAM3,             # segmentation ran on old frames
            StageId.CERTIFY,          # the acta certified the old geometry
            StageId.PGSR,             # PGSR trained on old poses/cloud
            StageId.TSDF,             # TSDF mesh integrated old depth/poses
            StageId.INSTANCE_CLEANER, # instance PLYs from old segmentation
        ],
        StageId.VLM: [
            StageId.SAM3,             # SAM3 uses VLM categories
            StageId.CERTIFY,          # instance loops come from the segmentation
            StageId.INSTANCE_CLEANER,
        ],
        StageId.SAM3: [
            StageId.CERTIFY,          # instance loops come from the segmentation
            StageId.PGSR,             # dynamic masks come from SAM3 artifacts
            StageId.INSTANCE_CLEANER, # instances depend on segmentation
        ],
        StageId.CLOUDCOMPY: [
            StageId.VLM,              # the semantic stages read THIS cloud
            StageId.SAM3,             # masks are projected onto THIS cloud
            StageId.CERTIFY,          # the loop certifies THIS cleaned cloud
            StageId.PGSR,             # the Gaussian seed is the cleaned cloud
            StageId.TSDF,             # TSDF masks to the old cleaned_cloud
            StageId.INSTANCE_CLEANER,
        ],
        StageId.CERTIFY: [
            StageId.PGSR,             # a mesh is built on the certified geometry
            StageId.TSDF,
        ],
        StageId.PGSR: [
            StageId.TSDF,             # precision TSDF integrates pgsr_render depths
        ],
    }

    # Stages whose worker REWRITES, in place and without staging, artifacts that
    # other stages read. Nothing here is deleted by the manager — these names are
    # what a re-run would take with it, so the manager can NAME them in a refusal.
    #
    # 2026-09-21 10:00, the incident that put this map here: a run_pipeline with
    # replace=False (relaunch from the VLM) enabled every stage, so RECONSTRUCTION
    # was in the list, and STARTING it wiped cleaned_cloud.ply,
    # cleaned_cloud_raw.ply, camera_poses.txt, intrinsic.txt, omega_run/, da3_run/,
    # chunk_plan.json, scale_diagnostics.json, the idempotency markers and
    # frames/selected_frames.json. Cancelled 40 s later: too late, 3.5 h gone.
    # replace=False now deletes NOTHING — pre-cleaning is what replace MEANS — and
    # a stage that cannot run without destroying a FINISHED predecessor refuses
    # and names the flag that authorises it.
    IN_PLACE_REWRITERS: Dict[StageId, List[str]] = {
        # map_worker's phase-2 chunked re-run unlinks these by name before it
        # starts over (workers/map_worker.py, "wipe phase-1 reconstruction
        # artifacts"), and re-selects the keyframes on top.
        StageId.RECONSTRUCTION: [
            "camera_poses.txt", "camera_frames.txt", "intrinsic.txt",
            "omega_run", "maplong_run", "da3_run",
            "chunk_plan.json", "scale_diagnostics.json",
            ".metric_scale_applied", ".orientation_applied",
        ],
    }

    @staticmethod
    def _wipe_outputs_for_replace(session_dir: Path, output_dir: Path):
        """Replace mode: delete the WHOLE output/ dir before anything runs.

        Per-stage cleanup is not enough. It only ran for stages that were about to
        run, and the resume probes ran first — so a session whose every stage
        probed "complete" (e.g. artifacts from an older architecture) skipped the
        cleanup entirely and Replace silently did nothing.

        Also removes the derived caches OUTSIDE output/ that are pure functions of
        it, or the stale ones get served: merged_cloud.ply, its Potree octree
        (convert_ply_to_potree skips when metadata.json exists), the floor
        transform, the frame-selection files reconstruction regenerates, and the
        BIM/sábana artifacts tied to the old cloud.
        """
        import os as _os
        import shutil as _shutil

        def _rm_tree(target: Path) -> bool:
            """Rename out of the way, THEN delete. rmtree walks a big Potree octree
            for seconds, and during that walk half its files are gone while the rest
            still answer exists() — the viewer's in-flight requests then died with
            FileNotFoundError between the route's exists() check and open(). The
            rename is one atomic syscall: the path is either fully there or fully
            gone, so a racing request gets a clean 404 instead of a 500."""
            if not target.exists():
                return False
            doomed = target.with_name(f".{target.name}.wiping-{_os.getpid()}")
            try:
                target.rename(doomed)
            except OSError:
                doomed = target          # racing / cross-device: delete in place
            _shutil.rmtree(doomed, ignore_errors=True)
            return True

        deleted = []
        # The ledger is the ONE thing a Replace must not take with it: a new
        # reconstruction is geometry epoch 0 again, but the history of every
        # human-directed correction survives it and replay re-keys those runs by
        # frame_global onto the new chain (USER 2026-09-08). Deleting output/
        # wholesale — which is what this function does, and rightly so — used to
        # take corrections.jsonl along, against the rule two comments in this
        # file already stated. Read it out before the wipe, put it back after.
        from correction.epoch import LEDGER_FILE
        ledger_bytes = None
        ledger_path = output_dir / LEDGER_FILE
        try:
            if ledger_path.is_file():
                ledger_bytes = ledger_path.read_bytes()
        except OSError as e:  # noqa: BLE001 — a Replace must not fail over this
            logger.warning(f"[Pipeline] Replace: could not preserve {LEDGER_FILE} ({e})")

        if _rm_tree(output_dir):
            deleted.append("output/")
        output_dir.mkdir(parents=True, exist_ok=True)
        if ledger_bytes is not None:
            try:
                ledger_path.write_bytes(ledger_bytes)
                n_records = ledger_bytes.count(b"\n")
                logger.info(f"[Pipeline] Replace: {LEDGER_FILE} preserved "
                            f"({n_records} record(s)) — append-only, a new "
                            f"reconstruction never erases the history")
            except OSError as e:  # noqa: BLE001
                logger.warning(f"[Pipeline] Replace: could not restore {LEDGER_FILE} ({e})")

        # frame selection + quality: reconstruction rebuilds these
        frames_dir = session_dir / "frames"
        if frames_dir.exists():
            for pattern in PipelineManager.FRAMES_DIR_FILES:
                for f in frames_dir.glob(pattern):
                    f.unlink(missing_ok=True)
                    deleted.append(f"frames/{f.name}")

        # project-level derivatives of output/ (merged cloud, its octree, floor)
        try:
            merged_dir = session_dir.parent.parent.parent / "merged"
            for name in ("merged_cloud.ply", "potree", "floor_transform.npz"):
                target = merged_dir / name
                if target.is_dir():
                    if _rm_tree(target):
                        deleted.append(f"merged/{name}")
                elif target.exists():
                    target.unlink(missing_ok=True)
                    deleted.append(f"merged/{name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Pipeline] Replace: could not clean merged/ ({e})")

        # BIM comparison / sábana: derived from the cloud that just went away
        for base in (session_dir, session_dir.parent.parent.parent / "bim_comparison"):
            if not base.exists():
                continue
            for pattern in PipelineManager.BIM_COMPARISON_FILES:
                for f in base.glob(pattern):
                    if f.is_dir():
                        _shutil.rmtree(f, ignore_errors=True)
                    else:
                        f.unlink(missing_ok=True)
                    deleted.append(f.name)

        logger.info(f"[Pipeline] 🗑️ Replace: wiped {', '.join(deleted) or 'nothing'} "
                    f"— every stage re-runs from scratch")

    @staticmethod
    def _cleanup_stage_outputs(output_dir: Path, stage_id: StageId,
                               session_dir: Path = None):
        """Delete existing output files for a stage AND all downstream dependents.

        Handles files in:
          - output/ dir (stage outputs)
          - frames/ dir (selected_frames.json, frame_quality.json)
          - session root or bim_comparison/ (sábana files)
        """
        import shutil as _shutil

        # Collect all stages to clean: this stage + cascade dependents
        stages_to_clean = [stage_id]
        stages_to_clean.extend(
            PipelineManager.CASCADE_INVALIDATION.get(stage_id, [])
        )

        all_deleted = []

        def _delete_patterns(base_dir: Path, patterns: List[str]):
            for pattern in patterns:
                if "*" in pattern:
                    for f in base_dir.glob(pattern):
                        if f.is_dir():
                            _shutil.rmtree(f, ignore_errors=True)
                        else:
                            f.unlink(missing_ok=True)
                        all_deleted.append(f.name)
                else:
                    target = base_dir / pattern
                    if target.exists():
                        if target.is_dir():
                            _shutil.rmtree(target, ignore_errors=True)
                        else:
                            target.unlink(missing_ok=True)
                        all_deleted.append(pattern)

        # 1) Clean stage output files in output/ dir
        for sid in stages_to_clean:
            patterns = PipelineManager.STAGE_OUTPUT_FILES.get(sid, [])
            # Protect gaus_slam_run/da3_full if nerfstudio_run symlinks to it
            # (nerfstudio resume depends on this data surviving cleanup)
            ns_da3_link = output_dir / "nerfstudio_run" / "da3_full"
            protected_dirs = set()
            if ns_da3_link.is_symlink():
                try:
                    protected_dirs.add(ns_da3_link.resolve().parent.name)
                except Exception:
                    pass
            filtered = [p for p in patterns if p not in protected_dirs]
            _delete_patterns(output_dir, filtered)

        # 2) Clean frames/ dir files when reconstruction is re-run
        if stage_id == StageId.RECONSTRUCTION and session_dir:
            frames_dir = session_dir / "frames"
            if frames_dir.exists():
                _delete_patterns(frames_dir, PipelineManager.FRAMES_DIR_FILES)

        # 3) Clean BIM comparison / sábana artifacts when cloud changes
        if stage_id in (StageId.RECONSTRUCTION, StageId.CLOUDCOMPY):
            # Check output dir (legacy sessions store sábana here)
            _delete_patterns(output_dir, PipelineManager.BIM_COMPARISON_FILES)
            # Check session root (some legacy layouts)
            if session_dir:
                _delete_patterns(session_dir, PipelineManager.BIM_COMPARISON_FILES)
                # Check bim_comparison/ subdir (new project layout)
                bim_dir = session_dir.parent.parent.parent / "bim_comparison"
                if bim_dir.exists():
                    _delete_patterns(bim_dir, PipelineManager.BIM_COMPARISON_FILES)

        if all_deleted:
            cascade_label = ""
            if len(stages_to_clean) > 1:
                cascade_label = f" (+cascade: {', '.join(s.value for s in stages_to_clean[1:])})"
            logger.info(f"[Pipeline] 🗑️ Replace mode{cascade_label}: deleted {', '.join(all_deleted)}")

    @classmethod
    def _destructive_conflict(cls, output_dir: Path, session_dir: Path,
                              stage_id: StageId) -> List[str]:
        """The artifacts re-running `stage_id` would rewrite in place, when a
        FINISHED version of that stage is already on disk. Empty list = there is
        nothing to lose and the stage may run.

        A HALF-finished stage returns nothing on purpose: resuming a crashed
        reconstruction is the whole point of resume mode, and its leftovers are
        not a deliverable. What this protects is the finished one — the 3.5 h
        chain of 2026-09-21 that a replace-less relaunch destroyed in 40 s."""
        names = cls.IN_PLACE_REWRITERS.get(stage_id)
        if not names:
            return []
        done, _why = cls._stage_is_complete(output_dir, session_dir, stage_id)
        if not done:
            return []
        at_risk = [n for n in names if (output_dir / n).exists()]
        if stage_id == StageId.RECONSTRUCTION:
            # the segmentation's keyframe list: map_worker re-selects it, and
            # without it SAM3 segments the raw video instead of the keyframes
            if (session_dir / "frames" / "selected_frames.json").exists():
                at_risk.append("frames/selected_frames.json")
        return at_risk

    async def _run_pipeline(
        self,
        job: PipelineJob,
        session_dir: str,
        config: dict,
        on_progress: Optional[ProgressCallback],
        on_complete: Optional[Callable[[str, bool], Awaitable[None]]],
        replace: bool = False,
    ):
        """Run stages sequentially, each as a subprocess."""
        job.status = JobStatus.RUNNING
        success = True
        output_dir = Path(session_dir) / "output"

        # REPLACE MODE: "start from scratch" — wipe output/ and its derived caches
        # up front, THEN run every stage. The resume probes are skipped entirely:
        # consulting them first is what made Replace a no-op on sessions whose
        # artifacts all probed complete.
        #
        # replace=False DELETES NOTHING. Pre-cleaning is what replace MEANS, by
        # definition, and the flag is documented right above as "resume is the
        # safe default, an omitted flag must never destroy a reconstruction".
        # It did not honour that: until 2026-09-21 the wipe also fired whenever
        # the RECONSTRUCTION stage merely appeared in the list, under the theory
        # that a reconstruction must never reuse a prior artifact. Since the
        # manager enables ALL stages (finding 21), a relaunch-from-the-VLM with
        # replace=False carried the reconstruction along and the wipe took the
        # session's cloud, poses, omega_run/, chunk_plan.json and the
        # segmentation's keyframe list with it — 3.5 h destroyed in 40 s. The
        # clean-slate contract now lives where it belongs: a FINISHED
        # reconstruction is refused a silent re-run (see _destructive_conflict
        # below), so the choice is explicit instead of destructive.
        #
        # A replace that carries an explicit stage SELECTION pre-cleans only the
        # selected stages and their cascade (_cleanup_stage_outputs, below in the
        # loop). Wiping the whole output/ for a partial relaunch would destroy
        # exactly what the caller did not ask to redo — the same shape of loss
        # this fix is about, one flag further along.
        _selective = any(s.stage.explicit for s in job.stages)
        if replace and not _selective:
            self._wipe_outputs_for_replace(Path(session_dir), output_dir)
        elif replace:
            logger.info("[Pipeline] Replace with an explicit stage selection → "
                        "pre-cleaning only "
                        f"{', '.join(s.stage.id.value for s in job.stages if s.stage.explicit)}"
                        " and their cascade; the rest of output/ is untouched")

        # RESUME MODE (no wipe): the pipeline detects on its own which stages
        # this session already completed (artifact + freshness probes) and only
        # runs what is missing/stale. Once any stage actually runs, everything
        # downstream is considered stale (its inputs just changed) and runs too.
        # An EXPLICITLY selected stage (stages= / from_stage=) skips the probe:
        # the caller already made that decision by naming it.
        upstream_ran = replace

        for idx, stage_state in enumerate(job.stages):
            if not stage_state.stage.enabled:
                stage_state.status = JobStatus.DONE
                stage_state.message = "Skipped (disabled)"
                continue

            if job.status == JobStatus.CANCELLED:
                break

            if not upstream_ran and not stage_state.stage.explicit:
                done, why = self._stage_is_complete(
                    output_dir, Path(session_dir), stage_state.stage.id)
                if done:
                    stage_state.status = JobStatus.DONE
                    stage_state.pct = 100
                    stage_state.message = f"Already complete ({why}) — resumed past"
                    logger.info(f"[Pipeline] {stage_state.stage.id.value}: "
                                f"already complete ({why}) — skipping")
                    continue
            upstream_ran = True

            # A stage about to destroy a FINISHED predecessor says so and stops:
            # the flag that authorises the destruction is `replace`, and nobody
            # else gets to imply it. Refusing is the whole finding-23 fix — the
            # run that cost the user his cloud never asked this question.
            if not replace:
                at_risk = self._destructive_conflict(
                    output_dir, Path(session_dir), stage_state.stage.id)
                if at_risk:
                    detail = (
                        f"{stage_state.stage.id.value} is already complete and "
                        f"re-running it rewrites {', '.join(at_risk)} in place. "
                        f"replace=False deletes nothing — resend with "
                        f"replace=true to authorise it, or drop "
                        f"'{stage_state.stage.id.value}' from the stage "
                        f"selection (stages=/from_stage=)")
                    stage_state.status = JobStatus.FAILED
                    stage_state.message = detail
                    logger.error(f"[Pipeline] ✋ refused: {detail}")
                    success = False
                    break

            # Clean up previous outputs if in replace mode (cascade invalidation)
            if replace and output_dir.exists():
                self._cleanup_stage_outputs(
                    output_dir, stage_state.stage.id,
                    session_dir=Path(session_dir)
                )

            job.current_stage_idx = idx
            stage_state.status = JobStatus.RUNNING
            stage_state.pct = 0
            stage_state.message = "Starting..."

            if on_progress:
                await on_progress(job.session_id, job.to_dict())

            try:
                ok = await self._run_stage(job, stage_state, session_dir, config, on_progress, replace)
                if not ok:
                    stage_state.status = JobStatus.FAILED
                    success = False
                    break
                stage_state.status = JobStatus.DONE
            except asyncio.CancelledError:
                stage_state.status = JobStatus.CANCELLED
                job.status = JobStatus.CANCELLED
                success = False
                break
            except Exception as e:
                stage_state.status = JobStatus.FAILED
                stage_state.message = str(e)
                logger.error(f"[Pipeline] Stage {stage_state.stage.id.value} error: {e}")
                success = False
                break

            if on_progress:
                await on_progress(job.session_id, job.to_dict())

        if job.status != JobStatus.CANCELLED:
            job.status = JobStatus.DONE if success else JobStatus.FAILED

        if on_progress:
            await on_progress(job.session_id, job.to_dict())

        if on_complete:
            await on_complete(job.session_id, success)

        logger.info(f"[Pipeline] {job.session_id} finished: {job.status.value}")

    async def _run_stage(
        self,
        job: PipelineJob,
        stage_state: StageState,
        session_dir: str,
        config: dict,
        on_progress: Optional[ProgressCallback],
        replace: bool = False,
    ) -> bool:
        """Run a single stage as a subprocess with Pipe IPC."""

        stage_id = stage_state.stage.id
        reg = STAGE_REGISTRY[stage_id]
        module_name = reg["module"]

        # 3D Reconstruction uses MapAnything worker directly

        # Import the worker module dynamically
        import importlib
        worker_mod = importlib.import_module(module_name)

        # Spawn process (MUST use 'spawn' to avoid CUDA fork errors)
        ctx = get_context('spawn')
        server_conn, worker_conn = ctx.Pipe()
        job._server_conn = server_conn

        # Merge stage-specific config with global config.
        # _pipeline_replace lets workers reuse pre-existing artifacts when the
        # user disabled "Replace existing outputs" (only map_worker reads it).
        merged_config = {**config, **stage_state.stage.config, "_pipeline_replace": replace}

        proc = ctx.Process(
            target=worker_mod.run,
            args=(worker_conn, session_dir, merged_config),
            daemon=True,
        )
        job._process = proc

        t0 = time.time()
        proc.start()
        worker_conn.close()  # Server doesn't write to worker side

        # Poll pipe for messages (non-blocking via asyncio)
        loop = asyncio.get_running_loop()
        done = False
        success = True

        while not done:
            # Non-blocking poll with short timeout
            has_data = await loop.run_in_executor(None, lambda: server_conn.poll(0.25))

            if has_data:
                try:
                    msg = server_conn.recv()
                except (EOFError, OSError):
                    break

                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "progress":
                    stage_state.pct = msg.get("pct", 0)
                    stage_state.message = msg.get("msg", "")
                    if on_progress:
                        await on_progress(job.session_id, job.to_dict())

                elif msg_type == "log":
                    level = msg.get("level", "info")
                    log_msg = f"[{stage_id.value}] {msg.get('msg', '')}"
                    getattr(logger, level, logger.info)(log_msg)
                    # Also print to terminal so logs are visible in start.sh console
                    print(log_msg, flush=True)
                    # Forward log to UI via progress callback
                    stage_state.message = msg.get("msg", "")
                    if on_progress:
                        await on_progress(job.session_id, job.to_dict())

                elif msg_type == "done":
                    done = True
                    success = msg.get("success", False)
                    stage_state.elapsed = msg.get("elapsed", time.time() - t0)
                    if not success:
                        stage_state.message = msg.get("detail", "Failed")

                elif msg_type == "error":
                    error_msg = f"[{stage_id.value}] ❌ {msg.get('msg', 'Unknown error')}"
                    logger.error(error_msg)
                    print(error_msg, flush=True)
                    if msg.get("traceback"):
                        logger.error(msg["traceback"])
                        print(msg["traceback"], flush=True)

            # Check if process died unexpectedly
            if not proc.is_alive() and not done:
                rc = proc.exitcode
                if rc != 0:
                    logger.error(f"[Pipeline] Worker {stage_id.value} died with code {rc}")
                    success = False
                done = True

        # Cleanup — reap the worker AND any children it left behind (a crashed
        # or killed worker must never orphan GPU-holding subprocesses)
        proc.join(timeout=10)
        if proc.is_alive():
            self._kill_stage_tree(proc, timeout=5)
        else:
            # worker exited: sweep stragglers of its process group. The worker
            # setsid()'d at startup, so its pgid == its pid — valid for killpg
            # even after the leader died, as long as children survive. No such
            # group (the normal case) raises ProcessLookupError and we move on.
            import os as _os
            import signal as _signal
            try:
                _os.killpg(proc.pid, _signal.SIGKILL)
                logger.warning(f"[Pipeline] {stage_id.value}: swept leftover "
                               f"children of exited worker (pgid={proc.pid})")
            except (ProcessLookupError, PermissionError):
                pass  # nothing left — the normal case

        try:
            server_conn.close()
        except Exception:
            pass

        job._process = None
        job._server_conn = None

        stage_state.elapsed = time.time() - t0

        return success


    # ── resume-mode probes ────────────────────────────────────
    @staticmethod
    def _stage_is_complete(output_dir: Path, session_dir: Path,
                           stage_id: StageId) -> tuple:
        """(complete, reason) — does this session already have the stage's
        outputs, fresh w.r.t. its inputs? Drives automatic resume: the user
        never picks stages, the pipeline continues from wherever the session
        actually is. Probes are ARTIFACT-based (survive restarts/crashes)."""
        def mt(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        if stage_id == StageId.RECONSTRUCTION:
            # NEVER chunk_*.ply. CloudCompy DELETES them when it finishes
            # ("removed 21 redundant chunk files, baked into cleaned_cloud",
            # workers/cloudcompy_worker.py), so a probe that asks for them says
            # "never ran" about every session that ever reached the cleaned
            # cloud — and then redoes the whole 3.5 h. The evidence that the
            # reconstruction ran is what SURVIVES the later stages: the poses,
            # the run directory that produced them, the chunk plan, the depth.
            poses = [output_dir / d / "camera_poses.txt"
                     for d in ("omega_run", "da3_run", "maplong_run", ".")]
            has_poses = any(p.exists() for p in poses)
            has_depth = any((output_dir / d).is_dir()
                            for d in ("omega_run/results_output",
                                      "da3_run/results_output", "results_output",
                                      "_tmp_results_aligned"))
            has_run = any((output_dir / d).is_dir()
                          for d in ("omega_run", "da3_run", "maplong_run"))
            # chunk_plan.json is persisted by map_worker on every chunked run and
            # nothing downstream removes it; it is the correction's own record of
            # the real chunks, so its presence means this session reconstructed.
            has_plan = (output_dir / "chunk_plan.json").exists()
            has_cloud = (output_dir / "cleaned_cloud.ply").exists()
            if has_poses and (has_depth or has_cloud or has_plan or has_run):
                why = ", ".join(w for w, ok in (
                    ("poses", has_poses), ("depth", has_depth),
                    ("chunk_plan", has_plan), ("run dir", has_run),
                    ("cleaned cloud", has_cloud)) if ok)
                return True, f"{why} on disk"
            if has_poses:
                return False, "poses on disk but no depth, chunk plan or run dir"
            return False, "no reconstruction artifacts"

        if stage_id == StageId.CLOUDCOMPY:
            cloud = output_dir / "cleaned_cloud.ply"
            if not cloud.exists():
                return False, "no cleaned cloud"
            # this stage also owns the deferred mask→cloud mapping
            seg = output_dir / "segmentation.json"
            res = output_dir / "segmentation_result.json"
            if seg.exists() and (not res.exists() or mt(res) < mt(seg)):
                return False, "mask→cloud mapping pending"
            return True, "cleaned_cloud.ply on disk"

        if stage_id == StageId.VLM:
            if (output_dir / "vlm_analysis.json").exists():
                return True, "vlm_analysis.json on disk"
            # a session segmented MANUALLY (Segmentation Manager) already
            # fulfilled this stage's purpose — never overwrite human work
            # with an auto-prompted re-segmentation
            if (output_dir / "segmentation.json").exists():
                return True, "segmentation exists (manual) — auto-prompt not needed"
            return False, "no VLM analysis"

        if stage_id == StageId.SAM3:
            seg = output_dir / "segmentation.json"
            masks = output_dir / "seg_masks.npz"
            if not (seg.exists() and masks.exists()):
                return False, "no segmentation"
            vlm = output_dir / "vlm_analysis.json"
            if vlm.exists() and mt(seg) < mt(vlm):
                return False, "segmentation older than VLM analysis"
            # partial-run guard: an interrupted SAM3 leaves an incremental
            # segmentation.json with only SOME of the VLM's categories — that
            # is not "complete", re-run the stage
            if vlm.exists():
                try:
                    import json as _json
                    want = {c.strip() for c in _json.load(open(vlm))
                            .get("prompt", "").split(";") if c.strip()}
                    have = set(_json.load(open(seg)).get("prompts", []))
                    if want and not want <= have:
                        return False, (f"segmentation partial "
                                       f"({len(have & want)}/{len(want)} categories)")
                except Exception:
                    pass
            return True, "segmentation.json + masks on disk"

        if stage_id == StageId.PGSR:
            # active only under backend vggtomega_pgsr — for every other backend the
            # worker no-ops, so "complete" here is simply "nothing pending" unless
            # the render products are expected and missing.
            from config import load_config as _lc
            try:
                _backend = str((_lc().get("reconstruction", {}) or {})
                               .get("backend", "")).lower()
            except Exception:
                _backend = ""
            if _backend != "vggtomega_pgsr":
                return True, "backend is not vggtomega_pgsr — stage not applicable"
            render_dir = output_dir / "pgsr_render"
            if (render_dir / "report.json").exists() and \
                    any(render_dir.glob("frame_*.npz")):
                return True, "pgsr_render depths + report on disk"
            return False, "no PGSR rendered depths"

        if stage_id == StageId.TSDF:
            scene_dir = output_dir / "tsdf" / "scene"
            meshes = list(scene_dir.glob("scene.*")) if scene_dir.is_dir() else []
            if not meshes:
                return False, "no scene TSDF mesh"
            return True, "scene mesh on disk"

        if stage_id == StageId.CERTIFY:
            acta = output_dir / "certify_acta.json"
            if not acta.exists():
                return False, "no certification acta"
            cloud = output_dir / "cleaned_cloud.ply"
            # the acta certifies ONE cloud + ONE segmentation: an older acta
            # than either of them is stale (the cloud may already be a later
            # epoch the acta itself produced — its mtime is then older than
            # the acta's, which is what this compares)
            for src in (output_dir / "segmentation_result.json", output_dir / "segmentation.json"):
                if src.exists() and mt(acta) < mt(src):
                    return False, f"acta older than {src.name}"
            if cloud.exists() and mt(acta) < mt(cloud) and not (output_dir / "geometry_epoch.json").exists():
                return False, "acta older than cleaned_cloud.ply"
            return True, "certification acta on disk"

        return False, "no probe for stage"


# ── Helpers ──────────────────────────────────────────────────

def _parse_stage(name) -> StageId:
    """One stage name → StageId. An unknown name is a PipelineSelectionError
    that lists what IS valid; it is never dropped, never coerced."""
    if isinstance(name, StageId):
        return name
    key = str(name).strip().lower()
    for sid in StageId:
        if sid.value == key:
            return sid
    raise PipelineSelectionError(
        f"unknown stage '{name}' — valid stages are "
        f"{', '.join(s.value for s in DEFAULT_STAGE_ORDER)}")


def build_pipeline_stages(backend: Optional[str] = None,
                          stages: Optional[List] = None,
                          from_stage: Optional[str] = None) -> List[PipelineStage]:
    """Build the pipeline stage list.

    With no selection this is THE pipeline — the whole DEFAULT_STAGE_ORDER,
    gated only by the config switches (auto_segment / auto_tsdf /
    auto_after_segmentation) and by the backend. That is the historical
    behaviour and it is unchanged: `build_pipeline_stages(backend)` returns
    exactly what it returned before.

    With a selection it runs what the caller NAMED. There was no way to ask for
    that until 2026-09-21 (USER: "quiero que relances desde segmentacion con
    todas sus etapas, ademas de la certificacion... es decir desde cloudcompy
    para abajo sin incluir claro a cloudcompy") — the only options were the full
    chain or nothing, which is how the reconstruction ended up in a relaunch
    that only wanted the semantic stages, and how it then destroyed them
    (finding 23). A named stage also RUNS: the resume probe is not consulted for
    it (PipelineStage.explicit), because naming it IS the decision.

    Args:
        backend: Reconstruction backend name (e.g., "da3", "gaus_slam").
                 GauS-SLAM backends skip CloudCompPy (Gaussian surfels are
                 already clean — the stage has nothing to consume).
        stages: Explicit stage ids/names to run, in any order — the list is
                re-sorted into pipeline order. Mutually exclusive with
                from_stage.
        from_stage: Run this stage and everything after it in pipeline order.

    Raises:
        PipelineSelectionError: unknown name, both selectors given, a stage this
        backend does not have, or a selection that comes out empty.
    """
    _gaus_backends = ("gaus_slam", "gaus_slam_lidar", "gaus_slam_da3", "gaus_slam_hybrid")
    skip_cloudcompy = backend in _gaus_backends
    if skip_cloudcompy:
        logger.info(f"[Pipeline] CloudCompPy skipped (backend={backend})")

    # `pipeline.auto_segment: false` disables the automatic semantic chain
    # (VLM auto-prompt → SAM3) and restores the on-demand flow.
    auto_segment = True
    try:
        from config import cfg
        auto_segment = bool(cfg.get("pipeline", {}).get("auto_segment", True))
    except Exception:
        pass
    _semantic_stages = {StageId.VLM, StageId.SAM3}
    if not auto_segment:
        logger.info("[Pipeline] auto_segment off — VLM/SAM3 stages disabled")

    # `pipeline.auto_tsdf: false` stops the pipeline at the cleaned cloud — the
    # TSDF/texrecon takes hours and the user wants the cloud on screen first;
    # the mesh is then triggered manually.
    auto_tsdf = True
    try:
        from config import cfg as _c2
        auto_tsdf = bool(_c2.get("pipeline", {}).get("auto_tsdf", True))
    except Exception:
        pass
    if not auto_tsdf:
        logger.info("[Pipeline] auto_tsdf off — pipeline ends at the cleaned cloud")

    # `certify.auto_after_segmentation` is the ONE switch of the automatic
    # certification, and it has to gate BOTH of its triggers: the CERTIFY stage
    # here and the re-run when the Segmentation Manager closes (certify.api
    # auto_run). Without this the stage ran whatever the switch said.
    auto_certify = True
    try:
        from config import cfg as _c3
        auto_certify = bool(_c3.get("certify", {}).get("auto_after_segmentation", True))
    except Exception:
        pass
    if not auto_certify:
        logger.info("[Pipeline] auto_after_segmentation off — CERTIFY stage disabled")

    def _auto_enabled(stage_id: StageId) -> bool:
        """What the AUTOMATIC chain runs — the config switches' verdict."""
        if skip_cloudcompy and stage_id == StageId.CLOUDCOMPY:
            return False
        if not auto_segment and stage_id in _semantic_stages:
            return False
        if stage_id == StageId.CERTIFY and not (auto_certify and auto_segment):
            return False   # config.yaml documents auto_segment:false as "no
                           # segmentation, no instance loops, no certification" 
        if not auto_tsdf and stage_id in (StageId.TSDF, StageId.PGSR):
            return False   # no mesh requested → the 2h PGSR stage has no consumer
        return True

    # The PGSR stage exists only for backend vggtomega_pgsr; for every other
    # backend it is dropped from the list entirely (user 2026-08-18: the UI
    # must not show a "Precision (PGSR)" step the pipeline will never run).
    order = [s for s in DEFAULT_STAGE_ORDER
             if s != StageId.PGSR or backend == "vggtomega_pgsr"]

    # ── explicit selection ────────────────────────────────────
    selection: Optional[List[StageId]] = None
    if stages is not None and from_stage is not None:
        raise PipelineSelectionError(
            "stages= and from_stage= are mutually exclusive — one names a set, "
            "the other names a starting point; pick one")
    if from_stage is not None:
        first = _parse_stage(from_stage)
        if first not in order:
            raise PipelineSelectionError(
                f"stage '{first.value}' is not part of this pipeline "
                f"(backend={backend}) — it runs "
                f"{', '.join(s.value for s in order)}")
        # A RANGE, not a naming: "from the VLM down" means the stages the
        # automatic chain would have run from there, not every stage that
        # exists. Without this filter `from_stage="vlm"` answered
        # vlm, sam3, certify, TSDF on a session whose config says
        # `pipeline.auto_tsdf: false` — "the pipeline ends at the cleaned
        # cloud, the mesh is on demand" (USER 2026-08-28) — and a relaunch
        # meant to redo the segmentation would have ended in the two-hour
        # mesh nobody asked for. Naming a stage in `stages=` is different:
        # that IS asking for it, switch or no switch.
        selection = [s for s in order[order.index(first):] if _auto_enabled(s)]
        if not selection:
            raise PipelineSelectionError(
                f"from_stage '{first.value}': every stage from there on is "
                f"disabled by the config switches (auto_segment / auto_tsdf / "
                f"certify.auto_after_segmentation) — name them explicitly with "
                f"stages= if you mean to override that")
    elif stages is not None:
        wanted = [_parse_stage(s) for s in stages]
        if not wanted:
            raise PipelineSelectionError(
                "empty stage selection — omit stages= to run the whole chain")
        for w in wanted:
            if w not in order:
                raise PipelineSelectionError(
                    f"stage '{w.value}' is not part of this pipeline "
                    f"(backend={backend}) — it runs "
                    f"{', '.join(s.value for s in order)}")
        wanted_set = set(wanted)
        selection = [s for s in order if s in wanted_set]   # pipeline order wins

    if selection is None:
        chosen = [s for s in order if _auto_enabled(s)]
    else:
        # A structurally absent stage was already refused above. What remains is
        # the config switches, and against a stage the caller named BY NAME they
        # do not get a vote — they configure the automatic chain, not a manual
        # relaunch. The override is logged, never silent.
        overridden = [s.value for s in selection if not _auto_enabled(s)]
        if overridden:
            logger.info(f"[Pipeline] explicit selection runs {', '.join(overridden)} "
                        f"although the config switches them off — named by the caller")
        chosen = list(selection)
        if not chosen:
            raise PipelineSelectionError(
                "the stage selection resolved to no runnable stage")
        logger.info(f"[Pipeline] explicit stage selection: "
                    f"{', '.join(s.value for s in chosen)} "
                    f"(the resume probe is not consulted for these)")

    chosen_set = set(chosen)
    return [PipelineStage(id=stage_id, enabled=stage_id in chosen_set,
                          explicit=selection is not None and stage_id in chosen_set)
            for stage_id in order]
