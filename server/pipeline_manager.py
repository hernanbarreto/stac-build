# STAC-Builder: Pipeline Manager
# Orchestrates reconstruction stages as independent subprocesses.
# Each stage runs in its own multiprocessing.Process with Pipe IPC.
# The server process never loads GPU models — it only spawns workers.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import asyncio
import json
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
    TSDF = "tsdf"
    VLM = "vlm"
    SAM3 = "sam3"
    INSTANCE_CLEANER = "instance_cleaner"


STAGE_REGISTRY = {
    StageId.RECONSTRUCTION:   {"label": "3D Reconstruction", "icon": "🔨", "module": "workers.map_worker"},
    StageId.CLOUDCOMPY:       {"label": "Cloud Cleaning",    "icon": "🧹", "module": "workers.cloudcompy_worker"},
    StageId.TSDF:             {"label": "TSDF Mesh",         "icon": "🧊", "module": "workers.tsdf_worker"},
    StageId.VLM:              {"label": "Scene Analysis",    "icon": "🔍", "module": "workers.vlm_worker"},
    StageId.SAM3:             {"label": "Segmentation",      "icon": "🏷️", "module": "workers.sam3_worker"},
    StageId.INSTANCE_CLEANER: {"label": "Instance Cleaning", "icon": "✨", "module": "workers.instance_cleaner_worker"},
}

# THE reconstruction pipeline — always runs end-to-end, no client selection.
#   Reconstruction (SIMPLE: sparse frames → ONE Omega pass → metric scale →
#                   upright orientation baked into cloud + poses)
#   → VLM   (Qwen3-VL understands the scene → RICH concept phrases, no boxes)
#   → SAM3  (one concept session per phrase — SAM3's tracking IS the identity;
#            instance store scene_r.db rebuilt from the clean instances)
#   → CloudCompy (merge/clean → cleaned cloud + Potree + mask→cloud mapping)
#   → scene TSDF (fusion + texrecon photo texture).
# The former Phase R (semantic anchoring) was REMOVED 2026-07-09: the one-pass
# reconstruction has no window seams to anchor, and the anchoring never beat
# its own A/B gate. Its shared primitives (instance store, geometry) live on
# under phase_r/ as the data layer for phases 2-6.
DEFAULT_STAGE_ORDER: List[StageId] = [
    StageId.RECONSTRUCTION,
    StageId.VLM,
    StageId.SAM3,
    StageId.CLOUDCOMPY,
    StageId.TSDF,
]


@dataclass
class PipelineStage:
    id: StageId
    enabled: bool = True
    config: dict = field(default_factory=dict)

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
    _process: Optional[Process] = field(default=None, repr=False)
    _server_conn: Optional[Connection] = field(default=None, repr=False)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "current_stage_idx": self.current_stage_idx,
            "stages": [
                {
                    "id": ss.stage.id.value,
                    "label": ss.stage.label,
                    "icon": ss.stage.icon,
                    "enabled": ss.stage.enabled,
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
                     an omitted flag must never destroy a reconstruction.
            scan_key: Optional "date/source" key (e.g. "2026-03-07/legacy") to target
                      a specific scan. If None, resolves to latest scan/first source.
        """
        # Cancel existing job for this session if any
        if session_id in self._jobs:
            await self.cancel_pipeline(session_id)

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
            else:
                # Legacy: scan_key is ignored, use normal resolution
                ctx = resolve_session(server_dir, session_id)
                session_dir = str(ctx.session_dir)
        else:
            ctx = resolve_session(server_dir, session_id)
            session_dir = str(ctx.session_dir)

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
                      ".metric_scale_applied", ".orientation_applied"],
                      # ↑ idempotency markers: MUST be cleared on Replace, else scale_align /
                      # orient skip on every re-run → cloud not metric, scene upside down.
        StageId.CLOUDCOMPY: ["cleaned_cloud.ply", "cleaned_cloud_raw.ply",
                             "floor_transform.npz", "scene_consolidate_report.json",
                             # the deferred mask→cloud mapping runs IN this stage
                             # → its products are THIS stage's outputs, not SAM3's
                             "segmentation_result.json", "seg_broadcast.json",
                             "classification.npy", "corrected_cloud.ply"],
        StageId.TSDF: ["tsdf/scene"],
        StageId.VLM: ["scene_analysis.json", "vlm_analysis.json",
                      "scene_understanding.json", "autoprompt_instances.json",
                      "autoprompt_review_queue.json"],
        StageId.SAM3: ["segmentation.json", "segmentation_result.json",
                       "seg_masks.npz", "seg_broadcast.json", "scene_r.db"],
        StageId.INSTANCE_CLEANER: ["instance_*.ply", "inst_cleaned_cloud.ply"],
    }

    # Files in frames/ dir that should be regenerated on reconstruction
    FRAMES_DIR_FILES: List[str] = [
        "selected_frames.json", "selected_frames_seg*.json", "frame_quality.json",
        "da3_frames.json",   # dense-set list (legacy da3/mapanything backends)
    ]

    # BIM comparison / sábana artifacts (may live in output/, session root, or bim_comparison/)
    BIM_COMPARISON_FILES: List[str] = [
        "sabana.npz", "sabana_cloud.ply", "sabana_meta.json", "sabana_potree",
    ]

    # Cascade: when a stage re-runs, these DOWNSTREAM stages' outputs are ALSO
    # invalidated. MUST mirror DEFAULT_STAGE_ORDER:
    #   RECONSTRUCTION → VLM → SAM3 → CLOUDCOMPY → TSDF
    # A cascade edge pointing UPSTREAM deletes freshly produced artifacts
    # mid-pipeline — that exact bug (CLOUDCOMPY → SAM3, a relic of the old
    # cloudcompy-before-sam3 order) silently erased segmentation.json +
    # seg_masks.npz right after SAM3 wrote them (test2, 2026-07-08).
    CASCADE_INVALIDATION: Dict[StageId, List[StageId]] = {
        StageId.RECONSTRUCTION: [
            StageId.VLM,              # scene analysis ran on old keyframes
            StageId.SAM3,             # segmentation ran on old frames
            StageId.CLOUDCOMPY,       # cleaned_cloud depends on chunks
            StageId.TSDF,             # TSDF mesh integrated old depth/poses
            StageId.INSTANCE_CLEANER, # instance PLYs from old segmentation
        ],
        StageId.VLM: [
            StageId.SAM3,             # SAM3 uses VLM categories
            StageId.INSTANCE_CLEANER,
        ],
        StageId.SAM3: [
            StageId.INSTANCE_CLEANER, # instances depend on segmentation
        ],
        StageId.CLOUDCOMPY: [
            StageId.TSDF,             # TSDF masks to the old cleaned_cloud
            StageId.INSTANCE_CLEANER,
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
        if _rm_tree(output_dir):
            deleted.append("output/")
        output_dir.mkdir(parents=True, exist_ok=True)

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
        # RECONSTRUCTION always starts from scratch: whenever the reconstruction
        # stage is part of this run, output/ is wiped UNCONDITIONALLY — a
        # reconstruction must never reuse any prior artifact (resume subtleties
        # inside the stages caused silent partial reuse; the only safe contract
        # is a clean slate). Stage-only runs (TSDF / segmentation) still resume
        # on the existing outputs.
        recon_requested = any(
            s.stage.enabled and s.stage.id == StageId.RECONSTRUCTION
            for s in job.stages)
        if replace or recon_requested:
            if recon_requested and not replace:
                logger.info("[Pipeline] reconstruction stage requested → output/ "
                            "wiped unconditionally (a reconstruction never reuses "
                            "prior artifacts)")
            self._wipe_outputs_for_replace(Path(session_dir), output_dir)

        # RESUME MODE (no wipe): the pipeline detects on its own which stages
        # this session already completed (artifact + freshness probes) and only
        # runs what is missing/stale — the user never selects stages. Once any
        # stage actually runs, everything downstream is considered stale (its
        # inputs just changed) and runs too.
        upstream_ran = replace or recon_requested

        for idx, stage_state in enumerate(job.stages):
            if not stage_state.stage.enabled:
                stage_state.status = JobStatus.DONE
                stage_state.message = "Skipped (disabled)"
                continue

            if job.status == JobStatus.CANCELLED:
                break

            if not upstream_ran:
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
            poses = [output_dir / d / "camera_poses.txt"
                     for d in ("omega_run", "da3_run", "maplong_run", ".")]
            has_poses = any(p.exists() for p in poses)
            has_depth = any((output_dir / d).is_dir()
                            for d in ("omega_run/results_output",
                                      "da3_run/results_output", "results_output",
                                      "_tmp_results_aligned"))
            if has_poses and (has_depth or (output_dir / "cleaned_cloud.ply").exists()):
                return True, "poses + depth on disk"
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

        if stage_id == StageId.TSDF:
            scene_dir = output_dir / "tsdf" / "scene"
            meshes = list(scene_dir.glob("scene.*")) if scene_dir.is_dir() else []
            if not meshes:
                return False, "no scene TSDF mesh"
            return True, "scene mesh on disk"

        return False, "no probe for stage"


# ── Helpers ──────────────────────────────────────────────────

def build_pipeline_stages(backend: Optional[str] = None) -> List[PipelineStage]:
    """Build THE pipeline stage list — always the full DEFAULT_STAGE_ORDER
    (reconstruction → cloudcompy → tsdf), end to end. There is deliberately no
    per-stage client selection: a reconstruction is only usable once the cloud
    is cleaned (Potree) and the scene TSDF exists, so partial runs just left
    sessions in in-between states that hung the on-load lazy paths.

    Args:
        backend: Reconstruction backend name (e.g., "da3", "gaus_slam").
                 GauS-SLAM backends skip CloudCompPy (Gaussian surfels are
                 already clean — the stage has nothing to consume).
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

    def _enabled(stage_id: StageId) -> bool:
        if skip_cloudcompy and stage_id == StageId.CLOUDCOMPY:
            return False
        if not auto_segment and stage_id in _semantic_stages:
            return False
        if not auto_tsdf and stage_id == StageId.TSDF:
            return False
        return True

    return [PipelineStage(id=stage_id, enabled=_enabled(stage_id))
            for stage_id in DEFAULT_STAGE_ORDER]
