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
    VLM = "vlm"
    SAM3 = "sam3"
    INSTANCE_CLEANER = "instance_cleaner"


STAGE_REGISTRY = {
    StageId.RECONSTRUCTION:   {"label": "3D Reconstruction", "icon": "🔨", "module": "workers.map_worker"},
    StageId.CLOUDCOMPY:       {"label": "Cloud Cleaning",    "icon": "🧹", "module": "workers.cloudcompy_worker"},
    StageId.VLM:              {"label": "Scene Analysis",    "icon": "🔍", "module": "workers.vlm_worker"},
    StageId.SAM3:             {"label": "Segmentation",      "icon": "🏷️", "module": "workers.sam3_worker"},
    StageId.INSTANCE_CLEANER: {"label": "Instance Cleaning", "icon": "✨", "module": "workers.instance_cleaner_worker"},
}

DEFAULT_STAGE_ORDER: List[StageId] = [
    StageId.RECONSTRUCTION,
    StageId.CLOUDCOMPY,
    StageId.VLM,
    StageId.SAM3,
    StageId.INSTANCE_CLEANER,
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
        replace: bool = True,
        scan_key: Optional[str] = None,
    ) -> PipelineJob:
        """Start a pipeline for the given session.
        
        Args:
            session_id: Session directory name (e.g. "2026-01-31-22-23-52")
            stages: Ordered list of stages to run (only enabled ones execute)
            config: Server config dict (from config.yaml)
            on_progress: Async callback(session_id, job_dict) for progress updates
            on_complete: Async callback(session_id, success) when pipeline finishes
            replace: If True, delete existing outputs before running each stage
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

        # Terminate process if still running
        if job._process and job._process.is_alive():
            job._process.terminate()
            job._process.join(timeout=5)
            if job._process.is_alive():
                job._process.kill()

        # Cancel the asyncio task
        if job._task and not job._task.done():
            job._task.cancel()

        job.status = JobStatus.CANCELLED
        if 0 <= job.current_stage_idx < len(job.stages):
            job.stages[job.current_stage_idx].status = JobStatus.CANCELLED

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
                      "gaus_slam_run",
                      "vggt_long_config.yaml", "da3_streaming_config.yaml",
                      "camera_poses_mapanything.json",
                      "intrinsic.txt", "lidar_complement.ply"],
        StageId.CLOUDCOMPY: ["cleaned_cloud.ply", "floor_transform.npz"],
        StageId.VLM: ["scene_analysis.json", "vlm_analysis.json"],
        StageId.SAM3: ["segmentation.json", "segmentation_result.json",
                       "seg_masks.npz", "seg_broadcast.json"],
        StageId.INSTANCE_CLEANER: ["instance_*.ply", "inst_cleaned_cloud.ply"],
    }

    # Files in frames/ dir that should be regenerated on reconstruction
    FRAMES_DIR_FILES: List[str] = [
        "selected_frames.json", "selected_frames_seg*.json", "frame_quality.json",
    ]

    # BIM comparison / sábana artifacts (may live in output/, session root, or bim_comparison/)
    BIM_COMPARISON_FILES: List[str] = [
        "sabana.npz", "sabana_cloud.ply", "sabana_meta.json", "sabana_potree",
    ]

    # Cascade: when a stage re-runs, these downstream stages' outputs are ALSO invalidated
    # Example: Reconstruction → new chunks → old cleaned_cloud is invalid → old sábana is invalid
    CASCADE_INVALIDATION: Dict[StageId, List[StageId]] = {
        StageId.RECONSTRUCTION: [
            StageId.CLOUDCOMPY,       # cleaned_cloud depends on chunks
            StageId.VLM,              # scene analysis ran on old keyframes
            StageId.SAM3,             # segmentation ran on old cloud
            StageId.INSTANCE_CLEANER, # instance PLYs from old segmentation
        ],
        StageId.CLOUDCOMPY: [
            StageId.SAM3,             # segmentation used old cleaned_cloud
            StageId.INSTANCE_CLEANER,
        ],
        StageId.VLM: [
            StageId.SAM3,             # SAM3 uses VLM categories
            StageId.INSTANCE_CLEANER,
        ],
        StageId.SAM3: [
            StageId.INSTANCE_CLEANER, # instances depend on segmentation
        ],
    }

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
        replace: bool = True,
    ):
        """Run stages sequentially, each as a subprocess."""
        job.status = JobStatus.RUNNING
        success = True
        output_dir = Path(session_dir) / "output"

        for idx, stage_state in enumerate(job.stages):
            if not stage_state.stage.enabled:
                stage_state.status = JobStatus.DONE
                stage_state.message = "Skipped (disabled)"
                continue

            if job.status == JobStatus.CANCELLED:
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
        replace: bool = True,
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

        # Cleanup
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)

        try:
            server_conn.close()
        except Exception:
            pass

        job._process = None
        job._server_conn = None

        stage_state.elapsed = time.time() - t0

        return success


# ── Helpers ──────────────────────────────────────────────────

def build_pipeline_stages(
    ordered_stages: Optional[List[str]] = None,
    enabled: Optional[Dict[str, bool]] = None,
    backend: Optional[str] = None,
) -> List[PipelineStage]:
    """Build the pipeline stage list.
    
    Args:
        ordered_stages: Optional list of stage IDs (e.g., ["vlm", "sam3", "reconstruction", "cloudcompy"]).
                        If not provided, defaults to DEFAULT_STAGE_ORDER.
        enabled: Optional dict like {"reconstruction": True, "vlm": False, ...}
                 to override which stages are actually executed.
        backend: Reconstruction backend name (e.g., "da3", "gaus_slam").
                 Used to auto-disable stages incompatible with certain backends.
    """
    if ordered_stages is None:
        stage_order = DEFAULT_STAGE_ORDER
    else:
        # Convert strings back to Enums, ignoring invalid ones
        stage_order = []
        for s in ordered_stages:
            try:
                stage_order.append(StageId(s))
            except ValueError:
                logger.warning(f"Unknown stage: {s}")

    enabled_dict = dict(enabled) if enabled else {}

    # Auto-disable stages based on reconstruction backend
    _gaus_backends = ("gaus_slam", "gaus_slam_lidar", "gaus_slam_da3", "gaus_slam_hybrid")
    if backend in _gaus_backends:
        # GauS-SLAM produces clean Gaussian surfels — CloudCompPy cleaning is unnecessary
        enabled_dict.setdefault("cloudcompy", False)
        logger.info(f"[Pipeline] Auto-disabled CloudCompPy (backend={backend})")
    stages = []
    
    for stage_id in stage_order:
        is_enabled = enabled_dict.get(stage_id.value, True)  # all enabled by default
        stages.append(PipelineStage(id=stage_id, enabled=is_enabled))
        
    return stages
