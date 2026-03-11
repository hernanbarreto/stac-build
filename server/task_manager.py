"""
STAC Build — Centralized Task Manager
In-memory tracking for all long-running server tasks.
No files, no DB — tasks die with server restart (correct behavior).
"""

import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class TaskInfo:
    task_id: str
    session_id: str
    task_type: str        # "pipeline", "propagation", "auto_seg", "dbscan_refresh", "instance_clean"
    label: str            # human-readable: "Running Pipeline", "Propagating sofa..."
    status: str = "running"    # "running", "done", "failed"
    pct: int = 0               # 0-100
    detail: str = ""           # current step description
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round(time.time() - self.started_at, 1)
        return d


class TaskManager:
    """Singleton task tracker. Thread-safe via lock."""

    _instance: Optional["TaskManager"] = None
    STALE_TIMEOUT = 4 * 60 * 60  # 4 hours — DA3 CPU inference can take 1h+ per segment

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tasks: Dict[str, TaskInfo] = {}
            cls._instance._lock = threading.Lock()
        return cls._instance

    def start(self, session_id: str, task_type: str, label: str) -> str:
        """Register a new running task. Returns task_id."""
        task_id = str(uuid.uuid4())[:8]
        info = TaskInfo(
            task_id=task_id,
            session_id=session_id,
            task_type=task_type,
            label=label,
        )
        with self._lock:
            self._cleanup_stale()
            self._tasks[task_id] = info
        print(f"[TaskManager] ▶ Started {task_type} [{task_id}] for {session_id}: {label}")
        return task_id

    def update(self, task_id: str, pct: Optional[int] = None, detail: Optional[str] = None):
        """Update progress of a running task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if pct is not None:
                task.pct = pct
            if detail is not None:
                task.detail = detail
            task.updated_at = time.time()

    def finish(self, task_id: str):
        """Mark task as done and remove it."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                print(f"[TaskManager] ✅ Finished {task.task_type} [{task_id}] "
                      f"({time.time() - task.started_at:.1f}s)")

    def fail(self, task_id: str, error: str = ""):
        """Mark task as failed and remove it."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                print(f"[TaskManager] ❌ Failed {task.task_type} [{task_id}]: {error}")

    def get_active(self, session_id: Optional[str] = None) -> List[dict]:
        """Get all running tasks, optionally filtered by session_id."""
        with self._lock:
            self._cleanup_stale()
            tasks = list(self._tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return [t.to_dict() for t in tasks if t.status == "running"]

    def _cleanup_stale(self):
        """Remove tasks not updated in STALE_TIMEOUT seconds. Called under lock."""
        now = time.time()
        stale = [tid for tid, t in self._tasks.items()
                 if now - t.updated_at > self.STALE_TIMEOUT]
        for tid in stale:
            t = self._tasks.pop(tid)
            print(f"[TaskManager] ⏰ Auto-expired stale task {t.task_type} [{tid}] "
                  f"(no update for {self.STALE_TIMEOUT/60:.0f}min)")


# Module-level singleton
task_manager = TaskManager()
