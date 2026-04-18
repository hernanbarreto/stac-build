"""
STAC Build — Project Paths
============================

Centralized path resolution for the hierarchical project structure.
Every module imports paths from here instead of building them manually.

Structure:
    projects/
      <project-slug>/
        project.json
        ifcs/
        scans/
          <YYYY-MM-DD>/
            scan_day.json
            src_<name>/
              source.json
              frames/
              output/
        merged/
          merged_cloud.ply
          potree/
          floor_transform.npz
        coverage_history/
        segmentation/
        bim_comparison/

Authors: Hernán Barreto — Ingerop IN3
"""

from pathlib import Path
from typing import Optional, List
import json
import time


# ═══════════════════════════════════════════════════════════════════
#  PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════

class ProjectPaths:
    """
    Resolves all filesystem paths for a STAC project.
    
    Usage:
        paths = ProjectPaths("/data/projects", "torre-norte")
        paths.source_output("2026-02-15", "planta1_juan")
        # → /data/projects/torre-norte/scans/2026-02-15/src_planta1_juan/output
    """

    def __init__(self, projects_root: str, project_slug: str):
        self.root = Path(projects_root)
        self.slug = project_slug
        self._project = self.root / project_slug

    # ── Project level ────────────────────────────────────────────

    @property
    def project_dir(self) -> Path:
        return self._project

    @property
    def project_json(self) -> Path:
        return self._project / "project.json"

    @property
    def ifcs_dir(self) -> Path:
        return self._project / "ifcs"

    # ── Scan day level ───────────────────────────────────────────

    @property
    def scans_dir(self) -> Path:
        return self._project / "scans"

    def scan_day_dir(self, date: str) -> Path:
        """e.g. date='2026-02-15'"""
        return self.scans_dir / date

    def scan_day_json(self, date: str) -> Path:
        return self.scan_day_dir(date) / "scan_day.json"

    def list_scan_days(self) -> List[str]:
        """Return sorted list of scan day dates."""
        if not self.scans_dir.exists():
            return []
        return sorted([
            d.name for d in self.scans_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    # ── Source level ─────────────────────────────────────────────

    def source_dir(self, date: str, source_name: str) -> Path:
        return self.scan_day_dir(date) / f"src_{source_name}"

    def source_json(self, date: str, source_name: str) -> Path:
        return self.source_dir(date, source_name) / "source.json"

    def source_frames(self, date: str, source_name: str) -> Path:
        return self.source_dir(date, source_name) / "frames"

    def source_output(self, date: str, source_name: str) -> Path:
        return self.source_dir(date, source_name) / "output"

    def list_sources(self, date: str) -> List[str]:
        """Return list of source names for a scan day."""
        day_dir = self.scan_day_dir(date)
        if not day_dir.exists():
            return []
        return sorted([
            d.name.replace("src_", "", 1)
            for d in day_dir.iterdir()
            if d.is_dir() and d.name.startswith("src_")
        ])

    # ── Merged / accumulated ─────────────────────────────────────

    @property
    def merged_dir(self) -> Path:
        return self._project / "merged"

    @property
    def merged_cloud(self) -> Path:
        return self.merged_dir / "merged_cloud.ply"

    @property
    def merged_potree(self) -> Path:
        return self.merged_dir / "potree"

    @property
    def floor_transform(self) -> Path:
        return self.merged_dir / "floor_transform.npz"

    # ── Coverage ─────────────────────────────────────────────────

    @property
    def coverage_dir(self) -> Path:
        return self._project / "coverage_history"

    @property
    def coverage_timeline(self) -> Path:
        return self.coverage_dir / "coverage_timeline.json"

    @property
    def coverage_snapshots_dir(self) -> Path:
        return self.coverage_dir / "snapshots"

    def coverage_snapshot(self, date: str) -> Path:
        return self.coverage_snapshots_dir / f"{date}.json"

    # ── Segmentation ─────────────────────────────────────────────

    @property
    def segmentation_dir(self) -> Path:
        return self._project / "segmentation"

    # ── BIM comparison ───────────────────────────────────────────

    @property
    def bim_comparison_dir(self) -> Path:
        return self._project / "bim_comparison"

    # ── Convenience: "current source" context ────────────────────

    def for_source(self, date: str, source_name: str) -> "SourceContext":
        """
        Return a convenience object pre-bound to a specific source.
        Useful when a worker processes a single source at a time.
        """
        return SourceContext(self, date, source_name)

    # ── Init project dirs ────────────────────────────────────────

    def ensure_dirs(self):
        """Create all top-level directories for a project."""
        for d in [
            self.project_dir,
            self.ifcs_dir,
            self.scans_dir,
            self.merged_dir,
            self.merged_potree,
            self.coverage_dir,
            self.coverage_snapshots_dir,
            self.segmentation_dir,
            self.bim_comparison_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def ensure_source_dirs(self, date: str, source_name: str):
        """Create directories for a specific source."""
        for d in [
            self.source_frames(date, source_name),
            self.source_output(date, source_name),
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Project metadata I/O ────────────────────────────────────

    def load_project_meta(self) -> dict:
        if self.project_json.exists():
            return json.loads(self.project_json.read_text())
        return {}

    def save_project_meta(self, meta: dict):
        self.project_json.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )

    def init_project_meta(self, name: str, location: str = ""):
        """Initialize project.json if it doesn't exist."""
        if self.project_json.exists():
            return
        self.save_project_meta({
            "name": name,
            "slug": self.slug,
            "location": location,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ifcs": [],
            "scan_days": [],
        })


# ═══════════════════════════════════════════════════════════════════
#  SOURCE CONTEXT (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════

class SourceContext:
    """
    Pre-bound to a specific (project, date, source) for workers
    that process a single source at a time.
    
    Usage:
        ctx = paths.for_source("2026-02-15", "planta1_juan")
        ctx.frames_dir   # → .../src_planta1_juan/frames
        ctx.output_dir   # → .../src_planta1_juan/output
        ctx.merged_dir   # → .../merged  (project-level)
    """

    def __init__(self, paths: ProjectPaths, date: str, source_name: str):
        self.paths = paths
        self.date = date
        self.source_name = source_name

    # Source-specific
    @property
    def source_dir(self) -> Path:
        return self.paths.source_dir(self.date, self.source_name)

    @property
    def frames_dir(self) -> Path:
        return self.paths.source_frames(self.date, self.source_name)

    @property
    def output_dir(self) -> Path:
        return self.paths.source_output(self.date, self.source_name)

    # Project-level (pass-through)
    @property
    def project_dir(self) -> Path:
        return self.paths.project_dir

    @property
    def merged_dir(self) -> Path:
        return self.paths.merged_dir

    @property
    def merged_cloud(self) -> Path:
        return self.paths.merged_cloud

    @property
    def merged_potree(self) -> Path:
        return self.paths.merged_potree

    @property
    def floor_transform(self) -> Path:
        return self.paths.floor_transform

    @property
    def coverage_dir(self) -> Path:
        return self.paths.coverage_dir

    @property
    def segmentation_dir(self) -> Path:
        # Segmentation data lives per-scan inside output/, not at project level
        return self.output_dir

    @property
    def bim_comparison_dir(self) -> Path:
        return self.paths.bim_comparison_dir

    @property
    def ifcs_dir(self) -> Path:
        return self.paths.ifcs_dir

    # ── Legacy compatibility ─────────────────────────────────────
    # These properties let old code that uses `session_dir` patterns
    # work with minimal changes during migration.

    @property
    def session_dir(self) -> Path:
        """
        Legacy compat: maps to source dir (parent of output/ and frames/).
        Workers do Path(session_dir) / "output" and / "frames".
        """
        return self.source_dir

    @property
    def scans_dir(self) -> Path:
        """Legacy compat: for frame_storage that uses scans_dir."""
        return self.paths.scans_dir


# ═══════════════════════════════════════════════════════════════════
#  LEGACY BRIDGE
# ═══════════════════════════════════════════════════════════════════

class LegacySourceContext:
    """
    Wraps the OLD flat session layout to expose the same interface
    as SourceContext. Used during migration so old sessions keep working.
    
    Old layout:  scans/{session_id}/frames/
                 scans/{session_id}/output/
                 scans/{session_id}/*.ifc
    
    Maps to:
        output_dir  → scans/{session_id}/output
        frames_dir  → scans/{session_id}/frames
        merged_dir  → scans/{session_id}/output  (same, no merge concept)
        ifcs_dir    → scans/{session_id}          (IFCs lived at root)
        etc.
    """

    def __init__(self, session_root: Path):
        self._root = session_root

    @property
    def source_dir(self) -> Path:
        return self._root

    @property
    def frames_dir(self) -> Path:
        return self._root / "frames"

    @property
    def output_dir(self) -> Path:
        return self._root / "output"

    @property
    def project_dir(self) -> Path:
        return self._root

    @property
    def merged_dir(self) -> Path:
        return self._root / "output"

    @property
    def merged_cloud(self) -> Path:
        return self._root / "output" / "cleaned_cloud.ply"

    @property
    def merged_potree(self) -> Path:
        return self._root / "output" / "potree"

    @property
    def floor_transform(self) -> Path:
        return self._root / "output" / "floor_transform.npz"

    @property
    def coverage_dir(self) -> Path:
        return self._root / "coverage_history"

    @property
    def segmentation_dir(self) -> Path:
        return self._root / "output"

    @property
    def bim_comparison_dir(self) -> Path:
        return self._root

    @property
    def ifcs_dir(self) -> Path:
        return self._root

    # Legacy aliases
    @property
    def session_dir(self) -> Path:
        return self._root

    @property
    def scans_dir(self) -> Path:
        return self._root.parent


def resolve_session(server_dir: str, session_id: str) -> "SourceContext | LegacySourceContext":
    """
    Factory: resolve a session_id to the correct path context.
    
    1. Check if it's a new-style project in projects/ dir
    2. Fall back to legacy flat layout in scans/ dir
    
    Usage:
        ctx = resolve_session("/path/to/server", "pccr-10022026")
        ctx.output_dir    # works regardless of old/new layout
        ctx.frames_dir
        ctx.merged_potree
    """
    server = Path(server_dir)
    
    # ── New-style project? ──
    from config import PROJECTS_DIR
    projects_dir = PROJECTS_DIR
    project_json = projects_dir / session_id / "project.json"
    if project_json.exists():
        paths = ProjectPaths(str(projects_dir), session_id)
        # Find the latest scan day + first source
        scan_days = paths.list_scan_days()
        if scan_days:
            latest = scan_days[-1]
            sources = paths.list_sources(latest)
            source = sources[0] if sources else "default"
            return paths.for_source(latest, source)
        else:
            return paths.for_source("default", "default")
    
    # ── Legacy flat session ──
    legacy_dir = server / "scans" / session_id
    if legacy_dir.exists():
        return LegacySourceContext(legacy_dir)
    
    # ── Not found — return new-style (will be created) ──
    paths = ProjectPaths(str(projects_dir), session_id)
    return paths.for_source(
        time.strftime("%Y-%m-%d"), "default"
    )

