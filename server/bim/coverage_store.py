"""
STAC Build — Coverage Store
============================

Cumulative per-element coverage tracking with occlusion awareness.

Each BIM element's surface is sampled at fixed points. Across multiple scans,
coverage can only INCREASE — once a sample is covered, it stays covered.

Storage: one .npz per element in session_dir/coverage_history/
Timeline: coverage_timeline.json tracks scan-by-scan snapshots.

Authors: Hernán Barreto — Ingerop IN3
"""
import json
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class SampleStatus(IntEnum):
    """Status of a BIM surface sample point."""
    NOT_VISIBLE = 0     # No camera had line-of-sight
    NOT_BUILT   = 1     # Camera could see, but no scan points nearby
    COVERED     = 2     # Scan points confirm construction at this location
    OCCLUDED    = 3     # Something blocks the view between camera and BIM


class OccluderType(IntEnum):
    """Classification of what is causing occlusion."""
    UNKNOWN    = 0
    TEMPORARY  = 1   # Debris, scaffold, tools → needs re-scan
    PERMANENT  = 2   # Installed fixture (cabinet, MEP) → freeze coverage


class ElementState(IntEnum):
    """Workflow state for a BIM element."""
    NOT_STARTED  = 0   # 0% coverage
    IN_PROGRESS  = 1   # 0% < coverage < completion threshold
    COMPLETED    = 2   # coverage >= threshold AND quality >= quality_threshold
    VERIFIED     = 3   # Inspector manually marked
    OCCLUDED_FROZEN = 4  # Permanent occlusion, coverage frozen at last value


@dataclass
class ScanCoverageResult:
    """Result of analyzing one scan against one BIM element."""
    scan_id: str
    date: str
    status: np.ndarray          # (M,) SampleStatus per sample
    deviation: np.ndarray       # (M,) C2M distance (NaN if not covered)
    occluder_labels: np.ndarray  # (M,) object dtype, SAM3 label or ""
    occluder_types: np.ndarray  # (M,) OccluderType per sample


@dataclass
class ElementCoverage:
    """Cumulative coverage state for one BIM element."""
    element_key: str
    surface_samples: np.ndarray   # (M, 3) fixed BIM surface points
    surface_normals: np.ndarray   # (M, 3) face normals at each sample
    best_covered: np.ndarray      # (M,) bool — ever covered?
    best_deviation: np.ndarray    # (M,) float — best C2M (NaN if never covered)
    last_status: np.ndarray       # (M,) SampleStatus from most recent scan
    occluder_labels: np.ndarray   # (M,) object dtype — SAM3 label if occluded
    occluder_types: np.ndarray    # (M,) OccluderType
    element_state: int = ElementState.NOT_STARTED
    scan_count: int = 0

    @property
    def n_samples(self) -> int:
        return len(self.surface_samples)

    @property
    def coverage_cumulative(self) -> float:
        """% of surface ever covered (monotonically increasing)."""
        if self.n_samples == 0:
            return 0.0
        return round(float(np.sum(self.best_covered) / self.n_samples * 100), 1)

    @property
    def coverage_current(self) -> float:
        """% of surface covered in the last scan."""
        if self.n_samples == 0:
            return 0.0
        covered = np.sum(self.last_status == SampleStatus.COVERED)
        return round(float(covered / self.n_samples * 100), 1)

    @property
    def occluded_pct(self) -> float:
        """% of surface currently occluded."""
        if self.n_samples == 0:
            return 0.0
        occluded = np.sum(self.last_status == SampleStatus.OCCLUDED)
        return round(float(occluded / self.n_samples * 100), 1)

    @property
    def quality_pct(self) -> float:
        """% of covered samples within tolerance (uses best deviation)."""
        covered_mask = self.best_covered
        if np.sum(covered_mask) == 0:
            return 0.0
        devs = self.best_deviation[covered_mask]
        valid = devs[~np.isnan(devs)]
        if len(valid) == 0:
            return 0.0
        # Default 20mm tolerance — overridden at call sites
        within = np.sum(np.abs(valid) <= 0.020)
        return round(float(within / len(valid) * 100), 1)


# ═══════════════════════════════════════════════════════════════════
#  BIM MESH SURFACE SAMPLING
# ═══════════════════════════════════════════════════════════════════

def sample_mesh_surface(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    density: float = 4.0,
    max_samples: int = 50000,
    seed: int = 42,
) -> tuple:
    """
    Sample points uniformly on a triangle mesh surface.
    
    Args:
        mesh_verts: (V, 3) vertex positions
        mesh_faces: (F, 3) triangle indices
        density: samples per m²
        max_samples: cap total samples for performance
        seed: random seed for reproducibility
    
    Returns:
        (samples, normals) where samples is (M, 3) and normals is (M, 3)
    """
    rng = np.random.RandomState(seed)

    v0 = mesh_verts[mesh_faces[:, 0]]
    v1 = mesh_verts[mesh_faces[:, 1]]
    v2 = mesh_verts[mesh_faces[:, 2]]

    # Face normals (un-normalized for area, then normalize)
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) * 0.5
    total_area = np.sum(areas)

    if total_area < 1e-10:
        return np.empty((0, 3)), np.empty((0, 3))

    # Normalize face normals
    norms = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

    all_samples = []
    all_normals = []

    for i in range(len(mesh_faces)):
        n = max(1, int(areas[i] * density))
        # Random barycentric coordinates for uniform sampling
        r1 = np.sqrt(rng.rand(n))
        r2 = rng.rand(n)
        pts = ((1 - r1)[:, None] * v0[i] +
               (r1 * (1 - r2))[:, None] * v1[i] +
               (r1 * r2)[:, None] * v2[i])
        all_samples.append(pts)
        all_normals.append(np.tile(norms[i], (n, 1)))

    samples = np.vstack(all_samples)
    normals = np.vstack(all_normals)

    # Cap at max_samples
    if len(samples) > max_samples:
        idx = rng.choice(len(samples), max_samples, replace=False)
        samples = samples[idx]
        normals = normals[idx]

    return samples.astype(np.float32), normals.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  COVERAGE STORE
# ═══════════════════════════════════════════════════════════════════

class CoverageStore:
    """
    Persistent cumulative coverage store for a session/project.
    
    Stores per-element coverage in:
        session_dir/coverage_history/element_{key}_coverage.npz
    
    Tracks timeline in:
        session_dir/coverage_history/coverage_timeline.json
    """

    def __init__(self, session_dir: str):
        self.dir = Path(session_dir) / "coverage_history"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._timeline_path = self.dir / "coverage_timeline.json"

    # ── Element I/O ──────────────────────────────────────────────

    def _element_path(self, element_key: str) -> Path:
        safe_key = element_key.replace("/", "_").replace("\\", "_")
        return self.dir / f"element_{safe_key}_coverage.npz"

    def has_element(self, element_key: str) -> bool:
        return self._element_path(element_key).exists()

    def load_element(self, element_key: str) -> Optional[ElementCoverage]:
        """Load persisted coverage for an element, or None if not found."""
        path = self._element_path(element_key)
        if not path.exists():
            return None

        data = np.load(str(path), allow_pickle=True)
        return ElementCoverage(
            element_key=element_key,
            surface_samples=data["surface_samples"],
            surface_normals=data["surface_normals"],
            best_covered=data["best_covered"],
            best_deviation=data["best_deviation"],
            last_status=data["last_status"],
            occluder_labels=data["occluder_labels"],
            occluder_types=data["occluder_types"],
            element_state=int(data["element_state"]),
            scan_count=int(data["scan_count"]),
        )

    def save_element(self, ec: ElementCoverage):
        """Persist coverage data for an element."""
        path = self._element_path(ec.element_key)
        np.savez_compressed(
            str(path),
            surface_samples=ec.surface_samples,
            surface_normals=ec.surface_normals,
            best_covered=ec.best_covered,
            best_deviation=ec.best_deviation,
            last_status=ec.last_status,
            occluder_labels=ec.occluder_labels,
            occluder_types=ec.occluder_types,
            element_state=np.array(ec.element_state),
            scan_count=np.array(ec.scan_count),
        )

    # ── Initialize element ───────────────────────────────────────

    def init_element(
        self,
        element_key: str,
        mesh_verts: np.ndarray,
        mesh_faces: np.ndarray,
        density: float = 4.0,
    ) -> ElementCoverage:
        """
        Create initial coverage record for a BIM element.
        Samples the mesh surface and initializes all arrays to empty/zero.
        
        If the element already exists, returns the existing data
        (samples are fixed — same mesh = same samples due to fixed seed).
        """
        existing = self.load_element(element_key)
        if existing is not None:
            return existing

        samples, normals = sample_mesh_surface(
            mesh_verts, mesh_faces, density=density
        )
        M = len(samples)

        ec = ElementCoverage(
            element_key=element_key,
            surface_samples=samples,
            surface_normals=normals,
            best_covered=np.zeros(M, dtype=bool),
            best_deviation=np.full(M, np.nan, dtype=np.float32),
            last_status=np.zeros(M, dtype=np.int8),
            occluder_labels=np.array([""] * M, dtype=object),
            occluder_types=np.zeros(M, dtype=np.int8),
            element_state=ElementState.NOT_STARTED,
            scan_count=0,
        )
        self.save_element(ec)
        return ec

    # ── Update with new scan ─────────────────────────────────────

    def update_element(
        self,
        element_key: str,
        scan_result: ScanCoverageResult,
        completion_threshold: float = 80.0,
        quality_threshold: float = 80.0,
    ) -> ElementCoverage:
        """
        Merge a new scan's coverage into the cumulative store.
        
        Rules:
        - best_covered = old | new  (coverage only grows)
        - best_deviation = min(old, new) for covered samples
        - last_status = new scan's status
        - element_state transitions based on cumulative metrics
        """
        ec = self.load_element(element_key)
        if ec is None:
            raise ValueError(f"Element {element_key} not initialized")

        # ── Merge coverage (OR — never lose coverage) ──
        new_covered = scan_result.status == SampleStatus.COVERED
        ec.best_covered = ec.best_covered | new_covered

        # ── Merge deviation (keep best, i.e., smallest absolute) ──
        for i in range(ec.n_samples):
            if new_covered[i]:
                new_dev = scan_result.deviation[i]
                if np.isnan(ec.best_deviation[i]):
                    ec.best_deviation[i] = new_dev
                elif not np.isnan(new_dev):
                    if abs(new_dev) < abs(ec.best_deviation[i]):
                        ec.best_deviation[i] = new_dev

        # ── Update current status ──
        ec.last_status = scan_result.status.astype(np.int8)
        ec.occluder_labels = scan_result.occluder_labels
        ec.occluder_types = scan_result.occluder_types.astype(np.int8)
        ec.scan_count += 1

        # ── State machine ──
        ec.element_state = self._compute_state(
            ec, completion_threshold, quality_threshold
        )

        self.save_element(ec)
        return ec

    # ── State machine ────────────────────────────────────────────

    @staticmethod
    def _compute_state(
        ec: ElementCoverage,
        completion_threshold: float,
        quality_threshold: float,
    ) -> int:
        """
        Derive element state from cumulative metrics.
        
        NOT_STARTED → IN_PROGRESS → COMPLETED → VERIFIED
                                        ↓
                                OCCLUDED_FROZEN
        """
        # Don't downgrade from VERIFIED
        if ec.element_state == ElementState.VERIFIED:
            return ElementState.VERIFIED

        cov = ec.coverage_cumulative

        if cov == 0.0:
            return ElementState.NOT_STARTED

        # Check for permanent occlusion (>50% of surface permanently occluded)
        perm_occluded = np.sum(ec.occluder_types == OccluderType.PERMANENT)
        if ec.n_samples > 0 and (perm_occluded / ec.n_samples * 100) > 50:
            return ElementState.OCCLUDED_FROZEN

        if cov >= completion_threshold:
            # Check quality too
            qual = ec.quality_pct
            if qual >= quality_threshold:
                return ElementState.COMPLETED
            else:
                return ElementState.IN_PROGRESS

        return ElementState.IN_PROGRESS

    # ── Timeline ─────────────────────────────────────────────────

    def append_timeline(
        self,
        scan_id: str,
        element_key: str,
        ec: ElementCoverage,
    ):
        """Add a timeline entry for this scan + element combination."""
        timeline = self._load_timeline()
        timeline.append({
            "scan_id": scan_id,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "element_key": element_key,
            "coverage_current": ec.coverage_current,
            "coverage_cumulative": ec.coverage_cumulative,
            "occluded_pct": ec.occluded_pct,
            "quality_pct": ec.quality_pct,
            "element_state": ElementState(ec.element_state).name,
            "scan_count": ec.scan_count,
        })
        self._save_timeline(timeline)

    def _load_timeline(self) -> list:
        if self._timeline_path.exists():
            return json.loads(self._timeline_path.read_text())
        return []

    def _save_timeline(self, timeline: list):
        self._timeline_path.write_text(
            json.dumps(timeline, indent=2, ensure_ascii=False)
        )

    # ── Summary / reporting ──────────────────────────────────────

    def get_all_elements(self) -> List[str]:
        """List all element keys that have coverage data."""
        keys = []
        for f in self.dir.glob("element_*_coverage.npz"):
            name = f.stem  # element_{key}_coverage
            key = name.replace("element_", "").replace("_coverage", "")
            keys.append(key)
        return sorted(keys)

    def get_summary(self) -> Dict[str, dict]:
        """Get cumulative coverage summary for all elements."""
        summary = {}
        for key in self.get_all_elements():
            ec = self.load_element(key)
            if ec is None:
                continue
            summary[key] = {
                "coverage_cumulative": ec.coverage_cumulative,
                "coverage_current": ec.coverage_current,
                "occluded_pct": ec.occluded_pct,
                "quality_pct": ec.quality_pct,
                "element_state": ElementState(ec.element_state).name,
                "scan_count": ec.scan_count,
                "n_samples": ec.n_samples,
            }
        return summary
