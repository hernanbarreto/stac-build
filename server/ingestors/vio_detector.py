"""
VIO Trajectory Detector
================================================================================
Detects an OPTIONAL visual-inertial-odometry (ARCore/ARKit) trajectory recorded
alongside the session video. Any modern phone (with or without LiDAR) can
produce one; the capture app exports it next to the video. When present, VIO
becomes the metric scale SOURCE (short-horizon VIO scale is excellent even
though its pose drifts) and DA3 degrades to a cross-check — see
reconstruction/vio_scale.py and docs/VIO_FORMAT.md.

Accepted locations, first match wins (ingestors.capture_inputs.vio_candidates):
the scan's ``inputs/`` directory FIRST — the capture data the replace wipe never
touches (docs/plan_determinismo.md point 73, DECIDIDO 2026-10-07) — then, legacy,
the scan directory itself (src_*/ — same level as source_video.*); never a
sibling scan:
    inputs/vio_trajectory.csv      vio_trajectory.csv
    inputs/vio_trajectory.json     vio_trajectory.json
    inputs/vio/trajectory.csv      vio/trajectory.csv
    inputs/vio/trajectory.json     vio/trajectory.json

Same auto-detection philosophy as ingestors/stray_detector.py: pure filesystem
probe, no parsing here (parsing + validation is vio_scale.load_vio_trajectory,
which FAILS HARD on malformed data — a present-but-broken VIO file must abort,
never silently fall back).
"""
from pathlib import Path

from ingestors.capture_inputs import find_vio_file


def detect_vio_data(session_dir) -> dict:
    """Probe the scan for a VIO trajectory file (``inputs/`` first, then the legacy names).

    Returns dict with keys:
        has_vio (bool):      True if a trajectory file is present
        vio_path (Path|None): the file found (first match in reading order)
        format (str|None):   "csv" | "json"
    """
    p = find_vio_file(Path(session_dir))
    if p is not None:
        return {"has_vio": True, "vio_path": p, "format": p.suffix.lstrip(".").lower()}
    return {"has_vio": False, "vio_path": None, "format": None}
