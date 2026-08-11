"""
VIO Trajectory Detector
================================================================================
Detects an OPTIONAL visual-inertial-odometry (ARCore/ARKit) trajectory recorded
alongside the session video. Any modern phone (with or without LiDAR) can
produce one; the capture app exports it next to the video. When present, VIO
becomes the metric scale SOURCE (short-horizon VIO scale is excellent even
though its pose drifts) and DA3 degrades to a cross-check — see
reconstruction/vio_scale.py and docs/VIO_FORMAT.md.

Accepted locations inside the session dir (src_*/ — same level as
source_video.*), first match wins:
    vio_trajectory.csv
    vio_trajectory.json
    vio/trajectory.csv
    vio/trajectory.json

Same auto-detection philosophy as ingestors/stray_detector.py: pure filesystem
probe, no parsing here (parsing + validation is vio_scale.load_vio_trajectory,
which FAILS HARD on malformed data — a present-but-broken VIO file must abort,
never silently fall back).
"""
from pathlib import Path


_CANDIDATES = (
    "vio_trajectory.csv",
    "vio_trajectory.json",
    "vio/trajectory.csv",
    "vio/trajectory.json",
)


def detect_vio_data(session_dir) -> dict:
    """Probe session_dir for a VIO trajectory file.

    Returns dict with keys:
        has_vio (bool):      True if a trajectory file is present
        vio_path (Path|None): the file found (first match in _CANDIDATES order)
        format (str|None):   "csv" | "json"
    """
    session = Path(session_dir)
    for rel in _CANDIDATES:
        p = session / rel
        if p.is_file():
            return {"has_vio": True, "vio_path": p, "format": p.suffix.lstrip(".").lower()}
    return {"has_vio": False, "vio_path": None, "format": None}
