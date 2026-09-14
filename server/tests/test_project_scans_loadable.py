"""USER 2026-09-13: opening a project must load the scan that IS reconstructed
when the active one has no cloud (its reconstruction was wiped or never ran)."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _project(tmp_path, scans):
    root = tmp_path / "projects"
    pdir = root / "p1"
    pdir.mkdir(parents=True)
    (pdir / "project.json").write_text(json.dumps({
        "name": "p1", "slug": "p1", "scans": [{"key": k, "label": k, "date": k.split("/")[0],
                                               "source": k.split("/")[1]} for k in scans],
        "composition": {"reference": scans[0], "transforms": {}}}))
    for k in scans:
        d, s = k.split("/")
        (pdir / "scans" / d / f"src_{s}" / "output").mkdir(parents=True)
        (pdir / "scans" / d / f"src_{s}" / "frames").mkdir(parents=True)
    from project_paths import ProjectPaths
    return ProjectPaths(str(root), "p1")


def test_loadable_scan_prefers_the_requested_scan_when_it_has_a_cloud(tmp_path):
    from project_scans import loadable_scan, scan_has_cloud
    pp = _project(tmp_path, ["2026-08-24/default", "2026-08-31/default"])
    (pp.for_source("2026-08-24", "default").output_dir / "cleaned_cloud.ply").write_bytes(b"ply")
    assert scan_has_cloud(pp, "2026-08-24/default")
    assert loadable_scan(pp, "2026-08-24/default") == "2026-08-24/default"


def test_loadable_scan_falls_back_to_the_newest_reconstructed_scan(tmp_path):
    from project_scans import loadable_scan
    pp = _project(tmp_path, ["2026-08-24/default", "2026-08-31/default", "2026-09-05/default"])
    old = pp.for_source("2026-08-24", "default").output_dir / "cleaned_cloud.ply"
    new = pp.for_source("2026-08-31", "default").output_dir / "cleaned_cloud.ply"
    old.write_bytes(b"ply")
    time.sleep(0.02)
    new.write_bytes(b"ply")
    # the active scan (09-05) lost its reconstruction → the newest cloud opens
    assert loadable_scan(pp, "2026-09-05/default") == "2026-08-31/default"
    # an older cloud touched later is "newest" by mtime
    os.utime(old)
    assert loadable_scan(pp, "2026-09-05/default") == "2026-08-24/default"
    assert loadable_scan(pp, None) == "2026-08-24/default"


def test_loadable_scan_is_none_when_nothing_is_reconstructed(tmp_path):
    from project_scans import loadable_scan
    pp = _project(tmp_path, ["2026-08-24/default", "2026-08-31/default"])
    assert loadable_scan(pp, "2026-08-31/default") is None
