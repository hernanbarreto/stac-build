"""map_worker._find_stray_dir takes a scan's Stray data from the scan's own directory or its
stray/ subdirectory only — never from a sibling scan (docs/plan_determinismo.md point 35)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.map_worker import _find_stray_dir                  # noqa: E402


def _stray(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "odometry.csv").write_text("timestamp,frame,x,y,z,qx,qy,qz,qw\n")
    (d / "depth").mkdir(exist_ok=True)


def test_only_the_scans_own_places_count(tmp_path):
    sess = tmp_path / "scan" / "src_default"
    sess.mkdir(parents=True)
    sib = tmp_path / "scan" / "src_other"
    _stray(sib)
    _stray(sib / "stray")
    assert _find_stray_dir(sess) is None                           # never the sibling
    _stray(sess / "stray")
    assert _find_stray_dir(sess) == sess / "stray"
    _stray(sess)
    assert _find_stray_dir(sess) == sess                           # the scan's own dir first
    src = (Path(__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    body = src[src.index("def _find_stray_dir"):src.index("def _run_lidar_only")]
    assert "iterdir" not in body and "parent" not in body and "sibling" in body
