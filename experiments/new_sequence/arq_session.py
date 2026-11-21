"""A test session for the architecture of 2026-11-21: a NEW project holding one scan — a copy of an existing scan's
frames, frames manifest and video, and its AMB3R pass over every frame (the same video: the same pass).

    python arq_session.py <source scan dir> <new project name>

Prints the new scan's directory. Nothing of the source scan is changed; the copies are real copies (no links)."""
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECTS = Path("/workspace/stac-build/server/projects")


def main() -> int:
    src, name = Path(sys.argv[1]).resolve(), sys.argv[2]
    date, source_dir = src.parent.name, src.name                  # scans/<date>/src_<source>
    if not source_dir.startswith("src_") or src.parent.parent.name != "scans":
        raise SystemExit(f"{src} is not a scan directory (…/scans/<date>/src_<source>)")
    source = source_dir[len("src_"):]
    project = PROJECTS / name
    if project.exists():
        raise SystemExit(f"{project} exists — nothing is overwritten")
    dst = project / "scans" / date / source_dir
    (dst / "frames").mkdir(parents=True)
    (dst / "intake").mkdir()
    (dst / "output").mkdir()
    frames = sorted(p for p in (src / "frames").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda p: shutil.copy2(p, dst / "frames" / p.name), frames))
    shutil.copy2(src / "frames" / "manifest.json", dst / "frames" / "manifest.json")
    videos = [p for p in src.iterdir() if p.is_file() and p.stem == "source_video"]
    for v in videos:
        shutil.copy2(v, dst / v.name)
    for f in ("amb3r_video.npz", "amb3r_video.stamp.json"):
        shutil.copy2(src / "intake" / f, dst / "intake" / f)
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    key = f"{date}/{source}"
    (project / "project.json").write_text(json.dumps({
        "name": name, "slug": name, "location": "", "created_at": now, "ifcs": [], "scan_days": [],
        "scans": [{"key": key, "label": f"scan {date}", "added_at": now, "date": date, "source": source}],
        "composition": {"reference": key, "transforms": {}}, "active_scan": key}, indent=1))
    print(f"{dst}  ({len(frames)} frames, {len(videos)} video file(s), AMB3R's pass)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
