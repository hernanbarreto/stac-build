"""The viewer's LIGHT copy of segmentation_result.json — every instance WITHOUT its per-point index
list (globalIndices): the viewer never reads it (the octree carries every point's instance).
zaragoza 2026-10-09: the whole 934 MB file, parsed and sent over the viewer socket, held the server
~40 s, the socket died and the session never showed.

The copy is keyed by the result's inode, size and mtime and rebuilt when it changes. It lives
OUTSIDE the session (``server/.cache/segments_view/``): a viewer GET never writes into a session's
output/ (docs/plan_determinismo.md point 111). A copy an earlier version left beside the result
(output/segmentation_view.json) is still READ when it names the same result. Shared by the server
(main.py) and the processes that rewrite the result (segmentation.label_overrides), so a rewrite
leaves the light copy ready instead of making the next viewer read parse the whole file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

LEGACY_NAME = "segmentation_view.json"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "segments_view"


def cache_path(res: Path) -> Path:
    return CACHE_DIR / (hashlib.sha256(str(Path(res).resolve()).encode("utf-8")).hexdigest()[:32]
                        + ".json")


def _put(cache: Path, key: str, data) -> None:
    try:
        from atomic_io import atomic_write_json
        cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache, {"_source": key, "data": data})
    except Exception as e:  # noqa: BLE001 — the light copy is a cache, never a failure
        print(f"[Segments] light copy not cached ({e}) — sent without it")


def segments_view(res: Path) -> Optional[dict]:
    res = Path(res)
    st = res.stat()
    key = f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
    legacy_key = f"{st.st_size}:{st.st_mtime_ns}"
    cache = cache_path(res)
    for side, want in ((cache, key), (res.with_name(LEGACY_NAME), legacy_key)):
        if not side.exists():
            continue
        try:
            doc = json.loads(side.read_text())
            if isinstance(doc, dict) and doc.get("_source") == want:
                data = doc.get("data")
                if side != cache:
                    _put(cache, key, data)
                return data
        except (OSError, ValueError):
            pass
    try:
        data = json.loads(res.read_text())
    except (OSError, ValueError) as e:
        print(f"[Segments] {res} unreadable ({e}) — nothing is sent; the projection stage "
              f"rewrites it")
        return None
    if not isinstance(data, dict):
        return None
    for inst in data.get("instances") or []:
        if isinstance(inst, dict):
            inst.pop("globalIndices", None)
    _put(cache, key, data)
    return data
