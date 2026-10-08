"""Undo the fusion judgement: the raw SAM3 masklets are the store again.

The fusion is a JUDGEMENT — the matcher deciding which masks are one object — and a
judgement has to be auditable and reversible. Since docs/plan_determinismo.md point 100
(2026-10-08) the raw SAM3 store (segmentation.json + seg_masks.npz) is IMMUTABLE and the
fusion lives apart, in ``fusion_map.json`` tied to the sha256 of the raw store; every reader
(projection, certification, census) applies it over the raw masks. Undoing it is therefore
deleting that map — nothing is copied back, nothing was ever rewritten.

    python -m segmentation.unfuse <output_dir>

Restoring does NOT re-run the matching: the next matching pass starts from the masklets again
(the mask-space cache is cleared).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List

from segmentation.fuse_parent import MAP, PARENT


def generations(output_dir) -> List[Path]:
    """The raw store is never archived any more (point 100): there are no generations."""
    return []


def unfuse(output_dir, gen=None, log: Callable[[str], None] = print) -> dict:
    """Delete the fusion map of the session. Returns what it did."""
    output_dir = Path(output_dir)
    mp = output_dir / MAP
    if not mp.exists():
        raise FileNotFoundError(f"{mp} does not exist — this session holds no fusion to undo")
    n_objects = _count(output_dir / PARENT)
    mp.unlink()
    from segmentation import mask_space
    mask_space.invalidate(output_dir)
    log(f"deleted {mp.name}: the store is its {n_objects} raw masklet(s) again. The matching has "
        f"NOT been re-run — segmentation_result.json still describes the fused state until it does.")
    return {"deleted": str(mp), "masklets": n_objects}


def _count(p: Path) -> int:
    import json
    try:
        return len(json.loads(p.read_text()).get("instances") or [])
    except Exception:  # noqa: BLE001
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir")
    a = ap.parse_args(argv)
    unfuse(a.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
