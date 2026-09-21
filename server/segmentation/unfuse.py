"""Put the raw SAM3 masklets back.

The fusion is a JUDGEMENT — the matcher deciding which masks are one object —
and a judgement has to be auditable and reversible. `fuse_parent.apply_fusion`
copies the pre-fusion pair into `output/_sam3_raw/gen_<NNN>/` before it rewrites
anything; this restores it.

    python -m segmentation.unfuse <output_dir> [--gen 0] [--list]

`gen_000` is the true raw SAM3 output and is the default. Restoring does NOT
re-run the matching: it puts the parent and the mask store back and clears the
mask-space cache, so the next matching pass starts from the masklets again.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable, List, Optional

from segmentation.fuse_parent import ARCHIVE, MAP, MASKS, PARENT


def generations(output_dir) -> List[Path]:
    base = Path(output_dir) / ARCHIVE
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir()
                  if d.is_dir() and d.name.startswith("gen_"))


def unfuse(output_dir, gen: Optional[int] = None,
           log: Callable[[str], None] = print) -> dict:
    """Restore the parent pair from a generation. Returns what it did."""
    output_dir = Path(output_dir)
    gens = generations(output_dir)
    if not gens:
        raise FileNotFoundError(
            f"{output_dir/ARCHIVE} has no generation to restore — this session "
            f"was never fused")
    src = gens[0] if gen is None else output_dir / ARCHIVE / f"gen_{int(gen):03d}"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} does not exist — available: "
            f"{', '.join(g.name for g in gens)}")
    n_before = _count(output_dir / PARENT)
    for f in (PARENT, MASKS):
        if (src / f).exists():
            shutil.copy2(src / f, output_dir / f)
    from segmentation import mask_space
    mask_space.invalidate(output_dir)
    # the ledger records that the rounds were undone; it is append-only history
    mp = output_dir / MAP
    if mp.exists():
        try:
            doc = json.loads(mp.read_text())
            doc.setdefault("unfused", []).append(
                {"restored_from": src.name, "objects_before": n_before})
            mp.write_text(json.dumps(doc))
        except Exception:  # noqa: BLE001
            pass
    n_after = _count(output_dir / PARENT)
    log(f"restored {src.name}: {n_before} object(s) -> {n_after} masklet(s). "
        f"The matching has NOT been re-run — segmentation_result.json still "
        f"describes the fused state until it does.")
    return {"restored_from": str(src), "before": n_before, "after": n_after}


def _count(p: Path) -> int:
    try:
        return len(json.loads(p.read_text()).get("instances") or [])
    except Exception:  # noqa: BLE001
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir")
    ap.add_argument("--gen", type=int, default=None,
                    help="generation to restore (default: gen_000, the raw "
                         "SAM3 output)")
    ap.add_argument("--list", action="store_true",
                    help="list the generations and exit")
    a = ap.parse_args(argv)
    if a.list:
        for g in generations(a.output_dir):
            print(f"{g.name}  {_count(g/PARENT)} masklet(s)")
        return 0
    unfuse(a.output_dir, a.gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
