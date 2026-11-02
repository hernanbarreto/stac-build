"""Omega's chunk files on the pod's LOCAL disk while the fork runs (USER 2026-11-01: "esta etapa se
está comiendo muchísimo tiempo … acá hay un cuello de botella claro").

MEASURED on station 2026-11-01: between the bridges and the pose graph the fork reads and rewrites
its eight 2 GB chunk files about ten times (the metric lock, the two loop measurements, the scale
close, the exact seams, the alignment, the uncertainty, the held-out pairs, the depth graph, the
outputs) — 360 MB/s over the network file system that holds /workspace, under one core busy and the
card at 0 % for ~8 minutes. The pod's local disk wrote 2.9 GB/s (fdatasync) and its page cache
serves the re-reads.

:func:`place` makes ``<save_dir>/_tmp_results_unaligned`` (and ``_tmp_results_aligned`` when the
run does not keep it) a SYMLINK to a directory under :data:`SCRATCH_ROOT` when the chunks fit the
local disk; the fork and every reader keep their paths, the bytes are the same files. When they do
not fit, nothing moves and it is said. :func:`remove` deletes a scratch directory through its link
(the link and its target); :func:`gc` deletes every scratch directory no live link points to (a
replace wipe or a crash leaves links gone and targets behind).

A session whose output was started on the network keeps it there (a resume finds a real directory
with the run's chunks and leaves it as it is). A link whose target is gone (the pod's local disk is
not persistent storage) gets an empty target again: the fork finds no chunk and infers it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

SCRATCH_ROOT = Path(os.environ.get("STAC_LOCAL_SCRATCH", "/root/stac_scratch"))
OWNER_FILE = "owner.json"
# a BOUND of a chunk's bytes per grid pixel per frame — what the fork stores per frame: world points
# (3 float32), their confidence, the depth and its confidence (float32 each) and the uint8 image =
# 27 B/px (station 2026-11-01, chunk_1.npy: 1.076 GB for 67 frames at 1024x576 = 27.2 B/px with the
# camera tokens); 32 leaves room for the fork's smaller arrays
BYTES_PER_PIXEL_BOUND = 32
# the share of the local disk's free bytes the chunks may take (the rest stays free for the pod)
FREE_FRACTION = 0.8


def scratch_dir_for(save_dir: Path) -> Path:
    """The scratch directory of one fork save directory (named by the save directory's path)."""
    key = hashlib.sha256(str(Path(save_dir).resolve()).encode()).hexdigest()[:16]
    return SCRATCH_ROOT / key


def estimate_bytes(chunk_lengths: Sequence[int], grid_wh: Sequence[int]) -> int:
    """The most the chunk files occupy at once: every chunk once plus the largest twice over (the
    alignment writes chunk k aligned before it deletes chunk k unaligned)."""
    px = int(grid_wh[0]) * int(grid_wh[1]) * BYTES_PER_PIXEL_BOUND
    lens = [int(n) for n in chunk_lengths]
    return (sum(lens) + (max(lens) if lens else 0)) * px


def _free_bytes(root: Path) -> int:
    p = root
    while not p.exists():
        p = p.parent
    return int(shutil.disk_usage(p).free)


def place(save_dir: Path, subs: Iterable[str], est_bytes: int, log: Callable = print) -> Dict:
    """Link each ``<save_dir>/<sub>`` to the local scratch when ``est_bytes`` fits; returns what was
    done ({"placed": [...], "kept": [...], "reason": ...})."""
    save_dir = Path(save_dir)
    subs = list(subs)
    sdir = scratch_dir_for(save_dir)
    free = _free_bytes(SCRATCH_ROOT)
    budget = int(FREE_FRACTION * (free + _scratch_bytes(sdir)))
    rec: Dict = {"scratch": str(sdir), "est_bytes": int(est_bytes), "budget_bytes": budget,
                 "placed": [], "kept": []}
    if est_bytes > budget:
        rec["reason"] = (f"the chunks need up to {est_bytes / 1e9:.1f} GB and the local disk allows "
                         f"{budget / 1e9:.1f} GB — they stay on the network file system")
        rec["kept"] = subs
        log(f"[scratch] {rec['reason']}")
        return rec
    save_dir.mkdir(parents=True, exist_ok=True)
    for sub in subs:
        link = save_dir / sub
        target = sdir / sub
        if link.is_symlink():
            if Path(os.readlink(link)) != target:
                link.unlink()                         # another scratch layout: relinked below
            else:
                target.mkdir(parents=True, exist_ok=True)   # a lost target comes back empty
                rec["placed"].append(sub)
                continue
        elif link.exists():
            rec["kept"].append(sub)                   # a run started on the network stays there
            continue
        target.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        rec["placed"].append(sub)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / OWNER_FILE).write_text(json.dumps({"save_dir": str(save_dir.resolve()),
                                               "subs": rec["placed"]}))
    if rec["placed"]:
        log(f"[scratch] {', '.join(rec['placed'])} on the local disk ({sdir}; up to "
            f"{est_bytes / 1e9:.1f} GB of {budget / 1e9:.1f} GB allowed)"
            + (f"; kept on the network: {', '.join(rec['kept'])}" if rec["kept"] else ""))
    return rec


def _scratch_bytes(sdir: Path) -> int:
    if not sdir.is_dir():
        return 0
    return sum(f.stat().st_size for f in sdir.rglob("*") if f.is_file())


def remove(path: Path) -> int:
    """Delete ``path`` — through its link when it is one (the target and the link) — returning
    the bytes freed. A missing path frees nothing."""
    path = Path(path)
    freed = 0
    if path.is_symlink():
        target = Path(os.readlink(path))
        if not target.is_absolute():
            target = path.parent / target
        if target.is_dir():
            freed = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target, ignore_errors=True)
        path.unlink()
    elif path.is_dir():
        freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        freed = path.stat().st_size
        path.unlink()
    return freed


def gc(log: Callable = print) -> int:
    """Delete every scratch directory no live link points to; returns the bytes freed."""
    if not SCRATCH_ROOT.is_dir():
        return 0
    freed = 0
    for sdir in sorted(SCRATCH_ROOT.iterdir()):
        if not sdir.is_dir():
            continue
        try:
            owner = json.loads((sdir / OWNER_FILE).read_text())
            save_dir = Path(owner["save_dir"])
            subs = list(owner.get("subs") or [])
        except (OSError, ValueError, KeyError):
            save_dir, subs = None, []
        live = any(save_dir is not None and (save_dir / s).is_symlink()
                   and Path(os.readlink(save_dir / s)) == sdir / s for s in subs)
        if not live:
            b = _scratch_bytes(sdir)
            shutil.rmtree(sdir, ignore_errors=True)
            freed += b
            log(f"[scratch] {sdir} had no live link — deleted ({b / 1e9:.1f} GB)")
    return freed
