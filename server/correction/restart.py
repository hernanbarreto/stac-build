"""Epoch 0 for a precision chain that starts again (zaragoza, 2026-11-09).

The precision core's first steps measure against Omega's epoch-0 geometry (F2's gauge refuses any other
epoch). After a chain publishes its cloud, the map worker used to DELETE every stored epoch (USER
2026-09-28: "la única para visualizar debe ser la N, el resto deben descartarse"), so a chain that had to
start again from F0 — its code or configuration changed, Omega's pass unchanged — found the session in the
published epoch with no way back: zaragoza 2026-11-09 failed at F2 ("the session is at geometry epoch 3 …
select epoch 0 first").

The fix keeps the WAY BACK, not the epochs: once the product is published, the stored epochs of the live
epoch's lineage (the pose-only states between Omega's epoch 0 and the published one) move into
``output/_restart_epochs/`` — out of the epochs the viewer lists, so only the published epoch is shown, as
he ordered — and every other stored epoch is deleted as before. When the chain starts again from its first
step with the session in another epoch, :func:`restore_epoch0` moves them back and selects epoch 0 with the
existing, journaled ``correction.apply.select_epoch``; nothing is recomputed.

This module is imported by the chain runner and the map worker only — never by a precision step — so it
enters no step's code stamp.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Tuple

from correction.epoch import EPOCH_DIR_PREFIX, current_epoch, epoch_lineage

RESTART_DIR = "_restart_epochs"


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _number(d: Path) -> int:
    return int(d.name[len(EPOCH_DIR_PREFIX):])


def stash_lineage(output_dir) -> Tuple[int, int]:
    """After the product is published: the stored epochs on the live epoch's lineage move to
    ``output/_restart_epochs/``; every other stored epoch (another branch) is deleted, and so is anything
    already in the stash that is no longer on the lineage. Returns (bytes kept, bytes freed). The lineage
    is read strictly: an unreadable epoch record raises, nothing is moved or deleted on a guess."""
    out = Path(output_dir)
    live = current_epoch(out)
    lineage = set(epoch_lineage(out, live, strict=True))
    lineage.discard(live)
    stash = out / RESTART_DIR
    kept = freed = 0
    stored = [d for d in out.glob(f"{EPOCH_DIR_PREFIX}*") if d.is_dir() and not d.is_symlink()]
    for d in sorted(stored):
        try:
            j = _number(d)
        except ValueError:
            continue
        if j in lineage:
            stash.mkdir(exist_ok=True)
            target = stash / d.name
            if target.exists():
                shutil.rmtree(target)
            kept += _size(d)
            d.rename(target)
        else:
            freed += _size(d)
            shutil.rmtree(d)
    if stash.is_dir():
        for d in sorted(stash.glob(f"{EPOCH_DIR_PREFIX}*")):
            try:
                j = _number(d)
            except ValueError:
                continue
            if j not in lineage:
                freed += _size(d)
                shutil.rmtree(d)
    return kept, freed


def restore_epoch0(output_dir, log: Callable = print) -> bool:
    """The session back in Omega's epoch 0 for a chain that starts again from its first step: the stashed
    stores return to ``output/`` and ``correction.apply.select_epoch(…, 0)`` restores epoch 0 (nothing is
    destroyed: the live epoch is filed under its own store). False when the session already is in epoch 0.
    Raises naming the problem when epoch 0 is not kept (a session published before this fix)."""
    out = Path(output_dir)
    cur = current_epoch(out)
    if cur == 0:
        return False
    stash = out / RESTART_DIR
    moved = 0
    if stash.is_dir():
        for d in sorted(stash.glob(f"{EPOCH_DIR_PREFIX}*")):
            target = out / d.name
            if target.exists():
                raise RuntimeError(f"{d} and {target} both exist — the session's stored epochs are "
                                   f"inconsistent; nothing was moved past this point")
            d.rename(target)
            moved += 1
    if not (out / f"{EPOCH_DIR_PREFIX}0").is_dir():
        raise RuntimeError(f"the chain must start again from its first step, the session is in epoch {cur} "
                           f"and Omega's epoch 0 is not kept ({out / RESTART_DIR} holds no epoch 0) — a "
                           f"session published before 2026-11-09: reconstruct it again")
    from correction.apply import select_epoch
    select_epoch(out, 0, log=log)
    log(f"[restart] back in Omega's epoch 0 from epoch {cur} ({moved} stored epoch(s) returned from "
        f"{RESTART_DIR}/) — the chain starts again on the geometry it measures against")
    return True
