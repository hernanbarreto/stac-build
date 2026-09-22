# STAC-Builder — chunked-metric Omega: walk measurement + chunk/anchor planning.
#
# Pure helpers (numpy only) behind the ONE-PASS reconstruction (USER ORDER
# 2026-09-22): `reconstruction.simple.chunk_frames` is both the single-pass
# capacity and the chunk size, decided from the keyframe COUNT before any
# inference — at or under it one chunk with no overlap, over it chunked-metric at
# 50 % overlap with every chunk metric-locked by DA3 anchors, glued SE(3), loop
# closure on. `walk_length_m` stays because the walk is still worth REPORTING;
# `plan_chunks` (chunk size from WALKED METERS) was deleted with the two-phase
# flow — on pccr it read 44.1 m over a ~19 m walk and sized the chunks from it.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def walk_length_m(poses_txt: Path) -> float:
    """Trajectory length from camera_poses.txt (one flattened 4x4 c2w per line, or
    frame_idx + 16 values). Metric ONLY after scale_align has been applied."""
    rows = [line.split() for line in open(poses_txt) if line.strip()]
    arr = np.array([[float(x) for x in r] for r in rows])
    if arr.shape[1] == 17:
        arr = arr[:, 1:]
    if arr.shape[1] != 16:
        raise ValueError(f"unexpected camera_poses.txt layout: {arr.shape[1]} cols")
    centers = arr.reshape(-1, 4, 4)[:, :3, 3]
    if len(centers) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())


def chunk_ranges(n_keyframes: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """[(start, end)) keyframe-index ranges, EXACTLY as VGGT-Long slices them
    (step = chunk_size - overlap; last chunk clipped to the end)."""
    if n_keyframes <= chunk_size:
        return [(0, n_keyframes)]
    step = chunk_size - overlap
    ranges = []
    start = 0
    while start < n_keyframes:
        end = min(start + chunk_size, n_keyframes)
        ranges.append((start, end))
        if end >= n_keyframes:
            break
        start += step
    return ranges


def plan_anchor_indices(n_keyframes: int, chunk_size: int, overlap: int,
                        per_chunk: int = 3) -> List[int]:
    """Keyframe indices (sorted, deduped) for the DA3 metric anchors: `per_chunk`
    spread inside EVERY chunk's range, so no chunk is left without a metric lock."""
    per_chunk = max(1, int(per_chunk))
    picks = set()
    for start, end in chunk_ranges(n_keyframes, chunk_size, overlap):
        span = end - start
        if span <= 0:
            continue
        if per_chunk == 1:
            fracs = [0.5]
        else:
            fracs = [0.15 + 0.7 * i / (per_chunk - 1) for i in range(per_chunk)]
        for fr in fracs:
            picks.add(start + min(span - 1, int(round(fr * (span - 1)))))
    return sorted(picks)
