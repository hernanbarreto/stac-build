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


def plan_chunks(n_keyframes: int, walk_m: float, chunk_walk_m: float,
                min_size: int = 24, max_size: int = 150) -> Tuple[int, int]:
    """(chunk_size, overlap) in KEYFRAMES so each chunk covers ~`chunk_walk_m`
    of WALK. 50 % overlap.

    USER 2026-09-23: *"implementemos el chunk walk 12, por algo estaban no?"* —
    and the reason is measured. A chunk size in FRAMES covers a different
    DISTANCE in every scene, because keyframes are parallax-uniform, not
    distance-uniform:

        pccr          216 kf / 18.8 m -> 0.087 m/kf -> 60 kf = 5.2 m per chunk
        test2         255 kf / 12.9 m -> 0.051 m/kf -> 60 kf = 3.1 m per chunk
        observatorio  213 kf / 11.2 m -> 0.053 m/kf -> 60 kf = 3.2 m per chunk

    A 3-metre chunk gives Omega almost no baseline across its 60 frames — nearly
    the same viewpoint sixty times — and puts seven seams inside twelve metres.
    pccr at 5.2 m per chunk was accepted; those two at 3 m came out broken.

    The walk this is sized from is PHASE 1's, which reads long when that pass
    drifted (pccr: 43.7 m over a ~19 m walk). That inflation is not a problem
    here: m/kf is inflated by the same factor, so the chunk lands at
    `chunk_walk_m / inflation` of REAL walk — on pccr, 59 kf, which is the 60/30
    layout that works. It is self-correcting in the direction that matters:
    a pass that drifted more gets shorter chunks.

    Clamped: below `min_size` the overlap alignment starves; above `max_size`
    the chunk re-enters the drift regime the chunking exists to avoid.
    """
    if n_keyframes < 2 or walk_m <= 0 or chunk_walk_m <= 0:
        raise ValueError("need n_keyframes>=2, walk_m>0, chunk_walk_m>0")
    m_per_kf = walk_m / n_keyframes
    size = int(round(chunk_walk_m / max(m_per_kf, 1e-6)))
    size = max(min_size, min(max_size, size, n_keyframes))
    return size, size // 2


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
