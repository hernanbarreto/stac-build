# STAC-Builder — chunked-metric Omega: walk measurement + chunk/anchor planning.
#
# Pure helpers (numpy only) behind the two-phase reconstruction:
#   phase 1: single Omega pass over the motion keyframes (the probe — and the final
#            result for short scans);
#   phase 2: when the measured walk exceeds the comfort limit, re-run in chunks sized
#            by WALKED METERS, each chunk metric-locked by DA3 anchors, glued SE(3),
#            loop closure on.
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


def trim_static_ends(centers: np.ndarray, fx: np.ndarray = None,
                     floor_ratio: float = 10.0, zoom_z_cut: float = 3.5) -> Tuple[int, int]:
    """(lo, hi): the slice of keyframes that actually WALKS (and does not ZOOM),
    per the probe trajectory.

    Two per-keyframe 'no 3D information' conditions, trimmed only as maximal runs
    at the HEAD and TAIL (cutting a mid-walk stretch would split the sequence into
    islands with no shared frames to glue — mid-walk weakness is the chunk health
    gate's job):

    1. STATIC: step an order of magnitude below the session's own walking pace
       (median step / floor_ratio — the same physical decade criterion as the
       health gate; measured on test4: body steps 29-47 cm vs tail 1-5.5 cm).
       Rotation in place holds no parallax. Scale-invariant.
    2. ZOOM (when `fx` per keyframe is given, from the probe's intrinsic.txt):
       robust z (Iglewicz-Hoaglin, cut 3.5) of the estimated focal against the
       trajectory's median. Optical zoom magnifies without adding baseline — no
       parallax no matter how the poses come out (on test4 the zoomed tail's
       garbage poses JUMPED metres instead of standing still, which is exactly
       why the step criterion alone missed it: fx 550 body vs 704-1334 tail).

    Returns keyframe indices with centers[lo:hi] kept; (0, len) when nothing trims.
    """
    centers = np.asarray(centers, np.float64)
    n = len(centers)
    if n < 3:
        return 0, n
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)   # steps[i]: kf i -> i+1
    med = float(np.median(steps))
    if med <= 0:
        return 0, n
    bad_kf = np.zeros(n, bool)
    if fx is not None and len(np.asarray(fx)) == n:
        f = np.asarray(fx, np.float64)
        fmed = float(np.median(f))
        fmad = float(np.median(np.abs(f - fmed)))
        if fmad > 0:
            bad_kf |= np.abs(f - fmed) / (1.4826 * fmad) > float(zoom_z_cut)
    # a step is unusable when it is static OR touches a zoomed keyframe
    bad = (steps < med / float(floor_ratio)) | bad_kf[:-1] | bad_kf[1:]
    lo = 0
    while lo < len(bad) and bad[lo]:
        lo += 1
    hi = n
    while hi - 2 >= lo and bad[hi - 2]:
        hi -= 1
    if hi - lo < 2:            # degenerate (everything static/zoomed) — untouched:
        return 0, n            # phase 2 would not even trigger on a ~0 m walk
    return lo, hi


def plan_chunks(n_keyframes: int, walk_m: float, chunk_walk_m: float,
                min_size: int = 24, max_size: int = 150) -> Tuple[int, int]:
    """(chunk_size, overlap) in KEYFRAMES so each chunk covers ~chunk_walk_m of walk.

    Keyframes are parallax-uniform, so count maps to walk through the measured
    m/keyframe of THIS scan (walk_m / n_keyframes). 50% overlap. chunk_size is clamped:
    below min_size the overlap alignment starves; above max_size the chunk re-enters
    the drift regime the chunking exists to avoid.
    """
    if n_keyframes < 2 or walk_m <= 0 or chunk_walk_m <= 0:
        raise ValueError("need n_keyframes>=2, walk_m>0, chunk_walk_m>0")
    m_per_kf = walk_m / n_keyframes
    size = int(round(chunk_walk_m / max(m_per_kf, 1e-6)))
    size = max(min_size, min(max_size, size, n_keyframes))
    overlap = size // 2
    return size, overlap


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
