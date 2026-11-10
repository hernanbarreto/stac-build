"""P4 — joint refinement of the keyframe poses, the session camera and the
landmarks, and the localisation of the witness frames (claude_stac.txt §4-F5).

Unknowns: an SE(3) per keyframe, ONE session camera (OPENCV: fx, fy, cx, cy, k1,
k2, p1, p2 — F0's ``camera.json``), the landmarks. Residuals: reprojection in
NATIVE pixels under a Huber loss (``huber_px``). Engine: pycolmap / COLMAP's Ceres
(``reconstruction.colmap_ba``: ``_tune_ceres``, ``_solve_checked`` — a rung that
does not converge fails, no fallback), Ceres on ONE thread (``ceres_threads`` must be
1: its multi-threaded accumulation is not bit-reproducible) and every COLMAP RANSAC
seeded (``seed``) — identical inputs give bit-identical outputs.

Initialisation: the current keyframe poses (Omega × F2's gauge when it was
applied), F0's camera, landmarks triangulated from the FIT tracks (F4
``track_split`` 0) seen from at least ``min_tri_deg`` apart.

The gauge: no fixed prior. Each keyframe's centre is held by a soft prior whose σ
grows with the walk the way consecutive-distance priors of relative σ_gauge
(F2's measured scatter of its applied instrument) accumulate: σ_k =
σ_gauge · sqrt(Σ d_i²) over the steps up to k.

The ladder, each rung judged ONLY on the held-out tracks (split 1: never
optimised — triangulated with the rung's own poses and camera and reprojected):
  R0 camera fixed → R1 + fx, fy, cx, cy → R2 + k1, k2, p1, p2; ONE camera for every keyframe
  (docs/plan_determinismo.md point 62). R3 (a focal per temporal block, measured and reported,
  never applied) was REMOVED 2026-11-10 with its temporal test, the rolling-shutter diagnostic
  and the prior-depth cross-check (USER: "lo que no se usa para corregir geometría no se pierde
  un solo segundo de máquina" — R3 alone took 18 min on pccr 2026-08-24).

THE JUDGE (docs/plan_determinismo.md points 46, 59, 60, 61 — USER 2026-10-07, after pccr
2026-08-31 took R2 on a 0.0005 px paired change over 128k tracks, R2 having ENDED AT A
HIGHER Ceres cost than R1 whose parameter space it contains, and F6 bent every depth map
through that lens: floor undulation 14.6 → 45.1 cm):
  * every rung is WARM-STARTED from the solution of the rung it contains (poses, landmarks,
    camera), the position priors staying at the gauge poses for all of them; a nested rung
    that still ends above its parent's final Ceres cost is logged and NOT taken (point 61);
  * after each rung a CONTINUATION solve runs from its end state with the same settings; the
    median paired change of the held-out statistic between the two IS the solver's own error
    (point 59), reported per rung;
  * a more complex rung enters only by THE USER'S RULE, ``metric_lock.decide_change`` (point
    1): the paired held-out improvement is significant under a CLUSTER bootstrap by keyframe
    (tracks seen from one keyframe share its pose and camera error — point 46), at least
    ``min_judge_closures(confidence)`` keyframes testify (5 at 0.95), and the median
    improvement is at least ``correction_graph.graph.improvement_error_factor`` × the larger
    solver error of the two states compared; otherwise the simpler rung stays;
  * a held-out track whose observations do not round-trip through a rung's lens is DROPPED
    and COUNTED for that rung, never a rejection of the rung (point 60); the comparison
    uses only the tracks valid in both rungs.
Every rung, verdict and margin is reported in ``refine.json``; the solver's wall-clock times
go to ``refine.timing.json`` (point 36: no clock in a compared artifact).

Witness frames (the second pass of ``refine_poses_ba_twopass``): PnP + refinement
of every witness against the FIXED landmarks with its witness→keyframe tracks,
the session camera fixed; a witness whose reprojection RMS exceeds the bootstrap
upper bound of the keyframes' held-out quantile stays ``unlocalized`` with the
reason.

Outputs (``output/precision``): ``refine.json``, ``refine.timing.json``, ``refine_residuals.npz``,
``witness_poses.txt`` (+ ``witness_frames.txt``, float64 round-trip exact — point 45);
``output/camera.json`` at camera_epoch + 1; the keyframes' rigid motion as an epoch through
``precision.poses_epoch.apply_pose_epoch`` (poses only — no cloud before F7).

CLI (the mapanything env: pycolmap 4): ``python -m precision.refine --session <dir>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REFINE_NAME = "refine.json"
TIMING_NAME = "refine.timing.json"
RESIDUALS_NAME = "refine_residuals.npz"
WITNESS_POSES_NAME = "witness_poses.txt"
WITNESS_FRAMES_NAME = "witness_frames.txt"
REFINE_VERSION = 3            # 2 (2026-10-07): the judge of points 46 / 59 / 60 / 61, R3 report-only;
                              # 3 (2026-11-10): R3, the rolling-shutter and prior-depth reports removed
PROVENANCE = "tool_measured"
LOG_TAG = "[refine]"
# the ladder of nested camera models, simplest first (point 62: one camera for every keyframe)
LADDER = ("R0", "R1", "R2")


class RefineError(RuntimeError):
    """A structural impossibility of the stage — always with the exact reason."""


def _vendor_path() -> None:
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)


# ── geometry (numpy / OpenCV) ────────────────────────────────────────────

def K_of(params: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = (float(v) for v in params[:4])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def normalized(uv: np.ndarray, params: Sequence[float], solver: dict) -> np.ndarray:
    """Distorted native pixels → normalised image coordinates (x/z, y/z) under the
    OPENCV model, the undistortion iterated to convergence (F0's solver bounds) and
    verified by re-distortion (``precision.camera.undistort_normalized``: a point that
    did not converge fails the call)."""
    from precision.camera import undistort_normalized
    return undistort_normalized(np.asarray(uv, np.float64).reshape(-1, 2), K_of(params),
                                np.asarray(params[4:8], np.float64), **solver)


def project(X: np.ndarray, w2c: np.ndarray, params: Sequence[float]) -> np.ndarray:
    import cv2
    R, t = w2c[:3, :3], w2c[:3, 3]
    rv, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rv, t,
                              K_of(params), np.asarray(params[4:8], np.float64))
    return uv.reshape(-1, 2)


def triangulate(xn: np.ndarray, w2c: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Multi-view DLT of one track from normalised coordinates ``xn`` (k, 2) seen by
    cameras ``w2c`` (k, 4, 4). Returns (X, max pairwise ray angle in degrees, in
    front of every camera)."""
    A = []
    for (x, y), T in zip(xn, w2c):
        P = T[:3, :4]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _u, _s, Vt = np.linalg.svd(np.asarray(A))
    Xh = Vt[-1]
    if abs(Xh[3]) < np.finfo(float).eps:
        return np.full(3, np.nan), 0.0, False
    X = Xh[:3] / Xh[3]
    z = np.array([(T[:3, :3] @ X + T[:3, 3])[2] for T in w2c])
    C = np.array([-T[:3, :3].T @ T[:3, 3] for T in w2c])
    rays = X[None] - C
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    cosm = np.clip(rays @ rays.T, -1.0, 1.0)
    return X, float(np.degrees(np.arccos(cosm.min()))), bool((z > 0).all())


class TrackGroups(dict):
    """{track: [(index of the frame in ``frames``, uv)]} — a dict of lists, AND the same content as arrays
    (2026-11-01: the conversions of the lists back to arrays — one list comprehension over 13 M fit
    observations per triangulation, per held-out, per solve — cost ~8 s each on pccr, ~80 s per F5):
    ``tracks`` (int64, the dict's order), ``counts``, ``starts``, ``ii`` (keyframe index per observation),
    ``uv`` (float64, one row per observation), observations in each track's order. The arrays are the
    lists' content, built from the same input by one stable sort; :func:`_track_arrays` reads them when
    the groups carry them and builds them from the lists otherwise (a plain dict — the tests' shape)."""
    # ``listed``: whether the dict holds the lists too. F5 builds its groups WITHOUT them (2026-11-10, pccr
    # 2026-08-24: 180 M keyframe observations at 367 bytes each as Python lists = 66 GB per grouping — F5 was
    # killed by the pod's 117 GB limit); a reader that needs the per-track lists calls :func:`listed`
    __slots__ = ("tracks", "counts", "starts", "ii", "uv", "listed")


def group_tracks(track: np.ndarray, frame: np.ndarray, uv: np.ndarray,
                 frames: Sequence[int], lists: bool = True) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """{track: [(index of the frame in ``frames``, uv)]} over the given frames — a :class:`TrackGroups`:
    tracks in the order of their first observation, observations in their order (what the per-observation
    loop produced; :func:`_group_tracks_reference` is that loop), with the arrays cached."""
    track = np.asarray(track, np.int64); frame = np.asarray(frame, np.int64)
    uv = np.asarray(uv)
    fr = np.asarray([int(f) for f in frames], np.int64)
    fo = np.argsort(fr, kind="stable")
    pos = np.searchsorted(fr[fo], frame)
    pos_c = np.minimum(pos, len(fr) - 1) if len(fr) else pos
    hit = (pos < len(fr)) & (fr[fo][pos_c] == frame) if len(fr) else np.zeros(len(frame), bool)
    t = track[hit]; i_of = fo[pos_c[hit]]; p = uv[hit]
    out = TrackGroups()
    out.listed = bool(lists)
    if not len(t):
        out.tracks = np.zeros(0, np.int64); out.counts = np.zeros(0, np.int64); out.starts = np.zeros(0, np.int64)
        out.ii = np.zeros(0, np.int64); out.uv = np.zeros((0, 2), np.float64)
        return out
    order = np.argsort(t, kind="stable")                      # by track, observations in their order
    ts = t[order]
    uniq, start_sorted, counts = np.unique(ts, return_index=True, return_counts=True)
    first_pos = order[start_sorted]                            # each track's first observation (original row)
    rank = np.argsort(first_pos, kind="stable")                # the tracks in first-appearance order
    tracks, counts, start_sorted = uniq[rank], counts[rank], start_sorted[rank]
    new_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos_in = np.arange(int(counts.sum())) - np.repeat(new_starts, counts)
    seg = order[np.repeat(start_sorted, counts) + pos_in]      # the observations, track by track
    ii = i_of[seg]; uvs = p[seg]
    out.tracks, out.counts, out.starts, out.ii = tracks, counts, new_starts, ii.astype(np.int64)
    out.uv = np.asarray(uvs, np.float64).reshape(-1, 2)
    if not lists:
        # ``lists=False``: the ARRAYS only, the dict left empty (F6's landmark rows read nothing else; building
        # 2.6 M Python lists was 10 of its 24 s on pccr) — a reader of the dict gets nothing, so only the
        # array-reading helpers (_track_arrays, _obs_of_tracks) may receive it
        return out
    ii_l = ii.tolist()
    for q in range(len(tracks)):
        a, b = int(new_starts[q]), int(new_starts[q] + counts[q])
        out[int(tracks[q])] = list(zip(ii_l[a:b], uvs[a:b]))
    return out


def listed(groups: Dict[int, List]) -> Dict[int, List]:
    """The groups as {track: [(keyframe index, uv)]} lists — ``groups`` itself when it holds them (a plain dict, or
    a :class:`TrackGroups` built with lists), else built from the arrays: the same tracks in the same order, the
    same observations in each track's order, uv as float64 rows (the values of the lists' rows). Only the per-track
    readers call it (the Ceres rung, the one-track-at-a-time references): on a large session it costs what the
    arrays-only groups save."""
    if not isinstance(groups, TrackGroups) or getattr(groups, "listed", True):
        return groups
    out: Dict[int, List] = {}
    ii_l = groups.ii.tolist()
    for q, t in enumerate(groups.tracks.tolist()):
        a = int(groups.starts[q]); b = a + int(groups.counts[q])
        out[int(t)] = list(zip(ii_l[a:b], groups.uv[a:b]))
    return out


def _group_tracks_reference(track, frame, uv, frames):
    """The per-observation loop :func:`group_tracks` replaced (kept for its test)."""
    idx = {int(f): i for i, f in enumerate(frames)}
    out: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for t, f, p in zip(track, frame, uv):
        i = idx.get(int(f))
        if i is not None:
            out.setdefault(int(t), []).append((i, p))
    return out


def count_dropped(dropped: Optional[dict], obs) -> None:
    """Point 60's ledger: a track whose observations' undistortion does not round-trip through
    the camera (``precision.camera`` verifies every point) is left out and COUNTED here — tracks
    and observations — never a failure of the caller."""
    if dropped is not None:
        dropped["tracks"] = int(dropped.get("tracks", 0)) + 1
        dropped["observations"] = int(dropped.get("observations", 0)) + len(obs)


def triangulate_tracks_per_track(groups: Dict[int, List], w2c: np.ndarray, params: Sequence[float],
                                 solver: dict, min_tri_deg: float,
                                 dropped: Optional[dict] = None) -> Dict[int, np.ndarray]:
    """The one-track-at-a-time reference of :func:`triangulate_tracks` (kept for the equivalence
    test: the batched form must give the same tracks and the same bits)."""
    from precision.camera import CameraError
    groups = listed(groups)
    out = {}
    for t, obs in groups.items():
        if len(obs) < 2:
            continue
        ii = np.array([i for i, _ in obs])
        try:
            xn = normalized(np.array([p for _, p in obs]), params, solver)
        except CameraError:
            count_dropped(dropped, obs)
            continue
        X, ang, front = triangulate(xn, w2c[ii])
        if front and ang >= min_tri_deg and np.all(np.isfinite(X)):
            out[t] = X
    return out


# observations per batch of the batched triangulation: a BOUND on memory (the DLT's U of a k-view
# track is (2k)^2 doubles), not a decision — the result does not depend on it
_TRI_BATCH_OBS = 400_000

# where the batched DLTs run (2026-11-01): numpy = the path of record (LAPACK, bit for bit the per-track
# reference) | cpu | cuda = torch's batched SVD (_dlt_stack_t), the same landmarks to 1e-9
TRIANGULATION_DEVICES = ("numpy", "cpu", "cuda")
_DEVICE_OVERRIDE: Optional[str] = None


def use_device(dev: Optional[str]) -> Optional[str]:
    """The device of every triangulation after this call; None = numpy, the path of record (the step CLIs
    set it from ``refine.triangulation_device``)."""
    global _DEVICE_OVERRIDE
    if dev is not None and str(dev) not in TRIANGULATION_DEVICES:
        raise RefineError(f"refine.triangulation_device must be one of {list(TRIANGULATION_DEVICES)}, got {dev!r}")
    if dev == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RefineError("refine.triangulation_device is 'cuda' and torch sees no card — the triangulation "
                              "does not fall back to the CPU by itself; declare cpu or numpy")
    _DEVICE_OVERRIDE = None if dev is None else str(dev)
    return _DEVICE_OVERRIDE


def device_of() -> str:
    return _DEVICE_OVERRIDE or "numpy"


# observations per call of the lens' batched undistortion: a BOUND on memory, not a decision — the call is per
# point and each track's tolerance its own, so whole tracks in chunks give the same bits (2026-11-10: Python's
# cv2.projectPoints inside it also returns its (2n × 14) Jacobian, 32 GB for pccr 2026-08-24's 144 M fit
# observations — with the rest F5 went over the pod's 117 GB)
_UNDISTORT_CHUNK_OBS = 2_000_000


def _undistort_tracks(uv: np.ndarray, K: np.ndarray, dist: np.ndarray, starts: np.ndarray, counts: np.ndarray,
                      solver: dict) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`precision.camera.undistort_normalized_tracks` over contiguous tracks (track q = rows
    starts[q] : starts[q] + counts[q], in order, no gap), whole tracks at a time, ≤ ``_UNDISTORT_CHUNK_OBS``
    observations per call (one track larger than that is one call)."""
    from precision.camera import undistort_normalized_tracks
    starts = np.asarray(starts, np.int64); counts = np.asarray(counts, np.int64)
    if len(uv) <= _UNDISTORT_CHUNK_OBS:
        return undistort_normalized_tracks(uv, K, dist, starts, counts, **solver)
    xn = np.empty((len(uv), 2), np.float64)
    ok = np.empty(len(starts), bool)
    ends = starts + counts
    q0 = 0
    while q0 < len(starts):
        q1 = max(q0 + 1, int(np.searchsorted(ends, starts[q0] + _UNDISTORT_CHUNK_OBS, side="right")))
        a, b = int(starts[q0]), int(ends[q1 - 1])
        xn[a:b], ok[q0:q1] = undistort_normalized_tracks(uv[a:b], K, dist, starts[q0:q1] - a, counts[q0:q1], **solver)
        q0 = q1
    return xn, ok


def triangulate_tracks(groups: Dict[int, List], w2c: np.ndarray, params: Sequence[float],
                       solver: dict, min_tri_deg: float,
                       dropped: Optional[dict] = None) -> Dict[int, np.ndarray]:
    """{track: X} of every track seen from ≥ 2 views whose rays span ≥ ``min_tri_deg`` in front
    of its cameras. A track whose observations cannot be undistorted under this camera (the
    round trip of ``precision.camera._undistort_verified`` fails) is dropped and counted in
    ``dropped`` (docs/plan_determinismo.md point 60) — it used to fail the whole call.

    BATCHED (USER 2026-10-08, speed: the per-track loop was 475 of F6's 655 CPU seconds on pccr,
    1.75 M tracks): every observation undistorted in ONE solver call with each TRACK's own
    round-trip tolerance (``precision.camera.undistort_normalized_tracks``), the DLTs of all the
    tracks with k views solved as one stack (numpy's batched SVD = the same LAPACK routine per
    matrix), the same front / angle tests — the result equals :func:`triangulate_tracks_per_track`
    (tests/test_triangulate_batched.py)."""
    tracks, counts, starts, ii, uv = _track_arrays(groups)
    if not tracks:
        return {}
    xn, ok_track = _undistort_tracks(uv, K_of(params), np.asarray(params[4:8], np.float64), starts, counts, solver)
    return _dlt_tracks(tracks, counts, starts, ii, xn, ok_track, w2c, min_tri_deg, dropped)


def _track_arrays(groups: Dict[int, List], min_views: int = 2):
    """The observations of every track with ≥ ``min_views`` views, contiguous per track: (tracks, counts,
    starts, keyframe index per observation, uv per observation) — from the cached arrays of a
    :class:`TrackGroups`, else from the lists."""
    if isinstance(groups, TrackGroups) and hasattr(groups, "counts"):
        keep = groups.counts >= int(min_views)
        if not keep.any():
            return [], np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2))
        counts = groups.counts[keep]
        m = np.repeat(keep, groups.counts)
        return (groups.tracks[keep].tolist(), counts, np.concatenate([[0], np.cumsum(counts)[:-1]]),
                groups.ii[m], groups.uv[m])
    tracks = [t for t, obs in groups.items() if len(obs) >= int(min_views)]
    if not tracks:
        return tracks, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2))
    counts = np.fromiter((len(groups[t]) for t in tracks), np.int64, len(tracks))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ii = np.fromiter((i for t in tracks for i, _ in groups[t]), np.int64, int(counts.sum()))
    uv = np.array([p for t in tracks for _, p in groups[t]], np.float64).reshape(-1, 2)
    return tracks, counts, starts, ii, uv


def _obs_of_tracks(groups: Dict[int, List], tracks: Sequence[int]):
    """(counts, keyframe index per observation, uv per observation) of ``tracks`` in that order — from the
    cached arrays when the groups carry them and ``tracks`` follows the groups' order (a filtered
    subsequence: every X of a triangulation is), else from the lists."""
    tracks = list(tracks)
    if isinstance(groups, TrackGroups) and hasattr(groups, "counts") and tracks:
        want = np.asarray(tracks, np.int64)
        keep = np.isin(groups.tracks, want)
        if int(keep.sum()) == len(want) and np.array_equal(groups.tracks[keep], want):
            m = np.repeat(keep, groups.counts)
            return groups.counts[keep], groups.ii[m], groups.uv[m]
        if not getattr(groups, "listed", True):
            # any order, from the arrays (no lists to read): each track's run of observations, in its order
            o = np.argsort(groups.tracks, kind="stable")
            pos = np.searchsorted(groups.tracks[o], want)
            q = o[np.minimum(pos, len(o) - 1)]
            if np.any(pos >= len(o)) or not np.array_equal(groups.tracks[q], want):
                raise RefineError("a track asked for is not in the groups")
            counts = groups.counts[q]
            idx = np.repeat(groups.starts[q], counts) + (np.arange(int(counts.sum()))
                                                          - np.repeat(np.cumsum(counts) - counts, counts))
            return counts, groups.ii[idx], groups.uv[idx]
    counts = np.fromiter((len(groups[t]) for t in tracks), np.int64, len(tracks))
    ii = np.fromiter((i for t in tracks for i, _ in groups[t]), np.int64, int(counts.sum()))
    uv = np.array([p for t in tracks for _, p in groups[t]], np.float64).reshape(-1, 2)
    return counts, ii, uv


def triangulate_tracks_blocks(groups: Dict[int, List], w2c: np.ndarray, params_by_block: Sequence[Sequence[float]],
                              block_of: np.ndarray, solver: dict, min_tri_deg: float,
                              dropped: Optional[dict] = None) -> Dict[int, np.ndarray]:
    """:func:`triangulate_tracks` with ONE CAMERA PER BLOCK (the rung's camera layout; F5 solves one
    block): every observation is undistorted
    under the camera of its keyframe's block — per block one solver call over that block's
    observations, each track's part judged with the track rule of ``undistort_normalized_tracks``;
    a track is dropped when any of its parts fails (what the per-track reference did) — then the
    same batched DLT. (The per-track loop this replaces took 270 s for a held-out on pccr.)"""
    tracks, counts, starts, ii, uv = _track_arrays(groups)
    if not tracks:
        return {}
    block_of = np.asarray(block_of, int)
    tid = np.repeat(np.arange(len(tracks)), counts)                     # track index per observation
    blk = block_of[ii]
    xn = np.full((len(uv), 2), np.nan)
    ok_track = np.ones(len(tracks), bool)
    for b in np.unique(blk):
        m = np.nonzero(blk == b)[0]                                      # track-ordered (observations are)
        tb = tid[m]
        first = np.r_[True, tb[1:] != tb[:-1]]
        starts_b = np.nonzero(first)[0]
        counts_b = np.diff(np.r_[starts_b, len(m)])
        pb = params_by_block[int(b)]
        xn_b, ok_b = _undistort_tracks(uv[m], K_of(pb), np.asarray(pb[4:8], np.float64), starts_b, counts_b, solver)
        xn[m] = xn_b
        ok_track[tb[starts_b][~ok_b]] = False
    return _dlt_tracks(tracks, counts, starts, ii, xn, ok_track, np.asarray(w2c, np.float64), min_tri_deg, dropped)


def _dlt_tracks(tracks, counts, starts, ii, xn, ok_track, w2c, min_tri_deg, dropped):
    """The batched multi-view DLT of :func:`triangulate_tracks` over normalised coordinates ``xn``
    (one row per observation), with its per-track gates (in front of every camera, ray span)."""
    if dropped is not None and not ok_track.all():
        bad = np.nonzero(~ok_track)[0]
        dropped["tracks"] = int(dropped.get("tracks", 0)) + len(bad)
        dropped["observations"] = int(dropped.get("observations", 0)) + int(counts[bad].sum())
    w2c = np.asarray(w2c, np.float64)
    keep = np.zeros(len(tracks), bool)
    Xall = np.full((len(tracks), 3), np.nan)
    for k in np.unique(counts):
        sel_all = np.nonzero((counts == k) & ok_track)[0]
        step = max(1, _TRI_BATCH_OBS // int(k))
        for b0 in range(0, len(sel_all), step):
            sel = sel_all[b0:b0 + step]
            idx = starts[sel][:, None] + np.arange(int(k))[None, :]          # (n, k) observations
            # the stacked DLT with the per-track gates (depth in every camera, the widest pair of rays:
            # X itself is the SVD's, bit for bit on numpy; these only gate it) — on the declared device
            X, good = _dlt_stack(xn[idx], w2c[ii[idx]], float(min_tri_deg))
            keep[sel[good]] = True
            Xall[sel[good]] = X[good]
    return {tracks[q]: Xall[q] for q in np.nonzero(keep)[0]}


def project_obs(X: np.ndarray, w2c: np.ndarray, params: np.ndarray) -> np.ndarray:
    """OPENCV projection of one point per row: ``X`` (M,3) world, ``w2c`` (M,4,4), ``params`` (M,8)
    or (8,) — cv2.projectPoints' polynomial (k1 k2 p1 p2), vectorised; equal to :func:`project`
    point by point (tests/test_heldout_batched.py)."""
    X = np.asarray(X, np.float64).reshape(-1, 3)
    w2c = np.asarray(w2c, np.float64).reshape(-1, 4, 4)
    P = np.broadcast_to(np.asarray(params, np.float64).reshape(-1, 8), (len(X), 8))
    Xc = np.einsum("mij,mj->mi", w2c[:, :3, :3], X) + w2c[:, :3, 3]
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    k1, k2, p1, p2 = P[:, 4], P[:, 5], P[:, 6], P[:, 7]
    r2 = x * x + y * y
    r4 = r2 * r2
    cd = 1.0 + k1 * r2 + k2 * r4
    a1 = 2.0 * x * y
    xd = x * cd + p1 * a1 + p2 * (r2 + 2.0 * x * x)
    yd = y * cd + p1 * (r2 + 2.0 * y * y) + p2 * a1
    return np.stack([P[:, 0] * xd + P[:, 2], P[:, 1] * yd + P[:, 3]], 1)


def _rms_per_track(groups: Dict[int, List], X: Dict[int, np.ndarray], w2c: np.ndarray,
                   params_of_obs) -> Dict[int, float]:
    """The RMS reprojection error per track of ``X`` over its own observations, every observation
    projected in ONE vectorised call (``params_of_obs(ii)`` → (M,8) camera per observation)."""
    tracks = list(X)
    if not tracks:
        return {}
    counts, ii, uv = _obs_of_tracks(groups, tracks)
    tid = np.repeat(np.arange(len(tracks)), counts)
    Xo = np.asarray([X[t] for t in tracks], np.float64)[tid]
    e2 = np.sum((project_obs(Xo, np.asarray(w2c, np.float64)[ii], params_of_obs(ii)) - uv) ** 2, axis=1)
    rms = np.sqrt(np.bincount(tid, weights=e2, minlength=len(tracks)) / counts)
    return {t: float(rms[q]) for q, t in enumerate(tracks)}


def heldout_rms(groups: Dict[int, List], w2c: np.ndarray, params: Sequence[float],
                solver: dict, min_tri_deg: float, dropped: Optional[dict] = None) -> Dict[int, float]:
    """Per held-out track: triangulated with these poses and this camera, the RMS
    of its reprojection over its own observations (native px). Tracks that do not
    round-trip through the lens are dropped and counted (``dropped``, point 60).
    BATCHED (2026-11-01): one projection call for every observation — the per-track loop
    (:func:`heldout_rms_per_track`, kept as the reference) cost 20-40 s per evaluation on pccr."""
    X = triangulate_tracks(groups, w2c, params, solver, min_tri_deg, dropped=dropped)
    pr = np.asarray(params, np.float64)
    return _rms_per_track(groups, X, w2c, lambda ii: pr)


def heldout_rms_per_track(groups: Dict[int, List], w2c: np.ndarray, params: Sequence[float],
                          solver: dict, min_tri_deg: float, dropped: Optional[dict] = None) -> Dict[int, float]:
    """The one-track-at-a-time reference of :func:`heldout_rms` (kept for the equivalence test)."""
    X = triangulate_tracks(groups, w2c, params, solver, min_tri_deg, dropped=dropped)
    groups = listed(groups)
    out = {}
    for t, P in X.items():
        e = [np.linalg.norm(project(P[None], w2c[i], params)[0] - np.asarray(p)) for i, p in groups[t]]
        out[t] = float(np.sqrt(np.mean(np.square(e))))
    return out


def heldout_rms_blocks(groups: Dict[int, List], w2c: np.ndarray, params_by_block: Sequence[Sequence[float]],
                       block_of: np.ndarray, solver: dict, min_tri_deg: float,
                       dropped: Optional[dict] = None) -> Dict[int, float]:
    """:func:`heldout_rms` with one camera per block (the rung's layout): each observation triangulated and
    reprojected under its keyframe's block camera, batched."""
    X = triangulate_tracks_blocks(groups, w2c, params_by_block, block_of, solver, min_tri_deg, dropped=dropped)
    pb = np.asarray(params_by_block, np.float64)
    bo = np.asarray(block_of, int)
    return _rms_per_track(groups, X, w2c, lambda ii: pb[bo[ii]])


def prior_sigmas(centres: np.ndarray, sigma_rel: float) -> np.ndarray:
    """σ_k = σ_rel · sqrt(Σ_{i≤k} d_i²): consecutive-distance priors of relative σ
    accumulated along the walk (the first keyframe carries its first step's)."""
    d = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    acc = np.sqrt(np.concatenate([[d[0] ** 2 if len(d) else 0.0], np.cumsum(d ** 2)]))
    return float(sigma_rel) * np.maximum(acc, acc[acc > 0].min() if (acc > 0).any() else 1.0)


# ── one rung (pycolmap) ──────────────────────────────────────────────────

@dataclass
class RungResult:
    name: str
    w2c: np.ndarray
    params_by_block: List[List[float]]      # the camera(s) by block — F5 solves one block
    block_of: np.ndarray                    # keyframe → camera block
    fit_rms_px: float                       # MEDIAN per-landmark reprojection error (px) of the
                                            # fit tracks — pccr 2026-09-29 R0: pycolmap's MEAN read
                                            # 1e146 px from a few landmarks the solve sent to
                                            # infinity while Ceres' Huber cost was 0.76 px
    n_points: int
    termination: str
    n_degenerate: int = 0                   # landmarks with a non-finite error or one beyond the
                                            # image diagonal — counted, never in the statistic
    X: Dict[int, np.ndarray] = field(default_factory=dict)   # the landmarks AS SOLVED (track → xyz):
                                            # the next rung's warm start and the witnesses' map
                                            # (points 55 / 61)
    initial_cost: float = float("nan")      # Ceres' total cost at the start / end of this solve
    final_cost: float = float("nan")        # (reprojection under Huber + position priors, the SAME
                                            # residual blocks for every rung of the ladder)
    iterations: int = 0                     # Ceres steps taken (successful + unsuccessful)
    seconds: float = 0.0                    # the solve's wall clock — refine.timing.json only


def run_rung(name: str, w2c_start: np.ndarray, params_start: Sequence[float], wh: Tuple[int, int],
             fit_groups: Dict[int, List], X_start: Dict[int, np.ndarray], sigmas: np.ndarray,
             cfg, block_of: Optional[np.ndarray] = None,
             params_by_block0: Optional[List[List[float]]] = None,
             prior_w2c: Optional[np.ndarray] = None) -> RungResult:
    """One pose-prior bundle adjustment. ``w2c_start`` / ``params_start`` / ``X_start``: where the
    solve STARTS (the gauge for R0; the rung it contains for R1, R2 and the continuation solves —
    point 61). ``prior_w2c``: the poses whose centres hold the gauge priors — the GAUGE poses for
    every rung (default: ``w2c_start``, the pre-2026-10-07 behaviour, kept for callers that start at
    the gauge). The position prior σ per keyframe is ``sigmas``."""
    if str(getattr(cfg, "solver", "ceres")) == "torch":
        from precision.ba_torch import run_rung_torch
        return run_rung_torch(name, w2c_start, params_start, wh, fit_groups, X_start, sigmas, cfg, block_of=block_of,
                              params_by_block0=params_by_block0, prior_w2c=prior_w2c)
    import pycolmap
    _vendor_path()
    from reconstruction.colmap_ba import _rigid, _tune_ceres, _solve_checked
    N = len(w2c_start)
    prior_w2c = w2c_start if prior_w2c is None else np.asarray(prior_w2c, np.float64)
    if prior_w2c.shape != w2c_start.shape:
        raise RefineError(f"{name}: {len(prior_w2c)} prior poses for {N} keyframes")
    block_of = np.zeros(N, int) if block_of is None else np.asarray(block_of, int)
    blocks = sorted(set(block_of.tolist()))
    pb0 = params_by_block0 or [list(params_start) for _ in blocks]
    rec = pycolmap.Reconstruction()
    for b in blocks:
        cam = pycolmap.Camera.create_from_model_name(b + 1, "OPENCV", float(pb0[b][0]),
                                                     int(wh[0]), int(wh[1]))
        cam.params = [float(v) for v in pb0[b]]
        cam.camera_id = b + 1
        rec.add_camera_with_trivial_rig(cam)
    fit_groups = listed(fit_groups)                       # this path reads the per-track lists
    per_img: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in range(N)}
    for t, obs in fit_groups.items():
        if t in X_start:
            for i, p in obs:
                per_img[i].append((t, p))
    p2d: Dict[Tuple[int, int], int] = {}
    for i in range(N):
        pts = [pycolmap.Point2D(np.asarray(p, float)) for _, p in per_img[i]]
        im = pycolmap.Image(name=f"{i}.jpg", camera_id=int(block_of[i]) + 1, points2D=pts)
        im.image_id = i + 1
        rec.add_image_with_trivial_frame(im, _rigid(w2c_start[i], pycolmap))
        for k, (t, _p) in enumerate(per_img[i]):
            p2d[(i, t)] = k
    n_pts = 0
    pid_of: Dict[int, int] = {}
    for t, X in X_start.items():
        tr = pycolmap.Track()
        for i, _p in fit_groups[t]:
            if (i, t) in p2d:
                tr.add_element(pycolmap.TrackElement(i + 1, p2d[(i, t)]))
        if tr.length() >= 2:
            pid_of[t] = int(rec.add_point3D(np.asarray(X, float), tr))
            n_pts += 1
    priors = []
    for i in range(N):
        pp = pycolmap.PosePrior()
        pp.position = -prior_w2c[i, :3, :3].T @ prior_w2c[i, :3, 3]
        pp.position_covariance = np.eye(3) * float(sigmas[i]) ** 2
        pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        pp.corr_data_id = rec.image(i + 1).data_id
        priors.append(pp)
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = name in ("R1", "R2")
    opts.refine_principal_point = name in ("R1", "R2")
    opts.refine_extra_params = name == "R2"
    opts.refine_points3D = True
    opts.refine_rig_from_world = True
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
    opts.ceres.loss_function_scale = float(cfg.huber_px)
    _tune_ceres(opts, int(cfg.max_iterations))
    # ONE thread (the config refuses any other value): Ceres' multi-threaded evaluation
    # and Schur accumulation sum in a run-dependent order — measured 1.8e-11 in the
    # poses between two identical solves at 8 threads, bit for bit the same at 1
    opts.ceres.solver_options.num_threads = int(cfg.ceres_threads)
    bcfg = pycolmap.BundleAdjustmentConfig()
    for i in range(N):
        bcfg.add_image(i + 1)
    popts = pycolmap.PosePriorBundleAdjustmentOptions()
    popts.prior_position_fallback_stddev = float(np.max(sigmas))
    # the adjuster first aligns the reconstruction to the priors by RANSAC; with COLMAP's
    # default seed (-1) and its PRNG unseeded, every solve of identical inputs differed
    # (measured) — both are fixed, so a rung does not depend on what ran before it
    popts.alignment_ransac.random_seed = int(cfg.seed)
    popts.alignment_ransac.num_threads = 1
    pycolmap.set_random_seed(int(cfg.seed))
    summary = _solve_checked(pycolmap.create_pose_prior_bundle_adjuster(
        opts, popts, bcfg, priors, rec), f"refine {name}")
    cs = _ceres_summary(summary)
    w2c = np.tile(np.eye(4), (N, 1, 1))
    for i in range(N):
        w2c[i, :3, :4] = np.asarray(rec.image(i + 1).cam_from_world().matrix())
    # pycolmap 4 reports the STORED point errors: refresh them, or they read 0
    rec.update_point_3d_errors()
    errs = np.array([float(pt.error) for pt in rec.points3D.values()], np.float64)
    diag = float(np.hypot(wh[0], wh[1]))
    ok = np.isfinite(errs) & (errs >= 0.0) & (errs <= diag)
    fit_med = float(np.median(errs[ok])) if ok.any() else float("nan")
    X_out = {t: np.asarray(rec.point3D(pid).xyz, np.float64).copy() for t, pid in pid_of.items()}
    return RungResult(name, w2c, [list(rec.camera(b + 1).params) for b in blocks], block_of,
                      fit_med, n_pts, str(getattr(summary, "termination_type", "")),
                      int((~ok).sum()), X=X_out,
                      initial_cost=float(cs["initial_cost"]), final_cost=float(cs["final_cost"]),
                      iterations=int(cs["num_successful_steps"]) + int(cs["num_unsuccessful_steps"]),
                      seconds=float(cs["total_time_in_seconds"]))


def _ceres_summary(summary) -> dict:
    """Ceres' own summary of a solve (pycolmap 4: ``BundleAdjustmentSummary.todict()
    ['ceres_summary']``): the costs and step counts the nested-rung check (point 61) and the
    report read. Missing fields FAIL — a rung whose cost cannot be read cannot be compared."""
    d = summary.todict() if hasattr(summary, "todict") else {}
    cs = d.get("ceres_summary") if isinstance(d, dict) else None
    need = ("initial_cost", "final_cost", "num_successful_steps", "num_unsuccessful_steps",
            "total_time_in_seconds")
    if not isinstance(cs, dict) or any(k not in cs for k in need):
        raise RefineError(f"pycolmap's bundle-adjustment summary carries no Ceres summary with "
                          f"{need} — the rungs' costs cannot be compared (point 61)")
    return cs


# ── the refinement ───────────────────────────────────────────────────────

def _heldout_of(r: RungResult, groups, solver, cfg) -> Tuple[Dict[int, float], dict]:
    """(per held-out track its reprojection RMS under this rung, the point-60 ledger of the
    tracks dropped because they do not round-trip through this rung's lens)."""
    from precision.camera import CameraError
    dropped: dict = {"tracks": 0, "observations": 0}
    if len(r.params_by_block) == 1:
        return heldout_rms(groups, r.w2c, r.params_by_block[0], solver, cfg.min_tri_deg,
                           dropped=dropped), dropped
    # per-block cameras: a track is judged with the camera of each observation's block —
    # triangulated in normalised coordinates of its own block (batched; the per-track loop
    # below it is the reference, tests/test_heldout_batched.py)
    return heldout_rms_blocks(groups, r.w2c, r.params_by_block, r.block_of, solver, cfg.min_tri_deg,
                              dropped=dropped), dropped


def _heldout_of_blocks_per_track(r: RungResult, groups, solver, cfg) -> Tuple[Dict[int, float], dict]:
    """The one-track-at-a-time reference of the per-block held-out."""
    from precision.camera import CameraError
    dropped: dict = {"tracks": 0, "observations": 0}
    groups = listed(groups)
    out = {}
    for t, obs in groups.items():
        if len(obs) < 2:
            continue
        ii = np.array([i for i, _ in obs])
        try:
            xn = np.array([normalized(np.asarray(p)[None], r.params_by_block[r.block_of[i]], solver)[0]
                           for i, p in obs])
        except CameraError:
            count_dropped(dropped, obs)
            continue
        X, ang, front = triangulate(xn, r.w2c[ii])
        if not (front and ang >= cfg.min_tri_deg and np.all(np.isfinite(X))):
            continue
        e = [np.linalg.norm(project(X[None], r.w2c[i], r.params_by_block[r.block_of[i]])[0]
                            - np.asarray(p)) for i, p in obs]
        out[t] = float(np.sqrt(np.mean(np.square(e))))
    return out, dropped


def track_clusters(groups: Dict[int, List]) -> Dict[int, int]:
    """The JUDGE a held-out track belongs to (point 46): the first keyframe (lowest index) it is
    observed in. Tracks seen from one keyframe share that keyframe's pose and camera error, so
    the bootstrap resamples keyframes, not tracks — independent of the emission order."""
    if isinstance(groups, TrackGroups) and hasattr(groups, "counts") and len(groups.counts):
        keep = groups.counts >= 2
        if not keep.any():
            return {}
        m = np.repeat(keep, groups.counts)
        counts = groups.counts[keep]
        first = np.minimum.reduceat(groups.ii[m], np.concatenate([[0], np.cumsum(counts)[:-1]]))
        return {int(t): int(i) for t, i in zip(groups.tracks[keep].tolist(), first.tolist())}
    return {t: int(min(i for i, _ in obs)) for t, obs in listed(groups).items() if len(obs) >= 2}


def solver_error(h_end: Dict[int, float], h_cont: Dict[int, float]) -> dict:
    """Point 59: the solver's own error = the paired change of the held-out statistic between
    a rung's END state and the CONTINUATION solve from it (same settings). ``error`` = the
    median ABSOLUTE paired change — how far one more solve alone moves a track's held-out RMS —
    the bar a rung-to-rung improvement is held to (× error_factor); the signed median and the
    sample size are reported next to it. 0.0 when the continuation reproduced every track
    exactly (a converged state); NaN with no common track (then nothing can be judged)."""
    common = sorted(set(h_end) & set(h_cont))
    if not common:
        return {"error": float("nan"), "median_signed_change": float("nan"), "n_tracks": 0}
    d = np.array([h_cont[t] - h_end[t] for t in common], np.float64)
    return {"error": float(np.median(np.abs(d))), "median_signed_change": float(np.median(d)),
            "n_tracks": int(len(common))}


def judge_rungs(h_best: Dict[int, float], h_cand: Dict[int, float], clusters: Dict[int, int],
                error: float, error_factor: float, cfg) -> dict:
    """THE USER'S RULE for 'does the more complex rung enter' (point 46 → metric_lock.decide_change):
    paired per-track held-out RMS of the best rung vs the candidate, over the tracks valid in BOTH
    (point 60), bootstrapped by KEYFRAME clusters (fixed seed), ≥ min_judge_closures(confidence)
    keyframes, median improvement ≥ error_factor × ``error`` (the larger solver error of the two
    states, point 59). Returns decide_change's dict (verdict + every margin) plus the per-rung
    medians on the common tracks, so the report compares like with like."""
    _vendor_path()
    from loop_utils.metric_lock import decide_change
    common = sorted(set(h_best) & set(h_cand))
    b = [h_best[t] for t in common]
    a = [h_cand[t] for t in common]
    verdict = decide_change(b, a, error=float(error), error_factor=float(error_factor),
                            confidence=float(cfg.heldout_confidence),
                            clusters=[clusters[t] for t in common] if common else None,
                            n_boot=int(cfg.permutations), seed=int(cfg.seed))
    verdict["n_tracks"] = len(common)
    verdict["median_best_px"] = float(np.median(b)) if common else None
    verdict["median_candidate_px"] = float(np.median(a)) if common else None
    return verdict


RUNGS_ENV = "STAC_REFINE_RUNGS"


def rung_ladder(log: Callable = print) -> Tuple[str, ...]:
    """The rungs F5 tries: the whole ladder, unless the DIAGNOSTIC variable ``STAC_REFINE_RUNGS``
    (e.g. ``R0,R1``) names a prefix of it (USER 2026-10-07, pccr: R2 entered on a 0.0005 px
    paired change and the cloud broke — a re-run forced to R1 showed the lens was the damage).
    Never set by the pipeline; declared in the log when set AND recorded in refine.json
    (``ladder`` / ``ladder_restricted_by``), so a restricted run never passes for a full one."""
    raw = os.environ.get(RUNGS_ENV, "").strip()
    if not raw:
        return LADDER
    want = tuple(x.strip().upper() for x in raw.split(",") if x.strip())
    if not want or want != LADDER[:len(want)]:
        raise RefineError(f"{RUNGS_ENV}={raw!r}: must name a prefix of the ladder {LADDER}")
    log(f"{LOG_TAG} DIAGNOSTIC: rungs restricted to {', '.join(want)} by {RUNGS_ENV}")
    return want


def _rung_report(r: RungResult, held_r: Dict[int, float], dropped: dict, cont: RungResult,
                 err: dict) -> dict:
    return {"params": r.params_by_block[0] if len(r.params_by_block) == 1 else None,
            "params_by_block": r.params_by_block if len(r.params_by_block) > 1 else None,
            "fit_rms_px": r.fit_rms_px, "n_points": r.n_points, "n_degenerate": r.n_degenerate,
            "termination": r.termination, "ceres": {"initial_cost": r.initial_cost,
                                                     "final_cost": r.final_cost,
                                                     "iterations": r.iterations},
            "heldout_median_px": float(np.median(list(held_r.values()))) if held_r else None,
            "n_heldout_tracks": len(held_r), "heldout_dropped": dict(dropped),
            "continuation": {"final_cost": cont.final_cost, "iterations": cont.iterations,
                             "termination": cont.termination, **err}}


def _rung_and_continuation(name: str, start: RungResult, wh, fit_g, held_g, sig, cfg, solver,
                           prior_w2c: np.ndarray, log: Callable, **kw):
    """One rung from ``start``'s solution, then the continuation solve from the rung's own end
    state (point 59). Returns (rung, its held-out, its dropped ledger, continuation, solver error)."""
    r = run_rung(name, start.w2c, start.params_by_block[0], wh, fit_g, start.X, sig, cfg,
                 prior_w2c=prior_w2c, **kw)
    cont = run_rung(name, r.w2c, r.params_by_block[0], wh, fit_g, r.X, sig, cfg,
                    prior_w2c=prior_w2c,
                    **({**kw, "params_by_block0": r.params_by_block} if "block_of" in kw else kw))
    held_r, dropped = _heldout_of(r, held_g, solver, cfg)
    held_c, _ = _heldout_of(cont, held_g, solver, cfg)
    err = solver_error(held_r, held_c)
    log(f"{LOG_TAG} {name}: Ceres cost {r.initial_cost:.6g} → {r.final_cost:.6g} in {r.iterations} "
        f"step(s); continuation → {cont.final_cost:.6g} in {cont.iterations}; solver error on the "
        f"held-out {err['error']:.3g} px (median |Δ| over {err['n_tracks']} tracks; signed "
        f"{err['median_signed_change']:+.3g}); {dropped['tracks']} held-out track(s) dropped "
        f"(no round trip through this lens)")
    return r, held_r, dropped, cont, err


def within_range(track: np.ndarray, frame: np.ndarray, uv: np.ndarray, kf_frames: Sequence[int], w2c: np.ndarray,
                 params: Sequence[float], solver: dict, min_tri_deg: float, range_m: float) -> np.ndarray:
    """THE RANGE (reconstruction.view_range_m, USER 2026-11-09: "ningún KF debe aportar puntos más allá de los 10 m
    … esos puntos tampoco deben ser considerados en etapas anteriores … ni de nada"): per observation, False when it
    is on a keyframe and its track, triangulated from every keyframe view with the poses ``w2c``, lies farther than
    ``range_m`` from that keyframe's camera. Observations off the keyframes and tracks that do not triangulate are
    kept (nothing to measure them with — the fit's own triangulation drops the latter)."""
    track = np.asarray(track, np.int64); frame = np.asarray(frame, np.int64)
    keep = np.ones(len(track), bool)
    X = triangulate_tracks(group_tracks(track, frame, uv, kf_frames, lists=False), w2c, params, solver, min_tri_deg)
    if not X:
        return keep
    tx = np.array(sorted(X), np.int64)
    Xs = np.stack([X[int(t)] for t in tx])
    fr = np.asarray([int(f) for f in kf_frames], np.int64)
    fo = np.argsort(fr, kind="stable")
    pos = np.searchsorted(fr[fo], frame)
    on_kf = (pos < len(fr)) & (fr[fo][np.minimum(pos, len(fr) - 1)] == frame)
    ti = np.searchsorted(tx, track)
    has = (ti < len(tx)) & (tx[np.minimum(ti, len(tx) - 1)] == track)
    m = on_kf & has
    C = np.linalg.inv(np.asarray(w2c, np.float64))[:, :3, 3]
    d = np.linalg.norm(Xs[ti[m]] - C[fo[pos[m]]], axis=1)
    keep[np.flatnonzero(m)[d > float(range_m)]] = False
    return keep


def refine_core(w2c0: np.ndarray, params0: Sequence[float], wh: Tuple[int, int],
                track: np.ndarray, frame: np.ndarray, uv: np.ndarray, split: np.ndarray,
                kf_frames: Sequence[int], sigma_rel: float, cfg, solver: dict, *,
                error_factor: float, log: Callable = print, solve: bool = True,
                range_m: float = 0.0) -> Dict[str, Any]:
    """The ladder over the keyframes. ``split`` per OBSERVATION (its track's split).
    ``error_factor``: ``correction_graph.graph.improvement_error_factor`` (the user's 2).
    ``solve=False`` (``refine.enabled: false``, the evaluation battery): no rung is solved — the
    initial state is the result (its held-out measured, its landmarks the witnesses' map)."""
    import pycolmap
    pycolmap.set_random_seed(int(cfg.seed))           # COLMAP's PRNG: identical runs, identical bits
    fac = float(error_factor)
    if not (np.isfinite(fac) and fac > 0.0):
        raise RefineError(f"error_factor {error_factor!r} must be finite and > 0 "
                          f"(correction_graph.graph.improvement_error_factor)")
    if range_m and range_m > 0:
        keep = within_range(track, frame, uv, kf_frames, w2c0, params0, solver, cfg.min_tri_deg, range_m)
        log(f"{LOG_TAG} range {range_m:g} m (reconstruction.view_range_m): {int((~keep).sum()):,} of {len(keep):,} "
            f"observation(s) of a landmark farther than it from their keyframe left out of the fit and the held-out")
        track, frame, uv, split = track[keep], frame[keep], uv[keep], split[keep]
    # the arrays only (TrackGroups.listed): every F5 reader takes them; the lists were 66 GB on pccr 2026-08-24
    fit_g = group_tracks(track[split == 0], frame[split == 0], uv[split == 0], kf_frames, lists=False)
    held_g = group_tracks(track[split == 1], frame[split == 1], uv[split == 1], kf_frames, lists=False)
    dropped_fit: dict = {"tracks": 0, "observations": 0}
    X0 = triangulate_tracks(fit_g, w2c0, params0, solver, cfg.min_tri_deg, dropped=dropped_fit)
    if len(X0) < 3:
        raise RefineError(f"{len(X0)} fit track(s) triangulate at ≥ {cfg.min_tri_deg:g}° — "
                          f"nothing to refine against")
    clusters = track_clusters(held_g)
    sig = prior_sigmas(np.array([-T[:3, :3].T @ T[:3, 3] for T in w2c0]), sigma_rel)
    base = RungResult("init", w2c0, [list(params0)], np.zeros(len(w2c0), int), float("nan"), 0, "",
                      X=X0)
    held: Dict[str, Dict[int, float]] = {}
    held["init"], dropped_init = _heldout_of(base, held_g, solver, cfg)
    if not solve:
        log(f"{LOG_TAG} refine.enabled: false — no rung solved; the initial poses and camera stand "
            f"(held-out median {np.median(list(held['init'].values())) if held['init'] else float('nan'):.3f} px "
            f"over {len(held['init'])} tracks)")
        return {"best": base, "rungs": {}, "held": held, "X": X0, "X0": X0, "fit_groups": fit_g,
                "held_groups": held_g, "prior_sigma_m": sig,
                "solver_errors": {}, "clusters": clusters, "ladder": [],
                "ladder_restricted_by": "refine.enabled: false", "error_factor": fac,
                "dropped": {"fit_init": dropped_fit, "heldout_init": dropped_init}, "timing": {}}
    errors: Dict[str, dict] = {}
    rungs: Dict[str, Any] = {}
    best: Optional[RungResult] = None       # the rung the session takes (the held-out's verdict)
    parent: RungResult = base               # the rung the next one is warm-started from and whose
                                            # final cost it must not exceed (point 61)
    ladder = rung_ladder(log)
    timing: Dict[str, float] = {}
    cont_costs: Dict[str, float] = {}       # each rung's continuation final cost (the cost resolution)
    for name in ladder:
        # point 61 (USER 2026-10-07): ONLY R2 starts at R1's solution. R1 starts at the gauge, as in
        # the validated recipe — started at R0's fixed-focal optimum it never moved (pccr
        # 2026-10-07: cost 0.587605 → 0.587605, the focal stayed at F0's 362.56)
        start = base if name in ("R0", "R1") else parent
        r, held_r, dropped, cont, err = _rung_and_continuation(
            name, start, wh, fit_g, held_g, sig, cfg, solver, w2c0, log)
        held[name] = held_r
        errors[name] = err
        timing[name] = r.seconds
        timing[name + "_continuation"] = cont.seconds
        rep = _rung_report(r, held_r, dropped, cont, err)
        # point 61: a nested rung starts AT its parent's solution, so its cost can only fall; one
        # that ends above it did not reach the optimum of a space that contains the parent's. The
        # bar is the MEASURED resolution of the PARENT's final cost: the pose-prior adjuster
        # re-aligns every start to the priors (RANSAC) before solving, so even a continuation from
        # a converged state ends a hair off it (synthetic: 1587.7093 vs 1587.7087, 3e-7 relative)
        # — the parent's own continuation change is what 'above' must exceed. Never the rung's own
        # continuation change: a rung that stopped early (pccr's R2 at 0.485 px over R1's 0.459)
        # would then excuse itself with the very distance it still had to fall.
        cost_tol = (abs(cont_costs[start.name] - start.final_cost) if start.name != "init" else 0.0)
        cont_costs[name] = cont.final_cost
        nested_ok = start.name == "init" or not (r.final_cost > start.final_cost + cost_tol)
        rep["nested_cost_check"] = {"parent": start.name, "parent_final_cost": start.final_cost,
                                    "final_cost": r.final_cost, "cost_tolerance": float(cost_tol),
                                    "margin": (float(start.final_cost + cost_tol - r.final_cost)
                                               if start.name != "init" else None),
                                    "passed": bool(nested_ok)}
        if not nested_ok:
            rep.update({"taken": False, "verdict_vs_best": None,
                        "rejected": f"ended at Ceres cost {r.final_cost:.6g} above {start.name}'s "
                                    f"{start.final_cost:.6g} (+ the measured cost resolution "
                                    f"{cost_tol:.3g}), whose parameter space it contains (point 61) — "
                                    f"the simpler rung stays"})
            rungs[name] = rep
            log(f"{LOG_TAG} {name}: NOT taken — {rep['rejected']}")
            continue
        if best is None:
            verdict, take = None, True          # R0: the base of the ladder
        else:
            bar = max(float(errors[best.name]["error"]), float(err["error"]))
            if not np.isfinite(bar):
                bar = float("nan")
            verdict = (judge_rungs(held[best.name], held_r, clusters, bar, fac, cfg)
                       if np.isfinite(bar) else
                       {"improves": False, "worsens": False, "n_tracks": 0,
                        "reason": "no held-out track common to both states — nothing to judge"})
            take = bool(verdict["improves"])
        rep.update({"verdict_vs_best": verdict, "taken": take})
        rungs[name] = rep
        parent = r
        if take:
            best = r
        log(f"{LOG_TAG} {name}: fit median {r.fit_rms_px:.3f} px"
            + (f" ({r.n_degenerate} degenerate landmark(s) left out)" if r.n_degenerate else "")
            + f", held-out median {rep['heldout_median_px']} px over {len(held_r)} tracks"
            + ("" if verdict is None else
               f" — vs {best.name if not take else 'previous best'} on {verdict.get('n_tracks', 0)} "
               f"common tracks: {'TAKEN' if take else 'not taken'} ({verdict.get('reason', '')})"))
    if best is None:
        raise RefineError("no rung could be judged — the session camera cannot be refined; see the "
                          "rung reasons in the log")
    return {"best": best, "rungs": rungs, "held": held, "X": best.X, "X0": X0, "fit_groups": fit_g,
            "held_groups": held_g, "prior_sigma_m": sig,
            "solver_errors": errors, "clusters": clusters, "ladder": list(ladder),
            "ladder_restricted_by": RUNGS_ENV if os.environ.get(RUNGS_ENV, "").strip() else None,
            "error_factor": fac, "dropped": {"fit_init": dropped_fit, "heldout_init": dropped_init},
            "timing": timing}


WITNESS_WORKERS = 8      # BOUND (cost): worker processes of the witnesses' PnP; the result does not depend on it


def _pnp_witnesses(args) -> List[Tuple[int, dict]]:
    """One worker of :func:`localize_witnesses`: COLMAP's PnP + refinement of each (frame, 2-D, 3-D) against the
    fixed landmarks, the LO-RANSAC seeded per call (the same options as the sequential loop: each call is
    independent of the others, so the split across workers changes no pose)."""
    params, wh, seed, bound_px, items = args
    import pycolmap
    pycolmap.set_random_seed(int(seed))
    cam = pycolmap.Camera.create_from_model_name(1, "OPENCV", float(params[0]), int(wh[0]), int(wh[1]))
    cam.params = [float(v) for v in params]
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.ransac.random_seed = int(seed)
    est.ransac.num_threads = 1
    ref = pycolmap.AbsolutePoseRefinementOptions()
    out = []
    for f, p2, p3 in items:
        res = pycolmap.estimate_and_refine_absolute_pose(p2, p3, cam, est, ref)
        if not res:
            out.append((f, {"localized": False, "reason": "pnp_failed", "n": int(len(p2))}))
            continue
        T = np.eye(4)
        T[:3, :4] = np.asarray(res["cam_from_world"].matrix())
        inl = np.asarray(res["inlier_mask"], bool)
        e = np.linalg.norm(project(p3[inl], T, params) - p2[inl], axis=1)
        rms = float(np.sqrt(np.mean(e ** 2))) if inl.any() else float("inf")
        ok = rms <= bound_px
        out.append((f, {"localized": bool(ok), "reason": None if ok else "rms_above_keyframe_heldout",
                        "rms_px": rms, "n": int(len(p2)), "n_inliers": int(inl.sum()),
                        "c2w": np.linalg.inv(T).tolist()}))
    return out


def localize_witnesses(best: RungResult, fit_track_X: Dict[int, np.ndarray],
                       track: np.ndarray, frame: np.ndarray, uv: np.ndarray,
                       witness_frames: Sequence[int], wh: Tuple[int, int], bound_px: float,
                       cfg, workers: Optional[int] = None) -> Dict[int, dict]:
    """PnP + refinement of every witness against the FIXED landmarks, the session
    camera fixed (the best rung's). The LO-RANSAC is seeded explicitly per call
    (its default seed is -1, the one that left the pose-prior alignment run-dependent) and
    single-threaded; the witnesses are split over ``workers`` processes (2026-11-01: COLMAP's PnP has
    no GPU path — 29 s on one core for pccr's 1,791 witnesses; each call is independent, so the poses
    are the same whatever the split: tests/test_refine_cpu_phases.py)."""
    params = [float(v) for v in best.params_by_block[0]]
    out = {}
    wset = set(int(w) for w in witness_frames)
    # the correspondences of every witness at once (2026-11-01: a Python loop over 16 M observations,
    # 11 s on pccr): the observations of a witness frame whose track is a landmark, in their order
    track = np.asarray(track, np.int64); frame = np.asarray(frame, np.int64); uv = np.asarray(uv)
    lm_ids = np.fromiter(fit_track_X.keys(), np.int64, len(fit_track_X))
    sel = np.flatnonzero(np.isin(frame, np.fromiter(wset, np.int64, len(wset))) & np.isin(track, lm_ids))
    sel = sel[np.argsort(frame[sel], kind="stable")]
    f_sorted = frame[sel]
    bounds = np.searchsorted(f_sorted, np.array(sorted(wset) + [np.iinfo(np.int64).max]))
    items = []
    for q, f in enumerate(sorted(wset)):
        rows = sel[bounds[q]:bounds[q + 1]]
        if len(rows) < int(cfg.min_witness_corr):
            out[f] = {"localized": False, "reason": "too_few_correspondences", "n": int(len(rows))}
            continue
        p2 = np.asarray(uv[rows], np.float64)
        p3 = np.array([fit_track_X[int(t)] for t in track[rows]], np.float64)
        out[f] = None                                        # the place of f in the sequential order
        items.append((f, p2, p3))
    n_w = max(1, min(int(WITNESS_WORKERS if workers is None else workers), len(items)))
    if items:
        chunks = [items[k::n_w] for k in range(n_w)]
        jobs = [(params, tuple(wh), int(cfg.seed), float(bound_px), c) for c in chunks if c]
        if n_w == 1:
            parts = [_pnp_witnesses(j) for j in jobs]
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_w, mp_context=multiprocessing.get_context("spawn")) as ex:
                parts = list(ex.map(_pnp_witnesses, jobs))
        for part in parts:
            for f, rec in part:
                out[f] = rec
    return out

def split_of_observations(track_ids: np.ndarray, track_split: np.ndarray, obs_track: np.ndarray) -> np.ndarray:
    """The split (0 fit / 1 held-out) of every observation from its track's — by searchsorted over the
    track ids (2026-11-01: a Python dict map over 16 M observations, 2.5 s on pccr, twice per chain)."""
    tid = np.asarray(track_ids, np.int64)
    order = np.argsort(tid, kind="stable")
    pos = np.searchsorted(tid[order], np.asarray(obs_track, np.int64))
    if np.any(pos >= len(tid)) or np.any(tid[order][np.minimum(pos, len(tid) - 1)] != np.asarray(obs_track, np.int64)):
        raise RefineError("an observation names a track that is not in the tracks' query list")
    return np.asarray(track_split, np.int8)[order][pos]


def _dlt_stack(x: np.ndarray, T: np.ndarray, min_tri_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """The multi-view DLT of a STACK of tracks with the same view count: ``x`` (n, k, 2) normalised
    coordinates, ``T`` (n, k, 4, 4) world-to-camera. Returns (X (n, 3), good (n,)) with the per-track
    gates of :func:`triangulate` (a finite homogeneous point, in front of every camera, rays spanning
    ≥ ``min_tri_deg``) — the same arithmetic as :func:`_dlt_tracks`, one LAPACK SVD per matrix."""
    dev = device_of()
    if dev != "numpy":
        return _dlt_stack_t(x, T, min_tri_deg, dev)
    n, k = x.shape[0], x.shape[1]
    P = T[:, :, :3, :4]
    rx = x[:, :, 0:1] * P[:, :, 2, :] - P[:, :, 0, :]
    ry = x[:, :, 1:2] * P[:, :, 2, :] - P[:, :, 1, :]
    A = np.stack([rx, ry], axis=2).reshape(n, 2 * k, 4)
    _u, _s, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    ok = ~(np.abs(Xh[:, 3]) < np.finfo(float).eps)
    X = np.where(ok[:, None], Xh[:, :3] / np.where(ok, Xh[:, 3], 1.0)[:, None], np.nan)
    R = T[:, :, :3, :3]
    t = T[:, :, :3, 3]
    z = (R[:, :, 2, 0] * X[:, None, 0] + R[:, :, 2, 1] * X[:, None, 1] + R[:, :, 2, 2] * X[:, None, 2] + t[:, :, 2])
    C = -np.einsum("nkji,nkj->nki", R, t)
    rays = X[:, None, :] - C
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    cosm = np.clip(np.einsum("nki,nli->nkl", rays, rays), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosm.reshape(n, -1).min(axis=1)))
    good = ok & (z > 0).all(axis=1) & (ang >= min_tri_deg) & np.isfinite(X).all(axis=1)
    return X, good


def _dlt_stack_t(x: np.ndarray, T: np.ndarray, min_tri_deg: float, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`_dlt_stack` on torch (float64, ``device``): the same design rows, torch's batched SVD (the
    reduced right factor holds the four right singular vectors; its last row is the DLT's solution), the
    same gates — the landmarks agree with numpy's to 1e-9 (tests/test_triangulate_torch.py)."""
    import torch
    xt = torch.as_tensor(np.ascontiguousarray(x, dtype=np.float64), device=device)
    Tt = torch.as_tensor(np.ascontiguousarray(T, dtype=np.float64), device=device)
    n, k = int(xt.shape[0]), int(xt.shape[1])
    P = Tt[:, :, :3, :4]
    rx = xt[:, :, 0:1] * P[:, :, 2, :] - P[:, :, 0, :]
    ry = xt[:, :, 1:2] * P[:, :, 2, :] - P[:, :, 1, :]
    A = torch.stack([rx, ry], 2).reshape(n, 2 * k, 4)
    # the right singular vectors of A are those of its QR factor R (A = Q·R, Q orthonormal): a batched QR
    # (cuBLAS, every matrix at once) then the SVD of the (4 × 4) R — on CUDA torch's batched SVD is fast
    # only up to 32 rows; a 24-view track has 48 and fell to one cusolver call per matrix
    R4 = torch.linalg.qr(A, mode="r").R                                   # (n, 4, 4)
    Vh = torch.linalg.svd(R4, full_matrices=False).Vh                     # (n, 4, 4)
    Xh = Vh[:, -1, :]
    ok = ~(Xh[:, 3].abs() < np.finfo(float).eps)
    denom = torch.where(ok, Xh[:, 3], torch.ones_like(Xh[:, 3]))
    X = torch.where(ok[:, None], Xh[:, :3] / denom[:, None], torch.full_like(Xh[:, :3], float("nan")))
    R = Tt[:, :, :3, :3]
    t = Tt[:, :, :3, 3]
    z = (R[:, :, 2, 0] * X[:, None, 0] + R[:, :, 2, 1] * X[:, None, 1] + R[:, :, 2, 2] * X[:, None, 2] + t[:, :, 2])
    C = -torch.einsum("nkji,nkj->nki", R, t)
    rays = X[:, None, :] - C
    rays = rays / torch.linalg.norm(rays, dim=2, keepdim=True)
    cosm = torch.clamp(torch.einsum("nki,nli->nkl", rays, rays), -1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(cosm.reshape(n, -1).min(1).values))
    good = ok & (z > 0).all(1) & (ang >= float(min_tri_deg)) & torch.isfinite(X).all(1)
    return X.cpu().numpy(), good.cpu().numpy()


def heldout_leave_one_view_out(held_g: Dict[int, List], best: RungResult, solver: dict,
                               cfg, max_px: float = float("inf")) -> Dict[int, float]:
    """Per keyframe: the RMS of its held-out observations reprojected from their track triangulated
    WITHOUT that view — the situation of a witness frame, measured against landmarks it never shaped.
    BATCHED (2026-11-01): the per-track loop ran 705,519 Python triangulations on pccr (316 s of F5's
    CPU); here every observation is undistorted once (the per-point solver gives the same normalised
    coordinates whatever its companions), the tracks with k views are stacked and, for each left-out
    position, triangulated from their other k − 1 views by one batched SVD (:func:`_dlt_stack`, the
    same LAPACK routine per matrix as :func:`triangulate`), the left-out view's residual by
    :func:`project_obs`; the residuals are gathered per keyframe in the loop's order (track by track,
    view by view) and reduced as the loop did. :func:`heldout_leave_one_view_out_per_track` is the
    loop. A track whose observations do not round-trip through the lens is left out (the loop would
    have failed the stage on it)."""
    params = best.params_by_block[0]
    tracks, counts, starts, ii, uv = _track_arrays(held_g, min_views=3)
    if not tracks:
        return {}
    xn, ok_track = _undistort_tracks(uv, K_of(params), np.asarray(params[4:8], np.float64), starts, counts, solver)
    w2c = np.asarray(best.w2c, np.float64)
    frame_out, res_out, key_out = [], [], []
    for k in np.unique(counts):
        sel_all = np.nonzero((counts == k) & ok_track)[0]
        step = max(1, _TRI_BATCH_OBS // int(k))
        for b0 in range(0, len(sel_all), step):
            sel = sel_all[b0:b0 + step]
            idx = starts[sel][:, None] + np.arange(int(k))[None, :]              # (n, k)
            for j in range(int(k)):
                rest = np.concatenate([np.arange(j), np.arange(j + 1, int(k))])
                X, good = _dlt_stack(xn[idx[:, rest]], w2c[ii[idx[:, rest]]], float(cfg.min_tri_deg))
                if not good.any():
                    continue
                i_left = ii[idx[good, j]]
                pred = project_obs(X[good], w2c[i_left], np.asarray(params, np.float64))
                e = np.linalg.norm(pred - uv[idx[good, j]], axis=1)
                frame_out.append(i_left); res_out.append(e)
                key_out.append(sel[good] * int(k) + j)                            # (track, position): the loop's order
    if not frame_out:
        return {}
    fr = np.concatenate(frame_out); res = np.concatenate(res_out); key = np.concatenate(key_out)
    order = np.lexsort((key, fr))                                                # per frame, in the loop's order
    fr, res = fr[order], res[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(fr)) + 1, len(fr)]
    out = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = res[a:b]
        v = v[np.isfinite(v) & (v >= 0.0) & (v <= max_px)]
        if v.size:
            out[int(fr[a])] = float(np.sqrt(np.mean(np.square(v))))
    return out


def heldout_leave_one_view_out_per_track(held_g: Dict[int, List], best: RungResult, solver: dict,
                                         cfg, max_px: float = float("inf")) -> Dict[int, float]:
    """The per-track loop :func:`heldout_leave_one_view_out` replaced (kept for its test)."""
    params = best.params_by_block[0]
    per: Dict[int, List[float]] = {}
    for t, obs in listed(held_g).items():
        if len(obs) < 3:
            continue
        for k, (i, p) in enumerate(obs):
            rest = [o for kk, o in enumerate(obs) if kk != k]
            ii = np.array([j for j, _ in rest])
            xn = normalized(np.array([q for _, q in rest]), params, solver)
            X, ang, front = triangulate(xn, best.w2c[ii])
            if not (front and ang >= cfg.min_tri_deg and np.all(np.isfinite(X))):
                continue
            per.setdefault(i, []).append(
                float(np.linalg.norm(project(X[None], best.w2c[i], params)[0] - np.asarray(p))))
    out = {}
    for i, v in per.items():
        a = np.asarray(v, np.float64)
        a = a[np.isfinite(a) & (a >= 0.0) & (a <= max_px)]
        if a.size:
            out[i] = float(np.sqrt(np.mean(np.square(a))))
    return out


def heldout_bound(per_frame: Dict[int, float], cfg) -> float:
    """The bootstrap upper bound of the keyframes' held-out quantile
    (``heldout_confidence``) — what a frame localised as well as a keyframe may show."""
    v = np.array(list(per_frame.values()))
    rng = np.random.default_rng(int(cfg.seed))
    q = 100.0 * float(cfg.heldout_confidence)
    boots = [np.percentile(v[rng.integers(0, len(v), len(v))], q) for _ in range(int(cfg.permutations))]
    return float(np.percentile(boots, q))


# ── session I/O ──────────────────────────────────────────────────────────

def _gauge_sigma(output_dir: Path) -> Tuple[float, str]:
    p = output_dir / "gauge.json"
    if not p.exists():
        raise RefineError(f"{p} is missing — the refinement's gauge priors come from F2 "
                          f"(python -m precision.gauge --session <dir>)")
    g = json.loads(p.read_text())
    inst = g["applied_instrument"]
    return float(g["sigma_by_instrument"][inst]), inst


def run_refine(session_dir: Path, pcfg, *, apply: bool = True, log: Callable = print) -> Dict[str, Any]:
    from precision.camera import load_camera_json, save_camera_json, undistort_solver
    from precision.tracks import load_tracks_v2, KIND_WITNESS
    session_dir = Path(session_dir)
    out = session_dir / "output"
    cfg = pcfg.refine
    cam = load_camera_json(out / "camera.json")
    solver = undistort_solver(pcfg.camera)
    poses = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
    if len(kf) != len(poses):
        raise RefineError(f"camera_poses.txt ({len(poses)}) and camera_frames.txt ({len(kf)}) "
                          f"disagree")
    w2c0 = np.linalg.inv(poses)
    use_device(cfg.triangulation_device)                 # every DLT of this stage on the declared device
    _t_load = time.perf_counter()
    tr = load_tracks_v2(session_dir)
    _phase = {"load_tracks_s": round(time.perf_counter() - _t_load, 1)}
    tid = tr["track_query_id"]
    split_obs = split_of_observations(tid, tr["track_split"], tr["obs_track"])
    sigma_rel, inst = _gauge_sigma(out)
    log(f"{LOG_TAG} {len(kf)} keyframes, {len(tid):,} tracks, gauge σ {sigma_rel:.4f} ({inst})")
    from precision.config import raw_of
    from reconstruction.loops.config import improvement_error_factor
    fac = float(improvement_error_factor(raw_of(pcfg)))   # the job's frozen configuration (point 69)
    core = refine_core(w2c0, cam.params, (cam.width, cam.height), tr["obs_track"], tr["obs_frame"],
                       tr["obs_uv_native"], split_obs, kf, sigma_rel, cfg, solver,
                       error_factor=fac, log=log, solve=bool(cfg.enabled), range_m=float(pcfg.view_range_m))
    best = core["best"]
    _t = time.perf_counter()
    bound = heldout_bound(heldout_leave_one_view_out(core["held_groups"], best, solver, cfg,
                                                     max_px=float(np.hypot(cam.width, cam.height))), cfg)
    _phase["heldout_leave_one_view_out_s"] = round(time.perf_counter() - _t, 1)
    wit = sorted(set(tr["obs_frame"][tr["frame_kind"] == KIND_WITNESS].tolist()) - set(kf))
    # the witnesses localise against the landmarks AS THE BEST RUNG SOLVED THEM (in its world,
    # with its camera) — not against X0, triangulated from the initial poses with F0's camera
    _t = time.perf_counter()
    wloc = localize_witnesses(best, core["X"], tr["obs_track"], tr["obs_frame"],
                              tr["obs_uv_native"], wit, (cam.width, cam.height), bound, cfg)
    _phase["witnesses_s"] = round(time.perf_counter() - _t, 1)
    core["timing"].update(_phase)                       # refine.timing.json: the CPU around the solves, measured
    log(f"{LOG_TAG} phases: " + ", ".join(f"{k} {v}" for k, v in _phase.items()))
    pdir = out / "precision"
    pdir.mkdir(parents=True, exist_ok=True)
    loc = [f for f, r in wloc.items() if r["localized"]]
    from repro import write_poses_exact
    if loc:
        write_poses_exact(pdir / WITNESS_POSES_NAME, np.array([wloc[f]["c2w"] for f in loc]))
    else:
        (pdir / WITNESS_POSES_NAME).write_text("")
    (pdir / WITNESS_FRAMES_NAME).write_text(" ".join(str(f) for f in loc) + "\n")
    from intake.quality import read_session_epochs
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    epochs = read_session_epochs(session_dir)
    doc = {"version": REFINE_VERSION, "provenance": PROVENANCE, **epochs,
           RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out),
           "params": dataclasses.asdict(cfg),            # nested dataclasses (refine.torch_ba) as dicts
           "judge": {"rule": "metric_lock.decide_change (USER 2026-10-07): significant under a "
                             "keyframe-cluster bootstrap AND >= min_judge_closures(confidence) "
                             "keyframes AND median improvement >= improvement_error_factor x the "
                             "larger solver error (continuation solve) of the two states",
                     "improvement_error_factor": core["error_factor"],
                     "heldout_confidence": float(cfg.heldout_confidence),
                     "n_judge_keyframes": len(set(core["clusters"].values())),
                     "solver_errors": core["solver_errors"]},
           "ladder": core["ladder"], "ladder_restricted_by": core["ladder_restricted_by"],
           "dropped": core["dropped"],
           "gauge": {"sigma_rel": sigma_rel, "instrument": inst,
                     "prior_sigma_m": {"first": float(core["prior_sigma_m"][0]),
                                       "last": float(core["prior_sigma_m"][-1])}},
           "rungs": core["rungs"], "applied_rung": best.name,
           "camera": {"before": list(cam.params), "after": best.params_by_block},
           "n_landmarks": len(core["X"]),
           "keyframe_heldout_bound_px": bound,
           "witnesses": {"n": len(wit), "localized": len(loc),
                         "unlocalized": {str(f): r["reason"] for f, r in wloc.items()
                                         if not r["localized"]},
                         "rms_px": {str(f): r.get("rms_px") for f, r in wloc.items()}},
           "applied": False}
    # the solver's wall clock, outside the compared artifact (point 36)
    (pdir / TIMING_NAME).write_text(json.dumps({"stage": "refine", "ceres_seconds": core["timing"]},
                                               indent=1, default=float))
    np.savez_compressed(pdir / RESIDUALS_NAME,
                        heldout_track=np.array(list(core["held"][best.name].keys()), np.int64),
                        heldout_rms_px=np.array(list(core["held"][best.name].values()), np.float32),
                        init_track=np.array(list(core["held"]["init"].keys()), np.int64),
                        init_rms_px=np.array(list(core["held"]["init"].values()), np.float32))
    if apply:
        new_cam = cam.with_params(best.params_by_block[0], "refine", cam.camera_epoch + 1,
                                  report={"rung": best.name, "blocks": len(best.params_by_block)})
        save_camera_json(out / "camera.json", new_cam, epochs.get("geometry_epoch"))
        T = np.linalg.inv(best.w2c) @ w2c0              # c2w_new · w2c_old: world motion
        from precision.poses_epoch import apply_pose_epoch
        res = apply_pose_epoch(out, T[:, :3, :3], T[:, :3, 3], np.ones(len(T)), "refine",
                                    [{"stage": "refine", "rung": best.name}], log=log)
        doc["applied"] = bool(res)
        doc["epoch_to"] = (res or {}).get("epoch_to")
        doc["camera_epoch_to"] = new_cam.camera_epoch
    (pdir / REFINE_NAME).write_text(json.dumps(doc, indent=1, default=float))
    log(f"{LOG_TAG} applied rung {best.name}; witnesses {len(loc)}/{len(wit)} localized "
        f"(bound {bound:.2f} px) → {pdir / REFINE_NAME}")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.refine",
                                 description="Joint pose + session camera refinement (F5).")
    ap.add_argument("--session", required=True)
    ap.add_argument("--no-apply", action="store_true")
    args = ap.parse_args(argv)
    run_refine(Path(args.session), load_precision_config(session_dir=Path(args.session)),
               apply=not args.no_apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
