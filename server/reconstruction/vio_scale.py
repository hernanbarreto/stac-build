"""
VIO scale source for scale_align v2 (precision task, Phase A).
================================================================================
A phone VIO trajectory (ARCore/ARKit) recorded with the video gives METRIC
positions whose short-horizon scale is excellent even though the pose drifts
over minutes. So the VIO↔Omega scale is estimated PER SEGMENT — the robust
median of trajectory-length ratios over time segments — never one global
length quotient (which drift would corrupt):

    s_vio = median over segments of  L_vio(seg) / L_omega(seg)

where L_* is the arc length walked inside the segment. Omega's keyframe camera
centres are UP-TO-SCALE at estimation time (scale_align runs before apply), so
s_vio is directly the similarity the pipeline applies.

Priority when VIO is present: VIO SETS the scale; DA3 stays as cross-check.
The VIO↔DA3 agreement (%) is ALWAYS reported — it is the best per-session
diagnostic of DA3's scale bias we have.

FAIL-HARD contract: a VIO file that is present but unusable (unparseable, too
short, no time overlap with the video, too few valid segments) RAISES with the
exact reason. Silent fallback to DA3 would hide a capture problem — the user
removes/fixes the file explicitly instead.

File format: docs/VIO_FORMAT.md (CSV: timestamp,x,y,z[,qx,qy,qz,qw] with
optional header; JSON: {"video_fps": …, "samples": [{"t","p",["q"]}…]} or a
plain list of samples). Timestamps are SECONDS from the start of the video.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("VioScale")

MIN_SAMPLES = 20                    # fewer VIO samples than this cannot support segments
DEFAULT_SEGMENT_S = 5.0             # segment length (s): long enough to walk measurable
                                    # distance, short enough that VIO drift stays negligible
DEFAULT_MIN_SEG_DISP_M = 0.05       # segments where VIO walked less than this are skipped
                                    # (standing still → ratio of two ~zeros = noise)
DEFAULT_MIN_SEGMENTS = 8
DEFAULT_MIN_COVERAGE = 0.5          # matched segments must span ≥ this fraction of the
                                    # keyframe time range


def load_vio_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Parse a VIO trajectory file → (t[N] seconds, pos[N,3] meters, video_fps|None).
    Fails hard with the exact reason on malformed data. Orientation columns are
    accepted and ignored (arc length needs positions only)."""
    path = Path(path)
    t: List[float] = []
    p: List[List[float]] = []
    fps: Optional[float] = None

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            raise RuntimeError(f"VIO file {path.name}: invalid JSON ({e})")
        samples = data.get("samples", data) if isinstance(data, dict) else data
        if isinstance(data, dict) and data.get("video_fps"):
            fps = float(data["video_fps"])
        if not isinstance(samples, list):
            raise RuntimeError(f"VIO file {path.name}: expected a list of samples "
                               f"(or {{'samples': [...]}}), got {type(samples).__name__}")
        for i, s in enumerate(samples):
            try:
                t.append(float(s["t"]))
                pos = s["p"]
                p.append([float(pos[0]), float(pos[1]), float(pos[2])])
            except Exception:
                raise RuntimeError(f"VIO file {path.name}: sample {i} malformed — need "
                                   f"{{'t': seconds, 'p': [x,y,z]}} (see docs/VIO_FORMAT.md)")
    else:
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            raise RuntimeError(f"VIO file {path.name}: empty")
        start = 0
        header = [c.strip().lower() for c in rows[0]]
        cols = {"t": 0, "x": 1, "y": 2, "z": 3}
        if any(not _is_float(c) for c in rows[0]):
            start = 1
            name_map = {"timestamp": "t", "time": "t", "t": "t",
                        "x": "x", "y": "y", "z": "z",
                        "px": "x", "py": "y", "pz": "z"}
            found = {}
            for i, c in enumerate(header):
                key = name_map.get(c)
                if key and key not in found:
                    found[key] = i
            if set(found) >= {"t", "x", "y", "z"}:
                cols = found
            elif len(header) < 4:
                raise RuntimeError(f"VIO file {path.name}: header {header} — need at "
                                   f"least timestamp,x,y,z (see docs/VIO_FORMAT.md)")
        for i, row in enumerate(rows[start:], start=start):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            try:
                t.append(float(row[cols["t"]]))
                p.append([float(row[cols["x"]]), float(row[cols["y"]]),
                          float(row[cols["z"]])])
            except Exception:
                raise RuntimeError(f"VIO file {path.name}: row {i + 1} malformed "
                                   f"({row[:6]}) — need timestamp,x,y,z[,qx,qy,qz,qw]")

    if len(t) < MIN_SAMPLES:
        raise RuntimeError(f"VIO file {path.name}: only {len(t)} samples "
                           f"(need ≥{MIN_SAMPLES}) — trajectory unusable")
    ta = np.asarray(t, np.float64)
    pa = np.asarray(p, np.float64)
    order = np.argsort(ta, kind="stable")
    ta, pa = ta[order], pa[order]
    if float(ta[-1] - ta[0]) <= 0:
        raise RuntimeError(f"VIO file {path.name}: timestamps span zero time")
    return ta, pa, fps


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def video_fps(session_dir: Path) -> Optional[float]:
    """FPS of the session's source video (frame number / fps = frame timestamp).
    Uses OpenCV; returns None when no video is found (caller decides fatality)."""
    session = Path(session_dir)
    vids = sorted(session.glob("source_video.*"))
    if not vids:
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(str(vids[0]))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps if fps > 0 else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not read fps from {vids[0].name}: {e}")
        return None


def _arc_length(t: np.ndarray, p: np.ndarray, t0: float, t1: float) -> Optional[float]:
    """Arc length of the sampled trajectory inside [t0, t1] (endpoints interpolated).
    None when fewer than 2 samples fall in the window."""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 2:
        return None
    seg_t = t[m]
    seg_p = p[m]
    # interpolate exact endpoints so segment boundaries do not quantize on samples
    ends = []
    for te in (t0, t1):
        if seg_t[0] <= te <= seg_t[-1]:
            continue
        i = np.searchsorted(t, te)
        if 0 < i < len(t):
            a = (te - t[i - 1]) / max(t[i] - t[i - 1], 1e-9)
            ends.append((te, p[i - 1] + a * (p[i] - p[i - 1])))
    for te, pe in ends:
        if te < seg_t[0]:
            seg_t = np.concatenate([[te], seg_t]); seg_p = np.vstack([pe, seg_p])
        else:
            seg_t = np.concatenate([seg_t, [te]]); seg_p = np.vstack([seg_p, pe])
    return float(np.sum(np.linalg.norm(np.diff(seg_p, axis=0), axis=1)))


def estimate_vio_scale(vio_t: np.ndarray, vio_p: np.ndarray,
                       kf_t: np.ndarray, kf_c: np.ndarray,
                       segment_s: float = DEFAULT_SEGMENT_S,
                       min_seg_disp_m: float = DEFAULT_MIN_SEG_DISP_M,
                       min_segments: int = DEFAULT_MIN_SEGMENTS,
                       min_coverage: float = DEFAULT_MIN_COVERAGE) -> dict:
    """Per-segment robust VIO↔Omega scale.

    kf_t: keyframe timestamps (s), kf_c: keyframe camera centres [N,3] in omega's
    UP-TO-SCALE units, both time-ordered. Segments of `segment_s` seconds tile the
    overlap of the two time ranges; a segment votes when it has ≥2 keyframes AND
    the VIO walked ≥ min_seg_disp_m in it. s_vio = median of votes.

    Raises RuntimeError (fail-hard) when the result would be statistically
    meaningless: too little overlap, too few voting segments, or coverage below
    min_coverage of the keyframe time span."""
    order = np.argsort(kf_t, kind="stable")
    kf_t, kf_c = np.asarray(kf_t, np.float64)[order], np.asarray(kf_c, np.float64)[order]

    t0 = max(float(vio_t[0]), float(kf_t[0]))
    t1 = min(float(vio_t[-1]), float(kf_t[-1]))
    kf_span = float(kf_t[-1] - kf_t[0])
    if t1 - t0 <= 0 or kf_span <= 0:
        raise RuntimeError(
            f"VIO/keyframe time ranges do not overlap (VIO {vio_t[0]:.1f}–{vio_t[-1]:.1f}s, "
            f"keyframes {kf_t[0]:.1f}–{kf_t[-1]:.1f}s) — check that VIO timestamps are "
            f"seconds from the start of the video (docs/VIO_FORMAT.md)")

    n_seg = max(1, int(np.ceil((t1 - t0) / max(segment_s, 1e-6))))
    ratios, segments = [], []
    for i in range(n_seg):
        a, b = t0 + i * segment_s, min(t0 + (i + 1) * segment_s, t1)
        km = (kf_t >= a) & (kf_t <= b)
        if km.sum() < 2:
            continue
        l_omega = float(np.sum(np.linalg.norm(np.diff(kf_c[km], axis=0), axis=1)))
        l_vio = _arc_length(vio_t, vio_p, float(kf_t[km][0]), float(kf_t[km][-1]))
        if l_vio is None or l_vio < min_seg_disp_m or l_omega <= 1e-9:
            continue
        r = l_vio / l_omega
        ratios.append(r)
        segments.append({"t_start": a, "t_end": b, "l_vio_m": l_vio,
                         "l_omega": l_omega, "ratio": r})
    if len(ratios) < min_segments:
        raise RuntimeError(
            f"VIO present but only {len(ratios)} usable segments (need ≥{min_segments}) — "
            f"VIO too short/static or keyframes too sparse. Fix or remove the VIO file "
            f"(no silent fallback).")
    coverage = float(sum(s["t_end"] - s["t_start"] for s in segments)) / kf_span
    if coverage < min_coverage:
        raise RuntimeError(
            f"VIO segments cover only {coverage:.0%} of the keyframe time span "
            f"(need ≥{min_coverage:.0%}) — VIO recording does not span the walk. Fix or "
            f"remove the VIO file (no silent fallback).")

    r = np.asarray(ratios, np.float64)
    s_vio = float(np.median(r))
    mad = float(np.median(np.abs(r - s_vio)))
    return {"s_vio": s_vio, "n_segments": int(r.size), "coverage_frac": coverage,
            "mad_rel": float(mad / s_vio) if s_vio else None,
            "ratio_min": float(r.min()), "ratio_max": float(r.max()),
            "segment_s": float(segment_s), "segments": segments}
