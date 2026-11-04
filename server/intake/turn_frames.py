"""Turn frames — Omega's CONTEXT views through a turn (USER 2026-11-04).

Hernán, after the SLAM trials of 2026-11-02/04: *"la optimización de la cantidad de poses en giros
… hay que mejorarla"*. VGGT-SLAM took a view every ~1° of turn and gave the best pccr by eye; I1
takes one keyframe per rotation quantum θ_q (~10.8° on pccr). But turn views have no parallax: as
KEYFRAMES they give F5 cameras without baseline and F6 bends their depth to ~0 (every SLAM output
of the trials). So two classes of views (*"programá las imágenes"*, the same day):

* the KEYFRAMES of I1 (intake/parallax.py) stay the only frames every stage after Omega knows —
  frame_list.json, camera_frames.txt, the per-frame records, F0–F6, the cloud;
* the TURN FRAMES enter only Omega's inference of the chunk that holds them, as extra views
  between its keyframes; the fork drops their slots before anything is saved (the loop bridges'
  extra frames are the precedent). Omega sees the turn with denser views; nothing downstream
  sees them.

WHICH FRAMES. The turn between two consecutive keyframes a → k is I1's window anchored at a: its
rotation reading of every frame is the angle of the rotation fitted from a (intake/parallax.py
step 4). That turn is cut into n equal steps,

    n = ceil(r_k / (θ_t · (1 + b))),   θ_t = θ_q / d,

r_k the keyframe's own reading, θ_q the session's rotation quantum (derived from the measured K),
b I1's keyframe_band_frac and d = ``reconstruction.vggtomega.turn_frames.per_rotation_quantum``:
no two consecutive Omega views of a window lie further apart than the top of θ_t's band — the
tolerance I1's keyframe rule closes a window on. The frame of step j is the SHARPEST (I0
sharp_rank, ties → the earliest, I1's own rule) whose reading lies within ±b of a step around
j·r_k/n, after the previous step's frame and before k; none in that band (the turn jumped over it
between two frames) → the frame whose reading is nearest (ties → the earliest). A window without
a keyframe, anchored on a non-keyframe (after a coverage break) or without a reading of its
keyframe gets none, and says so.

EVERY INTERVAL (USER 2026-11-04, the same night: *"¿qué pasa si la usamos en todo el recorrido
para dar completitud en todo el recorrido sin romper geometría?"* … *"con ambas mejoras"*): a
window the turn rule leaves without a frame — a stretch of walk, not a turn — is cut into
``min_per_interval`` + 1 equal steps of its PARALLAX reading (I1's parallax from the anchor, the
reading its keyframe rule closes a walking window on), each step's frame by the same band rule; a
keyframe without a parallax reading is cut on its rotation reading. 0 = turns only.

TWO ROLES (USER 2026-11-04, after the first run of every interval in Omega doubled its views and
halved its chunks: *"lo que quiero que cambie es que los chunks no se modifiquen por los frames
extra en los tramos rectos, que solo entren para afirmar o no en el F6"*): the TURN frames are
``context`` views — Omega infers them with the chunk, as validated on pccr (``files``: what the
chunk plan counts and the chunks are inferred with); the INTERVAL frames are ``witness`` views
only (``witness_files``): inferred in a pass of their own after each chunk (the fork's
_stac_witness_pass), never with it, and read by F6's vote alone — the chunks are the turn rule's.

Nothing is measured again: the readings are I1's own records (``intake/coverage_warnings.json``
'frames'), the sharpness I0's (``frames/quality_features.json``), the keyframes and windows
``frames/selected_frames.json``. The result, ``intake/turn_frames.json``, is stamped with the
sha256 of those three files and of this code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from intake.quality import _write_json_atomic

TURN_FRAMES_NAME = "turn_frames.json"
TURN_FRAMES_VERSION = 3          # 3: two roles — turn frames = Omega context, interval frames =
                                 #    F6 witnesses only (2: every interval as Omega context)
PROVENANCE = "tool_measured"
METHOD = "turn_rotation_steps+interval_parallax_steps"
CONFIG_PATH = "reconstruction.vggtomega.turn_frames"
SKIP_REASONS = ("no_keyframe", "anchor_not_keyframe", "no_rotation_reading", "no_reading")
# the reading a window is cut on: I1's per-frame record key and its unit
READINGS = {"rotation": ("rot_deg", "deg"), "parallax": ("parallax_px", "px")}


class TurnFramesError(RuntimeError):
    pass


@dataclass(frozen=True)
class TurnFramesConfig:
    enabled: bool
    per_rotation_quantum: int        # d: θ_t = θ_q / d
    min_per_interval: int            # the fewest context frames between two consecutive keyframes


def load_turn_frames_config(config: Dict[str, Any]) -> TurnFramesConfig:
    """``reconstruction.vggtomega.turn_frames`` of a run's config; a missing key fails naming it."""
    sec = (((config or {}).get("reconstruction") or {}).get("vggtomega") or {}).get("turn_frames")
    if not isinstance(sec, dict):
        raise TurnFramesError(f"{CONFIG_PATH} is missing from the config")
    for k in ("enabled", "per_rotation_quantum", "min_per_interval"):
        if k not in sec:
            raise TurnFramesError(f"{CONFIG_PATH}.{k} is missing from the config")
    d = sec["per_rotation_quantum"]
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise TurnFramesError(f"{CONFIG_PATH}.per_rotation_quantum must be an integer ≥ 1, got {d!r}")
    m = sec["min_per_interval"]
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise TurnFramesError(f"{CONFIG_PATH}.min_per_interval must be an integer ≥ 0, got {m!r}")
    return TurnFramesConfig(enabled=bool(sec["enabled"]), per_rotation_quantum=int(d),
                            min_per_interval=int(m))


def _finite(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def select_turn_frames(selection: Dict[str, Any], frame_records: Sequence[Dict[str, Any]],
                       quality_rows: Sequence[Dict[str, Any]], per_rotation_quantum: int,
                       min_per_interval: int) -> Dict[str, Any]:
    """The context frames of one session (see the module docstring). ``selection``: I1's
    selected_frames.json; ``frame_records``: I1's per-frame records (coverage_warnings.json
    'frames': frame, anchor, lost, rot_deg, parallax_px); ``quality_rows``: I0's frames (frame,
    file, sharp_rank, usable); ``min_per_interval``: the fewest frames between two consecutive
    keyframes (0 = turns only). Pure and deterministic. Returns {theta_q_deg, theta_t_deg,
    band_frac, step_max_deg, frames: [...], files: [...], windows: [...]}."""
    d = int(per_rotation_quantum)
    if d < 1:
        raise TurnFramesError(f"per_rotation_quantum must be ≥ 1, got {per_rotation_quantum}")
    if isinstance(min_per_interval, bool) or not isinstance(min_per_interval, int) or min_per_interval < 0:
        raise TurnFramesError(f"min_per_interval must be an integer ≥ 0, got {min_per_interval!r}")
    n_min = 1 + int(min_per_interval)
    theta_q = _finite(selection.get("rotation_quantum_deg"))
    b = _finite(selection.get("keyframe_band_frac"))
    if theta_q is None or theta_q <= 0.0 or b is None or not 0.0 <= b < 1.0:
        raise TurnFramesError(f"selected_frames.json carries no usable rotation quantum / band "
                              f"(rotation_quantum_deg {selection.get('rotation_quantum_deg')!r}, "
                              f"keyframe_band_frac {selection.get('keyframe_band_frac')!r}) — "
                              f"re-run intake I1")
    theta_t = theta_q / d
    step_max = theta_t * (1.0 + b)
    q = {int(r["frame"]): r for r in quality_rows}
    rec = {int(r["frame"]): r for r in frame_records}
    kfs = sorted(int(k["frame"]) for k in selection["keyframes"])
    kf_rot = {int(k["frame"]): _finite(k.get("rotation_deg")) for k in selection["keyframes"]}
    kf_par = {int(k["frame"]): _finite(k.get("parallax_px")) for k in selection["keyframes"]}
    kf_set = set(kfs)
    frames: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    for w in selection["windows"]:
        a, k = w.get("anchor"), w.get("keyframe")
        skip = ("no_keyframe" if k is None else
                "anchor_not_keyframe" if not w.get("anchor_is_keyframe") else None)
        if skip is None:
            r_k, p_k = kf_rot.get(int(k)), kf_par.get(int(k))
            n_rot = max(1, int(math.ceil(r_k / step_max))) if r_k is not None else 0
            if n_rot >= 2:                  # a turn: its rotation steps, Omega's context views
                kind, total, n, role = "rotation", r_k, n_rot, "context"
            elif n_min >= 2 and p_k is not None and p_k > 0.0:   # a walk: F6's witnesses only
                kind, total, n, role = "parallax", p_k, n_min, "witness"
            elif n_min >= 2 and r_k is not None and r_k > 0.0:
                kind, total, n, role = "rotation", r_k, n_min, "witness"
            elif r_k is None or n_min >= 2:
                skip = "no_rotation_reading" if n_min < 2 else "no_reading"
            else:
                continue                                     # a short turn, turns only: none
        if skip is not None:
            windows.append({"anchor": a, "keyframe": k, "skipped": skip})
            continue
        a, k = int(a), int(k)
        key, unit = READINGS[kind]
        cands = []
        for f in range(a + 1, k):
            r = rec.get(f)
            if (r is None or f in kf_set or int(r.get("anchor", -1)) != a or r.get("lost")
                    or not (q.get(f) or {}).get("usable")):
                continue
            x = _finite(r.get(key))
            if x is not None:
                cands.append((f, x))
        chosen: List[Dict[str, Any]] = []
        step = float(total) / n
        prev = a
        for j in range(1, n):
            target = j * step
            pool = [(f, x) for f, x in cands if f > prev]
            if not pool:
                break
            band = [(f, x) for f, x in pool if abs(x - target) <= b * step]
            if band:
                f, x = max(band, key=lambda c: (float(q[c[0]]["sharp_rank"]), -c[0]))
                how = "band"
            else:
                f, x = min(pool, key=lambda c: (abs(c[1] - target), c[0]))
                how = "nearest"
            chosen.append({"frame": f, "file": q[f]["file"], "anchor": a, "keyframe": k,
                           "role": role, "reading": kind, "unit": unit, "step": j, "steps": n,
                           "target": target, "value": x, "chosen_by": how})
            prev = f
        frames.extend(chosen)
        windows.append({"anchor": a, "keyframe": k, "role": role, "reading": kind, "unit": unit,
                        "total": float(total), "steps": n, "n_candidates": len(cands),
                        "n_turn_frames": len(chosen), "skipped": None})
    ordered = sorted(frames, key=lambda t: t["frame"])
    files = [t["file"] for t in ordered if t["role"] == "context"]
    witness_files = [t["file"] for t in ordered if t["role"] == "witness"]
    return {"theta_q_deg": theta_q, "per_rotation_quantum": d, "theta_t_deg": theta_t,
            "band_frac": b, "step_max_deg": step_max, "min_per_interval": int(min_per_interval),
            "frames": frames, "files": files, "witness_files": witness_files, "windows": windows}


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _inputs(scan_dir: Path) -> Dict[str, Path]:
    scan_dir = Path(scan_dir)
    return {"selected_frames": scan_dir / "frames" / "selected_frames.json",
            "coverage_warnings": scan_dir / "intake" / "coverage_warnings.json",
            "quality_features": scan_dir / "frames" / "quality_features.json"}


def session_turn_frames(scan_dir: Path, cfg: TurnFramesConfig, log=print) -> Dict[str, Any]:
    """The session's turn frames, written to ``<scan>/intake/turn_frames.json`` (atomic) and
    returned. Disabled → no frame (the document says so). Missing I0 / I1 products fail naming
    the file."""
    scan_dir = Path(scan_dir)
    paths = _inputs(scan_dir)
    for name, p in paths.items():
        if not p.exists():
            raise TurnFramesError(f"{p} is missing — the turn frames are read from intake I0/I1's "
                                  f"products ({name})")
    import repro
    head = {"version": TURN_FRAMES_VERSION, "provenance": PROVENANCE, "method": METHOD,
            "enabled": bool(cfg.enabled), "per_rotation_quantum": int(cfg.per_rotation_quantum),
            "min_per_interval": int(cfg.min_per_interval),
            "inputs": {f"{k}_sha256": _sha_file(p) for k, p in paths.items()},
            "code_sha256": repro.stamp(code=[__file__])["code"],
            "derivation": "a turn: n = ceil(r_k / (theta_t * (1 + b))) rotation steps between "
                          "consecutive keyframes, theta_t = theta_q / per_rotation_quantum; any other "
                          "interval: min_per_interval + 1 steps of its parallax reading p_k; step j = "
                          "the sharpest frame within +-b of a step around j * (r_k or p_k) / n "
                          "(I1's readings from the anchor)"}
    if not cfg.enabled:
        doc = {**head, "frames": [], "files": [], "witness_files": [], "windows": [],
               "n_turn_frames": 0, "n_witness_frames": 0}
        _write_json_atomic(scan_dir / "intake" / TURN_FRAMES_NAME, doc)
        log(f"[turn-frames] OFF ({CONFIG_PATH}.enabled false) — Omega sees the keyframes only")
        return doc
    sel = json.loads(paths["selected_frames"].read_text())
    cov = json.loads(paths["coverage_warnings"].read_text())
    qual = json.loads(paths["quality_features"].read_text())
    res = select_turn_frames(sel, cov.get("frames") or [], qual.get("frames") or [],
                             cfg.per_rotation_quantum, cfg.min_per_interval)
    by, by_reading = {}, {}
    for t in res["frames"]:
        by[t["chosen_by"]] = by.get(t["chosen_by"], 0) + 1
        by_reading[t["reading"]] = by_reading.get(t["reading"], 0) + 1
    skipped = {}
    for w in res["windows"]:
        if w.get("skipped"):
            skipped[w["skipped"]] = skipped.get(w["skipped"], 0) + 1
    doc = {**head, **res, "n_turn_frames": len(res["files"]),
           "n_witness_frames": len(res["witness_files"]),
           "n_windows_with_turn_frames": sum(1 for w in res["windows"]
                                             if w.get("n_turn_frames")),
           "chosen_by": by, "by_reading": by_reading, "skipped": skipped}
    _write_json_atomic(scan_dir / "intake" / TURN_FRAMES_NAME, doc)
    log(f"[turn-frames] {doc['n_turn_frames']} turn frame(s) — Omega's context views (θ_t = θ_q "
        f"{res['theta_q_deg']:.2f}° / {res['per_rotation_quantum']} = {res['theta_t_deg']:.2f}°, no "
        f"step over {res['step_max_deg']:.2f}°) — and {doc['n_witness_frames']} witness frame(s) "
        f"for F6 only (≥ {res['min_per_interval']} in every other interval, by its parallax), "
        f"{doc['n_windows_with_turn_frames']} window(s) of {len(sel['keyframes'])} keyframes; band "
        f"±{res['band_frac']:g}, chosen in the band {by.get('band', 0)}, nearest reading "
        f"{by.get('nearest', 0)}" + (f"; windows skipped {skipped}" if skipped else "")
        + f" → {TURN_FRAMES_NAME}")
    return doc


def _frame_no(name: str) -> int:
    return int(Path(name).stem)


def chunk_context(ranges: Sequence[Sequence[int]], keyframe_files: Sequence[str],
                  turn_files: Sequence[str]) -> List[List[str]]:
    """Per chunk [s, e) of the keyframe list ``keyframe_files``: the turn frames strictly
    between its first and its last keyframe, in frame order — the views Omega adds to that
    chunk's inference. A turn frame lies between two CONSECUTIVE keyframes, so it is in every
    chunk that holds both. A turn file that is a keyframe fails."""
    kf_no = [_frame_no(f) for f in keyframe_files]
    if kf_no != sorted(kf_no):
        raise TurnFramesError("the keyframe list is not in frame order")
    kf_set = set(kf_no)
    turns = sorted(set(turn_files), key=_frame_no)
    bad = [f for f in turns if _frame_no(f) in kf_set]
    if bad:
        raise TurnFramesError(f"turn frame(s) {bad[:5]} are keyframes — a keyframe is never a "
                              f"context view")
    out: List[List[str]] = []
    for s, e in ranges:
        s, e = int(s), int(e)
        if not 0 <= s < e <= len(kf_no):
            raise TurnFramesError(f"chunk [{s}, {e}) is outside the {len(kf_no)} keyframes")
        lo, hi = kf_no[s], kf_no[e - 1]
        out.append([f for f in turns if lo < _frame_no(f) < hi])
    return out

