"""Intake I3 → the metric WALK (claude_stac.txt §4-F2), measured BEFORE Omega.

DA3 runs jointly over overlapping windows of keyframes (``extract_da3_depth.py
--windows_json``); the NESTED model's window poses are metric (its multi-view
depth is aligned to its own metric branch by one least-squares factor per
window, the translations scaled by the same factor). Consecutive windows share
``window_frames × window_overlap_frac`` keyframes: each window is placed in the
frame of the chain by the rigid motion its shared frames agree on (rotation =
chordal mean of the per-frame rotations, translation = mean of the per-frame
translations), so the chained trajectory carries only LOCAL error — measured,
per seam, as the disagreement of the shared cameras' centres after placement.

The walk is the sum of |c_{k+1} − c_k| over the chained keyframe centres; the
chainage of every keyframe is written to ``<session>/intake/walk.json``. The
windows are what the chunk plan is measured on (I4, the co-visibility plan,
``reconstruction.chunk_covis.plan_session``, USER 2026-10-06); the walk an Omega
pass measures is evidence and never decides — a single Omega
pass over pccr 2026-08-24 read 1526.6 m over a walk its chunked run measured at
104.8 m.

Every window also yields the per-frame metric anchors the scale stages read
(``da3_run/results_output/frame_<num>.npz``: depth, conf, intrinsics), each
frame taken from the window where it sits most centrally — one DA3 round.

REPRODUCIBLE (docs/plan_determinismo.md points 4, 21, 26, 27, 41, 42, 44, 2026-10-07):
``windows.json`` is the plan AND what it depends on — the windows, process_res, model, the
window size from the committed card table with its numbers (``window_sizing``) and the DA3
identity of the extracting interpreter (``da3_environment``: pinned weights, extractor and DA3
code, torch / CUDA / cuDNN, card, driver, numerics — ``extract_da3_depth.py --identity``). The
card must be free before DA3 runs (repro.require_exclusive_gpu) and a window that does not fit
FAILS — never halved. ``walk.json`` records the spec it was measured on (its sha256), every
window's content sha256 and the reference view DA3 picked: a REGENERATION of deleted window
files (F2, a resume of the co-visibility plan) must be of that exact spec and reproduce those
exact windows, or it fails and says what differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

WALK_NAME = "walk.json"
WINDOWS_DIRNAME = "da3_windows"
REFERENCE_VIEWS_NAME = "reference_views.json"   # extract_da3_depth.REFERENCE_VIEWS_NAME
WALK_VERSION = 2          # 2: the spec sha, the DA3 identity and every window's content sha
PROVENANCE = "tool_measured"
LOG_TAG = "[intake.walk]"


class WalkError(RuntimeError):
    """A structural impossibility of the walk (no keyframes, a window missing,
    two windows that share too few frames) — always with the exact reason."""


# ── windows ──────────────────────────────────────────────────────────────

def plan_windows(n: int, window_frames: int, overlap_frac: float) -> List[Tuple[int, int]]:
    """[start, end) keyframe-index windows of ``window_frames`` with
    ``round(window_frames × overlap_frac)`` frames shared between consecutive
    ones; the last window ends at ``n`` (shifted back so it keeps its size) —
    every keyframe lies in at least one window and every pair of consecutive
    windows shares at least the planned overlap."""
    if n < 1:
        raise WalkError("no keyframes to window")
    w = int(window_frames)
    ov = int(round(w * overlap_frac))
    if w < 4 or ov < 2 or ov >= w:
        raise WalkError(f"window of {w} with overlap {ov} cannot chain (need 2 ≤ overlap < window)")
    if n <= w:
        return [(0, n)]
    step = w - ov
    out = []
    s = 0
    while True:
        e = s + w
        if e >= n:
            out.append((max(0, n - w), n))
            break
        out.append((s, e))
        s += step
    # the shifted last window may coincide with the previous one
    dedup = []
    for r in out:
        if not dedup or r != dedup[-1]:
            dedup.append(r)
    return dedup


def _to_c2w(ext: np.ndarray) -> np.ndarray:
    """(N, 3|4, 4) world-to-camera (DA3's convention, api.py) → (N, 4, 4) camera-to-world."""
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape[-2] == 3:
        pad = np.zeros(ext.shape[:-2] + (1, 4))
        pad[..., 0, 3] = 1.0
        ext = np.concatenate([ext, pad], axis=-2)
    R = ext[:, :3, :3]
    t = ext[:, :3, 3]
    out = np.tile(np.eye(4), (len(ext), 1, 1))
    out[:, :3, :3] = np.transpose(R, (0, 2, 1))
    out[:, :3, 3] = -np.einsum("nji,nj->ni", R, t)
    return out


def load_window(path: Path) -> Dict[str, Any]:
    with np.load(path) as z:
        return {"frames": [int(f) for f in z["frames"]],
                "c2w": _to_c2w(z["extrinsics"]),
                "scale_factor": float(z["scale_factor"]),
                "is_metric": int(z["is_metric"]),
                "path": str(path)}


def window_content_sha256(path: Path) -> str:
    """sha256 of a window file's ARRAYS (name, dtype, shape, bytes — sorted by name): the npz's
    own bytes carry zip timestamps, its arrays are what the walk and the gauge read."""
    import hashlib
    h = hashlib.sha256()
    with np.load(path) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}\0{a.dtype.str}\0{list(a.shape)}\0".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def spec_sha256(spec: Dict[str, Any]) -> str:
    """The identity of a windows plan (canonical JSON — the extractor keys its recorded reference
    views by the same value)."""
    import repro
    return repro.sha256_json(spec)


# ── chaining ─────────────────────────────────────────────────────────────

def _chordal_mean(Rs: Sequence[np.ndarray]) -> np.ndarray:
    U, _s, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def chain_windows(windows: Sequence[Dict[str, Any]]) -> Tuple[Dict[int, np.ndarray],
                                                              List[Dict[str, Any]]]:
    """Every window placed in the first window's frame by the rigid motion its
    frames shared with the chain agree on. Returns ({frame: c2w}, seams): a
    frame keeps the placement of the FIRST window that holds it; each seam
    records the shared count, the centres' disagreement after placement (median
    and max, m — the chain's local error) and the ratio of the shared cameras'
    spans (chain / window — ~1 when both are metric; reported, not applied)."""
    if not windows:
        raise WalkError("no DA3 window to chain")
    poses: Dict[int, np.ndarray] = {}
    for f, T in zip(windows[0]["frames"], windows[0]["c2w"]):
        poses[f] = T
    seams: List[Dict[str, Any]] = []
    for k in range(1, len(windows)):
        w = windows[k]
        local = dict(zip(w["frames"], w["c2w"]))
        shared = [f for f in w["frames"] if f in poses]
        if len(shared) < 2:
            raise WalkError(f"window {k} shares {len(shared)} frame(s) with the chain — it "
                            f"cannot be placed (window_overlap_frac too small?)")
        Ms = [poses[f] @ np.linalg.inv(local[f]) for f in shared]
        R = _chordal_mean([M[:3, :3] for M in Ms])
        t = np.mean([poses[f][:3, 3] - R @ local[f][:3, 3] for f in shared], axis=0)
        G = np.eye(4)
        G[:3, :3] = R
        G[:3, 3] = t
        cg = np.array([poses[f][:3, 3] for f in shared])
        cw = np.array([(G @ local[f])[:3, 3] for f in shared])
        dis = np.linalg.norm(cg - cw, axis=1)
        span_g = float(np.linalg.norm(cg[-1] - cg[0]))
        span_w = float(np.linalg.norm(np.array([local[f][:3, 3] for f in shared])[-1]
                                      - np.array([local[f][:3, 3] for f in shared])[0]))
        seams.append({"window": k, "n_shared": len(shared),
                      "centre_disagreement_median_m": float(np.median(dis)),
                      "centre_disagreement_max_m": float(dis.max()),
                      "span_ratio": (span_g / span_w) if span_w > 0 else None})
        for f in w["frames"]:
            if f not in poses:
                poses[f] = G @ local[f]
    return poses, seams


def walk_of(frames: Sequence[int], poses: Dict[int, np.ndarray]) -> Tuple[float, List[float]]:
    """(walk length, chainage per frame) over the frames in order (m)."""
    c = np.array([poses[f][:3, 3] for f in frames])
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1) if len(c) > 1 else np.zeros(0)
    chain = np.concatenate([[0.0], np.cumsum(steps)])
    return float(chain[-1]), [float(x) for x in chain]


# ── anchors ──────────────────────────────────────────────────────────────

def write_anchors(windows_dir: Path, windows: Sequence[Dict[str, Any]],
                  anchors_dir: Path) -> int:
    """``frame_<num>.npz`` (depth, conf, intrinsics) for every keyframe, from the
    window where it sits most centrally. Returns the number written."""
    best: Dict[int, Tuple[float, int, int]] = {}
    for k, w in enumerate(windows):
        n = len(w["frames"])
        for i, f in enumerate(w["frames"]):
            d = abs(i - (n - 1) / 2.0)
            if f not in best or d < best[f][0]:
                best[f] = (d, k, i)
    anchors_dir.mkdir(parents=True, exist_ok=True)
    by_window: Dict[int, List[Tuple[int, int]]] = {}
    for f, (_d, k, i) in best.items():
        by_window.setdefault(k, []).append((f, i))
    for k, items in by_window.items():
        with np.load(windows[k]["path"]) as z:
            depth, conf, K = z["depth"], z["conf"], z["intrinsics"]
            for f, i in items:
                np.savez_compressed(anchors_dir / f"frame_{f}.npz",
                                    depth=depth[i].astype(np.float32),
                                    conf=conf[i].astype(np.float32),
                                    intrinsics=K[i].astype(np.float64))
    return len(best)


# ── run ──────────────────────────────────────────────────────────────────

def keyframe_files(frames_dir: Path) -> List[str]:
    p = Path(frames_dir) / "selected_frames.json"
    if not p.exists():
        raise WalkError(f"{p} does not exist — the walk windows the KEYFRAMES")
    doc = json.loads(p.read_text())
    files = doc.get("selected_files") if isinstance(doc, dict) else doc
    if not files:
        raise WalkError(f"{p} lists no keyframe")
    return sorted(files, key=lambda f: int("".join(ch for ch in Path(f).stem if ch.isdigit())))


DA3_PATCH = 14


def da3_process_res(value, frames_dir: Path) -> int:
    """DA3's process_res: an int as configured, or ``native`` — the frames' long side
    rounded up to DA3's patch (USER 2026-09-28: maximum resolution; DA3 upper-bound-
    resizes the long side, so this is the native frame and never an upsampling)."""
    if value != "native":
        return int(value)
    import cv2
    from intake.quality import list_frames
    paths = list_frames(Path(frames_dir))
    if not paths:
        raise WalkError(f"no frame in {frames_dir} — the native resolution cannot be read")
    img = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise WalkError(f"cannot read {paths[0]}")
    return int(-(-max(img.shape[:2]) // DA3_PATCH) * DA3_PATCH)


def da3_identity(python: str, model_id: str, log: Callable = print) -> Dict[str, Any]:
    """The DA3 identity of ``python`` (the interpreter that will extract): what a window depends
    on beyond its frames, from ``extract_da3_depth.py --identity`` in a subprocess (it touches
    CUDA there and exits — this process never holds a context on the card)."""
    import da3_weights
    import repro
    server_dir = Path(__file__).resolve().parent.parent
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--identity", "--model",
           str(model_id)]
    env = da3_weights.hf_env(repro.deterministic_env())
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=repro.PROBE_TIMEOUT_S)
    tag = "[DA3 identity] "
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag)]
    if proc.returncode != 0 or len(lines) != 1:
        tail = (proc.stderr or proc.stdout).strip()[-800:]
        raise WalkError(f"the DA3 identity probe failed (exit {proc.returncode}): {tail}")
    return json.loads(lines[0][len(tag):])


def intake_config_for(session_dir: Path, run_cfg: Optional[Dict[str, Any]], log: Callable = print):
    """(IntakeConfig, frozen sha256 | None) I3 sizes its windows with — docs/plan_determinismo.md
    point 69: NEVER ``config.yaml`` re-read from disk when the worker spawns. The session's FROZEN
    ``output/run_config.yaml`` when there is one (verified; a ``run_cfg`` handed by the worker must
    agree with it — a difference FAILS naming the fields); else the handed ``run_cfg`` (the job's
    own snapshot); else — a command-line job — the server configuration frozen NOW, this being the
    job's start (intake.run_config.cli_intake_config)."""
    from intake.config import load_intake_config
    from intake.run_config import (cli_intake_config, effective_config, has_run_config,
                                   load_run_config, verify_intake_config)
    session_dir = Path(session_dir)
    if has_run_config(session_dir):
        frozen, sha = load_run_config(session_dir)
        icfg = load_intake_config(effective_config(frozen, "reconstruction"))
        if run_cfg is not None:
            verify_intake_config(session_dir, load_intake_config(run_cfg), "reconstruction")
        return icfg, sha
    if run_cfg is not None:
        return load_intake_config(run_cfg), None
    return cli_intake_config(session_dir, log=log)


def planned_spec(session_dir: Path, gcfg, python: str, log: Callable = print, *,
                 frames_dir: Optional[Path] = None, files: Optional[Sequence[str]] = None,
                 run_cfg: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Dict[str, Any], List[str], Path]:
    """The windows plan this run would extract — (spec, keyframe files, frames dir): the windows,
    process_res and model, the window size from the committed card table (with every number) and
    the DA3 identity of ``python``. Deterministic: the same frames, config, card, weights and code
    give the same spec. ``run_cfg``: the job's configuration dict when the caller holds it (the
    worker); see :func:`intake_config_for`."""
    from intake.vram import window_size
    session_dir = Path(session_dir)
    frames_dir = Path(frames_dir) if frames_dir is not None else session_dir / "frames"
    files = (sorted(files, key=lambda f: int("".join(ch for ch in Path(f).stem
                                                     if ch.isdigit())))
             if files else keyframe_files(frames_dir))
    res = da3_process_res(gcfg.process_res, frames_dir)
    icfg, _rc_sha = intake_config_for(session_dir, run_cfg, log)
    import cv2
    _img = cv2.imread(str(frames_dir / files[0]), cv2.IMREAD_UNCHANGED)
    if _img is None:
        raise WalkError(f"cannot read {frames_dir / files[0]}")
    native_wh = (int(_img.shape[1]), int(_img.shape[0]))
    ident = da3_identity(python, gcfg.model_id, log)
    # the window the CARD holds at this (native) resolution — intake/vram.py, the committed card
    # table; the overlap keeps its share, so consecutive windows still chain through shared frames
    sizing = window_size(gcfg.model_id, res, native_wh, requested=int(gcfg.window_frames),
                         margin_frac=float(icfg.parallax.vram_margin_frac),
                         card_key=str(ident["card"]), log=log)
    w_frames = int(sizing["window_frames"])
    plan = plan_windows(len(files), w_frames, gcfg.window_overlap_frac)
    spec = {"windows": [[str(frames_dir / f) for f in files[a:b]] for a, b in plan],
            "process_res": res, "model_id": gcfg.model_id,
            "window_sizing": sizing, "da3_environment": ident}
    return spec, list(files), frames_dir


def _walk_spec_check(session_dir: Path, spec: Dict[str, Any]) -> Optional[str]:
    """None when walk.json was measured on exactly ``spec``; otherwise why not."""
    walk = load_walk(session_dir)
    if walk is None:
        return "there is no walk.json"
    if int(walk.get("version", 0)) != WALK_VERSION or not walk.get("windows_spec_sha256"):
        return (f"walk.json (version {walk.get('version')}) predates the window stamps "
                f"(version {WALK_VERSION})")
    if walk["windows_spec_sha256"] != spec_sha256(spec):
        old_env = walk.get("da3_environment") or {}
        new_env = spec.get("da3_environment") or {}
        what = sorted(k for k in set(old_env) | set(new_env) if old_env.get(k) != new_env.get(k))
        old_sz = (walk.get("window_sizing") or {}).get("window_frames")
        return (f"walk.json was measured on another windows plan (DA3 environment differs in "
                f"{what or 'nothing'}; window {old_sz} → "
                f"{(spec.get('window_sizing') or {}).get('window_frames')} frames)")
    return None


def run_da3_windows(session_dir: Path, gcfg, python: str, log: Callable = print,
                    check_cancel: Optional[Callable[[], bool]] = None, *,
                    frames_dir: Optional[Path] = None, files: Optional[Sequence[str]] = None,
                    for_new_walk: bool = False,
                    run_cfg: Optional[Dict[str, Any]] = None) -> Tuple[Path, List[List[str]]]:
    """I3: the DA3 multi-view windows over the keyframes (GPU, in ``python`` — the da3 env).
    ``files`` (keyframe basenames in ``frames_dir``) default to
    ``<session>/frames/selected_frames.json``.

    ``for_new_walk=True`` (I3 itself, followed by measure_walk): the windows of the plan this
    run computes; another plan's window files are deleted first. Otherwise (F2, a co-visibility
    resume — a REGENERATION of deleted window files): the plan must be the one walk.json was
    measured on, and every regenerated window must reproduce the window the walk read (content
    sha256 and DA3's reference view) — or this FAILS and says what differs (plan points 26, 42).

    The card is checked FREE before DA3 starts (point 4) and a window that does not fit fails —
    never halved. Window files on disk are reused only when every one carries this run's stamp
    (the extractor; point 44). Returns (windows dir, windows)."""
    import da3_weights
    import repro
    session_dir = Path(session_dir)
    spec, files, frames_dir = planned_spec(session_dir, gcfg, python, log,
                                           frames_dir=frames_dir, files=files, run_cfg=run_cfg)
    windows = spec["windows"]
    walk = load_walk(session_dir)
    if not for_new_walk and walk is None:
        # nothing to regenerate: the session has no walk yet, so this IS its first I3 (the by-hand
        # `python -m precision.gauge` runs I3 + the walk when missing) — declared, not refused
        log(f"{LOG_TAG} no walk.json in this session: a first I3, not a regeneration — the "
            f"windows of this run's plan are extracted and the walk follows")
        for_new_walk = True
    if not for_new_walk:
        why = _walk_spec_check(session_dir, spec)
        if why is not None:
            raise WalkError(f"the I3 windows cannot be regenerated for this walk: {why} — re-run "
                            f"I3 (the windows AND the walk) instead of mixing two plans")
    elif walk is not None:
        old_card = (walk.get("da3_environment") or {}).get("card")
        if old_card is not None and old_card != spec["da3_environment"]["card"]:
            try:
                repro.parse_card_key(old_card)
                old_format = False
            except repro.ReproError:
                old_format = True          # keyed on torch's bytes before 2026-10-07
            if old_format:
                log(f"{LOG_TAG} the previous walk recorded its card under the old key "
                    f"({old_card}: torch's usable bytes, not the model) — this run's card is "
                    f"{spec['da3_environment']['card']}; window {spec['window_sizing']['window_frames']} "
                    f"frames from the card table (declared)")
            else:
                log(f"{LOG_TAG} ⚠ the card changed: the previous walk was measured on {old_card}, "
                    f"this one runs on {spec['da3_environment']['card']} — its window size is the "
                    f"new card's ({spec['window_sizing']['window_frames']} frames; declared)")
    wdir = session_dir / "output" / WINDOWS_DIRNAME
    wdir.mkdir(parents=True, exist_ok=True)
    # the per-session footprint cache of the old sizing (output/intake/da3_vram.json): nothing
    # reads it since 2026-10-07 (the size is the committed card table's) — a leftover goes, said
    stale_cache = session_dir / "output" / "intake" / "da3_vram.json"
    if stale_cache.exists():
        stale_cache.unlink()
        log(f"{LOG_TAG} {stale_cache.relative_to(session_dir)} deleted: the window size is the "
            f"committed card table's (server/card_table.json), no run measures or caches it")
    spec_path = wdir / "windows.json"
    old = json.loads(spec_path.read_text()) if spec_path.exists() else None
    if old != spec:
        for p in wdir.glob("window_*.npz"):            # another plan: its windows are not ours
            p.unlink()
        (wdir / REFERENCE_VIEWS_NAME).unlink(missing_ok=True)
        spec_path.write_text(json.dumps(spec))
    w_frames = int(spec["window_sizing"]["window_frames"])
    server_dir = Path(__file__).resolve().parent.parent
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--image_dir",
           str(frames_dir), "--output_dir", str(wdir), "--model", gcfg.model_id,
           "--process_res", str(spec["process_res"]), "--windows_json", str(spec_path)]
    log(f"{LOG_TAG} I3: DA3 {gcfg.model_id} over {len(windows)} window(s) of "
        f"{w_frames} keyframes ({len(files)} keyframes, overlap "
        f"{gcfg.window_overlap_frac:g}; configured {gcfg.window_frames})")
    repro.require_exclusive_gpu(log=log)                # point 4: nothing else on the card
    env = da3_weights.hf_env(repro.deterministic_env())
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env)
    for line in proc.stdout:
        line = line.strip()
        if line:
            log(line)
        if check_cancel is not None and check_cancel():
            proc.terminate()
            raise WalkError("cancelled")
    proc.wait()
    from intake.vram import IDENTITY_EXIT, OOM_EXIT, REF_VIEW_EXIT
    if proc.returncode == OOM_EXIT:
        raise WalkError(f"a window of {w_frames} keyframes did not fit "
                        f"{spec['da3_environment']['card']} although the card table sizes it "
                        f"to (predicted peak {spec['window_sizing']['predicted_peak_gib']:.1f} GiB) "
                        f"— nothing is halved (another window size is another walk): re-measure "
                        f"the card's entry or free the card")
    if proc.returncode == REF_VIEW_EXIT:
        raise WalkError("a regenerated DA3 window picked another reference view than the one "
                        "recorded — it is not the window the walk was measured on (see the line "
                        "above); re-run I3")
    if proc.returncode == IDENTITY_EXIT:
        raise WalkError("the extracting interpreter is not the DA3 environment windows.json was "
                        "planned for (see the line above)")
    if proc.returncode != 0:
        raise WalkError(f"DA3 window extraction exited with code {proc.returncode}")
    if not for_new_walk:
        rec = {int(w["index"]): w for w in walk.get("windows", []) if "index" in w}
        for i in range(len(windows)):
            p = wdir / f"window_{i:04d}.npz"
            want = (rec.get(i) or {}).get("content_sha256")
            got = window_content_sha256(p)
            if want != got:
                raise WalkError(f"the regenerated {p.name} is not the window walk.json was "
                                f"measured on (content sha256 {str(want)[:12]} → {got[:12]}) — "
                                f"the DA3 extraction does not reproduce on this stack; re-run I3")
        log(f"{LOG_TAG} regenerated {len(windows)} window(s): every one bit-identical to the "
            f"windows walk.json was measured on")
    return wdir, windows


def walk_is_current(session_dir: Path, files: Sequence[str], frames_dir: Path, gcfg,
                    python: str, log: Callable = print,
                    run_cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True when ``intake/walk.json`` was measured on EXACTLY the window plan this run would
    extract (same keyframes, window size from the card table, model, resolution AND DA3
    identity — weights, code, torch, card) and the revisit reference is on disk — the walk, the
    anchors and the revisit reference are then reused and the window depth files are NOT
    regenerated (USER 2026-10-06: a resume regenerated 153 windows, 20 min of GPU, only because
    their files had been deleted to save disk). F2 regenerates the files when IT needs them
    (run_da3_windows, which then verifies they reproduce the walk's)."""
    session_dir = Path(session_dir)
    walk = load_walk(session_dir)
    spec_path = session_dir / "output" / WINDOWS_DIRNAME / "windows.json"
    ref = session_dir / "output" / REVISIT_REFERENCE_NAME
    if walk is None or not spec_path.exists() or not ref.exists():
        return False
    try:
        old = json.loads(spec_path.read_text())
        ref_doc = json.loads(ref.read_text())
    except (OSError, ValueError):
        return False
    spec, files, _fd = planned_spec(session_dir, gcfg, python, log=lambda *_: None,
                                    frames_dir=frames_dir, files=files, run_cfg=run_cfg)
    why = _walk_spec_check(session_dir, spec)
    if why is None and not revisit_reference_is_of(ref_doc, spec):
        why = (f"{REVISIT_REFERENCE_NAME} was measured on another windows plan (or predates the "
               f"stamp, version {ref_doc.get('version')})")
    same = (old == spec and why is None and int(walk.get("n_keyframes", -1)) == len(files)
            and int(walk.get("n_windows", -1)) == len(spec["windows"]))
    if same:
        log(f"{LOG_TAG} I3 reused: walk.json measured on this exact plan "
            f"({len(spec['windows'])} windows of {spec['window_sizing']['window_frames']} "
            f"keyframes, {len(files)} keyframes, same DA3 identity) — the window depth is not "
            f"regenerated")
    elif why is not None:
        log(f"{LOG_TAG} I3 not reusable: {why}")
    return same


def delete_windows(session_dir: Path, log: Callable = print) -> int:
    """Remove the window depth files (``window_*.npz``) of ``output/da3_windows`` — F0's only
    reader is through; ``windows.json`` (the plan), ``walk.json`` and the anchors stay, so
    ``run_da3_windows`` regenerates exactly these windows when a step needs them again. Returns
    the bytes freed."""
    wdir = Path(session_dir) / "output" / WINDOWS_DIRNAME
    freed = 0
    for p in sorted(wdir.glob("window_*.npz")) if wdir.is_dir() else []:
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError as e:
            log(f"{LOG_TAG} could not delete {p.name}: {e}")
    if freed:
        log(f"{LOG_TAG} deleted the I3 window depth files ({freed / 1e9:.1f} GB; windows.json, "
            f"walk.json and the anchors stay — regenerated on demand)")
    return freed


def measure_walk(session_dir: Path, gcfg, log: Callable = print, *,
                 geometry_epoch: int = 0, camera_epoch: int = 0) -> Dict[str, Any]:
    """Chain the I3 windows, write ``intake/walk.json`` and the per-frame anchors.
    Returns the walk document."""
    session_dir = Path(session_dir)
    wdir = session_dir / "output" / WINDOWS_DIRNAME
    spec = json.loads((wdir / "windows.json").read_text())
    windows = []
    for i in range(len(spec["windows"])):
        p = wdir / f"window_{i:04d}.npz"
        if not p.exists():
            raise WalkError(f"{p} is missing — run I3 (the DA3 windows) first")
        windows.append(load_window(p))
    poses, seams = chain_windows(windows)
    frames = sorted(poses)
    walk_m, chainage = walk_of(frames, poses)
    n_anchor = write_anchors(wdir, windows, session_dir / "output" / "da3_run" / "results_output")
    dis = [s["centre_disagreement_median_m"] for s in seams]
    import repro
    win_rec = []
    for i, w in enumerate(windows):
        with np.load(w["path"]) as z:
            refs = [int(x) for x in z["ref_views"]] if "ref_views" in z.files else None
            stamp_i = str(z["stamp"]) if "stamp" in z.files else None
        win_rec.append({"index": i, "frames": [w["frames"][0], w["frames"][-1]],
                        "n": len(w["frames"]), "scale_factor": w["scale_factor"],
                        "is_metric": w["is_metric"], "reference_views": refs, "stamp": stamp_i,
                        "content_sha256": window_content_sha256(Path(w["path"]))})
    doc = {
        "version": WALK_VERSION,
        "provenance": PROVENANCE,
        "geometry_epoch": int(geometry_epoch),
        "camera_epoch": int(camera_epoch),
        "method": "da3_windows_chained",
        "params": {"window_frames": gcfg.window_frames,
                   "window_overlap_frac": gcfg.window_overlap_frac,
                   "process_res": spec["process_res"],
                   "model_id": gcfg.model_id},
        # what the walk was measured on (point 21): the windows plan's sha256, its DA3 identity
        # and window sizing, the code that chained it
        "windows_spec_sha256": spec_sha256(spec),
        "da3_environment": spec.get("da3_environment"),
        "window_sizing": spec.get("window_sizing"),
        "code": repro.stamp(code=[__file__])["code"],
        # the job's frozen configuration this walk belongs to (point 69; None on a session whose
        # job predates the frozen copy)
        "run_config_sha256": _frozen_config_sha(session_dir),
        "n_keyframes": len(frames),
        "n_windows": len(windows),
        "all_windows_metric": all(w["is_metric"] == 1 for w in windows),
        "walk_length_m": walk_m,
        "local_error_median_m": float(np.median(dis)) if dis else 0.0,
        "local_error_max_m": float(max(dis)) if dis else 0.0,
        "chainage": [{"frame": f, "chainage_m": c} for f, c in zip(frames, chainage)],
        "windows": win_rec,
        "seams": seams,
        "n_anchors_written": n_anchor,
    }
    out = session_dir / "intake" / WALK_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out)
    log(f"{LOG_TAG} walk {walk_m:.1f} m over {len(frames)} keyframes, {len(windows)} "
        f"window(s); seam disagreement median {doc['local_error_median_m'] * 100:.1f} cm, "
        f"max {doc['local_error_max_m'] * 100:.1f} cm; {n_anchor} anchor(s) → {out}")
    if not doc["all_windows_metric"]:
        log(f"{LOG_TAG} WARNING: a DA3 window came back up-to-scale (is_metric 0) — its "
            f"poses are chained at the chain's scale only through the shared frames")
    return doc


REVISIT_REFERENCE_NAME = "salad_revisit_reference.json"


def revisit_reference(session_dir: Path) -> Dict[str, Any]:
    """The GEOMETRIC revisits the loop detector's appearance bar is calibrated on
    (``LoopModels.LoopModel.calibrate_threshold``): the I3 windows chained into one
    metric trajectory give every keyframe's camera centre and viewing direction;
    two keyframes see the same place when their cameras stand closer than the
    scene's median depth (every window's median DA3 depth, median over the
    windows) AND look less than half the field of view apart (from the windows'
    own K) — both bars measured on the session. Written to
    ``output/salad_revisit_reference.json``; returns it.

    Why: SALAD's fixed 0.65 (measured on pccr 2026-08-31, 216 keyframes) proposed
    six pairs on pccr 2026-08-24 (746 keyframes), all 11-13 keyframes apart,
    while this trajectory held 26 revisit episodes > 10 m of walk apart."""
    session_dir = Path(session_dir)
    wdir = session_dir / "output" / WINDOWS_DIRNAME
    spec = json.loads((wdir / "windows.json").read_text())
    windows = [load_window(wdir / f"window_{i:04d}.npz") for i in range(len(spec["windows"]))]
    poses, _seams = chain_windows(windows)
    frames = sorted(poses)
    med_depth, hfov_w = [], []          # per WINDOW: its median depth, its median half-FOV input
    for w in windows:
        with np.load(w["path"]) as z:
            d, c, K = z["depth"], z["conf"], z["intrinsics"]
            valid = (c > 0) & np.isfinite(d) & (d > 0)
            if valid.any():
                med_depth.append(float(np.median(d[valid])))
            width = d.shape[-1]
            hfov_w.append(float(np.median(2.0 * np.arctan(width / (2.0 * K[:, 0, 0])))))
    if not med_depth or not hfov_w:
        raise WalkError("the I3 windows carry no valid depth — no revisit reference")
    import repro
    # DECIDIDO (docs/plan_determinismo.md point 71, 2026-10-07): each bar is a median over the
    # windows and carries its MEASURED error — the standard deviation of that median under a
    # fixed-key bootstrap over the windows (n_boot 2000, seed 0: the repo's one bootstrap
    # convention, loop_utils/metric_lock.heldout_change / decide_change). The calibration
    # leaves pairs within error_factor x the error of either bar out of both classes; the
    # factor is THE USER's (point 1), read from the job's FROZEN configuration (point 69).
    from intake.run_config import effective_config, load_run_config
    from reconstruction.loops.config import improvement_error_factor
    frozen, _rc_sha = load_run_config(session_dir)
    factor = float(improvement_error_factor(effective_config(frozen, "reconstruction")))
    rng = np.random.default_rng(BAR_BOOTSTRAP_SEED)
    md, hw = np.asarray(med_depth, np.float64), np.asarray(hfov_w, np.float64)
    pick = rng.integers(0, md.size, size=(BAR_BOOTSTRAP_N, md.size))
    pick_h = rng.integers(0, hw.size, size=(BAR_BOOTSTRAP_N, hw.size))
    dist_err = float(np.std(np.median(md[pick], axis=1)))
    cos_err = float(np.std(np.cos(0.5 * np.median(hw[pick_h], axis=1))))
    doc = {"version": REVISIT_REFERENCE_VERSION, "provenance": PROVENANCE,
           "frames": [int(f) for f in frames],
           "centres": [poses[f][:3, 3].tolist() for f in frames],
           "forward": [poses[f][:3, 2].tolist() for f in frames],
           "dist_bar_m": float(np.median(md)),
           "cos_bar": float(np.cos(0.5 * float(np.median(hw)))),       # half the FOV
           "hfov_rad": float(np.median(hw)),
           "dist_bar_err_m": dist_err, "cos_bar_err": cos_err,
           "error_factor": factor,
           "bar_bootstrap": {"n_windows": int(md.size), "n_boot": BAR_BOOTSTRAP_N,
                             "seed": BAR_BOOTSTRAP_SEED},
           # what it was measured on (docs/plan_determinismo.md point 16: the SALAD bar is
           # calibrated on this reference; it is reused only for the exact windows plan — the
           # fork stamps the file's bytes into its loop stamp, point 8)
           "windows_spec_sha256": spec_sha256(spec),
           "code": repro.stamp(code=[__file__])["code"]}
    out = session_dir / "output" / REVISIT_REFERENCE_NAME
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, out)
    return doc


REVISIT_REFERENCE_VERSION = 3       # 2: stamped with the windows plan it was measured on
                                    # 3: each bar with its measured error + the user's error
                                    #    factor (plan point 71, 2026-10-07)
# the repo's fixed-key bootstrap (loop_utils/metric_lock.heldout_change / decide_change)
BAR_BOOTSTRAP_N = 2000
BAR_BOOTSTRAP_SEED = 0


def revisit_reference_is_of(doc: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    """True when a revisit reference was measured on exactly the windows plan ``spec``."""
    return (isinstance(doc, dict) and int(doc.get("version", 0)) == REVISIT_REFERENCE_VERSION
            and doc.get("windows_spec_sha256") == spec_sha256(spec))


def load_walk(session_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(session_dir) / "intake" / WALK_NAME
    return json.loads(p.read_text()) if p.exists() else None


def _frozen_config_sha(session_dir: Path) -> Optional[str]:
    from intake.run_config import has_run_config, load_run_config
    if not has_run_config(session_dir):
        return None
    return load_run_config(session_dir)[1]


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m intake.walk",
                                 description="I3 DA3 windows (if missing) + the metric walk.")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    g = load_precision_config().gauge
    run_da3_windows(Path(args.session), g, sys.executable, for_new_walk=True)
    measure_walk(Path(args.session), g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
