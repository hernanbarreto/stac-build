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
chainage of every keyframe is written to ``<session>/intake/walk.json``. THIS is
the walk that sizes the chunks (I4, ``reconstruction.chunk_plan.plan_chunks``);
the one an Omega pass measures is evidence and never decides — a single Omega
pass over pccr 2026-08-24 read 1526.6 m over a walk its chunked run measured at
104.8 m.

Every window also yields the per-frame metric anchors the scale stages read
(``da3_run/results_output/frame_<num>.npz``: depth, conf, intrinsics), each
frame taken from the window where it sits most centrally — one DA3 round.
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
WALK_VERSION = 1
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


def run_da3_windows(session_dir: Path, gcfg, python: str, log: Callable = print,
                    check_cancel: Optional[Callable[[], bool]] = None, *,
                    frames_dir: Optional[Path] = None, files: Optional[Sequence[str]] = None
                    ) -> Tuple[Path, List[List[str]]]:
    """I3: the DA3 multi-view windows over the keyframes (GPU, in ``python`` — the da3
    env). ``files`` (keyframe basenames in ``frames_dir``) default to
    ``<session>/frames/selected_frames.json``. Existing window files of the same plan
    are reused (resume). Returns (windows dir, windows)."""
    session_dir = Path(session_dir)
    frames_dir = Path(frames_dir) if frames_dir is not None else session_dir / "frames"
    files = (sorted(files, key=lambda f: int("".join(ch for ch in Path(f).stem
                                                     if ch.isdigit())))
             if files else keyframe_files(frames_dir))
    res = da3_process_res(gcfg.process_res, frames_dir)
    wdir = session_dir / "output" / WINDOWS_DIRNAME
    wdir.mkdir(parents=True, exist_ok=True)
    # the window the CARD holds at this (native) resolution — intake/vram.py; the overlap keeps
    # its share, so consecutive windows still chain through shared frames
    from intake.vram import OOM_EXIT, window_size
    from intake.config import load_intake_config
    from config import cfg as _raw_cfg
    icfg = load_intake_config(_raw_cfg)
    import cv2
    _img = cv2.imread(str(frames_dir / files[0]), cv2.IMREAD_UNCHANGED)
    if _img is None:
        raise WalkError(f"cannot read {frames_dir / files[0]}")
    native_wh = (int(_img.shape[1]), int(_img.shape[0]))
    w_frames = window_size(session_dir, files, frames_dir, gcfg.model_id, res, native_wh, python,
                           requested=int(gcfg.window_frames), calibration_frames=int(icfg.parallax.vram_calibration_frames),
                           margin_frac=float(icfg.parallax.vram_margin_frac), log=log, cancelled=check_cancel)
    server_dir = Path(__file__).resolve().parent.parent
    while True:
        plan = plan_windows(len(files), w_frames, gcfg.window_overlap_frac)
        windows = [[str(frames_dir / f) for f in files[a:b]] for a, b in plan]
        spec = {"windows": windows, "process_res": res, "model_id": gcfg.model_id}
        spec_path = wdir / "windows.json"
        old = json.loads(spec_path.read_text()) if spec_path.exists() else None
        if old != spec:
            for p in wdir.glob("window_*.npz"):        # another plan: its windows are not ours
                p.unlink()
            spec_path.write_text(json.dumps(spec))
        cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--image_dir",
               str(frames_dir), "--output_dir", str(wdir), "--model", gcfg.model_id,
               "--process_res", str(res), "--windows_json", str(spec_path)]
        log(f"{LOG_TAG} I3: DA3 {gcfg.model_id} over {len(windows)} window(s) of "
            f"{w_frames} keyframes ({len(files)} keyframes, overlap "
            f"{gcfg.window_overlap_frac:g}; configured {gcfg.window_frames})")
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")    # deterministic cuBLAS
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
        if proc.returncode == OOM_EXIT and w_frames > 2:
            w_frames = max(2, w_frames // 2)
            log(f"{LOG_TAG} a window did not fit the card after all — halving to {w_frames} keyframes")
            continue
        if proc.returncode != 0:
            raise WalkError(f"DA3 window extraction exited with code {proc.returncode}")
        break
    return wdir, windows


def walk_is_current(session_dir: Path, files: Sequence[str], frames_dir: Path, gcfg,
                    python: str, log: Callable = print) -> bool:
    """True when ``intake/walk.json`` was measured on EXACTLY the window plan this run would
    extract (same keyframes, same deterministic window size, same model and resolution) and
    the revisit reference is on disk — the walk, the anchors and the revisit reference are
    then reused and the window depth files are NOT regenerated (USER 2026-10-06: a resume
    regenerated 153 windows, 20 min of GPU, only because their files had been deleted to
    save disk). F2 regenerates the files when IT needs them (gauge.run_gauge)."""
    session_dir = Path(session_dir)
    walk = load_walk(session_dir)
    spec_path = session_dir / "output" / WINDOWS_DIRNAME / "windows.json"
    ref = session_dir / "output" / REVISIT_REFERENCE_NAME
    if walk is None or not spec_path.exists() or not ref.exists():
        return False
    try:
        old = json.loads(spec_path.read_text())
    except (OSError, ValueError):
        return False
    res = da3_process_res(gcfg.process_res, frames_dir)
    from intake.vram import window_size
    from intake.config import load_intake_config
    from config import cfg as _raw_cfg
    icfg = load_intake_config(_raw_cfg)
    import cv2
    _img = cv2.imread(str(Path(frames_dir) / files[0]), cv2.IMREAD_UNCHANGED)
    if _img is None:
        return False
    w_frames = window_size(session_dir, files, frames_dir, gcfg.model_id, res,
                           (int(_img.shape[1]), int(_img.shape[0])), python,
                           requested=int(gcfg.window_frames),
                           calibration_frames=int(icfg.parallax.vram_calibration_frames),
                           margin_frac=float(icfg.parallax.vram_margin_frac), log=lambda *_: None)
    plan = plan_windows(len(files), w_frames, gcfg.window_overlap_frac)
    spec = {"windows": [[str(Path(frames_dir) / f) for f in files[a:b]] for a, b in plan],
            "process_res": res, "model_id": gcfg.model_id}
    same = (old == spec and int(walk.get("n_keyframes", -1)) == len(files)
            and int(walk.get("n_windows", -1)) == len(plan))
    if same:
        log(f"{LOG_TAG} I3 reused: walk.json measured on this exact plan ({len(plan)} windows of "
            f"{w_frames} keyframes, {len(files)} keyframes) — the window depth is not regenerated")
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
    doc = {
        "version": WALK_VERSION,
        "provenance": PROVENANCE,
        "geometry_epoch": int(geometry_epoch),
        "camera_epoch": int(camera_epoch),
        "method": "da3_windows_chained",
        "params": {"window_frames": gcfg.window_frames,
                   "window_overlap_frac": gcfg.window_overlap_frac,
                   "process_res": json.loads((wdir / "windows.json").read_text())["process_res"],
                   "model_id": gcfg.model_id},
        "n_keyframes": len(frames),
        "n_windows": len(windows),
        "all_windows_metric": all(w["is_metric"] == 1 for w in windows),
        "walk_length_m": walk_m,
        "local_error_median_m": float(np.median(dis)) if dis else 0.0,
        "local_error_max_m": float(max(dis)) if dis else 0.0,
        "chainage": [{"frame": f, "chainage_m": c} for f, c in zip(frames, chainage)],
        "windows": [{"frames": [w["frames"][0], w["frames"][-1]], "n": len(w["frames"]),
                     "scale_factor": w["scale_factor"], "is_metric": w["is_metric"]}
                    for w in windows],
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
    med_depth, hfov = [], []
    for w in windows:
        with np.load(w["path"]) as z:
            d, c, K = z["depth"], z["conf"], z["intrinsics"]
            valid = (c > 0) & np.isfinite(d) & (d > 0)
            if valid.any():
                med_depth.append(float(np.median(d[valid])))
            width = d.shape[-1]
            hfov.extend(2.0 * np.arctan(width / (2.0 * K[:, 0, 0])))
    if not med_depth or not hfov:
        raise WalkError("the I3 windows carry no valid depth — no revisit reference")
    doc = {"version": 1, "provenance": PROVENANCE,
           "frames": [int(f) for f in frames],
           "centres": [poses[f][:3, 3].tolist() for f in frames],
           "forward": [poses[f][:3, 2].tolist() for f in frames],
           "dist_bar_m": float(np.median(med_depth)),
           "cos_bar": float(np.cos(0.5 * float(np.median(hfov)))),     # half the FOV
           "hfov_rad": float(np.median(hfov))}
    out = session_dir / "output" / REVISIT_REFERENCE_NAME
    out.write_text(json.dumps(doc))
    return doc


def load_walk(session_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(session_dir) / "intake" / WALK_NAME
    return json.loads(p.read_text()) if p.exists() else None


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m intake.walk",
                                 description="I3 DA3 windows (if missing) + the metric walk.")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    g = load_precision_config().gauge
    run_da3_windows(Path(args.session), g, sys.executable)
    measure_walk(Path(args.session), g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
