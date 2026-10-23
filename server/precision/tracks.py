"""P3 — sub-pixel correspondences in NATIVE pixels (claude_stac.txt §4-F4).

The VGGSfM tracker vendored in the fork (``base_models/vggt/dependency``: coarse
tracker + ``track_refine`` on crops) runs at the frames' own resolution, capped at
``tracker_long_side`` and never upsampled; every observation is carried to native
pixels by the exact grid mapping of F0 (``precision.camera.grid_full_frame_resize``
/ ``grid_to_native``). Each observation's σ is the tracker's OWN uncertainty — the
standard deviation of its fine similarity heatmap (``compute_score``, which the
vendor's ``forward`` switches off) — carried to native pixels as well.

Queries: a grid plus Shi-Tomasi corners of the query frame, dropped where the prior's
depth jumps (``depth_edge_tol_rel`` over a 2×2 neighbourhood of Omega's depth). Nothing
else excludes a query: I2 (content tags + exclusion masks) was ELIMINATED from the pipeline
(docs/plan_determinismo.md point 75, USER 2026-10-08 "ya no corre VLM ahí") — no reader of
``intake/content_tags.json`` / ``intake/exclusion_masks`` is left here.

Windows:
  keyframes  overlapping runs of ``window_frames`` keyframes (``window_overlap_frac``)
  loop       the keyframes around both ends of every SALAD pair
             (``maplong_run/loop_closures.txt`` — INDICES into its own image list,
             resolved through that list) and every re-identified object's visits
             (``instance_loops.json``, F3)
  witness    each run of witness frames with the two keyframes around it — the
             correspondences F5 localises the witnesses with

Determinism: the tracker's weights are the file whose sha256 is declared
(``tracker_weights_url`` at a pinned revision, ``tracker_weights_sha256``; recorded in
tracks.json, any other file refused) and every tracker call runs with torch's
deterministic algorithms on (STRICT), TF32 off and the ``seed`` set
(:func:`deterministic_torch` = ``repro.deterministic_torch``), on the card only (no CPU
fallback: other kernels, other bits), with the card checked free before the step
(``repro.require_exclusive_gpu`` — the runner checks it before launching, ``main`` again
before torch touches CUDA) and the environment (card, driver, torch / CUDA / cuDNN, BLAS,
CPU, libraries, code) recorded in tracks.json (docs/plan_determinismo.md points 4, 37).

Every track is assigned to the fit or the held-out set once (``heldout_frac``) by a
hash of ITS OWN stable key — the seed, its query frame and the exact float32 bits of its
query pixel (:func:`track_split_of`) — never by its position in the emission order:
one query more or less anywhere used to reshuffle ~32 % of every later track between
F5's fit and held-out sets (point 28). The same query point tracked in two windows
lands in the same set (no leak between fit and held-out).

Instance loops (``instance_loops.json``) enter only when stamped with THIS
reconstruction's id (``correction.epoch.same_reconstruction``, point 34) AND when their
EVIDENCE STAMP — the ``repro.stamp`` ``correction.visit_drift_run`` wrote over the session
files the closures were measured on (cloud, poses, keyframes, camera, segmentation and masks,
the measurement's code and parameters) — is the session's as it is now
(``visit_drift_run.read_evidence``, point 134: never the equality of an epoch number); the
report says why one was not taken. Timings go to ``tracks.timing.json`` (point 36).

Output ``output/precision/tracks.npz`` (v2 = the v1 keys at the
tracker grid + ``obs_uv_native``, ``obs_uv_sigma``, ``track_split``,
``window_id``, ``is_loop_pair``, ``frame_kind``) and ``tracks.json``.

CLI (the mapanything env, GPU): ``python -m precision.tracks --session <dir>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TRACKS_NAME = "tracks.npz"
REPORT_NAME = "tracks.json"
TIMING_NAME = "tracks.timing.json"
INSTANCE_LOOPS_NAME = "instance_loops.json"
TRACKS_VERSION = 2
PROVENANCE = "tool_measured"
LOG_TAG = "[tracks]"
KIND_KEYFRAME, KIND_WITNESS = 0, 1
GRAY_MAX = np.iinfo(np.uint8).max           # 8-bit frames: the tracker takes [0, 1]
SPLIT_FIT, SPLIT_HELDOUT = 0, 1

# (images (S,H,W,3) float [0,1] at the tracker grid, queries (N,2) grid px on image 0,
#  the frame numbers of the images) → (uv (S,N,2) grid px, visibility (S,N), sigma (S,N)
#  grid px). The frame numbers let a ground-truth tracker stand in for the network.
TrackFn = Callable[[np.ndarray, np.ndarray, List[int]],
                   Tuple[np.ndarray, np.ndarray, np.ndarray]]


class TracksError(RuntimeError):
    """A structural impossibility of the stage — always with the exact reason."""


@dataclass
class Window:
    frames: List[int]
    kind: str                           # keyframes | loop | witness
    query_frames: List[int] = field(default_factory=list)
    sides: Optional[Tuple[List[int], List[int]]] = None   # loop: the frames around each end


def _num(name: str) -> int:
    return int("".join(ch for ch in Path(name).stem if ch.isdigit()))


# ── windows ──────────────────────────────────────────────────────────────

def keyframe_windows(kf: Sequence[int], window_frames: int, overlap_frac: float,
                     n_query: int) -> List[Window]:
    kf = list(kf)
    w = int(window_frames)
    step = max(1, w - int(round(w * overlap_frac)))
    out, s = [], 0
    while True:
        fr = kf[s:s + w]
        if len(fr) >= 2:
            q = sorted({fr[int(round(i))] for i in np.linspace(0, len(fr) - 1, n_query)})
            out.append(Window(fr, "keyframes", q))
        if s + w >= len(kf):
            break
        s += step
    return out


def loop_windows(kf: Sequence[int], pairs: Sequence[Tuple[int, int]], half: int) -> List[Window]:
    """``pairs``: keyframe INDICES (i, j); the keyframes within ``half`` of each end."""
    kf = list(kf)
    out = []
    for i, j in pairs:
        if not (0 <= i < len(kf) and 0 <= j < len(kf)) or i == j:
            continue
        a = list(range(max(0, i - half), min(len(kf), i + half + 1)))
        b = list(range(max(0, j - half), min(len(kf), j + half + 1)))
        idx = sorted(set(a) | set(b))
        out.append(Window([kf[k] for k in idx], "loop", [kf[i], kf[j]], ([kf[k] for k in a], [kf[k] for k in b])))
    return out


def witness_windows(kf: Sequence[int], witnesses: Sequence[int], window_frames: int) -> List[Window]:
    """Each run of witnesses strictly between two consecutive keyframes, with both
    keyframes; a run longer than the window is cut, every piece keeping both."""
    kf = sorted(kf)
    kfs = set(kf)
    ws = sorted(w for w in witnesses if w not in kfs)
    out = []
    j = 0
    for a, b in zip(kf, kf[1:]):
        run = []
        while j < len(ws) and ws[j] < b:
            if ws[j] > a:
                run.append(ws[j])
            j += 1
        per = max(1, int(window_frames) - 2)
        for s in range(0, len(run), per):
            out.append(Window([a] + run[s:s + per] + [b], "witness", [a, b]))
    return out


def salad_pairs(loop_file: Path) -> List[Tuple[int, int]]:
    """SALAD's pairs as KEYFRAME FRAME NUMBERS: the file holds indices into its own
    image list, written below the pairs as ``# i: path``."""
    if not loop_file.exists():
        return []
    idx_pairs, path_of = [], {}
    for line in loop_file.read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            body = s[1:].strip()
            if ":" in body and body.split(":", 1)[0].strip().isdigit():
                k, p = body.split(":", 1)
                path_of[int(k)] = p.strip()
            continue
        parts = [p for p in s.replace(",", " ").split() if p]
        if len(parts) >= 2:
            try:
                idx_pairs.append((int(float(parts[0])), int(float(parts[1]))))
            except ValueError:
                continue
    return [(_num(path_of[i]), _num(path_of[j])) for i, j in idx_pairs
            if i in path_of and j in path_of]


# ── queries ──────────────────────────────────────────────────────────────

def depth_edges(depth: np.ndarray, tol_rel: float) -> np.ndarray:
    """True where the 2×2 neighbourhood's depth spread exceeds ``tol_rel`` of its
    nearest depth (the loops.witness.tracks convention)."""
    d = np.asarray(depth, np.float64)
    pad = np.pad(d, ((0, 1), (0, 1)), mode="edge")
    blk = np.stack([pad[:-1, :-1], pad[1:, :-1], pad[:-1, 1:], pad[1:, 1:]])
    lo, hi = blk.min(0), blk.max(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        edge = (hi - lo) / lo > tol_rel
    return edge | ~(lo > 0)


def query_points(gray: np.ndarray, grid_side: int, max_corners: int, corner_quality: float,
                 corner_min_distance_px: float) -> np.ndarray:
    """Grid + Shi-Tomasi corners of one image (its own pixel grid)."""
    import cv2
    h, w = gray.shape
    nw = int(grid_side)
    nh = max(2, int(round(nw * h / w)))
    xs = (np.arange(nw) + 0.5) * w / nw - 0.5
    ys = (np.arange(nh) + 0.5) * h / nh - 0.5
    gx, gy = np.meshgrid(xs, ys)
    pts = [np.stack([gx.ravel(), gy.ravel()], 1)]
    c = cv2.goodFeaturesToTrack(gray, int(max_corners), float(corner_quality),
                                float(corner_min_distance_px))
    if c is not None:
        pts.append(c.reshape(-1, 2).astype(np.float64))
    return np.concatenate(pts).astype(np.float32)


def keep_mask(q_native: np.ndarray, exclusion: Optional[np.ndarray],
              edges: Optional[np.ndarray], edges_map) -> np.ndarray:
    """Queries outside the exclusion mask (native grid) and off the depth edges
    (``edges`` on ``edges_map``'s grid)."""
    from precision.camera import native_to_grid
    keep = np.ones(len(q_native), bool)
    if exclusion is not None:
        H, W = exclusion.shape
        u = np.clip(np.rint(q_native[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.rint(q_native[:, 1]).astype(int), 0, H - 1)
        keep &= ~exclusion[v, u]
    if edges is not None:
        g = native_to_grid(q_native, edges_map)
        H, W = edges.shape
        u = np.clip(np.rint(g[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.rint(g[:, 1]).astype(int), 0, H - 1)
        keep &= ~edges[v, u]
    return keep


# ── the tracker ──────────────────────────────────────────────────────────

def tracker_grid(native_w: int, native_h: int, long_side: int, stride: int) -> Tuple[int, int]:
    """(w, h) of the tracker input: the native size capped at ``long_side`` (never
    upsampled), each side a multiple of the network's ``stride``."""
    s = min(1.0, float(long_side) / max(native_w, native_h))
    w = max(stride, int(round(native_w * s / stride)) * stride)
    h = max(stride, int(round(native_h * s / stride)) * stride)
    return w, h


# ── determinism (shared with F6) ─────────────────────────────────────────

_SERVER = str(Path(__file__).resolve().parents[1])
if _SERVER not in sys.path:
    sys.path.insert(0, _SERVER)

# ONE implementation for every stage (server/repro.py, 2026-10-07): torch deterministic STRICT,
# cuDNN deterministic without benchmark, TF32 off, seeds fixed, the cuBLAS workspace pinned to the
# one value every launcher uses. Re-exported here for the stages that import it from F4.
from repro import (CUBLAS_WORKSPACE_ENV, deterministic_torch,  # noqa: E402,F401
                   ensure_cublas_workspace)


# ── the tracker weights, pinned by content ───────────────────────────────

def file_sha256(path: Path) -> str:
    from repro import sha256_file
    return sha256_file(path)


def verified_checkpoint(url: str, sha256: str, log: Callable = print) -> Path:
    """The checkpoint at ``url`` (a pinned revision) in torch hub's cache — downloaded
    once, the download itself hash-checked — whose content hashes to ``sha256``. A cached
    file that does not (another revision, a corrupt copy) is refused: the tracks are
    never made by weights nobody declared."""
    import torch
    from urllib.parse import urlparse
    dst = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        log(f"{LOG_TAG} downloading {url} → {dst}")
        torch.hub.download_url_to_file(url, str(dst), hash_prefix=sha256, progress=False)
    got = file_sha256(dst)
    if got != sha256:
        raise TracksError(f"{dst} hashes to sha256 {got}, the configured tracker weights are "
                          f"{sha256} (precision.tracks.tracker_weights_sha256) — another file "
                          f"with the same name; delete it to fetch {url}")
    return dst


def vggsfm_track_fn(weights_url: str, weights_sha256: str, device: str = "cuda",
                    log: Callable = print) -> TrackFn:
    """The fork's VGGSfM tracker: coarse tracks, then ``refine_track`` WITH its score
    (the fine heatmap's standard deviation, px — the observation's σ). The weights are
    the file whose sha256 is ``weights_sha256`` (:func:`verified_checkpoint`)."""
    import torch
    root = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long" / "base_models" / "vggt"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dependency.vggsfm_tracker import TrackerPredictor
    from dependency.track_modules.track_refine import refine_track
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        # no CPU fallback: the CPU kernels round otherwise — the tracks would silently be
        # another product (docs/plan_determinismo.md point 12's rule, applied to F4)
        raise TracksError("torch sees no CUDA device — the tracker runs on the card, never "
                          "falls back to the CPU")
    dev = torch.device(device)
    tk = TrackerPredictor()
    ckpt = verified_checkpoint(weights_url, weights_sha256, log=log)
    tk.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    tk = tk.to(dev).eval()

    def fn(images: np.ndarray, queries: np.ndarray, frames: List[int]):
        with torch.no_grad():
            im = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().to(dev)
            fm = tk.process_images_to_fmaps(im)[None]
            q = torch.from_numpy(queries).float().to(dev)[None]
            coarse, vis = tk.coarse_predictor(query_points=q, fmaps=fm, iters=6,
                                              down_ratio=tk.coarse_down_ratio)
            fine, score = refine_track(im[None], tk.fine_fnet, tk.fine_predictor, coarse[-1],
                                       compute_score=True)
        return (fine[0].float().cpu().numpy(), vis[0].float().cpu().numpy(),
                score[0].float().cpu().numpy())
    return fn


# ── the held-out split, per track ───────────────────────────────────────

_SPLIT_KEY = np.dtype([("seed", "<i8"), ("frame", "<i8"), ("u", "<f4"), ("v", "<f4")])


def track_split_of(seed: int, query_frames: np.ndarray, query_uv: np.ndarray,
                   heldout_frac: float) -> np.ndarray:
    """SPLIT_HELDOUT / SPLIT_FIT per track from a hash of the track's OWN stable key: (``seed``,
    its query frame, the exact float32 bits of its query pixel) packed little-endian, sha256, the
    first 8 bytes as an unsigned integer u: held out when u < ``heldout_frac`` × 2⁶⁴ (exact: the
    fraction scaled by a power of two). Membership depends on the track alone — not on how many
    tracks were emitted before it (point 28) — and is the same on every machine."""
    qf = np.asarray(query_frames, np.int64).ravel()
    uv = np.asarray(query_uv, np.float32).reshape(-1, 2)
    if len(qf) != len(uv):
        raise TracksError(f"track_split_of: {len(qf)} query frames for {len(uv)} query pixels")
    frac = float(heldout_frac)
    if not 0.0 <= frac <= 1.0:
        raise TracksError(f"heldout_frac {heldout_frac!r} must lie in [0, 1]")
    key = np.zeros(len(qf), _SPLIT_KEY)
    key["seed"], key["frame"], key["u"], key["v"] = int(seed), qf, uv[:, 0], uv[:, 1]
    raw, step = key.tobytes(), _SPLIT_KEY.itemsize
    threshold = int(frac * 2.0 ** 64)
    out = np.empty(len(qf), np.int8)
    sha = hashlib.sha256
    for i in range(len(qf)):
        u = int.from_bytes(sha(raw[i * step:(i + 1) * step]).digest()[:8], "little")
        out[i] = SPLIT_HELDOUT if u < threshold else SPLIT_FIT
    return out


# ── inputs that come from outside the chain ─────────────────────────────

def instance_loops_owned(out_dir: Path, rid: Optional[str] = None) -> Tuple[Optional[dict], dict]:
    """(the ``instance_loops.json`` document when it belongs to THIS reconstruction — its
    ``reconstruction_id``, point 34 — else None, report). The same rule as the gauge's
    ``_evidence_doc`` for the same evidence class: an absent file is ``present: False``; a file
    that is there and cannot be read FAILS the step naming it — never a default in its place
    (``precision.gauge`` raises GaugeError for the identical situation; F4 applies the same
    rule, not its opposite); a file of another reconstruction is refused, reported."""
    from correction.epoch import reconstruction_id_or_none, same_reconstruction
    p = Path(out_dir) / INSTANCE_LOOPS_NAME
    if not p.exists():
        return None, {"present": False, "taken": False}
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise TracksError(f"{p} is unreadable ({e}) — re-run the measurement that writes it "
                          f"(python -m correction.visit_drift_run --mode measure)") from e
    if not isinstance(doc, dict):
        raise TracksError(f"{p} is not an evidence document (a JSON {type(doc).__name__}) — "
                          f"re-run the measurement that writes it")
    rid = reconstruction_id_or_none(out_dir) if rid is None else rid
    taken, why_rid = same_reconstruction(doc, rid)
    if not taken:
        return None, {"present": True, "taken": False, "reason": why_rid}
    return doc, {"present": True, "taken": True}


def instance_loops(out_dir: Path, rid: Optional[str] = None,
                   cfg=None) -> Tuple[List[Tuple[int, int]], dict]:
    """(keyframe-index pairs, report) of ``instance_loops.json`` — taken only when it was measured
    on THIS reconstruction (its ``reconstruction_id``; point 34 — :func:`instance_loops_owned`)
    AND on the session AS IT IS NOW: ``correction.visit_drift_run.evidence_fresh`` verifies the
    EVIDENCE STAMP the measurement wrote (point 134 — the cloud, the poses, the keyframe list,
    the camera, the segmentation and its masks, the fused objects, the chunk plan, the code and
    the parameters of the measurement); a file without that stamp, or with one that differs, is
    REFUSED and the report says what differs. The reconstruction id alone used to decide: the
    loops measured on the previous masks (an epoch re-selected in the kit and re-segmented) read
    as current. ``rid``: the session's id when the caller already has it; ``cfg``: the correction
    configuration the stamp is computed with (the job's frozen one on the pipeline path — point
    69; None = the process's). A file that exists and cannot be read raises TracksError."""
    from correction.visit_drift_run import evidence_fresh
    doc, rep = instance_loops_owned(out_dir, rid)
    if doc is None:
        return [], rep
    fresh, why = evidence_fresh(Path(out_dir), doc, cfg)
    if not fresh:
        return [], {"present": True, "taken": False,
                    "reason": ("its evidence stamp is not the session's as it is now (point 134): "
                               + "; ".join(why[:4])),
                    "stamp_differences": list(why)}
    pairs = [(int(l["i"]), int(l["j"])) for l in doc.get("loops", [])]
    return pairs, {"present": True, "taken": True, "n": len(pairs)}


def chain_inputs(session_dir: Path, rid: Optional[str] = None) -> Dict[str, Path]:
    """The files F4 reads that no step of the chain writes (the runner stamps them, point 31):
    the keyframe and witness lists, SALAD's pairs and the instance loops of THIS reconstruction
    (point 34). The frames, Omega's records (depth edges) and the config are stamped by the
    runner.

    Whether instance_loops.json is an INPUT of F4 is decided by its OWNERSHIP alone (this
    reconstruction's id), the gauge's rule for scale_loop_rows.json: a file of this
    reconstruction is an input whether or not its evidence stamp is fresh today — when it is
    re-measured its bytes change and the chain's stamp with them, so F4 re-runs and takes it.
    The freshness test (point 134: a sha256 over the cloud, the masks, the poses…) is
    ``run_tracks``'s, once; running it here on every resume probe of the chain re-hashed the
    published cloud for a listing it never changed."""
    sd = Path(session_dir)
    out = sd / "output"
    cand = {"frames/selected_frames.json": sd / "frames" / "selected_frames.json",
            "frames/witness_frames.json": sd / "frames" / "witness_frames.json",
            "output/maplong_run/loop_closures.txt": out / "maplong_run" / "loop_closures.txt"}
    found = {k: p for k, p in cand.items() if p.exists()}
    if instance_loops_owned(out, rid)[1]["taken"]:
        found[f"output/{INSTANCE_LOOPS_NAME}"] = out / INSTANCE_LOOPS_NAME
    return found


# ── run ──────────────────────────────────────────────────────────────────

def _read_list(p: Path, key: str) -> List[int]:
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    if key == "frames":
        return sorted(int(r["frame"]) for r in d.get("frames", []))
    return sorted(_num(f) for f in d.get(key, []))


def run_tracks(session_dir: Path, tcfg, track_fn: Optional[TrackFn] = None,
               log: Callable = print, ccfg=None, match_fn=None) -> Dict[str, Any]:
    """F4 on a session. ``ccfg``: the correction configuration the instance loops' evidence
    stamp is verified with (point 134) — the job's frozen one on the pipeline path (point 69);
    None = the process's (the tests' path). ``match_fn``: an injected loop-bridge matcher
    (tests); None = the backend ``tcfg.loop_bridge`` names (precision.loop_bridge)."""
    import cv2
    from precision.camera import grid_full_frame_resize, grid_to_native, grid_like, native_to_grid
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    out_dir = session_dir / "output"
    kf = _read_list(frames_dir / "selected_frames.json", "selected_files")
    if len(kf) < 2:
        raise TracksError(f"{frames_dir / 'selected_frames.json'} lists {len(kf)} keyframe(s)")
    wit = _read_list(frames_dir / "witness_frames.json", "frames")
    probe = cv2.imread(str(frames_dir / f"{kf[0]:06d}.jpg"), cv2.IMREAD_COLOR)
    if probe is None:
        raise TracksError(f"{frames_dir / f'{kf[0]:06d}.jpg'} is unreadable")
    nh, nw = probe.shape[:2]
    tw, th = tracker_grid(nw, nh, tcfg.tracker_long_side, tcfg.tracker_stride)
    gmap = grid_full_frame_resize(nw, nh, tw, th, "tracker")
    s_mean = 0.5 * (nw / tw + nh / th)             # grid px → native px (σ)

    pairs_f = salad_pairs(out_dir / "maplong_run" / "loop_closures.txt")
    kidx = {f: i for i, f in enumerate(kf)}
    idx_pairs = [(kidx[a], kidx[b]) for a, b in pairs_f if a in kidx and b in kidx]
    n_salad = len(idx_pairs)
    il_pairs, il_rep = instance_loops(out_dir, cfg=ccfg)
    if il_rep["present"] and not il_rep["taken"]:
        log(f"{LOG_TAG} {INSTANCE_LOOPS_NAME} NOT taken: {il_rep['reason']}")
    idx_pairs += il_pairs
    # the classical cross-check's loop candidates (phase 3d): COLMAP's verified two-view geometries
    # between keyframes far apart, taken only for THIS reconstruction and THIS keyframe list
    from precision.classical_check import classical_loops
    from correction.epoch import reconstruction_id_or_none
    cl_pairs, cl_rep = classical_loops(out_dir, kf, reconstruction_id_or_none(out_dir))
    if cl_rep["present"] and not cl_rep["taken"]:
        log(f"{LOG_TAG} loop_candidates.json NOT taken: {cl_rep['reason']}")
    known = {(min(a, b), max(a, b)) for a, b in idx_pairs}
    cl_new = [(a, b) for a, b in cl_pairs if (min(a, b), max(a, b)) not in known]
    idx_pairs += cl_new
    n_classical = len(cl_new)
    windows = (keyframe_windows(kf, tcfg.window_frames, tcfg.window_overlap_frac,
                                tcfg.query_frames_per_window)
               + loop_windows(kf, idx_pairs, tcfg.loop_half_window)
               + witness_windows(kf, wit, tcfg.window_frames))
    log(f"{LOG_TAG} {len(kf)} keyframes, {len(wit)} witnesses → {len(windows)} window(s) "
        f"({sum(w.kind == 'keyframes' for w in windows)} keyframe, "
        f"{sum(w.kind == 'loop' for w in windows)} loop from {n_salad} SALAD + "
        f"{len(idx_pairs) - n_salad - n_classical} instance + {n_classical} classical pair(s), "
        f"{sum(w.kind == 'witness' for w in windows)} witness); tracker grid {tw}x{th} "
        f"for native {nw}x{nh}")

    # the prior's depth edges need Omega's exact grid (F0's camera.json); without it
    # the filter is skipped and SAID, never approximated
    omap = None
    cam_json = out_dir / "camera.json"
    if cam_json.exists():
        from precision.camera import load_camera_json
        omap = load_camera_json(cam_json).omega_grid
    else:
        log(f"{LOG_TAG} {cam_json} is missing — the depth-edge filter is skipped "
            f"(run python -m precision.camera first)")

    def image(f):
        im = cv2.imread(str(frames_dir / f"{f:06d}.jpg"), cv2.IMREAD_COLOR)
        if im is None:
            raise TracksError(f"frame {f:06d}.jpg is unreadable")
        return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (tw, th),
                          interpolation=cv2.INTER_AREA)

    # no exclusion mask touches a query: I2 is gone (point 75, USER 2026-10-08) — a PNG under
    # intake/exclusion_masks or a content_tags.json left by an older run is never read

    def edges(f):
        p = out_dir / "omega_run" / "results_output" / f"frame_{f}.npz"
        if omap is None or not p.exists():
            return None, None
        with np.load(p) as z:
            d = z["depth"]
        g = omap if (omap.w, omap.h) == (d.shape[1], d.shape[0]) else \
            grid_like(omap, d.shape[1], d.shape[0], "omega_npz")
        return depth_edges(d, tcfg.depth_edge_tol_rel), g

    # the tracker's weights are recorded by content; an injected track_fn (a ground-truth
    # stand-in) is said to be one
    weights = {"injected_track_fn": True}
    on_card = track_fn is None
    if on_card:
        ensure_cublas_workspace()                 # before the tracker touches the GPU
        track_fn = vggsfm_track_fn(tcfg.tracker_weights_url, tcfg.tracker_weights_sha256,
                                   log=log)
        weights = {"url": tcfg.tracker_weights_url, "sha256": tcfg.tracker_weights_sha256}
    numerics = None
    OBS_KEYS = ("track", "frame", "uv", "uv_native", "sigma", "vis", "window", "loop", "kind", "bridged")
    obs = {k: [] for k in OBS_KEYS}
    tq = {k: [] for k in ("id", "frame", "uv")}
    wit_set = set(wit) - set(kf)
    next_tid, t0 = 0, time.time()
    n_dropped_edge = 0
    # THE LOOP BRIDGE (phase 3d): wide-baseline matches across every loop pair, verified by the
    # session camera's geometry, become queries on both visits; their tracks merge into one
    bridge_rep: Dict[str, Any] = {"backend": str(tcfg.loop_bridge), "pairs": {}, "bridged_tracks": 0,
                                  "links": 0}
    by_pair: Dict[Tuple[int, int], Any] = {}
    if not idx_pairs:
        bridge_rep["skipped"] = "no loop pair (SALAD, instance or classical) — nothing to bridge"
    if idx_pairs and (match_fn is not None or str(tcfg.loop_bridge) != "none"):
        from precision import loop_bridge as LB
        from precision.camera import load_camera_json
        if not cam_json.exists():
            raise TracksError(f"{cam_json} is missing — the loop bridge verifies every pair with the session "
                              f"camera (python -m precision.camera first)")
        cam = load_camera_json(cam_json)
        mfn = match_fn if match_fn is not None else LB.make_match_fn(str(tcfg.loop_bridge), tcfg, log=log)

        def image_native(f):
            im = cv2.imread(str(frames_dir / f"{f:06d}.jpg"), cv2.IMREAD_COLOR)
            if im is None:
                raise TracksError(f"frame {f:06d}.jpg is unreadable")
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        matched = LB.match_pairs(idx_pairs, kf, image_native, mfn, cam.K(), cam.dist(),
                                 ransac_px=float(tcfg.loop_bridge_ransac_px), min_inliers=int(tcfg.loop_bridge_min_inliers),
                                 max_matches=int(tcfg.loop_bridge_max_matches), ransac_prob=float(tcfg.loop_bridge_ransac_prob),
                                 seed=int(tcfg.seed), min_inlier_ratio=float(tcfg.loop_bridge_min_inlier_ratio), log=log)
        for m in matched:
            by_pair[(m.i, m.j)] = m
            bridge_rep["pairs"][f"{kf[m.i]}-{kf[m.j]}"] = {"matches": m.n_raw, "verified": int(len(m.uv_i)),
                                                           "inlier_ratio": m.inlier_ratio, "reason": m.reason}
        bridge_rep["match_fn"] = "injected" if match_fn is not None else str(tcfg.loop_bridge)
    for wi, win in enumerate(windows):
        imgs = np.stack([image(f) for f in win.frames]).astype(np.float32) / GRAY_MAX
        wobs = {k: [] for k in OBS_KEYS}
        wtq = {k: [] for k in ("id", "frame", "uv")}
        pm = None
        if win.kind == "loop" and by_pair:
            a, b = kidx[win.query_frames[0]], kidx[win.query_frames[1]]
            pm = by_pair.get((min(a, b), max(a, b)))
            if pm is not None and (pm.reason or len(pm.uv_i) == 0):
                pm = None
        bridge_tids: Dict[int, Dict[int, int]] = {}        # query frame → match index → tid
        for qf in win.query_frames:
            qi = win.frames.index(qf)
            gray = cv2.cvtColor((imgs[qi] * GRAY_MAX).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            q = query_points(gray, tcfg.grid_side, tcfg.max_corners, tcfg.corner_quality,
                             tcfg.corner_min_distance_px)
            n_base = len(q)
            bq_idx = np.zeros(0, np.int64)
            if pm is not None:
                # the bridge queries of this end, in the tracker's grid
                uv_b = pm.uv_i if kidx[qf] == pm.i else pm.uv_j
                qb = native_to_grid(np.asarray(uv_b, np.float64), gmap).astype(q.dtype)
                q = np.concatenate([q, qb]) if len(q) else qb
                bq_idx = np.arange(len(uv_b))
            e, emap = edges(qf)
            keep = keep_mask(grid_to_native(q, gmap), None, e, emap)
            n_dropped_edge += int((~keep[:n_base]).sum())
            kept_pos = np.cumsum(keep) - 1                 # position of each kept query
            q = q[keep]
            if len(q) == 0:
                continue
            order = [qi] + [k for k in range(len(win.frames)) if k != qi]
            # deterministic kernels, re-seeded per call: a window's tracks do not depend on
            # the windows tracked before it
            with deterministic_torch(tcfg.seed) as num:
                uv, vis, sig = track_fn(imgs[order], q, [win.frames[k] for k in order])
            numerics = num if numerics is None else numerics
            inv = np.argsort(order)
            uv, vis, sig = uv[inv], vis[inv], sig[inv]
            good = vis > tcfg.vis_thresh
            good[qi] = True                               # the query itself
            seen = good.sum(0)
            for n in np.flatnonzero(seen >= 2):
                tid = next_tid + int(n)
                for s_ in np.flatnonzero(good[:, n]):
                    f = win.frames[s_]
                    wobs["track"].append(tid)
                    wobs["frame"].append(f)
                    wobs["uv"].append(uv[s_, n])
                    wobs["sigma"].append(float(sig[s_, n]) * s_mean)
                    wobs["vis"].append(float(vis[s_, n]))
                    wobs["window"].append(wi)
                    wobs["loop"].append(win.kind == "loop")
                    wobs["kind"].append(KIND_WITNESS if f in wit_set else KIND_KEYFRAME)
                    wobs["bridged"].append(False)
                wtq["id"].append(tid)
                wtq["frame"].append(qf)
                wtq["uv"].append(q[n])
            if pm is not None:
                # the tid of every bridge query that survived the edge filter (a single-frame track
                # — the query alone — still exists for the merge: it joins the other side's)
                tids_of: Dict[int, int] = {}
                for k in bq_idx:
                    pos = n_base + int(k)
                    if keep[pos]:
                        n_ = int(kept_pos[pos])
                        if seen[n_] >= 1:
                            tids_of[int(k)] = next_tid + n_
                            if seen[n_] < 2:
                                # the query alone: one observation, so the merge has it
                                wobs["track"].append(next_tid + n_); wobs["frame"].append(qf)
                                wobs["uv"].append(uv[qi, n_]); wobs["sigma"].append(float(sig[qi, n_]) * s_mean)
                                wobs["vis"].append(float(vis[qi, n_])); wobs["window"].append(wi)
                                wobs["loop"].append(True); wobs["kind"].append(KIND_WITNESS if qf in wit_set else KIND_KEYFRAME)
                                wobs["bridged"].append(False)
                                wtq["id"].append(next_tid + n_); wtq["frame"].append(qf); wtq["uv"].append(q[n_])
                bridge_tids[qf] = tids_of
            next_tid += len(q)
        if pm is not None and len(bridge_tids) == 2:
            fa, fb = win.query_frames[0], win.query_frames[1]
            ta_of, tb_of = bridge_tids.get(fa, {}), bridge_tids.get(fb, {})
            links = [(ta_of[k], tb_of[k]) for k in sorted(set(ta_of) & set(tb_of))]
            if links:
                from precision import loop_bridge as LB
                side_a, side_b = (set(win.sides[0]), set(win.sides[1])) if win.sides else (set(), set())
                mrep = LB.merge_bridged(wobs, wtq, links, side_a, side_b)
                merged = {ta for ta, _tb in links}
                tr_arr = np.asarray(wobs["track"], np.int64)
                br = np.isin(tr_arr, list(merged))
                wobs["bridged"] = br.tolist()
                bridge_rep["bridged_tracks"] += int(mrep["bridged_tracks"])
                bridge_rep["links"] += int(mrep["links"])
                bridge_rep["pairs"][f"{fa}-{fb}"]["bridged_tracks"] = int(mrep["bridged_tracks"])
        for k in OBS_KEYS:
            obs[k].extend(wobs[k])
        for k in ("id", "frame", "uv"):
            tq[k].extend(wtq[k])
        if (wi + 1) % 25 == 0 or wi == len(windows) - 1:
            log(f"{LOG_TAG} window {wi + 1}/{len(windows)}: {len(obs['track']):,} obs "
                f"({time.time() - t0:.0f} s)")
    if not obs["track"]:
        raise TracksError("no track survived — nothing to write")
    uv = np.asarray(obs["uv"], np.float32)
    tids = np.asarray(tq["id"], np.int64)
    split = track_split_of(int(tcfg.seed), np.asarray(tq["frame"], np.int64),
                           np.asarray(tq["uv"], np.float32), float(tcfg.heldout_frac))
    pdir = out_dir / "precision"
    pdir.mkdir(parents=True, exist_ok=True)
    from intake.quality import read_session_epochs
    epochs = read_session_epochs(session_dir)
    meta = {"version": TRACKS_VERSION, "provenance": PROVENANCE, **epochs,
            "tracker_grid": gmap.to_dict(), "native_wh": [nw, nh],
            "tracker_weights": weights,
            "params": {k: getattr(tcfg, k) for k in tcfg.__dataclass_fields__}}
    tmp = pdir / (TRACKS_NAME + ".tmp.npz")
    np.savez_compressed(
        tmp,
        obs_track=np.asarray(obs["track"], np.int64), obs_frame=np.asarray(obs["frame"], np.int64),
        obs_uv=uv, obs_vis=np.asarray(obs["vis"], np.float32),
        obs_score=np.asarray(obs["vis"], np.float32),
        track_query_id=tids, track_query_frame=np.asarray(tq["frame"], np.int64),
        track_query_uv=np.asarray(tq["uv"], np.float32),
        res_h=np.int64(th), res_w=np.int64(tw),
        obs_uv_native=grid_to_native(uv.astype(np.float64), gmap).astype(np.float32),
        obs_uv_sigma=np.asarray(obs["sigma"], np.float32),
        track_split=split, window_id=np.asarray(obs["window"], np.int32),
        is_loop_pair=np.asarray(obs["loop"], bool), frame_kind=np.asarray(obs["kind"], np.int8),
        is_bridged=np.asarray(obs["bridged"], bool),
        meta=np.asarray(json.dumps(meta)))
    tmp.replace(pdir / TRACKS_NAME)
    sig = np.asarray(obs["sigma"])
    # what the tracks depend on beyond their inputs (point 37): the card, driver, torch / CUDA /
    # cuDNN, the numerics the tracker ran with, BLAS core, CPU, libraries and code — no clock
    from repro import environment_record
    env = {"environment": environment_record(gpu=on_card), "numerics": numerics}
    rep = {**meta, **env, "n_windows": len(windows),
           "n_windows_by_kind": {k: sum(w.kind == k for w in windows)
                                 for k in ("keyframes", "loop", "witness")},
           "n_obs": len(obs["track"]), "n_tracks": int(len(tids)),
           "n_heldout_tracks": int(split.sum()),
           "n_queries_dropped_edge": n_dropped_edge,
           "depth_edge_filter": omap is not None,
           "keyframes_covered": len(set(obs["frame"]) & set(kf)), "n_keyframes": len(kf),
           "witnesses_covered": len(set(obs["frame"]) & wit_set), "n_witnesses": len(wit_set),
           "sigma_native_px": {"median": float(np.median(sig)),
                               "p90": float(np.percentile(sig, 90))},
           "instance_loops": il_rep,
           "classical_loops": dict(cl_rep, n_new_pairs=n_classical),
           "loop_bridge": bridge_rep,
           "split": "sha256 of (seed, query frame, query pixel float32 bits) per track"}
    (pdir / REPORT_NAME).write_text(json.dumps(rep, indent=1))
    # the wall clock lives apart from the compared report (point 36)
    (pdir / TIMING_NAME).write_text(json.dumps({"elapsed_s": round(time.time() - t0, 1)}, indent=1))
    log(f"{LOG_TAG} {rep['n_obs']:,} obs, {rep['n_tracks']:,} tracks ({rep['n_heldout_tracks']:,} "
        f"held out), keyframes {rep['keyframes_covered']}/{len(kf)}, witnesses "
        f"{rep['witnesses_covered']}/{len(wit_set)}, σ median {rep['sigma_native_px']['median']:.2f} "
        f"px → {pdir / TRACKS_NAME}")
    return rep


def load_tracks_v2(session_dir: Path) -> Dict[str, np.ndarray]:
    p = Path(session_dir) / "output" / "precision" / TRACKS_NAME
    if not p.exists():
        raise TracksError(f"{p} is missing — run python -m precision.tracks first")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.tracks",
                                 description="Native-pixel sub-pixel tracks (F4).")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    # the card is this step's alone (point 4): checked here, before torch touches CUDA, also
    # when the step is launched by hand; nothing is lowered to fit a shared card
    from repro import require_exclusive_gpu
    require_exclusive_gpu(log=print)
    ensure_cublas_workspace()                   # before torch initialises CUDA
    # the job's frozen configuration (point 69): the tracks' parameters and the correction
    # configuration the instance loops' evidence stamp is verified with (point 134)
    from correction.config import load_correction_config
    from precision.config import raw_of
    pcfg = load_precision_config(session_dir=Path(args.session))
    run_tracks(Path(args.session), pcfg.tracks, ccfg=load_correction_config(raw_of(pcfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
