"""P3 — sub-pixel correspondences in NATIVE pixels (claude_stac.txt §4-F4).

The VGGSfM tracker vendored in the fork (``base_models/vggt/dependency``: coarse
tracker + ``track_refine`` on crops) runs at the frames' own resolution, capped at
``tracker_long_side`` and never upsampled; every observation is carried to native
pixels by the exact grid mapping of F0 (``precision.camera.grid_full_frame_resize``
/ ``grid_to_native``). Each observation's σ is the tracker's OWN uncertainty — the
standard deviation of its fine similarity heatmap (``compute_score``, which the
vendor's ``forward`` switches off) — carried to native pixels as well.

Queries: a grid plus Shi-Tomasi corners of the query frame, dropped where I2's
exclusion masks cover them (``intake/exclusion_masks``) and where the prior's depth
jumps (``depth_edge_tol_rel`` over a 2×2 neighbourhood of Omega's depth).

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
deterministic algorithms on, TF32 off and the ``seed`` set (:func:`deterministic_torch`).

Every track is assigned to the fit or the held-out set once (``heldout_frac``,
fixed ``seed``). Output ``output/precision/tracks.npz`` (v2 = the v1 keys at the
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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TRACKS_NAME = "tracks.npz"
REPORT_NAME = "tracks.json"
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
        idx = sorted(set(range(max(0, i - half), min(len(kf), i + half + 1)))
                     | set(range(max(0, j - half), min(len(kf), j + half + 1))))
        out.append(Window([kf[k] for k in idx], "loop", [kf[i], kf[j]]))
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

# cuBLAS is deterministic only with a fixed workspace (NVIDIA / torch docs); with
# ``torch.use_deterministic_algorithms`` on, torch refuses every cuBLAS call without it
CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_DETERMINISTIC = (":4096:8", ":16:8")


def ensure_cublas_workspace() -> None:
    """Set CUBLAS_WORKSPACE_CONFIG (:4096:8) when unset — it must be in the environment
    BEFORE torch initialises CUDA, so every GPU CLI calls this first."""
    import os
    os.environ.setdefault(CUBLAS_WORKSPACE_ENV, CUBLAS_WORKSPACE_DETERMINISTIC[0])


@contextmanager
def deterministic_torch(seed: int):
    """torch with deterministic algorithms ON (an op without a deterministic kernel
    RAISES instead of running), cuDNN deterministic and not benchmarking, TF32 off (full
    float32 arithmetic on Ampere) and the RNGs seeded; the previous state is restored on
    exit. On CUDA the cuBLAS workspace must already be fixed (:func:`ensure_cublas_workspace`
    before CUDA initialises) — refused otherwise, naming why."""
    import os
    import torch
    if torch.cuda.is_available() and \
            os.environ.get(CUBLAS_WORKSPACE_ENV) not in CUBLAS_WORKSPACE_DETERMINISTIC:
        if torch.cuda.is_initialized():
            raise RuntimeError(f"{CUBLAS_WORKSPACE_ENV} is not one of "
                               f"{CUBLAS_WORKSPACE_DETERMINISTIC} and CUDA is already "
                               f"initialised — cuBLAS cannot be made deterministic now; set it "
                               f"before torch touches the GPU (the precision CLIs do)")
        ensure_cublas_workspace()
    prev = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(int(seed))                      # CPU and every CUDA device
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = prev[2:]


# ── the tracker weights, pinned by content ───────────────────────────────

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
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


# ── run ──────────────────────────────────────────────────────────────────

def _read_list(p: Path, key: str) -> List[int]:
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    if key == "frames":
        return sorted(int(r["frame"]) for r in d.get("frames", []))
    return sorted(_num(f) for f in d.get(key, []))


def run_tracks(session_dir: Path, tcfg, track_fn: Optional[TrackFn] = None,
               log: Callable = print) -> Dict[str, Any]:
    import cv2
    from precision.camera import grid_full_frame_resize, grid_to_native, grid_like
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
    il = out_dir / "instance_loops.json"
    kidx = {f: i for i, f in enumerate(kf)}
    idx_pairs = [(kidx[a], kidx[b]) for a, b in pairs_f if a in kidx and b in kidx]
    n_salad = len(idx_pairs)
    if il.exists():
        idx_pairs += [(int(l["i"]), int(l["j"])) for l in json.loads(il.read_text()).get("loops", [])]
    windows = (keyframe_windows(kf, tcfg.window_frames, tcfg.window_overlap_frac,
                                tcfg.query_frames_per_window)
               + loop_windows(kf, idx_pairs, tcfg.loop_half_window)
               + witness_windows(kf, wit, tcfg.window_frames))
    log(f"{LOG_TAG} {len(kf)} keyframes, {len(wit)} witnesses → {len(windows)} window(s) "
        f"({sum(w.kind == 'keyframes' for w in windows)} keyframe, "
        f"{sum(w.kind == 'loop' for w in windows)} loop from {n_salad} SALAD + "
        f"{len(idx_pairs) - n_salad} instance pair(s), "
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

    def exclusion(f):
        p = session_dir / "intake" / "exclusion_masks" / f"{f:06d}.png"
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
        return (m == 255) if m is not None else None

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
    if track_fn is None:
        ensure_cublas_workspace()                 # before the tracker touches the GPU
        track_fn = vggsfm_track_fn(tcfg.tracker_weights_url, tcfg.tracker_weights_sha256,
                                   log=log)
        weights = {"url": tcfg.tracker_weights_url, "sha256": tcfg.tracker_weights_sha256}
    obs = {k: [] for k in ("track", "frame", "uv", "uv_native", "sigma", "vis", "window",
                           "loop", "kind")}
    tq = {k: [] for k in ("id", "frame", "uv")}
    wit_set = set(wit) - set(kf)
    next_tid, t0 = 0, time.time()
    n_dropped_excl = 0
    for wi, win in enumerate(windows):
        imgs = np.stack([image(f) for f in win.frames]).astype(np.float32) / GRAY_MAX
        for qf in win.query_frames:
            qi = win.frames.index(qf)
            gray = cv2.cvtColor((imgs[qi] * GRAY_MAX).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            q = query_points(gray, tcfg.grid_side, tcfg.max_corners, tcfg.corner_quality,
                             tcfg.corner_min_distance_px)
            e, emap = edges(qf)
            keep = keep_mask(grid_to_native(q, gmap), exclusion(qf), e, emap)
            n_dropped_excl += int((~keep).sum())
            q = q[keep]
            if len(q) == 0:
                continue
            order = [qi] + [k for k in range(len(win.frames)) if k != qi]
            # deterministic kernels, re-seeded per call: a window's tracks do not depend on
            # the windows tracked before it
            with deterministic_torch(tcfg.seed):
                uv, vis, sig = track_fn(imgs[order], q, [win.frames[k] for k in order])
            inv = np.argsort(order)
            uv, vis, sig = uv[inv], vis[inv], sig[inv]
            good = vis > tcfg.vis_thresh
            good[qi] = True                               # the query itself
            seen = good.sum(0)
            for n in np.flatnonzero(seen >= 2):
                tid = next_tid + int(n)
                for s in np.flatnonzero(good[:, n]):
                    f = win.frames[s]
                    obs["track"].append(tid)
                    obs["frame"].append(f)
                    obs["uv"].append(uv[s, n])
                    obs["sigma"].append(float(sig[s, n]) * s_mean)
                    obs["vis"].append(float(vis[s, n]))
                    obs["window"].append(wi)
                    obs["loop"].append(win.kind == "loop")
                    obs["kind"].append(KIND_WITNESS if f in wit_set else KIND_KEYFRAME)
                tq["id"].append(tid)
                tq["frame"].append(qf)
                tq["uv"].append(q[n])
            next_tid += len(q)
        if (wi + 1) % 25 == 0 or wi == len(windows) - 1:
            log(f"{LOG_TAG} window {wi + 1}/{len(windows)}: {len(obs['track']):,} obs "
                f"({time.time() - t0:.0f} s)")
    if not obs["track"]:
        raise TracksError("no track survived — nothing to write")
    uv = np.asarray(obs["uv"], np.float32)
    tids = np.asarray(tq["id"], np.int64)
    rng = np.random.default_rng(int(tcfg.seed))
    split = (rng.random(len(tids)) < float(tcfg.heldout_frac)).astype(np.int8)
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
        meta=np.asarray(json.dumps(meta)))
    tmp.replace(pdir / TRACKS_NAME)
    sig = np.asarray(obs["sigma"])
    rep = {**meta, "n_windows": len(windows),
           "n_windows_by_kind": {k: sum(w.kind == k for w in windows)
                                 for k in ("keyframes", "loop", "witness")},
           "n_obs": len(obs["track"]), "n_tracks": int(len(tids)),
           "n_heldout_tracks": int(split.sum()),
           "n_queries_dropped_excluded_or_edge": n_dropped_excl,
           "depth_edge_filter": omap is not None,
           "keyframes_covered": len(set(obs["frame"]) & set(kf)), "n_keyframes": len(kf),
           "witnesses_covered": len(set(obs["frame"]) & wit_set), "n_witnesses": len(wit_set),
           "sigma_native_px": {"median": float(np.median(sig)),
                               "p90": float(np.percentile(sig, 90))},
           "elapsed_s": round(time.time() - t0, 1)}
    (pdir / REPORT_NAME).write_text(json.dumps(rep, indent=1))
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
    ensure_cublas_workspace()                   # before torch initialises CUDA
    run_tracks(Path(args.session), load_precision_config().tracks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
