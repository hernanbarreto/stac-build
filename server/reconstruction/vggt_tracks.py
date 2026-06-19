"""
VGGSfM track extraction — the CORRESPONDENCE front-end for the global bundle
adjustment (Tier-1 pose refinement).
================================================================================
The active backbone (MapAnything) has no tracker, so we use the VGGSfM tracker
that ships vendored inside VGGT-Long (`dependency.vggsfm_tracker.TrackerPredictor`,
weights auto-downloaded from HuggingFace, ~few-hundred-MB — NOT the 5 GB VGGT model).
It is a learned, CoTracker-style tracker that produces sub-pixel 2D-2D
correspondences robust to the low-texture indoor scenes where classical features
(SIFT/ORB) fail — exactly why we don't hand-roll keypoint matching.

Per window of S keyframes we seed query points on a few query frames, track them
across the window, and keep observations with high visibility+score. Windows
overlap (shared keyframes) so the downstream BA is globally coupled; loop-closure
pairs get their own windows so long-range constraints exist.

Output: `<output_dir>/ba_run/tracks.npz` — a flat observation table the BA consumes:
    obs_track  (P,) int64   global track id (unique across windows)
    obs_frame  (P,) int64   GLOBAL keyframe number (matches camera_frames.txt)
    obs_uv     (P,2) float32 pixel coords at `res` (x=col, y=row, top-left origin)
    obs_vis    (P,) float32 visibility [0,1]
    obs_score  (P,) float32 confidence [0,1]
plus track_query_frame (T,), track_query_uv (T,2), and meta (res_h, res_w).

Run (in the mapanything env):
    python -m reconstruction.vggt_tracks --output-dir <out> --frames-dir <frames> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("VGGTTracks")

# ── working resolution (W,H) the tracker + BA operate at. AUTO-DETECTED from the
#    backbone per-frame npz so tracks, intrinsics and depth all share one grid.
#    These are overwritten by _detect_res() at the start of extract_tracks(); the
#    portrait phone capture here is W280×H504 (DA3), not landscape. ──
RES_W = 280
RES_H = 504


def _detect_res(output_dir: Path) -> Tuple[int, int]:
    """Return (W, H) from the first backbone per-frame npz depth — the resolution
    whose intrinsics the BA will use, so the tracker must run there too."""
    import glob
    for sub in ("da3_run/results_output", "results_output"):
        cands = sorted(glob.glob(str(output_dir / sub / "frame_*.npz")))
        if cands:
            try:
                z = np.load(cands[0])
                h, w = z["depth"].shape[:2]
                return int(w), int(h)
            except Exception:
                continue
    return RES_W, RES_H


def _add_vendor_to_path() -> None:
    """The VGGSfM tracker uses package-relative imports (`from .track_modules…`),
    so its parent dir must be on sys.path as the import root."""
    here = Path(__file__).resolve()
    # server/reconstruction/vggt_tracks.py → repo root is parents[2] (…/stac-build)
    root = here.parents[2]
    vggt_pkg = root / "vendor" / "VGGT-Long" / "base_models" / "vggt"
    if not vggt_pkg.exists():
        raise FileNotFoundError(f"VGGT-Long vggt pkg not found at {vggt_pkg}")
    p = str(vggt_pkg)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_keyframe_list(output_dir: Path, frames_dir: Path) -> List[Tuple[int, Path]]:
    """Return [(global_frame_num, image_path), …] for the loop-closed keyframes,
    in trajectory order. Prefers camera_frames.txt (the numbers the poses use);
    falls back to selected_frames.json."""
    # camera_frames.txt: one global frame number per pose line (keyframe order)
    for base in (output_dir / "maplong_run", output_dir, output_dir / "da3_run"):
        cf = base / "camera_frames.txt"
        if cf.exists():
            nums = [int(float(x)) for x in cf.read_text().split()]
            sel = frames_dir / "selected_frames.json"
            name_map: Dict[int, str] = {}
            if sel.exists():
                try:
                    names = json.loads(sel.read_text())
                    if isinstance(names, dict):
                        names = names.get("frames", names.get("selected", []))
                    for nm in names:
                        stem = Path(str(nm)).stem
                        digits = "".join(ch for ch in stem if ch.isdigit())
                        if digits:
                            name_map[int(digits)] = str(nm)
                except Exception:
                    pass
            out = []
            for n in nums:
                img = _resolve_image(frames_dir, n, name_map.get(n))
                if img is not None:
                    out.append((n, img))
            if out:
                logger.info(f"loaded {len(out)} keyframes from {cf}")
                return out
    # fallback: selected_frames.json only
    sel = frames_dir / "selected_frames.json"
    if sel.exists():
        names = json.loads(sel.read_text())
        if isinstance(names, dict):
            names = names.get("frames", names.get("selected", []))
        out = []
        for nm in names:
            stem = Path(str(nm)).stem
            digits = "".join(ch for ch in stem if ch.isdigit())
            n = int(digits) if digits else len(out)
            img = _resolve_image(frames_dir, n, str(nm))
            if img is not None:
                out.append((n, img))
        logger.info(f"loaded {len(out)} keyframes from selected_frames.json")
        return out
    raise FileNotFoundError("no camera_frames.txt or selected_frames.json found")


def _resolve_image(frames_dir: Path, num: int, name: Optional[str]) -> Optional[Path]:
    if name:
        p = frames_dir / Path(name).name
        if p.exists():
            return p
    for pat in (f"{num:06d}.jpg", f"{num:06d}.png", f"{num}.jpg", f"frame_{num:06d}.jpg"):
        p = frames_dir / pat
        if p.exists():
            return p
    return None


def _load_loop_pairs(output_dir: Path) -> List[Tuple[int, int]]:
    """Loop-closure frame pairs (global numbers) from loop_closures.txt, if present."""
    for base in (output_dir / "maplong_run", output_dir):
        lc = base / "loop_closures.txt"
        if lc.exists():
            pairs = []
            for line in lc.read_text().splitlines():
                parts = [p for p in line.replace(",", " ").split() if p.strip()]
                if len(parts) >= 2:
                    try:
                        pairs.append((int(float(parts[0])), int(float(parts[1]))))
                    except ValueError:
                        continue
            logger.info(f"loaded {len(pairs)} loop pairs from {lc}")
            return pairs
    return []


def _load_image_tensor(path: Path, torch_mod) -> "object":
    """Load one image → (3, RES_H, RES_W) float tensor in [0,1] (RGB)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((RES_W, RES_H), Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch_mod.from_numpy(arr).permute(2, 0, 1).contiguous()


def _grid_query_points(torch_mod, device, n_side_w: int = 48) -> "object":
    """A grid of query points across the frame (x=col, y=row). The tracker's own
    visibility/score then prunes the ones that don't actually track — denser and
    better-distributed than sparse keypoints for pose BA."""
    n_w = n_side_w
    n_h = max(2, int(round(n_side_w * RES_H / RES_W)))
    xs = np.linspace(RES_W * 0.04, RES_W * 0.96, n_w)
    ys = np.linspace(RES_H * 0.04, RES_H * 0.96, n_h)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return torch_mod.from_numpy(pts).to(device)


def _build_windows(n_kf: int, win: int, stride: int) -> List[List[int]]:
    """Overlapping sequential windows of keyframe INDICES (into the kf list)."""
    if n_kf <= win:
        return [list(range(n_kf))]
    wins = []
    i = 0
    while i < n_kf:
        w = list(range(i, min(i + win, n_kf)))
        wins.append(w)
        if w[-1] == n_kf - 1:
            break
        i += stride
    return wins


def extract_tracks(
    output_dir: Path,
    frames_dir: Path,
    win: int = 24,
    stride: int = 12,
    query_frames_per_win: int = 3,
    grid_side: int = 48,
    vis_thresh: float = 0.2,
    score_thresh: float = 0.2,
    loop_window: int = 8,
    smoke: bool = False,
    device: str = "cuda",
) -> Optional[Path]:
    """Extract VGGSfM tracks over keyframe windows → tracks.npz. Returns the path."""
    import torch

    _add_vendor_to_path()
    from dependency.vggsfm_tracker import TrackerPredictor

    def build_vggsfm_tracker():
        # Build the tracker WITHOUT importing dependency.vggsfm_utils (which does a
        # top-level `import pycolmap` → ModuleNotFoundError). The tracker itself needs
        # only its own weights, auto-downloaded + cached by torch.hub.
        tk = TrackerPredictor()
        url = "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
        tk.load_state_dict(torch.hub.load_state_dict_from_url(url))
        tk.eval()
        return tk

    t0 = time.time()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        logger.warning("CUDA not available — tracker on CPU will be very slow")

    global RES_W, RES_H
    RES_W, RES_H = _detect_res(output_dir)
    logger.info(f"working resolution (W,H) = {RES_W}×{RES_H} (auto from backbone npz)")

    kfs = _load_keyframe_list(output_dir, frames_dir)
    if len(kfs) < 2:
        logger.error("need ≥2 keyframes")
        return None
    kf_nums = [n for n, _ in kfs]
    num_to_idx = {n: i for i, n in enumerate(kf_nums)}

    if smoke:
        kfs = kfs[: min(len(kfs), 6)]
        kf_nums = [n for n, _ in kfs]
        win, stride, grid_side, query_frames_per_win = min(win, 6), 3, 12, 2
        logger.info(f"[SMOKE] {len(kfs)} keyframes, win={win} grid={grid_side}")

    logger.info("building VGGSfM tracker (auto-downloads weights on first run)…")
    tracker: TrackerPredictor = build_vggsfm_tracker().to(dev).eval()

    # preload all keyframe image tensors once (RES_H×RES_W is small)
    imgs = [_load_image_tensor(p, torch) for _, p in kfs]

    windows = _build_windows(len(kfs), win, stride)
    # loop-closure windows: a small bundle around each end of every loop pair
    if not smoke:
        for a, b in _load_loop_pairs(output_dir):
            if a in num_to_idx and b in num_to_idx:
                ia, ib = num_to_idx[a], num_to_idx[b]
                h = loop_window // 2
                wa = range(max(0, ia - h), min(len(kfs), ia + h))
                wb = range(max(0, ib - h), min(len(kfs), ib + h))
                windows.append(sorted(set([*wa, *wb])))
    logger.info(f"{len(windows)} windows (incl. loop windows)")

    obs_track, obs_frame, obs_uv, obs_vis, obs_score = [], [], [], [], []
    tq_id, tq_frame, tq_uv = [], [], []
    next_tid = 0

    for wi, widx in enumerate(windows):
        if len(widx) < 2:
            continue
        stack = torch.stack([imgs[i] for i in widx], dim=0).to(dev)  # (S,3,H,W)
        S = stack.shape[0]
        with torch.no_grad():
            fmaps = tracker.process_images_to_fmaps(stack)            # (S,C,h,w)
            fmaps = fmaps[None]                                       # (1,S,C,h,w)
            # query frames spread across the window (first, middle(s), last)
            qpos = sorted(set(np.linspace(0, S - 1, query_frames_per_win).round().astype(int).tolist()))
            for qf in qpos:
                qpts = _grid_query_points(torch, dev, grid_side)      # (N,2) in frame qf
                # reorder so the query frame is at position 0 (tracker convention)
                order = [qf] + [k for k in range(S) if k != qf]
                inv = np.argsort(order)
                stack_o = stack[order][None]                          # (1,S,3,H,W)
                fmaps_o = fmaps[:, order]
                # fine_tracking ON: sub-pixel refinement (better tracks → better BA). The
                # H!=W out-of-bounds bug in refine_track (clamped x to H instead of W) is
                # patched in our VGGT-Long fork, so this is safe on 280x504 inputs.
                ft, _ct, vis, score = tracker(
                    stack_o, qpts[None], fmaps=fmaps_o, fine_tracking=(not smoke))
                ft = ft[0][inv].float().cpu().numpy()                 # (S,N,2) back to window order
                vis = vis[0][inv].float().cpu().numpy()               # (S,N)
                # fine_tracking refines with compute_score=False → score is None; visibility
                # is then the only quality signal (use it for both gates).
                if score is not None:
                    score = score[0][inv].float().cpu().numpy()      # (S,N)
                else:
                    score = vis
                N = ft.shape[1]
                good = (vis > vis_thresh) & (score > score_thresh)
                seen = good.sum(axis=0)                               # per-track #frames seen
                keep = np.where(seen >= 2)[0]                         # a track needs ≥2 views
                for n in keep:
                    tid = next_tid + int(n)
                    fr = good[:, n]
                    fs = np.where(fr)[0]
                    for s in fs:
                        obs_track.append(tid)
                        obs_frame.append(kf_nums[widx[s]])
                        obs_uv.append(ft[s, n])
                        obs_vis.append(float(vis[s, n]))
                        obs_score.append(float(score[s, n]))
                    tq_id.append(tid)
                    tq_frame.append(kf_nums[widx[qf]])
                    tq_uv.append(qpts[int(n)].cpu().numpy())
                next_tid += N
        if (wi + 1) % 10 == 0 or wi == len(windows) - 1:
            logger.info(f"  window {wi+1}/{len(windows)}  obs={len(obs_track):,}  "
                        f"tracks≈{len(tq_frame):,}  ({time.time()-t0:.0f}s)")

    if not obs_track:
        logger.error("no tracks survived filtering")
        return None

    out_dir = output_dir / "ba_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("tracks_smoke.npz" if smoke else "tracks.npz")
    np.savez_compressed(
        out,
        obs_track=np.asarray(obs_track, np.int64),
        obs_frame=np.asarray(obs_frame, np.int64),
        obs_uv=np.asarray(obs_uv, np.float32),
        obs_vis=np.asarray(obs_vis, np.float32),
        obs_score=np.asarray(obs_score, np.float32),
        track_query_id=np.asarray(tq_id, np.int64),
        track_query_frame=np.asarray(tq_frame, np.int64),
        track_query_uv=np.asarray(tq_uv, np.float32),
        res_h=np.int64(RES_H), res_w=np.int64(RES_W),
    )
    # quick stats
    uniq_tracks = len(set(obs_track))
    per_track = np.bincount(np.asarray(obs_track))
    per_track = per_track[per_track > 0]
    logger.info(f"✅ wrote {out.name}: {len(obs_track):,} obs, {uniq_tracks:,} tracks, "
                f"mean {per_track.mean():.1f} views/track, "
                f"frames covered {len(set(obs_frame))}/{len(kf_nums)} "
                f"in {time.time()-t0:.0f}s")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--win", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--grid-side", type=int, default=48)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = extract_tracks(
        Path(args.output_dir), Path(args.frames_dir),
        win=args.win, stride=args.stride, grid_side=args.grid_side, smoke=args.smoke,
    )
    print(f"[TRACKS-RESULT] {out if out else 'NONE'}")
    if out is None:                       # NO FALLBACK: signal failure to the caller
        sys.exit(1)


if __name__ == "__main__":
    main()
