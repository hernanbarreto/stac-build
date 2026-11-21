"""VERSION 2, step 2 — "Selección nueva de keyframes sobre ese mapa, del frame 0 al n, comparando cada frame con el
último keyframe elegido."

THE RULE (USER 2026-11-21, his design and his limits: "si, me cierran"):

  * the video is walked from its first frame to its last, each frame compared with THE LAST CHOSEN KEYFRAME;
  * overlap above 90 %: the same view again — skipped;
  * overlap between 90 % and 85 %: a candidate;
  * when the overlap falls under 85 % the BEST candidate is chosen and the walk goes on from it;
  * when the overlap jumps from above 90 % to under 85 % between two consecutive frames, BOTH are taken;
  * the first frame of the video is a keyframe (as today: the first usable frame);
  * no frame is rejected by its brightness, no 8-frame rule, no coverage fill.

OVERLAP of two frames a, b — on AMB3R's whole-video pass (intake/amb3r_video.npz), with the pipeline's own map of
what every reconstructed frame saw (reconstruction.coverage_fill.CoverageMap, on the card, inside THE RANGE): the
points frame a sees are the points of its record (its own frame's when AMB3R reconstructed it, else the nearest
reconstructed frame's — reconstruction.amb3r_probe.source_records, the pipeline's rule) that fall inside a's image at
a's own pose (AMB3R gives every frame a pose); o(a→b) = the share of them that fall inside b's image at b's pose;
overlap(a, b) = the smaller of o(a→b) and o(b→a). It is the footprint overlap of photogrammetric practice (forward
overlap), measured on the scene AMB3R reconstructed.

THE BEST FRAME of a band — ONE score: the image's detail with the light equalised. The detail is I0's own measure
(intake.quality.fft_score: the high-frequency energy of the frame at I0's analysis scale), taken on the frame after its
brightness and contrast are equalised ((gray − mean) / std): a darker or flatter exposure of the same view scores the
same. Ties → the earlier frame (I1's rule). The transform runs on the card.

    python v2_select.py <scan dir> [<amb3r_video.npz>]      → <scan>/intake/keyframes_v2.json
"""
import json
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

OVERLAP_REPEAT = 0.90      # above: the same view again (USER 2026-11-21)
OVERLAP_HOLE = 0.85        # below: a hole (USER 2026-11-21)
BATCH = 64                 # frames compared with the last keyframe at once
NAME = "keyframes_v2.json"


class Overlap:
    """overlap(a, b) of two video positions, on the card (module docstring).

    WHAT A FRAME SEES. AMB3R reconstructs one frame in four; a frame between two reconstructed ones sees what BOTH
    saw, each weighted by how near it is along the video (a reconstructed frame: its own record alone), and is projected
    with the K interpolated the same way. With the nearest record alone (the first version of this file) the measure
    stepped at every change of record — the view entering a frame's leading edge is in the NEXT record, not in the
    previous one — and the walk read four false jumps over the band on bufferStop."""

    def __init__(self, video, range_m: float, device: str = "cuda"):
        import torch
        from reconstruction.coverage_fill import CoverageMap
        self.t = torch
        self.cmap = CoverageMap(video, device, state=False, range_m=float(range_m))
        dev = self.cmap.dev
        self.frame_no = [int(f) for f in video["frame_no"]]
        self.n = len(self.frame_no)
        rec = np.asarray(video["rec_index"], np.int64)                 # video positions of the reconstructed frames
        if np.any(np.diff(rec) <= 0):
            raise ValueError("AMB3R's reconstructed frames are not in video order")
        pos = np.arange(self.n)
        nxt = np.clip(np.searchsorted(rec, pos, side="left"), 0, len(rec) - 1)     # first record at or after
        prv = np.clip(np.searchsorted(rec, pos, side="right") - 1, 0, len(rec) - 1)  # last record at or before
        span = (rec[nxt] - rec[prv]).astype(np.float64)
        w = np.where(span > 0, (pos - rec[prv]) / np.where(span > 0, span, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)                                       # before the first / after the last record: 0
        self.prv, self.nxt, self.w = prv, nxt, w
        Kr = self.cmap.K.double()
        wt = torch.as_tensor(w, dtype=torch.float64, device=dev)[:, None, None]
        self.K = ((1.0 - wt) * Kr[torch.as_tensor(prv, device=dev)] + wt * Kr[torch.as_tensor(nxt, device=dev)]).float()
        c2w = torch.as_tensor(np.asarray(video["pred"], np.float64), device=dev)
        R, C = c2w[:, :3, :3], c2w[:, :3, 3]
        self.R_w2c = R.transpose(1, 2).float()
        self.t_w2c = (-(R.transpose(1, 2) @ C[:, :, None])[:, :, 0]).float()
        self._own = {}

    def _inside(self, idx, X):
        """(len(idx), len(X)) — which of the points X fall inside the image of each video position, at its own pose."""
        t, cm = self.t, self.cmap
        Xc = t.einsum("kab,nb->kna", self.R_w2c[idx], X) + self.t_w2c[idx][:, None, :]
        z = Xc[..., 2]
        zs = t.where(z > 1e-6, z, t.ones_like(z))
        K = self.K[idx]
        u = t.round(K[:, 0, 0, None] * Xc[..., 0] / zs + K[:, 0, 2, None])
        v = t.round(K[:, 1, 1, None] * Xc[..., 1] / zs + K[:, 1, 2, None])
        return (z > 1e-3) & (u >= 0) & (u < cm.Ws) & (v >= 0) & (v < cm.Hs)

    def sees(self, i: int):
        """What video position i sees: [(points inside its image, weight)] over its one or two records."""
        if i not in self._own:
            t = self.t
            parts = []
            for r, wr in ((int(self.prv[i]), 1.0 - float(self.w[i])), (int(self.nxt[i]), float(self.w[i]))):
                if wr <= 0.0:
                    continue
                X = self.cmap.X[self.cmap.own(r)]
                P = X[self._inside(t.as_tensor([i], device=X.device), X)[0]]
                if len(P):
                    parts.append((P, wr))
            tot = sum(wr for _, wr in parts)
            self._own[i] = [(P, wr / tot) for P, wr in parts]
            if len(self._own) > 4 * BATCH:
                for k in list(self._own)[:-2 * BATCH]:          # the oldest entries (the newest is the one returned)
                    del self._own[k]
        return self._own[i]

    def with_last(self, a: int, bs):
        """overlap(a, b) for every b of ``bs`` (video positions) — float64 numpy."""
        t = self.t
        dev = self.cmap.dev
        idx = t.as_tensor([int(b) for b in bs], device=dev)
        ia = t.as_tensor([a], device=dev)
        ab = t.zeros(len(bs), dtype=t.float64, device=dev)
        for P, wr in self.sees(a):
            ab += wr * self._inside(idx, P).double().mean(1)
        ba = []
        for b in bs:
            x = t.zeros((), dtype=t.float64, device=dev)
            for P, wr in self.sees(int(b)):
                x = x + wr * self._inside(ia, P)[0].double().mean()
            ba.append(x)
        return t.minimum(ab, t.stack(ba)).cpu().numpy()


class Detail:
    """The score of a frame: I0's detail measure on the frame with its light equalised (module docstring)."""

    def __init__(self, frames_dir: Path, fft_max_side: int, device: str = "cuda"):
        import torch
        self.t, self.dir, self.side = torch, Path(frames_dir), int(fft_max_side)
        self.dev = torch.device(device)
        self.cache = {}

    def __call__(self, frame: int) -> float:
        if frame not in self.cache:
            import cv2
            from intake.quality import downscale_gray, frame_file, read_gray
            t = self.t
            g = downscale_gray(read_gray(frame_file(self.dir, frame)), self.side, cv2.INTER_AREA)
            x = t.as_tensor(np.ascontiguousarray(g), dtype=t.float64, device=self.dev)
            sd = x.std(unbiased=False)
            if float(sd) <= 0.0:
                self.cache[frame] = 0.0                       # a uniform image holds no detail
            else:
                mag = t.fft.fftshift(t.fft.fft2((x - x.mean()) / sd)).abs()
                h, w = mag.shape
                cy, cx, r = h // 2, w // 2, min(h, w) // 8
                mag[cy - r:cy + r, cx - r:cx + r] = 0           # intake.quality.fft_score's low-frequency block
                self.cache[frame] = float(mag.mean())
        return self.cache[frame]


def select(ov: Overlap, score, log=print):
    """The keyframes (video positions, in order) and the record of how each was chosen."""
    n = ov.n
    chosen = [{"pos": 0, "rule": "first_frame", "overlap_with_previous": None, "candidates": 1}]
    last, i = 0, 1
    cands = []                                   # (pos, overlap) inside the band since the last keyframe
    prev_ov = 1.0                                # the overlap of frame i - 1 with the last keyframe
    while i < n:
        bs = list(range(i, min(n, i + BATCH)))
        o = ov.with_last(last, bs)
        restart = None
        for b, x in zip(bs, o):
            x = float(x)
            if x > OVERLAP_REPEAT:
                prev_ov = x
                continue
            if x >= OVERLAP_HOLE:
                cands.append((b, x))
                prev_ov = x
                continue
            # under 85 %: the band is closed
            if cands:
                best = max(cands, key=lambda c: (score(ov.frame_no[c[0]]), -c[0]))
                chosen.append({"pos": best[0], "rule": "best_of_band", "overlap_with_previous": best[1],
                               "candidates": len(cands), "score": score(ov.frame_no[best[0]])})
                last = best[0]
            else:
                # the overlap jumped over the band between two consecutive frames: both are taken
                if b - 1 > last:
                    chosen.append({"pos": b - 1, "rule": "jump_before", "overlap_with_previous": prev_ov,
                                   "candidates": 0})
                chosen.append({"pos": b, "rule": "jump_after", "overlap_with_previous": x, "candidates": 0})
                last = b
            restart = last + 1
            break
        if restart is None:
            i = bs[-1] + 1
        else:
            i, cands, prev_ov = restart, [], 1.0
    if cands:                                     # the video ended inside a band: its best frame closes the walk
        best = max(cands, key=lambda c: (score(ov.frame_no[c[0]]), -c[0]))
        chosen.append({"pos": best[0], "rule": "best_of_last_band", "overlap_with_previous": best[1],
                       "candidates": len(cands), "score": score(ov.frame_no[best[0]])})
    for c in chosen:
        c["frame"] = ov.frame_no[c.pop("pos")]
    return chosen


def consecutive_overlaps(ov: Overlap, frames):
    """overlap of every keyframe with the one before it (the check of the result)."""
    pos = {f: i for i, f in enumerate(ov.frame_no)}
    out = []
    for a, b in zip(frames[:-1], frames[1:]):
        out.append(float(ov.with_last(pos[a], [pos[b]])[0]))
    return np.asarray(out)


def main() -> int:
    scan = Path(sys.argv[1])
    npz = Path(sys.argv[2]) if len(sys.argv) > 2 else scan / "intake" / "amb3r_video.npz"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else scan / "intake" / NAME
    import yaml
    from intake.config import load_intake_config
    from intake.run_config import run_config_path
    from reconstruction import amb3r_probe as A3
    from view_range import view_range_m
    cfg_p = run_config_path(scan)
    config = yaml.safe_load((cfg_p if cfg_p.exists() else Path(SERVER) / "config.yaml").read_text())
    video = A3.load_video(npz)
    if "metric_scale" not in video:
        raise SystemExit(f"{npz} is not in metres (no metric_scale): the range cannot be applied")
    rng = float(view_range_m(config))
    ov = Overlap(video, rng)
    score = Detail(scan / "frames", load_intake_config(config).quality.fft_max_side)
    log = lambda m: print(m, flush=True)   # noqa: E731
    log(f"[select] {ov.n} video frames with a pose, {ov.cmap.F} reconstructed by AMB3R (their records say what every "
        f"frame saw), range {rng:g} m; band: overlap {OVERLAP_HOLE:.0%} .. {OVERLAP_REPEAT:.0%} with the last keyframe")
    chosen = select(ov, score, log)
    frames = [c["frame"] for c in chosen]
    co = consecutive_overlaps(ov, frames)
    rules = {}
    for c in chosen:
        rules[c["rule"]] = rules.get(c["rule"], 0) + 1
    gaps = np.diff(frames)
    doc = {"version": 1, "rule": {"overlap_repeat": OVERLAP_REPEAT, "overlap_hole": OVERLAP_HOLE,
                                  "overlap": "min of the two directions, AMB3R's records inside each frame's image at "
                                             "its own pose, inside the range",
                                  "best": "intake.quality.fft_score on the frame with its light equalised "
                                          "((gray - mean) / std)", "range_m": rng},
           "video": str(npz.name), "n_video_frames": ov.n, "n_keyframes": len(frames), "frames": frames,
           "keyframes": chosen, "by_rule": rules,
           "overlap_between_consecutive_keyframes": {"min": float(co.min()), "median": float(np.median(co)),
                                                     "max": float(co.max()),
                                                     "under_hole_bar": int((co < OVERLAP_HOLE).sum()),
                                                     "over_repeat_bar": int((co > OVERLAP_REPEAT).sum())},
           "frames_between_keyframes": {"min": int(gaps.min()), "median": float(np.median(gaps)),
                                        "max": int(gaps.max())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out)
    log(f"[select] {len(frames)} keyframes of {ov.n} frames; by rule {rules}; overlap between consecutive keyframes "
        f"min {co.min():.3f} / median {np.median(co):.3f} / max {co.max():.3f} ({int((co < OVERLAP_HOLE).sum())} under "
        f"{OVERLAP_HOLE:.0%}, {int((co > OVERLAP_REPEAT).sum())} over {OVERLAP_REPEAT:.0%}); video frames between "
        f"keyframes min {gaps.min()} / median {np.median(gaps):g} / max {gaps.max()} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
