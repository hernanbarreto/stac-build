"""Step 1 of the architecture (ARQUITECTURA_mapa.md): THE GOOD FRAMES — on the card.

A frame is bad when (in this order):
  * "burnt": its mean luma is above LUMA_BURNT (220);
  * "saturated": more than BURNT_SHARE_MAX (25 %) of its pixels are burnt (≥ clip_hi, 250);
  * "detail": its detail is under SHARP_MIN (0.5) of the median detail of its SHARP_NEIGHBOURS (15) neighbours on each
    side (every frame of the video, the frame itself left out; fewer at the two ends).
Darkness rejects nothing. AMB3R reads every frame anyway; a bad frame can be neither a keyframe nor an F4 witness.

THE MEASURES ARE I0's (intake.quality), the ones the thresholds were measured with on the five sessions (2026-11-22):
the gray is the frame read as gray (cv2.IMREAD_GRAYSCALE: the JPEG's Y plane); the luma is its mean; the burnt share
the share of its pixels ≥ clip_hi; the detail is fft_score of the gray downscaled to fft_max_side with cv2's
INTER_AREA.

TWO PATHS. "numpy" = I0's own functions, frame by frame (the record). "cuda" = the same arithmetic on the card: the Y
plane decoded by nvJPEG (jpeg_gpu: libjpeg-turbo's integer IDCT), luma and burnt share as integer counts, cv2's
INTER_AREA downscale written in float32 in OpenCV's own order (computeResizeAreaTab, then rows then columns, rounded
half to even), the FFT in float64, numpy's median (the mean of the two middle values when the count is even).

    python arq3_frames.py <scan dir> <out.json> [cuda|numpy]"""
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

LUMA_BURNT = 220.0
BURNT_SHARE_MAX = 0.25
SHARP_MIN = 0.5
SHARP_NEIGHBOURS = 15
REASONS = ("burnt", "saturated", "detail")


def quality_knobs():
    import yaml
    from intake.config import load_intake_config
    q = load_intake_config(yaml.safe_load((Path(SERVER) / "config.yaml").read_text())).quality
    return int(q.fft_max_side), int(q.clip_hi)


def frame_list(frames_dir: Path):
    from intake.quality import list_frames
    paths = list_frames(Path(frames_dir))
    return [int(p.stem) for p in paths], paths


def digest(a: np.ndarray) -> str:
    """The downscaled gray's identity: its shape and the sha1 of its bytes."""
    a = np.ascontiguousarray(a)
    return f"{a.shape[0]}x{a.shape[1]}:" + hashlib.sha1(a.tobytes()).hexdigest()


# ── the record: I0's own functions ────────────────────────────────────────────────────────────────────────────────
def _worker_init():
    import cv2
    cv2.setNumThreads(1)


def _numpy_one(job):
    path, fft_max_side, clip_hi = job
    import cv2
    from intake.quality import downscale_gray, fft_score, read_gray
    g = read_gray(path)
    small = downscale_gray(g, fft_max_side, cv2.INTER_AREA)
    return (float(g.mean()), float(np.count_nonzero(g >= clip_hi)) / float(g.size), fft_score(small), digest(small))


def measures_numpy(paths, fft_max_side: int, clip_hi: int, workers: int = 1):
    """luma, burnt share, fft, digests — I0's functions, frame by frame (in `workers` processes, in order)."""
    jobs = [(str(p), fft_max_side, clip_hi) for p in paths]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_worker_init) as ex:
            rows = list(ex.map(_numpy_one, jobs, chunksize=32))
    else:
        rows = [_numpy_one(j) for j in jobs]
    luma, burnt, fft, dig = zip(*rows)
    return np.asarray(luma), np.asarray(burnt), np.asarray(fft), list(dig)


# ── the card ──────────────────────────────────────────────────────────────────────────────────────────────────────
def area_tab(ssize: int, dsize: int, scale: float):
    """OpenCV's computeResizeAreaTab (imgproc/resize.cpp): for every output index, its (source index, weight) in
    OpenCV's order — the partial cell first, the whole ones, the partial last; weights as float32."""
    out = []
    for d in range(dsize):
        fs1 = d * scale
        fs2 = fs1 + scale
        cell = min(scale, ssize - fs1)
        s1, s2 = math.ceil(fs1), math.floor(fs2)
        s2 = min(s2, ssize - 1)
        s1 = min(s1, s2)
        taps = []
        if s1 - fs1 > 1e-3:
            taps.append((s1 - 1, np.float32((s1 - fs1) / cell)))
        for s in range(s1, s2):
            taps.append((s, np.float32(1.0 / cell)))
        if fs2 - s2 > 1e-3:
            taps.append((s2, np.float32(min(min(fs2 - s2, 1.0), cell) / cell)))
        out.append(taps)
    return out


def _tap_tensors(tab, device):
    import torch
    T = max(len(t) for t in tab)
    idx = np.zeros((T, len(tab)), np.int64)
    w = np.zeros((T, len(tab)), np.float32)
    for d, taps in enumerate(tab):
        for k, (s, a) in enumerate(taps):
            idx[k, d], w[k, d] = s, a
    return torch.as_tensor(idx, device=device), torch.as_tensor(w, device=device)


class AreaDown:
    """cv2.resize(gray, None, fx=s, fy=s, interpolation=INTER_AREA) for a downscale by a non-integer factor, on the
    card: the generic path of OpenCV (ResizeArea_Invoker with float sums) in the same order of operations."""

    def __init__(self, h: int, w: int, max_side: int, device: str = "cuda"):
        self.hw = (h, w)
        s = max_side / max(h, w)                                       # intake.quality.downscale_gray
        self.dw, self.dh = int(round(w * s)), int(round(h * s))      # cv::resize: saturate_cast<int>(w * fx)
        sx, sy = 1.0 / s, 1.0 / s                                    # hal::resize: scale = 1 / inv_scale
        if abs(sx - round(sx)) < 2.2e-16:
            raise ValueError(f"an integer downscale ({sx}) takes OpenCV's fast area path, not written here")
        self.xi, self.xw = _tap_tensors(area_tab(w, self.dw, sx), device)
        self.yi, self.yw = _tap_tensors(area_tab(h, self.dh, sy), device)

    def __call__(self, g):
        """g: (n, h, w) uint8 on the card → (n, dh, dw) uint8."""
        import torch
        if tuple(g.shape[1:]) != self.hw:
            raise ValueError(f"a frame of {tuple(g.shape[1:])} among frames of {self.hw}")
        S = g.to(torch.float32)
        buf = torch.zeros((S.shape[0], S.shape[1], self.dw), dtype=torch.float32, device=S.device)
        for k in range(self.xi.shape[0]):                            # buf[dx] += S[si] * alpha, in the tab's order
            buf = buf + S[:, :, self.xi[k]] * self.xw[k]
        acc = torch.zeros((S.shape[0], self.dh, self.dw), dtype=torch.float32, device=S.device)
        for k in range(self.yi.shape[0]):                            # sum[dx] += beta * buf[dx], in the tab's order
            acc = acc + buf[:, self.yi[k], :] * self.yw[k][None, :, None]
        return torch.round(acc).clamp(0, 255).to(torch.uint8)       # saturate_cast<uchar>: cvRound, half to even


def fft_score_t(small):
    """intake.quality.fft_score on the card, float64, a batch (n, h, w)."""
    import torch
    f = torch.fft.fftshift(torch.fft.fft2(small.to(torch.float64)), dim=(-2, -1))
    mag = f.abs()
    h, w = mag.shape[-2:]
    cy, cx, r = h // 2, w // 2, min(h, w) // 8
    mag[:, cy - r:cy + r, cx - r:cx + r] = 0
    return mag.mean(dim=(-2, -1))


def measures_cuda(paths, fft_max_side: int, clip_hi: int, digests: bool = False):
    """luma, burnt share, fft (and the downscaled grays' digests) — on the card, a batch of frames at a time."""
    import torch
    import jpeg_gpu as JG
    nv = JG._nvjpeg()
    luma, burnt, fft, dig = [], [], [], []
    down = None
    with ThreadPoolExecutor(8) as io:
        for s in range(0, len(paths), JG.BATCH):
            part = paths[s:s + JG.BATCH]
            datas = list(io.map(lambda p: Path(p).read_bytes(), part))
            names = [str(p) for p in part]
            lay = JG.read_layout(datas[0], names[0])
            for d, nm in zip(datas[1:], names[1:]):
                if JG.read_layout(d, nm) != lay:
                    raise ValueError(f"{nm}: another JPEG layout than {names[0]}")
            y = nv.planes(lay, datas, names, None)[0]                 # (n, H, W) uint8: the Y plane
            n, H, W = y.shape
            # the pixel count as a tensor ON THE CARD: divided by a Python number, PyTorch's CUDA division multiplies
            # by its reciprocal (one bit off at times); by a device tensor it is IEEE division, numpy's own
            tot = torch.tensor(float(H * W), dtype=torch.float64, device=y.device)
            luma.append((y.sum(dim=(1, 2), dtype=torch.int64).to(torch.float64) / tot).cpu())
            burnt.append(((y >= clip_hi).sum(dim=(1, 2), dtype=torch.int64).to(torch.float64) / tot).cpu())
            if max(H, W) > fft_max_side:
                if down is None:
                    down = AreaDown(H, W, fft_max_side, "cuda")
                small = down(y)
            else:
                small = y
            fft.append(fft_score_t(small).cpu())
            if digests:
                sm = small.cpu().numpy()
                dig.extend(digest(sm[k]) for k in range(n))
    return torch.cat(luma).numpy(), torch.cat(burnt).numpy(), torch.cat(fft).numpy(), dig


def neighbour_median(fft: np.ndarray, device: str):
    """numpy's median of the SHARP_NEIGHBOURS frames on each side of every frame (itself left out)."""
    n, W = len(fft), SHARP_NEIGHBOURS
    if device == "numpy":
        return np.array([np.median(np.r_[fft[max(0, i - W):i], fft[i + 1:i + 1 + W]]) for i in range(n)])
    import torch
    v = torch.as_tensor(fft, dtype=torch.float64, device=device)
    i = torch.arange(n, device=device)[:, None]
    off = torch.cat([torch.arange(-W, 0, device=device), torch.arange(1, W + 1, device=device)])[None, :]
    j = i + off
    ok = (j >= 0) & (j < n)
    near = v[j.clamp(0, n - 1)]
    vals = torch.where(ok, near, torch.full_like(near, float("inf")))
    srt, _ = torch.sort(vals, dim=1)
    c = ok.sum(1)
    lo = torch.gather(srt, 1, ((c - 1) // 2)[:, None])[:, 0]
    hi = torch.gather(srt, 1, (c // 2)[:, None])[:, 0]
    return ((lo + hi) / 2.0).cpu().numpy()                           # odd count: lo = hi; even: their mean


def decide(luma, burnt, fft, med):
    why = []
    for i in range(len(luma)):
        if luma[i] > LUMA_BURNT:
            why.append("burnt")
        elif burnt[i] > BURNT_SHARE_MAX:
            why.append("saturated")
        elif fft[i] < SHARP_MIN * med[i]:
            why.append("detail")
        else:
            why.append(None)
    return why


def good_frames(scan: Path, device: str = "cuda"):
    """{frame: None | reason} of every frame of the scan."""
    fft_max_side, clip_hi = quality_knobs()
    frames, paths = frame_list(Path(scan) / "frames")
    if device == "numpy":
        luma, burnt, fft, _d = measures_numpy(paths, fft_max_side, clip_hi)
    else:
        luma, burnt, fft, _d = measures_cuda(paths, fft_max_side, clip_hi)
    med = neighbour_median(fft, device)
    return dict(zip(frames, decide(luma, burnt, fft, med)))


def main() -> int:
    scan, dst = Path(sys.argv[1]), Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    t0 = time.time()
    why = good_frames(scan, device)
    bad = {f: r for f, r in why.items() if r}
    counts = {r: sum(1 for x in bad.values() if x == r) for r in REASONS}
    doc = {"rule": {"luma_burnt": LUMA_BURNT, "burnt_share_max": BURNT_SHARE_MAX, "sharp_min": SHARP_MIN,
                    "sharp_neighbours": SHARP_NEIGHBOURS},
           "n_frames": len(why), "n_good": len(why) - len(bad), "bad": {str(f): r for f, r in sorted(bad.items())},
           "good": [f for f, r in sorted(why.items()) if r is None]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc))
    print(f"[frames] {len(why) - len(bad)} good frames of {len(why)}; bad: {counts} — {sorted(bad)} "
          f"({device}, {time.time() - t0:.0f} s) → {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
