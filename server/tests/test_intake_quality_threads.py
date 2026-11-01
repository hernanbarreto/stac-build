"""I0 measured in worker threads (intake.quality.analyze_frames, 2026-11-01) against the frame-by-frame loop it
replaced: the same features for every frame, the inter-frame difference against the previous frame's diff-scale
gray, the same report."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from intake import quality as Q  # noqa: E402
from intake.config import load_intake_config  # noqa: E402


def _frames(tmp_path, n=23):
    d = tmp_path / "frames"
    d.mkdir()
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (180, 320), dtype=np.uint8)
    for i in range(n):
        img = np.roll(base, i * 3, axis=1)
        img = cv2.GaussianBlur(img, (0, 0), 0.5 + (i % 5))
        cv2.imwrite(str(d / f"{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return d


def _reference(frames_dir, cfg):
    rows, prev = [], None
    for p in Q.list_frames(frames_dir):
        f = Q.frame_features(Q.read_gray(p), prev, cfg)
        prev = f.pop("gray_small")
        rows.append({"frame": int(p.stem), "file": p.name, **f})
    return rows


def test_threads_give_the_loops_features(tmp_path, monkeypatch):
    d = _frames(tmp_path)
    cfg = load_intake_config().quality
    want = _reference(d, cfg)
    for threads in (1, 3, 8):
        monkeypatch.setattr(Q, "QUALITY_THREADS", threads)
        rep = Q.analyze_frames(d, cfg, log=lambda m: None, heartbeat_s=60.0, environment={"x": 1})
        got = [{k: fr[k] for k in ("frame", "fft", "laplacian", "luma_mean", "clip_frac", "inter_frame_diff")}
               for fr in rep["frames"]]
        ref = [{k: r[k] for k in ("frame", "fft", "laplacian", "luma_mean", "clip_frac", "inter_frame_diff")}
               for r in want]
        assert got == ref, threads
