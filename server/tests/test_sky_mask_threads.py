"""The fork's sky masks computed on threads (vendor/VGGT-Long/loop_utils/sky_mask.py
apply_sky_masks → _segment_all, SKY_THREADS): the same masks, the same counts and the same
masked confidences as one thread; a frame listed twice is segmented once and counted reused the
second time, as the loop counted it; a session is used by one thread at a time and kept for the
next call; an error stops the call."""
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))
from loop_utils import sky_mask as SM  # noqa: E402


def _frames(tmp_path, n=17):
    imgs = []
    for k in range(n):
        p = tmp_path / f"{k:06d}.jpg"
        p.write_bytes(f"img-{k}".encode() * (k + 1))
        imgs.append(str(p))
    return imgs


def _segmenter(log):
    import cv2

    def segment(path, session, mask_path):
        with log["lock"]:
            log["calls"].append(os.path.basename(path))
            assert session not in log["busy"], "a session used by two threads at once"
            log["busy"].add(session)
        k = int(Path(path).stem)
        m = np.full((8, 10), 255, np.uint8)
        m[: (k % 5) + 1] = 0
        cv2.imwrite(mask_path, m)
        with log["lock"]:
            log["busy"].discard(session)
    return segment


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_threads_give_the_one_threads_masks_and_counts(tmp_path, monkeypatch, threads):
    model = tmp_path / "skyseg.onnx"
    model.write_bytes(b"model-bytes")
    imgs = _frames(tmp_path)
    paths = imgs + [imgs[3], imgs[0]]                     # two frames listed twice
    log = {"calls": [], "busy": set(), "lock": threading.Lock(), "made": 0}

    def factory(p):
        with log["lock"]:
            log["made"] += 1
            return f"session-{log['made']}"
    monkeypatch.setattr(SM, "SKY_THREADS", threads)
    conf = np.ones((len(paths), 8, 10), np.float32)
    r = SM.apply_sky_masks(conf, paths, str(tmp_path / "cache"), model_path=str(model),
                           segment=_segmenter(log), session_factory=factory, log=lambda *a: None)
    assert r["n_computed"] == len(imgs) and r["n_reused"] == 2
    assert sorted(log["calls"]) == sorted(os.path.basename(p) for p in imgs)
    for i, p in enumerate(paths):
        k = int(Path(p).stem)
        assert conf[i, : (k % 5) + 1].sum() == 0 and conf[i, (k % 5) + 1:].min() == 1.0
    assert log["made"] <= threads
    # the next call reuses the sessions it made (no new one) and every mask
    made = log["made"]
    (tmp_path / "000099.jpg").write_bytes(b"one more")
    conf2 = np.ones((2, 8, 10), np.float32)
    r2 = SM.apply_sky_masks(conf2, [imgs[1], str(tmp_path / "000099.jpg")], str(tmp_path / "cache"),
                            model_path=str(model), segment=_segmenter(log), session_factory=factory,
                            log=lambda *a: None)
    assert r2["n_computed"] == 1 and r2["n_reused"] == 1 and log["made"] == made


def test_an_error_stops_the_call(tmp_path, monkeypatch):
    model = tmp_path / "skyseg.onnx"
    model.write_bytes(b"m")
    imgs = _frames(tmp_path, 9)
    monkeypatch.setattr(SM, "SKY_THREADS", 4)

    def bad(path, session, mask_path):
        if path.endswith("000005.jpg"):
            raise RuntimeError("segmenter failed on 000005")
        import cv2
        cv2.imwrite(mask_path, np.full((8, 10), 255, np.uint8))
    with pytest.raises(RuntimeError, match="000005"):
        SM.apply_sky_masks(np.ones((9, 8, 10), np.float32), imgs, str(tmp_path / "c"),
                           model_path=str(model), segment=bad, session_factory=lambda p: object(),
                           log=lambda *a: None)
