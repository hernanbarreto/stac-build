"""I1 prerequisite — the session camera measured by the DA3 focal probe (no GPU:
the DA3 window is written by hand)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import focal as F                                   # noqa: E402
from intake.parallax import QUALITY_VERSION                     # noqa: E402


def _quality(n=100, unusable=(3, 4, 50)):
    frames = [{"frame": i, "file": f"{i:06d}.jpg", "sharp_rank": 0.5, "usable": i not in unusable}
              for i in range(n)]
    return {"version": QUALITY_VERSION, "frames": frames, "native_w": 464, "native_h": 832}


def test_probe_frames_are_usable_and_spread_over_the_video():
    files = F.probe_files(_quality(), 16)
    assert len(files) == 16 and files[0] == "000000.jpg" and files[-1] == "000099.jpg"
    assert not {"000003.jpg", "000004.jpg", "000050.jpg"} & set(files)
    assert files == sorted(files)
    with pytest.raises(F.FocalError):
        F.probe_files(_quality(n=3, unusable=(0, 1)), 16)


def test_grid_intrinsics_carry_to_native_pixels_exactly():
    # a native camera, the same camera as DA3 sees it on a full-frame resize
    W, H, w, h = 464, 832, 280, 504
    K = np.array([[420.0, 0.0, 231.5], [0.0, 421.0, 415.5], [0.0, 0.0, 1.0]])
    sx, sy = w / W, h / H
    Kg = K.copy()
    Kg[0, 0], Kg[1, 1] = K[0, 0] * sx, K[1, 1] * sy
    Kg[0, 2], Kg[1, 2] = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    Kn = F.native_intrinsics(np.stack([Kg, Kg]), (w, h), (W, H))
    assert np.allclose(Kn[0], K, atol=1e-9) and np.allclose(Kn[1], K, atol=1e-9)


def test_summary_is_the_median_with_its_spread():
    ks = []
    for fx in (400.0, 410.0, 420.0, 430.0, 1000.0):          # one outlier frame
        ks.append(np.array([[fx, 0, 232.0], [0, fx, 416.0], [0, 0, 1.0]]))
    s = F.summarise(np.stack(ks))
    assert s["fx"] == 420.0 and s["cx"] == 232.0
    assert np.allclose(np.asarray(s["K"]), [[420.0, 0, 232.0], [0, 420.0, 416.0], [0, 0, 1]])
    assert s["spread_pct"]["fx"] > 0 and s["spread_pct"]["cx"] == 0.0


def test_probe_is_reused_while_its_spec_is_unchanged(tmp_path, monkeypatch):
    from intake.config import load_intake_config
    from dataclasses import replace
    pc = replace(load_intake_config().parallax, focal_probe_res=1008)
    sess = tmp_path / "s"
    (sess / "frames").mkdir(parents=True)
    q = _quality()
    spec = {"windows": [[str(sess / "frames" / f) for f in F.probe_files(q, pc.focal_probe_frames)]],
            "process_res": int(pc.focal_probe_res), "model_id": str(pc.focal_probe_model)}
    (sess / "intake").mkdir()
    doc = {"version": F.FOCAL_VERSION, "spec": spec, "K": np.eye(3).tolist(),
           "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    (sess / "intake" / F.FOCAL_NAME).write_text(json.dumps(doc))
    monkeypatch.setattr(F.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DA3 ran")))
    out = F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)
    assert out["K"] == np.eye(3).tolist()
