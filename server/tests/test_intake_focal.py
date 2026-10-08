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


A100 = "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"
IDENT = {"card": A100, "weights": {"revision": "b2359bdf"}, "torch": {"version": "2.5"},
         "code": {"server/extract_da3_depth.py": "x"}, "autocast_dtype": "bfloat16"}


def _write_frames(sess, files, salt=0):
    (sess / "frames").mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        (sess / "frames" / f).write_bytes(bytes([i % 251, salt, 7]) * 5)


def test_probe_is_reused_on_its_stamp_only(tmp_path, monkeypatch):
    """The reuse key (docs/plan_determinismo.md points 66 / 70) is the stamp of the frames'
    BYTES, the committed layout, resolution, model, the card table's sizing, the DA3 identity
    (weights, code, torch, card), the probe parameters and the CPU environment — the product
    names its frames relative to the session, so a copied session keeps it; frames re-extracted
    under the same names, another identity or another card do not."""
    import shutil
    from intake.config import load_intake_config
    from dataclasses import replace
    from intake.vram import window_size
    from intake import focal_probe as FP
    import card_table
    pc = replace(load_intake_config().parallax, focal_probe_res=840)       # the table's entry
    sess = tmp_path / "s"
    q = _quality()
    files = F.probe_files(q, pc.focal_probe_frames)
    _write_frames(sess, files)
    import intake.vram as V
    import intake.walk as W
    monkeypatch.setattr(W, "da3_identity", lambda python, model_id, log=print: dict(IDENT))
    sizing = window_size(str(pc.focal_probe_model), 840, (464, 832), requested=len(files),
                         margin_frac=float(pc.vram_margin_frac), card_key=A100, log=lambda m: None)
    assert sizing["window_frames"] == len(files) == 16                      # pccr: one window
    layout = card_table.probe_window_frames(str(pc.focal_probe_model), 840)
    spec = FP.probe_spec([files], process_res=840, model_id=str(pc.focal_probe_model),
                         probe_layout=layout, window_sizing=sizing, da3_environment=IDENT)
    assert spec["windows"] == [[f"frames/{f}" for f in files]]              # session-relative
    (sess / "intake").mkdir()
    doc = {"version": F.FOCAL_VERSION, "spec": spec, "stamp": FP.probe_stamp(sess, spec, pc),
           "K": np.eye(3).tolist(), "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    (sess / "intake" / F.FOCAL_NAME).write_text(json.dumps(doc))
    monkeypatch.setattr(F.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DA3 ran")))
    out = F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)
    assert out["K"] == np.eye(3).tolist()
    # the session copied elsewhere: still reused (no absolute path in the key)
    dst = tmp_path / "elsewhere" / "s"
    shutil.copytree(sess, dst)
    assert F.run_focal_probe(dst, q, pc, python="python", log=lambda *a: None)["K"] == np.eye(3).tolist()
    # a frame re-extracted under the same name (other bytes): NOT reused — DA3 would run
    logs = []
    _write_frames(dst, files[:1], salt=9)
    with pytest.raises(AssertionError, match="DA3 ran"):
        F.run_focal_probe(dst, q, pc, python="python", log=logs.append)
    assert any("not reused" in m and f"frames/{files[0]}" in m for m in logs)
    # another DA3 identity (other weights, another card): the probe is NOT reused — DA3 would run
    monkeypatch.setattr(W, "da3_identity", lambda python, model_id, log=print: dict(IDENT, card="X"))
    monkeypatch.setattr(V, "window_size", lambda *a, **k: sizing)
    with pytest.raises(AssertionError, match="DA3 ran"):
        F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)
    # an older document (version 1, absolute paths as the key) is never reused
    old = {"version": 1, "spec": {"windows": [[str(sess / "frames" / f) for f in files]]},
           "K": np.eye(3).tolist(), "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    (sess / "intake" / F.FOCAL_NAME).write_text(json.dumps(old))
    monkeypatch.setattr(W, "da3_identity", lambda python, model_id, log=print: dict(IDENT))
    with pytest.raises(AssertionError, match="DA3 ran"):
        F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)


def test_the_probe_never_halves_a_window_and_needs_the_card_free():
    src = Path(F.__file__).read_text()
    body = src[src.index("def run_focal_probe"):src.index("def default_probe")]
    assert "repro.require_exclusive_gpu(log=log)" in body and "da3_weights.hf_env(" in body
    assert "OOM_EXIT" in body and "never halved" in body and "// 2" not in body
    assert body.index("require_exclusive_gpu") < body.index("subprocess.Popen(")


def test_probe_windows_never_exceed_the_layout_and_never_hold_a_lone_frame():
    """Point 64: the validated layouts stand (16 → [16]; 6 → [6, 6, 4]); a lone last frame is paired
    from the previous window, never folded INTO it (the old rule made a window of n_win + 1)."""
    items = [f"{i:06d}.jpg" for i in range(16)]
    assert F.probe_windows(items, 16) == ([items], [])
    w, d = F.probe_windows(items, 6)
    assert [len(x) for x in w] == [6, 6, 4] and d == [] and sum(w, []) == items
    w, d = F.probe_windows(items, 3)
    assert [len(x) for x in w] == [3, 3, 3, 3, 2, 2] and d == [] and sum(w, []) == items
    w, d = F.probe_windows(items, 5)
    assert [len(x) for x in w] == [5, 5, 4, 2] and d == []
    w, d = F.probe_windows(items[:5], 2)                         # [2, 2, 1]: no pairing keeps ≥ 2
    assert [len(x) for x in w] == [2, 2] and d == [items[4]]
    assert F.probe_windows(items[:1], 4) == ([items[:1]], [])     # a single frame: the caller's problem
    assert all(len(x) <= 3 for x in F.probe_windows(items, 3)[0])


def test_the_probe_runs_the_committed_layout_and_fails_on_a_card_too_small(tmp_path, monkeypatch):
    import intake.vram as V
    import intake.walk as W
    from intake.config import load_intake_config
    from dataclasses import replace
    pc = replace(load_intake_config().parallax, focal_probe_res=840)
    sess = tmp_path / "s"
    q = _quality()
    _write_frames(sess, F.probe_files(q, pc.focal_probe_frames))      # the stamp hashes them
    monkeypatch.setattr(W, "da3_identity", lambda python, model_id, log=print: dict(IDENT))
    monkeypatch.setattr(F.repro if hasattr(F, "repro") else sys.modules["repro"], "require_exclusive_gpu",
                        lambda log=print: {})
    launched = []

    class P:
        returncode = 0
        stdout = iter([])

        def wait(self):
            return 0

        def terminate(self):
            pass

    def _popen(cmd, **kw):
        args = dict(zip(cmd, cmd[1:]))
        spec = json.loads(Path(args["--windows_json"]).read_text())
        launched.append(spec)
        out = Path(args["--output_dir"])
        for i, win in enumerate(spec["windows"]):
            n = len(win)
            np.savez(out / f"window_{i:04d}.npz", depth=np.zeros((n, 4, 6), np.float32),
                     intrinsics=np.tile(np.array([[20.0, 0, 3.0], [0, 20.0, 2.0], [0, 0, 1.0]]), (n, 1, 1)))
        return P()
    monkeypatch.setattr(F.subprocess, "Popen", _popen)
    doc = F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)
    assert doc["spec"]["probe_layout"]["window_frames"] == 16
    assert [len(w) for w in doc["spec"]["windows"]] == [16]                  # pccr: one window
    assert launched[0]["probe_layout"]["window_frames"] == 16 and doc["fx"] > 0
    # the product: relative paths, a stamp, epoch 0, no time; the extractor's transient input
    # (absolute paths) is gone
    assert doc["spec"]["windows"][0][0] == "frames/000000.jpg" and str(sess) not in json.dumps(doc)
    assert doc["stamp"]["sha256"] and set(doc["stamp"]["inputs"]) == {f"frames/{f}" for f in doc["frames"]}
    assert doc["geometry_epoch"] == 0 and doc["camera_epoch"] == 0
    assert launched[0]["windows"][0][0] == str(sess / "frames" / "000000.jpg")
    assert not (sess / "intake" / F.DIRNAME / "windows.json").exists()
    # the same call again: reused, DA3 not launched
    assert F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)["fx"] == doc["fx"]
    assert len(launched) == 1
    # a card that holds fewer than the layout: FAIL, never split smaller (another layout is another K)
    small = dict(V.window_sizing(str(pc.focal_probe_model), 840, (464, 832), 16, 0.15, card_key=A100),
                 window_frames=14, limited_by="card")
    monkeypatch.setattr(V, "window_size", lambda *a, **k: small)
    (sess / "intake" / F.FOCAL_NAME).unlink()
    with pytest.raises(F.FocalError, match="never split smaller"):
        F.run_focal_probe(sess, q, pc, python="python", log=lambda *a: None)
    assert len(launched) == 1, "DA3 was not launched for a layout the card cannot hold"
