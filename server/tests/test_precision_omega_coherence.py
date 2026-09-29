"""precision.omega_coherence: the chunk size is measured — the largest window over which
Omega keeps one scale against the DA3 walk (USER 2026-09-29, decision 2B)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from precision import omega_coherence as OC

CFG = SimpleNamespace(enabled=True, lengths=(8, 16, 32, 64), heldout_confidence=0.95,
                      bootstrap=400, seed=0)


def _walk(n: int, step: float = 0.06) -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.concatenate([[0.0], np.cumsum(step + rng.normal(0, 0.004, n - 1))])


def _centres(chainage: np.ndarray, scale_curve) -> np.ndarray:
    """Omega centres along x whose step scale follows ``scale_curve(k)`` (Omega is up to
    scale: a constant curve is a coherent pass)."""
    d = np.diff(chainage)
    steps = d * np.array([scale_curve(k) for k in range(len(d))])
    x = np.concatenate([[0.0], np.cumsum(steps)])
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)


def test_coherent_pass_is_coherent_and_a_drifted_one_is_not():
    ch = _walk(64)
    rng = np.random.default_rng(2)
    same = _centres(ch, lambda k: 3.0 * (1 + rng.normal(0, 0.03)))      # one scale + noise
    r = OC.coherence_of(same, ch, confidence=0.95, n_boot=400, seed=0)
    assert r["coherent"] and abs(r["drift_pct"]) < 5
    drift = _centres(ch, lambda k: 3.0 * (1 + 0.012 * k))               # +75 % across the window
    r2 = OC.coherence_of(drift, ch, confidence=0.95, n_boot=400, seed=0)
    assert not r2["coherent"] and r2["drift_pct"] > 30
    assert r2["quarters_rel_scale"][-1] > r2["quarters_rel_scale"][0]


def test_probe_lengths_include_the_whole_set_when_it_fits():
    assert OC.probe_lengths(289, 883, (32, 64, 128, 256, 512)) == [32, 64, 128, 256, 289]
    assert OC.probe_lengths(2000, 883, (32, 64, 128, 256, 512)) == [32, 64, 128, 256, 512, 883]
    assert OC.probe_lengths(20, 883, (32, 64)) == [20]
    with pytest.raises(OC.CoherenceError):
        OC.probe_lengths(3, 883, (32,))


def test_decision_takes_the_largest_coherent_window():
    res = [{"length": 8, "coherent": True}, {"length": 16, "coherent": True},
           {"length": 32, "coherent": False}, {"length": 64, "coherent": False}]
    d = OC.decide(res, 64, 883)
    assert d["chunk_keyframes"] == 16 and d["overlap_keyframes"] == 8 and not d["single_pass"]
    d2 = OC.decide([{"length": 8, "coherent": True}, {"length": 64, "coherent": True}], 64, 883)
    assert d2["single_pass"] and d2["chunk_keyframes"] == 64 and d2["overlap_keyframes"] == 0
    d3 = OC.decide([{"length": 8, "coherent": False}, {"length": 16, "coherent": False}], 64, 883)
    assert d3["none_coherent"] and d3["chunk_keyframes"] == 8


def test_run_writes_the_report_and_sizes_the_chunks(tmp_path):
    n = 64
    frames = [10 * i for i in range(n)]
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "selected_frames.json").write_text(json.dumps(
        {"version": "2.0", "selected_files": [f"{f:06d}.jpg" for f in frames]}))
    ch = _walk(n)
    (tmp_path / "intake").mkdir()
    (tmp_path / "intake" / "walk.json").write_text(json.dumps(
        {"version": 1, "walk_length_m": float(ch[-1]),
         "chainage": [{"frame": f, "chainage_m": float(c)} for f, c in zip(frames, ch)]}))
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    idx = {f: i for i, f in enumerate(frames)}

    # Omega keeps one scale over 16 keyframes and drifts 1.2 %/keyframe beyond
    def infer(paths, resolution, mode):
        ks = [idx[int(Path(p).stem)] for p in paths]
        k0 = ks[0]
        c2w = np.tile(np.eye(4), (len(ks), 1, 1))
        x = 0.0
        for j, k in enumerate(ks):
            if j:
                rate = 1.0 if (k - k0) <= 16 else 1.0 + 0.012 * (k - k0 - 16)
                x += (ch[k] - ch[k - 1]) * 3.0 * rate
            c2w[j, 0, 3] = x
        return {"c2w": c2w}

    rep = OC.run_coherence(tmp_path, CFG, capacity=883, infer=infer, resolution=512,
                           mode="balanced", log=lambda *a: None)
    lengths = [w["length"] for w in rep["windows"]]
    assert lengths == [8, 16, 32, 64]
    assert rep["windows"][0]["coherent"] and rep["windows"][1]["coherent"]
    assert not rep["windows"][3]["coherent"]
    assert rep["chunk_keyframes"] in (16, 32) and not rep["single_pass"]
    assert rep["overlap_keyframes"] == rep["chunk_keyframes"] // 2
    saved = json.loads((tmp_path / "output" / OC.PROBE_NAME).read_text())
    assert saved["chunk_keyframes"] == rep["chunk_keyframes"] and saved["provenance"] == "tool_measured"


def test_run_refuses_without_the_walk(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "selected_frames.json").write_text(json.dumps(
        {"version": "2.0", "selected_files": [f"{f:06d}.jpg" for f in range(0, 80, 10)]}))
    with pytest.raises(OC.CoherenceError, match="walk.json"):
        OC.run_coherence(tmp_path, CFG, capacity=883, infer=lambda *a: None, resolution=512,
                         mode="balanced", log=lambda *a: None)
