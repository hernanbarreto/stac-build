"""Omega's resolution probe (claude_stac.txt §4-F3): the central window, the pair
mismatch measured on the surfaces both frames see, and a report that ranks the
resolutions without deciding the config."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import omega_probe as P                          # noqa: E402
from precision.config import load_precision_config              # noqa: E402

PCFG = load_precision_config().omega.resolution_probe


def _pred(S=8, H=48, W=64, noise=0.0, seed=0):
    """A slanted wall z = 4 + 0.25 x + 0.1 y seen by cameras stepping along x;
    ``noise``: each frame's depth off by its own factor (1 + ε)."""
    rng = np.random.default_rng(seed)
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    wp, c2w = [], []
    for s in range(S):
        c = np.array([0.15 * s, 0.0, 0.0])
        # (c + t d)_z = 4 + 0.25 (c + t d)_x + 0.1 (c + t d)_y
        t = (4 + 0.25 * c[0] + 0.1 * c[1] - c[2]) / (d[..., 2] - 0.25 * d[..., 0] - 0.1 * d[..., 1])
        t = t * (1.0 + rng.normal(0, noise)) if noise else t
        wp.append(c + t[..., None] * d)
        T = np.eye(4)
        T[:3, 3] = c
        c2w.append(T)
    return {"world_points": np.array(wp), "conf": np.ones((S, H, W)),
            "c2w": np.array(c2w), "K": np.tile(K, (S, 1, 1))}


def test_probe_window_is_the_central_one():
    files = [f"{i:06d}.jpg" for i in range(100)]
    w = P.probe_window(files, 32)
    assert len(w) == 32 and w[0] == "000034.jpg"
    assert P.probe_window(files[:5], 32) == files[:5]


def test_consistent_frames_read_no_mismatch_noisy_ones_do():
    good = P.window_disagreement(_pred(), PCFG.pair_samples)
    bad = P.window_disagreement(_pred(noise=0.05, seed=1), PCFG.pair_samples)
    assert good["n_measured"] == good["n_pairs"] == 28
    assert good["median_rel_mismatch"] < 2e-3
    assert bad["median_rel_mismatch"] > 10 * good["median_rel_mismatch"]


def test_run_reports_every_resolution_and_decides_nothing(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    files = [f"{i:06d}.jpg" for i in range(40)]
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    calls = []

    def infer(paths, res, mode):
        calls.append((len(paths), res, mode))
        return _pred(S=len(paths[:8]), noise=0.0 if res == 768 else 0.04, seed=res)

    cfg = type("C", (), {"resolutions": (512, 768), "mode": "balanced", "window_frames": 32,
                         "pair_samples": PCFG.pair_samples})()
    rep = P.run_probe(tmp_path, cfg, infer=infer, log=lambda *a: None)
    assert calls == [(32, 512, "balanced"), (32, 768, "balanced")]
    assert rep["lowest_mismatch_resolution"] == 768
    assert rep["window"][0] == 4 and len(rep["window"]) == 32
    doc = json.loads((tmp_path / "output" / P.PROBE_NAME).read_text())
    assert [r["resolution"] for r in doc["results"]] == [512, 768]
    assert doc["provenance"] == "tool_measured" and "user sets" in doc["decides"]


def test_no_keyframes_names_the_file(tmp_path):
    (tmp_path / "frames").mkdir()
    with pytest.raises(P.ProbeError, match="selected_frames.json"):
        P.run_probe(tmp_path, PCFG, infer=lambda *a: None, log=lambda *a: None)
