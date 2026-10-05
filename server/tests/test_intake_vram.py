"""DA3 windows sized by the card at native resolution (intake/vram.py, zaragoza 2026-10-04)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import vram as V  # noqa: E402


def test_tokens_follow_the_native_grid():
    assert V.da3_grid(1920, 1080, 1932) == (1932, 1092)                 # zaragoza's grid, as DA3 printed it
    assert V.tokens_per_frame(1920, 1080, 1932) == 138 * 78             # 10,764 tokens
    assert V.da3_grid(464, 832, 840) == (462, 840) or V.da3_grid(464, 832, 840)[1] == 840
    assert V.tokens_per_frame(464, 832, 840) < 2200                     # pccr: ~2,000 tokens


def test_footprint_parsing_and_window_sizing():
    fp = V.parse_footprint(["[DA3 windows] model loaded: 6.10 GiB allocated",
                            "[DA3 windows] window 0: 2 frame(s) at process_res 1932 → peak 25.80 GiB allocated, 40.72 GiB reserved"])
    assert fp == {"weights_gb": 6.1, "peak_gb": 25.8}
    tpf = V.tokens_per_frame(1920, 1080, 1932)
    per_token = (25.8 - 6.1) / (2 * tpf)
    doc = {"weights_gb": 6.1, "per_token_gb": per_token}
    # the measured card (80 GB, 79 free, vLLM down): 16 frames of 1080p do not fit, ~5 do
    n = V.max_frames_per_window(doc, tpf, free_gb=79.0, margin_frac=0.15, at_most=16)
    assert 4 <= n <= 7
    # pccr's frames on the same card: the 16-frame probe window fits
    assert V.max_frames_per_window(doc, V.tokens_per_frame(464, 832, 840), 79.0, 0.15, 16) == 16
    # a 48 GB card (A6000) at 1080p: 2-3 frames
    assert 2 <= V.max_frames_per_window(doc, tpf, free_gb=47.0, margin_frac=0.15, at_most=16) <= 3
    with pytest.raises(V.VramError):
        V.parse_footprint(["nothing measured"])


def test_production_config_declares_the_bounds():
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    p = raw["intake"]["parallax"]
    assert p["vram_calibration_frames"] >= 2 and 0 <= p["vram_margin_frac"] < 1
    from intake.config import load_intake_config
    c = load_intake_config(raw)
    assert c.parallax.vram_calibration_frames == p["vram_calibration_frames"]


def test_window_size_is_sized_on_the_total_never_on_the_free_memory(monkeypatch, tmp_path):
    """USER 2026-10-05 ("debe ser determinista"): pccr 2408 measured its walk over 221
    windows of 18 keyframes with an orphan holding 21 GB and 153 of 25 on the next run —
    the same frames. The size comes from the card's TOTAL memory; the free memory only
    produces a declared warning."""
    import intake.vram as V
    fp = {"weights_gb": 6.4, "per_token_gb": 1.194 / 1024.0, "key": {"tokens_per_frame": 2000}}
    monkeypatch.setattr(V, "footprint", lambda *a, **k: fp)
    monkeypatch.setattr(V, "total_vram_gb", lambda: 80.0)
    logs = []
    sizes = []
    for free in (79.0, 58.0, 12.0):
        monkeypatch.setattr(V, "free_vram_gb", lambda free=free: free)
        sizes.append(V.window_size(tmp_path, ["a.jpg"], tmp_path, "m", 840, (832, 464), "py",
                                   requested=32, calibration_frames=2, margin_frac=0.15,
                                   log=logs.append))
    assert len(set(sizes)) == 1, sizes
    assert sizes[0] == V.max_frames_per_window(fp, 2000, 80.0, 0.15, 32)
    assert any("only 12.0 GiB free" in m for m in logs), logs
