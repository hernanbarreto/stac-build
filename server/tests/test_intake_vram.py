"""The I3 DA3 window size is a FUNCTION OF COMMITTED NUMBERS (intake/vram.py + server/card_table.json,
docs/plan_determinismo.md points 5, 6, 13, 25, 41): the card read through torch, its table entry
(total memory, DA3 weights + 2-frame peak measured once by the calibration CLI), the same formula
over constants. No run-time calibration, no per-session cache, no 'unknown' card, no halving."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import card_table                                               # noqa: E402
from intake import vram as V                                    # noqa: E402

A100 = "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"
MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def test_tokens_follow_the_native_grid():
    assert V.da3_grid(1920, 1080, 1932) == (1932, 1092)                 # zaragoza's grid, as DA3 printed it
    assert V.tokens_per_frame(1920, 1080, 1932) == 138 * 78             # 10,764 tokens
    assert V.tokens_per_frame(464, 832, 840) == 1980                    # pccr: the table's tokens_per_frame


def test_the_committed_table_reproduces_the_validated_windows():
    """pccr 2026-10-06: 'window 32 → 26 frame(s) … 26.06 would fit' — the number in the logs, from
    the committed weights 6.36 GiB / peak 11.09 GiB; zaragoza 1080p at 1932: 6 frames."""
    s = V.window_sizing(MODEL, 840, (464, 832), requested=32, margin_frac=0.15, card_key=A100)
    assert s["window_frames"] == 26 and s["limited_by"] == "card" and s["source"] == "card_table"
    assert abs(s["frames_exact"] - 26.063) < 1e-3 and abs(s["headroom_frames"] - 0.063) < 1e-3
    assert s["card_total_gib"] == 80.0 and s["tokens_per_frame"] == 1980
    # the logs printed per_token_gb × 1e3 as '1.194 MiB/token' (GiB × 1000, their label): that number
    assert abs(s["per_token_gib"] * 1e3 - 1.194) < 1e-3
    assert s["predicted_peak_gib"] <= 80.0 * 0.85 and "pccr" in s["footprint_provenance"]
    z = V.window_sizing(MODEL, 1932, (1920, 1080), requested=32, margin_frac=0.15, card_key=A100)
    assert z["window_frames"] == 6 and abs(z["per_token_gib"] * 1e3 - 0.903) < 1e-3
    # the 16-frame focal probe of pccr fits in one window; zaragoza's needs three
    assert V.window_sizing(MODEL, 840, (464, 832), 16, 0.15, card_key=A100)["window_frames"] == 16
    assert V.window_sizing(MODEL, 1932, (1920, 1080), 16, 0.15, card_key=A100)["window_frames"] == 6
    # deterministic: the same call, the same record
    assert s == V.window_sizing(MODEL, 840, (464, 832), requested=32, margin_frac=0.15, card_key=A100)


def test_an_unknown_card_or_resolution_fails_naming_the_calibration_cli():
    with pytest.raises(card_table.CardTableError, match="intake.vram --calibrate") as e:
        V.window_sizing(MODEL, 840, (464, 832), 32, 0.15, card_key="NVIDIA RTX A6000 | 49140 MiB | sm_8.6")
    assert "NVIDIA RTX A6000" in str(e.value) and "card_table" in str(e.value)
    with pytest.raises(card_table.CardTableError, match="process_res 1008") as e:
        V.window_sizing(MODEL, 1008, (464, 832), 32, 0.15, card_key=A100)
    assert "--calibrate" in str(e.value)
    with pytest.raises(card_table.CardTableError, match="no DA3 footprint"):
        V.window_sizing("another/model", 840, (464, 832), 32, 0.15, card_key=A100)


def test_window_size_logs_the_decision_and_never_reads_the_card_at_run_time(tmp_path):
    """USER 2026-10-05 ('debe ser determinista'): nothing of the moment enters — no nvidia-smi,
    no free memory, no calibration subprocess. The module has no such reader any more."""
    for gone in ("footprint", "card_name", "free_vram_gb", "total_vram_gb", "CACHE_NAME"):
        assert not hasattr(V, gone), gone
    logs = []
    s = V.window_size(MODEL, 840, (464, 832), requested=32, margin_frac=0.15, card_key=A100,
                      log=logs.append)
    assert s["window_frames"] == 26
    assert any("window 32 → 26" in m and "card table" in m and "headroom" in m for m in logs), logs
    # a table copy with another peak gives another size — the table decides, nothing else
    doc = card_table.load_table()
    doc["cards"][A100]["da3"][MODEL]["840"]["peak_gib"] = 11.11       # the audit's knife edge
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    assert V.window_sizing(MODEL, 840, (464, 832), 32, 0.15, card_key=A100,
                           table_path=p)["window_frames"] == 25


def test_the_exit_codes_are_the_extractors():
    src = (Path(__file__).resolve().parents[1] / "extract_da3_depth.py").read_text()
    for name, val in (("OOM_EXIT", V.OOM_EXIT), ("REF_VIEW_EXIT", V.REF_VIEW_EXIT),
                      ("IDENTITY_EXIT", V.IDENTITY_EXIT)):
        assert f"{name} = {val}" in src, name


def test_production_config_declares_the_bounds():
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    p = raw["intake"]["parallax"]
    assert p["vram_calibration_frames"] >= 2 and 0 <= p["vram_margin_frac"] < 1
    from intake.config import load_intake_config
    c = load_intake_config(raw)
    assert c.parallax.vram_calibration_frames == p["vram_calibration_frames"]
    # the margin every validated window was sized with (H_CALIBRATION_LAYOUT records it)
    from reconstruction.chunk_covis import H_CALIBRATION_LAYOUT
    assert c.parallax.vram_margin_frac == H_CALIBRATION_LAYOUT["vram_margin_frac"]
    assert H_CALIBRATION_LAYOUT["card_key"] == A100


def test_the_calibration_cli_is_the_only_writer_and_needs_the_gpu_free():
    src = (Path(__file__).resolve().parents[1] / "intake" / "vram.py").read_text()
    body = src[src.index("def calibrate("):src.index("def main(")]
    assert "repro.require_exclusive_gpu" in body and "card_table.write_da3_entry" in body
    assert "write_da3_entry" not in src[:src.index("def calibrate(")]
    assert "da3_weights.hf_env" in body
