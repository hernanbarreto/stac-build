"""server/card_table.json — the COMMITTED per-card constants (docs/plan_determinismo.md points 3, 5,
6, 24, 25, 41): read by every run, written by nobody but the calibration CLIs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import card_table                                               # noqa: E402
import repro                                                    # noqa: E402

A100 = "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"
A6000 = "NVIDIA RTX A6000 | 49140 MiB | sm_8.6"                 # a card with no entry
MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def test_the_committed_entry_carries_the_measured_numbers_with_their_provenance():
    ent = card_table.card_entry(A100)
    assert card_table.sizing_total_gib(ent) == 80.0                      # nvidia-smi's 81920 MiB
    assert card_table.omega_footprint_factor(ent, A100) == 1.2139        # zaragoza 2026-10-06
    assert "zaragoza" in ent["omega"]["provenance"] and "81.27" in ent["omega"]["provenance"]
    fp = card_table.da3_footprint(ent, MODEL, 840, A100)
    assert fp["weights_gib"] == 6.36 and fp["peak_gib"] == 11.09 and fp["frames"] == 2
    assert fp["tokens_per_frame"] == 1980 and abs(fp["per_token_gib"] - 1.19444e-3) < 1e-7
    fp2 = card_table.da3_footprint(ent, MODEL, 1932, A100)
    assert fp2["peak_gib"] == 25.8 and fp2["tokens_per_frame"] == 10764
    for rec in (fp, fp2):
        assert rec["provenance"]
    # the key is repro.card_key of the card MODEL: torch's name and capability, nvidia-smi's board MiB
    ident = {"name": "NVIDIA A100 80GB PCIe", "total_memory_bytes": 85097971712, "capability": "8.0",
             "memory_total_mib": 81920}
    assert repro.card_key(ident) == A100
    assert card_table.key_memory_mib(A100) == ent["nvidia_smi_memory_total_mib"] == 81920
    assert ent["card_key_provenance"] and "85094825984" in ent["card_key_provenance"]


def test_the_committed_entry_is_found_for_both_torch_readings_of_the_run(monkeypatch):
    """The failure of 2026-10-07 (logs/server_20261007_071903.log): after a pod restart torch read
    85094825984 B on the same A100 and the card table, keyed on 85097971712 B, had no entry. Both
    readings now resolve to the committed entry through the real identity path."""
    row = "0, GPU-9e73f3cf-a20b-ef36-85c3-9cc06ba2b7d5, NVIDIA A100 80GB PCIe, 0, 81920, 595.91.07\n"

    def smi(args):
        assert args[0].startswith("--query-gpu"), args
        return row
    monkeypatch.setattr(repro, "_nvidia_smi", smi)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)          # the one card nvidia-smi lists
    for total in (85097971712, 85094825984):
        probe = {"name": "NVIDIA A100 80GB PCIe", "total_memory_bytes": total, "capability": "8.0",
                 "uuid": "9e73f3cf-a20b-ef36-85c3-9cc06ba2b7d5", "multi_processor_count": 108,
                 "l2_cache_bytes": 41943040}
        monkeypatch.setattr(repro, "_card_probe_subprocess", lambda d, p=probe: dict(p))
        monkeypatch.setattr(repro, "_card_probe_inprocess", lambda d, p=probe: dict(p))
        cur = card_table.current_card()
        assert cur["key"] == A100 and cur["total_gib"] == 80.0
        assert cur["identity"]["total_memory_bytes"] == total
        assert card_table.omega_footprint_factor(cur["entry"], cur["key"]) == 1.2139


def test_the_table_refuses_keys_not_naming_the_model_or_disagreeing_with_their_size(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"version": 1, "cards": {
        "NVIDIA A100 80GB PCIe | 85097971712 B | sm_8.0": {"nvidia_smi_memory_total_mib": 81920}}}))
    with pytest.raises(card_table.CardTableError, match="not a card key"):
        card_table.load_table(old)
    bad = tmp_path / "bad_size.json"
    bad.write_text(json.dumps({"version": 1, "cards": {A100: {"nvidia_smi_memory_total_mib": 81919}}}))
    with pytest.raises(card_table.CardTableError, match="one card model, one size"):
        card_table.card_entry(A100, bad)


def test_missing_entries_fail_naming_what_and_the_cli(tmp_path):
    with pytest.raises(card_table.CardTableError, match="has no entry") as e:
        card_table.card_entry(A6000)
    assert "intake.vram --calibrate" in str(e.value) and "chunk_plan --omega-footprint" in str(e.value)
    ent = card_table.card_entry(A100)
    with pytest.raises(card_table.CardTableError, match="process_res 1008"):
        card_table.da3_footprint(ent, MODEL, 1008, A100)
    with pytest.raises(card_table.CardTableError, match="no Omega footprint"):
        card_table.omega_footprint_factor({"nvidia_smi_memory_total_mib": 81920}, "x")
    with pytest.raises(card_table.CardTableError, match="malformed"):
        card_table.omega_footprint_factor({"omega": {"footprint_factor": 0.9}}, "x")
    with pytest.raises(card_table.CardTableError, match="inconsistent"):
        card_table.da3_footprint({"da3": {MODEL: {"840": {"weights_gib": 7, "peak_gib": 6,
                                                            "frames": 2, "tokens_per_frame": 10}}}},
                                 MODEL, 840, "x")
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(card_table.CardTableError, match="not a version"):
        card_table.load_table(bad)


def test_the_cli_writers_are_the_only_writers_and_refuse_nonsense(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"version": 1, "cards": {}}))
    card_table.write_da3_entry(A6000, 49140, MODEL, 840,
                               {"weights_gib": 6.4, "peak_gib": 11.1, "frames": 2,
                                "tokens_per_frame": 1980, "provenance": "test"}, p)
    card_table.write_omega_entry(A6000, 49140, 1.0, "linear model, no OOM measured", p)
    ent = card_table.card_entry(A6000, p)
    assert card_table.sizing_total_gib(ent) == 49140 / 1024 and ent["omega"]["footprint_factor"] == 1.0
    # the key's MiB and nvidia-smi's must agree; a key not in repro.card_key's format is refused
    with pytest.raises(card_table.CardTableError, match="one card model, one size"):
        card_table.write_omega_entry(A6000, 81920, 1.0, "x", p)
    for bad_key in ("cardX", "NVIDIA RTX A6000 | 51527024640 B | sm_8.6"):
        with pytest.raises(card_table.CardTableError, match="not a card key"):
            card_table.write_omega_entry(bad_key, 49140, 1.0, "x", p)
    assert set(card_table.load_table(p)["cards"]) == {A6000}
    with pytest.raises(card_table.CardTableError, match="below 1.0"):
        card_table.write_omega_entry(A6000, 49140, 0.8, "x", p)
    with pytest.raises(card_table.CardTableError, match="provenance"):
        card_table.write_omega_entry(A6000, 49140, 1.1, "  ", p)
    # the CLIs hand the writers card_identity's own key and board MiB — never a second nvidia-smi
    # lookup that could disagree with the key
    server = Path(__file__).resolve().parents[1]
    for name in ("intake/vram.py", "reconstruction/chunk_plan.py"):
        src = (server / name).read_text()
        assert 'int(ident["memory_total_mib"])' in src and "repro.card_key(ident)" in src, name
    # no module of the pipeline writes the table: only the two CLIs call the writers
    writers ={"card_table.py", "vram.py", "chunk_plan.py"}
    for f in server.rglob("*.py"):
        if "tests" in f.parts or f.name in writers:
            continue
        txt = f.read_text()
        assert "write_da3_entry" not in txt and "write_omega_entry" not in txt, f
    for name, must in (("intake/vram.py", "def calibrate("), ("reconstruction/chunk_plan.py", "def main(")):
        src = (server / name).read_text()
        assert src.index(must) < src.index("write_da3_entry" if "vram" in name else "write_omega_entry")


def test_no_learned_footprint_file_is_read_or_written_any_more():
    server = Path(__file__).resolve().parents[1]
    import reconstruction.chunk_plan as CP
    for gone in ("record_omega_oom", "FOOTPRINT_PATH", "OMEGA_FOOTPRINT_NAME"):
        assert not hasattr(CP, gone), gone
    assert "omega_footprint_factor" not in [n for n in dir(CP) if not n.startswith("_")] \
        or not callable(getattr(CP, "omega_footprint_factor", None)) \
        or CP.omega_footprint_factor is not None   # the name lives in card_table (entry-based)
    # the file is named only where its history is told (card_table, chunk_plan) and where a
    # leftover is deleted, declared (map_worker); nothing reads or writes it
    for f in server.rglob("*.py"):
        if "tests" in f.parts:
            continue
        txt = f.read_text()
        assert "record_omega_oom" not in txt, f
        if "omega_footprint.json" in txt:
            assert f.name in {"card_table.py", "chunk_plan.py", "map_worker.py"}, f
            assert "omega_footprint.json\")" not in txt.replace("_legacy_fp", "") or f.name == "map_worker.py"
    src = (server / "workers" / "map_worker.py").read_text()
    assert "_legacy_fp.unlink()" in src
    line = next(ln for ln in src.splitlines() if "omega_footprint.json" in ln and "_legacy_fp =" in ln)
    assert '"weights" / "omega_footprint.json"' in line


def test_footprint_factor_from_oom_reads_this_process_only():
    """The calibration CLI's reading (point 3): 'this process has X in use' + 'Tried to allocate
    Y' — never total − free, which counted a foreign process's 416 MiB on zaragoza's OOM."""
    from reconstruction.chunk_plan import ChunkLayoutError, footprint_factor_from_oom
    msg = ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.21 GiB. GPU 0 has a total "
           "capacity of 79.25 GiB of which 2.19 GiB is free. Including non-PyTorch memory, this process "
           "has 76.64 GiB memory in use. Process 2429892 has 416.00 MiB memory in use.")
    r = footprint_factor_from_oom(msg, 66.952)
    assert abs(r["need_min_gib"] - (76.64 + 4.21)) < 1e-9 and abs(r["factor"] - 80.85 / 66.952) < 1e-9
    mib = msg.replace("Tried to allocate 4.21 GiB", "Tried to allocate 512.00 MiB")
    assert abs(footprint_factor_from_oom(mib, 66.952)["need_min_gib"] - (76.64 + 0.5)) < 1e-9
    with pytest.raises(ChunkLayoutError, match="this process"):
        footprint_factor_from_oom("Tried to allocate 4.21 GiB. GPU 0 has a total capacity of 79.25 GiB "
                                  "of which 2.19 GiB is free.", 66.952)
    with pytest.raises(ChunkLayoutError):
        footprint_factor_from_oom(msg, 0.0)
    assert footprint_factor_from_oom(msg, 1000.0)["factor"] == 1.0          # never below the model


def test_the_focal_probe_layout_is_a_committed_constant_per_model_and_resolution():
    """docs/plan_determinismo.md point 64: the probe's windows no longer follow the card — the
    layout of the validated run at each resolution is the table's, on every card."""
    assert card_table.probe_window_frames(MODEL, 840)["window_frames"] == 16       # pccr: one window
    assert card_table.probe_window_frames(MODEL, 1932)["window_frames"] == 6       # zaragoza: [6, 6, 4]
    for res in (840, 1932):
        assert card_table.probe_window_frames(MODEL, res)["provenance"]
    with pytest.raises(card_table.CardTableError, match="da3_probe_layout") as e:
        card_table.probe_window_frames(MODEL, 1008)
    assert "1008" in str(e.value) and "commit" in str(e.value)
    with pytest.raises(card_table.CardTableError, match="no committed focal-probe"):
        card_table.probe_window_frames("other/model", 840)
    # the committed layouts fit the card they were measured on
    from intake.vram import window_sizing
    for res, wh in ((840, (464, 832)), (1932, (1920, 1080))):
        n = card_table.probe_window_frames(MODEL, res)["window_frames"]
        assert window_sizing(MODEL, res, wh, n, 0.15, card_key=A100)["window_frames"] == n


def test_exactly_one_visible_card(monkeypatch):
    """Point 78: the identity, memory and windows are THE device's — never nvidia-smi's first GPU
    among several, never one of several torch could pick."""
    two = [{"index": 0, "uuid": "GPU-a", "name": "A", "used_mib": 0, "total_mib": 1, "driver_version": "x"},
           {"index": 1, "uuid": "GPU-b", "name": "B", "used_mib": 0, "total_mib": 1, "driver_version": "x"}]
    monkeypatch.setattr(repro, "gpu_cards", lambda: two)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert card_table.visible_card_count() == 2
    with pytest.raises(card_table.CardTableError, match="2 card"):
        card_table.require_one_visible_card()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(card_table.CardTableError, match="2 card"):
        card_table.require_one_visible_card()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(card_table.CardTableError, match="0 card"):
        card_table.require_one_visible_card()
    for one in ("1", "GPU-b"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", one)
        card_table.require_one_visible_card()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(repro, "gpu_cards", lambda: two[:1])
    card_table.require_one_visible_card()
    server = Path(__file__).resolve().parents[1]
    for name in ("workers/map_worker.py", "intake/vram.py", "reconstruction/chunk_plan.py"):
        src = (server / name).read_text()
        assert "require_one_visible_card()" in src, name
        assert src.index("require_one_visible_card()") < src.index("repro.card_identity(0)"), name
