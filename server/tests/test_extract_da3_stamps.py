"""extract_da3_depth.py's reproducibility contract on CPU (docs/plan_determinismo.md points 27,
42, 44): strict deterministic torch, bf16 fixed, per-window stamps (one mismatch clears all), the
reference views recorded per plan, the exit codes the launchers read."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro                                                    # noqa: E402
import extract_da3_depth as X                                   # noqa: E402

SRC = Path(X.__file__).read_text()


def _spec(tmp_path, n_frames=4, win=2):
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    paths = []
    for i in range(n_frames):
        p = frames / f"{i:06d}.jpg"
        p.write_bytes(bytes([i, 1, 2]))
        paths.append(str(p))
    windows = [paths[a:a + win] for a in range(0, n_frames, win)]
    return {"windows": windows, "process_res": 840, "model_id": "m"}


IDENT = {"weights": {"revision": "abc"}, "card": "A100", "torch": {"version": "2.5"}}


def test_window_stamp_is_of_the_identity_the_plan_and_the_frames_bytes(tmp_path):
    spec = _spec(tmp_path)
    s0 = X.window_stamp(IDENT, spec, 0)
    assert len(s0) == 64 and s0 == X.window_stamp(IDENT, spec, 0)
    assert s0 != X.window_stamp(IDENT, spec, 1)
    assert s0 != X.window_stamp(dict(IDENT, card="A6000"), spec, 0)
    assert s0 != X.window_stamp(IDENT, dict(spec, process_res=1008), 0)
    Path(spec["windows"][0][1]).write_bytes(b"\x09\x01\x02")
    assert s0 != X.window_stamp(IDENT, spec, 0)
    # keyed by the frames' NAMES and bytes, not their directory: a copied session keeps its stamps
    spec2 = json.loads(json.dumps(spec))
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for i, w in enumerate(spec["windows"]):
        for j, p in enumerate(w):
            q = moved / Path(p).name
            q.write_bytes(Path(p).read_bytes())
            spec2["windows"][i][j] = str(q)
    assert X.window_stamp(IDENT, spec2, 1) == X.window_stamp(IDENT, spec, 1)


def test_one_stale_window_file_sends_all_of_them(tmp_path):
    spec = _spec(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    stamps = [X.window_stamp(IDENT, spec, i) for i in range(2)]

    def _w(i, stamp):
        np.savez(out / f"window_{i:04d}.npz", frames=np.arange(2), depth=np.zeros((2, 2, 2), np.float32),
                 stamp=np.asarray(stamp))
    _w(0, stamps[0]); _w(1, stamps[1])
    on_disk, bad = X.stale_window_files(str(out), stamps)
    assert len(on_disk) == 2 and bad == []
    _w(1, "other")
    on_disk, bad = X.stale_window_files(str(out), stamps)
    assert bad == ["window_0001.npz: stamp differs (or none)"]
    np.savez(out / "window_0001.npz", frames=np.arange(2))                 # no stamp at all
    assert X.stale_window_files(str(out), stamps)[1] == ["window_0001.npz: stamp differs (or none)"]
    _w(1, stamps[1]); _w(7, "x")
    assert "window_0007.npz: index beyond the plan's 2 windows" in X.stale_window_files(str(out), stamps)[1]
    (out / "window_x.npz").write_bytes(b"")
    assert any("not a window of this plan" in b for b in X.stale_window_files(str(out), stamps)[1])
    # the run deletes EVERY file on disk when any is stale
    blk = SRC[SRC.index("on_disk, bad = stale_window_files("):SRC.index("todo = [i for i")]
    assert "for path in on_disk:\n            os.unlink(path)" in blk and "deleting ALL" in blk


def test_reference_views_are_recorded_per_plan_and_a_regeneration_must_reproduce_them(tmp_path):
    p = str(tmp_path / X.REFERENCE_VIEWS_NAME)
    X._save_reference_views(p, "sha-A", {0: [1], 1: [0]})
    assert X._load_reference_views(p, "sha-A") == {0: [1], 1: [0]}
    assert X._load_reference_views(p, "sha-B") == {}                       # another plan's record
    assert X._load_reference_views(str(tmp_path / "none.json"), "sha-A") == {}
    blk = SRC[SRC.index("refs = list(captured.get(\"ref\", []))"):SRC.index("ref_views[i] = refs")]
    assert "sys.exit(REF_VIEW_EXIT)" in blk and "does not reproduce it" in blk
    assert "ref_views=np.asarray(refs" in SRC and "stamp=np.asarray(stamps[i])" in SRC
    assert "_vt.select_reference_view = _select" in SRC, "DA3's own choice is kept and recorded"


def test_strict_deterministic_torch_bf16_fixed_pinned_weights_no_cpu_fallback(monkeypatch):
    import torch
    assert X.AUTOCAST_DTYPE == "bfloat16" and X.DETERMINISTIC_SEED == 0
    assert "warn_only=True" not in SRC and "repro.enable_deterministic_torch" in SRC
    assert "no CPU fallback" in SRC and "is_bf16_supported" in SRC
    assert "da3_weights.verified_snapshot(" in SRC and "write_environment_record(" in SRC
    prev = repro._torch_state(torch)
    monkeypatch.delenv(repro.CUBLAS_WORKSPACE_ENV, raising=False)
    try:
        rec = X._deterministic()
        assert rec["deterministic_algorithms"] and not rec["deterministic_warn_only"]
        assert not rec["cudnn_allow_tf32"] and not rec["matmul_allow_tf32"] and rec["seed"] == 0
        assert os.environ[repro.CUBLAS_WORKSPACE_ENV] == repro.CUBLAS_WORKSPACE_VALUE
    finally:
        repro._restore_torch_state(torch, prev)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        X._require_cuda()


def test_exit_codes_and_identity_fields():
    from intake import vram as V
    assert (X.OOM_EXIT, X.REF_VIEW_EXIT, X.IDENTITY_EXIT) == (V.OOM_EXIT, V.REF_VIEW_EXIT, V.IDENTITY_EXIT)
    blk = SRC[SRC.index("def da3_identity"):SRC.index("def _load_model")]
    for key in ("weights", "code", "da3_git", "torch", "libs", "card", "driver_version",
                "autocast_dtype", "numerics"):
        assert f'"{key}"' in blk, key
    assert "repro.card_identity(dev)" in blk and "_require_one_device()" in blk
    assert "sys.exit(IDENTITY_EXIT)" in SRC and "sys.exit(OOM_EXIT)" in SRC


def test_the_identity_is_of_the_one_visible_device(monkeypatch):
    """Point 78: exactly one card visible; the identity is read from that device."""
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="2 card"):
        X._require_one_device()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert X._require_one_device() == 0
    blk = SRC[SRC.index("def da3_identity"):SRC.index("def _load_model")]
    assert "dev = _require_one_device()" in blk and "repro.card_identity(dev)" in blk
    assert '"device_count": 1' in blk


def test_the_da3_identity_names_the_card_model_never_the_instance(monkeypatch):
    """The DA3 identity is COMPARED (every window stamp, windows.json, the walk's spec sha, the
    focal probe's stamp): two cards of one model — another uuid, torch's usable bytes 3 MiB apart
    as across the pod restart of 2026-10-07 — give the SAME identity, so their products are reused
    instead of refused. The instance is recorded in da3_environment.json, not here."""
    import torch
    monkeypatch.setattr(X, "_require_cuda", lambda: None)
    monkeypatch.setattr(X, "_require_one_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(X.da3_weights, "identity", lambda m: {"model_id": m, "revision": "r"})
    monkeypatch.setattr(repro, "git_state", lambda root: {"commit": "c"})
    monkeypatch.setattr(X.intake_stamps, "jpeg_decoder_record", lambda: {"pillow": "x"})
    ids = []
    for uuid, total in (("GPU-aaaaaaaa", 85097971712), ("GPU-bbbbbbbb", 85094825984)):
        card = {"name": "NVIDIA A100 80GB PCIe", "total_memory_bytes": total, "capability": "8.0",
                "uuid": uuid, "memory_total_mib": 81920}
        card["key"] = repro.card_key(card)
        monkeypatch.setattr(repro, "card_identity", lambda dev=0, c=card: dict(c))
        monkeypatch.setattr(repro, "gpu_cards", lambda c=card: [
            {"index": 0, "uuid": c["uuid"], "name": c["name"], "used_mib": 0, "total_mib": 81920,
             "driver_version": "595.91.07"}])
        ids.append(X.da3_identity("m"))
    assert ids[0] == ids[1]
    assert ids[0]["card"] == "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"
    assert ids[0]["driver_version"] == "595.91.07"
    text = repro.canonical_json(ids[0])
    assert "GPU-aaaaaaaa" not in text and "85097971712" not in text and "card_uuid" not in text
