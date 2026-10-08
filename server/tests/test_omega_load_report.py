"""Omega's strict load report (claude_stac.txt §4-F3): every skipped key is reported,
and a checkpoint missing a tensor of aggregator / camera_head / depth_head fails
naming it instead of running on random weights."""

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
V = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
sys.path.insert(0, str(V))

from base_models import vggtomega_adapter as A                  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = torch.nn.Linear(2, 2)
        self.camera_head = torch.nn.Linear(2, 2)
        self.depth_head = torch.nn.Linear(2, 2)


def test_report_names_missing_and_unexpected_keys():
    r = A.omega_load_report(["depth_head.weight", "extra.bias"], ["junk"])
    assert r["missing_required"] == ["depth_head.weight"] and not r["ok"]
    assert r["unexpected_keys"] == ["junk"]
    assert A.omega_load_report([], [])["ok"]


def _adapter(tmp_path, monkeypatch, drop=None):
    sd = _Tiny().state_dict()
    if drop:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop)}
    sd["unused.tensor"] = torch.zeros(1)
    ck = tmp_path / "omega.pt"
    torch.save(sd, ck)
    fake = types.ModuleType("vggt_omega.models")
    fake.VGGTOmega = _Tiny
    monkeypatch.setitem(sys.modules, "vggt_omega", types.ModuleType("vggt_omega"))
    monkeypatch.setitem(sys.modules, "vggt_omega.models", fake)
    monkeypatch.setattr(A, "_omega_pkg_on_path", lambda: None)
    # plan point 12: the adapter is built for a CUDA card with bf16 only (no CPU / fp16
    # fallback) — the card is faked to build it; the load report itself is read on the CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    ad = A.VGGTOmegaAdapter({"Model": {}, "Weights": {"VGGTOmega": str(ck)}})
    assert ad.device == "cuda" and ad.dtype == torch.bfloat16
    ad.device = "cpu"
    return ad


def test_a_complete_checkpoint_loads_and_reports(tmp_path, monkeypatch):
    ad = _adapter(tmp_path, monkeypatch)
    ad.load()
    assert ad.load_report["ok"] and ad.load_report["unexpected_keys"] == ["unused.tensor"]


def test_a_missing_head_fails_naming_it(tmp_path, monkeypatch):
    ad = _adapter(tmp_path, monkeypatch, drop="camera_head")
    with pytest.raises(RuntimeError, match="camera_head"):
        ad.load()


def test_the_fork_runs_on_a_bf16_card_only(monkeypatch):
    """plan point 12: ONE autocast dtype (bf16) on every card — no CPU and no fp16 fallback;
    the device is never chosen by what the machine has."""
    from base_models import base_model as B
    assert B.AMP_DTYPE == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA card only"):
        B.require_amp_card()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Old Card")
    with pytest.raises(RuntimeError, match="does not support bfloat16"):
        B.require_amp_card()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    assert B.require_amp_card() is torch.bfloat16
    with pytest.raises(RuntimeError, match="CUDA card only"):
        A.VGGTOmegaAdapter({"Model": {}, "Weights": {}}, device="cpu")
