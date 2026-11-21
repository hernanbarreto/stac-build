"""Phase 1 of the mono-detail work (claude_stac.txt 2026-10-04): the PointDiT runner. The model is
mocked except in the GPU test, which skips itself when the weights or the card are not there."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import pointdit_runner as PR  # noqa: E402
from precision.config import load_precision_config  # noqa: E402


def _cfg(**kw):
    return replace(load_precision_config().mono_detail, **kw)


def _fake_generate(x):
    """A deterministic stand-in: z = the image's green channel centred, x/y = a ramp; one pixel far out."""
    import torch
    g = x[:, 1:2]
    z = g - g.mean()
    out = torch.cat([torch.zeros_like(z), torch.zeros_like(z), z], 1)
    out[:, :, 0, 0] = 5.0                                     # norm > norm_max → invalid
    return out


def test_same_input_same_output_bit_for_bit_and_validity_mask():
    r = PR.PointDiTRunner(_cfg(), log=lambda m: None, device="cpu", generate=_fake_generate)
    img = (np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8)
    z1, v1 = r.depth(img)
    z2, v2 = r.depth(img)
    assert z1.shape == (48, 64) and z1.dtype == np.float32
    assert np.array_equal(z1, z2) and np.array_equal(v1, v2)
    assert not v1[0, 0] and v1[1:, 1:].all()


def test_runs_at_the_requested_size_and_refuses_non_multiples_of_16():
    r = PR.PointDiTRunner(_cfg(), log=lambda m: None, device="cpu", generate=_fake_generate)
    img = np.zeros((50, 70, 3), np.uint8)
    z, _ = r.depth(img, size=(32, 64))
    assert z.shape == (32, 64)
    with pytest.raises(PR.PointDiTError):
        r.depth(img)                                          # 50 x 70 is not patch-aligned
    assert PR.working_size(832, 464) == (688, 384)            # pccr's frame at the 32x32-token budget


def test_checksum_prefix_is_verified_and_cached(tmp_path):
    p = tmp_path / "x-deadbeef.pth"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(PR.PointDiTError):
        PR.verify_checkpoint(p, log=lambda m: None)
    import hashlib
    d = hashlib.sha256(b"ok").hexdigest()
    q = tmp_path / f"y-{d[:8]}.pth"; q.write_bytes(b"ok")
    assert PR.verify_checkpoint(q, log=lambda m: None) == d
    side = json.loads((tmp_path / f"y-{d[:8]}.pth.sha256").read_text())
    assert side["sha256"] == d
    q.write_bytes(b"changed")                                 # same name, other content → the cache is stale
    with pytest.raises(PR.PointDiTError):
        PR.verify_checkpoint(q, log=lambda m: None)


def test_missing_encoder_is_a_clear_error(tmp_path):
    cfg = _cfg(repo_dir=str(tmp_path), weights_dir=str(tmp_path), dinov3_dir=str(tmp_path))
    r = PR.PointDiTRunner(cfg, log=lambda m: None, device="cpu")
    with pytest.raises(PR.PointDiTError):
        r.load()


def _weights_present():
    try:
        p = PR.resolve_paths(_cfg())
        import torch
        return p.checkpoint.is_file() and p.dinov3.is_file() and torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _weights_present(), reason="PointDiT / DINOv3 weights or a GPU are not on this machine")
def test_gpu_real_model_is_deterministic_and_sharp():
    r = PR.PointDiTRunner(_cfg(), log=print).load()
    rng = np.random.default_rng(0)
    img = np.zeros((256, 256, 3), np.uint8)
    img[:, :128] = 200; img[:, 128:] = 40                     # a vertical brightness edge
    img += rng.integers(0, 10, img.shape).astype(np.uint8)
    z1, v1 = r.depth(img)
    z2, _ = r.depth(img)
    assert np.array_equal(z1, z2) and v1.mean() > 0.9 and np.isfinite(z1).all()
