"""segmentation/crc32_gpu.py — the device CRC-32 is zlib's, bit for bit."""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.crc32_gpu import BLOCK, crc32_rows  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


@pytest.mark.parametrize("device", DEVICES)
def test_it_is_zlibs_at_every_length_around_the_blocks(device):
    rng = np.random.default_rng(0)
    lengths = [0, 1, 2, 7, BLOCK - 1, BLOCK, BLOCK + 1, 2 * BLOCK - 1, 2 * BLOCK, 2 * BLOCK + 1, 3 * BLOCK,
               255, 256, 257, 1000, 4095, 4096, 4097, 65535, 65536, 65537]
    data = rng.integers(0, 256, size=(len(lengths), max(lengths)), dtype=np.uint8)
    got = crc32_rows(torch.as_tensor(data, device=device), lengths)
    assert got == [zlib.crc32(data[i, :n].tobytes()) for i, n in enumerate(lengths)]


@pytest.mark.parametrize("device", DEVICES)
def test_rows_of_one_length_and_of_any_and_nothing_beyond_a_length_is_read(device):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, size=(37, 5000), dtype=np.uint8)
    t = torch.as_tensor(data, device=device)
    assert crc32_rows(t, [5000] * 37) == [zlib.crc32(r.tobytes()) for r in data]
    lengths = [int(x) for x in rng.integers(0, 5001, size=37)]
    want = [zlib.crc32(data[i, :n].tobytes()) for i, n in enumerate(lengths)]
    assert crc32_rows(t, lengths) == want
    other = t.clone()
    for i, n in enumerate(lengths):
        other[i, n:] = 255 - other[i, n:]
    assert crc32_rows(other, lengths) == want


@pytest.mark.parametrize("device", DEVICES)
def test_known_messages(device):
    msgs = [b"123456789", b"\x00" * 1000, b"\xff" * 1000, b"a"]
    data = torch.zeros((len(msgs), 1000), dtype=torch.uint8, device=device)
    for i, m in enumerate(msgs):
        data[i, :len(m)] = torch.frombuffer(bytearray(m), dtype=torch.uint8).to(device)
    got = crc32_rows(data, [len(m) for m in msgs])
    assert got == [zlib.crc32(m) for m in msgs] and got[0] == 0xCBF43926


def test_no_row_and_bad_lengths():
    assert crc32_rows(torch.zeros((0, 8), dtype=torch.uint8), []) == []
    t = torch.zeros((2, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="2 rows and 1 lengths"):
        crc32_rows(t, [8])
    with pytest.raises(ValueError, match="outside the rows' 8 bytes"):
        crc32_rows(t, [8, 9])


@pytest.mark.skipif(not CUDA, reason="no card")
def test_mask_sized_rows_on_the_card_under_strict_deterministic_torch():
    rng = np.random.default_rng(2)
    data = rng.integers(0, 256, size=(48, 386_000), dtype=np.uint8)
    lengths = [386_000] * 46 + [128, 385_999]
    want = [zlib.crc32(data[i, :n].tobytes()) for i, n in enumerate(lengths)]
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        assert crc32_rows(torch.as_tensor(data, device="cuda"), lengths) == want
    finally:
        torch.use_deterministic_algorithms(before)
