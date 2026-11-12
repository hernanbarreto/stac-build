"""CRC-32 (zlib's: reflected 0xEDB88320, initial and final XOR 0xFFFFFFFF) of many byte strings ON THE DEVICE
(USER 2026-11-11/12: "Nada debe hacerse en cpu"): the zip members of the SAM3 mask store need it — zlib.crc32
was 60 s of CPU for pccr 2026-08-24's 290,418 masks.

The CRC register is LINEAR over GF(2): the CRC of a message from register 0 is unchanged by leading zero bytes,
and appending ``n`` bytes multiplies the register by a fixed 32x32 matrix M_n. So each message is padded in
FRONT with zeros to ``BLOCK · 2^k`` bytes, every BLOCK-byte block runs the byte table in parallel (BLOCK steps
for all blocks of all messages), and the blocks are folded pairwise in a tree (left · M_len XOR right), the
lengths doubling; the standard CRC is then reg0(msg) XOR M_L · 0xFFFFFFFF XOR 0xFFFFFFFF. Integer arithmetic:
the same 32 bits as zlib.crc32 (tests/test_crc32_gpu.py).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import torch

_POLY = 0xEDB88320
_MASK = 0xFFFFFFFF
BLOCK = 64


@lru_cache(maxsize=1)
def _table() -> List[int]:
    t = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ _POLY if c & 1 else c >> 1
        t.append(c)
    return t


def _advance_zero_bytes(c: int, n: int) -> int:
    T = _table()
    for _ in range(n):
        c = T[c & 0xFF] ^ (c >> 8)
    return c


def _apply_cols(cols: Sequence[int], v: int) -> int:
    out = 0
    for i in range(32):
        if (v >> i) & 1:
            out ^= cols[i]
    return out


@lru_cache(maxsize=64)
def _shift_cols(nbytes: int) -> tuple:
    """Columns of M_nbytes (the register advanced through ``nbytes`` zero bytes), by squaring."""
    if nbytes <= BLOCK:
        return tuple(_advance_zero_bytes(1 << i, nbytes) for i in range(32))
    if nbytes % 2:
        half = _shift_cols(nbytes - 1)
        one = _shift_cols(1)
        return tuple(_apply_cols(one, half[i]) for i in range(32))
    half = _shift_cols(nbytes // 2)
    return tuple(_apply_cols(half, half[i]) for i in range(32))


def _apply_t(cols: Sequence[int], v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(v)
    for i in range(32):
        out ^= ((v >> i) & 1) * int(cols[i])
    return out


def _reg0_same_length(rows: torch.Tensor) -> List[int]:
    """The register from 0 over each row of ``rows`` (m, L) uint8, L > 0 — every row its L bytes."""
    dev = rows.device
    m, L = int(rows.shape[0]), int(rows.shape[1])
    k = 0
    while BLOCK * (1 << k) < L:
        k += 1
    lp = BLOCK * (1 << k)
    # right-aligned in a zero buffer of lp bytes (leading zeros leave the register from 0 unchanged) — a
    # slice copy, no indexed write (strict deterministic torch sorts an index_put's every element)
    buf = torch.zeros((m, lp), dtype=torch.uint8, device=dev)
    buf[:, lp - L:] = rows
    T = torch.as_tensor(_table(), dtype=torch.int64, device=dev)
    cols_major = buf.view(m * (lp // BLOCK), BLOCK).t().contiguous()           # (BLOCK, blocks): rows contiguous
    reg = torch.zeros(int(cols_major.shape[1]), dtype=torch.int64, device=dev)
    for j in range(BLOCK):
        reg = T[(reg ^ cols_major[j].to(torch.int64)) & 0xFF] ^ (reg >> 8)
    reg = reg.view(m, lp // BLOCK)
    length = BLOCK
    while int(reg.shape[1]) > 1:
        reg = _apply_t(_shift_cols(length), reg[:, 0::2]) ^ reg[:, 1::2]
        length *= 2
    return reg[:, 0].tolist()


def crc32_rows(data: torch.Tensor, lengths: Sequence[int]) -> List[int]:
    """zlib.crc32 of each row of ``data`` (n, Lmax) uint8 on its device, row ``r`` being its first
    ``lengths[r]`` bytes (what lies beyond them is not read). Rows of one length run as one block — the
    store's masks all share theirs."""
    n = int(data.shape[0])
    lengths = [int(L) for L in lengths]
    if len(lengths) != n:
        raise ValueError(f"{n} rows and {len(lengths)} lengths")
    if any(L < 0 or L > int(data.shape[1]) for L in lengths):
        raise ValueError(f"a length outside the rows' {int(data.shape[1])} bytes")
    out = [0] * n                                                              # zlib.crc32(b"") is 0
    for L in sorted(set(lengths)):
        if L == 0:
            continue
        idx = [r for r, x in enumerate(lengths) if x == L]
        sub = data[:, :L]
        if len(idx) != n:
            sub = sub.index_select(0, torch.as_tensor(idx, dtype=torch.int64, device=data.device))
        final = _apply_cols(_shift_cols(L), _MASK) ^ _MASK
        for r, reg0 in zip(idx, _reg0_same_length(sub)):
            out[r] = (int(reg0) ^ final) & _MASK
    return out
