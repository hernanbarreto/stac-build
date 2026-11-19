"""The exact point-to-triangle distance on torch tensors — float64, ONE declared device.

The kernel of the comparison's device path (USER 2026-11-01: "todo debe correr en GPU, en modo
determinista"; 2026-11-11: "todo debe ser GPU nada en cpu"): ``point_to_tri`` is Ericson's closest
point on a triangle, region by region, with the SAME operations in the same order as the record's
``bim.comparison._point_to_tri_batch`` — sums of three products left to right, one division, one
correctly rounded square root (``_sqrt``: torch's own is not, on the CPU), no fused multiply-add
across calls: a distance is the record's, bit for bit (tests/test_bim_scan_comparison.py — 1e-9 is
the rule; equality is what is held).

WHICH triangles a point is measured against is bim/c2m_grid.py's (the search, on the device too
since 2026-11-19; until then the candidates came from scipy's k-d tree on the CPU: 98 s of pccr
2026-08-31's comparison against 28 s).

Selected by ``bim.deviation.device``: numpy (the record) | cpu (the device path on the CPU — the
tests) | cuda. The server never calls this on the card itself: the comparison is a child process
(bim/compare_job.py) that ends.
"""
from __future__ import annotations

import numpy as np

DEVICES = ("numpy", "cpu", "cuda")


def _torch():
    import torch
    return torch


def resolve_device(name: str) -> str:
    """'numpy' | 'cpu' | 'cuda', validated; cuda with no visible card FAILS naming the key."""
    name = str(name)
    if name not in DEVICES:
        raise ValueError(f"bim.deviation.device must be one of {list(DEVICES)}, got {name!r}")
    if name == "cuda" and not _torch().cuda.is_available():
        raise RuntimeError("bim.deviation.device is 'cuda' and torch sees no card — the comparison does not "
                           "fall back to the CPU by itself; declare cpu or numpy")
    return name


def tensor(a, device: str, dtype=None):
    torch = _torch()
    return torch.as_tensor(np.ascontiguousarray(a), device=device).to(torch.float64 if dtype is None else dtype)


def _dot(u, v):
    return u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]


def point_to_tri(P, A, B, C):
    """(N,) distances from N points to N triangles (A, B, C) — ``_point_to_tri_batch``, the same
    regions decided by the same comparisons and, where two regions meet, the same one winning (the
    record writes vertex A, B, C, then edge AB, AC, BC, then the inside: the last one written)."""
    torch = _torch()
    tiny = 1e-30
    AB, AC, AP = B - A, C - A, P - A
    d1, d2 = _dot(AB, AP), _dot(AC, AP)
    BP = P - B
    d3, d4 = _dot(AB, BP), _dot(AC, BP)
    CP = P - C
    d5, d6 = _dot(AB, CP), _dot(AC, CP)
    mask_a = (d1 <= 0) & (d2 <= 0)
    mask_b = (d3 >= 0) & (d4 <= d3)
    mask_c = (d6 >= 0) & (d5 <= d6)
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / torch.clamp(d1 - d3, min=tiny)
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / torch.clamp(d2 - d6, min=tiny)
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / torch.clamp((d4 - d3) + (d5 - d6), min=tiny)
    denom = 1.0 / torch.clamp(va + vb + vc, min=tiny)
    v_in, w_in = vb * denom, vc * denom
    closest = A + v_in[:, None] * AB + w_in[:, None] * AC          # inside the triangle, unless …
    closest = torch.where(mask_a[:, None], A, closest)
    closest = torch.where(mask_b[:, None], B, closest)
    closest = torch.where(mask_c[:, None], C, closest)
    closest = torch.where(mask_ab[:, None], A + v_ab[:, None] * AB, closest)
    closest = torch.where(mask_ac[:, None], A + w_ac[:, None] * AC, closest)
    closest = torch.where(mask_bc[:, None], B + w_bc[:, None] * (C - B), closest)
    D = P - closest
    return _sqrt(D[:, 0] * D[:, 0] + D[:, 1] * D[:, 1] + D[:, 2] * D[:, 2])


def _sqrt(x):
    """The square root, CORRECTLY ROUNDED on every device. torch's own on the CPU is not (measured
    2026-11-18, torch 2.9.1: its vectorised float64 sqrt is one unit in the last place off
    math.sqrt for 1.4 % of 200,000 values — numpy's and the card's for none): on the CPU the root
    is numpy's, over the tensor's own memory."""
    if x.device.type == "cpu":
        return _torch().from_numpy(np.sqrt(x.numpy()))
    return _torch().sqrt(x)
