"""The resolution GenRecon's mesh step retries its remesh at when the card runs out of memory.

cumesh's narrow-band remesh (cumesh.remeshing.remesh_narrow_band_dc) builds its grid by doubling a base of at
most 32 cells, so it ACCEPTS a resolution only when halving it down to <= 32 never meets an odd number
("Failed to find a base resolution that is a multiple of 2"). The out-of-memory retry of
vendor/GenRecon/inference/chunked_glb.py took int(0.85 x resolution): 3712 -> 3155, odd, refused — pccr
2026-08-24's GenRecon died there on 2026-11-12, after the whole reconstruction had finished (zaragoza's retry of
2026-11-05 had landed on an accepted value by chance). The retry now runs at the LARGEST ACCEPTED resolution not
above that 85 %: the value it took before whenever that one was accepted, the next accepted one below it
otherwise (3712 -> 3072).

Pure integers, no library: the vendor file loads this module by its path (tests/test_genrecon_remesh_resolution.py).
"""
from __future__ import annotations

#: the retry's step, as it was (inference/chunked_glb.py, 2026-11-05)
RETRY_FACTOR = 0.85
#: cumesh's largest base grid (remeshing.remesh_narrow_band_dc halves the resolution while it is above this)
BASE_MAX = 32


def accepted(resolution: int) -> bool:
    """cumesh's own rule: the resolution halved while above BASE_MAX, never odd on the way."""
    r = int(resolution)
    if r < 1:
        return False
    while r > BASE_MAX:
        if r % 2:
            return False
        r //= 2
    return True


def largest_accepted(limit: int) -> int:
    """The largest accepted resolution <= ``limit`` (0 when there is none)."""
    r = int(limit)
    while r >= 1 and not accepted(r):
        r -= 1
    return max(r, 0)


def coarser(resolution: int, factor: float = RETRY_FACTOR) -> int:
    """The resolution the out-of-memory retry runs at: the largest accepted one <= int(resolution x factor).
    Raises when no coarser resolution is left."""
    res = int(resolution)
    out = largest_accepted(int(res * float(factor)))
    if out < 1 or out >= res:
        raise ValueError(f"no accepted remesh resolution below {res} (x {factor}) — the remesh cannot be retried coarser")
    return out
