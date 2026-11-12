"""tools/genrecon/remesh_resolution.py — the out-of-memory retry of GenRecon's remesh runs at a resolution cumesh
accepts (pccr 2026-08-24, 2026-11-12: int(0.85 x 3712) = 3155 was refused, the run died after the reconstruction)."""
import importlib.util
import re
from pathlib import Path

import pytest

_SERVER = Path(__file__).resolve().parent.parent
_TOOLS = _SERVER / "tools" / "genrecon"
_spec = importlib.util.spec_from_file_location("remesh_resolution", _TOOLS / "remesh_resolution.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _cumesh_accepts(resolution: int) -> bool:
    """cumesh.remeshing.remesh_narrow_band_dc's own loop, verbatim (its assertion as the verdict)."""
    base_resolution = resolution
    while base_resolution > 32:
        if base_resolution % 2 != 0:
            return False                      # "Failed to find a base resolution that is a multiple of 2"
        base_resolution //= 2
    return True


def test_accepted_is_cumesh_rule():
    for r in range(1, 9000):
        assert R.accepted(r) == _cumesh_accepts(r), r
    assert not R.accepted(0) and not R.accepted(-4)


def test_the_failed_retry_of_pccr_2408():
    assert not R.accepted(int(3712 * 0.85))            # 3155: what the retry asked cumesh for
    assert R.coarser(3712) == 3072                     # 24 x 128
    assert R.accepted(3072)


def test_coarser_is_the_largest_accepted_not_above_the_factor():
    for res in range(40, 9000):
        if not R.accepted(res):
            continue
        target = int(res * R.RETRY_FACTOR)
        c = R.coarser(res)
        assert R.accepted(c) and c <= target and c < res, res
        assert not any(R.accepted(x) for x in range(c + 1, target + 1)), res      # none finer was left out
        if R.accepted(target):
            assert c == target, res                    # the value the retry took before, when cumesh accepted it


def test_retries_end_at_one_then_raise():
    res, seen = 3712, []
    while res > 1:
        res = R.coarser(res)
        seen.append(res)
    assert seen[0] == 3072 and seen[-1] == 1 and all(a > b for a, b in zip(seen, seen[1:]))
    with pytest.raises(ValueError, match="cannot be retried coarser"):
        R.coarser(1)


def test_largest_accepted():
    assert R.largest_accepted(3155) == 3072
    assert R.largest_accepted(3200) == 3200
    assert R.largest_accepted(33) == 32 and R.largest_accepted(0) == 0


def test_the_vendor_retry_uses_it():
    """The STAC patch (what setup_env.sh applies) carries the retry through this module, and the tree it was cut
    from does too when the vendor clone is here."""
    patch = (_TOOLS / "genrecon_stac.patch").read_text()
    assert "_remesh_resolution.coarser(remesh_res)" in patch
    assert not re.search(r"^\+.*int\(remesh_res \* 0\.85\)", patch, re.M)
    vendor = _SERVER.parent / "vendor" / "GenRecon" / "inference" / "chunked_glb.py"
    if vendor.is_file():
        src = vendor.read_text()
        assert "_remesh_resolution.coarser(remesh_res)" in src and "int(remesh_res * 0.85)" not in src
