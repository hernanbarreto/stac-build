"""docs/plan_determinismo.md point 57: the octree of an epoch is built on ONE PotreeConverter thread,
and two conversions of the same PLY are byte-identical. MEASURED 2026-10-07 (2.2 M points, exact
coordinate ties): the vendor's default thread count gave different octree.bin / hierarchy.bin on
every run; one thread, the same bytes three times. The vendored source carries the patches that
make it so (thread pin, sorted chunk files and hierarchy batches, a total order in the Poisson
samplers' sort)."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import potree_converter as PC                                       # noqa: E402

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "PotreeConverter" / "Converter"


def test_the_converter_is_always_launched_on_one_thread(tmp_path, monkeypatch):
    fake_bin = tmp_path / "PotreeConverter"
    fake_bin.write_text("")
    monkeypatch.setattr(PC, "POTREE_BIN", fake_bin)
    monkeypatch.setenv(PC.POTREE_THREADS_ENV, "7")                 # a stray value in the shell never wins
    seen = {}

    def run(cmd, **kw):
        seen.update(kw)
        od = Path(cmd[cmd.index("-o") + 1])
        od.mkdir(parents=True)
        (od / "metadata.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(PC.subprocess, "run", run)
    assert PC._run_potree_converter(tmp_path / "x.las", tmp_path / "octree")
    assert seen["env"][PC.POTREE_THREADS_ENV] == "1" and PC.POTREE_THREADS == "1"
    assert "timeout" not in seen


def _ply_with_ties(path: Path, n: int) -> None:
    from precision.epoch0_cloud import _write_ply_xyzrgb
    rng = np.random.default_rng(0)
    xyz = np.concatenate([
        np.c_[rng.uniform(0, 6, n // 2), np.zeros(n // 2), rng.uniform(0, 8, n // 2)],
        np.c_[np.zeros(n // 4), rng.uniform(0, 3, n // 4), rng.uniform(0, 8, n // 4)],
        np.c_[rng.uniform(0, 6, n // 4), rng.uniform(0, 3, n // 4), np.zeros(n // 4)]])
    xyz = np.round(xyz, 3)                                          # 1 mm: exact distance ties
    xyz = np.concatenate([xyz, xyz[: n // 10]]).astype(np.float32)   # and exact duplicates
    rgb = rng.integers(0, 255, (len(xyz), 3)).astype(np.uint8)
    _write_ply_xyzrgb(path, xyz, rgb)


def _digests(d: Path) -> dict:
    # log.txt carries wall-clock times and is not a product
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.iterdir()) if f.is_file() and f.name != "log.txt"}


@pytest.mark.skipif(not PC.POTREE_BIN.exists(), reason="PotreeConverter is not built on this machine")
def test_two_conversions_of_one_ply_are_byte_identical(tmp_path):
    pytest.importorskip("laspy")
    ply = tmp_path / "cloud.ply"
    _ply_with_ties(ply, 200_000)
    las = tmp_path / "cloud.las"
    PC._ply_to_las(ply, las)
    runs = []
    for k in range(2):
        od = tmp_path / f"octree_{k}"
        assert PC._run_potree_converter(las, od)
        runs.append(_digests(od))
    assert set(runs[0]) >= {"octree.bin", "hierarchy.bin", "metadata.json"}
    assert runs[0] == runs[1], {k: (runs[0].get(k), runs[1].get(k)) for k in runs[0] if runs[0].get(k) != runs[1].get(k)}


def test_the_vendored_source_carries_the_determinism_patches():
    unsuck = (VENDOR / "modules" / "unsuck" / "unsuck_platform_specific.cpp").read_text()
    assert "POTREE_NUM_THREADS" in unsuck and "stacConfiguredProcessors" in unsuck
    indexer = (VENDOR / "src" / "indexer.cpp").read_text()
    assert "std::sort(chunksToLoad.begin(), chunksToLoad.end()" in indexer
    assert 'std::getenv("POTREE_NUM_THREADS") != nullptr ? numSampleThreads()' in indexer
    hb = (VENDOR / "include" / "HierarchyBuilder.h").read_text()
    assert "std::sort(batchFiles.begin(), batchFiles.end())" in hb
    for name in ("sampler_poisson.h", "sampler_poisson_average.h"):
        src = (VENDOR / "include" / name).read_bytes().decode()
        assert "return a.pointIndex < b.pointIndex;" in src and "a.childIndex < b.childIndex" in src, name
        assert "\r\n" in src                                        # the vendor's CRLF endings, untouched
