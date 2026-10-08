"""docs/plan_determinismo.md points 103 / 124 (2026-10-08): the vendored PotreeConverter
lays octree.bin out in a CANONICAL pass (breadth-first node order, offsets rewritten in
the flushed hierarchy records), sorts the Poisson samplers by (distance, x, y, z, child,
index), flushes the chunk files in name order and writes NO log.txt into the octree. Two
conversions of one cloud are byte-identical over EVERY file of the octree — nothing is
excluded from the comparison."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import potree_converter as PC                                       # noqa: E402

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "PotreeConverter" / "Converter"


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
    return {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(d.rglob("*")) if f.is_file()}


def _records(d: Path):
    h = (d / "hierarchy.bin").read_bytes()
    out = []
    for i in range(len(h) // 22):
        typ = h[22 * i]
        off = int.from_bytes(h[22 * i + 6: 22 * i + 14], "little")
        bs = int.from_bytes(h[22 * i + 14: 22 * i + 22], "little")
        out.append((typ, off, bs))
    return out


@pytest.mark.skipif(not PC.POTREE_BIN.exists(), reason="PotreeConverter is not built on this machine")
def test_two_conversions_are_byte_identical_over_every_file_and_write_no_log(tmp_path):
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
    assert set(runs[0]) == {"octree.bin", "hierarchy.bin", "metadata.json"}, sorted(runs[0])
    assert "log.txt" not in runs[0]                                     # point 124
    assert runs[0] == runs[1], {k: (runs[0].get(k), runs[1].get(k))
                                for k in runs[0] if runs[0].get(k) != runs[1].get(k)}
    # the canonical layout: the non-proxy nodes' byte ranges tile octree.bin exactly, in
    # breadth-first order of the hierarchy — the first record (the root) at offset 0
    od = tmp_path / "octree_0"
    size = (od / "octree.bin").stat().st_size
    ranges = sorted({(off, bs) for typ, off, bs in _records(od) if typ != 2 and bs > 0})
    cur = 0
    for off, bs in ranges:
        assert off == cur, (off, cur)
        cur += bs
    assert cur == size
    assert _records(od)[0][1] == 0
    meta = json.loads((od / "metadata.json").read_text())
    assert meta["points"] == 220_000


def test_the_vendored_source_carries_the_canonical_patches():
    indexer = (VENDOR / "src" / "indexer.cpp").read_bytes().decode()
    assert "stac_canonical::canonicalizeOctree(targetDir, hierarchyDir);" in indexer
    assert "void canonicalizeOctree(string targetDir, string hierarchyDir)" in indexer
    main = (VENDOR / "src" / "main.cpp").read_bytes().decode()
    assert 'logger::addOutputFile(targetDir + "/log.txt")' not in main
    writer = (VENDOR / "include" / "ConcurrentWriter.h").read_bytes().decode()
    assert "cand->first < it->first" in writer
    for name in ("sampler_poisson.h", "sampler_poisson_average.h"):
        src = (VENDOR / "include" / name).read_bytes().decode()
        assert "if (a.x != b.x) {" in src and "if (a.z != b.z) {" in src, name
        assert "return a.pointIndex < b.pointIndex;" in src
        assert "\r\n" in src                                        # the vendor's CRLF endings, untouched
