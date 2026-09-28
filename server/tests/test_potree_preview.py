"""The raw-reconstruction preview octree (potree_converter.convert_chunks_preview_to_potree):
the chunks are concatenated into one LAS, and an octree marked as a preview is never
taken for the clean cloud's — neither "up to date" nor reused."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import potree_converter as PC                                    # noqa: E402

pytest.importorskip("laspy")

DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                  ("b", "u1"), ("confidence", "<f4")])


def _write_ply(path, data):
    names = {"r": "red", "g": "green", "b": "blue"}
    types = {"<f4": "float", "|u1": "uchar"}
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    head += [f"property {types[data.dtype[n].str]} {names.get(n, n)}" for n in data.dtype.names]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode())
        f.write(data.tobytes())


def _cloud(n, seed):
    rng = np.random.default_rng(seed)
    d = np.zeros(n, DTYPE)
    for k in ("x", "y", "z"):
        d[k] = rng.uniform(-5, 5, n)
    for k in ("r", "g", "b"):
        d[k] = rng.integers(0, 256, n)
    d["confidence"] = rng.uniform(1, 10, n)
    return d


def test_chunks_concatenate_into_one_las(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    a, b = _cloud(100, 0), _cloud(50, 1)
    _write_ply(out / "chunk_0.ply", a)
    _write_ply(out / "chunk_1.ply", b)
    seen = {}

    def fake_converter(las_path, potree_dir):
        import laspy
        las = laspy.read(las_path)
        seen["n"] = len(las.x)
        Path(potree_dir).mkdir(parents=True)
        (Path(potree_dir) / "metadata.json").write_text('{"points": %d}' % len(las.x))
        return True

    monkeypatch.setattr(PC, "_run_potree_converter", fake_converter)
    assert PC.convert_chunks_preview_to_potree(tmp_path)
    assert seen["n"] == 150
    assert (out / "potree" / PC.PREVIEW_MARKER).exists()


def test_no_chunks_no_preview(tmp_path):
    (tmp_path / "output").mkdir()
    assert PC.convert_chunks_preview_to_potree(tmp_path) is False


def test_a_preview_octree_is_never_up_to_date_for_the_clean_cloud(tmp_path, monkeypatch):
    out = tmp_path / "output"
    potree = out / "potree"
    potree.mkdir(parents=True)
    ply = out / "cleaned_cloud.ply"
    _write_ply(ply, _cloud(10, 2))
    (potree / "metadata.json").write_text("{}")          # newer than the PLY
    (potree / PC.PREVIEW_MARKER).write_text("preview")
    calls = []
    monkeypatch.setattr(PC, "_ply_to_las", lambda p, l: calls.append(p) or 10)
    monkeypatch.setattr(PC, "_run_potree_converter", lambda l, d: False)
    PC._convert_ply_to_potree_inner(out, ply, potree, force=False)
    assert calls == [ply], "a preview octree must not skip the clean conversion"


# ── 2026-09-28: the preview and the clean build are serialised across
# processes, never mixed, and the octree keeps 0.1 mm ──

def _fake_converter(tag, seen=None):
    def run(las_path, potree_dir):
        potree_dir = Path(potree_dir)
        potree_dir.mkdir(parents=True)
        (potree_dir / "metadata.json").write_text("{}")
        (potree_dir / f"{tag}.bin").write_text(tag)
        if seen is not None:
            seen.append(tag)
        return True
    return run


def test_preview_skipped_when_cleaned_cloud_exists(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "potree").mkdir(parents=True)
    (out / "potree" / "clean.bin").write_text("clean")
    _write_ply(out / "chunk_0.ply", _cloud(20, 3))
    _write_ply(out / "cleaned_cloud.ply", _cloud(20, 4))
    seen = []
    monkeypatch.setattr(PC, "_run_potree_converter", _fake_converter("preview", seen))
    assert PC.convert_chunks_preview_to_potree(tmp_path) is False
    assert seen == [] and sorted(p.name for p in (out / "potree").iterdir()) == ["clean.bin"]


def test_preview_discarded_when_clean_cloud_appears_during_build(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "potree").mkdir(parents=True)
    (out / "potree" / "old.bin").write_text("old")
    _write_ply(out / "chunk_0.ply", _cloud(20, 5))

    def conv(las_path, potree_dir):
        _write_ply(out / "cleaned_cloud.ply", _cloud(20, 6))   # CloudCompy finished
        return _fake_converter("preview")(las_path, potree_dir)
    monkeypatch.setattr(PC, "_run_potree_converter", conv)
    assert PC.convert_chunks_preview_to_potree(tmp_path) is False
    assert sorted(p.name for p in (out / "potree").iterdir()) == ["old.bin"]


def test_lock_is_an_os_lock(tmp_path):
    """flock, not a threading.Lock: another open file description (another
    process) cannot take it while a build holds it."""
    import fcntl
    import os
    out = tmp_path / "output"
    with PC._potree_lock(out):
        fh = os.open(str(out / PC.LOCK_NAME), os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fh)
    fh = os.open(str(out / PC.LOCK_NAME), os.O_RDWR)
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)      # released on exit
    finally:
        os.close(fh)


def test_back_to_back_builds_never_mix(tmp_path, monkeypatch):
    """Two builds inside the same second used to share potree_old_<int(time)>:
    the second rename failed silently and the new octree was copied ON TOP of
    the old one."""
    out = tmp_path / "output"
    (out / "potree").mkdir(parents=True)
    (out / "potree" / "old.bin").write_text("old")
    ply = out / "cleaned_cloud.ply"
    _write_ply(ply, _cloud(10, 7))
    monkeypatch.setattr(PC, "_ply_to_las", lambda p, l: 10)
    for tag in ("a", "b"):
        monkeypatch.setattr(PC, "_run_potree_converter", _fake_converter(tag))
        assert PC.convert_ply_to_potree(tmp_path, force=True)
    assert sorted(p.name for p in (out / "potree").iterdir()) == ["b.bin", "metadata.json"]


def test_rename_failure_fails_the_build_and_keeps_the_old_octree(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "potree").mkdir(parents=True)
    (out / "potree" / "old.bin").write_text("old")
    ply = out / "cleaned_cloud.ply"
    _write_ply(ply, _cloud(10, 8))
    monkeypatch.setattr(PC, "_ply_to_las", lambda p, l: 10)
    monkeypatch.setattr(PC, "_run_potree_converter", _fake_converter("new"))
    real = Path.rename

    def rename(self, target):
        if self == out / "potree":
            raise OSError(22, "Invalid argument")
        return real(self, target)
    monkeypatch.setattr(Path, "rename", rename)
    assert PC.convert_ply_to_potree(tmp_path, force=True) is False
    assert sorted(p.name for p in (out / "potree").iterdir()) == ["old.bin"]


def test_converter_has_no_wall_clock_timeout(tmp_path, monkeypatch):
    import subprocess
    fake_bin = tmp_path / "PotreeConverter"
    fake_bin.write_text("")
    monkeypatch.setattr(PC, "POTREE_BIN", fake_bin)
    seen = {}

    def run(cmd, **kw):
        seen.update(kw)
        Path(cmd[cmd.index("-o") + 1]).mkdir(parents=True)
        (Path(cmd[cmd.index("-o") + 1]) / "metadata.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(PC.subprocess, "run", run)
    assert PC._run_potree_converter(tmp_path / "x.las", tmp_path / "octree")
    assert "timeout" not in seen


def test_las_keeps_a_tenth_of_a_millimetre(tmp_path):
    import laspy
    d = _cloud(500, 9)
    d["x"] += np.float32(123.45678)
    PC._vertices_to_las(d, tmp_path / "c.las", tmp_path, "test")
    las = laspy.read(tmp_path / "c.las")
    assert list(las.header.scales) == [PC.LAS_SCALE_M] * 3
    for a in ("x", "y", "z"):
        err = np.abs(np.asarray(las[a], np.float64) - d[a].astype(np.float64))
        assert err.max() <= PC.LAS_SCALE_M / 2 + 1e-9, a


def test_las_extent_beyond_30_bits_fails(tmp_path):
    d = _cloud(10, 10)
    d["x"][0] = 2.0e5                                      # 200 km at 0.1 mm
    with pytest.raises(ValueError, match="30-bit"):
        PC._vertices_to_las(d, tmp_path / "c.las", tmp_path, "test")
