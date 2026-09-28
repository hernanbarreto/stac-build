"""The raw-reconstruction preview octree (potree_converter.convert_chunks_preview_to_potree):
the chunks are concatenated into one LAS, and an octree marked as a preview is never
taken for the clean cloud's — neither "up to date" nor reused."""

import os
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


def test_las_quantum_is_the_finest_power_of_two_the_extent_allows(tmp_path):
    import laspy
    d = _cloud(500, 9)                                     # 10 m cube
    d["x"] += np.float32(123.45678)
    PC._vertices_to_las(d, tmp_path / "c.las", tmp_path, "test")
    las = laspy.read(tmp_path / "c.las")
    extent = max(float(d[a].max()) - float(d[a].min()) for a in ("x", "y", "z"))
    s = las.header.scales[0]
    assert list(las.header.scales) == [s] * 3
    assert np.frexp(s)[0] == 0.5                           # a power of two
    assert extent / s < 2 ** PC.LAS_EXTENT_BITS <= extent / (s / 2)   # the finest
    for a in ("x", "y", "z"):
        err = np.abs(np.asarray(las[a], np.float64) - d[a].astype(np.float64))
        assert err.max() <= s / 2, a


def test_las_quantum_follows_the_extent_and_never_meets_potrees_clamp():
    for extent in (0.0, 1e-3, 12.9, 43.7, 2.0e5):
        s = PC._las_scale(extent)
        assert np.frexp(s)[0] == 0.5
        assert extent / s < 2 ** 29                        # 2x under PotreeConverter's 2**30
        assert extent == 0.0 or extent / (s / 2) >= 2 ** 29
    assert PC._las_scale(2.0e5) > PC._las_scale(43.7)      # a larger scene, a coarser grid


def test_las_non_finite_coordinates_fail(tmp_path):
    d = _cloud(10, 10)
    d["x"][0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        PC._vertices_to_las(d, tmp_path / "c.las", tmp_path, "test")


def test_failed_rename_in_restores_the_old_octree(tmp_path, monkeypatch):
    """The old octree is moved aside, then the NEW rename fails: the old one
    goes back — the session is never left without an octree — and nothing is
    deleted in the background."""
    out = tmp_path / "output"
    (out / "potree").mkdir(parents=True)
    (out / "potree" / "old.bin").write_text("old")
    ply = out / "cleaned_cloud.ply"
    _write_ply(ply, _cloud(10, 11))
    monkeypatch.setattr(PC, "_ply_to_las", lambda p, l: 10)
    monkeypatch.setattr(PC, "_run_potree_converter", _fake_converter("new"))
    popen = []
    monkeypatch.setattr(PC.subprocess, "Popen", lambda *a, **k: popen.append(a))
    real = Path.rename

    def rename(self, target):
        if Path(target) == out / "potree" and self.name == "potree" \
                and not self.parent.name.startswith("potree_old_"):
            raise OSError(28, "No space left on device")
        return real(self, target)
    monkeypatch.setattr(Path, "rename", rename)
    assert PC.convert_ply_to_potree(tmp_path, force=True) is False
    assert sorted(p.name for p in (out / "potree").iterdir()) == ["old.bin"]
    assert not list(out.glob("potree_old_*")) and popen == []


def test_lock_wait_is_logged_with_the_holder(tmp_path, monkeypatch, caplog):
    """No deadline on the wait — but it is visible, and it names the holder."""
    import logging
    import threading
    out = tmp_path / "output"
    monkeypatch.setattr(PC, "LOCK_POLL_S", 0.01)
    monkeypatch.setattr(PC, "LOCK_LOG_EVERY_S", 0.05)
    got = threading.Event()

    def waiter():
        with PC._potree_lock(out):
            got.set()
    with caplog.at_level(logging.INFO, logger=PC.logger.name):
        with PC._potree_lock(out):
            t = threading.Thread(target=waiter)
            t.start()
            assert not got.wait(0.3)                       # still waiting
        t.join(5)
    assert got.is_set()
    waits = [r.getMessage() for r in caplog.records if "waiting" in r.getMessage()]
    assert len(waits) >= 2 and all(f"pid={os.getpid()}" in m for m in waits)


def test_epoch_swap_waits_for_the_octree_lock(tmp_path, monkeypatch):
    """correction.apply.swap_transaction moves output/potree under the same
    cross-process lock as every octree build."""
    import threading
    from correction.apply import swap_transaction
    monkeypatch.setattr(PC, "LOCK_POLL_S", 0.01)
    out = tmp_path / "output"
    (out / "potree").mkdir(parents=True)
    (out / "potree" / "old.bin").write_text("old")
    tx = out / "_tx_epoch_1"
    (tx / "potree").mkdir(parents=True)
    (tx / "potree" / "new.bin").write_text("new")
    info = {"tx_dir": str(tx), "epoch_from": 0, "epoch_to": 1,
            "artifacts": [{"rel": "potree", "existed_before": True}]}
    done = threading.Event()

    def swap():
        swap_transaction(out, info, log=lambda m: None)
        done.set()
    with PC._potree_lock(out):
        t = threading.Thread(target=swap)
        t.start()
        assert not done.wait(0.3)                          # blocked on the build's lock
        assert sorted(p.name for p in (out / "potree").iterdir()) == ["old.bin"]
    t.join(5)
    assert done.is_set()
    assert sorted(p.name for p in (out / "potree").iterdir()) == ["new.bin"]
