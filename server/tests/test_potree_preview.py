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
