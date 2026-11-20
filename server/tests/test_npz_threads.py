"""An .npz written and read on threads against numpy's own (USER 2026-11-20: "todo lo que no cambie el resultado
debe programarse mas rapido"): the same members, dtype, shape and bits — the tracks file of F4."""
import json
import subprocess
import zipfile

import numpy as np
import pytest

from precision import npz_threads as NT


def _tracks(rng, n_obs=250_000, n_tracks=40_000):
    """The seventeen members F4 writes, in its order, with its dtypes (0-d ints, a 0-d unicode string, bools)."""
    uv = rng.random((n_obs, 2)).astype(np.float32) * 800
    return dict(
        obs_track=rng.integers(0, n_tracks, n_obs).astype(np.int64), obs_frame=rng.integers(0, 14000, n_obs).astype(np.int64),
        obs_uv=uv, obs_vis=rng.random(n_obs).astype(np.float32), obs_score=rng.random(n_obs).astype(np.float32),
        track_query_id=np.arange(n_tracks, dtype=np.int64), track_query_frame=rng.integers(0, 14000, n_tracks).astype(np.int64),
        track_query_uv=rng.random((n_tracks, 2)).astype(np.float32), res_h=np.int64(464), res_w=np.int64(832),
        obs_uv_native=(uv * 2.3).astype(np.float32), obs_uv_sigma=rng.random(n_obs).astype(np.float32),
        track_split=rng.random(n_tracks) < 0.1, window_id=rng.integers(0, 4712, n_obs).astype(np.int32),
        is_loop_pair=rng.random(n_obs) < 0.2, frame_kind=rng.integers(0, 3, n_obs).astype(np.int8),
        meta=np.asarray(json.dumps({"version": 2, "params": {"seed": 0}, "ñ": "línea"})))


def _same(a, b):
    assert list(a) == list(b), "the same members, in the same order"
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert x.tobytes() == y.tobytes(), k


def _raw(path):
    """Each member's compressed bytes, as they lie in the file."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as fp:
        for i in z.infolist():
            fp.seek(i.header_offset + 26)
            n, e = int.from_bytes(fp.read(2), "little"), int.from_bytes(fp.read(2), "little")
            fp.seek(i.header_offset + 30 + n + e)
            out[i.filename] = fp.read(i.compress_size)
    return out


def _is_numpys_npy(record, threaded):
    """Member by member the UNCOMPRESSED stream is numpy's own .npy: the same name, length and CRC-32."""
    with zipfile.ZipFile(record) as r, zipfile.ZipFile(threaded) as t:
        assert t.testzip() is None
        assert [i.filename for i in t.infolist()] == [i.filename for i in r.infolist()]
        for ir, it in zip(r.infolist(), t.infolist()):
            assert (it.CRC, it.file_size) == (ir.CRC, ir.file_size), it.filename
            assert it.date_time == (1980, 1, 1, 0, 0, 0) and it.compress_type == zipfile.ZIP_DEFLATED


@pytest.mark.parametrize("threads", [1, 3, 8])
def test_np_load_reads_the_threaded_file_as_numpys_own(tmp_path, threads):
    m = _tracks(np.random.default_rng(3))
    np.savez_compressed(tmp_path / "record.npz", **m)
    NT.save(tmp_path / "threads.npz", m, threads=threads)
    with np.load(tmp_path / "record.npz") as a, np.load(tmp_path / "threads.npz") as b:
        _same({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
        assert b.files == list(m), "laid out in the order given"
    _is_numpys_npy(tmp_path / "record.npz", tmp_path / "threads.npz")


def test_a_member_of_one_piece_is_numpys_own_deflate_stream_and_the_file_has_no_clock(tmp_path):
    m = _tracks(np.random.default_rng(5), n_obs=60_000, n_tracks=9_000)
    assert max(np.asarray(v).nbytes for v in m.values()) < NT.BLOCK_BYTES
    np.savez_compressed(tmp_path / "record.npz", **m)
    NT.save(tmp_path / "a.npz", m, threads=4)
    NT.save(tmp_path / "b.npz", m, threads=2)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes(), "the same arrays give the same file"
    _is_numpys_npy(tmp_path / "record.npz", tmp_path / "a.npz")
    assert _raw(tmp_path / "record.npz") == _raw(tmp_path / "a.npz")


@pytest.mark.parametrize("block", [1, 7, 4096, 65536, 480_000, 1_000_000])
def test_members_cut_in_many_pieces_are_one_stream_each(tmp_path, monkeypatch, block):
    """The pieces put end to end are one deflate stream: numpy, zipfile and unzip read it; a member whose data is
    a whole number of pieces, one of a single byte and an empty one among them."""
    rng = np.random.default_rng(block)
    n = 60_000 if block >= 4096 else 300                    # (a piece of 1 or 7 bytes: a short member)
    m = _tracks(rng, n_obs=n, n_tracks=max(40, n // 6))
    m["whole_pieces"] = rng.integers(0, 255, 4 * max(block, 16), dtype=np.uint8)
    m["one_byte"] = np.uint8(7)
    m["empty"] = np.zeros((0, 2), np.float32)
    monkeypatch.setattr(NT, "BLOCK_BYTES", block)
    np.savez_compressed(tmp_path / "record.npz", **m)
    NT.save(tmp_path / "a.npz", m, threads=5)
    NT.save(tmp_path / "b.npz", m, threads=1)
    assert (tmp_path / "a.npz").read_bytes() == (tmp_path / "b.npz").read_bytes(), "never a function of the threads"
    _is_numpys_npy(tmp_path / "record.npz", tmp_path / "a.npz")
    with np.load(tmp_path / "record.npz") as a, np.load(tmp_path / "a.npz") as b:
        _same({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
    _same(NT.load(tmp_path / "record.npz", threads=3), NT.load(tmp_path / "a.npz", threads=3))
    r = subprocess.run(["unzip", "-tq", str(tmp_path / "a.npz")], capture_output=True, text=True)
    if not (r.returncode == 127 or "not found" in (r.stderr or "")):
        assert r.returncode == 0, r.stdout + r.stderr


def test_arrays_numpy_writes_in_its_own_ways_are_written_as_numpy_writes_them(tmp_path):
    """Fortran order, strides, another byte order, records, dates, bytes, an empty string, complex: the raw member
    is numpy's .npy whatever the array (the ones that are not C-contiguous go through numpy's own writer)."""
    rng = np.random.default_rng(17)
    c = rng.random((37, 5, 3))
    m = {
        "fortran": np.asfortranarray(c), "strided": c[::2, 1:, ::-1], "transposed": c.T,
        "big_endian": rng.integers(0, 1 << 40, 500).astype(">i8"), "half": rng.random(77).astype(np.float16),
        "records": np.array([(1, 2.5, b"ab"), (3, -1.0, b"c")], dtype=[("i", "<i4"), ("f", "<f8"), ("s", "S3")]),
        "dates": np.array(["2026-11-20", "1999-01-01"], dtype="datetime64[D]"), "steps": np.arange(5).astype("timedelta64[s]"),
        "bytes_": np.array([b"abc", b"", b"zz"]), "text": np.array(["uno", "dós", ""]), "no_text": np.asarray(""),
        "complex": (rng.random(9) + 1j * rng.random(9)).astype(np.complex64), "scalar": np.float64(1.5),
        "flag": np.bool_(True), "empty": np.zeros((0, 2), np.float32), "empty_f": np.zeros((3, 0), np.int16, order="F"),
        "unsigned": rng.integers(0, 65535, 1000).astype(np.uint16), "ints": np.arange(3),
    }
    np.savez_compressed(tmp_path / "record.npz", **m)
    NT.save(tmp_path / "threads.npz", m, threads=4)
    _is_numpys_npy(tmp_path / "record.npz", tmp_path / "threads.npz")
    with np.load(tmp_path / "record.npz") as a, np.load(tmp_path / "threads.npz") as b:
        _same({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
        assert b["fortran"].flags.f_contiguous == a["fortran"].flags.f_contiguous
    _same(NT.load(tmp_path / "record.npz", threads=4), NT.load(tmp_path / "threads.npz", threads=1))


def test_objects_are_refused_a_name_twice_cannot_be_given_and_an_empty_file_is_an_npz(tmp_path):
    with pytest.raises(ValueError, match="objects"):
        NT.save(tmp_path / "o.npz", {"a": np.arange(3), "o": np.array([{"x": 1}, None], dtype=object)}, threads=2)
    NT.save(tmp_path / "empty.npz", {}, threads=2)
    with np.load(tmp_path / "empty.npz") as z:
        assert z.files == []
    assert NT.load(tmp_path / "empty.npz") == {}


def test_a_failed_piece_ends_the_write_with_its_error(tmp_path, monkeypatch):
    m = _tracks(np.random.default_rng(23), n_obs=50_000, n_tracks=5_000)
    monkeypatch.setattr(NT, "BLOCK_BYTES", 4096)
    real, seen = NT._deflate, []

    def failing(head, block, last):
        seen.append(1)
        if len(seen) == 40:
            raise OSError(122, "Disk quota exceeded")
        return real(head, block, last)
    monkeypatch.setattr(NT, "_deflate", failing)
    with pytest.raises(OSError, match="quota"):
        NT.save(tmp_path / "t.npz", m, threads=3)
    pieces = sum(max(1, -(-np.asarray(v).nbytes // 4096)) for v in m.values())
    assert len(seen) < pieces, "the pieces still waiting were dropped, not deflated for a lost file"


@pytest.mark.parametrize("threads", [1, 4])
def test_the_threaded_reader_gives_np_loads_arrays(tmp_path, threads):
    m = _tracks(np.random.default_rng(11))
    np.savez_compressed(tmp_path / "record.npz", **m)
    NT.save(tmp_path / "threads.npz", m, threads=4)
    with np.load(tmp_path / "record.npz") as a:
        rec = {k: a[k] for k in a.files}
    for path in ("record.npz", "threads.npz"):              # either file, either reader
        _same(rec, NT.load(tmp_path / path, threads=threads))
    few = ("track_query_id", "track_split", "obs_track", "obs_frame", "meta")
    got = NT.load(tmp_path / "threads.npz", members=few, threads=threads)
    _same({k: rec[k] for k in few}, got)
    with pytest.raises(KeyError, match="lacks"):
        NT.load(tmp_path / "threads.npz", members=("nope",))


def test_a_member_over_4_gib_is_a_plain_case(tmp_path):
    """ZIP64 for real: one member of 4 GiB + 3 bytes of zeros (it deflates to a few MB), read back by numpy."""
    big = np.zeros((1 << 32) + 3, np.uint8)
    big[-1], big[1 << 31] = 9, 5
    NT.save(tmp_path / "big.npz", {"small": np.arange(4), "big": big, "after": np.float32(2.5)}, threads=4)
    with zipfile.ZipFile(tmp_path / "big.npz") as z:
        info = z.getinfo("big.npy")
        assert info.file_size > (1 << 32) and z.getinfo("after.npy").header_offset > 0
    got = NT.load(tmp_path / "big.npz", threads=3)
    assert got["big"].shape == big.shape and int(got["big"][-1]) == 9 and int(got["big"][1 << 31]) == 5
    assert int(got["big"].sum()) == 14 and float(got["after"]) == 2.5 and got["small"].tolist() == [0, 1, 2, 3]
    with np.load(tmp_path / "big.npz") as z:
        assert z["after"].dtype == np.float32 and z["small"].tolist() == [0, 1, 2, 3]
