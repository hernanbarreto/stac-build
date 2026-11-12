"""segmentation/store_deflate_gpu.py — a mask store deflated on the card holds the record's members: every
member's bytes, size and CRC-32 are the CPU writer's, in the same order under the same entry headers; only the
Deflate streams are another encoder's."""
from __future__ import annotations

import io
import sys
import threading
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import erase as E                       # noqa: E402
from segmentation import mask_space                       # noqa: E402
from segmentation import pipeline as P                    # noqa: E402
from segmentation import store_deflate_gpu as G           # noqa: E402
from segmentation.mask_store_gpu import DeviceMaskStore   # noqa: E402

CUDA = torch.cuda.is_available()
card = pytest.mark.skipif(not CUDA, reason="no card")


def _npy(a) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.asanyarray(a), allow_pickle=False)
    return buf.getvalue()


def _arrays(rng) -> dict:
    """Every kind of array a writer may be handed: masks, and what numpy serialises in its own ways."""
    big = np.zeros((480, 640), np.uint8)
    big[100:300, 200:500] = 1
    noise = rng.integers(0, 256, (61, 47), dtype=np.uint8)
    return {
        "f3_o1": big,
        "f3_o2": big.astype(bool),
        "f4_o1": noise,
        "f5_o7": noise > 128,
        "zero_d": np.array(7, np.int64),
        "text": np.array("2026-11-12 a store"),
        "names": np.array(["a", "bcd", ""]),
        "empty": np.zeros((0,), np.float32),
        "empty_2d": np.zeros((0, 5), np.uint8),
        "strided": big[::2, ::3],
        "fortran": np.asfortranarray(noise),
        "big_endian": np.arange(12, dtype=">i8").reshape(3, 4),
        "obj_ids": np.arange(40, dtype=np.int64),
        "f8": rng.random((5, 3)),
    }


def _mask_store(path: Path, rng, n=30, h=41, w=53) -> dict:
    """A store as the SAM3 stage leaves it before its canonical rewrite, one member STORED."""
    arrs = {}
    for k in range(n):
        m = np.zeros((h, w), np.uint8)
        r, c = int(rng.integers(0, h - 8)), int(rng.integers(0, w - 8))
        m[r:r + 8, c:c + 8] = 1
        arrs[f"f{k % 5}_o{k}"] = m if k % 2 else m.astype(bool)
    arrs["obj_ids"] = np.arange(n, dtype=np.int64)
    arrs["frames"] = np.arange(5, dtype=np.int64)
    np.savez_compressed(path, **arrs)
    arrs["f9_o999"] = (rng.random((h, w)) > 0.5).astype(np.uint8)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("f9_o999.npy", _npy(arrs["f9_o999"]))
    return arrs


def _same_members(record: Path, on_card: Path) -> None:
    with zipfile.ZipFile(record) as za, zipfile.ZipFile(on_card) as zb:
        assert zb.testzip() is None
        assert zb.namelist() == za.namelist()
        for n in za.namelist():
            assert zb.read(n) == za.read(n), n
            ia, ib = za.getinfo(n), zb.getinfo(n)
            assert (ib.file_size, ib.CRC, ib.compress_type) == (ia.file_size, ia.CRC, ia.compress_type), n


# ── the record ───────────────────────────────────────────────────────────

def test_no_device_and_cpu_are_the_record_and_another_device_fails():
    arrs = _arrays(np.random.default_rng(0))
    items = list(arrs.values())
    want = []
    for a in items:
        raw = _npy(a)
        comp = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
        want.append((len(raw), zlib.crc32(raw), comp.compress(raw) + comp.flush()))
    assert G.deflate_members(items, None) == want
    assert G.deflate_members(items, "cpu") == want
    with pytest.raises(ValueError, match="device"):
        G.deflate_members(items, "gpu")
    with pytest.raises(ValueError, match="names"):
        G.deflate_members(items, None, ["one"])
    assert not E.store_on_card(None) and not E.store_on_card("cpu") and E.store_on_card("cuda")


def test_batches_are_bounded_by_members_and_by_bytes(monkeypatch):
    assert list(G._batches([])) == []
    assert list(G._batches([10] * (G.BATCH + 3))) == [(0, G.BATCH), (G.BATCH, G.BATCH + 3)]
    monkeypatch.setattr(G, "BATCH_BYTES", 64)
    assert list(G._batches([8, 8, 8, 8, 40, 100, 8, 8])) == [(0, 4), (4, 5), (5, 6), (6, 8)]


def test_the_read_back_check_names_the_member():
    s = [torch.arange(10, dtype=torch.uint8), torch.arange(7, dtype=torch.uint8)]
    G._check_read_back([t.clone() for t in s], s, ["a", "b"])
    bad = [s[0].clone(), s[1].clone()]
    bad[1][3] ^= 1
    with pytest.raises(RuntimeError, match="b: its nvCOMP Deflate stream does not inflate back"):
        G._check_read_back(bad, s, ["a", "b"])
    with pytest.raises(RuntimeError, match="a: its nvCOMP Deflate stream inflates to 9 bytes"):
        G._check_read_back([s[0][:9], s[1]], s, ["a", "b"])


def test_the_layout_is_numpys_own():
    for k, a in _arrays(np.random.default_rng(0)).items():
        h, body = G._npy_layout(np.asarray(a), k)
        assert h + np.ascontiguousarray(body).tobytes() == _npy(a), k
    with pytest.raises(ValueError, match="x: object arrays"):
        G._npy_layout(np.array([{}], dtype=object), "x")


# ── the card ─────────────────────────────────────────────────────────────

@card
def test_on_the_card_every_member_is_the_records_bytes():
    arrs = _arrays(np.random.default_rng(0))
    got = G.deflate_members(list(arrs.values()), "cuda", list(arrs))
    for (k, a), (size, crc, stream) in zip(arrs.items(), got):
        raw = _npy(a)
        assert (size, crc) == (len(raw), zlib.crc32(raw)), k
        assert zlib.decompress(stream, -15) == raw, k


@card
def test_a_stream_is_the_same_twice_alone_among_others_and_across_batches(monkeypatch):
    items = list(_arrays(np.random.default_rng(1)).values())
    a = G.deflate_members(items, "cuda")
    assert G.deflate_members(items, "cuda") == a
    assert G.deflate_members(items[::-1], "cuda") == a[::-1]
    for i in (0, 3, len(items) - 1):
        assert G.deflate_members([items[i]], "cuda") == [a[i]]
    monkeypatch.setattr(G, "BATCH", 3)
    monkeypatch.setattr(G, "BATCH_BYTES", 4096)
    assert G.deflate_members(items, "cuda") == a


@card
def test_what_is_not_a_member_fails_named():
    with pytest.raises(ValueError, match="x: object arrays"):
        G.deflate_members([np.array([{}], dtype=object)], "cuda", ["x"])
    with pytest.raises(ValueError, match="y: a member's .npy bytes come as a 1-D byte tensor"):
        G.deflate_members([torch.zeros(4, dtype=torch.float32)], "cuda", ["y"])


@card
def test_savez_on_the_card_is_a_zip_every_reader_opens(tmp_path):
    arrs = _arrays(np.random.default_rng(2))
    p = tmp_path / "seg_masks.npz"
    E._atomic_savez(p, arrs, device="cuda")
    with zipfile.ZipFile(p) as zf:
        assert zf.testzip() is None
        assert zf.namelist() == [k + ".npy" for k in arrs]
        for k, a in arrs.items():
            assert zf.read(k + ".npy") == _npy(a), k
            i = zf.getinfo(k + ".npy")
            assert (i.compress_type, i.date_time, i.external_attr, i.flag_bits) == \
                (zipfile.ZIP_DEFLATED, P.STORE_ZIP_TIME, 0o600 << 16, 0), k
    with np.load(p) as z:
        for k, a in arrs.items():
            b = z[k]
            assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(b, a), k
    first = p.read_bytes()
    E._atomic_savez(p, arrs, device="cuda")
    assert p.read_bytes() == first                       # no clock in it: the same file twice
    assert [f.name for f in tmp_path.iterdir()] == ["seg_masks.npz"]


@card
def test_append_on_the_card_carries_the_kept_entries_as_stored(tmp_path):
    rng = np.random.default_rng(3)

    def m():
        return rng.integers(0, 2, (40, 50), dtype=np.uint8)
    old = {"f0_o1": m(), "f0_o2": m(), "f1_o3": m(), "obj_ids": np.array([1, 2, 3])}
    new = {"f0_o2": m(), "f2_o9": m().astype(bool), "obj_ids": np.array([1, 2, 9])}
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(a, **old)
    b.write_bytes(a.read_bytes())
    E._atomic_append_npz(a, new, keep=["f0_o1", "f0_o2"])                       # the record
    E._atomic_append_npz(b, new, keep=["f0_o1", "f0_o2"], device="cuda")
    _same_members(a, b)
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        assert zb.namelist() == ["f0_o1.npy", "f0_o2.npy", "f2_o9.npy", "obj_ids.npy"]
        ia, ib = za.getinfo("f0_o1.npy"), zb.getinfo("f0_o1.npy")
        assert (ib.header_offset, ib.compress_size, ib.CRC) == (ia.header_offset, ia.compress_size, ia.CRC)
    with np.load(b) as z:
        assert np.array_equal(z["f0_o1"], old["f0_o1"]) and np.array_equal(z["f0_o2"], new["f0_o2"])
        assert np.array_equal(z["f2_o9"], new["f2_o9"]) and z["f2_o9"].dtype == bool


@card
def test_the_canonical_rewrite_on_the_card_holds_the_records_members(tmp_path):
    a, b = tmp_path / "rec" / "seg_masks.npz", tmp_path / "card" / "seg_masks.npz"
    a.parent.mkdir()
    b.parent.mkdir()
    arrs = _mask_store(a, np.random.default_rng(4))
    b.write_bytes(a.read_bytes())
    masks = [k for k in arrs if P._mask_key_parts(k) is not None]
    assert P.compact_mask_store(a) == P.compact_mask_store(b, device="cuda") == len(masks)
    _same_members(a, b)
    with zipfile.ZipFile(a) as za, zipfile.ZipFile(b) as zb:
        for n in za.namelist():
            ia, ib = za.getinfo(n), zb.getinfo(n)
            assert (ib.date_time, ib.external_attr, ib.flag_bits) == (ia.date_time, ia.external_attr, ia.flag_bits)
    first = b.read_bytes()
    P.compact_mask_store(b, device="cuda")               # a card's store rewritten from itself: the same bytes
    assert b.read_bytes() == first
    P.compact_mask_store(b)                              # ... and by the record: the record's bytes
    assert b.read_bytes() == a.read_bytes()
    P.compact_mask_store(b, device="cuda")
    st = DeviceMaskStore(b, "cuda")                      # the device reader on the card's store
    assert np.array_equal(st.stack(masks).cpu().numpy(), np.stack([arrs[k] != 0 for k in masks]))


@card
def test_a_member_that_is_not_what_the_source_declares_is_refused(tmp_path):
    p = tmp_path / "seg_masks.npz"
    _mask_store(p, np.random.default_rng(5), n=6)
    z = P._load_store_in_ram(p)
    members = {k: (None if P._mask_key_parts(k) is not None else np.asarray(z[k])) for k in z.files}
    victim = next(k for k in members if members[k] is None)
    z.zip.getinfo(victim + ".npy").CRC ^= 1
    before = p.read_bytes()
    with pytest.raises(RuntimeError, match=f"{victim}: inflated on the card"):
        P.write_canonical_store(p, members, source=z, device="cuda")
    assert p.read_bytes() == before
    with pytest.raises(ValueError, match="none was given"):
        P.write_canonical_store(p, members, device="cuda")
    assert p.read_bytes() == before


@card
def test_the_saver_on_the_card_writes_the_records_store(tmp_path):
    """The SAM3 stage's own sequence — a first save, an append, the compaction — on both devices."""
    cfg = {"visualization": {"segment_colors": [[1, 2, 3], [4, 5, 6]]}}
    stores = []
    for name, device in (("rec", None), ("card", "cuda")):
        rng = np.random.default_rng(6)

        def m():
            return (rng.random((24, 32)) > 0.5).astype(np.uint8)
        out = tmp_path / name / "output"
        out.mkdir(parents=True)
        (out / "camera_frames.txt").write_text("0\n30\n60\n97\n")
        (tmp_path / name / "frames").mkdir()
        P._save_masks(out, {0: {0: m(), 1: m()}, 1: {0: m()}}, ["thing"], {0: "thing", 1: "thing"}, cfg,
                      frame_space=mask_space.SPACE_KEYFRAME, compact=False, device=device)
        P._save_masks(out, {2: {2: m()}, 3: {1: m()}}, ["thing"], {1: "thing", 2: "thing"}, cfg,
                      frame_space=mask_space.SPACE_KEYFRAME, compact=False, device=device)
        stores.append(out / "seg_masks.npz")
        P._save_masks(out, {3: {2: m()}}, ["thing"], {2: "thing"}, cfg,
                      frame_space=mask_space.SPACE_KEYFRAME, compact=True, device=device)
    _same_members(*stores)


@card
def test_the_card_writers_under_strict_deterministic_torch(tmp_path):
    a, b = tmp_path / "rec" / "seg_masks.npz", tmp_path / "card" / "seg_masks.npz"
    a.parent.mkdir()
    b.parent.mkdir()
    _mask_store(a, np.random.default_rng(7))
    b.write_bytes(a.read_bytes())
    arrs = _arrays(np.random.default_rng(7))
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        P.compact_mask_store(b, device="cuda")
        got = G.deflate_members(list(arrs.values()), "cuda", list(arrs))
    finally:
        torch.use_deterministic_algorithms(before)
    P.compact_mask_store(a)
    _same_members(a, b)
    assert got == G.deflate_members(list(arrs.values()), "cuda", list(arrs))


@card
def test_a_write_from_a_thread_while_the_card_works():
    """The SAM3 stage saves behind, on its writer thread, while the model runs."""
    items = list(_arrays(np.random.default_rng(8)).values())
    want = G.deflate_members(items, "cuda")
    got, err = [], []

    def work():
        try:
            for _ in range(5):
                got.append(G.deflate_members(items, "cuda"))
        except Exception as e:                           # noqa: BLE001 — handed to the assertion below
            err.append(e)
    t = threading.Thread(target=work)
    t.start()
    x = torch.rand(512, 512, device="cuda")
    while t.is_alive():
        x = (x @ x).clamp_(0, 1)
    t.join()
    assert not err, err
    assert len(got) == 5 and all(g == want for g in got)


@card
def test_a_slice_of_a_real_store_on_the_card(tmp_path):
    real = Path(__file__).resolve().parents[1] / "projects/pccr/scans/2026-08-31/src_default/output/seg_masks.npz"
    if not real.exists():
        pytest.skip("pccr 2026-08-31's store is not here")
    z = P._load_store_in_ram(real)                       # read only
    keys = [k for k in z.files if P._mask_key_parts(k) is not None][::97][:600]
    members = {k: None for k in keys}
    members.update({k: np.asarray(z[k]) for k in P.STORE_META_KEYS if k in z.files})
    a, b = tmp_path / "rec" / "seg_masks.npz", tmp_path / "card" / "seg_masks.npz"
    a.parent.mkdir()
    b.parent.mkdir()
    P.write_canonical_store(a, members, source=z)
    P.write_canonical_store(b, members, source=z, device="cuda")
    _same_members(a, b)
    st = DeviceMaskStore(b, "cuda")
    assert np.array_equal(st.stack(keys[:50]).cpu().numpy(), np.stack([np.asarray(z[k]) != 0 for k in keys[:50]]))
