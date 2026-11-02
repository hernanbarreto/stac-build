"""The SAM3 stage writes a FRESH, CANONICAL, SEALED store (docs/plan_determinismo.md
points 83, 101, 114, 125 — 2026-10-08).

* 83: `run_segmentation` builds the store in a staging directory and swaps it into
  output/ when the run finished — a previous run's store (its ids, its masks in the
  frames the new run did not write) never leaks into the new one; the products of the
  old store (the projection result, the fusion map) are removed with it.
* 101: the npz members are written in (object, frame) order, compact, with a fixed
  zip time — two runs over the same input give the same bytes.
* 114: the store declares the keyframe list it was segmented on; a session whose
  `camera_frames.txt` differs is refused by every reader (mask_space.resolve).
* 125: the per-prompt seconds go to `segmentation_timing.json`, never into the status
  the census copies.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import mask_space                                   # noqa: E402
from segmentation import pipeline as P                                # noqa: E402

KF = [0, 51, 61, 75, 91, 120]


class _FakeSAM3:
    """`process_batch` as the per-prompt loop sees it: {frame: {local id: lit pixel}}."""

    def __init__(self, script):
        self.script = script

    def process_batch(self, batch_dir, category, index_mapping, boxes_by_local=None):
        out = {}
        for fr, objs in self.script.get(category, {}).items():
            ids = sorted(objs)
            masks = np.zeros((len(ids), 4, 4), np.uint8)
            for k, oid in enumerate(ids):
                masks[k].flat[objs[oid]] = 1
            out[fr] = {"out_binary_masks": masks, "out_obj_ids": np.array(ids)}
        return out

    def release_batch_session(self):
        pass

    def set_upcoming(self, prompts):
        pass

    def drop_group_state(self):
        pass

    def unload_model(self):
        pass

    def load_model(self):
        pass


def _session(tmp_path, keyframes=KF):
    frames = tmp_path / "frames"
    frames.mkdir()
    import cv2
    for v in keyframes:
        cv2.imwrite(str(frames / f"{v:06d}.jpg"), np.zeros((8, 16, 3), np.uint8))
    (frames / "selected_frames.json").write_text(json.dumps(
        {"selected_files": [f"{v:06d}.jpg" for v in keyframes]}))
    out = tmp_path / "output"
    out.mkdir()
    (out / "camera_frames.txt").write_text("\n".join(str(v) for v in keyframes) + "\n")
    return frames, out


SCRIPT = {"chair": {0: {0: 1}, 1: {0: 1, 1: 2}, 2: {1: 2}, 5: {1: 2}},
          "box": {0: {0: 3}, 3: {0: 3}}}


def _run(tmp_path, monkeypatch, prompt="chair;box"):
    import torch
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: _FakeSAM3(SCRIPT))
    frames, out = _session(tmp_path) if not (tmp_path / "output").exists() else \
        (tmp_path / "frames", tmp_path / "output")
    status = {}
    res = P.run_segmentation(str(frames), str(out), prompt, prompt_status=status)
    return out, status, res


def _store_bytes(out):
    return (out / "seg_masks.npz").read_bytes(), (out / "segmentation.json").read_bytes()


def test_the_store_is_fresh_canonical_and_sealed(tmp_path, monkeypatch):
    out, status, res = _run(tmp_path, monkeypatch)
    assert not (out / P.SAM3_STAGING_DIRNAME).exists()
    z = zipfile.ZipFile(out / "seg_masks.npz")
    names = [i.filename for i in z.infolist()]
    import re
    masks = [n for n in names if re.match(r"^f\d+_o\d+\.npy$", n)]
    assert len(masks) == 7
    assert masks == sorted(masks, key=lambda n: (int(n.split("_o")[1][:-4]), int(n[1:].split("_o")[0]))), \
        "members in (object, frame) order"
    assert names[: len(masks)] == masks, "the masks first, the metadata members after them"
    assert all(i.date_time == P.STORE_ZIP_TIME for i in z.infolist()), "no clock in the zip entries"
    npz = np.load(out / "seg_masks.npz")
    assert mask_space.declared_keyframes(npz) == KF                   # point 114
    assert "reconstruction_id" in npz.files                            # point 51 / 83
    doc = json.loads((out / "segmentation.json").read_text())
    assert doc["stamp"]["inputs"]["seg_masks.npz"] and doc["n_keyframes"] == len(KF)
    assert "sam3_model" in doc, "the model that drew the masks is recorded (points 91 / 165)"
    assert doc["stamp"]["config"]["prompts"] and doc["stamp"]["config"]["sam3"]
    assert P.check_store_seal(out) == []
    assert sorted(i["label"] for i in doc["instances"]) == ["box", "chair", "chair"]
    # point 125: no seconds in the status the census copies; a timing sidecar instead
    assert all("seconds" not in st for st in status.values()), status
    timing = json.loads((out / P.SEGMENTATION_TIMING_NAME).read_text())
    assert set(timing["prompts"]) == {"chair", "box"} and timing["seconds"] is not None


def test_two_runs_write_the_same_bytes(tmp_path, monkeypatch):
    out, _, _ = _run(tmp_path, monkeypatch)
    first = _store_bytes(out)
    (out / "segmentation_result.json").write_text("{}")                 # a product of the old store
    out, _, _ = _run(tmp_path, monkeypatch)
    assert _store_bytes(out) == first
    assert not (out / "segmentation_result.json").exists(), "the old store's products go with it"


def test_a_previous_store_never_leaks_into_the_new_one(tmp_path, monkeypatch):
    """pccr 2026-10-05: raw ids 1..k of the new first category equalled the old ids and
    'kept their ID, overwrote masks' in the new frames only — chimera masklets."""
    frames, out = _session(tmp_path)
    old = {"f9_o0": np.ones((4, 4), np.uint8), "f9_o7": np.ones((4, 4), np.uint8),
           "obj_ids": np.array([0, 7], np.int32), "frames": np.array([9], np.int32),
           "scaled_res": np.array([4, 4], np.int32),
           mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME)}
    np.savez_compressed(out / "seg_masks.npz", **old)
    (out / "segmentation.json").write_text(json.dumps({
        "version": "3.0", "prompts": ["ghost"], "mask_file": "seg_masks.npz",
        "instances": [{"id": 0, "label": "ghost", "instance_id": 1, "color": [1, 1, 1]},
                      {"id": 7, "label": "ghost", "instance_id": 8, "color": [1, 1, 1]}]}))
    (out / "fusion_map.json").write_text("{}")
    out, _, _ = _run(tmp_path, monkeypatch)
    npz = np.load(out / "seg_masks.npz")
    assert not any(k.startswith("f9_") for k in npz.files), "the old run's frames are gone"
    assert npz["obj_ids"].tolist() == [0, 1, 2], "ids from the first masklet, no high-water offset"
    doc = json.loads((out / "segmentation.json").read_text())
    assert [i["label"] for i in doc["instances"]] == ["chair", "chair", "box"]
    assert doc["prompts"] == ["chair", "box"], "no prompt of the previous store"
    assert not (out / "fusion_map.json").exists()


def test_a_store_segmented_on_another_keyframe_list_is_refused(tmp_path, monkeypatch):
    out, _, _ = _run(tmp_path, monkeypatch)
    assert mask_space.resolve(out).keyframes == KF
    other = list(KF)
    other[2] = 62                                                       # one keyframe re-selected
    (out / "camera_frames.txt").write_text("\n".join(str(v) for v in other) + "\n")
    with pytest.raises(mask_space.KeyframeListMismatch, match="position 2: video frame 61 vs 62"):
        mask_space.resolve(out)
    (out / "camera_frames.txt").write_text("\n".join(str(v) for v in KF[:-1]) + "\n")
    with pytest.raises(mask_space.KeyframeListMismatch, match="6 keyframes and camera_frames.txt lists 5"):
        mask_space.resolve(out)


def test_a_run_that_fails_leaves_the_previous_store_untouched(tmp_path, monkeypatch):
    out, _, _ = _run(tmp_path, monkeypatch)
    first = _store_bytes(out)
    import segmentation.sam3_wrapper as w

    class _Boom(_FakeSAM3):
        def process_batch(self, *a, **k):
            raise RuntimeError("CUDA out of memory")                  # twice → the prompt fails

    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: _Boom(SCRIPT))
    status = {}
    with pytest.raises(RuntimeError, match="CUDA out of memory"):      # point 92: no retry, no skip
        P.run_segmentation(str(tmp_path / "frames"), str(out), "chair;box", prompt_status=status)
    assert status["chair"]["status"] == "failed" and status["box"]["status"] == "not_reached"
    assert _store_bytes(out) == first and not (out / P.SAM3_STAGING_DIRNAME).exists()


def test_the_canonical_writer_refuses_unknown_members(tmp_path):
    with pytest.raises(ValueError, match="cannot hold members"):
        P.write_canonical_store(tmp_path / "x.npz", {"f0_o0": np.ones((2, 2), np.uint8),
                                                     "stray": np.ones(2)})


def test_the_saves_run_behind_in_order_and_write_the_inline_store(tmp_path, monkeypatch):
    """The per-category saves on the store writer's thread (USER 2026-11-01: the card is never
    idle — station's 75 saves held it 116 s): in submission order, off the main thread, and the
    store they leave is byte for byte the one the saves made inline leave."""
    import threading
    calls = []
    real = P._save_masks

    def spy(output_dir, all_masks, categories, *a, **k):
        calls.append((threading.current_thread().name, list(categories)))
        return real(output_dir, all_masks, categories, *a, **k)

    monkeypatch.setattr(P, "_save_masks", spy)
    out, status, _ = _run(tmp_path, monkeypatch)
    behind = _store_bytes(out)
    assert [c for _t, c in calls] == [["chair"], ["chair", "box"]], "one save per category, in order"
    assert all(t.startswith("sam3-store") for t, _c in calls), "the saves ran on the writer's thread"

    class _Inline:
        """the saves made inline, as before the writer"""
        def __init__(self):
            self.last = None

        def submit(self, category, fn, *a, **k):
            self.last = fn(*a, **k)

        def check(self):
            pass

        def wait(self):
            return self.last

        def drain(self):
            pass

    monkeypatch.setattr(P, "_StoreWriter", _Inline)
    calls.clear()
    out, _, _ = _run(tmp_path, monkeypatch)
    assert all(t == threading.main_thread().name for t, _c in calls)
    assert _store_bytes(out) == behind


def test_a_save_that_fails_behind_fails_the_run_naming_its_category(tmp_path, monkeypatch):
    out, _, _ = _run(tmp_path, monkeypatch)
    first = _store_bytes(out)
    real = P._save_masks

    def broken(output_dir, all_masks, categories, *a, **k):
        if categories[-1] == "box":
            raise OSError("No space left on device")
        return real(output_dir, all_masks, categories, *a, **k)

    monkeypatch.setattr(P, "_save_masks", broken)
    status = {}
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: _FakeSAM3(SCRIPT))
    with pytest.raises(OSError, match="No space left"):
        P.run_segmentation(str(tmp_path / "frames"), str(out), "chair;box", prompt_status=status)
    assert status["box"]["status"] == "failed"
    assert "could not be saved" in status["box"]["reason"]
    assert _store_bytes(out) == first and not (out / P.SAM3_STAGING_DIRNAME).exists()
    assert P._ACTIVE_STORE_WRITER is None


def _write_canonical_reference(path, members, source=None):
    """write_canonical_store as it was before the in-memory build (member by member into the
    temporary file) — the byte sequence the new writer must reproduce."""
    import io as _io
    import os
    import tempfile
    path = Path(path)
    mask_keys = sorted((k for k in members if P._mask_key_parts(k) is not None), key=P._mask_key_parts)
    meta_keys = [k for k in P.STORE_META_KEYS if k in members]
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for k in mask_keys + meta_keys:
            arr = members[k] if members[k] is not None else source[k]
            buf = _io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=P.STORE_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def test_the_store_built_in_memory_is_the_same_bytes(tmp_path):
    """write_canonical_store builds the zip in memory and writes it once (station 2026-11-02: the
    member-by-member rewrite of a 31 MB store took over two minutes on the network file system),
    and the stores it rewrites are read in one read: the bytes are the ones the member-by-member
    writer wrote, members read from the file or from memory alike."""
    rng = np.random.default_rng(7)
    members = {f"f{f}_o{o}": (rng.random((24, 40)) > 0.6) for o in range(1, 40) for f in range(0, 30, 3)}
    members["scaled_res"] = np.array([24, 40])
    members[mask_space.KEYFRAMES_KEY] = np.arange(30, dtype=np.int32)
    members["reconstruction_id"] = np.array("abc")
    a, b = tmp_path / "a" / "seg_masks.npz", tmp_path / "b" / "seg_masks.npz"
    a.parent.mkdir(); b.parent.mkdir()
    _write_canonical_reference(a, dict(members))
    P.write_canonical_store(b, dict(members))
    assert a.read_bytes() == b.read_bytes()
    # rewritten from its own members (compact_mask_store reads it in memory now)
    P.compact_mask_store(b)
    assert a.read_bytes() == b.read_bytes()
    # the members read lazily from the file vs from the in-memory copy
    with np.load(a) as z:
        lazy = {k: None for k in z.files if P._mask_key_parts(k) is not None}
        lazy.update({k: np.asarray(z[k]) for k in z.files if P._mask_key_parts(k) is None})
        c = tmp_path / "c" / "seg_masks.npz"
        c.parent.mkdir()
        _write_canonical_reference(c, lazy, source=z)
    z2 = P._load_store_in_ram(a)
    d = tmp_path / "d" / "seg_masks.npz"
    d.parent.mkdir()
    P.write_canonical_store(d, {k: (None if v is None else v) for k, v in lazy.items()}, source=z2)
    assert c.read_bytes() == d.read_bytes() == a.read_bytes()
    # write_npz_canonical: the same bytes as the zip written member by member
    arrays = {"s": np.arange(5.0), "R": np.eye(3)}
    e, f = tmp_path / "e.npz", tmp_path / "f.npz"
    P.write_npz_canonical(e, arrays)
    import io as _io
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for k in sorted(arrays):
            buf = _io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=P.STORE_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())
    assert e.read_bytes() == f.read_bytes()
