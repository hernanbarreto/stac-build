"""The class byte is NOT the instance id, and every reader must go through the map.

`write_classification` stores an ENCODED class (`class_map.json` → `class_of`):
the instance id while every id fits in a byte, a compact 1..N index when they do
not — because instance ids are a 1-based counter over SAM3 masklets and a busy
session runs past 255.

MEASURED 2026-09-23 on pccr (136 instances, ids 44..402, encoding `compact`):
the erase transaction's own verification compared `bincount(classification)[iid]`
with the RAW id and printed 88 `VERIFY FAILED` lines over data that balanced
exactly — 0 mismatches through the map, and the independent octree check agreed
at 99.98 %. It also skipped every id above 255, so the instances most likely to
be mis-encoded were the ones never checked. A check that cries wolf is worse
than no check: it teaches the reader to scroll past the real one.
"""
import json

import numpy as np
import pytest


@pytest.fixture
def writer():
    from segmentation.republish import write_classification
    return write_classification


def _instances(ids, per=10):
    """Disjoint index blocks, one per id — the exclusivity the brush guarantees."""
    out, k = [], 0
    for iid in ids:
        out.append({"instance_id": iid, "globalIndices": list(range(k, k + per))})
        k += per
    return out, k


def _check(dirpath, instances, cls):
    cmap = json.loads((dirpath / "class_map.json").read_text())
    code = {int(k): int(v) for k, v in cmap["class_of"].items()}
    counts = np.bincount(cls, minlength=256)
    return cmap["encoding"], [
        int(i["instance_id"]) for i in instances
        if int(counts[code[int(i["instance_id"])]]) != len(i["globalIndices"])]


def test_identity_encoding_when_every_id_fits(tmp_path, writer):
    ids = [1, 7, 44, 255]
    inst, n = _instances(ids)
    cls = writer(tmp_path, inst, n)
    enc, bad = _check(tmp_path, inst, cls)
    assert enc == "identity" and not bad
    # under identity the raw id happens to work — which is why the bug hid
    counts = np.bincount(cls, minlength=256)
    assert all(int(counts[i["instance_id"]]) == len(i["globalIndices"]) for i in inst)


def test_compact_encoding_balances_through_the_map(tmp_path, writer):
    """pccr's shape: ids well past 255, far fewer than 255 instances."""
    ids = [44, 139, 191, 260, 402]
    inst, n = _instances(ids)
    cls = writer(tmp_path, inst, n)
    enc, bad = _check(tmp_path, inst, cls)
    assert enc == "compact", "ids above 255 must force the compact index"
    assert not bad, f"instances that do not balance through class_map: {bad}"


def test_the_raw_id_is_what_produced_the_false_alarm(tmp_path, writer):
    """Pin the reason the old check was wrong, so nobody 'simplifies' it back."""
    ids = [44, 139, 191, 260, 402]
    inst, n = _instances(ids)
    cls = writer(tmp_path, inst, n)
    counts = np.bincount(cls, minlength=256)
    wrong = [int(i["instance_id"]) for i in inst
             if int(i["instance_id"]) <= 255
             and int(counts[int(i["instance_id"])]) != len(i["globalIndices"])]
    assert wrong, ("under compact encoding the raw instance id must NOT agree "
                   "with the class byte — if it does, this test's premise is "
                   "stale and the erase verification can be simplified")


def test_no_instance_is_left_without_a_class_byte(tmp_path, writer):
    ids = [44, 260, 402]
    inst, n = _instances(ids)
    writer(tmp_path, inst, n)
    code = json.loads((tmp_path / "class_map.json").read_text())["class_of"]
    assert {int(k) for k in code} == set(ids), (
        "an instance with no class byte cannot be painted in the octree")


def test_more_than_255_instances_overflow_the_byte_never_the_session(tmp_path, writer):
    """docs/plan_determinismo.md point 107 (DECIDIDO 2026-10-08): no ceiling on the object
    count. The 256th object used to raise here and the SAM3 stage shipped no segmentation;
    now the byte runs out for the LARGEST ids (byte 0, listed in class_map.json
    `byte_overflow`) and every object keeps its full id in the 16-bit instance_ids.npy."""
    import json
    import numpy as np
    ids = list(range(1, 300))
    inst, n = _instances(ids, per=1)
    cls = writer(tmp_path, inst, n)
    cmap = json.loads((tmp_path / "class_map.json").read_text())
    assert cmap["encoding"] == "compact" and cmap["byte_overflow"] == list(range(256, 300))
    assert len(set(cls[cls > 0].tolist())) == 255
    full = np.load(tmp_path / "instance_ids.npy")
    assert full.dtype == np.uint16 and sorted(set(full.tolist())) == ids
