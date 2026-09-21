"""USER 2026-09-20: "cuando aplicas una solucion debe ser de fondo y
profesional, no un parche y a medias".

A cloud never travels alone. Beside it live arrays indexed BY ROW —
`classification.npy` (the per-point class the Potree converter bakes into the
octree) and `out_of_place.npy` (the geometric-cleanup marks) — and a census
that has to agree with it. A correction epoch republished the cloud and left
them behind, and NOTHING said so: the converter compares the two lengths,
writes nothing and logs one line, so pccr's epoch-1 octree shipped with every
class byte at zero and `geometric_cleanup` refused to run at all.

These tests pin the invariant: after a republish, no sibling of the cloud
describes a different cloud.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.republish import (CLASSIFICATION, OUT_OF_PLACE, RESULT,
                                    BROADCAST, republish_membership)


def _result(dirpath: Path, instances, total=999):
    """A result file with a deliberately WRONG census — republishing has to
    be what fixes it, not the caller."""
    (dirpath / RESULT).write_text(json.dumps({
        "instances": instances, "total_points": total,
        "segmented_points": 123456, "coverage": 0.5}))


def _inst(iid, idx, label="wall"):
    return {"id": iid, "instance_id": iid, "label": label,
            "globalIndices": list(idx), "total_points": len(idx)}


# ── classification.npy ───────────────────────────────────────────────────

def test_the_class_byte_is_the_instance_id_and_zero_is_unsegmented(tmp_path):
    _result(tmp_path, [_inst(7, [0, 1]), _inst(9, [3])])
    rep = republish_membership(tmp_path, n_points=5, log=lambda m: None)
    cls = np.load(tmp_path / CLASSIFICATION)
    assert cls.dtype == np.uint8 and len(cls) == 5
    assert list(cls) == [7, 7, 0, 9, 0]
    assert CLASSIFICATION in rep["files"] and rep["max_instance_id"] == 9


def test_the_class_array_is_as_long_as_the_cloud_not_as_the_old_one(tmp_path):
    """The whole defect in one line: 22,770,025 values against 22,431,995
    rows. The converter writes nothing and says one line nobody reads."""
    _result(tmp_path, [_inst(1, [0, 1, 2])], total=22_770_025)
    republish_membership(tmp_path, n_points=3, log=lambda m: None)
    assert len(np.load(tmp_path / CLASSIFICATION)) == 3


def test_an_index_past_the_end_cannot_reach_the_array(tmp_path):
    _result(tmp_path, [_inst(4, [0, 99])])
    republish_membership(tmp_path, n_points=2, log=lambda m: None)
    assert list(np.load(tmp_path / CLASSIFICATION)) == [4, 0]


# ── the census ───────────────────────────────────────────────────────────

def test_the_census_is_rewritten_to_agree_with_the_cloud(tmp_path):
    _result(tmp_path, [_inst(1, [0, 1]), _inst(2, [2])], total=22_770_025)
    republish_membership(tmp_path, n_points=10, log=lambda m: None)
    doc = json.loads((tmp_path / RESULT).read_text())
    assert doc["total_points"] == 10
    assert doc["segmented_points"] == 3
    assert doc["coverage"] == pytest.approx(0.3)


# ── out_of_place.npy: a MEASUREMENT, carried, never recomputed ───────────

def test_the_cleanup_marks_are_carried_through_the_deletion(tmp_path):
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    np.save(src / OUT_OF_PLACE, np.array([True, False, True, False], bool))
    _result(dst, [_inst(1, [0, 1])])
    keep = np.array([True, False, True, True])
    rep = republish_membership(dst, n_points=3, keep=keep, source_dir=src,
                               log=lambda m: None)
    assert list(np.load(dst / OUT_OF_PLACE)) == [True, True, False]
    assert OUT_OF_PLACE in rep["files"]


def test_marks_that_are_already_stale_are_not_carried(tmp_path):
    """Carrying a wrong-length array would launder the staleness into the new
    epoch. It is left behind and said out loud instead."""
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    np.save(src / OUT_OF_PLACE, np.zeros(99, bool))
    _result(dst, [_inst(1, [0])])
    said = []
    rep = republish_membership(dst, n_points=2, keep=np.array([True, True]),
                               source_dir=src, log=said.append)
    assert not (dst / OUT_OF_PLACE).exists()
    assert OUT_OF_PLACE not in rep["files"]
    assert any("stale already" in m for m in said)


# ── the mask has to describe the cloud it is handed ─────────────────────

def test_a_keep_mask_that_does_not_match_the_cloud_is_refused(tmp_path):
    _result(tmp_path, [_inst(1, [0])])
    with pytest.raises(ValueError, match="keep mask keeps"):
        republish_membership(tmp_path, n_points=5,
                             keep=np.array([True, False, True]),
                             log=lambda m: None)


# ── seg_broadcast.json: kept in step, never invented ─────────────────────

def test_the_broadcast_is_kept_in_step_only_when_the_session_has_one(tmp_path):
    src = tmp_path / "src"; dst = tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _result(dst, [_inst(1, [0, 1])])
    rep = republish_membership(dst, n_points=2, source_dir=src,
                               log=lambda m: None)
    assert not (dst / BROADCAST).exists() and BROADCAST not in rep["files"]

    (src / BROADCAST).write_text(json.dumps({"instances": [_inst(9, [0])]}))
    rep = republish_membership(dst, n_points=2, source_dir=src,
                               log=lambda m: None)
    assert [i["instance_id"] for i in
            json.loads((dst / BROADCAST).read_text())["instances"]] == [1]


# ── only ONE writer of the class byte ────────────────────────────────────

def test_erase_and_the_epoch_write_the_same_bytes(tmp_path):
    """They used to be two implementations that disagreed on the key — `id`
    in the pipeline, `instance_id` in erase — equal only by convention."""
    from segmentation.erase import _write_classification
    from segmentation.republish import write_classification
    insts = [_inst(3, [0, 2]), _inst(5, [1])]
    a = _write_classification(tmp_path, insts, 4)
    b = write_classification(tmp_path, insts, 4)
    assert np.array_equal(a, b)


# ── the store stops holding objects that no longer exist ────────────────

def _store(tmp_path, ids):
    from phase_r.instance_store import InstanceStore
    st = InstanceStore(tmp_path / "scene_r.db")
    for i in ids:
        st.upsert_instance(i, f"obj{i}")
        st.set_points(i, np.zeros((3, 3), np.float32))
    return st


def test_reconcile_drops_the_ghosts_and_keeps_the_findings(tmp_path):
    st = _store(tmp_path, [1, 2, 3])
    st.add_finding(instance_id=2, type="gap", severity="low",
                   description="a gap", confidence=0.9,
                   point3d=np.zeros(3, np.float32))
    rep = st.reconcile([1, 3])
    assert rep["removed"] == 1 and rep["removed_ids"] == [2]
    assert [r["instance_id"] for r in st.list_instances()] == [1, 3]
    f = st.list_findings()
    assert len(f) == 1 and f[0]["instance_id"] is None, \
        "a finding is an observation: it loses its instance, not its existence"
    assert st.get_points(2) is None
    st.close()


def test_reconcile_on_a_store_that_is_already_right_changes_nothing(tmp_path):
    st = _store(tmp_path, [1, 2])
    assert st.reconcile([1, 2, 99])["removed"] == 0
    assert len(st.list_instances()) == 2
    st.close()


# ── the transaction registers each path once ────────────────────────────

def test_the_manifest_never_carries_a_path_twice(tmp_path):
    """Step 4 stages the segmentation and step 9c republishes it. The swap
    moves the manifest row by row, so a second row renames a file that is
    already gone and aborts the whole swap."""
    src = Path(__file__).resolve().parents[1] / "correction" / "apply.py"
    body = src.read_text()
    assert "_art_seen" in body and "if rel in _art_seen:" in body, \
        "_art must stay idempotent"


# ── the class byte: one byte, ids that do not fit in one byte ────────────
# USER 2026-09-20: "si detectaste un error corregilo, nunca detectes algo que
# no corregiste". pccr had EIGHT live instances above 254 — exposed_ceiling_
# beams, four black_electrical_cable, three exposed_ceiling_wiring — and a
# `min(id, 255)` painted all eight the same class: 45,091 points that were one
# block in the viewer, where one toggle switched all of them.

def test_while_the_ids_fit_the_byte_is_the_instance_id(tmp_path):
    """Every ordinary session must come out byte-identical to before, so a
    reader that has not learned about the map keeps working."""
    from segmentation.republish import CLASS_MAP
    _result(tmp_path, [_inst(3, [0]), _inst(200, [1])])
    republish_membership(tmp_path, n_points=2, log=lambda m: None)
    assert list(np.load(tmp_path / CLASSIFICATION)) == [3, 200]
    assert json.loads((tmp_path / CLASS_MAP).read_text())["encoding"] == "identity"


def test_ids_past_the_byte_get_a_compact_index_not_the_same_colour(tmp_path):
    from segmentation.republish import CLASS_MAP, class_byte, as_instance_ids
    ids = [44, 257, 258, 259, 262, 264, 266, 267, 270]
    _result(tmp_path, [_inst(i, [n]) for n, i in enumerate(ids)])
    republish_membership(tmp_path, n_points=len(ids), log=lambda m: None)
    cls = np.load(tmp_path / CLASSIFICATION)
    assert json.loads((tmp_path / CLASS_MAP).read_text())["encoding"] == "compact"
    assert len(set(cls.tolist())) == len(ids), \
        "every object has to keep its own byte — that is the whole defect"
    assert cls.max() <= 255
    # and the byte still resolves back to the object it belongs to
    assert list(as_instance_ids(tmp_path, cls)) == ids
    assert class_byte(tmp_path, 270) == cls[-1]


def test_more_objects_than_codes_is_refused_not_collapsed(tmp_path):
    from segmentation.republish import _encode
    with pytest.raises(ValueError, match="one class byte"):
        _encode(list(range(300, 300 + 256)))


def test_a_session_with_no_map_reads_as_identity(tmp_path):
    """Every session written before the map existed carries the id in the
    byte, and nothing may reinterpret it."""
    from segmentation.republish import class_byte, as_instance_ids, instance_of
    assert instance_of(tmp_path) == {}
    assert class_byte(tmp_path, 77) == 77
    assert list(as_instance_ids(tmp_path, np.array([0, 5, 200], np.uint8))) == [0, 5, 200]
