"""Manual segment merge (USER 2026-10-09: "3 segmentos son lo mismo — fusionarlos"):
`segmentation.erase.merge_instances` moves every point and every mask of the sources into the
target, the store stays sealed, a re-projection keeps the merge, the merged ids retire, and the
undo restores the session — or refuses when the session changed since. CPU only, synthetic
session (the projection fixture of test_p3_projection)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import erase as E                                   # noqa: E402
from segmentation import pipeline as P                                # noqa: E402
from segmentation.fuse_parent import high_water                       # noqa: E402

from test_erase_store_reseal import _seal                             # noqa: E402
from test_p3_projection import _session, fake_potree                  # noqa: E402,F401


def _by_iid(out):
    res = json.loads((out / "segmentation_result.json").read_text())
    return {int(i["instance_id"]): i for i in res["instances"]}


def _two(out):
    inst = _by_iid(out)
    assert len(inst) >= 2, "the fixture projects at least two objects"
    a, b = sorted(inst)[:2]
    return inst, a, b


def test_merge_moves_points_and_masks_and_survives_a_reprojection(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    inst, tgt, src = _two(out)
    pts = set(inst[tgt]["globalIndices"]) | set(inst[src]["globalIndices"])
    hw_before = high_water(out)

    rep = E.merge_instances(out, tgt, [src])
    assert rep["merged"] == [src] and rep["moved_points"] == len(inst[src]["globalIndices"])
    after = _by_iid(out)
    assert src not in after and set(after[tgt]["globalIndices"]) == pts
    assert P.check_store_seal(out) == [], "the merged store is sealed to its bytes"
    doc = json.loads((out / "segmentation.json").read_text())
    assert doc[E.STORE_EDITS_KEY] == {"merge": 1}
    assert high_water(out) == hw_before, "the merged ids retire — never handed out again"

    # the merge lives in the masks: a re-projection gives the target every merged point
    P._match_and_save_result(out)
    again = _by_iid(out)
    assert src not in again
    assert set(again[tgt]["globalIndices"]) >= set(inst[src]["globalIndices"]) & pts


def test_merge_undo_restores_the_store_and_the_result(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    inst, tgt, src = _two(out)
    npz_before = (out / "seg_masks.npz").read_bytes()
    seg_before = (out / "segmentation.json").read_bytes()

    rep = E.merge_instances(out, tgt, [src])
    back = E.undo_erase(out, rep["undo"])
    assert back["restored"] == int(inst[src]["total_points"])
    assert (out / "seg_masks.npz").read_bytes() == npz_before
    assert (out / "segmentation.json").read_bytes() == seg_before
    restored = _by_iid(out)
    assert set(restored) == set(inst)
    for i in (tgt, src):
        assert restored[i]["globalIndices"] == inst[i]["globalIndices"]
    assert not Path(rep["undo"]["snapshot"]).exists(), "the snapshot leaves with the undo"


def test_merge_undo_is_refused_when_the_session_changed(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    _, tgt, src = _two(out)
    rep = E.merge_instances(out, tgt, [src])
    # something rewrites the store after the merge (an erase, a SAM3 round, a manager edit)
    seg = json.loads((out / "segmentation.json").read_text())
    seg["masks_total"] = int(seg.get("masks_total") or 0) + 1
    (out / "segmentation.json").write_text(json.dumps(seg))
    npz_now = (out / "seg_masks.npz").read_bytes()
    back = E.undo_erase(out, rep["undo"])
    assert back["restored"] == 0 and "changed since the merge" in back["detail"]
    assert (out / "seg_masks.npz").read_bytes() == npz_now, "nothing was restored over it"


def test_merge_needs_a_source_and_existing_segments(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    _, tgt, _ = _two(out)
    import pytest
    with pytest.raises(ValueError):
        E.merge_instances(out, tgt, [tgt])
    with pytest.raises(ValueError):
        E.merge_instances(out, tgt, [987654])


def test_snapshots_no_undo_refers_to_are_pruned(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    _, tgt, src = _two(out)
    stale = out / E.MERGE_UNDO_DIR / "7"
    stale.mkdir(parents=True)
    (stale / "seg_masks.npz").write_bytes(b"x")
    rep = E.merge_instances(out, tgt, [src], keep_snapshots=[])
    assert not stale.exists() and Path(rep["undo"]["snapshot"]).exists()
    assert np.asarray(rep["undo"]["target_before"]).size > 0
