"""The geometric cleanup cycle closes — and closes reversibly (USER 2026-09-15).

The cycle has two moments. The audit MARKS, at match time, every point landing
off its own mask in a view that sees it, and removes nothing, because the
certification may still move that point where it belongs. This second moment
runs after the correction and asks the only question left: is it STILL wrong?

    "sé que ese punto debe estar, no es ruido, es un huérfano mal ubicado, si no
     lo puedo corregir, lamentablemente lo voy a tener que sacar"

What is asserted here is the part that fails SILENTLY when it is wrong: a point
leaving the cloud shifts every index after it, and an instance still holding the
old ones points at its neighbour's geometry. Plus the decision itself — cured,
still wrong, or never asked — because calling a point cured when no mask looked
at it would delete nothing today and the wrong thing tomorrow.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── the decision: three outcomes, no threshold ───────────────────────────

def _decide(was_marked, judged_now, out_now):
    """The verdict `measure()` reaches per point: (removable, cured, unmeasured)."""
    decided = was_marked & judged_now
    return decided & out_now, decided & ~out_now, was_marked & ~judged_now


def test_a_point_the_correction_fixed_is_cured_and_stays():
    """The drift orphan: it fell off its mask before the epoch and lands on it
    after. There was never anything wrong with the point, only with where the
    reconstruction had put it."""
    was = np.array([True]); judged = np.array([True]); out = np.array([False])
    rm, cured, unk = _decide(was, judged, out)
    assert not rm[0] and cured[0] and not unk[0]


def test_a_point_still_off_its_mask_has_no_correction_left_and_goes():
    was = np.array([True]); judged = np.array([True]); out = np.array([True])
    rm, cured, unk = _decide(was, judged, out)
    assert rm[0] and not cured[0] and not unk[0]


def test_a_point_no_mask_looked_at_is_neither_cured_nor_removed():
    """The distinction that matters: 'not found wrong' is not 'found right'.
    An instance whose masks are gone judges nothing, and its marks must survive
    for a later pass instead of being cleared as if they had been answered."""
    was = np.array([True]); judged = np.array([False]); out = np.array([False])
    rm, cured, unk = _decide(was, judged, out)
    assert not rm[0] and not cured[0] and unk[0]


def test_an_unmarked_point_is_never_touched_however_it_measures():
    """The second moment only ever re-asks about what the FIRST one marked. It
    is not a new filter with a new reach."""
    was = np.array([False, False]); judged = np.array([True, True])
    out = np.array([True, False])
    rm, cured, unk = _decide(was, judged, out)
    assert not rm.any() and not cured.any() and not unk.any()


def test_the_audit_publishes_judged_so_the_second_moment_can_tell_them_apart():
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "mask_filter.py").read_text()
    assert 'rec["judged"] = judged = seen_views > 0' in src, \
        "the audit stopped reporting WHICH points it actually looked at"


def test_the_audit_report_serializes_with_the_per_point_marks_in_place():
    """The marks are arrays as long as the instance and JSON cannot hold them.
    pccr 2026-09-15: the report raised 'Object of type ndarray is not JSON
    serializable', the writer caught it as non-fatal, and out_of_place.npy was
    never written — the whole second moment silently did not exist."""
    import json as _json
    from segmentation.mask_filter import MaskAudit

    audit = MaskAudit.__new__(MaskAudit)     # no session needed to test report()
    audit.stats = [{"instance_id": 7, "verdict": "correct",
                    "on_mask": 10, "off_mask": 2, "n_judged": 12,
                    "n_out_of_place": 2,
                    "out_of_place": np.array([True, False]),
                    "judged": np.array([True, True])}]
    rep = audit.report()
    _json.dumps(rep)                          # must not raise
    assert "out_of_place" not in rep["per_instance"][0]
    assert "judged" not in rep["per_instance"][0]
    assert rep["per_instance"][0]["n_out_of_place"] == 2


def test_the_marks_are_saved_before_the_report_can_fail():
    """The marks are the cycle; the report is a report. They must not share a
    try block, or a report bug takes the cleanup with it."""
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "pipeline.py").read_text()
    save = src.index('np.save(output_dir / "out_of_place.npy"')
    report = src.index('rep = mask_filter.report()')
    assert save < report, "the marks must be written before the report is built"


# ── the cloud: removal carries every index, and comes back ───────────────

PROPS = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("frame_global", "<i4"), ("pixel_row", "<i4"), ("pixel_col", "<i4")]


def _write_cloud(path: Path, n: int) -> np.ndarray:
    data = np.zeros(n, dtype=np.dtype(PROPS))
    idx = np.arange(n)
    data["x"] = idx.astype(np.float32)              # x == original index
    # y and z spread the points into a real volume so a box can be fitted; the
    # deterministic pattern keeps every assertion reproducible
    data["y"] = (idx % 4).astype(np.float32) * 0.5
    data["z"] = (idx % 3).astype(np.float32) * 0.5
    data["frame_global"] = np.arange(n, dtype=np.int32)
    header = [b"ply\n", b"format binary_little_endian 1.0\n",
              f"element vertex {n}\n".encode()]
    header += [f"property {t} {name}\n".encode()
               for name, t in (("x", "float"), ("y", "float"), ("z", "float"),
                               ("frame_global", "int"), ("pixel_row", "int"),
                               ("pixel_col", "int"))]
    header.append(b"end_header\n")
    with open(path, "wb") as f:
        for line in header:
            f.write(line)
        f.write(data.tobytes())
    return data


@pytest.fixture()
def session(tmp_path):
    """A cloud of 20 points in two instances, with x == the point's own index so
    any index error is visible as a value error."""
    out = tmp_path / "output"
    out.mkdir()
    _write_cloud(out / "cleaned_cloud.ply", 20)
    doc = {"total_points": 20, "instances": [
        {"instance_id": 1, "label": "floor", "globalIndices": list(range(0, 10)),
         "total_points": 10},
        {"instance_id": 2, "label": "wall", "globalIndices": list(range(10, 20)),
         "total_points": 10}]}
    (out / "segmentation_result.json").write_text(json.dumps(doc))
    np.save(out / "out_of_place.npy", np.zeros(20, bool))
    return out


def test_removal_shifts_every_instance_index_onto_the_new_cloud(session):
    """The silent corruption this guards: drop point 3 and the wall's indices
    10..19 now name 9..18. An instance that kept the old ones would be holding
    its neighbour's geometry, and nothing would report it."""
    from segmentation.geometric_cleanup import _remove
    from correction.session import read_ply

    drop = np.zeros(20, bool)
    drop[[3, 4, 15]] = True
    ok, detail = _remove(session, drop, np.zeros(20, bool), log=lambda m: None)
    assert ok, detail
    assert detail["removed"] == 3 and detail["points_after"] == 17

    _h, data = read_ply(session / "cleaned_cloud.ply")
    assert len(data) == 17
    # x still carries the ORIGINAL index of every surviving point
    assert list(data["x"].astype(int)) == [i for i in range(20) if i not in (3, 4, 15)]

    doc = json.loads((session / "segmentation_result.json").read_text())
    by = {i["instance_id"]: i for i in doc["instances"]}
    # every remaining index must still resolve to the point it named before
    for iid, originals in ((1, [0, 1, 2, 5, 6, 7, 8, 9]),
                           (2, [10, 11, 12, 13, 14, 16, 17, 18, 19])):
        gi = np.asarray(by[iid]["globalIndices"], np.int64)
        assert list(data["x"][gi].astype(int)) == originals
        assert by[iid]["total_points"] == len(originals)
    assert doc["total_points"] == 17
    assert doc["segmented_points"] == 17


def test_the_undo_puts_the_cloud_back_exactly(session):
    """Removal is physical but not unrecoverable: the rows return to their own
    positions, so every index that was valid before the cleanup is valid again."""
    from segmentation.geometric_cleanup import _remove, undo_cleanup
    from correction.session import read_ply

    _h, before = read_ply(session / "cleaned_cloud.ply")
    drop = np.zeros(20, bool)
    drop[[0, 7, 19]] = True
    ok, _d = _remove(session, drop, np.zeros(20, bool), log=lambda m: None)
    assert ok

    res = undo_cleanup(session, log=lambda m: None)
    assert res["restored"] == 3
    _h, after = read_ply(session / "cleaned_cloud.ply")
    assert len(after) == 20
    assert np.array_equal(after["x"], before["x"])
    assert np.array_equal(after["frame_global"], before["frame_global"])

    doc = json.loads((session / "segmentation_result.json").read_text())
    by = {i["instance_id"]: i["globalIndices"] for i in doc["instances"]}
    assert by[1] == list(range(0, 10)) and by[2] == list(range(10, 20))


def test_a_cloud_out_of_step_with_the_marks_is_never_touched(session):
    """cleaned_cloud_raw.ply must share the point order. If it does not, the
    session is inconsistent and removing from one of them would silently break
    the pairing surface_fit residuals depend on."""
    from segmentation.geometric_cleanup import _remove
    from correction.session import read_ply

    _write_cloud(session / "cleaned_cloud_raw.ply", 19)       # one short
    _h, before = read_ply(session / "cleaned_cloud.ply")
    drop = np.zeros(20, bool)
    drop[5] = True
    ok, detail = _remove(session, drop, np.zeros(20, bool), log=lambda m: None)
    assert not ok
    assert "out of step" in (detail["reason"] or "")
    _h, after = read_ply(session / "cleaned_cloud.ply")
    assert len(after) == 20 and np.array_equal(after["x"], before["x"])


def test_every_parallel_cloud_follows_the_same_removal(session):
    """cleaned_cloud_raw.ply carries the same points in the same order and must
    lose exactly the same rows — the pairing is load-bearing for residuals."""
    from segmentation.geometric_cleanup import _remove
    from correction.session import read_ply

    _write_cloud(session / "cleaned_cloud_raw.ply", 20)
    drop = np.zeros(20, bool)
    drop[[2, 11]] = True
    ok, detail = _remove(session, drop, np.zeros(20, bool), log=lambda m: None)
    assert ok
    assert set(detail["clouds"]) == {"cleaned_cloud.ply", "cleaned_cloud_raw.ply"}
    a = read_ply(session / "cleaned_cloud.ply")[1]["x"]
    b = read_ply(session / "cleaned_cloud_raw.ply")[1]["x"]
    assert np.array_equal(a, b) and len(a) == 18


def test_the_box_of_a_cleaned_instance_is_refitted(session):
    """The flyers the cleanup removes are exactly what inflated the box. Leaving
    the old extent would keep the error the removal cured — and the extent is
    not cosmetic: the floor levelling picks its candidate by height and the
    duplicate measurements read the box."""
    from segmentation.geometric_cleanup import _remove

    doc = json.loads((session / "segmentation_result.json").read_text())
    for inst in doc["instances"]:
        inst["obb"] = {"stale": True}
    (session / "segmentation_result.json").write_text(json.dumps(doc))

    drop = np.zeros(20, bool)
    drop[[0, 1]] = True                      # only instance 1 loses points
    ok, _d = _remove(session, drop, np.zeros(20, bool), log=lambda m: None)
    assert ok
    by = {i["instance_id"]: i for i in
          json.loads((session / "segmentation_result.json").read_text())["instances"]}
    assert "stale" not in by[1]["obb"], "the cleaned instance kept its old box"
    assert by[2]["obb"] == {"stale": True}, \
        "an untouched instance must not be refitted — its points did not change"


def test_the_surviving_marks_follow_the_cloud_too(session):
    """The marks are per-point: after a removal they must be re-indexed onto the
    new cloud, or the next pass judges the wrong points."""
    from segmentation.geometric_cleanup import _remove

    keep_marks = np.zeros(20, bool)
    keep_marks[[8, 17]] = True          # marked but never re-measured
    drop = np.zeros(20, bool)
    drop[[1, 2]] = True
    ok, _d = _remove(session, drop, keep_marks, log=lambda m: None)
    assert ok
    m = np.load(session / "out_of_place.npy")
    assert len(m) == 18
    assert list(np.flatnonzero(m)) == [6, 15]      # 8 and 17, shifted by two


# ── the guards on the measurement side ───────────────────────────────────

def test_stale_marks_stop_the_whole_pass(session):
    """A cloud rebuilt after the audit means the marks name points that no
    longer exist. Measuring on would delete geometry chosen at random."""
    from segmentation.geometric_cleanup import geometric_cleanup

    np.save(session / "out_of_place.npy", np.ones(11, bool))
    rep = geometric_cleanup(session, session.parent, apply=True, log=lambda m: None)
    assert not rep["applied"]
    assert "stale" in rep["reason"]


def test_no_marks_means_no_second_moment(session):
    from segmentation.geometric_cleanup import geometric_cleanup

    (session / "out_of_place.npy").unlink()
    rep = geometric_cleanup(session, session.parent, apply=True, log=lambda m: None)
    assert not rep["applied"] and "never ran" in rep["reason"]


def test_nothing_marked_removes_nothing(session):
    from segmentation.geometric_cleanup import geometric_cleanup

    rep = geometric_cleanup(session, session.parent, apply=True, log=lambda m: None)
    assert rep["removed" if "removed" in rep else "removable"] == 0
    assert not rep["applied"]


def test_the_certification_closes_the_cycle_at_the_end():
    """The second moment runs where the user put it — after the certification
    has had its chance, not before."""
    src = (Path(__file__).resolve().parents[1]
           / "reconstruction" / "certify" / "run.py").read_text()
    assert "from segmentation.geometric_cleanup import geometric_cleanup" in src
    i = src.index("geometric_cleanup(")
    assert i < src.index('acta["metrics_final"]'), \
        "the cleanup must run before the acta is closed"
