"""The SAM3 store stays SEALED through the Segmentation Manager's edits (wave 2 cross,
2026-10-08): the projection and the certification refuse a store whose seal names other mask
bytes than the ones on disk (`segmentation.pipeline.check_store_seal`, point 83), and an erase /
undo / delete / mask paint used to leave exactly that behind. `segmentation.erase.reseal_store`
re-seals the store after every rewrite of seg_masks.npz; the edit is recorded; a store that
never had a seal is left as it is. The erase tools read floor_transform.npz under the viewer's
ONE rule (point 158, `segmentation.floor_frame`). CPU only, synthetic session."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import erase as E                                   # noqa: E402
from segmentation import floor_frame as FF                            # noqa: E402
from segmentation import pipeline as P                                # noqa: E402

from test_p3_projection import FRAMES, H, W, _session, fake_potree    # noqa: E402,F401


CFG = {"models": {"segmentation": {"batch_size": 1}}, "segmentation": {}}


def _seal(out, monkeypatch):
    """Seal the synthetic store exactly as the SAM3 stage does (sam3_store_seal)."""
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: object())
    doc = json.loads((out / "segmentation.json").read_text())
    doc["stamp"] = P.sam3_store_seal(CFG, ["monitor", "wall"], FRAMES, [H, W], None,
                                     out / "seg_masks.npz")
    (out / "segmentation.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    assert P.check_store_seal(out) == []
    return doc["stamp"]


def _monitor_sphere(out, res):
    """A sphere around one point of the monitor instance (display frame = identity here)."""
    xyz = P._load_ply_origins(out / "cleaned_cloud.ply")[0]
    gi = np.asarray(next(i for i in res["instances"] if i["instance_id"] == 1)["globalIndices"])
    c = xyz[gi[len(gi) // 2]]
    return [{"center": c.tolist(), "radius": 0.05}]


def test_an_erase_and_its_undo_keep_the_store_sealed_and_record_the_edit(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    sealed = _seal(out, monkeypatch)
    res = P._match_and_save_result(out)
    assert res["decisions"]["store_seal"]["intact"]
    npz_before = (out / "seg_masks.npz").read_bytes()

    rep = E.erase_spheres(out, _monitor_sphere(out, res))
    assert rep["total_removed"] > 0 and rep["undo"]["pixels"], "the erase cleared mask pixels"
    assert (out / "seg_masks.npz").read_bytes() != npz_before, "the store was rewritten"
    assert P.check_store_seal(out) == [], "the seal names the bytes on disk"
    doc = json.loads((out / "segmentation.json").read_text())
    assert doc[E.STORE_EDITS_KEY] == {"erase": 1}
    st = doc["stamp"]
    from repro import sha256_file, sha256_json
    assert st["inputs"]["seg_masks.npz"] == sha256_file(out / "seg_masks.npz")
    assert st["code"] == sealed["code"] and st["config"] == sealed["config"], \
        "what drew the masks is carried over unchanged"
    assert st["stamp_version"] == sealed["stamp_version"] and st["sha256"] != sealed["sha256"]
    body = {k: st[k] for k in ("stamp_version", "inputs", "code", "config")}
    assert st["sha256"] == sha256_json(body), "the digest is computed as repro.stamp computes it"
    # the projection ACCEPTS the edited store (it used to refuse it: 'changed since it was sealed')
    res2 = P._match_and_save_result(out)
    assert res2["decisions"]["store_seal"] == {"intact": True, "declared": []}

    E.undo_erase(out, rep["undo"])
    assert P.check_store_seal(out) == []
    doc = json.loads((out / "segmentation.json").read_text())
    assert doc[E.STORE_EDITS_KEY] == {"erase": 1, "undo_erase": 1}


def test_a_store_without_a_seal_is_left_unsealed_never_given_one(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)                       # the fixture writes no stamp
    res = P._match_and_save_result(out)
    assert res["decisions"]["store_seal"]["intact"] is False
    logs = []
    rep = E.reseal_store(out, by="erase", log=logs.append)
    assert rep["resealed"] is False and "carries no seal" in rep["reason"]
    assert any("DECLARED" in m for m in logs)
    doc = json.loads((out / "segmentation.json").read_text())
    assert "stamp" not in doc and E.STORE_EDITS_KEY not in doc
    # a real erase on it: still unsealed, still declared by the projection, never refused
    E.erase_spheres(out, _monitor_sphere(out, res))
    fatal, declared = P.store_seal_verdict(out)
    assert fatal == [] and declared and "carries no seal" in declared[0]


def test_reseal_without_a_store_declares_and_writes_nothing(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    logs = []
    rep = E.reseal_store(out, by="delete", log=logs.append)
    assert rep == {"resealed": False, "reason": "no segmentation.json / seg_masks.npz to seal",
                   "sha256": None}
    assert not (out / "segmentation.json").exists() and logs


def _rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_the_erase_tools_read_the_floor_transform_under_the_viewers_rule(tmp_path, fake_potree):
    """A zone is drawn in the frame the viewer SHOWS: the sealed transform, else the cloud's
    own frame. An unsealed npz (the gizmo's before it sealed, another cloud's) is identity for
    the erase too — or the brush would land beside the points it covered."""
    from repro import sha256_file
    out, _, _ = _session(tmp_path)
    assert E._load_floor_transform(out)[0] == 1.0 and np.array_equal(E._load_floor_transform(out)[1], np.eye(3))
    np.savez(out / "floor_transform.npz", s=np.array(2.0), R=_rot_y(30.0), t=np.array([0.0, 0.5, 0.0]))
    s, R, t = E._load_floor_transform(out)                          # unsealed: identity
    assert s == 1.0 and np.array_equal(R, np.eye(3)) and np.array_equal(t, np.zeros(3))
    ok, why = FF.floor_seal_verdict(out)
    assert ok is False and "no floor_level.json" in why
    # sealed for this reconstruction and this cloud: applied — by the erase, the viewer, the stages alike
    from test_p3_wave2_closing import _name_reconstruction
    rid = _name_reconstruction(out)
    seal = {"reconstruction_id": rid, "cloud_sha256": sha256_file(out / "cleaned_cloud.ply"),
            "floor_transform_sha256": sha256_file(out / "floor_transform.npz")}
    (out / "floor_level.json").write_text(json.dumps({"stamp": seal}))
    s, R, t = E._load_floor_transform(out)
    assert s == 2.0 and np.allclose(R, _rot_y(30.0)) and np.allclose(t, [0.0, 0.5, 0.0])
    assert FF.floor_seal_verdict(out)[0] is True
    assert P._display_transform(out, out / "cleaned_cloud.ply", np.zeros((1, 3)), log=lambda m: None)[0] == 2.0
    # the memo follows the files: the npz rewritten after the seal → judged again, refused
    np.savez(out / "floor_transform.npz", s=np.array(2.0), R=_rot_y(31.0), t=np.array([0.0, 0.5, 0.0]))
    ok, why = FF.floor_seal_verdict(out)
    assert ok is False and "rewritten after it was sealed" in why
    logs = []
    assert FF.sealed_floor_transform(out, log=logs.append, tag="T") is None
    assert logs and logs[0].startswith("[T] DECLARED: floor_transform.npz ignored") and "point 158" in logs[0]
    assert E._load_floor_transform(out)[0] == 1.0
    # no cloud: nothing to hold the seal to
    (out / "cleaned_cloud.ply").unlink()
    ok, why = FF.floor_seal_verdict(out)
    assert ok is False and "no cleaned_cloud.ply" in why


# ── the delete mode rewrites the cloud: the floor seal is carried to the edited bytes ────────

def _low_confidence_unsegmented(out, data, n=40):
    """The last ``n`` wall points become unsegmented at confidence 0 — what the delete mode of
    the brush (``conf_below`` with the Unsegmented toggle on) removes physically. Returns them."""
    from test_p3_projection import _write_ply
    res_path = out / "segmentation_result.json"
    res = json.loads(res_path.read_text())
    wall = next(i for i in res["instances"] if i["instance_id"] == 2)
    gi = np.asarray(wall["globalIndices"], np.int64)
    drop = gi[-n:]
    wall["globalIndices"] = gi[:-n].tolist()
    wall["total_points"] = len(wall["globalIndices"])
    res_path.write_text(json.dumps(res))
    data = data.copy()
    data["confidence"][drop] = 0.0
    _write_ply(out / "cleaned_cloud.ply", data)
    return drop


def _seal_floor(out, R, t):
    """floor_transform.npz + the seal every writer puts in floor_level.json (point 158)."""
    from repro import sha256_file
    from test_p3_wave2_closing import _name_reconstruction
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=R, t=np.asarray(t, np.float64))
    seal = {"reconstruction_id": _name_reconstruction(out),
            "cloud_sha256": sha256_file(out / "cleaned_cloud.ply"),
            "floor_transform_sha256": sha256_file(out / "floor_transform.npz")}
    (out / "floor_level.json").write_text(json.dumps({"leveled_by": "t", "stamp": seal}))
    return seal


def test_a_physical_delete_carries_the_floor_seal_to_the_edited_cloud(tmp_path, fake_potree, capsys):
    """The erase's delete mode rewrites cleaned_cloud.ply; the seal named the old bytes, so
    the next projection refused the npz and the viewer dropped the levelled frame (skeptic,
    2026-10-08). The edit removed points and moved none: the seal is carried to the new bytes,
    rid and npz unchanged, the edit recorded — the stages and the viewer keep the frame."""
    from repro import sha256_file
    out, data, _ = _session(tmp_path)
    P._match_and_save_result(out)
    drop = _low_confidence_unsegmented(out, data)
    seal = _seal_floor(out, _rot_y(30.0), [0.0, 0.5, 0.0])
    assert FF.floor_seal_verdict(out)[0] is True
    cloud_before = (out / "cleaned_cloud.ply").read_bytes()

    rep = E.erase_spheres(out, [], conf_below=0.5, include_unsegmented=True)
    assert rep["ledger"]["deleted_points"] == len(drop) and rep["undo"] is None, rep["ledger"]
    assert (out / "cleaned_cloud.ply").read_bytes() != cloud_before, "the cloud was rewritten"
    assert len(P._load_ply_origins(out / "cleaned_cloud.ply")[0]) == len(data) - len(drop)
    # the stages: the seal names the edited cloud, the projection takes the npz
    ok, why = P._floor_seal_status(out, out / "cleaned_cloud.ply")
    assert ok is True, why
    assert P._display_transform(out, out / "cleaned_cloud.ply", np.zeros((1, 3)),
                                log=lambda m: None)[3].startswith("floor_transform.npz (sealed")
    # the viewer and the tools: the same verdict, the same transform
    assert FF.floor_seal_verdict(out)[0] is True
    s, R, t = FF.sealed_floor_transform(out, log=lambda m: None)
    assert s == 1.0 and np.allclose(R, _rot_y(30.0)) and np.allclose(t, [0.0, 0.5, 0.0])
    assert E._load_floor_transform(out)[0] == 1.0 and np.allclose(E._load_floor_transform(out)[1], _rot_y(30.0))
    fl = json.loads((out / "floor_level.json").read_text())
    assert fl["stamp"] == {**seal, "cloud_sha256": sha256_file(out / "cleaned_cloud.ply")}
    assert fl["stamp"]["cloud_sha256"] != seal["cloud_sha256"]
    assert fl[E.STORE_EDITS_KEY] == {"erase_delete": 1} and fl["leveled_by"] == "t"
    assert "floor seal carried to the edited cloud after erase_delete" in capsys.readouterr().out


def test_a_floor_seal_that_did_not_hold_before_the_edit_is_never_carried(tmp_path, fake_potree, capsys):
    """A seal of another cloud (stale before the edit) stays exactly as it is: the viewer showed
    the cloud in its own frame and the zones were tested there — sealing it to the new bytes
    would make the viewer jump to a frame nobody validated."""
    out, data, _ = _session(tmp_path)
    P._match_and_save_result(out)
    drop = _low_confidence_unsegmented(out, data)
    seal = _seal_floor(out, _rot_y(30.0), [0.0, 0.5, 0.0])
    stale = {**seal, "cloud_sha256": "0" * 64}
    (out / "floor_level.json").write_text(json.dumps({"stamp": stale}))
    assert FF.floor_seal_verdict(out)[0] is False
    rep = E.erase_spheres(out, [], conf_below=0.5, include_unsegmented=True)
    assert rep["ledger"]["deleted_points"] == len(drop)
    fl = json.loads((out / "floor_level.json").read_text())
    assert fl["stamp"] == stale and E.STORE_EDITS_KEY not in fl
    assert "DECLARED: floor seal not carried after erase_delete" in capsys.readouterr().out
    assert FF.floor_seal_verdict(out)[0] is False
    assert FF.sealed_floor_transform(out, log=lambda m: None) is None
    assert E._load_floor_transform(out)[0] == 1.0 and np.array_equal(E._load_floor_transform(out)[1], np.eye(3))
    # the helper on its own: a verdict that did not hold is declared, nothing written
    logs = []
    rep = E.reseal_floor(out, by="t", held_before=(False, "another cloud"), log=logs.append)
    assert rep["resealed"] is False and "did not hold before the edit (another cloud)" in rep["reason"]
    assert logs == [m for m in logs if m.startswith("[Erase] DECLARED: floor seal not carried after t")] and logs
    assert json.loads((out / "floor_level.json").read_text())["stamp"] == stale
    # a file with no seal is left as it is
    (out / "floor_level.json").write_text(json.dumps({"selected_instance_id": 1}))
    rep = E.reseal_floor(out, by="t", held_before=(True, "x"), log=logs.append)
    assert rep["resealed"] is False and "carries no seal" in rep["reason"]
    assert json.loads((out / "floor_level.json").read_text()) == {"selected_instance_id": 1}
    # no npz: nothing applies a transform to the cloud, nothing to carry, silent
    (out / "floor_transform.npz").unlink()
    n_logs = len(logs)
    rep = E.reseal_floor(out, by="t", held_before=(True, "x"), log=logs.append)
    assert rep["resealed"] is False and "nothing to carry" in rep["reason"] and len(logs) == n_logs


# ── the memo follows every file the verdict is a function of; one entry per session ─────────

def test_the_memo_follows_an_omega_record_rewritten_in_place(tmp_path, fake_potree):
    """`correction.epoch.reconstruction_id` hashes omega_run/results_output; a record rewritten
    IN PLACE (a resume that re-runs Omega without wiping) keeps the directory's own inode /
    size / mtime, so a memo keyed on the directory alone served the verdict of a reconstruction
    that no longer existed while the stages judged right (skeptic, 2026-10-08). The signature
    now carries every file under the directory; and the memo holds ONE entry per session."""
    import os
    out, _, _ = _session(tmp_path)
    _seal_floor(out, _rot_y(10.0), [0.0, 0.1, 0.0])
    ok, why = FF.floor_seal_verdict(out)
    assert ok is True, why
    key = str(out.resolve())
    assert key in FF._VERDICTS and FF._VERDICTS[key][1] == (True, why)
    rec = out / "omega_run" / "results_output" / "frame_000000.npz"
    dir_before = os.stat(rec.parent)
    ino_before = os.stat(rec).st_ino
    np.savez(rec, depth=np.ones((3, 3), np.float32))              # the same path, truncated in place
    dir_after = os.stat(rec.parent)
    assert os.stat(rec).st_ino == ino_before, "rewritten in place: the same inode"
    assert (dir_before.st_ino, dir_before.st_size, dir_before.st_mtime_ns) == \
        (dir_after.st_ino, dir_after.st_size, dir_after.st_mtime_ns), "the directory's own triple did not move"
    ok, why = FF.floor_seal_verdict(out)
    assert ok is False and "the seal names reconstruction" in why, why
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply") == (ok, why), "the viewer and the stages agree"
    assert [k for k in FF._VERDICTS if k == key] == [key] and FF._VERDICTS[key][1] == (False, why)
    # a record that comes back as it was: judged again, the one entry replaced again
    np.savez(rec, depth=np.zeros((2, 2), np.float32))
    assert FF.floor_seal_verdict(out)[0] is True
    assert len([k for k in FF._VERDICTS if k.startswith(str(tmp_path.resolve()))]) == 1


def test_the_rule_logs_once_per_change_of_verdict(tmp_path, fake_potree):
    """The erase tools ask per zone and per stroke, the scan list on every poll: the
    declaration is ONE line per change of verdict per session, not one per call."""
    out, _, _ = _session(tmp_path)
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(5.0), t=np.zeros(3))
    logs = []
    assert FF.sealed_floor_transform(out, log=logs.append, tag="A") is None
    assert FF.sealed_floor_transform(out, log=logs.append, tag="B") is None
    assert E._load_floor_transform(out)[0] == 1.0
    assert len(logs) == 1 and logs[0].startswith("[A] DECLARED: floor_transform.npz ignored")
    _seal_floor(out, _rot_y(5.0), [0.0, 0.0, 0.0])
    for _ in range(3):
        assert FF.sealed_floor_transform(out, log=logs.append, tag="C") is not None
    assert len(logs) == 2 and logs[1].startswith("[C] floor transform applied — sealed for this reconstruction")
    # the npz rewritten after the seal: a new verdict, a new line
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(6.0), t=np.zeros(3))
    assert FF.sealed_floor_transform(out, log=logs.append, tag="D") is None
    assert len(logs) == 3 and "rewritten after it was sealed" in logs[2]
