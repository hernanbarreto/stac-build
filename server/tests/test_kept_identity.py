"""The reconstruction's identity and the keyframes' chunks OUTLIVE the Omega records (USER 2026-11-19: "Esto de
liberar los intermedios debería hacerse de forma automática ... el sistema debe liberar ese espacio" / "Ok lo de
la huella con archivos chicos dale hacelo y no te mandes más cagadas"): kept in two small files written FROM the
records; recovered, for a session that lost them, only from artifacts that all name one reconstruction."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from correction import epoch as E
from correction import release as R
from correction.floor import omega_chunk_of_frames

FRAMES = [0, 6, 12, 18, 24, 30]
OWNER = [0, 0, 0, 1, 1, 2]            # the cube that owns each keyframe (the plan's)


def _session(tmp_path: Path, *, orientation: bool = True) -> Path:
    out = tmp_path / "output"
    rec = out / "omega_run" / "results_output"
    rec.mkdir(parents=True)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in FRAMES) + "\n")
    (out / "vggt_omega_config.yaml").write_text("Model:\n  name: omega\n")
    (out / ".metric_scale_applied").write_text("2.6381\n")
    if orientation:
        (out / ".orientation_applied").write_text("baked\n")
    rng = np.random.default_rng(7)
    for f, k in zip(FRAMES, OWNER):
        np.savez_compressed(rec / f"frame_{f}.npz", depth=rng.random((4, 3)).astype(np.float32),
                            chunk=np.int64(k), frame_global=np.int64(FRAMES.index(f)))
    run = out / "maplong_run"
    run.mkdir()
    (run / "chunk_sim3.json").write_text(json.dumps({
        "layout": "cubes", "owner": OWNER, "chunk_sets": [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5]],
        "poses": [{"s": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0]}] * 3}))
    return out


def _release(out: Path) -> None:
    shutil.rmtree(out / "omega_run")


def test_the_records_decide_while_they_exist(tmp_path):
    out = _session(tmp_path)
    rid = E.reconstruction_id(out)
    E.write_reconstruction_id_record(out, "0" * 64, source="recovered")      # a kept file naming ANOTHER id
    E._RID_MEMO.clear()
    assert E.reconstruction_id(out) == rid, "the kept file is never read beside the records"


def test_the_identity_outlives_the_records(tmp_path):
    out = _session(tmp_path)
    rid = E.reconstruction_id(out)
    assert E.keep_reconstruction_id(out) == rid
    doc = json.loads((out / E.RECONSTRUCTION_ID_RECORD).read_text())
    assert doc["reconstruction_id"] == rid and doc["source"] == "records" and doc["records"] == len(FRAMES)
    assert set(doc["files"]) == set(E.KEPT_ID_FILES) and all(doc["files"].values())
    _release(out)
    E._RID_MEMO.clear()
    assert E.reconstruction_id(out) == rid
    assert E.reconstruction_id_or_none(out) == rid
    assert E.same_reconstruction({"reconstruction_id": rid}, E.reconstruction_id_or_none(out))[0]


def test_a_session_that_kept_nothing_has_no_identity_once_the_records_go(tmp_path):
    out = _session(tmp_path)
    _release(out)
    with pytest.raises(E.ReconstructionIdError, match="holds no Omega record"):
        E.reconstruction_id(out)
    assert E.reconstruction_id_or_none(out) is None


def test_the_kept_identity_is_held_to_the_other_identity_files(tmp_path):
    out = _session(tmp_path, orientation=False)
    rid = E.keep_reconstruction_id(out)
    assert json.loads((out / E.RECONSTRUCTION_ID_RECORD).read_text())["files"][".orientation_applied"] is None
    _release(out)
    assert E.reconstruction_id(out) == rid
    frames = (out / "camera_frames.txt").read_text()
    (out / "camera_frames.txt").write_text(frames + "36\n")                  # another keyframe list
    with pytest.raises(E.ReconstructionIdError, match="no longer the reconstruction"):
        E.reconstruction_id(out)
    (out / "camera_frames.txt").write_text(frames)
    assert E.reconstruction_id(out) == rid
    (out / ".orientation_applied").write_text("baked\n")                     # absent when it was kept
    with pytest.raises(E.ReconstructionIdError, match="was absent when the identity was kept"):
        E.reconstruction_id(out)
    (out / ".orientation_applied").unlink()
    (out / ".metric_scale_applied").unlink()                                 # present when it was kept
    with pytest.raises(E.ReconstructionIdError, match="is missing"):
        E.reconstruction_id(out)


def test_an_unreadable_or_foreign_kept_file_is_no_identity(tmp_path):
    out = _session(tmp_path)
    E.keep_reconstruction_id(out)
    _release(out)
    p = out / E.RECONSTRUCTION_ID_RECORD
    good = p.read_text()
    p.write_text("{not json")
    with pytest.raises(E.ReconstructionIdError, match="unreadable"):
        E.reconstruction_id(out)
    doc = json.loads(good)
    doc["version"] = 99
    p.write_text(json.dumps(doc))
    with pytest.raises(E.ReconstructionIdError, match="not a kept identity"):
        E.reconstruction_id(out)
    doc = json.loads(good)
    doc["reconstruction_id"] = "abc"
    p.write_text(json.dumps(doc))
    with pytest.raises(E.ReconstructionIdError, match="not a kept identity"):
        E.reconstruction_id(out)


def test_keeping_the_same_identity_twice_rewrites_nothing(tmp_path):
    out = _session(tmp_path)
    E.keep_reconstruction_id(out)
    before = (out / E.RECONSTRUCTION_ID_RECORD).stat()
    E.keep_reconstruction_id(out)
    after = (out / E.RECONSTRUCTION_ID_RECORD).stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_an_identity_is_kept_from_the_records_never_without_them(tmp_path):
    out = _session(tmp_path)
    _release(out)
    with pytest.raises(E.ReconstructionIdError, match="kept FROM the records"):
        E.keep_reconstruction_id(out)


def test_the_chunk_of_a_keyframe_outlives_its_record(tmp_path):
    out = _session(tmp_path)
    from_records = omega_chunk_of_frames(out, FRAMES)
    assert from_records.tolist() == OWNER
    got = R.keep(out, log=lambda m: None)
    assert got["keyframes"] == len(FRAMES)
    _release(out)
    E._RID_MEMO.clear()
    assert np.array_equal(omega_chunk_of_frames(out, FRAMES), from_records)
    assert np.array_equal(omega_chunk_of_frames(out, FRAMES[::-1]), from_records[::-1])
    with pytest.raises(RuntimeError, match="the chunk of a keyframe is unknown"):
        omega_chunk_of_frames(out, FRAMES + [999])


def test_kept_chunks_of_another_reconstruction_are_not_taken(tmp_path):
    out = _session(tmp_path)
    R.keep(out, log=lambda m: None)
    _release(out)
    doc = json.loads((out / R.KEPT_CHUNKS_FILE).read_text())
    doc["reconstruction_id"] = "f" * 64
    (out / R.KEPT_CHUNKS_FILE).write_text(json.dumps(doc))
    assert R.kept_chunks(out) == {}
    with pytest.raises(RuntimeError, match="the chunk of a keyframe is unknown"):
        omega_chunk_of_frames(out, FRAMES)


def test_nothing_is_kept_from_an_incomplete_set_of_records(tmp_path):
    out = _session(tmp_path)
    (out / "omega_run" / "results_output" / "frame_12.npz").unlink()
    with pytest.raises(R.ReleaseError, match="keyframe 12 has no Omega record"):
        R.keep(out, log=lambda m: None)
    assert not (out / E.RECONSTRUCTION_ID_RECORD).exists() and not (out / R.KEPT_CHUNKS_FILE).exists()


# ── a session that already lost its records ────────────────────────────────────────────────────────
def _stripped(tmp_path: Path, *, carriers=("gauge.json", "precision/refine.json", "floor_level.json")):
    out = _session(tmp_path)
    rid = E.reconstruction_id(out)
    chunks = omega_chunk_of_frames(out, FRAMES)
    (out / "precision").mkdir()
    for rel in carriers:
        doc = ({"stamp": {"reconstruction_id": rid, "cloud_sha256": "c" * 64}} if rel == "floor_level.json"
               else {"reconstruction_id": rid, "rows": [{"reconstruction_id": rid}]} if rel == "gauge.json"
               else {"stamp": {"reconstruction_id": rid}})
        (out / rel).write_text(json.dumps(doc))
    _release(out)
    E._RID_MEMO.clear()
    return out, rid, chunks


def test_recover_takes_the_one_id_every_artifact_names(tmp_path):
    out, rid, chunks = _stripped(tmp_path)
    logs = []
    rep = R.recover(out, log=logs.append)
    assert rep["reconstruction_id"] == rid and rep["written"] is False
    assert rep["carriers"] == ["floor_level.json", "gauge.json", "precision/refine.json"]
    assert not (out / E.RECONSTRUCTION_ID_RECORD).exists(), "a dry run writes nothing"
    rep = R.recover(out, write=True, log=logs.append)
    assert rep["written"] is True
    assert E.reconstruction_id(out) == rid
    doc = json.loads((out / E.RECONSTRUCTION_ID_RECORD).read_text())
    assert doc["source"] == "recovered" and doc["carriers"] == rep["carriers"]
    assert np.array_equal(omega_chunk_of_frames(out, FRAMES), chunks), "the plan's owner IS the record's chunk"


def test_recover_refuses_artifacts_that_name_two_reconstructions(tmp_path):
    out, rid, _ = _stripped(tmp_path)
    (out / "precision" / "tracks.json").write_text(json.dumps({"reconstruction_id": "a" * 64}))
    with pytest.raises(R.ReleaseError, match="2 different reconstructions"):
        R.recover(out, write=True, log=lambda m: None)
    assert not (out / E.RECONSTRUCTION_ID_RECORD).exists()


def _step_stamp(rid: str) -> dict:
    """What a step's ``repro.stamp`` record looks like: digests of what it read, never the values."""
    import repro
    return repro.stamp(config={"reconstruction_id": rid, "prompts": ["floor", "wall"]})


def test_a_step_stamp_computed_with_the_identity_is_not_a_second_reconstruction(tmp_path):
    out, rid, _ = _stripped(tmp_path)
    st = _step_stamp(rid)
    assert st["config"]["reconstruction_id"] == R._id_digest(rid) != rid, "a stamp keeps the digest of the id"
    (out / "segmentation.json").write_text(json.dumps({"reconstruction_id": rid, "stamp": st, "instances": []}))
    rep = R.recover(out, write=True, log=lambda m: None)
    assert rep["written"] is True and rep["stamped_with_it"] == ["segmentation.json"]
    assert "segmentation.json" in rep["carriers"] and E.reconstruction_id(out) == rid


def test_recover_refuses_a_step_stamp_computed_with_another_identity(tmp_path):
    out, rid, _ = _stripped(tmp_path)
    (out / "segmentation.json").write_text(json.dumps({"stamp": _step_stamp("b" * 64), "instances": []}))
    with pytest.raises(R.ReleaseError, match="computed with another identity"):
        R.recover(out, write=True, log=lambda m: None)
    assert not (out / E.RECONSTRUCTION_ID_RECORD).exists()


def test_recover_refuses_a_seal_that_vouches_for_itself(tmp_path):
    out, rid, _ = _stripped(tmp_path, carriers=("floor_level.json",))
    with pytest.raises(R.ReleaseError, match="does not vouch for itself"):
        R.recover(out, write=True, log=lambda m: None)
    assert not (out / E.RECONSTRUCTION_ID_RECORD).exists()


def test_recover_refuses_with_nothing_to_recover_from_and_beside_the_records(tmp_path):
    out, rid, _ = _stripped(tmp_path, carriers=())
    with pytest.raises(R.ReleaseError, match="no artifact carries"):
        R.recover(out, write=True, log=lambda m: None)
    with_records = _session(tmp_path / "b")
    with pytest.raises(R.ReleaseError, match="holds Omega records"):
        R.recover(with_records, write=True, log=lambda m: None)


def test_recover_reports_what_was_sealed_without_an_id(tmp_path):
    out, rid, _ = _stripped(tmp_path, carriers=("gauge.json", "precision/refine.json"))
    (out / "floor_level.json").write_text(json.dumps({"stamp": {"reconstruction_id": None, "cloud_sha256": "c" * 64}}))
    logs = []
    rep = R.recover(out, write=True, log=logs.append)
    assert rep["stamped_without_an_id"] == ["floor_level.json"] and rep["written"] is True
    assert any("WITHOUT an id" in m for m in logs)


def test_recover_keeps_the_identity_when_the_run_was_not_cut_in_cubes(tmp_path):
    out, rid, _ = _stripped(tmp_path)
    (out / "maplong_run" / "chunk_sim3.json").write_text(json.dumps({"layout": "chain", "ranges": [[0, 6]]}))
    logs = []
    rep = R.recover(out, write=True, log=logs.append)
    assert rep["written"] is True and E.reconstruction_id(out) == rid
    assert not (out / R.KEPT_CHUNKS_FILE).exists() and any("chunks NOT recovered" in m for m in logs)


# ── what the identity is for: the floor transform's seal ──────────────────────────────────────────
def _sealed(tmp_path: Path):
    from repro import sha256_file
    out = _session(tmp_path)
    (out / "cleaned_cloud.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n" + bytes(64))
    np.savez(out / "floor_transform.npz", s=1.0, R=np.eye(3), t=np.zeros(3))
    rid = E.reconstruction_id(out)
    (out / "gauge.json").write_text(json.dumps({"reconstruction_id": rid}))
    (out / "floor_level.json").write_text(json.dumps({"stamp": {
        "reconstruction_id": rid, "cloud_sha256": sha256_file(out / "cleaned_cloud.ply"),
        "floor_transform_sha256": sha256_file(out / "floor_transform.npz")}}))
    return out, rid


def test_the_floor_seal_holds_after_the_records_are_released_with_the_identity_kept(tmp_path):
    from segmentation.pipeline import _floor_seal_status
    out, rid = _sealed(tmp_path)
    assert _floor_seal_status(out, out / "cleaned_cloud.ply")[0] is True
    R.keep(out, log=lambda m: None)
    _release(out)
    E._RID_MEMO.clear()
    ok, why = _floor_seal_status(out, out / "cleaned_cloud.ply")
    assert ok is True and rid[:12] in why


def test_the_memoised_verdict_follows_the_kept_identity(tmp_path, monkeypatch):
    from segmentation import floor_frame as FF
    monkeypatch.setattr(FF, "_DISK", tmp_path / "memo")
    FF._VERDICTS.clear()
    out, rid = _sealed(tmp_path)
    assert FF.floor_seal_verdict(out)[0] is True
    _release(out)                                                            # the records go, nothing kept
    E._RID_MEMO.clear()
    ok, why = FF.floor_seal_verdict(out)
    assert ok is False and "no reconstruction id" in why
    assert FF.sealed_floor_transform(out, log=lambda m: None) is None
    R.recover(out, write=True, log=lambda m: None)                           # the identity comes back
    ok, why = FF.floor_seal_verdict(out)
    assert ok is True, "the verdict saved without the identity is not served once the identity is there"
    assert FF.sealed_floor_transform(out, log=lambda m: None) is not None
    FF._VERDICTS.clear()                                                     # another process: the on-disk memo
    assert FF.floor_seal_verdict(out)[0] is True
    (out / E.RECONSTRUCTION_ID_RECORD).unlink()
    E._RID_MEMO.clear()
    assert FF.floor_seal_verdict(out)[0] is False
