"""Wave 2 cross items of the server package (2026-10-08): the alignment gizmo SEALS the
floor transform it writes (point 158: the same floor_level.json seal the cloud stage and the
explicit level_floor write, so the stages and the viewer apply it); the explicit level_floor
seals the npz's own bytes too; the viewer reads floor_transform.npz under the ONE rule of
`segmentation.floor_frame` (sealed, else the cloud's own frame, declared); the Segmentation
Manager's delete and mask paint re-seal the SAM3 store; the interactive SAM3 runs on the scan's
frozen configuration. ``import main`` loads the FastAPI app (no server, no GPU): ~25 s, once
per module; the endpoints are driven through Starlette's TestClient without the lifespan."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERVER_DIR = Path(__file__).resolve().parents[1]

from test_p3_projection import FRAMES, H, W, _session, fake_potree   # noqa: E402,F401
from test_p3_wave2_closing import _name_reconstruction, _rot_y        # noqa: E402


@pytest.fixture(scope="module")
def main_mod():
    import main
    return main


@pytest.fixture
def client(main_mod):
    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)              # no `with`: the lifespan (vLLM) never runs


def _ctx_to(monkeypatch, main_mod, out: Path):
    ctx = types.SimpleNamespace(output_dir=out, session_dir=out.parent, frames_dir=out.parent / "frames")
    monkeypatch.setattr(main_mod, "_ctx", lambda session_id, scan_key=None: ctx)
    return ctx


def _col_major(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M.flatten(order="F").tolist()


# ── the gizmo seals what it writes; one seal shape for every writer ─────────────────────────

def test_the_alignment_gizmo_seals_the_transform_it_writes(tmp_path, monkeypatch, main_mod, client):
    from repro import sha256_file
    from segmentation import floor_frame as FF
    from segmentation import pipeline as P
    out, _, _ = _session(tmp_path)
    rid = _name_reconstruction(out)
    (out / "floor_level.json").write_text(json.dumps({"selected_instance_id": 2, "stamp": {}}))
    _ctx_to(monkeypatch, main_mod, out)
    r = client.post("/api/sessions/s/alignment", json={"transform": _col_major(1.0, _rot_y(20.0), [0, 0.3, 0])})
    assert r.status_code == 200 and r.json()["ok"], r.text
    fl = json.loads((out / "floor_level.json").read_text())
    assert fl["stamp"] == {"reconstruction_id": rid,
                           "cloud_sha256": sha256_file(out / "cleaned_cloud.ply"),
                           "floor_transform_sha256": sha256_file(out / "floor_transform.npz")}
    assert fl["leveled_by"].startswith("alignment gizmo") and fl["selected_instance_id"] == 2
    # the stages and the viewer take it — the same verdict
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply")[0] is True
    s, R, t = FF.sealed_floor_transform(out, log=lambda m: None)
    assert s == pytest.approx(1.0) and np.allclose(R, _rot_y(20.0)) and np.allclose(t, [0, 0.3, 0])
    # the same gesture on a session with NO cloud: the seal names none — declared, applied by nobody
    (out / "cleaned_cloud.ply").unlink()
    r = client.post("/api/sessions/s/alignment", json={"transform": _col_major(1.0, _rot_y(5.0), [0, 0, 0])})
    assert r.status_code == 200
    fl = json.loads((out / "floor_level.json").read_text())
    assert fl["stamp"]["cloud_sha256"] is None and fl["stamp"]["reconstruction_id"] == rid
    assert FF.sealed_floor_transform(out, log=lambda m: None) is None


def test_level_floor_core_seals_the_npz_bytes_too(main_mod, tmp_path):
    from repro import sha256_file
    from segmentation import pipeline as P
    from test_p5_main_floor import _floor, _session_with_floor
    out = _session_with_floor(tmp_path, _floor(120, 6.0, tilt_deg=2.0, height_m=0.05, noise_m=0.003))
    res = main_mod.level_floor_core(out, out / "segmentation_result.json", out / "cleaned_cloud.ply",
                                    out / "floor_level.json", None, "auto", "t")
    assert res["leveled"], res
    fl = json.loads((out / "floor_level.json").read_text())
    assert fl["stamp"]["floor_transform_sha256"] == sha256_file(out / "floor_transform.npz")
    assert fl["stamp"]["cloud_sha256"] == sha256_file(out / "cleaned_cloud.ply")
    assert fl["stamp"]["reconstruction_id"] is None, "no Omega records: taken by no reader, said so"
    assert P._floor_seal_status(out, out / "cleaned_cloud.ply")[0] is False


# ── the viewer applies the npz only under the seal ──────────────────────────────────────────

def test_the_camera_markers_follow_the_sealed_transform_and_the_clouds_own_frame_otherwise(tmp_path, main_mod, capsys):
    from repro import sha256_file
    out, _, _ = _session(tmp_path)
    (out / "camera_poses.txt").write_text(" ".join(repr(float(x)) for x in np.eye(4).ravel()) + "\n")
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(90.0), t=np.array([0.0, 1.0, 0.0]))
    poses, _, _ = main_mod._camera_poses_payload(out, None)
    assert np.allclose(poses[0], np.eye(4)), "unsealed: the markers sit in the cloud's own frame"
    assert "DECLARED: floor_transform.npz ignored" in capsys.readouterr().out
    rid = _name_reconstruction(out)
    (out / "floor_level.json").write_text(json.dumps({"stamp": {
        "reconstruction_id": rid, "cloud_sha256": sha256_file(out / "cleaned_cloud.ply")}}))
    poses, _, _ = main_mod._camera_poses_payload(out, None)
    M = np.asarray(poses[0])
    assert np.allclose(M[:3, :3], _rot_y(90.0)) and np.allclose(M[:3, 3], [0.0, 1.0, 0.0])


def _raw_reads_of_the_floor_npz(src: str):
    """The functions of ``src`` that call ``np.load(<name>)`` on a name bound, in that same
    function, to a path spelled with 'floor_transform.npz' — whatever the name."""
    import ast
    found = []
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(
                    isinstance(c, ast.Constant) and isinstance(c.value, str) and "floor_transform.npz" in c.value
                    for c in ast.walk(node.value)):
                bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "np"
                    and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in bound):
                found.append(fn.name)
    return found


def test_every_viewer_read_of_the_floor_transform_goes_through_the_one_rule():
    src = (SERVER_DIR / "main.py").read_text()
    # no raw np.load of the npz is left where the server APPLIES it: the one raw read is the
    # explicit level's BASE (`_load_floor_transform_srt`, which composes a delta onto the file
    # on disk and seals the result — it applies nothing to a cloud); found by the AST, so a
    # raw read under any other variable name is caught and a comment breaks nothing
    assert _raw_reads_of_the_floor_npz(src) == ["_load_floor_transform_srt"]
    assert src.count("sealed_floor_matrix(") + src.count("sealed_floor_transform(") >= 7
    # the two writers seal with ONE helper; the gizmo seals right after the npz
    assert src.count("_floor_seal(") >= 2
    i = src.index("def save_alignment(")
    body = src[i:i + 2500]
    assert body.index("_write_floor_transform(output_dir, s, R, t)") < body.index("_seal_floor_level(")


def test_the_viewer_drops_an_unsealed_transform_and_carries_a_sealed_one(tmp_path, monkeypatch, main_mod):
    """Behavioural twin of the source guard above: the potree_ready a correction (or an erase)
    broadcasts carries ``floorTransform`` and levelled camera markers ONLY under the seal — an
    unsealed npz on disk leaves the message without the transform and the markers in the
    cloud's own frame (point 158: shown as projected)."""
    import asyncio
    from repro import sha256_file
    out, _, _ = _session(tmp_path)
    (out / "potree").mkdir()
    (out / "potree" / "metadata.json").write_text(json.dumps({"points": 7}))
    np.savez(out / "floor_transform.npz", s=np.array(1.0), R=_rot_y(90.0), t=np.array([0.0, 1.0, 0.0]))
    _ctx_to(monkeypatch, main_mod, out)
    sent = []

    async def _capture(text):
        sent.append(json.loads(text))

    monkeypatch.setattr(main_mod.viewer_manager, "broadcast_text", _capture)
    asyncio.run(main_mod._correction_notify_viewer("s", out))
    assert sent and sent[-1]["type"] == "potree_ready" and sent[-1]["points"] == 7
    assert "floorTransform" not in sent[-1], "unsealed: the viewer shows the cloud in its own frame"
    cam0 = np.asarray(sent[-1]["cameraPoses"][0])
    assert np.allclose(cam0[:3, 3], [-0.6, 0.0, 0.0]), "the markers in the cloud's own frame too"
    # sealed for this reconstruction and this cloud: the transform rides the message, the
    # markers follow it
    rid = _name_reconstruction(out)
    (out / "floor_level.json").write_text(json.dumps({"stamp": {
        "reconstruction_id": rid, "cloud_sha256": sha256_file(out / "cleaned_cloud.ply")}}))
    asyncio.run(main_mod._correction_notify_viewer("s", out))
    M = np.asarray(sent[-1]["floorTransform"]).reshape(4, 4).T          # column-major on the wire
    assert np.allclose(M[:3, :3], _rot_y(90.0)) and np.allclose(M[:3, 3], [0.0, 1.0, 0.0])
    cam0 = np.asarray(sent[-1]["cameraPoses"][0])
    assert np.allclose(cam0[:3, 3], _rot_y(90.0) @ [-0.6, 0.0, 0.0] + [0.0, 1.0, 0.0])


# ── the store is re-sealed by the Manager's delete and mask paint ───────────────────────────

def _sealed_session(tmp_path, monkeypatch):
    import segmentation.sam3_wrapper as w
    from segmentation import pipeline as P
    out, _, _ = _session(tmp_path)
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: object())
    doc = json.loads((out / "segmentation.json").read_text())
    doc["stamp"] = P.sam3_store_seal({"models": {"segmentation": {}}, "segmentation": {}},
                                     ["monitor", "wall"], FRAMES, [H, W], None, out / "seg_masks.npz")
    (out / "segmentation.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    assert P.check_store_seal(out) == []
    return out


def test_deleting_an_instance_re_seals_the_store(tmp_path, monkeypatch, main_mod, client, fake_potree):
    from segmentation import pipeline as P
    from segmentation.erase import STORE_EDITS_KEY
    out = _sealed_session(tmp_path, monkeypatch)
    P._match_and_save_result(out)
    _ctx_to(monkeypatch, main_mod, out)
    r = client.post("/api/segmentation/delete", json={"session_id": "s", "instance_id": 2, "label": "wall"})
    assert r.status_code == 200, r.text
    assert r.json()["removed_obj_ids"] == [1]
    assert not any(k.endswith("_o1") for k in np.load(out / "seg_masks.npz").files)
    assert P.check_store_seal(out) == [], "the seal names the rewritten bytes"
    doc = json.loads((out / "segmentation.json").read_text())
    assert doc[STORE_EDITS_KEY] == {"delete": 1}
    assert [i["label"] for i in doc["instances"]] == ["monitor"]


def test_painting_a_mask_re_seals_the_store(tmp_path, monkeypatch, main_mod, client):
    from segmentation import pipeline as P
    from segmentation.erase import STORE_EDITS_KEY
    out = _sealed_session(tmp_path, monkeypatch)

    import re

    def painted():
        z = np.load(out / "seg_masks.npz")
        return {k: int(z[k].sum()) for k in z.files if re.match(r"^f\d+_o\d+$", k)}

    before, npz_before = painted(), (out / "seg_masks.npz").read_bytes()
    _ctx_to(monkeypatch, main_mod, out)
    # the endpoint resolves the key by the raw id first (f0_o1 here, which covers the corner):
    # a REMOVE stroke there clears pixels whatever mask it lands in
    r = client.post("/api/segmentation/paint_mask", json={
        "session_id": "s", "instance_id": 1, "kf_index": 0, "x": 0.04, "y": 0.04, "radius": 3, "action": "remove"})
    assert r.status_code == 200, r.text
    after = painted()
    assert (out / "seg_masks.npz").read_bytes() != npz_before, "the store was rewritten"
    assert sum(after.values()) < sum(before.values()), "the stroke cleared pixels of frame 0"
    assert all(after[k] == before[k] for k in before if not k.startswith("f0_")), "frame 0 only"
    assert P.check_store_seal(out) == []
    assert json.loads((out / "segmentation.json").read_text())[STORE_EDITS_KEY] == {"mask_paint": 1}


# ── the interactive SAM3 runs on the scan's frozen configuration ────────────────────────────

def test_the_interactive_sam3_runs_on_the_sessions_frozen_configuration(tmp_path, main_mod):
    src = (SERVER_DIR / "main.py").read_text()
    for name in ("async def run_auto_segmentation(", "def _retro_process_active():"):
        i = src.index(name)
        body = src[i:i + 3000]
        j = body.index("run_segmentation(")
        call = body[j:body.index(")", body.index("boxes_map", j))]
        assert "config=config" in call, name
        assert body.index("_session_job_config(") < j and "_segmentation_config_source(" in body
    # the helper names the frozen file, or DECLARES the server's
    from intake.run_config import freeze_run_config, run_config_path
    session = tmp_path / "scan"
    frozen = {"segmentation": {"marker": 1}}
    freeze_run_config(session, frozen, log=lambda m: None)
    cfg, p = main_mod._session_job_config(session)
    assert cfg == frozen and p == run_config_path(session)
    assert str(p) in main_mod._segmentation_config_source("AutoSeg", p)
    assert "DECLARED" in main_mod._segmentation_config_source("AutoSeg", None)
