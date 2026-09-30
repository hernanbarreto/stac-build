"""Selecting an epoch across a FUSED (new-cloud) epoch — pccr 2026-09-29.

"Show epoch 0" failed with *"epoch 3 has no persisted transform, so the
instance store cannot follow the geometry to epoch 0"*. Epoch 3 was published
by F7 (`precision/fuse.py`): a cloud REBUILT from the depth maps under the same
cameras, not a per-keyframe warp of epoch 2 — so no `epoch_3.npz` can exist and
none is needed: the store follows a new-cloud edge with the identity and refits
its points from the swapped cloud.

Behind the gate lay a second defect: the stored directories are DELTAS (on
pccr `_epoch_1/` holds poses and sidecar only, `_epoch_2/` the cloud, octree
and segmentation only) and `select_epoch` restored files from the target's
directory alone, so any target but 0 would have shown one epoch's cloud under
another epoch's poses. The fixture here is pccr's exact layout.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction import run as run_mod                       # noqa: E402
from correction.apply import (MANIFEST_NAME, PREV_PREFIX,   # noqa: E402
                              available_epochs, select_epoch)
from correction.epoch import (EPOCH_FILE, EPOCH_KIND_NEW_CLOUD,  # noqa: E402
                              EPOCH_KIND_TRANSFORM, epoch_kind,
                              make_epoch_record)
from correction.ledger import save_epoch_npz                # noqa: E402
from correction.run import compose_moves, run_select        # noqa: E402

FRAMES = [10, 20, 30]


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _move(seed):
    """A per-keyframe warp with every degree of freedom non-trivial."""
    rng = np.random.default_rng(seed)
    R = np.stack([_rot(rng.normal(size=3), rng.uniform(1, 5)) for _ in FRAMES])
    t = rng.normal(scale=0.3, size=(len(FRAMES), 3))
    k = rng.uniform(0.9, 1.1, size=len(FRAMES))
    b = rng.normal(scale=0.02, size=len(FRAMES))
    return {"R_kf": R, "t_kf": t, "k_kf": k, "b_kf": b, "frames": list(FRAMES)}


MOVE1, MOVE2 = _move(1), _move(2)


def _manifest(d: Path, epoch: int, epoch_to: int, arts):
    d.mkdir(parents=True, exist_ok=True)
    (d / MANIFEST_NAME).write_text(json.dumps(
        {"epoch": epoch, "epoch_from": epoch, "epoch_to": epoch_to,
         "artifacts": [{"rel": r, "existed_before": ex} for r, ex in arts]}))


def _write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _record(root: Path, epoch: int, parent: int, kind=None):
    rec = make_epoch_record(epoch, f"cid{epoch}", parent,
                            **({"kind": kind} if kind else {}))
    if kind is None:
        rec.pop("kind")             # a record written before `kind` existed
    (root / EPOCH_FILE).write_text(json.dumps(rec))


def _read(root: Path, rel: str):
    p = root / rel
    return p.read_text() if p.exists() else None


@pytest.fixture()
def pccr(tmp_path):
    """pccr's layout on 2026-09-29, live at epoch 3 (the fused cloud):

        epoch 0  cloud A, poses 0, segmentation 0, octree 0
        epoch 1  gauge  — poses 1 + depth sidecar (transform, epoch_1.npz)
        epoch 2  refine — poses 2 + depth sidecar (transform, epoch_2.npz)
        epoch 3  fuse   — cloud B, raw twin, fuse report, pending segmentation,
                          octree 3; NO npz (kind new_cloud)

    `_epoch_<j>/` holds exactly what the departure from j replaced."""
    out = tmp_path / "output"
    out.mkdir()
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in FRAMES))
    # live: epoch 3
    _write(out, "cleaned_cloud.ply", "cloud-B")
    _write(out, "cleaned_cloud_raw.ply", "raw-B")
    _write(out, "fuse_report.json", "{}")
    _write(out, "segmentation_result.json", "seg-pending-3")
    _write(out, "camera_poses.txt", "poses-2")
    _write(out, "depth_correction.json", "depth-2")
    _write(out, "potree/metadata.json", "potree-3")
    _record(out, 3, 2, EPOCH_KIND_NEW_CLOUD)
    # _epoch_0: what epoch 1 replaced (and introduced)
    d0 = out / f"{PREV_PREFIX}0"
    _manifest(d0, 0, 1, [("cleaned_cloud.ply", True), ("camera_poses.txt", True),
                         ("segmentation_result.json", True),
                         ("depth_correction.json", False), (EPOCH_FILE, False),
                         ("corrections/epoch_1.npz", False), ("potree", True)])
    _write(d0, "cleaned_cloud.ply", "cloud-A")
    _write(d0, "camera_poses.txt", "poses-0")
    _write(d0, "segmentation_result.json", "seg-0")
    _write(d0, "potree/metadata.json", "potree-0")
    # _epoch_1: what epoch 2 replaced — poses and sidecar only
    d1 = out / f"{PREV_PREFIX}1"
    _manifest(d1, 1, 2, [("camera_poses.txt", True), ("depth_correction.json", True),
                         (EPOCH_FILE, True), ("corrections/epoch_2.npz", False)])
    _write(d1, "camera_poses.txt", "poses-1")
    _write(d1, "depth_correction.json", "depth-1")
    _record(d1, 1, 0, EPOCH_KIND_TRANSFORM)
    # _epoch_2: what the fusion replaced — cloud, segmentation, octree only
    d2 = out / f"{PREV_PREFIX}2"
    _manifest(d2, 2, 3, [("cleaned_cloud.ply", True), ("cleaned_cloud_raw.ply", False),
                         ("fuse_report.json", False), ("segmentation_result.json", True),
                         (EPOCH_FILE, True), ("potree", True)])
    _write(d2, "cleaned_cloud.ply", "cloud-A")
    _write(d2, "segmentation_result.json", "seg-0")
    _write(d2, "potree/metadata.json", "potree-0")
    _record(d2, 2, 1, EPOCH_KIND_TRANSFORM)
    for e, mv in ((1, MOVE1), (2, MOVE2)):
        save_epoch_npz(out, e, mv["R_kf"], mv["t_kf"], mv["k_kf"], mv["frames"],
                       b_kf=mv["b_kf"])
    return out


@pytest.fixture()
def store_calls(monkeypatch):
    """`update_instance_store` recorded, never run (it needs a real cloud)."""
    calls = []

    def fake(output_dir, R, t, k, frames, log=print, b_kf=None):
        calls.append({"R": np.array(R), "t": np.array(t), "k": np.array(k),
                      "b": None if b_kf is None else np.array(b_kf),
                      "frames": list(frames)})
        return {"store": "updated"}

    monkeypatch.setattr(run_mod, "update_instance_store", fake)
    return calls


def _state(out: Path) -> dict:
    return {rel: _read(out, rel) for rel in (
        "cleaned_cloud.ply", "cleaned_cloud_raw.ply", "fuse_report.json",
        "segmentation_result.json", "camera_poses.txt", "depth_correction.json",
        "potree/metadata.json")}


STATE = {
    0: {"cleaned_cloud.ply": "cloud-A", "cleaned_cloud_raw.ply": None,
        "fuse_report.json": None, "segmentation_result.json": "seg-0",
        "camera_poses.txt": "poses-0", "depth_correction.json": None,
        "potree/metadata.json": "potree-0"},
    1: {"cleaned_cloud.ply": "cloud-A", "cleaned_cloud_raw.ply": None,
        "fuse_report.json": None, "segmentation_result.json": "seg-0",
        "camera_poses.txt": "poses-1", "depth_correction.json": "depth-1",
        "potree/metadata.json": "potree-0"},
    2: {"cleaned_cloud.ply": "cloud-A", "cleaned_cloud_raw.ply": None,
        "fuse_report.json": None, "segmentation_result.json": "seg-0",
        "camera_poses.txt": "poses-2", "depth_correction.json": "depth-2",
        "potree/metadata.json": "potree-0"},
    3: {"cleaned_cloud.ply": "cloud-B", "cleaned_cloud_raw.ply": "raw-B",
        "fuse_report.json": "{}", "segmentation_result.json": "seg-pending-3",
        "camera_poses.txt": "poses-2", "depth_correction.json": "depth-2",
        "potree/metadata.json": "potree-3"},
}


def _identity(call):
    n = len(call["frames"])
    assert np.allclose(call["R"], np.tile(np.eye(3), (n, 1, 1)))
    assert np.allclose(call["t"], 0) and np.allclose(call["k"], 1)
    assert call["b"] is not None and np.allclose(call["b"], 0)


def _inverse_of(call, fwd):
    """`call` undoes `fwd` exactly: composing the two is the identity."""
    n = len(call["frames"])
    for i in range(n):
        assert np.allclose(call["R"][i] @ fwd["R"][i], np.eye(3), atol=1e-12)
        assert np.allclose(call["R"][i] @ fwd["t"][i] + call["t"][i], 0, atol=1e-12)
        assert np.isclose(call["k"][i] * fwd["k"][i], 1.0)
        # z'' = k_c (k_f z + b_f) + b_c must be z
        assert np.isclose(call["k"][i] * fwd["b"][i] + call["b"][i], 0.0, atol=1e-12)


def _forward_1_then_2():
    R, t, k, b = compose_moves([(MOVE1, False), (MOVE2, False)], FRAMES,
                               log=lambda m: None)
    return {"R": R, "t": t, "k": k, "b": b, "frames": FRAMES}


# ── the failure that was reported ─────────────────────────────────────────

def test_show_epoch_0_across_the_fused_epoch(pccr, store_calls):
    res = run_select(pccr, 0, "test", log=lambda m: None)
    assert res["ok"] and res["epoch"] == 0 and res["changed"]
    assert _state(pccr) == STATE[0]
    assert not (pccr / EPOCH_FILE).exists()          # epoch 0 has no record
    assert [e["epoch"] for e in available_epochs(pccr)] == [0, 1, 2, 3]
    # the store followed ONCE, with epochs 2 and 1 undone (3 is the identity)
    assert len(store_calls) == 1
    assert store_calls[0]["frames"] == FRAMES
    _inverse_of(store_calls[0], _forward_1_then_2())
    # the transforms never travel
    assert (pccr / "corrections" / "epoch_1.npz").exists()
    assert (pccr / "corrections" / "epoch_2.npz").exists()


def test_back_up_to_the_fused_cloud(pccr, store_calls):
    run_select(pccr, 0, "test", log=lambda m: None)
    res = run_select(pccr, 3, "test", log=lambda m: None)
    assert res["epoch"] == 3
    assert _state(pccr) == STATE[3]
    assert json.loads((pccr / EPOCH_FILE).read_text())["epoch"] == 3
    assert len(store_calls) == 2
    fwd = _forward_1_then_2()
    for key in ("R", "t", "k", "b"):
        assert np.allclose(store_calls[1][key], fwd[key])


# ── the delta directories: every target, not only 0 ───────────────────────

def test_epoch_1_gets_cloud_A_with_its_own_poses(pccr, store_calls):
    """Its directory holds poses and sidecar only; the cloud it had is the one
    epoch 2 still had, stored where the fusion replaced it (`_epoch_2/`)."""
    run_select(pccr, 1, "test", log=lambda m: None)
    assert _state(pccr) == STATE[1]
    assert json.loads((pccr / EPOCH_FILE).read_text())["epoch"] == 1
    assert len(store_calls) == 1
    _inverse_of(store_calls[0], {"R": MOVE2["R_kf"], "t": MOVE2["t_kf"],
                                 "k": MOVE2["k_kf"], "b": MOVE2["b_kf"],
                                 "frames": FRAMES})
    # `_epoch_2/` gave its cloud away and says so; the fused state is filed
    man2 = json.loads((pccr / f"{PREV_PREFIX}2" / MANIFEST_NAME).read_text())
    cloud_entry = [a for a in man2["artifacts"] if a["rel"] == "cleaned_cloud.ply"][0]
    assert cloud_entry["same_as_epoch"] == 1
    assert not (pccr / f"{PREV_PREFIX}2" / "cleaned_cloud.ply").exists()
    assert _read(pccr / f"{PREV_PREFIX}3", "cleaned_cloud.ply") == "cloud-B"
    assert _read(pccr / f"{PREV_PREFIX}3", "camera_poses.txt") == "poses-2"


def test_epoch_2_is_the_identity_edge_alone(pccr, store_calls):
    run_select(pccr, 2, "test", log=lambda m: None)
    assert _state(pccr) == STATE[2]
    assert len(store_calls) == 1
    _identity(store_calls[0])
    assert store_calls[0]["frames"] == FRAMES


def test_every_epoch_survives_a_full_tour(pccr, store_calls):
    tour = [0, 3, 1, 2, 0, 2, 3, 1, 0]
    for n, e in enumerate(tour, start=1):
        run_select(pccr, e, "test", log=lambda m: None)
        assert _state(pccr) == STATE[e], f"step {n}: epoch {e}"
        rec = json.loads((pccr / EPOCH_FILE).read_text())["epoch"] if e else 0
        assert rec == e
        assert [x["epoch"] for x in available_epochs(pccr)] == [0, 1, 2, 3]
        assert len(store_calls) == n
    # the pointers left behind never break a stored epoch: each still holds
    # every file its manifest claims
    for d in pccr.glob(f"{PREV_PREFIX}*"):
        man = json.loads((d / MANIFEST_NAME).read_text())
        for a in man["artifacts"]:
            if a.get("existed_before") and not a["rel"].startswith("corrections/"):
                assert (d / a["rel"]).exists(), f"{d.name}/{a['rel']} missing"


def test_a_transform_epoch_without_its_npz_still_selects_and_says_so(pccr, store_calls):
    """pccr 2026-09-30 ("¿por qué no puedo ver la época 0?"): a precision run deleted
    its poses-only epochs' transforms; the geometry is selected anyway — the transform
    only carried the store's finding anchors — and the log says what did not follow."""
    (pccr / "corrections" / "epoch_2.npz").unlink()
    logs = []
    run_select(pccr, 0, "test", log=logs.append)
    assert _state(pccr) == STATE[0]
    assert any("epoch 2 has no persisted transform" in m for m in logs)


# ── the kind: record, then ledger, then transform ─────────────────────────

def test_epoch_kind_reads_the_record(pccr):
    assert epoch_kind(pccr, 3) == EPOCH_KIND_NEW_CLOUD
    assert epoch_kind(pccr, 2) == EPOCH_KIND_TRANSFORM
    assert epoch_kind(pccr, 0) == EPOCH_KIND_TRANSFORM
    assert make_epoch_record(5, "x", 4)["kind"] == EPOCH_KIND_TRANSFORM
    with pytest.raises(ValueError):
        make_epoch_record(5, "x", 4, kind="rebuilt")


def test_a_record_without_kind_is_classified_from_the_ledger(pccr, store_calls):
    """pccr's epoch 3 was published before `kind` existed: the record says
    nothing, the ledger says the run was a `fuse`."""
    _record(pccr, 3, 2, None)
    assert epoch_kind(pccr, 3) == EPOCH_KIND_TRANSFORM       # no ledger yet
    (pccr / "corrections.jsonl").write_text(json.dumps(
        {"type": "run", "correction_id": "80f8f537", "epoch_from": 2,
         "epoch_to": 3, "kind": "fuse", "verdict": "applied"}) + "\n")
    assert epoch_kind(pccr, 3) == EPOCH_KIND_NEW_CLOUD
    run_select(pccr, 0, "test", log=lambda m: None)
    assert _state(pccr) == STATE[0]
    assert epoch_kind(pccr, 3) == EPOCH_KIND_NEW_CLOUD       # stored record now


def test_a_record_without_kind_and_no_ledger_is_a_transform(pccr, store_calls):
    _record(pccr, 3, 2, None)
    assert epoch_kind(pccr, 3) == EPOCH_KIND_TRANSFORM
    logs = []
    run_select(pccr, 0, "test", log=logs.append)
    assert any("epoch 3 has no persisted transform" in m for m in logs)


# ── the swap stays atomic ─────────────────────────────────────────────────

def _files(root: Path) -> dict:
    """Every file under the session, byte for byte (the octree lock file the
    cross-process lock creates is not session state)."""
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file() and p.name != ".potree.lock"}


def test_a_failure_mid_select_rolls_everything_back(pccr, monkeypatch):
    before = _state(pccr)
    stored_before = _files(pccr)
    real_rename = Path.rename
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("disk went away")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        select_epoch(pccr, 1, log=lambda m: None)
    monkeypatch.setattr(Path, "rename", real_rename)
    assert _state(pccr) == before
    assert not (pccr / "_tx_swap_journal.json").exists()
    assert not (pccr / f"{PREV_PREFIX}3").exists()
    assert _files(pccr) == stored_before


def test_replay_refuses_to_cross_a_new_cloud_epoch(pccr):
    from correction.replay import replay
    with pytest.raises(RuntimeError, match="epoch 3 is a NEW CLOUD"):
        replay(pccr, 3, log=lambda m: None)
    assert not (pccr / "replay").exists()
