"""Stage 10 — ONE FINAL EPOCH (USER 2026-09-30: *"debe quedar una sola época que es la
final"*, docs/pipeline_final.md §10).

`certify.single_final_epoch` is a mandatory typed key. When on: the reconstruction
stage builds no Omega comparison cloud and drops `_epoch_0/` with the rest; the
certification, once its epoch is published and the chunk check ran, deletes every
`_epoch_<N>/` and `_tx_epoch_*/` and NOTHING else — the ledger, the per-epoch warps
and the live record stay — and the acta lists the discarded epochs.
`publish_in_flight` names a transaction directory still on disk so an operation that
rewrites the live cloud can refuse to race a publish."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction import apply as AP                                               # noqa: E402
from correction.epoch import EPOCH_FILE, LEDGER_FILE                             # noqa: E402
from correction.ledger import EPOCH_NPZ_DIR, save_epoch_npz                      # noqa: E402
from reconstruction.loops.config import LoopsConfigError, load_loops_config      # noqa: E402

SERVER = Path(__file__).resolve().parents[1]


def _raw():
    with open(SERVER / "config.yaml") as f:
        return yaml.safe_load(f)


# ── the key ──────────────────────────────────────────────────────────────

def test_the_key_is_mandatory_typed_and_on_in_production():
    raw = _raw()
    assert load_loops_config(raw).certify.single_final_epoch is True
    missing = copy.deepcopy(raw)
    del missing["certify"]["single_final_epoch"]
    with pytest.raises(LoopsConfigError, match="certify.single_final_epoch"):
        load_loops_config(missing)
    wrong = copy.deepcopy(raw)
    wrong["certify"]["single_final_epoch"] = "yes"
    with pytest.raises(LoopsConfigError, match="certify.single_final_epoch must be true/false"):
        load_loops_config(wrong)


# ── keep_only_live_epoch on a synthetic output dir ───────────────────────

def _epoch_dir(out: Path, epoch: int, with_manifest=True):
    d = out / f"{AP.PREV_PREFIX}{epoch}"
    d.mkdir()
    (d / "cleaned_cloud.ply").write_bytes(b"ply\n")
    if with_manifest:
        (d / AP.MANIFEST_NAME).write_text(json.dumps(
            {"epoch": epoch, "epoch_from": epoch, "epoch_to": epoch + 1,
             "artifacts": [{"rel": "cleaned_cloud.ply", "existed_before": True}]}))
    return d


def _session(tmp_path: Path, live: int = 3):
    out = tmp_path / "output"
    out.mkdir()
    for e in range(live):
        _epoch_dir(out, e)
    (out / f"{AP.TX_PREFIX}{live + 1}").mkdir()                    # a crashed publish
    (out / f"{AP.TX_PREFIX}{live + 1}" / "cleaned_cloud.ply").write_bytes(b"ply\n")
    (out / EPOCH_FILE).write_text(json.dumps({"epoch": live, "parent_epoch": live - 1,
                                              "kind": "transform"}))
    (out / LEDGER_FILE).write_text("\n".join(json.dumps(
        {"type": "run", "correction_id": f"c{e}", "epoch_from": e - 1, "epoch_to": e,
         "kind": "certify", "verdict": "applied"}) for e in range(1, live + 1)) + "\n")
    frames = [0, 10, 20]
    for e in range(1, live + 1):
        save_epoch_npz(out, e, np.tile(np.eye(3), (3, 1, 1)), np.zeros((3, 3)), np.ones(3), frames)
    (out / "cleaned_cloud.ply").write_bytes(b"live\n")
    (out / "potree").mkdir()
    (out / "potree" / "metadata.json").write_text("{}")
    (out / "_epoch_notes.txt").write_text("a FILE whose name looks like an epoch dir")
    return out


def test_keep_only_live_epoch_deletes_the_epoch_dirs_and_nothing_else(tmp_path):
    out = _session(tmp_path, live=3)
    before = {p.relative_to(out) for p in out.rglob("*")}
    msgs = []
    discarded = AP.keep_only_live_epoch(out, log=msgs.append)
    assert discarded == [0, 1, 2]
    assert not any(out.glob(f"{AP.PREV_PREFIX}*/")) or all(
        not p.is_dir() for p in out.glob(f"{AP.PREV_PREFIX}*"))
    assert not list(out.glob(f"{AP.TX_PREFIX}*"))
    after = {p.relative_to(out) for p in out.rglob("*")}
    gone = before - after
    assert all(str(p).startswith((AP.PREV_PREFIX, AP.TX_PREFIX)) for p in gone), gone
    # what stays: the live record, the ledger, every warp, the cloud, the octree, the file
    for rel in (EPOCH_FILE, LEDGER_FILE, "cleaned_cloud.ply", "potree/metadata.json", "_epoch_notes.txt",
                f"{EPOCH_NPZ_DIR}/epoch_1.npz", f"{EPOCH_NPZ_DIR}/epoch_2.npz", f"{EPOCH_NPZ_DIR}/epoch_3.npz"):
        assert (out / rel).exists(), rel
    assert json.loads((out / EPOCH_FILE).read_text())["epoch"] == 3
    # only the live epoch is left to select, and numbers are never recycled
    eps = AP.available_epochs(out)
    assert [(e["epoch"], e["live"]) for e in eps] == [(3, True)]
    assert AP.next_epoch(out) == 4
    assert any("[0, 1, 2]" in m for m in msgs)
    # idempotent: nothing left, nothing deleted, declared
    assert AP.keep_only_live_epoch(out, log=lambda m: None) == []


def test_a_half_swapped_session_is_refused_and_untouched(tmp_path):
    out = _session(tmp_path, live=2)
    (out / AP.SWAP_JOURNAL).write_text("{}")
    with pytest.raises(RuntimeError, match="interrupted correction swap"):
        AP.keep_only_live_epoch(out, log=lambda m: None)
    assert (out / f"{AP.PREV_PREFIX}0").is_dir() and (out / f"{AP.PREV_PREFIX}1").is_dir()
    assert (out / f"{AP.TX_PREFIX}3").is_dir()


# ── publish_in_flight ────────────────────────────────────────────────────

def test_publish_in_flight_names_a_transaction_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    assert AP.publish_in_flight(out) is None
    (out / "_tx_swap_journal.json").write_text("{}")             # a file, not a transaction dir
    (out / "_epoch_0").mkdir()
    assert AP.publish_in_flight(out) is None
    (out / "_tx_depth_on_f5").mkdir()                             # the bend's staging dir
    assert AP.publish_in_flight(out) == "_tx_depth_on_f5"
    (out / "_tx_epoch_4").mkdir()
    assert AP.publish_in_flight(out) in ("_tx_depth_on_f5", "_tx_epoch_4")
    (out / "_tx_depth_on_f5").rmdir()
    assert AP.publish_in_flight(out) == "_tx_epoch_4"
    (out / "_tx_epoch_4").rmdir()
    assert AP.publish_in_flight(out) is None


# ── the reconstruction stage under the key ───────────────────────────────

def test_discard_previous_epochs_drops_epoch_0_only_under_the_key(tmp_path):
    from workers.map_worker import _discard_previous_epochs
    for keep0 in (True, False):
        out = tmp_path / f"k{int(keep0)}" / "output"
        out.mkdir(parents=True)
        for e in (0, 1, 2):
            _epoch_dir(out, e)
        (out / "_tx_epoch_3").mkdir()
        (out / "chunk_000.ply").write_bytes(b"ply\n")
        (out / LEDGER_FILE).write_text("")
        save_epoch_npz(out, 1, np.tile(np.eye(3), (1, 1, 1)), np.zeros((1, 3)), np.ones(1), [0])
        (out / EPOCH_FILE).write_text(json.dumps({"epoch": 3}))
        _discard_previous_epochs(out, keep_epoch0=keep0)
        assert (out / "_epoch_0").is_dir() == keep0
        assert not (out / "_epoch_1").exists() and not (out / "_epoch_2").exists()
        assert not (out / "_tx_epoch_3").exists() and not (out / "chunk_000.ply").exists()
        assert (out / LEDGER_FILE).exists() and (out / EPOCH_NPZ_DIR / "epoch_1.npz").exists()
        assert (out / EPOCH_FILE).exists()


def test_the_map_worker_builds_no_comparison_cloud_under_the_key():
    import inspect
    import workers.map_worker as M
    src = inspect.getsource(M._run_precision_core)
    assert "single_final_epoch" in src and "build_epoch0_cloud" in src
    # the comparison cloud is built on the OTHER branch of the key, never unconditionally
    assert "if single_final:" in src and "keep_epoch0=not single_final" in src
    sig = inspect.signature(M._discard_previous_epochs)
    assert "keep_epoch0" in sig.parameters


def test_the_certification_discards_after_the_check_and_before_the_acta():
    src = (SERVER / "reconstruction" / "certify" / "run.py").read_text()
    assert 'getattr(ccert, "single_final_epoch", False)' in src
    i_check = src.index("run_check(session_dir, _pcfg, log=log)")
    i_keep = src.index("keep_only_live_epoch(output_dir, log=log)")
    i_acta = src.index("(output_dir / ACTA_JSON).write_text")
    assert i_check < i_keep < i_acta
    assert 'acta["epochs_discarded"]' in src


# ── end to end: a floor correction, then one epoch left ──────────────────

def test_after_a_real_correction_only_the_live_epoch_remains(tmp_path):
    """The transactional floor run stores epoch 0 in `_epoch_0/` (test_correction_tx
    pins that); with the key the stored epoch goes and what makes the epoch
    reproducible — the ledger and the warp — stays."""
    from correction.run import run_floor
    from tests.synth_correction import build_scene, make_correction_cfg
    scene = build_scene(tmp_path, floor="ramp", floor_slope=0.04, drift_yaw_deg=0.0,
                        drift_t=(0.0, 0.10, 0.0))
    rep = run_floor(scene.output_dir, "plane", None, "test", cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
    out = scene.output_dir
    assert (out / "_epoch_0").is_dir()
    assert AP.keep_only_live_epoch(out, log=lambda m: None) == [0]
    assert not (out / "_epoch_0").exists()
    assert json.loads((out / EPOCH_FILE).read_text())["epoch"] == 1
    assert (out / EPOCH_NPZ_DIR / "epoch_1.npz").exists()
    from correction.ledger import applied_runs
    assert [r["epoch_to"] for r in applied_runs(out)] == [1]
    assert [(e["epoch"], e["live"]) for e in AP.available_epochs(out)] == [(1, True)]
    assert AP.next_epoch(out) == 2
    with pytest.raises(RuntimeError, match="not in this session"):
        AP.select_epoch(out, 0, log=lambda m: None)


def test_a_deliverable_only_certification_leaves_one_epoch_and_declares_it(tmp_path):
    """The pipeline's own path: the certification (deliverable_only) publishes its
    epoch, runs the chunk check (declared, not run on a synthetic session without
    Omega records), and under the key leaves ONE epoch on disk with the acta
    naming what it discarded."""
    from tests.synth_metric import corridor_loop_scene, loop_trajectory, make_session
    from tests.test_certify_f3 import N_KF, _cfg, _drift, _run, _write
    sess = make_session(H=40, W=56, scene=corridor_loop_scene(), poses=loop_trajectory(N_KF, extra_laps=0.15))
    root = _write(tmp_path / "s", sess, _drift(sess))
    out = root / "output"
    cfg = _cfg(**{"certify.single_final_epoch": True, "certify.deliverable_only": True})
    assert cfg.certify.single_final_epoch and cfg.certify.deliverable_only
    acta = _run(root, sess, cfg)
    final = int(acta["epoch_final"])
    assert final == acta["epoch_after_correction"] >= 1, acta["correction"]
    assert acta["epochs_discarded"] == list(range(0, final)), acta["epochs_discarded"]
    assert not [d for d in out.glob("_epoch_*") if d.is_dir()]
    assert not list(out.glob("_tx_epoch_*"))
    assert json.loads((out / EPOCH_FILE).read_text())["epoch"] == final
    for e in range(1, final + 1):
        assert (out / EPOCH_NPZ_DIR / f"epoch_{e}.npz").exists(), e
    assert (out / LEDGER_FILE).exists() and (out / "cleaned_cloud.ply").exists()
    assert [(e["epoch"], e["live"]) for e in AP.available_epochs(out)] == [(final, True)]
    saved = json.loads((out / "certify_acta.json").read_text())
    assert saved["epochs_discarded"] == acta["epochs_discarded"]
