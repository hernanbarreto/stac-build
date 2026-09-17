"""claude_stac.txt F3 — §12.6 witnesses: injected floaters end
``single_witness``, a SAM3 mask leak ends ``mask_conflict``, nothing is
deleted, the counts land in the report; §12.9 provenance: every field
survives the witness epoch and the segmentation's globalIndices stay valid;
§12.10 config validation of the witness section."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops.config import LoopsConfigError, load_loops_config      # noqa: E402
from reconstruction.witness.fields import read_fields                              # noqa: E402
from reconstruction.witness.frames import frames_from_arrays                       # noqa: E402
from reconstruction.witness.run import run_witnesses                               # noqa: E402
from reconstruction.witness.status import STATUS_CODES                             # noqa: E402
from tests.synth_metric import (make_session, write_session_dir, write_aligned_chunks,  # noqa: E402
                                raw_server_cfg, inject_floaters, leak_mask,
                                corridor_loop_scene, loop_trajectory)
from tests.synth_correction import make_correction_cfg                             # noqa: E402

N_KF = 48


def _instances(sess):
    """Walls, columns and boxes are SAM3 instances; the floor and the ceiling
    stay unsegmented (where the floaters are injected: a point of no
    instance has no mask witness, only the multi-view one)."""
    prims = sess.scene.prims
    inst, iid = {}, 1
    for p in prims:
        lab = getattr(p, "label", "")
        if lab in ("wall", "column", "box"):
            inst[iid] = {"label": lab, "oids": [p.oid]}
            iid += 1
    return inst


def _unsegmented_oids(sess):
    return [p.oid for p in sess.scene.prims if getattr(p, "label", "") in ("floor", "ceiling")]


def _cfg(**over):
    base = {"witness.mask_erosion_px": 1}
    base.update(over)
    return load_loops_config(raw_server_cfg(**base))


def _frames(sess):
    return frames_from_arrays(sess.depth, sess.K, sess.poses, sess.frame_numbers)


@pytest.fixture(scope="module")
def base_session():
    return make_session(H=40, W=56, scene=corridor_loop_scene(),
                        poses=loop_trajectory(N_KF, extra_laps=0.0))


def test_floaters_single_witness_leak_mask_conflict_nothing_deleted(tmp_path, base_session):
    sess, picks = inject_floaters(base_session, n=150, seed=3, oids=_unsegmented_oids(base_session))
    inst = _instances(sess)
    root = write_session_dir(tmp_path / "s", sess, inst, point_stride=1)
    write_aligned_chunks(root, sess, chunk_size=24, overlap=12)
    out = root / "output"
    # a wall's mask leaks over one column's pixels in every frame
    labels = {iid: v["label"] for iid, v in inst.items()}
    wall = next(i for i, l in labels.items() if l == "wall")
    cols = [i for i, l in labels.items() if l == "column"]
    res = json.loads((out / "segmentation_result.json").read_text())["instances"]
    by_id = {int(r["instance_id"]): r for r in res}
    # the column the wall is most often seen with
    masks = np.load(out / "seg_masks.npz")
    def _n_shared(c):
        return sum(1 for k in masks.files if k.endswith(f"_o{wall - 1}")
                   and (k[:k.index('_o')] + f"_o{c - 1}") in masks.files)
    victim = max(cols, key=_n_shared)
    control = max((c for c in cols if c != victim), key=lambda c: by_id[c]["total_points"])
    leaked = leak_mask(out, wall, victim, px=40)
    assert leaked > 0
    cfg = _cfg()
    n_before = len(np.load(out / "seg_masks.npz").files)
    rep = run_witnesses(out, cfg, frames=_frames(sess), device="cpu", log=lambda m: None, epoch=False)
    f = read_fields(out / "cleaned_cloud.ply")
    from correction.session import read_ply
    _, data = read_ply(out / "cleaned_cloud.ply")
    n = len(data)
    # nothing deleted: every point still there, provenance intact
    assert rep["n_points"] == n == sum(rep["status_counts"].values())
    assert set(data.dtype.names) >= {"frame_global", "pixel_row", "pixel_col", "mv_votes",
                                     "mask_votes", "mask_conflicts", "status"}
    # the floaters: their pixel's depth disagrees with every neighbour
    fg, pr, pc = data["frame_global"], data["pixel_row"], data["pixel_col"]
    key = {(int(sess.frame_numbers[g]), int(r), int(c)): k for k, (g, r, c) in enumerate(picks)}
    idx = [i for i in range(n) if (int(fg[i]), int(pr[i]), int(pc[i])) in key]
    assert len(idx) == len(picks)
    st = f["status"][idx]
    frac_single = float(np.mean(st == STATUS_CODES["single_witness"]))
    assert frac_single > 0.9, frac_single
    assert not np.any(st == STATUS_CODES["verified"])
    # the leaked column: its points land in the wall's mask, not in its own
    gv = np.asarray(by_id[victim]["globalIndices"], np.int64)
    stv = f["status"][gv]
    assert float(np.mean(stv == STATUS_CODES["mask_conflict"])) > 0.9, rep["status_counts"]
    assert np.all(f["mask_conflicts"][gv] >= f["mask_votes"][gv])
    # the control column keeps its own witnesses (a thin cylinder a few
    # pixels wide: its silhouette pixels blend with the background in the
    # neighbours' depth and lose the vote — the body is verified, and no
    # view ever puts it in another instance's mask)
    gc = np.asarray(by_id[control]["globalIndices"], np.int64)
    verified_control = float(np.mean(f["status"][gc] == STATUS_CODES["verified"]))
    assert verified_control > 0.6, verified_control
    assert verified_control > 3 * float(np.mean(stv == STATUS_CODES["verified"]))
    assert int(f["mask_conflicts"][gc].sum()) == 0
    # the bulk of an exact synthetic cloud is verified; the acta has the counts
    assert rep["status_fraction"]["verified"] > 0.7, rep["status_fraction"]
    assert rep["mask_conflicts_total"] >= len(gv)
    assert (out / "witness_report.json").exists()
    assert len(np.load(out / "seg_masks.npz").files) == n_before
    # the raw cloud carries the same fields
    assert set(read_fields(out / "cleaned_cloud.ply")) == set(f)


def test_witness_epoch_keeps_provenance_and_global_indices(tmp_path, base_session):
    sess = base_session
    inst = _instances(sess)
    root = write_session_dir(tmp_path / "s", sess, inst, point_stride=2)
    write_aligned_chunks(root, sess, chunk_size=24, overlap=12)
    out = root / "output"
    from correction.session import read_ply
    from correction.session import read_poses
    _, before = read_ply(out / "cleaned_cloud.ply")
    poses_before = read_poses(out / "camera_poses.txt")
    cfg = _cfg()
    rep = run_witnesses(out, cfg, frames=_frames(sess), device="cpu", log=lambda m: None,
                        epoch=True, operator="test", correction_cfg=make_correction_cfg())
    assert rep["epoch"] == 1
    _, after = read_ply(out / "cleaned_cloud.ply")
    assert len(after) == len(before)
    for fld in ("x", "y", "z", "frame_global", "pixel_row", "pixel_col", "confidence"):
        assert np.array_equal(after[fld], before[fld]), fld
    assert np.array_equal(read_poses(out / "camera_poses.txt"), poses_before)
    res = json.loads((out / "segmentation_result.json").read_text())["instances"]
    for r in res:
        gi = np.asarray(r["globalIndices"], np.int64)
        if len(gi) == 0:
            continue                      # an object the walk never saw
        assert gi.min() >= 0 and gi.max() < len(after)
        assert np.all(after["frame_global"][gi] == before["frame_global"][gi])
    # the epoch is real: ledger record, and the previous epoch kept on disk
    ledger = [json.loads(l) for l in (out / "corrections.jsonl").read_text().splitlines()]
    assert ledger[-1]["kind"] == "witness" and ledger[-1]["verdict"] == "applied"
    assert (out / "_epoch_0").exists()
    assert json.loads((out / "geometry_epoch.json").read_text())["epoch"] == 1
    # selecting epoch 0 shows the cloud as it was, and epoch 1 is NOT destroyed
    from correction.run import run_select
    from correction.apply import available_epochs
    res_v = run_select(out, 0, "test")
    assert res_v["epoch"] == 0 and res_v["changed"]
    assert [e["epoch"] for e in available_epochs(out)] == [0, 1]
    _, restored = read_ply(out / "cleaned_cloud.ply")
    assert restored.dtype.names == before.dtype.names
    assert np.array_equal(restored["x"], before["x"])


def test_witness_config_validation():
    with pytest.raises(LoopsConfigError, match="witness.rules.verified_min_mv_votes"):
        load_loops_config(raw_server_cfg(**{"witness.rules.verified_min_mv_votes": 0}))
    with pytest.raises(LoopsConfigError, match="witness status"):
        load_loops_config(raw_server_cfg(**{"witness.clean_statuses": ["verified", "nope"]}))
    with pytest.raises(LoopsConfigError, match="certify.gates.max_verified_drop_frac"):
        load_loops_config(raw_server_cfg(**{"certify.gates.max_verified_drop_frac": 2.0}))
    with pytest.raises(LoopsConfigError, match="certify.envelope.loop_densities"):
        load_loops_config(raw_server_cfg(**{"certify.envelope.loop_densities": [2.0]}))
    with pytest.raises(LoopsConfigError, match="witness.tracks.python"):
        load_loops_config(raw_server_cfg(**{"witness.tracks.python": None}))
