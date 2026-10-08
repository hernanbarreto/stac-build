"""docs/plan_determinismo.md, package C — the compared artifacts of the F0–F4 chain carry no
random id and no wall clock (points 36 / 45): correction ids derive from what the correction
is, poses are written float64 round-trip exact, timings live in sibling files, the ledger and
epoch record are identical across two runs of the same chain; evidence written by hand is
stamped with the reconstruction it was measured on (point 34)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import repro


def _session(root: Path, n: int = 5) -> Path:
    out = root / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    frames = [10 * i for i in range(n)]
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    rng = np.random.default_rng(0)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, 3] = rng.uniform(-3, 3, (n, 3)) / 7.0            # not representable in 9 decimals
    from correction.session import write_poses
    write_poses(out / "camera_poses.txt", poses)
    write_poses(out / "omega_run" / "camera_poses.txt", poses)
    for f in frames:
        np.savez(out / "omega_run" / "results_output" / f"frame_{f}.npz",
                 depth=np.full((4, 6), 2.0, np.float32), conf=np.ones((4, 6), np.float32))
    (out / "vggt_omega_config.yaml").write_text("Model: {omega_mode: balanced, omega_resolution: 512}\n")
    (out / ".metric_scale_applied").write_text("scale=1.0\n")
    return out


def test_write_poses_round_trips_bit_exact_through_every_reader(tmp_path):
    """Plan point 45: correction.session.write_poses (the writer of F2's and F5's epochs) writes
    float64 that read_poses and np.loadtxt read back bit for bit."""
    from correction.session import read_poses, write_poses
    rng = np.random.default_rng(1)
    poses = rng.standard_normal((7, 4, 4)) * np.array([1e3, 1.0, 1e-3, 1e-9])
    poses[0, 0, 0] = 1.0 / 3.0
    poses[1, 1, 1] = np.nextafter(1.0, 2.0)
    p = tmp_path / "camera_poses.txt"
    write_poses(p, poses)
    back = read_poses(p)
    assert back.dtype == np.float64 and np.array_equal(back.view(np.uint64), poses.view(np.uint64))
    assert np.array_equal(np.loadtxt(p).reshape(-1, 4, 4).view(np.uint64), poses.view(np.uint64))
    # the same poses give the same bytes (no temp-file residue, no clock)
    write_poses(tmp_path / "again.txt", poses)
    assert (tmp_path / "again.txt").read_bytes() == p.read_bytes()
    assert not list(tmp_path.glob(".*tmp*"))


def test_an_epoch_of_poses_is_the_same_bytes_on_two_sessions_and_its_id_is_derived(tmp_path):
    """Plan point 36: two identical sessions → identical correction id, ledger line, epoch record,
    report and poses; the clock is in corrections.timing.jsonl only."""
    from correction.ledger import LEDGER_FILE, TIMING_FILE, read_ledger
    from precision.poses_epoch import apply_pose_epoch, load_poses
    outs = []
    for name in ("a", "b"):
        out = _session(tmp_path / name)
        frames, poses = load_poses(out)
        n = len(frames)
        R = np.tile(np.eye(3), (n, 1, 1))
        t = np.zeros((n, 3)); t[:, 2] = 1.0 / 3.0
        res = apply_pose_epoch(out, R, t, np.full(n, 1.01), "gauge",
                               [{"stage": "gauge", "instrument": "da3_windows"}], log=lambda *a: None)
        outs.append((out, res))
    (oa, ra), (ob, rb) = outs
    assert ra["correction_id"] == rb["correction_id"] and len(ra["correction_id"]) == 8
    for rel in ("camera_poses.txt", "omega_run/camera_poses.txt", "geometry_epoch.json",
                "depth_correction.json", LEDGER_FILE, f"corrections/report_{ra['correction_id']}.json",
                "corrections/epoch_1.npz"):
        assert (oa / rel).read_bytes() == (ob / rel).read_bytes(), rel
    rec = json.loads((oa / "geometry_epoch.json").read_text())
    assert "created_at" not in rec and rec["correction_id"] == ra["correction_id"]
    assert all("created_at" not in e for e in read_ledger(oa))
    timing = [json.loads(l) for l in (oa / TIMING_FILE).read_text().splitlines()]
    assert timing[0]["correction_id"] == ra["correction_id"] and timing[0]["created_at"]
    # another transform is another id; the id does not depend on when
    out_c = _session(tmp_path / "c")
    frames, poses = load_poses(out_c)
    n = len(frames)
    res_c = apply_pose_epoch(out_c, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), np.full(n, 1.01),
                             "gauge", [{"stage": "gauge", "instrument": "da3_windows"}], log=lambda *a: None)
    assert res_c["correction_id"] != ra["correction_id"]


def test_correction_ids_are_derived_from_their_parts():
    from correction.ledger import new_correction_id
    a = new_correction_id("poses_epoch", "gauge", 0, 1, [1, 2], np.ones(3))
    assert a == new_correction_id("poses_epoch", "gauge", 0, 1, [1, 2], np.ones(3)) and len(a) == 8
    assert a != new_correction_id("poses_epoch", "gauge", 0, 1, [1, 2], np.ones(3) * 2)
    assert a == repro.stable_id("correction", "poses_epoch", "gauge", 0, 1, [1, 2], np.ones(3), n_hex=8)


def test_the_reconstruction_id_names_omegas_output_and_nothing_later(tmp_path):
    """Plan point 34: the id every by-hand evidence file carries — the same for the same Omega
    output whatever epoch the session is in, another when one Omega record changes, none for a
    session without Omega records."""
    from correction.epoch import (RECONSTRUCTION_ID_KEY, ReconstructionIdError, reconstruction_id,
                                  reconstruction_id_or_none, same_reconstruction)
    out = _session(tmp_path / "s")
    rid = reconstruction_id(out)
    assert len(rid) == 64 and reconstruction_id_or_none(out) == rid
    # an epoch moves the poses: the reconstruction is still the same one
    from precision.poses_epoch import apply_pose_epoch, load_poses
    frames, _p = load_poses(out)
    n = len(frames)
    apply_pose_epoch(out, np.tile(np.eye(3), (n, 1, 1)), np.ones((n, 3)), np.ones(n), "gauge", [],
                     log=lambda *a: None)
    assert reconstruction_id(out) == rid
    # one Omega record changes: another reconstruction
    rec = sorted((out / "omega_run" / "results_output").glob("frame_*.npz"))[0]
    with np.load(rec) as z:
        d = {k: z[k] for k in z.files}
    d["depth"] = d["depth"] * 1.001
    np.savez(rec, **d)
    assert reconstruction_id(out) != rid
    assert same_reconstruction({RECONSTRUCTION_ID_KEY: rid}, reconstruction_id(out))[0] is False
    assert same_reconstruction({RECONSTRUCTION_ID_KEY: rid}, rid) == (True, "same reconstruction")
    assert same_reconstruction({"measured_on_epoch": 0}, rid)[0] is False
    assert same_reconstruction({RECONSTRUCTION_ID_KEY: rid}, None)[0] is False
    with pytest.raises(ReconstructionIdError):
        reconstruction_id(tmp_path / "nowhere")
    assert reconstruction_id_or_none(tmp_path / "nowhere") is None


def test_visit_drift_stamps_its_evidence_with_the_reconstruction(tmp_path):
    """The writers of scale_loop_rows.json / instance_loops.json (visit_drift_run) stamp the
    reconstruction id F2 / F4 verify (point 34)."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
    from correction.visit_drift_run import _stamp_out, _write_scale_rows
    out = _session(tmp_path / "s")
    p = _stamp_out(out, "instance_loops.json", {"loops": [{"i": 0, "j": 3}]})
    doc = json.loads(p.read_text())
    assert doc[RECONSTRUCTION_ID_KEY] == reconstruction_id(out) and doc["measured_on_epoch"] == 0
    _write_scale_rows(out, {"scale_rows": [{"i": 0, "j": 3, "k_b": 1.02, "residual_m": 0.01, "D_b_m": 2.0}]},
                      log=lambda *a: None)
    rows = json.loads((out / "scale_loop_rows.json").read_text())
    assert rows[RECONSTRUCTION_ID_KEY] == reconstruction_id(out)
    # a session with no Omega records writes None — which no reader takes
    out2 = tmp_path / "bare" / "output"
    out2.mkdir(parents=True)
    doc2 = json.loads(_stamp_out(out2, "instance_loops.json", {"loops": []}).read_text())
    assert doc2[RECONSTRUCTION_ID_KEY] is None


def test_the_confidence_calibration_carries_the_reconstruction_id(tmp_path):
    """Plan point 33: the writer stamps it; F2 takes it only with this id and at epoch 0."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
    from precision import confidence as CAL
    out = _session(tmp_path / "s")
    rng = np.random.default_rng(0)
    c = rng.uniform(0.0, 1.0, 2000)
    table = CAL.calibrate(np.abs(rng.normal(0, 0.01, c.size)), c, rng.uniform(1, 6, c.size),
                          conf_bins=2, dist_bins=1, quantile=0.95, min_bin_samples=50)
    CAL.write_calibration(out, "tier0", {"da3": table}, {"geometry_epoch": 0, "camera_epoch": 0}, {})
    doc = CAL.load_calibration(out, reference="tier0")
    assert doc[RECONSTRUCTION_ID_KEY] == reconstruction_id(out)
