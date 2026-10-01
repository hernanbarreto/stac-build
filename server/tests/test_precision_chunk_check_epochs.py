"""The chunk check after the certification measures the PUBLISHED cloud: the certify
stage applies a further per-keyframe depth factor about each keyframe's camera
(corrections/epoch_<N>.npz, k_kf and the affine b_kf), composed in lineage order on
the cloud epoch's own s_k + bend. The rigid part is already in camera_poses.txt.

Same synthetic room as test_precision_chunk_check: clean Omega depth everywhere;
the certification's epoch stretches chunk 1's depth by 1.15 → the check must say
`depth` on chunk 1 with factor ≈ 1/1.15, because that is the cloud that is live."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.epoch import EPOCH_FILE, LEDGER_FILE          # noqa: E402
from correction.ledger import EPOCH_NPZ_DIR, save_epoch_npz   # noqa: E402
from precision import chunk_check as CC                        # noqa: E402
from tests.test_precision_chunk_check import N_KF, PER_CHUNK, build_session   # noqa: E402


@pytest.fixture(scope="module")
def pcfg():
    from precision.config import load_precision_config
    p = load_precision_config()
    return replace(p, chunk_check=replace(p.chunk_check, pixel_stride=1, min_points=50, bootstrap=200))


def _cloud_report(out: Path, frames, epoch_to: int, s_k=1.0, bend=(0.0, 0.0)):
    (out / "corrected_cloud.json").write_text(json.dumps(
        {"epoch_to": epoch_to, "per_frame": {str(f): {"s_k": s_k, "bend": list(bend)} for f in frames}}))


def _npz(out: Path, epoch: int, frames, k, b=None):
    n = len(frames)
    save_epoch_npz(out, epoch, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)),
                   np.asarray(k, np.float64), list(frames), b_kf=b)


def test_the_check_measures_the_depth_the_certification_published(tmp_path, pcfg):
    frames, chain = build_session(tmp_path, "clean")                 # live epoch 2, no parent recorded
    out = tmp_path / "output"
    _cloud_report(out, frames, epoch_to=1)
    k = np.ones(N_KF)
    k[PER_CHUNK:2 * PER_CHUNK] = 1.15                                  # the certify epoch stretched chunk 1
    _npz(out, 2, frames, k)
    rep = CC.run_check(tmp_path, pcfg, log=lambda m: None, chainage=chain)
    assert rep["params"]["depth_epochs_composed"] == [2]
    assert "transform epoch(s) [2]" in rep["params"]["s_k_source"] and "k_kf" in rep["params"]["s_k_source"]
    assert "b_kf" not in rep["params"]["s_k_source"]
    by_kf = {r["i"]: r["s_k"] for r in rep["keyframes"]}
    assert all(abs(by_kf[i] - (1.15 if PER_CHUNK <= i < 2 * PER_CHUNK else 1.0)) < 1e-12 for i in range(N_KF))
    by = {c["chunk"]: c for c in rep["chunks"]}
    assert by[1]["verdict"] == "depth", by[1]["why"]
    assert by[0]["verdict"] != "depth" and by[2]["verdict"] != "depth"
    fix = [c for c in rep["to_correct"] if c["chunk"] == 1]
    assert len(fix) == 1 and fix[0]["kind"] == "scale_about_camera"
    assert abs(fix[0]["factor"] - 1 / 1.15) < 0.03 * (1 / 1.15)


def test_without_a_transform_epoch_the_cloud_epoch_is_what_is_measured(tmp_path, pcfg):
    frames, chain = build_session(tmp_path, "clean")
    out = tmp_path / "output"
    _cloud_report(out, frames, epoch_to=2)                             # live epoch == cloud epoch
    rep = CC.run_check(tmp_path, pcfg, log=lambda m: None, chainage=chain)
    assert rep["params"]["depth_epochs_composed"] == []
    assert rep["params"]["s_k_source"].endswith("(live epoch 2 is the cloud epoch)")
    assert all(r["s_k"] == 1.0 for r in rep["keyframes"])
    assert [c["verdict"] for c in rep["chunks"]] == ["ok", "ok", "ok"]


def test_two_epochs_compose_in_lineage_order_with_the_affine_offset(tmp_path):
    """z' = k·z + b per epoch: K ← k·K, B ← k·B + b; the cloud's k(u, v) scales by K."""
    frames, _ = build_session(tmp_path, "clean")
    out = tmp_path / "output"
    _cloud_report(out, frames, epoch_to=0, s_k=1.05, bend=(0.01, 0.02))
    n = N_KF
    _npz(out, 1, frames, np.full(n, 1.1), np.full(n, 0.2))
    _npz(out, 2, frames, np.full(n, 2.0), np.full(n, 0.1))
    (*_, s_k, src, _c, _ch, _o, _d, bend, composed) = CC.load_inputs(tmp_path, log=lambda m: None)
    assert composed["epochs_composed"] == [1, 2]
    f = frames[3]
    assert abs(s_k[f] - 1.05 * 2.2) < 1e-12
    assert np.allclose(bend[f], (0.01 * 2.2, 0.02 * 2.2))
    assert abs(composed["offset"][f] - (2.0 * 0.2 + 0.1)) < 1e-12
    assert "k_kf, b_kf" in src and "[1, 2]" in src
    # the offset lands only where Omega measured a depth
    d0 = np.array([[0.0, 2.0], [np.nan, 4.0]])
    d = d0 * CC.scale_map(s_k[f], bend[f], 2, 2)
    off = composed["offset"][f]
    d = np.where(np.isfinite(d0) & (d0 > 0), d + off, d)
    assert d[0, 0] == 0.0 and np.isnan(d[1, 0]) and d[0, 1] > 2.0 * s_k[f]


def test_a_new_cloud_epoch_above_the_cloud_epoch_is_refused(tmp_path):
    frames, _ = build_session(tmp_path, "clean")
    out = tmp_path / "output"
    _cloud_report(out, frames, epoch_to=1)
    (out / LEDGER_FILE).write_text(json.dumps(
        {"type": "run", "correction_id": "x", "epoch_from": 1, "epoch_to": 2, "kind": "fuse",
         "verdict": "applied"}) + "\n")
    with pytest.raises(CC.ChunkCheckError, match="epoch 2 is a new cloud"):
        CC.load_inputs(tmp_path, log=lambda m: None)


def test_a_live_epoch_that_does_not_descend_from_the_cloud_is_refused(tmp_path):
    frames, _ = build_session(tmp_path, "clean")
    out = tmp_path / "output"
    _cloud_report(out, frames, epoch_to=5)                             # the cloud epoch is above the live one
    with pytest.raises(CC.ChunkCheckError, match="does not descend from the cloud epoch 5"):
        CC.load_inputs(tmp_path, log=lambda m: None)


def test_an_epoch_that_does_not_name_a_keyframe_is_refused(tmp_path):
    frames, _ = build_session(tmp_path, "clean")
    out = tmp_path / "output"
    _cloud_report(out, frames, epoch_to=1)
    _npz(out, 2, frames[:-1], np.ones(N_KF - 1))
    with pytest.raises(CC.ChunkCheckError, match=f"epoch_2.npz names no transform for keyframe"):
        CC.load_inputs(tmp_path, log=lambda m: None)
    assert (out / EPOCH_NPZ_DIR / "epoch_2.npz").exists() and (out / EPOCH_FILE).exists()
