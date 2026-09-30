"""Floor model "chunk" (USER 2026-09-30: "tenemos la segmentación de piso, el piso por
chunk llevarlo a cero"): ONE rigid motion per Omega chunk from its SEGMENTED floor —
every keyframe of a chunk moves together (the per-keyframe models layered the floor)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from correction import floor as FL


def _session(tmp_path, tilt_deg=3.0, offset=0.12):
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    rng = np.random.default_rng(0)
    n_kf, per = 20, 400
    xyz, ks = [], []
    th = np.radians(tilt_deg)
    for k in range(n_kf):
        chunk = 0 if k < 10 else 1
        np.savez(out / "omega_run" / "results_output" / f"frame_{100 + k}.npz", chunk=np.int64(chunk))
        x = rng.uniform(-2, 2, per) + (0 if chunk == 0 else 5)
        z = rng.uniform(-2, 2, per)
        y = np.zeros(per)
        if chunk == 0:                        # chunk 0: tilted about z and 12 cm up
            y = x * np.tan(th) + offset
        xyz.append(np.stack([x, y, z], 1)); ks.append(np.full(per, k))
    xyz = np.concatenate(xyz); ks = np.concatenate(ks)
    (out / "segmentation_result.json").write_text(json.dumps(
        {"instances": [{"id": 1, "label": "floor", "globalIndices": list(range(len(xyz)))}]}))
    s = SimpleNamespace(output_dir=out, xyz=xyz, ks=ks, n_kf=n_kf, frames=[100 + k for k in range(n_kf)])
    return s


def _cfg():
    fl = SimpleNamespace(chunk_floor_labels=("floor",), min_inliers=100, ransac_sample=2000,
                         ransac_iters=200, ransac_tol_m=0.01, ransac_refit_band_m=0.02)
    return SimpleNamespace(floor=fl)


def test_each_chunk_moves_rigidly_to_a_level_floor_at_zero(tmp_path):
    s = _session(tmp_path)
    sol = FL.solve_floor_by_chunk(s, _cfg(), np.random.default_rng(0), log=lambda m: None)
    R, t = sol["R_kf"], sol["t_kf"]
    # every keyframe of a chunk has the SAME transform
    assert np.allclose(R[:10], R[0]) and np.allclose(t[:10], t[0])
    assert np.allclose(R[10:], R[10]) and np.allclose(t[10:], t[10])
    # the corrected floor of chunk 0 is horizontal at y = 0
    P = s.xyz[s.ks < 10]
    Q = P @ R[0].T + t[0]
    assert np.abs(Q[:, 1]).max() < 0.005
    # chunk 1 was already level at 0 → (near) identity
    assert np.allclose(R[10], np.eye(3), atol=1e-3) and np.abs(t[10]).max() < 0.005


def test_the_segmented_floor_is_required(tmp_path):
    s = _session(tmp_path)
    (s.output_dir / "segmentation_result.json").write_text(json.dumps({"instances": []}))
    try:
        FL.solve_floor_by_chunk(s, _cfg(), np.random.default_rng(0), log=lambda m: None)
    except RuntimeError as e:
        assert "segmented" in str(e)
    else:
        raise AssertionError("no segmented floor must fail loudly")
