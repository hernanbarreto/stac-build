"""Floor model "chunk" (USER 2026-09-30: "tenemos la segmentación de piso, el piso por
chunk llevarlo a cero"): ONE rigid motion per Omega chunk — every keyframe of a chunk moves
together (the per-keyframe models layered the floor).

Since 2026-10-08 (docs/plan_determinismo.md points 128 / 129, the user's floor concept) the
floor of a chunk is GEOMETRIC: the dominant support plane below its cameras, a deterministic
robust fit (no RANSAC, no 5000-point cut); the segmented floor is a hint. These tests changed
with that decision: the fixture carries the cameras the floor is measured under and the keys
the estimator reads, and 'the segmented floor is required' became 'labels are only a hint'."""
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
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    for k in range(n_kf):
        chunk = 0 if k < 10 else 1
        np.savez(out / "omega_run" / "results_output" / f"frame_{100 + k}.npz", chunk=np.int64(chunk))
        x = rng.uniform(-2, 2, per) + (0 if chunk == 0 else 5)
        z = rng.uniform(-2, 2, per)
        y = np.zeros(per)
        if chunk == 0:                        # chunk 0: tilted about z and 12 cm up
            y = x * np.tan(th) + offset
        xyz.append(np.stack([x, y, z], 1)); ks.append(np.full(per, k))
        # the camera walks 1.6 m over its floor (the floor is measured under the cameras)
        poses[k, :3, 3] = [(0 if chunk == 0 else 5) + 0.2 * (k % 10) - 1.0, 1.6 + (offset if chunk == 0 else 0.0), 0.0]
    xyz = np.concatenate(xyz); ks = np.concatenate(ks)
    (out / "segmentation_result.json").write_text(json.dumps(
        {"instances": [{"id": 1, "label": "floor", "globalIndices": list(range(len(xyz)))}]}))
    s = SimpleNamespace(output_dir=out, xyz=xyz, ks=ks, n_kf=n_kf, frames=[100 + k for k in range(n_kf)],
                        poses=poses)
    return s


def _cfg():
    fl = SimpleNamespace(chunk_floor_labels=("floor",), band_m=0.5, low_band_pct=5.0,
                         ransac_tol_m=0.01, ransac_refit_band_m=0.02)
    return SimpleNamespace(floor=fl, visit_drift=SimpleNamespace(default_repeatability_m=0.0477))


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
    # the measurement is DETERMINISTIC: the same session gives the same bytes
    sol2 = FL.solve_floor_by_chunk(s, _cfg(), np.random.default_rng(5), log=lambda m: None)
    assert np.array_equal(sol2["R_kf"], R) and np.array_equal(sol2["t_kf"], t)
    per = sol["per_kf_report"]
    # chunk 0 moves by its own measurement; chunk 1 (level, measured to the millimetre) would
    # take the neighbours' blend by point 129 — its own measurement contradicts that blend
    # beyond its error, so its own (identity) value stands, declared
    assert [c["role"] for c in per] == ["anchor", "own_within_error"], per
    assert per[1]["blend_contradicted"] and per[1]["blended_from"] == [0]
    assert all(c["levels"][0]["real"] for c in per)


def test_the_floor_is_geometric_and_labels_are_only_a_hint(tmp_path):
    """Point 128: no instance labelled 'floor' (pccr 2026-10-04: no epoch was published for
    that) — the floor under the cameras is still measured and levelled."""
    s = _session(tmp_path)
    (s.output_dir / "segmentation_result.json").write_text(json.dumps({"instances": []}))
    sol = FL.solve_floor_by_chunk(s, _cfg(), np.random.default_rng(0), log=lambda m: None)
    P = s.xyz[s.ks < 10]
    Q = P @ sol["R_kf"][0].T + sol["t_kf"][0]
    assert np.abs(Q[:, 1]).max() < 0.005
    assert all(c["hint_points"] == 0 for c in sol["per_kf_report"])
    assert sol["model_params"]["labels_hint"] == ["floor"]


def test_a_second_level_is_a_separate_floor_and_a_step_is_kept(tmp_path):
    """The user's floor concept (2026-10-08): a platform and the track bed 1 m below it are two
    floors; the camera steps down in chunk 1 — the level change is a STEP, never corrected as
    drift, and both surfaces keep their separation."""
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    rng = np.random.default_rng(1)
    n_kf, per = 20, 600
    xyz, ks = [], []
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    for k in range(n_kf):
        chunk = 0 if k < 10 else 1
        np.savez(out / "omega_run" / "results_output" / f"frame_{100 + k}.npz", chunk=np.int64(chunk))
        x0 = 0.0 if chunk == 0 else 5.0
        # every keyframe sees the platform (y = 0) and the track bed (y = -1) side by side
        xp = rng.uniform(-2, 2, per) + x0
        zp = rng.uniform(-2, 0, per)
        xb = rng.uniform(-2, 2, per) + x0
        zb = rng.uniform(0, 2, per)
        xyz.append(np.stack([xp, np.zeros(per) + 0.05 * (k < 10), zp], 1))   # chunk 0 sits 5 cm high
        xyz.append(np.stack([xb, np.full(per, -1.0) + 0.05 * (k < 10), zb], 1))
        ks.append(np.full(2 * per, k))
        # chunk 0: the camera walks on the platform; chunk 1: it stepped down to the track bed
        cam_y = (1.6 + 0.05) if chunk == 0 else 0.6
        cam_z = -1.0 if chunk == 0 else 1.0
        poses[k, :3, 3] = [x0 + 0.2 * (k % 10) - 1.0, cam_y, cam_z]
    xyz = np.concatenate(xyz); ks = np.concatenate(ks)
    (out / "segmentation_result.json").write_text(json.dumps({"instances": []}))
    s = SimpleNamespace(output_dir=out, xyz=xyz, ks=ks, n_kf=n_kf,
                        frames=[100 + k for k in range(n_kf)], poses=poses)
    sol = FL.solve_floor_by_chunk(s, _cfg(), None, log=lambda m: None)
    per = sol["per_kf_report"]
    # both levels are real floors in both chunks (the user's rule on the keyframes)
    assert all(len(c["levels"]) >= 2 and c["levels"][1]["real"] for c in per), per
    # chunk 0 walks on the platform, chunk 1 on the track bed — chosen by the camera's height
    assert per[0]["chosen_rank"] != per[1]["chosen_rank"] or \
        abs(per[0]["floor_y_at_first_kf_m"] - per[1]["floor_y_at_first_kf_m"]) > 0.5
    assert per[1]["seam"]["verdict"].startswith("level_change"), per[1]["seam"]
    assert len(sol["model_params"]["level_changes"]) == 1
    R, t = sol["R_kf"], sol["t_kf"]
    Q = xyz @ R[ks].transpose(0, 2, 1)[:, :, :] if False else np.einsum("nij,nj->ni", R[ks], xyz) + t[ks]
    plat = Q[(xyz[:, 2] < 0)][:, 1]
    bed = Q[(xyz[:, 2] > 0)][:, 1]
    # the platform of chunk 0 lands at y = 0; the track bed stays ~1 m below it everywhere
    assert abs(np.median(plat[ks[xyz[:, 2] < 0] < 10])) < 0.01
    assert abs((np.median(plat) - np.median(bed)) - 1.0) < 0.02
    # chunk 1 kept the step: its platform is NOT flattened onto its track bed
    assert abs(np.median(plat[ks[xyz[:, 2] < 0] >= 10]) - np.median(bed[ks[xyz[:, 2] > 0] >= 10]) - 1.0) < 0.02
