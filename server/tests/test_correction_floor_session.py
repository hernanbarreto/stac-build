"""Floor model "session" (USER 2026-10-16, after bufferStop came out "de costado" under the chunk
model): ONE rigid motion for the whole walk — the SEGMENTATION's support surfaces, over the whole
session, their level under the cameras → horizontal at y = 0. No chunk turns against another,
nothing assumes which way is up, a second level keeps its step, no support instance → the
geometric candidates, declared."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from correction import floor as FL


def _scene(tmp_path, *, tilt_deg=3.0, offset=0.12, frame_rot_deg=0.0, platform=False,
           labels=("floor",), noise_m=0.004, seed=3):
    """A 20-keyframe walk in two Omega chunks over ONE floor tilted ``tilt_deg`` about z and
    ``offset`` m up, with Gaussian scatter; a 'gravel_pile' blob above the floor (not support);
    optionally a 'platform' 1 m higher on the far side (a second support level, under no camera).
    ``frame_rot_deg`` turns the WHOLE scene (points and cameras) about z: the session's frame
    is then not y-up. ``labels``: the label given to the floor instance(s)."""
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    n_kf, per = 20, 500
    th = np.radians(tilt_deg)
    xyz, ks, inst = [], [], []
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    start = 0
    on_platform = (lambda k: platform and k < 6)
    for k in range(n_kf):
        chunk = 0 if k < 10 else 1
        np.savez(out / "omega_run" / "results_output" / f"frame_{100 + k}.npz", chunk=np.int64(chunk))
        cx = 0.6 * k - 2.0
        n = 120 if on_platform(k) else per            # a keyframe on the platform sees little floor
        x = rng.uniform(cx - 1.5, cx + 1.5, n)
        z = rng.uniform(-2, 2, n)
        y = x * np.tan(th) + offset + rng.normal(0.0, noise_m, n)
        xyz.append(np.stack([x, y, z], 1)); ks.append(np.full(n, k))
        poses[k, :3, 3] = [cx, cx * np.tan(th) + offset + 1.6 + (1.0 if on_platform(k) else 0.0), 0.0]
    n_floor = sum(len(a) for a in xyz)
    inst.append({"id": 1, "label": labels[0], "globalIndices": list(range(start, start + n_floor))})
    start += n_floor
    # a heap above the floor, seen from every keyframe: NOT a support surface
    m = 3000
    hx = rng.uniform(2, 5, m); hz = rng.uniform(-1, 1, m)
    hy = 0.3 + 0.8 * np.exp(-((hx - 3.5) ** 2 + hz ** 2)) + hx * np.tan(th) + offset
    xyz.append(np.stack([hx, hy, hz], 1)); ks.append(rng.integers(0, n_kf, m))
    inst.append({"id": 2, "label": "gravel_pile", "globalIndices": list(range(start, start + m))})
    start += m
    if platform:
        # keyframes 0-5 walk on a platform 1 m above the floor (x < 1.3): their points are mostly its
        # (the user's rule needs at least five keyframes standing on a level to call it real)
        m = 6 * per
        px = rng.uniform(-4, 1.3, m); pz = rng.uniform(-2, 2, m)
        py = px * np.tan(th) + offset + 1.0 + rng.normal(0.0, noise_m, m)
        xyz.append(np.stack([px, py, pz], 1)); ks.append(np.repeat(np.arange(6), per))
        inst.append({"id": 3, "label": "concrete_platform", "globalIndices": list(range(start, start + m))})
        start += m
    xyz = np.concatenate(xyz); ks = np.concatenate(ks)
    if frame_rot_deg:
        Rz = Rotation.from_euler("z", frame_rot_deg, degrees=True).as_matrix()
        xyz = xyz @ Rz.T
        for k in range(n_kf):
            poses[k, :3, 3] = Rz @ poses[k, :3, 3]
            poses[k, :3, :3] = Rz @ poses[k, :3, :3]
    (out / "segmentation_result.json").write_text(json.dumps({"instances": inst}))
    s = SimpleNamespace(output_dir=out, xyz=xyz, ks=ks, n_kf=n_kf, frames=[100 + k for k in range(n_kf)],
                        poses=poses)
    return s, n_floor


def _cfg():
    fl = SimpleNamespace(chunk_floor_labels=("floor",), support_labels=("floor", "ground", "platform", "slab"),
                         band_m=0.5, low_band_pct=5.0, ransac_tol_m=0.01, ransac_refit_band_m=0.02,
                         ransac_sample=120000)
    return SimpleNamespace(floor=fl, visit_drift=SimpleNamespace(default_repeatability_m=0.0477))


def _apply(s, sol):
    R, t = sol["R_kf"], sol["t_kf"]
    return np.einsum("nij,nj->ni", R[s.ks], s.xyz) + t[s.ks]


def test_one_rigid_motion_levels_the_segmented_floor_at_zero(tmp_path):
    s, n_floor = _scene(tmp_path)
    sol = FL.solve_floor_session(s, _cfg(), log=lambda m: None)
    rec = sol["model_params"]
    assert rec["source"] == "segmentation" and set(rec["support_instances"]) == {"floor"}
    assert rec["tilt_applied"] and rec["height_applied"] and rec["under_cameras"]
    assert abs(rec["tilt_deg"] - 3.0) < 0.05
    # every keyframe moves by the SAME transform
    assert np.allclose(sol["R_kf"], sol["R_kf"][0]) and np.allclose(sol["t_kf"], sol["t_kf"][0])
    moved = _apply(s, sol)
    y = moved[:n_floor, 1]
    assert abs(float(np.median(y))) < 0.01 and float(np.std(y)) < 0.01     # flat, at zero
    # the heap went with it, 0.3 m and more above the floor — never flattened
    assert float(np.median(moved[n_floor:n_floor + 3000, 1])) > 0.25


def test_a_frame_on_its_side_is_levelled_all_the_same(tmp_path):
    s, n_floor = _scene(tmp_path, frame_rot_deg=50.0)
    sol = FL.solve_floor_session(s, _cfg(), log=lambda m: None)
    rec = sol["model_params"]
    assert rec["tilt_applied"] and abs(rec["tilt_deg"] - 53.0) < 0.1
    moved = _apply(s, sol)
    y = moved[:n_floor, 1]
    assert abs(float(np.median(y))) < 0.01 and float(np.std(y)) < 0.01
    cams = np.einsum("nij,nj->ni", sol["R_kf"], s.poses[:, :3, 3]) + sol["t_kf"]
    assert np.all(cams[:, 1] > 1.5)                                        # the walk above its floor


def test_a_second_support_level_keeps_its_step(tmp_path):
    s, n_floor = _scene(tmp_path, platform=True)
    sol = FL.solve_floor_session(s, _cfg(), log=lambda m: None)
    rec = sol["model_params"]
    assert set(rec["support_instances"]) == {"floor", "concrete_platform"}
    real = [l for l in rec["levels"] if l["real"]]
    assert len(real) == 2 and rec["under_cameras"]
    chosen = [l for l in rec["levels"] if l["chosen"]][0]
    other = [l for l in rec["levels"] if not l["chosen"]][0]
    assert chosen["cameras_under"] >= 12 and other["cameras_under"] >= 6      # the walk is over the floor
    assert chosen["support"] > 7000                                           # the WHOLE floor, not the band's strip
    moved = _apply(s, sol)
    assert abs(float(np.median(moved[:n_floor, 1]))) < 0.01
    plat = moved[n_floor + 3000:, 1]
    assert abs(float(np.median(plat)) - 1.0) < 0.02                         # the step stays
    assert abs(sorted(rec["surfaces"].values())[-1] - 1.0) < 0.02


def test_without_a_support_instance_the_points_below_the_cameras_decide_declared(tmp_path):
    s, n_floor = _scene(tmp_path, labels=("wall",))
    sol = FL.solve_floor_session(s, _cfg(), log=lambda m: None)
    rec = sol["model_params"]
    assert rec["source"] == "geometric" and rec["support_instances"] == {}
    assert rec["tilt_applied"]
    moved = _apply(s, sol)
    assert abs(float(np.median(moved[:n_floor, 1]))) < 0.02


def test_head_noun_names_the_surface():
    assert FL.head_noun("concrete_slab") == "slab"
    assert FL.head_noun("White tiled floor") == "floor"
    assert FL.head_noun("platforms") == "platform"
    assert FL.head_noun("floor 2") == "floor" and FL.head_noun("floor#3") == "floor"
    assert FL.head_noun("gravel_pile") == "pile"
    assert FL.head_noun("andén") == "andén"


def test_rot_between_handles_antiparallel_vectors():
    R = FL._rot_between(np.array([0.0, -1.0, 0.0]), FL.UP)
    assert np.allclose(R @ np.array([0.0, -1.0, 0.0]), FL.UP) and np.isclose(np.linalg.det(R), 1.0)


def test_config_reads_the_session_model_and_the_segmentations_support_list():
    from correction.config import CorrectionConfigError, load_correction_config
    from tests.synth_correction import make_correction_raw_cfg
    import pytest
    raw = make_correction_raw_cfg(**{"floor.model_default": "session"})
    cfg = load_correction_config(raw)
    assert cfg.floor.model_default == "session" and "platform" in cfg.floor.support_labels
    raw.pop("autoprompt")
    with pytest.raises(CorrectionConfigError, match="autoprompt.support_surfaces"):
        load_correction_config(raw)
