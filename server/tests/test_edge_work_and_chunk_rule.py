"""docs/plan_determinismo.md points 138 (the acta's edge metric bounded by WORK, never by a clock),
140 (the chunk check's floor plane always fitted and reported; per-chunk verdicts by the user's
rule) and 45 / 56 (witness poses round-trip exact). CPU, synthetic."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUIET = lambda *a: None  # noqa: E731


def _fac() -> float:
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    return float(improvement_error_factor(raw_cfg))


# ── point 138: the edge metric's work ─────────────────────────────────────

def test_the_largest_objects_are_chosen_by_a_stable_rule():
    from precision.cloud_metrics import select_edge_objects
    inst = [{"instance_id": 7, "globalIndices": [0, 1, 2]}, {"id": 3, "globalIndices": [0, 1, 2]},
            {"instance_id": 5, "total_points": 10, "globalIndices": [1]}, {"instance_id": 9, "globalIndices": [4]}]
    got = [(n, i) for n, i, _ in select_edge_objects(inst, 3)]
    assert got == [(10, 5), (3, 3), (3, 7)]                   # largest first, ties by instance id
    assert [(n, i) for n, i, _ in select_edge_objects(list(reversed(inst)), 3)] == got
    assert select_edge_objects(inst, 0) == []


def test_the_point_sample_is_the_same_points_whatever_the_order_or_the_cloud_size():
    from precision.cloud_metrics import stable_point_sample
    rng = np.random.default_rng(0)
    n = 5000
    fields = {"x": rng.standard_normal(n).astype(np.float32), "y": rng.standard_normal(n).astype(np.float32),
              "z": rng.standard_normal(n).astype(np.float32),
              "frame_global": rng.integers(0, 300, n), "pixel_row": rng.integers(0, 800, n),
              "pixel_col": rng.integers(0, 460, n)}
    rows = np.arange(n)
    a = stable_point_sample(fields, rows, 700)
    assert len(a) == 700 and np.all(np.diff(a) > 0)
    # the same rows in another order → the same sample
    assert np.array_equal(stable_point_sample(fields, rows[::-1], 700), a)
    # the sample of a superset keeps every point whose key ranks in the sample of the subset's keys
    sub = rows[:2500]
    b = stable_point_sample(fields, sub, 700)
    assert set(b) <= set(sub) and len(b) == 700
    # under the cap: untouched
    assert np.array_equal(stable_point_sample(fields, sub[:100], 700), sub[:100])
    # without provenance the float32 bits of xyz are the key — the same rule, the same determinism
    xyz_only = {k: fields[k] for k in ("x", "y", "z")}
    c = stable_point_sample(xyz_only, rows, 700)
    assert len(c) == 700 and np.array_equal(stable_point_sample(xyz_only, rows[::-1], 700), c)


def _edge_session(tmp_path: Path, n_inst: int = 3, n_pts: int = 400) -> Path:
    from precision.epoch0_cloud import _write_ply_xyzrgb
    out = tmp_path / "output"
    (out / "precision").mkdir(parents=True)
    rng = np.random.default_rng(1)
    N = n_inst * n_pts
    _write_ply_xyzrgb(out / "cleaned_cloud.ply", rng.standard_normal((N, 3)).astype(np.float32),
                      rng.integers(0, 255, (N, 3)).astype(np.uint8))
    inst = [{"instance_id": k + 1, "label": f"obj{k}", "globalIndices": list(range(k * n_pts, (k + 1) * n_pts))}
            for k in range(n_inst)]
    (out / "segmentation_result.json").write_text(json.dumps({"instances": inst}))
    np.savetxt(out / "camera_poses.txt", np.tile(np.eye(4).reshape(1, 16), (2, 1)))
    (out / "camera_frames.txt").write_text("0\n1\n")
    return out


def test_the_subprocess_gets_bounded_work_a_fresh_report_and_no_clock(tmp_path, monkeypatch):
    from precision import cloud_metrics as CM
    from precision.config import load_precision_config
    out = _edge_session(tmp_path)
    p = load_precision_config()
    pcfg = replace(p, cloud_metrics=replace(p.cloud_metrics, edge_max_objects=2, edge_max_points_per_object=150))
    stale = out / "precision" / CM.EDGE_REPORT
    stale.write_text(json.dumps({"stale": True}))
    seen = {}

    def run(cmd, **kw):
        seen["kw"] = kw
        seen["cmd"] = cmd
        assert not stale.exists(), "the previous report must be gone before the metric runs"
        seg = json.loads(Path(cmd[cmd.index("--seg") + 1]).read_text())
        seen["seg"] = seg
        objs = [{"instance_id": i["instance_id"], "label": i["label"], "n_points": len(i["globalIndices"]),
                 "creases": [{}], "summary": {"n_creases": 1, "r_hat_m": 0.004}} for i in seg["instances"]]
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"reference": {"objects": objs}, "seconds": 12.5}))
        return subprocess.CompletedProcess(cmd, 0, "[edge] fake\n", "")
    monkeypatch.setattr(CM.subprocess, "run", run) if hasattr(CM, "subprocess") else None
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", run)
    rep = CM.run_cloud_metrics(tmp_path, pcfg, stage="acta", log=QUIET, edges=True)
    assert "timeout" not in seen["kw"]
    ids = [int(x) for x in seen["cmd"][seen["cmd"].index("--instance-ids") + 1:]]
    assert ids == [1, 2]                                           # the two largest (ties by id)
    assert [len(i["globalIndices"]) for i in seen["seg"]["instances"]] == [150, 150]
    assert all(i["total_points"] == 400 for i in seen["seg"]["instances"])
    w = rep["edges"]["work"]
    assert w == {"objects_bound": 2, "points_per_object_bound": 150,
                 "objects": [{"instance_id": 1, "label": "obj0", "points": 400, "points_measured": 150, "sampled": True},
                             {"instance_id": 2, "label": "obj1", "points": 400, "points_measured": 150, "sampled": True}]}
    assert rep["edges"]["objects_measured"] == 2 and rep["edges"]["r_hat_m_median"] == 0.004
    assert "seconds" not in rep and "seconds" not in rep["edges"]
    written = json.loads((out / "precision" / CM.EDGE_REPORT).read_text())
    assert "seconds" not in written and "stale" not in written
    times = json.loads((out / "precision" / "cloud_metrics.timing.json").read_text())
    assert times["edges_seconds"] == 12.5 and "edges_seconds_total" in times and "measured_at" in times
    assert not (out / "precision" / CM.EDGE_OBJECTS).exists()      # the transient object list is gone


def test_a_failed_edge_run_leaves_no_stale_report(tmp_path, monkeypatch):
    from precision import cloud_metrics as CM
    from precision.config import load_precision_config
    out = _edge_session(tmp_path)
    stale = out / "precision" / CM.EDGE_REPORT
    stale.write_text(json.dumps({"stale": True}))
    import subprocess as _sp
    monkeypatch.setattr(_sp, "run", lambda cmd, **kw: _sp.CompletedProcess(cmd, 1, "", "boom"))
    rep = CM.run_cloud_metrics(tmp_path, load_precision_config(), stage="acta", log=QUIET, edges=True)
    assert "error" in rep["edges"] and "boom" in rep["edges"]["error"]
    assert not stale.exists()


@pytest.mark.parametrize("section,key", [("cloud_metrics", "edge_timeout_s"), ("chunk_check", "plane_min_inlier_frac")])
def test_a_leftover_removed_key_fails_the_load(section, key):
    import copy
    from config import cfg as raw_cfg
    from precision.config import PrecisionConfigError, load_precision_config
    bad = copy.deepcopy(raw_cfg)
    bad["reconstruction"]["precision"][section][key] = 1
    with pytest.raises(PrecisionConfigError, match=f"{section}.{key}"):
        load_precision_config(bad)
    assert load_precision_config().cloud_metrics.edge_max_points_per_object >= 1


# ── point 140: the floor plane is reported, the verdicts judged by the rule ──

def test_a_floor_holding_nineteen_percent_is_reported_with_its_share_and_interval():
    from precision import chunk_check as CC
    rng = np.random.default_rng(0)
    n_floor, n_rest = 19_000, 81_000
    floor = np.c_[rng.uniform(-5, 5, n_floor), rng.normal(0, 0.005, n_floor), rng.uniform(-5, 5, n_floor)]
    rest = np.c_[rng.uniform(-5, 5, n_rest), rng.uniform(0.3, 3.0, n_rest), rng.uniform(-5, 5, n_rest)]
    P = np.concatenate([floor, rest])
    n, c, info = CC.dominant_plane(P, 0.08, 0, 0.95, 200)        # no acceptance bar
    assert abs(abs(n[1]) - 1.0) < 1e-3 and abs(c[1]) < 0.01
    assert abs(info["inlier_frac"] - 0.19) < 0.01
    lo, hi = info["inlier_frac_ci"]
    assert lo <= info["inlier_frac"] <= hi and hi - lo < 0.01 and info["n_points"] == 100_000
    assert "none" in info["acceptance_bar"]
    # the same plane twice (seeded)
    n2, c2, info2 = CC.dominant_plane(P, 0.08, 0, 0.95, 200)
    assert np.array_equal(n, n2) and np.array_equal(c, c2) and info == info2
    with pytest.raises(CC.NoFloorPlane):
        CC.dominant_plane(P[:2], 0.08, 0, 0.95, 200)


def _rows(n_per_chunk, floor_by_chunk, ceil_by_chunk, res=0.01, seed=0):
    from precision.chunk_check import Row
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    for k, (fh, ch) in enumerate(zip(floor_by_chunk, ceil_by_chunk)):
        for _ in range(n_per_chunk[k]):
            f = fh + rng.normal(0, res); c = ch + rng.normal(0, res)
            rows.append(Row(i, 10 * i, k, 0.1 * i, f, 500, c, 500, 1.3, 1.3 - f, 1.2, 1.0, res, res))
            i += 1
    return rows


def test_chunk_verdicts_are_the_users_rule_with_keyframe_judges_and_margins():
    from precision import chunk_check as CC
    from precision.config import load_precision_config
    cfg = replace(load_precision_config().chunk_check, bootstrap=200)
    fac = _fac()
    # chunk 1's floor and ceiling move together by 14 cm (pose); chunk 3 has only 4 keyframes. The
    # sound chunks are larger than the off one: a chunk is judged against ALL the others pooled, and a
    # pool half made of the off chunk puts its median in the sound half's upper tail (a median's bias
    # on a mixed sample, the pre-2026-10-07 statistic had it too) — with 40 sound keyframes against 10
    # the reference is the sound level
    rows = _rows([20, 10, 20, 4], [0.0, 0.14, 0.0, 0.0], [3.0, 3.14, 3.0, 3.0])
    rep = CC.judge(rows, cfg, 1.0, log=QUIET, error_factor=fac)
    by = {c["chunk"]: c for c in rep["chunks"]}
    assert by[1]["verdict"] == "pose", by[1]["why"]
    fr = by[1]["floor_rule"]
    assert fr["departs"] is True and fr["n_judges"] == 10 and fr["min_judges"] == 5 and fr["error_factor"] == fac
    assert fr["margin_m"] > 0 and fr["ci_margin_m"] > 0 and fr["judges_margin"] == 5 and fr["failed"] == []
    assert by[0]["verdict"] == "ok" and by[2]["verdict"] == "ok", (by[0]["why"], by[2]["why"])
    assert abs(fr["median_delta_m"] - 0.14) < 0.02 and fr["required_m"] == pytest.approx(fac * fr["error_m"])
    assert fr["form"].startswith("two-sample") and fr["n_reference"] == 44
    assert by[1]["same_amount_rule"]["departs"] is False and by[1]["same_amount_rule"]["form"].startswith("paired")
    assert by[3]["verdict"] == "undecided" and "4 keyframe(s) judge" in by[3]["why"]
    assert by[0]["verdict"] == "ok" and by[0]["floor_rule"]["departs"] is False
    assert by[2]["verdict"] == "ok"
    assert rep["session"]["improvement_error_factor"] == fac and "decide_change" in rep["session"]["rule"]
    # a 1.5 cm floor move: significant over 10 keyframes at 1 cm noise, but under 2 x the 1 cm resolution
    rows2 = _rows([10, 10, 10], [0.0, 0.015, 0.0], [3.0, 3.0, 3.0])
    rep2 = CC.judge(rows2, cfg, 1.0, log=QUIET, error_factor=fac)
    c1 = {c["chunk"]: c for c in rep2["chunks"]}[1]
    assert c1["verdict"] == "ok" and "error" in c1["floor_rule"]["failed"] and c1["floor_rule"]["margin_m"] < 0
    # a real level change: the floor departs, the ceiling does not
    rows3 = _rows([10, 10, 10], [0.0, -0.14, 0.0], [3.0, 3.0, 3.0])
    rep3 = CC.judge(rows3, cfg, 1.0, log=QUIET, error_factor=fac)
    c1 = {c["chunk"]: c for c in rep3["chunks"]}[1]
    assert c1["verdict"] == "level" and c1["ceiling_rule"]["departs"] is False and rep3["to_correct"] == []


def test_the_session_report_carries_the_plane_share_instead_of_refusing(tmp_path):
    import test_precision_chunk_check as TC
    from precision import chunk_check as CC
    from precision.config import load_precision_config
    p = load_precision_config()
    pcfg = replace(p, chunk_check=replace(p.chunk_check, pixel_stride=1, min_points=50, bootstrap=100))
    _, chain = TC.build_session(tmp_path, "clean")
    rep = CC.run_check(tmp_path, pcfg, log=QUIET, chainage=chain)
    pl = rep["plane"]
    assert 0 < pl["inlier_frac"] <= 1 and pl["inlier_frac_ci"][0] <= pl["inlier_frac"] <= pl["inlier_frac_ci"][1]
    assert rep["params"]["improvement_error_factor"] == _fac() and "point 140" in rep["params"]["floor_plane"]
    assert all(c["verdict"] == "ok" for c in rep["chunks"]) and all("floor_rule" in c for c in rep["chunks"])


# ── point 45 / 56: the witness poses round-trip exactly ───────────────────

def test_witness_poses_round_trip_bit_exact(tmp_path):
    from precision.depth_sweep import _read_poses
    from repro import write_poses_exact
    rng = np.random.default_rng(3)
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, :3, :] = rng.standard_normal((5, 3, 4)) * np.pi
    p = write_poses_exact(tmp_path / "witness_poses.txt", poses)
    (tmp_path / "witness_frames.txt").write_text(" ".join(str(f) for f in range(5)) + "\n")
    frames, back = _read_poses(p, tmp_path / "witness_frames.txt")
    assert frames == [0, 1, 2, 3, 4] and np.array_equal(back, poses)
