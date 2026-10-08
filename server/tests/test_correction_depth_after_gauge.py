"""The depth correction from the visit closures, applied AFTER the precision gauge on
the RESIDUAL (USER 2026-09-29).

The gauge (precision/gauge.py, DA3 on every keyframe) already set the scale along
the walk. What the depth stage may still do there is what the closures MEASURED ON
THE CURRENT EPOCH say is left — and nothing else: the DA3 trend rows, the per-chunk
anchor rows and the unstamped absolute rows are not relative to that geometry and
would apply the gauge's drift a second time, so they stand down
(`scale_stage.stand_down_for_gauge`). The closures are relative; the minimum-norm
gauge of the scale graph keeps the session's size.

Covered here: an injected radial residual in one chunk is corrected; no residual is
identity (the floor still publishes its epoch); a tangential false identity cannot
move a chunk; the DA3 / anchor rows do not double-apply the gauge; stale closures do
not drive anything; depth and floor publish ONE epoch; the per-chunk factor pivots on
the first keyframe each chunk OWNS, so an overlapping plan keeps the walk continuous.

CHANGED 2026-10-08 with docs/plan_determinismo.md points 127 and 134: the depth solution is
applied only by THE USER'S RULE on the closures (leave-one-out judges, >= 5 of them, 95 % CI,
median improvement >= the factor x their error), so every fixture that expects an applied
depth carries at least 5 closures; the rows file is taken on its evidence STAMP, never on
the equality of an epoch number, so the fixture writes stamped rows (``stale=True`` writes
a stamp that is not the session's).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import correction.visit_drift_run as V                                   # noqa: E402
from correction import visit_drift as vd                                 # noqa: E402
from reconstruction.certify import scale_stage as S                      # noqa: E402
from reconstruction.loops.config import load_loops_config                # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg      # noqa: E402

N_KF = 30
FRAME_STEP = 10
# 50 % overlap, like the reconstruction's chunked plans: nearest-centre ownership
# gives chunk 0 keyframes 0-15 and chunk 1 keyframes 16-29
PLAN = {"chunk_ranges": [[0, 20], [10, 30]], "chunk_size": 20, "overlap": 10,
        "n_keyframes": N_KF}
C_INJECTED = 0.92          # chunk 1 reconstructed 8 % small against chunk 0
LOCK_DRIFT = 1.15          # what the DA3 anchors said at the lock (the gauge removed it);
                           # 2026-10-08: 1.25 -> 1.15 — the drift-rate ramp pinned at the
                           # start (point 127) puts the whole correction on the last chunk,
                           # and log 1.25 = 0.223 lies beyond the declared bound
                           # certify.scale.max_correction_log 0.2 (a refusal, not a judge)


def _scfg():
    return load_loops_config().certify.scale


def _gauge(out: Path, applied: bool = True) -> None:
    """gauge.json as F2 leaves it: stamped (plan point 146 — gauge_applied verifies the stamp
    against the session and the applied epoch against the live lineage)."""
    from correction.epoch import current_epoch, reconstruction_id_or_none
    from precision import gauge as G
    from precision.config import load_precision_config
    from reconstruction.loops.config import improvement_error_factor
    from config import cfg as raw_cfg
    params = G.gauge_params(load_precision_config().gauge, improvement_error_factor(raw_cfg))
    (out / "gauge.json").write_text(json.dumps(
        {"version": 1, "applied": applied, "params": params,
         "epoch_to": int(current_epoch(out)) if applied else None,
         G.GAUGE_STAMP_KEY: G.gauge_stamp(out.parent, params, reconstruction_id_or_none(out))}))


def _rows(out: Path, rows, epoch: int = 0, stale: bool = False, cfg=None) -> None:
    """scale_loop_rows.json as visit_drift writes it (stamped, point 134; ``cfg`` = the
    correction configuration the reader will verify the stamp with, None = the server's);
    ``epoch`` is the recorded measured_on_epoch (a record, not the freshness test); ``stale``
    writes a stamp that is not this session's (what a file measured on another geometry
    carries)."""
    V._stamp_out(out, "scale_loop_rows.json", {"rows": rows}, cfg=cfg)
    p = out / "scale_loop_rows.json"
    doc = json.loads(p.read_text())
    doc["measured_on_epoch"] = int(epoch)
    if stale:
        doc[V.EVIDENCE_STAMP_KEY] = dict(doc[V.EVIDENCE_STAMP_KEY], sha256="0" * 64)
    p.write_text(json.dumps(doc))


def _row(i, j, s_ab, residual_m=0.005, extent_m=1.0, iid=1, label="desk"):
    return {"instance_id": iid, "label": label, "i": int(i), "j": int(j),
            "s_ab": float(s_ab), "residual_m": float(residual_m), "extent_m": float(extent_m),
            "scale_trusted": True, "source": "visit_drift", "provenance": "tool_measured"}


def _closures(s_ab):
    """What six objects seen at the start (chunk 0) and again at the end
    (chunk 1) read when chunk 1 is `s_ab` times chunk 0 — radial, no tangential
    part (the silhouette instrument itself is tested in test_visit_drift_*). Six:
    the user's rule needs at least 5 judges (point 127)."""
    return [_row(4, 24, s_ab, iid=1, label="wall1"), _row(6, 26, s_ab, iid=2, label="box1"),
            _row(8, 28, s_ab, iid=3, label="col1"), _row(3, 23, s_ab, iid=4, label="shelf1"),
            _row(5, 25, s_ab, iid=5, label="wall2"), _row(7, 27, s_ab, iid=6, label="box2")]


def _lock_diag(out: Path, frames, owner, drift_by_chunk, rounded_history: bool = True):
    """scale_diagnostics.json as the precision flow leaves it: the lock-time DA3
    anchors (carrying the drift the gauge removed) and an `epochs` entry appended
    by the gauge's epoch with NO depth factor, its agreements rounded to 6 decimals
    (`diagnose.regenerate_scale_diagnostics`)."""
    rng = np.random.default_rng(3)
    anchors = []
    for g, f in enumerate(frames):
        s_f = float(drift_by_chunk[int(owner[g])] * (1.0 + rng.normal(0, 0.002)))
        anchors.append({"num": int(f), "s_f": s_f, "n_px": 5000,
                        "z_median_omega": 2.5, "d_median_m": 2.5})
    diag = {"version": 1, "s_applied": 1.0, "mode_used": "global_median",
            "anchors": {"count": len(anchors), "mad_rel": 0.01, "frames": anchors}}
    if rounded_history:
        diag["epochs"] = [{"epoch": 1, "correction_id": "gauge",
                           "agreement": {str(a["num"]): round(a["s_f"], 6) for a in anchors}}]
    (out / "scale_diagnostics.json").write_text(json.dumps(diag))


def _mock_session(n_kf=N_KF, step=0.5):
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    poses[:, 0, 3] = np.arange(n_kf) * step
    poses[:, 1, 3] = 1.6
    return SimpleNamespace(n_kf=n_kf, frames=[k * FRAME_STEP for k in range(n_kf)],
                           poses=poses)


def _mock_out(tmp_path, plan=PLAN):
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunk_plan.json").write_text(json.dumps(plan))
    return out


# ── the rows: only what is measured on the current geometry ─────────────

def test_da3_trend_and_anchor_rows_do_not_double_apply_the_gauge(tmp_path):
    out = _mock_out(tmp_path)
    sess = _mock_session()
    _ranges, owner = S.chunk_of_keyframes(out, N_KF)
    _lock_diag(out, sess.frames, owner, {0: 1.0, 1: LOCK_DRIFT})
    # control — no gauge: the lock-time trend is still a row and moves the chunks (the
    # closures, the judges of point 127, corroborate it: chunk 1 reads small by the drift)
    _rows(out, _closures(1.0 / LOCK_DRIFT))
    ctl = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert ctl["applied"] and ctl["da3_trend_rows"], ctl
    assert ctl["r"][1] / ctl["r"][0] > 1.05, ctl["r"]
    # the gauge applied: the drift it removed is not asked for again — the closures measured
    # on the current geometry say nothing is left
    _rows(out, _closures(1.0))
    _gauge(out)
    rep = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert not rep["applied"] and rep["r"] == [1.0, 1.0], rep
    sd = rep["stood_down_for_gauge"]
    assert sd["anchor_rows"] == [0, 1] and sd["absolute_rows"] == 0
    assert len(sd["da3_trend_rows"]) == 1
    assert abs(sd["da3_trend_rows"][0]["log_r"] - np.log(LOCK_DRIFT)) < 0.01
    assert rep["da3_trend_rows"] == [] and rep["anchor_rows"] == {}
    assert rep["loop_rows"], "the current closures are what the graph saw"
    json.dumps(rep)                                             # the acta carries it


def test_stale_closures_do_not_drive_the_graph_after_the_gauge(tmp_path):
    out = _mock_out(tmp_path)
    sess = _mock_session()
    (out / "geometry_epoch.json").write_text(json.dumps(
        {"epoch": 2, "parent_epoch": 1, "kind": "new_cloud", "correction_id": "x"}))
    _gauge(out)
    _rows(out, _closures(0.80), epoch=1, stale=True)   # measured on another geometry
    rep = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert not rep["applied"] and rep["loop_rows"] == [], rep
    assert rep["stood_down_for_gauge"]["visit_drift_rows_stale"] == {
        "n": 6, "measured_on_epoch": 1}
    assert rep["loop_rows_refused"]["why"], rep["loop_rows_refused"]


def test_solve_depth_remeasures_a_stale_stamp_and_never_solves_the_old_rows(tmp_path, monkeypatch):
    out = _mock_out(tmp_path)
    (out / "geometry_epoch.json").write_text(json.dumps(
        {"epoch": 2, "parent_epoch": 1, "kind": "new_cloud", "correction_id": "x"}))
    _gauge(out)
    _rows(out, _closures(0.80), epoch=1, stale=True)
    called = []

    def _measure(o, c, r, log=print):
        called.append(int(json.loads((Path(o) / "scale_loop_rows.json").read_text())
                          ["measured_on_epoch"]))
        return {"scale_rows": [], "rejected": []}  # nothing measurable on epoch 2
    monkeypatch.setattr(V, "measure_epoch", _measure)
    monkeypatch.setattr(V, "_repeatability_m", lambda o, d, log: 0.05)
    k_kf, t_kf, rep = V.solve_depth(out, log=lambda m: None, cfg=make_correction_cfg())
    assert k_kf is None and t_kf is None and not rep["applied"]
    assert "epoch 2" in rep["reason"] and "not used" in rep["reason"], rep
    assert called == [1], "the stale stamp must trigger a re-measurement"


def test_a_tangential_false_identity_cannot_move_a_chunk(tmp_path):
    """pccr epoch 6: a monitor matched to another monitor 2.8 m away (286 cm
    tangential, 64 cm radial). `scale_rows` prices the tangential part into the
    row's σ; alone it moves nothing, beside the genuine closures it moves the
    solution by a negligible amount."""
    C = np.stack([np.arange(N_KF) * 0.5, np.full(N_KF, 1.6), np.zeros(N_KF)], 1)
    poses = np.tile(np.eye(4), (N_KF, 1, 1))
    poses[:, :3, 3] = C
    rng = np.random.default_rng(0)

    def blob(c, n=400):
        return np.asarray(c, float) + rng.uniform(-0.3, 0.3, (n, 3))

    def cand(iid, label, A, B, va=(2, 6), vb=(22, 26)):
        ks = np.concatenate([rng.integers(va[0], va[1] + 1, len(A)),
                             rng.integers(vb[0], vb[1] + 1, len(B))])
        return (SimpleNamespace(copies=[A, B], visits=[va, vb], instance_id=iid, label=label,
                                oid=iid - 1, points=np.arange(len(ks))), ks)

    rows = {}
    # the genuine closures: copy B sits 10 % short ALONG the rays of its cameras — five of
    # them (the user's rule judges with at least 5 closures, point 127)
    cam_b = C[22:27].mean(0)
    genuine = []
    for n_, truth in enumerate([np.array([12.0, 0.8, 3.5]), np.array([11.0, 1.2, 2.5]),
                                np.array([13.0, 0.5, 4.0]), np.array([12.5, 1.0, -2.0]),
                                np.array([11.5, 0.7, -3.0])]):
        A = blob(truth)
        B = blob(cam_b + (truth - cam_b) / 1.10)
        (c, ks) = cand(1 + n_, "desk", A, B)
        dr = SimpleNamespace(t=A.mean(0) - B.mean(0), worst_disagreement=0.01)
        genuine.append(vd.scale_rows([(c, dr)], poses, ks, log=lambda m: None)[0])
    rows["desk"] = genuine[0]
    truth = np.array([12.0, 0.8, 3.5])
    # the false identity: another monitor 2.8 m to the side and 0.6 m nearer
    A = blob(truth + np.array([2.8, 0.0, 0.0]))
    B = blob(truth + np.array([0.0, 0.0, 0.6]))
    (c, ks) = cand(9, "monitor", A, B)
    dr = SimpleNamespace(t=A.mean(0) - B.mean(0), worst_disagreement=0.01)
    rows["monitor"] = vd.scale_rows([(c, dr)], poses, ks, log=lambda m: None)[0]
    mon, desk = rows["monitor"], rows["desk"]
    assert mon["tangential_m"] > 2.5 and abs(np.log(mon["s_ab"])) > 0.1, mon
    assert mon["residual_m"] / mon["extent_m"] > 20 * desk["residual_m"] / desk["extent_m"]

    out = _mock_out(tmp_path)
    sess = _mock_session()
    _gauge(out)
    _rows(out, [mon])
    alone = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert not alone["applied"] and alone["r"] == [1.0, 1.0], alone
    _rows(out, genuine)
    good = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    _rows(out, genuine + [mon])
    both = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert good["applied"] and both["applied"]
    assert np.max(np.abs(np.log(both["r"]) - np.log(good["r"]))) < 0.002, (good["r"], both["r"])


def test_pccr_epoch6_closures_the_monitor_does_not_move_the_walk(tmp_path):
    """The rows pccr measured on epoch 6 (scale_loop_rows.json), on its own chunk
    plan: with and without the false monitor the per-chunk factors agree to 0.1 %."""
    plan = {"chunk_ranges": [[0, 83], [42, 125], [84, 167], [126, 209], [168, 251],
                             [210, 289]], "chunk_size": 83, "overlap": 41, "n_keyframes": 289}
    out = _mock_out(tmp_path, plan)
    sess = _mock_session(289, 0.06)
    _gauge(out)
    chair = _row(4, 275, 0.9823934639702284, 0.04147365062129851, 0.7506464725357292, 100, "chair")
    monitor = _row(10, 271, 1.2055422981406194, 2.864091322872773, 2.1950020562916133, 120, "monitor")
    desk_a = _row(5, 276, 0.9494368951481736, 0.14880905739557532, 1.3337705381357634, 166, "desk")
    desk_b = _row(7, 278, 0.9434048538705396, 0.07149632858811254, 1.2212902202386302,
                  166, "desk")
    _rows(out, [chair, desk_a, desk_b])
    good = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    _rows(out, [chair, monitor, desk_a, desk_b])
    both = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert both["applied"] == good["applied"], "the monitor must not change the verdict"
    assert np.max(np.abs(np.log(both["r"]) - np.log(good["r"]))) < 0.001, (good["r"], both["r"])
    # the start <-> end closures are radial and one direction: the end of the walk grows
    assert both["r"][-1] >= both["r"][0]
    assert abs(float(np.mean(np.log(both["r"])))) < 1e-9, "the gauge's size is kept"


# ── the chunk unit ───────────────────────────────────────────────────────

def test_an_overlapping_plan_keeps_the_walk_continuous():
    """Each chunk pivots on the first keyframe it OWNS (the records' `chunk`
    field): every step of the walk is scaled by the factor of the chunk that
    owns its first keyframe — no camera jump at an ownership boundary."""
    plan_ranges = [(0, 20), (10, 30), (20, 40)]
    sess = _mock_session(40)
    rng = np.random.default_rng(1)
    sess.poses[:, :3, 3] += np.cumsum(rng.normal(0, 0.05, (40, 3)), 0)
    S._vendor_on_path()
    from loop_utils.metric_lock import frame_owner
    owner = frame_owner(plan_ranges, 40)
    r = np.array([1.0, 1.10, 0.95])
    _k, t = S.scale_transforms(sess, plan_ranges, owner, r)
    C = sess.poses[:, :3, 3]
    old = np.diff(C, axis=0)
    new = np.diff(C + t, axis=0)
    want = old * r[owner[:-1]][:, None]
    assert np.abs(new - want).max() < 1e-12, np.abs(new - want).max(axis=1)
    assert np.allclose(t[0], 0.0)


# ── the whole stage on a real (synthetic) session ────────────────────────

def _scene(tmp_path, c=C_INJECTED):
    """A session whose chunk 1 is `c` times chunk 0 (injected with the
    known-answer machinery, which uses the stage's own `scale_transforms`), after
    a gauge whose lock-time DA3 drift would ask for LOCK_DRIFT again."""
    from reconstruction.quality.known_answer import perturb_chunk
    scene = build_scene(tmp_path, n_kf=N_KF, chunk_plan=PLAN)
    out = scene.output_dir
    if c != 1.0:
        perturb_chunk(out, "last", 0.0, 0.0, c, log=lambda m: None)
    _ranges, owner = S.chunk_of_keyframes(out, N_KF)
    _lock_diag(out, scene.frames, owner, {0: 1.0, 1: LOCK_DRIFT})
    _gauge(out)
    return scene, owner


def test_a_radial_residual_in_one_chunk_is_corrected_after_the_gauge(tmp_path):
    from correction.session import load_session
    from reconstruction.certify.run import transformed_session
    scene, owner = _scene(tmp_path)
    out = scene.output_dir
    _rows(out, _closures(C_INJECTED), cfg=make_correction_cfg())
    k_kf, t_kf, srep = V.solve_depth(out, log=lambda m: None, cfg=make_correction_cfg())
    assert k_kf is not None and srep["applied"], srep
    r = np.asarray(srep["r"])
    # the relative factor the closures measure, within the seam prior's pull
    assert abs(np.log(r[1] * C_INJECTED / r[0])) < 0.10 * abs(np.log(C_INJECTED)), r
    assert abs(float(np.mean(np.log(r)))) < 1e-9, "the gauge's size is kept"
    assert srep["stood_down_for_gauge"]["da3_trend_rows"], "the lock drift stood down"
    # the geometry: the corrected cloud is the truth up to ONE similarity (the
    # min-norm split of the correction: chunk 0 scaled by r0 about camera 0)
    sess = load_session(out)
    xyz = transformed_session(sess, np.tile(np.eye(3), (N_KF, 1, 1)), t_kf, k_kf).xyz
    c0 = scene.poses_true[0][:3, 3]
    expect = c0 + r[0] * (scene.xyz_true - c0)
    ch1 = owner[scene.ks] == 1
    before = np.linalg.norm(sess.xyz - scene.xyz_true, axis=1)[ch1]
    after = np.linalg.norm(xyz - expect, axis=1)[ch1]
    assert np.median(before) > 0.2, np.median(before)
    assert np.median(after) < 0.15 * np.median(before), (np.median(before), np.median(after))
    assert np.linalg.norm(xyz - expect, axis=1)[~ch1].max() < 1e-4


def _epochs(out):
    from correction import ledger
    from correction.epoch import current_epoch
    return current_epoch(out), ledger.applied_runs(out), sorted(
        p.name for p in (out / "corrections").glob("epoch_*.npz"))


def test_depth_and_floor_compose_into_one_epoch_after_the_gauge(tmp_path):
    scene, owner = _scene(tmp_path)
    out = scene.output_dir
    cfg = make_correction_cfg(**{"gates.mode": "advisory"})
    _rows(out, _closures(C_INJECTED), cfg=cfg)
    assert cfg.visit_drift.skip_when_gauge_applied is False
    res = V.run(tmp_path, log=lambda m: None, cfg=cfg)
    names = [s["stage"] for s in res["stages"]]
    assert names == ["depth", "floor_plane+depth"], res["stages"]
    assert res["stages"][0]["after_gauge"] is True and res["stages"][0]["applied"] is True
    assert res["stages"][1]["status"] == "applied", res["stages"]
    epoch, runs, npz = _epochs(out)
    assert epoch == 1 and len(runs) == 1 and runs[0]["kind"] == "floor", runs
    assert npz == ["epoch_1.npz"]
    k = np.load(out / "corrections" / "epoch_1.npz")["k_kf"]
    assert np.ptp(k[owner == 0]) < 1e-12 and np.ptp(k[owner == 1]) < 1e-12
    ratio = float(k[owner == 1][0] / k[owner == 0][0])
    assert abs(np.log(ratio * C_INJECTED)) < 0.10 * abs(np.log(C_INJECTED)), ratio
    json.dumps(res)


def test_the_switch_still_stands_the_depth_down_when_asked(tmp_path):
    scene, owner = _scene(tmp_path)
    out = scene.output_dir
    cfg = make_correction_cfg(**{"gates.mode": "advisory",
                                 "visit_drift.skip_when_gauge_applied": True})
    _rows(out, _closures(C_INJECTED), cfg=cfg)
    res = V.run(tmp_path, log=lambda m: None, cfg=cfg)
    assert [s["stage"] for s in res["stages"]] == ["depth", "floor_plane"], res["stages"]
    dep = res["stages"][0]
    assert dep["applied"] is False and dep["after_gauge"] is True
    assert "skip_when_gauge_applied" in dep["reason"], dep
    k = np.load(out / "corrections" / "epoch_1.npz")["k_kf"]
    assert np.all(k == 1.0)


def test_no_residual_is_identity_and_the_floor_still_applies(tmp_path):
    scene, _owner = _scene(tmp_path, c=1.0)
    out = scene.output_dir
    _rows(out, _closures(1.0), cfg=make_correction_cfg(**{"gates.mode": "advisory"}))
    res = V.run(tmp_path, log=lambda m: None,
                cfg=make_correction_cfg(**{"gates.mode": "advisory"}))
    assert [s["stage"] for s in res["stages"]] == ["depth", "floor_plane"], res["stages"]
    # the acta says WHY there is no depth: it ran after the gauge, on the
    # closures alone, and solved to identity — not "the switch stood it down"
    dep = res["stages"][0]
    assert dep["applied"] is False and dep["after_gauge"] is True, dep
    assert dep["stood_down_for_gauge"]["da3_trend_rows"], dep
    assert "within the row" in dep["reason"], dep
    assert res["stages"][1]["status"] == "applied"
    epoch, runs, npz = _epochs(out)
    assert epoch == 1 and len(runs) == 1 and npz == ["epoch_1.npz"]
    k = np.load(out / "corrections" / "epoch_1.npz")["k_kf"]
    assert np.all(k == 1.0), "no depth epoch: the gauge's drift was not applied again"


@pytest.mark.parametrize("n_rows", [0, 3])
def test_a_single_pass_session_is_one_degree_of_freedom(tmp_path, n_rows):
    """DECLARED: one chunk — its only degree of freedom is its size, which the
    gauge holds; closures inside it cannot move anything."""
    out = tmp_path / "output"
    out.mkdir()
    sess = _mock_session()
    _gauge(out)
    _rows(out, _closures(0.9)[:n_rows])
    rep = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert rep["n_chunks"] == 1 and not rep["applied"] and rep["r"] == [1.0]
    assert rep["loop_rows_intra_chunk"] == n_rows
    assert "ONE chunk" in rep["reason"]


# ── the self-validation after the gauge (earn_the_right) ────────────────

PCCR_PLAN = {"chunk_ranges": [[0, 83], [42, 125], [84, 167], [126, 209], [168, 251],
                              [210, 289]], "chunk_size": 83, "overlap": 41, "n_keyframes": 289}
# pccr epoch 6's false identity, as scale_loop_rows.json holds it
MONITOR = _row(10, 271, 1.2055422981406194, 2.864091322872773, 2.1950020562916133, 120,
               "monitor")
# a second one of the same kind (the light fixture 5 m away; synthetic numbers)
FIXTURE = _row(12, 280, 0.80, 5.1, 1.0, 130, "light_fixture")


def _pccr(tmp_path, gauge=True):
    out = _mock_out(tmp_path, PCCR_PLAN)
    if gauge:
        _gauge(out)
    return out, _mock_session(289, 0.06)


def _genuine(n):
    """`n` radial start <-> end closures agreeing on +3.9 % — enough (>= 9) for
    `earn_the_right` to hold some out and judge the ramp."""
    return [_row(2 + i, 270 + i, 1 / 1.039, 0.01, 1.0, 200 + i, "desk") for i in range(n)]


def test_after_the_gauge_the_ramp_keeps_the_gauges_size(tmp_path):
    """The ramp used to be pinned at chunk 0 (k = 1 + eps*d from the start):
    after the gauge that resized the session by ~2 % against the gauge's
    absolute scale. It keeps the minimum-norm size, as the free solution does."""
    out, sess = _pccr(tmp_path)
    _rows(out, _genuine(12))
    rep = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert rep["earned"]["verdict"] == "ramp" and rep["earned"]["hold_size"], rep["earned"]
    assert rep["applied"], rep
    r = np.asarray(rep["r"])
    assert abs(float(np.mean(np.log(r)))) < 1e-9, r
    assert np.all(np.diff(r) > 0), "the end of the walk grows against the start"
    # control, no gauge: the drift-rate ramp still starts exact at chunk 0
    out0, _ = _pccr(tmp_path / "nogauge", gauge=False)
    _rows(out0, _genuine(12))
    ctl = S.solve_scale_stage(out0, sess, [], _scfg(), log=lambda m: None)
    assert ctl["earned"]["verdict"] == "ramp" and not ctl["earned"]["hold_size"]
    assert ctl["r"][0] == 1.0
    assert np.allclose(np.diff(np.log(ctl["r"])), np.diff(np.log(r)))


@pytest.mark.parametrize("false_at", [(12,), (2, 5)], ids=["in_the_fit", "held_out"])
def test_false_identities_do_not_steer_the_judge_after_the_gauge(tmp_path, false_at):
    """A closure the graph prices out (σ 1.3-5 against 0.01) must not decide the
    ramp's slope (in the fitted rows) nor the verdict (in the held-out rows)."""
    out, sess = _pccr(tmp_path)
    rows = _genuine(12)
    _rows(out, rows)
    good = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    for pos, bad in zip(false_at, (MONITOR, FIXTURE)):
        rows.insert(pos, bad)
    _rows(out, rows)
    both = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None)
    assert good["applied"] and both["applied"] == good["applied"], both
    assert both["earned"]["verdict"] == good["earned"]["verdict"], (good["earned"],
                                                                   both["earned"])
    assert np.max(np.abs(np.log(both["r"]) - np.log(good["r"]))) < 0.002, (good["r"],
                                                                            both["r"])


def test_weighted_median_is_the_median_on_equal_weights():
    for v in ([3.0, 1.0, 2.0], [4.0, 1.0, 3.0, 2.0], [0.5]):
        assert S._weighted_median(v, [7.0] * len(v)) == float(np.median(v))
    assert S._weighted_median([0.0, 0.0, 9.0, 9.0], [1e4, 1e4, 1.0, 1.0]) == 0.0


def test_a_depth_applied_alone_after_the_gauge_says_what_drove_it(tmp_path, monkeypatch):
    """The floor failed, the depth goes out alone: its ledger evidence names the
    closures of the current epoch, not "cross-checked against the DA3 anchors"
    (every anchor and trend row stood down after the gauge)."""
    import correction.run as CR
    from correction import ledger
    scene, _owner = _scene(tmp_path)
    out = scene.output_dir
    _rows(out, _closures(C_INJECTED), cfg=make_correction_cfg(**{"gates.mode": "advisory"}))

    def _no_floor(*a, **k):
        raise RuntimeError("floor unavailable in this test")
    monkeypatch.setattr(CR, "run_floor", _no_floor)
    res = V.run(tmp_path, log=lambda m: None,
                cfg=make_correction_cfg(**{"gates.mode": "advisory"}))
    assert [s["stage"] for s in res["stages"]] == ["depth", "floor_plane", "depth_alone"]
    assert res["stages"][2]["status"] == "applied", res["stages"]
    run = ledger.applied_runs(out)[-1]
    ev = " ".join(d.get("evidence", "") for d in run["diagnosis"])
    assert "closures measured on epoch 0" in ev and "stood down" in ev, ev
    assert "cross-checked against the DA3 anchors" not in ev, ev
