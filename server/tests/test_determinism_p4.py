"""Wave 2, package P4 (corrections + certification) of docs/plan_determinismo.md, 2026-10-08:
the mechanisms behind points 127, 130, 131, 132, 134, 135, 136, 137, 142, 144, 147, 160 and 166,
each on the REAL code path the pipeline calls, synthetic, CPU only."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction import chain as CH                                       # noqa: E402
from correction import ledger                                            # noqa: E402
from correction import visit_drift as vd                                 # noqa: E402
from correction import visit_drift_run as V                              # noqa: E402
from correction.epoch import EPOCH_FILE, corrections_summary             # noqa: E402
from reconstruction.certify import scale_stage as S                      # noqa: E402
from reconstruction.loops.config import load_loops_config                # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg      # noqa: E402

N_KF = 30
FRAME_STEP = 10
PLAN = {"chunk_ranges": [[0, 20], [10, 30]], "chunk_size": 20, "overlap": 10, "n_keyframes": N_KF}


def _scfg():
    return load_loops_config().certify.scale


def _graph():
    return load_loops_config().graph


def _mock_session(n_kf=N_KF, step=0.5):
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    poses[:, 0, 3] = np.arange(n_kf) * step
    poses[:, 1, 3] = 1.6
    return SimpleNamespace(n_kf=n_kf, frames=[k * FRAME_STEP for k in range(n_kf)], poses=poses)


def _mock_out(tmp_path, plan=PLAN):
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunk_plan.json").write_text(json.dumps(plan))
    return out


def _row(i, j, s_ab, residual_m=0.005, extent_m=1.0, iid=1, label="desk", **extra):
    return {"instance_id": iid, "label": label, "i": int(i), "j": int(j),
            "s_ab": float(s_ab), "residual_m": float(residual_m), "extent_m": float(extent_m),
            "scale_trusted": True, "source": "visit_drift", "provenance": "tool_measured", **extra}


def _rows(out, rows):
    V._stamp_out(out, "scale_loop_rows.json", {"rows": rows})


# ── point 127: the judge is the closures, by the user's rule ───────────────

def test_127_the_depth_solution_needs_five_judges_and_is_order_invariant(tmp_path):
    out = _mock_out(tmp_path)
    sess = _mock_session()
    # three closures: fewer than the five judges the rule needs → identity, declared
    _rows(out, [_row(4, 24, 1 / 1.08, iid=i) for i in range(3)])
    rep = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None, graph=_graph())
    assert not rep["applied"] and rep["r"] == [1.0, 1.0]
    assert rep["earned"]["verdict"] == "identity" and "judges" in rep["earned"]["judge"]["failed"]
    # six closures agreeing on +8 %: applied by the rule — and the verdict does not depend on
    # the order of the rows (the old held-out split by list position flipped in 27 of 40 cases)
    rows = [_row(2 + i, 22 + i, 1 / 1.08, iid=10 + i, label=f"obj{i}") for i in range(6)]
    verdicts, factors = set(), []
    for shift in range(len(rows)):
        _rows(out, rows[shift:] + rows[:shift])
        rep = S.solve_scale_stage(out, sess, [], _scfg(), log=lambda m: None, graph=_graph())
        assert rep["applied"], rep["earned"]
        verdicts.add(rep["earned"]["verdict"])
        factors.append(rep["r"])
    assert len(verdicts) == 1 and all(np.allclose(f, factors[0]) for f in factors)
    assert abs(np.log(factors[0][1] / factors[0][0]) - np.log(1.08)) < 0.02
    j = rep["earned"]["judge"]
    assert j["n_judges"] == 6 and j["significant"] and j["beyond_error"]
    assert rep["earned"]["error_unit"].startswith("closure sigma")
    # the bound stays a declared gate, never a judge
    assert rep["gate"]["name"] == "max_correction_log" and rep["gate"]["passed"]


# ── point 147: an anchor moved when |log K| clears the anchors' own error ──

def test_147_anchor_rows_stand_down_by_the_applied_factor_not_float_equality(tmp_path):
    out = _mock_out(tmp_path)
    frames = [k * FRAME_STEP for k in range(N_KF)]
    _, owner = S.chunk_of_keyframes(out, N_KF)
    anchors = [{"num": f, "s_f": 1.0 + 1e-7 * (i % 3), "n_px": 5000, "z_median_omega": 2.5,
                "d_median_m": 2.5} for i, f in enumerate(frames)]
    (out / "scale_diagnostics.json").write_text(json.dumps({
        "version": 1, "s_applied": 1.0, "mode_used": "global_median",
        "anchors": {"count": len(anchors), "mad_rel": 0.01, "frames": anchors},
        # a rounded history, the trap of the old any(x != 1.0) test
        "epochs": [{"epoch": 1, "correction_id": "x",
                    "agreement": {str(a["num"]): round(a["s_f"] * 1.0000004, 6) for a in anchors}}]}))
    k = np.ones(N_KF)
    k[owner == 1] = 1.05                         # chunk 1 had a 5 % depth factor applied
    s, n, moved, margins = S.anchor_rows(out, owner, frames, applied_k=k, error_factor=1.1)
    assert moved == {0: False, 1: True}, (moved, margins)
    assert margins[1]["margin"] > 0 > margins[0]["margin"]
    assert abs(margins[1]["anchor_error"] - 0.01 / np.sqrt(n[1])) < 1e-12


# ── point 130: a continuous peak with its ambiguity ────────────────────────

def _bars(offsets, n_per=400, seed=0, width=0.06, length=2.0):
    rng = np.random.default_rng(seed)
    pts = []
    for o in offsets:
        pts.append(np.stack([rng.uniform(o, o + width, n_per), rng.uniform(0, length, n_per),
                             np.zeros(n_per)], 1))
    return np.concatenate(pts)


def _lattice_bars(offsets, width=0.06, length=2.0, step=0.005):
    """Bars sampled on one lattice: every bar is the same point set shifted — a periodic
    object whose two correlation peaks are equal to the last bit."""
    u = np.arange(0.0, width, step)
    v = np.arange(0.0, length, step)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    one = np.stack([U.ravel(), Vv.ravel(), np.zeros(U.size)], 1)
    return np.concatenate([one + np.array([o, 0.0, 0.0]) for o in offsets])


def test_130_the_correlation_peak_is_sub_cell_and_a_repeated_object_pays_its_ambiguity():
    ax, ay = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # one bar, shifted by 15.3 mm: the shift reads to a fraction of the 1 cm cell
    A = _bars([0.0])
    B = A + np.array([0.0153, 0.0, 0.0])
    du, dv, peak, info = vd.view_shift(A, B, ax, ay, 0.01, 1.2, 5, 2.0, error_factor=1.1)
    assert abs(abs(du) - 0.0153) < 0.004 and abs(dv) < 0.004, (du, dv)
    assert abs(du * 100 - round(du * 100)) > 0.05, "the shift is not quantised to a whole cell"
    assert info["ambiguity_m"] == 0.0 and info["peak2"] is not None and info["margin"] > 0
    # the skeptic's rails (point 130): three bars of period 0.40 m against two bars shifted
    # by +0.15 — two bars overlap at +0.15 and two at −0.25; the second peak sits one period
    # away, within the correlation's own sampling noise of the first: the view is ambiguous
    # and pays the separation instead of the argmax deciding (it read −0.25 in 3 of 8 draws)
    A = _lattice_bars([0.0, 0.4, 0.8])
    B = _lattice_bars([0.0, 0.4]) + np.array([0.15, 0.0, 0.0])
    du, dv, peak, info = vd.view_shift(A, B, ax, ay, 0.01, 1.2, 5, 2.0, error_factor=1.1)
    assert info["peak2"] is not None and abs(info["separation_m"] - 0.4) < 0.03, info
    assert info["ambiguity_m"] > 0.0 and info["margin"] <= 0.0, info
    assert info["noise_peaks"] is not None and info["noise"] >= info["noise_mad"]
    # the same rails sampled at random (the skeptic's draws): the peaks' difference and the
    # noise are both measured — whatever the draw, the record says which won by how much
    A = _bars([0.0, 0.4, 0.8], seed=1)
    B = _bars([0.0, 0.4], seed=2) + np.array([0.15, 0.0, 0.0])
    du, dv, peak, info = vd.view_shift(A, B, ax, ay, 0.01, 1.2, 5, 2.0, error_factor=1.1)
    assert abs(info["separation_m"] - 0.4) < 0.03 and info["noise_peaks"] > 0.0, info
    assert (info["ambiguity_m"] > 0.0) == (info["margin"] <= 0.0)
    # DETERMINISM: the same silhouettes, the same bytes
    again = vd.view_shift(A, B, ax, ay, 0.01, 1.2, 5, 2.0, error_factor=1.1)
    assert again[0] == du and again[1] == dv and again[3] == info


def test_130_obb_axes_use_the_principal_direction_only_when_it_clears_its_error():
    rng = np.random.default_rng(0)
    up = np.array([0.0, 1.0, 0.0])
    # a long object: the principal direction wins, its sign fixed
    P = np.stack([rng.uniform(-2, 2, 4000), rng.uniform(0, 0.5, 4000), rng.uniform(-0.2, 0.2, 4000)], 1)
    axes, info = vd.obb_axes_info(P, up, 1.1)
    assert info["rule"] == "principal" and info["margin"] > 0
    assert abs(axes[0] @ np.array([1.0, 0.0, 0.0])) > 0.99 and axes[0][0] > 0
    # a square table: the two eigenvalues are within their error — the stable rule, declared
    Q = np.stack([rng.uniform(-1, 1, 4000), rng.uniform(0, 0.5, 4000), rng.uniform(-1, 1, 4000)], 1)
    axes2, info2 = vd.obb_axes_info(Q, up, 1.1)
    assert info2["rule"] == "world_x" and info2["margin"] <= 0
    assert np.allclose(axes2[0], [1.0, 0.0, 0.0])
    assert np.array_equal(vd.obb_axes_info(Q, up, 1.1)[0], axes2)


def test_130_a_non_converging_refinement_returns_the_mean_of_its_cycle():
    t = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
         np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    mean, period = vd._cycle_mean(t, 0.01)
    assert period == 2 and np.allclose(mean, [0.5, 0.0, 0.0])


# ── point 131: the rivals, exactly ─────────────────────────────────────────

def test_131_rivals_are_counted_on_every_point_without_a_generator():
    rng = np.random.default_rng(3)
    A = rng.uniform(0, 1, (6000, 3))
    xyz = np.concatenate([A, A + np.array([1.3, 0, 0]), A + np.array([3.0, 0, 0])])
    pm = {0: np.arange(0, 6000), 1: np.arange(6000, 12000), 2: np.arange(12000, 18000)}
    labels = {0: "tile", 1: "tile", 2: "tile"}
    n, hits, margins = vd.ambiguity(A, 0, "tile", pm, labels, xyz, magnitude_m=0.5)
    # masklet 1's nearest point is ~0.30 m away (bar 0.5: a rival, its margin recorded);
    # masklet 2's ~2.0 m (clear, not listed)
    assert (n, hits) == (1, [1]) and set(margins) == {1}
    assert abs(margins[1] - (0.5 - 0.3)) < 0.02
    # EXACT: the nearest pair of the full sets, not a subsample's
    from scipy.spatial import cKDTree
    exact = float(cKDTree(A).query(xyz[pm[1]], k=1)[0].min())
    assert margins[1] == 0.5 - exact
    # a masklet exactly AT the bar is a rival (the bar is inclusive)
    assert vd.ambiguity(A, 0, "tile", pm, labels, xyz, magnitude_m=exact)[1] == [1]
    assert vd.ambiguity(A, 0, "tile", pm, labels, xyz, magnitude_m=0.5) == (n, hits, margins)


# ── points 132 / 144: every visit against the earliest, split by chunk shares ──

def test_132_144_closures_are_attributed_by_the_chunk_shares_of_their_copies(tmp_path):
    out = _mock_out(tmp_path)
    sess = _mock_session()
    _, owner = S.chunk_of_keyframes(out, N_KF)
    rng = np.random.default_rng(0)
    C = sess.poses[:, :3, 3]

    def blob(c, n=300):
        return np.asarray(c, float) + rng.uniform(-0.3, 0.3, (n, 3))

    truth = np.array([12.0, 0.8, 3.5])
    # three visits: the earliest at kf 2-6, then kf 13-18 (half owned by each chunk: the
    # ownership boundary lies at 15) and kf 22-26 — each later visit its own row (point 144)
    visits = [(2, 6), (13, 18), (22, 26)]
    copies, ks = [], []
    for (a, b) in visits:
        cam = C[a:b + 1].mean(0)
        copies.append(blob(cam + (truth - cam) / 1.10) if a > 2 else blob(truth))
        ks.append(rng.integers(a, b + 1, 300))
    ks = np.concatenate(ks)
    cand = SimpleNamespace(copies=copies, visits=visits, instance_id=1, label="desk", oid=0,
                           points=np.arange(len(ks)))
    kept = []
    for v in visits[1:]:
        dr = SimpleNamespace(t=copies[0].mean(0) - copies[visits.index(v)].mean(0),
                             worst_disagreement=0.01, visit_a=visits[0], visit_b=v)
        kept.append((cand, dr))
    rows = vd.scale_rows(kept, sess.poses, ks, log=lambda m: None, owner=owner)
    assert [r["visit_b"] for r in rows] == [[13, 18], [22, 26]]
    assert rows[0]["chunks_a"] == {"0": 1.0} and set(rows[0]["chunks_b"]) == {"0", "1"}
    assert abs(sum(rows[0]["chunks_b"].values()) - 1.0) < 1e-12
    assert rows[1]["chunks_b"] == {"1": 1.0}
    # the graph's rows: the mid visit is a pair row at the share's weight, its intra share weighs nothing
    closures, n_intra = S._closure_rows(rows, owner, 0.01)
    assert n_intra == 0 and len(closures) == 2
    mid = closures[0]
    assert [p[:2] for p in mid["pairs"]] == [(0, 1)] and 0.0 < mid["pairs"][0][2] < 1.0
    assert abs(mid["pairs"][0][2] + mid["intra_share"] - 1.0) < 1e-12
    # the prediction of a row from per-chunk factors follows the shares, continuously
    x = np.array([0.0, 0.1])
    assert abs(S._predict(mid, x) - 0.1 * mid["chunks_b"][1]) < 1e-12
    # a one-chunk closure constrains nothing a per-chunk factor can change
    intra = [_row(3, 7, 0.95, chunks_a={"0": 1.0}, chunks_b={"0": 1.0})]
    assert S._closure_rows(intra, owner, 0.01) == ([], 1)


# ── point 134: the evidence is taken on its stamp, never on an epoch number ──

def test_134_evidence_files_are_taken_only_on_a_matching_stamp(tmp_path):
    scene = build_scene(tmp_path, n_kf=N_KF, chunk_plan=PLAN)
    out = scene.output_dir
    cfg = make_correction_cfg()
    assert V._write_scale_rows(out, {"scale_rows": [_row(4, 24, 0.9)]}, log=lambda m: None, cfg=cfg)
    doc, fresh, why = V.read_evidence(out, "scale_loop_rows.json", cfg)
    assert fresh and doc[V.EVIDENCE_STAMP_KEY]["sha256"] and why == []
    # another geometry under the same epoch number: the cloud changed → refused, by name
    (out / "cleaned_cloud.ply").write_bytes((out / "cleaned_cloud.ply").read_bytes() + b"\0")
    doc, fresh, why = V.read_evidence(out, "scale_loop_rows.json", cfg)
    assert not fresh and any("cleaned_cloud.ply" in w for w in why), why
    rows, ep, refused = S.loop_rows_artifact(out, cfg)
    assert rows == [] and refused["n"] == 1 and refused["why"]
    # the same geometry, another configuration of the measurement → refused too
    (out / "cleaned_cloud.ply").write_bytes((out / "cleaned_cloud.ply").read_bytes()[:-1])
    assert V.read_evidence(out, "scale_loop_rows.json", cfg)[1]
    other = make_correction_cfg(**{"visit_drift.min_walk_m": 0.2})
    assert not V.read_evidence(out, "scale_loop_rows.json", other)[1]
    # a re-measurement that finds no row deletes the stale file instead of leaving it
    assert not V._write_scale_rows(out, {"scale_rows": []}, log=lambda m: None, cfg=cfg)
    assert not (out / "scale_loop_rows.json").exists()


# ── points 135 / 136 / 147: the chain is the source of what was applied ─────

def _transform_epoch(out, epoch, parent, frames, k, kind="transform"):
    n = len(frames)
    ledger.save_epoch_npz(out, epoch, np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)),
                          np.full(n, float(k)), frames)
    d = out / f"_epoch_{epoch}"
    d.mkdir(exist_ok=True)
    (d / EPOCH_FILE).write_text(json.dumps({"epoch": epoch, "parent_epoch": parent, "kind": kind,
                                            "correction_id": f"c{epoch}"}))
    (d / "_manifest.json").write_text(json.dumps({"epoch": epoch, "epoch_from": epoch, "epoch_to": epoch + 1,
                                                  "artifacts": []}))


def test_135_136_the_sidecar_and_the_counts_come_from_the_lineage_of_the_product(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    frames = [0, 10, 20, 30]
    # epochs 1 (F2's, k 1.02) and 2 (F5's) BEFORE the product epoch 3 (a new cloud), then the
    # certification's epoch 4 over it; a leftover epoch 7 of another branch sits on disk
    _transform_epoch(out, 1, 0, frames, 1.02)
    _transform_epoch(out, 2, 1, frames, 1.0)
    (out / "_epoch_3").mkdir()
    (out / "_epoch_3" / EPOCH_FILE).write_text(json.dumps({"epoch": 3, "parent_epoch": 2, "kind": "new_cloud",
                                                           "correction_id": "p"}))
    (out / "_epoch_3" / "_manifest.json").write_text(json.dumps({"epoch": 3, "epoch_to": 4, "artifacts": []}))
    _transform_epoch(out, 4, 3, frames, 1.05)
    _transform_epoch(out, 7, 3, frames, 1.5)
    (out / EPOCH_FILE).write_text(json.dumps({"epoch": 4, "parent_epoch": 3, "kind": "transform",
                                              "correction_id": "c4"}))
    (out / "depth_correction.json").write_text(json.dumps({"version": 2, "k": {"0": 9.0}, "b": {"0": 0.0}}))
    assert CH.base_epoch(out) == 3 and CH.transform_epochs(out, 3) == [4]
    K = CH.applied_depth_factor(out, frames)
    assert np.allclose(K, 1.05), "the chain over the PRODUCT: F2's 1.02 before it is not part of it"
    side = CH.depth_sidecar(out, frames, 5, k_new=np.full(4, 1.1))
    assert side["base_epoch"] == 3 and side["chain"] == [4, 5] and side["chain_stamp"]
    assert side["k"] == {str(f): round(1.05 * 1.1, 6) for f in frames}, side["k"]
    assert "9.0" not in json.dumps(side), "the sidecar on disk was never read"
    # the count every derived artifact carries: the transform epochs of the live lineage
    assert corrections_summary(out)["applied"] == 1
    # the leftovers outside the lineage go; the ledger (history) is not touched
    (out / "corrections.jsonl").write_text("")
    gone = CH.lineage_cleanup(out, [0, 1, 2, 3, 4], log=lambda m: None)
    assert gone == {"dirs": [7], "npz": [7], "quality": []}
    assert (out / "corrections" / "epoch_4.npz").exists() and not (out / "_epoch_7").exists()
    from correction.apply import next_epoch
    assert next_epoch(out) == 5
    with pytest.raises(CH.ChainError):
        CH.chain_transform(out, frames, base=0)         # the chain from 0 crosses the product


# ── point 137: the ids come from the inputs ────────────────────────────────

def test_137_two_identical_floor_runs_write_the_same_id_and_ledger_bytes(tmp_path):
    from correction.run import run_floor
    outs = []
    for name in ("a", "b"):
        scene = build_scene(tmp_path / name, floor="ramp", floor_slope=0.04, drift_t=(0.0, 0.10, 0.0))
        rep = run_floor(scene.output_dir, "plane", None, "test", cfg=make_correction_cfg())
        assert rep["status"] == "applied", rep.get("rejection_reason")
        assert "created_at" not in rep and "elapsed_s" not in rep
        outs.append((scene.output_dir, rep))
    (oa, ra), (ob, rb) = outs
    assert ra["correction_id"] == rb["correction_id"] and len(ra["correction_id"]) == 8
    for rel in ("corrections.jsonl", "geometry_epoch.json", "depth_correction.json",
                f"corrections/report_{ra['correction_id']}.json", "scale_diagnostics.json"):
        assert (oa / rel).read_bytes() == (ob / rel).read_bytes(), rel
    timing = [json.loads(l) for l in (oa / ledger.TIMING_FILE).read_text().splitlines()]
    assert timing[-1]["correction_id"] == ra["correction_id"] and "elapsed_s" in timing[-1]
    diag = json.loads((oa / "scale_diagnostics.json").read_text())
    assert len(diag["epochs"]) == 1 and "created_at" not in diag["epochs"][0]


# ── point 142: the object store rebuilt into a new file, annotations carried ──

def test_142_the_store_is_rebuilt_from_the_result_in_a_fixed_order(tmp_path):
    from correction.invalidate import rebuild_store_fresh
    from phase_r.instance_store import InstanceStore
    scene = build_scene(tmp_path, n_kf=N_KF)
    out = scene.output_dir
    st = InstanceStore(out / "scene_r.db")
    st.upsert_instance(99, "ghost")                       # an instance the segmentation no longer has
    st.add_user_volume("box", (1.0, 0.5, 0.0), (1.0, 1.0, 1.0))
    st.set_meta("chat_notes", json.dumps({"a": "note"}))
    st.set_meta("corrections_ledger", "[]")               # the old mirror (point 136): not carried
    st.close()
    s1 = rebuild_store_fresh(out, log=lambda m: None)
    b1 = (out / "scene_r.db").read_bytes()
    s2 = rebuild_store_fresh(out, log=lambda m: None)
    assert (out / "scene_r.db").read_bytes() == b1 and s1 == s2
    assert s1["store"] == "rebuilt" and s1["user_volumes"] == 1 and s1["scene_meta"] == 1
    st = InstanceStore(out / "scene_r.db")
    try:
        assert {int(r["instance_id"]) for r in st.list_instances()} == {1, 2, 3, 4}
        assert st.get_meta("chat_notes") == json.dumps({"a": "note"})
        assert st.get_meta("corrections_ledger") is None
        assert len(st.list_user_volumes()) == 1
    finally:
        st.close()
    assert not (out / "scene_r.db.before_rebuild").exists()


# ── point 160: the server runs no certification code ───────────────────────

def test_160_the_server_process_runs_no_certification():
    """ONE launcher: the certification runs in the pipeline's stage process
    (workers/certify_worker.py); the Segmentation Manager's close orders a job of the
    manager (main.py, P5). The certification HTTP surface holds no stage code and no
    thread hook that would run it inside the server."""
    from reconstruction.certify import api
    src = Path(api.__file__).read_text()
    assert not hasattr(api, "auto_run") and not hasattr(api, "certify_job")
    assert "certify_session" not in src and "run_correction" not in src
    main_src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    assert "certify_session" not in main_src and "auto_run" not in main_src
    worker = (Path(__file__).resolve().parents[1] / "workers" / "certify_worker.py").read_text()
    assert "certify_session(session_path" in worker


# ── points 126 / 139 / 166: the acta — base epoch, stamps, no clock; reuse ──

def test_126_166_the_certification_runs_on_the_product_epoch_and_reuses_its_stamp(tmp_path, monkeypatch):
    from reconstruction.certify.run import ACTA_JSON, ACTA_TIMING_JSON, certify_session
    scene = build_scene(tmp_path, n_kf=N_KF, chunk_plan=PLAN)
    out = scene.output_dir
    cfg = make_correction_cfg(**{"gates.mode": "advisory"})
    # six closures saying nothing is left (identity by the rule): the floor still publishes
    _rows(out, [_row(2 + i, 22 + i, 1.0, iid=10 + i) for i in range(6)])
    V._stamp_out(out, "scale_loop_rows.json",
                 {"rows": [_row(2 + i, 22 + i, 1.0, iid=10 + i) for i in range(6)]}, cfg=cfg)
    lcfg = load_loops_config()
    assert lcfg.certify.deliverable_only
    acta = certify_session(tmp_path, cfg=lcfg, operator="test", log=lambda m: None,
                           correction_cfg=cfg)
    assert acta["base"]["base_epoch"] == 0 and acta["epoch_initial"] == 0
    assert acta["epoch_final"] == 1 and acta["input_stamp"] and acta["environment"]["git"]
    assert acta.get("reused") is None
    for k in ("started_at", "elapsed_s"):
        assert k not in acta and k not in acta["correction"]
    timing = json.loads((out / ACTA_TIMING_JSON).read_text())
    assert timing["started_at"] and timing["elapsed_s"] >= 0
    saved = json.loads((out / ACTA_JSON).read_text())
    assert saved["input_stamp"] == acta["input_stamp"]
    rec = json.loads((out / EPOCH_FILE).read_text())
    assert rec["certify_input_stamp"] == acta["input_stamp"] and "created_at" not in rec
    assert acta["instance_store"]["store"] == "rebuilt"
    cloud1 = (out / "cleaned_cloud.ply").read_bytes()
    # the same inputs again: the certified epoch is REUSED, nothing recomputed, nothing stacked
    monkeypatch.setattr(V, "run", lambda *a, **k: pytest.fail("the correction must not run again"))
    acta2 = certify_session(tmp_path, cfg=lcfg, operator="test", log=lambda m: None,
                            correction_cfg=cfg)
    assert acta2["reused"] is True and acta2["epoch_final"] == 1
    assert (out / "cleaned_cloud.ply").read_bytes() == cloud1
    from correction.apply import available_epochs
    assert [e["epoch"] for e in available_epochs(out)] == [0, 1]


# ── points 106 / 133: every deleted point's ballot, with its pixels' edge distances ──

def test_106_133_the_mask_filter_records_the_ballot_of_every_deleted_point(tmp_path):
    """The certification's mask filter step itself (visit_drift_run.filter_staged_cloud) on the
    rendered plate-and-wall scene of test_certify_mask_filter_camera: the user's rules decide
    unchanged (the skirt leaves, nothing else) and the record lists exactly the deleted points,
    each with its views and every counting vote's signed distance to the mask edge; two runs
    write the same bytes."""
    from correction import visit_drift_run as VR
    from correction.visit_drift import write_votes
    from tests.test_certify_mask_filter_camera import _session
    out, xyz, fg, pr, pc, ks, skirt, label, poses = _session(tmp_path)
    data = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("frame_global", "<i4"), ("pixel_row", "<i2"),
                                     ("pixel_col", "<i2")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["frame_global"], data["pixel_row"], data["pixel_col"] = fg, pr, pc
    header = [b"ply\n", b"format binary_little_endian 1.0\n", b"element vertex 0\n",
              b"property float x\n", b"property float y\n", b"property float z\n",
              b"property int frame_global\n", b"property short pixel_row\n",
              b"property short pixel_col\n", b"end_header\n"]
    session = SimpleNamespace(output_dir=out, ks=ks, header=header, raw_header=None)
    cfg = SimpleNamespace(visit_drift=SimpleNamespace(min_points=1, min_visit_share=0.0))
    xyz64 = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    blobs = []
    for run in ("a", "b"):
        tx = tmp_path / f"tx_{run}"
        tx.mkdir()
        rep, keep = VR.filter_staged_cloud(tx, session, data, xyz64, poses, None, cfg,
                                           log=lambda m: None)
        assert np.array_equal(~keep, skirt)
        votes = rep["_votes"]
        assert sorted(votes["point_index"].tolist()) == np.flatnonzero(skirt).tolist()
        n = len(votes["point_index"])
        off = votes["vote_offsets"]
        assert len(off) == n + 1 and off[0] == 0 and off[-1] == len(votes["vote_kf"])
        # the skirt leaves by rule 4 (on the wall's surface in the majority of the views that
        # saw it): its recorded votes are the ones the verdict counted
        assert set(votes["rule"].tolist()) == {4}
        for i in range(n):
            r4 = votes["vote_rule"][off[i]:off[i + 1]] == 4
            assert int(r4.sum()) == int(votes["n_votes"][i])
            assert int(votes["vote_on_other"][off[i]:off[i + 1]][r4].sum()) == int(votes["n_other"][i])
        # every counting vote carries the distance of its pixel to the own mask edge, and a
        # vote ON the other object lies inside that object's mask (distance >= 0)
        assert np.all(np.isfinite(votes["vote_dist_own_px"]))
        on = votes["vote_on_other"]
        assert np.all(votes["vote_dist_other_px"][on] >= 0.0)
        p = tx / "votes.npz"
        write_votes(p, votes)
        blobs.append(p.read_bytes())
        assert rep["rules"]["min_votes"] == 2 and rep["margins"]
    assert blobs[0] == blobs[1]


# ── points 126 / 150: a stage on the product reads the product epoch ──────────

def test_126_150_the_product_epoch_is_made_live_and_a_missing_one_refuses(tmp_path):
    from correction.chain import ChainError, select_product_epoch
    from correction.epoch import current_epoch
    from correction.run import run_floor
    scene = build_scene(tmp_path, floor="ramp", floor_slope=0.04, drift_t=(0.0, 0.10, 0.0))
    out = scene.output_dir
    cloud0 = (out / "cleaned_cloud.ply").read_bytes()
    # the reconstruction published epoch 0 as its product (corrected_cloud.json names it)
    (out / "corrected_cloud.json").write_text(json.dumps({"epoch_to": 0, "correction_id": "p0"}))
    rep = run_floor(out, "plane", None, "test", cfg=make_correction_cfg())
    assert rep["status"] == "applied" and current_epoch(out) == 1
    # a certified / user-picked epoch is live: the product comes back, bit for bit
    info = select_product_epoch(out, log=lambda m: None)
    assert info["selected"] and info["base_epoch"] == 0 and current_epoch(out) == 0
    assert (out / "cleaned_cloud.ply").read_bytes() == cloud0
    assert not select_product_epoch(out, log=lambda m: None)["selected"]
    # the product epoch gone from disk: refused, by name — never another epoch in its place
    from correction.run import run_select
    run_select(out, 1, "test")
    import shutil
    shutil.rmtree(out / "_epoch_0")
    with pytest.raises(ChainError, match="product epoch 0"):
        select_product_epoch(out, log=lambda m: None)
    assert current_epoch(out) == 1
