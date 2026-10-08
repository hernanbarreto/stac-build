"""The co-visibility chunk planner (reconstruction/chunk_covis.py, design §1.5–1.8) on synthetic
budgets, plus the session path (measure → persist → resume without the window files).

USER 2026-10-06: the plan depends on co-visibility ONLY — no card capacity splits or shrinks a
chunk (Omega's resolution adapts to the largest one) and the scan horizon is the whole walk."""

import json

import numpy as np
import pytest

from reconstruction.chunk_covis import (H_LENGTHS_PER_CHUNK, MIN_CHUNK_FRAMES, CovisError, PlanError,
                                        inputs_sha256, plan, plan_detail, plan_session)


def _flat(n, z=1.0, t=1.0):
    return np.full(n, z), np.full(n - 1, t)


def _check_structure(ranges, n, mcf=MIN_CHUNK_FRAMES):
    """The fork's rules (design §1.6) and the 50 % block structure."""
    assert ranges[0][0] == 0 and ranges[-1][1] == n
    starts = [a for a, _ in ranges]
    ends = [b for _, b in ranges]
    assert all(x < y for x, y in zip(starts, starts[1:]))
    assert all(x < y for x, y in zip(ends, ends[1:]))
    for a, b in ranges:
        assert mcf <= b - a
    for k in range(len(ranges) - 1):
        assert ranges[k + 1][0] < ranges[k][1]            # consecutive chunks overlap
    for k in range(len(ranges) - 2):
        assert ranges[k][1] == ranges[k + 2][0]           # chunk k ends where chunk k+2 starts
    for k in range(1, len(ranges) - 1):                   # the seams are whole blocks: chunk k is
        assert ranges[k - 1][1] == ranges[k + 1][0]       # its left seam + its right seam


def test_single_pass_when_within_budget():
    d = np.full(100, 0.01)
    z, t = _flat(100)
    assert plan(d, z, t) == [(0, 100)]
    det = plan_detail(d, z, t)
    assert det["single_pass"] and det["D_total"] <= H_LENGTHS_PER_CHUNK and det["flags"] == []
    # the same walk over budget is chunked
    assert len(plan(np.full(100, 0.5), z, t)) > 1


def test_uniform_budget_gives_equal_blocks():
    n = 300
    d = np.full(n, 0.1)                       # 68 frames per block fit H/2 → 5 blocks; 60 each
    z, t = _flat(n)
    det = plan_detail(d, z, t)
    assert [b["frames"] for b in det["blocks"]] == [60] * 5
    assert det["ranges"] == [(0, 120), (60, 180), (120, 240), (180, 300)]
    assert all(c["D"] <= H_LENGTHS_PER_CHUNK for c in det["chunks"])


def test_mixed_walk_long_then_short_chunks():
    d = np.concatenate([np.full(200, 0.02), np.full(195, 0.5)])     # cheap, then expensive
    n = len(d)
    z, t = _flat(n)
    det = plan_detail(d, z, t)
    r = det["ranges"]
    _check_structure(r, n)
    lengths = [b - a for a, b in r]
    assert lengths[0] >= 200 and max(lengths[1:]) <= 26
    assert all(c["D"] <= H_LENGTHS_PER_CHUNK for c in det["chunks"])
    assert all(b["D"] <= H_LENGTHS_PER_CHUNK / 2 for b in det["blocks"])
    assert det["objective"]["blocks"] == 16 and det["flags"] == []


def test_the_card_never_splits_a_chunk():
    """No capacity in the planner (USER 2026-10-06): a free walk of any length is ONE pass,
    and a chunked walk keeps chunks as long as its budget allows — Omega's resolution adapts."""
    n = 3000
    d = np.zeros(n)                            # D_total 0: nothing to chunk, however long
    z, t = _flat(n)
    det = plan_detail(d, z, t)
    assert det["single_pass"] and det["ranges"] == [(0, n)]
    d = np.concatenate([np.zeros(1500), np.full(200, 0.5)])     # a free stretch, then a costly one
    z, t = _flat(len(d))
    r = plan(d, z, t)
    _check_structure(r, len(d))
    assert r[0][1] - r[0][0] > 1500            # the free stretch stays in one chunk


def test_seam_lands_on_near_structure_and_cuts_avoid_the_turn():
    n = 300
    d = np.full(n, (H_LENGTHS_PER_CHUNK / 2.0) / 140.0)   # a block holds ≤ 140 frames: 3 blocks
    z = np.full(n, 10.0)
    z[150:220] = 2.0
    det = plan_detail(d, z, np.ones(n - 1))
    assert det["objective"]["blocks"] == 3
    assert det["blocks"][1]["z_med_m"] == 2.0 and det["objective"]["worst_seam_depth_m"] == 2.0
    t = np.ones(n - 1)
    t[55:115] = 30.0                           # a turn: no cut between keyframes 56..115
    det = plan_detail(d, np.full(n, 5.0), t)
    assert all(c["theta_deg"] == 1.0 for c in det["cuts"])
    assert all(not (56 <= c["index"] <= 115) for c in det["cuts"])


def test_minimum_size_over_budget_is_flagged():
    d = np.full(240, 0.1)
    z, t = _flat(240)
    det = plan_detail(d, z, t, H=0.1)                     # even 12 frames exceed H/2
    assert [b["frames"] for b in det["blocks"]] == [12] * 20
    assert det["flags"] == ["over_budget_at_minimum"]
    assert all(b["flag"] == "over_budget_at_minimum" for b in det["blocks"])
    _check_structure(det["ranges"], 240)
    # 245 frames: the integer remainder forces ONE block above the minimum, and only one
    z, t = _flat(245)
    det = plan_detail(np.full(245, 0.1), z, t, H=0.1)
    lens = sorted(b["frames"] for b in det["blocks"])
    assert lens == [12] * 19 + [17]
    assert det["objective"]["over_budget_above_minimum"] == 1
    assert "over_budget_partition" in det["flags"]


def test_too_short_to_chunk_and_malformed_inputs():
    z, t = _flat(20)
    det = plan_detail(np.full(20, 1.0), z, t)
    assert det["single_pass"] and det["flags"] == ["too_short_to_chunk"]
    z, t = _flat(100)
    with pytest.raises(PlanError):
        plan(np.zeros(100), z, t[:-1])                    # n − 1 cut angles are required
    with pytest.raises(PlanError):
        plan(np.full(100, -0.1), z, t)                    # a budget is ≥ 0


def test_deterministic():
    rng = np.random.default_rng(7)
    n = 600
    d = rng.gamma(1.0, 0.12, n)
    z = rng.uniform(1.0, 9.0, n)
    t = rng.uniform(0.0, 6.0, n - 1)
    a = plan_detail(d, z, t)
    b = plan_detail(d.copy(), z.copy(), t.copy())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert inputs_sha256(d, z, t, 24, 13.67) == inputs_sha256(d.copy(), z, t, 24, 13.67)
    _check_structure(a["ranges"], n)
    assert all(c["D"] <= H_LENGTHS_PER_CHUNK for c in a["chunks"])


def test_block_overlap_structure():
    rng = np.random.default_rng(3)
    for n, scale in ((500, 0.2), (900, 0.05), (333, 0.1)):
        d = rng.gamma(1.0, scale, n)
        det = plan_detail(d, rng.uniform(1, 5, n), rng.uniform(0, 3, n - 1))
        r = det["ranges"]
        assert len(r) > 2
        _check_structure(r, n)
        bl = [tuple(b["range"]) for b in det["blocks"]]
        assert r == [(bl[k][0], bl[k + 1][1]) for k in range(len(bl) - 1)]
        for k in range(len(r) - 1):                       # the overlap of k and k+1 IS block k+1
            assert (r[k + 1][0], r[k][1]) == bl[k + 1]


# ── the session path: synthetic I3 windows ───────────────────────────────────────────────────

def _synthetic_session(root, n=40, wf=8):
    """A camera sliding sideways along x in front of a wall 3 m away, DA3 windows of `wf`
    keyframes at 50 %, each window's depth carrying its own small per-pixel disagreement."""
    from intake.walk import plan_windows
    h, w = 24, 32
    K = np.array([[20.0, 0, 16.0], [0, 20.0, 12.0], [0, 0, 1.0]])
    wdir = root / "output" / "da3_windows"
    wdir.mkdir(parents=True)
    (root / "intake").mkdir()
    vv, uu = np.mgrid[0:h, 0:w]
    spec = []
    for k, (a, b) in enumerate(plan_windows(n, wf, 0.5)):
        idx = np.arange(a, b)
        ext = np.zeros((len(idx), 3, 4))
        ext[:, :3, :3] = np.eye(3)
        ext[:, 0, 3] = -0.1 * idx                         # w2c of a camera at x = 0.1·i
        depth = np.stack([3.0 * (1.0 + 0.01 * np.sin(0.7 * uu + 1.3 * vv + k + 0.3 * i))
                          for i in idx]).astype(np.float32)
        np.savez(wdir / f"window_{k:04d}.npz", frames=(10 * idx).astype(np.int64), depth=depth,
                 depth_mono=depth, conf=np.ones_like(depth), extrinsics=ext,
                 intrinsics=np.repeat(K[None], len(idx), 0), scale_factor=np.float64(1.0),
                 is_metric=np.int64(1))
        spec.append([f"frames/{10 * i:06d}.jpg" for i in idx])
    (wdir / "windows.json").write_text(json.dumps({"windows": spec, "process_res": w,
                                                   "model_id": "synthetic"}))
    (root / "intake" / "walk.json").write_text(json.dumps({"version": 1, "n_keyframes": n}))
    return wdir


def test_session_persists_and_resumes_without_windows(tmp_path):
    wdir = _synthetic_session(tmp_path)
    logs = []
    first = plan_session(tmp_path, log=logs.append)
    rep = first["report"]
    assert rep["measurement"] == "measured" and rep["n"] == 40
    doc = json.loads((tmp_path / "intake" / "covis.json").read_text())
    assert doc["scan_horizon"] == "whole_walk" and doc["stamp_parts"]["scan_horizon"] == "whole_walk"
    assert 0.0 < rep["tol_rel"] < 0.05
    # a camera sliding 0.1 m/kf past a wall 3 m away (4.8 m of view): co-visible for ~33 kf,
    # the whole 40-kf walk is a few lengths — ONE pass, however few frames a card would hold
    assert rep["single_pass"] and first["ranges"] == [(0, 40)]
    assert rep["D_total"] <= H_LENGTHS_PER_CHUNK
    for p in wdir.glob("window_*.npz"):                   # F2 deletes the windows
        p.unlink()
    again = plan_session(tmp_path, log=logs.append)
    # the session's FROZEN plan (point 15), served from the persisted measurement without windows
    assert again["report"]["measurement"] == "frozen"
    assert again["ranges"] == first["ranges"]
    a = dict(rep); b = dict(again["report"])
    a.pop("measurement"); b.pop("measurement")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert (tmp_path / "intake" / "chunk_plan_frozen.json").exists()
    # the stamp holds the code of the planner and of the walk (point 21)
    assert set(doc["stamp_parts"]["code_sha256"]) == {"server/intake/walk.py",
                                                       "server/reconstruction/chunk_covis.py"}
    # every bar's margin is recorded (point 15)
    assert rep["margins"]["D_total_over_H"] < 0 and rep["tau_margin_min"] is not None
    assert len(doc["tau_margin"]) == 40 and doc["tau_margin_min"] >= 0
    # a new walk (I3 re-measured) invalidates the stamp; without the windows that is an error
    (tmp_path / "intake" / "walk.json").write_text(json.dumps({"version": 1, "n_keyframes": 40,
                                                               "walk_length_m": 3.9}))
    with pytest.raises(CovisError):
        plan_session(tmp_path, log=logs.append)


def test_the_window_layout_is_recorded_next_to_h_calibration_layout(tmp_path):
    """Review 2026-10-06: the measurement depends on the I3 windows, whose size is the card's.
    covis.json and the plan report RECORD the windows it was read from (size, seams, model,
    process_res, the card that sized them) next to the layout H was calibrated at — recorded,
    never acted on: the plan is the same with or without the card on record."""
    from reconstruction.chunk_covis import H_CALIBRATION_LAYOUT, format_plan
    _synthetic_session(tmp_path)
    first = plan_session(tmp_path, log=lambda m: None)
    wl = json.loads((tmp_path / "intake" / "covis.json").read_text())["window_layout"]
    assert wl["window_frames"] == 8 and wl["seam_frames"] == [4, 4] and wl["n_windows"] == 9
    assert wl["process_res"] == 32 and wl["model_id"] == "synthetic"
    assert wl["card"] is None and wl["card_matches_calibration"] is None
    rep = first["report"]
    assert rep["window_layout"] == wl and rep["H_calibration_layout"] == H_CALIBRATION_LAYOUT
    assert H_CALIBRATION_LAYOUT["card"] == "NVIDIA A100 80GB PCIe, 81920 MiB"
    assert any("is not recorded" in ln and "H 13.67 was calibrated" in ln for ln in format_plan(rep))
    # the card that sized the windows (windows.json's own sizing record, the committed card
    # table's key — since 2026-10-07; the da3_vram.json cache is gone) — another card is DECLARED only
    spec_p = tmp_path / "output" / "da3_windows" / "windows.json"
    spec = json.loads(spec_p.read_text())
    spec["window_sizing"] = {"card": "NVIDIA RTX A6000 | 49140 MiB | sm_8.6", "window_frames": 8}
    spec_p.write_text(json.dumps(spec))
    again = plan_session(tmp_path, log=lambda m: None)
    assert again["ranges"] == first["ranges"] and again["report"]["measurement"] == "frozen"
    wl2 = again["report"]["window_layout"]
    assert wl2["card"] == "NVIDIA RTX A6000 | 49140 MiB | sm_8.6" and wl2["card_matches_calibration"] is False
    assert any("another card" in ln for ln in format_plan(again["report"]))
    assert H_CALIBRATION_LAYOUT["card_key"] == "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"


def test_the_frozen_plan_declines_a_replan_on_noise_and_plans_anew_for_other_keyframes(tmp_path, monkeypatch):
    """Point 15: the first plan of a keyframe set is FROZEN per session; a later measurement whose
    integer ℓ or DP thresholds would flip the cut never re-plans it (declared, with the plan it
    would have given); another keyframe set plans again and freezes that."""
    import shutil
    from reconstruction import chunk_covis as CC
    _synthetic_session(tmp_path)
    first = plan_session(tmp_path, log=lambda m: None)
    assert first["ranges"] == [(0, 40)] and first["report"]["measurement"] == "measured"
    fz = json.loads((tmp_path / "intake" / CC.FROZEN_NAME).read_text())
    assert fz["ranges"] == [[0, 40]] and len(fz["keyframes"]) == 40 and fz["planner"]["H"] == CC.H_LENGTHS_PER_CHUNK
    # the measurement now reads differently (a flipped test): the planner would cut — the frozen plan stays
    real = CC.plan_detail

    def _flipped(*a, **k):
        det = real(*a, **k)
        det = dict(det, ranges=[(0, 24), (16, 40)], single_pass=False)
        return det
    monkeypatch.setattr(CC, "plan_detail", _flipped)
    logs = []
    again = plan_session(tmp_path, log=logs.append)
    assert again["ranges"] == [(0, 40)] and again["report"]["measurement"] == "frozen"
    assert again["report"]["replan_declined"]["ranges_now"] == [(0, 24), (16, 40)]
    assert any("FROZEN plan" in m and "declared" in m for m in logs)
    monkeypatch.setattr(CC, "plan_detail", real)
    # another keyframe set (a session with 48 keyframes carrying the first one's frozen file)
    other = tmp_path / "other"
    _synthetic_session(other, n=48)
    shutil.copy(tmp_path / "intake" / CC.FROZEN_NAME, other / "intake" / CC.FROZEN_NAME)
    logs = []
    new = plan_session(other, log=logs.append)
    assert new["ranges"][-1][1] == 48 and new["report"]["measurement"] == "measured"
    assert json.loads((other / "intake" / CC.FROZEN_NAME).read_text())["keyframes"][-1] == 470
    assert any("froze another keyframe set" in m for m in logs)
