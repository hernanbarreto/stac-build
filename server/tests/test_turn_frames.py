"""Turn frames (USER 2026-11-04, intake/turn_frames.py): Omega's context views between two
consecutive keyframes, chosen from I1's own rotation readings — the steps, the sharpest frame of
each step's band, the windows that get none and why, the per-chunk assignment, the config, the
stamped document; the map worker sizes the card on keyframes + context views and records them in
the plan (reconstruction/chunk_plan.py, workers/map_worker.py). CPU only, no model."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake.turn_frames import (TURN_FRAMES_NAME, TurnFramesConfig, TurnFramesError,  # noqa: E402
                                chunk_context, load_turn_frames_config, select_turn_frames,
                                session_turn_frames)

THETA_Q = 10.8           # → θ_t 3.6°, steps ≤ 4.5° at band 0.25


def _fn(f):
    return f"{f:06d}.jpg"


def _session_docs(rot_per_frame=0.54, sharp=None, lost=(), unusable=(), kf=(0, 10, 30),
                  kf_rot=None, anchors=None, windows=None, par_per_frame=1.2, kf_par=None):
    """Keyframes ``kf``; the window kf[i] → kf[i+1] reads (f − kf[i]) · rot_per_frame degrees and
    (f − kf[i]) · par_per_frame pixels of parallax from its anchor; frame 30's readings are the
    keyframe's own."""
    sharp = dict(sharp or {})
    n = kf[-1] + 5
    quality = [{"frame": f, "file": _fn(f), "sharp_rank": sharp.get(f, 0.5),
                "usable": f not in unusable} for f in range(n)]
    recs = []
    for i in range(len(kf) - 1):
        a, k = kf[i], kf[i + 1]
        for f in range(a + 1, k + 1):
            recs.append({"frame": f, "anchor": (anchors or {}).get(f, a), "lost": f in lost,
                         "rot_deg": None if f in lost else (f - a) * rot_per_frame,
                         "parallax_px": None if f in lost else (f - a) * par_per_frame})
    kf_rot, kf_par = kf_rot or {}, kf_par or {}
    keyframes = [{"frame": kf[0], "file": _fn(kf[0]), "anchor": None, "rotation_deg": 0.0,
                  "parallax_px": 0.0, "closed_by": "first_usable_frame"}]
    for i in range(1, len(kf)):
        keyframes.append({"frame": kf[i], "file": _fn(kf[i]), "anchor": kf[i - 1],
                          "rotation_deg": kf_rot.get(kf[i], (kf[i] - kf[i - 1]) * rot_per_frame),
                          "parallax_px": kf_par.get(kf[i], (kf[i] - kf[i - 1]) * par_per_frame),
                          "closed_by": "rotation"})
    if windows is None:
        windows = [{"anchor": kf[i - 1], "keyframe": kf[i], "anchor_is_keyframe": True}
                   for i in range(1, len(kf))]
    sel = {"rotation_quantum_deg": THETA_Q, "keyframe_band_frac": 0.25, "keyframes": keyframes,
           "windows": windows, "selected_files": [_fn(f) for f in kf]}
    return sel, recs, quality


def test_a_turn_is_cut_into_equal_steps_and_each_step_takes_its_sharpest_frame():
    # window 10 → 30 turns 20 · 0.54 = 10.8°: n = ceil(10.8 / 4.5) = 3 steps of 3.6°; the band
    # of step 1 is 3.6 ± 0.9 (frames 15-18), of step 2 7.2 ± 0.9 (frames 22-25, 25 on the edge)
    sel, recs, q = _session_docs(sharp={17: 0.9, 16: 0.8, 23: 0.95, 24: 0.99})
    r = select_turn_frames(sel, recs, q, 3, 0)
    assert math.isclose(r["theta_t_deg"], 3.6) and math.isclose(r["step_max_deg"], 4.5)
    w1 = [(t["frame"], t["step"], t["steps"], t["chosen_by"]) for t in r["frames"]
          if t["anchor"] == 10]
    assert w1 == [(17, 1, 3, "band"), (24, 2, 3, "band")]
    # window 0 → 10 turns 5.4°: n = 2, one frame at 2.7 ± 0.675 (frames 4, 5, 6, all sharp 0.5
    # → the earliest)
    w0 = [(t["frame"], t["step"], t["steps"]) for t in r["frames"] if t["anchor"] == 0]
    assert w0 == [(4, 1, 2)]
    assert r["files"] == [_fn(4), _fn(17), _fn(24)]
    assert [w["steps"] for w in r["windows"]] == [2, 3]


def test_two_steps_and_the_tie_rule():
    sel, recs, q = _session_docs(rot_per_frame=0.6, kf=(0, 10))     # 6.0° → n = 2, step 3.0
    r = select_turn_frames(sel, recs, q, 3, 0)
    # band 3.0 ± 0.75: readings 2.4, 3.0, 3.6 → frames 4, 5, 6, all sharp 0.5 → the earliest
    assert [(t["frame"], t["step"], t["steps"]) for t in r["frames"]] == [(4, 1, 2)]
    sel, recs, q = _session_docs(rot_per_frame=0.6, kf=(0, 10), sharp={6: 0.7})
    assert [t["frame"] for t in select_turn_frames(sel, recs, q, 3, 0)["frames"]] == [6]


def test_a_short_turn_gets_no_turn_frame():
    sel, recs, q = _session_docs(rot_per_frame=0.4, kf=(0, 10))     # 4.0° ≤ 4.5°: one step
    r = select_turn_frames(sel, recs, q, 3, 0)
    assert r["frames"] == [] and r["windows"] == []
    # a finer divisor cuts the same turn: d = 6 → θ_t 1.8°, steps ≤ 2.25° → n = 2
    assert [t["steps"] for t in select_turn_frames(sel, recs, q, 6, 0)["frames"]] == [2]


def test_a_turn_that_jumps_over_a_band_takes_the_nearest_reading():
    # 10.8° in 4 frames: readings 2.7, 5.4, 8.1 between the keyframes; steps 3.6 / 7.2 with bands
    # ± 0.9 → 2.7 is in band 1 (edge), 7.2 ± 0.9 = [6.3, 8.1] → 8.1 (edge) — make it jump: 3 frames
    sel, recs, q = _session_docs(rot_per_frame=10.8 / 3, kf=(0, 3))     # readings 3.6, 7.2 …
    r = select_turn_frames(sel, recs, q, 3, 0)
    assert [(t["frame"], t["chosen_by"]) for t in r["frames"]] == [(1, "band"), (2, "band")]
    sel, recs, q = _session_docs(rot_per_frame=10.8 / 2, kf=(0, 2))     # one frame at 5.4
    r = select_turn_frames(sel, recs, q, 3, 0)
    # step 1 (3.6 ± 0.9) has no reading → the nearest, frame 1 (5.4); step 2: no frame left
    assert [(t["frame"], t["step"], t["chosen_by"]) for t in r["frames"]] == [(1, 1, "nearest")]
    assert r["windows"][0]["n_turn_frames"] == 1 and r["windows"][0]["steps"] == 3


def test_lost_unusable_and_other_anchor_frames_are_never_chosen():
    sel, recs, q = _session_docs(rot_per_frame=0.6, kf=(0, 10), lost={4}, unusable={5},
                                 anchors={6: 99}, sharp={4: 1.0, 5: 1.0, 6: 1.0})
    r = select_turn_frames(sel, recs, q, 3, 0)
    # band 3.0 ± 0.75 holds frames 4, 5, 6 — all excluded → the nearest reading left: frames 3
    # (1.8) and 7 (4.2) are 1.2° away, the tie → the earlier
    assert [(t["frame"], t["chosen_by"]) for t in r["frames"]] == [(3, "nearest")]


def test_windows_without_a_measured_turn_get_none_and_say_why():
    sel, recs, q = _session_docs(kf=(0, 10, 30), windows=[
        {"anchor": 0, "keyframe": 10, "anchor_is_keyframe": True},
        {"anchor": 10, "keyframe": 30, "anchor_is_keyframe": False},
        {"anchor": 30, "keyframe": None, "anchor_is_keyframe": True}])
    r = select_turn_frames(sel, recs, q, 3, 0)
    skipped = [w["skipped"] for w in r["windows"] if w.get("skipped")]
    assert skipped == ["anchor_not_keyframe", "no_keyframe"]
    assert all(t["anchor"] == 0 for t in r["frames"])
    sel, recs, q = _session_docs(kf=(0, 10, 30), kf_rot={30: float("nan")})
    r = select_turn_frames(sel, recs, q, 3, 0)
    assert [w["skipped"] for w in r["windows"] if w.get("skipped")] == ["no_rotation_reading"]


def test_every_interval_gets_a_frame_at_half_its_parallax():
    # USER 2026-11-04 (night): 1° and 2° of turn — one rotation step each, the turn rule leaves
    # them alone; at min_per_interval 1 each window takes the sharpest frame at half its parallax:
    # window 0 → 10 (12 px) 6 ± 1.5 px = frames 4, 5, 6; window 10 → 30 (24 px) 12 ± 3 px =
    # frames 18-22
    sel, recs, q = _session_docs(rot_per_frame=0.1, sharp={5: 0.9, 21: 0.8})
    assert select_turn_frames(sel, recs, q, 3, 0)["frames"] == []
    r = select_turn_frames(sel, recs, q, 3, 1)
    got = [(t["frame"], t["role"], t["reading"], t["step"], t["steps"], t["chosen_by"])
           for t in r["frames"]]
    assert got == [(5, "witness", "parallax", 1, 2, "band"), (21, "witness", "parallax", 1, 2, "band")]
    assert [(w["reading"], w["unit"], w["total"], w["steps"]) for w in r["windows"]] == [
        ("parallax", "px", 12.0, 2), ("parallax", "px", 24.0, 2)]
    # USER 2026-11-04: they only affirm or deny in F6 — never Omega's context views
    assert r["min_per_interval"] == 1 and r["files"] == [] and r["witness_files"] == [_fn(5), _fn(21)]
    # all sharp 0.5 → the earliest frame of the band
    sel, recs, q = _session_docs(rot_per_frame=0.1)
    assert select_turn_frames(sel, recs, q, 3, 1)["witness_files"] == [_fn(4), _fn(18)]
    # two per interval: thirds of the parallax, 4 ± 1 px → frames 3, 4 (3.6, 4.8), 8 ± 1 px →
    # frames 6, 7 (7.2, 8.4); all sharp 0.5 → the earliest of each band
    r2 = select_turn_frames(sel, recs, q, 3, 2)
    assert [(t["frame"], t["step"], t["steps"]) for t in r2["frames"] if t["anchor"] == 0] == [
        (3, 1, 3), (6, 2, 3)]


def test_a_turn_keeps_exactly_its_rotation_steps_whatever_the_interval_rule():
    # the chunks are the turn rule's (USER 2026-11-04): Omega's context views never change with
    # min_per_interval, and a turn gets no witness of the interval rule
    sel, recs, q = _session_docs(sharp={17: 0.9, 16: 0.8, 23: 0.95, 24: 0.99})
    turns = select_turn_frames(sel, recs, q, 3, 0)
    for m in (1, 2):
        r = select_turn_frames(sel, recs, q, 3, m)
        assert r["frames"] == turns["frames"] and r["files"] == turns["files"]
        assert r["witness_files"] == []
    assert all(t["reading"] == "rotation" and t["role"] == "context" for t in turns["frames"])
    assert turns["files"] == [_fn(4), _fn(17), _fn(24)]


def test_an_interval_without_a_parallax_reading_is_cut_on_its_rotation_or_skipped():
    sel, recs, q = _session_docs(rot_per_frame=0.1, kf=(0, 10), kf_par={10: None})
    r = select_turn_frames(sel, recs, q, 3, 1)
    assert [(t["role"], t["reading"], t["frame"]) for t in r["frames"]] == [("witness", "rotation", 4)]
    assert r["files"] == [] and r["witness_files"] == [_fn(4)]                     # 0.5° ± 0.125
    sel, recs, q = _session_docs(rot_per_frame=0.1, kf=(0, 10), kf_par={10: None},
                                 kf_rot={10: float("nan")})
    r = select_turn_frames(sel, recs, q, 3, 1)
    assert r["frames"] == [] and [w["skipped"] for w in r["windows"]] == ["no_reading"]
    # turns only: the old reason
    assert [w["skipped"] for w in select_turn_frames(sel, recs, q, 3, 0)["windows"]] == [
        "no_rotation_reading"]
    for bad in (-1, 1.0, True):
        with pytest.raises(TurnFramesError, match="min_per_interval"):
            select_turn_frames(sel, recs, q, 3, bad)


def test_the_selection_refuses_a_session_without_a_rotation_quantum():
    sel, recs, q = _session_docs()
    sel["rotation_quantum_deg"] = None
    with pytest.raises(TurnFramesError, match="rotation quantum"):
        select_turn_frames(sel, recs, q, 3, 0)
    with pytest.raises(TurnFramesError):
        select_turn_frames(_session_docs()[0], recs, q, 0, 0)


def test_chunk_context_follows_the_keyframes_each_chunk_holds():
    kfs = [_fn(f) for f in (0, 10, 20, 30, 40)]
    turns = [_fn(5), _fn(15), _fn(25), _fn(35)]
    # chunks [0, 3) = kf 0..20, [1, 5) = kf 10..40 (seam of 2 keyframes)
    assert chunk_context([(0, 3), (1, 5)], kfs, turns) == [[_fn(5), _fn(15)],
                                                           [_fn(15), _fn(25), _fn(35)]]
    assert chunk_context([(0, 5)], kfs, list(reversed(turns))) == [turns]
    assert chunk_context([(0, 5)], kfs, []) == [[]]
    with pytest.raises(TurnFramesError, match="are keyframes"):
        chunk_context([(0, 5)], kfs, [_fn(10)])
    with pytest.raises(TurnFramesError, match="outside"):
        chunk_context([(0, 6)], kfs, turns)


def test_the_config_is_read_strictly():
    ok = {"reconstruction": {"vggtomega": {"turn_frames": {"enabled": True,
                                                           "per_rotation_quantum": 3,
                                                           "min_per_interval": 1}}}}
    assert load_turn_frames_config(ok) == TurnFramesConfig(True, 3, 1)
    with pytest.raises(TurnFramesError, match="turn_frames is missing"):
        load_turn_frames_config({"reconstruction": {"vggtomega": {}}})
    with pytest.raises(TurnFramesError, match="per_rotation_quantum is missing"):
        load_turn_frames_config({"reconstruction": {"vggtomega": {"turn_frames": {"enabled": 1}}}})
    for bad in (0, 2.5, True, "3"):
        with pytest.raises(TurnFramesError, match="integer"):
            load_turn_frames_config({"reconstruction": {"vggtomega": {"turn_frames": {
                "enabled": True, "per_rotation_quantum": bad, "min_per_interval": 1}}}})
    with pytest.raises(TurnFramesError, match="min_per_interval is missing"):
        load_turn_frames_config({"reconstruction": {"vggtomega": {"turn_frames": {
            "enabled": True, "per_rotation_quantum": 3}}}})
    for bad in (-1, 1.5, False, "1"):
        with pytest.raises(TurnFramesError, match="min_per_interval must be an integer"):
            load_turn_frames_config({"reconstruction": {"vggtomega": {"turn_frames": {
                "enabled": True, "per_rotation_quantum": 3, "min_per_interval": bad}}}})
    import yaml
    live = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert load_turn_frames_config(live) == TurnFramesConfig(True, 3, 1)


def _write_session(tmp_path, **kw):
    sel, recs, q = _session_docs(**kw)
    (tmp_path / "frames").mkdir()
    (tmp_path / "intake").mkdir()
    (tmp_path / "frames" / "selected_frames.json").write_text(json.dumps(sel))
    (tmp_path / "intake" / "coverage_warnings.json").write_text(json.dumps({"frames": recs}))
    (tmp_path / "frames" / "quality_features.json").write_text(json.dumps({"frames": q}))
    return tmp_path


def test_the_session_document_is_stamped_and_reproducible(tmp_path):
    scan = _write_session(tmp_path, sharp={17: 0.9, 24: 0.99})
    logs = []
    d1 = session_turn_frames(scan, TurnFramesConfig(True, 3, 0), log=logs.append)
    first = (scan / "intake" / TURN_FRAMES_NAME).read_text()
    d2 = session_turn_frames(scan, TurnFramesConfig(True, 3, 0), log=logs.append)
    assert (scan / "intake" / TURN_FRAMES_NAME).read_text() == first and d1 == d2
    assert d1["files"] == [_fn(4), _fn(17), _fn(24)] and d1["n_turn_frames"] == 3
    assert set(d1["inputs"]) == {"selected_frames_sha256", "coverage_warnings_sha256",
                                 "quality_features_sha256"} and d1["code_sha256"]
    assert any("[turn-frames] 3 turn frame(s) — Omega's context views" in m for m in logs), logs
    assert d1["min_per_interval"] == 0 and d1["by_reading"] == {"rotation": 3}
    assert d1["n_turn_frames"] == 3 and d1["n_witness_frames"] == 0 and d1["witness_files"] == []
    off = session_turn_frames(scan, TurnFramesConfig(False, 3, 1), log=logs.append)
    assert off["files"] == [] and off["enabled"] is False
    (scan / "intake" / "coverage_warnings.json").unlink()
    with pytest.raises(TurnFramesError, match="coverage_warnings"):
        session_turn_frames(scan, TurnFramesConfig(True, 3, 1))


def test_the_session_document_records_every_interval(tmp_path):
    scan = _write_session(tmp_path, rot_per_frame=0.1, sharp={5: 0.9, 21: 0.8})
    logs = []
    d = session_turn_frames(scan, TurnFramesConfig(True, 3, 1), log=logs.append)
    assert d["files"] == [] and d["witness_files"] == [_fn(5), _fn(21)]
    assert d["by_reading"] == {"parallax": 2} and d["n_witness_frames"] == 2
    assert d["version"] == 3 and d["min_per_interval"] == 1
    assert any("and 2 witness frame(s) for F6 only (≥ 1 in every other interval" in m for m in logs), logs


def test_the_card_holds_the_keyframes_and_the_context_views(tmp_path):
    from reconstruction.chunk_plan import session_omega_resolution
    card = {"card": "A100-SXM4-80GB", "total_gb": 80.0, "footprint_factor": 1.0}
    ranges = [(0, 150), (100, 183)]
    plain = session_omega_resolution(tmp_path, ranges, (1920, 1080), "max_size", 1920, 0.15, card,
                                     log=lambda m: None)
    assert plain["largest_chunk_frames"] == 150 and "context_counts" not in plain
    # zero context views: the same plan, the same key — the decision on disk is reused
    again = session_omega_resolution(tmp_path, ranges, (1920, 1080), "max_size", 1920, 0.15, card,
                                     log=lambda m: None, context_counts=[0, 0])
    assert again["persisted"] == "reused" and again["resolution"] == plain["resolution"]
    ctx = session_omega_resolution(tmp_path, ranges, (1920, 1080), "max_size", 1920, 0.15, card,
                                   log=lambda m: None, context_counts=[10, 90])
    assert ctx["persisted"] == "decided" and ctx["largest_chunk_frames"] == 173
    assert ctx["chunk_views"] == [160, 173] and ctx["context_counts"] == [10, 90]
    assert ctx["resolution"] <= plain["resolution"]
    doc = json.loads((tmp_path / "intake" / "omega_resolution.json").read_text())
    assert doc["key"]["context_counts"] == [10, 90]
    from reconstruction.chunk_plan import ChunkLayoutError
    with pytest.raises(ChunkLayoutError, match="context_counts"):
        session_omega_resolution(tmp_path, ranges, (1920, 1080), "max_size", 1920, 0.15, card,
                                 log=lambda m: None, context_counts=[1])


def test_the_map_worker_sizes_the_card_on_the_context_views_and_records_them():
    src = (Path(__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    # the CUBES (2026-11-06 on): each cube's context and witness views size the card and are recorded
    i_ctx = src.index('vggt_config["Model"]["chunk_context"] = _ctx')
    i_turn = src.rindex("_turn = session_turn_frames(", 0, i_ctx)
    i_res = src.index("_res = session_omega_resolution(", i_ctx)
    assert i_turn < i_ctx < i_res
    assert "context_counts=[max(len(c), len(w)) for c, w in zip(_ctx, _wit)]," in src[i_res:i_res + 600]
    assert "_n_max = max(n + max(len(c), len(w)) for n, c, w in zip(_chunk_frame_counts, _ctx, _wit))" in src
    assert 'vggt_config["Model"]["chunk_witnesses"] = _wit' in src
    assert "_persist_cube_plan(_cplan, _sets, _owner, _n_selected, _walk0, _res, _ctx)" in src
    assert '"chunk_context": _ctx}, pipe.send_log)' in src          # the single pass's check


def test_a_plan_with_other_context_views_wipes_the_old_products(tmp_path):
    from test_chunk_plan_invalidation import PCCR, _plan, _session, _wiped
    from workers.map_worker import _invalidate_on_new_chunk_plan, chunk_plan_doc
    ctx = [[], [_fn(100)], [], [], []]
    with_ctx = chunk_plan_doc(PCCR, 289, 17.3, {"D_total": 1.0}, {"resolution": 832,
                              "mode": "max_size"}, context=ctx)
    assert with_ctx["chunk_context"] == ctx and with_ctx["chunk_context_counts"] == [0, 1, 0, 0, 0]
    assert "chunk_context" not in _plan(PCCR)                   # no context: the record as before
    out = _session(tmp_path, _plan(PCCR))
    assert _invalidate_on_new_chunk_plan(out, with_ctx, log=lambda m: None) is True
    assert _wiped(out)
    out2 = _session(tmp_path / "b", with_ctx)
    assert _invalidate_on_new_chunk_plan(out2, with_ctx, log=lambda m: None) is False
    # a single pass records its context in its own run config
    out3 = _session(tmp_path / "c", old_config={"chunk_ranges": [[0, 289]], "omega_resolution": 832,
                                                "omega_mode": "max_size",
                                                "chunk_context": [[_fn(100)]]})
    one = {"chunk_ranges": [[0, 289]], "n_keyframes": 289,
           "omega_resolution": {"resolution": 832, "mode": "max_size"}}
    assert _invalidate_on_new_chunk_plan(out3, dict(one, chunk_context=[[_fn(100)]]),
                                         log=lambda m: None) is False
    assert _invalidate_on_new_chunk_plan(out3, dict(one, chunk_context=[[]]),
                                         log=lambda m: None) is True
