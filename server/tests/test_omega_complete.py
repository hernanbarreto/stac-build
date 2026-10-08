"""Omega's stamped completion marker (reconstruction/omega_complete.py) and the map worker's wiring
of docs/plan_determinismo.md points 3, 4, 12, 13, 14, 22, 23: the fork is skipped only on a matching
stamp, the card comes from torch + the committed table, an OOM fails, the environment is recorded."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro                                                    # noqa: E402
from reconstruction import omega_complete as OC                 # noqa: E402

SRC = (Path(__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()


def _session(tmp_path, n=3):
    sess = tmp_path / "s"
    frames = sess / "frames"
    frames.mkdir(parents=True)
    names = [f"{i:06d}.jpg" for i in range(n)]
    for i, nm in enumerate(names):
        (frames / nm).write_bytes(bytes([i] * 16))
    (sess / "intake").mkdir()
    (sess / "intake" / "walk.json").write_text(json.dumps({"version": 2, "n_keyframes": n}))
    out = sess / "output"
    ml = out / "maplong_run"
    ml.mkdir(parents=True)
    for p, txt in ((ml / "camera_poses.txt", "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"),
                   (out / "camera_poses.txt", "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"),
                   (ml / "chunk_sim3.json", '{"chunk_indices": [[0, 3]]}'),
                   (ml / "frame_list.json", json.dumps(names)),
                   (out / "camera_frames.txt", "0\n1\n2\n")):
        p.write_text(txt)
    cfg = {"Model": {"chunk_ranges": [[0, n]], "omega_resolution": 832, "mask_sky": False,
                     "omega_resolution_report": {"resolution": 832, "persisted": "decided"}},
           "Weights": {"model": "Other"}, "Loop": {"SALAD": {"min_gap": 20}}}
    return sess, frames, names, cfg


def test_launch_stamp_is_of_the_keyframes_walk_code_and_config(tmp_path):
    sess, frames, names, cfg = _session(tmp_path)
    a = OC.launch_stamp(sess, frames, names, cfg)
    assert set(a["inputs"]) == {"frames/000000.jpg", "frames/000001.jpg", "frames/000002.jpg",
                                "intake/walk.json"}
    assert "vendor/VGGT-Long/vggt_long.py" in a["code"] and "server/repro.py" in a["code"]
    assert any(k.startswith("vendor/VGGT-Long/loop_utils/") for k in a["code"])
    assert any(k.startswith("vendor/vggt-omega/") for k in a["code"])
    assert a == OC.launch_stamp(sess, frames, names, cfg), "deterministic"
    # the log-only field of the resolution report is not part of it
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["Model"]["omega_resolution_report"]["persisted"] = "reused"
    assert OC.launch_stamp(sess, frames, names, cfg2)["sha256"] == a["sha256"]
    # a keyframe byte, the order, the walk, the config: each another launch
    (frames / names[1]).write_bytes(b"\x07" * 16)
    b = OC.launch_stamp(sess, frames, names, cfg)
    assert repro.check_stamp(a, b) == ["input 'frames/000001.jpg' changed "
                                       f"({a['inputs']['frames/000001.jpg'][:12]} -> {b['inputs']['frames/000001.jpg'][:12]})"]
    assert OC.launch_stamp(sess, frames, names[::-1], cfg)["sha256"] != b["sha256"]
    cfg3 = json.loads(json.dumps(cfg)); cfg3["Loop"]["SALAD"]["min_gap"] = 21
    assert "config 'fork_config' changed" in " ".join(repro.check_stamp(b, OC.launch_stamp(sess, frames, names, cfg3)))
    (sess / "intake" / "walk.json").write_text("{}")
    assert any("intake/walk.json" in d for d in repro.check_stamp(b, OC.launch_stamp(sess, frames, names, cfg)))
    with pytest.raises(OC.OmegaCompleteError, match="keyframe"):
        OC.launch_stamp(sess, frames, names + ["nope.jpg"], cfg)
    (sess / "intake" / "walk.json").unlink()
    with pytest.raises(OC.OmegaCompleteError, match="walk.json"):
        OC.launch_stamp(sess, frames, names, cfg)
    # the revisit reference enters when present
    (sess / "intake" / "walk.json").write_text("{}")
    (sess / "output" / "salad_revisit_reference.json").write_text("{}")
    assert "salad_revisit_reference.json" in OC.launch_stamp(sess, frames, names, cfg)["inputs"]


def test_check_passes_only_on_the_same_stamp_and_untouched_products(tmp_path):
    sess, frames, names, cfg = _session(tmp_path)
    out = sess / "output"
    now = OC.launch_stamp(sess, frames, names, cfg)
    ok, why = OC.check(out, now)
    assert not ok and "no completion marker" in why[0]
    p = OC.write(out, now)
    assert p == out / "maplong_run" / OC.MARKER_NAME
    doc = json.loads(p.read_text())
    assert doc["products"]["maplong_run/camera_poses.txt"] is None          # transformed later: existence
    assert len(doc["products"]["maplong_run/chunk_sim3.json"]) == 64            # never transformed: sha
    assert OC.check(out, now) == (True, [])
    # scale_align / orient rewrite the poses afterwards: still complete
    (out / "camera_poses.txt").write_text("2 0 0 0 0 2 0 0 0 0 2 0 0 0 0 1\n")
    assert OC.check(out, now)[0]
    # a product the fork wrote changed, or is gone: not complete, named
    (out / "maplong_run" / "chunk_sim3.json").write_text('{"chunk_indices": [[0, 2]]}')
    ok, why = OC.check(out, now)
    assert not ok and any("chunk_sim3.json changed" in d for d in why)
    (out / "maplong_run" / "chunk_sim3.json").unlink()
    assert any("chunk_sim3.json is gone" in d for d in OC.check(out, now)[1])
    # another launch: not complete, every difference named
    cfg2 = json.loads(json.dumps(cfg)); cfg2["Model"]["omega_resolution"] = 816
    ok, why = OC.check(out, OC.launch_stamp(sess, frames, names, cfg2))
    assert not ok and any("fork_config" in d for d in why)
    OC.clear(out)
    assert not p.exists() and OC.check(out, now)[0] is False
    (out / "maplong_run" / "camera_poses.txt").unlink()
    with pytest.raises(OC.OmegaCompleteError, match="nothing to mark complete"):
        OC.write(out, now)


def test_no_cleanup_deletes_the_marker_and_a_new_plan_does():
    """_cleanup_recon_temps and _discard_previous_epochs never touch maplong_run/omega_complete.json;
    _invalidate_on_new_chunk_plan (another plan) wipes it with the products — right, they go too."""
    from workers import map_worker as MW
    assert OC.MARKER_NAME not in MW._PLAN_INDEPENDENT_MAPLONG
    body = SRC[SRC.index("def _cleanup_recon_temps"):SRC.index("def _postprocess_reconstruction")]
    for d in ("_tmp_results_unaligned", "pcd", "_tmp_results_loop", "uncert"):
        assert d in body
    assert "omega_complete" not in body and "glob(" not in body.split("da3_run")[0].replace("rglob(", "")
    body2 = SRC[SRC.index("def _discard_previous_epochs"):SRC.index("def _run_pose_refine_step")]
    assert "maplong_run" not in body2 and "omega_complete" not in body2


def test_the_worker_skips_the_fork_only_on_a_matching_stamp_and_fails_on_oom():
    blk = SRC[SRC.index('_tag1 = "chunked-metric"'):SRC.index("    if not _ok:\n        return")]
    assert "_launch = _OC.launch_stamp(output_dir.parent, frames_dir, _sel_files, vggt_config)" in blk
    assert "_done, _why = _OC.check(output_dir, _launch)" in blk
    assert blk.index("_OC.check(") < blk.index("repro.require_exclusive_gpu(log=pipe.send_log)") \
        < blk.index("_ok = _omega_pass(vggt_config, _tag1)") < blk.index("_OC.write(output_dir, _launch)")
    assert "_wipe_plan_products(output_dir" in blk, "a stale marker wipes the products of the other launch"
    assert "ran OUT OF MEMORY" in blk and "the run FAILS" in blk and "is lowered to fit" in blk
    assert "--omega-footprint" in blk and "--from-oom-log" in blk and "--predicted-peak-gib" in blk
    for gone in ("record_omega_oom", "_prev - _OPS", "oom_fallback_from", "retrying at", "while True:"):
        assert gone not in SRC[SRC.index("def _run_vggtomega"):SRC.index("_PLAN_INDEPENDENT_OUTPUT =")], gone
    assert SRC.count("_omega_pass(vggt_config") == 1, "ONE Omega pass, never a re-run"


def test_the_worker_reads_the_card_through_torch_and_the_table_and_persists_the_resolution():
    blk = SRC[SRC.index("def _run_vggtomega"):SRC.index("_PLAN_INDEPENDENT_OUTPUT =")]
    assert "_ident = repro.card_identity(0)" in blk and '_card = omega_card(_ident["key"])' in blk
    assert "_res = session_omega_resolution(output_dir.parent, _ranges" in blk
    for gone in ("card_name", "_gpu_total_gb", "_gpu_free_gb", "omega_footprint_factor(", "nvidia-smi"):
        assert gone not in blk, gone
    assert "_env_rec = repro.environment_record(gpu=True)" in blk
    assert "omega_plan_environment.json" in blk
    assert "chunk_plan_doc(_ranges, _n_kf, _walk, _covis, _res, environment=_env_rec)" in blk
    assert 'if k != "persisted"' in blk, "the log-only field stays out of the stamped config"
    assert "_legacy_fp.unlink()" in blk and "weights" in blk.split("_legacy_fp = ")[1].split("\n")[0]


def test_the_worker_checks_the_card_free_before_every_gpu_step_and_regenerates_i3_as_a_new_walk():
    anchor = SRC[SRC.index("def _run_da3_anchor"):SRC.index("def math_deg")]
    assert "repro.require_exclusive_gpu(log=pipe.send_log)" in anchor
    assert anchor.index("require_exclusive_gpu") < anchor.index("extract_anchor_depths(frames_dir")
    blk = SRC[SRC.index("def _run_vggtomega"):SRC.index("_PLAN_INDEPENDENT_OUTPUT =")]
    i3 = blk[blk.index("if walk_is_current("):blk.index("_walk_doc = measure_walk(")]
    assert "for_new_walk=True" in i3
    regen = blk[blk.index("except CovisError as _ce"):blk.index("_ranges = [(int(a), int(b)) for a, b in _cplan")]
    assert "run_da3_windows(" in regen and "for_new_walk" not in regen, "a regeneration of the walk's windows"
    # the fork's launch environment: repro's deterministic env + the one HF cache
    omega = blk[blk.index("def _omega_pass"):blk.index("_chunked_already = False")]
    assert "env = repro.deterministic_env()" in omega and 'env["HF_HOME"] = da3_weights.HF_HOME' in omega
    assert 'env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"' not in omega and 'env["PYTHONHASHSEED"]' not in omega


def test_the_absolute_rows_enter_only_for_this_plan():
    blk = SRC[SRC.index("def _apply_chunked_metric"):SRC.index("def _ensure_anchors")]
    assert "absolute_rows_for_plan(_abs_path, _ranges, _n_selected)" in blk
    assert "IGNORED" in blk and 'json.loads(_abs_path.read_text()).get("rows"' not in blk


def test_chunk_plan_doc_carries_the_environment_and_drops_the_log_only_field():
    from workers.map_worker import chunk_plan_doc
    env = {"gpu": {"name": "A100"}, "torch": {"version": "2.5"}}
    doc = chunk_plan_doc([(0, 60), (30, 90)], 90, 12.0, {"D_total": 20.0},
                         {"resolution": 832, "mode": "max_size", "persisted": "decided"}, environment=env)
    assert doc["environment"] == env and "persisted" not in doc["omega_resolution"]
    assert "environment" not in chunk_plan_doc([(0, 60)], 60, 1.0, {}, {"resolution": 832, "mode": "max_size"})
    json.dumps(doc)
