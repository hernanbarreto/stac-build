"""Wave 2, package P5 — the server side (main.py): the explicit floor levelling under the
user's rule (docs/plan_determinismo.md points 158 / 161), and the server never building,
projecting or levelling at load or at a run's end (points 111 / 157 / 158 / 160).

``import main`` loads the FastAPI app (no server, no GPU): ~25 s, once per module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERVER_DIR = Path(__file__).resolve().parents[1]
UI_APP = SERVER_DIR.parent / "ui" / "src" / "App.tsx"


@pytest.fixture(scope="module")
def main_mod():
    import main
    return main


def _floor(n_side: int, side_m: float, *, tilt_deg: float, height_m: float, noise_m: float,
           seed: int = 0) -> np.ndarray:
    """A square floor of ``side_m`` m (n_side² points), rotated by ``tilt_deg`` about x,
    lifted by ``height_m``, with Gaussian noise ``noise_m`` on its normal."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0.0, side_m, n_side)
    x, z = np.meshgrid(g, g)
    y = rng.normal(0.0, noise_m, x.shape)
    P = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    a = np.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    P = P @ Rx.T
    P[:, 1] += height_m
    return P


RULE = dict(cell_m=1.0, cell_min_points=300, error_factor=1.1, confidence=0.95, seed=7)


def test_a_tilted_offset_floor_is_levelled_and_a_level_one_is_left_alone(main_mod):
    tilted = _floor(120, 6.0, tilt_deg=2.0, height_m=0.05, noise_m=0.003)
    d = main_mod.floor_level_decision(tilted, **RULE)
    assert d["apply"] is True, d["reason"]
    assert d["n_judges"] >= 5 and d["decision"]["improves"]
    # the floor was rotated about x by 2° around the origin, then lifted 5 cm: its centroid
    # (3, 0, 3) sits at y = 0.05 - 3 sin 2°
    assert abs(d["tilt_deg"] - 2.0) < 0.2
    assert abs(d["height_m"] - (0.05 - 3.0 * np.sin(np.radians(2.0)))) < 0.01
    assert d["sigma_height_m"] > 0 and np.isfinite(d["sigma_tilt_deg"])
    # the same input gives the same decision, bit for bit (seeded plane, seeded bootstrap)
    d2 = main_mod.floor_level_decision(tilted, **RULE)
    assert json.dumps(d["decision"], sort_keys=True) == json.dumps(d2["decision"], sort_keys=True)
    assert np.array_equal(d["n_disp"], d2["n_disp"])

    level = _floor(120, 6.0, tilt_deg=0.0, height_m=0.0, noise_m=0.003)
    d = main_mod.floor_level_decision(level, **RULE)
    assert d["apply"] is False and d["decision"] is not None
    assert "improves" in d["decision"] and not d["decision"]["improves"]


def test_a_floor_too_small_for_five_judges_is_never_moved(main_mod):
    small = _floor(60, 1.5, tilt_deg=3.0, height_m=0.10, noise_m=0.002)
    d = main_mod.floor_level_decision(small, **RULE)
    assert d["apply"] is False and d["n_judges"] < 5
    assert not d["decision"]["enough_judges"]


def _session_with_floor(tmp_path, P: np.ndarray) -> Path:
    import open3d as o3d
    out = tmp_path / "output"
    out.mkdir()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(np.full((len(P), 3), 0.5))
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), pcd)
    c = P.mean(0).tolist()
    (out / "segmentation_result.json").write_text(json.dumps({
        "type": "segmentation", "version": "3.0", "instances": [{
            "instance_id": 1, "label": "white tiled floor", "total_points": int(len(P)),
            "globalIndices": list(range(len(P))),
            "obb": {"center": c, "half_extents": [3, 0.01, 3], "rotation": np.eye(3).tolist()}}]}))
    return out


def test_level_floor_core_writes_sealed_atomic_results_only_when_the_rule_says_so(main_mod, tmp_path):
    P = _floor(120, 6.0, tilt_deg=2.0, height_m=0.05, noise_m=0.003)
    out = _session_with_floor(tmp_path, P)
    res = main_mod.level_floor_core(out, out / "segmentation_result.json", out / "cleaned_cloud.ply",
                                    out / "floor_level.json", None, "auto", "t")
    assert res["ok"] and res["leveled"] and res["changed"], res
    assert (out / "floor_transform.npz").exists()
    fl = json.loads((out / "floor_level.json").read_text())
    assert fl["decision"]["improves"] and fl["measured"]["n_judges"] >= 5
    assert fl["stamp"]["cloud_sha256"] and "reconstruction_id" in fl["stamp"]
    assert fl["stamp"]["rule"]["error_factor"] > 0 and 0.5 < fl["stamp"]["rule"]["confidence"] < 1
    assert not list(out.glob(".floor_transform.*.tmp")), "the temporary file is gone"
    # under the new frame the floor sits at y = 0: a second request changes nothing, writes nothing
    m1 = (out / "floor_transform.npz").stat().st_mtime_ns
    res2 = main_mod.level_floor_core(out, out / "segmentation_result.json", out / "cleaned_cloud.ply",
                                     out / "floor_level.json", None, "auto", "t")
    assert res2["ok"] and not res2["changed"], res2
    assert (out / "floor_transform.npz").stat().st_mtime_ns == m1
    assert abs(res2["residual_mm"]) < 10


def test_the_server_never_builds_projects_or_levels_at_load_or_at_a_runs_end():
    src = (SERVER_DIR / "main.py").read_text()
    for gone in ("_run_cloudcompy_postprocess", "_onload_projection", "_stop_onload_projection",
                 "compute_leveling_from_points", "ply_override=_corr", "np.savez(transform_path"):
        assert gone not in src, gone
    # the only writer of floor_transform.npz in this process is the explicit floor action's
    assert src.count("floor_transform.npz") >= 3 and src.count("np.savez(") == 1
    # the completion handler and the load path READ; the octree is the stage's
    i = src.index("async def _notify_cloud_ready(sid):")
    j = src.index("async def _on_pipeline_complete(sid, success, restart_chat=True):")
    assert "convert_ply_to_potree_async" not in src[i:j]
    k = src.index("    # Start pipeline — support sequential multi-scan")
    assert "apply_segmentation_to_cloud" not in src[j:k] and "_session_intel_when_chat_up" not in src[j:k]
    assert "_read_segments_payload" in src[j:k]
    # the floor endpoint refuses the on-load mode before reading anything, and a busy session
    e = src.index('@app.post("/api/segmentation/level_floor")')
    body = src[e:e + 3000]
    assert 'if mode == "auto_if_needed":' in body and body.index('if mode == "auto_if_needed":') < body.index("ctx = _ctx(session_id)")
    assert "pipeline_manager.is_active(session_id)" in body
    assert "tilt_deg < 0.5" not in src and "abs(height) < 0.01" not in src
    # the Segmentation Manager's close enqueues a job; the server runs no matching
    r = src.index('@app.post("/api/segmentation/refresh")')
    rb = src[r:r + 4000]
    assert "_enqueue_service_job(" in rb and "_match_and_save_result" not in rb and "auto_run" not in rb
    # opening a session orders jobs instead of building
    assert 'kind="cloud"' in src and 'kind="projection"' in src
    # the UI never levels at load nor after the Manager closes
    ui = UI_APP.read_text()
    assert "applyFloorLevel(activeSession, 'auto_if_needed')" not in ui
    assert "'auto' | 'explicit' | 'auto_if_needed'" not in ui, "the on-load mode is gone from the type"
    assert "applyFloorLevel(sid, 'auto')" not in ui, "no automatic level after the Manager closes"
    assert "data.queued" in ui


def test_autosegment_save_refuses_a_change_while_a_job_of_the_scan_is_pending():
    src = (SERVER_DIR / "main.py").read_text()
    a = src.index('@app.post("/api/autosegment/{session_id}")')
    body = src[a:a + 3500]
    assert "pipeline_manager.is_active(session_id, scan)" in body and "409" in body
    assert body.index("changes") < body.index("save_vlm_prompt(ctx.output_dir")
