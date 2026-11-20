"""F6 (docs/plan_determinismo.md points 51, 52, 53, 60 — USER 2026-10-07): SAM3 masks enter only when stamped for
this reconstruction, the ONE confidence floor's arithmetic is pinned, every bar's margins are recorded, the camera
file round-trips exactly. CPU, synthetic."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import depth_on_f5 as D                              # noqa: E402

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])
QUIET = lambda *a: None  # noqa: E731


# ── point 52: THE ONE confidence floor, pinned ─────────────────────────────

def test_the_confidence_floor_is_min_plus_norm_times_the_span():
    v = np.array([0.2, 0.5, 1.0, 4.0])
    thr, cmax = D.confidence_floor(v, 0.10)
    assert thr == 0.2 + 0.10 * (4.0 - 0.2) and cmax == 4.0
    assert D.confidence_floor(np.array([3.0]), 0.5) == (3.0, 3.0)
    with pytest.raises(D.DepthOnF5Error):
        D.confidence_floor(np.zeros(0), 0.1)
    # the key it reads is the pipeline's ONE floor, shared with the viewer's slider arithmetic
    import inspect
    from config import cfg as raw_cfg
    assert float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"]) == 0.10
    src = inspect.getsource(D.compute)
    assert 'raw_cfg["reconstruction"]["simple"]["conf_min_norm"]' in src and "confidence_floor(" in src


# ── point 51: masks only when stamped for this reconstruction ─────────────

def _session_with_reconstruction(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    np.savez(out / "omega_run" / "results_output" / "frame_1.npz", depth=np.ones((4, 4), np.float32))
    (out / "camera_frames.txt").write_text("1\n")
    return out


def test_masks_without_a_stamp_or_from_another_reconstruction_are_ignored(tmp_path):
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
    out = _session_with_reconstruction(tmp_path)
    rid = reconstruction_id(out)
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8))
    masks = np.load(out / "seg_masks.npz")
    taken, why = D.masks_stamp_check(out, masks)
    assert not taken and RECONSTRUCTION_ID_KEY in why
    msgs = []
    assert D.mask_labels(out, H, W, log=msgs.append) is None
    assert any("IGNORED" in m and "point 51" in m for m in msgs)
    # another reconstruction's stamp: refused with the reason
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8), **{RECONSTRUCTION_ID_KEY: "0" * 64})
    taken, why = D.masks_stamp_check(out, np.load(out / "seg_masks.npz"))
    assert not taken and "another reconstruction" in why
    # this reconstruction's stamp in the store itself, or in segmentation.json next to it: taken
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8), **{RECONSTRUCTION_ID_KEY: rid})
    assert D.masks_stamp_check(out, np.load(out / "seg_masks.npz"))[0] is True
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8))
    (out / "segmentation.json").write_text(json.dumps({RECONSTRUCTION_ID_KEY: rid}))
    assert D.masks_stamp_check(out, np.load(out / "seg_masks.npz"))[0] is True
    # a session with no reconstruction to name takes no masks at all
    bare = tmp_path / "bare" / "output"
    bare.mkdir(parents=True)
    np.savez(bare / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8), **{RECONSTRUCTION_ID_KEY: rid})
    assert D.masks_stamp_check(bare, np.load(bare / "seg_masks.npz"))[0] is False


# ── point 53: the vote's bar and its margins ───────────────────────────────

def test_the_vote_reports_tau_and_every_frames_margins_to_it():
    cams = {}
    for f, x in zip((0, 1, 2), (-0.2, 0.0, 0.2)):
        c = np.eye(4); c[0, 3] = x; cams[f] = c
    order = [0, 1, 2]
    dep = {f: np.full((H, W), 3.0, np.float32) for f in order}
    dep[1][10:20, 30:40] = 1.5
    ok = {f: np.ones((H, W), bool) for f in order}
    bars = {}
    _, tau, tot = D.edge_keeping_vote(order, dep, ok, {f: ok[f].copy() for f in order}, K, cams, None,
                                      (-1, 1), 75.0, 2, False, log=QUIET, bars_out=bars)
    assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in tot.values())   # numeric for its readers
    assert bars["tau"] == tau and bars["tau_quantile"] == 75.0
    pf = bars["tau_margin_rel"]["per_frame"]
    assert set(pf) == {"0", "1", "2"} and pf["1"]["n"] > 0
    assert set(pf["1"]) == {"n", "p05", "p25", "p50", "p75", "p95", "share_beyond"}
    assert 0.0 <= pf["1"]["share_beyond"] <= 1.0 and bars["tau_margin_rel"]["median_of_frame_medians"] is not None


def test_margin_quantiles_report_the_share_beyond_the_bar():
    import torch
    from precision.f6_torch import margin_quantiles
    q = margin_quantiles(torch.tensor([1.0, 0.5, -0.5, float("nan")], dtype=torch.float64))
    assert q["n"] == 3 and abs(q["share_beyond"] - 1 / 3) < 1e-12 and q["p50"] == 0.5
    assert margin_quantiles(torch.zeros(0, dtype=torch.float64)) is None


# ── point 45 / 56: the camera file is exact ───────────────────────────────

def test_the_camera_file_is_float64_round_trip_exact(tmp_path):
    params = [391.123456789012345, 388.7, 234.8, 414.1, 0, 0, 0, 0]
    p = D.camera_travels(tmp_path, params, 2, log=QUIET)
    rows = np.loadtxt(p).reshape(-1, 4)
    assert rows.shape == (2, 4) and rows[0, 0] == params[0] and rows[1, 1] == 388.7
