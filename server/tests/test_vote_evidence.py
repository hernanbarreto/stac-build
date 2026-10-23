"""The edge-keeping vote after the audit of 2026-10-23: a neighbour's MIXED pixels (Omega's contour
ramps, in free space between object and wall) do not judge; a depth STEP is a kink of the inverse
depth, never a smooth oblique surface; PointDiT's refinement is judged on the held-out rows it
changed. Synthetic scenes, no GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import f6_torch as FT                                  # noqa: E402
from precision.corrected_cloud import depth_steps, mixed_pixels       # noqa: E402
from precision.depth_on_f5 import edge_keeping_vote, mono_heldout_gate  # noqa: E402

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])


def _cam(x):
    c = np.eye(4); c[0, 3] = x; return c


def _box_scene(cx_cam):
    """A wall at 4 m; a 0.6 m box at 2 m in front of it; Omega's 2-px ramp around the box's contour."""
    z = np.full((H, W), 4.0, np.float32)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    X2 = (uu - K[0, 2]) / K[0, 0] * 2.0 + cx_cam
    Y2 = (vv - K[1, 2]) / K[1, 1] * 2.0
    box = (np.abs(X2) <= 0.3) & (np.abs(Y2) <= 0.3)
    z[box] = 2.0
    ring1 = binary_dilation(box) & ~box
    ring2 = binary_dilation(ring1 | box) & ~(ring1 | box)
    z[ring1] = 2.6; z[ring2] = 3.3
    return z, box, ring1 | ring2


def _vote(judges_exclude_mixed: bool, edge_rule: str):
    """Keyframe 1 judged by two neighbours BEHIND it on the same side (a walk's ±3/±6 that both lie to
    one side, or a window's end): their ramps land on the same background pixels of keyframe 1. The
    three maps carry a 0.4 % scale disagreement, so the measured tau is a real tolerance."""
    order = [0, 1, 2]
    c2w = {0: _cam(-0.4), 1: _cam(0.0), 2: _cam(-0.2)}
    dep, box, ramp = {}, {}, {}
    for f, s in zip(order, (0.996, 1.0, 1.004)):
        dep[f], box[f], ramp[f] = _box_scene(c2w[f][0, 3])
        dep[f] = (dep[f] * s).astype(np.float32)
    ok = {f: np.ones((H, W), bool) for f in order}
    final, tau, tot = edge_keeping_vote(order, {f: dep[f].copy() for f in order}, ok, {f: ok[f].copy() for f in order},
                                        K, c2w, None, (-1, 1), 75.0, 2, log=lambda *a: None,
                                        judges_exclude_mixed=judges_exclude_mixed, edge_rule=edge_rule)
    z1, a1, c1 = final[1]
    bg = ~box[1] & ~ramp[1]
    near = bg & (np.abs(np.arange(W)[None, :] - W // 2) < 14) & (np.abs(np.arange(H)[:, None] - H // 2) < 14)
    return z1, a1, c1, near, box[1], tau, tot


def test_the_neighbours_contour_ramps_no_longer_contradict_the_wall_around_an_object():
    """Epoch 8 (judges_exclude_mixed false): the ramp pixels of the neighbours, floor-passing, splat
    in front of the wall next to the box and contradict it — the background around the object is
    repaired to the ramp's depth (a halo of flyers at 2.6-3.3 m) or dropped (holes). With the
    audit's rule those pixels do not judge: every wall pixel next to the box stays at 4 m, the box
    itself at 2 m either way. (pccr: measured on 116 background pixels around the box, epoch 8 left
    85 at the wall, 8 at ramp depth, 23 holes.)"""
    z_old, a_old, c_old, near, box, tau, _ = _vote(False, FT.EDGE_RULE_WINDOW)
    z_new, a_new, c_new, near2, _, _, tot = _vote(True, FT.EDGE_RULE_KINK)
    assert 0.002 < tau < 0.01
    wall_old = np.abs(z_old[near] - 4.0) < 0.06
    wall_new = np.abs(z_new[near] - 4.0) < 0.06
    at_ramp_old = (z_old[near] > 2.4) & (z_old[near] < 3.6)
    holes_old = z_old[near] == 0
    assert at_ramp_old.sum() > 0 and holes_old.sum() > 0, "the defect must reproduce under epoch 8's rules"
    assert wall_new.all(), f"{(~wall_new).sum()} wall pixel(s) next to the box are not at the wall"
    assert wall_new.sum() > wall_old.sum()
    assert np.abs(z_new[box] - 2.0).max() < 0.06 and np.abs(z_old[box] - 2.0).max() < 0.06
    # a wall pixel is confirmed, or occluded by the box in both neighbours (unjudgeable: 0 / 0);
    # a contradiction from the box's own silhouette (one pixel of splat resolution) never
    # outnumbers the confirmations
    assert (c_new[near] <= a_new[near]).all() and (a_new[near] >= 1).sum() > 0.6 * near.sum()
    assert tot["mixed_not_judging"] > 0
    # the window rule alone does not cure it — the judges do
    z_w, *_ = _vote(True, FT.EDGE_RULE_WINDOW)
    assert (np.abs(z_w[near] - 4.0) < 0.06).all()


def _slanted_floor(far_m: float, f_px: float = 70.0):
    """A floor 1.5 m under the camera up to ``far_m``: near the horizon the per-pixel |dz|/z grows beyond
    any tau while the surface is one plane."""
    vv = np.arange(H)[:, None].repeat(W, 1).astype(np.float64)
    yt = (vv - H / 2) / f_px
    z = np.where(yt > 0, 1.5 / np.maximum(yt, 1e-9), np.nan)
    valid = np.isfinite(z) & (z < far_m)
    return np.where(valid, z, 0.0).astype(np.float32), valid


def test_a_slanted_plane_near_the_horizon_is_not_an_edge_under_the_kink_rule():
    tau = 0.02
    z, valid = _slanted_floor(far_m=60.0)
    jump = np.abs(np.diff(z, axis=0)) / np.maximum(z[1:], 1e-9)
    assert (jump[valid[1:] & valid[:-1]] > tau).sum() > 100           # 1-px jumps above tau abound …
    old = depth_steps(z, valid, tau, FT.EDGE_RULE_WINDOW)
    new = depth_steps(z, valid, tau, FT.EDGE_RULE_KINK)
    assert old.sum() > 100 and new[valid].sum() <= 2 * W               # … the window rule flags them, the kink does not
    assert not mixed_pixels(z, valid, tau, FT.EDGE_RULE_KINK)[1:-1].any()
    # a real step on the same floor: a 0.3 m box standing on it — flagged on both sides, its ramp mixed
    z2 = z.copy(); v2 = valid.copy()
    r0, r1, c0, c1 = 40, 50, 30, 50
    z2[r0:r1, c0:c1] = z[r0:r1, c0:c1] * 0.7
    z2[r0 - 1, c0:c1] = (z[r0 - 1, c0:c1] * 0.85)                      # a 1-px ramp above the box
    steps = depth_steps(z2, v2, tau, FT.EDGE_RULE_KINK)
    assert steps[r0 - 2:r0 + 1, c0 + 5:c1 - 5].all() and steps[r1 - 1:r1 + 1, c0 + 5:c1 - 5].all()
    assert not steps[r0 + 3:r1 - 3, c0 + 3:c1 - 3].any()               # the box's interior is no edge
    mixed = mixed_pixels(z2, v2, tau, FT.EDGE_RULE_KINK)
    assert mixed[r0 - 1, c0 + 5:c1 - 5].all()
    # the band PointDiT works in follows the same rule
    disc, band = FT.discontinuity_band(FT._t(z2), FT._t(v2, FT.torch.bool), tau, 1, FT.EDGE_RULE_KINK)
    assert bool(FT._np(disc)[r0 - 1, c0 + 10]) and not FT._np(disc)[5:30, 5:W - 5].any()
    with pytest.raises(FT.F6TorchError):
        FT.depth_steps(FT._t(z2), FT._t(v2, FT.torch.bool), tau, "nonsense")


def test_pointdit_is_reverted_where_the_held_out_rows_it_changed_get_worse():
    """mono_heldout_gate: keyframe 0's refinement worsens its touched rows → reverted; keyframe 1's
    improves them → kept; a keyframe with too few judges follows the pooled verdict; an untouched
    one is untouched."""
    rng = np.random.default_rng(3)
    frames = [0, 1, 2, 3]
    n = 40
    u = rng.uniform(1, W - 2, n); v = rng.uniform(1, H - 2, n)
    r = np.rint(v).astype(int); c = np.rint(u).astype(int)
    z_lm = np.full(n, 3.0)
    before, after, rows = {}, {}, {}
    for i, f in enumerate(frames):
        before[f] = np.full((H, W), 3.0, np.float32) * (1 + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
        after[f] = before[f].copy()
        rows[i] = (u, v, z_lm, before[f][r, c].astype(np.float64))
    after[0][r, c] = before[0][r, c] * 1.06                              # worse on every judged row
    after[1][r, c] = 3.0                                                 # exact on every judged row
    rows[2] = (u[:2], v[:2], z_lm[:2], before[2][r[:2], c[:2]].astype(np.float64))
    after[2][r[:2], c[:2]] = 3.0                                         # two judges only
    keep, verdict = mono_heldout_gate(frames, before, after, rows, 2.0, 0.95, 200, 0,
                                      {f: 0.01 for f in frames}, log=lambda *a: None)
    assert keep == {0: False, 1: True, 2: True, 3: True}
    pf = verdict["per_frame"]
    assert pf["0"]["status"] == "reverted" and pf["1"]["status"] == "kept"
    assert pf["2"]["status"] == "unjudged" and pf["2"]["follows"] == "session pooled verdict"
    assert pf["3"]["status"] == "untouched" and verdict["pooled"]["rows"] == 82
    assert verdict["keyframes"] == {"kept": 1, "reverted": 1, "unjudged": 1, "untouched": 1}


def test_origins_without_the_contradiction_column_read_as_zeros(tmp_path):
    """The origins v2 field added by the audit (n_contradict): a file written before it reads 0 there;
    a writer that does not carry it writes 0."""
    import json
    import precision.provenance as PV
    n = 5
    cols = {k: np.arange(n, dtype=PV.V2_DTYPES[k]) for k in PV.V2_FIELDS if k != "n_contradict"}
    p = PV.write_origins(tmp_path / "o.npz", cols, {"stage": "test"})
    with np.load(p) as z:
        assert "n_contradict" in z.files and (z["n_contradict"] == 0).all()
    # an older file: the field absent on disk
    with np.load(p) as z:
        old = {k: z[k] for k in z.files if k != "n_contradict"}
    (tmp_path / "output").mkdir()
    np.savez(tmp_path / "output" / PV.ORIGINS_NAME, **old)
    from correction.session import write_ply
    header = [(s + "\n").encode("ascii") for s in
              ("ply", "format binary_little_endian 1.0", f"element vertex {n}", "property float x", "property float y",
               "property float z", "property int frame_global", "property int pixel_row", "property int pixel_col",
               "end_header")]
    data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("frame_global", "<i4"),
                              ("pixel_row", "<i4"), ("pixel_col", "<i4")])
    data["frame_global"] = np.arange(n)
    write_ply(tmp_path / "output" / "cleaned_cloud.ply", header, data)
    (tmp_path / "output" / "geometry_epoch.json").write_text(json.dumps({"epoch": int(old["geometry_epoch"][0])}))
    got = PV.load_origins(tmp_path / "output")
    assert got["version"] == 2 and (got["n_contradict"] == 0).all() and len(got["n_contradict"]) == n
