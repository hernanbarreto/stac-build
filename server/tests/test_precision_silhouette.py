"""precision/silhouette_filter.py — the SAM3-silhouette flyer filter of the corrected cloud,
on a synthetic scene with ground truth (CPU only):

    a post (0.3 × 2 × 0.3 m) 4 m in front of the birth keyframe, a wall at 7 m, six more
    keyframes on an arc around the post (±20°, ±35°, ±50°). The birth keyframe's points
    on the post are the BODY; the same pixels pushed 1.5 m back along their rays are a
    TAIL (Omega's feed-forward smear behind a post); wall points in front of the wall,
    born where no mask exists, are UNSEGMENTED flyers.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from precision import silhouette_filter as SF                      # noqa: E402

W, H = 160, 120
K = np.array([[150.0, 0, 79.5], [0, 150.0, 59.5], [0, 0, 1]])
POST = (np.array([-0.15, -1.0, 3.85]), np.array([0.15, 1.0, 4.15]))
WALL_Z = 7.0
YAWS = (0.0, 20.0, -20.0, 35.0, -35.0, 50.0, -50.0)
BIRTH = 0
TAIL_M = 1.5


def _c2w(yaw_deg: float) -> np.ndarray:
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([0.0, 0.0, 4.0]) - 4.0 * R[:, 2]            # 4 m from the post, facing it
    return T


def render(c2w: np.ndarray):
    """(depth = camera z, id: 1 post / 2 wall, world Y of the hit) per pixel."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ c2w[:3, :3].T
    C = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (POST[0] - C) / d
        t2 = (POST[1] - C) / d
    tn = np.nanmax(np.minimum(t1, t2), -1)
    tf = np.nanmin(np.maximum(t1, t2), -1)
    hit_post = (tf >= tn) & (tn > 0)
    t_wall = (WALL_Z - C[2]) / d[..., 2]
    t = np.where(hit_post, tn, t_wall)                    # the post is in front of the wall
    ident = np.where(hit_post, 1, 2)
    Y = C[1] + t * d[..., 1]
    return t, ident, Y                                     # d has camera z = 1: t IS the depth


@pytest.fixture(scope="module")
def scene():
    c2w = np.stack([_c2w(a) for a in YAWS])
    w2c = np.linalg.inv(c2w)
    rend = [render(T) for T in c2w]
    depth = [r[0] for r in rend]
    ident = [r[1] for r in rend]
    # masks on the native grid: the post everywhere; the wall only on the LEFT half
    masks = {j: {0: ident[j] == 1, 1: (ident[j] == 2) & (np.arange(W)[None, :] < W // 2)}
             for j in range(len(YAWS))}
    # points born in the birth keyframe
    t, idb, Yb = rend[BIRTH]
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    C = c2w[BIRTH][:3, 3]
    R = c2w[BIRTH][:3, :3]

    def pts(sel, extra):
        z = t[sel] + extra
        return (ray[sel] * z[:, None]) @ R.T + C, v[sel], u[sel]
    post = idb == 1
    right_wall = (idb == 2) & (u >= W // 2 + 20) & (u % 3 == 0) & (v % 3 == 0)   # clear of the post
    Xb, rb, cb = pts(post, 0.0)
    Xt, rt, ct = pts(post, TAIL_M)
    Xf, rf, cf = pts(right_wall, -1.5)                     # flyers in front of the wall, no mask
    xyz = np.concatenate([Xb, Xt, Xf])
    rows = np.concatenate([rb, rt, rf])
    cols = np.concatenate([cb, ct, cf])
    kind = np.concatenate([np.zeros(len(Xb), int), np.ones(len(Xt), int), np.full(len(Xf), 2)])
    Ypt = np.concatenate([Yb[post], Yb[post], Yb[right_wall]])
    return {"c2w": c2w, "w2c": w2c, "depth": depth, "masks": masks, "xyz": xyz, "rows": rows,
            "cols": cols, "kind": kind, "Y": Ypt}


@pytest.fixture(scope="module")
def params():
    from config import cfg as raw
    from precision.config import load_precision_config
    p = SF.params_from(load_precision_config(), raw)
    return replace(p, device="cpu")


def _members(sc, masks):
    """{oid: points whose birth pixel is inside the oid's mask in the birth keyframe}."""
    out = {}
    for oid, m in masks[BIRTH].items():
        ix = np.flatnonzero(m[sc["rows"], sc["cols"]])
        if len(ix):
            out[oid] = ix
    return out


def _run(sc, params, masks=None, labels=None, depth=None):
    masks = masks or sc["masks"]
    labels = labels or {0: "post", 1: "wall"}
    depth = depth or sc["depth"]
    n = len(sc["xyz"])
    mkf = {o: [j for j in masks if o in masks[j] and masks[j][o].any()] for o in labels}
    ident = np.arange(H * W, dtype=np.int64)
    return SF.silhouette_verdict(sc["xyz"], np.full(n, BIRTH), _members(sc, masks), labels, mkf, sc["w2c"], K,
                                 (H, W), lambda j: depth[j], lambda j: masks[j], ident, (H, W), params,
                                 log=lambda m: None)


def test_a_post_with_a_backward_tail_loses_the_tail_and_keeps_the_body(scene, params):
    keep, rep = _run(scene, params)
    k = scene["kind"]
    assert (k == 1).sum() > 200
    assert not keep[k == 1].any(), f"{int(keep[k == 1].sum())} tail points survived"
    assert keep[k == 0].all(), f"{int((~keep[k == 0]).sum())} body points left"
    assert keep[k == 2].all()                                   # unsegmented: never touched
    assert rep["dropped"] == int((k == 1).sum())
    assert rep["dropped_by_label"]["post"]["own_silhouette"] == rep["dropped"]
    assert rep["n_unsegmented_points"] == int((k == 2).sum())
    assert rep["eligible_views_per_membership"]["median"] >= params.min_votes
    # the cap is a cost cap: fewer views tested, never more than asked
    keep2, rep2 = _run(scene, replace(params, max_views=3))
    assert rep2["views_tested_per_membership"]["max"] <= 3
    assert rep2["pairs_tested"] < rep["pairs_tested"]


def test_occluded_views_do_not_vote(scene, params):
    """Every other keyframe's own depth says something stands 1 m in front of its camera:
    the post is occluded in all of them, nobody may judge it — the tail stays."""
    depth = [d if j == BIRTH else np.full_like(d, 1.0) for j, d in enumerate(scene["depth"])]
    keep, rep = _run(scene, params, depth=depth)
    assert keep.all()
    assert rep["eligible_views_per_membership"]["max"] == 0
    # a view with NO measured depth there does not occlude (the witness module's rule)
    depth0 = [d if j == BIRTH else np.zeros_like(d) for j, d in enumerate(scene["depth"])]
    keep0, _ = _run(scene, params, depth=depth0)
    assert not keep0[scene["kind"] == 1].any()


def _split_masks(sc):
    """The post as TWO masklets in the other keyframes — oid 0 above y = 0, oid 2 below —
    and as oid 0 alone in the birth keyframe (every body point belongs to oid 0 only)."""
    out = {}
    for j in sc["masks"]:
        m = dict(sc["masks"][j])
        if j != BIRTH:
            _, ident, Y = render(sc["c2w"][j])
            m[0] = (ident == 1) & (Y > 0)
            m[2] = (ident == 1) & (Y <= 0)
        out[j] = m
    return out


def test_same_label_masklets_never_conflict(scene, params):
    masks = _split_masks(scene)
    keep, rep = _run(scene, params, masks=masks, labels={0: "post", 1: "wall", 2: "post"})
    body = scene["kind"] == 0
    assert keep[body].all(), f"{int((~keep[body]).sum())} body points left"
    assert not keep[scene["kind"] == 1].any()


def test_another_labels_mask_removes(scene, params):
    """The lower half of the post is labelled 'pedestal' in the other views: the body points
    below y = 0 land inside another label's mask and never inside their own — they leave,
    by rule 2 even when rule 1 cannot judge (min_votes above the views available)."""
    masks = _split_masks(scene)
    labels = {0: "post", 1: "wall", 2: "pedestal"}
    body = scene["kind"] == 0
    low = body & (scene["Y"] < -0.2)
    high = body & (scene["Y"] > 0.2)
    assert low.sum() > 50 and high.sum() > 50
    keep, rep = _run(scene, params, masks=masks, labels=labels)
    assert not keep[low].any() and keep[high].all()
    keep2, rep2 = _run(scene, replace(params, min_votes=len(YAWS) + 1), masks=masks, labels=labels)
    assert not keep2[low].any() and keep2[high].all()
    assert rep2["dropped_by_rule"]["other_mask"] > 0 and rep2["dropped_by_rule"]["own_silhouette"] == 0
    assert keep2[scene["kind"] == 2].all()


def test_a_point_of_two_masklets_leaves_only_when_both_say_so(scene, params):
    """The tail pixels also inside a second masklet of the birth keyframe whose mask
    follows them in every view: that membership keeps them."""
    masks = {j: dict(m) for j, m in scene["masks"].items()}
    tail = scene["kind"] == 1
    for j in masks:
        P = scene["xyz"][tail] @ scene["w2c"][j][:3, :3].T + scene["w2c"][j][:3, 3]
        uu = np.clip(np.rint(K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]).astype(int), 0, W - 1)
        vv = np.clip(np.rint(K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]).astype(int), 0, H - 1)
        m = np.zeros((H, W), bool)
        m[vv, uu] = True
        masks[j][3] = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    keep, _ = _run(scene, params, masks=masks, labels={0: "post", 1: "wall", 3: "smear"})
    assert keep[tail].all()


def test_the_mask_lut_follows_the_lens_to_the_mask_grid():
    """F6's undistorted pixel → the original frame through the lens → a half-resolution
    full-frame mask grid, against the camera's own forward distortion."""
    from precision.camera import CameraModel, GridMap, distort_points, mask_grid_for, native_to_grid, undistort_maps
    g = GridMap("omega", W, H, W, H, 0, 0, 0, 0, W, H, W, H)
    cam = CameraModel(W, H, (150.0, 150.0, 79.5, 59.5, -0.08, 0.02, 0.001, -0.001), "synthetic", 1, g)
    m1, m2, _ = undistort_maps(cam)
    lut = SF.mask_lut((m1, m2), (W, H), (H // 2, W // 2))
    v, u = np.mgrid[0:H, 0:W]
    uv = distort_points(np.stack([u, v], -1).reshape(-1, 2).astype(np.float64), cam)
    gg = mask_grid_for(W, H, (H // 2, W // 2))
    q = native_to_grid(uv, gg)
    c, r = np.rint(q[:, 0]), np.rint(q[:, 1])
    ok = (c >= 0) & (c < gg.w) & (r >= 0) & (r < gg.h)
    want = np.where(ok, r * gg.w + c, -1).astype(np.int64)
    inside = (lut >= 0) & (want >= 0)
    assert inside.mean() > 0.9
    assert (lut[inside] == want[inside]).mean() > 0.99
    # the ones that differ sit on a rounding boundary: one mask pixel at most
    d = np.abs(lut[inside] // gg.w - want[inside] // gg.w) + np.abs(lut[inside] % gg.w - want[inside] % gg.w)
    assert d.max() <= 2


def _write_ply(path: Path, xyz, fg, pr, pc, tag, extra=0.0):
    from correction.session import write_ply
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"),
          ("pixel_col", "<i2"), ("mv_votes", "u1")]
    a = np.empty(len(xyz), dt)
    a["x"], a["y"], a["z"] = xyz[:, 0] + extra, xyz[:, 1], xyz[:, 2]
    a["frame_global"], a["pixel_row"], a["pixel_col"], a["mv_votes"] = fg, pr, pc, tag
    hdr = [b"ply\n", b"format binary_little_endian 1.0\n", f"element vertex {len(a)}\n".encode()]
    hdr += [b"property float x\n", b"property float y\n", b"property float z\n", b"property int frame_global\n",
            b"property short pixel_row\n", b"property short pixel_col\n", b"property uchar mv_votes\n",
            b"end_header\n"]
    write_ply(path, hdr, a)


def test_the_step_filters_both_plys_with_the_same_rows(tmp_path, scene, params):
    """run_filter on a session: segmentation.json + a HALF-resolution mask store keyed by
    keyframe position, camera_frames.txt, the two PLYs of the cloud stage (the raw one
    with other coordinates, as after the consolidation) — the same rows leave both."""
    from correction.session import read_ply
    out = tmp_path / "output"
    tmp = out / "_tx"
    tmp.mkdir(parents=True)
    frames_n = [10 * (j + 1) for j in range(len(YAWS))]
    (out / "camera_frames.txt").write_text("\n".join(map(str, frames_n)) + "\n")
    (out / "segmentation.json").write_text(json.dumps({
        "mask_file": "seg_masks.npz",
        "instances": [{"id": 0, "label": "post", "instance_id": 1}, {"id": 1, "label": "wall", "instance_id": 2}]}))
    store = {"mask_frame_space": np.array("keyframe_position")}
    for j, mm in scene["masks"].items():
        for oid, m in mm.items():
            store[f"f{j}_o{oid}"] = cv2.resize(m.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    np.savez_compressed(out / "seg_masks.npz", **store)
    n = len(scene["xyz"])
    fg = np.full(n, frames_n[BIRTH])
    tag = scene["kind"] + 1                                  # the row's kind travels in mv_votes
    _write_ply(tmp / "cleaned_cloud.ply", scene["xyz"], fg, scene["rows"], scene["cols"], tag)
    _write_ply(tmp / "cleaned_cloud_raw.ply", scene["xyz"], fg, scene["rows"], scene["cols"], tag, extra=0.001)
    frames = {f: {"depth": scene["depth"][j]} for j, f in enumerate(frames_n)}
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    rep = SF.run_filter(out, tmp, frames, frames_n, scene["w2c"], K, (W, H), (u, v), params, log=lambda m: None)
    assert rep["ran"] and rep["mask_grid"] == [H // 2, W // 2]
    _, a = read_ply(tmp / "cleaned_cloud.ply")
    _, b = read_ply(tmp / "cleaned_cloud_raw.ply")
    assert len(a) == len(b) == rep["kept"] == n - rep["dropped"]
    for k in ("frame_global", "pixel_row", "pixel_col", "mv_votes", "y", "z"):
        assert np.array_equal(a[k], b[k]), k
    assert np.allclose(b["x"] - a["x"], 0.001, atol=1e-5)          # row i of raw IS row i of the cloud
    kind = scene["kind"]
    kept = np.bincount(a["mv_votes"].astype(int) - 1, minlength=3)
    assert kept[1] <= 0.05 * (kind == 1).sum()                       # the tail, at half mask resolution
    assert kept[0] >= 0.98 * (kind == 0).sum()                       # the body
    assert kept[2] == (kind == 2).sum()                              # unsegmented: untouched


def test_no_segmentation_touches_nothing(tmp_path, params):
    out = tmp_path / "output"
    out.mkdir()
    rep = SF.run_filter(out, out, {}, [1], np.eye(4)[None], K, (W, H), (None, None), params, log=lambda m: None)
    assert rep["ran"] is False


def test_config_keys_are_mandatory_and_the_device_never_falls_back():
    import copy
    from config import cfg as raw
    from precision.config import PrecisionConfigError, load_precision_config
    c = load_precision_config().cloud
    assert c.silhouette_filter is True and c.silhouette_max_views >= 1 and 0 < c.silhouette_min_inside_frac <= 1
    for key in ("silhouette_filter", "silhouette_max_views", "silhouette_min_votes", "silhouette_min_inside_frac",
                "silhouette_device"):
        bad = copy.deepcopy(raw)
        del bad["reconstruction"]["precision"]["cloud"][key]
        with pytest.raises(PrecisionConfigError, match=f"cloud.{key}"):
            load_precision_config(bad)
    bad = copy.deepcopy(raw)
    del bad["segmentation"]["mask_filter"]["dilate_px"]
    with pytest.raises(SF.SilhouetteError, match="dilate_px"):
        SF.params_from(load_precision_config(), bad)
    if not torch.cuda.is_available():
        with pytest.raises(SF.SilhouetteError, match="no CUDA"):
            SF.torch_device("cuda")
