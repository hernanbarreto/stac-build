"""Phase 0 of the mono-detail work (claude_stac.txt 2026-10-04): the flyer diagnosis. Synthetic, no GPU:
a plate in front of a wall seen by three cameras; one injected flyer of each class must land in its class."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import flyers as FL  # noqa: E402
from precision.config import load_precision_config  # noqa: E402

H, W = 64, 96
F = 80.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
Z_WALL, Z_PLATE = 3.0, 1.5
NEIGHBORS = (-2, -1, 1, 2)


def _cams():
    # camera 0 at x = 0; 1 and 2 shifted right, so the left border of frame 0 is seen by nobody else
    out = {}
    for f, x in zip((0, 1, 2), (0.0, 0.9, 1.8)):
        c = np.eye(4); c[0, 3] = x; out[f] = c
    return out


def _true_depth(c2w):
    """Analytic depth of the wall (z = 3) with a plate (z = 1.5, |x| <= 0.35, |y| <= 0.25) in front."""
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dx, dy = (uu - K[0, 2]) / F, (vv - K[1, 2]) / F
    cx = c2w[0, 3]
    xw = cx + dx * Z_PLATE; yw = dy * Z_PLATE
    plate = (np.abs(xw) <= 0.35) & (np.abs(yw) <= 0.25)
    return np.where(plate, Z_PLATE, Z_WALL)


def _scene():
    cams = _cams()
    frames = [0, 1, 2]
    xyz, fr, row, col = [], [], [], []
    for f in frames:
        z = _true_depth(cams[f])
        for v in range(0, H, 2):
            for u in range(0, W, 2):
                d = z[v, u]
                p = np.array([(u - K[0, 2]) / F * d, (v - K[1, 2]) / F * d, d, 1.0])
                xyz.append((cams[f] @ p)[:3]); fr.append(f); row.append(v); col.append(u)
    xyz = np.array(xyz); fr = np.array(fr); row = np.array(row); col = np.array(col)
    # textured gray image (random), with a FLAT patch on the wall's left side (seen by camera 0 only)
    rng = np.random.default_rng(1)
    imgs = {f: rng.integers(0, 255, (H, W)).astype(np.float64) for f in frames}
    flat = (slice(40, 62), slice(2, 20))
    imgs[0][flat] = 120.0
    return cams, frames, xyz, fr, row, col, imgs, flat


def _replace_point(xyz, fr, row, col, f, v, u, depth, cams):
    """Move the point born at (f, v, u) to ``depth`` along its own ray (it stays that pixel's point)."""
    i = np.nonzero((fr == f) & (row == v) & (col == u))[0]
    assert len(i) == 1
    p = np.array([(u - K[0, 2]) / F * depth, (v - K[1, 2]) / F * depth, depth, 1.0])
    xyz[i[0]] = (cams[f] @ p)[:3]
    return int(i[0])


def _cfg():
    p = load_precision_config()
    return replace(p.flyers, knn_k=4, isolated_quantile=99.0, step_quantile=95.0, band_px=2,
                   view_quantile=75.0, texture_quantile=10.0, texture_window_px=5, specular_gray=250,
                   sample_points=100000, seed=0)


def _energy(imgs, fr, row, col):
    e = np.full(len(fr), np.nan); g = np.full(len(fr), np.nan)
    for f, img in imgs.items():
        E, M = FL.texture_maps(img, 5)
        m = fr == f
        e[m] = E[row[m], col[m]]; g[m] = M[row[m], col[m]]
    return e, g


def test_one_flyer_of_each_class_lands_in_its_class():
    cams, frames, xyz, fr, row, col, imgs, flat = _scene()
    w2c = {f: np.linalg.inv(cams[f]) for f in frames}
    z0 = _true_depth(cams[0])
    # (a) a mixed edge pixel: the plate's right edge in frame 0, at a depth between plate and wall
    edge_u = int(np.nonzero(np.diff(z0[H // 2]) != 0)[0][-1])
    edge_u -= edge_u % 2
    ia = _replace_point(xyz, fr, row, col, 0, H // 2, edge_u, 2.2, cams)
    # (b) a point floating in free space in front of the wall where cameras 1 and 2 see the wall behind it
    ib = _replace_point(xyz, fr, row, col, 0, 10, 70, 2.0, cams)
    # (c) an isolated point on the FLAT patch of the wall, seen by camera 0 only (left border)
    ic = _replace_point(xyz, fr, row, col, 0, 50, 10, 2.6, cams)
    # (d) an isolated point in a textured region seen by camera 0 only, nobody contradicts it
    idd = _replace_point(xyz, fr, row, col, 0, 20, 10, 2.6, cams)
    energy, gray = _energy(imgs, fr, row, col)
    cls, bars = FL.diagnose(xyz, fr, row, col, K, H, W, frames, w2c, _cfg(), NEIGHBORS, energy, gray,
                            log=lambda m: None)
    assert cls[ia] == 1, (cls[ia], bars)
    assert cls[ib] == 2, (cls[ib], bars)
    assert cls[ic] == 3, (cls[ic], bars)
    assert cls[idd] == 4, (cls[idd], bars)
    # the regular surface points are not flyers (the isolation bar is a percentile: a few land above it,
    # but never a majority)
    assert (cls > 0).mean() < 0.02
    rep = FL.counts(cls)
    assert rep["by_class"]["mixed_edge"]["count"] >= 1 and rep["flyers"] >= 4


def test_depth_maps_keep_the_nearest_point_per_pixel_and_steps_are_relative():
    cams, frames, xyz, fr, row, col, _, _ = _scene()
    w2c = {f: np.linalg.inv(cams[f]) for f in frames}
    xyz2 = np.concatenate([xyz, xyz[:1] * 0.5 + cams[0][:3, 3] * 0.5]); fr2 = np.r_[fr, fr[:1]]
    row2 = np.r_[row, row[:1]]; col2 = np.r_[col, col[:1]]
    zm, im, z_own = FL.depth_maps_from_cloud(xyz2, fr2, row2, col2, frames, w2c, H, W)
    assert np.isclose(z_own[-1], z_own[0] * 0.5)
    assert im[0][row[0], col[0]] == len(xyz)                    # the nearer duplicate wins the pixel
    assert np.isclose(zm[0][row[0], col[0]], _true_depth(cams[0])[row[0], col[0]] * 0.5)
    zf, zb, sp = FL.window_spread(zm[0], 2)
    v = zm[0] > 0
    assert np.isclose(sp[v].max(), 1.0, atol=0.02) and np.median(sp[v]) == 0.0   # plate/wall (3-1.5)/1.5; flats 0


def test_classification_precedence_and_layer_file(tmp_path):
    is_f = np.array([True, True, True, True, False])
    mixed = np.array([True, False, False, False, True])
    agree = np.array([0, 0, 2, 0, 0]); contra = np.array([1, 1, 1, 0, 3])
    low = np.array([True, False, True, False, True])
    cls = FL.classify(is_f, mixed, agree, contra, low)
    assert cls.tolist() == [1, 2, 3, 4, 0]                     # a beats b beats c; a non-flyer stays 0
    p = FL.write_layer_glb(tmp_path / "flyers.glb", np.random.default_rng(0).random((5, 3)), cls)
    import trimesh
    g = trimesh.load(str(p), force="scene")
    n = sum(len(x.vertices) for x in g.geometry.values())
    assert n == 4
    FL.write_csv(tmp_path / "f.csv", "s", FL.counts(cls))
    assert "mixed_edge" in (tmp_path / "f.csv").read_text()


def test_production_config_declares_every_bar():
    f = load_precision_config().flyers
    assert 0 < f.isolated_quantile <= 100 and 0 < f.step_quantile <= 100 and f.knn_k >= 1
