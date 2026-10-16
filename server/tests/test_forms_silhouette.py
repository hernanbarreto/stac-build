"""FORMS P3 (USER 2026-10-16): the corrector — the form against the masks of every view, the
consensus the objective, convergence and verdict by the views' own noise. Synthetic scene on
the CPU: a true cylinder, masks rendered from it, a cloud that is DEFORMED (thinner, shorter)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms.catalog import FORMS, parse_answer  # noqa: E402
from reconstruction.forms.fit import fit_cloud  # noqa: E402
from reconstruction.forms.silhouette import (  # noqa: E402
    Silhouettes, boundary_pixels, compare_forms, correct, per_view_chamfer, pinhole_view, worst_views)
from reconstruction.forms.templates import Extrusion, Frame, Pose, Section, build_template  # noqa: E402

DEV = "cpu"
CAT = list(FORMS)
UP = np.array([0.0, 1.0, 0.0])
FRAME = Frame(UP, np.array([1.0, 0.0, 0.0]))
H, W, F = 720, 960, 900.0      # a camera like the scans' (a pixel ~3 mm at 3 m, under the cloud's σ)
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
SIGMA = 0.005
FIT = dict(lr=0.02, block_iterations=20, max_blocks=30, huber_k=1.345, max_points=20000, seed=0, dev=DEV)


def _ans(form, orientation="vertical"):
    return parse_answer({"form": form, "orientation": orientation, "axis_aligned": True, "hollow": False,
                         "outline": "unknown", "profile": "unknown", "parts": [], "proportions": "",
                         "alternatives": [], "confidence": "high", "what_would_help": "", "notes": ""}, CAT)


def _cylinder(r, half, c=(0.0, 1.0, 0.0)):
    return Extrusion("cylinder", Section("circle", {"r": r}, DEV),
                     Pose(FRAME, "fixed", "snapped", DEV, np.asarray(c, float), UP), half, False, DEV)


def _cameras(n=6, radius=3.0, height=1.0, target=(0.0, 1.0, 0.0)):
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pos = np.array([radius * math.cos(a), height, radius * math.sin(a)])
        z = np.asarray(target) - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, UP); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
        out.append(c2w)
    return out


def _project(c2w, X):
    w2c = np.linalg.inv(c2w)
    pc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    u = F * pc[:, 0] / z + W / 2
    v = F * pc[:, 1] / z + H / 2
    return u, v, z


def _mask_of(c2w, truth_pts):
    from precision.f6_torch import dilate, erode3
    u, v, z = _project(c2w, truth_pts)
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = torch.zeros((H, W), dtype=torch.bool)
    m[np.rint(v[ok]).astype(int), np.rint(u[ok]).astype(int)] = True
    m = erode3(dilate(m, 1))
    return m.numpy()


def _zbuf_of(c2w, cloud_pts):
    from precision.f6_torch import min_filter
    from reconstruction.surface_fit.hole_audit import zbuffer_min
    u, v, z = _project(c2w, cloud_pts)
    ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix = torch.as_tensor(np.rint(v[ok]).astype(np.int64) * W + np.rint(u[ok]).astype(np.int64))
    zb = zbuffer_min(pix, torch.as_tensor(z[ok], dtype=torch.float64), H, W)
    return min_filter(zb, 5).numpy().astype(np.float32)


def _dense_cylinder(r, half, c=(0.0, 1.0, 0.0), step=0.0015):
    """The true surface sampled under half a pixel (the silhouette raster must be FILLED —
    the template sampler's rings are pixels apart and would stripe the mask)."""
    na = max(16, int(round(2 * math.pi * r / step)))
    nz = max(2, int(round(2 * half / step)))
    ang = np.linspace(0, 2 * math.pi, na, endpoint=False)
    zz = np.linspace(-half, half, nz)
    A, Z = np.meshgrid(ang, zz, indexing="ij")
    lat = np.column_stack([r * np.cos(A).ravel(), Z.ravel(), r * np.sin(A).ravel()])
    rr = np.linspace(0, r, max(2, int(round(r / step))))
    Rr, Ar = np.meshgrid(rr, ang, indexing="ij")
    disc = np.column_stack([Rr.ravel() * np.cos(Ar).ravel(), np.zeros(Rr.size), Rr.ravel() * np.sin(Ar).ravel()])
    caps = np.vstack([disc + [0, half, 0], disc + [0, -half, 0]])
    return np.vstack([lat, caps]) + np.asarray(c, float)


def _scene(r_true=0.10, half_true=1.0, r_cloud=0.095, half_cloud=0.8, seed=0):
    truth = _cylinder(r_true, half_true)
    T = _dense_cylinder(r_true, half_true)
    with torch.no_grad():
        C = _cylinder(r_cloud, half_cloud).samples(12000).numpy()
    rng = np.random.default_rng(seed)
    cloud = C + rng.normal(0, SIGMA, C.shape)
    views = []
    for i, c2w in enumerate(_cameras()):
        views.append(pinhole_view(i, c2w, K, _mask_of(c2w, T), _zbuf_of(c2w, cloud), SIGMA, 256, 0, DEV))
    return truth, cloud, views


def test_boundary_pixels_drop_occlusion_edges():
    truth, cloud, views = _scene()
    v = views[0]
    assert v.boundary.shape[0] > 50 and v.area > 500 and torch.isfinite(v.z_boundary).all()
    # an occluder right outside the mask: a closer depth ring around the mask's left edge
    m = v.mask.clone()
    zb = v.zbuf.clone()
    rows, cols = torch.nonzero(m, as_tuple=True)
    left = int(cols.min())
    zb[:, max(0, left - 3):left] = 1.0           # something at 1 m, in front of the object at ~3 m
    b1, _ = boundary_pixels(m, v.zbuf, SIGMA, F, 10 ** 6, 0)   # every usable boundary pixel
    b2, _ = boundary_pixels(m, zb, SIGMA, F, 10 ** 6, 0)
    # the left edge's pixels left the boundary
    assert int((b2[:, 0] <= left).sum()) == 0 and b2.shape[0] < b1.shape[0]
    assert int((b1[:, 0] <= left).sum()) > 0                   # they were there before the occluder


def test_the_masks_correct_a_deformed_cloud_length_and_radius():
    truth, cloud, views = _scene()
    t = build_template(_ans("cylinder"), cloud, FRAME, DEV)
    fit_cloud(t, cloud, SIGMA, **FIT)
    d0 = t.dims()
    assert d0["length_m"] == pytest.approx(1.6, abs=0.05) and d0["radius_m"] == pytest.approx(0.095, abs=0.004)
    sil = Silhouettes(views, SIGMA, 1.345, DEV)
    rec = correct(t, cloud, SIGMA, sil, lr=0.02, block_iterations=20, max_blocks=30, huber_k=1.345,
                  max_points=20000, surface_samples=8192, seed=0, dev=DEV, confidence=0.95)
    d1 = t.dims()
    # the images see where the object ends: the length the cloud did not have comes back
    assert d1["length_m"] == pytest.approx(2.0, abs=0.06), d1
    # the radius follows the silhouettes (the image's resolution outweighs the thinner cloud)
    assert d1["radius_m"] == pytest.approx(0.10, abs=0.003), d1
    assert rec["verdict"] == "coincides" and rec["n_views_judging"] == 6
    assert rec["consensus_final_m"] < rec["consensus_start_m"]
    assert rec["stop"] in ("judge", "max_blocks") and rec["blocks"] >= 2
    assert rec["iou_median"] > 0.6 and len(rec["per_view"]) == 6
    assert np.allclose(t.pose.axis().detach().numpy(), UP)          # "vertical" stays exact


def test_a_wrong_form_loses_to_the_right_alternative_on_the_views():
    truth, cloud, views = _scene()
    sil = Silhouettes(views, SIGMA, 1.345, DEV)
    recs = {}
    for form in ("box", "cylinder"):
        t = build_template(_ans(form), cloud, FRAME, DEV)
        fit_cloud(t, cloud, SIGMA, **FIT)
        recs[form] = correct(t, cloud, SIGMA, sil, lr=0.02, block_iterations=20, max_blocks=30,
                             huber_k=1.345, max_points=20000, surface_samples=8192, seed=0, dev=DEV,
                             confidence=0.95)
    # a box fitted to a round column: against a DEFORMED cloud (2 cm off the masks) it may still
    # coincide — the bar is the cloud's own agreement, declared — but against the right form,
    # judged on the same views, it clearly loses: the alternatives decide by consensus
    c = compare_forms(per_view_chamfer(recs["cylinder"]), per_view_chamfer(recs["box"]), 0.95, 0)
    assert c["a_better"] and not c["tie"]
    assert recs["box"]["consensus_final_m"] > 1.5 * recs["cylinder"]["consensus_final_m"]
    assert len(worst_views(recs["box"], 3)) == 3


def test_the_corrector_is_deterministic():
    outs = []
    for _ in range(2):
        truth, cloud, views = _scene()
        t = build_template(_ans("cylinder"), cloud, FRAME, DEV)
        fit_cloud(t, cloud, SIGMA, **FIT)
        sil = Silhouettes(views, SIGMA, 1.345, DEV)
        rec = correct(t, cloud, SIGMA, sil, lr=0.02, block_iterations=10, max_blocks=6, huber_k=1.345,
                      max_points=20000, surface_samples=2048, seed=0, dev=DEV, confidence=0.95)
        outs.append((rec["consensus_m"], t.state()))
    assert outs[0][0] == outs[1][0]
    assert all(np.array_equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))
