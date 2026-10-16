"""FORMS P2 (USER 2026-10-16): the cloud measures pose + dimensions of the VLM's form — the
parametric templates and the robust fit, on synthetic clouds with a known answer. CPU."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms.catalog import parse_answer, FORMS  # noqa: E402
from reconstruction.forms.fit import fit_cloud, residual_report, subsample  # noqa: E402
from reconstruction.forms.templates import (  # noqa: E402
    Extrusion, Frame, Pose, Revolution, Section, Sphere, build_template)

DEV = "cpu"
CAT = list(FORMS)
FRAME = Frame(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))     # Y-up, X along x
SIGMA = 0.01
FIT = dict(lr=0.02, block_iterations=25, max_blocks=40, huber_k=1.345, max_points=50000, seed=0, dev=DEV)


def _ans(form, orientation="vertical", **kw):
    d = {"form": form, "orientation": orientation, "axis_aligned": kw.pop("axis_aligned", True),
         "hollow": kw.pop("hollow", None), "outline": kw.pop("outline", "unknown"),
         "profile": kw.pop("profile", "unknown"), "parts": [], "proportions": "", "alternatives": [],
         "confidence": "high", "what_would_help": "", "notes": ""}
    d.update(kw)
    return parse_answer(d, CAT)


def _cloud(template, n, seed, noise=SIGMA, shift=np.zeros(3)):
    """Noisy points on a TRUTH template's surface (+ an offset the init must find)."""
    with torch.no_grad():
        S = template.samples(n).cpu().numpy()
    rng = np.random.default_rng(seed)
    return S + rng.normal(0, noise, S.shape) + shift


def _pose(axis_mode, roll_mode, c, axis, roll=0.0, psi=0.0):
    return Pose(FRAME, axis_mode, roll_mode, DEV, np.asarray(c, float), np.asarray(axis, float), roll, psi)


# ── the templates themselves ───────────────────────────────────────────────────

def test_samples_lie_on_the_surface_and_meshes_are_valid():
    cyl = Extrusion("cylinder", Section("circle", {"r": 0.15}, DEV),
                    _pose("fixed", "snapped", [1, 2, 3], [0, 1, 0]), 1.2, False, DEV)
    box = Extrusion("box", Section("rect", {"bx": 0.3, "by": 0.2}, DEV),
                    _pose("free", "free", [0, 0, 0], [0.3, 0.9, 0.1], 0.4), 0.5, False, DEV)
    ibeam = Extrusion("profile", Section("I", {"b": 0.2, "h": 0.4, "tf": 0.02, "tw": 0.012}, DEV),
                      _pose("horizontal", "snapped", [0, 3, 0], [1, 0, 0]), 3.0, False, DEV)
    panel = Extrusion("panel", Section("arch", {"b": 1.0, "H": 2.2}, DEV),
                      _pose("horizontal", "snapped", [0, 1, 0], [0, 0, 1]), None, True, DEV)
    cone = Revolution("cone", _pose("fixed", "snapped", [0, 0, 0], [0, 1, 0]), [0.4, 0.1], 0.6, DEV)
    sph = Sphere(np.array([1.0, 1.0, 1.0]), 0.3, DEV)
    for t in (cyl, box, ibeam, panel, cone, sph):
        S = t.samples(3000)
        assert S.shape[1] == 3 and len(S) > 500
        d = t.distance(S).detach().cpu().numpy()
        assert d.max() < 2e-3, (t.form, d.max())
        V, F = t.mesh()
        assert len(V) > 3 and len(F) > 1 and F.max() < len(V)
        # the mesh vertices are on the surface too
        dv = t.distance(torch.as_tensor(V, dtype=torch.float64)).detach().cpu().numpy()
        assert np.percentile(dv, 95) < 1e-2, (t.form, np.percentile(dv, 95))
    # an I-beam's dims and a hollow section's wall
    assert ibeam.dims()["flange_width_m"] == pytest.approx(0.2) and ibeam.dims()["length_m"] == pytest.approx(6.0)
    tube = Extrusion("cylinder", Section("ring", {"r": 0.1, "t": 0.02}, DEV),
                     _pose("fixed", "snapped", [0, 0, 0], [0, 1, 0]), 1.0, True, DEV)
    assert tube.dims()["wall_m"] == pytest.approx(0.02)
    P = torch.tensor([[0.0, 0.0, 0.09], [0.0, 0.0, 0.1], [0.0, 1.5, 0.1]], dtype=torch.float64)
    d = tube.distance(P).detach().numpy()
    assert d[0] == pytest.approx(0.01, abs=1e-5) and d[1] < 1e-5 and d[2] == pytest.approx(0.5, abs=1e-5)


def test_the_vertical_word_fixes_the_axis_exactly():
    ans = _ans("cylinder", "vertical")
    truth = Extrusion("cylinder", Section("circle", {"r": 0.12}, DEV),
                      _pose("fixed", "snapped", [2.0, 1.2, -1.0], [0, 1, 0]), 1.2, False, DEV)
    P = _cloud(truth, 6000, 1)
    t = build_template(ans, P, FRAME, DEV)
    assert t.pose.axis_mode == "fixed" and np.allclose(t.pose.axis().numpy(), [0, 1, 0])
    rec = fit_cloud(t, P, SIGMA, **FIT)
    assert rec["stop"] in ("converged", "max_blocks") and rec["blocks"] >= 2
    d = t.dims()
    assert d["radius_m"] == pytest.approx(0.12, abs=0.004)
    assert d["length_m"] == pytest.approx(2.4, abs=0.05)
    assert np.allclose(t.pose.c.detach().numpy(), [2.0, 1.2, -1.0], atol=0.02)
    assert np.allclose(t.pose.axis().numpy(), [0, 1, 0])          # still exact after the fit
    rep = residual_report(t, P, SIGMA, DEV)
    # the cap points sit ~2σ inside the percentile length (the declared bias of the cloud-only
    # extents); the lateral surface is at the noise
    assert rep["within_2sigma"] > 0.85 and rep["within_3sigma"] > 0.97 and rep["median_m"] < 1.5 * SIGMA


def test_an_inclined_cylinder_keeps_its_measured_axis():
    ans = _ans("cylinder", "inclined", axis_aligned=False)
    a = np.array([0.5, 0.8, 0.33]); a /= np.linalg.norm(a)
    truth = Extrusion("cylinder", Section("circle", {"r": 0.08}, DEV),
                      _pose("free", "snapped", [0.0, 0.5, 0.0], a), 1.5, False, DEV)
    P = _cloud(truth, 6000, 2)
    t = build_template(ans, P, FRAME, DEV)
    assert t.pose.axis_mode == "free"
    fit_cloud(t, P, SIGMA, **FIT)
    got = t.pose.axis().detach().numpy()
    ang = math.degrees(math.acos(min(1.0, abs(float(got @ a)))))
    assert ang < 1.0 and t.dims()["radius_m"] == pytest.approx(0.08, abs=0.004)


def test_a_box_and_a_panel_recover_their_extents():
    ans = _ans("box", "vertical", axis_aligned=True)
    truth = Extrusion("box", Section("rect", {"bx": 0.4, "by": 0.25}, DEV),
                      _pose("fixed", "snapped", [1.0, 0.45, 2.0], [0, 1, 0]), 0.45, False, DEV)
    P = _cloud(truth, 8000, 3)
    t = build_template(ans, P, FRAME, DEV)
    fit_cloud(t, P, SIGMA, **FIT)
    d = t.dims()
    # the extents of the cloud-only fit are robust percentiles of the points: the noise tail of a
    # face biases them by up to ~2σ (declared); the silhouette corrector refines them (P3)
    assert d["width_m"] == pytest.approx(0.8, abs=0.05) and d["height_m"] == pytest.approx(0.5, abs=0.05)
    assert d["length_m"] == pytest.approx(0.9, abs=0.05)
    # a vertical rectangular panel (a wall-like plate): its normal horizontal, exactly in the plane
    pans = _ans("panel", "vertical", outline="rectangle", axis_aligned=True)
    wall = Extrusion("panel", Section("rect", {"bx": 1.5, "by": 1.2}, DEV),
                     _pose("horizontal", "snapped", [0, 1.2, 3.0], [0, 0, 1], psi=math.pi / 2), None, True, DEV)
    P2 = _cloud(wall, 6000, 4)
    w = build_template(pans, P2, FRAME, DEV)
    assert w.flat and w.pose.axis_mode == "horizontal"
    fit_cloud(w, P2, SIGMA, **FIT)
    assert abs(float(w.pose.axis().detach().numpy() @ FRAME.up)) < 1e-9        # stays horizontal
    assert w.dims()["width_m"] == pytest.approx(3.0, abs=0.06) and w.dims()["height_m"] == pytest.approx(2.4, abs=0.06)
    assert abs(float(w.pose.axis().detach().numpy()[2])) > 0.999                  # snapped to the room's axis


def test_an_i_profile_a_cone_a_revolution_and_a_sphere():
    ans = _ans("profile", "horizontal", profile="I", axis_aligned=True)
    truth = Extrusion("profile", Section("I", {"b": 0.2, "h": 0.4, "tf": 0.02, "tw": 0.012}, DEV),
                      _pose("horizontal", "snapped", [0, 3.0, 0], [1, 0, 0]), 2.0, False, DEV)
    P = _cloud(truth, 12000, 5, noise=0.004)
    t = build_template(ans, P, FRAME, DEV)
    fit_cloud(t, P, 0.004, **FIT)
    d = t.dims()
    assert d["section"] == "I" and d["flange_width_m"] == pytest.approx(0.2, abs=0.02)
    assert d["depth_m"] == pytest.approx(0.4, abs=0.02) and d["length_m"] == pytest.approx(4.0, abs=0.06)
    # a cone standing up
    cans = _ans("cone", "vertical")
    ct = Revolution("cone", _pose("fixed", "snapped", [0, 0.6, 0], [0, 1, 0]), [0.4, 0.1], 0.6, DEV)
    P = _cloud(ct, 8000, 6)
    c = build_template(cans, P, FRAME, DEV)
    fit_cloud(c, P, SIGMA, **FIT)
    assert c.dims()["radius_base_m"] == pytest.approx(0.4, abs=0.02)
    assert c.dims()["radius_top_m"] == pytest.approx(0.1, abs=0.02)
    assert c.dims()["length_m"] == pytest.approx(1.2, abs=0.05)
    # a surface of revolution (a bollard: wide base, narrow waist, wider head)
    rans = _ans("revolution", "vertical")
    rt = Revolution("revolution", _pose("fixed", "snapped", [0, 0.5, 0], [0, 1, 0]),
                    [0.25, 0.15, 0.1, 0.1, 0.12, 0.18, 0.2, 0.16], 0.5, DEV)
    P = _cloud(rt, 12000, 7)
    r = build_template(rans, P, FRAME, DEV, revolution_stations=8)
    fit_cloud(r, P, SIGMA, **FIT)
    got = r.radii().detach().numpy()
    assert np.allclose(got, [0.25, 0.15, 0.1, 0.1, 0.12, 0.18, 0.2, 0.16], atol=0.02)
    # a sphere
    st = Sphere(np.array([0.5, 0.5, 0.5]), 0.3, DEV)
    P = _cloud(st, 4000, 8)
    s = build_template(_ans("sphere"), P, FRAME, DEV)
    fit_cloud(s, P, SIGMA, **FIT)
    assert s.dims()["radius_m"] == pytest.approx(0.3, abs=0.005)
    assert np.allclose(s.c.detach().numpy(), [0.5, 0.5, 0.5], atol=0.01)


def test_the_fit_is_deterministic_and_subsampled_by_seed():
    ans = _ans("cylinder", "vertical")
    truth = Extrusion("cylinder", Section("circle", {"r": 0.1}, DEV),
                      _pose("fixed", "snapped", [0, 1, 0], [0, 1, 0]), 1.0, False, DEV)
    P = _cloud(truth, 9000, 9)
    outs = []
    for _ in range(2):
        t = build_template(ans, P, FRAME, DEV)
        rec = fit_cloud(t, P, SIGMA, **{**FIT, "max_points": 3000})
        outs.append((t.state(), rec["loss"], rec["n_points_fit"]))
    assert outs[0][2] == 3000 and outs[0][1] == outs[1][1]
    assert all(np.array_equal(a, b) for a, b in zip(outs[0][0], outs[1][0]))
    assert np.array_equal(subsample(P, 100, 0), subsample(P, 100, 0))
    assert not np.array_equal(subsample(P, 100, 0), subsample(P, 100, 1))


def test_composite_and_freeform_have_no_single_template():
    with pytest.raises(ValueError):
        build_template(_ans("freeform"), np.random.default_rng(0).normal(size=(100, 3)), FRAME, DEV)
