"""Edge-definition metric (precision/edge_metric.py) — the crease profile of the
2026-10-01 edge audit, §3, on synthetic boxes with KNOWN edges.

USER 2026-10-01: "lo más importante es que los objetos deben tener mucha
definición, corte en los filos, las aristas". The metric must read a sharp
box as sharp (r̂ ≈ 0, below its own resolution floor r_res), recover a known
fillet radius above r_res, lose coverage when the edge band is eroded (the
confidence floor's signature) and gain overshoot when a mixed-pixel skirt
hangs off the edges — and the A-vs-B verdict (heldout_change at the declared
confidence) must say so. No GPU; three cloud detections in total.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import edge_metric as em                                # noqa: E402
from tests import synth_edges as se                                     # noqa: E402

FILLET_R = 0.025


def _raw_cfg():
    """The keys the metric reads, at the values config.yaml declares."""
    return {"surface_fit": {"p2c_inlier_dist_m": 0.02, "p2c_detect_max_pts": 150000,
                            "p2c_max_primitives": 24, "p2c_freeform_min_pts": 2000},
            "correction_graph": {"graph": {"heldout_confidence": 0.95}}}


@pytest.fixture(scope="module")
def cfg():
    return em.load_edge_config(_raw_cfg())


def _quiet(*_a, **_k):
    pass


# ── config ────────────────────────────────────────────────────────────────

def test_config_is_strict_and_repo_config_loads():
    with pytest.raises(em.EdgeMetricConfigError, match="surface_fit"):
        em.load_edge_config({"correction_graph": {"graph": {"heldout_confidence": 0.95}}})
    raw = _raw_cfg()
    del raw["correction_graph"]["graph"]["heldout_confidence"]
    with pytest.raises(em.EdgeMetricConfigError, match="heldout_confidence"):
        em.load_edge_config(raw)
    raw = _raw_cfg()
    del raw["surface_fit"]["p2c_inlier_dist_m"]
    with pytest.raises(em.EdgeMetricConfigError, match="p2c_inlier_dist_m"):
        em.load_edge_config(raw)
    c = em.load_edge_config(em._read_config_yaml())                # the live config.yaml
    assert c.confidence == pytest.approx(0.95) and c.inlier_dist_m > 0
    assert em.z_of(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert em.bootstrap_count() == 2000                             # heldout_change's own


# ── geometry of the filleted L ────────────────────────────────────────────

@pytest.mark.parametrize("phi_deg", [60.0, 90.0, 120.0])
def test_profile_distances(phi_deg):
    phi, r, delta = np.radians(phi_deg), 0.02, 0.004
    T = r / np.tan(phi / 2)
    c = np.array([T, r])                                            # fillet centre
    mid = np.array([np.cos(phi / 2), np.sin(phi / 2)])              # bisector
    q = np.array([
        [0.10, 0.0],                                                # on arm A, past T
        [0.10 * np.cos(phi), 0.10 * np.sin(phi)],                   # on arm B, past T
        c - r * mid,                                                # arc midpoint
        [0.0, 0.0],                                                 # the sharp corner
        [0.10, -0.003],                                             # outside, off arm A
        -0.01 * mid,                                                # beyond the corner
    ])
    p = em.profile(q, r, phi, delta)
    corner = r * (1.0 / np.sin(phi / 2) - 1.0)                      # what r_res inverts
    np.testing.assert_allclose(p.dist[:3], 0.0, atol=1e-12)
    assert p.dist[3] == pytest.approx(corner, rel=1e-9)
    assert p.signed[3] > 0                                          # cut off by the fillet
    assert p.dist[4] == pytest.approx(0.003) and p.signed[4] > 0
    assert p.signed[5] > 0 and p.in_band[5]
    assert p.in_band[2] and p.in_band[3] and not p.in_band[0] and not p.in_band[1]
    sharp = em.profile(q, 0.0, phi, delta)
    assert sharp.dist[3] == 0.0 and sharp.in_band[3]


def _section(r, phi=np.pi / 2, h=0.002, arm=0.15, noise=0.0008, erode=0.0, rows=20, seed=0):
    """2-D cross-section of an L (optionally filleted) with noise: ``rows``
    noisy copies of the profile, as the slices along a crease stack up."""
    rng = np.random.default_rng(seed)
    T = r / np.tan(phi / 2) if r > 0 else 0.0
    ua, ub = np.array([1.0, 0.0]), np.array([np.cos(phi), np.sin(phi)])
    t = np.arange(T, arm, h)
    pts = [np.outer(t, ua), np.outer(t, ub)]
    if r > 0:
        # from the centre (T, r): towards arm A's tangent at -90°, sweeping the
        # (π − φ) of arc that faces the corner, to arm B's tangent
        c = np.array([T, r])
        ang = -0.5 * np.pi - np.linspace(0.0, np.pi - phi, max(3, int(r * (np.pi - phi) / h)))
        pts.append(c + r * np.column_stack([np.cos(ang), np.sin(ang)]))
    Q = np.tile(np.concatenate(pts), (rows, 1))
    Q = Q + noise * rng.standard_normal(Q.shape)
    if erode > 0:
        Q = Q[np.hypot(Q[:, 0], Q[:, 1]) > erode]
    return Q


def test_fillet_radius_on_a_section():
    delta, z = 0.002, em.z_of(0.95)
    r_max = 0.15
    assert em.fillet_radius(_section(0.0), np.pi / 2, delta, r_max, z) == 0.0
    r_hat = em.fillet_radius(_section(FILLET_R), np.pi / 2, delta, r_max, z)
    assert abs(r_hat - FILLET_R) <= delta
    # an eroded corner leaves the cost flat up to the gap: within the sample's
    # own noise the data support no rounding, so r̂ must not read erosion as one
    assert em.fillet_radius(_section(0.0, erode=0.012), np.pi / 2, delta, r_max, z) == 0.0


# ── label transfer, PLY loader, detector adjacency ────────────────────────

def test_carry_labels_by_key():
    ref_keys = np.array([10, 11, 12, 12, 13, 14])
    ref_lab = np.array([0, 0, 1, 2, 1, -1])                         # key 12: two labels
    other = np.array([14, 13, 99, 12, 10, 10])
    np.testing.assert_array_equal(em.carry_labels(ref_keys, ref_lab, other),
                                  [-1, 1, -1, -1, 0, 0])
    assert len(em.carry_labels(np.array([], np.int64), np.array([], np.int64), other)) == 6


def test_loader_ply_path_and_indices_keep_default_behaviour(tmp_path):
    from segmentation.perfect_object import _load_instance_cloud, _read_ply_fields
    rng = np.random.default_rng(0)
    P = rng.normal(size=(500, 3))
    se.write_ply(tmp_path / "cleaned_cloud.ply", P, np.arange(500))
    f = _read_ply_fields(tmp_path / "cleaned_cloud.ply")
    f["confidence"][:] = np.linspace(0.0, 1.0, 500, dtype=np.float32)
    inst = {"globalIndices": list(range(0, 500, 2))}
    old = _load_instance_cloud(tmp_path, inst, {}, "x", _quiet, fields=f)   # 20 % trim default
    P2, gi = _load_instance_cloud(tmp_path, inst, {}, "x", _quiet, fields=f,
                                  return_indices=True)
    np.testing.assert_array_equal(old, P2)
    assert len(old) == 200 and gi.min() >= 100                      # the lowest 20 % left
    other = tmp_path / "epoch.ply"
    se.write_ply(other, P + 1.0)
    P3, gi3 = _load_instance_cloud(tmp_path, inst, {"p2c_conf_trim_pct": 0.0}, "x", _quiet,
                                   ply_path=other, return_indices=True)
    assert len(P3) == 250
    np.testing.assert_allclose(P3, (P + 1.0)[gi3], atol=1e-5)


def test_region_adjacency_counts_contact_edges():
    from segmentation.perfect_object import region_adjacency
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 0]])
    lab = np.array([0, 1, 1, -1, 2])
    assert region_adjacency(E, lab) == {(0, 1): 2, (0, 2): 1}


# ── S4: T-junction and coplanar fragments (labels given, no detector) ─────

def _plane_patch(origin, u, v, nu, nv, h, rng, noise, n):
    gu, gv = np.meshgrid(np.arange(nu) * h, np.arange(nv) * h, indexing="ij")
    P = origin + np.outer(gu.ravel(), u) + np.outer(gv.ravel(), v)
    return P + noise * rng.standard_normal(len(P))[:, None] * n


def test_t_junction_splits_and_coplanar_fragments_skip(cfg):
    rng = np.random.default_rng(3)
    h, noise = 0.005, 0.0008
    ex, ey, ez = np.eye(3)
    wall = _plane_patch(np.array([0.0, -0.2, -0.15]), ey, ez, 81, 61, h, rng, noise, ex)
    shelf = _plane_patch(np.array([h, 0.0, -0.15]), ex, ez, 40, 61, h, rng, noise, ey)
    P = np.concatenate([wall, shelf])
    lab = np.r_[np.zeros(len(wall), np.int64), np.ones(len(shelf), np.int64)]
    z = em.z_of(cfg.confidence)
    creases, skipped = em.define_creases(P, lab, [(0, 1)], cfg, z, em.bootstrap_count(), _quiet)
    assert not skipped and len(creases) == 2 and all(c.split_a and not c.split_b for c in creases)
    res = em.measure_object(P, lab, creases, em.resolution(P), cfg, z)
    for c in res["creases"]:
        assert c["measured"] and abs(c["phi_deg"] - 90.0) < 1.0
        assert c["r_hat_m"] <= c["r_res_m"] and c["sharp_within_instrument"]
    # one plane cut in two labels: a majority of each lies in the other's band
    flat = _plane_patch(np.zeros(3), ex, ez, 80, 60, h, rng, noise, ey)
    lab2 = (flat[:, 0] > 0.2).astype(np.int64)
    creases2, skipped2 = em.define_creases(flat, lab2, [(0, 1)], cfg, z, em.bootstrap_count(),
                                           _quiet)
    assert creases2 == [] and len(skipped2) == 1


# ── the box: detector + full pipeline through run() ───────────────────────

@pytest.fixture(scope="module")
def boxes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("edge_boxes")
    sharp = se.box(r=0.0)
    keys = np.arange(len(sharp))
    fillet = se.box(r=FILLET_R)
    sk = se.skirt()
    keep = se.eroded_keep(sharp)
    return {
        "dir": tmp,
        "sharp": se.scene(tmp, "sharp", sharp, keys),
        "fillet": se.scene(tmp, "fillet", fillet, np.arange(len(fillet))),
        "skirt": se.scene(tmp, "skirt", np.concatenate([sharp, sk]),
                          np.concatenate([keys, 10 ** 6 + np.arange(len(sk))])),
        "eroded": se.scene(tmp, "eroded", sharp[keep], keys[keep]),
    }


@pytest.fixture(scope="module")
def fillet_report(boxes, cfg):
    ply, seg = boxes["fillet"]
    return em.run(boxes["dir"], ply, seg, out_path=boxes["dir"] / "fillet.json", cfg=cfg,
                  log=_quiet)


@pytest.fixture(scope="module")
def skirt_report(boxes, cfg):
    return em.run(boxes["dir"], *boxes["sharp"], compare=boxes["skirt"], cfg=cfg, log=_quiet)


@pytest.fixture(scope="module")
def eroded_report(boxes, cfg):
    return em.run(boxes["dir"], *boxes["sharp"], compare=boxes["eroded"], cfg=cfg, log=_quiet)


def _measured(obj):
    return [c for c in obj["creases"] if c["measured"]]


def test_sharp_box_reads_sharp_within_the_instrument(skirt_report):
    ob = skirt_report["reference"]["objects"][0]
    cr = _measured(ob)
    assert len(cr) >= 8                                             # a box has 12 edges
    for c in cr:
        assert abs(c["phi_deg"] - 90.0) < 1.0
        assert c["r_hat_m"] <= c["r_res_m"], c                       # r̂ ~ 0: sharp
        assert c["coverage"] > 0.5 and c["overshoot"] < 0.15
    s = ob["summary"]
    assert s["r_hat_m"] == 0.0 and s["tau_over_sigma"] < 1.5


def test_filleted_box_recovers_the_radius_above_r_res(fillet_report):
    ob = fillet_report["reference"]["objects"][0]
    cr = _measured(ob)
    assert len(cr) >= 8
    for c in cr:
        assert c["r_res_m"] < c["r_hat_m"], c
        assert abs(c["r_hat_m"] - FILLET_R) <= c["delta_m"], c       # grid step = Δ
        assert c["coverage"] > 0.9
    assert abs(ob["summary"]["r_hat_m"] - FILLET_R) <= ob["creases"][0]["delta_m"]
    rep = json.loads((Path(fillet_report["reference"]["ply"]).parent / "fillet.json").read_text())
    assert rep["provenance"] == "tool_measured" and rep["reference"]["summary"]["n_creases"] >= 8


def test_skirt_raises_overshoot_and_the_verdict_says_worse(skirt_report):
    a = skirt_report["reference"]["objects"][0]
    b = skirt_report["compare"]["objects"][0]
    assert b["paired_with"] == a["instance_id"]
    pairs = [(x, y) for x, y in zip(a["creases"], b["creases"]) if x["measured"] and y["measured"]]
    assert len(pairs) >= 8
    assert all(y["overshoot"] > x["overshoot"] for x, y in pairs)
    assert b["summary"]["overshoot"] > a["summary"]["overshoot"] + 0.2
    assert all(y["r_hat_m"] <= y["r_res_m"] for _, y in pairs)       # still sharp
    v = skirt_report["verdict"]
    assert v["pairing"] == "crease"
    assert v["metrics"]["overshoot"]["verdict"] == "worsens"
    assert v["metrics"]["tau_over_sigma"]["verdict"] == "worsens"


def test_eroded_edge_band_drops_coverage(eroded_report, skirt_report):
    a = eroded_report["reference"]["objects"][0]
    b = eroded_report["compare"]["objects"][0]
    pairs = [(x, y) for x, y in zip(a["creases"], b["creases"]) if x["measured"] and y["measured"]]
    assert len(pairs) >= 8
    assert all(y["coverage"] < x["coverage"] for x, y in pairs)
    assert b["summary"]["coverage"] < 0.2 < a["summary"]["coverage"]
    assert all(y["r_hat_m"] == 0.0 for _, y in pairs)                # erosion is not a fillet
    v = eroded_report["verdict"]
    assert v["metrics"]["one_minus_coverage"]["verdict"] == "worsens"
    # faces were detected ONCE per reference and the detection is seeded: the
    # two runs over the same reference found the same faces and creases
    ref2 = skirt_report["reference"]["objects"][0]
    assert [c["faces"] for c in a["creases"]] == [c["faces"] for c in ref2["creases"]]
    assert a["skipped_pairs"] == ref2["skipped_pairs"]


def test_without_provenance_keys_objects_pair_by_instance_id(boxes, cfg, tmp_path):
    """No (frame_global, pixel_row, pixel_col) in the clouds: labels cannot be
    carried point by point, so each epoch is detected on its own and the
    verdict pairs OBJECTS by instance_id (audit §3)."""
    sharp = se.scene(tmp_path, "sharp_nokey", se.box(r=0.0), None)
    fillet = se.scene(tmp_path, "fillet_nokey", se.box(r=FILLET_R), None)
    rep = em.run(tmp_path, *sharp, compare=fillet, cfg=cfg, log=_quiet)
    v = rep["verdict"]
    assert v["pairing"] == "object" and v["n_pairs"] == 1
    a = rep["reference"]["objects"][0]["summary"]
    b = rep["compare"]["objects"][0]["summary"]
    assert a["r_hat_m"] <= a["r_res_m"]
    assert abs(b["r_hat_m"] - FILLET_R) <= rep["compare"]["objects"][0]["creases"][0]["delta_m"]
    assert v["metrics"]["r_hat_m"]["verdict"] == "worsens"         # rounder than the reference
