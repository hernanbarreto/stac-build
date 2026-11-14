"""The chat's measurements on the device (phase5_qa/tools_torch.py) against the path of record
(numpy / scipy in phase5_qa/tools.py): exact for counts, indices and verdicts, 1e-9 for geometry
(USER 2026-11-13: "el vlm y todas sus herramientas deben funcionar en GPU")."""
import math

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from phase5_qa import tools as chat_tools
from phase5_qa import tools_torch as tt
from phase5_qa.tools import POINT_TOOLS, SpatialTools
from phase_r.geometry import fit_gravity_aligned_obb
from phase_r.instance_store import InstanceStore

CARD = pytest.mark.skipif(not torch.cuda.is_available(), reason="no card visible")
# the hierarchical search forced on small sets: tiny budgets, a coarse start, sparse probes
FORCED = [dict(enum_total=1, pair_budget=5000, start_level=1, probe_pairs=4, probe_samples=3),
          dict(enum_total=20000, pair_budget=3000, start_level=2),
          dict(enum_total=1, pair_budget=700, start_level=3, probe_pairs=2, probe_samples=1)]


def _tree_pair(a, b):
    d, i = cKDTree(b).query(a, k=1)
    j = int(np.argmin(d))
    return float(d[j]), j, int(i[j])


def _clouds():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 10, size=(6000, 2))
    return {
        "apart": (rng.normal(size=(3000, 3)), rng.normal(size=(2500, 3)) + [4, 0, 0]),
        "mixed": (rng.uniform(size=(4000, 3)), rng.uniform(size=(3500, 3))),
        "parallel planes": (np.c_[x[:, 0], 0.01 * rng.normal(size=6000), x[:, 1]],
                            np.c_[rng.uniform(0, 10, 5000), 2.5 + 0.01 * rng.normal(size=5000),
                                  rng.uniform(0, 10, 5000)]),
        "floor and wall": (np.c_[rng.uniform(0, 10, 6000), 0.005 * rng.normal(size=6000),
                                 rng.uniform(0, 8, 6000)],
                           np.c_[rng.uniform(0, 10, 5000), rng.uniform(0, 3, 5000),
                                 8 + 0.005 * rng.normal(size=5000)]),
        "small and far": (rng.normal(size=(40, 3)) * 0.05,
                          rng.normal(size=(35, 3)) * 0.05 + [30, 2, -7]),
        "one point": (rng.normal(size=(1, 3)), rng.normal(size=(500, 3))),
        "a line": (np.c_[np.linspace(0, 5, 900), np.zeros(900), np.zeros(900)],
                   np.c_[np.linspace(0, 5, 800), np.full(800, 0.3), np.zeros(800)] + 1e-3),
    }


@pytest.mark.parametrize("name", sorted(_clouds()))
def test_closest_pair_is_the_trees(name):
    a, b = _clouds()[name]
    want = _tree_pair(a, b)
    for kw in [{}] + FORCED:
        d, i, j = tt.closest_pair(tt.tensor(a, "cpu"), tt.tensor(b, "cpu"), **kw)
        assert abs(d - want[0]) <= 1e-12, (name, kw)
        assert (i, j) == want[1:], (name, kw)


def test_closest_pair_exact_ties_take_the_first_of_each_set():
    # two lattices one cell apart: every facing pair is at exactly the same distance
    g = np.stack(np.meshgrid(np.arange(12.0), np.arange(12.0), indexing="ij"), -1).reshape(-1, 2)
    a = np.c_[g, np.zeros(len(g))]
    b = np.c_[g, np.full(len(g), 2.0)][::-1].copy()
    for kw in [{}] + FORCED:
        d, i, j = tt.closest_pair(tt.tensor(a, "cpu"), tt.tensor(b, "cpu"), **kw)
        assert d == 2.0
        assert i == 0                                   # numpy's first argmin over the first set
        assert j == int(np.flatnonzero((b[:, :2] == a[0, :2]).all(1)).min())
    # coincident points: the distance is zero, the first pair wins
    d, i, j = tt.closest_pair(tt.tensor(a, "cpu"), tt.tensor(a[::-1].copy(), "cpu"), **FORCED[0])
    assert (d, i, j) == (0.0, 0, len(a) - 1)


def test_closest_pair_refuses_an_empty_set():
    with pytest.raises(ValueError):
        tt.closest_pair(tt.tensor(np.zeros((0, 3)), "cpu"), tt.tensor(np.zeros((4, 3)), "cpu"))


def test_median_and_percentile_are_numpys():
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 10, 11, 1000, 1001):
        v = np.round(rng.normal(size=n), 2)            # repeated values too
        t = torch.as_tensor(v)
        assert tt.median(t) == pytest.approx(float(np.median(v)), abs=1e-15)
        for q in (0, 5, 50, 95, 99.5, 100):
            assert tt.percentile(t, q) == pytest.approx(float(np.percentile(v, q)), abs=1e-13)


def test_principal_axes_are_the_svds():
    rng = np.random.default_rng(5)
    for scale in ([5.0, 3.0, 0.004], [0.1, 4.0, 0.1], [1.0, 1.1, 0.9]):
        p = rng.normal(size=(5000, 3)) * scale + [3, -2, 7]
        c, vt = tt.principal_axes(tt.tensor(p, "cpu"))
        _u, _s, want = np.linalg.svd(p - p.mean(0), full_matrices=False)
        assert np.allclose(c, p.mean(0), atol=1e-12)
        for k in range(3):
            assert min(np.abs(vt[k] - want[k]).max(), np.abs(vt[k] + want[k]).max()) <= 1e-9


def test_free_placement_is_the_records_loops():
    rng = np.random.default_rng(11)
    for trial in range(40):
        n_vox = rng.integers(1, 9, size=3)
        need = np.minimum(rng.integers(1, 5, size=3), n_vox)
        occ = rng.uniform(size=tuple(n_vox)) < rng.choice([0.0, 0.05, 0.3, 0.9])
        keys = np.argwhere(occ)
        free = (~occ).astype(np.int64)
        want = SpatialTools._find_empty_box(free.cumsum(0).cumsum(1).cumsum(2), n_vox, need)
        got, n_free = tt.free_placement(torch.as_tensor(keys, dtype=torch.int64), n_vox, need)
        assert got == want, (trial, n_vox, need)
        assert n_free == int(free.sum())


# ── the tools, numpy against the device, on a store ─────────────────────────────────────────
def _store(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    store = InstanceStore(tmp_path / "scene_r.db")
    store.set_meta("built_from", "test_display_frame")
    xz = rng.uniform(-4, 4, size=(20000, 2))
    objects = {
        1: ("floor", np.c_[xz[:, 0], 0.004 * rng.normal(size=20000), xz[:, 1]]),
        2: ("wall", np.c_[rng.uniform(-4, 4, 9000), rng.uniform(0, 2.6, 9000),
                          4 + 0.004 * rng.normal(size=9000)]),
        3: ("ceiling", np.c_[rng.uniform(-4, 4, 8000), 2.6 + 0.02 * np.sin(rng.uniform(0, 6, 8000)),
                             rng.uniform(-4, 4, 8000)]),
        4: ("column", np.c_[1 + 0.15 * np.cos(t := rng.uniform(0, 2 * np.pi, 5000)),
                            rng.uniform(0, 2.6, 5000) + 0.0, -1 + 0.15 * np.sin(t)]),
        5: ("beam", np.c_[rng.uniform(-3, 3, 6000), 2.2 + rng.uniform(0, 0.3, 6000),
                          0.5 + rng.uniform(0, 0.2, 6000)]),
        6: ("fire_extinguisher", rng.normal(size=(400, 3)) * [0.06, 0.2, 0.06] + [3.5, 0.5, 3.6]),
        7: ("sign", rng.normal(size=(60, 3)) * [0.1, 0.1, 0.004] + [-2, 1.7, 3.95]),
        8: ("sign", rng.normal(size=(12, 3)) * 0.02 + [0, 1.5, 3.9]),        # under 30 points
    }
    for iid, (label, pts) in objects.items():
        store.upsert_instance(iid, label, source="test")
        store.set_points(iid, pts)
        obb = fit_gravity_aligned_obb(np.asarray(pts, np.float32).astype(np.float64))
        store.set_obb(iid, obb[0], obb[1], obb[2], n_points=len(pts))
    return store


def _calls():
    box = dict(center=[1.0, 1.2, -1.0], size=[2.0, 2.4, 2.0])
    return [
        ("get_clearance", dict(id1=1, id2=2)), ("get_clearance", dict(id1=4, id2=5)),
        ("get_clearance", dict(id1=6, id2=7)), ("get_clearance", dict(id1=7, id2=8)),
        ("measure_between", dict(id1=1, id2=3, feature1="top", feature2="lower", axis="vertical")),
        ("measure_between", dict(id1=4, id2=6, feature1="highest", feature2="lowest")),
        ("measure_between", dict(id1=4, id2=6, feature1="closest", feature2="closest")),
        ("measure_between", dict(id1=1, id2=8, feature1="centroid", feature2="top")),
        ("get_extent", dict(id=3)), ("get_extent", dict(id=8)),
        ("fits_through", dict(item_size=[0.4, 0.4, 1.0], id1=4, id2=6)),
        ("get_plumb", dict(id=4)), ("get_plumb", dict(id=2)), ("get_level", dict(id=1)),
        ("get_level", dict(id=3)),
        ("get_height_profile", dict(id=5, floor_id=1)),
        ("get_height_profile", dict(id=3, n_stations=7)),
        ("get_flatness_report", dict(id=1)), ("get_flatness_report", dict(id=3, tolerance_mm=30.0)),
        ("evaluate_volume", box), ("evaluate_volume", dict(box, yaw_deg=33.0, voxel_m=0.05)),
        ("evaluate_volume", dict(center=[40, 40, 40], size=[1, 1, 1])),
        ("objects_in_volume", box), ("objects_in_volume", dict(box, yaw_deg=33.0)),
        ("fits_in_volume", dict(box, item_size=[0.5, 0.5, 0.5])),
        ("fits_in_volume", dict(box, item_size=[1.9, 2.3, 1.9])),
        ("fits_in_volume", dict(box, item_size=[3.0, 1.0, 1.0])),
        ("fits_in_volume", dict(center=[0, 1.3, 0], size=[7.5, 2.5, 7.5], item_size=[2, 2, 2],
                                voxel_m=0.25)),
    ]


def _same(a, b, where=""):
    """Exact for everything that is not a float; 1e-9 for floats."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None \
            or isinstance(a, (str, int)) and not isinstance(a, bool) and isinstance(b, (str, int)):
        assert a == b and type(a) is type(b), (where, a, b)
    elif isinstance(a, float) or isinstance(b, float):
        assert float(a) == pytest.approx(float(b), abs=1e-9), (where, a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (where, list(a), list(b))
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), (where, a, b)
        for k, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{k}]")
    else:
        assert a == b, (where, a, b)


def _add(store, iid, label, pts):
    store.upsert_instance(iid, label, source="test")
    store.set_points(iid, pts)
    obb = fit_gravity_aligned_obb(np.asarray(pts, np.float32).astype(np.float64))
    store.set_obb(iid, obb[0], obb[1], obb[2], n_points=len(pts))


def test_plumb_and_level_follow_the_elements_shape(tmp_path):
    """A wall is measured on its FACE and a column on its AXIS (until 2026-11-14 the long axis was
    taken for everything: pccr 3108's largest wall read "out of plumb 89°") — and the same on
    both paths."""
    store = _store(tmp_path)
    rng = np.random.default_rng(5)
    lean = math.radians(2.0)
    # a wall 8 m long leaning 2° about its length (x): the face's normal tilts 2° off horizontal
    h = rng.uniform(0, 2.6, 9000)
    _add(store, 9, "wall", np.c_[rng.uniform(-4, 4, 9000), h * math.cos(lean),
                                 -3 + h * math.sin(lean) + 0.002 * rng.normal(size=9000)])
    # a post leaning 3° towards +x
    t, hh = rng.uniform(0, 2 * np.pi, 4000), rng.uniform(0, 3.0, 4000)
    tilt = math.radians(3.0)
    _add(store, 10, "post", np.c_[2 + 0.05 * np.cos(t) + hh * math.sin(tilt), hh * math.cos(tilt),
                                  2 + 0.05 * np.sin(t)])
    try:
        rec = SpatialTools(store, device="numpy")
        dev = SpatialTools(store, device="cpu")
        for name, iid in (("get_plumb", 2), ("get_plumb", 9), ("get_plumb", 4), ("get_plumb", 10),
                          ("get_plumb", 1), ("get_plumb", 5), ("get_level", 1), ("get_level", 5),
                          ("get_level", 2), ("get_level", 4), ("get_level", 9)):
            _same(getattr(rec, name)(iid), getattr(dev, name)(iid), f"{name}({iid})")

        wall, leaning = rec.get_plumb(2), rec.get_plumb(9)
        assert wall["measured_on"] == "face" and wall["vertical_element"] is True
        assert wall["plumb_deviation_deg"] < 0.05                      # built vertical
        assert leaning["measured_on"] == "face"
        assert leaning["plumb_deviation_deg"] == pytest.approx(2.0, abs=0.02)
        assert leaning["deviation_mm_per_m"] == pytest.approx(math.tan(lean) * 1000, abs=0.4)

        column, post = rec.get_plumb(4), rec.get_plumb(10)
        assert column["measured_on"] == "axis" and column["plumb_deviation_deg"] < 0.5
        assert post["measured_on"] == "axis"
        assert post["plumb_deviation_deg"] == pytest.approx(3.0, abs=0.05)

        # plumb does not apply to what lies flat: said, with no mm/m
        floor, beam = rec.get_plumb(1), rec.get_plumb(5)
        assert floor["measured_on"] == "face" and floor["vertical_element"] is False
        assert floor["plumb_deviation_deg"] > 89.9 and "deviation_mm_per_m" not in floor
        assert beam["measured_on"] == "axis" and beam["vertical_element"] is False

        # level: the floor's face, the beam's axis; a wall is not a horizontal element
        assert rec.get_level(1)["measured_on"] == "face" and rec.get_level(1)["level_deviation_deg"] < 0.05
        lvl = rec.get_level(5)
        assert lvl["measured_on"] == "axis" and lvl["horizontal_element"] is True
        assert lvl["level_deviation_deg"] < 0.5
        up = rec.get_level(2)
        assert up["horizontal_element"] is False and "deviation_mm_per_m" not in up

        # the sentence the tool layer writes says what was measured
        from phase5_qa import answers
        say = lambda tool, iid, lang: answers.compose(     # noqa: E731
            tool, {"id": iid}, {**getattr(rec, tool)(iid), "id": iid, "label": "x"}, lang)
        assert "desplome de su cara" in say("get_plumb", 9, "es")
        assert "desplome de su eje" in say("get_plumb", 10, "es")
        assert "no es un elemento vertical" in say("get_plumb", 1, "es")
        assert "its face is out of plumb" in say("get_plumb", 9, "en")
        assert "no es un elemento horizontal" in say("get_level", 2, "es")
        assert "desnivel de su eje" in say("get_level", 5, "es")
    finally:
        store.close()


def test_the_devices_singular_values_are_numpys():
    rng = np.random.default_rng(3)
    for scale in ([5, 2, 0.01], [3, 0.2, 0.2], [1, 1, 1]):
        p = rng.normal(size=(4000, 3)) * scale + [10, -3, 7]
        _u, s, _vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
        _c, _v, got = tt.principal_axes_values(tt.tensor(p, "cpu"))
        assert np.allclose(got, s, rtol=1e-12, atol=0)


def _compare(store, device):
    rec = SpatialTools(store, device="numpy")
    dev = SpatialTools(store, device=device)
    names = set()
    for name, kw in _calls():
        _same(getattr(rec, name)(**kw), getattr(dev, name)(**kw), name)
        names.add(name)
    # define_volume writes a volume: the same placement on both paths, then measured from
    for t in (rec, dev):
        v = t.define_volume("bay", volume_m3=1.0)
        v2 = t.define_volume("by the column", size=[0.5, 1.0, 0.5], anchor_id=4)
        t._vols = (v, v2)
    _same({k: x for k, x in rec._vols[0].items() if k != "volume_id"},
          {k: x for k, x in dev._vols[0].items() if k != "volume_id"}, "define_volume")
    _same({k: x for k, x in rec._vols[1].items() if k != "volume_id"},
          {k: x for k, x in dev._vols[1].items() if k != "volume_id"}, "define_volume anchored")
    for oid in (1, 4, 6):
        for axis in ("straight", "vertical"):
            _same(rec.measure_volume(rec._vols[1]["volume_id"], id=oid, axis=axis),
                  dev.measure_volume(rec._vols[1]["volume_id"], id=oid, axis=axis), "measure_volume")
    names |= {"define_volume", "measure_volume"}
    assert names == set(POINT_TOOLS), "a point tool is not compared"
    dev.release()


def test_every_point_tool_is_equal_on_the_cpu_device(tmp_path):
    store = _store(tmp_path)
    try:
        _compare(store, "cpu")
    finally:
        store.close()


def test_the_hierarchical_search_inside_a_tool(tmp_path, monkeypatch):
    store = _store(tmp_path, seed=4)
    try:
        want = SpatialTools(store, device="numpy").get_clearance(1, 2)
        monkeypatch.setattr(tt, "ENUM_TOTAL", 1)
        monkeypatch.setattr(tt, "PAIR_BUDGET", 4000)
        real = tt.closest_pair
        monkeypatch.setattr(tt, "closest_pair", lambda a, b: real(a, b, enum_total=1, pair_budget=4000))
        _same(want, SpatialTools(store, device="cpu").get_clearance(1, 2))
    finally:
        store.close()


@CARD
def test_every_point_tool_is_equal_on_the_card_in_strict_mode(tmp_path):
    store = _store(tmp_path, seed=2)
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)      # the tools process's own mode: no kernel may refuse it
    try:
        _compare(store, "cuda")
        a, b = _clouds()["parallel planes"]
        for kw in [{}] + FORCED:
            d, i, j = tt.closest_pair(tt.tensor(a, "cuda"), tt.tensor(b, "cuda"), **kw)
            assert abs(d - _tree_pair(a, b)[0]) <= 1e-12 and (i, j) == _tree_pair(a, b)[1:]
    finally:
        torch.use_deterministic_algorithms(before)
        store.close()


def test_points_and_boxes_are_measured_in_float64(tmp_path):
    store = _store(tmp_path)
    try:
        t = SpatialTools(store, device="numpy")
        assert t._points(1).dtype == np.float64
        assert all(v.dtype == np.float64 for v in t._obb(1))
        assert np.array_equal(t._points(1), store.get_points(1).astype(np.float64))
    finally:
        store.close()


def test_the_device_key_is_typed():
    with pytest.raises(ValueError, match="semantic.chat.tools_device"):
        tt.resolve_device("gpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="semantic.chat.tools_device"):
            tt.resolve_device("cuda")
    assert chat_tools.chat_config()["tools_device"] in tt.DEVICES


def test_the_suite_runs_the_chat_tools_on_the_cpu(tmp_path):
    store = _store(tmp_path)
    try:
        assert chat_tools.effective_device() == "cpu"          # tests/conftest.py
        assert SpatialTools(store)._dev == "cpu"
    finally:
        store.close()


# ── the tools process ───────────────────────────────────────────────────────────────────────
def test_the_tools_process_measures_and_ends(tmp_path, monkeypatch):
    from phase5_qa import tools_worker
    store = _store(tmp_path)
    want = SpatialTools(store, device="numpy").get_clearance(1, 2)
    box = dict(center=[1.0, 1.2, -1.0], size=[2.0, 2.4, 2.0])
    want_vol = SpatialTools(store, device="numpy").evaluate_volume(**box)
    store.close()
    sp = str(tmp_path / "scene_r.db")
    monkeypatch.setattr(tools_worker, "_lease_refusal", lambda: None)
    said = []
    try:
        _same(want, tools_worker.call(sp, "get_clearance", dict(id1=1, id2=2), device="cpu",
                                      log=said.append))
        st = tools_worker.status()
        assert st["running"] and st["device"] == "cpu"
        pid = st["pid"]
        _same(want_vol, tools_worker.call(sp, "evaluate_volume", box, device="cpu"))
        assert tools_worker.status()["pid"] == pid               # the same process, kept
        # a tool that is not a point tool is not this process's; the tool's own error is told
        with pytest.raises(tools_worker.ToolsWorkerError, match="not a point tool"):
            tools_worker.call(sp, "list_objects", {}, device="cpu")
        with pytest.raises(tools_worker.ToolsWorkerError, match="TypeError"):
            tools_worker.call(sp, "get_clearance", dict(id1=1), device="cpu")
        assert tools_worker.status()["pid"] == pid               # it survives a tool's error
    finally:
        assert tools_worker.stop("test done", log=said.append) is True
    assert tools_worker.status() == {"running": False}
    import os
    assert not os.path.exists(f"/proc/{pid}") or open(f"/proc/{pid}/stat").read().split()[2] == "Z"


def test_the_tools_process_is_not_started_beside_a_job(tmp_path, monkeypatch):
    from phase5_qa import tools_worker
    store = _store(tmp_path)
    store.close()
    monkeypatch.setattr(tools_worker, "_lease_refusal", lambda: "pipeline holds the GPU (job pccr)")
    with pytest.raises(tools_worker.ToolsWorkerBusy, match="pipeline holds the GPU"):
        tools_worker.call(str(tmp_path / "scene_r.db"), "get_clearance", dict(id1=1, id2=2),
                          device="cpu")
    assert tools_worker.warm(str(tmp_path / "scene_r.db"), device="cpu") is False
    assert tools_worker.status() == {"running": False}


def test_a_new_segmentation_restarts_the_tools_process(tmp_path, monkeypatch):
    from phase5_qa import tools_worker
    store = _store(tmp_path)
    store.close()
    sp = str(tmp_path / "scene_r.db")
    monkeypatch.setattr(tools_worker, "_lease_refusal", lambda: None)
    try:
        tools_worker.call(sp, "get_extent", dict(id=1), device="cpu")
        first = tools_worker.status()["pid"]
        (tmp_path / "segmentation_result.json").write_text("{}")   # the scan was segmented again
        tools_worker.call(sp, "get_extent", dict(id=1), device="cpu")
        assert tools_worker.status()["pid"] != first
    finally:
        tools_worker.stop("test done")
