"""The space dedupe's voxel sets on the device (segmentation/dedupe_torch.py) against the numpy path of
record (segmentation.pipeline.VoxelSetsNumpy): the same intersections, judges, decisions and merges; and
the attach step's neighbour vote on the device (segmentation/attach_torch.py) against the cKDTree recipe
(_attach_unsegmented_numpy): the same owners, the same instances."""
import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from segmentation import pipeline as SP  # noqa: E402
from segmentation.dedupe_torch import VoxelSetsTorch  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")
DECISION = (1.1, 0.95, 5)        # (error factor, confidence, minimum judges) — the configured rule's shape


def _scene(seed=0):
    """Six instances: a floor and its duplicate under another name (the survivor is a support surface), a
    desk and a drift copy of it 2 cm apart (same label), a wall and a sign resting on it (contained, not
    identical); points born in 40 keyframes; a 5 cm voxel grid."""
    rng = np.random.default_rng(seed)
    n_kf = 40

    def blob(center, ext, n):
        return rng.uniform(-1, 1, (n, 3)) * np.asarray(ext) + np.asarray(center)
    parts = [("floor", blob((0, 0, 0), (3, 0.01, 3), 6000)),
             ("ground", blob((0.05, 0, 0.03), (3, 0.01, 3), 5000)),      # the same floor, another name
             ("desk", blob((1, 0.75, 1), (0.6, 0.02, 0.4), 3000)),
             ("desk", blob((1.02, 0.75, 1.01), (0.6, 0.02, 0.4), 2500)),  # a drift copy
             ("wall", blob((-2, 1.2, 0), (0.02, 1.2, 2), 4000)),
             ("sign", blob((-1.97, 1.5, 0.5), (0.005, 0.2, 0.3), 800))]
    xyz = np.concatenate([p for _, p in parts])
    frame = rng.integers(0, n_kf, len(xyz))
    instances, start = [], 0
    for k, (label, p) in enumerate(parts):
        instances.append({"id": k + 1, "instance_id": k + 1, "label": label,
                          "globalIndices": list(range(start, start + len(p))), "total_points": len(p)})
        start += len(p)
    return instances, xyz, frame


def _run(backend_sets, instances, xyz, frame, support_only=False):
    inst = copy.deepcopy(instances)
    sets = backend_sets(inst)
    absorbed_into, decisions = {}, []
    policy = SimpleNamespace(support={"floor", "ground"})
    out = SP._space_dedupe(inst, sets, xyz, {"segmentation": {}}, 0.5, True, policy, DECISION, absorbed_into, decisions,
                           support_only=support_only)
    return out, absorbed_into, decisions


def test_space_dedupe_on_the_device_takes_the_numpy_decisions(monkeypatch):
    monkeypatch.setattr(SP, "_compute_obb", lambda pts, cfg=None, **k: {"stub": int(len(pts))})
    for seed in range(3):
        instances, xyz, frame = _scene(seed)
        a = _run(lambda inst: SP.VoxelSetsNumpy(inst, xyz, frame, 0.05), instances, xyz, frame)
        b = _run(lambda inst: VoxelSetsTorch(inst, xyz, frame, 0.05, DEV), instances, xyz, frame)
        assert [i["id"] for i in a[0]] == [i["id"] for i in b[0]]
        for ia, ib in zip(a[0], b[0]):
            assert ia["globalIndices"] == ib["globalIndices"] and ia["total_points"] == ib["total_points"]
        assert a[1] == b[1]
        assert len(a[2]) == len(b[2]) > 0
        for da, db in zip(a[2], b[2]):
            assert da == db, (da, db)
        assert any(d["merged"] for d in a[2])             # the floor's duplicate and the desk's copy merge


def test_only_the_floor_merges_when_merges_are_support_only(monkeypatch):
    """USER 2026-11-11 ("no quiero hacer una pasada para fusionar objetos, solo el piso … a veces une dos
    objetos que están muy alejados" / "lo que dije que se evitara es después de sam3 fusionar lo
    segmentado"): with segmentation.merge_support_only the floor and its duplicate under another name
    still merge; the desk and its drift copy stay two, the pair declared with the reason."""
    monkeypatch.setattr(SP, "_compute_obb", lambda pts, cfg=None, **k: {"stub": int(len(pts))})
    instances, xyz, frame = _scene(0)
    # the 'ground' instance made the floor seen again: the floor's own points, 1 mm apart
    fl = np.asarray(instances[0]["globalIndices"])
    gr = np.asarray(instances[1]["globalIndices"])
    rng = np.random.default_rng(5)
    xyz = xyz.copy()
    xyz[gr] = xyz[fl[:len(gr)]] + rng.normal(0, 0.001, (len(gr), 3))
    sets = lambda inst: SP.VoxelSetsNumpy(inst, xyz, frame, 0.05)  # noqa: E731
    out, absorbed_into, decisions = _run(sets, instances, xyz, frame, support_only=True)
    assert sorted(i["label"] for i in out) == ["desk", "desk", "floor", "sign", "wall"]
    assert [v["into_label"] for v in absorbed_into.values()] == ["floor"]
    desk = [d for d in decisions if set(d["instance_ids"]) == {3, 4}]
    assert desk and not desk[0]["merged"] and "only the floor merges" in desk[0]["reason"]
    # without the rule the desk's copy merges too (the general path, unchanged)
    out_all, _, _ = _run(sets, instances, xyz, frame, support_only=False)
    assert sorted(i["label"] for i in out_all) == ["desk", "floor", "sign", "wall"]


def test_voxel_sets_intersections_and_judges_equal():
    instances, xyz, frame = _scene(1)
    A = SP.VoxelSetsNumpy(instances, xyz, frame, 0.05)
    B = VoxelSetsTorch(instances, xyz, frame, 0.05, DEV)
    n = len(instances)
    for i in range(n):
        assert A.n(i) == B.n(i)
        js = [j for j in range(n) if j != i]
        assert A.inter_counts(i, js) == B.inter_counts(i, js)
        for j in js:
            common = A.common_keyframes(i, j)
            assert common == B.common_keyframes(i, j)
            assert A.judges(i, j, common, True) == B.judges(i, j, common, True)
            assert A.judges(i, j, common, False) == B.judges(i, j, common, False)
    A.absorb(0, 1)
    B.absorb(0, 1)
    assert A.n(0) == B.n(0)
    assert A.inter_counts(0, [2, 3, 4, 5]) == B.inter_counts(0, [2, 3, 4, 5])


def test_attach_on_the_device_equals_the_kd_tree_recipe(monkeypatch):
    rng = np.random.default_rng(4)
    a = np.column_stack([rng.uniform(0, 2, 20000), rng.normal(0, 0.004, 20000), rng.uniform(0, 1, 20000)])
    b = np.column_stack([rng.uniform(0, 2, 15000), 0.5 + rng.normal(0, 0.004, 15000), rng.uniform(0, 1, 15000)])
    c = np.column_stack([rng.uniform(0, 2, 30000), rng.uniform(-0.05, 0.55, 30000), rng.uniform(0, 1, 30000)])
    xyz = np.concatenate([a, b, c])
    inst = [{"id": 1, "label": "floor", "globalIndices": list(range(0, 20000)), "total_points": 20000},
            {"id": 2, "label": "table", "globalIndices": list(range(20000, 35000)), "total_points": 15000}]
    ia, ib = copy.deepcopy(inst), copy.deepcopy(inst)
    na, grown_a = SP._attach_unsegmented_numpy(ia, xyz, 0.03, 4, 3)
    nb, grown_b = SP._attach_unsegmented(ib, xyz, 0.03, 4, 3, device=DEV)
    assert na == nb > 0 and grown_a == grown_b
    for x, y in zip(ia, ib):
        assert x["globalIndices"] == y["globalIndices"] and x["total_points"] == y["total_points"]
