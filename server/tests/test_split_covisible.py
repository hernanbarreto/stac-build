"""Several objects under one instance are split — and one surface with holes is not (pccr 2026-09-30).

Desk #174 held three desks 2 m apart (one SAM3 mask over neighbouring desks + the space dedupe).
The first split also cut the floor in 2 and the ceiling in 3: their fragments, separated by the
holes the cleaning left, are seen together too. What tells them apart is what the cameras see
through the gap: air between desks shows the floor behind; nothing lies behind a hole in a floor.
A drift duplicate (never in the same frames) is not split either — the certification corrects it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.pipeline import _split_covisible_components  # noqa: E402

RNG = np.random.default_rng(0)


def _box(c, n, ext=(1.3, 0.05, 0.7)):
    return RNG.uniform(-0.5, 0.5, (n, 3)) * np.asarray(ext) + np.asarray(c, float)


def _run(blobs, frames_of_blob, cams, extra=None):
    """blobs belong to ONE instance; `extra` = other points of the scene (other objects)."""
    xyz = np.vstack(blobs + ([extra] if extra is not None else []))
    fr = np.concatenate([RNG.choice(list(fs), len(b)) for b, fs in zip(blobs, frames_of_blob)]
                        + ([np.full(len(extra), 999)] if extra is not None else [])).astype(np.int32)
    n_inst = sum(len(b) for b in blobs)
    inst = [{"id": 9, "instance_id": 10, "label": "x", "globalIndices": list(range(n_inst))}]
    cam_centre = {f: np.asarray(c, float) for f, c in cams.items()}
    added = _split_covisible_components(inst, xyz, fr, gap_m=0.10, min_points=1000, covis_share=0.5,
                                        cam_centre=cam_centre, min_walk_m=1.0)
    return inst, added


CAMS = {f: (x, 1.5, 3.0) for f, x in zip(range(10), np.linspace(-1, 5, 10))}   # above, in front


def test_desks_with_air_between_are_split():
    desks = [_box((x, 0.75, 0.0), 8000) for x in (0.0, 2.0, 4.0)]            # desk tops at 0.75 m
    floor = _box((2.0, 0.0, -1.0), 60000, ext=(8.0, 0.02, 6.0))              # the floor behind/below
    inst, added = _run(desks, [range(10)] * 3, CAMS, extra=floor)
    assert added == 2 and len(inst) == 3


def test_a_floor_with_holes_is_not_split():
    frags = [_box((x, 0.0, 0.0), 8000, ext=(1.5, 0.02, 1.5)) for x in (0.0, 2.0, 4.0)]   # nothing below
    inst, added = _run(frags, [range(10)] * 3, CAMS)
    assert added == 0 and len(inst) == 1


def test_a_drift_duplicate_is_not_split():
    """Seen from two passes: the walk goes away and COMES BACK to the same place (a loop)."""
    loop = {f: (x, 1.5, 3.0) for f, x in zip(range(10), [0.0, 0.3, 0.6, 2.0, 4.0, 6.0, 4.0, 2.0, 0.5, 0.2])}
    copies = [_box((0.0, 0.75, 0.0), 8000), _box((0.8, 0.75, 0.0), 8000)]
    floor = _box((2.0, 0.0, -1.0), 60000, ext=(8.0, 0.02, 6.0))
    inst, added = _run(copies, [range(0, 3), range(8, 10)], loop, extra=floor)
    assert added == 0 and len(inst) == 1


def test_without_cameras_nothing_is_split():
    desks = [_box((x, 0.75, 0.0), 8000) for x in (0.0, 2.0, 4.0)]
    xyz = np.vstack(desks); fr = np.zeros(len(xyz), np.int32)
    inst = [{"id": 9, "instance_id": 10, "label": "x", "globalIndices": list(range(len(xyz)))}]
    assert _split_covisible_components(inst, xyz, fr, 0.10, 1000, 0.5, cam_centre=None) == 0


def test_a_row_of_desks_walked_past_one_by_one_is_split():
    """pccr desk #174: never all in one frame, but all in ONE pass — a pass cannot double an object."""
    desks = [_box((x, 0.75, 0.0), 8000) for x in (0.0, 2.0, 4.0)]
    floor = _box((2.0, 0.0, -1.0), 60000, ext=(8.0, 0.02, 6.0))
    inst, added = _run(desks, [range(0, 3), range(3, 6), range(6, 9)], CAMS, extra=floor)
    assert added == 2 and len(inst) == 3


def test_the_gap_grid_is_world_anchored_an_extra_point_changes_only_its_cell(monkeypatch):
    """docs/plan_determinismo.md point 102 (DECIDIDO): the gap grid used to start at the instance's
    minimum. Two fragments 0.10 m apart sit in ADJACENT world cells (x ∈ [0, 0.05] → cell 0,
    x ∈ [0.15, 0.199] → cell 1): ONE component. One extra point at x = −0.002 moved the old grid
    by 2 mm and put the fragments in cells 0 and 2 — two components, a split candidate out of a
    flyer. World-anchored, the extra point lands in cell −1, adjacent to cell 0: still one."""
    import scipy.ndimage as ndi                       # the module the split calls `ndimage.label` on
    seen = []
    real = ndi.label

    def _label(grid, structure=None):
        lab, n = real(grid, structure=structure)
        seen.append(int(n))
        return lab, n
    monkeypatch.setattr(ndi, "label", _label)
    rng = np.random.default_rng(5)
    a = np.column_stack([rng.uniform(0.0, 0.05, 3000), rng.uniform(0.0, 0.05, 3000), rng.uniform(0.0, 0.05, 3000)])
    b = np.column_stack([rng.uniform(0.15, 0.199, 3000), rng.uniform(0.0, 0.05, 3000), rng.uniform(0.0, 0.05, 3000)])
    for extra in (None, np.array([[-0.002, 0.02, 0.02]])):
        xyz = np.vstack([a, b] + ([extra] if extra is not None else []))
        fr = rng.choice(10, len(xyz)).astype(np.int32)
        inst = [{"id": 9, "instance_id": 10, "label": "x", "globalIndices": list(range(len(xyz)))}]
        added = _split_covisible_components(inst, xyz, fr, gap_m=0.10, min_points=1000, covis_share=0.5,
                                            cam_centre={f: np.asarray(c, float) for f, c in CAMS.items()},
                                            min_walk_m=1.0)
        assert added == 0 and len(inst) == 1
    assert seen == [1, 1], f"components with / without the extra point: {seen} — the grid moved"
