"""segmentation.pipeline.voxel_mesh_entry — the cleaning's 5 cm quads reach the result only when
models.segmentation.segment_cleaning.voxel_mesh is true (USER 2026-11-02: off)."""
from pathlib import Path

import yaml

from segmentation.pipeline import voxel_mesh_entry

_DATA = [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.05, 0.0, 0.0, 0.0, 1.0, 0.0]]


def _cfg(v):
    return {"models": {"segmentation": {"segment_cleaning": {"voxel_mesh": v}}}}


def test_off_unless_asked():
    assert voxel_mesh_entry(_DATA, _cfg(False)) is None
    assert voxel_mesh_entry(_DATA, {"models": {"segmentation": {"segment_cleaning": {}}}}) is None   # a frozen config without it


def test_on_when_asked():
    assert voxel_mesh_entry(_DATA, _cfg(True)) == {"voxel_size": 0.05, "count": 2, "data": _DATA}
    assert voxel_mesh_entry([], _cfg(True)) is None


def test_the_shipped_configuration_keeps_it_off():
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    assert cfg["models"]["segmentation"]["segment_cleaning"]["voxel_mesh"] is False
