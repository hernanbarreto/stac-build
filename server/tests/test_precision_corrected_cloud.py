"""precision/corrected_cloud.py — the corrected Omega cloud as the product: the camera
on Omega's record grid, the chain the product selects, the publication as a new-cloud
epoch the product helper recognises (no GPU: the publication is exercised with the
octree stubbed, the recipe's GPU steps are the cloud stage's own, tested there)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from precision import corrected_cloud as CC                      # noqa: E402
from precision import product as PR                              # noqa: E402


def test_camera_on_the_record_grid_identity_and_mapped():
    from precision.camera import GridMap
    K = np.array([[400.0, 0, 232.0], [0, 402.0, 416.0], [0, 0, 1]])
    g0 = GridMap(name="omega", w=464, h=832, content_w=464, content_h=832, pad_left=0, pad_top=0,
                 crop_x=0, crop_y=0, crop_w=464, crop_h=832, native_w=464, native_h=832)
    assert np.allclose(CC.K_on_grid(K, g0), K)
    # a half-resolution grid of a crop: 2 native pixels per grid pixel, padded
    g1 = replace(g0, w=232, h=416, content_w=232, content_h=416, crop_x=10, crop_y=20, pad_left=3, pad_top=4)
    assert g1.scale_x == 2.0
    Kg = CC.K_on_grid(K, g1)
    assert Kg[0, 0] == 200.0 and Kg[1, 1] == 201.0
    assert Kg[0, 2] == (232.0 - 10) / 2 + 3 and Kg[1, 2] == (416.0 - 20) / 2 + 4


def test_the_product_selects_the_chain():
    from precision import runner as RN
    from precision.config import load_precision_config
    p = load_precision_config()
    corrected = [s.key for s in RN.chain_steps(replace(p, cloud=replace(p.cloud, source="omega_corrected")))]
    fusion = [s.key for s in RN.chain_steps(replace(p, cloud=replace(p.cloud, source="fusion")))]
    assert "f7_cloud" in corrected and not {"f6_sweep", "f7_fuse", "f6_colmap"} & set(corrected)
    assert "f7_cloud" not in fusion and {"f6_sweep", "f7_fuse"} <= set(fusion)
    assert corrected[-1] == fusion[-1] == "f6_check"
    assert corrected.index("f7_cloud") == corrected.index("f5_refine") + 1


def test_config_rejects_an_unknown_source():
    import copy
    from precision.config import PrecisionConfigError, load_precision_config
    from config import cfg as raw
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["cloud"]["source"] = "magic"
    with pytest.raises(PrecisionConfigError, match="cloud.source"):
        load_precision_config(bad)


def _ply(path: Path, n: int) -> None:
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((n, 3)).astype(np.float32)
    rgb = rng.integers(0, 255, (n, 3)).astype(np.uint8)
    CC_ = __import__("precision.epoch0_cloud", fromlist=["x"])
    CC_._write_ply_xyzrgb(path, xyz, rgb)


def test_publish_is_a_new_cloud_epoch_the_product_helper_recognises(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    _ply(out / "cleaned_cloud.ply", 20)                     # the session's previous cloud (epoch 0)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    (out / "segmentation_result.json").write_text(json.dumps({"instances": [{"id": 1}]}))
    tmp = out / CC.TX_TMP
    tmp.mkdir()
    _ply(tmp / "cleaned_cloud.ply", 30)

    def fake_potree(session_dir, force=False, ply_override=None, potree_dir_override=None):
        Path(potree_dir_override).mkdir(parents=True)
        (Path(potree_dir_override) / "metadata.json").write_text(json.dumps({"points": 30}))
        return True
    import potree_converter
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree", fake_potree)
    assert PR.product_report(out) is None
    rep = CC.publish(tmp_path, tmp, {"stage": "corrected_cloud", "per_frame": {}}, log=lambda m: None)
    assert rep["epoch_from"] == 0 and rep["epoch_to"] == 1 and rep["n_points"] == 30
    assert json.loads((out / "geometry_epoch.json").read_text())["epoch"] == 1
    from correction.epoch import epoch_kind
    assert epoch_kind(out, 1) == "new_cloud"
    assert (out / "_epoch_0" / "cleaned_cloud.ply").exists()          # the previous cloud stays selectable
    (out / CC.CLOUD_REPORT).write_text(json.dumps(rep, default=float))
    live, why = PR.product_is_live(out)
    assert live and "corrected_cloud" in why
    seg = json.loads((out / "segmentation_result.json").read_text())
    assert seg["instances"] == [] and "pending" in seg                 # the cloud stage projects on it
    # the fusion's report, if an older one were present, does not outrank the live product
    (out / "fuse_report.json").write_text(json.dumps({"epoch_to": 0}))
    assert PR.product_report(out)["product_file"] == CC.CLOUD_REPORT
