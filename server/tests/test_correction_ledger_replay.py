"""Ledger + replay: N corrections (one undone) → the replay reproduces the
final epoch from the epoch-0 artifacts; the ledger keeps everything."""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.ledger import ledger_view                    # noqa: E402
from correction.replay import replay                         # noqa: E402
from correction.run import run_floor, run_objects, run_verdict  # noqa: E402
from correction.session import read_ply, read_poses          # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def test_ledger_and_replay(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08), floor="flat")
    out = scene.output_dir
    # keep the epoch-0 artifacts for the replay
    epoch0 = tmp_path / "epoch0"
    epoch0.mkdir()
    shutil.copy2(out / "cleaned_cloud.ply", epoch0 / "cleaned_cloud.ply")
    shutil.copy2(out / "camera_poses.txt", epoch0 / "camera_poses.txt")

    cfg = make_correction_cfg()
    # epoch 1: objects correction, approved
    rep1 = run_objects(out, [1, 2], "op", cfg=cfg)
    assert rep1["status"] == "pending"
    run_verdict(out, "approved", "op")
    # a floor alignment applied then UNDONE (stays in the ledger)
    repu = run_floor(out, "level", None, "op", cfg=cfg)
    assert repu["status"] == "pending", repu.get("rejection_reason")
    run_verdict(out, "undone", "op")
    # epoch 2: floor alignment (plane), approved
    rep2 = run_floor(out, "plane", None, "op", cfg=cfg)
    assert rep2["status"] == "pending", rep2.get("rejection_reason")
    run_verdict(out, "approved", "op")

    rows = ledger_view(out)
    assert [r["verdict"] for r in rows] == ["approved", "undone", "approved"]
    assert [r["kind"] for r in rows] == ["objects", "floor", "floor"]
    # the undone run is preserved with its verdict — history is never erased
    assert rows[1]["epoch_to"] == 2 and rows[1]["verdict"] == "undone"

    # replay epoch 0 → 2 reproduces the current cloud and poses
    res = replay(out, 2, cloud_path=epoch0 / "cleaned_cloud.ply",
                 poses_path=epoch0 / "camera_poses.txt",
                 out_dir=tmp_path / "replayed")
    _, cur = read_ply(out / "cleaned_cloud.ply")
    _, rep = read_ply(Path(res["cloud"]))
    for f in ("x", "y", "z"):
        assert np.allclose(cur[f], rep[f], atol=1e-4), f
    p_cur = read_poses(out / "camera_poses.txt")
    p_rep = read_poses(Path(res["poses"]))
    assert np.allclose(p_cur, p_rep, atol=1e-6)
