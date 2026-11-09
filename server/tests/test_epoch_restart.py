"""The way back to epoch 0 for a precision chain that starts again (zaragoza 2026-11-09: its code changed,
Omega's pass did not, the publish had deleted every stored epoch — F2 failed on epoch 3).
correction/restart.py: the publish keeps the lineage's stores out of the listed epochs; a restart returns
them and selects epoch 0 with the journaled select."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.apply import MANIFEST_NAME, PREV_PREFIX, available_epochs  # noqa: E402
from correction.restart import RESTART_DIR, restore_epoch0, stash_lineage  # noqa: E402

ARTS = ["cleaned_cloud.ply", "camera_poses.txt", "geometry_epoch.json"]


def _store(out: Path, e: int, payload: str):
    d = out / f"{PREV_PREFIX}{e}"
    d.mkdir(parents=True)
    for rel in ARTS:
        (d / rel).write_text(payload if rel != "geometry_epoch.json" else json.dumps({"epoch": e}))
    (d / MANIFEST_NAME).write_text(json.dumps(
        {"epoch_from": e - 1, "epoch_to": e, "artifacts": [{"rel": r, "existed_before": True} for r in ARTS]}))


@pytest.fixture()
def published(tmp_path):
    """A session whose chain published epoch 3 over 0 (Omega), 1 (F2), 2 (F5)."""
    out = tmp_path / "output"
    out.mkdir()
    for rel in ARTS:
        (out / rel).write_text("live-3" if rel != "geometry_epoch.json" else json.dumps({"epoch": 3}))
    for e in (0, 1, 2):
        _store(out, e, f"state-{e}")
    return out


def test_the_publish_keeps_the_way_back_out_of_the_listed_epochs(published):
    stash_lineage(published)
    assert [e["epoch"] for e in available_epochs(published)] == [3], "the viewer lists the published epoch only"
    assert sorted(d.name for d in (published / RESTART_DIR).iterdir()) == ["_epoch_0", "_epoch_1", "_epoch_2"]


def test_a_chain_that_starts_again_is_back_in_omegas_epoch_0(published):
    stash_lineage(published)
    assert restore_epoch0(published, log=lambda *a: None) is True
    assert json.loads((published / "geometry_epoch.json").read_text())["epoch"] == 0
    assert (published / "camera_poses.txt").read_text() == "state-0", "Omega's poses, not the published ones"
    assert restore_epoch0(published, log=lambda *a: None) is False, "already in epoch 0: nothing to do"


def test_a_session_without_epoch_0_kept_fails_naming_it(published):
    import shutil
    shutil.rmtree(published / f"{PREV_PREFIX}0")
    stash_lineage(published)
    with pytest.raises(RuntimeError, match="epoch 0 is not kept"):
        restore_epoch0(published, log=lambda *a: None)
