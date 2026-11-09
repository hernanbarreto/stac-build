"""The cleaner subprocess gets the card: the F6 process hands its cached memory back BEFORE launching it
(bufferStop 2026-11-08: the F6 process held 78.6 GB reserved and the cleaner of 130.7 M points died of OOM)."""
from pathlib import Path

import precision.corrected_cloud as CC


def test_clean_releases_the_card_before_the_cleaner_runs(monkeypatch, tmp_path: Path):
    order = []
    cleaned = tmp_path / "cleaned_cloud.ply"

    class FakePopen:
        """The cleaner as a live stream (2026-11-09: its every line reaches the log as it is printed)."""
        def __init__(self, cmd, **kw):
            order.append("cleaner")
            cleaned.write_bytes(b"ply\n")
            self.stdout = iter(["[Step 3/6] Voxel spatial subsampling (1.0mm)...\n", "✅ done\n"])

        def wait(self):
            return 0

    logs = []
    monkeypatch.setattr(CC, "_release_card", lambda log: order.append("release"))
    monkeypatch.setattr(CC.subprocess, "Popen", FakePopen)
    monkeypatch.setattr("precision.epoch0_cloud.clean_cmd", lambda cfg, tmp, out: ["true"])
    CC.clean({}, tmp_path, cleaned, log=logs.append)
    assert order == ["release", "cleaner"]
    assert any("[Step 3/6]" in m for m in logs), "every cleaner line reaches the log, not only the ✅ ones"


def test_release_is_a_no_op_without_an_initialised_card():
    logs = []
    CC._release_card(logs.append)          # tests run on the CPU: no CUDA context was ever created
    assert logs == []
