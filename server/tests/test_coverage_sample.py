"""Coverage is geometry: a pose, its intrinsics and the cloud — never a mask.

The bug this guards (pccr 2026-09-17): `cover_keyframes` asked
`surface_fit.hole_audit._evidence`, whose `__init__` returns on its second line
when `output/seg_masks.npz` is missing — the SAM3 masks. The sampler runs in the
VLM stage, BEFORE SAM3, so that file cannot exist yet: the fallback to eight
evenly spaced keyframes fired on EVERY run the module ever had, and the log
blamed the cameras ("no camera evidence") while camera_poses.txt and
intrinsic.txt had been on disk for an hour.

What it costs: in the SIMPLE pipeline those frames are the only thing the VLM
looks at, and its phrases become the SAM3 prompts one for one. What is not in
them is never named, never segmented, never measured.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PATH = (Path(__file__).resolve().parents[1]
         / "segmentation" / "autoprompt" / "coverage_sample.py")
_SRC = _PATH.read_text()
# CODE only: the comments explain the bug and name the very things the code
# must not do, so scanning the raw text would flag the explanation itself
_CODE = "\n".join(l.split("#", 1)[0] for l in _SRC.splitlines())


class TestNoMaskDependency:
    def test_it_does_not_ask_hole_audit_for_evidence(self):
        assert "_evidence" not in _CODE, \
            "cover_keyframes asks for the SAM3 masks again — they cannot exist in its stage"

    def test_it_loads_the_cameras_directly(self):
        assert "_load_camera_source" in _CODE
        assert "_k_grid" in _CODE          # the pixel grid comes from the cloud

    def test_the_message_names_the_file_that_is_missing(self):
        """"no camera evidence" sent a reader hunting for poses that were
        there. Each refusal now says which artifact is absent."""
        assert "no camera evidence" not in _CODE
        assert "camera_poses.txt/intrinsic.txt" in _CODE
        assert "no cloud on disk" in _CODE


class TestRefusalsAreDeclared:
    def test_every_early_return_logs_first(self):
        """A None return is the caller's cue to fall back and SAY so; a silent
        one would look like a deliberate 8-frame choice."""
        body = _CODE[_CODE.index("def cover_keyframes"):]
        head = body[:body.index("rng = np.random.default_rng")]
        returns = [i for i, l in enumerate(head.splitlines())
                   if l.strip() == "return None"]
        lines = head.splitlines()
        for i in returns:
            assert any("log(" in lines[j] for j in range(max(0, i - 4), i)), \
                f"a `return None` at line {i} of cover_keyframes says nothing"
