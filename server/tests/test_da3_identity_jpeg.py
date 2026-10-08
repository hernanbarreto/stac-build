"""docs/plan_determinismo.md point 79: the DA3 identity (the stamp of every I3 window and of the
focal probe) records BOTH JPEG decoders of the pipeline with their libjpeg-turbo builds — DA3
reads the frames with Pillow, I0 / I1 / F4 with OpenCV. Runs once point79.patch is applied to
server/extract_da3_depth.py; until then it is skipped, saying so."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import stamps as St                                 # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
SRC = (SERVER / "extract_da3_depth.py").read_text()
pytestmark = pytest.mark.skipif("jpeg_decoders" not in SRC,
                                reason="point79.patch not applied yet (scratchpad/point79.patch)")


def test_the_da3_identity_records_both_jpeg_decoders():
    body = SRC[SRC.index("def da3_identity"):SRC.index("def _load_model")]
    assert re.search(r'"jpeg_decoders":\s*intake_stamps\.jpeg_decoder_record\(\)', body)
    assert "from intake import stamps as intake_stamps" in SRC
    rec = St.jpeg_decoder_record()
    assert rec["pillow"]["jpg_codec"] and rec["pillow"]["version"]
    assert "libjpeg" in rec["opencv"]["libjpeg"].lower() and rec["opencv"]["version"]
