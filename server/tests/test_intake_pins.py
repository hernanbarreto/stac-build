"""The CPU libraries whose numerics reach the intake readings are PINNED in the repo's
environment files to the versions of env da3 (docs/plan_determinismo.md 74 / 79): numpy, scipy,
OpenCV and Pillow (each bringing its own libjpeg-turbo). The pins must equal what is installed
here — the stack every validated run used — and the intake stamp records the same versions."""

import re
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import stamps as St                                 # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PINNED = {"numpy": "numpy", "scipy": "scipy", "opencv-python": "opencv-python",
          "pillow": "Pillow"}                        # dist name → the name used in the files


def _installed():
    return {dist: metadata.version(dist) for dist in PINNED}


def _pins(text, sep):
    out = {}
    for ln in text.splitlines():
        s = ln.strip().lstrip("- ").split("#", 1)[0].strip()
        m = re.match(rf"^([A-Za-z0-9_.-]+){re.escape(sep)}([0-9][A-Za-z0-9.]*)$", s)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def test_requirements_pin_the_installed_versions():
    pins = _pins((REPO / "requirements.txt").read_text(), "==")
    inst = _installed()
    for dist, name in PINNED.items():
        assert pins.get(name.lower()) == inst[dist], (dist, pins.get(name.lower()), inst[dist])


def test_environment_yml_pins_the_installed_versions():
    pins = _pins((REPO / "environment.yml").read_text(), "==")
    inst = _installed()
    for dist, name in PINNED.items():
        assert pins.get(name.lower()) == inst[dist], (dist, pins.get(name.lower()), inst[dist])


def test_the_intake_stamp_records_the_pinned_versions_and_both_libjpeg_builds():
    rec = St.cpu_environment_record()
    inst = _installed()
    assert rec["libs"]["numpy"] == inst["numpy"] and rec["libs"]["scipy"] == inst["scipy"]
    assert rec["libs"]["opencv-python"] == inst["opencv-python"]
    assert rec["libs"]["pillow"] == inst["pillow"]
    jd = rec["jpeg_decoders"]
    assert "libjpeg" in jd["opencv"]["libjpeg"].lower()          # OpenCV's bundled decoder
    assert jd["pillow"]["jpg_codec"]                              # Pillow's decoder version
    # the DA3 extractor records Pillow and OpenCV too (its identity, point 79)
    src = (REPO / "server" / "extract_da3_depth.py").read_text()
    m = re.search(r"IDENTITY_LIBS\s*=\s*\((.*?)\)", src, re.S)
    assert m and "pillow" in m.group(1) and "opencv-python" in m.group(1)
