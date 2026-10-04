"""USER 2026-09-30: plain CATEGORY prompts to SAM3; when a bare category confirms nothing
("column" found no masklet on pccr, "metal support column" did in 62 keyframes), SAM3 is
retried with the prompt's origins — the VLM's visual descriptions, then merged names."""
from types import SimpleNamespace
import json

from segmentation.autoprompt.scene_understanding import (FrameUnderstanding, SceneUnderstanding,
                                                        understand_frame)
from segmentation.autoprompt.session_builder import build_fallback_prompts


class _VLM:
    def __init__(self, answer):
        self.answer = answer

    def chat(self, msgs, max_tokens=0, consumer=""):
        return SimpleNamespace(content=json.dumps(self.answer))


def test_the_vlm_answer_carries_category_and_description():
    fu = understand_frame(_VLM({"scene_type": "server room", "summary": "",
                                "objects": [{"category": "Column", "description": "metal support column"},
                                            {"category": "desk", "description": "black metal desk"},
                                            "floor"]}), image=__import__("PIL.Image", fromlist=["x"]).new("RGB", (8, 8)), frame_id=3)
    assert fu.objects == ["column", "desk", "floor"]
    assert fu.descriptions == {"column": "metal support column", "desk": "black metal desk"}


def test_fallbacks_are_the_descriptions_then_the_merged_names():
    frames = [FrameUnderstanding(1, "room", "", ["column", "desk"],
                                 descriptions={"column": "metal support column", "desk": "black metal desk"}),
              FrameUnderstanding(2, "room", "", ["column", "beam", "table"],
                                 descriptions={"column": "metal support column", "beam": "steel beam"}),
              FrameUnderstanding(3, "room", "", ["columns"],
                                 descriptions={"columns": "grey concrete column"})]
    und = SceneUnderstanding("room", "", ["column", "desk"], per_frame=frames,
                             merged={"columns": "column"})
    fb = build_fallback_prompts(und, ["column", "desk"], {"beam": "column", "table": "desk"}, n_max=3)
    # most frequent description first, then the others, then merged names; capped; every
    # description carries the category it describes (USER 2026-10-01)
    assert fb["column"] == ["metal support column", "steel beam column", "grey concrete column"]
    assert fb["desk"] == ["black metal desk", "table"]


def test_the_retry_phrase_carries_the_category():
    """pccr 2026-10-01: a 'window' fallback 'dark, rectangular, closed' got a door's masklet —
    bare adjectives name nothing; the retry says what kind of object they describe."""
    from segmentation.autoprompt.session_builder import with_category
    assert with_category("dark, rectangular, closed", "window") == "dark rectangular closed window"
    assert with_category("white tiled", "floor") == "white tiled floor"
    assert with_category("metal support column", "column") == "metal support column"
    assert with_category("Fire Extinguisher, red", "fire extinguisher") == "Fire Extinguisher red"
    frames = [FrameUnderstanding(1, "room", "", ["window"], descriptions={"window": "dark, rectangular, closed"}),
              FrameUnderstanding(2, "room", "", ["window"], descriptions={"window": "window"})]
    und = SceneUnderstanding("room", "", ["window"], per_frame=frames, merged={})
    assert build_fallback_prompts(und, ["window"], {}, n_max=3) == {"window": ["dark rectangular closed window"]}


def test_production_config_has_the_bound():
    import yaml, pathlib
    raw = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert int(raw["autoprompt"]["sam3_fallback_max"]) >= 1
