"""Two things a run of 2026-09-21 did twice, and one it did in silence.

FINDING 26 — the mask→cloud matching ran TWICE on every run. CloudCompy moved
ahead of the semantic stages, so `run_segmentation` step 4 now always finds
cleaned_cloud.ply and matches; the SAM3 worker then called
`map_segmentation_to_cloud`, which matched the same three files again:

    12:18  Matching masks against cleaned_cloud (22,771,938 points)…
    12:26  75 instances, 20,803,417/22,771,938 pts (91.4 %); fusion round 1
    12:26  Matching masks against cleaned_cloud (22,771,938 points)…
    12:32  75 instances, 20,803,704/22,771,938 pts (91.4 %); fusion round 2

Eight minutes and one extra 22 M-point octree for 287 points of difference.
What is pinned here is the CONTRACT the reuse rests on, the one
`segmentation/fuse_parent` documents: the fusion writes the parent first and
the caller writes `segmentation_result.json` LAST, so a result newer than the
parent covers that parent — and a result that is missing, older or unreadable
is matched, never trusted.

FINDING 22 — the detail of the segmentation was decided by a pass that could
fail in silence. Two runs of the SAME scan:

    04:57  61 object types understood; "the pass returned nothing parseable"
           61 → 61 concept(s) in 0 pass(es)          → 61 categories to SAM3
    11:34  61 object types understood; pass 1: 61 → 33 ; pass 2: 33 → 32
           61 → 32 in 3 pass(es)                     → 32 categories to SAM3

The VLM was stable; the vocabulary of the whole session was not, and nothing
on disk said which of the two had happened. What is pinned here is that the
outcome is now RECORDED (raw per-frame names, every pass, what was folded into
what, the parse_failed flag) and DECLARED (a warning on the consolidation and
next to `prompts` in the parent).

Since 2026-09-21 the pass only APPLIES a literal restatement of one name
(`consolidate_prompts.same_name`) and a PART keeps its own prompt, so the
fixtures here are built out of restatements — a plural, a collective suffix —
whenever a list has to shrink across passes. The record grew a map with it:
`folded` answers "this phrase is not a prompt any more, where did it go?" and
so may only hold concepts that really went, while a surviving part is declared
in `part_of`.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import segmentation.pipeline as sp  # noqa: E402
from segmentation import mask_space  # noqa: E402
from segmentation.autoprompt.consolidate_prompts import consolidate  # noqa: E402
from segmentation.autoprompt.scene_understanding import (  # noqa: E402
    FrameUnderstanding, SceneUnderstanding)
from segmentation.autoprompt.session_builder import AutoPrompter  # noqa: E402

PCCR = ["white tiled floor", "checkered floor tiles", "black server rack",
        "server rack fan", "dark wooden door"]

# The same list, rewritten so the consolidation can still move it: 'server
# racks' restates 'server rack' (a plural) and is APPLIED; 'server rack fan' is
# a PART and since 2026-09-21 keeps its own prompt. 'checkered floor tiles' was
# neither — head 'tile' against head 'floor' — so it is now a refusal and a
# list built on it never shrinks at all.
RESTATED = ["white tiled floor", "server rack", "server racks",
            "server rack fan", "dark wooden door"]


def _quiet(_m):
    pass


# ═════════════════════════════════════════════════════════════════════
#  FINDING 26 — one matching per set of inputs
# ═════════════════════════════════════════════════════════════════════

def _session(tmp_path: Path, *, result=True, parent_t=1000.0, masks_t=1000.0,
             cloud_t=900.0, result_t=2000.0, result_body=None) -> Path:
    """A session directory whose four files have DECIDED mtimes.

    Only the timestamps matter to the freshness test, so the payloads are the
    smallest thing that still reads back as what it claims to be.
    """
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "segmentation.json").write_text(json.dumps(
        {"version": "3.0", "prompts": ["black server rack"],
         "instances": [{"id": 0, "instance_id": 1, "label": "black_server_rack"}]}))
    (out / "seg_masks.npz").write_bytes(b"not really an npz, only an mtime")
    (out / "cleaned_cloud.ply").write_bytes(b"ply\n")
    os.utime(out / "segmentation.json", (parent_t, parent_t))
    os.utime(out / "seg_masks.npz", (masks_t, masks_t))
    os.utime(out / "cleaned_cloud.ply", (cloud_t, cloud_t))
    if result:
        body = result_body if result_body is not None else json.dumps(
            {"type": "segmentation", "instances": [{"id": 0, "total_points": 12}],
             "coverage": 0.914})
        (out / "segmentation_result.json").write_text(body)
        os.utime(out / "segmentation_result.json", (result_t, result_t))
    return out


@pytest.fixture()
def counted(monkeypatch):
    """Counts the matchings `map_segmentation_to_cloud` actually launches."""
    calls = []

    def _fake(output_dir, ply_path=None, new_obj_ids=None):
        calls.append(Path(output_dir))
        return {"type": "segmentation", "instances": [{"id": 99}], "coverage": 0.5}

    monkeypatch.setattr(sp, "_match_and_save_result", _fake)
    return calls


def test_a_result_newer_than_the_parent_is_reused(tmp_path, counted):
    """The fuse_parent write order IS the freshness test: parent first, result
    last. The second call of a normal run must cost nothing."""
    out = _session(tmp_path)
    res = sp.map_segmentation_to_cloud(out)
    assert counted == [], "the cloud was matched a second time"
    assert res["coverage"] == 0.914
    assert res["instances"][0]["id"] == 0, "it must be the CACHED result"


def test_a_result_older_than_the_parent_is_rematched(tmp_path, counted):
    """A fusion round rewrote segmentation.json after the result was written:
    the result describes masks that no longer exist as such."""
    out = _session(tmp_path, parent_t=3000.0)
    res = sp.map_segmentation_to_cloud(out)
    assert counted == [out]
    assert res["instances"][0]["id"] == 99


def test_a_result_older_than_the_masks_is_rematched(tmp_path, counted):
    out = _session(tmp_path, masks_t=3000.0)
    sp.map_segmentation_to_cloud(out)
    assert counted == [out]


def test_a_result_older_than_the_cloud_is_rematched(tmp_path, counted):
    """A re-reconstruction or a correction epoch republishes cleaned_cloud.ply;
    the global indices of the old result point into a cloud that is gone."""
    out = _session(tmp_path, cloud_t=3000.0)
    sp.map_segmentation_to_cloud(out)
    assert counted == [out]


def test_a_missing_result_is_matched(tmp_path, counted):
    out = _session(tmp_path, result=False)
    sp.map_segmentation_to_cloud(out)
    assert counted == [out]


def test_an_unreadable_result_is_matched(tmp_path, counted):
    """Fresh by mtime, truncated on disk — the reuse must not hand a caller
    half a JSON document because the timestamps looked right."""
    out = _session(tmp_path, result_body='{"instances": [{"id": 0}')
    res = sp.map_segmentation_to_cloud(out)
    assert counted == [out]
    assert res["instances"][0]["id"] == 99


def test_a_session_without_a_cloud_still_refuses_with_its_reason(tmp_path, counted):
    out = tmp_path / "output"
    out.mkdir(parents=True)
    (out / "segmentation.json").write_text("{}")
    res = sp.map_segmentation_to_cloud(out)
    assert res["error"] == "no cleaned_cloud.ply"
    assert counted == []


def test_the_worker_still_writes_the_broadcast():
    """The viewer loses nothing: seg_broadcast.json is written from whatever
    map_segmentation_to_cloud returns, cached or freshly matched."""
    src = (Path(__file__).resolve().parents[1] / "workers" / "sam3_worker.py").read_text()
    i = src.index("seg_data = map_segmentation_to_cloud(output_dir)")
    j = src.index('seg_broadcast_path.write_text(json.dumps(seg_data))')
    assert j > i
    assert 'output_dir / "seg_broadcast.json"' in src


# ═════════════════════════════════════════════════════════════════════
#  FINDING 22 — the vocabulary says what happened to it
# ═════════════════════════════════════════════════════════════════════

class _Reply:
    def __init__(self, content, finish_reason=None):
        self.content = content
        self.finish_reason = finish_reason


class _Client:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        r = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        if isinstance(r, _Reply):
            return r
        return _Reply(r if isinstance(r, str) else json.dumps(r))


def test_an_unparseable_pass_is_recorded_and_declared():
    """04:57: the session segmented with the RAW 61-phrase list and the only
    trace was a log line. The list is still left alone — that part is right —
    but the outcome is now a record and a sentence."""
    res = consolidate(_Client("I think the floor and the walls are the same."),
                      "server room", PCCR, log=_quiet)
    assert res.objects == PCCR, "the list must still be left untouched"
    assert res.passes == 0
    assert res.parse_failed is True
    assert len(res.history) == 1
    note = res.history[0]
    assert note["status"] == "unparsed"
    assert note["before"] == len(PCCR) and note["after"] == len(PCCR)
    assert "no JSON object" in note["reason"]
    assert str(len("I think the floor and the walls are the same.")) in note["reason"]
    warn = res.warning()
    assert warn and "RAW per-frame list" in warn and str(len(PCCR)) in warn
    d = res.to_dict()
    assert d["parse_failed"] is True and d["warning"] == warn
    assert d["input_objects"] == PCCR and d["history"] == res.history


def test_a_truncated_answer_says_it_was_cut_at_max_tokens():
    """finish_reason 'length' is a different defect from a confused model: the
    concept list grows with the scene and max_tokens does not."""
    res = consolidate(_Client(_Reply('{"objects": [{"name": "white tiled f',
                                     finish_reason="length")),
                      "server room", PCCR, max_tokens=4096, log=_quiet)
    assert res.parse_failed is True
    assert "max_tokens=4096" in res.history[0]["reason"]
    assert res.history[0]["finish_reason"] == "length"


def test_an_empty_response_is_named_as_such():
    res = consolidate(_Client(""), "server room", PCCR, log=_quiet)
    assert "empty response" in res.history[0]["reason"]
    assert res.history[0]["chars"] == 0


def test_a_failing_client_is_recorded_too():
    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("semantic service down")

    res = consolidate(_Boom(), "server room", PCCR, log=_quiet)
    assert res.objects == PCCR and res.passes == 0
    assert res.history[0]["status"] == "failed"
    assert "semantic service down" in res.history[0]["reason"]
    assert "RAW per-frame list" in res.warning()


def test_a_converged_run_carries_no_warning_and_a_full_history():
    """11:34: the good run. Every pass is written down with what it did, so
    the 61 → 32 of a session can be read back a month later.

    The shrink is narrower than it was that day: one literal restatement is
    APPLIED ('server racks' → 'server rack') and the part is RECORDED and kept
    ('server rack fan' is still a prompt), so 5 → 4 and then convergence. What
    the history has to carry is unchanged — before, after, what was merged,
    what was named as a part — and it is the same whether the pass applied the
    proposal or refused it.
    """
    first = {"objects": [
        {"name": "server rack", "aliases": ["server racks"],
         "parts": ["server rack fan"]},
        {"name": "white tiled floor", "aliases": [], "parts": []},
        {"name": "dark wooden door", "aliases": [], "parts": []}]}
    second = {"objects": [{"name": p, "aliases": [], "parts": []}
                          for p in ["server rack", "server rack fan",
                                    "white tiled floor", "dark wooden door"]]}
    res = consolidate(_Client(first, second), "server room", RESTATED,
                      max_passes=3, log=_quiet)
    assert res.objects == ["server rack", "server rack fan", "white tiled floor",
                           "dark wooden door"]
    assert res.parse_failed is False and res.max_passes_reached is False
    assert res.warning() is None
    assert [h["status"] for h in res.history] == ["applied", "converged"]
    assert res.history[0]["before"] == 5 and res.history[0]["after"] == 4
    assert res.history[0]["merged"] == {"server rack": ["server racks"]}
    assert res.history[0]["parts"] == {"server rack": ["server rack fan"]}
    assert res.input_objects == RESTATED


def test_reaching_the_bound_with_a_moving_list_is_declared():
    """`consolidate_passes` is a BOUND, not a decision — but stopping on it
    means the list was still changing, which is not convergence.

    A pass only CHANGES the list when it applies something, so two restatements
    the model spots one pass at a time are what it takes: the plural in pass 1,
    the collective suffix in pass 2, and the bound falls while the list is
    still moving.
    """
    listing = ["server rack", "server racks", "exposed ceiling ducts",
               "exposed ceiling ductwork", "red fire extinguisher",
               "computer monitor"]
    shrink = [{"objects": [{"name": "server rack",
                            "aliases": ["server racks"], "parts": []},
                           {"name": "exposed ceiling ducts", "aliases": [],
                            "parts": []},
                           {"name": "exposed ceiling ductwork", "aliases": [],
                            "parts": []},
                           {"name": "red fire extinguisher", "aliases": [], "parts": []},
                           {"name": "computer monitor", "aliases": [], "parts": []}]},
              {"objects": [{"name": "exposed ceiling ducts",
                            "aliases": ["exposed ceiling ductwork"], "parts": []},
                           {"name": "server rack", "aliases": [], "parts": []},
                           {"name": "red fire extinguisher", "aliases": [], "parts": []},
                           {"name": "computer monitor", "aliases": [], "parts": []}]}]
    res = consolidate(_Client(*shrink), "server room", listing, max_passes=2,
                      log=_quiet)
    assert [h["status"] for h in res.history] == ["applied", "applied"]
    assert [(h["before"], h["after"]) for h in res.history] == [(6, 5), (5, 4)]
    assert res.max_passes_reached is True
    assert "still changing" in res.warning()


# ── the record on disk ───────────────────────────────────────────────────

def _understanding():
    return SceneUnderstanding(
        scene_type="server room", summary="a server room",
        objects=list(PCCR),
        per_frame=[FrameUnderstanding(0, "server room", "",
                                      ["white tiled floor", "black server rack"]),
                   FrameUnderstanding(60, "server room", "",
                                      ["checkered floor tiles", "server rack fan",
                                       "dark wooden door"])])


def _understanding_restated():
    """`_understanding` over RESTATED — the list the pass can still fold
    something out of, so the record has both a real fold and a surviving
    part to declare."""
    return SceneUnderstanding(
        scene_type="server room", summary="a server room",
        objects=list(RESTATED),
        per_frame=[FrameUnderstanding(0, "server room", "",
                                      ["white tiled floor", "server rack"]),
                   FrameUnderstanding(60, "server room", "",
                                      ["server racks", "server rack fan",
                                       "dark wooden door"])])


class _Prompter:
    """Just enough AutoPrompter to exercise the record writer — the VLM, the
    poses and the keyframes play no part in it."""

    def __init__(self, out):
        self.output_dir = Path(out)

    _write_concepts_record = AutoPrompter._write_concepts_record


def test_the_record_carries_the_raw_names_the_passes_and_the_folds(tmp_path):
    """The record answers "where did this phrase go?", so it may only claim a
    phrase went somewhere when it did.

    Until 2026-09-21 a PART was removed, and `folded` — whose question is
    literally "this phrase is not a prompt any more, where did it go?" — listed
    it with the aliases. A part now keeps its own prompt ("segment
    EVERYTHING"), so listing it there would say the opposite of what happened
    to it. The relationship still has to travel, because it is provenance the
    VLM proposed: it is declared in `part_of`, which says "still a prompt, and
    known to be part of that one". `folded` keeps only the concepts that are
    genuinely gone.
    """
    answer = {"objects": [
        {"name": "server rack", "aliases": ["server racks"],
         "parts": ["server rack fan"]},
        {"name": "white tiled floor", "aliases": [], "parts": []},
        {"name": "dark wooden door", "aliases": [], "parts": []}]}
    cons = consolidate(_Client(answer), "server room", RESTATED, log=_quiet)
    rec = _Prompter(tmp_path)._write_concepts_record(
        _understanding_restated(), RESTATED, cons.objects, cons, "")
    on_disk = json.loads((tmp_path / "autoprompt_concepts.json").read_text())
    assert on_disk == rec
    assert on_disk["origin"] == "vlm_proposed"          # the provenance rule
    assert on_disk["raw"]["objects"] == RESTATED
    assert [f["frame_id"] for f in on_disk["raw"]["per_frame"]] == [0, 60]
    assert on_disk["raw"]["per_frame"][1]["objects"] == [
        "server racks", "server rack fan", "dark wooden door"]
    assert on_disk["consolidated"]["objects"] == cons.objects
    assert on_disk["consolidation"]["history"][0]["status"] == "applied"
    # "what was folded into what", asked the way a reader asks it — and only
    # the alias really left the list
    assert on_disk["folded"] == {
        "server racks": {"into": "server rack", "as": "alias"}}
    assert "server racks" not in on_disk["consolidated"]["objects"]
    # the part survived, so it is a RELATIONSHIP and not a fold
    assert on_disk["part_of"] == {"server rack fan": "server rack"}
    assert "server rack fan" in on_disk["consolidated"]["objects"]
    assert not set(on_disk["folded"]) & set(on_disk["part_of"])
    assert on_disk["consolidation"]["parts"] == {"server rack": ["server rack fan"]}
    assert on_disk["consolidation"]["parts_kept_as_objects"] is True
    assert on_disk["parse_failed"] is False and on_disk["warning"] is None
    assert on_disk["prompt"] == ";".join(cons.objects)


def test_the_record_exists_even_when_the_pass_did_not_parse(tmp_path):
    """The run nobody could reconstruct: prompt_consolidation.json is only
    written when the pass RUNS, so the failure left no file at all."""
    cons = consolidate(_Client("sorry, I cannot"), "server room", PCCR, log=_quiet)
    rec = _Prompter(tmp_path)._write_concepts_record(
        _understanding(), PCCR, cons.objects, cons, "")
    assert (tmp_path / "autoprompt_concepts.json").exists()
    assert rec["parse_failed"] is True
    assert "RAW per-frame list" in rec["warning"]
    assert rec["consolidated"]["objects"] == PCCR


def test_the_record_exists_when_the_pass_is_switched_off(tmp_path):
    rec = _Prompter(tmp_path)._write_concepts_record(
        _understanding(), PCCR, PCCR, None, "autoprompt.consolidate_prompts is off")
    assert rec["consolidation"]["ran"] is False
    assert rec["consolidation"]["reason"] == "autoprompt.consolidate_prompts is off"
    assert rec["parse_failed"] is False and rec["warning"] is None


def test_the_simple_branch_writes_the_record_before_the_contract():
    """Source check: the record must be built from the list SAM3 is handed,
    and vlm_analysis.json must carry the same summary the worker warns on."""
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "autoprompt"
           / "session_builder.py").read_text()
    i = src.index("concepts = self._write_concepts_record(")
    assert src.index('prompt = ";".join(phrases)') > i
    assert '"consolidation": concepts["consolidation"],' in src


def test_the_worker_warns_when_the_vocabulary_is_degraded():
    src = (Path(__file__).resolve().parents[1] / "workers" / "sam3_worker.py").read_text()
    assert 'vlm_data.get("consolidation")' in src
    assert 'level="warning"' in src[src.index('_cons = vlm_data.get("consolidation")'):]


# ── the parent carries it, next to `prompts` ─────────────────────────────

_CFG = {"visualization": {"segment_colors": ["#ff0000", "#00ff00"]}}


def _save_one_mask(out: Path):
    sp._save_masks(out, {0: {1: np.ones((4, 4), dtype=np.uint8)}},
                   ["black server rack"], {1: "black server rack"}, _CFG,
                   frame_space=mask_space.SPACE_KEYFRAME)
    return json.loads((out / "segmentation.json").read_text())


def test_the_parent_carries_the_consolidation_next_to_the_prompts(tmp_path):
    """`prompts: [61 strings]` and not one word about why there were 61."""
    out = tmp_path / "output"
    out.mkdir()
    cons = consolidate(_Client("not json at all"), "server room", PCCR, log=_quiet)
    _Prompter(out)._write_concepts_record(_understanding(), PCCR, cons.objects,
                                          cons, "")
    meta = _save_one_mask(out)
    keys = list(meta)
    assert keys.index("consolidation") == keys.index("prompts") + 1
    summary = meta["consolidation"]
    assert summary["origin"] == "vlm_proposed"
    assert summary["parse_failed"] is True
    assert summary["n_raw"] == len(PCCR) and summary["n_objects"] == len(PCCR)
    assert summary["passes"] == 0
    assert "RAW per-frame list" in summary["warning"]
    assert summary["record"] == "autoprompt_concepts.json"


def test_a_session_with_no_autoprompter_record_gets_no_invented_verdict(tmp_path):
    """Manual prompt, or a session older than the record: silence, not a
    fabricated 'it all went fine'."""
    out = tmp_path / "output"
    out.mkdir()
    meta = _save_one_mask(out)
    assert meta["consolidation"] is None


def test_an_incremental_save_does_not_lose_the_summary(tmp_path):
    """`_save_masks` runs once per concept and rebuilds the parent each time;
    the summary is written by the auto-prompter, long before."""
    out = tmp_path / "output"
    out.mkdir()
    cons = consolidate(_Client("not json at all"), "server room", PCCR, log=_quiet)
    _Prompter(out)._write_concepts_record(_understanding(), PCCR, cons.objects,
                                          cons, "")
    first = _save_one_mask(out)
    (out / "autoprompt_concepts.json").unlink()      # e.g. a re-run from masks
    second = _save_one_mask(out)
    assert second["consolidation"] == first["consolidation"]
