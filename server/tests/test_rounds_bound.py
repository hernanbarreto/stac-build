"""A segmentation round whose SAM3 prompt bound is already full is not run (USER 2026-11-11: "si ya
se alcanzó el total en la primera ronda ya no entre en la ronda 2 y nos ahorramos ese paso").

pccr 2026-08-24 reached autoprompt.max_sam3_prompts (150) in round 1; its round 2 still started the
VLM and showed it 2765 keyframes, though with no room left every name found could only be recorded
"not prompted". The round's answer is known before it starts: no new prompt, the rounds end."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.autoprompt import rounds as R  # noqa: E402


def _out(tmp_path, n_round1, n_store_extra=0):
    out = tmp_path / "output"
    out.mkdir(parents=True)
    first = [f"thing {i}" for i in range(n_round1)]
    (out / "vlm_analysis.json").write_text(json.dumps({"prompt": ";".join(first)}))
    (out / "segmentation.json").write_text(json.dumps(
        {"prompts": first + [f"extra {i}" for i in range(n_store_extra)]}))
    return out


def _cfg(bound):
    return {"autoprompt": {"max_sam3_prompts": bound}}


def test_a_full_bound_records_the_round_with_no_new_prompt(tmp_path):
    out = _out(tmp_path, 150)
    rec = R.bound_full_round(out, _cfg(150), 2)
    assert rec is not None and rec["new_prompts"] == [] and rec["n_prompts_before"] == 150
    assert "already reached" in rec["skipped"]
    assert R.round_prompts(out, 2) == [], "the manager's loop reads no new prompt: the rounds end"
    assert R.load_rounds(out)["rounds"][0]["round"] == 2


def test_the_count_is_every_prompt_sam3_was_given(tmp_path):
    out = _out(tmp_path, 148, n_store_extra=2)            # 148 + 2 earlier-round prompts = 150
    assert R.bound_full_round(out, _cfg(150), 3) is not None


def test_with_room_the_round_runs(tmp_path):
    out = _out(tmp_path, 149)
    assert R.bound_full_round(out, _cfg(150), 2) is None
    assert not (out / R.ROUNDS_FILE).exists()


class _Pipe:
    def __init__(self):
        self.logs, self.progress = [], []

    def send_log(self, msg, level="info"):
        self.logs.append(msg)

    def send_progress(self, pct, msg, stage=None):
        self.progress.append((pct, msg))

    def check_cancel(self):
        return False


def test_the_worker_starts_no_engine_for_a_full_bound(tmp_path, monkeypatch):
    import semantic.service as S
    from workers import vlm_worker

    def _no_engine(*a, **k):
        raise AssertionError("a round with a full bound must not start the VLM engine")

    monkeypatch.setattr(S, "job_engine", _no_engine)
    monkeypatch.setattr(R, "run_round", _no_engine)
    _out(tmp_path, 150)
    pipe = _Pipe()
    vlm_worker._vlm_round(pipe, tmp_path, _cfg(150), 2)
    assert any("round 2 not run" in m for m in pipe.logs)
    assert pipe.progress[-1][0] == 100
    assert R.round_prompts(tmp_path / "output", 2) == []


def test_the_bound_is_read_strictly(tmp_path):
    out = _out(tmp_path, 3)
    with pytest.raises(Exception, match="autoprompt.max_sam3_prompts"):
        R.bound_full_round(out, {"autoprompt": {}}, 2)
