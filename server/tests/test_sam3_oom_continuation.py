"""SAM3 never fails on GPU memory (USER 2026-10-09: "no debe nunca fallar por colapso del GPU",
replacing docs/plan_determinismo.md points 90 / 92's "an out-of-memory fails the stage").

pccr's Autosegment died at prompt 42/45 'tile': one SAM3 session of 250 frames grew 8 → 77.7 GB
with every floor tile an object, and torch.OutOfMemoryError failed the stage. Now an
out-of-memory is a STOP like the vendor tracker's (2026-10-08): the wrapper releases the
session (memory verified back) and hands over the leading frames it completed; the batch loop
keeps them and continues in a fresh session from (stop − overlap), linked by the overlap rule,
declared in the census as ``oom_continuation``. A continuation with no progress shortens its
overlap (down to 0: linked by nothing); only a session that cannot complete ONE frame on the
card fails. Any other error still fails. CPU only — the SAM3 predictor and wrapper are fakes,
the batch loop and the wrapper's process_batch are the real ones."""

import contextlib
import gc
import json
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import yaml

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

CONFIG = yaml.safe_load((SERVER / "config.yaml").read_text())
CARD = {"at_stop": {"free_mib": 20, "total_mib": 81153, "allocated_mib": 79100, "reserved_mib": 80300},
        "after_release": {"free_mib": 72500, "total_mib": 81153, "allocated_mib": 6900,
                          "reserved_mib": 7400}}


# ── the batch loop (segmentation.pipeline._run_sam3_batched) ─────────────────────────

class _SessionFakeSAM3:
    """``process_batch`` as the batch loop sees it — every call is a FRESH SAM3 session whose
    local ids restart (numbered by first appearance, offset per call so a link can only come
    from the overlap rule, never from a shared number). ``scene[frame] = {name: lit pixel}``.
    ``actions`` are consumed one per call: None = the batch completes; ("oom", k) = CUDA out
    of memory after k frames; ("prompt_oom",) = out of memory in add_prompt; ("error", msg) =
    any other SAM3 error; ``capacity`` = every session runs out of memory once it has
    propagated that many frames (a card that is always too small)."""

    def __init__(self, scene, actions=(), capacity=None):
        self.scene, self.actions, self.capacity = scene, list(actions), capacity
        self.calls = []

    def _outputs(self, frames, call):
        ids, out = {}, {}
        for fr in frames:
            objs = self.scene.get(fr) or {}
            if not objs:
                continue
            names = sorted(objs)
            for n in names:
                ids.setdefault(n, 100 * call + len(ids))
            masks = np.zeros((len(names), 4, 4), np.uint8)
            for k, n in enumerate(names):
                masks[k].flat[objs[n]] = 1
            out[fr] = {"out_binary_masks": masks, "out_obj_ids": np.array([ids[n] for n in names])}
        return out

    def process_batch(self, batch_dir, category, index_mapping, boxes_by_local=None):
        from segmentation import sam3_wrapper as w
        frames = [index_mapping[i] for i in sorted(index_mapping)]
        call = len(self.calls)
        self.calls.append((frames[0], frames[-1] + 1))
        act = self.actions.pop(0) if self.actions else None
        if act is None and self.capacity is not None and len(frames) > self.capacity:
            act = ("oom", self.capacity)
        if act is None:
            return self._outputs(frames, call)
        if act[0] == "error":
            raise w.SAM3RunError(act[1])
        if act[0] == "prompt_oom":
            raise w.SAM3PropagationStopped(
                f"SAM3 add_prompt for prompt '{category}' ran out of CUDA memory after 0 frame(s)",
                partial={}, frames_done=0, cause=w.STOP_OUT_OF_MEMORY, phase="add_prompt",
                memory=CARD, prompt_frame=0)
        k = act[1]
        raise w.SAM3PropagationStopped(
            f"SAM3 propagate_in_video for prompt '{category}' ran out of CUDA memory after {k} "
            f"of {len(frames)} frame(s)",
            partial=self._outputs(frames[:k], call), frames_done=k, cause=w.STOP_OUT_OF_MEMORY,
            phase="propagate_in_video", memory=CARD, prompt_frame=0)

    def release_batch_session(self):
        pass

    def unload_model(self):
        pass

    def load_model(self):
        pass


def _run(tmp_path, monkeypatch, fake, n_frames, batch_size, overlap, category="tile"):
    import torch
    import segmentation.sam3_wrapper as w
    from segmentation import pipeline as P
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: fake)
    frames = tmp_path / "frames_valid"
    frames.mkdir()
    files = [f"{i:06d}.jpg" for i in range(n_frames)]
    for f in files:
        (frames / f).write_bytes(b"")
    out = tmp_path / "output"
    out.mkdir()
    status = {}
    try:
        P._run_sam3_batched(frames, files, [category], batch_size, overlap,
                            CONFIG["models"]["segmentation"]["iou_match_threshold"], 0.9,
                            output_dir=out, cfg=CONFIG, prompt_status=status)
    finally:
        P._clear_batch_dirs()
    return out, status


def _objects(out):
    """{store object id: {(frame, lit pixel), …}} read back from seg_masks.npz."""
    z = np.load(out / "seg_masks.npz")
    got = {}
    for k in z.files:
        if k.startswith("f") and "_o" in k:
            f, o = k[1:].split("_o")
            got.setdefault(int(o), set()).update((int(f), int(p)) for p in np.flatnonzero(z[k]))
    return got


def _oom_decisions(status, category="tile"):
    return [d for d in status[category].get("identity_decisions") or []
            if d.get("kind") == "oom_continuation"]


def _scene(n):
    # tile A (pixel 1) in every frame; tile B (pixel 2) in frames 3 … n − 4
    return {f: ({"a": 1, "b": 2} if 3 <= f < n - 3 else {"a": 1}) for f in range(n)}


def test_an_oom_midway_is_continued_and_the_objects_across_the_stop_stay_one(tmp_path, monkeypatch):
    fake = _SessionFakeSAM3(_scene(30), actions=[("oom", 18)])
    out, status = _run(tmp_path, monkeypatch, fake, 30, 30, 10)
    assert fake.calls == [(0, 30), (8, 30)], "the rest propagated again from stop − overlap"
    st = status["tile"]
    assert st["status"] == "ran" and "note" not in st, \
        "a continuation is not a recovery note: the SAM3 worker passes the stage"
    objs = _objects(out)
    assert sorted(tuple(sorted(v)) for v in objs.values()) == sorted([
        tuple((f, 1) for f in range(30)),                     # A: ONE object across the stop
        tuple((f, 2) for f in range(3, 27)),                  # B: one object too
    ]), "every frame has its masks and the overlap rule linked both objects"
    assert st["n_objects"] == 2 and st["n_frames"] == 30
    (d,) = _oom_decisions(status)
    assert d["frames_done"] == 18 and d["kept"] == [0, 18] and d["batch"] == [0, 30]
    assert d["continuation"] == [8, 30] and d["overlap"] == 10 and d["progress"] is True
    assert d["card_memory"] == {"total_mib": 81153, "free_mib_at_stop": 20,
                                "free_mib_after_release": 72500}
    # the census carries the decision (its prompt row copies the prompt's SAM3 record)
    from segmentation.census import CENSUS_NAME, build_census
    lines = []
    build_census(out, prompt="tile", vlm_doc=None, gap_kf=1, frames_dir=tmp_path / "frames",
                 prompt_status=status, log=lines.append)
    doc = json.loads((out / CENSUS_NAME).read_text())
    (row,) = doc["prompts"]
    assert [x["kind"] for x in row["sam3_status"]["identity_decisions"]
            if x["kind"].endswith("continuation")] == ["oom_continuation"]
    assert doc["prompts_not_completed_by_sam3"] == [] and row["n_masklets"] == 2


def test_a_continuation_with_no_progress_shortens_its_overlap(tmp_path, monkeypatch):
    # the continuation from 8 stops after 6 frames — all inside the 10 already kept
    fake = _SessionFakeSAM3(_scene(30), actions=[("oom", 18), ("oom", 6)])
    out, status = _run(tmp_path, monkeypatch, fake, 30, 30, 10)
    assert fake.calls == [(0, 30), (8, 30), (13, 30)], "frontier 18 − overlap reduced to 5"
    first, second = _oom_decisions(status)
    assert second["progress"] is False and second["action"] == "reduce"
    assert second["leading_overlap"] == 10 and second["overlap"] == 5
    assert second["kept"] is None and second["continuation"] == [13, 30]
    objs = _objects(out)
    assert sorted(tuple(sorted(v)) for v in objs.values()) == sorted([
        tuple((f, 1) for f in range(30)), tuple((f, 2) for f in range(3, 27))]), \
        "five overlap frames still link (the identity judge's minimum)"


def test_the_overlap_goes_down_to_zero_and_that_seam_is_linked_by_nothing(tmp_path, monkeypatch):
    fake = _SessionFakeSAM3(_scene(30), actions=[("oom", 18), ("oom", 0)])
    out, status = _run(tmp_path, monkeypatch, fake, 30, 30, 10)
    assert fake.calls == [(0, 30), (8, 30), (18, 30)], "a fresh session from the stop itself"
    second = _oom_decisions(status)[1]
    assert second["overlap"] == 0 and "linked by nothing" in second["seam"]
    objs = _objects(out)
    frames_seen = {f for v in objs.values() for f, p in v if p == 1}
    assert frames_seen == set(range(30)), "every frame keeps its masks"
    a_parts = sorted(tuple(sorted(f for f, _ in v)) for v in objs.values()
                     if all(p == 1 for _, p in v))
    assert a_parts == [tuple(range(18)), tuple(range(18, 30))], \
        "the objects of that seam get fresh ids (declared), nothing is dropped"


def test_a_card_that_is_always_too_small_still_completes_the_category(tmp_path, monkeypatch):
    """The progress guarantee: every session runs out of memory after 4 frames, shorter than
    the 10-frame overlap — each continuation's overlap is kept under the measured capacity,
    so every step keeps at least one new frame."""
    fake = _SessionFakeSAM3(_scene(12), capacity=4)
    out, status = _run(tmp_path, monkeypatch, fake, 12, 12, 10)
    assert status["tile"]["status"] == "ran"
    assert {f for v in _objects(out).values() for f, p in v if p == 1} == set(range(12))
    ds = _oom_decisions(status)
    assert all(d["progress"] and d["overlap"] == 3 for d in ds)
    assert [d["kept"][1] for d in ds] == list(range(4, 12)), "one new frame per session"


def test_an_oom_before_any_frame_is_retried_once_in_a_fresh_session(tmp_path, monkeypatch):
    fake = _SessionFakeSAM3(_scene(10), actions=[("prompt_oom",)])
    out, status = _run(tmp_path, monkeypatch, fake, 10, 10, 2)
    assert fake.calls == [(0, 10), (0, 10)] and status["tile"]["status"] == "ran"
    (d,) = _oom_decisions(status)
    assert d["action"] == "retry_prompt" and d["phase"] == "add_prompt"


def test_a_session_that_cannot_complete_one_frame_is_the_one_failure(tmp_path, monkeypatch):
    import segmentation.sam3_wrapper as w
    fake = _SessionFakeSAM3(_scene(10), actions=[("oom", 0)])
    with pytest.raises(w.SAM3OutOfMemory, match=r"ONE frame of prompt 'tile'.*000000\.jpg"
                                                r".*total 81153 MiB.*physically impossible"):
        _run(tmp_path, monkeypatch, fake, 10, 10, 2)


def test_a_second_oom_before_any_frame_fails_naming_it(tmp_path, monkeypatch):
    import segmentation.sam3_wrapper as w
    fake = _SessionFakeSAM3(_scene(10), actions=[("prompt_oom",), ("prompt_oom",)])
    with pytest.raises(w.SAM3OutOfMemory, match="add_prompt ran out of CUDA memory twice"):
        _run(tmp_path, monkeypatch, fake, 10, 10, 2)
    assert len(fake.calls) == 2


def test_any_other_sam3_error_still_fails_the_run(tmp_path, monkeypatch):
    import segmentation.sam3_wrapper as w
    fake = _SessionFakeSAM3(_scene(10), actions=[("error", "SAM3 propagate_in_video failed for "
                                                           "prompt 'tile' (RuntimeError: boom)")])
    with pytest.raises(w.SAM3RunError, match="boom"):
        _run(tmp_path, monkeypatch, fake, 10, 10, 2)


def test_the_plan_always_progresses_or_names_the_physical_failure():
    """_plan_oom_continuation over every (overlap kept, frames done) a batch can stop with:
    either frames are kept past the frontier, or the overlap strictly shrinks, or it is the
    one physical failure (nothing done with no overlap left)."""
    from segmentation.pipeline import _plan_oom_continuation
    for lead in range(0, 11):
        for n_done in range(0, 31):
            frontier = None if lead == 0 else 100 + lead
            p = _plan_oom_continuation(100, 130, frontier, n_done, 10, "propagate_in_video", False)
            if p["action"] in ("continue", "complete"):
                assert p["kept"][1] > (frontier or 100)
                if p["action"] == "continue":
                    assert p["continuation"][0] > 100 and p["overlap"] < n_done
            elif p["action"] == "reduce":
                assert p["kept"] is None and p["overlap"] < lead
                assert p["continuation"][0] == frontier - p["overlap"]
            else:
                assert p["action"] == "fail" and lead == 0 and n_done == 0
    assert _plan_oom_continuation(0, 30, None, 0, 10, "add_prompt", False)["action"] == "retry_prompt"
    assert _plan_oom_continuation(0, 30, None, 0, 10, "add_prompt", True)["action"] == "fail"


# ── the wrapper (segmentation.sam3_wrapper.SAM3Wrapper.process_batch) ───────────────

class _Held:
    """Stands for the GPU tensors the vendor's propagation holds in its frame locals."""


class _Predictor:
    """A SAM3 predictor: one session in its store; the propagation holds ``held`` in a local
    and yields ``k`` frames before raising ``err``. ``prompt_err`` fails add_prompt instead."""

    def __init__(self, k, err=None, prompt_err=None):
        self.k, self.err, self.prompt_err = k, err, prompt_err
        self.store, self.closed = {}, []
        self.held_ref = None

    def handle_request(self, request):
        t = request["type"]
        if t == "start_session":
            self.store["s"] = {"state": {"memory_bank": _Held()}}
            return {"session_id": "s"}
        if t == "add_prompt":
            if self.prompt_err is not None:
                raise self.prompt_err
            return {"out_obj_ids": [1]}
        if t == "close_session":
            self.closed.append(request["session_id"])
            sess = self.store.pop(request["session_id"], None)
            if sess:
                sess["state"].clear()
            return {}
        return {}

    def handle_stream_request(self, request):
        held = _Held()                                  # a local of the vendor's frame
        self.held_ref = weakref.ref(held)
        for f in range(self.k):
            yield {"frame_index": f, "outputs": {"out_binary_masks": np.ones((1, 2, 2), bool),
                                                 "out_obj_ids": np.array([1])}}
        raise self.err


def _wrapper(monkeypatch, predictor, readings=None):
    import torch
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    it = iter(readings or [])
    monkeypatch.setattr(w, "card_memory_mib", lambda: next(it, None))
    wrapper = w.SAM3Wrapper()
    wrapper.is_loaded = True
    wrapper.predictor = predictor
    return w, wrapper


def test_the_wrapper_turns_an_oom_into_a_stop_with_the_session_released(monkeypatch):
    import torch
    oom = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    pred = _Predictor(3, err=oom)
    w, wrapper = _wrapper(monkeypatch, pred, readings=[CARD["at_stop"], CARD["after_release"]])
    try:
        wrapper.process_batch("/tmp/b", "tile", {0: 40, 1: 41, 2: 42, 3: 43, 4: 44})
    except w.SAM3PropagationStopped as stop:
        gc.collect()
        assert pred.held_ref() is None, \
            "nothing the stopped propagation held survives while the caller handles the stop"
        assert stop.cause == w.STOP_OUT_OF_MEMORY and stop.phase == "propagate_in_video"
        assert stop.frames_done == 3 and sorted(stop.partial) == [40, 41, 42]
        assert stop.memory == {"at_stop": CARD["at_stop"], "after_release": CARD["after_release"]}
        assert stop.__cause__ is None and stop.__context__ is None, "raised unchained"
    else:
        pytest.fail("an out-of-memory must come back as a stop")
    assert pred.closed == ["s"] and not pred.store and not wrapper._batch_sessions, \
        "the session is closed: a continuation opens a FRESH one"


def test_the_wrapper_reports_an_oom_before_any_frame_and_an_oom_on_frame_zero(monkeypatch):
    import torch
    w, wrapper = _wrapper(monkeypatch, _Predictor(0, prompt_err=torch.cuda.OutOfMemoryError(
        "CUDA out of memory")))
    with pytest.raises(w.SAM3PropagationStopped) as e:
        wrapper.process_batch("/tmp/b", "tile", {0: 0, 1: 1})
    assert e.value.phase == "add_prompt" and e.value.frames_done == 0 and e.value.partial == {}
    assert e.value.memory == {"at_stop": None, "after_release": None}, "unreadable is None"
    w, wrapper = _wrapper(monkeypatch, _Predictor(0, err=RuntimeError(
        "CUDA error: out of memory")))
    with pytest.raises(w.SAM3PropagationStopped) as e:
        wrapper.process_batch("/tmp/b", "tile", {0: 0, 1: 1})
    assert e.value.phase == "propagate_in_video" and e.value.frames_done == 0


def test_the_wrapper_still_fails_any_other_error_and_keeps_the_vendor_stop(monkeypatch):
    w, wrapper = _wrapper(monkeypatch, _Predictor(2, err=RuntimeError("tracker exploded")))
    with pytest.raises(w.SAM3RunError, match="tracker exploded") as e:
        wrapper.process_batch("/tmp/b", "tile", {0: 0, 1: 1, 2: 2})
    assert not isinstance(e.value, w.SAM3PropagationStopped)
    pred = _Predictor(2, err=RuntimeError(w.VENDOR_STATE_WITHOUT_CONDITIONING))
    w, wrapper = _wrapper(monkeypatch, pred)
    try:
        wrapper.process_batch("/tmp/b", "tile", {0: 0, 1: 1, 2: 2})
    except w.SAM3PropagationStopped as stop:
        gc.collect()
        assert stop.cause == w.STOP_VENDOR_STATE and stop.frames_done == 2
        assert pred.held_ref() is None, "the vendor stop's frames are cleared too"
    else:
        pytest.fail("the vendor tracker's stop is still a stop")


def test_an_oom_is_recognised_through_a_wrapping_error():
    import torch
    from segmentation.sam3_wrapper import is_cuda_oom
    assert is_cuda_oom(torch.cuda.OutOfMemoryError("Tried to allocate 2 GiB"))
    try:
        try:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        except Exception as inner:
            raise RuntimeError("tracker step failed") from inner
    except RuntimeError as outer:
        assert is_cuda_oom(outer)
    assert not is_cuda_oom(RuntimeError("tracker exploded")) and not is_cuda_oom(None)
