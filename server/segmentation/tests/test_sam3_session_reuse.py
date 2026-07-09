# STAC-Builder — SAM3 batch-session reuse + single add_prompt.
#
# SAM3's text pathway is one concept per pass (add_prompt calls reset_state), but the
# SESSION — the decoded frames — is shared. These tests pin the two things that used
# to burn ~75 minutes of a 100-minute segmentation:
#   1. ONE session for N concepts over the same frame set (was: N sessions, each
#      re-decoding every frame from disk).
#   2. ONE add_prompt per concept (was: 4, of which reset_state silently killed 3).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class FakePredictor:
    """Records the request stream a real SAM3 predictor would receive."""

    def __init__(self):
        self.calls = []          # list of (type, payload)
        self.open_sessions = set()
        self._next = 0

    def handle_request(self, request):
        rtype = request["type"]
        self.calls.append((rtype, request))
        if rtype == "start_session":
            self._next += 1
            sid = f"s{self._next}"
            self.open_sessions.add(sid)
            return {"session_id": sid}
        if rtype == "close_session":
            self.open_sessions.discard(request["session_id"])
            return {}
        if rtype == "add_prompt":
            return {"out_obj_ids": [1], "out_binary_masks": None}
        return {}

    def handle_stream_request(self, request):
        self.calls.append((request["type"], request))
        yield {"frame_index": 0, "outputs": {"out_obj_ids": [1]}}

    def count(self, rtype):
        return sum(1 for t, _ in self.calls if t == rtype)


def _wrapper(predictor):
    from segmentation.sam3_wrapper import SAM3Wrapper
    w = SAM3Wrapper.__new__(SAM3Wrapper)      # no model load, no CUDA
    w.predictor = predictor
    w.is_loaded = True
    w._batch_session = None
    return w


def test_one_session_and_one_prompt_across_concepts():
    fp = FakePredictor()
    w = _wrapper(fp)
    concepts = ["a red metal door", "a gray concrete wall", "black rubber train wheels"]
    for c in concepts:
        w.process_batch("/tmp/batch_A", c, {0: 0, 1: 1, 2: 2, 3: 3})

    assert fp.count("start_session") == 1, "the frame set is decoded once, not per concept"
    assert fp.count("add_prompt") == len(concepts), "exactly one add_prompt per concept"
    assert fp.count("close_session") == 0, "the session stays open between concepts"
    assert fp.count("propagate_in_video") == len(concepts)

    w.release_batch_session()
    assert fp.count("close_session") == 1 and not fp.open_sessions


def test_new_frame_set_reopens_a_session():
    fp = FakePredictor()
    w = _wrapper(fp)
    w.process_batch("/tmp/batch_A", "a door", {0: 0})
    w.process_batch("/tmp/batch_B", "a door", {0: 0})   # different frames → new session
    assert fp.count("start_session") == 2
    assert fp.count("close_session") == 1, "the previous session is closed, not leaked"
    assert len(fp.open_sessions) == 1, "only ever one session open at a time"


def test_prompt_lands_on_the_seeded_frame():
    """add_prompt resets the session, so only ONE seed frame can survive: pick the
    frame carrying the most boxes rather than silently keeping the last one."""
    fp = FakePredictor()
    w = _wrapper(fp)
    w.process_batch("/tmp/batch_A", "a door", {i: i for i in range(10)},
                    boxes_by_local={2: [[0, 0, 1, 1]], 7: [[0, 0, 1, 1], [0, 0, 2, 2]]})
    prompts = [r for t, r in fp.calls if t == "add_prompt"]
    assert len(prompts) == 1
    assert prompts[0]["frame_index"] == 7
    assert len(prompts[0]["bounding_boxes"]) == 2


def test_batch_dir_is_reused_for_the_same_frames(tmp_path=None):
    import tempfile
    from pathlib import Path
    from segmentation.pipeline import _prepare_batch_dir, _clear_batch_dirs

    _clear_batch_dirs()
    frames = Path(tempfile.mkdtemp())
    for i in range(3):
        (frames / f"{i:06d}.jpg").write_bytes(b"x")
    files = ["000000.jpg", "000001.jpg", "000002.jpg"]

    d1, m1 = _prepare_batch_dir(frames, files, 0)
    d2, m2 = _prepare_batch_dir(frames, files, 0)
    assert d1 == d2, "same frame list → same symlink dir (so SAM3 reuses the session)"
    assert m1 == m2 == {0: 0, 1: 1, 2: 2}

    d3, _ = _prepare_batch_dir(frames, files[:2], 0)
    assert d3 != d1, "a different frame list gets its own dir"

    _clear_batch_dirs()
    assert not d1.exists() and not d3.exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all SAM3 session-reuse tests passed")
