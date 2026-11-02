"""SAM3's concepts advanced together over one frame set (segmentation/sam3_group.py + the group path
of segmentation/sam3_wrapper.process_batch): the vision backbone of a chunk computed once and served
to every member through the vendor's own Sam3Image._get_img_feats, the same features; the vendor's
per-frame mask cache not written; a session cloned over the same frames; and the wrapper's group
path giving every concept the masks it gets alone — served from the group, after a member's stop,
after an out-of-memory group — with no GPU."""
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import vendor_paths  # noqa: F401,E402 — vendor/sam31 on the path
from segmentation import sam3_group as G  # noqa: E402
from segmentation import sam3_wrapper as W  # noqa: E402

N_FRAMES, CHUNK, H, Wd = 40, 16, 4, 6


# ── the vendor's _get_img_feats on a stand-in detector ───────────────────────

class _Backbone:
    def __init__(self):
        self.calls = []

    def forward_image(self, image):
        self.calls.append(int(image.shape[0]))
        b = image.shape[0]
        lvl = [image.mean(dim=1, keepdim=True).expand(b, 3, 8, 8) * (k + 1) for k in range(3)]
        pos = [torch.full_like(x, 0.5) for x in lvl]
        return {"vision_features": lvl[-1], "vision_pos_enc": pos, "backbone_fpn": lvl,
                "sam2_backbone_out": {"vision_features": lvl[-1] * 2, "backbone_fpn": lvl}}


def _detector():
    from sam3.model.sam3_image import Sam3Image

    class Det:
        _get_img_feats = Sam3Image._get_img_feats
        num_feature_levels = 1
        device = torch.device("cpu")

        def __init__(self):
            self.backbone = _Backbone()
    return Det()


class _Nested:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, i):
        return self.tensors[i]

    def __len__(self):
        return len(self.tensors)


def _frames():
    g = torch.Generator().manual_seed(0)
    return torch.rand(N_FRAMES, 3, 8, 8, generator=g)


def test_shared_vision_computes_a_chunk_once_and_serves_the_same_features():
    det = _detector()
    frames = _Nested(_frames())
    ids = torch.arange(16, 32)
    plain = det._get_img_feats({"img_batch_all_stages": frames, "language_features": 1}, ids)
    assert det.backbone.calls == [16]
    with G.SharedVision(det, capacity=2) as sv:
        a = det._get_img_feats({"img_batch_all_stages": frames, "language_features": "concept A"}, ids)
        b = det._get_img_feats({"img_batch_all_stages": frames, "language_features": "concept B"}, ids)
        assert (sv.computed, sv.served) == (1, 1)
    assert det.backbone.calls == [16, 16]              # one forward for the two concepts
    for got in (a, b):
        for x, y in zip(got[1], plain[1]):              # the image features the encoder reads
            assert torch.equal(x, y)
        for x, y in zip(got[2], plain[2]):
            assert torch.equal(x, y)
        assert got[3] == plain[3]
    assert a[0]["language_features"] == "concept A" and b[0]["language_features"] == "concept B"
    assert a[0]["backbone_fpn"][0] is b[0]["backbone_fpn"][0]   # the SAME tensors, served
    # outside the block the detector is the vendor's again
    assert "_get_img_feats" not in vars(det)
    det._get_img_feats({"img_batch_all_stages": frames}, ids)
    assert det.backbone.calls == [16, 16, 16]


def test_shared_vision_keys_on_the_frame_set_and_the_ids_and_holds_its_capacity():
    det = _detector()
    f1, f2 = _Nested(_frames()), _Nested(_frames() + 1)
    with G.SharedVision(det, capacity=2) as sv:
        for _ in range(2):
            det._get_img_feats({"img_batch_all_stages": f1}, torch.arange(0, 16))
            det._get_img_feats({"img_batch_all_stages": f2}, torch.arange(0, 16))   # other frames
        assert (sv.computed, sv.served) == (2, 2)
        det._get_img_feats({"img_batch_all_stages": f1}, torch.arange(16, 32))     # evicts the oldest
        det._get_img_feats({"img_batch_all_stages": f1}, torch.arange(0, 16))
        assert sv.computed == 4
        # a call that already carries features is the vendor's own fast path, untouched
        bo = det._get_img_feats({"img_batch_all_stages": f1}, torch.arange(0, 16))[0]
        det._get_img_feats(bo, torch.arange(0, 16))
        assert sv.computed == 4 and sv.served == 3
    with pytest.raises(ValueError):
        G.SharedVision(det, capacity=0)


def test_frame_outputs_not_kept_while_the_block_runs():
    class Model:
        def __init__(self):
            self.cache = {}

        def _cache_frame_outputs(self, state, frame_idx, obj_id_to_mask, **_k):
            state.setdefault("cached_frame_outputs", {})[frame_idx] = obj_id_to_mask

    m, st = Model(), {}
    with G.frame_outputs_not_kept(m):
        m._cache_frame_outputs(st, 3, {1: "mask"})
    assert st == {}
    m._cache_frame_outputs(st, 3, {1: "mask"})
    assert st == {"cached_frame_outputs": {3: {1: "mask"}}}
    assert "_cache_frame_outputs" not in vars(m)


def test_clone_session_shares_the_frames_and_restores_the_loader():
    mod = types.ModuleType("fake_sam3_tracking_mod")
    loads = []

    def load_resource_as_video_frames(resource_path, **_k):
        loads.append(resource_path)
        return _Nested(_frames()), 576, 1024

    mod.load_resource_as_video_frames = load_resource_as_video_frames

    def init_state(self, resource_path):
        images, h, w = mod.load_resource_as_video_frames(resource_path=resource_path)
        return {"input_batch": types.SimpleNamespace(img_batch=_Nested(images)),
                "orig_height": h, "orig_width": w}

    Model = type("Model", (), {"init_state": init_state, "__module__": mod.__name__})
    sys.modules[mod.__name__] = mod
    try:
        class Pred:
            def __init__(self):
                self.model = Model()
                self._all_inference_states = {}

            def handle_request(self, request):
                assert request["type"] == "start_session"
                sid = f"s{len(self._all_inference_states)}"
                self._all_inference_states[sid] = {
                    "state": self.model.init_state(request["resource_path"])}
                return {"session_id": sid}

        pr = Pred()
        base = pr.handle_request({"type": "start_session", "resource_path": "/frames"})["session_id"]
        clone = G.clone_session(pr, base)
        st0 = pr._all_inference_states[base]["state"]
        st1 = pr._all_inference_states[clone]["state"]
        assert loads == ["/frames"]                     # decoded once
        assert st1["input_batch"].img_batch.tensors is st0["input_batch"].img_batch.tensors
        assert (st1["orig_height"], st1["orig_width"]) == (576, 1024)
        assert mod.load_resource_as_video_frames is load_resource_as_video_frames
    finally:
        sys.modules.pop(mod.__name__, None)


# ── the wrapper's group path on a stand-in predictor ─────────────────────────

def _mask(text, frame):
    rng = np.random.default_rng(abs(hash((text, frame))) % (2 ** 32))
    return rng.random((2, H, Wd)) > 0.5


_LOADS = []


def load_resource_as_video_frames(resource_path, **_k):
    """The stand-in model's frame loader, looked up by name in this module (as the vendor's
    init_state looks up its own) — clone_session hands a clone the base session's frames."""
    _LOADS.append(resource_path)
    return _frames(), H, Wd


class _GroupModel:
    """Each session's propagation reads the vision features of every 16-frame chunk through the
    detector's _get_img_feats (as the vendor's batched grounding does) and yields its frames."""

    def __init__(self):
        self.detector = _detector()
        self.cached = 0

    def init_state(self, resource_path):
        images, h, w = load_resource_as_video_frames(resource_path=resource_path)
        return {"input_batch": types.SimpleNamespace(img_batch=_Nested(images)), "text": None,
                "orig_height": h, "orig_width": w}

    def _cache_frame_outputs(self, *_a, **_k):
        self.cached += 1


class _GroupPredictor:
    def __init__(self, stop=None, oom_at=None):
        self.model = _GroupModel()
        self._all_inference_states = {}
        self.calls = []
        self.stop = stop or {}          # text → frame where the propagation raises (vendor stop)
        self.oom_at = oom_at            # (text, frame) where the card runs out of memory
        self.open = set()
        self.n = 0

    def handle_request(self, request):
        t = request["type"]
        self.calls.append((t, request.get("text"), request.get("session_id")))
        if t == "start_session":
            sid = f"s{self.n}"
            self.n += 1
            self._all_inference_states[sid] = {"state": self.model.init_state(request["resource_path"])}
            self.open.add(sid)
            return {"session_id": sid}
        st = self._all_inference_states[request["session_id"]]["state"]
        if t == "add_prompt":
            st["text"] = request["text"]
            return {"out_obj_ids": [1]}
        if t == "reset_session":
            st["text"] = None
            return {"is_success": True}
        if t == "close_session":
            self.open.discard(request["session_id"])
            self._all_inference_states.pop(request["session_id"], None)
            return {}
        raise AssertionError(t)

    def handle_stream_request(self, request):
        st = self._all_inference_states[request["session_id"]]["state"]
        text = st["text"]
        self.calls.append(("propagate_in_video", text, request["session_id"]))
        img_batch = st["input_batch"].img_batch
        for f in range(N_FRAMES):
            if f % CHUNK == 0:
                ids = torch.arange(f, min(f + CHUNK, N_FRAMES))
                self.model.detector._get_img_feats({"img_batch_all_stages": img_batch}, ids)
            if self.oom_at == (text, f):
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 822.00 MiB")
            if self.stop.get(text) == f:
                raise RuntimeError(f"{W.VENDOR_STATE_WITHOUT_CONDITIONING} (stand-in)")
            self.model._cache_frame_outputs(st, f, {})
            yield {"frame_index": f, "outputs": {"out_binary_masks": torch.from_numpy(_mask(text, f)),
                                                 "out_obj_ids": torch.tensor([1, 2])}}


def _wrapper(pred, group=3, keep=False):
    w = W.SAM3Wrapper(device="cpu", config={"models": {"segmentation": {}}})
    w.predictor = pred
    w.is_loaded = True
    w._concept_group = group
    w._keep_frame_outputs = keep
    return w


MAP = {i: 100 + i for i in range(N_FRAMES)}
CONCEPTS = ["door", "pipe", "light", "sign"]


def _alone(text):
    w = _wrapper(_GroupPredictor(), group=1)
    return w.process_batch("/b0", text, MAP)


def _same(a, b):
    assert a.keys() == b.keys()
    for f in a:
        assert np.array_equal(a[f]["out_binary_masks"], b[f]["out_binary_masks"])
        assert np.array_equal(a[f]["out_obj_ids"], b[f]["out_obj_ids"])


def test_a_group_gives_every_concept_its_own_masks_with_one_backbone_per_chunk():
    pred = _GroupPredictor()
    w = _wrapper(pred, group=3)
    w.set_upcoming(CONCEPTS[1:])
    got = {c: None for c in CONCEPTS}
    got["door"] = w.process_batch("/b0", "door", MAP)
    n_chunks = -(-N_FRAMES // CHUNK)
    # door + the next two advanced together: the backbone ran once per chunk for the three
    assert pred.model.detector.backbone.calls == [CHUNK, CHUNK, N_FRAMES - 2 * CHUNK]
    assert w.group_stats["chunks_computed"] == n_chunks
    assert w.group_stats["chunks_served"] == 2 * n_chunks
    assert sum(1 for c in pred.calls if c[0] == "add_prompt") == 3
    assert sum(1 for c in pred.calls if c[0] == "start_session") == 3   # 1 decode: the clones share it
    w.set_upcoming(CONCEPTS[2:])
    got["pipe"] = w.process_batch("/b0", "pipe", MAP)           # served: no new propagation
    w.set_upcoming(CONCEPTS[3:])
    got["light"] = w.process_batch("/b0", "light", MAP)
    assert sum(1 for c in pred.calls if c[0] == "propagate_in_video") == 3
    w.set_upcoming([])
    got["sign"] = w.process_batch("/b0", "sign", MAP)            # alone: nothing upcoming
    assert w.group_stats["served_from_group"] == 2
    for c in CONCEPTS:
        _same(got[c], _alone(c))
    # the group's sessions over the same frames are gone; the batch's own session stays open
    assert pred.open == {w._batch_sessions["/b0"]}
    assert pred.model.cached == 0                                # the per-frame cache not written


def test_keep_frame_outputs_true_writes_the_vendor_cache_as_before():
    pred = _GroupPredictor()
    w = _wrapper(pred, group=1, keep=True)
    w.process_batch("/b0", "door", MAP)
    assert pred.model.cached == N_FRAMES
    assert not any(c[0] == "reset_session" for c in pred.calls)


def test_the_session_is_reset_once_its_masks_are_copied_out():
    pred = _GroupPredictor()
    w = _wrapper(pred, group=1, keep=False)
    w.process_batch("/b0", "door", MAP)
    assert [c[0] for c in pred.calls][-1] == "reset_session"


def test_a_member_that_stops_runs_alone_and_stops_the_same_way():
    pred = _GroupPredictor(stop={"pipe": 20})
    w = _wrapper(pred, group=3)
    w.set_upcoming(["pipe", "light"])
    door = w.process_batch("/b0", "door", MAP)
    _same(door, _alone("door"))
    w.set_upcoming(["light"])
    with pytest.raises(W.SAM3PropagationStopped) as stop:
        w.process_batch("/b0", "pipe", MAP)                     # its own call: the stop again
    assert stop.value.frames_done == 20
    w.set_upcoming([])
    light = w.process_batch("/b0", "light", MAP)
    _same(light, _alone("light"))
    assert w.group_stats["served_from_group"] == 1


def test_out_of_memory_in_a_group_runs_every_member_alone():
    pred = _GroupPredictor(oom_at=("light", 30))
    w = _wrapper(pred, group=3)
    w.set_upcoming(["pipe", "light"])
    door = w.process_batch("/b0", "door", MAP)                   # the group stopped: door alone
    assert w.group_stats["groups_out_of_memory"] == 1 and w.group_stats["fell_back"] == 1
    _same(door, _alone("door"))
    w.set_upcoming(["light"])
    pipe = w.process_batch("/b0", "pipe", MAP)                   # refused for groups: alone
    _same(pipe, _alone("pipe"))
    assert w.group_stats["groups"] == 1


def test_masks_kept_for_a_later_call_are_bit_packed_and_restored_exactly():
    rng = np.random.default_rng(3)
    res = {7: {"out_binary_masks": rng.random((3, 5, 7)) > 0.5, "out_obj_ids": np.array([1, 2, 3])},
           8: {"out_binary_masks": np.zeros((0, 5, 7), bool), "out_obj_ids": np.zeros(0, int)}}
    packed = W._pack_results(res)
    assert packed[7]["out_binary_masks"][0] == "packbits"
    back = W._unpack_results(packed)
    for f in res:
        assert back[f]["out_binary_masks"].dtype == np.bool_
        assert np.array_equal(back[f]["out_binary_masks"], res[f]["out_binary_masks"])


def test_concept_group_config_fails_naming_the_key():
    assert W.concept_group_config({"concept_group": 4, "keep_frame_outputs": False}) == (4, False)
    with pytest.raises(W.SAM3ConfigError, match="concept_group"):
        W.concept_group_config({"keep_frame_outputs": False})
    with pytest.raises(W.SAM3ConfigError, match="keep_frame_outputs"):
        W.concept_group_config({"concept_group": 4})
    for bad in (0, -1, True, 2.5, "4"):
        with pytest.raises(W.SAM3ConfigError):
            W.concept_group_config({"concept_group": bad, "keep_frame_outputs": False})
    with pytest.raises(W.SAM3ConfigError):
        W.concept_group_config({"concept_group": 2, "keep_frame_outputs": "no"})
