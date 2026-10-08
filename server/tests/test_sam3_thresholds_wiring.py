"""SAM3's detection / confirmation thresholds reach the MODEL object
(models.segmentation.sam3_thresholds; USER 2026-09-29, with the census).

The vendor builders hard-code six numbers that decide which detections become
masklets — an object seen in fewer than 3 consecutive (parallax-spaced)
keyframes may never be confirmed. They are now declared per version in
config.yaml and written onto the built model in `SAM3Wrapper.load_model`, the
builder path. The bug class this guards is the 2026-09-23 one: a config key that
never reaches the object that uses it, and a run that succeeds anyway. So:

  · every configured value is read back off the model the (stubbed) vendor
    builder returned, for the 3.1 GPU, 3.0 GPU and CPU paths;
  · a missing key fails the load naming it — BEFORE the vendor builder runs —,
    a vendor rename fails it too, and the per-prompt loop RE-RAISES that error
    instead of skipping the prompt (every prompt would otherwise end at zero
    masklets, which the census would read as "thresholds too strict"); the SAM3
    worker checks the block before SAM3 starts;
  · the production values are the vendor's OWN builder values (no behaviour
    change) — parsed from the vendor source, so a vendor bump that moves them
    shows up here.
"""

import ast
import contextlib
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERVER = Path(__file__).resolve().parents[1]
KEYS = ("score_threshold_detection", "new_det_thresh", "hotstart_delay",
        "hotstart_unmatch_thresh", "hotstart_dup_thresh",
        "masklet_confirmation_consecutive_det_thresh")
# deliberately NOT the vendor values, so reading them back proves the wiring
TUNED = {"score_threshold_detection": 0.31, "new_det_thresh": 0.52, "hotstart_delay": 6,
         "hotstart_unmatch_thresh": 4, "hotstart_dup_thresh": 5,
         "masklet_confirmation_consecutive_det_thresh": 1}


def _vendor_model(vals):
    return types.SimpleNamespace(**vals)


class _Predictor:
    def __init__(self, vals):
        self.model = _vendor_model(vals)


VENDOR_31 = dict(score_threshold_detection=0.4, new_det_thresh=0.65, hotstart_delay=15,
                 hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
                 masklet_confirmation_consecutive_det_thresh=3)
VENDOR_30 = dict(score_threshold_detection=0.5, new_det_thresh=0.7, hotstart_delay=15,
                 hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
                 masklet_confirmation_consecutive_det_thresh=3)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """The vendor builders replaced by stubs that return a model carrying the
    vendor's attributes; CUDA reported present; autocast neutralised."""
    import torch
    import segmentation.sam3_wrapper as w

    built = {"n_builds": 0}
    pkg = types.ModuleType("sam3")
    pkg.__path__ = []
    mb = types.ModuleType("sam3.model_builder")

    def _multiplex(**kw):
        built["n_builds"] += 1
        built["sam3.1"] = _Predictor(VENDOR_31)
        return built["sam3.1"]

    def _video(**kw):
        built["sam3"] = _Predictor(VENDOR_30)
        return built["sam3"]

    mb.build_sam3_multiplex_video_predictor = _multiplex
    mb.build_sam3_video_predictor = _video
    model_pkg = types.ModuleType("sam3.model")
    model_pkg.__path__ = []
    vp = types.ModuleType("sam3.model.sam3_video_predictor")

    def _cpu():
        built["cpu"] = _Predictor(VENDOR_30)
        return built["cpu"]

    vp.Sam3VideoPredictor = _cpu
    for name, mod in (("sam3", pkg), ("sam3.model_builder", mb), ("sam3.model", model_pkg),
                      ("sam3.model.sam3_video_predictor", vp)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    ckpt = tmp_path / "sam3.1_multiplex.pt"
    ckpt.write_bytes(b"")

    def load(version, thresholds, cuda=True):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(w, "cfg", {"models": {"segmentation": {
            "version": version, "checkpoint_path": str(ckpt),
            "sam3_thresholds": thresholds}}})
        wrapper = w.SAM3Wrapper()
        wrapper.load_model()
        return wrapper

    return load, built


@pytest.mark.parametrize("version,key", [("sam3.1", "sam3.1"), ("sam3", "sam3")])
def test_every_configured_value_reaches_the_gpu_model(stubbed, version, key):
    load, built = stubbed
    wrapper = load(version, {version: dict(TUNED)})
    model = built[key].model
    for k in KEYS:
        assert getattr(model, k) == TUNED[k], f"{k} did not reach the {version} model"
    assert wrapper.applied_thresholds["applied"] == TUNED
    assert wrapper.applied_thresholds["vendor_built"] == (VENDOR_31 if version == "sam3.1"
                                                          else VENDOR_30)


def test_without_cuda_the_configured_version_fails_instead_of_the_cpu_fallback(stubbed):
    """docs/plan_determinismo.md point 165 (2026-10-08): the configured SAM is a requirement;
    without CUDA the load FAILS — the 3.0 CPU model is another model and is never built
    under a 3.1 configuration (it used to be, with only a warning)."""
    from segmentation.sam3_wrapper import SAM3DeviceError
    load, built = stubbed
    with pytest.raises(SAM3DeviceError, match="CUDA is not visible"):
        load("sam3.1", {"sam3.1": dict(VENDOR_31), "sam3": dict(TUNED)}, cuda=False)
    assert "cpu" not in built and "sam3" not in built, "a model was built without CUDA"


@pytest.mark.parametrize("missing", KEYS)
def test_a_missing_key_fails_the_load_naming_it_before_the_vendor_builder(stubbed, missing):
    from segmentation.sam3_wrapper import SAM3ConfigError
    load, built = stubbed
    vals = dict(TUNED)
    del vals[missing]
    with pytest.raises(SAM3ConfigError, match=f"sam3_thresholds.sam3.1.{missing}"):
        load("sam3.1", {"sam3.1": vals})
    assert built["n_builds"] == 0, "the vendor model was built before the key was checked"


def test_a_vendor_rename_fails_instead_of_silently_not_applying():
    from segmentation.sam3_wrapper import SAM3ConfigError, apply_sam3_thresholds
    renamed = dict(VENDOR_31)
    renamed["new_detection_thresh"] = renamed.pop("new_det_thresh")
    with pytest.raises(SAM3ConfigError, match="new_det_thresh"):
        apply_sam3_thresholds(_Predictor(renamed), "sam3.1", dict(TUNED))


def test_the_hotstart_assertion_the_constructor_runs_is_kept():
    from segmentation.sam3_wrapper import SAM3ConfigError, sam3_thresholds
    bad = dict(TUNED, hotstart_delay=3, hotstart_unmatch_thresh=4)
    with pytest.raises(SAM3ConfigError, match="hotstart"):
        sam3_thresholds({"sam3_thresholds": {"sam3.1": bad}}, "sam3.1")


# ── a config error is never a per-prompt failure ─────────────────────────

class _ConfigBrokenSAM3:
    """The wrapper as the per-prompt loop sees it: its lazy load (inside the
    first process_batch) raises the config error, every time it is asked."""

    def __init__(self):
        self.calls = 0

    def process_batch(self, *a, **k):
        from segmentation.sam3_wrapper import SAM3ConfigError
        self.calls += 1
        raise SAM3ConfigError("'models.segmentation.sam3_thresholds.sam3.1': "
                              "hotstart_unmatch_thresh and hotstart_dup_thresh must not "
                              "exceed hotstart_delay")

    def release_batch_session(self):
        pass

    def unload_model(self):
        pass

    def load_model(self):
        pass


def test_the_per_prompt_loop_reraises_a_config_error_instead_of_skipping(tmp_path,
                                                                        monkeypatch):
    import torch
    import segmentation.sam3_wrapper as w
    from segmentation import pipeline as P
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake = _ConfigBrokenSAM3()
    monkeypatch.setattr(w, "get_sam3_wrapper", lambda: fake)
    frames = tmp_path / "frames_valid"
    frames.mkdir()
    files = [f"{i:06d}.jpg" for i in range(3)]
    for f in files:
        (frames / f).write_bytes(b"")
    status = {}
    with pytest.raises(w.SAM3ConfigError, match="hotstart"):
        P._run_sam3_batched(frames, files, ["chair", "box", "conduit"], 10, 2, 0.3, 0.9,
                            output_dir=tmp_path, cfg={"visualization": {
                                "segment_colors": [[1, 2, 3]]}},
                            prompt_status=status)
    assert fake.calls == 1, "the next prompts were attempted after a config error"
    assert status["chair"]["status"] == "failed"
    assert "SAM3ConfigError" in status["chair"]["reason"]


def test_the_sam3_worker_checks_the_thresholds_before_sam3_starts(tmp_path, monkeypatch):
    """The wrapper loads lazily inside the first prompt; the worker must fail in
    seconds, naming the key, before any frame is touched."""
    import segmentation.sam3_wrapper as w
    import segmentation_pipeline
    from workers import sam3_worker
    called = []
    monkeypatch.setattr(segmentation_pipeline, "run_segmentation",
                        lambda **kw: called.append(kw) or {"instances": []})
    broken = yaml.safe_load((SERVER / "config.yaml").read_text())
    for block in broken["models"]["segmentation"]["sam3_thresholds"].values():
        del block["new_det_thresh"]
    monkeypatch.setattr(w, "cfg", broken)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "vlm_analysis.json").write_text('{"prompt": "chair"}')

    class _Pipe:
        def send_log(self, *a, **k):
            pass

        def send_progress(self, *a, **k):
            pass

        def check_cancel(self):
            return False

    with pytest.raises(w.SAM3ConfigError, match="new_det_thresh"):
        sam3_worker._sam3_work(_Pipe(), str(tmp_path), broken)
    assert not called, "SAM3 started before its thresholds were checked"


# ── production = the vendor's own values (no behaviour change) ───────────

def _vendor_file(*parts):
    """vendor/<...> of this checkout; git worktrees lack the gitignored
    vendor/sam31, so the enclosing checkouts are searched too."""
    for root in [SERVER.parent, *SERVER.parent.parents]:
        p = root.joinpath("vendor", *parts)
        if p.is_file():
            return p
    pytest.skip(f"vendor/{'/'.join(parts)} not on disk")


def _call_kwargs(path, func, callee):
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == callee)
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
            if kw.arg in KEYS}


def _init_default(path, cls, key):
    tree = ast.parse(path.read_text())
    c = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = init.args.args
    defaults = dict(zip([a.arg for a in args[len(args) - len(init.args.defaults):]],
                        init.args.defaults))
    return ast.literal_eval(defaults[key])


def test_production_sam31_values_are_the_multiplex_builders_own():
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())
    conf = raw["models"]["segmentation"]["sam3_thresholds"]["sam3.1"]
    mb = _vendor_file("sam31", "sam3", "model_builder.py")
    vendor = _call_kwargs(mb, "build_sam3_multiplex_video_predictor",
                          "Sam3MultiplexTrackingWithInteractivity")
    vendor["masklet_confirmation_consecutive_det_thresh"] = _init_default(
        mb.parent / "model" / "sam3_multiplex_base.py", "Sam3MultiplexBase",
        "masklet_confirmation_consecutive_det_thresh")
    assert set(vendor) == set(KEYS)
    assert conf == vendor, "config.yaml's sam3.1 block moved off the vendor's values"


def test_production_sam30_values_are_the_video_builders_own():
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())
    conf = raw["models"]["segmentation"]["sam3_thresholds"]["sam3"]
    mb = _vendor_file("sam3", "sam3", "model_builder.py")
    # the FIRST construction = the apply_temporal_disambiguation branch, the
    # default the wrapper builds (Sam3VideoPredictor passes True)
    vendor = _call_kwargs(mb, "build_sam3_video_model",
                          "Sam3VideoInferenceWithInstanceInteractivity")
    vendor["masklet_confirmation_consecutive_det_thresh"] = _init_default(
        mb.parent / "model" / "sam3_video_base.py", "Sam3VideoBase",
        "masklet_confirmation_consecutive_det_thresh")
    assert set(vendor) == set(KEYS)
    assert conf == vendor, "config.yaml's sam3 block moved off the vendor's values"


def test_production_blocks_load_through_the_strict_reader():
    from segmentation.sam3_wrapper import sam3_thresholds
    scfg = yaml.safe_load((SERVER / "config.yaml").read_text())["models"]["segmentation"]
    for v in ("sam3.1", "sam3"):
        assert set(sam3_thresholds(scfg, v)) == set(KEYS)
    assert scfg["version"] in scfg["sam3_thresholds"]
