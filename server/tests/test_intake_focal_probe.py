"""The focal probe's reuse key and spec (docs/plan_determinismo.md 66, 70): the spec names
frames relative to the session (the extractor gets absolute paths as its transient input); the
reuse key is a stamp of the frames' bytes, the code, the spec (layout, card, weights, dtype,
versions), the parameters and the CPU environment — a copied session keeps it, a changed frame,
card, dtype, resolution, parameter or library stack does not. (The layout itself is point 64's:
card_table.probe_window_frames + intake.focal.probe_windows, tested in test_intake_focal.)"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake import focal_probe as FP                            # noqa: E402
from intake import stamps as St                                 # noqa: E402

A100 = "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"
IDENT = {"card": A100, "weights": {"revision": "b2359bdf", "sha256": {"model.safetensors": "8e"}},
         "torch": {"version": "2.5", "cuda": "12.4", "cudnn": 90100},
         "libs": {"pillow": "12.0.0", "numpy": "1.26.4"},
         "code": {"server/extract_da3_depth.py": "x"}, "autocast_dtype": "bfloat16"}
ENV = {"libs": {"numpy": "1.26.4"}, "cpu_model": "x"}
LAYOUT = {"window_frames": 16, "provenance": "pccr: one window of 16"}


class PC:
    focal_probe_frames = 16
    focal_probe_res = "native"
    focal_probe_model = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    vram_margin_frac = 0.15


def _session(root: Path, n=4):
    sess = root / "s"
    fd = sess / "frames"
    fd.mkdir(parents=True)
    files = [f"{i * 5:06d}.jpg" for i in range(n)]
    for i, f in enumerate(files):
        (fd / f).write_bytes(bytes([i, 3, 5]) * 7)
    return sess, files


def test_spec_is_session_relative_and_the_extractor_gets_absolute_paths(tmp_path):
    sess, files = _session(tmp_path)
    sizing = {"window_frames": 2, "card": A100, "process_res": 840}
    windows = [[str(sess / "frames" / f) for f in files[:2]], files[2:]]   # paths or names
    spec = FP.probe_spec(windows, process_res=840, model_id=PC.focal_probe_model,
                         probe_layout={"window_frames": 2}, window_sizing=sizing,
                         da3_environment=IDENT)
    assert spec["windows"] == [["frames/000000.jpg", "frames/000005.jpg"],
                               ["frames/000010.jpg", "frames/000015.jpg"]]
    assert spec["sizes"] == [2, 2] and spec["da3_environment"]["card"] == A100
    assert spec["probe_layout"] == {"window_frames": 2}
    assert str(sess) not in json.dumps(spec)
    assert FP.extractor_windows(spec, sess) == [[str(sess / "frames" / f) for f in files[:2]],
                                                [str(sess / "frames" / f) for f in files[2:]]]


def test_stamp_keys_on_frame_bytes_code_spec_params_and_environment(tmp_path):
    sess, files = _session(tmp_path)
    sizing = {"window_frames": 4, "card": A100, "process_res": 840}
    spec = FP.probe_spec([files], process_res=840, model_id=PC.focal_probe_model,
                         probe_layout=LAYOUT, window_sizing=sizing, da3_environment=IDENT)
    st = FP.probe_stamp(sess, spec, PC, run_config_sha256="r" * 64, environment=ENV)
    assert set(st["inputs"]) == {f"frames/{f}" for f in files}
    assert "server/intake/focal.py" in st["code"] and "server/extract_da3_depth.py" in st["code"]
    assert set(st["config"]) == {"spec", "params", "environment", "run_config_sha256"}
    assert FP.probe_reusable({"version": FP.FOCAL_VERSION, "stamp": st}, st) == []
    # a copied session: the same stamp
    dst = tmp_path / "copy" / "s"
    shutil.copytree(sess, dst)
    assert FP.probe_stamp(dst, spec, PC, run_config_sha256="r" * 64, environment=ENV) == st
    # a frame re-extracted under the same name (another byte, same size): not reusable
    (sess / "frames" / files[1]).write_bytes(bytes([9, 3, 5]) * 7)
    st2 = FP.probe_stamp(sess, spec, PC, run_config_sha256="r" * 64, environment=ENV)
    assert [d for d in FP.probe_reusable({"version": FP.FOCAL_VERSION, "stamp": st}, st2)
            if files[1] in d]
    (sess / "frames" / files[1]).write_bytes(bytes([1, 3, 5]) * 7)
    # another card, dtype, weights, resolution or probe parameter: not reusable either
    for other in (dict(spec, da3_environment=dict(IDENT, card="NVIDIA RTX A6000 | 49140 MiB | sm_8.6")),
                  dict(spec, da3_environment=dict(IDENT, autocast_dtype="float16")),
                  dict(spec, da3_environment=dict(IDENT, weights={"revision": "ffff"})),
                  dict(spec, process_res=1932)):
        st3 = FP.probe_stamp(sess, other, PC, run_config_sha256="r" * 64, environment=ENV)
        assert FP.probe_reusable({"version": FP.FOCAL_VERSION, "stamp": st}, st3) == [
            f"config 'spec' changed ({st['config']['spec'][:12]} -> {st3['config']['spec'][:12]})"]

    class PC2(PC):
        focal_probe_frames = 8
    st4 = FP.probe_stamp(sess, spec, PC2, run_config_sha256="r" * 64, environment=ENV)
    assert any(d.startswith("config 'params'") for d in
               FP.probe_reusable({"version": FP.FOCAL_VERSION, "stamp": st}, st4))
    # another library stack
    st5 = FP.probe_stamp(sess, spec, PC, run_config_sha256="r" * 64,
                         environment={"libs": {"numpy": "2.0.0"}, "cpu_model": "x"})
    assert any(d.startswith("config 'environment'") for d in
               FP.probe_reusable({"version": FP.FOCAL_VERSION, "stamp": st}, st5))
    # an older document, no stamp, not a document
    assert FP.probe_reusable({"version": 1, "spec": spec}, st) == [
        f"focal_probe.json is version 1, this code writes {FP.FOCAL_VERSION}"]
    assert FP.probe_reusable({"version": FP.FOCAL_VERSION}, st) == [
        "no stamp saved with the product (or it is unreadable)"]
    assert FP.probe_reusable("nope", st) == ["focal_probe.json is not a JSON object"]
    # the environment defaults to the measured one
    import repro
    st6 = FP.probe_stamp(sess, spec, PC)
    assert st6["config"]["environment"] == repro.sha256_json(St.cpu_environment_record())
    assert st6["config"]["run_config_sha256"] == repro.sha256_json(None)
