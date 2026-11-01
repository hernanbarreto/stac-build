"""reconstruction/sky_precompute.py — the fork's sky masks computed before Omega, in parallel.

1. keyframe_paths is vggt_long.VGGT_Long.run's image list (sorted *.jpg + *.png, the selected
   files, the stride).
2. The orchestration over a stand-in fork (the same function names as loop_utils): every frame's
   mask under its cache name, split over the workers, a mask already cached is not recomputed, no
   temporary file left, a failing worker FAILS the wait with its log.
3. With the REAL fork, model and Omega's interpreter (skipped where they are absent): the masks the
   workers write are byte for byte the masks the fork's own apply_sky_masks computes.
4. The map worker starts nothing when the fork will not mask the sky.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from reconstruction import sky_precompute as SP

ROOT = Path(__file__).resolve().parents[2]


def _img(path: Path, seed: int, h: int = 48, w: int = 64):
    import cv2
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[: h // 3] = (235, 206, 135)              # a band the segmenter may call sky
    cv2.imwrite(str(path), a)


def test_keyframe_paths_is_the_forks_image_list(tmp_path):
    for n in ("000003.jpg", "000001.png", "000002.jpg", "000010.jpg", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    got = SP.keyframe_paths(tmp_path, ["000002.jpg", "000010.jpg", "000001.png"])
    assert [os.path.basename(p) for p in got] == ["000001.png", "000002.jpg", "000010.jpg"]
    # an empty selection = every image (the fork filters only on a non-empty set)
    assert len(SP.keyframe_paths(tmp_path, [])) == 4
    assert [os.path.basename(p) for p in SP.keyframe_paths(tmp_path, [], stride=2)] == \
        ["000001.png", "000003.jpg"]


FAKE_SKY_MASK = '''
import hashlib, os
def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
def skyseg_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.bin")
def _sha256_cached(p):
    return _sha(p)
def segmenter_code_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_util.py")
def mask_cache_dir(root, m, c):
    return os.path.join(root, f"{m[:16]}-{c[:16]}")
def mask_path(cdir, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cdir, f"{stem}_{_sha(image_path)[:16]}.png")
def cpu_session(model):
    return {"model": model}
'''
FAKE_REPRO = '''
import hashlib
class _R:
    @staticmethod
    def sha256_file(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
def repro():
    return _R
'''
FAKE_VISUAL = '''
import os
import cv2
import numpy as np
def segment_sky(image_path, session, mask_filename):
    if os.path.basename(image_path).startswith("bad"):
        raise RuntimeError("cannot segment " + image_path)
    im = cv2.imread(image_path)
    m = np.where(im[..., 0] > 200, 0, 255).astype(np.uint8)
    cv2.imwrite(mask_filename, m)
    return m
'''


def _fake_fork(tmp_path: Path) -> Path:
    fork = tmp_path / "fork"
    (fork / "loop_utils").mkdir(parents=True)
    (fork / "loop_utils" / "__init__.py").write_text("")
    (fork / "loop_utils" / "sky_mask.py").write_text(FAKE_SKY_MASK)
    (fork / "loop_utils" / "stac_repro.py").write_text(FAKE_REPRO)
    (fork / "loop_utils" / "visual_util.py").write_text(FAKE_VISUAL)
    (fork / "model.bin").write_bytes(b"model")
    return fork


def test_orchestration_over_a_stand_in_fork(tmp_path):
    import cv2
    fork = _fake_fork(tmp_path)
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(7):
        _img(frames / f"{i:06d}.jpg", i)
    imgs = SP.keyframe_paths(frames, [f"{i:06d}.jpg" for i in range(7)])
    root = tmp_path / "sky_masks"
    # the stand-in's cache names, computed here (never by importing its loop_utils into this
    # process: the real fork's package may already be loaded under that name)
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cdir = str(root / f"{sha(fork / 'model.bin')[:16]}-{sha(fork / 'loop_utils' / 'visual_util.py')[:16]}")
    names = [os.path.join(cdir, f"{Path(p).stem}_{sha(p)[:16]}.png") for p in imgs]
    # one mask already cached: reused, not recomputed (its bytes are kept)
    os.makedirs(cdir, exist_ok=True)
    Path(names[3]).write_bytes(b"cached")
    h = SP.start(imgs, root, fork, sys.executable, workers=3, log=lambda s: None)
    rep = h.wait()
    assert rep["computed"] == 6 and rep["reused"] == 1 and rep["workers"] == 3
    assert Path(names[3]).read_bytes() == b"cached"
    for k, n in enumerate(names):
        if k == 3:
            continue
        m = cv2.imread(n, cv2.IMREAD_GRAYSCALE)
        im = cv2.imread(imgs[k])
        assert np.array_equal(m, np.where(im[..., 0] > 200, 0, 255).astype(np.uint8))
    assert not [f for f in os.listdir(cdir) if ".tmp" in f]
    # a frame the segmenter cannot do: the wait FAILS naming it
    _img(frames / "bad_000099.jpg", 99)
    h = SP.start([str(frames / "bad_000099.jpg")], root, fork, sys.executable, workers=2,
                 log=lambda s: None)
    with pytest.raises(RuntimeError, match="cannot segment"):
        h.wait()


def _real_fork_and_python():
    fork = ROOT / "vendor" / "VGGT-Long"
    model = fork / "skyseg.onnx"
    try:
        py = SP.omega_python()
    except RuntimeError:
        py = None
    if not model.is_file() or py is None:
        pytest.skip("the pinned skyseg.onnx or Omega's interpreter is not on this machine")
    return fork, py


def test_masks_are_the_bytes_the_forks_own_masking_writes(tmp_path):
    fork, py = _real_fork_and_python()
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        _img(frames / f"{i:06d}.jpg", 10 + i, h=96, w=160)
    imgs = SP.keyframe_paths(frames, [])
    pre = tmp_path / "pre"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    SP.start(imgs, pre, fork, py, env=env, workers=2, log=lambda s: None).wait()
    # the fork itself, in Omega's interpreter, from the fork's directory: apply_sky_masks computes
    # every mask (empty cache) through segment_sky
    own = tmp_path / "own"
    code = textwrap.dedent(f"""
        import json, sys
        import numpy as np
        sys.path[0] = {str(fork)!r}
        from loop_utils.sky_mask import apply_sky_masks, skyseg_path
        from loop_utils.visual_util import segment_sky
        conf = np.ones((3, 96, 160), np.float32)
        r = apply_sky_masks(conf, {imgs!r}, {str(own)!r}, model_path=skyseg_path(),
                            segment=segment_sky, log=lambda s: None)
        print(json.dumps({{"cache_dir": r["cache_dir"], "n_computed": r["n_computed"]}}))
    """)
    out = subprocess.run([py, "-c", code], cwd=str(fork), env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    assert rec["n_computed"] == 3
    own_dir = Path(rec["cache_dir"])
    pre_dir = pre / own_dir.name
    assert sorted(os.listdir(own_dir)) == sorted(f for f in os.listdir(pre_dir))
    for f in os.listdir(own_dir):
        assert (own_dir / f).read_bytes() == (pre_dir / f).read_bytes(), f


def test_the_map_worker_starts_nothing_when_the_fork_does_not_mask_the_sky(tmp_path):
    from workers import map_worker as MW

    class _Pipe:
        def send_log(self, *a, **k):
            pass
    cfg = {"reconstruction": {"simple": {"mask_sky": False}}}
    assert MW._start_sky_precompute(cfg, tmp_path, tmp_path, ["000000.jpg"], _Pipe()) is None
