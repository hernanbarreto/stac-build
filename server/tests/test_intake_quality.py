"""I0 — features are measured for every frame and none is rejected for being
blurrier than its neighbours; only an impossible exposure rejects, with its
reason; sharp_rank ranks fft over ALL frames; the legacy frame_quality.json
shim is consumed by the existing selectors without raising."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from intake import quality as Q                                     # noqa: E402
from intake.config import QualityConfig                            # noqa: E402

W, H = 320, 240

CFG = QualityConfig(luma_lo=40.0, luma_hi=220.0, clip_frac_max=0.25, clip_lo=5, clip_hi=250,
                    fft_max_side=640, diff_max_side=320)


# ── synthetic frames ─────────────────────────────────────────────────────

def _texture(seed=0):
    """A textured gray image: noise + a checkerboard + a few edges, mean ~128."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    checker = (((xx // 20) + (yy // 20)) % 2) * 60.0
    noise = rng.normal(0.0, 25.0, size=(H, W))
    img = 98.0 + checker + noise
    img[60:180, 100:104] = 250.0                # a bright vertical bar
    return np.clip(img, 0, 255).astype(np.uint8)


def _sharp(seed=0):
    return _texture(seed)


def _blurred(seed=0, sigma=3.0):
    return cv2.GaussianBlur(_texture(seed), (0, 0), sigma)


def _dark(seed=0):
    return (_texture(seed).astype(np.float64) * 0.1).astype(np.uint8)          # mean ≈ 13


def _bright(seed=0):
    return np.clip(_texture(seed).astype(np.float64) + 110.0, 0, 255).astype(np.uint8)  # ≈ 238


def _clipped(seed=0):
    """Half the pixels black, half white: mean ≈ 128 (inside the band) but
    nearly everything sits at the ends of the range."""
    img = _texture(seed).astype(np.float64)
    out = np.where(img > np.median(img), 255.0, 0.0)
    return out.astype(np.uint8)


def _write(frames_dir: Path, grays, frame_numbers=None, ext=".jpg"):
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frame_numbers is None:
        frame_numbers = list(range(len(grays)))
    names = []
    for g, k in zip(grays, frame_numbers):
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        name = f"{k:06d}{ext}"
        ok = cv2.imwrite(str(frames_dir / name), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        names.append(name)
    return names


def _mixed_session(tmp_path):
    """sharp, blurred, dark, bright, clipped, sharp — strided frame numbers."""
    grays = [_sharp(0), _blurred(0), _dark(0), _bright(0), _clipped(0), _sharp(1)]
    numbers = [0, 3, 6, 9, 12, 15]
    frames_dir = tmp_path / "sess" / "frames"
    names = _write(frames_dir, grays, numbers)
    return frames_dir, names, numbers


# ── features ─────────────────────────────────────────────────────────────

def test_features_are_monotonic_in_blur():
    f_sharp = Q.frame_features(_sharp(), None, CFG)
    f_blur = Q.frame_features(_blurred(), None, CFG)
    f_blur2 = Q.frame_features(_blurred(sigma=6.0), None, CFG)
    assert f_sharp["fft"] > f_blur["fft"] > f_blur2["fft"]
    assert f_sharp["laplacian"] > f_blur["laplacian"] > f_blur2["laplacian"]
    assert f_sharp["inter_frame_diff"] is None
    assert f_sharp["gray_small"].shape == (H, W)                   # 320 ≤ diff_max_side: untouched
    assert abs(f_sharp["luma_mean"] - float(_sharp().mean())) < 1e-9
    assert 0.0 <= f_sharp["clip_frac"] <= 1.0


def test_exposure_features_and_verdicts():
    d = Q.frame_features(_dark(), None, CFG)
    b = Q.frame_features(_bright(), None, CFG)
    c = Q.frame_features(_clipped(), None, CFG)
    s = Q.frame_features(_sharp(), None, CFG)
    assert d["luma_mean"] < CFG.luma_lo and b["luma_mean"] > CFG.luma_hi
    assert CFG.luma_lo < c["luma_mean"] < CFG.luma_hi and c["clip_frac"] > 0.9
    assert Q.exposure_verdict(d["luma_mean"], d["clip_frac"], CFG) == "dark"
    assert Q.exposure_verdict(b["luma_mean"], b["clip_frac"], CFG) == "bright"
    assert Q.exposure_verdict(c["luma_mean"], c["clip_frac"], CFG) == "clipped"
    assert Q.exposure_verdict(s["luma_mean"], s["clip_frac"], CFG) is None


def test_inter_frame_diff_matches_legacy_arithmetic():
    a, b = _sharp(0), _sharp(1)
    fa = Q.frame_features(a, None, CFG)
    fb = Q.frame_features(b, fa["gray_small"], CFG)
    assert fb["inter_frame_diff"] == pytest.approx(float(cv2.absdiff(a, b).mean()))
    # a big frame is brought to ≤ diff_max_side with the legacy scale
    big = cv2.resize(a, (1280, 960), interpolation=cv2.INTER_LINEAR)
    fbig = Q.frame_features(big, None, CFG)
    assert max(fbig["gray_small"].shape) == 320
    fbig2 = Q.frame_features(big, fbig["gray_small"], CFG)
    assert fbig2["inter_frame_diff"] == pytest.approx(0.0)
    # another aspect ratio lands on another diff-scale shape → the contract breaks loudly
    wide = cv2.resize(a, (1280, 720), interpolation=cv2.INTER_LINEAR)
    fwide = Q.frame_features(wide, None, CFG)
    assert fwide["gray_small"].shape == (180, 320)
    with pytest.raises(Q.QualityError, match="native resolution"):
        Q.frame_features(a, fwide["gray_small"], CFG)


def test_fft_matches_the_legacy_formula():
    g = _sharp()
    from frames.quality import compute_fft_score
    assert Q.fft_score(g) == pytest.approx(compute_fft_score(g), rel=1e-6)


def test_fft_is_explicit_float64_and_bit_identical_to_the_numpy_1x_float32_path():
    """Plan point 74: the legacy call fed float32, which numpy 1.x upcasts to complex128 and
    numpy 2.x computes natively — the explicit float64 FFT gives EXACTLY the numpy-1.26 result
    (uint8 pixels are exact in float32 and float64, so the inputs are the same numbers)."""
    import ast
    src = Path(Q.__file__).read_text()
    body = src[src.index("def fft_score"):src.index("def laplacian_score")]
    assert "astype(np.float64)" in body and "float32" not in body.split('"""')[-1]
    for seed in range(5):
        for g in (_sharp(seed), _blurred(seed, sigma=1.0 + seed), _dark(seed), _clipped(seed)):
            small = Q.downscale_gray(g, CFG.fft_max_side, cv2.INTER_AREA)
            # the numpy-1.26 arithmetic of the old call, spelled out: complex128 throughout
            f = np.fft.fftshift(np.fft.fft2(small.astype(np.float32).astype(np.complex128)))
            mag = np.abs(f)
            h, w = mag.shape
            cy, cx, r = h // 2, w // 2, min(h, w) // 8
            mag[cy - r:cy + r, cx - r:cx + r] = 0
            legacy = float(np.mean(mag))
            assert Q.fft_score(small) == legacy                      # bit-identical, not approx
    assert np.fft.fft2(np.zeros((2, 2), np.float64)).dtype == np.complex128


def test_sharp_ranks():
    r = Q.sharp_ranks(np.array([3.0, 1.0, 2.0, 5.0]))
    assert np.allclose(r, [2 / 3, 0.0, 1 / 3, 1.0])
    assert np.allclose(Q.sharp_ranks(np.array([7.0])), [1.0])
    tied = Q.sharp_ranks(np.array([1.0, 2.0, 2.0, 3.0]))
    assert np.allclose(tied, [0.0, 0.5, 0.5, 1.0])
    assert tied.min() >= 0.0 and tied.max() <= 1.0


# ── the stage ────────────────────────────────────────────────────────────

def test_analyze_frames_report_contract(tmp_path):
    frames_dir, names, numbers = _mixed_session(tmp_path)
    logs = []
    rep = Q.analyze_frames(frames_dir, CFG, log=logs.append, heartbeat_s=1e-6)
    assert rep["version"] == Q.QUALITY_VERSION == 2 and rep["provenance"] == "tool_measured"
    assert rep["geometry_epoch"] == 0 and rep["camera_epoch"] == 0 == Q.INTAKE_EPOCH
    assert (rep["native_w"], rep["native_h"]) == (W, H)
    # the CPU environment the readings depend on (points 74 / 79), paths relative to the session
    env = rep["environment"]
    assert env["libs"]["numpy"] == np.__version__
    assert env["libs"]["opencv-python"].startswith(cv2.__version__)
    assert env["jpeg_decoders"]["opencv"]["libjpeg"] and env["jpeg_decoders"]["pillow"]["version"]
    assert env["cpu_model"] and env["numpy_fft_dtype"] == "complex128"
    assert rep["inputs"]["frames_dir"] == "frames" and "/" not in json.dumps(rep["inputs"])
    assert rep["n_frames"] == 6 and rep["n_usable"] == 3
    assert rep["rejected"] == {"dark": 1, "bright": 1, "clipped": 1}
    assert rep["params"]["luma_lo"] == CFG.luma_lo and rep["params"]["clip_lo"] == 5
    assert rep["inputs"]["n_frames"] == 6
    assert rep["inputs"]["first"] == names[0] and rep["inputs"]["last"] == names[-1]
    frames = rep["frames"]
    assert [f["frame"] for f in frames] == numbers          # video frame numbers, sorted
    assert [f["file"] for f in frames] == names
    assert set(frames[0]) == {"frame", "file", "fft", "laplacian", "luma_mean", "clip_frac",
                              "inter_frame_diff", "sharp_rank", "usable", "reject_reason"}
    assert frames[0]["inter_frame_diff"] is None
    assert all(isinstance(f["inter_frame_diff"], float) and f["inter_frame_diff"] > 0
               for f in frames[1:])
    by_name = {f["file"]: f for f in frames}
    assert by_name[names[0]]["usable"] and by_name[names[0]]["reject_reason"] is None
    assert by_name[names[1]]["usable"]                          # blurred is NOT rejected
    assert by_name[names[2]]["reject_reason"] == "dark"
    assert by_name[names[3]]["reject_reason"] == "bright"
    assert by_name[names[4]]["reject_reason"] == "clipped"
    assert all(not f["usable"] for f in frames if f["reject_reason"])
    ranks = [f["sharp_rank"] for f in frames]
    assert min(ranks) >= 0.0 and max(ranks) == 1.0
    assert by_name[names[0]]["sharp_rank"] > by_name[names[1]]["sharp_rank"]
    assert by_name[names[0]]["fft"] > by_name[names[1]]["fft"]
    # heartbeat lines carry the rate
    assert any("frames/s" in m for m in logs)
    assert any("done: 3/6 usable" in m for m in logs)


def test_no_percentile_rejection(tmp_path):
    """Twelve frames spanning a wide blur range at normal exposure: all usable.
    The legacy module would have cut the bottom 15 % by construction."""
    grays = [_blurred(i, sigma=0.5 + 0.75 * i) if i else _sharp(0) for i in range(12)]
    frames_dir = tmp_path / "frames"
    _write(frames_dir, grays)
    rep = Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    assert rep["n_usable"] == 12 and rep["rejected"] == {"dark": 0, "bright": 0, "clipped": 0}
    ffts = [f["fft"] for f in rep["frames"]]
    assert max(ffts) / min(ffts) > 3                             # the range really is wide
    ranks = [f["sharp_rank"] for f in rep["frames"]]
    assert ranks[0] == 1.0 and min(ranks) == 0.0
    from frames.quality import analyze_frames as legacy_analyze
    legacy = legacy_analyze(str(frames_dir))
    assert legacy["rejected_frames"] > 0                          # the gate this module removes


def test_determinism(tmp_path):
    frames_dir, _, _ = _mixed_session(tmp_path)
    a = Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    b = Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    assert a["frames"] == b["frames"] and a["rejected"] == b["rejected"]


def test_structural_failures_carry_the_reason(tmp_path):
    frames_dir = tmp_path / "frames"
    with pytest.raises(Q.QualityError, match="not a directory"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    frames_dir.mkdir()
    with pytest.raises(Q.QualityError, match="no frame"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    _write(frames_dir, [_sharp(0), _sharp(1)])
    (frames_dir / "notes.txt").write_text("ignored")
    (frames_dir / "thumb.jpg").write_bytes(b"not an image")
    with pytest.raises(Q.QualityError, match="video frame number"):
        Q.list_frames(frames_dir)
    (frames_dir / "thumb.jpg").unlink()
    (frames_dir / "000005.jpg").write_bytes(b"not an image")
    with pytest.raises(Q.QualityError, match="unreadable"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    (frames_dir / "000005.jpg").unlink()
    small = cv2.resize(_sharp(2), (160, 120))
    _write(frames_dir, [small], [7])
    with pytest.raises(Q.QualityError, match="mixes resolutions"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    with pytest.raises(Q.QualityError, match="heartbeat_s"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=0.0)


def test_heartbeat_is_required_and_never_read_from_the_global_config(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    _write(frames_dir, [_sharp(0), _sharp(1)])
    with pytest.raises(TypeError, match="heartbeat_s"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None)
    with pytest.raises(TypeError, match="heartbeat_s"):
        Q.run_quality(frames_dir, CFG, log=lambda *a: None)
    # a stage never reaches for the server config: break the loader, it still runs
    monkeypatch.setattr(Q, "load_intake_config", None)
    assert Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)["n_frames"] == 2


def test_cancel_stops_the_loop_naming_where(tmp_path):
    frames_dir, _, _ = _mixed_session(tmp_path)
    polls = []

    def cancel_at_frame_2():
        polls.append(1)
        return len(polls) > 2

    with pytest.raises(Q.IntakeCancelled, match=r"intake I0 \(quality\) at frame 2/6"):
        Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6,
                         cancelled=cancel_at_frame_2)
    assert not (frames_dir / "quality_features.json").exists()       # nothing half-written


def test_products_carry_epoch_zero_whatever_the_session_holds(tmp_path, monkeypatch):
    """Plan points 63 / 70: an intake product belongs to the reconstruction's epoch 0 by
    construction (the intake precedes every reconstruction); the session's epochs at run time
    are read for the RUN RECORD only, so the same frames give the same bytes whatever ran
    before. The CLI reads the frozen run configuration (point 69)."""
    frames_dir, _, _ = _mixed_session(tmp_path)
    session = frames_dir.parent
    assert Q.read_session_epochs(session) == {"geometry_epoch": 0, "camera_epoch": 0}
    a = Q.run_quality(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    bytes_a = {n: (frames_dir / n).read_bytes() for n in ("quality_features.json", "frame_quality.json")}
    out = session / "output"
    out.mkdir()
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 4}))
    from precision.camera import save_camera_json
    from tests import synth_precision as S
    cam = replace(S.default_camera(W, H).to_precision_camera(), camera_epoch=2)
    save_camera_json(out / "camera.json", cam, geometry_epoch=4)
    assert Q.read_session_epochs(session) == {"geometry_epoch": 4, "camera_epoch": 2}
    with pytest.raises(TypeError):                         # epochs are no longer a parameter
        Q.run_quality(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6, geometry_epoch=4)
    b = Q.run_quality(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    assert a == b and (b["geometry_epoch"], b["camera_epoch"]) == (0, 0)
    for name, before in bytes_a.items():
        assert (frames_dir / name).read_bytes() == before, name       # byte-identical
    # the CLI: no frozen run configuration in the session → the server's is frozen first
    import intake.run_config as RC
    monkeypatch.setattr(RC, "cli_intake_config",
                        lambda session_dir, log=print: (
                            __import__("intake.config", fromlist=["load_intake_config"])
                            .load_intake_config(_raw_with(CFG)), "0" * 64))
    assert Q.main(["--frames-dir", str(frames_dir)]) == 0
    assert json.loads((frames_dir / "frame_quality.json").read_text())["geometry_epoch"] == 0


def _raw_with(qcfg):
    import yaml
    from dataclasses import asdict
    with open(Path(Q.__file__).resolve().parents[1] / "config.yaml") as f:
        raw = yaml.safe_load(f)
    raw["intake"]["quality"] = asdict(qcfg)
    return raw


def test_bounds_come_from_the_config_object(tmp_path):
    frames_dir, names, _ = _mixed_session(tmp_path)
    loose = replace(CFG, luma_lo=1.0, luma_hi=254.0, clip_frac_max=1.0)
    rep = Q.analyze_frames(frames_dir, loose, log=lambda *a: None, heartbeat_s=1e-6)
    assert rep["n_usable"] == 6 and sum(rep["rejected"].values()) == 0
    strict = replace(CFG, clip_frac_max=0.0)
    rep = Q.analyze_frames(frames_dir, strict, log=lambda *a: None, heartbeat_s=1e-6)
    by_name = {f["file"]: f for f in rep["frames"]}
    assert by_name[names[0]]["reject_reason"] == "clipped"        # the bright bar clips


# ── artifacts ────────────────────────────────────────────────────────────

def test_write_and_load_round_trip_atomic(tmp_path):
    frames_dir, _, _ = _mixed_session(tmp_path)
    rep = Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    p = Q.write_quality(frames_dir, rep)
    assert p == frames_dir / "quality_features.json" and p.exists()
    assert not list(frames_dir.glob("*.tmp"))
    assert Q.load_quality(frames_dir) == json.loads(json.dumps(rep))
    with pytest.raises(Q.QualityError, match="does not exist"):
        Q.load_quality(tmp_path / "elsewhere")
    stale = dict(rep, version=0)
    Q.write_quality(frames_dir, stale)
    with pytest.raises(Q.QualityError, match="version"):
        Q.load_quality(frames_dir)


def test_legacy_shim_shape(tmp_path):
    frames_dir, names, _ = _mixed_session(tmp_path)
    rep = Q.analyze_frames(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    p = Q.write_legacy_frame_quality(frames_dir, rep)
    assert p == frames_dir / "frame_quality.json"
    leg = json.load(open(p))
    # the four artifact stamps ride along (extra top-level keys the legacy readers ignore)
    assert leg["version"] == Q.QUALITY_VERSION and leg["provenance"] == "tool_measured"
    assert leg["geometry_epoch"] == rep["geometry_epoch"] == 0
    assert leg["camera_epoch"] == rep["camera_epoch"] == 0
    assert leg["method"] == "intake_features"
    assert leg["threshold_fft"] == 0.0 and leg["threshold"] == 0.0
    assert leg["threshold_percentile"] == 0.0
    assert leg["total_frames"] == 6 and leg["valid_frames"] == 3 and leg["rejected_frames"] == 3
    assert [e["index"] for e in leg["frames"]] == list(range(6))
    assert [e["file"] for e in leg["frames"]] == names
    for e, f in zip(leg["frames"], rep["frames"]):
        assert set(e) == {"index", "file", "fft_score", "blur_score", "inter_frame_diff", "valid"}
        assert e["fft_score"] == f["fft"] and e["blur_score"] == f["laplacian"]
        assert e["valid"] == f["usable"]
    assert leg["frames"][0]["inter_frame_diff"] == 0.0
    assert leg["frames"][1]["inter_frame_diff"] == rep["frames"][1]["inter_frame_diff"]
    # the legacy reader of frames/quality.py consumes it too
    from frames.quality import load_valid_frames
    assert load_valid_frames(str(frames_dir)) == [f["file"] for f in rep["frames"] if f["usable"]]


def test_legacy_shim_feeds_frames_selector(tmp_path):
    frames_dir, names, _ = _mixed_session(tmp_path)
    rep = Q.run_quality(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    from frames.selector import _load_valid_frame_list
    files = _load_valid_frame_list(frames_dir)
    assert files == [f["file"] for f in rep["frames"] if f["usable"]]
    assert files == sorted(files, key=lambda f: int(Path(f).stem))
    assert len(files) == 3


def test_legacy_shim_feeds_map_worker_motion_keyframes(tmp_path):
    """workers.map_worker._motion_keyframes reads the shim and must not raise.
    Its import is heavy (like frames.selector's it ends up loading torch, ~10 s
    cold); torch is part of the da3 environment contract, so only a torch-free
    import failure is a reason to skip."""
    frames_dir, names, _ = _mixed_session(tmp_path)
    rep = Q.run_quality(frames_dir, CFG, log=lambda *a: None, heartbeat_s=1e-6)
    try:
        from workers.map_worker import _motion_keyframes
    except ImportError as e:                       # pragma: no cover - environment gap
        if "torch" in str(e):
            raise
        pytest.skip(f"workers.map_worker not importable here: {e}")
    shim = json.loads((frames_dir / "frame_quality.json").read_text())
    assert "provenance" in shim and "geometry_epoch" in shim     # the stamps do not disturb it
    diffs = [f["inter_frame_diff"] or 0.0 for f in rep["frames"]]
    quantum = sum(diffs) / 2.0
    chosen, n_total, soft = _motion_keyframes(frames_dir, quantum)
    assert n_total == 6 and len(chosen) >= 2
    assert set(chosen) <= set(names)
    assert chosen == sorted(chosen, key=lambda f: int(Path(f).stem))
    # the windows the selector cuts, replayed from the shim: a window holding a
    # usable frame picks a usable one (the sharpest); only a window with none is
    # 'soft' — exactly those, counted
    windows, win, acc = [], [], 0.0
    for e in shim["frames"]:
        win.append(e)
        acc += e["inter_frame_diff"]
        if acc >= quantum:
            windows.append(win)
            win, acc = [], 0.0
    if win:
        windows.append(win)
    assert len(windows) == len(chosen)
    expected_soft = sum(1 for w in windows if not any(e["valid"] for e in w))
    assert soft == expected_soft
    for w, c in zip(windows, chosen):
        valid = [e for e in w if e["valid"]]
        if valid:
            assert c == max(valid, key=lambda e: e["fft_score"])["file"]
    # the whole walk in one window: the sharpest usable frame wins
    chosen1, _, soft1 = _motion_keyframes(frames_dir, sum(diffs) * 10.0)
    best = max((f for f in rep["frames"] if f["usable"]), key=lambda f: f["fft"])
    assert chosen1 == [best["file"]] and soft1 == 0


def test_cli_writes_both_artifacts(tmp_path, capsys):
    sess = tmp_path / "sess"
    _write(sess / "frames", [_sharp(0), _sharp(1), _blurred(0)])
    assert Q.main(["--session", str(sess)]) == 0
    assert (sess / "frames" / "quality_features.json").exists()
    assert (sess / "frames" / "frame_quality.json").exists()
    out = capsys.readouterr().out
    assert "3/3 usable" in out
    assert Q.main(["--frames-dir", str(sess / "frames")]) == 0
