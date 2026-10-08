"""The VGGT-Long fork's determinism package (docs/plan_determinismo.md — package A):

- 8   loop_closures.txt stamped (keyframes, SALAD weights / code / parameters, revisit reference,
      the bar it applied); reused only on a matching stamp; post-hoc candidates in their own
      stamped file;
- 9   the run stamp (fork_stamp.json): products of another code / config / input are deleted
      and recomputed, never replayed; the digest travels in every artifact;
- 11  every capped subsample keeps the elements whose STABLE KEY ranks first — one element in
      or out moves at most one other;
- 12  deterministic torch STRICT, TF32 off, ONE autocast dtype, no CPU fallback, the
      environment recorded;
- 17  a bridge's measured scale disagreement enters its σ in quadrature, continuously;
- 20  the sky segmenter pinned by sha256, masks cached by model + code + image bytes;
- 45  camera_poses.txt / intrinsic.txt float64 round-trip exact.

CPU only, no model: the fork's modules are exercised directly; what lives inside the
VGGT_Long class (which needs the GPU-only SALAD stack to import) is checked on its source."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FORK = ROOT / "vendor" / "VGGT-Long"
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "server" / "tests"))
sys.path.insert(0, str(FORK))

import repro  # noqa: E402
from loop_utils import loop_bridges as LB  # noqa: E402
from loop_utils import fork_stamp as FS  # noqa: E402
from loop_utils import sky_mask as SM  # noqa: E402
from loop_utils.stable_sample import pixel_keys, stable_pick, stable_half  # noqa: E402

SRC = (FORK / "vggt_long.py").read_text()


def _method(name):
    cls = SRC[SRC.index("class VGGT_Long:"):]
    body = cls[cls.index(f"    def {name}("):]
    nxt = body.find("\n    def ", 10)
    return body if nxt < 0 else body[:nxt]


# ── point 11: stable subsamples ──────────────────────────────────────────────────────────

def test_stable_pick_moves_at_most_one_element_when_one_enters_or_leaves():
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(200_000, 50_000, replace=False))
    keys = pixel_keys(7, idx)
    sel = set(idx[stable_pick(keys, 8000)].tolist())
    for drop in (0, 123, 49_999):
        keep = np.delete(np.arange(len(idx)), drop)
        sel2 = set(idx[keep][stable_pick(keys[keep], 8000)].tolist())
        assert len(sel ^ sel2) <= 2, len(sel ^ sel2)
    # the old draw: default_rng(0).choice(len(valid), k) re-draws almost everything
    old = set(idx[np.random.default_rng(0).choice(len(idx), 8000, replace=False)].tolist())
    keep = np.delete(np.arange(len(idx)), 0)
    old2 = set(idx[keep][np.random.default_rng(0).choice(len(keep), 8000, replace=False)].tolist())
    assert len(old ^ old2) > 1000
    # deterministic, and the halves are per element
    assert np.array_equal(stable_pick(keys, 8000), stable_pick(keys, 8000))
    h = stable_half(keys)
    assert np.array_equal(stable_half(keys[1:]), h[1:])


def _plane_frame(H=100, W=100, f=80.0, z0=2.0):
    """A slanted plane seen by a camera at the origin: world points (H, W, 3), all valid."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    z = z0 + 0.002 * u + 0.001 * v
    x = (u - W / 2) / f * z
    y = (v - H / 2) / f * z
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return np.stack([x, y, z], -1).astype(np.float32), np.ones((H, W), np.float32), K


def test_surface_and_depth_pair_samples_are_stable_against_one_pixel():
    from loop_utils.metric_lock import depth_pair_samples, surface_pair_correspondences
    wp, cf, K = _plane_frame()
    w2c = np.eye(4)
    a = surface_pair_correspondences(wp, cf, wp, cf, w2c, K, max_samples=8000, seed=11,
                                     return_keys=True)
    cf2 = cf.copy()
    cf2[3, 7] = 0.0                                  # ONE pixel leaves the valid set
    b = surface_pair_correspondences(wp, cf2, wp, cf, w2c, K, max_samples=8000, seed=11,
                                     return_keys=True)
    ka, kb = set(a[2].tolist()), set(b[2].tolist())
    assert len(ka ^ kb) <= 2 and len(ka) == len(kb) == 8000
    # the key-free call returns the same pairs
    a2 = surface_pair_correspondences(wp, cf, wp, cf, w2c, K, max_samples=8000, seed=11)
    assert np.array_equal(a2[0], a[0]) and np.array_equal(a2[1], a[1])
    za = depth_pair_samples(wp, cf, wp, cf, w2c, K, max_samples=8000, seed=11)[0]
    zb = depth_pair_samples(wp, cf2, wp, cf, w2c, K, max_samples=8000, seed=11)[0]
    assert len(set(za.tolist()) ^ set(zb.tolist())) <= 2


def test_exact_correspondences_keep_their_pixels_when_one_leaves():
    wp, cf, _ = _plane_frame()
    bridge = {"world_points": np.stack([wp, wp]), "world_points_conf": np.stack([cf, cf])}
    chunk = {"world_points": np.stack([wp, wp, wp]), "world_points_conf": np.stack([cf] * 3)}
    p, q, k = LB.exact_correspondences(bridge, [0, 1], chunk, [1, 2], 3000, seed=5,
                                       frame_keys=[41, 42], return_keys=True)
    chunk2 = {"world_points": chunk["world_points"],
              "world_points_conf": chunk["world_points_conf"].copy()}
    chunk2["world_points_conf"][1, 50, 50] = 0.0
    p2, q2, k2 = LB.exact_correspondences(bridge, [0, 1], chunk2, [1, 2], 3000, seed=5,
                                          frame_keys=[41, 42], return_keys=True)
    assert len(set(k.tolist()) ^ set(k2.tolist())) <= 2
    # the keys name the GLOBAL keyframe: frame 41's and 42's pixels
    assert set((k >> np.uint64(32)).tolist()) == {41, 42}


def test_no_seeded_redraw_is_left_in_the_fork_production_path():
    for rel in ("vggt_long.py", "loop_utils/metric_lock.py", "loop_utils/loop_bridges.py"):
        s = (FORK / rel).read_text()
        assert ".choice(" not in s and ".permutation(" not in s and "RandomState(" not in s, rel
    s = (FORK / "loop_utils/sim3utils.py").read_text()
    wa = s[s.index("def weighted_align_point_maps"):]
    assert "stable_pick(" in wa and "np.random.RandomState(0)\n" not in wa
    assert "rng.choice(" not in wa
    # the intra-chunk judges are chosen by key, not the first 2000 of a key-ordered list
    assert "pq[0][:2000]" not in SRC and "stable_pick(pq[2], 2000)" in SRC
    # the STAC chunk writer never subsamples (the vendor's reservoir draws the global RNG)
    assert "Model.Pointcloud_Save.sample_ratio is" in SRC


# ── point 17: σ grows continuously with the measured scale disagreement ─────────────────

def _cfg():
    from synth_metric import fork_loops_cfg
    return fork_loops_cfg()


def _meas(s_ab=1.0, range_m=4.0, holdout=0.02):
    return {"ok": True, "s_ab": s_ab, "residual_m": 0.015, "holdout_residual_m": holdout,
            "n_corr": 50_000, "range_m": range_m, "R_ab": np.eye(3).tolist(),
            "t_ab": [0.0, 0.0, 0.0]}


def test_scale_disagreement_enters_sigma_in_quadrature_without_a_step():
    cfg = _cfg()
    assert "scale_tol_log" not in cfg and "scale_break_sigma_factor" not in cfg
    sig = []
    for d in (0.0, 0.049, 0.050, 0.051, 0.2):
        v = LB.verify_loop(_meas(), cfg, reference_m=0.05, scale_disagreement_log=d)
        assert v["status"] == "accepted"                    # no 'scale_break' any more
        assert v["sigma_m"] == pytest.approx(np.hypot(0.02, d * 4.0))
        assert v["checks"]["scale"]["sigma_scale_m"] == pytest.approx(d * 4.0)
        sig.append(v["sigma_m"])
    assert sig == sorted(sig)
    assert sig[3] - sig[1] < 1e-2                           # no ×4 jump at the old 5 % bar
    # the measurement's own log s_ab is the disagreement when the caller measured none
    v = LB.verify_loop(_meas(s_ab=np.exp(0.03)), cfg, reference_m=0.05)
    assert v["sigma_m"] == pytest.approx(np.hypot(0.02, 0.03 * 4.0))
    # a disagreement with no lever arm to read it in metres is refused, not guessed
    m = _meas(); m["range_m"] = None
    with pytest.raises(ValueError, match="lever arm"):
        LB.verify_loop(m, cfg, reference_m=0.05, scale_disagreement_log=0.01)
    # the scale row of the scale graph: the same rule in log units
    assert LB.scale_row_sigma(0.005, 0.0) == pytest.approx(0.005)
    assert LB.scale_row_sigma(0.005, -0.06) == pytest.approx(np.hypot(0.005, 0.06))
    assert "scale_row_sigma(sigma_loop, log_r_meas)" in SRC
    assert "sb_factor" not in SRC and "is_break" not in SRC
    assert "scale_disagreement_log=_sres" in SRC
    # the residual is persisted so a resumed run weighs the edge exactly as the first did
    assert '"bridge_scale_residual_log"' in SRC


def test_leftover_scale_break_keys_fail_the_load():
    import yaml
    from reconstruction.loops.config import LoopsConfigError, load_loops_config
    raw = yaml.safe_load((ROOT / "server" / "config.yaml").read_text())
    assert not hasattr(load_loops_config(raw).loop, "scale_tol_log")
    for k, v in (("scale_tol_log", 0.05), ("scale_break_sigma_factor", 4.0)):
        bad = yaml.safe_load((ROOT / "server" / "config.yaml").read_text())
        bad["correction_graph"]["loop"][k] = v
        with pytest.raises(LoopsConfigError, match=k):
            load_loops_config(bad)


# ── point 8: the loop candidate files ───────────────────────────────────────────────────

def _salad_session(tmp_path, n=6):
    frames = tmp_path / "frames"
    frames.mkdir()
    imgs = []
    for k in range(n):
        p = frames / f"{100 + k:06d}.jpg"
        p.write_bytes(f"jpeg-{k}".encode())
        imgs.append(str(p))
    w = tmp_path / "salad.ckpt"
    w.write_bytes(b"salad-weights")
    ref = tmp_path / "salad_revisit_reference.json"
    ref.write_text(json.dumps({"frames": [], "dist_bar_m": 1.0, "cos_bar": 0.5}))
    cfg = {"Weights": {"SALAD": str(w)},
           "Loop": {"SALAD": {"image_size": [322, 434], "batch_size": 32,
                              "similarity_threshold": 0.6, "top_k": 5, "use_nms": True,
                              "nms_threshold": 3, "min_gap": 10, "min_gap_frac": 0.05,
                              "revisit_reference": str(ref)}},
           "Model": {"loop_chunk_size": 20, "frame_stride": 1}}
    run = tmp_path / "maplong_run"
    run.mkdir()
    return imgs, cfg, run, ref


def _write_salad(run, cands, stamp):
    p = run / "loop_closures.txt"
    p.write_text("# Loop Detection Results (index1, index2, similarity)\n\n# Loop pairs:\n"
                 + "".join(f"{i}, {j}, {LB.format_similarity(s)}\n" for i, j, s in cands)
                 + "\n# Image path list:\n# 0: /x/000100.jpg\n")
    LB.write_loop_stamp(str(p), stamp)
    return p


def test_salad_stamp_names_every_input_that_makes_the_candidates(tmp_path):
    imgs, cfg, run, ref = _salad_session(tmp_path)
    st = LB.salad_loop_stamp(imgs, cfg)
    assert set(st["inputs"]) == {f"frames/{Path(p).name}" for p in imgs} | {
        "weights/SALAD", "revisit_reference"}
    assert any(k.startswith("vendor/VGGT-Long/LoopModels/") for k in st["code"])
    assert LB.salad_loop_stamp(imgs, cfg) == st                 # deterministic
    Path(imgs[2]).write_bytes(b"another frame")                 # a frame's bytes
    assert repro.check_stamp(st, LB.salad_loop_stamp(imgs, cfg))
    st2 = LB.salad_loop_stamp(imgs, cfg)
    cfg["Loop"]["SALAD"]["top_k"] = 6                           # a SALAD parameter
    assert repro.check_stamp(st2, LB.salad_loop_stamp(imgs, cfg))
    cfg["Loop"]["SALAD"]["top_k"] = 5
    ref.write_text(json.dumps({"frames": [1], "dist_bar_m": 1.0, "cos_bar": 0.5}))
    assert repro.check_stamp(st2, LB.salad_loop_stamp(imgs, cfg))   # the revisit reference
    assert repro.check_stamp(st2, LB.salad_loop_stamp(imgs[:-1], cfg))  # the frame list


def test_a_stamped_salad_file_is_reused_only_on_a_matching_stamp(tmp_path):
    imgs, cfg, run, _ = _salad_session(tmp_path)
    now = LB.salad_loop_stamp(imgs, cfg)
    lt = _write_salad(run, [(5, 0, 0.8123456789)], dict(now, salad_threshold=0.61))
    (run / "salad_calibration.json").write_text("{}")
    rec = LB.reconcile_loop_files(str(lt), now, log=lambda *a: None)
    assert rec["salad"] == "reused" and lt.exists()
    c = LB.load_loop_candidates(str(lt))
    assert c == [{"i": 5, "j": 0, "sim": 0.8123456789, "source": "salad"}]   # exact similarity
    # another keyframe list: the file, its calibration and the post-hoc candidates go
    LB.write_loop_candidates(str(lt), c + [{"i": 4, "j": 1, "sim": None, "source": "instance"}])
    other = LB.salad_loop_stamp(imgs[:-1], cfg)
    rec = LB.reconcile_loop_files(str(lt), other, log=lambda *a: None)
    assert rec["salad"] == "deleted" and rec["diffs"]
    for fn in ("loop_closures.txt", "salad_calibration.json", "loop_closures_posthoc.txt"):
        assert not (run / fn).exists(), fn
    # an UNSTAMPED file (written before stamps existed) is never reused
    (run / "loop_closures.txt").write_text("5, 0, 0.8\n")
    assert LB.reconcile_loop_files(str(lt), now, log=lambda *a: None)["salad"] == "deleted"


def test_post_hoc_candidates_live_in_their_own_stamped_file(tmp_path):
    imgs, cfg, run, _ = _salad_session(tmp_path)
    now = LB.salad_loop_stamp(imgs, cfg)
    lt = _write_salad(run, [(5, 0, 0.81)], now)
    salad_text = lt.read_text()
    salad = LB.load_loop_candidates(str(lt))
    post = [{"i": 4, "j": 1, "sim": None, "source": "instance"},
            {"i": 3, "j": 0, "sim": 0.5, "source": "instance:movable"}]
    LB.write_loop_candidates(str(lt), salad + post, header="merged by reconstruction.loops")
    assert lt.read_text() == salad_text, "the server never rewrites SALAD's stamped file"
    pp = run / "loop_closures_posthoc.txt"
    st = LB.read_loop_stamp(str(pp))
    assert st["salad_stamp"] == now["sha256"] and st["n"] == 2
    assert LB.load_loop_candidates(str(lt)) == salad + post
    assert LB.load_loop_candidates(str(lt), include_posthoc=False) == salad
    # the server cannot change SALAD's candidates
    with pytest.raises(ValueError, match="SALAD candidates are the fork's"):
        LB.write_loop_candidates(str(lt), [{"i": 9, "j": 0, "sim": 0.9, "source": "salad"}])
    # a post-hoc file edited by hand, or merged against another SALAD stamp, is not consumed
    pp.write_text(pp.read_text().replace("4, 1", "4, 2"))
    assert LB.load_loop_candidates(str(lt)) == salad
    assert LB.reconcile_posthoc(str(lt), log=lambda *a: None) == "deleted" and not pp.exists()
    LB.write_loop_candidates(str(lt), salad + post)
    LB.write_loop_stamp(str(lt), dict(now, sha256="another-salad-run"))
    assert LB.load_loop_candidates(str(lt)) == salad
    # no post-hoc candidate left: the file goes
    LB.write_loop_candidates(str(lt), salad)
    assert not pp.exists()


def test_get_loop_pairs_reads_the_stamped_files_and_stamps_a_fresh_run():
    body = _method("get_loop_pairs")
    assert "salad_loop_stamp(self.img_list, self.config)" in body
    assert "reconcile_loop_files(loop_txt, now" in body
    assert "salad_threshold=self.loop_detector.applied_threshold" in body
    assert "reconcile_posthoc(loop_txt" in body and "load_loop_candidates(loop_txt)" in body
    lm = (FORK / "LoopModels" / "LoopModel.py").read_text()
    assert "self.applied_threshold = float(threshold)" in lm
    assert "{repr(float(sim))}" in lm and "{sim:.4f}\\n" not in lm


# ── point 9: the run stamp ──────────────────────────────────────────────────────────────

def _fork_run(tmp_path, n=4):
    frames = tmp_path / "frames"
    frames.mkdir()
    imgs = []
    for k in range(n):
        p = frames / f"{200 + k:06d}.jpg"
        p.write_bytes(f"kf-{k}".encode())
        imgs.append(str(p))
    (frames / "000999.jpg").write_bytes(b"not a keyframe")
    w = tmp_path / "omega.pt"
    w.write_bytes(b"omega-weights")
    anchors = tmp_path / "da3_run" / "results_output"
    anchors.mkdir(parents=True)
    for k in range(n):
        (anchors / f"frame_{200 + k}.npz").write_bytes(f"anchor-{k}".encode())
    cfg = {"Weights": {"model": "VGGTOmega", "VGGTOmega": str(w)},
           "Model": {"metric_lock": {"anchor_dir": str(anchors)}, "loop_enable": True,
                     "loops": {"bridge_extra_frames": 0}, "omega_resolution": 518}}
    run = tmp_path / "maplong_run"
    for d in FS.RUN_SUBDIRS:
        (run / d).mkdir(parents=True)
    return imgs, cfg, run, anchors


CODE = [FORK / "vggt_long.py", FORK / "loop_utils" / "fork_stamp.py"]
ENV = {"card": "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0", "torch": "2.x", "amp_dtype": "bf16"}


def _stamp(imgs, cfg, run, **kw):
    return FS.run_stamp(img_list=imgs, img_dir=str(Path(imgs[0]).parent), config=cfg,
                        loop_cands=kw.pop("loop_cands", [{"i": 3, "j": 0, "sim": 0.7}]),
                        output_dir=str(run), environment=kw.pop("environment", ENV),
                        code_files=CODE, **kw)


def test_the_run_stamp_names_inputs_code_config_and_environment(tmp_path):
    imgs, cfg, run, anchors = _fork_run(tmp_path)
    st = _stamp(imgs, cfg, run)
    assert set(st["inputs"]) == ({f"frames/{Path(p).name}" for p in imgs} | {"weights/VGGTOmega"}
                                 | {f"anchors/frame_{200 + k}.npz" for k in range(4)})
    assert set(st["config"]) == {"fork_config", "frame_list", "loop_candidates", "environment"}
    assert _stamp(imgs, cfg, run) == st
    for change in (lambda: cfg["Model"].__setitem__("omega_resolution", 504),
                   lambda: Path(imgs[1]).write_bytes(b"re-extracted frame"),
                   lambda: (anchors / "frame_201.npz").write_bytes(b"another DA3")):
        before = _stamp(imgs, cfg, run)
        change()
        assert repro.check_stamp(before, _stamp(imgs, cfg, run))
    before = _stamp(imgs, cfg, run)
    assert repro.check_stamp(before, _stamp(imgs, cfg, run, loop_cands=[]))
    assert repro.check_stamp(before, _stamp(imgs, cfg, run, environment=dict(ENV, card="RTX A6000")))
    # a bridge may add NON-keyframe frames: then every image of the directory is an input
    cfg["Model"]["loops"]["bridge_extra_frames"] = 4
    assert "frames/000999.jpg" in _stamp(imgs, cfg, run)["inputs"]
    # the default code set: the fork, the Omega package, the server modules it imports
    code = [str(p) for p in FS.fork_code_files()]
    assert any(c.endswith("vendor/VGGT-Long/vggt_long.py") for c in code)
    assert any("/vggt-omega/vggt_omega/" in c for c in code)
    assert any(c.endswith("server/reconstruction/loops/spatial_gate.py") for c in code)
    assert any(c.endswith("vendor/VGGT-Long/loop_utils/metric_lock.py") for c in code)


def test_products_of_another_stamp_are_deleted_and_recomputed(tmp_path):
    imgs, cfg, run, anchors = _fork_run(tmp_path)
    now = _stamp(imgs, cfg, run)
    # a session written before stamps existed: every product goes, the stamped files stay
    for rel in ("_tmp_results_unaligned/chunk_0.npy", "pcd/0_pcd.ply", "pose_graph.json",
                "metric_lock.json", "camera_poses.txt", "loop_closures.txt", "frame_list.json",
                "vggt_omega_config.yaml", "salad_calibration.json"):
        (run / rel).write_text("x")
    (run / "sky_masks").mkdir()
    res = FS.reconcile(str(run), now, anchor_dir=str(anchors), log=lambda *a: None)
    assert not res["resumed"] and res["diffs"]
    left = sorted(os.listdir(run))
    assert set(left) == {"loop_closures.txt", "frame_list.json", "vggt_omega_config.yaml",
                         "salad_calibration.json", "sky_masks", "fork_stamp.json",
                         *FS.RUN_SUBDIRS}, left
    assert os.listdir(run / "_tmp_results_unaligned") == [] and os.listdir(run / "pcd") == []
    assert FS.read_saved(str(run))["sha256"] == now["sha256"]
    # the same stamp: the products are this run's — resumed, nothing deleted
    (run / "_tmp_results_unaligned" / "chunk_0.npy").write_text("y")
    res = FS.reconcile(str(run), _stamp(imgs, cfg, run), anchor_dir=str(anchors),
                       log=lambda *a: None)
    assert res["resumed"] and (run / "_tmp_results_unaligned" / "chunk_0.npy").exists()
    # another config: never replayed
    cfg["Model"]["omega_resolution"] = 504
    res = FS.reconcile(str(run), _stamp(imgs, cfg, run), anchor_dir=str(anchors),
                       log=lambda *a: None)
    assert not res["resumed"] and not (run / "_tmp_results_unaligned" / "chunk_0.npy").exists()
    assert any("fork_config" in d for d in res["diffs"])


def test_anchors_the_run_extracted_are_its_products(tmp_path):
    imgs, cfg, run, anchors = _fork_run(tmp_path, n=4)
    (anchors / "frame_203.npz").unlink()                      # the map worker gave 3 anchors
    now = _stamp(imgs, cfg, run)
    FS.reconcile(str(run), now, anchor_dir=str(anchors), log=lambda *a: None)
    (anchors / "frame_203.npz").write_bytes(b"bridge anchor")  # the run extracts one itself
    FS.record_run_products(str(run), anchors=["frame_203.npz"])
    saved = FS.read_saved(str(run))
    assert FS.run_products(saved)["anchors"] == ["frame_203.npz"]
    # a resume of the same run leaves it out of its stamp: the same stamp, resumed
    again = _stamp(imgs, cfg, run, exclude_anchors=FS.run_products(saved)["anchors"])
    assert FS.reconcile(str(run), again, anchor_dir=str(anchors), log=lambda *a: None)["resumed"]
    # a new stamp deletes it with the other products: the next run starts from the map
    # worker's inputs
    cfg["Model"]["omega_resolution"] = 504
    res = FS.reconcile(str(run), _stamp(imgs, cfg, run), anchor_dir=str(anchors),
                       log=lambda *a: None)
    assert "anchors/frame_203.npz" in res["deleted"] and not (anchors / "frame_203.npz").exists()


def test_an_artifact_of_another_stamp_stops_the_run():
    class Mismatch(RuntimeError):
        pass
    FS.check_artifact({"fork_stamp": "abc"}, "abc", "x", error=Mismatch)
    FS.check_artifact({"_stac_fork_stamp": "abc"}, "abc", "x", error=Mismatch)
    FS.check_artifact({}, None, "x", error=Mismatch)            # a stage driven alone
    with pytest.raises(Mismatch, match="no fork stamp"):
        FS.check_artifact({}, "abc", "x", error=Mismatch)
    with pytest.raises(Mismatch, match="is not this run's"):
        FS.check_artifact({"fork_stamp": "old"}, "abc", "x", error=Mismatch)


def test_the_fork_reconciles_its_stamp_before_resuming_anything():
    run = _method("run")
    assert run.index("self._stac_verify_run_stamp()") < run.index("_poses_done = ")
    assert run.index("self.get_loop_pairs()") < run.index("self._stac_verify_run_stamp()")
    v = _method("_stac_verify_run_stamp")
    assert "FS.reconcile(" in v and "environment=self._stac_environment_key()" in v
    assert "exclude_anchors=FS.run_products(saved)" in v
    # every resume artifact carries the digest and is checked when reloaded
    for name, check in (("process_single_chunk", "self._stac_check_fork_stamp(predictions"),
                        ("_stac_metric_lock", "self._stac_check_fork_stamp("),
                        ("_stac_elastic_seams", "self._stac_check_fork_stamp(prev, seams_path)"),
                        ("_stac_intra_chunk", "self._stac_check_fork_stamp(prev, ic_path)"),
                        ("_stac_depth_graph", "self._stac_check_fork_stamp(prev, dg_path)"),
                        ("_stac_uncertainty", "self._stac_check_fork_stamp(prev, rep_path)"),
                        ("_stac_pose_graph", "self._stac_check_fork_stamp(prev, pg_path)"),
                        ("_stac_ensemble_uncertainty", "self._stac_check_fork_stamp(")):
        assert check in _method(name), name
    assert "predictions['_stac_fork_stamp'] = self._stac_stamp_digest()" in SRC
    assert "record_run_products(self.output_dir" in _method("_stac_ensure_bridge_anchors")


# ── point 12: the fork's numerics ───────────────────────────────────────────────────────

def test_the_fork_process_runs_deterministic_strict_with_a_fixed_dtype():
    main = SRC[SRC.index("if __name__ == '__main__':"):]
    assert "_stac_repro().enable_deterministic_torch(42)" in main
    assert "warn_only=True" not in SRC
    assert "self._stac_require_deterministic_numerics(" in _method("run")
    env = _method("_stac_write_environment")
    assert "environment_record(gpu=True)" in env and "omega_environment.json" in env
    assert "_stac_require_deterministic_numerics(" in env
    req = _method("_stac_require_deterministic_numerics")
    for k in ('"deterministic_warn_only": False', '"cudnn_allow_tf32": False',
              '"matmul_allow_tf32": False', '"cudnn_benchmark": False'):
        assert k in req
    init = _method("__init__")
    assert "self.dtype = require_amp_card()" in init and 'self.device = "cuda"' in init
    assert "get_device_capability" not in SRC
    lm = (FORK / "LoopModels" / "LoopModel.py").read_text()
    assert "torch.device('cuda' if" not in lm
    assert 'raise RuntimeError("SALAD loop detection needs a CUDA device' in lm


# ── point 20: the pinned sky segmenter and its cache ────────────────────────────────────

def test_the_vendored_sky_segmenter_is_the_pinned_one():
    p = SM.skyseg_path()
    assert p == str(FORK / "skyseg.onnx")
    assert repro.sha256_file(p) == SM.SKYSEG_SHA256


def test_a_missing_or_different_sky_segmenter_fails(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        SM.skyseg_path(fork_dir=str(tmp_path))
    (tmp_path / "skyseg.onnx").write_bytes(b"another model")
    with pytest.raises(RuntimeError, match="has sha256"):
        SM.skyseg_path(fork_dir=str(tmp_path))
    ok = repro.sha256_file(tmp_path / "skyseg.onnx")
    assert SM.skyseg_path(fork_dir=str(tmp_path), expected_sha256=ok)
    assert "resolve/main/skyseg.onnx" not in SRC and "download_file_from_url" not in SRC


def test_sky_masks_are_cached_by_model_code_and_image_bytes(tmp_path):
    import cv2
    model = tmp_path / "skyseg.onnx"
    model.write_bytes(b"model-bytes")
    imgs = []
    for k in range(3):
        p = tmp_path / f"{k:06d}.jpg"
        p.write_bytes(f"img-{k}".encode())
        imgs.append(str(p))
    calls = []

    def segment(path, session, mask_path):
        calls.append(os.path.basename(path))
        m = np.full((8, 10), 255, np.uint8)
        m[:2] = 0                                           # the top two rows are sky
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, m)

    cache = tmp_path / "sky_masks"
    cache.mkdir()
    (cache / "000000.png").write_bytes(b"legacy mask keyed on the name alone")
    conf = np.ones((3, 8, 10), np.float32)
    r = SM.apply_sky_masks(conf, imgs, str(cache), model_path=str(model), segment=segment,
                           session_factory=lambda p: "session", log=lambda *a: None)
    assert r["n_computed"] == 3 and r["n_reused"] == 0
    assert conf[:, :2].sum() == 0 and conf[:, 2:].min() == 1.0
    assert Path(r["cache_dir"]).name.startswith(repro.sha256_file(model)[:16])
    # the same model, code and bytes: reused, the same mask
    conf2 = np.ones((3, 8, 10), np.float32)
    r2 = SM.apply_sky_masks(conf2, imgs, str(cache), model_path=str(model), segment=segment,
                            session_factory=lambda p: "session", log=lambda *a: None)
    assert r2["n_reused"] == 3 and np.array_equal(conf, conf2) and len(calls) == 3
    # a re-extracted frame with the SAME NAME is segmented again (the old cache keyed on the
    # name alone)
    Path(imgs[1]).write_bytes(b"new bytes, same name")
    r3 = SM.apply_sky_masks(np.ones((3, 8, 10), np.float32), imgs, str(cache),
                            model_path=str(model), segment=segment,
                            session_factory=lambda p: "session", log=lambda *a: None)
    assert r3["n_computed"] == 1 and calls[-1] == "000001.jpg"
    # another model: another cache directory, every mask recomputed
    model.write_bytes(b"other-model")
    r4 = SM.apply_sky_masks(np.ones((3, 8, 10), np.float32), imgs, str(cache),
                            model_path=str(model), segment=segment,
                            session_factory=lambda p: "session", log=lambda *a: None)
    assert r4["n_computed"] == 3 and r4["cache_dir"] != r["cache_dir"]
    # what can never be reused is gone: the legacy name-keyed mask, the other model's cache
    assert sorted(os.listdir(cache)) == [Path(r4["cache_dir"]).name]
    # a frame without its image cannot be masked
    with pytest.raises(RuntimeError, match="image path"):
        SM.apply_sky_masks(np.ones((4, 8, 10), np.float32), imgs, str(cache),
                           model_path=str(model), segment=segment,
                           session_factory=lambda p: "session", log=lambda *a: None)
    body = _method("_stac_mask_sky")
    assert "apply_sky_masks(" in body and "model_path=_stac_skyseg_path()" in body
    assert "except Exception" not in body


def test_the_sky_session_is_cpu_single_thread():
    import inspect
    src = inspect.getsource(SM.cpu_session)
    assert 'providers=["CPUExecutionProvider"]' in src
    assert "intra_op_num_threads = 1" in src and "inter_op_num_threads = 1" in src


# ── point 45: lossless pose files ───────────────────────────────────────────────────────

def test_the_fork_writes_poses_and_intrinsics_round_trip_exact(tmp_path):
    from loop_utils.camera_files import write_camera_files
    rng = np.random.default_rng(9)
    poses = []
    for k in range(5):
        T = np.eye(4)
        T[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        T[:3, 3] = rng.normal(0, 7, 3)
        poses.append(T.astype(np.float32) if k % 2 else T)       # float32 copies too
    K = [np.array([[392.123456789, 0, 207.5], [0, 391.987654321, 116.25], [0, 0, 1]],
                  np.float32 if k % 2 else np.float64) for k in range(5)]
    out = write_camera_files(str(tmp_path), poses, K)
    back = np.loadtxt(out["poses"]).reshape(-1, 4, 4)
    want = np.stack([np.asarray(p, np.float64) for p in poses])
    assert back.dtype == np.float64 and np.array_equal(back, want)   # bit for bit
    kb = np.loadtxt(out["intrinsics"])
    kw = np.array([[float(k_[0, 0]), float(k_[1, 1]), float(k_[0, 2]), float(k_[1, 2])]
                   for k_ in K])
    assert np.array_equal(kb, kw)
    # the old writer (str of a float32) did NOT read back as the same float64
    assert float(str(np.float32(0.1))) != float(np.float32(0.1))
    with pytest.raises(RuntimeError, match="without a pose"):
        write_camera_files(str(tmp_path), [poses[0], None])
    body = _method("save_camera_poses")
    assert "write_camera_files(" in body and "f.write(' '.join([str(x)" not in body
    assert "f.write(f'{fx} {fy} {cx} {cy}" not in body
