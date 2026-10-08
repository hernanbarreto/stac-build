#!/usr/bin/env python3
"""
Simple script to extract DA3 Giant depth maps to NumPy arrays.
Designed to run in the `da3` conda environment to avoid VGGT-Long dependency clashes.

Reproducibility (docs/plan_determinismo.md points 27, 32, 42, 43, 44):
- torch STRICT deterministic (an op without a deterministic kernel raises, no warn-only), TF32
  off for cuDNN and matmul, cuBLAS workspace pinned (repro.enable_deterministic_torch);
- the weights of the PINNED revision from the ONE Hugging Face cache, every file sha256-checked,
  offline (da3_weights); CUDA required (no CPU fallback) and DA3's autocast dtype fixed to bf16;
- ``--identity`` prints what a window depends on beyond its frames (model / revision / weights,
  this extractor's and DA3's code, torch / CUDA / cuDNN, the card and its driver, the numerics):
  the launcher records it in windows.json and every window is stamped with it;
- windows mode: every window_<i>.npz carries a stamp (identity + its frames' sha256); any window
  on disk whose stamp differs from this run's (or has none) makes ALL window files go before
  anything is reused; the reference view DA3 picks per window is recorded (``ref_views`` in the
  npz and output_dir/reference_views.json) and a regeneration must pick the same one or FAIL.
"""
import argparse
import os
import sys
import glob
import json
import numpy as np

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import repro  # noqa: E402  (numpy only at import)
import da3_weights  # noqa: E402
from intake import stamps as intake_stamps  # noqa: E402 — the JPEG decoders' record (point 79)

# exit codes the launchers read (intake/vram.py re-exports them)
OOM_EXIT = 3            # a window does not fit the card — the caller FAILS (never halves)
REF_VIEW_EXIT = 5       # a regenerated window picked another reference view than the recorded one
IDENTITY_EXIT = 6       # this process is not the DA3 environment windows.json was planned for

DA3_SRC = os.path.join(_SERVER_DIR, "..", "vendor", "depth-anything-3", "src")
DA3_ROOT = os.path.join(_SERVER_DIR, "..", "vendor", "depth-anything-3")
REFERENCE_VIEWS_NAME = "reference_views.json"
# the libraries whose versions a window's numerics depend on (image decode, tensors, kernels)
IDENTITY_LIBS = ("numpy", "pillow", "opencv-python", "opencv-python-headless", "torch",
                 "torchvision", "safetensors", "huggingface-hub", "xformers", "einops", "timm")
# THE dtype DA3 runs in on every validated run (its api picks bf16 whenever the card supports
# it — A100 sm_80, A6000 sm_86); a card without bf16 would silently run fp16: refused instead
AUTOCAST_DTYPE = "bfloat16"
DETERMINISTIC_SEED = 0

import torch  # noqa: E402
import cv2  # noqa: E402,F401
from PIL import Image  # noqa: E402,F401


def _deterministic() -> dict:
    """Identical inputs → bit-identical depths (USER 2026-09-28; STRICT since 2026-10-07, plan
    point 27). Runs before anything touches CUDA: cuBLAS reads its workspace config when its
    handle is created."""
    return repro.enable_deterministic_torch(DETERMINISTIC_SEED)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("DA3 needs the GPU and torch sees no CUDA device — no CPU fallback "
                           "(another device gives other numbers)")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"this card does not support {AUTOCAST_DTYPE}: DA3 would run in fp16, "
                           f"not the dtype of every validated run — refused")
    return torch.device("cuda")


def _require_one_device() -> int:
    """Exactly ONE card visible to this process (plan point 78): the identity, the memory and
    the windows are those of the device DA3 runs on — never nvidia-smi's first GPU, never one of
    several torch could pick. Returns torch's device index (0)."""
    n = int(torch.cuda.device_count())
    if n != 1:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} leaves "
                           f"{n} card(s) visible — DA3 runs on exactly one card and its identity is "
                           f"read from that device; set CUDA_VISIBLE_DEVICES to the one card")
    return 0


def da3_identity(model_id: str) -> dict:
    """What a DA3 output depends on beyond its frames — the same on every run of the same
    machine, weights and code (no time, pid or host). CUDA is initialised here (in the probe
    subprocess, or in the extracting process), on the ONE visible device (point 78)."""
    _require_cuda()
    dev = _require_one_device()
    torch.cuda.init()
    card = repro.card_identity(dev)
    drivers = {c["uuid"]: c["driver_version"] for c in repro.gpu_cards()}
    if card["uuid"] not in drivers:
        raise RuntimeError(f"nvidia-smi does not list the card {card['uuid']}")
    code = repro.stamp(code=[os.path.abspath(__file__), repro.__file__, da3_weights.__file__])
    return {
        "weights": da3_weights.identity(model_id),
        "code": code["code"],
        "da3_git": repro.git_state(os.path.abspath(DA3_ROOT)),
        "torch": {"version": str(torch.__version__), "cuda": str(torch.version.cuda),
                  "cudnn": int(torch.backends.cudnn.version() or 0),
                  "git_version": str(getattr(torch.version, "git_version", ""))},
        "libs": {n: repro._dist_version(n) for n in IDENTITY_LIBS},
        # the two JPEG decoders of the pipeline with their libjpeg-turbo builds (plan point
        # 79): DA3 reads the frames with Pillow (input_processor.Image.open), I0 / I1 / F4
        # read the same files with OpenCV — a change of either changes the pixels one side sees
        "jpeg_decoders": intake_stamps.jpeg_decoder_record(),
        # the card MODEL (repro.card_key: name | board MiB | sm) — never the instance: this
        # identity is compared by every window stamp, windows.json, the walk and the focal probe,
        # and a uuid (or torch's usable-memory bytes) there invalidated or refused their products
        # on another card of the same model; the instance is recorded in da3_environment.json
        "card": card["key"],
        "device_count": 1,
        "driver_version": drivers[card["uuid"]],
        "autocast_dtype": AUTOCAST_DTYPE,
        "numerics": {k: v for k, v in repro.torch_numerics_record().items()},
    }


def _load_model(model_id: str, log=print):
    """The pinned DA3 weights (verified) on the GPU, in eval mode."""
    device = _require_cuda()
    snap = da3_weights.verified_snapshot(model_id, log=log)
    if DA3_SRC not in sys.path:
        sys.path.insert(0, DA3_SRC)
    from depth_anything_3.api import DepthAnything3
    # PyTorchModelHubMixin.from_pretrained SILENTLY IGNORES a `device=` kwarg — the model stayed
    # on CPU (23 cores pinned, minutes per frame, GPU at 0%). Move it explicitly and verify.
    model = DepthAnything3.from_pretrained(str(snap)).to(device)
    model.eval()
    p = next(model.parameters())
    if p.device.type != "cuda":
        raise RuntimeError("model did not reach the GPU — aborting instead of "
                           "silently burning CPU")
    return model, device


def main():
    _deterministic()
    parser = argparse.ArgumentParser("Extract DA3 relative depth to NPY")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--identity", action="store_true",
                        help="print this process's DA3 identity (weights, code, torch, card, "
                             "numerics) as one '[DA3 identity] {json}' line and exit — what the "
                             "launcher records in windows.json; loads no weights")
    parser.add_argument("--per_frame", action="store_true",
                        help="Run inference one image at a time (ISOLATED monocular depth "
                             "— no cross-frame attention). Used for the metric scale "
                             "anchor, where frames are seconds apart and must not be "
                             "treated as a multi-view set.")
    parser.add_argument("--process_res", type=int, default=None,
                        help="DA3 processing resolution (upper-bound long side; model "
                             "default 504). Phase C detail transfer uses ~1008 so the "
                             "depth carries detail above the omega grid's Nyquist. "
                             "ViT cost grows ~quadratically — keyframes only.")
    parser.add_argument("--windows_json", type=str, default=None,
                        help="MULTI-VIEW WINDOWS (intake I3, claude_stac.txt §4-F2): a JSON "
                             "{'windows': [[image path, ...], ...]}; each window is ONE joint "
                             "inference and lands in <output_dir>/window_<i:04d>.npz (frames, "
                             "depth, conf, extrinsics w2c, intrinsics, scale_factor, "
                             "is_metric, stamp, ref_views). Window files are reused only when "
                             "EVERY one on disk carries this run's stamp.")
    args = parser.parse_args()

    if args.identity:
        print("[DA3 identity] " + repro.canonical_json(da3_identity(args.model)), flush=True)
        return
    if not args.image_dir or not args.output_dir:
        parser.error("--image_dir and --output_dir are required")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.windows_json:
        return run_windows(args)

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.image_dir, "*.png")))

    # Skip entirely if all outputs already exist (before loading model)
    missing = []
    for img_path in images:
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")
        if not os.path.exists(depth_path) or not os.path.exists(conf_path):
            missing.append(img_path)

    if not missing:
        print(f"[DA3 Extractor] All {len(images)} depth+conf maps already exist. Skipping.")
        return

    print(f"[DA3 Extractor] {len(missing)} of {len(images)} need processing")

    # Only now load the heavy model (pinned weights, verified; GPU only)
    print(f"[DA3 Extractor] Loading {args.model} (pinned revision, one HF cache)")
    model, device = _load_model(args.model)
    p = next(model.parameters())
    print(f"[DA3 Extractor] model device: {p.device} (dtype {p.dtype})")

    # Optional processing-resolution override (Phase C hi-res detail source)
    _res_kw = {}
    if args.process_res:
        _res_kw["process_res"] = int(args.process_res)
        print(f"[DA3 Extractor] process_res override: {args.process_res}")

    # Run inference: one joint batch (default) or strictly per-frame (--per_frame)
    if args.per_frame:
        d_list, c_list, k_list = [], [], []
        with torch.no_grad():
            for i, img_path in enumerate(images):
                print(f"[DA3 Extractor] isolated inference {i+1}/{len(images)}: "
                      f"{os.path.basename(img_path)}")
                pred = model.inference([img_path], **_res_kw)
                d = pred.depth
                c = pred.conf
                k = getattr(pred, "intrinsics", None)
                d_list.append(d.cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d))
                c_list.append(c.cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c))
                if k is not None:
                    k_list.append(k.cpu().numpy() if isinstance(k, torch.Tensor) else np.asarray(k))
        depths = np.concatenate(d_list, axis=0)
        confs = np.concatenate(c_list, axis=0)
        intrinsics = np.concatenate(k_list, axis=0) if len(k_list) == len(images) else None
    else:
        with torch.no_grad():
            prediction = model.inference(images, **_res_kw)
        # prediction.depth has shape [N, H, W], prediction.conf has shape [N, H, W]
        depths = prediction.depth
        confs = prediction.conf
        intrinsics = getattr(prediction, "intrinsics", None)
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        if isinstance(confs, torch.Tensor):
            confs = confs.cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()

    # DA3 conf uses expp1 activation (exp(x)+1), range ~1-60+
    # Subtract 1.0 so minimum is 0 (same as DA3-streaming does)
    confs = confs - 1.0
    confs = np.clip(confs, 0, None)

    print(f"[DA3 Extractor] Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
    print(f"[DA3 Extractor] Conf range:  [{confs.min():.3f}, {confs.max():.3f}]")
    if device.type == "cuda":
        print(f"[DA3 Extractor] peak VRAM: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)   # own your output directory
    for i, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")

        if os.path.exists(depth_path) and os.path.exists(conf_path):
            continue

        print(f"[{i+1}/{len(images)}] {basename} → depth + conf")
        np.save(depth_path, depths[i])
        np.save(conf_path, confs[i])
        if intrinsics is not None:
            # per-frame predicted K — dense_pose_fusion unprojects DA3 depth with it
            np.save(os.path.join(args.output_dir, stem + "_intrinsics.npy"), intrinsics[i])

    print("[DA3 Extractor] Finished successfully.")

def _np(x):
    if x is None:
        return None
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def window_stamp(identity: dict, spec: dict, i: int) -> str:
    """The stamp of window ``i``: everything its depth depends on — the DA3 identity, the model
    and process_res of the plan, the window's index and its frames (basename + sha256 of each
    file's bytes). sha256 hex of their canonical JSON."""
    paths = spec["windows"][i]
    return repro.sha256_json({"identity": identity, "model_id": spec.get("model_id"),
                              "process_res": spec.get("process_res"), "index": int(i),
                              "frames": [[os.path.basename(p), repro.sha256_file(p)]
                                         for p in paths]})


def _window_stamp_on_disk(path: str):
    try:
        with np.load(path) as z:
            return str(z["stamp"]) if "stamp" in z.files else None
    except Exception:  # noqa: BLE001 — an unreadable window is a mismatching one
        return None


def stale_window_files(output_dir: str, stamps: list) -> tuple:
    """(window files on disk, the reasons each one that is NOT this run's is stale) — plan
    point 44: a window file is reused only when it carries the stamp this run would write for its
    index; a file with no stamp, another stamp, a foreign name or an index beyond the plan is
    stale, and ONE stale file sends ALL of them (the caller deletes every file on disk)."""
    on_disk = sorted(glob.glob(os.path.join(output_dir, "window_*.npz")))
    bad = []
    for path in on_disk:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            k = int(stem.split("_")[1])
        except (IndexError, ValueError):
            bad.append(f"{os.path.basename(path)}: not a window of this plan")
            continue
        if k >= len(stamps):
            bad.append(f"{os.path.basename(path)}: index beyond the plan's {len(stamps)} windows")
        elif _window_stamp_on_disk(path) != stamps[k]:
            bad.append(f"{os.path.basename(path)}: stamp differs (or none)")
    return on_disk, bad


def write_environment_record(output_dir: str) -> str:
    """``<output_dir>/da3_environment.json`` (plan point 27): repro.environment_record(gpu=True)
    — card and driver, torch / CUDA / cuDNN, BLAS core, CPU, library versions, git state of the
    repo and the forks — plus the numerics this process runs DA3 with (deterministic STRICT, TF32
    off) and the fixed autocast dtype. Deterministic content: no time, host or pid."""
    rec = repro.environment_record(gpu=True)
    rec["torch_numerics"] = repro.torch_numerics_record()
    rec["autocast_dtype"] = AUTOCAST_DTYPE
    path = os.path.join(output_dir, "da3_environment.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def _load_reference_views(path: str, spec_sha: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    if doc.get("windows_spec_sha256") != spec_sha:
        return {}
    return {int(k): [int(x) for x in v] for k, v in (doc.get("windows") or {}).items()}


def _save_reference_views(path: str, spec_sha: str, views: dict) -> None:
    doc = {"version": 1, "windows_spec_sha256": spec_sha,
           "windows": {str(k): [int(x) for x in v] for k, v in sorted(views.items())}}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def run_windows(args):
    """One joint DA3 inference per window. The NESTED model aligns its multi-view
    depth to its own metric branch with one least-squares factor per window and
    scales the extrinsics' translations by the same factor (model/da3.py
    _apply_depth_alignment: is_metric = 1), so each window's poses are metric."""
    import time
    with open(args.windows_json) as f:
        spec = json.load(f)
    spec_sha = repro.sha256_json(spec)          # intake.walk.spec_sha256: the plan's identity
    windows = spec["windows"]
    # THIS process must be the DA3 environment the plan was made for (point 27): the launcher
    # recorded it in windows.json from an --identity probe of the same interpreter
    identity = da3_identity(args.model)
    planned = spec.get("da3_environment")
    if planned is not None and repro.canonical_json(planned) != repro.canonical_json(identity):
        diff = sorted(k for k in set(planned) | set(identity) if planned.get(k) != identity.get(k))
        print(f"[DA3 windows] this process is not the DA3 environment windows.json was planned "
              f"for (differs in: {diff}) — refusing to mix windows of two environments",
              flush=True)
        sys.exit(IDENTITY_EXIT)
    # PER-WINDOW STAMPS (point 44): a window file on disk is reused only when EVERY one carries
    # the stamp this run would write; one mismatch and all of them go
    stamps = [window_stamp(identity, spec, i) for i in range(len(windows))]
    on_disk, bad = stale_window_files(args.output_dir, stamps)
    if bad:
        print(f"[DA3 windows] {len(bad)} window file(s) on disk are not this run's "
              f"({'; '.join(bad[:3])}{' …' if len(bad) > 3 else ''}) — deleting ALL "
              f"{len(on_disk)} window file(s) before anything is reused", flush=True)
        for path in on_disk:
            os.unlink(path)
    todo = [i for i in range(len(windows))
            if not os.path.exists(os.path.join(args.output_dir, f"window_{i:04d}.npz"))]
    if not todo:
        print(f"[DA3 windows] all {len(windows)} windows already exist with this run's stamps. "
              f"Skipping.")
        return
    print(f"[DA3 windows] {len(todo)} of {len(windows)} windows need processing")
    ref_path = os.path.join(args.output_dir, REFERENCE_VIEWS_NAME)
    ref_views = _load_reference_views(ref_path, spec_sha)
    model, device = _load_model(args.model)
    # the weights' footprint — the calibration CLI (intake.vram --calibrate) reads it
    torch.cuda.synchronize()
    print(f"[DA3 windows] model loaded: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)
    # the environment these windows are made in (point 27), beside them
    print(f"[DA3 windows] environment → {write_environment_record(args.output_dir)}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    res_kw = {"process_res": int(args.process_res)} if args.process_res else {}
    # the NESTED model computes each frame's MONOCULAR metric depth (its metric
    # branch) and discards it after aligning the multi-view depth to it — kept here
    # as `depth_mono`, the gauge's per-frame instrument, from the SAME inference
    captured = {}
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "_apply_depth_alignment"):
        _orig_align = inner._apply_depth_alignment

        def _capture(output, metric_output):
            captured["mono"] = metric_output.depth.detach().float().cpu().numpy()
            return _orig_align(output, metric_output)

        inner._apply_depth_alignment = _capture
    # THE REFERENCE VIEW DA3 PICKS (point 42): for windows of ≥ 3 frames it reorders the views
    # around argmin of a bf16 balance score; the choice is kept (the model's own) and RECORDED,
    # and a regeneration must pick the same one
    import depth_anything_3.model.dinov2.vision_transformer as _vt
    _orig_select = _vt.select_reference_view

    def _select(x, *a, **k):
        b_idx = _orig_select(x, *a, **k)
        captured.setdefault("ref", []).extend(int(v) for v in b_idx.reshape(-1).tolist())
        return b_idx

    _vt.select_reference_view = _select
    t0 = time.time()
    for n, i in enumerate(todo):
        paths = windows[i]
        captured.clear()
        torch.manual_seed(i)            # a window's draws depend on the window alone
        try:
            with torch.no_grad():
                pred = model.inference(paths, **res_kw)
        except torch.OutOfMemoryError as e:
            # a window that does not fit the card is NOT a crash to decode from a traceback:
            # exit code 3 (intake.vram.OOM_EXIT) and the caller FAILS — the window size is the
            # card table's, never lowered to fit (plan point 4)
            print(f"[DA3 windows] window {i}: {len(paths)} frame(s) at process_res {res_kw.get('process_res')} "
                  f"do NOT fit the card: {str(e).splitlines()[0][:160]}", flush=True)
            sys.exit(OOM_EXIT)
        refs = list(captured.get("ref", []))
        if i in ref_views and ref_views[i] != refs:
            print(f"[DA3 windows] window {i}: DA3 picked reference view(s) {refs}, the window was "
                  f"recorded with {ref_views[i]} — the regeneration does not reproduce it "
                  f"(declared; nothing written)", flush=True)
            sys.exit(REF_VIEW_EXIT)
        ref_views[i] = refs
        frames = np.array([int("".join(ch for ch in os.path.splitext(os.path.basename(p))[0]
                                       if ch.isdigit())) for p in paths], dtype=np.int64)
        conf = np.clip(_np(pred.conf) - 1.0, 0, None)       # expp1 activation, as above
        ext = _np(pred.extrinsics)
        out = os.path.join(args.output_dir, f"window_{i:04d}.npz")
        tmp = out + ".tmp.npz"
        extra = {}
        depth = _np(pred.depth).astype(np.float32)
        mono = captured.get("mono")
        if mono is not None:
            mono = mono.reshape((-1,) + mono.shape[-2:])
            if mono.shape == depth.shape:
                extra["depth_mono"] = mono.astype(np.float32)
            else:
                print(f"[DA3 windows] window {i}: mono depth {mono.shape} ≠ depth "
                      f"{depth.shape} — not stored")
        # the window's VRAM footprint (the calibration CLI reads the 2-frame window's peak)
        print(f"[DA3 windows] window {i}: {len(paths)} frame(s) at process_res {res_kw.get('process_res')} → "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
        torch.cuda.reset_peak_memory_stats()
        np.savez(tmp, frames=frames, depth=depth, **extra,
                 conf=conf.astype(np.float32), extrinsics=ext.astype(np.float64),
                 intrinsics=_np(pred.intrinsics).astype(np.float64),
                 scale_factor=np.float64(pred.scale_factor if pred.scale_factor is not None
                                         else np.nan),
                 is_metric=np.int64(pred.is_metric),
                 ref_views=np.asarray(refs, dtype=np.int64),
                 stamp=np.asarray(stamps[i]))
        os.replace(tmp, out)
        _save_reference_views(ref_path, spec_sha, ref_views)
        el = time.time() - t0
        print(f"[DA3 windows] window {n + 1}/{len(todo)} ({len(paths)} frames, "
              f"is_metric={int(pred.is_metric)}, scale {pred.scale_factor}, reference view(s) "
              f"{refs}) — {el:.0f}s, ~{el / (n + 1) * (len(todo) - n - 1):.0f}s left", flush=True)
    print(f"[DA3 windows] peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("[DA3 windows] Finished successfully.")


if __name__ == "__main__":
    main()
