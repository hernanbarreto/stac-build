"""I1 prerequisite — the session camera's intrinsics, MEASURED once before the parallax.

The parallax reference is the pure ROTATION of the camera that shot the video:
H = K·R·K⁻¹ with K the camera's, only R fitted per frame. A focal fitted per frame
(or a free K) is not the camera: it bends to absorb translation parallax — measured
on pccr 2026-08-24, a per-frame focal ran to its bound on half the frames and the
free-K model settled at fx≈1e-4 px, skew≈−15, so the reading was decided by a bound
or by where the solver stopped, not by the tracks.

The instrument: ONE DA3 multi-view inference over ``focal_probe_frames`` usable frames
spread uniformly over the whole video (intake I0's usable list), its per-frame K on
its grid carried to native pixels by F0's exact mapping (DA3 resizes the full frame),
and the per-parameter MEDIAN. The spread across the frames is reported (a lens that
zooms shows it there).

Output: ``<session>/intake/focal_probe.json`` (+ the window in ``intake/da3_focal/``).
GPU (DA3); the probe is reused on its STAMP (docs/plan_determinismo.md points 66 / 70,
2026-10-07 — ``intake.focal_probe``): the sha256 of every probe frame's bytes, the intake + DA3
extractor code, the spec (the committed layout, resolution, model, the card table's sizing, the
DA3 identity: weights, card, dtype, torch / CUDA / cuDNN, libraries), the probe parameters and
the CPU environment — never by the window paths alone (frames re-extracted under the same names
used to get the old video's K; a copied session re-measured K on the GPU). The product names
its frames relative to the session and carries no epoch, time or absolute path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from intake import focal_probe as FP

FOCAL_NAME = "focal_probe.json"
DIRNAME = "da3_focal"
FOCAL_VERSION = FP.FOCAL_VERSION       # 2: session-relative spec + stamp as the reuse key
PROVENANCE = "tool_measured"
LOG_TAG = "[intake.focal]"
MAD_TO_SIGMA = 1.0 / 0.6744897501960817


class FocalError(RuntimeError):
    """The intrinsics cannot be measured — with the exact reason."""


def probe_files(quality: Dict[str, Any], n: int) -> List[str]:
    """``n`` usable frames spread uniformly over the video (basenames, in order)."""
    from intake.parallax import _frame_table
    rows = [r for r in _frame_table(quality) if r["usable"]]
    if len(rows) < 2:
        raise FocalError(f"{len(rows)} usable frame(s) — the probe needs at least two")
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, min(int(n), len(rows)))).astype(int))
    return [rows[i]["file"] for i in idx]


def native_intrinsics(K_grid: np.ndarray, grid_wh, native_wh) -> np.ndarray:
    """(S,3,3) DA3 grid K → native px (DA3 resizes the whole frame)."""
    from precision.camera import K_grid_to_native, grid_full_frame_resize
    g = grid_full_frame_resize(native_wh[0], native_wh[1], grid_wh[0], grid_wh[1], "da3_focal")
    return np.stack([K_grid_to_native(K, g) for K in np.asarray(K_grid, np.float64)])


def summarise(K_native: np.ndarray) -> Dict[str, Any]:
    """The median K and each parameter's robust spread (% of its median)."""
    p = np.stack([K_native[:, 0, 0], K_native[:, 1, 1], K_native[:, 0, 2], K_native[:, 1, 2]], 1)
    med = np.median(p, axis=0)
    spread = MAD_TO_SIGMA * np.median(np.abs(p - med), axis=0) / np.abs(med) * 100.0
    K = np.array([[med[0], 0.0, med[2]], [0.0, med[1], med[3]], [0.0, 0.0, 1.0]])
    return {"K": K.tolist(), "fx": float(med[0]), "fy": float(med[1]), "cx": float(med[2]),
            "cy": float(med[3]),
            "spread_pct": {k: float(v) for k, v in zip(("fx", "fy", "cx", "cy"), spread)},
            "per_frame": p.tolist()}


def probe_windows(items: List[str], n_win: int) -> Tuple[List[List[str]], List[str]]:
    """The probe's windows of at most ``n_win`` items, in order (``range(0, N, n_win)`` — the
    layout of every validated run: 16 at 840 → [16]; 16 at 1932 → [6, 6, 4]). A lone last item
    (which measures no multi-view K) is paired by taking one item from the previous window when
    that window keeps two or more; otherwise it is DROPPED and returned — no window ever exceeds
    the layout (the old fold rule made one window of n_win + 1, which could not fit)."""
    n = max(2, int(n_win))
    windows = [list(items[a:a + n]) for a in range(0, len(items), n)]
    dropped: List[str] = []
    if len(windows) > 1 and len(windows[-1]) == 1:
        if len(windows[-2]) >= 3:
            windows[-1].insert(0, windows[-2].pop())
        else:
            dropped = windows.pop()
    return windows, dropped


def run_focal_probe(session_dir: Path, quality: Dict[str, Any], pcfg, *, python: str,
                    log: Callable = print,
                    cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Measure (or reuse) the session K; returns the probe document."""
    from intake.walk import da3_identity, da3_process_res
    import da3_weights
    import repro
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    from intake.vram import OOM_EXIT, window_size
    files = probe_files(quality, pcfg.focal_probe_frames)
    process_res = da3_process_res(pcfg.focal_probe_res, frames_dir)
    model_id = str(pcfg.focal_probe_model)
    native_wh = (int(quality["native_w"]), int(quality["native_h"]))
    # the DA3 identity of the extracting interpreter (weights, code, torch, card) and the window
    # the CARD holds at this (native) resolution, from the committed card table — zaragoza
    # 2026-10-04: 16 frames of 1080p in one window were 57 GB; the probe's frames go in as many
    # windows as the table allows (docs/plan_determinismo.md points 5, 13, 21)
    ident = da3_identity(python, model_id, log)
    # THE LAYOUT IS A COMMITTED CONSTANT per (model, process_res) — docs/plan_determinismo.md
    # point 64: the probe's windows used to be sized by the card's memory (and halved on an OOM),
    # so the session K, the rotation reference and every keyframe depended on the card and on its
    # co-tenants. The card table only says whether THIS card holds that layout: if not, the probe
    # FAILS (another layout is another K) — it is never split smaller.
    import card_table
    layout = card_table.probe_window_frames(model_id, process_res)
    n_win = int(layout["window_frames"])
    sizing = window_size(model_id, process_res, native_wh, requested=n_win,
                         margin_frac=float(pcfg.vram_margin_frac), card_key=str(ident["card"]),
                         log=log)
    if int(sizing["window_frames"]) < n_win:
        raise FocalError(f"the committed focal-probe layout at process_res {process_res} is windows "
                         f"of {n_win} frames; {ident['card']} holds {sizing['window_frames']} "
                         f"(predicted peak of {n_win}: {sizing['weights_gib'] + sizing['per_token_gib'] * sizing['tokens_per_frame'] * n_win:.1f} GiB "
                         f"of {sizing['card_total_gib']:.1f}) — the probe is never split smaller "
                         f"(another layout is another K); run it on a card that holds the layout")
    out = session_dir / "intake" / FOCAL_NAME
    wdir = session_dir / "intake" / DIRNAME
    server_dir = Path(__file__).resolve().parent.parent
    env = da3_weights.hf_env(repro.deterministic_env())
    windows, dropped = probe_windows(list(files), n_win)
    if dropped:
        log(f"{LOG_TAG} ⚠ {len(dropped)} probe frame(s) left out: a lone last frame measures no "
            f"multi-view K and no window may exceed the layout's {n_win} frames ({dropped})")
    # the PRODUCT's spec names its frames relative to the session (point 70); the reuse key is
    # the stamp of everything the K depends on (point 66) — the frames' bytes included
    spec = FP.probe_spec(windows, process_res=process_res, model_id=model_id,
                         probe_layout=layout, window_sizing=sizing, da3_environment=ident)
    stamp = FP.probe_stamp(session_dir, spec, pcfg)
    if out.exists():
        try:
            doc = json.loads(out.read_text())
        except ValueError:
            doc = None
        why = FP.probe_reusable(doc, stamp)
        if not why:
            log(f"{LOG_TAG} reusing {out} (stamp {stamp['sha256'][:12]}: same frames' bytes, "
                f"layout, resolution, model, DA3 identity, code and environment): fx "
                f"{doc['fx']:.2f} fy {doc['fy']:.2f} cx {doc['cx']:.2f} cy {doc['cy']:.2f} px")
            return doc
        log(f"{LOG_TAG} {out.name} not reused — " + "; ".join(why[:6]))
    wdir.mkdir(parents=True, exist_ok=True)
    for p in wdir.glob("window_*.npz"):
        p.unlink()
    # the extractor's input (transient, deleted after the probe): absolute paths
    spec_path = wdir / "windows.json"
    spec_path.write_text(json.dumps({**spec, "windows": FP.extractor_windows(spec, session_dir)}))
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--image_dir", str(frames_dir),
           "--output_dir", str(wdir), "--model", model_id,
           "--process_res", str(process_res), "--windows_json", str(spec_path)]
    log(f"{LOG_TAG} DA3 {model_id} on {len(files)} frames spread over the video in {len(windows)} "
        f"window(s) of ≤ {n_win} (process_res {process_res}) → the session K")
    repro.require_exclusive_gpu(log=log)                # point 4: nothing else on the card
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env)
    for line in proc.stdout:
        if line.strip():
            log(line.strip())
        if cancelled is not None and cancelled():
            proc.terminate()
            raise FocalError("cancelled")
    proc.wait()
    if proc.returncode == OOM_EXIT:
        # never halved (point 4): another window split is another probe
        raise FocalError(f"a focal-probe window of {n_win} frame(s) did not fit "
                         f"{ident['card']} although the card table sizes it to — re-measure the "
                         f"card's entry or free the card")
    if proc.returncode != 0:
        raise FocalError(f"DA3 focal probe exited with code {proc.returncode}")
    Ks, depth = [], None
    for i in range(len(windows)):
        with np.load(wdir / f"window_{i:04d}.npz") as z:
            Ks.append(np.asarray(z["intrinsics"], np.float64)); depth = z["depth"]
    K_grid = np.concatenate(Ks, 0)
    K_nat = native_intrinsics(K_grid, (depth.shape[-1], depth.shape[-2]), native_wh)
    from intake.quality import intake_epochs
    doc = {"version": FOCAL_VERSION, "provenance": PROVENANCE, **intake_epochs(),
           "spec": spec, "stamp": stamp, "frames": list(files),
           "native_wh": list(native_wh), "grid_wh": [int(depth.shape[-1]), int(depth.shape[-2])],
           **summarise(K_nat)}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out)
    for p in wdir.glob("window_*.npz"):          # K is in the json; the depth (389 MB at 1080p) is dead
        p.unlink()
    spec_path.unlink(missing_ok=True)            # the extractor's input held absolute paths
    s = doc["spread_pct"]
    log(f"{LOG_TAG} K: fx {doc['fx']:.2f} fy {doc['fy']:.2f} cx {doc['cx']:.2f} cy "
        f"{doc['cy']:.2f} px (spread fx {s['fx']:.2f} %, cx {s['cx']:.2f} %) → {out}")
    return doc


def default_probe(python: Optional[str] = None) -> Callable:
    """The production probe (DA3 in ``python``, default this interpreter)."""
    py = python or sys.executable

    def probe(session_dir, quality, pcfg, log, cancelled) -> np.ndarray:
        return np.asarray(run_focal_probe(session_dir, quality, pcfg, python=py, log=log,
                                          cancelled=cancelled)["K"], np.float64)
    return probe
