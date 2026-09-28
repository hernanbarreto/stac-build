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
GPU (DA3); the probe is reused while its frames, resolution and model are unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

FOCAL_NAME = "focal_probe.json"
DIRNAME = "da3_focal"
FOCAL_VERSION = 1
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


def run_focal_probe(session_dir: Path, quality: Dict[str, Any], pcfg, *, python: str,
                    log: Callable = print,
                    cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Measure (or reuse) the session K; returns the probe document."""
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    files = probe_files(quality, pcfg.focal_probe_frames)
    spec = {"windows": [[str(frames_dir / f) for f in files]],
            "process_res": int(pcfg.focal_probe_res), "model_id": str(pcfg.focal_probe_model)}
    out = session_dir / "intake" / FOCAL_NAME
    wdir = session_dir / "intake" / DIRNAME
    if out.exists():
        doc = json.loads(out.read_text())
        if doc.get("version") == FOCAL_VERSION and doc.get("spec") == spec:
            log(f"{LOG_TAG} reusing {out} (same {len(files)} frames, resolution and model): "
                f"fx {doc['fx']:.2f} fy {doc['fy']:.2f} cx {doc['cx']:.2f} cy {doc['cy']:.2f} px")
            return doc
    wdir.mkdir(parents=True, exist_ok=True)
    for p in wdir.glob("window_*.npz"):
        p.unlink()
    spec_path = wdir / "windows.json"
    spec_path.write_text(json.dumps(spec))
    server_dir = Path(__file__).resolve().parent.parent
    cmd = [str(python), str(server_dir / "extract_da3_depth.py"), "--image_dir", str(frames_dir),
           "--output_dir", str(wdir), "--model", spec["model_id"],
           "--process_res", str(spec["process_res"]), "--windows_json", str(spec_path)]
    log(f"{LOG_TAG} DA3 {spec['model_id']} on {len(files)} frames spread over the video "
        f"(process_res {spec['process_res']}) → the session K")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env)
    for line in proc.stdout:
        if line.strip():
            log(line.strip())
        if cancelled is not None and cancelled():
            proc.terminate()
            raise FocalError("cancelled")
    proc.wait()
    if proc.returncode != 0:
        raise FocalError(f"DA3 focal probe exited with code {proc.returncode}")
    with np.load(wdir / "window_0000.npz") as z:
        K_grid, depth = z["intrinsics"], z["depth"]
    native_wh = (int(quality["native_w"]), int(quality["native_h"]))
    K_nat = native_intrinsics(K_grid, (depth.shape[-1], depth.shape[-2]), native_wh)
    from intake.quality import read_session_epochs
    doc = {"version": FOCAL_VERSION, "provenance": PROVENANCE,
           **read_session_epochs(session_dir), "spec": spec, "frames": files,
           "native_wh": list(native_wh), "grid_wh": [int(depth.shape[-1]), int(depth.shape[-2])],
           **summarise(K_nat)}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out)
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
