"""The focal probe's reuse key and session-relative spec (docs/plan_determinismo.md points 66
and 70, 2026-10-07). :mod:`intake.focal` runs DA3 with them; the window LAYOUT itself is point
64's and lives in ``card_table.probe_window_frames`` + ``intake.focal.probe_windows`` (a committed
constant per model and resolution, never exceeded, never halved).

Point 66 / 70 — the probe used to be reused when ``{absolute window paths, process_res,
model_id}`` matched: frames re-extracted under the same names got the old video's K, a copied
session re-measured K on the GPU. The spec now names frames relative to the session, and the
reuse key is :func:`probe_stamp`: the sha256 of every probe frame's bytes, the intake and DA3
extractor code, the spec (window layout, resolution, model, the card table's sizing, the DA3
identity: weights, card, dtype, torch / CUDA / cuDNN, libraries), the probe's parameters and
the CPU environment. Reuse only on a full match (:func:`probe_reusable`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from intake import stamps as St

FOCAL_VERSION = 2            # 2: session-relative spec + repro stamp as the reuse key (2026-10-07)


def probe_spec(windows: Sequence[Sequence[str]], *, process_res: int, model_id: str,
               probe_layout: Mapping[str, Any], window_sizing: Mapping[str, Any],
               da3_environment: Mapping[str, Any], frames_dirname: str = St.FRAMES_DIRNAME
               ) -> Dict[str, Any]:
    """The probe's spec as the PRODUCT records it: the windows (``intake.focal.probe_windows``,
    frame names or paths) as session-relative paths (``frames/000123.jpg``), resolution, model,
    the committed layout (``card_table.probe_window_frames``), the card table's sizing with every
    number behind it and the DA3 identity of the extracting interpreter."""
    rel = [[f"{frames_dirname}/{Path(f).name}" for f in x] for x in windows]
    return {"windows": rel, "process_res": int(process_res), "model_id": str(model_id),
            "probe_layout": dict(probe_layout), "window_sizing": dict(window_sizing),
            "da3_environment": dict(da3_environment),
            "sizes": [len(x) for x in rel]}


def extractor_windows(spec: Mapping[str, Any], session_dir: os.PathLike) -> List[List[str]]:
    """The spec's windows as ABSOLUTE paths for ``extract_da3_depth.py --windows_json`` (the
    extractor's transient input, not a product)."""
    return [[str(St.in_session(p, session_dir)) for p in x] for x in spec["windows"]]


def probe_params(pcfg: Any) -> Dict[str, Any]:
    """The intake.parallax parameters the probe depends on."""
    return {"focal_probe_frames": int(pcfg.focal_probe_frames),
            "focal_probe_res": pcfg.focal_probe_res,
            "focal_probe_model": str(pcfg.focal_probe_model),
            "vram_margin_frac": float(pcfg.vram_margin_frac)}


def probe_stamp(session_dir: os.PathLike, spec: Mapping[str, Any], pcfg: Any, *,
                run_config_sha256: Optional[str] = None,
                environment: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The reuse key of ``focal_probe.json`` (point 66): every probe frame's bytes (keyed by
    its session-relative path), the intake + DA3 extractor code, and the config sections —
    the spec (layout, resolution, model, card-table sizing, DA3 identity: weights sha, card,
    dtype, torch / CUDA / cuDNN, libraries), the probe parameters, the CPU environment
    (:func:`intake.stamps.cpu_environment_record` unless given) and the frozen run
    configuration's sha256 when the caller has one."""
    session_dir = Path(session_dir)
    inputs = {p: St.in_session(p, session_dir) for x in spec["windows"] for p in x}
    env = dict(environment) if environment is not None else St.cpu_environment_record()
    return St.step_stamp(inputs, {"spec": dict(spec), "params": probe_params(pcfg),
                                  "environment": env,
                                  "run_config_sha256": run_config_sha256})


def probe_reusable(doc: Any, stamp_now: Mapping[str, Any]) -> List[str]:
    """[] when ``doc`` (a focal_probe.json) is the product of ``stamp_now``; otherwise what
    differs (an older version, a missing or different stamp — each named)."""
    if not isinstance(doc, dict):
        return ["focal_probe.json is not a JSON object"]
    if doc.get("version") != FOCAL_VERSION:
        return [f"focal_probe.json is version {doc.get('version')!r}, this code writes "
                f"{FOCAL_VERSION}"]
    return St.stamp_differences(doc.get("stamp"), stamp_now)
