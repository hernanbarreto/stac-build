"""The masks of every observation a published point REPRESENTS (audit 2026-10-23, phase 3e).

Since the owner fusion (precision/owner_fusion.py) a surface element is published ONCE, by its
owner observation; the other keyframes' observations of it are CONSUMED and live in
``rejected_points.npz`` (reason ``fused``) with their winner. The mask projection used to read
only the owner's own birth pixel: an object masked in keyframe B but owned by a point born in
keyframe A, where SAM saw nothing, lost its label — the fusion had moved the pixel the mask
indexed. Here every published point gets its LOOKUP PIXELS: its own, plus the pixel of every
consumed observation whose chain of winners ends at it (``member_lookups``). The projection
reads the masks at all of them; the smaller-area rule decides as before.

Grids: the fusion's pixels are the BEND's (the undistorted native grid, origins.npz
``pixel_u_und / pixel_v_und``); the masks index the RECORD grid through the lens — each member
pixel is carried like the owner's own (``corrected_cloud.to_record_grid``) when the two grids
differ, exactly as the publish did for the owner's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REASON_FUSED = 10


def _keys(frame: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """One int64 key per (frame, row, col)."""
    return (np.asarray(frame, np.int64) << 32) | (np.asarray(row, np.int64).clip(0, 65535) << 16) \
        | np.asarray(col, np.int64).clip(0, 65535)


def member_lookups(output_dir: Path, frame_global: np.ndarray, u_und: np.ndarray, v_und: np.ndarray,
                   max_hops: int = 8, log=print) -> Optional[Dict[str, Any]]:
    """For the published points (``frame_global``, bend-grid pixel ``u_und / v_und``): the consumed
    observations each represents. Returns {point_index, frame, row, col (bend grid), report} or
    None when the session holds no fusion record."""
    rp = Path(output_dir) / "rejected_points.npz"
    if not rp.exists():
        return None
    with np.load(rp) as z:
        if "winner_frame" not in z.files or "reason" not in z.files:
            return None
        reason = np.asarray(z["reason"])
        m = reason == REASON_FUSED
        fg = np.asarray(z["frame_global"], np.int64)[m]; pr = np.asarray(z["pixel_row"], np.int64)[m]
        pc = np.asarray(z["pixel_col"], np.int64)[m]
        wf = np.asarray(z["winner_frame"], np.int64)[m]; wr = np.asarray(z["winner_row"], np.int64)[m]
        wc = np.asarray(z["winner_col"], np.int64)[m]
    if fg.size == 0:
        return {"point_index": np.zeros(0, np.int64), "frame": np.zeros(0, np.int64), "row": np.zeros(0, np.int64),
                "col": np.zeros(0, np.int64), "report": {"consumed": 0, "resolved": 0, "unresolved": 0}}
    own = _keys(frame_global, v_und, u_und)
    order = np.argsort(own, kind="stable")
    own_sorted = own[order]
    cons_key = _keys(fg, pr, pc)
    cons_order = np.argsort(cons_key, kind="stable")
    cons_sorted = cons_key[cons_order]
    win = _keys(wf, wr, wc)
    # follow the winners: an owner (in the cloud) ends the chain; a consumed winner continues
    target = win.copy()
    resolved = np.zeros(len(win), bool)
    idx = np.full(len(win), -1, np.int64)
    for _hop in range(int(max_hops)):
        pos = np.searchsorted(own_sorted, target)
        pos_c = np.clip(pos, 0, len(own_sorted) - 1)
        hit = (~resolved) & (own_sorted[pos_c] == target)
        idx[hit] = order[pos_c[hit]]
        resolved |= hit
        if resolved.all():
            break
        # not an owner: the winner is itself consumed — its own winner is the next target
        q = np.searchsorted(cons_sorted, target)
        q_c = np.clip(q, 0, len(cons_sorted) - 1)
        step = (~resolved) & (cons_sorted[q_c] == target)
        if not step.any():
            break
        target[step] = win[cons_order[q_c[step]]]
    n_un = int((~resolved).sum())
    if n_un:
        log(f"[members] {n_un:,} consumed observation(s) of {len(win):,} lead to no published point within "
            f"{max_hops} hops (their winner left the cloud) — no lookup pixel for them")
    keep = resolved
    return {"point_index": idx[keep], "frame": fg[keep], "row": pr[keep], "col": pc[keep],
            "report": {"consumed": int(len(win)), "resolved": int(keep.sum()), "unresolved": n_un}}


def lookup_pixels(output_dir: Path, frame_global: np.ndarray, pixel_row: np.ndarray, pixel_col: np.ndarray,
                  log=print) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """The projection's lookup set on the RECORD grid: (point_index, frame, row, col, report) — the
    published points' own pixels first (as the PLY carries them), then every member pixel the
    fusion record names, carried through the lens when the publish carried the owners'."""
    import precision.provenance as PV
    from precision.camera import CAMERA_JSON_NAME, load_camera_json, undistort_maps
    n = len(frame_global)
    base_idx = np.arange(n, dtype=np.int64)
    rep: Dict[str, Any] = {"members": None}
    try:
        org = PV.load_origins(Path(output_dir))
    except Exception as e:          # a cloud without origins v2: the owner's pixel alone, declared
        rep["members"] = {"taken": False, "reason": f"origins not readable: {e}"}
        return base_idx, np.asarray(frame_global, np.int64), np.asarray(pixel_row, np.int64), np.asarray(pixel_col, np.int64), rep
    if int(org.get("version", 1)) < 2 or "pixel_u_und" not in org:
        rep["members"] = {"taken": False, "reason": "origins v1 (no undistorted pixel): the owner's pixel alone"}
        return base_idx, np.asarray(frame_global, np.int64), np.asarray(pixel_row, np.int64), np.asarray(pixel_col, np.int64), rep
    u_und = np.asarray(org["pixel_u_und"], np.int64); v_und = np.asarray(org["pixel_v_und"], np.int64)
    mem = member_lookups(Path(output_dir), np.asarray(frame_global, np.int64), u_und, v_und, log=log)
    if mem is None or len(mem["point_index"]) == 0:
        rep["members"] = {"taken": False, "reason": "no fusion record (rejected_points.npz with winners)"}
        return base_idx, np.asarray(frame_global, np.int64), np.asarray(pixel_row, np.int64), np.asarray(pixel_col, np.int64), rep
    # the member pixels are the bend's: onto the record grid the way the owners' went
    carried = not (np.array_equal(u_und, np.asarray(pixel_col, np.int64)) and np.array_equal(v_und, np.asarray(pixel_row, np.int64)))
    mr, mc = mem["row"], mem["col"]
    if carried:
        from precision.corrected_cloud import to_record_grid
        cam = load_camera_json(Path(output_dir) / CAMERA_JSON_NAME)
        maps = undistort_maps(cam)
        with np.load(Path(output_dir) / "omega_run" / "results_output" / f"frame_{int(frame_global[0])}.npz") as z:
            rec_hw = tuple(int(x) for x in np.asarray(z["depth"]).shape[:2])
        mr, mc = to_record_grid(mc, mr, cam, maps, rec_hw)
    rep["members"] = {"taken": True, "carried_through_lens": bool(carried), **mem["report"],
                      "lookups_per_point": float(1.0 + len(mem["point_index"]) / max(n, 1))}
    idx = np.concatenate([base_idx, mem["point_index"]])
    fr = np.concatenate([np.asarray(frame_global, np.int64), mem["frame"]])
    rows = np.concatenate([np.asarray(pixel_row, np.int64), np.asarray(mr, np.int64)])
    cols = np.concatenate([np.asarray(pixel_col, np.int64), np.asarray(mc, np.int64)])
    log(f"[members] {len(mem['point_index']):,} member pixel(s) of {mem['report']['consumed']:,} consumed observations "
        f"join the {n:,} owners' own: {rep['members']['lookups_per_point']:.2f} lookup pixels per point"
        + (" (carried through the lens onto the record grid)" if carried else ""))
    return idx, fr, rows, cols, rep
