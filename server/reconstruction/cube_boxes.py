"""The cubes of the cut, as boxes in the scene the viewer shows (USER 2026-11-06: "me gustaría ver los chunks como
cubos en la escena, y que se puedan apagar y prender, para entender cómo se separó la escena").

The cut (reconstruction/cubes.py, intake/cubes.json) is made in the structure probe's own frame: boxes of space in its
gravity frame (a plan of 2026-11-06 morning: k-d cells, every cube two neighbour blocks — drawn as its cells' union). A
cube is carried into the session's live frame by the similarity that best maps the probe's keyframe cameras onto the
live ones (output/camera_poses.txt: Umeyama over the keyframes' centres, its residual reported), then through the
sealed floor transform like the camera markers. A cube of space is drawn as the box it is, its own height included (a
plan of k-d cells spans the cloud vertically: the octree's bounds along the up axis — far points made those columns
18 m tall on pccr and they seemed to hold each other). Nothing here is a measurement."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _umeyama(A: np.ndarray, B: np.ndarray):
    """(s, R, t) with B ≈ s R A + t (least squares over the rows)."""
    ma, mb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((A - ma).var(0).sum(), 1e-18))
    return s, R, mb - s * R @ ma


def cube_boxes(scan_dir: Path) -> Optional[Dict]:
    """{"cubes": [{id, center, half_extents, rotation (columns = the box axes), color, keyframes, views, blocks}],
    "fit": {...}} in the viewer's display frame, or None when the session has no cube plan."""
    scan_dir = Path(scan_dir)
    out = scan_dir / "output"
    cp, npz = scan_dir / "intake" / "cubes.json", scan_dir / "intake" / "structure_probe.npz"
    poses_p, frames_p = out / "camera_poses.txt", out / "camera_frames.txt"
    if not (cp.exists() and poses_p.exists() and frames_p.exists()):
        return None
    plan = json.loads(cp.read_text())
    if "kf_centres" not in plan and not npz.exists():
        return None
    sel = json.loads((scan_dir / "frames" / "selected_frames.json").read_text())
    files = sel if isinstance(sel, list) else sel.get("selected_files", [])
    kf_no = [int(Path(f).stem) for f in sorted(files)]
    if len(kf_no) != int(plan.get("n_keyframes", -1)):
        return None                                   # the plan is of another keyframe set
    if "kf_centres" in plan:                           # the keyframes' centres in the probe frame, with the plan
        C_probe = np.asarray(plan["kf_centres"], np.float64)
    else:
        with np.load(npz) as z:
            C_probe = np.asarray(z["c2w"], np.float64)[np.asarray(plan["kf_slot"])][:, :3, 3]
    poses = np.loadtxt(poses_p).reshape(-1, 4, 4)
    nums = [int(float(x)) for x in frames_p.read_text().split()]
    live = {n: p for n, p in zip(nums, poses)}
    have = [k for k, n in enumerate(kf_no) if n in live]
    if len(have) < 3:
        return None
    C_live = np.stack([live[kf_no[k]][:3, 3] for k in have])
    s, R, t = _umeyama(C_probe[have], C_live)
    res = np.linalg.norm(s * C_probe[have] @ R.T + t - C_live, axis=1)
    # the viewer's display frame: the sealed floor transform, as the camera markers
    from segmentation.floor_frame import sealed_floor_transform
    fs, fR, ft = sealed_floor_transform(out, tag="Viewer") or (1.0, np.eye(3), np.zeros(3))
    S_all, R_all, t_all = fs * s, fR @ R, fs * (fR @ t) + ft          # probe → display
    G = np.asarray(plan["gravity_frame"], np.float64)                   # rows e1, e2, up (probe frame)
    axes = R_all @ G.T                                                  # columns: e1, e2, up in the display frame
    # the vertical span: the octree's bounds (cloud frame → display), along the mapped up axis
    lo_up = hi_up = None
    meta = out / "potree" / "metadata.json"
    if meta.exists():
        bb = json.loads(meta.read_text()).get("boundingBox") or {}
        if "min" in bb and "max" in bb:
            mn, mx = np.asarray(bb["min"], np.float64), np.asarray(bb["max"], np.float64)
            corners = np.array([[x, y, zz] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for zz in (mn[2], mx[2])])
            disp = fs * corners @ fR.T + ft
            up = axes[:, 2] / np.linalg.norm(axes[:, 2])
            h = disp @ up
            lo_up, hi_up = float(h.min()), float(h.max())
    cells, blocks = plan.get("cells"), plan.get("blocks")
    cubes: List[Dict] = []
    n = max(1, len(plan["cubes"]))
    for ci, cb in enumerate(plan["cubes"]):
        if "lo" in cb:                                   # a cube of space (cut by what the views see)
            lo, hi = np.asarray(cb["lo"], np.float64), np.asarray(cb["hi"], np.float64)
        else:                                            # a pair of k-d cells (the cut of 2026-11-06 morning)
            lo = np.min([cells[blocks[b]["cell"]]["lo"] for b in cb["blocks"]], axis=0)
            hi = np.max([cells[blocks[b]["cell"]]["hi"] for b in cb["blocks"]], axis=0)
        c_g = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * S_all
        center = S_all * (R_all @ (G.T @ c_g)) + t_all
        if lo_up is not None and "lo" not in cb:          # a floor-plan cell spans the cloud; a cube of space is its own box
            up = axes[:, 2] / np.linalg.norm(axes[:, 2])
            h0 = float(center @ up)
            center = center + up * (0.5 * (lo_up + hi_up) - h0)
            half[2] = 0.5 * (hi_up - lo_up)
        rgb = np.array(__import__("colorsys").hsv_to_rgb(ci / n, 0.85, 0.95)) * 255
        cubes.append({"id": ci, "center": center.tolist(), "half_extents": half.tolist(),
                      "rotation": (axes / np.linalg.norm(axes, axis=0)).tolist(),
                      "color": "#%02x%02x%02x" % tuple(int(v) for v in rgb),
                      "keyframes": len(cb["keyframes"]), "views": int(cb.get("omega_views", cb.get("views", 0))),
                      "units": len(cb.get("units", [])), "size_m": cb.get("size_m")})
    return {"cubes": cubes, "fit": {"keyframes": len(have), "scale": s, "residual_median_m": float(np.median(res)),
                                    "residual_max_m": float(res.max())}}
