# STAC-Builder — which keyframes the VLM sees: the fewest that SEE the whole scene.
#
# USER 2026-09-16. It used to be `autoprompt.understand_sample: 8` — eight
# keyframes picked by `np.linspace` over the list, whatever the walk. Eight for
# 80 keyframes, eight for 216, eight for 2000: the same place walked twenty-five
# times more slowly still summarised in eight images.
#
# And this is not one gate among many. In the SIMPLE pipeline these frames are
# the ONLY thing the VLM looks at, and the phrases it writes become the SAM3
# prompts one for one. What is not in these frames is never named, and what is
# never named is never segmented, never measured and never corrected. pccr's
# main door was in none of the eight — *"puertas, ventanas, paredes, pisos,
# techos, columnas, eso es estructural, debe estar"*.
#
# ── THE ALGORITHM, IN THE USER'S WORDS ──────────────────────────────────────
#
#   *"empezando por el primer kf, su posición, matemática pura, posición de
#    cámara, intrínsecas de cámara, con rayo, tenemos cobertura de nube,
#    generamos cubo de cobertura, paso al siguiente mismo cálculo, si hay
#    superposición de sector, se saltea, paso al próximo que ve lo que el
#    anterior no ve, así sucesivamente"*
#
# Keyframe 0 is kept and what it SEES is marked covered. Then each keyframe in
# walk order: project the cloud through its pose and intrinsics, ask which
# voxels it sees, and keep it only when most of that is new.
#
# SEES, not WROTE. The first attempt used each keyframe's own points — what it
# CONTRIBUTED to the cloud — and got 215 of 216, because a moving camera always
# writes some fresh geometry even while looking at the same wall. Two adjacent
# keyframes facing one wall must cover the same thing, and only the frustum
# says so.
#
# ── THE ONE DECISION ────────────────────────────────────────────────────────
#
# *"todos van a aportar tal vez un poco de lo que el otro no vio, pero va a
#  aportar mucho de lo que el otro vio"* — a keyframe is kept when most of what
# it sees is new. Without it every frame qualifies on a sliver of fresh scene
# and the walk stops being selective at all.
#
# Set to 50 % after measuring pccr's four options, and the measurement is the
# point: the TOTAL barely moves (68 / 67 / 67 / 71 for 10 / 25 / 50 / 75 %),
# because what fixes the count is covering the scene, not the tolerance. All it
# moves is the split between the two passes — 9+59 at 10 %, 16+51 at 50 %. So
# this number is not load-bearing, and that is worth knowing before anyone
# spends time tuning it.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

MAX_OVERLAP = 0.50          # USER 2026-09-16, after measuring the four options
VOXEL_M = 0.10              # physical size at which two points are the same scene


def _load_cloud(output_dir: Path):
    for name in ("cleaned_cloud.ply", "corrected_cloud.ply"):
        p = Path(output_dir) / name
        if not p.is_file():
            continue
        try:
            from segmentation.pipeline import _load_ply_origins
            got = _load_ply_origins(p)
        except Exception:  # noqa: BLE001
            continue
        if got and got[0] is not None and len(got[0]):
            return np.asarray(got[0])
    return None


def _visible_voxels(xyz: np.ndarray, vox: np.ndarray, c2w: np.ndarray,
                    K: np.ndarray, w: int, h: int,
                    max_points: int) -> Optional[np.ndarray]:
    """The voxels this camera SEES: cloud points inside its frustum.

    Pure geometry — pose, intrinsics, ray. No occlusion test: a wall in front
    of another surface still means the camera is LOOKING at that sector, which
    is what the cover is about.
    """
    M = np.linalg.inv(c2w)
    p = (M[:3, :3] @ xyz.T).T + M[:3, 3]
    z = p[:, 2]
    front = z > 0.05
    if not front.any():
        return None
    u = K[0, 0] * p[front, 0] / z[front] + K[0, 2]
    v = K[1, 1] * p[front, 1] / z[front] + K[1, 2]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not inb.any():
        return None
    idx = np.flatnonzero(front)[inb]
    return np.unique(vox[idx])


def cover_keyframes(output_dir, session_dir, keyframe_files: Sequence[str],
                    frame_of: Callable[[str], int],
                    voxel_m: float = VOXEL_M,
                    max_overlap: float = MAX_OVERLAP,
                    sample_points: int = 400_000,
                    log: Callable[[str], None] = print) -> Optional[List[str]]:
    """The fewest keyframes that see the whole scene, in walk order.

    Returns None when the cloud or the cameras are missing, so the caller can
    fall back to its own sampling and say so.
    """
    # Cameras and the trace grid ONLY — never the masks. This used to ask
    # `hole_audit._evidence`, which returns on its second line when
    # `output/seg_masks.npz` is missing; that file is written by SAM3, and this
    # runs in the VLM stage, BEFORE SAM3. So the fallback fired on every run
    # ever (pccr 2026-09-17: "no camera evidence" while camera_poses.txt and
    # intrinsic.txt had been on disk for an hour). Coverage is geometry: a
    # pose, its intrinsics and the cloud answer it completely.
    from segmentation.session_io import _load_camera_source
    from reconstruction.surface_fit.hole_audit import _k_grid

    output_dir, session_dir = Path(output_dir), Path(session_dir)
    xyz = _load_cloud(output_dir)
    if xyz is None:
        log("[cover] no cloud on disk — coverage cannot be measured")
        return None
    cam = _load_camera_source(session_dir, output_dir)
    if cam is None:
        log("[cover] no camera_poses.txt/intrinsic.txt — coverage cannot be measured")
        return None
    grid = _k_grid(output_dir)
    if grid is None:
        log("[cover] the cloud carries no pixel provenance (pixel_row/col) — "
            "coverage cannot be measured")
        return None
    kw, kh, _cloud_for_zbuf = grid

    # the whole cloud is overkill for a 10 cm question
    rng = np.random.default_rng(0)
    if len(xyz) > sample_points:
        xyz = xyz[rng.choice(len(xyz), sample_points, replace=False)]
    key = np.floor(xyz / float(voxel_m)).astype(np.int64)
    _u, vox = np.unique(key, axis=0, return_inverse=True)
    n_vox = int(vox.max()) + 1
    covered = np.zeros(n_vox, bool)

    # what every keyframe sees, computed once
    seen_of: Dict[str, np.ndarray] = {}
    for fn in keyframe_files:
        f = int(frame_of(fn))
        c2w = cam.pose_map.get(f)
        K = cam.K_for(f)
        if c2w is None or K is None:
            continue
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        s_ = _visible_voxels(xyz, vox, c2w4, K, kw, kh, sample_points)
        if s_ is not None and len(s_):
            seen_of[fn] = s_
    if not seen_of:
        log("[cover] no keyframe sees any of the cloud — coverage unavailable")
        return None

    # FIRST PASS — the user's walk: keep a keyframe when at most `max_overlap`
    # of what it sees is already covered.
    chosen: List[str] = []
    for fn in keyframe_files:
        s_ = seen_of.get(fn)
        if s_ is None:
            continue
        if chosen and int(covered[s_].sum()) / float(len(s_)) > max_overlap:
            continue
        covered[s_] = True
        chosen.append(fn)
    first = len(chosen)
    reachable = np.zeros(n_vox, bool)
    for s_ in seen_of.values():
        reachable[s_] = True

    # SECOND PASS (USER 2026-09-16) — one walk stops before the scene is
    # covered: the moment a keyframe is skipped for overlapping, whatever IT
    # alone could see is lost with it. pccr's first pass took 9 of 216 and left
    # 22 % of the scene unseen. So go back over the ones that were skipped and
    # take whichever adds the most of what is still missing, until nothing
    # reachable is missing. The overlap rule does not apply here: these frames
    # are being taken FOR the part nobody showed.
    taken = set(chosen)
    while True:
        missing = reachable & ~covered
        if not missing.any():
            break
        best, gain = None, 0
        for fn, s_ in seen_of.items():
            if fn in taken:
                continue
            g = int(missing[s_].sum())
            if g > gain:
                best, gain = fn, g
        if best is None or gain == 0:
            break
        covered[seen_of[best]] = True
        taken.add(best)
        chosen.append(best)
    chosen = [fn for fn in keyframe_files if fn in taken]

    pct = 100.0 * int(covered.sum()) / max(int(reachable.sum()), 1)
    log(f"[cover] {len(chosen)} of {len(keyframe_files)} keyframe(s) see the "
        f"whole scene at {voxel_m * 100:.0f} cm — {first} in the walk "
        f"(≤{max_overlap * 100:.0f}% overlap) + {len(chosen) - first} to close "
        f"the gaps; {pct:.1f}% of what any camera can see is covered")
    return chosen
