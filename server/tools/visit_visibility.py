"""Which part of an object did each visit SEE — not measure. Three states.

USER 2026-09-18: *"podemos separar en voxeles la mascara y determinar que kf
vieron cada parte para saber donde hay solapamiento … partes que se ven en
ambas visitas y son las que tienen discrepancia, porque si solo se ve de una
visita no puede haber discrepancia"*.

VISTO is not MEASURED. A voxel can lie inside the object's SAM mask and carry
no points (low confidence, grazing angle). Today's silhouette correlation reads
"no points" as "the object is not here", and that is what makes a partial visit
drag the peak. So every voxel gets, per visit, one of: seen / not seen — decided
by projecting the voxel centre into every keyframe of that visit and asking the
object's own MASK, never the points.

A mask is 2-D, so landing inside it is not enough: a voxel BEHIND the object is
inside the same silhouette and the camera never saw it. Every projection is
verified against a per-keyframe Z-buffer, and the machinery is the module's —
``correction.visit_drift.Visibility`` — never a second copy of it here.

The voxel grid is aligned to the object's OBB (L, W, U), so the three
orthogonal views are clean slices of it.

    python -m tools.visit_visibility --session <dir> [--voxel-m 0.10]
                                     [--instance 203]

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

C_BOTH = "#1f9d55"      # seen by both visits   -> the only place discrepancy lives
C_V1 = "#1c7ed6"        # seen only by visit 1
C_V2 = "#e03131"        # seen only by visit 2
C_NONE = "#d8d8d8"      # seen by neither


def _view(ax, g, coords, i, j, names, title):
    """Collapse the voxel grid along the third axis; priority both > v1 > v2."""
    import matplotlib.patches as mp
    lo_i, lo_j, step = coords[i][0], coords[j][0], coords["step"]
    k = [a for a in (0, 1, 2) if a not in (i, j)][0]
    col = np.max(g, axis=k)                      # 0 none, 1 v2, 2 v1, 3 both
    for a in range(col.shape[0]):
        for b in range(col.shape[1]):
            s = col[a, b]
            if s == 0:
                continue
            c = {3: C_BOTH, 2: C_V1, 1: C_V2}[s]
            ax.add_patch(mp.Rectangle((lo_i + a * step, lo_j + b * step),
                                      step, step, facecolor=c, edgecolor="none"))
    ax.set_xlim(lo_i, lo_i + col.shape[0] * step)
    ax.set_ylim(lo_j, lo_j + col.shape[1] * step)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{names[0]} (m)", fontsize=8)
    ax.set_ylabel(f"{names[1]} (m)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.2, alpha=0.3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--epoch-dir", default="_epoch_0")
    ap.add_argument("--voxel-m", type=float, default=0.10)
    ap.add_argument("--instance", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    from correction.config import load_correction_config
    from correction.distribute import chainage
    from correction.session import read_ply
    from correction import visit_drift as vd
    from segmentation import mask_space

    out = Path(a.session) / "output"
    src = out / a.epoch_dir if (out / a.epoch_dir).is_dir() else out
    dst = Path(a.out) if a.out else out / "reproject" / "visibilidad"
    dst.mkdir(parents=True, exist_ok=True)
    cfg = load_correction_config().visit_drift

    _, d = read_ply(src / "cleaned_cloud.ply")
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    poses = np.loadtxt(src / "camera_poses.txt").reshape(-1, 4, 4)
    K_all = np.loadtxt(out / "intrinsic.txt").reshape(-1, 4)
    Ht, Wt = vd.trace_grid(out)
    up = -poses[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    ch = chainage(poses)

    kfs = mask_space.keyframe_numbers(out) or []
    kf_of = np.full(int(max(kfs)) + 2, -1, np.int64)
    for k, f in enumerate(kfs):
        kf_of[int(f)] = k
    ks = kf_of[np.clip(d["frame_global"].astype(np.int64), 0, len(kf_of) - 1)]

    pm = vd.points_of_masklets(out, d["frame_global"], d["pixel_row"],
                               d["pixel_col"], log=lambda m: None)
    ms = vd.masklet_visits(out, log=lambda m: None)
    label_of = {m.oid: m.label for m in ms}
    cands, _ = vd.filter_chain(ms, pm, ks, ch, cfg.min_points, cfg.min_walk_m,
                               cfg.min_visit_share, xyz=xyz, log=lambda m: None)

    pairs = []
    for c in cands:
        A, B = c.copies[0], c.copies[1]
        axes = vd.obb_axes(A if len(A) >= len(B) else B, up)
        t, pv, dis = vd.drift_by_views(A, B, axes, cfg.silhouette_cell_m,
                                       cfg.search_margin_m, cfg.silhouette_close_px,
                                       cfg.silhouette_blur_px)
        pairs.append((c, vd.Drift(c.instance_id, c.label, c.visits[0], c.visits[1],
                                  c.walked[0], 0.0, axes, t, pv, dis, len(A), len(B))))
    kept, _rep = vd.drop_ambiguous(pairs, pm, label_of, xyz, cfg.max_ambiguity,
                                   log=lambda m: None)
    if a.instance is not None:
        kept = [(c, dr) for c, dr in kept if c.instance_id == a.instance]

    from config import get_param
    tol = float(get_param("segmentation.mask_filter.depth_tol_m", 0.15))
    step = float(a.voxel_m)
    vis = None
    rows = []
    for c, dr in sorted(kept, key=lambda x: -x[1].magnitude):
        axes = dr.axes
        P = np.vstack(c.copies)
        loc = P @ axes.T
        lo = loc.min(0) - step
        hi = loc.max(0) + step
        n = np.maximum(((hi - lo) / step).astype(int) + 1, 1)
        gi, gj, gk = np.meshgrid(np.arange(n[0]), np.arange(n[1]),
                                 np.arange(n[2]), indexing="ij")
        local = np.stack([lo[0] + (gi.ravel() + 0.5) * step,
                          lo[1] + (gj.ravel() + 0.5) * step,
                          lo[2] + (gk.ravel() + 0.5) * step], 1)
        centres = local @ axes           # back to world

        if vis is None:
            vis = vd.Visibility(out, xyz, ks, poses, K_all, tol)
        s1 = vis.seen(centres, vis.masks_of(c.oid, c.visits[0]))
        s2 = vis.seen(centres - dr.t, vis.masks_of(c.oid, c.visits[1]))
        code = (s1.astype(np.uint8) * 2 + s2.astype(np.uint8))   # 3 both,2 v1,1 v2,0 none
        g = code.reshape(tuple(n))

        nb, n1, n2, n0 = int((code == 3).sum()), int((code == 2).sum()), \
            int((code == 1).sum()), int((code == 0).sum())
        rows.append((c, dr, nb, n1, n2, n0))

        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=110)
        coords = {0: (lo[0],), 1: (lo[1],), 2: (lo[2],), "step": step}
        _view(axs[0], g, coords, 0, 1, ("L", "W"), "planta  (L-W)")
        _view(axs[1], g, coords, 0, 2, ("L", "U"), "costado  (L-U)")
        _view(axs[2], g, coords, 1, 2, ("W", "U"), "frente  (W-U)")
        tot = nb + n1 + n2 + n0
        fig.suptitle(
            f"{c.label}#{c.instance_id}   voxel {step*100:.0f} cm   "
            f"{tot:,} voxeles   |   ambas {nb:,} ({100*nb/tot:.1f}%)   "
            f"solo v1 {n1:,}   solo v2 {n2:,}   ninguna {n0:,}   "
            f"(z-buffer del propio keyframe, tolerancia {tol*100:.0f} cm)\n"
            f"visita 1: kf {c.visits[0][0]}-{c.visits[0][1]}   "
            f"visita 2: kf {c.visits[1][0]}-{c.visits[1][1]}   "
            f"|t| medido {dr.magnitude*100:.1f} cm", fontsize=10)
        h = [mp.Patch(color=C_BOTH, label="vistas por las DOS"),
             mp.Patch(color=C_V1, label="solo visita 1"),
             mp.Patch(color=C_V2, label="solo visita 2"),
             mp.Patch(color=C_NONE, label="por ninguna")]
        fig.legend(handles=h, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 0.88))
        p = dst / f"{c.instance_id}_{c.label}.jpg"
        fig.savefig(p, format="jpg", pil_kwargs={"quality": 92})
        plt.close(fig)
        print(f"  {p.name}  ambas {nb:,} ({100*nb/tot:.1f}%)  v1 {n1:,}  "
              f"v2 {n2:,}  ninguna {n0:,}")

    print(f"\n{len(rows)} jpg en {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
