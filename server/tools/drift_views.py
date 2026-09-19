"""Paso 3, para validar a ojo: las dos copias ANTES y DESPUES de aplicar t.

Arriba, las tres vistas ortogonales del OBB con las dos visitas donde estan.
Abajo, las mismas tres vistas con la copia de la visita 2 corrida por el vector
que la medicion dice. Si la medicion es correcta, el rojo cae sobre el verde.

    python -m tools.drift_views --session <dir> [--instance 203]

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C_A = "#21c85a"
C_B = "#e03131"
MAX_PLOT = 120_000


def _panel(ax, A, B, u, v, nu, nv, title):
    for P, c in ((A, C_A), (B, C_B)):
        if len(P):
            ax.scatter(P @ u, P @ v, s=0.4, c=c, alpha=0.45, linewidths=0,
                       rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{nu} (m)", fontsize=8)
    ax.set_ylabel(f"{nv} (m)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--epoch-dir", default="_epoch_0")
    ap.add_argument("--instance", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from correction.config import load_correction_config
    from correction.distribute import chainage
    from correction.session import read_ply
    from correction import visit_drift as vd
    from segmentation import mask_space

    out = Path(a.session) / "output"
    src = out / a.epoch_dir if (out / a.epoch_dir).is_dir() else out
    dst = Path(a.out) if a.out else out / "reproject" / "paso3_deriva"
    dst.mkdir(parents=True, exist_ok=True)
    cfg = load_correction_config().visit_drift

    _, d = read_ply(src / "cleaned_cloud.ply")
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    poses = np.loadtxt(src / "camera_poses.txt").reshape(-1, 4, 4)
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
    cands, steps = vd.filter_chain(ms, pm, ks, ch, cfg.min_points, cfg.min_walk_m,
                                   cfg.min_visit_share, xyz=xyz)
    if a.instance is not None:
        cands = [c for c in cands if c.instance_id == a.instance]

    rng = np.random.default_rng(0)
    rows = []
    for c in cands:
        A, B = c.copies[0], c.copies[1]
        axes = vd.obb_axes(A if len(A) >= len(B) else B, up)
        t, pv, dis = vd.drift_by_views(A, B, axes, cfg.silhouette_cell_m,
                                       cfg.search_margin_m, cfg.silhouette_close_px,
                                       cfg.silhouette_blur_px)
        mag = float(np.linalg.norm(t))
        comp = [float(t @ axes[i]) for i in range(3)]
        rows.append({"objeto": f"{c.label}#{c.instance_id}", "oid": c.oid,
                     "L_cm": round(comp[0] * 100, 1), "W_cm": round(comp[1] * 100, 1),
                     "U_cm": round(comp[2] * 100, 1), "t_cm": round(mag * 100, 1),
                     "desac_L": round(dis[0] * 100, 1), "desac_W": round(dis[1] * 100, 1),
                     "desac_U": round(dis[2] * 100, 1),
                     "pico_min": round(min(v[2] for v in pv.values()), 3),
                     "picos": {k: round(v[2], 3) for k, v in pv.items()}})

        As = A if len(A) <= MAX_PLOT else A[rng.choice(len(A), MAX_PLOT, replace=False)]
        Bs = B if len(B) <= MAX_PLOT else B[rng.choice(len(B), MAX_PLOT, replace=False)]
        L, W, U = axes
        fig, axs = plt.subplots(2, 3, figsize=(16.5, 10.4), dpi=105)
        for r, (Bp, tag) in enumerate(((Bs, "ANTES"), (Bs + t, "DESPUES"))):
            _panel(axs[r][0], As, Bp, L, W, "L", "W", f"planta (L-W)  {tag}")
            _panel(axs[r][1], As, Bp, L, U, "L", "U", f"costado (L-U)  {tag}")
            _panel(axs[r][2], As, Bp, W, U, "W", "U", f"frente (W-U)  {tag}")
        fig.suptitle(
            f"{c.label}#{c.instance_id}   visita 1 kf {c.visits[0][0]}-{c.visits[0][1]} "
            f"({c.shares[0]*100:.0f}%)  ·  visita 2 kf {c.visits[1][0]}-{c.visits[1][1]} "
            f"({c.shares[1]*100:.0f}%)   ·  {c.walked[0]:.1f} m de recorrido\n"
            f"t = L {comp[0]*100:+.1f}   W {comp[1]*100:+.1f}   U {comp[2]*100:+.1f} cm"
            f"   |t| {mag*100:.1f} cm   ·   desacuerdo entre vistas "
            f"{dis[0]*100:.0f} / {dis[1]*100:.0f} / {dis[2]*100:.0f} cm"
            f"   ·   pico min {min(v[2] for v in pv.values()):.2f}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = dst / f"{c.instance_id}_{c.label}.jpg"
        fig.savefig(p, format="jpg", pil_kwargs={"quality": 90})
        plt.close(fig)

    rows.sort(key=lambda r: max(r["desac_L"], r["desac_W"], r["desac_U"]))
    (dst / "_paso3.json").write_text(json.dumps(
        {"epoch_dir": src.name, "chain": steps, "objetos": rows,
         "provenance": "tool_measured"}, indent=1))
    print("%-28s %7s %7s %7s %7s   %5s %5s %5s  %5s" % (
        "objeto", "L cm", "W cm", "U cm", "|t|", "dsL", "dsW", "dsU", "pico"))
    for r in rows:
        print("%-28s %7.1f %7.1f %7.1f %7.1f   %5.0f %5.0f %5.0f  %5.2f" % (
            r["objeto"][:28], r["L_cm"], r["W_cm"], r["U_cm"], r["t_cm"],
            r["desac_L"], r["desac_W"], r["desac_U"], r["pico_min"]))
    print(f"\n{len(rows)} jpg en {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
