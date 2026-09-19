"""One object, its three orthogonal OBB views, one colour per visit.

The check the correction rests on: if two visits of the same object put it in
two different places, the three views of its own OBB show it — plan (L-W),
side (L-U) and front (W-U). Each view fixes two of the three components, so a
displacement invisible in one is visible in another. A single 3-D view, or a
point-to-point ICP, hides exactly the lateral slide these panels expose.

The OBB axes are computed on ONE visit — the best sampled — never on the union
of the two: the union has its principal axis dragged by the displacement
itself.

    python -m tools.ortho_obb_views --session <dir> [--epoch-dir _epoch_0]
                                    [--out <dir>] [--instance 203]

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# green first visit, red second, blue third, then on
VISIT_COLORS = ["#21c85a", "#e03131", "#1c7ed6", "#f08c00", "#9c36b5", "#0ca678"]
MAX_PLOT = 150_000          # points drawn per visit; above it, a random sample


def _panel(ax, P_by_visit, ax_u, ax_v, name_u, name_v, title):
    for i, P in enumerate(P_by_visit):
        if not len(P):
            continue
        u, v = P @ ax_u, P @ ax_v
        ax.scatter(u, v, s=0.4, c=VISIT_COLORS[i % len(VISIT_COLORS)],
                   alpha=0.45, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{name_u} (m)", fontsize=8)
    ax.set_ylabel(f"{name_v} (m)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)


def render(cand, xyz, ks_of_point, up, out_dir: Path, walked) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from correction.visit_drift import obb_axes

    per_visit = []
    for (a, b) in cand.visits:
        ks = ks_of_point[cand.points]
        P = xyz[cand.points[(ks >= a) & (ks <= b)]]
        if len(P) > MAX_PLOT:
            P = P[np.random.default_rng(0).choice(len(P), MAX_PLOT, replace=False)]
        per_visit.append(P)

    # the axes come from the best-sampled visit, never from the union
    axes = obb_axes(max(per_visit, key=len), up)
    L, W, U = axes[0], axes[1], axes[2]

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=110)
    _panel(axs[0], per_visit, L, W, "L", "W", "planta  (L-W)")
    _panel(axs[1], per_visit, L, U, "L", "U", "costado  (L-U)")
    _panel(axs[2], per_visit, W, U, "W", "U", "frente  (W-U)")

    legend = "   ".join(
        f"visita {i + 1}: kf {a}-{b}  {sh * 100:.0f}%  {len(P):,} pts"
        for i, ((a, b), sh, P) in enumerate(zip(cand.visits, cand.shares, per_visit)))
    fig.suptitle(
        f"{cand.label}#{cand.instance_id}   (masklet oid {cand.oid})   "
        f"{len(cand.points):,} puntos   |   recorrido entre visitas: "
        + ", ".join(f"{w:.2f} m" for w in walked) + f"\n{legend}",
        fontsize=10)
    for i, c in enumerate(VISIT_COLORS[:len(per_visit)]):
        fig.text(0.012, 0.955 - i * 0.030, "■", color=c, fontsize=13)

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{cand.instance_id}_{cand.label}.jpg"
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(p, format="jpg", pil_kwargs={"quality": 92})
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--epoch-dir", default="_epoch_0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--instance", type=int, default=None)
    a = ap.parse_args(argv)

    from correction.config import load_correction_config
    from correction.distribute import chainage
    from correction.session import read_ply
    from correction.visit_drift import (filter_chain, masklet_visits,
                                        points_of_masklets)
    from segmentation import mask_space

    out = Path(a.session) / "output" if (Path(a.session) / "output").is_dir() \
        else Path(a.session)
    src = out / a.epoch_dir if a.epoch_dir and (out / a.epoch_dir).is_dir() else out
    dst = Path(a.out) if a.out else out / "reproject" / "ortho_obb"
    cfg = load_correction_config().visit_drift

    _, data = read_ply(src / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    poses = np.loadtxt(src / "camera_poses.txt").reshape(-1, 4, 4)
    up = -poses[:, :3, 1].mean(0)
    up = up / np.linalg.norm(up)
    ch = chainage(poses)

    kfs = mask_space.keyframe_numbers(out) or []
    kf_of = np.full(int(max(kfs)) + 2, -1, np.int64)
    for k, f in enumerate(kfs):
        kf_of[int(f)] = k
    ks = kf_of[np.clip(data["frame_global"].astype(np.int64), 0, len(kf_of) - 1)]

    pm = points_of_masklets(out, data["frame_global"], data["pixel_row"],
                            data["pixel_col"])
    ms = masklet_visits(out)
    cands, steps = filter_chain(ms, pm, ks, ch, cfg.min_points, cfg.min_walk_m,
                                cfg.min_visit_share)
    if a.instance is not None:
        cands = [c for c in cands if c.instance_id == a.instance]

    written = []
    for c in sorted(cands, key=lambda x: -x.walked[0]):
        written.append(render(c, xyz, ks, up, dst, c.walked))
        print(f"  {written[-1].name}")
    (dst / "_chain.json").write_text(json.dumps(
        {"epoch_dir": str(src.name), "steps": steps,
         "objects": [c.as_dict() for c in cands],
         "provenance": "tool_measured"}, indent=1))
    print(f"{len(written)} jpg en {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
