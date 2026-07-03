"""
Stage 4 exports — the deviation record an engineer can open:

  - deviation PLY: the ORIGINAL measured points colored by signed residual
    (diverging colormap, symmetric around 0) → inspect in CloudCompare/Potree.
  - heatmap PNG: the gridded residual field over the surface's UV frame with
    a colorbar in mm, findings marked, axes in metres.

matplotlib is used headless (Agg) for the PNG; the PLY colors are computed
with the same colormap so both artifacts read identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

from .models import Finding
from .spatial_test import GridField

_CMAP = "RdBu_r"   # blue = behind the surface, red = in front (along +normal)


def pick_vmax_mm(signed_mm: np.ndarray, vmax_mm: Optional[float] = None) -> float:
    """Symmetric color range: config override, else p98 of |residual| with a
    2 mm floor so near-perfect surfaces don't amplify sensor noise into a
    dramatic-looking map."""
    if vmax_mm is not None and vmax_mm > 0:
        return float(vmax_mm)
    if signed_mm.size == 0:
        return 2.0
    return float(max(np.percentile(np.abs(signed_mm), 98), 2.0))


def write_deviation_ply(points: np.ndarray, signed_m: np.ndarray, path: Path,
                        vmax_mm: float) -> Optional[str]:
    """Original points colored by signed residual (mm), diverging colormap."""
    try:
        import open3d as o3d
    except Exception:
        return None
    signed_mm = np.asarray(signed_m, dtype=np.float64) * 1000.0
    norm = mcolors.Normalize(vmin=-vmax_mm, vmax=vmax_mm, clip=True)
    rgba = matplotlib.colormaps[_CMAP](norm(signed_mm))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(rgba[:, :3].astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)
    return str(path)


def write_heatmap_png(field: GridField, path: Path, vmax_mm: float,
                      title: str = "", subtitle: str = "",
                      findings: Optional[List[Finding]] = None,
                      findings_uv: Optional[np.ndarray] = None) -> Optional[str]:
    """Residual field (cell means, mm) over the surface UV frame → PNG."""
    h, w = field.mean.shape
    img = field.mean * 1000.0                       # mm
    img = np.ma.masked_invalid(img)
    extent = (field.u0, field.u0 + w * field.cell,
              field.v0, field.v0 + h * field.cell)

    fig_w = 10.0
    fig_h = max(3.0, min(14.0, fig_w * (h / max(w, 1)) * 0.9 + 1.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    cmap = plt.get_cmap(_CMAP).copy()
    cmap.set_bad(color="#d9d9d9")                   # empty cells = light grey
    im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap,
                   vmin=-vmax_mm, vmax=vmax_mm, interpolation="nearest",
                   aspect="equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("signed deviation [mm]")
    ax.set_xlabel("u [m]")
    ax.set_ylabel("v [m]")
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=26)
    if subtitle:
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=8, color="#444444", va="bottom")

    if findings and findings_uv is not None and len(findings_uv) == len(findings):
        for f, (fu, fv) in zip(findings, findings_uv):
            if f.kind == "tilt":                    # global finding → no point marker
                continue
            ax.plot(fu, fv, marker="x", ms=9, mew=2.0, color="#111111")
            ax.annotate(f"{f.kind} {f.peak_dev_mm:+.1f} mm",
                        xy=(fu, fv), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color="#111111",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#888888", alpha=0.75))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
