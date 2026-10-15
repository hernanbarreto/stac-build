"""Coverage of a variant (phase 3): registered keyframes, disconnected models, reconstructible
test ends and the share of each reference plane's surface that holds points. A distance a
variant cannot rebuild is "not reconstructed" and enters the coverage; nothing is dropped."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from evaluation import geometry as G
from evaluation.product import Product


def _polygon_cells(poly: np.ndarray, cell_px: int) -> tuple:
    """(cell grid origin, mask of cells inside the polygon, shape)."""
    from matplotlib.path import Path as MplPath
    lo = np.floor(poly.min(axis=0) / cell_px).astype(int)
    hi = np.ceil(poly.max(axis=0) / cell_px).astype(int)
    nx, ny = int(hi[0] - lo[0]), int(hi[1] - lo[1])
    if nx <= 0 or ny <= 0:
        return lo, np.zeros((0, 0), bool), (0, 0)
    cx = (np.arange(nx) + lo[0] + 0.5) * cell_px
    cy = (np.arange(ny) + lo[1] + 0.5) * cell_px
    gx, gy = np.meshgrid(cx, cy)
    inside = MplPath(poly).contains_points(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(ny, nx)
    return lo, inside, (ny, nx)


def plane_surface_coverage(plane: Dict[str, Any], product: Product, cell_px: int) -> Dict[str, Any]:
    """Share of the polygon cells (cell_px × cell_px, keyframe pixels) holding ≥ 1 product point:
    by provenance when the product has it, else by projecting the dense cloud into the keyframe."""
    total, filled = 0, 0
    per = []
    for poly in plane["polygons"]:
        f = int(poly["frame"])
        if not product.registered(f):
            per.append({"frame": f, "cells": 0, "filled": 0, "reason": "keyframe not registered"})
            continue
        pts = product.undistort(f, np.asarray(poly["points"], np.float64))
        lo, inside, shape = _polygon_cells(pts, cell_px)
        n_cells = int(inside.sum())
        if n_cells == 0:
            per.append({"frame": f, "cells": 0, "filled": 0, "reason": "degenerate polygon"})
            continue
        if product.has_provenance():
            _, uv = product.points_of_frame(f)
        else:
            fp = product.projection(f)
            uv = fp.uv
        hit = np.zeros(shape, bool)
        if len(uv):
            ix = np.floor(uv[:, 0] / cell_px).astype(int) - lo[0]
            iy = np.floor(uv[:, 1] / cell_px).astype(int) - lo[1]
            ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
            hit[iy[ok], ix[ok]] = True
        n_filled = int((hit & inside).sum())
        total += n_cells; filled += n_filled
        per.append({"frame": f, "cells": n_cells, "filled": n_filled})
    return {"id": plane["id"], "label": plane.get("label", ""), "cells": total, "cells_with_points": filled,
            "surface_with_points_pct": (100.0 * filled / total) if total else None, "cell_px": cell_px,
            "per_polygon": per}


def coverage(product: Optional[Product], keyframes: Sequence[int], doc: Dict[str, Any],
             recs: Optional[List[Dict[str, Any]]], cell_px: int, n_models: int = 1,
             models: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """``keyframes`` = the reference keyframe list (what every variant was given);
    ``recs`` = phase 1's raw records (ends with or without a point)."""
    kf = [int(k) for k in keyframes]
    registered = [k for k in kf if product is not None and product.registered(k)]
    test_ends = [(r["ref"]["id"], e, getattr(r[e], "point", None) is not None)
                 for r in (recs or []) if r["ref"]["role"] == "test" for e in ("a", "b")]
    all_ends = [(r["ref"]["id"], e, getattr(r[e], "point", None) is not None)
                for r in (recs or []) for e in ("a", "b")]
    planes = ([plane_surface_coverage(p, product, cell_px) for p in doc.get("planes", [])]
              if product is not None else
              [{"id": p["id"], "label": p.get("label", ""), "cells": 0, "cells_with_points": 0,
                "surface_with_points_pct": None, "cell_px": cell_px, "per_polygon": []} for p in doc.get("planes", [])])
    return {
        "keyframes_total": len(kf), "keyframes_registered": len(registered),
        "keyframes_registered_pct": (100.0 * len(registered) / len(kf)) if kf else None,
        "keyframes_unregistered": [k for k in kf if k not in set(registered)],
        "n_models": int(n_models), "models": list(models or []),
        "test_ends_total": len(test_ends), "test_ends_reconstructed": sum(1 for *_, ok in test_ends if ok),
        "test_ends_reconstructed_pct": (100.0 * sum(1 for *_, ok in test_ends if ok) / len(test_ends)) if test_ends else None,
        "ends_total": len(all_ends), "ends_reconstructed": sum(1 for *_, ok in all_ends if ok),
        "distances_not_reconstructed": sorted({i for i, _, ok in all_ends if not ok}),
        "planes": planes,
        "planes_surface_with_points_pct_mean": (float(np.mean([p["surface_with_points_pct"] for p in planes
                                                              if p["surface_with_points_pct"] is not None]))
                                                if any(p["surface_with_points_pct"] is not None for p in planes) else None),
    }
