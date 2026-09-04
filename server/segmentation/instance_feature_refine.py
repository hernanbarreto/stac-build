"""
FASE 4 of the DINOv3 plan (USER 2026-09-04): object/background separation
refinement AFTER segmentation.

Each instance's points carry their own DINOv3 feature (source frame + pixel,
via the fase-1 cache). The instance's feature signature (k robust modes) is
contrasted against the signature of its spatial surroundings (background
points near the instance but not in it): points inside the instance whose
feature clearly belongs to the background — 2D mask bleed, border
contamination — are FLAGGED, never deleted. The result is a reversible
sidecar (``output/feature_refine/<safe>_refine.json`` + store meta) that
downstream consumers (per-object meshing, p2c, OBBs) can opt into.

Provenance: tool_measured (feature affinity margins over measured points).

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _kmeans_cosine(F: np.ndarray, k: int, iters: int = 12,
                   seed: int = 0) -> np.ndarray:
    """Spherical k-means on unit features → (k, D) unit centroids."""
    rng = np.random.default_rng(seed)
    k = min(k, len(F))
    C = F[rng.choice(len(F), k, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(F @ C.T, axis=1)
        for j in range(k):
            m = a == j
            if m.any():
                c = F[m].mean(axis=0)
                C[j] = c / max(np.linalg.norm(c), 1e-8)
    return C


def refine_instance(output_dir: Path, instance_id: int,
                    cfg: Optional[dict] = None, log=logger.info
                    ) -> Optional[Path]:
    """Flag background-contaminated points of one instance by feature
    affinity. Returns the sidecar path (None when inputs are missing)."""
    from scipy.spatial import cKDTree
    from segmentation.tsdf_export import _safe_label
    from segmentation.perfect_object import _read_ply_fields
    from reconstruction.dino_features import (FeatureCache,
                                              calibrate_provenance_grid,
                                              extract_session_features,
                                              load_session_cameras)

    cfg = cfg or {}
    k_modes = int(cfg.get("modes", 3))
    margin = float(cfg.get("margin", 0.05))
    bg_radius = float(cfg.get("bg_radius_m", 0.5))
    max_fit = int(cfg.get("max_fit_pts", 50_000))
    t0 = time.time()
    out = Path(output_dir)

    result = json.loads((out / "segmentation_result.json").read_text())
    inst = next((i for i in result.get("instances", [])
                 if int(i.get("instance_id", i.get("id"))) == int(instance_id)),
                None)
    if inst is None:
        raise ValueError(f"instance {instance_id} not found")
    label = str(inst.get("label", "segment"))
    safe = _safe_label(label, int(instance_id))
    gi = np.asarray(inst.get("globalIndices") or [], np.int64)
    if len(gi) == 0:
        log(f"[feat-refine:{safe}] no globalIndices — skipped")
        return None

    frames_dir = out.parent / "frames"
    extract_session_features(out, frames_dir, cfg, log=log)
    fc = FeatureCache(out)
    fields = _read_ply_fields(out / "cleaned_cloud.ply")
    xyz = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(
        np.float64)
    fg = np.asarray(fields["frame_global"], np.int64)
    pr = np.asarray(fields["pixel_row"], np.int64)
    pcl = np.asarray(fields["pixel_col"], np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]

    poses, frames, Ks = load_session_cameras(out)
    Hg, Wg, _err = calibrate_provenance_grid(xyz, fg, pr, pcl, poses,
                                             frames, Ks, log=log)

    def _own_features(idx: np.ndarray) -> np.ndarray:
        F = np.zeros((len(idx), fc.dim), np.float32)
        okm = np.zeros(len(idx), bool)
        for f in np.unique(fg[idx]):
            if not fc.has(int(f)):
                continue
            m = fg[idx] == f
            g = fc.grid(int(f))
            F[m] = fc.sample(g, (pr[idx[m]] + 0.5) / Hg,
                             (pcl[idx[m]] + 0.5) / Wg)
            okm[m] = True
        return F, okm

    rng = np.random.default_rng(0)
    # instance signature
    fit_idx = gi if len(gi) <= max_fit else \
        gi[rng.choice(len(gi), max_fit, replace=False)]
    Fi, oki = _own_features(fit_idx)
    if oki.sum() < 100:
        log(f"[feat-refine:{safe}] too few featured points — skipped")
        return None
    Cin = _kmeans_cosine(Fi[oki], k_modes)

    # background signature: near the instance, not of it
    in_mask = np.zeros(len(xyz), bool)
    in_mask[gi] = True
    sub_inst = xyz[fit_idx]
    kd = cKDTree(sub_inst[:: max(1, len(sub_inst) // 20_000)])
    cand = np.flatnonzero(~in_mask)
    cand = cand[rng.choice(len(cand), min(len(cand), 800_000),
                           replace=False)]
    d, _ = kd.query(xyz[cand], k=1, workers=8,
                    distance_upper_bound=bg_radius)
    bg_idx = cand[np.isfinite(d)]
    if len(bg_idx) < 500:
        log(f"[feat-refine:{safe}] no background neighbourhood — skipped")
        return None
    if len(bg_idx) > max_fit:
        bg_idx = bg_idx[rng.choice(len(bg_idx), max_fit, replace=False)]
    Fb, okb = _own_features(bg_idx)
    Cbg = _kmeans_cosine(Fb[okb], k_modes)

    # margins for EVERY instance point (blocked)
    flagged = []
    margins_all = []
    for b0 in range(0, len(gi), 2_000_000):
        blk = gi[b0:b0 + 2_000_000]
        F, okm = _own_features(blk)
        m_in = (F @ Cin.T).max(axis=1)
        m_bg = (F @ Cbg.T).max(axis=1)
        mg = np.where(okm, m_in - m_bg, np.nan)
        margins_all.append(mg)
        flagged.append(blk[okm & (mg < -margin)])
    flagged = np.concatenate(flagged) if flagged else np.array([], np.int64)
    margins = np.concatenate(margins_all)
    ok_m = np.isfinite(margins)

    dst = out / "feature_refine"
    dst.mkdir(exist_ok=True)
    sidecar = dst / f"{safe}_refine.json"
    payload = {
        "instance_id": int(instance_id), "label": label,
        "n_points": int(len(gi)),
        "n_flagged_background": int(len(flagged)),
        "flagged_frac": round(float(len(flagged) / max(len(gi), 1)), 4),
        "margin_thr": margin, "modes": k_modes,
        "margin_median": round(float(np.nanmedian(margins)), 4)
        if ok_m.any() else None,
        "flagged_global_indices": [int(x) for x in flagged],
        "provenance": "tool_measured",
        "elapsed_s": round(time.time() - t0, 1),
    }
    sidecar.write_text(json.dumps(payload))
    try:
        from phase_r.instance_store import InstanceStore
        st = InstanceStore(out / "scene_r.db")
        st.set_meta(f"feature_refine_{int(instance_id)}", json.dumps(
            {k: payload[k] for k in ("n_points", "n_flagged_background",
                                     "flagged_frac", "margin_median")}))
        st.close()
    except Exception as e:  # noqa: BLE001 — sidecar is the artifact of record
        log(f"[feat-refine:{safe}] store meta not written ({e})")
    log(f"[feat-refine:{safe}] ✅ {len(flagged):,}/{len(gi):,} pts flagged "
        f"as background ({payload['flagged_frac'] * 100:.1f}%) → "
        f"{sidecar.name} ({payload['elapsed_s']}s)")
    return sidecar


def flagged_background_indices(output_dir: Path, safe: str
                               ) -> Optional[np.ndarray]:
    """Consumer helper: global indices of the points FLAGGED as background
    for this instance (subtract them from globalIndices to opt in), or None
    when no sidecar exists. Reversible by construction."""
    p = Path(output_dir) / "feature_refine" / f"{safe}_refine.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return np.asarray(d.get("flagged_global_indices") or [], np.int64)
