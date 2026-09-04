"""
VLM SHAPE PROPOSER (user 2026-09-03) — "describir correctamente el objeto".

Qwen3-VL looks at the object exactly the way the user does — annotated crops
of the real scan frames — plus the tool-measured facts, and PROPOSES:
  - what the object IS (identity + description);
  - for every detected part/region: the intended shape family, the part's
    role, its outline intent and expected relations to other parts.

Evidence given to the VLM per call:
  * isolated crops (SAM mask bbox, darkened background) of the best views;
  * region-annotated crops: each detected region's points projected into the
    frame (Z-buffer occlusion-verified, same math as hole_audit), tinted with
    a stable color and numbered at its 2-D centroid;
  * the measured inventory (kind, size, orientation, radius, rms) as TEXT.

Provenance rule (inviolable): the VLM proposes, it NEVER measures. Every
number it receives is tool_measured context; every field it returns is
vlm_proposed intent. Consumers (model rebuild, Point2CAD trim) gate on
`agrees_with_fit` + confidence and keep the deterministic fit as truth.

Published as output/shape_proposals/<safe>_proposal.json (+ the annotated
crops for the user's own eyes) and cached in scene_r.db meta.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_UP = np.array([0.0, 1.0, 0.0])   # display frame is three.js Y-up

# stable, high-contrast region palette (RGB 0-255)
_PALETTE = [
    (230, 60, 60), (60, 130, 230), (60, 200, 90), (240, 180, 40),
    (170, 90, 230), (40, 210, 210), (240, 120, 200), (150, 200, 60),
    (240, 140, 60), (100, 100, 240), (90, 220, 160), (220, 220, 90),
    (200, 80, 120), (80, 180, 240), (160, 160, 160), (120, 220, 60),
]

_KINDS = ["plane", "cylinder", "cone", "sphere", "torus", "extrusion",
          "opening", "freeform"]
_OUTLINES = ["circle", "rectangle", "rounded_rect", "arch", "polygon",
             "irregular", "unknown"]

# proposed kind → does it agree with our deterministic fit of that region?
# 'opening' NEVER agrees: it says the fitted surface is not material at all
# (a hole's rim wrongly fitted as a sphere/cylinder — user 2026-09-03:
# "no hay esferas, son huecos").
_KIND_MATCH = {"plane": {"plane", "extrusion"},
               "cylinder": {"cylinder", "extrusion"},
               "sphere": {"sphere"}}


# ── tolerant JSON extraction (fences / prose / truncation) ───────────────

def _extract_json(text: str):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                continue
    return None


# ── measured context (tool_measured — GIVEN to the VLM, never asked) ─────

def _orientation(kind: str, model) -> str:
    v = None
    if kind == "plane":
        v = np.asarray(model.normal, np.float64)
    elif kind == "cylinder":
        v = np.asarray(model.axis_dir, np.float64)
    if v is None:
        return "n/a"
    c = abs(float(v @ _UP))
    ang = float(np.degrees(np.arccos(np.clip(c, 0, 1))))
    if kind == "plane":
        # angle of the NORMAL vs up: 0° → horizontal surface (floor/top)
        if ang <= 15:
            return "horizontal surface (floor/top-like)"
        if ang >= 75:
            return "vertical surface (wall/side-like)"
        return f"tilted surface (normal {ang:.0f} deg from vertical)"
    if ang <= 15:
        return "vertical axis"
    if ang >= 75:
        return "horizontal axis"
    return f"tilted axis ({ang:.0f} deg from vertical)"


def _interior_void_ratio(r: dict, Vd: np.ndarray) -> Optional[float]:
    """Tool-measured VOID evidence from the cloud itself (the doctrine: the
    measured points are the truth): on the region's UV support grid, the
    fraction of its interior that is ENCLOSED emptiness — support cells vs
    empty cells fully surrounded by support. A hole's fake rim-fit (ring of
    real material around a void) scores high; a solid plate scores ~0."""
    from reconstruction.surface_fit.support import support_grid
    from scipy import ndimage
    try:
        uv = np.asarray(r["model"].to_uv(Vd[r["v_idx"]]))
        occ, _u0, _v0 = support_grid(uv, 0.02, 0.04)
    except Exception:  # noqa: BLE001
        return None
    lab, n = ndimage.label(~occ)
    if n == 0:
        return 0.0
    border = np.unique(np.concatenate(
        [lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    enclosed = np.isin(lab, [i for i in range(1, n + 1) if i not in border])
    denom = int(occ.sum()) + int(enclosed.sum())
    if denom < 30:
        return None
    return round(float(enclosed.sum()) / denom, 2)


def _region_facts(idx: int, r: dict, P_disp: np.ndarray) -> dict:
    ext = P_disp.ptp(axis=0)
    e = {
        "region": idx,
        "detected_kind": r["kind"],
        "orientation": _orientation(r["kind"], r["model"])
        if r.get("model") is not None else None,
        "size_m": [round(float(x), 3) for x in ext],
        # display-frame centroid: lets a later rebuild MATCH these proposals
        # to its own re-detected regions (RANSAC seeds drift the indices)
        "centroid_m": [round(float(x), 4) for x in P_disp.mean(axis=0)],
        "n_vertices": int(len(r["v_idx"])),
        "provenance": "tool_measured",
    }
    if r.get("model") is not None:
        d = np.abs(np.asarray(r["model"].signed_distance(P_disp)))
        e["fit_rms_mm"] = round(float(np.sqrt(np.mean(d ** 2))) * 1000, 1)
    if r["kind"] in ("cylinder", "sphere"):
        e["radius_m"] = round(float(r["model"].radius), 4)
    return e


# ── frame ranking + annotated crops ──────────────────────────────────────

def _project_frame(ev, fidx: int, pts: np.ndarray):
    """K-grid pixel coords + camera depth for raw-frame points; None if the
    frame has no pose. Same math as hole_audit's vote()."""
    c2w = ev.cam.pose_map.get(fidx)
    K = ev.cam.K_for(fidx)
    if c2w is None or K is None:
        return None
    c2w4 = np.eye(4)
    c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
    M = np.linalg.inv(c2w4)
    p = (M[:3, :3] @ pts.T).T + M[:3, 3]
    z = p[:, 2]
    front = z > 0.05
    u = np.full(len(pts), -1e9)
    v = np.full(len(pts), -1e9)
    u[front] = K[0, 0] * p[front, 0] / z[front] + K[0, 2]
    v[front] = K[1, 1] * p[front, 1] / z[front] + K[1, 2]
    return u, v, z, front


def _inst_zbuf(ev, fidx: int, mh: int, mw: int, inst_pts: np.ndarray,
               cache: Dict[int, Optional[np.ndarray]]):
    """Z-buffer of the INSTANCE's own points only, NO minimum filter — used
    for self-occlusion. hole_audit's full-cloud buffer + 5-px min filter is
    right for walls but flags most of a compact object as 'occluded' by its
    own front surface seen obliquely (measured on bufferStop: 70% of the
    points, mask hit ratio 95%)."""
    if fidx in cache:
        return cache[fidx]
    pr = _project_frame(ev, fidx, inst_pts)
    if pr is None:
        cache[fidx] = None
        return None
    u, v, z, front = pr
    mu = (u * mw / ev.kw).astype(np.int64)
    mv = (v * mh / ev.kh).astype(np.int64)
    inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
    zb = np.full((mh, mw), np.inf, dtype=np.float64)
    np.minimum.at(zb, (mv[inb], mu[inb]), z[inb])
    cache[fidx] = zb
    return zb


def _visible_in_frame(ev, fidx: int, mh: int, mw: int, pts: np.ndarray,
                      inst_pts: Optional[np.ndarray] = None,
                      inst_zbuf_cache: Optional[dict] = None,
                      depth_tol: float = 0.15, self_tol: float = 0.10):
    """(mask-grid u, v, visible) — visible = in bounds AND not occluded.
    External occlusion: the full-cloud Z-buffer says something ≥depth_tol in
    front AND the instance's own buffer does NOT (so the occluder is another
    object, not the object's own surface). Self-occlusion: the instance's own
    un-filtered Z-buffer has a point ≥self_tol closer on the same pixel."""
    pr = _project_frame(ev, fidx, pts)
    if pr is None:
        return None
    u, v, z, front = pr
    mu = (u * mw / ev.kw).astype(np.int64)
    mv = (v * mh / ev.kh).astype(np.int64)
    inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
    vis = inb.copy()
    if not inb.any():
        return mu, mv, vis
    zb_self = None
    if inst_pts is not None and inst_zbuf_cache is not None:
        zb_self = _inst_zbuf(ev, fidx, mh, mw, inst_pts, inst_zbuf_cache)
    with np.errstate(invalid="ignore"):   # inf-inf on unmeasured pixels
        zb = ev._zbuf(fidx, mh, mw)
        if zb is not None:
            zpix = np.full(len(pts), np.inf)
            zpix[inb] = zb[mv[inb], mu[inb]]
            ext_occ = zpix < (z - depth_tol)
            if zb_self is not None:
                zself = np.full(len(pts), np.inf)
                zself[inb] = zb_self[mv[inb], mu[inb]]
                ext_occ &= ~(np.abs(zpix - zself) < 0.05)  # occluder is itself
            vis &= ~ext_occ
        if zb_self is not None:
            zself = np.full(len(pts), np.inf)
            zself[inb] = zb_self[mv[inb], mu[inb]]
            vis &= ~(zself < (z - self_tol))
    return mu, mv, vis


def _calibrate_oid_lenient(ev, inst_pts: np.ndarray, instance_id: int,
                           log=print) -> Optional[int]:
    """npz object id by OWN-mask hit ratio (in-bounds projections only, no
    Z-buffer): hole_audit.calibrate_oid's depth-verified vote is right for
    walls but rejects compact objects whose own front surface 'occludes'
    their sampled points (bufferStop: coverage 0.24 with depth, 0.95 raw)."""
    sample = inst_pts[:: max(1, len(inst_pts) // 1500)]
    candidates = [int(instance_id) - 1]
    try:
        candidates += [int(o) for o in ev.masks["obj_ids"].tolist()
                       if int(o) != instance_id - 1]
    except Exception:  # noqa: BLE001
        pass
    for oid in candidates:
        frames = ev.frames_for(oid)
        if not frames:
            continue
        by_area = sorted(frames, key=lambda fk: int(
            (ev.masks[fk[1]] > 0).sum()), reverse=True)[:12]
        hits = tot = 0
        for fidx, key in by_area:
            m = ev.masks[key]
            mh, mw = m.shape
            pr = _project_frame(ev, fidx, sample)
            if pr is None:
                continue
            u, v, z, front = pr
            mu = (u * mw / ev.kw).astype(np.int64)
            mv = (v * mh / ev.kh).astype(np.int64)
            inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
            hits += int((m[mv[inb], mu[inb]] > 0).sum())
            tot += int(inb.sum())
        if tot >= 200 and hits / tot > 0.5:
            if oid != instance_id - 1:
                log(f"  mask oid {oid} matched (non-standard mapping)")
            return oid
    return None


def _pick_frames(ev, oid: int, mask_frames: List[Tuple[int, str]],
                 region_pts: List[np.ndarray], n_views: int,
                 inst_pts: np.ndarray, inst_zbuf_cache: dict,
                 min_gap: int = 25, log=print) -> List[Tuple[int, str]]:
    """Rank candidate keyframes: object mask area first (cheap), then how many
    regions each shows un-occluded; greedy pick with a frame-index gap so the
    views actually differ."""
    by_area = sorted(
        mask_frames,
        key=lambda fk: int((ev.masks[fk[1]] > 0).sum()), reverse=True)[:24]
    scored = []
    for fidx, key in by_area:
        m = ev.masks[key]
        mh, mw = m.shape
        score = 0.0
        for P in region_pts:
            r = _visible_in_frame(ev, fidx, mh, mw, P,
                                  inst_pts=inst_pts,
                                  inst_zbuf_cache=inst_zbuf_cache)
            if r is None:
                score = -1.0
                break
            _mu, _mv, vis = r
            score += float(np.sqrt(vis.sum()))
        if score >= 0:
            scored.append((score, fidx, key))
    scored.sort(reverse=True)
    picked: List[Tuple[int, str]] = []
    for _s, fidx, key in scored:
        if all(abs(fidx - f0) >= min_gap for f0, _k in picked):
            picked.append((fidx, key))
        if len(picked) >= n_views:
            break
    for f0, _k in picked:
        log(f"  view: frame {f0}")
    return picked


def _load_frame_rgb(session_dir: Path, fidx: int):
    from PIL import Image
    p = Path(session_dir) / "frames" / f"{fidx:06d}.jpg"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def _font(size: int):
    from PIL import ImageFont
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:  # noqa: BLE001
            continue
    from PIL import ImageFont as F
    return F.load_default()


def _isolated_crop(img, mask_rgb: np.ndarray):
    from segmentation.object_captioner import _create_isolated_crop
    return _create_isolated_crop(img, mask_rgb, padding_ratio=0.15,
                                 bg_darken=0.15)


def _annotated_crop(ev, fidx: int, key: str, session_dir: Path,
                    region_pts: List[np.ndarray],
                    inst_pts: np.ndarray, inst_zbuf_cache: dict,
                    max_side: int = 1100, min_pts: int = 40):
    """RGB crop around the object with every region's projected points tinted
    (stable palette) and numbered at its 2-D centroid. Returns (PIL, legend)
    where legend maps region index → color name shown."""
    from PIL import Image, ImageDraw
    img = _load_frame_rgb(session_dir, fidx)
    if img is None:
        return None
    W, H = img.size
    m = ev.masks[key]
    mh, mw = m.shape
    arr = np.asarray(img, np.float32)

    centroids: Dict[int, Tuple[float, float]] = {}
    for ri, P in enumerate(region_pts):
        r = _visible_in_frame(ev, fidx, mh, mw, P,
                              inst_pts=inst_pts,
                              inst_zbuf_cache=inst_zbuf_cache)
        if r is None:
            continue
        mu, mv, vis = r
        if vis.sum() < min_pts:
            continue
        ru = (mu[vis] * W / mw).astype(np.int64).clip(0, W - 1)
        rv = (mv[vis] * H / mh).astype(np.int64).clip(0, H - 1)
        col = np.array(_PALETTE[ri % len(_PALETTE)], np.float32)
        # 3-px stamps so the sparse projection reads as a surface tint
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                uu = (ru + du).clip(0, W - 1)
                vv = (rv + dv).clip(0, H - 1)
                arr[vv, uu] = 0.45 * arr[vv, uu] + 0.55 * col
        centroids[ri] = (float(ru.mean()), float(rv.mean()))
    if not centroids:
        return None

    # crop to the object's own mask bbox (padded) on the RGB grid
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min() * W / mw), int(xs.max() * W / mw)
    y0, y1 = int(ys.min() * H / mh), int(ys.max() * H / mh)
    px, py = int(0.12 * (x1 - x0)), int(0.12 * (y1 - y0))
    x0, x1 = max(0, x0 - px), min(W - 1, x1 + px)
    y0, y1 = max(0, y0 - py), min(H - 1, y1 + py)
    crop = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)) \
        .crop((x0, y0, x1 + 1, y1 + 1))
    s = min(1.0, max_side / max(crop.size))
    if s < 1.0:
        crop = crop.resize((int(crop.width * s), int(crop.height * s)),
                           Image.LANCZOS)

    draw = ImageDraw.Draw(crop)
    fnt = _font(max(16, int(0.035 * max(crop.size))))
    for ri, (cu, cv) in centroids.items():
        tx, ty = (cu - x0) * s, (cv - y0) * s
        label = str(ri)
        bb = draw.textbbox((tx, ty), label, font=fnt)
        pad = 3
        draw.rectangle([bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad],
                       fill=(0, 0, 0))
        draw.text((tx, ty), label, font=fnt,
                  fill=_PALETTE[ri % len(_PALETTE)])
    return crop, sorted(centroids.keys())


def _cloud_views(dst: Path, Pcloud_disp: np.ndarray,
                 region_pts_disp: List[Optional[np.ndarray]],
                 log=print) -> List[Path]:
    """Renders of the MEASURED point cloud (display frame, upright), each
    detected region tinted with its photo-overlay color and numbered — so the
    VLM can tell material from void (a 'sphere' on a hole's rim has no cloud
    behind it) and reason about WHERE each part sits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = Pcloud_disp[:: max(1, len(Pcloud_disp) // 25_000)]
    paths: List[Path] = []
    for k, (el, az) in enumerate([(18, 35), (18, -125), (70, 35)]):
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(sub[:, 0], sub[:, 2], sub[:, 1], s=0.3, c="0.82",
                   alpha=0.35, depthshade=False, rasterized=True)
        for ri, P in enumerate(region_pts_disp):
            if P is None or len(P) == 0:
                continue
            Q = P[:: max(1, len(P) // 1200)]
            col = np.asarray(_PALETTE[ri % len(_PALETTE)], float) / 255.0
            ax.scatter(Q[:, 0], Q[:, 2], Q[:, 1], s=5.0, c=[col],
                       depthshade=False, rasterized=True)
            c = P.mean(axis=0)
            ax.text(c[0], c[2], c[1], str(ri), color="white", fontsize=9,
                    weight="bold",
                    bbox=dict(facecolor=tuple(col), pad=1.2,
                              edgecolor="none", alpha=0.9))
        mn, mx = sub.min(axis=0), sub.max(axis=0)
        ctr, r = (mn + mx) / 2, float((mx - mn).max()) / 2 * 1.02
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[2] - r, ctr[2] + r)
        ax.set_zlim(ctr[1] - r, ctr[1] + r)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=el, azim=az)
        ax.set_axis_off()
        p = dst / f"cloud_view{k}.jpg"
        fig.tight_layout()
        fig.savefig(p, dpi=95)
        plt.close(fig)
        paths.append(p)
    log(f"  {len(paths)} measured-cloud view(s) rendered")
    return paths


# ── VLM calls ────────────────────────────────────────────────────────────

def _chat_json(client, messages, schema: Optional[dict], max_tokens: int,
               log=print):
    """chat() with vLLM structured output (response_format json_schema —
    verified working on our vLLM; the legacy `guided_json` key is silently
    IGNORED by it), tolerant parse + one correction turn as fallback.
    Returns (parsed, raw_text)."""
    from semantic.types import user as user_msg
    resp = None
    if schema is not None:
        try:
            resp = client.chat(messages, temperature=0.1,
                               max_tokens=max_tokens,
                               extra_body={"response_format": {
                                   "type": "json_schema",
                                   "json_schema": {"name": "proposal",
                                                   "schema": schema}}})
        except Exception as e:  # noqa: BLE001
            log(f"  structured output rejected ({e}) — free-form retry")
            resp = None
    if resp is None:
        resp = client.chat(messages, temperature=0.1, max_tokens=max_tokens)
    parsed = _extract_json(resp.content or "")
    if parsed is None:
        fix = messages + [
            user_msg("Your previous answer was not valid JSON. Reply again "
                     "with ONLY the JSON, no prose, no code fences.")]
        resp = client.chat(fix, temperature=0.0, max_tokens=max_tokens)
        parsed = _extract_json(resp.content or "")
    return parsed, (resp.content or "")


_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "que_es": {"type": "string"},
        "descripcion_detallada": {"type": "string"},
        "caracteristicas": {"type": "array", "items": {"type": "string"}},
        "materiales": {"type": "array", "items": {"type": "string"}},
        "estado_aparente": {"type": "string"},
        "funcion": {"type": "string"},
        "interaccion_con_entorno": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["que_es", "descripcion_detallada", "caracteristicas",
                 "materiales", "funcion", "interaccion_con_entorno",
                 "confidence"],
}


_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "identity": {"type": "string"},
        "description": {"type": "string"},
        "category": {"type": "string"},
        "expected_symmetry": {
            "type": "string",
            "enum": ["mirror", "rotational", "both", "none", "unknown"]},
        "confidence": {"type": "number"},
    },
    "required": ["identity", "description", "confidence"],
}


def _regions_schema(indices: List[int]) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "region": {"type": "integer", "enum": indices},
                "proposed_kind": {"type": "string", "enum": _KINDS},
                "part_role": {"type": "string"},
                "location": {"type": "string"},
                "outline": {"type": "string", "enum": _OUTLINES},
                "relations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["region", "proposed_kind", "part_role", "location",
                         "outline", "relations", "confidence"],
        },
    }


# ── main entry (worker mode 'propose') ───────────────────────────────────

def propose_object(output_dir: Path, instance_id: int,
                   cfg: Optional[dict] = None,
                   source: Optional[str] = None,
                   log=print) -> Optional[Path]:
    """Describe the object + propose per-region shape intent with Qwen3-VL.
    Writes output/shape_proposals/<safe>_proposal.json (+ the evidence crops)
    and caches the proposal in scene_r.db meta. Returns the json path."""
    from PIL import Image
    from segmentation.tsdf_export import _safe_label
    from segmentation.perfect_object import (_detect_and_snap,
                                             _detect_mirror_symmetry,
                                             _to_display)
    from reconstruction.surface_fit.hole_audit import _evidence
    from semantic.client import get_semantic_client
    from semantic.types import system as sys_msg, user as user_msg

    t0 = time.time()
    cfg = cfg or {}
    n_views = int(cfg.get("propose_views", 4))
    n_obj_views = int(cfg.get("propose_object_views", 3))
    per_call = int(cfg.get("propose_regions_per_call", 14))
    out = Path(output_dir)
    session_dir = out.parent

    result = json.loads((out / "segmentation_result.json").read_text())
    inst = next((i for i in result.get("instances", [])
                 if int(i.get("instance_id", i.get("id"))) == int(instance_id)),
                None)
    if inst is None:
        raise ValueError(f"instance {instance_id} not found")
    label = str(inst.get("label", "segment"))
    safe = _safe_label(label, int(instance_id))

    # 1) detection engine — the same regions the model rebuild / p2c use
    if bool(cfg.get("p2c_from_cloud", True)):
        # USER 2026-09-04: label CLOUD points, never mesh vertices — the
        # mesh already carries its own errors
        from segmentation.perfect_object import (_detect_and_snap_cloud,
                                                 _load_instance_cloud)
        P_raw_full = _load_instance_cloud(out, inst, cfg, safe, log)
        detect_max_pts = int(cfg.get("p2c_detect_max_pts", 150_000))
        rng0 = np.random.default_rng(11)
        sub = (np.arange(len(P_raw_full)) if len(P_raw_full) <= detect_max_pts
               else np.sort(rng0.choice(len(P_raw_full), detect_max_pts,
                                        replace=False)))
        V_raw = P_raw_full[sub]
        Vd = _to_display(out, V_raw)
        src_name = f"cloud (conf-trimmed, detect on {len(sub):,} pts)"
        regions, _ = _detect_and_snap_cloud(Vd, cfg, safe, log)
    else:
        from segmentation.perfect_object import _load_source_mesh
        tm, src = _load_source_mesh(out, safe, source)
        V_raw = np.asarray(tm.vertices, np.float64)
        F = np.asarray(tm.faces, np.int64)
        Vd = _to_display(out, V_raw)
        src_name = src.parent.name
        regions, _ = _detect_and_snap(tm, F, Vd, cfg, safe, log)
    if not regions:
        raise RuntimeError(f"{safe}: no regions detected — nothing to propose")
    log(f"[propose:{safe}] {len(regions)} region(s) from {src_name}")

    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(out / "cleaned_cloud.ply"))
    xyz = np.asarray(pc.points)
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    Pcloud_disp = _to_display(out, xyz[gi])
    sym = _detect_mirror_symmetry(Pcloud_disp, regions, log, safe)

    # 2) evidence: masks + cameras (raw frame — the GLB verts already are)
    ev = _evidence(out, session_dir)
    if not ev.ok:
        raise RuntimeError(f"{safe}: no mask/camera evidence "
                           "(seg_masks.npz / poses missing)")
    inst_pts = xyz[gi]
    if len(inst_pts) > 200_000:
        inst_pts = inst_pts[:: len(inst_pts) // 200_000]
    oid = _calibrate_oid_lenient(ev, inst_pts, int(instance_id), log=log)
    if oid is None:
        raise RuntimeError(f"{safe}: could not calibrate mask object id")
    mask_frames = ev.frames_for(oid)
    log(f"[propose:{safe}] oid={oid}, {len(mask_frames)} mask keyframes")

    # cap: propose only on the N largest regions — a noisy source mesh can
    # shatter into 100+ slivers and drown both the overlay and the VLM
    max_regions = int(cfg.get("propose_max_regions", 40))
    order = sorted(range(len(regions)),
                   key=lambda i: len(regions[i]["v_idx"]), reverse=True)
    keep = set(order[:max_regions])
    if len(regions) > max_regions:
        log(f"[propose:{safe}] {len(regions)} regions — proposing on the "
            f"{max_regions} largest")

    rng = np.random.default_rng(3)
    region_pts_raw: List[Optional[np.ndarray]] = []
    for i, r in enumerate(regions):
        if i not in keep:
            region_pts_raw.append(None)
            continue
        P = V_raw[r["v_idx"]]
        if len(P) > 3000:
            P = P[rng.choice(len(P), 3000, replace=False)]
        region_pts_raw.append(P)

    # 3) evidence images
    dst = out / "shape_proposals" / safe
    dst.mkdir(parents=True, exist_ok=True)
    inst_zbuf_cache: dict = {}
    score_pts = [(P[:: max(1, len(P) // 400)] if P is not None
                  else np.zeros((0, 3))) for P in region_pts_raw]
    picked = _pick_frames(ev, oid, mask_frames, score_pts, n_views,
                          inst_pts, inst_zbuf_cache, log=log)
    if not picked:
        raise RuntimeError(f"{safe}: no usable keyframes")

    annotated = []
    covered_regions: set = set()
    draw_pts = [(P if P is not None else np.zeros((0, 3)))
                for P in region_pts_raw]
    for fidx, key in picked:
        ac = _annotated_crop(ev, fidx, key, session_dir, draw_pts,
                             inst_pts, inst_zbuf_cache)
        if ac is None:
            continue
        crop, present = ac
        p = dst / f"regions_f{fidx}.jpg"
        crop.save(p, quality=90)
        annotated.append({"frame": fidx, "path": p, "regions": present})
        covered_regions.update(present)

    # TARGETED second pass: regions the main views missed get their own
    # views — only THEY are drawn (original numbering kept), and the
    # visibility bar drops (they are small; 15 projected points suffice)
    missing = sorted(set(keep) - covered_regions)
    extra_budget = int(cfg.get("propose_extra_views", 3))
    if missing and extra_budget > 0:
        log(f"[propose:{safe}] {len(missing)} region(s) not covered — "
            f"targeted views: {missing}")
        used = {f for f, _k in picked}
        cand = [(f, k) for f, k in mask_frames
                if all(abs(f - u) >= 10 for u in used)]
        miss_score = [(score_pts[i] if i in missing else np.zeros((0, 3)))
                      for i in range(len(regions))]
        extra = _pick_frames(ev, oid, cand, miss_score, extra_budget,
                             inst_pts, inst_zbuf_cache, min_gap=15, log=log)
        miss_draw = [(draw_pts[i] if i in missing else np.zeros((0, 3)))
                     for i in range(len(regions))]
        for fidx, key in extra:
            ac = _annotated_crop(ev, fidx, key, session_dir, miss_draw,
                                 inst_pts, inst_zbuf_cache, min_pts=8)
            if ac is None:
                continue
            crop, present = ac
            p = dst / f"regions_f{fidx}_extra.jpg"
            crop.save(p, quality=90)
            annotated.append({"frame": fidx, "path": p, "regions": present})
            covered_regions.update(present)

    # measured-cloud views (display frame): material vs void + placement
    region_pts_disp = [(Vd[r["v_idx"]][:: max(1, len(r["v_idx"]) // 1500)]
                        if i in keep else None)
                       for i, r in enumerate(regions)]
    cloud_paths = _cloud_views(dst, Pcloud_disp, region_pts_disp, log=log)

    iso = []
    by_area = sorted(mask_frames,
                     key=lambda fk: int((ev.masks[fk[1]] > 0).sum()),
                     reverse=True)
    for fidx, key in by_area[:n_obj_views]:
        img = _load_frame_rgb(session_dir, fidx)
        if img is None:
            continue
        m = ev.masks[key] > 0
        mrgb = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                          .resize(img.size, Image.NEAREST)) > 0
        crop = _isolated_crop(img, mrgb)
        p = dst / f"object_f{fidx}.jpg"
        crop.save(p, quality=90)
        iso.append({"frame": fidx, "path": p})
    if not annotated and not iso:
        raise RuntimeError(f"{safe}: could not build any evidence image")

    # 4) VLM — object identity first (isolated crops)
    client = get_semantic_client(consumer="shape_proposer")
    sym_line = ""
    if sym and sym.get("accepted"):
        sym_line = ("Tool-measured fact: the object HAS a vertical mirror "
                    "symmetry plane (verified against the point cloud). ")
    obj_prop: dict = {}
    if iso:
        msgs = [
            sys_msg("You are an expert in construction and industrial "
                    "equipment identifying a 3D-scanned object. You only "
                    "PROPOSE identity and intent; you never estimate "
                    "dimensions — measurements come from instruments."),
            user_msg(
                "These photos show one object segmented from a 3D scan "
                f"(segmentation label: '{label}'; background darkened). "
                f"{sym_line}"
                "The LAST image is a render of its measured 3-D point cloud. "
                "Identify it. Reply ONLY with JSON: "
                '{"identity": short name, "description": 2-3 sentences in '
                'Spanish (what it is, its function, its overall shape), '
                '"category": general category, "expected_symmetry": one of '
                'mirror|rotational|both|none|unknown, "confidence": 0-1}',
                images=[str(e["path"]) for e in iso]
                + [str(cloud_paths[0])] if cloud_paths else
                [str(e["path"]) for e in iso]),
        ]
        parsed, _raw = _chat_json(client, msgs, _OBJECT_SCHEMA, 500, log=log)
        if isinstance(parsed, dict):
            obj_prop = parsed
            log(f"[propose:{safe}] object: "
                f"{obj_prop.get('identity')} ({obj_prop.get('confidence')})")

    # 4b) DEEP ANALYSIS for the chat (user 2026-09-03: "debe conocer
    # perfectamente qué es cada elemento, características, materiales, cómo
    # interactúa con el resto"): context frame (object in its surroundings)
    # + isolated crops + measured extent → detailed Spanish dossier, cached
    # in the store as object_analysis_<iid> (provenance vlm_proposed).
    analysis: dict = {}
    ctx_path = None
    if iso:
        try:
            from PIL import ImageDraw
            fidx0, key0 = by_area[0]
            img = _load_frame_rgb(session_dir, fidx0)
            if img is not None:
                m0 = ev.masks[key0]
                ys, xs = np.where(m0 > 0)
                W, H = img.size
                mh0, mw0 = m0.shape
                dr = ImageDraw.Draw(img)
                dr.rectangle([int(xs.min() * W / mw0), int(ys.min() * H / mh0),
                              int(xs.max() * W / mw0), int(ys.max() * H / mh0)],
                             outline=(255, 40, 40), width=6)
                s = min(1.0, 1280 / max(img.size))
                if s < 1.0:
                    img = img.resize((int(img.width * s), int(img.height * s)))
                ctx_path = dst / "context_frame.jpg"
                img.save(ctx_path, quality=88)
        except Exception as e:  # noqa: BLE001
            log(f"[propose:{safe}] context frame failed ({e})")
        try:
            from segmentation.object_analysis import analyze_object
            analysis = analyze_object(
                out, int(instance_id),
                evidence_images=([str(ctx_path)] if ctx_path else [])
                + [str(e["path"]) for e in iso[:3]]
                + ([str(cloud_paths[0])] if cloud_paths else []),
                extent_m=Pcloud_disp.ptp(axis=0),
                identity_hint=str(obj_prop.get("identity") or "") or None,
                sym_line=sym_line, refresh=True, log=log) or {}
            if analysis:
                log(f"[propose:{safe}] analysis: "
                    f"{str(analysis.get('que_es'))[:60]} · materiales "
                    f"{analysis.get('materiales')}")
        except Exception as e:  # noqa: BLE001
            log(f"[propose:{safe}] deep analysis failed ({e})")

    # 5) VLM — per-region intent (annotated crops, chunked asks)
    facts = [_region_facts(i, r, Vd[r["v_idx"]])
             for i, r in enumerate(regions)]
    for i, r in enumerate(regions):
        if region_pts_raw[i] is None:
            continue
        vr = _interior_void_ratio(r, Vd)
        if vr is not None:
            facts[i]["interior_void_ratio"] = vr
    proposals: Dict[int, dict] = {}
    ask_idx = sorted(covered_regions) if covered_regions else sorted(keep)
    for c0 in range(0, len(ask_idx), per_call):
        chunk = ask_idx[c0:c0 + per_call]
        chunk_facts = [facts[i] for i in chunk]
        # only the views that actually show this chunk's regions, plus the
        # measured-cloud renders (8-image budget: ≤5 photos + ≤3 cloud)
        chunk_views = [e for e in annotated
                       if set(e["regions"]) & set(chunk)] or annotated
        chunk_views = chunk_views[:5]
        chunk_images = [str(e["path"]) for e in chunk_views] \
            + [str(p) for p in cloud_paths[:3]]
        msgs = [
            sys_msg("You are an expert in CAD reverse-engineering of scanned "
                    "objects. You PROPOSE design intent for detected surface "
                    "regions; you never measure — all numbers you see were "
                    "measured by deterministic tools and are context only."),
            user_msg(
                f"Object: {obj_prop.get('identity') or label}. "
                f"{obj_prop.get('description') or ''}\n"
                f"{sym_line}"
                "In the attached PHOTOS, each detected surface region is "
                "tinted with a color and numbered (white number on black "
                "box). The 3-D POINT-CLOUD renders (gray points = measured "
                "material, same region colors/numbers) show the real "
                "geometry: where the cloud has NO material, there is VOID. "
                "In the facts, 'interior_void_ratio' is tool-measured VOID "
                "evidence: how much of the region's surface interior has NO "
                "measured points (enclosed emptiness). A high value (>0.5) "
                "means a ring of material around a void — almost certainly "
                "an 'opening' (hole), not a solid part. "
                "Measured facts per region (tool_measured):\n"
                + json.dumps(chunk_facts, ensure_ascii=False)
                + "\n\nFor EACH region listed above, propose the DESIGN "
                "INTENT of that part of the real object:\n"
                "- proposed_kind: one of "
                + "|".join(_KINDS)
                + " (what the designer intended, e.g. a slightly bumpy face "
                "of a concrete block is a 'plane'; a pipe is a 'cylinder'; "
                "'extrusion' = constant cross-section swept along a line; "
                "'opening' = NOT a part at all — a HOLE/void in a plate "
                "whose rim the detector wrongly fitted as a surface (check "
                "the photos AND the cloud: an opening shows background "
                "through it and has no material of its own); 'freeform' "
                "only when nothing regular applies)\n"
                "- part_role: short English noun phrase — what this part IS "
                "on the object (e.g. 'front face of the beam', 'bolt hole', "
                "'rail head')\n"
                "- location: short phrase saying WHERE on the object the "
                "part sits, consistent with BOTH the photos and the cloud "
                "views (e.g. 'top of the front plate', 'left end of the "
                "base', 'centre, above the pipe')\n"
                "- outline: the intended 2-D boundary shape of the region "
                "(one of " + "|".join(_OUTLINES) + ")\n"
                "- relations: expected geometric relations as short tags, "
                "only when clearly intended: 'axis_vertical', "
                "'axis_horizontal', 'coaxial_with:<n>', 'same_radius_as:<n>',"
                " 'parallel_to:<n>', 'perpendicular_to:<n>', 'mirror_of:<n>',"
                " 'coplanar_with:<n>'\n"
                "- confidence: 0-1.\n"
                "If the detected_kind looks wrong for the intent (e.g. a "
                "sphere fitted on a hole's rim, a cylinder fitted on what "
                "is clearly a flat face), say so via "
                "proposed_kind. Reply ONLY with a JSON array with EXACTLY "
                f"one entry per region, in this order: {chunk}. Every entry "
                "MUST carry its \"region\" number.",
                images=chunk_images),
        ]
        for attempt in range(2):
            parsed, _raw = _chat_json(client, msgs, _regions_schema(chunk),
                                      220 * max(4, len(chunk)), log=log)
            got = 0
            if isinstance(parsed, list):
                # order-zip fallback: a full-length answer with the region
                # ids missing is still usable (the ask fixes the order)
                if (len(parsed) == len(chunk)
                        and not any(isinstance(e, dict) and "region" in e
                                    for e in parsed)):
                    for ri, e in zip(chunk, parsed):
                        if isinstance(e, dict):
                            e["region"] = ri
                for e in parsed:
                    if not isinstance(e, dict):
                        continue
                    try:
                        ri = int(e.get("region"))
                    except Exception:  # noqa: BLE001
                        continue
                    if ri in chunk and ri not in proposals:
                        proposals[ri] = e
                        got += 1
            if got or attempt:
                break
            log(f"[propose:{safe}] chunk {chunk} came back empty — retrying")
        log(f"[propose:{safe}] regions {chunk[0]}–{chunk[-1]}: "
            f"{sum(1 for i in chunk if i in proposals)}/{len(chunk)} proposed")

    # 6) deterministic reconciliation + deliverable
    region_entries = []
    n_agree = 0
    for i, r in enumerate(regions):
        p = proposals.get(i)
        entry = dict(facts[i])
        if p:
            kind = str(p.get("proposed_kind", "")).strip().lower()
            agrees = kind in _KIND_MATCH.get(r["kind"], {r["kind"]})
            n_agree += int(agrees)
            entry["proposal"] = {
                "proposed_kind": kind if kind in _KINDS else "freeform",
                "part_role": str(p.get("part_role", ""))[:120],
                "location": str(p.get("location", ""))[:120],
                "outline": (str(p.get("outline", "unknown")).strip().lower()
                            if str(p.get("outline", "")).strip().lower()
                            in _OUTLINES else "unknown"),
                "relations": [str(x)[:40] for x in
                              (p.get("relations") or [])][:8],
                "confidence": float(np.clip(
                    float(p.get("confidence", 0.0) or 0.0), 0.0, 1.0)),
                "agrees_with_fit": bool(agrees),
                "provenance": "vlm_proposed",
            }
        else:
            entry["proposal"] = None
        region_entries.append(entry)

    deliverable = {
        "method": "shape_proposal",
        "instance_id": int(instance_id),
        "label": label,
        "source_mesh": src_name,
        "object": ({**obj_prop, "provenance": "vlm_proposed"}
                   if obj_prop else None),
        "object_analysis": analysis or None,
        "symmetry_mirror": sym,
        "n_regions": len(regions),
        "n_proposed": len(proposals),
        "n_agree_with_fit": n_agree,
        "regions": region_entries,
        "views": {
            "annotated": [{"frame": e["frame"], "file": e["path"].name,
                           "regions": e["regions"]} for e in annotated],
            "isolated": [{"frame": e["frame"], "file": e["path"].name}
                         for e in iso],
            "cloud": [p.name for p in cloud_paths],
        },
        "mask_oid": int(oid),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_json = out / "shape_proposals" / f"{safe}_proposal.json"
    out_json.write_text(json.dumps(deliverable, indent=2, ensure_ascii=False))

    try:  # cache for the chat (best-effort — the json file is the truth)
        from phase_r.instance_store import InstanceStore
        db = out / "scene_r.db"
        if db.exists():
            store = InstanceStore(str(db))
            store.set_meta(f"shape_proposal_{int(instance_id)}",
                           json.dumps(deliverable, ensure_ascii=False))
            if analysis:
                store.set_meta(f"object_analysis_{int(instance_id)}",
                               json.dumps(analysis, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        log(f"[propose:{safe}] store cache skipped ({e})")

    log(f"[propose:{safe}] ✅ {len(proposals)}/{len(regions)} regions "
        f"proposed, {n_agree} agree with the fit · "
        f"object: {obj_prop.get('identity', '?')} → {out_json.name} "
        f"({deliverable['elapsed_s']}s)")
    return out_json
