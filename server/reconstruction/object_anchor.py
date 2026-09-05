"""
FASE 5 v2 — OBJECT-LEVEL SEMANTIC ANCHORING (USER DESIGN 2026-09-05).

The appearance-only landmark idea was proven WRONG by the user's eye (the
precheck: semantic features cannot answer identity — every long-range pair
was false). This is his corrected architecture:

  1. Qwen (the session VLM) looks at keyframes and produces SHORT,
     DESCRIPTIVE object prompts ("matafuego rojo colgado", "gabinete
     eléctrico ventilado") — not bare nouns.
  2. SAM 3.1 segments ALL keyframes with those prompts (the pipeline's own
     batched runner, with tracking) → per-frame instance masks.
  3. Every observation = mask ∩ the frame's OWN measured points (per-point
     provenance), cleaned with a PER-SEGMENT SOR (user order: tight bbox,
     no floaters) → robust position + viewing ray + measured ray-depth.
  4. IDENTITY BY RAYS, never by measured world position (the measured
     position is displaced by the very depth error being corrected): the
     same physical object's rays from different poses CONVERGE; another
     look-alike exemplar's rays triangulate somewhere else.
  5. Instance consensus = robust ray triangulation. Each observation's
     depth error = measured ray-depth − depth-to-consensus along its ray.
  6. Correction unit = THE CHUNK (Ω's error anatomy: a chunk misplaces its
     objects as a block): per-chunk log depth-scale solved globally with
     neighbour smoothness; applying one chunk re-balances the rest through
     the graph, never by hand.
  7. Gates (industrial): CROSS-VALIDATION BY OBJECTS — solve on instance
     set A, instance set B (never seen by the solver) must ALSO tighten;
     DA3 metric agreement must not degrade; capacity follows evidence
     (caps wide enough for the measured error class), every failure RAISES.
  8. Apply = REBUILD the chunk: depths rescaled along each point's own ray,
     PLYs rewritten. instances.json = the session's semantic memory.

Provenance: VLM output is vlm_proposed (identity proposals); every number
is tool_measured. Nothing outside the pipeline counts as validation.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ObjectAnchorError(RuntimeError):
    """Fase-5v2 failure — always raised, never swallowed."""


# ── session geometry ─────────────────────────────────────────────────────

def _load_frame_points(output_dir: Path, log):
    """Per real-frame-number: (pixel_row, pixel_col, xyz) from the chunk
    PLYs + provenance, plus frame→chunk ownership and the ply handles for
    the final rebuild. frame_ownership ON ⇒ one writer per frame."""
    from plyfile import PlyData
    chunk_files = [p for p in sorted(output_dir.glob("chunk_*.ply"))
                   if not p.stem.startswith(("chunk_997", "chunk_998",
                                             "chunk_999"))]
    if not chunk_files:
        raise ObjectAnchorError("no chunk PLYs — this stage runs before the "
                                "cloud merge")
    frame_pts: Dict[int, list] = {}
    frame_chunk: Dict[int, int] = {}
    plys = []
    Hg = Wg = None
    import re
    for p in chunk_files:
        ci = int(re.search(r"(\d+)", p.stem).group(1))
        org = output_dir / f"{p.stem}_origins.npz"
        if not org.exists():
            raise ObjectAnchorError(f"{org.name} missing — provenance is "
                                    f"mandatory")
        z = np.load(org)
        pd = PlyData.read(str(p))
        v = np.array(pd["vertex"].data)
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        fg = z["frame_global"].astype(np.int64)
        if len(fg) != len(xyz):
            raise ObjectAnchorError(f"{p.name}: points != origins")
        if Hg is None and "scaled_resolution" in z.files:
            Hg, Wg = (int(x) for x in z["scaled_resolution"])
        pr = z["pixel_row"].astype(np.int32)
        pc = z["pixel_col"].astype(np.int32)
        plys.append((p, ci, v, fg))
        for f in np.unique(fg):
            m = fg == f
            frame_pts.setdefault(int(f), []).append(
                (pr[m], pc[m], xyz[m]))
            frame_chunk[int(f)] = ci
    if Hg is None:
        raise ObjectAnchorError("no scaled_resolution in origins")
    out = {f: (np.concatenate([a for a, _b, _c in L]),
               np.concatenate([b for _a, b, _c in L]),
               np.concatenate([c for _a, _b, c in L]))
           for f, L in frame_pts.items()}
    log(f"[obj-anchor] {len(out)} frames with points from "
        f"{len(chunk_files)} chunks (grid {Hg}x{Wg})")
    return out, frame_chunk, plys, (Hg, Wg), chunk_files


def _segment_sor(P: np.ndarray, knn: int = 8, sigma: float = 2.0
                 ) -> np.ndarray:
    """PER-SEGMENT SOR (USER 2026-09-05: 'a cada segmento correle los
    filtros de cloudcompy — bbox ajustado y sin voladores'). Boolean keep."""
    from scipy.spatial import cKDTree
    n = len(P)
    if n < knn + 2:
        return np.ones(n, bool)
    d, _ = cKDTree(P).query(P, k=knn + 1, workers=8)
    md = d[:, 1:].mean(axis=1)
    return md <= md.mean() + sigma * md.std()


# ── 1: VLM vocabulary (short descriptive prompts) ────────────────────────

def _vlm_vocabulary(output_dir: Path, frames_dir: Path, frame_files: list,
                    cfg: dict, log) -> List[str]:
    from PIL import Image
    from semantic.client import get_semantic_client
    from semantic.types import system as sys_msg, user as user_msg
    from segmentation.shape_proposer import _chat_json

    stride = int(cfg.get("vlm_stride", 8))
    max_calls = int(cfg.get("vlm_max_calls", 200))
    picked = frame_files[::stride][:max_calls]
    client = get_semantic_client(consumer="object_anchor")
    schema = {"type": "object", "properties": {"objects": {
        "type": "array", "maxItems": 10, "items": {
            "type": "object", "properties": {
                "prompt": {"type": "string", "maxLength": 60}},
            "required": ["prompt"]}}}, "required": ["objects"]}
    vocab: Dict[str, int] = {}
    for i, ff in enumerate(picked):
        img = Image.open(frames_dir / ff).convert("RGB")
        img.thumbnail((768, 768))
        msgs = [
            sys_msg("You list the distinct PHYSICAL OBJECTS visible in a "
                    "technical-room scan frame, as SHORT but DESCRIPTIVE "
                    "Spanish phrases usable as segmentation prompts "
                    "(e.g. 'matafuego rojo colgado', 'gabinete eléctrico "
                    "gris ventilado', 'puerta metálica con barral'). "
                    "Distinct fixed objects only — no floors, walls, "
                    "ceilings, cables in bulk, or people."),
            user_msg("List the distinct objects in this frame. Reply ONLY "
                     "with JSON: {\"objects\":[{\"prompt\": \"...\"}]}",
                     images=[img]),
        ]
        parsed, _raw = _chat_json(client, msgs, schema, 400, log=log)
        for o in (parsed or {}).get("objects", []):
            p = str(o.get("prompt", "")).strip().lower()
            if 3 <= len(p) <= 60:
                vocab[p] = vocab.get(p, 0) + 1
        if (i + 1) % 20 == 0:
            log(f"[obj-anchor] VLM {i + 1}/{len(picked)} frames — "
                f"{len(vocab)} prompt(s)")
    # keep prompts seen in ≥2 sampled frames (noise floor), cap the list
    kept = sorted((p for p, c in vocab.items() if c >= 2),
                  key=lambda p: -vocab[p])[:int(cfg.get("max_prompts", 24))]
    if not kept:
        raise ObjectAnchorError("VLM produced no reusable object prompts")
    log(f"[obj-anchor] vocabulary ({len(kept)}): {kept}")
    return kept


# ── 4/5: identity by rays + consensus ────────────────────────────────────

def _ray_point(obs):
    """(camera C, unit ray r, measured depth d) of one observation."""
    return obs["cam"], obs["ray"], obs["depth"]


def _rays_close(o1, o2, tol_m: float) -> bool:
    C1, r1, _ = _ray_point(o1)
    C2, r2, _ = _ray_point(o2)
    n = np.cross(r1, r2)
    nn = np.linalg.norm(n)
    if nn < 1e-6:                       # near-parallel: distance line-line
        return float(np.linalg.norm(np.cross(C2 - C1, r1))) <= tol_m
    return abs(float((C2 - C1) @ (n / nn))) <= tol_m


def _triangulate(obs_list) -> Tuple[np.ndarray, float]:
    """Robust LSQ point minimizing distance to all rays (IRLS)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    w = np.ones(len(obs_list))
    x = None
    for _ in range(3):
        A[:] = 0
        b[:] = 0
        for wi, o in zip(w, obs_list):
            C, r, _ = _ray_point(o)
            P = np.eye(3) - np.outer(r, r)
            A += wi * P
            b += wi * (P @ C)
        x = np.linalg.solve(A + 1e-9 * np.eye(3), b)
        res = np.array([np.linalg.norm(
            (np.eye(3) - np.outer(o["ray"], o["ray"])) @ (x - o["cam"]))
            for o in obs_list])
        c = max(1.5 * float(np.median(res)), 1e-3)
        w = 1.0 / (1.0 + (res / c) ** 2)
    spread = float(np.median(res))
    return x, spread


# ── main ─────────────────────────────────────────────────────────────────

def run(output_dir: Path, frames_dir: Optional[Path] = None,
        cfg: Optional[dict] = None, log=logger.info) -> int:
    from plyfile import PlyData, PlyElement
    from reconstruction.dino_features import load_session_cameras
    cfg = cfg or {}
    ray_tol_m = float(cfg.get("ray_tol_m", 0.35))
    min_obs = int(cfg.get("min_obs_per_instance", 3))
    min_instances = int(cfg.get("min_instances", 6))
    holdout_frac = float(cfg.get("holdout_frac", 0.3))
    min_gain = float(cfg.get("min_gain", 0.10))
    max_log_scale = float(np.log(float(cfg.get("max_scale", 1.5))))
    smooth_w = float(cfg.get("smooth_w", 3.0))
    leash_w = float(cfg.get("leash_w", 1.0))
    mask_min_px = int(cfg.get("mask_min_px", 400))
    t0 = time.time()

    output_dir = Path(output_dir)
    if frames_dir is None:
        frames_dir = output_dir.parent / "frames"
    poses, frames, Ks = load_session_cameras(output_dir)
    fidx = {f: k for k, f in enumerate(frames)}
    frame_files = json.loads((output_dir / "frame_list.json").read_text())
    fnum_by_file = {ff: int(Path(ff).stem) for ff in frame_files}
    frame_pts, frame_chunk, plys, (Hg, Wg), chunk_files = \
        _load_frame_points(output_dir, log)
    n_chunks = 1 + max(ci for _p, ci, _v, _f in plys)

    # 1) VLM vocabulary — the semantic service must be up (GPU is free here)
    from config import cfg as full_cfg
    from semantic.service import ensure_running
    if not ensure_running(full_cfg, log=lambda m: log(f"[obj-anchor] {m}")):
        raise ObjectAnchorError("semantic service (Qwen) could not start — "
                                "the VLM pass is mandatory")
    vocab = _vlm_vocabulary(output_dir, frames_dir, frame_files, cfg, log)

    # 2) SAM 3.1 over ALL keyframes with the VLM prompts (pipeline's own
    # batched runner + tracking). Masks land in our own store for reuse.
    from segmentation.pipeline import _run_sam3_batched
    seg_cfg = full_cfg.get("segmentation", {}) or {}
    all_masks, obj_labels = _run_sam3_batched(
        Path(frames_dir), list(frame_files), list(vocab),
        int(seg_cfg.get("batch_size", 50)),
        int(seg_cfg.get("batch_overlap", 10)),
        float(seg_cfg.get("iou_threshold", 0.6)),
        output_dir=None, cfg=full_cfg)
    n_masks = sum(len(m) for m in all_masks.values())
    log(f"[obj-anchor] SAM: {n_masks} masks, "
        f"{len(obj_labels)} tracked object ids")
    if n_masks == 0:
        raise ObjectAnchorError("SAM produced no masks from the VLM prompts")

    # 3) observations: mask ∩ own measured points → SOR → ray + depth
    observations = []
    for f_i, masks in all_masks.items():
        ff = frame_files[f_i] if f_i < len(frame_files) else None
        if ff is None:
            continue
        fnum = fnum_by_file[ff]
        if fnum not in frame_pts or fnum not in fidx:
            continue
        pr, pc, xyz = frame_pts[fnum]
        k = fidx[fnum]
        C = poses[k][:3, 3]
        for oid, mask in masks.items():
            mh, mw = mask.shape
            rr = np.clip(pr.astype(np.int64) * mh // Hg, 0, mh - 1)
            cc = np.clip(pc.astype(np.int64) * mw // Wg, 0, mw - 1)
            sel = mask[rr, cc] > 0
            if sel.sum() < mask_min_px:
                continue
            P = xyz[sel]
            keep = _segment_sor(P)          # USER: tight bbox, no floaters
            if keep.sum() < mask_min_px // 2:
                continue
            P = P[keep]
            ctr = np.median(P, axis=0)
            ray = ctr - C
            d = float(np.linalg.norm(ray))
            if d < 0.3:
                continue
            observations.append({
                "frame": fnum, "chunk": frame_chunk[fnum],
                "obj_id": int(oid),
                "label": obj_labels.get(oid, "?"),
                "cam": C, "ray": ray / d, "depth": d,
                "n_pts": int(keep.sum()),
                "bbox": [[float(x) for x in P.min(0)],
                         [float(x) for x in P.max(0)]],
            })
    log(f"[obj-anchor] {len(observations)} clean observations "
        f"(mask ∩ own points, per-segment SOR)")
    if len(observations) < min_obs * min_instances:
        raise ObjectAnchorError(f"only {len(observations)} observations — "
                                f"not enough consensus material")

    # 4) instances: same tracked id OR (same label + ray convergence)
    from collections import defaultdict
    n_obs = len(observations)
    parent = list(range(n_obs))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_track = defaultdict(list)
    for i, o in enumerate(observations):
        by_track[(o["label"], o["obj_id"])].append(i)
    for idxs in by_track.values():                 # SAM tracking continuity
        for i in idxs[1:]:
            union(idxs[0], i)
    by_label = defaultdict(list)
    for i, o in enumerate(observations):
        by_label[o["label"]].append(i)
    # ray convergence with VERIFY-ON-MERGE (synthetic: a twin 4 m away
    # fooled 20% of single ray pairs — one false edge would fuse two
    # exemplars; the JOINT triangulation of a wrong merge blows its spread)
    def _members(root):
        return [i for i in range(n_obs) if find(i) == root]
    for idxs in by_label.values():
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                a, b = idxs[ii], idxs[jj]
                ra, rb = find(a), find(b)
                if ra == rb or not _rays_close(
                        observations[a], observations[b], ray_tol_m):
                    continue
                joint = [observations[i] for i in _members(ra)] +                         [observations[i] for i in _members(rb)]
                _x, spread = _triangulate(joint)
                if spread <= ray_tol_m:
                    union(a, b)
    groups = defaultdict(list)
    for i in range(n_obs):
        groups[find(i)].append(i)
    instances = []
    for gid, idxs in groups.items():
        if len(idxs) < min_obs:
            continue
        obs = [observations[i] for i in idxs]
        if len({o["chunk"] for o in obs}) < 2:
            continue                    # single-chunk: no anchoring power
        x, spread = _triangulate(obs)
        instances.append({"label": obs[0]["label"], "obs": obs,
                          "consensus": x, "ray_spread_m": spread})
    log(f"[obj-anchor] {len(instances)} multi-chunk instances "
        f"(≥{min_obs} obs, rays convergent)")
    if len(instances) < min_instances:
        raise ObjectAnchorError(
            f"only {len(instances)} usable instances (<{min_instances}) — "
            f"refusing to correct without cross-validation material")

    # registry — the session's semantic memory
    (output_dir / "instances.json").write_text(json.dumps({
        "n_instances": len(instances),
        "vocabulary": vocab,
        "instances": [{
            "label": it["label"],
            "consensus": [round(float(v), 4) for v in it["consensus"]],
            "ray_spread_m": round(it["ray_spread_m"], 4),
            "frames": sorted({o["frame"] for o in it["obs"]}),
            "chunks": sorted({o["chunk"] for o in it["obs"]}),
            "n_obs": len(it["obs"]),
        } for it in instances],
        "provenance": {"identity": "vlm_proposed",
                       "geometry": "tool_measured"},
    }, indent=1))

    # 5/6/7) per-chunk log depth-scale, cross-validated BY OBJECTS
    rng = np.random.default_rng(2)
    hold = rng.random(len(instances)) < holdout_frac
    train = [it for it, h in zip(instances, hold) if not h]
    held = [it for it, h in zip(instances, hold) if h]
    if not held:
        held = instances[-max(1, len(instances) // 4):]

    def _mismatch(insts, s_log):
        vals = []
        for it in insts:
            derr = []
            for o in it["obs"]:
                d_cons = float((it["consensus"] - o["cam"]) @ o["ray"])
                if d_cons <= 0.1:
                    continue
                derr.append(abs(np.log(o["depth"]) + s_log[o["chunk"]]
                                - np.log(d_cons)))
            if derr:
                vals.append(float(np.median(derr)))
        return float(np.median(vals)) if vals else float("inf")

    s = np.zeros(n_chunks)
    for _ in range(3):
        JtJ = np.zeros((n_chunks, n_chunks))
        Jty = np.zeros(n_chunks)
        for it in train:
            x, _sp = _triangulate(it["obs"])   # re-triangulate as s evolves
            it["consensus"] = x
            for o in it["obs"]:
                d_cons = float((x - o["cam"]) @ o["ray"])
                if d_cons <= 0.1:
                    continue
                r = np.log(d_cons) - np.log(o["depth"])   # target for s_c
                c = o["chunk"]
                w = 1.0 / (1.0 + (abs(r - s[c]) / 0.05) ** 2)
                JtJ[c, c] += w
                Jty[c] += w * r
        for c in range(n_chunks - 1):
            JtJ[c, c] += smooth_w
            JtJ[c + 1, c + 1] += smooth_w
            JtJ[c, c + 1] -= smooth_w
            JtJ[c + 1, c] -= smooth_w
        JtJ += leash_w * np.eye(n_chunks)
        s = np.clip(np.linalg.solve(JtJ, Jty),
                    -max_log_scale, max_log_scale)

    zero = np.zeros(n_chunks)
    h0 = _mismatch(held, zero)
    h1 = _mismatch(held, s)
    t0_all = _mismatch(train, zero)
    t1_all = _mismatch(train, s)
    log(f"[obj-anchor] depth mismatch (median |log|): train "
        f"{t0_all:.4f}→{t1_all:.4f} | HELD-OUT OBJECTS {h0:.4f}→{h1:.4f} | "
        f"chunk scale corr: median {np.exp(np.median(np.abs(s))):.4f}x "
        f"max {np.exp(np.abs(s).max()):.4f}x")

    # DA3 metric gate: corrected depths vs isolated DA3 anchors
    da3 = _da3_gate(output_dir, frame_pts, frame_chunk, poses, fidx, s, log)
    accepted = h1 <= h0 * (1.0 - min_gain)
    da3_ok = True if da3 is None else (da3[1] <= da3[0] * 1.02)
    report = {
        "n_observations": len(observations),
        "n_instances": len(instances),
        "n_train": len(train), "n_heldout": len(held),
        "mismatch_logdepth": {"train_before": t0_all, "train_after": t1_all,
                              "heldout_before": h0, "heldout_after": h1},
        "da3_gate_logratio": ({"before": da3[0], "after": da3[1]}
                              if da3 else None),
        "chunk_scale": {"median_x": float(np.exp(np.median(np.abs(s)))),
                        "max_x": float(np.exp(np.abs(s).max()))},
        "accepted": bool(accepted and da3_ok),
        "min_gain": min_gain,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    (output_dir / "object_anchor_report.json").write_text(
        json.dumps(report, indent=1))
    if not (accepted and da3_ok):
        why = ("held-out objects did not improve enough" if not accepted
               else "DA3 metric gate degraded")
        log(f"[obj-anchor] {why} — NOTHING APPLIED (identity). "
            f"Evidence in object_anchor_report.json")
        return 0

    # 8) REBUILD the corrected chunks: depths along each point's own ray
    moved = 0
    for p, ci, v, fg in plys:
        if abs(s[ci]) < 1e-9:
            continue
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        sc = float(np.exp(s[ci]))
        for f in np.unique(fg):
            k = fidx.get(int(f))
            if k is None:
                continue
            C = poses[k][:3, 3]
            m = fg == f
            xyz[m] = C + (xyz[m] - C) * sc
        v["x"] = xyz[:, 0].astype(v["x"].dtype)
        v["y"] = xyz[:, 1].astype(v["y"].dtype)
        v["z"] = xyz[:, 2].astype(v["z"].dtype)
        PlyData([PlyElement.describe(v, "vertex")], text=False,
                byte_order="<").write(str(p))
        moved += 1
    log(f"[obj-anchor] ✅ APPLIED: {moved}/{n_chunks} chunks REBUILT "
        f"(depth rescaled along rays) — {report['elapsed_s']}s")
    return moved


def _da3_gate(output_dir, frame_pts, frame_chunk, poses, fidx, s, log
              ) -> Optional[Tuple[float, float]]:
    """Median |log depth-ratio| vs isolated DA3 anchor depths, before vs
    after the chunk scales. None when no anchors are on disk."""
    import re
    adir = output_dir / "da3_run" / "results_output"
    if not adir.is_dir():
        return None
    bef, aft = [], []
    for f in sorted(adir.glob("frame_*.npz")):
        m = re.search(r"(\d+)", f.stem)
        if not m:
            continue
        num = int(m.group(1))
        if num not in frame_pts or num not in fidx:
            continue
        try:
            da = np.load(f)
            key = [x for x in da.files if "depth" in x.lower()]
            dd = np.asarray(da[key[0]]) if key else None
        except Exception:  # noqa: BLE001
            continue
        if dd is None or dd.ndim != 2:
            continue
        pr, pc, xyz = frame_pts[num]
        C = poses[fidx[num]][:3, 3]
        step = max(1, len(xyz) // 4000)
        P = xyz[::step]
        rr = pr[::step]
        cc = pc[::step]
        d_ours = np.linalg.norm(P - C, axis=1)
        Hg = rr.max() + 1
        Wg = cc.max() + 1
        dv = dd[np.clip(rr * dd.shape[0] // Hg, 0, dd.shape[0] - 1),
                np.clip(cc * dd.shape[1] // Wg, 0, dd.shape[1] - 1)]
        ok = (dv > 0.1) & (d_ours > 0.1)
        if ok.sum() < 30:
            continue
        sc = float(np.exp(s[frame_chunk[num]]))
        bef.append(np.median(np.abs(np.log(d_ours[ok] / dv[ok]))))
        aft.append(np.median(np.abs(np.log(d_ours[ok] * sc / dv[ok]))))
    if not bef:
        return None
    b, a = float(np.median(bef)), float(np.median(aft))
    log(f"[obj-anchor] DA3 metric gate: median |log ratio| {b:.4f} → "
        f"{a:.4f}")
    return b, a


def main():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    try:
        os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Fase 5 v2 — object-level semantic anchoring")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", default=None)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import cfg as _cfg
    dcfg = _cfg.get("dino_features") or {}
    run(Path(args.output_dir),
        Path(args.frames_dir) if args.frames_dir else None,
        cfg={**dcfg, **(dcfg.get("object_anchor") or {})},
        log=lambda m: logging.getLogger(__name__).info(m))


if __name__ == "__main__":
    main()
