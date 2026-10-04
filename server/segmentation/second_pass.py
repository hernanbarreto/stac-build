"""The SECOND VLM PASS over what stayed unsegmented (pending A "VLM refinement", 2026-10-04).

USER 2026-10-01: 100 % of the points segmented — a second VLM pass looks at what stayed unsegmented
and proposes the concepts it sees there. Measured on pccr's live epoch (2026-10-04): 14.6 % of the
points carry NO SAM3 mask at their source pixel; the 99.4 % coverage of today is the geometric growth
of ``_attach_unsegmented``, which hands them a neighbour's label. This pass gives them their own.

After the cloud stage projected the intake's masks on the published cloud:
 1. UNSEGMENTED PIXELS per keyframe, from the cloud's own provenance: a point whose source pixel lies
    in no masklet of its keyframe (``seg_masks.npz``) marks that pixel; keyframes ranked by how many
    such points they hold, at most ``max_calls`` of them looked at (BOUND: cost).
 2. THE VLM (the same Qwen3-VL, the same understanding prompt, one call per keyframe) sees the
    keyframe with the already-segmented pixels DIMMED to ``dim`` and the unsegmented region at full
    brightness, and is told to name only the kinds of objects in the bright region — per CONCEPT,
    never per object, the prompt's own rules.
 3. NEW CONCEPTS = the names no existing prompt already covers (same-name key, the merge map, the
    synonyms of the first pass); bounded by what remains of ``autoprompt.max_sam3_prompts``.
 4. SAM3 on the new prompts over every keyframe (``run_segmentation`` APPENDS to the mask store), the
    fallback phrases of the new concepts carried like the first pass's.
 5. The projection runs again on the published cloud; the census and ``vlm_analysis.json`` record
    the pass: keyframes seen, every name proposed and its fate, the new prompts, the masklets they
    produced, the unmasked share before and after.
The GPU hand-over is the stage's: vLLM up for the calls, down for SAM3. Nothing is deleted; a pass
that proposes nothing new is recorded as such and the first projection stands.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[second-pass]"
REPORT = "second_pass.json"

_EXTRA = ("ONLY the bright region of this image matters: the DIMMED pixels are already segmented. "
          "List only the kinds of objects that lie inside the BRIGHT region (whole or in part). If the "
          "bright region holds nothing nameable, return an empty objects list.")


class SecondPassError(RuntimeError):
    pass


# ── 1. the unsegmented pixels ─────────────────────────────────────────────────

def unmasked_pixels(output_dir: Path, log: Callable = print) -> Tuple[Dict[int, np.ndarray], dict]:
    """{keyframe position: bool map of the pixels whose cloud point carries no mask} and the
    accounting (points, unmasked points, share). Read from the live cloud's provenance."""
    from correction.session import read_ply
    from precision.silhouette_filter import masks_by_keyframe
    from segmentation.mask_space import keyframe_numbers
    out = Path(output_dir)
    ply = out / "cleaned_cloud.ply"
    if not ply.exists():
        raise SecondPassError(f"{ply} is missing")
    _, data = read_ply(ply)
    names = data.dtype.names or ()
    for k in ("frame_global", "pixel_row", "pixel_col"):
        if k not in names:
            raise SecondPassError(f"{ply.name} carries no per-point provenance ('{k}')")
    fg = np.asarray(data["frame_global"]).astype(np.int64)
    pr = np.asarray(data["pixel_row"]).astype(np.int64); pc = np.asarray(data["pixel_col"]).astype(np.int64)
    kfs = keyframe_numbers(out) or []
    pos_of = {int(f): i for i, f in enumerate(kfs)}
    mpath = out / "seg_masks.npz"
    masks = np.load(mpath) if mpath.exists() else None
    by_kf = masks_by_keyframe(out, masks) if masks is not None and len(masks.files) else {}
    shape = None
    if masks is not None and len(masks.files):
        shape = tuple(np.asarray(masks[masks.files[0]]).shape)
    maps: Dict[int, np.ndarray] = {}
    n_un = 0
    order = np.argsort(fg, kind="stable"); fs = fg[order]
    for f in np.unique(fg):
        i = pos_of.get(int(f))
        if i is None:
            continue
        a, b = np.searchsorted(fs, f, "left"), np.searchsorted(fs, f, "right")
        idx = order[a:b]
        if shape is None:
            H, W = int(pr[idx].max()) + 1, int(pc[idx].max()) + 1
        else:
            H, W = shape
        U = np.zeros((H, W), bool)
        for _, key in by_kf.get(i, []):
            U |= np.asarray(masks[key]) > 0
        ok = (pr[idx] >= 0) & (pr[idx] < H) & (pc[idx] >= 0) & (pc[idx] < W)
        covered = np.zeros(len(idx), bool); covered[ok] = U[pr[idx][ok], pc[idx][ok]]
        un = ~covered & ok
        if un.any():
            M = np.zeros((H, W), bool)
            M[pr[idx][un], pc[idx][un]] = True
            maps[i] = M
            n_un += int(un.sum())
    acct = {"points": int(len(fg)), "unmasked_points": int(n_un), "unmasked_share": n_un / max(len(fg), 1),
            "keyframes_with_unmasked": len(maps)}
    log(f"{LOG_TAG} {n_un:,} of {len(fg):,} points ({acct['unmasked_share'] * 100:.1f} %) carry no mask at their "
        f"source pixel, over {len(maps)} keyframes")
    return maps, acct


def highlight(image: np.ndarray, unmasked: np.ndarray, dim: float, grow_px: int) -> np.ndarray:
    """The keyframe with the already-segmented pixels dimmed to ``dim`` (0..1) and the unmasked pixels,
    grown by ``grow_px`` so the sparse cloud points become a region, at full brightness."""
    from scipy.ndimage import binary_dilation
    img = np.asarray(image)
    H, W = img.shape[:2]
    m = unmasked
    if m.shape != (H, W):
        import cv2
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    if grow_px > 0:
        m = binary_dilation(m, structure=np.ones((2 * grow_px + 1, 2 * grow_px + 1), bool))
    out = img.astype(np.float32)
    out[~m] *= float(dim)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── 2-3. the calls and the new concepts ──────────────────────────────────────

def new_concepts(proposed: Counter, existing_prompts: Sequence[str], merged: Dict[str, str],
                 synonyms: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    """(new prompt names, fate per proposed name): a name already covered by an existing prompt (same
    name key, the first pass's merge map or synonyms) is 'covered'; the rest are new, most frequent
    first, one per same-name key."""
    from segmentation.autoprompt.scene_understanding import same_name_key
    keys = {same_name_key(p): p for p in existing_prompts}
    fate: Dict[str, str] = {}
    new: Dict[str, Tuple[int, str]] = {}
    for name, n in proposed.most_common():
        c0 = merged.get(name, name)
        c = synonyms.get(c0, c0)
        k = same_name_key(c)
        if c in existing_prompts or k in keys:
            fate[name] = f"covered by '{keys.get(k, c)}'"
            continue
        if k in new:
            fate[name] = f"same name as '{new[k][1]}'"
            continue
        new[k] = (n, c)
        fate[name] = "new prompt"
    return [c for _, c in sorted(new.values(), key=lambda t: -t[0])], fate


def propose(client, frames_dir: Path, maps: Dict[int, np.ndarray], keyframes: Sequence[int], spcfg,
            log: Callable = print, cancelled: Optional[Callable[[], bool]] = None) -> Tuple[Counter, Dict[int, dict], List[dict]]:
    """One understanding call per selected keyframe (most unmasked first, at most ``max_calls``) on the
    highlighted image. Returns (name counts, per-keyframe descriptions, the calls' records)."""
    from PIL import Image
    from intake.content import frame_file
    from segmentation.autoprompt.scene_understanding import understand_frame
    ranked = sorted(maps.items(), key=lambda kv: -int(kv[1].sum()))[:int(spcfg.max_calls)]
    counts: Counter = Counter()
    descs: Dict[str, Counter] = {}
    calls: List[dict] = []
    for n, (i, M) in enumerate(ranked):
        if cancelled is not None and cancelled():
            break
        f = int(keyframes[i])
        try:
            img = np.asarray(Image.open(frame_file(frames_dir, f)).convert("RGB"))
        except (FileNotFoundError, OSError) as e:
            calls.append({"keyframe": i, "frame": f, "error": str(e)}); continue
        hi = highlight(img, M, float(spcfg.dim), int(spcfg.grow_px))
        fu = understand_frame(client, Image.fromarray(hi), frame_id=f, tile="second_pass", extra=_EXTRA)
        rec = {"keyframe": i, "frame": f, "unmasked_px": int(M.sum()), "objects": (fu.objects if fu else None)}
        calls.append(rec)
        if fu is None:
            continue
        for o in dict.fromkeys(fu.objects):
            counts[o] += 1
            de = (fu.descriptions or {}).get(o)
            if de and de != o:
                descs.setdefault(o, Counter())[de] += 1
        if n % 20 == 0:
            log(f"{LOG_TAG} {n + 1}/{len(ranked)} keyframes seen, {len(counts)} distinct names so far")
    return counts, {k: dict(v) for k, v in descs.items()}, calls


# ── the pass ─────────────────────────────────────────────────────────────────

def run_second_pass(session_dir: Path, config: dict, log: Callable = print,
                    progress: Optional[Callable[[float, str], None]] = None,
                    cancelled: Optional[Callable[[], bool]] = None,
                    client_factory: Optional[Callable] = None,
                    segment: Optional[Callable] = None,
                    project: Optional[Callable] = None,
                    stop_vllm: Optional[Callable] = None) -> dict:
    """The whole pass on a session whose cloud already carries the first projection. ``client_factory``
    / ``segment`` / ``project`` / ``stop_vllm`` are injectable (tests): the VLM client, SAM3 on a prompt
    string, the mask→cloud projection, the GPU hand-over before SAM3."""
    from segmentation.autoprompt.session_builder import with_category
    t0 = time.time()
    session_dir = Path(session_dir); out = session_dir / "output"
    ap = (config.get("autoprompt") or {})
    sp_raw = ap.get("second_pass")
    if not isinstance(sp_raw, dict):
        raise SecondPassError("config.yaml is missing 'autoprompt.second_pass' (enabled, max_calls, dim, grow_px)")
    from types import SimpleNamespace
    for k in ("enabled", "max_calls", "dim", "grow_px"):
        if k not in sp_raw:
            raise SecondPassError(f"config.yaml is missing 'autoprompt.second_pass.{k}'")
    spcfg = SimpleNamespace(**sp_raw)
    rep: dict = {"version": 1, "provenance": "vlm_proposed", "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def _p(pct, msg):
        log(f"{LOG_TAG} {msg}")
        if progress:
            progress(pct, msg)

    if not bool(spcfg.enabled):
        rep.update(skipped="autoprompt.second_pass.enabled is false")
        return rep
    vlm_path = out / "vlm_analysis.json"
    if not vlm_path.exists():
        raise SecondPassError(f"{vlm_path} is missing — the first pass must have run")
    vlm = json.loads(vlm_path.read_text())
    prompts = [p for p in str(vlm.get("prompt") or "").split(";") if p]
    su = vlm.get("scene_understanding") or {}
    merged = dict(su.get("merged") or {})
    synonyms = dict(((vlm.get("consolidation") or {}).get("synonyms") or {}) or {})
    try:
        cj = json.loads((out / "autoprompt_concepts.json").read_text())
        synonyms.update(cj.get("synonyms") or {})
    except (OSError, ValueError):
        pass
    maps, acct = unmasked_pixels(out, log)
    rep["before"] = acct
    from segmentation.mask_space import keyframe_numbers
    keyframes = keyframe_numbers(out) or []
    if not maps:
        rep.update(new_prompts=[], note="every point's source pixel carries a mask — nothing to look at")
        _write(out, vlm, rep); return rep
    _p(5, f"VLM on {min(len(maps), int(spcfg.max_calls))} keyframes with unsegmented pixels")
    if client_factory is None:
        from semantic.client import get_semantic_client
        from semantic.service import ensure_service
        if not ensure_service(config, log=log, cancelled=cancelled):
            raise SecondPassError("the semantic service (Qwen3-VL) did not come up")
        client = get_semantic_client(consumer="segmentation.second_pass")
    else:
        client = client_factory()
    counts, descs, calls = propose(client, session_dir / "frames", maps, keyframes, spcfg, log, cancelled)
    rep["calls"] = len(calls)
    rep["proposed"] = dict(counts)
    bound = int(ap.get("max_sam3_prompts", 0) or 0)
    new, fate = new_concepts(counts, prompts, merged, synonyms)
    rep["fates"] = fate
    if bound and len(prompts) + len(new) > bound:
        cut = max(bound - len(prompts), 0)
        for c in new[cut:]:
            fate[c] = fate.get(c, "new prompt") + " — not prompted (autoprompt.max_sam3_prompts)"
        new = new[:cut]
    rep["new_prompts"] = new
    log(f"{LOG_TAG} {len(counts)} distinct names proposed over {len(calls)} calls → {len(new)} new prompt(s): {new}")
    if not new:
        _write(out, vlm, rep); return rep
    # the fallbacks of the new concepts carry the category (session_builder.with_category)
    fb = {}
    for c in new:
        cand = [with_category(d, c) for d, _ in Counter(descs.get(c, {})).most_common()]
        cand = [x for x in cand if x != c][:int(ap.get("sam3_fallback_max", 3) or 3)]
        if cand:
            fb[c] = cand
    if stop_vllm is not None:
        stop_vllm()
    _p(40, f"SAM3 on {len(new)} new prompt(s) over every keyframe")
    prompt_status: dict = {}
    if segment is None:
        from segmentation.pipeline import run_segmentation
        res = run_segmentation(frames_dir=str(session_dir / "frames"), output_dir=str(out), prompt=";".join(new),
                               frame_map={}, boxes_map=None, prompt_status=prompt_status, fallback_prompts=fb,
                               on_progress=lambda pct, m: _p(40 + 0.4 * pct, m), defer_cloud_mapping=True)
    else:
        res = segment(";".join(new), prompt_status, fb)
    if isinstance(res, dict) and res.get("error"):
        raise SecondPassError(f"SAM3 on the new prompts failed: {res['error']}")
    rep["prompt_status"] = prompt_status
    _p(85, "projecting the masks on the published cloud again")
    if project is None:
        from segmentation.pipeline import map_segmentation_to_cloud
        seg = map_segmentation_to_cloud(out)
    else:
        seg = project(out)
    if isinstance(seg, dict) and seg.get("error"):
        raise SecondPassError(f"re-projection failed: {seg['error']}")
    rep["coverage_after_projection"] = (seg or {}).get("coverage")
    _, after = unmasked_pixels(out, log)
    rep["after"] = after
    rep["seconds"] = round(time.time() - t0, 1)
    vlm["prompt"] = ";".join(prompts + new)
    vlm.setdefault("fallback_prompts", {}).update(fb)
    _write(out, vlm, rep)
    log(f"{LOG_TAG} unmasked points {acct['unmasked_share'] * 100:.1f} % → {after['unmasked_share'] * 100:.1f} % "
        f"({rep['seconds']} s)")
    return rep


def _write(out: Path, vlm: dict, rep: dict) -> None:
    vlm["second_pass"] = rep
    (out / "vlm_analysis.json").write_text(json.dumps(vlm, indent=2, ensure_ascii=False, default=float))
    (out / REPORT).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
