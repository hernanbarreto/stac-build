"""SEGMENTATION IN ROUNDS.

USER 2026-10-08 (decided, docs/pipeline_final.md) and 2026-10-09 (ordered): "las pasadas son
recursivas es decir vlm sam 3 que queda sin segmentar vuelve a vlm con lo segmentado grisado genera
prompts sam 3 etc hasta que no se pueda generar más nada y sale".

Round 1 is the VLM stage and the SAM3 stage as always. Round k >= 2 (pipeline_manager runs it):
  VLM   every keyframe round 1 looked at, with every pixel a masklet already covers painted flat
        GREY, asked the session's understanding question plus one instruction: name only what is
        not grey. Names already prompted (same name key) are dropped; the rest go through the merge
        pass TOGETHER with the prompts so far — a name that is the same kind as, or a part of, a
        prompt already segmented is covered; what remains is the round's NEW prompts
        (output/seg_rounds.json).
  SAM3  the new prompts over every keyframe, APPENDED to the store (run_segmentation append_store).
The loop ends when a round names nothing new, or when autoprompt.max_sam3_prompts is reached (every
round that goes on adds at least one prompt, so that bound also ends it). The masks are then
projected on the cloud once.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ROUNDS_FILE = "seg_rounds.json"
GREY = 128
LOG_TAG = "[rounds]"

_EXTRA = ("Every region painted FLAT GREY in this image is already identified: ignore it "
          "completely. Name ONLY the objects whose visible part is NOT grey. If nothing outside the "
          "grey regions can be named, return an empty objects list.")

_MASK_KEY = re.compile(r"^f(\d+)_o(\d+)$")


def load_rounds(output_dir) -> dict:
    p = Path(output_dir) / ROUNDS_FILE
    if not p.is_file():
        return {"rounds": []}
    return json.loads(p.read_text())


def reset_rounds(output_dir) -> None:
    """A new chain of rounds starts (round 1 just segmented): the previous chain's record goes."""
    p = Path(output_dir) / ROUNDS_FILE
    if p.is_file():
        p.unlink()


def round_prompts(output_dir, k: int) -> List[str]:
    for r in load_rounds(output_dir).get("rounds", []):
        if int(r.get("round", 0)) == int(k):
            return list(r.get("new_prompts") or [])
    return []


def _save_round(output_dir, rec: dict) -> None:
    from atomic_io import atomic_write_json
    doc = load_rounds(output_dir)
    doc["rounds"] = [r for r in doc.get("rounds", []) if int(r.get("round", 0)) != int(rec["round"])]
    doc["rounds"].append(rec)
    doc["rounds"].sort(key=lambda r: int(r["round"]))
    atomic_write_json(Path(output_dir) / ROUNDS_FILE, doc, indent=1)


def store_prompts(output_dir) -> List[str]:
    """Every prompt the session's mask store was segmented with, every round so far."""
    p = Path(output_dir) / "segmentation.json"
    if not p.is_file():
        return []
    return list(json.loads(p.read_text()).get("prompts") or [])


def tried_prompts(output_dir, before_round: int) -> List[str]:
    """Every prompt SAM3 has already been given — round 1's, the store's and every earlier
    round's new ones — whether it found anything or not (zaragoza 2026-10-09: 'trough' found
    nothing in round 3, is not in the store, and came back as new in round 4; a name that keeps
    coming back would never let the rounds end)."""
    out = Path(output_dir)
    seen: List[str] = []
    vlm_p = out / "vlm_analysis.json"
    first = (str(json.loads(vlm_p.read_text()).get("prompt") or "").split(";")
             if vlm_p.is_file() else [])
    later = [p for r in load_rounds(out).get("rounds", [])
             if int(r.get("round", 0)) < int(before_round) for p in (r.get("new_prompts") or [])]
    for p in [*first, *store_prompts(out), *later]:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return seen


#: head nouns that name no kind of thing (USER 2026-10-09, bufferStop round 2: 'object',
#: 'red object', 'hanging object', 'yellow object' added 329 masklets — SAM3 searches the
#: ORIGINAL frames, so a prompt that names nothing finds everything again)
GENERIC_HEADS = frozenset({"object", "thing", "item", "stuff", "something", "anything",
                           "entity", "element", "unknown", "misc", "miscellaneous"})


def is_generic_name(name: str) -> bool:
    """True when the name's head noun (its last word, plural folded) names no kind of thing —
    whatever colour, size, position or material is put in front of it ('red object',
    'hanging object', 'metal item')."""
    words = [w for w in str(name).lower().replace("-", " ").replace("_", " ").split() if w]
    if not words:
        return True
    head = words[-1]
    if head.endswith("s") and head[:-1] in GENERIC_HEADS:
        head = head[:-1]
    return head in GENERIC_HEADS


def decided_names(output_dir, before_round: int) -> List[str]:
    """Every name an earlier merge already JUDGED — round 1's census (prompted or folded into a
    prompt) and every earlier round's fates (new, covered, merged). A judged name is not fresh:
    re-asked against a different list it can land elsewhere (observatorio 2026-10-09: round 1
    folded 'railway track' into 'railroad track'; round 2 judged it again, kept it, and SAM3
    segmented the track twice). A name dropped for one witness, or for the prompt bound, was
    never judged and may come back."""
    out = Path(output_dir)
    names: List[str] = []
    cen = out / "segmentation_census.json"
    if cen.is_file():
        for c in json.loads(cen.read_text()).get("concepts") or []:
            if c.get("concept") and c.get("fate") in ("prompt", "merged"):
                names.append(str(c["concept"]))
    for r in load_rounds(out).get("rounds", []):
        if int(r.get("round", 0)) >= int(before_round):
            continue
        for n, fate in (r.get("fates") or {}).items():
            if not str(fate).startswith("not prompted"):
                names.append(str(n))
    return names


def segmented_masks(output_dir) -> Dict[int, np.ndarray]:
    """{video frame: bool map of the pixels some masklet of the store covers}, at the store's grid."""
    from segmentation import mask_space
    out = Path(output_dir)
    union: Dict[int, np.ndarray] = {}
    with np.load(out / "seg_masks.npz") as npz:
        space = mask_space.declared_space(npz) or mask_space.SPACE_KEYFRAME
        kfs = mask_space.declared_keyframes(npz) or mask_space.keyframe_numbers(out) or []
        for key in npz.files:
            m = _MASK_KEY.match(key)
            if not m:
                continue
            f = int(m.group(1))
            if space == mask_space.SPACE_KEYFRAME:
                if f >= len(kfs):
                    continue
                f = int(kfs[f])
            a = np.asarray(npz[key]) > 0
            if f in union:
                union[f] |= a
            else:
                union[f] = a.copy()
    return union


def greyed(image, mask: Optional[np.ndarray]):
    """The image with the pixels of ``mask`` (any grid; resized nearest) painted flat grey."""
    from PIL import Image
    img = np.asarray(image.convert("RGB")).copy()
    if mask is not None and mask.any():
        m = mask
        if m.shape != img.shape[:2]:
            import cv2
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = GREY
    return Image.fromarray(img)


def run_round(session_dir, config: dict, k: int, *, log: Callable[[str], None] = print,
              progress: Optional[Callable[[float, str], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """Round ``k`` (>= 2) of the VLM: returns (and records in output/seg_rounds.json) the round,
    its ``new_prompts`` first."""
    from PIL import Image
    from semantic.client import get_semantic_client
    from semantic.parallel import (Job, concurrency_plan, cost_of, engine_kv_cache, plan_line,
                                   run_ordered, self_check_judges)
    from segmentation.autoprompt.autosegment import vlm_prompt_for
    from segmentation.autoprompt.consolidate_prompts import classify_phrases
    from segmentation.autoprompt.scene_understanding import same_name_key, understand_frame
    from segmentation.autoprompt.vlm_sampling import load_max_sam3_prompts
    from segmentation.object_captioner import load_object_captions

    t0 = time.time()
    session_dir = Path(session_dir)
    out = session_dir / "output"
    frames_dir = session_dir / "frames"

    def _p(pct: float, msg: str) -> None:
        log(f"{LOG_TAG} {msg}")
        if progress:
            progress(pct, msg)

    vlm = json.loads((out / "vlm_analysis.json").read_text())
    census = vlm.get("census") or {}
    seen, frames = set(), []
    for c in census.get("calls") or []:
        if c.get("tile") is None and c.get("file") and c["file"] not in seen:
            seen.add(c["file"])
            frames.append((int(c["frame"]), str(c["file"])))
    if not frames:
        raise RuntimeError("vlm_analysis.json lists no keyframe the first round looked at — the "
                           "round cannot show the VLM the same frames")
    existing = tried_prompts(out, k)
    scene_type = str((vlm.get("scene_understanding") or {}).get("scene_type") or "")
    base_prompt, _ = vlm_prompt_for(out)
    max_tokens = int(load_object_captions(config).understand_max_tokens)
    backend = (config.get("autoprompt") or {}).get("backend", "qwen_local")

    _p(2, f"round {k}: painting what {len(existing)} prompt(s) already segmented grey on "
          f"{len(frames)} keyframe(s)")
    masks = segmented_masks(out)

    client = get_semantic_client(backend=backend, consumer=f"phase1.understand.round{k}")
    kv = engine_kv_cache(client, log=lambda m: log(f"{LOG_TAG} {m}"))
    # one request's upper bound: round 1's measured prompt tokens (same images, same question)
    # + the bytes of the round's extra instruction (a byte-level BPE has no more tokens than
    # bytes) + the answer's bound
    measured = [int(c["prompt_tokens"]) for c in census.get("calls") or []
                if c.get("tile") is None and c.get("prompt_tokens")]
    foot = (max(measured) + len(_EXTRA.encode("utf-8")) + 1 + max_tokens) if measured else None
    conc = concurrency_plan(kv, [foot] if foot else [], n_requests=len(frames),
                            footprint_source="round 1's measured prompt tokens + the round's "
                                             "instruction bytes + max_tokens")
    counts: Counter = Counter()
    per_frame: List[dict] = []

    def _jobs():
        for f, fn in frames:
            img = greyed(Image.open(frames_dir / fn), masks.get(f))
            yield Job(fn=(lambda img=img, f=f: understand_frame(
                client, img, f, max_tokens=max_tokens, extra=_EXTRA, prompt=base_prompt)),
                cost=cost_of(conc, foot))

    def _collect(i: int, fu) -> None:
        # ';' separates SAM3 categories: a name carrying one is split into its parts, and a
        # name with no letter or digit is no name at all (it would reach SAM3 as nothing)
        objs = list(dict.fromkeys(
            q for o in (fu.objects if fu is not None else []) for q in
            (part.strip() for part in str(o).split(";")) if any(ch.isalnum() for ch in q)))
        per_frame.append({"frame": frames[i][0], "objects": objs})
        counts.update(objs)
        if (i + 1) % 25 == 0:
            _p(5 + 60 * (i + 1) / len(frames), f"round {k}: {i + 1}/{len(frames)} keyframes seen")

    stats = run_ordered(_jobs(), capacity=conc["capacity_blocks"], threads=conc["threads"],
                        on_result=_collect, cancelled=cancelled)
    conc["observed"] = stats
    log(f"{LOG_TAG} {plan_line(conc, stats)}")

    have = {same_name_key(p) for p in existing} | {same_name_key(p) for p in decided_names(out, k)}
    fresh: List[str] = []
    fkeys = set()
    single_witness: List[str] = []
    generic: List[str] = []
    for name, n in counts.most_common():
        key = same_name_key(name)
        if key in have or key in fkeys:
            continue
        if is_generic_name(name):
            # NAMES NO KIND OF THING (USER 2026-10-09): 'object', 'red object', 'item' — never a
            # SAM3 prompt, recorded as dropped
            generic.append(name)
            continue
        if n < 2:
            # TWO WITNESSES (pccr 2026-10-09): on images mostly painted grey the VLM fills one
            # answer with a list of things that are not there (round 3: sink, toaster, blender,
            # washing machine … all 19 from ONE keyframe); a name counts when two keyframes
            # name it — the rule the project already applies to single-witness points
            single_witness.append(name)
            continue
        fkeys.add(key)
        fresh.append(name)
    rec: dict = {"round": int(k), "keyframes": len(frames), "proposed": dict(counts),
                 "prompts_before": existing, "fresh": fresh,
                 "single_witness_dropped": single_witness, "generic_dropped": generic,
                 "concurrency": conc,
                 "per_frame": per_frame, "provenance": "vlm_proposed"}
    if single_witness:
        log(f"{LOG_TAG} round {k}: {len(single_witness)} name(s) named by ONE keyframe only — "
            f"not prompted (two witnesses): {single_witness}")
    if generic:
        log(f"{LOG_TAG} round {k}: {len(generic)} name(s) that name no kind of thing — not "
            f"prompted (generic): {generic}")
    if not fresh:
        rec.update(new_prompts=[], fates={}, seconds=round(time.time() - t0, 1))
        _save_round(out, rec)
        _p(100, f"round {k}: the VLM names nothing that is not already segmented — the rounds end")
        return rec

    _p(70, f"round {k}: {len(fresh)} name(s) not prompted yet — merging them with the "
           f"{len(existing)} prompt(s) so far")
    merge_client = get_semantic_client(backend=backend, consumer="phase1.merge_synonyms",
                                       no_read_timeout=True)
    merge_calls: list = []
    merge_exec: dict = {}
    # a generic prompt an earlier round let through ('object') is no whole to fold a real name
    # into: it stays tried (never prompted again) but is not offered to the merge
    synonyms = classify_phrases(merge_client, scene_type,
                                [p for p in existing if not is_generic_name(p)] + fresh,
                                log=lambda m: log(f"{LOG_TAG} {m}"), calls=merge_calls, kv=kv,
                                judges=self_check_judges(config), record=merge_exec)

    def _carrier(p: str) -> str:
        return synonyms.get(p, p)

    existing_set = set(existing)
    existing_carriers = {_carrier(p) for p in existing}
    fates: Dict[str, str] = {}
    new: List[str] = []
    for n in fresh:
        c = _carrier(n)
        if c in existing_set or c in existing_carriers:
            fates[n] = f"covered by '{c}' (already segmented)"
        elif c != n:
            fates[n] = f"merged into '{c}'"
        else:
            new.append(n)
            fates[n] = "new prompt"
    bound = int(load_max_sam3_prompts(config))
    room = max(bound - len(existing), 0)
    for n in new[room:]:
        fates[n] = "not prompted (autoprompt.max_sam3_prompts reached)"
    new = new[:room]
    rec.update(new_prompts=new, fates=fates, merge_calls=merge_calls,
               merge_concurrency=merge_exec.get("concurrency"),
               merge_self_check=merge_exec.get("self_check"),
               seconds=round(time.time() - t0, 1))
    _save_round(out, rec)
    _p(100, f"round {k}: {len(new)} new prompt(s) for SAM3: {new}"
            + ("" if new else " — the rounds end"))
    return rec
