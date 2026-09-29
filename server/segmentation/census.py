# STAC-Builder — the SEGMENTATION CENSUS: output/segmentation_census.json.
#
# USER 2026-09-29: *"la segmentación VLM no es lo suficientemente detallada …
# debe segmentar todo, absolutamente preciso y completo"*. On pccr desks, chairs,
# a folding table, a doorbell panel, conduits and ceiling pipes were missing or
# fragmentary, and nothing on disk said WHERE each was lost: never looked at by
# the VLM, never named, named and then folded into another phrase, or prompted
# and never born as a SAM3 masklet. This file answers that for every run.
#
# It is written by the SAM3 worker right after SAM3, on every run, from what the
# workers already wrote — no inference:
#   · vlm_analysis.json → ``census`` (session_builder): the keyframes and crops
#     the VLM was shown, every call (parsed or not), every phrase it proposed
#     and its ONE fate — prompt / merged into the same name / not prompted, each
#     non-prompt with its reason — plus the consolidation's grouping as evidence;
#   · segmentation.json + seg_masks.npz: per prompt the SAM3 masklets, their
#     keyframe spans and their VISITS (runs of keyframes; a gap of more than
#     ``segmentation.mask_filter.visit_gap_kf`` opens a new one — the user's one
#     definition of a visit, read from the same key the mask filter reads).
# Prompts with ZERO masklets are flagged, and so is every concept they carry —
# but only when SAM3 RAN them (``run_segmentation``'s per-prompt status): a
# prompt SAM3 never completed (skipped, failed, OOM twice, never reached) is
# listed apart, with its reason, because it says nothing about the thresholds.
# The accounting closes when every proposed phrase has exactly one fate and
# every fate that names a prompt names one SAM3 actually received.
#
# PROVENANCE: the phrases are vlm_proposed; spans, visits and counts are
# tool_measured over SAM3's masks. Nothing here decides anything.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

CENSUS_NAME = "segmentation_census.json"
CENSUS_VERSION = 1
FATES = ("prompt", "merged", "not_prompted", "unaccounted")
# the run_segmentation statuses of a prompt SAM3 did NOT run to the end (its
# zero is not a zero); "ran" is the one whose zero is real
_SAM3_NOT_COMPLETED = ("skipped", "failed", "not_reached")
_MASK_KEY = re.compile(r"^f(\d+)_o(\d+)$")


def concept_label(prompt: str) -> str:
    """The label SAM3's masklets of ``prompt`` are persisted under — the ONE
    place a rich phrase becomes an id-like label (``pipeline._save_masks``
    calls this), so the census attributes masklets by the same rule."""
    return re.sub(r"[^a-z0-9]+", "_", str(prompt).strip().lower()).strip("_")[:48] or "object"


def split_prompt(prompt: str) -> List[str]:
    """The concepts of a ';'-joined prompt, exactly as ``run_segmentation`` splits it."""
    return [c.strip() for c in str(prompt or "").split(";") if c.strip()]


def visit_gap_kf(config: dict) -> int:
    """``segmentation.mask_filter.visit_gap_kf`` — strict (a missing key fails
    naming it)."""
    try:
        v = config["segmentation"]["mask_filter"]["visit_gap_kf"]
    except (KeyError, TypeError):
        raise KeyError("config.yaml is missing 'segmentation.mask_filter.visit_gap_kf' — "
                       "the census splits masklets into visits by the same key the mask "
                       "filter reads") from None
    if isinstance(v, bool) or int(v) != v or int(v) < 1:
        raise ValueError(f"'segmentation.mask_filter.visit_gap_kf' must be an integer "
                         f">= 1, got {v!r}")
    return int(v)


def segmentation_keyframes(frames_dir) -> Optional[List[str]]:
    """The keyframe files in the order SAM3 numbered them — mirrors
    ``pipeline._prepare_valid_frames``: ``frames/selected_frames.json``, with the
    files missing on disk dropped BEFORE numbering. None without the file."""
    p = Path(frames_dir) / "selected_frames.json"
    if not p.is_file():
        return None
    try:
        files = json.loads(p.read_text()).get("selected_files") or []
    except Exception:  # noqa: BLE001 — an unreadable selection is no selection
        return None
    return [f for f in files if (Path(frames_dir) / f).exists()] or None


def visits_of(positions: Sequence[int], gap: int) -> List[List[int]]:
    """``[[first, last], …]`` runs of keyframe positions; a step larger than
    ``gap`` keyframes ends a visit (``mask_filter._visits``' rule)."""
    out: List[List[int]] = []
    for k in sorted(set(int(p) for p in positions)):
        if out and k - out[-1][1] <= gap:
            out[-1][1] = k
        else:
            out.append([k, k])
    return out


def _masklet_positions(output_dir: Path, seg_doc: dict,
                       keyframe_files: Optional[List[str]]) -> Tuple[Dict[int, List[int]], str]:
    """{oid: sorted keyframe positions with a NON-EMPTY mask} for the instances
    of ``segmentation.json``, and a note on the frame space the store is in."""
    from segmentation import mask_space
    masks_path = output_dir / str(seg_doc.get("mask_file") or "seg_masks.npz")
    if not masks_path.exists():
        return {}, f"{masks_path.name} not on disk"
    oids = {int(i["id"]) for i in seg_doc.get("instances") or [] if "id" in i}
    npz = np.load(masks_path)
    space = mask_space.declared_space(npz)
    to_pos = None
    note = space or "undeclared (read as keyframe_position, what run_segmentation writes)"
    if space == mask_space.SPACE_VIDEO:
        if not keyframe_files:
            return {}, "video_frame store and no frames/selected_frames.json to translate it"
        idx = {int(os.path.splitext(f)[0]): i for i, f in enumerate(keyframe_files)}
        to_pos = idx.get
    out: Dict[int, set] = {}
    for key in npz.files:
        m = _MASK_KEY.match(key)
        if not m:
            continue
        oid = int(m.group(2))
        if oid not in oids:
            continue
        pos = int(m.group(1)) if to_pos is None else to_pos(int(m.group(1)))
        if pos is None:
            continue
        a = npz[key]
        if a.size == 0 or not a.any():
            continue                      # a stored but EMPTY mask did not see the object
        out.setdefault(oid, set()).add(pos)
    return {k: sorted(v) for k, v in out.items()}, note


def build_census(output_dir, *, prompt: str, vlm_doc: Optional[dict], gap_kf: int,
                 frames_dir=None, seg_error: Optional[str] = None,
                 prompt_status: Optional[Dict[str, dict]] = None,
                 sam3_thresholds: Optional[dict] = None,
                 log: Callable[[str], None] = print) -> dict:
    """Build, write (atomically) and log ``output/segmentation_census.json``.

    ``prompt`` is what the SAM3 worker handed to ``run_segmentation`` — the
    authority on what SAM3 was asked; ``vlm_doc`` the vlm_analysis.json it read
    (None when there was none). ``seg_error`` is the segmentation's own error
    when it failed — the census is still written, every prompt then at zero.
    ``prompt_status`` is ``run_segmentation``'s per-prompt execution record
    (``ran`` / ``skipped`` / ``failed`` / ``not_reached`` + reason); without it
    every prompt's status is ``unknown``."""
    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir) if frames_dir else output_dir.parent / "frames"
    prompts = split_prompt(prompt)
    kf_files = segmentation_keyframes(frames_dir)
    rec = (vlm_doc or {}).get("census") if isinstance(vlm_doc, dict) else None

    # ── SAM3: masklets per prompt ────────────────────────────────────────
    seg_path = output_dir / "segmentation.json"
    seg_doc = json.loads(seg_path.read_text()) if (seg_path.exists() and not seg_error) else {}
    positions, space_note = (_masklet_positions(output_dir, seg_doc, kf_files)
                             if seg_doc else ({}, "no segmentation.json from this run"))
    by_label: Dict[str, List[dict]] = {}
    for inst in seg_doc.get("instances") or []:
        by_label.setdefault(str(inst.get("label")), []).append(inst)
    labels = {p: concept_label(p) for p in prompts}
    sharing: Dict[str, List[str]] = {}
    for p, lb in labels.items():
        sharing.setdefault(lb, []).append(p)

    def _video(pos: int) -> Optional[int]:
        if kf_files and 0 <= pos < len(kf_files):
            return int(os.path.splitext(kf_files[pos])[0])
        return None

    concepts = list((rec or {}).get("concepts") or [])
    carried: Dict[str, List[str]] = {}
    role_of: Dict[str, dict] = {}
    for c in concepts:
        if c.get("prompt") and c.get("fate") in ("prompt", "merged"):
            carried.setdefault(c["prompt"], []).append(c["concept"])
        if c.get("fate") == "prompt" and c.get("consolidation"):
            role_of[c["prompt"]] = c["consolidation"]

    prompt_rows = []
    for p in prompts:
        ms = []
        for inst in by_label.get(labels[p], []):
            ks = positions.get(int(inst["id"]), [])
            vis = visits_of(ks, gap_kf)
            ms.append({"oid": int(inst["id"]), "instance_id": inst.get("instance_id"),
                       "n_keyframes": len(ks),
                       "first_kf": ks[0] if ks else None, "last_kf": ks[-1] if ks else None,
                       "span_kf": (ks[-1] - ks[0] + 1) if ks else 0,
                       "first_frame": _video(ks[0]) if ks else None,
                       "last_frame": _video(ks[-1]) if ks else None,
                       "visits": vis, "n_visits": len(vis)})
        seen = sorted({k for m in ms for k in positions.get(m["oid"], [])})
        st = dict((prompt_status or {}).get(p) or {"status": "unknown",
                                                    "reason": "no SAM3 execution record"})
        row = {"prompt": p, "label": labels[p],
               "raw_concepts": carried.get(p, []),
               "sam3_status": st,
               "n_masklets": len(ms), "zero_masklets": not ms,
               "n_keyframes_seen": len(seen),
               "masklets": sorted(ms, key=lambda m: (m["first_kf"] is None, m["first_kf"] or 0,
                                                     m["oid"]))}
        if len(sharing[labels[p]]) > 1:
            row["label_shared_with"] = [q for q in sharing[labels[p]] if q != p]
        if p in role_of:
            row["consolidation"] = role_of[p]
        prompt_rows.append(row)
    # a ZERO is a zero only when SAM3 ran the prompt (or when there is no record
    # to say otherwise); a prompt SAM3 did not complete is not a threshold question
    not_completed = [{"prompt": r["prompt"], **r["sam3_status"]} for r in prompt_rows
                     if r["sam3_status"].get("status") in _SAM3_NOT_COMPLETED]
    nc_set = {e["prompt"] for e in not_completed}
    # (a prompt SAM3 ran WITH objects whose masklets this census cannot attribute —
    # a run that raised after saving them — is not a zero either)
    zero = [r["prompt"] for r in prompt_rows
            if r["zero_masklets"] and r["prompt"] not in nc_set
            and not int(r["sam3_status"].get("n_objects") or 0)]
    zero_set = set(zero)
    prompt_set = set(prompts)

    # ── accounting: every phrase exactly one fate, every named prompt real ──
    by_fate = {f: 0 for f in FATES}
    dangling = []
    for c in concepts:
        fate = c.get("fate") if c.get("fate") in FATES else "unaccounted"
        if fate in ("prompt", "merged") and c.get("prompt") not in prompt_set:
            # the VLM record names a prompt SAM3 never received (a config
            # prompt override, a hand edit): this concept did not reach SAM3
            dangling.append(c.get("concept"))
            c["fate"], fate = "unaccounted", "unaccounted"
            c["reason"] = (f"its prompt '{c.get('prompt')}' is not among the prompts "
                           f"SAM3 received")
        by_fate[fate] += 1
        if c.get("prompt") in zero_set:
            c["zero_masklets"] = True
        if c.get("prompt") in nc_set:
            c["sam3_not_completed"] = (prompt_status or {}).get(c["prompt"])
    names = [c.get("concept") for c in concepts]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    accounting = {
        "concept_record": ("present" if rec is not None else
                           "MISSING — vlm_analysis.json carries no census record (InternVL3 "
                           "fallback or an explicit config prompt): the raw proposals are unknown"),
        "n_concepts": len(concepts),
        "by_fate": by_fate,
        "concepts_not_reaching_sam3_prompt": dangling,
        "duplicated_concepts": duplicated,
        "n_prompts": len(prompts),
        "n_prompts_with_masklets": sum(1 for r in prompt_rows if not r["zero_masklets"]),
        "n_prompts_zero_masklets": len(zero),
        "n_prompts_not_completed_by_sam3": len(not_completed),
        "n_masklets": sum(r["n_masklets"] for r in prompt_rows),
        "prompts_without_proposal": [p for p in prompts if not carried.get(p)],
        # masklets in segmentation.json under a label no prompt of this run makes
        # (an older run's upsert left in the store): reported, never attributed
        "labels_without_prompt": sorted(set(by_label) - set(labels.values())),
        "closed": bool(rec is not None and by_fate["unaccounted"] == 0 and not duplicated),
    }

    sampling = (rec or {}).get("sampling")
    calls = list((rec or {}).get("calls") or [])
    looked = {}
    for c in calls:
        e = looked.setdefault(c.get("frame"), {"frame": c.get("frame"), "file": c.get("file"),
                                               "keyframe_index": c.get("keyframe_index"),
                                               "position": c.get("position"),
                                               "n_calls": 0, "n_parsed": 0})
        e["n_calls"] += 1
        e["n_parsed"] += int(bool(c.get("parsed")))
    doc = {
        "version": CENSUS_VERSION,
        "provenance": {"concepts": "vlm_proposed", "masklets": "tool_measured"},
        "vlm": {
            "source": (vlm_doc or {}).get("source") if isinstance(vlm_doc, dict) else None,
            "sampling": sampling,
            "looked_at": sorted(looked.values(), key=lambda e: (e["keyframe_index"] is None,
                                                               e["keyframe_index"] or 0)),
            "n_calls": len(calls),
            "n_parsed": sum(1 for c in calls if c.get("parsed")),
            "calls": calls,
            "vocabulary_reused": (rec or {}).get("vocabulary_reused"),
            "prompt_bound": (rec or {}).get("prompt_bound"),
        },
        "concepts": concepts,
        "prompts": prompt_rows,
        "zero_masklet_prompts": zero,
        "prompts_not_completed_by_sam3": not_completed,
        "sam3": {"error": seg_error, "frame_space": space_note, "visit_gap_kf": gap_kf,
                 "n_keyframes": len(kf_files) if kf_files else None,
                 "thresholds": sam3_thresholds},
        "accounting": accounting,
    }
    out = output_dir / CENSUS_NAME
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    os.replace(tmp, out)

    # ── the summary lines ────────────────────────────────────────────────
    if rec is None:
        log("[census] no VLM concept record in vlm_analysis.json — the raw proposals of "
            "this run are unknown (InternVL3 fallback or a config prompt)")
    elif sampling:
        log(f"[census] VLM looked at {len(looked)} keyframe(s) of {sampling.get('n_keyframes')} "
            f"(axis {sampling.get('axis')}), {len(calls)} call(s) incl. crops "
            f"(bound {sampling.get('max_calls')}"
            f"{', BOUND REACHED' if sampling.get('bound_reached') else ''}), "
            f"{doc['vlm']['n_parsed']} parsed")
    else:
        log("[census] the VLM record carries no sampling plan (scene understanding off)")
    bound = (rec or {}).get("prompt_bound") or {}
    if bound.get("bound_reached"):
        log(f"[census] SAM3 prompt BOUND REACHED (autoprompt.max_sam3_prompts = "
            f"{bound.get('max_sam3_prompts')}): {len(bound.get('not_prompted') or [])} "
            f"name(s) not prompted, the least proposed first: {bound.get('not_prompted')}")
    log(f"[census] {len(concepts)} concept(s) proposed → {by_fate['prompt']} prompt(s), "
        f"{by_fate['merged']} merged as the same name, {by_fate['not_prompted']} not "
        f"prompted, {by_fate['unaccounted']} UNACCOUNTED; accounting "
        f"{'closed' if accounting['closed'] else 'NOT closed'}")
    log(f"[census] SAM3: {accounting['n_prompts_with_masklets']}/{len(prompts)} prompt(s) "
        f"produced {accounting['n_masklets']} masklet(s)"
        + (f"; ZERO masklets (SAM3 ran them and confirmed nothing): {zero}" if zero else "")
        + (f"; segmentation error: {seg_error}" if seg_error else ""))
    if not_completed:
        log(f"[census] SAM3 did NOT complete {len(not_completed)} prompt(s) — not a threshold "
            f"question: " + "; ".join(f"'{e['prompt']}' {e.get('status')} "
                                      f"({e.get('reason')})" for e in not_completed))
    log(f"[census] → {out}")
    return doc
