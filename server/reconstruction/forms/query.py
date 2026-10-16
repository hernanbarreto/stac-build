"""The form query: one structured VLM answer per object (``vlm_proposed``).

The object is shown ISOLATED in its best SAM3-mask views (the captioner's rule: the largest
masks, ties by the measured area dispersion) and Qwen3-VL answers the closed JSON of
``catalog.answer_schema`` — what the object IS and the form it SHOULD have, never a number in
metres. The answer, its raw text, the views and a STAMP (the exact crop bytes, the prompt, the
engine) are recorded in ``output/forms/<safe>.form.json``; a record whose stamp is the one
computed now is kept (no second call), ``refresh`` asks again. An object without a usable mask
view, or an answer that does not parse, is DECLARED in its record (``error``) — it is meshed as
measured downstream, never silently as some form.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .catalog import (FORMS, AnswerError, FormAnswer, FormsConfig, answer_from_dict,
                      answer_schema, parse_answer)

logger = logging.getLogger("Forms")
DIR = "forms"
VERSION = 1
CONSUMER = "forms.query"

_PROMPT = """You see ONE isolated 3D object (everything else is darkened) in {n} view(s) of the same object from different viewpoints, scanned inside a building or an industrial site.
Decide its IDEAL GEOMETRIC FORM for a CAD/BIM model: what the object IS and the shape it SHOULD have — the designed, regular shape, regardless of scan noise, holes or deformation.

Answer ONLY a JSON object with these fields:
- form: one of {forms}.
{catalog}
- orientation: "vertical" | "horizontal" | "inclined" | "unknown" — of the object's main axis (for a panel: "vertical" = a wall-like plate, "horizontal" = a floor-like or top-like plate; for a cylinder/profile: the direction of its length).
- axis_aligned: true when the object's edges follow the room's walls and floor (a Manhattan object), false when it is rotated or oblique.
- hollow: true for a tube/pipe/shell, false for a solid, null when it cannot be told.
- outline: for a panel, its outline — "rectangle" | "circle" | "arch" | "polygon" | "irregular" | "unknown"; "unknown" for other forms.
- profile: for a profile, its section — "I" | "L" | "C" | "T" | "rect" | "rect_hollow" | "round_tube" | "rail" | "unknown"; "unknown" for other forms.
- parts: for a composite, the list of its parts, each {{"name", "form", "count", "orientation", "relation"}} — "relation" in words ("four equal legs under the top", "two parallel rails joined by equally spaced rungs", "symmetric"); an empty list for other forms.
- proportions: ONE short sentence in words, never numbers in metres ("tall and thin, the height about ten times the diameter").
- alternatives: other plausible forms of the same list, the most plausible first (empty when sure).
- confidence: "high" | "medium" | "low".
- what_would_help: what view or detail would settle a doubt, in a few words, or "".
- notes: anything a modeller must know (an opening, a cap, a flange, a bend), or "".

Rules: decide the form the object SHOULD have; the model will be measured on the scan afterwards, so never give a dimension; JSON only, no prose.
The element is tagged as: {label}.{caption}"""


def _catalog_lines(catalog: List[str]) -> str:
    return "\n".join(f'  "{f}" = {FORMS[f]["what"]}' for f in catalog if f in FORMS)


def build_prompt(catalog: List[str], label: str, n_views: int, caption: Optional[str]) -> str:
    cap = f"\nAn earlier description of it: {caption.strip()}" if caption else ""
    return _PROMPT.format(n=int(n_views), forms=", ".join(f'"{f}"' for f in catalog),
                          catalog=_catalog_lines(catalog), label=label, caption=cap)


# ── records ────────────────────────────────────────────────────────────────────

def record_path(output_dir: Path, safe: str) -> Path:
    return Path(output_dir) / DIR / f"{safe}.form.json"


def load_record(output_dir: Path, safe: str) -> Optional[dict]:
    p = record_path(output_dir, safe)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_record(output_dir: Path, safe: str, doc: dict) -> Path:
    from atomic_io import atomic_write_json
    p = record_path(output_dir, safe)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, doc, indent=1, sort_keys=True)
    return p


def form_of_record(rec: Optional[dict]) -> Optional[FormAnswer]:
    """The form a record carries (the LAST round's), or None (no record / declared error)."""
    if not rec or not isinstance(rec.get("form"), dict):
        return None
    try:
        return answer_from_dict(rec["form"])
    except AnswerError:
        return None


def _stamp(refs: list, prompt: str, client) -> dict:
    imgs = [str(getattr(r, "sha1", None) or getattr(r, "short_sha1", "")) for r in refs]
    backend = getattr(client, "backend", None)
    return {"images_sha1": imgs, "n_views": len(refs),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model": str(getattr(backend, "model", "") or ""),
            "backend": str(getattr(backend, "name", "") or "")}


def stamp_key(stamp: dict) -> str:
    return hashlib.sha256(json.dumps(stamp, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ── one call ───────────────────────────────────────────────────────────────────

def ask_form(client, prompt: str, refs: list, catalog: List[str], max_tokens: int,
             log: Callable[[str], None] = logger.info) -> Dict[str, Any]:
    """One structured call. Returns ``{answer: FormAnswer|None, raw, error, finish_reason}``."""
    from segmentation.shape_proposer import _extract_json
    from semantic.types import user as user_msg
    schema = answer_schema(catalog)
    msgs = [user_msg(prompt, images=refs)]
    resp = None
    try:
        resp = client.chat(msgs, temperature=0.1, max_tokens=int(max_tokens), consumer=CONSUMER,
                           extra_body={"response_format": {"type": "json_schema",
                                                           "json_schema": {"name": "form",
                                                                           "schema": schema}}})
    except Exception as e:  # noqa: BLE001 — the engine without structured output: plain retry
        log(f"[forms] structured output rejected ({type(e).__name__}: {e}) — plain retry")
        resp = None
    if resp is None:
        resp = client.chat(msgs, temperature=0.1, max_tokens=int(max_tokens), consumer=CONSUMER)
    raw = resp.content or ""
    out: Dict[str, Any] = {"answer": None, "raw": raw, "error": None,
                           "finish_reason": getattr(resp, "finish_reason", None)}
    doc = _extract_json(raw)
    if doc is None:
        out["error"] = "the answer is not JSON"
        return out
    try:
        out["answer"] = parse_answer(doc, catalog)
    except AnswerError as e:
        out["error"] = f"AnswerError: {e}"
    return out


# ── the session ────────────────────────────────────────────────────────────────

def _instances(output_dir: Path) -> List[dict]:
    p = Path(output_dir) / "segmentation_result.json"
    if not p.exists():
        return []
    return list(json.loads(p.read_text()).get("instances") or [])


def _views_of(z, space, frames_dir: Path, oids: List[int], n_views: int):
    from segmentation.object_captioner import _instance_views_record
    return _instance_views_record(z, space, frames_dir, oids, n_views)


def query_session_forms(output_dir, session_dir, fcfg: FormsConfig, *,
                        instance_ids: Optional[List[int]] = None, refresh: bool = False,
                        client=None, log: Callable[[str], None] = print,
                        cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """One form record per instance (the requested ones, else every instance of
    ``segmentation_result.json``): kept when its stamp is the one computed now, asked
    otherwise — ONE call per object, the crops encoded once (the stamp's input).
    Returns ``{asked, kept, declared, n_instances, records: {iid: path}}``."""
    import numpy as np
    from segmentation import mask_space
    from segmentation.object_captioner import (CAPTION_KEY, _crops_for, _oids_by_instance)
    from segmentation.tsdf_export import _safe_label
    from semantic.types import _encode_image

    out = Path(output_dir)
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    t0 = time.time()
    instances = _instances(out)
    wanted = None if instance_ids is None else {int(i) for i in instance_ids}
    stats = {"asked": 0, "kept": 0, "declared": 0, "n_instances": 0, "records": {}}
    todo = [i for i in instances
            if wanted is None or int(i.get("instance_id", i.get("id"))) in wanted]
    stats["n_instances"] = len(todo)
    if not todo:
        return stats
    parent_p, masks_p = out / "segmentation.json", out / "seg_masks.npz"
    if not parent_p.exists() or not masks_p.exists():
        raise RuntimeError("no segmentation.json / seg_masks.npz — the objects have no mask "
                           "views to be asked about")
    parent = json.loads(parent_p.read_text())
    result = json.loads((out / "segmentation_result.json").read_text())
    oids_of = _oids_by_instance(parent, result.get("absorbed") or {})
    z = np.load(masks_p, allow_pickle=True)
    space = mask_space.resolve(out, masks=z)
    catalog = [f for f in fcfg.catalog if f in FORMS]
    if client is None:
        from segmentation.object_captioner import _service_client
        client = _service_client(None, log=log, cancelled=cancelled)
    for n, inst in enumerate(todo, 1):
        if cancelled and cancelled():
            raise RuntimeError("cancelled")
        iid = int(inst.get("instance_id", inst.get("id")))
        label = str(inst.get("label") or "segment")
        safe = _safe_label(label, iid)
        caption = ((inst.get(CAPTION_KEY) or {}).get("caption")
                   if isinstance(inst.get(CAPTION_KEY), dict) else None)
        prev = load_record(out, safe)
        try:
            frames, masks, vrec = _views_of(z, space, frames_dir, oids_of.get(iid, []), fcfg.views)
            crops = _crops_for(frames, masks, fcfg.views) if frames else []
        except Exception as e:  # noqa: BLE001 — declared below
            frames, crops, vrec = [], [], {"views": [], "view_selection": {"error": str(e)}}
        if not crops:
            doc = {"version": VERSION, "instance_id": iid, "label": label, "provenance": "vlm_proposed",
                   "form": None, "rounds": [], "views": vrec.get("views", []),
                   "view_selection": vrec.get("view_selection"),
                   "error": "no usable mask view — the object cannot be shown to the VLM"}
            if not (prev and prev.get("error") == doc["error"] and prev.get("form") is None):
                write_record(out, safe, doc)
            stats["declared"] += 1
            stats["records"][iid] = str(record_path(out, safe))
            log(f"[forms] {safe}: no usable mask view — declared ({n}/{len(todo)})")
            continue
        refs = [_encode_image(c) for c in crops]
        prompt = build_prompt(catalog, label, len(refs), caption)
        stamp = _stamp(refs, prompt, client)
        key = stamp_key(stamp)
        if (prev and not refresh and prev.get("stamp_key") == key
                and (prev.get("form") is not None or prev.get("error"))):
            stats["kept"] += 1
            stats["records"][iid] = str(record_path(out, safe))
            continue
        t1 = time.time()
        got = ask_form(client, prompt, refs, catalog, fcfg.max_tokens, log=log)
        ans: Optional[FormAnswer] = got["answer"]
        rnd = {"round": 1, "kind": "query", "answer": (ans.to_dict() if ans else None),
               "raw": got["raw"][:4000], "error": got["error"],
               "finish_reason": got["finish_reason"], "seconds": round(time.time() - t1, 1)}
        doc = {"version": VERSION, "instance_id": iid, "label": label, "provenance": "vlm_proposed",
               "source": "object", "caption_seen": caption, "catalog": catalog,
               "form": (ans.to_dict() if ans else None), "rounds": [rnd],
               "views": vrec.get("views", []), "view_selection": vrec.get("view_selection"),
               "stamp": stamp, "stamp_key": key, "error": got["error"]}
        write_record(out, safe, doc)
        stats["records"][iid] = str(record_path(out, safe))
        if ans is None:
            stats["declared"] += 1
            log(f"[forms] {safe}: answer not usable ({got['error']}) — declared ({n}/{len(todo)})")
        else:
            stats["asked"] += 1
            log(f"[forms] {safe}: {ans.form} ({ans.orientation}"
                f"{', axis-aligned' if ans.axis_aligned else ''}"
                f"{', hollow' if ans.hollow else ''}"
                f"{', ' + ans.outline if ans.form == 'panel' else ''}"
                f"{', ' + ans.profile if ans.form == 'profile' else ''}"
                f"{', parts ' + '+'.join(f'{p.count}x{p.form}' for p in ans.parts) if ans.parts else ''}"
                f"; {ans.confidence}) in {rnd['seconds']} s ({n}/{len(todo)})")
    stats["seconds"] = round(time.time() - t0, 1)
    log(f"[forms] {stats['asked']} asked, {stats['kept']} kept, {stats['declared']} declared "
        f"of {stats['n_instances']} in {stats['seconds']} s")
    return stats


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="the form query per object (Qwen3-VL, structured)")
    ap.add_argument("--session", required=True, help="the session dir (…/src_default)")
    ap.add_argument("--instance-id", type=int, action="append", default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    from config import cfg
    fcfg = load_forms_config(cfg)
    session_dir = Path(args.session)
    st = query_session_forms(session_dir / "output", session_dir, fcfg,
                             instance_ids=args.instance_id, refresh=bool(args.refresh))
    print(json.dumps({k: v for k, v in st.items() if k != "records"}))
    return 0


from .catalog import load_forms_config  # noqa: E402  (the CLI's loader)

if __name__ == "__main__":
    raise SystemExit(_main())
