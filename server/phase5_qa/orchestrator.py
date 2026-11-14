# STAC-Builder — Phase 5: spatial Q&A orchestrator.
#
# System prompt (scene inventory from the R.8 store + tool catalog + rules) +
# a bounded tool-calling loop over the shared semantic service. Every figure
# comes from a tool (the VLM never measures); answers cite the tool + arguments,
# declare uncertainty, and reply in the question's language. Multimodal: a
# keyframe of the referred object can be attached to disambiguate. Every session
# is persisted as a JSON log (question, tool trace, answer, latency).
#
# PROVENANCE: ours. Tool semantics ported from R3D (see tools.py); the loop uses
# our SemanticClient.run_tool_loop (Qwen3 native tool-calling), not R3D's text
# protocol.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase_r.instance_store import InstanceStore

from .tools import SpatialTools

_SYSTEM_TEMPLATE = """You are the spatial intelligence of STAC-Build: you supervise a 3D-reconstructed construction scene, you KNOW what session/scene you are looking at, and every figure you give is tool-measured.

SESSION:
{session}

SCENE INVENTORY (from the reconstruction; ids are stable):
{inventory}

OBJECT REFERENCES: the user names an object as label#id (sign#797, fire_extinguisher#425), #id
or "id N" — exactly as the Instances list of the viewer writes it. The number IS the object id:
use it directly in the tools. The inventory above names EVERY label of this scan. NEVER say an
id does not exist without a tool saying so (get_object(id)). When the question carries a
[REFERENCED OBJECTS] block, it was resolved by the tool layer on this scan: trust it.

HOW A TURN WORKS: you answer with TOOL CALLS only — every tool the question needs, all in this
one turn (several calls are fine: focus an object AND measure it). The tool layer runs them and
writes the answer to the user from their results; you do not write it.

NAMES: the inventory's names are the CURRENT ones. "name=given-by-user" marks a name the user
gave the object — always call it by that name. Notes from earlier conversations
(recall_notes) may use an object's older name: the id is what identifies it.

SPATIAL AWARENESS:
- Objects have PARTS. measure_between accepts feature1/feature2: top/upper,
  bottom/base/lower (real surface bands), highest/lowest (extreme points),
  centroid, closest — a curved ceiling has BOTH a lower and an upper part, a
  ladder has a base and a top. Use axis="vertical" for free heights
  (e.g. floor → ceiling upper part). get_extent tells where an object's lower
  and upper parts are.
- CLEARANCE / gap / separation ("holgura", "separación", "distancia libre") between TWO objects
  — ANY two, the floor and a wall included → get_clearance(id1, id2).
- FREE HEIGHT / headroom — ONLY when the question says height ("altura libre", "gálibo",
  "headroom") UNDER something overhead (a ceiling, a beam, a duct) → get_height_profile(
  id=<the upper object>, floor_id=<the floor>): the clearance at every station.
- PLUMB ("a plomo", "desplome", "vertical") → get_plumb(id). LEVEL ("a nivel", "desnivel",
  "horizontal") → get_level(id).
- "THE floor", "the ceiling", "the largest wall": pass the LABEL as the id ("floor", "wall") —
  the tool takes that label's largest object and tells you which one it took.
- "What session/scene is this?" → get_session_info. "What are we looking at?"
  → describe_scene (the VLM looks at the actual scan frames; interpretation,
  vlm_proposed). Store durable conclusions with remember_note; check
  recall_notes for context from earlier conversations.
- Evaluation volumes: define_volume RESTS the box ON the floor by default,
  centred on the floor (or anchor_id) — "a 2 m³ cube at the centre of the
  floor" is define_volume(name, volume_m3=2). Measure from a volume with
  measure_volume; list them with list_volumes. The user can then move/rotate/
  resize the box with the viewer gizmo.

YOU DRIVE THE VIEWER (USER 2026-10-09: the chat must be interactive with the cloud):
- Whenever your answer is about a specific object, call focus_object(id) so the user SEES it
  (OBB + dimensions + the camera travels there). Two objects: focus the first, measure the pair.
- "show / hide / only / all" → show_objects, show_only, show_all_objects. "box / OBB" → set_obb.
- "from above / plan / front / side / iso" → set_view. "rotate / turn around" → rotate_view.
- "take me from X to Y / go to" → travel / focus_object. "confidence / colour" → color_cloud.
- Questions about HOW the scene was reconstructed, the method, the segmentation or how reliable
  the points are → explain_reconstruction, then explain in plain words.

RULES:
- You NEVER estimate, compute or invent numbers. EVERY figure must come from a tool call made
  in THIS turn — not from memory, not from an earlier answer, not from arithmetic on positions.
  A distance between two objects is get_distance / get_clearance / measure_between, never a
  subtraction of their positions. The tool layer checks every figure of your answer against
  the tool results and sends the answer back when one is not there.
- Cite only tools you actually called in this turn. A count read from the inventory is cited
  as "inventory".
- When the question names a KIND of object ("how many signs", "is there a door"), pass that
  kind as the label of count_objects / list_objects, in the inventory's English label.
- Call list_objects(label) first if you are unsure which id an object refers to.
- Earlier turns of the conversation may precede the question: "it", "that one", "and the
  other" refer to the objects named there.
- Units are SI (metres, m^2, m^3, degrees). State the unit.
- Cite the tool and its arguments for every figure you report.
- If a tool returns {{"insufficient_data": true}}, say the data is insufficient —
  do NOT extrapolate or guess.
- Declare uncertainty when a tool reports low confidence.
- Descriptions from describe_scene / notes are vlm_proposed context, never
  measurements.
- Answer in the SAME LANGUAGE as the question.
- BE BRIEF (USER 2026-11-13: the chat "tarda muchisimo en contestar"; every word is generated at the engine's
  pace): one or two sentences — what was asked, with its figure and unit and the objects as label#id. No
  introduction, no explanation of what a tool does, no closing question, no offer of further help.
"""


def _inventory(store: InstanceStore, tools: SpatialTools | None = None) -> str:
    """The WHOLE scene for the system prompt. Up to ``semantic.chat.list_limit`` objects: one
    line each, with class, vertical extent and the name's origin. Beyond it: every LABEL with
    its count and its ids as ranges — the model knows every id that exists (the first 60 of
    pccr 2408's 8,777 were listed before, and it answered that sign#797 was outside "ids 1 to
    63", USER 2026-11-13)."""
    from .tools import chat_config
    tools = tools or SpatialTools(store)
    insts = store.list_instances()
    lines = []
    scene_type = store.get_meta("scene_type")
    if scene_type:
        lines.append(f"scene: {scene_type}")
    if not insts:
        return "\n".join(lines + ["  (no objects in store)"])
    if len(insts) > chat_config()["list_limit"]:
        # every LABEL with its count and its largest object. The ids are not listed: an id or a
        # label written in the question is resolved by the tool layer (references, resolve_id),
        # and every token of the prompt is paid at each call of the engine
        lines.append(f"{len(insts)} objects, by label (count, the largest object's id):")
        for r in tools.labels_summary(insts):
            big = r.get("largest_id", r["ids"].split(",")[0].split("-")[0])
            lines.append(f"  {r['label']} ({r['count']}, #{big})")
        named = [i for i in insts if i.get("label_origin") == "human_validated"]
        for i in named:
            lines.append(f"  id={i['instance_id']} {i['label']} name=given-by-user")
        return "\n".join(lines)
    cls = store.list_classifications()
    obbs = store.list_obbs()
    for i in insts:
        iid = i["instance_id"]
        extra = ""
        c = cls.get(int(iid))
        if c and c.get("class_final"):
            extra = f" class={c['class_final']}"
            if c.get("material"):
                extra += f" material={c['material']}"
        m = store.get_metrics(iid).get("onion")
        if m and m.get("bimodal"):
            extra += " ⚠onion"
        # vertical extent from the OBB corners — the model should KNOW each
        # object's y-range up front (curved ceilings span a lower→upper band)
        try:
            obb = obbs.get(int(iid))
            if obb is not None:
                T, aabb, _pos = obb
                T = np.asarray(T, float)
                lo = np.array([aabb[0], aabb[2], aabb[4]])
                hi = np.array([aabb[1], aabb[3], aabb[5]])
                corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                    for y in (lo[1], hi[1])
                                    for z in (lo[2], hi[2])])
                w = (T[:3, :3] @ corners.T).T + T[:3, 3]
                extra += f" y[{w[:, 1].min():.2f}..{w[:, 1].max():.2f}]m"
        except Exception:  # noqa: BLE001
            pass
        if i.get("label_origin") == "human_validated":
            extra += " name=given-by-user"
        lines.append(f"  id={iid} {i['label']}{extra} "
                     f"(status={i['status']}, views={i['n_views']})")
    return "\n".join(lines)


def _session_block(store: InstanceStore) -> str:
    """Session header for the system prompt — the assistant must know WHAT it
    is looking at (user 2026-08-29)."""
    from pathlib import Path as _P
    out_dir = _P(store.path).parent
    session_dir = out_dir.parent
    parts = session_dir.resolve().parts
    project = parts[parts.index("projects") + 1] \
        if "projects" in parts and parts.index("projects") + 1 < len(parts) else "?"
    insts = store.list_instances()
    # project / scan date / source: the store lives in <project>/scans/<date>/src_<source>/output
    scan = "/".join(p for p in parts[parts.index("projects") + 2:] if p != "scans") \
        if "projects" in parts else session_dir.name
    lines = [f"project: {project}   scan: {scan}   objects: {len(insts)}"]
    desc = store.get_meta("scene_description")
    if desc:
        lines.append(f"scene description (vlm_proposed): {desc}")
    else:
        lines.append("scene description: not generated yet — call describe_scene "
                     "when asked what the scene is")
    # per-object deep analyses (shape proposer's dossier): one line each so
    # the assistant KNOWS what every element is; full text via describe_object
    import json as _json
    # ONE query for every dossier (a get_meta per instance was 9.9 s of every question on
    # pccr 2408's 8,777 objects, measured 2026-11-13)
    analyses = store.list_meta("object_analysis_")
    for i in insts:
        iid = i["instance_id"]
        raw = analyses.get(f"object_analysis_{iid}")
        if not raw:
            continue
        try:
            a = _json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        mats = ", ".join((a.get("materiales") or [])[:3])
        lines.append(f"object {iid} ({i['label']}) — {a.get('que_es', '?')}"
                     + (f" · materiales: {mats}" if mats else "")
                     + " (dossier completo: describe_object)")
    return "\n".join(lines)


class SpatialQA:
    def __init__(self, store_path: str, backend: str = "qwen_local",
                 session_log_dir: str | Path | None = None):
        self.store_path = str(store_path)
        self.store = InstanceStore(store_path)
        # WHERE THE MEASUREMENTS RUN (semantic.chat.tools_device): with cuda the tools that read
        # points run in the chat's own tools process on the card (tools_worker) and this
        # process — the backend, which never touches the GPU — keeps a tool set that only reads
        # the store; with numpy / cpu everything runs here
        from .tools import effective_device
        self.tools_device = effective_device()
        self.tools = SpatialTools(                   # display frame (viewer-consistent)
            self.store, device=("numpy" if self.tools_device == "cuda" else self.tools_device))
        self.backend = backend
        # session logs default next to the scene store: output/spatial_qa_logs/
        self.session_log_dir = Path(session_log_dir) if session_log_dir \
            else Path(store_path).parent / "spatial_qa_logs"

    def system_prompt(self) -> str:
        return _SYSTEM_TEMPLATE.format(session=_session_block(self.store),
                                       inventory=_inventory(self.store, self.tools))

    def question_with_references(self, question: str) -> tuple[str, list[dict]]:
        """The question followed by the objects it names by id, resolved on this scan by the
        tool layer (never by the model): what exists, its label, where it is."""
        refs = self.tools.resolve_references(question)
        if not refs:
            return question, refs
        lines = ["", "[REFERENCED OBJECTS — resolved on this scan by the tool layer]"]
        for r in refs:
            if not r["exists"]:
                ln = f"- id {r['id']}: NO object with this id in this scan"
                if r.get("ids_with_that_label"):
                    ln += f" (the ids labelled '{r['label_asked']}' are: {r['ids_with_that_label']})"
            else:
                ln = f"- id {r['id']}: {r['label']}"
                if r.get("label_asked"):
                    ln += f" (the user wrote '{r['label_asked']}' — the id's own label is '{r['label']}')"
                # its position and size are NOT written here: read in the prompt they were
                # subtracted by the model into "distances" no tool had measured (2026-11-13) —
                # get_object(id) returns them, measured
            lines.append(ln)
        return question + "\n".join(lines), refs

    def ask(self, question: str, images: list | None = None,
            max_iterations: int = 8, history: list | None = None,
            lang: str | None = None) -> dict[str, Any]:
        """``history``: the conversation's earlier turns, oldest first, as
        [{"role": "user" | "assistant", "text": str}] — the last
        ``semantic.chat.history_turns`` of them go with the question. ``lang``: the language
        of the answer ('es' | 'en'); None = the question's own, by its function words.

        ONE CALL OF THE MODEL (USER 2026-11-13: "Determinista pero en 1 o 2 segundos ninguna es
        negociable"): the model chooses the tool(s) and their arguments, the tool layer runs
        them and WRITES the answer from their results (answers.py) — the same sentence for the
        same result, the tool's own figures, no second call of the engine. Only a tool with no
        wording there sends the turn back to the model, with the provenance check."""
        from semantic.client import get_semantic_client
        from semantic.types import assistant, system, user
        from .tools import chat_config

        t0 = time.time()
        client = get_semantic_client(backend=self.backend, consumer="phase5.qa")
        messages = [system(self.system_prompt())]
        keep = chat_config()["history_turns"]
        turns = [h for h in (history or []) if isinstance(h, dict)
                 and h.get("role") in ("user", "assistant") and str(h.get("text") or "").strip()]
        for h in (turns[-keep:] if keep else []):
            messages.append(user(str(h["text"])) if h["role"] == "user"
                            else assistant(text=str(h["text"])))
        asked, refs = self.question_with_references(question)
        messages.append(user(asked, images=images))
        t_prompt = time.time() - t0
        tools = self.tools.openai_schemas()
        impls = self.tools.impls(remote=self._remote if self.tools_device == "cuda" else None)
        # A TOOL CALL COMES FIRST, always (2026-11-13): left to choose, the model answered "only the
        # extinguishers are shown" having shown nothing, and "1.14 m" having measured nothing — an
        # answer is preceded by at least one call of the tool layer, then checked
        cap = chat_config()["max_tokens"]         # a call or a brief answer never needs more
        from . import answers
        lang = lang if lang in answers.LANGS else answers.language_of(question)
        t_llm = time.time()
        first = client.chat(messages, tools=tools, tool_choice="required", max_tokens=cap,
                            consumer="phase5.qa#call")
        llm_s = time.time() - t_llm
        if first.has_tool_calls:
            trace = []
            for tc in first.tool_calls:
                impl = impls.get(tc.name)
                try:
                    res = impl(**tc.arguments) if impl else {"error": f"unknown tool '{tc.name}'"}
                except Exception as e:  # noqa: BLE001 — told as the tool's result
                    res = {"error": f"{type(e).__name__}: {e}"}
                trace.append({"iteration": 0, "tool": tc.name, "arguments": tc.arguments, "result": res})
            written = answers.compose_turn(trace, lang)
            if written is not None:
                result = {
                    "question": question, "answer": written, "tool_trace": trace, "iterations": 1,
                    "stopped": "composed_by_tools", "references": refs,
                    # every figure IS a tool's: the sentence was written from the results
                    "provenance": {"ok": True, "figures": [], "tools": [], "retried": False,
                                   "written_by": "tool_layer"},
                    "lang": lang, "prompt_s": round(t_prompt, 2), "model_s": round(llm_s, 2),
                    "latency_s": round(time.time() - t0, 2),
                }
                self._log_session(result, n_images=len(images or []))
                return result
            # a tool the tool layer has no wording for: the model words the turn (below), its
            # calls already made — they are handed to it as the loop's first step
            convo = list(messages) + [assistant(text=first.content, tool_calls=first.tool_calls)]
            from semantic.types import tool_result as _tool_result
            from semantic.client import _to_text
            for tc, e in zip(first.tool_calls, trace):
                convo.append(_tool_result(tc.id, _to_text(e["result"]), name=tc.name))
            rest = client.run_tool_loop(convo, tools=tools, tool_impls=impls,
                                        max_iterations=max_iterations, max_tokens=cap,
                                        consumer="phase5.qa")
            for e in rest["tool_trace"]:
                e["iteration"] = int(e.get("iteration", 0)) + 1
            out = {"answer": rest["answer"], "stopped": rest["stopped"],
                   "tool_trace": trace + list(rest["tool_trace"]),
                   "iterations": 1 + int(rest["iterations"]), "messages": rest["messages"]}
        else:
            out = {"answer": first.content or "", "stopped": "final_answer", "tool_trace": [],
                   "iterations": 1, "messages": list(messages)}
        if not (out["answer"] or "").strip() and not out["tool_trace"]:
            # the required call ran to the token bound without closing (seen 2026-11-13: a list of
            # 56 ids written under the forced format) — asked again, free to choose
            print("[SpatialQA] the required tool call did not close within the bound — asked again")
            out = client.run_tool_loop(
                messages, tools=tools, tool_impls=impls, max_iterations=max_iterations,
                max_tokens=cap, consumer="phase5.qa.retry")
        # PROVENANCE CHECK (phase5_qa/provenance.py): a figure no tool returned in this turn,
        # or a tool cited and never called, sends the answer back ONCE with a tool call
        # required; what still fails is declared with the answer, never hidden
        from . import provenance
        context = (provenance.decimals_in(messages[0].text or "")       # the inventory
                   + provenance.numbers_of(refs)                        # the referenced objects
                   + provenance.numbers_in(question))                   # the user's own numbers
        verdict = provenance.check(out["answer"], out["tool_trace"], tool_names=impls,
                                   context_numbers=context)
        retried = False
        if not verdict["ok"]:
            retried = True
            convo = list(out["messages"]) + [assistant(text=out["answer"]),
                                             user(provenance.retry_message(verdict))]
            try:
                again = client.run_tool_loop(
                    convo, tools=tools, tool_impls=impls, max_iterations=max_iterations,
                    max_tokens=cap, consumer="phase5.qa.provenance", first_tool_choice="required")
            except Exception as e:  # noqa: BLE001 — an engine without tool_choice=required
                print(f"[SpatialQA] provenance retry with a required tool call failed ({e}); "
                      f"asked again with tool_choice auto")
                again = client.run_tool_loop(
                    convo, tools=tools, tool_impls=impls, max_iterations=max_iterations,
                    max_tokens=cap, consumer="phase5.qa.provenance")
            for e in again["tool_trace"]:
                e["iteration"] = int(e.get("iteration", 0)) + int(out["iterations"])
            out = {"answer": again["answer"], "stopped": again["stopped"],
                   "tool_trace": list(out["tool_trace"]) + list(again["tool_trace"]),
                   "iterations": int(out["iterations"]) + int(again["iterations"]),
                   "messages": again["messages"]}
            verdict = provenance.check(out["answer"], out["tool_trace"], tool_names=impls,
                                       context_numbers=context)
        result = {
            "question": question,
            "answer": out["answer"],
            "tool_trace": out["tool_trace"],     # traceability: tool + args + result
            "iterations": out["iterations"],
            "stopped": out["stopped"],
            "provenance": {**verdict, "retried": retried},
            "references": refs,
            "prompt_s": round(t_prompt, 2),          # building the prompt (store reads)
            "latency_s": round(time.time() - t0, 2),
        }
        self._log_session(result, n_images=len(images or []))
        return result

    def _log_session(self, result: dict, n_images: int = 0) -> None:
        """Persist the full JSON session log (spec: log JSON completo por sesión) — in its own
        thread: the write goes to the network volume and the answer does not wait for it."""
        import threading
        doc = dict(result)
        threading.Thread(target=self._write_log, args=(doc, n_images), daemon=True,
                         name="spatial-qa-log").start()

    def _write_log(self, result: dict, n_images: int = 0) -> None:
        """Best-effort — logging must never break an answer."""
        try:
            self.session_log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.session_log_dir / f"qa_{ts}_{int((time.time() % 1) * 1000):03d}.json"
            path.write_text(json.dumps({
                "ts": time.time(), "backend": self.backend,
                "store_path": self.store_path, "n_images": n_images, **result,
            }, indent=2, default=str))
        except Exception:
            pass

    def _remote(self, tool: str, kwargs: dict):
        from . import tools_worker
        return tools_worker.call(self.store_path, tool, kwargs, device="cuda")

    def close(self):
        self.tools.release()
        self.store.close()
