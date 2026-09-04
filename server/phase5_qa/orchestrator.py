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

SCENE INVENTORY (from the reconstruction; ids are stable; y[lo..hi] is each object's vertical extent in metres):
{inventory}

SPATIAL AWARENESS:
- Objects have PARTS. measure_between accepts feature1/feature2: top/upper,
  bottom/base/lower (real surface bands), highest/lowest (extreme points),
  centroid, closest — a curved ceiling has BOTH a lower and an upper part, a
  ladder has a base and a top. Use axis="vertical" for free heights
  (e.g. floor → ceiling upper part). get_extent tells where an object's lower
  and upper parts are.
- "What session/scene is this?" → get_session_info. "What are we looking at?"
  → describe_scene (the VLM looks at the actual scan frames; interpretation,
  vlm_proposed). Store durable conclusions with remember_note; check
  recall_notes for context from earlier conversations.
- Evaluation volumes: define_volume RESTS the box ON the floor by default,
  centred on the floor (or anchor_id) — "a 2 m³ cube at the centre of the
  floor" is define_volume(name, volume_m3=2). Measure from a volume with
  measure_volume; list them with list_volumes. The user can then move/rotate/
  resize the box with the viewer gizmo.

RULES:
- You NEVER estimate or invent numbers. EVERY figure must come from a tool call.
- Call list_objects first if you are unsure which id an object refers to.
- Units are SI (metres, m^2, m^3, degrees). State the unit.
- Cite the tool and its arguments for every figure you report.
- If a tool returns {{"insufficient_data": true}}, say the data is insufficient —
  do NOT extrapolate or guess.
- Declare uncertainty when a tool reports low confidence.
- Descriptions from describe_scene / notes are vlm_proposed context, never
  measurements.
- Answer in the SAME LANGUAGE as the question.
"""


def _inventory(store: InstanceStore, limit: int = 60) -> str:
    lines = []
    scene_type = store.get_meta("scene_type")
    if scene_type:
        lines.append(f"scene: {scene_type}")
    n = len(store.list_instances())
    if n > limit:
        lines.append(f"({n} objects total; first {limit} listed — filter with list_objects)")
    for i in store.list_instances()[:limit]:
        iid = i["instance_id"]
        extra = ""
        c = store.get_classification(iid)
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
            obb = store.get_obb(iid)
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
        lines.append(f"  id={iid} {i['label']}{extra} "
                     f"(status={i['status']}, views={i['n_views']})")
    return "\n".join(lines) if lines else "  (no objects in store)"


def _session_block(store: InstanceStore) -> str:
    """Session header for the system prompt — the assistant must know WHAT it
    is looking at (user 2026-08-29)."""
    from pathlib import Path as _P
    out_dir = _P(store.path).parent
    session_dir = out_dir.parent
    parts = session_dir.resolve().parts
    project = parts[parts.index("projects") + 1] \
        if "projects" in parts and parts.index("projects") + 1 < len(parts) else "?"
    n_frames = len(list((session_dir / "frames").glob("*.jpg"))) \
        if (session_dir / "frames").is_dir() else 0
    lines = [f"project: {project}   session: {session_dir.name}   "
             f"scan frames: {n_frames}   objects: {len(store.list_instances())}"]
    desc = store.get_meta("scene_description")
    if desc:
        lines.append(f"scene description (vlm_proposed): {desc}")
    else:
        lines.append("scene description: not generated yet — call describe_scene "
                     "when asked what the scene is")
    # per-object deep analyses (shape proposer's dossier): one line each so
    # the assistant KNOWS what every element is; full text via describe_object
    import json as _json
    for i in store.list_instances():
        iid = i["instance_id"]
        raw = store.get_meta(f"object_analysis_{iid}")
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
        self.tools = SpatialTools(self.store)  # display frame (viewer-consistent)
        self.backend = backend
        # session logs default next to the scene store: output/spatial_qa_logs/
        self.session_log_dir = Path(session_log_dir) if session_log_dir \
            else Path(store_path).parent / "spatial_qa_logs"

    def system_prompt(self) -> str:
        return _SYSTEM_TEMPLATE.format(session=_session_block(self.store),
                                       inventory=_inventory(self.store))

    def ask(self, question: str, images: list | None = None,
            max_iterations: int = 8) -> dict[str, Any]:
        from semantic.client import get_semantic_client
        from semantic.types import system, user

        t0 = time.time()
        client = get_semantic_client(backend=self.backend, consumer="phase5.qa")
        messages = [system(self.system_prompt()), user(question, images=images)]
        out = client.run_tool_loop(
            messages, tools=self.tools.openai_schemas(),
            tool_impls=self.tools.impls(), max_iterations=max_iterations,
            consumer="phase5.qa")
        result = {
            "question": question,
            "answer": out["answer"],
            "tool_trace": out["tool_trace"],     # traceability: tool + args + result
            "iterations": out["iterations"],
            "stopped": out["stopped"],
            "latency_s": round(time.time() - t0, 2),
        }
        self._log_session(result, n_images=len(images or []))
        return result

    def _log_session(self, result: dict, n_images: int = 0) -> None:
        """Persist the full JSON session log (spec: log JSON completo por
        sesión). Best-effort — logging must never break an answer."""
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

    def close(self):
        self.store.close()
