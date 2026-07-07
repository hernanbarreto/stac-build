# STAC-Builder — Phase 5: spatial Q&A orchestrator.
#
# System prompt (scene inventory from the R.8 store + tool catalog + rules) +
# a bounded tool-calling loop over the shared semantic service. Every figure
# comes from a tool (the VLM never measures); answers cite the tool + arguments,
# declare uncertainty, and reply in the question's language. Multimodal: a
# keyframe of the referred object can be attached to disambiguate.
#
# PROVENANCE: ours. Tool semantics ported from R3D (see tools.py); the loop uses
# our SemanticClient.run_tool_loop (Qwen3 native tool-calling), not R3D's text
# protocol.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from typing import Any

from phase_r.instance_store import InstanceStore

from .tools import SpatialTools

_SYSTEM_TEMPLATE = """You are a spatial assistant for construction-site 3D reconstruction supervision.

SCENE INVENTORY (from the reconstruction; ids are stable):
{inventory}

RULES:
- You NEVER estimate or invent numbers. EVERY figure must come from a tool call.
- Call list_objects first if you are unsure which id an object refers to.
- Units are SI (metres, m^2, m^3, degrees). State the unit.
- Cite the tool and its arguments for every figure you report.
- If a tool returns {{"insufficient_data": true}}, say the data is insufficient —
  do NOT extrapolate or guess.
- Declare uncertainty when a tool reports low confidence.
- Answer in the SAME LANGUAGE as the question.
"""


def _inventory(store: InstanceStore, limit: int = 60) -> str:
    lines = []
    scene_type = store.get_meta("scene_type")
    if scene_type:
        lines.append(f"scene: {scene_type}")
    for i in store.list_instances()[:limit]:
        lines.append(f"  id={i['instance_id']} {i['label']} "
                     f"(status={i['status']}, views={i['n_views']})")
    return "\n".join(lines) if lines else "  (no objects in store)"


class SpatialQA:
    def __init__(self, store_path: str, backend: str = "qwen_local"):
        self.store = InstanceStore(store_path)
        self.tools = SpatialTools(self.store)
        self.backend = backend

    def system_prompt(self) -> str:
        return _SYSTEM_TEMPLATE.format(inventory=_inventory(self.store))

    def ask(self, question: str, images: list | None = None,
            max_iterations: int = 8) -> dict[str, Any]:
        from semantic.client import get_semantic_client
        from semantic.types import system, user

        client = get_semantic_client(backend=self.backend, consumer="phase5.qa")
        messages = [system(self.system_prompt()), user(question, images=images)]
        out = client.run_tool_loop(
            messages, tools=self.tools.openai_schemas(),
            tool_impls=self.tools.impls(), max_iterations=max_iterations,
            consumer="phase5.qa")
        return {
            "question": question,
            "answer": out["answer"],
            "tool_trace": out["tool_trace"],     # traceability: tool + args + result
            "iterations": out["iterations"],
            "stopped": out["stopped"],
        }

    def close(self):
        self.store.close()
