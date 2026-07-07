# STAC-Builder — Semantic Service: response parsing (tolerant).
#
# Turns a raw OpenAI /chat/completions payload into an LLMResponse, parsing
# tool-call arguments with tolerance (the hermes parser occasionally emits
# slightly-off JSON on the 8B). Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from typing import Any

from .types import LLMResponse, ToolCall


def _loads_tolerant(s: str) -> dict[str, Any]:
    """Best-effort JSON parse of tool-call arguments."""
    if s is None or s == "":
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    # Strip code fences / leading label, retry on the first {...} block.
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"_raw": s}


def parse_response(data: dict[str, Any], latency_ms: float) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    content = msg.get("content")
    finish = choice.get("finish_reason")
    model = data.get("model", "")
    usage = data.get("usage", {}) or {}

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        tool_calls.append(
            ToolCall(
                id=tc.get("id", f"call_{len(tool_calls)}"),
                name=fn.get("name", ""),
                arguments=_loads_tolerant(fn.get("arguments", "")),
            )
        )
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish,
        model=model,
        usage=usage,
        latency_ms=latency_ms,
        raw=data,
    )
