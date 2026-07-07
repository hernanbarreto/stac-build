# STAC-Builder — Semantic Service: the single client every consumer uses.
#
# SemanticClient wraps an LLMBackend and adds:
#   - centralized JSONL call logging (consumer, prompt, image refs, response…)
#   - a bounded tool-calling loop (run_tool_loop) for the Q&A orchestrator and
#     any consumer that needs deterministic tools to do the measuring.
# Nobody talks to vLLM directly; this is the only door. Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import time
from typing import Any, Callable

from .backends import LLMBackend, make_backend
from .call_log import log_call
from .semantic_config import load_semantic_config
from .types import LLMResponse, Message, assistant, tool_result

# A tool implementation takes parsed kwargs and returns a JSON-serializable result.
ToolImpl = Callable[..., Any]


class SemanticClient:
    def __init__(self, backend: LLMBackend, consumer: str = "unknown", cfg: dict | None = None):
        self.backend = backend
        self.consumer = consumer
        self._cfg = cfg or load_semantic_config()
        self._max_iterations = self._cfg.get("tool_calling", {}).get("max_iterations", 8)

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def health(self) -> dict[str, Any]:
        return self.backend.health()

    # ── single call ─────────────────────────────────────────────────
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        consumer: str | None = None,
    ) -> LLMResponse:
        consumer = consumer or self.consumer
        error = None
        resp: LLMResponse | None = None
        t0 = time.time()
        try:
            resp = self.backend.chat(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return resp
        except Exception as e:  # noqa: BLE001
            error = str(e)
            raise
        finally:
            log_call(
                {
                    "consumer": consumer,
                    "backend": self.backend.name,
                    "model": self.backend.model,
                    "messages": [m.log_view() for m in messages],
                    "tools_offered": [t["function"]["name"] for t in (tools or [])],
                    "response": (
                        {
                            "content": (resp.content[:2000] if resp and resp.content else None),
                            "tool_calls": [tc.log_view() for tc in resp.tool_calls] if resp else [],
                            "finish_reason": resp.finish_reason if resp else None,
                            "usage": resp.usage if resp else {},
                        }
                        if resp
                        else None
                    ),
                    "latency_ms": round((time.time() - t0) * 1000.0, 1),
                    "error": error,
                }
            )

    # ── bounded tool-calling loop ───────────────────────────────────
    def run_tool_loop(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        tool_impls: dict[str, ToolImpl],
        max_iterations: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        consumer: str | None = None,
    ) -> dict[str, Any]:
        """Run the model with tools until it produces a final text answer or the
        iteration budget is exhausted. Returns
        {answer, iterations, tool_trace, messages}. Every tool result is
        appended to the conversation and every LLM turn is logged.
        """
        consumer = consumer or self.consumer
        max_iterations = max_iterations or self._max_iterations
        convo = list(messages)
        trace: list[dict[str, Any]] = []

        for it in range(max_iterations):
            resp = self.chat(
                convo,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens,
                consumer=f"{consumer}#it{it}",
            )
            if not resp.has_tool_calls:
                return {
                    "answer": resp.content or "",
                    "iterations": it + 1,
                    "tool_trace": trace,
                    "messages": convo,
                    "stopped": "final_answer",
                }

            convo.append(assistant(text=resp.content, tool_calls=resp.tool_calls))
            for tc in resp.tool_calls:
                impl = tool_impls.get(tc.name)
                if impl is None:
                    result: Any = {"error": f"unknown tool '{tc.name}'"}
                else:
                    try:
                        result = impl(**tc.arguments)
                    except Exception as e:  # noqa: BLE001
                        result = {"error": f"{type(e).__name__}: {e}"}
                trace.append({"iteration": it, "tool": tc.name, "arguments": tc.arguments, "result": result})
                convo.append(tool_result(tc.id, _to_text(result), name=tc.name))

        # Budget exhausted — ask for a final answer without tools.
        resp = self.chat(convo, temperature=temperature, max_tokens=max_tokens, consumer=f"{consumer}#final")
        return {
            "answer": resp.content or "",
            "iterations": max_iterations,
            "tool_trace": trace,
            "messages": convo,
            "stopped": "max_iterations",
        }


def _to_text(result: Any) -> str:
    import json

    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


# ── module-level accessors ──────────────────────────────────────────
_clients: dict[str, SemanticClient] = {}


def get_semantic_client(backend: str | None = None, consumer: str = "unknown") -> SemanticClient:
    """Return a SemanticClient for the named backend (default from config).

    Cached per backend name; `consumer` is set on the returned view so logs are
    attributed. Safe to call from any conda env / process.
    """
    cfg = load_semantic_config()
    name = backend or cfg.get("default_backend", "qwen_local")
    if name not in _clients:
        _clients[name] = SemanticClient(make_backend(name, cfg), consumer=consumer, cfg=cfg)
    client = _clients[name]
    client.consumer = consumer
    return client


def available_backends() -> list[str]:
    return sorted(load_semantic_config().get("backends", {}).keys())
