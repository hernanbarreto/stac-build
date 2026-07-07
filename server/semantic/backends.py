# STAC-Builder — Semantic Service: LLM backends.
#
# `LLMBackend` is the single interface every consumer codes against. Two
# concrete backends are configured (spec Phase 0):
#   - qwen_local        (default) : Qwen/Qwen3-VL-8B-Instruct  — R3D baseline
#   - qwen_local_large  (candidate): larger quantized Qwen3-VL for comparison
# Both are thin HTTP clients over the vLLM OpenAI-compatible endpoint; they
# differ only in the `served_model_name` they request. A single vLLM process
# serves ONE model at a time, so the comparison suite (Phase 7) serves each in
# turn. Ours (the interface + HTTP transport). vLLM is external.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from .semantic_config import backend_config, base_url, load_semantic_config
from .types import LLMResponse, Message, ToolCall
from ._parse import parse_response


class LLMBackend(ABC):
    """Interface all pipeline consumers use to reach the semantic service."""

    name: str
    model: str  # served_model_name requested from vLLM

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Single chat completion (optionally with tools / images)."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return {ok, served_models, error} — does the service answer and
        serve this backend's model?"""


class QwenLocalBackend(LLMBackend):
    """HTTP client to the vLLM OpenAI-compatible endpoint for a Qwen3-VL model."""

    def __init__(self, name: str = "qwen_local", cfg: dict[str, Any] | None = None):
        self._root_cfg = cfg or load_semantic_config()
        bcfg = backend_config(name, self._root_cfg)
        self.name = name
        self.model = bcfg["served_model_name"]
        self._bcfg = bcfg
        self._base_url = base_url(self._root_cfg)
        svc = self._root_cfg["service"]
        self._api_key = svc.get("api_key", "EMPTY")
        self._timeout = svc.get("request_timeout_s", 600)
        gen = self._root_cfg.get("generation", {})
        self._default_temp = gen.get("temperature", 0.0)
        self._default_max_tokens = gen.get("max_tokens", 2048)

    # ── OpenAI /chat/completions ────────────────────────────────────
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self._default_temp if temperature is None else temperature,
            "max_tokens": self._default_max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if extra_body:
            payload.update(extra_body)

        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        t0 = time.time()
        resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
        latency_ms = (time.time() - t0) * 1000.0
        if resp.status_code != 200:
            raise RuntimeError(
                f"semantic service HTTP {resp.status_code} at {url}: {resp.text[:500]}"
            )
        data = resp.json()
        return parse_response(data, latency_ms)

    def health(self) -> dict[str, Any]:
        try:
            url = f"{self._base_url}/models"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return {"ok": False, "error": f"HTTP {resp.status_code}", "served_models": []}
            served = [m["id"] for m in resp.json().get("data", [])]
            return {
                "ok": self.model in served,
                "served_models": served,
                "expected": self.model,
                "error": None if self.model in served else "model not served",
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "served_models": [], "error": str(e)}


def make_backend(name: str | None = None, cfg: dict[str, Any] | None = None) -> LLMBackend:
    """Factory. Today every backend is a Qwen3-VL HTTP client; the indirection
    exists so a future non-Qwen local backend can slot in without touching
    consumers."""
    cfg = cfg or load_semantic_config()
    name = name or cfg.get("default_backend", "qwen_local")
    return QwenLocalBackend(name=name, cfg=cfg)
