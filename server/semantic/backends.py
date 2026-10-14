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

import re
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

    def engine_kv_cache(self) -> dict[str, Any]:
        """The served engine's KV cache ``{num_gpu_blocks, block_size, source}`` — what bounds
        how many requests may be in flight (semantic.parallel). A backend that cannot read it
        RAISES; the caller then sends one request at a time, declared."""
        raise NotImplementedError(f"backend '{getattr(self, 'name', '?')}' publishes no record "
                                  f"of its engine's KV cache")


# the Prometheus gauge vLLM sets ONCE the engine is initialised, after num_gpu_blocks is known
# (vllm 0.19 v1/metrics/loggers.py:1007-1029, PrometheusStatLogger.log_engine_initialized): an
# info-style gauge whose labels are the engine's CacheConfig (config/cache.py metrics_info)
CACHE_CONFIG_INFO = "vllm:cache_config_info"
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_cache_config_info(text: str, source: str = "/metrics") -> dict[str, Any]:
    """``{num_gpu_blocks, block_size, source}`` from a Prometheus exposition: the labels of every
    ``vllm:cache_config_info`` sample (one per engine; the smallest cache wins — a request may
    land on any). RAISES naming what is missing (the gauge, a label, a label that is no
    integer — 'None' while the engine has not sized its cache)."""
    found: list[tuple[int, int]] = []
    for line in (text or "").splitlines():
        if not line.startswith(CACHE_CONFIG_INFO + "{"):
            continue
        labels = dict(_LABEL_RE.findall(line[len(CACHE_CONFIG_INFO) + 1:line.rfind("}")]))
        try:
            found.append((int(labels["num_gpu_blocks"]), int(labels["block_size"])))
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"{source}: {CACHE_CONFIG_INFO} carries no integer "
                               f"num_gpu_blocks / block_size ({e}; labels "
                               f"num_gpu_blocks={labels.get('num_gpu_blocks')!r}, "
                               f"block_size={labels.get('block_size')!r})") from e
    if not found:
        raise RuntimeError(f"{source}: no {CACHE_CONFIG_INFO} sample — the engine publishes no "
                           f"record of its KV cache")
    if len({bs for _n, bs in found}) != 1:
        raise RuntimeError(f"{source}: the engines disagree on the KV block size {found}")
    n, bs = min(found)
    return {"num_gpu_blocks": n, "block_size": bs,
            "source": f"GET {source} {CACHE_CONFIG_INFO}"}


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
        # REPRODUCIBILITY (USER 2026-09-22: *"debe ser lo mas determinista
        # posible"*). temperature 0 already asks for the argmax, but the engine
        # batches requests and the batch composition changes between runs: the
        # reduction order changes, the logits move in the last decimal and a
        # near-tie falls the other way — one token, one different phrase, a
        # different vocabulary downstream. A fixed seed pins the sampler so
        # everything the engine CAN make deterministic is.
        self._seed = gen.get("seed")

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
        if self._seed is not None:
            payload["seed"] = int(self._seed)
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

    def engine_kv_cache(self) -> dict[str, Any]:
        """The engine's KV cache from its own Prometheus endpoint (GET <root>/metrics, served by
        every ``vllm serve`` — vllm/entrypoints/serve/instrumentator/metrics.py): the
        ``vllm:cache_config_info`` labels num_gpu_blocks and block_size (USER 2026-10-09,
        semantic.parallel: how many requests may be in flight is derived from them). A read
        bound like the chat call's (``service.request_timeout_s``). RAISES when it cannot be
        read."""
        root = self._base_url[:-len("/v1")] if self._base_url.endswith("/v1") else self._base_url
        url = f"{root}/metrics"
        resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"},
                            timeout=self._timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} at {url}")
        return parse_cache_config_info(resp.text, url)


def make_backend(name: str | None = None, cfg: dict[str, Any] | None = None) -> LLMBackend:
    """Factory. Today every backend is a Qwen3-VL HTTP client; the indirection
    exists so a future non-Qwen local backend can slot in without touching
    consumers."""
    cfg = cfg or load_semantic_config()
    name = name or cfg.get("default_backend", "qwen_local")
    return QwenLocalBackend(name=name, cfg=cfg)
