# STAC-Builder — Semantic Service (Phase 0)
#
# Single persistent semantic backend (one vLLM OpenAI-compatible endpoint
# serving Qwen3-VL) shared by every downstream consumer of the pipeline:
#   Phase 1 (SAM3 auto-prompting), Phase 2 (segment classification),
#   Phase 3 (visual findings), Phase 4 (capture QC), Phase 5 (spatial Q&A),
#   Phase 6 (report drafting).
#
# ARCHITECTURE
#   vLLM runs in its OWN conda env (`semantic`) as a standalone service. The
#   backend (env `da3`) and all other consumers NEVER load the model in-process
#   and NEVER talk to vLLM directly — they go through `semantic_client` (this
#   package), a dependency-light HTTP client (requests + Pillow only). This
#   avoids the transformers version conflict (`da3` pins transformers<4.48;
#   vLLM needs a newer one) and keeps a single shared GPU-resident model.
#
# PROVENANCE
#   Ours. The vLLM *reference* client shipped by R3D
#   (vendor/r3d/r3d/pipeline/eval/vllm_client.py) runs vLLM in-process (offline
#   `LLM`); we deliberately diverge to a persistent OpenAI-compatible SERVICE so
#   multiple consumers across multiple conda envs can share one model. The exact
#   baseline model id (Qwen/Qwen3-VL-8B-Instruct) is taken from R3D's
#   generate_responses/rgb_overlay_evals scripts (spec Phase 0.a).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from .types import (
    ImageRef,
    LLMResponse,
    Message,
    ToolCall,
    assistant,
    system,
    tool_result,
    user,
)
from .backends import LLMBackend, QwenLocalBackend
from .client import SemanticClient, get_semantic_client, available_backends
from .semantic_config import load_semantic_config, backend_config

__all__ = [
    "ImageRef",
    "LLMResponse",
    "Message",
    "ToolCall",
    "assistant",
    "system",
    "tool_result",
    "user",
    "LLMBackend",
    "QwenLocalBackend",
    "SemanticClient",
    "get_semantic_client",
    "available_backends",
    "load_semantic_config",
    "backend_config",
]
