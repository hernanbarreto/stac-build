# STAC-Builder — Semantic Service: message / response data types.
#
# OpenAI chat-completions shaped structures, plus tolerant helpers to attach
# images (PIL / numpy / path / bytes / data-URL) to a message. Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Pillow is required only for image encoding; text-only callers never import it.
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - Pillow always present in our envs
    Image = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


ImageLike = Any  # PIL.Image | np.ndarray | str | Path | bytes


@dataclass
class ImageRef:
    """A reference to an image attached to a message.

    We keep both the encoded data-URL (sent to vLLM) and a short content hash
    (logged instead of the bytes, so JSONL logs stay small yet traceable).
    """

    data_url: str
    sha1: str
    source: str  # human-readable origin, e.g. a file path or "<ndarray HxW>"

    @property
    def short_sha1(self) -> str:
        return self.sha1[:12]


def _encode_image(img: ImageLike, fmt: str = "JPEG", quality: int = 90) -> ImageRef:
    """Encode any supported image type to a base64 data-URL + content hash."""
    source = "<image>"
    raw: bytes

    if isinstance(img, (str, Path)):
        source = str(img)
        raw = Path(img).read_bytes()
        mime = _mime_for_path(str(img))
        b64 = base64.b64encode(raw).decode("ascii")
        sha1 = hashlib.sha1(raw).hexdigest()
        return ImageRef(f"data:{mime};base64,{b64}", sha1, source)

    if isinstance(img, bytes):
        source = f"<bytes {len(img)}B>"
        raw = img
        b64 = base64.b64encode(raw).decode("ascii")
        sha1 = hashlib.sha1(raw).hexdigest()
        return ImageRef(f"data:image/jpeg;base64,{b64}", sha1, source)

    if isinstance(img, str) and img.startswith("data:"):
        # already a data URL
        sha1 = hashlib.sha1(img.encode("ascii")).hexdigest()
        return ImageRef(img, sha1, "<data-url>")

    # numpy array -> PIL
    if np is not None and isinstance(img, np.ndarray):
        source = f"<ndarray {'x'.join(map(str, img.shape))}>"
        if Image is None:
            raise RuntimeError("Pillow required to encode numpy images")
        arr = img
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8) if arr.max() > 1.0 else (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr)

    if Image is not None and isinstance(img, Image.Image):
        buf = io.BytesIO()
        rgb = img.convert("RGB")
        rgb.save(buf, format=fmt, quality=quality)
        raw = buf.getvalue()
        mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else f"image/{fmt.lower()}"
        b64 = base64.b64encode(raw).decode("ascii")
        sha1 = hashlib.sha1(raw).hexdigest()
        return ImageRef(f"data:{mime};base64,{b64}", sha1, source)

    raise TypeError(f"Unsupported image type: {type(img)!r}")


def _mime_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


@dataclass
class Message:
    """A single chat message in OpenAI format.

    `images` are attached to a `user` message as image_url content parts. For
    `assistant` tool-call turns, `tool_calls` carries the model's requested
    calls; for `tool` role, `tool_call_id` links the result back to the call.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    text: str | None = None
    images: list[ImageRef] = field(default_factory=list)
    tool_calls: list["ToolCall"] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name for role="tool"

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.role == "user" and self.images:
            content: list[dict[str, Any]] = []
            if self.text:
                content.append({"type": "text", "text": self.text})
            for ref in self.images:
                content.append({"type": "image_url", "image_url": {"url": ref.data_url}})
            msg["content"] = content
        elif self.role == "assistant" and self.tool_calls:
            msg["content"] = self.text or ""
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        else:
            msg["content"] = self.text or ""
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg

    def log_view(self, max_text: int = 400) -> dict[str, Any]:
        """Compact, image-byte-free view for JSONL logging."""
        v: dict[str, Any] = {"role": self.role}
        if self.text:
            v["text"] = self.text if len(self.text) <= max_text else self.text[:max_text] + "…"
        if self.images:
            v["images"] = [{"sha1": r.short_sha1, "source": r.source} for r in self.images]
        if self.tool_calls:
            v["tool_calls"] = [tc.log_view() for tc in self.tool_calls]
        if self.tool_call_id:
            v["tool_call_id"] = self.tool_call_id
        return v


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }

    def log_view(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    model: str
    usage: dict[str, Any]
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ── message constructors ────────────────────────────────────────────

def _as_image_refs(images: Iterable[ImageLike] | None) -> list[ImageRef]:
    if not images:
        return []
    return [img if isinstance(img, ImageRef) else _encode_image(img) for img in images]


def system(text: str) -> Message:
    return Message(role="system", text=text)


def user(text: str, images: Iterable[ImageLike] | None = None) -> Message:
    return Message(role="user", text=text, images=_as_image_refs(images))


def assistant(text: str | None = None, tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role="assistant", text=text, tool_calls=tool_calls or [])


def tool_result(tool_call_id: str, content: str, name: str | None = None) -> Message:
    return Message(role="tool", text=content, tool_call_id=tool_call_id, name=name)
