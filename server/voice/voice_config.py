"""The ``voice:`` block of config.yaml, typed: a missing or mistyped key fails naming it."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEVICES = ("cpu", "cuda")
_CACHE: Dict[str, Any] = {}


def _need(node: Any, key: str, types, where: str, check=None):
    v = node.get(key) if isinstance(node, dict) else None
    if isinstance(v, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
        raise RuntimeError(f"config.yaml: {where}.{key} is missing or mistyped")
    if not isinstance(v, types) or (check is not None and not check(v)):
        raise RuntimeError(f"config.yaml: {where}.{key} is missing or mistyped")
    return v


def validate(block: Any) -> Dict[str, Any]:
    if not isinstance(block, dict):
        raise RuntimeError("config.yaml: the voice block is missing")
    out: Dict[str, Any] = {
        "enabled": _need(block, "enabled", bool, "voice"),
        "start_with_backend": _need(block, "start_with_backend", bool, "voice"),
        "name": _need(block, "name", str, "voice", lambda s: bool(s.strip())),
        "device": _need(block, "device", str, "voice", lambda s: s in DEVICES),
        "languages": _need(block, "languages", list, "voice",
                           lambda v: v and all(isinstance(x, str) and x for x in v)),
    }
    svc = _need(block, "service", dict, "voice")
    out["service"] = {
        "host": _need(svc, "host", str, "voice.service"),
        "port": _need(svc, "port", int, "voice.service", lambda p: 0 < p < 65536),
        "startup_timeout_s": float(_need(svc, "startup_timeout_s", (int, float), "voice.service", lambda x: x > 0)),
        "request_timeout_s": float(_need(svc, "request_timeout_s", (int, float), "voice.service", lambda x: x > 0)),
    }
    stt = _need(block, "stt", dict, "voice")
    out["stt"] = {
        "model": _need(stt, "model", str, "voice.stt", lambda s: bool(s.strip())),
        "max_audio_s": float(_need(stt, "max_audio_s", (int, float), "voice.stt", lambda x: 0 < x <= 30)),
        "max_new_tokens": _need(stt, "max_new_tokens", int, "voice.stt", lambda x: x > 0),
        "no_speech_threshold": float(_need(stt, "no_speech_threshold", (int, float), "voice.stt", lambda x: 0 <= x <= 1)),
        "logprob_threshold": float(_need(stt, "logprob_threshold", (int, float), "voice.stt")),
        "prompt_max_labels": _need(stt, "prompt_max_labels", int, "voice.stt", lambda x: x >= 0),
        "vad_threshold": float(_need(stt, "vad_threshold", (int, float), "voice.stt", lambda x: 0 < x < 1)),
    }
    tts = _need(block, "tts", dict, "voice")
    voices = _need(tts, "voices", dict, "voice.tts")
    for lang in out["languages"]:
        v = voices.get(lang)
        if not isinstance(v, dict) or not isinstance(v.get("voice"), str) or not isinstance(v.get("lang_code"), str):
            raise RuntimeError(f"config.yaml: voice.tts.voices.{lang} is missing (voice + lang_code)")
    out["tts"] = {
        "model": _need(tts, "model", str, "voice.tts", lambda s: bool(s.strip())),
        "voices": {lang: {"voice": voices[lang]["voice"], "lang_code": voices[lang]["lang_code"]}
                   for lang in out["languages"]},
        "speed": float(_need(tts, "speed", (int, float), "voice.tts", lambda x: 0.5 <= x <= 2.0)),
        "max_chars": _need(tts, "max_chars", int, "voice.tts", lambda x: x > 0),
    }
    return out


def load(path: Path | str | None = None) -> Dict[str, Any]:
    """The validated block; config.yaml is parsed once per file on disk (inode, size, mtime)."""
    path = Path(path) if path is not None else CONFIG_PATH
    st = path.stat()
    stamp = (str(path), st.st_ino, st.st_size, st.st_mtime_ns)
    if _CACHE.get("stamp") != stamp:
        full = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _CACHE.update(stamp=stamp, value=validate(full.get("voice")))
    import copy
    return copy.deepcopy(_CACHE["value"])


def base_url(cfg: Dict[str, Any] | None = None) -> str:
    cfg = cfg or load()
    return f"http://{cfg['service']['host']}:{cfg['service']['port']}"
