"""A TRANSCRIPTION THAT REPEATS ITSELF IS NOT WHAT WAS SAID (USER 2026-11-24: "a veces cuando hablo en manos
libres aparece ese error de stac stac stac").

Traced that day on the recording that gave it (kept in /workspace/tmp/voice_trace_20261124): a weak copy of an
utterance with no clear speech in it. Without the vocabulary the recogniser writes "Thank you." (what Whisper
writes for noise); with the vocabulary, which begins "STAC, STAC Build.", greedy decoding loops on "STAC, " up to
``voice.stt.max_new_tokens``, with a mean log-probability of -0.05 — so the silence rule (a high no-speech
probability AND a low log-probability) never drops it, and since it names STAC, hands-free took it for an order.

The vendor's own test of such a decoding (openai-whisper, ``transcribe(compression_ratio_threshold=2.4)``): the
text's UTF-8 bytes over their zlib-compressed bytes. A transcription over the threshold is decoded again WITHOUT
the vocabulary; one still over it is not understood — nothing is written, nothing is acted on. Greedy both times:
the same audio still gives the same text.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable, Dict, Tuple


def compression_ratio(text: str) -> float:
    """openai-whisper's ``utils.compression_ratio``: the text's bytes over their zlib-compressed bytes."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def without_loops(decode: Callable[[bool], Tuple[str, float]], has_vocabulary: bool,
                  threshold: float) -> Dict[str, Any]:
    """``decode(with_vocabulary) -> (text, mean_logprob)``. The first decoding is the vocabulary's when there is
    one; a decoding over ``threshold`` is done again without it; still over it, nothing is kept.

    Returns ``text`` and ``mean_logprob`` (of the decoding kept; "" and 0.0 when none is), ``compression_ratio``
    (of the decoding kept, or of the last one tried), ``retried_without_vocabulary`` and ``repetitive`` (nothing
    kept because it repeats itself)."""
    text, logp = decode(has_vocabulary)
    ratio = compression_ratio(text)
    retried = False
    if ratio > threshold and has_vocabulary:
        retried = True
        text, logp = decode(False)
        ratio = compression_ratio(text)
    repetitive = ratio > threshold
    return {"text": "" if repetitive else text, "mean_logprob": 0.0 if repetitive else logp,
            "compression_ratio": round(ratio, 3), "retried_without_vocabulary": retried, "repetitive": repetitive}
