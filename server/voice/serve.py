"""THE VOICE SERVICE — its own process, in the `voice` env, on ONE device.

    python -m voice.serve            (launched by the backend: voice/service.py)

    GET  /health        {status, device, languages, stt, tts, voices, loaded_s}
    POST /transcribe    body = the audio as the browser recorded it (webm / ogg / wav / …)
                        ?prompt=<vocabulary: the system's name, the scene's labels>
                        → {text, language, probabilities, audio_s, seconds, no_speech}
    POST /speak         {"text", "lang"} → audio/wav (PCM 16, mono, 24 kHz);
                        headers X-Audio-Seconds, X-Synthesis-Seconds, X-Spoken-Text (URL-encoded)

speech → text: Whisper through transformers. The LANGUAGE is decided among the system's own
(``voice.languages``) from the first decoder step — never one of Whisper's other 96. Silence is
not transcribed (the vendor's rule: no-speech probability over its threshold AND a low mean
log-probability). Greedy, temperature 0: the same audio gives the same text.
text → speech: Kokoro, one voice per language, the text first turned into what is SAID
(voice/spoken.py).
"""
from __future__ import annotations

import io
import os
import sys
import time
import urllib.parse
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from fastapi import FastAPI, Request          # noqa: E402 — module level: the routes' annotations
from fastapi.responses import JSONResponse, Response      # noqa: E402

from voice import spoken                     # noqa: E402
from voice import voice_config               # noqa: E402

SAMPLE_RATE_IN = 16000                        # Whisper's
SAMPLE_RATE_OUT = 24000                       # Kokoro's


class VoiceError(RuntimeError):
    pass


def decode_audio(data: bytes, max_seconds: float) -> "Any":
    """Any container the browser records (webm / ogg opus, mp4 aac, wav) → float32 mono 16 kHz.
    More than ``max_seconds`` is refused naming the bound, never cut silently."""
    import av
    import numpy as np
    if not data:
        raise VoiceError("no audio was sent")
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise VoiceError("the file sent has no audio stream")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE_IN)
            chunks: List[Any] = []
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):                     # the resampler's tail
                chunks.append(out.to_ndarray().reshape(-1))
    except VoiceError:
        raise
    except Exception as e:  # noqa: BLE001 — told to the caller as what it is
        raise VoiceError(f"the audio could not be decoded ({type(e).__name__}: {e})") from e
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    seconds = len(audio) / SAMPLE_RATE_IN
    if seconds > max_seconds:
        raise VoiceError(f"the audio lasts {seconds:.1f} s — more than voice.stt.max_audio_s ({max_seconds:g} s)")
    return audio


def wav_bytes(samples: "Any", rate: int = SAMPLE_RATE_OUT) -> bytes:
    """float samples in [-1, 1] → a PCM-16 mono WAV."""
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767.0).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Engines:
    """Both engines on one device, loaded once."""

    def __init__(self, cfg: Dict[str, Any]):
        import torch
        self.cfg = cfg
        self.torch = torch
        dev = cfg["device"]
        if dev == "cuda" and not torch.cuda.is_available():
            raise VoiceError("config.yaml: voice.device is cuda and no card is visible to the voice service")
        self.device = dev
        torch.manual_seed(0)
        t0 = time.time()
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(cfg["stt"]["model"])
        self.dtype = torch.float16 if dev == "cuda" else torch.float32
        self.stt = WhisperForConditionalGeneration.from_pretrained(
            cfg["stt"]["model"], dtype=self.dtype).to(dev).eval()
        tok = self.proc.tokenizer
        self.langs: List[str] = list(cfg["languages"])
        self.lang_tokens = [tok.convert_tokens_to_ids(f"<|{l}|>") for l in self.langs]
        if any(t is None or t == tok.unk_token_id for t in self.lang_tokens):
            raise VoiceError(f"voice.languages {self.langs}: the recogniser does not know one of them")
        self.no_speech_token = tok.convert_tokens_to_ids("<|nospeech|>")
        if self.no_speech_token is None or self.no_speech_token == tok.unk_token_id:
            self.no_speech_token = tok.convert_tokens_to_ids("<|nocaptions|>")
        self.prefix_tokens = [tok.convert_tokens_to_ids("<|transcribe|>"),
                              tok.convert_tokens_to_ids("<|notimestamps|>")]
        # IS THERE SPEECH AT ALL — asked first of a voice-activity detector (Silero, MIT): the
        # recogniser writes a word for silence ("you") and its own no-speech flag stayed at 0.0
        # on silence, a tone and noise (measured 2026-11-14)
        from silero_vad import get_speech_timestamps, load_silero_vad
        self.vad = load_silero_vad()
        self._speech_timestamps = get_speech_timestamps
        from kokoro import KModel, KPipeline
        self.tts_model = KModel(repo_id=cfg["tts"]["model"]).to(dev).eval()
        self.pipes = {lang: KPipeline(lang_code=v["lang_code"], repo_id=cfg["tts"]["model"], model=self.tts_model)
                      for lang, v in cfg["tts"]["voices"].items()}
        self.loaded_s = round(time.time() - t0, 1)
        # the first call of each engine (kernels, the voices' files): paid here, not by the user
        for lang in self.langs:
            self.speak("STAC.", lang)
        import numpy as np
        self.transcribe(np.zeros(SAMPLE_RATE_IN, np.float32), "")

    # ── speech → text ───────────────────────────────────────────────────────────────────────
    def transcribe(self, audio: "Any", prompt: str = "") -> Dict[str, Any]:
        torch = self.torch
        t0 = time.time()
        seconds = len(audio) / SAMPLE_RATE_IN
        langs0 = {l: 0.0 for l in self.langs}
        if seconds == 0.0:
            return {"text": "", "language": self.langs[0], "probabilities": langs0, "no_speech": True,
                    "speech_s": 0.0, "audio_s": 0.0, "seconds": round(time.time() - t0, 3)}
        with torch.no_grad():
            spans = self._speech_timestamps(torch.from_numpy(audio), self.vad, sampling_rate=SAMPLE_RATE_IN,
                                            threshold=float(self.cfg["stt"]["vad_threshold"]))
        speech_s = sum(sp["end"] - sp["start"] for sp in spans) / SAMPLE_RATE_IN
        if not spans:
            return {"text": "", "language": self.langs[0], "probabilities": langs0, "no_speech": True,
                    "speech_s": 0.0, "audio_s": round(seconds, 2), "seconds": round(time.time() - t0, 3)}
        enc = self.proc.feature_extractor(audio, sampling_rate=SAMPLE_RATE_IN, return_tensors="pt",
                                          return_attention_mask=True)
        feats = enc.input_features.to(self.device, self.dtype)
        mask = enc.attention_mask.to(self.device)
        tok = self.proc.tokenizer
        sot = self.stt.config.decoder_start_token_id
        with torch.no_grad():
            first = self.stt(feats, decoder_input_ids=torch.tensor([[sot]], device=self.device)).logits[0, -1].float()
            no_speech = float(torch.softmax(first, dim=-1)[self.no_speech_token])
            lp = torch.log_softmax(first[self.lang_tokens], dim=-1)
            k_lang = int(torch.argmax(lp))
            lang = self.langs[k_lang]
            kwargs: Dict[str, Any] = dict(language=lang, task="transcribe", do_sample=False, num_beams=1,
                                          max_new_tokens=int(self.cfg["stt"]["max_new_tokens"]),
                                          attention_mask=mask)
            prompt_ids: List[int] = []
            if prompt.strip():
                p = self.proc.get_prompt_ids(prompt.strip(), return_tensors="pt")
                prompt_ids = [int(x) for x in p.tolist()]
                kwargs["prompt_ids"] = p.to(self.device)
            ids = self.stt.generate(feats, **kwargs)[0].tolist()
            # THE TEXT: the tokens after the decoder's own prefix (… <|notimestamps|>), whatever
            # of that prefix this version of generate returns; special tokens are not text
            last = max((k for k, t in enumerate(ids) if t in (sot, self.prefix_tokens[-1])), default=-1)
            said = [t for t in ids[last + 1:] if t < tok.eos_token_id]
            text = tok.decode(said, skip_special_tokens=True).strip()
            # the mean log-probability of what was said, end token included (the vendor's
            # definition): one teacher-forced pass over the decoder's full input
            mean_logp = 0.0
            if said:
                head = prompt_ids + [sot, self.lang_tokens[k_lang]] + self.prefix_tokens
                target = said + [tok.eos_token_id]
                dec = torch.tensor([head + said], device=self.device)
                logits = self.stt(feats, decoder_input_ids=dec).logits[0, len(head) - 1:].float()
                logp = torch.log_softmax(logits, dim=-1).gather(
                    1, torch.tensor(target, device=self.device).unsqueeze(1)).squeeze(1)
                mean_logp = float(logp.mean())
        silent = not text or (no_speech > self.cfg["stt"]["no_speech_threshold"]
                              and mean_logp < self.cfg["stt"]["logprob_threshold"])
        return {"text": "" if silent else text, "language": lang,
                "probabilities": {l: round(float(p), 4) for l, p in zip(self.langs, torch.exp(lp).tolist())},
                "no_speech": bool(silent), "no_speech_probability": round(no_speech, 4),
                "mean_logprob": round(mean_logp, 4), "speech_s": round(speech_s, 2),
                "audio_s": round(seconds, 2), "seconds": round(time.time() - t0, 3)}

    # ── text → speech ───────────────────────────────────────────────────────────────────────
    def speak(self, text: str, lang: str) -> Dict[str, Any]:
        torch = self.torch
        if lang not in self.pipes:
            raise VoiceError(f"no voice for language {lang!r} (voice.tts.voices has {sorted(self.pipes)})")
        said = spoken.say(text, lang)
        if not said:
            raise VoiceError("nothing to say")
        if len(said) > self.cfg["tts"]["max_chars"]:
            raise VoiceError(f"the text has {len(said)} characters — more than voice.tts.max_chars "
                             f"({self.cfg['tts']['max_chars']})")
        t0 = time.time()
        voice = self.cfg["tts"]["voices"][lang]["voice"]
        with torch.no_grad():
            parts = [r.audio for r in self.pipes[lang](said, voice=voice, speed=float(self.cfg["tts"]["speed"]))]
        parts = [p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in parts if p is not None]
        if not parts:
            raise VoiceError("the voice produced no audio")
        samples = torch.cat([p.reshape(-1).float() for p in parts]).cpu().numpy()
        return {"wav": wav_bytes(samples), "said": said, "audio_s": round(len(samples) / SAMPLE_RATE_OUT, 2),
                "seconds": round(time.time() - t0, 3)}


def build_app(engines: Engines):
    app = FastAPI()
    cfg = engines.cfg

    @app.get("/health")
    def health():
        return {"status": "ok", "device": engines.device, "languages": engines.langs,
                "stt": cfg["stt"]["model"], "tts": cfg["tts"]["model"],
                "voices": {l: v["voice"] for l, v in cfg["tts"]["voices"].items()},
                "loaded_s": engines.loaded_s, "pid": os.getpid()}

    @app.post("/transcribe")
    async def transcribe(request: Request, prompt: str = ""):
        data = await request.body()
        try:
            audio = decode_audio(data, cfg["stt"]["max_audio_s"])
            return engines.transcribe(audio, prompt)
        except VoiceError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    @app.post("/speak")
    async def speak(body: Dict[str, Any]):
        try:
            res = engines.speak(str(body.get("text") or ""), str(body.get("lang") or ""))
        except VoiceError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return Response(res["wav"], media_type="audio/wav", headers={
            "X-Audio-Seconds": str(res["audio_s"]), "X-Synthesis-Seconds": str(res["seconds"]),
            "X-Spoken-Text": urllib.parse.quote(res["said"])})

    return app


def main(argv: Optional[List[str]] = None) -> int:
    cfg = voice_config.load()
    if not cfg["enabled"]:
        print("[Voice] voice.enabled is false — not starting")
        return 0
    # the pod exports HF_HUB_ENABLE_HF_TRANSFER=1 and the voice env has no hf_transfer: plain
    # downloads (the weights are fetched once into the pod's one cache)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    print(f"[Voice] loading {cfg['stt']['model']} + {cfg['tts']['model']} on {cfg['device']} …", flush=True)
    engines = Engines(cfg)
    print(f"[Voice] ready in {engines.loaded_s} s — languages {engines.langs}", flush=True)
    import uvicorn
    uvicorn.run(build_app(engines), host=cfg["service"]["host"], port=int(cfg["service"]["port"]),
                log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
