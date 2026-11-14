"""STAC's voice (USER 2026-11-14: "incorpora herramientas para hablarle y que traduzca a texto y que
responda con voz sintetica en los 3 idiomas del sistema, ingles, español y frances"): what the
backend holds — the typed configuration, the written answer as it is SAID, the recogniser's
vocabulary, the service's life. The engines themselves live in the `voice` env and are measured
on the card (scripts in the commit message); nothing here loads a model."""
from __future__ import annotations

import io
import sys
import time
import wave

import pytest

from voice import api as voice_api
from voice import service, spoken, voice_config


def test_the_configuration_is_typed_and_a_missing_key_fails_naming_it():
    cfg = voice_config.load()
    assert cfg["languages"] == ["es", "en", "fr"] and cfg["name"] == "STAC"
    assert set(cfg["tts"]["voices"]) == {"es", "en", "fr"}
    assert cfg["start_with_backend"] is True and cfg["device"] in voice_config.DEVICES
    import copy
    import yaml
    block = yaml.safe_load(voice_config.CONFIG_PATH.read_text())["voice"]
    for path, said in ((("stt", "model"), "voice.stt.model"), (("service", "port"), "voice.service.port"),
                       (("name",), "voice.name"), (("tts", "voices", "fr"), "voice.tts.voices.fr"),
                       (("stt", "vad_threshold"), "voice.stt.vad_threshold")):
        bad = copy.deepcopy(block)
        node = bad
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
        with pytest.raises(RuntimeError) as e:
            voice_config.validate(bad)
        assert said in str(e.value), (path, str(e.value))
    bad = copy.deepcopy(block)
    bad["device"] = "gpu"
    with pytest.raises(RuntimeError):
        voice_config.validate(bad)


@pytest.mark.parametrize("lang, written, said", [
    ("es", "La holgura entre floor#75 y wall#110 es 0.0018 m (1.8 mm).",
     "La holgura entre floor 75 y wall 110 es 0,0018 metros, 1,8 milímetros."),
    ("es", "wall#110: desplome de su cara 0.064° (1.1 mm/m).",
     "wall 110: desplome de su cara 0,064 grados, 1,1 milímetros por metro."),
    ("es", "Hay 3 objetos «sign».", "Hay 3 objetos sign."),
    ("es", "El volumen es 1 m³ y la posición -3.9, 1.489, -0.809.",
     "El volumen es 1 metro cúbico y la posición menos 3,9, 1,489, menos 0,809."),
    ("en", "fire_extinguisher#425 measures 0.25 × 0.6 × 0.2 m (width × height × depth).",
     "fire extinguisher 425 measures 0.25 by 0.6 by 0.2 metres, width by height by depth."),
    ("en", "The distance is 1 m.", "The distance is 1 metre."),
    ("fr", "Hauteur libre sous ceiling#141 : minimum 1.66 m, maximum 2.07 m sur 10 stations.",
     "Hauteur libre sous ceiling 141 : minimum 1,66 mètres, maximum 2,07 mètres sur 10 stations."),
    ("fr", "Le volume est 2.5 m³.", "Le volume est 2,5 mètres cubes."),
])
def test_a_written_answer_as_it_is_said(lang, written, said):
    assert spoken.say(written, lang) == said


def test_what_is_said_keeps_every_figure_of_what_is_written():
    """Only the FORM changes: the digits said are the digits written (the tool's)."""
    import re
    for lang in spoken.LANGS:
        for text in ("12.345 m y 0.5 mm, 27 objetos, 89.058° (60810.7 mm/m), 3 × 2.25 m², -0.22 m",
                     "sign#797 fire_extinguisher#425 1.05 m"):
            digits = lambda t: re.findall(r"\d+", t)       # noqa: E731
            assert digits(spoken.say(text, lang)) == digits(text), (lang, text)
    assert spoken.say("", "es") == "" and spoken.say("hola", "xx") == "hola"


def test_the_recognisers_vocabulary_is_the_systems_name_and_the_scans_labels(tmp_path):
    from phase_r.instance_store import InstanceStore
    cfg = voice_config.load()
    assert voice_api.vocabulary(None, cfg) == "STAC, STAC Build."
    assert voice_api.vocabulary(str(tmp_path / "missing.db"), cfg) == "STAC, STAC Build."
    sp = tmp_path / "scene_r.db"
    store = InstanceStore(sp)
    for iid, label in ((1, "wall"), (2, "fire_extinguisher"), (3, "wall"), (4, "door"), (5, "wall"), (6, "door")):
        store.upsert_instance(iid, label, source="test")
    store.close()
    # the most frequent first, as people say them
    assert voice_api.vocabulary(str(sp), cfg) == "STAC, STAC Build. wall, door, fire extinguisher."
    two = {**cfg, "stt": {**cfg["stt"], "prompt_max_labels": 2}}
    assert voice_api.vocabulary(str(sp), two) == "STAC, STAC Build. wall, door."
    none = {**cfg, "stt": {**cfg["stt"], "prompt_max_labels": 0}}
    assert voice_api.vocabulary(str(sp), none) == "STAC, STAC Build."


def _wav(seconds=0.1, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def test_the_service_starts_once_is_refused_beside_a_job_and_stops_verified(tmp_path, monkeypatch):
    """The launcher is a stand-in that sleeps: what is tested is the life — one process, never
    beside a job, ended (its whole group) and verified when the card is taken."""
    launcher = tmp_path / "serve_voice.sh"
    launcher.write_text(f"#!/bin/bash\nexec {sys.executable} -c 'import time; time.sleep(120)'\n")
    monkeypatch.setattr(service, "LAUNCHER", launcher)
    monkeypatch.setattr(service, "LOG", tmp_path / "voice.log")
    monkeypatch.setattr(service, "_STATE", {"proc": None, "log": None})
    monkeypatch.setattr(service, "health", lambda timeout_s=2.0: None)          # never "up"
    busy = {"why": "a pipeline is using the GPU"}
    monkeypatch.setattr(service, "_busy", lambda: busy["why"])
    said = []
    assert service.start(log=said.append) is False and service._STATE["proc"] is None
    assert "not started" in said[-1] and "pipeline" in said[-1]
    assert service.status() == {"status": "busy", "detail": busy["why"]}

    busy["why"] = None
    assert service.start(log=said.append) is True
    proc = service._STATE["proc"]
    assert proc is not None and proc.poll() is None
    assert service.start(log=said.append) is True and service._STATE["proc"] is proc      # one process
    assert service.status() == {"status": "loading"}
    assert service.stop("a job took the GPU", log=said.append, wait_s=10) is True
    assert proc.poll() is not None and service._STATE["proc"] is None
    assert service.stop("again", log=said.append) is True                                 # nothing left
    assert service.status() == {"status": "down"}


def test_the_status_of_a_service_that_answers(monkeypatch):
    monkeypatch.setattr(service, "health", lambda timeout_s=2.0: {"status": "ok", "device": "cuda", "pid": 7})
    st = service.status()
    assert st["status"] == "up" and st["device"] == "cuda"       # the service's own "ok" is not the status


def test_the_wav_the_service_answers_with_and_the_decoder_bounds():
    pytest.importorskip("av")
    import numpy as np
    from voice import serve
    data = serve.wav_bytes(np.array([0.0, 1.0, -1.0, 2.0, -2.0]))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 5)
        assert np.frombuffer(w.readframes(5), "<i2").tolist() == [0, 32767, -32767, 32767, -32767]   # clipped
    audio = serve.decode_audio(_wav(0.5), max_seconds=30)
    assert audio.dtype == np.float32 and abs(len(audio) / 16000 - 0.5) < 0.02
    with pytest.raises(serve.VoiceError) as e:
        serve.decode_audio(_wav(2.0), max_seconds=1.0)
    assert "voice.stt.max_audio_s" in str(e.value)
    with pytest.raises(serve.VoiceError):
        serve.decode_audio(b"", 30)
    with pytest.raises(serve.VoiceError):
        serve.decode_audio(b"this is not audio", 30)


@pytest.mark.parametrize("heard, addressed, command", [
    ("STAC, ¿cuántas puertas hay?", True, "cuántas puertas hay?"),
    ("Stack mostrame solo las puertas", True, "mostrame solo las puertas"),
    ("Estac, abrí la consola", True, "abrí la consola"),
    ("Stak cerrá la sesión", True, "cerrá la sesión"),
    ("Hey Stac Build, open pccr", True, "open pccr"),
    ("STAC-Build conectá con el servidor", True, "conectá con el servidor"),
    ("Dis STAC, ouvre la session", True, "ouvre la session"),
    ("OK, STAC: pantalla completa", True, "pantalla completa"),
    ("¿cuántas puertas hay, STAC?", True, "¿cuántas puertas hay"),
    ("hola STAC", True, "hola"),                       # a greeting alone is what was said to it
    ("STAC", True, ""), ("stac.", True, ""),           # only its name: it listens
    ("Hey SDAC Build, show me the doors.", True, "show me the doors."),     # as a recogniser wrote it
    ("S.T.A.C., abrí pccr", True, "abrí pccr"), ("s t a c mostrame las puertas", True, "mostrame las puertas"),
    ("stacy is here", False, "stacy is here"), ("estado de la obra", False, "estado de la obra"),
    ("mostrame las puertas", False, "mostrame las puertas"),
    ("the stacking of boxes", False, "the stacking of boxes"),
    ("estacionamiento lleno", False, "estacionamiento lleno"),
    ("un stack de cajas", False, "un stack de cajas"),          # the word inside a sentence is not a call
    ("hasta acá llegamos", False, "hasta acá llegamos"),
    ("", False, ""),
])
def test_stac_is_spoken_to_when_what_was_heard_names_it(heard, addressed, command):
    """USER 2026-11-14: "el sistema se llama stac, no hace falta llamarlo por el nombre completo
    stac build, aunque tambien se acepta"."""
    from voice import wake
    assert wake.split(heard) == (addressed, command)
    assert set(wake.ACK) == {"es", "en", "fr"}


def test_the_interfaces_names_join_the_recognisers_vocabulary():
    """A recogniser that has not read "pccr" writes "PCC R" (2026-11-14): the sessions' and
    scans' names the interface sends are added, cleaned and bounded."""
    base = "STAC, STAC Build."
    assert voice_api.with_names(base, None) == base and voice_api.with_names(base, " , ,") == base
    assert voice_api.with_names(base, "pccr,zaragoza,pccr,2026-08-31 default") == base + " pccr, zaragoza, 2026-08-31 default."
    assert voice_api.with_names(base, "<script>x</script>,a" + "b" * 80) == base + " script x script, a" + "b" * 39 + "."
    assert voice_api.with_names(base, ",".join(f"s{k}" for k in range(100))).count(",") == 1 + 39


def test_a_long_answer_is_said_up_to_its_last_whole_sentence():
    long = "Primera frase corta. " + "Segunda frase algo más larga que la primera. " + "palabra " * 80
    cut = spoken.cut_for_speech(long, 80)
    assert cut == "Primera frase corta. Segunda frase algo más larga que la primera." and len(cut) <= 80
    assert spoken.cut_for_speech("corto", 80) == "corto"
    assert spoken.cut_for_speech("palabra " * 30, 50) == ("palabra " * 6).strip()          # no sentence end: whole words
    assert len(spoken.cut_for_speech("x" * 500, 50)) == 50
