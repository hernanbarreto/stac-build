"""STAC's VOICE — speech → text and text → speech, local, on the card, in the system's three
languages (USER 2026-11-14: "incorpora herramientas para hablarle y que traduzca a texto y que
responda con voz sintetica en los 3 idiomas del sistema, ingles, español y frances").

    voice_config.py   the typed ``voice:`` block of config.yaml
    serve.py          THE SERVICE — its own process, in the `voice` env (scripts/build_voice_env.sh):
                      Whisper (speech → text, the language detected among the system's) and Kokoro
                      (text → speech) on one device, behind HTTP on 127.0.0.1
    spoken.py         a written answer as it is SAID (units, object references, decimals) per language
    service.py        the service's life, from the backend: started with it, stopped when a job
                      takes the card (it holds a CUDA context), started again when the card frees
    api.py            /api/voice/* — what the interface calls

The backend process never loads a model and never touches the card (CLAUDE.md §3): it launches
the service and talks to it over HTTP, as it does with the chat's engine.
"""
