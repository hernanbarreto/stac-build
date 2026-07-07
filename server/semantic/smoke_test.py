# STAC-Builder — Semantic Service: smoke tests.
#
#   python -m semantic.smoke_test [--backend NAME]
# Exercises the three modalities the pipeline relies on:
#   1) text-only chat
#   2) image + text (multimodal)
#   3) tool-calling with a dummy deterministic tool
# Reports pass/fail, latency, and the resting/under-load GPU budget. Ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image, ImageDraw  # noqa: E402

from semantic.client import get_semantic_client  # noqa: E402
from semantic.healthcheck import gpu_mem_used_mib  # noqa: E402
from semantic.types import system, user  # noqa: E402


def _synthetic_image() -> Image.Image:
    """A trivial scene: a red rectangle (a 'column') on grey."""
    img = Image.new("RGB", (512, 384), (150, 150, 150))
    d = ImageDraw.Draw(img)
    d.rectangle([210, 60, 300, 340], fill=(200, 40, 40))
    return img


DUMMY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_distance",
        "description": "Return the deterministic distance in meters between two named objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "first object name"},
                "b": {"type": "string", "description": "second object name"},
            },
            "required": ["a", "b"],
        },
    },
}


def _dummy_get_distance(a: str, b: str) -> dict:
    # A tool 'measures'; the VLM never does. Fixed value for the smoke test.
    return {"distance_m": 2.5, "a": a, "b": b, "unit": "m"}


def run(backend: str | None) -> int:
    client = get_semantic_client(backend=backend, consumer="phase0.smoke")
    print(f"[smoke] backend={client.backend_name} model={client.backend.model}")

    h = client.health()
    if not h["ok"]:
        print(f"[smoke] FAIL health: {h}")
        return 1
    print(f"[smoke] health OK; GPU used (MiB, resting): {gpu_mem_used_mib()}")

    results = {}

    # 1) text only
    t0 = time.time()
    r = client.chat([system("You are terse."), user("Reply with exactly the word: OK")], max_tokens=16)
    dt = (time.time() - t0) * 1000
    ok1 = r.content is not None and "ok" in (r.content or "").lower()
    results["text"] = ok1
    print(f"[smoke] 1/3 text-only  {'PASS' if ok1 else 'FAIL'} ({dt:.0f} ms) -> {r.content!r}")

    # 2) image + text
    t0 = time.time()
    r = client.chat(
        [user("What single primary color is the tall vertical object? One word.", images=[_synthetic_image()])],
        max_tokens=16,
    )
    dt = (time.time() - t0) * 1000
    ok2 = r.content is not None and "red" in (r.content or "").lower()
    results["image"] = ok2
    print(f"[smoke] 2/3 image+text {'PASS' if ok2 else 'FAIL'} ({dt:.0f} ms) -> {r.content!r}")

    # 3) tool calling
    t0 = time.time()
    out = client.run_tool_loop(
        [
            system("You measure distances ONLY via tools. Never invent numbers."),
            user("What is the distance between the column and the wall? Use the tool, then answer in meters."),
        ],
        tools=[DUMMY_TOOL],
        tool_impls={"get_distance": _dummy_get_distance},
        max_iterations=4,
    )
    dt = (time.time() - t0) * 1000
    used_tool = any(t["tool"] == "get_distance" for t in out["tool_trace"])
    ok3 = used_tool and "2.5" in (out["answer"] or "")
    results["tools"] = ok3
    print(f"[smoke] 3/3 tool-call  {'PASS' if ok3 else 'FAIL'} ({dt:.0f} ms) "
          f"[used_tool={used_tool}] -> {out['answer']!r}")

    print(f"[smoke] GPU used (MiB, after load): {gpu_mem_used_mib()}")
    allok = all(results.values())
    print(f"[smoke] RESULT: {'ALL PASS' if allok else 'FAILURES: ' + str([k for k,v in results.items() if not v])}")
    return 0 if allok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    return run(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
