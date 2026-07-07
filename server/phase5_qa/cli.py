# STAC-Builder — Phase 5: spatial Q&A CLI.
#
#   conda activate da3 && cd server
#   python -m phase5_qa.cli --scene <store.db> --backend qwen_local "¿pregunta?"
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from phase5_qa.orchestrator import SpatialQA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Spatial Q&A over the Phase R instance store")
    ap.add_argument("--scene", required=True, help="path to the scene instance store (.db)")
    ap.add_argument("--backend", default="qwen_local")
    ap.add_argument("--image", action="append", default=None, help="keyframe(s) to attach")
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="print full JSON (with tool trace)")
    ap.add_argument("question")
    args = ap.parse_args()

    qa = SpatialQA(args.scene, backend=args.backend)
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in (args.image or [])]
    res = qa.ask(args.question, images=imgs or None, max_iterations=args.max_iterations)
    qa.close()

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\nQ: {res['question']}\nA: {res['answer']}\n")
        if res["tool_trace"]:
            print("traces:")
            for t in res["tool_trace"]:
                print(f"  {t['tool']}({t['arguments']}) -> {t['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
