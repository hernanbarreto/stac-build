# STAC-Builder — Phase 7 validation CLI.
#   Q&A suite:     python -m phase7_validation.cli qa --scene <db> [--backend ...] [--out f.json]
#   Coexistence:   python -m phase7_validation.cli coexist [--out f.json]
# Hernán Barreto - Ingerop IN3 Session IV - STAC
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def _qa(a) -> int:
    from phase7_validation.qa_runner import QASuiteRunner
    runner = QASuiteRunner(a.scene, suite_path=a.suite)
    backends = a.backend.split(",")
    allrep = {}
    for b in backends:
        rep = runner.run(backend=b.strip(),
                         on_progress=lambda p, m: print(f"[qa:{b} {p:3d}%] {m}", flush=True))
        allrep[b.strip()] = rep
        print(f"\n=== {b} ===  accuracy={rep['accuracy']}  "
              f"latency(mean/p90)={rep['latency_s']['mean']}/{rep['latency_s']['p90']}s")
        for c, s in sorted(rep["by_category"].items()):
            print(f"  {c:12s} {s['passed']}/{s['n']}  acc={s['accuracy']}")
        tf = rep["tool_failure_patterns"]
        print(f"  failed: {tf['failed_ids']}")
        print(f"  no-tool(meas/health): {tf['measurement_health_no_tool']}  "
              f"mean_iters={tf['mean_iterations']}")
    if a.out:
        Path(a.out).write_text(json.dumps(allrep, indent=2))
        print(f"\nwrote {a.out}")
    return 0


def _coexist(a) -> int:
    from phase7_validation.coexistence import probe
    rep = probe()
    print(json.dumps(rep, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 7 integral validation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("qa", help="run the spatial Q&A suite")
    p.add_argument("--scene", required=True)
    p.add_argument("--suite", default=None)
    p.add_argument("--backend", default="qwen_local", help="comma-separated backends")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_qa)
    c = sub.add_parser("coexist", help="probe GPU coexistence (vLLM + reconstruction)")
    c.add_argument("--out", default=None)
    c.set_defaults(fn=_coexist)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
