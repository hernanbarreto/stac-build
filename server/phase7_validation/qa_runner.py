# STAC-Builder — Phase 7 Q&A suite runner + scorer.
#
# Runs the validation suite through the Phase 5 SpatialQA orchestrator on one or
# more backends, scores each question by its declared check, and reports accuracy
# per category, latency, and chained tool-calling failure patterns of the 8B.
#
# Geometric magnitudes are relayed by the VLM from the deterministic tools; this
# runner checks the VLM called the right tool and grounded its answer in a
# number+unit, and records the tool's own value as the reference. Questions with
# `cloudcompare_pending` still require a human CloudCompare cross-check for the
# absolute magnitude — that is the external validation step, flagged in output.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from phase5_qa.orchestrator import SpatialQA
from phase_r.instance_store import InstanceStore

_INSUFF = re.compile(
    r"insufficient|no data|not (?:available|enough|measur|record|present|include|provide)|"
    r"cannot|can't|unable|don't have|do not have|not able|no such|does not exist|"
    r"doesn't exist|do(?:es)? not (?:include|provide|support)|not (?:in|part of)|"
    r"no (?:tool|function|way|capability)|out of scope|beyond",
    re.IGNORECASE)
_NUM_UNIT = re.compile(r"-?\d+(?:\.\d+)?\s?(?:m\b|m³|m\^?3|cm\b|mm\b|deg|°|metre|meter)",
                       re.IGNORECASE)
_FINDING_WORDS = re.compile(
    r"crack|moisture|damp|wet|spalling|flak|corros|exposed steel|obstruction|"
    r"finding|defect|deterior", re.IGNORECASE)
_HEALTH_WORDS = re.compile(
    r"bimodal|double[- ]?surface|onion|separation|registration|entropy|"
    r"well[- ]aligned|misalign|ghost", re.IGNORECASE)


@dataclass
class QAResult:
    id: str
    category: str
    passed: bool
    latency_s: float
    iterations: int
    stopped: str
    tools_called: list = field(default_factory=list)
    answer: str = ""
    reason: str = ""
    cloudcompare_pending: bool = False


def _tools_in(trace) -> list[str]:
    return [t["tool"] for t in (trace or [])]


class QASuiteRunner:
    def __init__(self, store_path: str, suite_path: str | None = None):
        self.store_path = store_path
        self.suite_path = suite_path or str(Path(__file__).with_name("qa_suite.yaml"))
        self.suite = yaml.safe_load(open(self.suite_path))["questions"]
        st = InstanceStore(store_path)
        self._instances = st.list_instances()
        st.close()

    # ── expected values derived from the store (store-agnostic) ─────
    def _count_all(self) -> int:
        return len(self._instances)

    def _count_label(self, label: str) -> int:
        return sum(1 for i in self._instances if label.lower() in i["label"].lower())

    # ── scoring ─────────────────────────────────────────────────────
    def _score(self, q: dict, answer: str, trace) -> tuple[bool, str]:
        chk = q["check"]
        t = chk["type"]
        tools = _tools_in(trace)
        called = [x for x in chk.get("tools", []) if x in tools]
        if t == "count_all":
            n = self._count_all()
            ok = bool(re.search(rf"\b{n}\b", answer))
            return ok, f"expected count {n}"
        if t == "count_label":
            n = self._count_label(chk["label"])
            ok = bool(re.search(rf"\b{n}\b", answer))
            return ok, f"expected {n} {chk['label']}(s)"
        if t == "insufficient":
            ok = bool(_INSUFF.search(answer))
            return ok, "must express insufficient/no-data"
        if t == "tool_called":
            return bool(called), f"one of {chk['tools']} (got {tools})"
        if t == "measured":
            ok = bool(called) and bool(_NUM_UNIT.search(answer))
            return ok, f"measurement tool + number/unit (tools={tools})"
        if t == "findings_present":
            ok = bool(_FINDING_WORDS.search(answer))
            return ok, "must mention a finding/defect type"
        if t == "health_bimodal":
            ok = bool(called) and bool(_HEALTH_WORDS.search(answer))
            return ok, f"health tool + double-surface/separation (tools={tools})"
        return False, f"unknown check {t}"

    def run(self, backend: str = "qwen_local", on_progress=None) -> dict:
        qa = SpatialQA(self.store_path, backend=backend)
        results: list[QAResult] = []
        for i, q in enumerate(self.suite):
            t0 = time.time()
            out = qa.ask(q["question"])
            dt = time.time() - t0
            passed, reason = self._score(q, out["answer"] or "", out["tool_trace"])
            results.append(QAResult(
                id=q["id"], category=q["category"], passed=passed, latency_s=dt,
                iterations=out["iterations"], stopped=out["stopped"],
                tools_called=_tools_in(out["tool_trace"]), answer=out["answer"] or "",
                reason=reason, cloudcompare_pending=bool(q.get("cloudcompare_pending"))))
            if on_progress:
                on_progress(int(100 * (i + 1) / len(self.suite)),
                            f"{q['id']} [{q['category']}] {'PASS' if passed else 'FAIL'} "
                            f"({dt:.1f}s)")
        qa.close()
        return self._aggregate(backend, results)

    def _aggregate(self, backend: str, results: list[QAResult]) -> dict:
        cats: dict[str, list[QAResult]] = {}
        for r in results:
            cats.setdefault(r.category, []).append(r)
        by_cat = {c: {"n": len(rs), "passed": sum(r.passed for r in rs),
                      "accuracy": round(sum(r.passed for r in rs) / len(rs), 3)}
                  for c, rs in cats.items()}
        lat = sorted(r.latency_s for r in results)
        # tool-calling failure patterns
        fails = [r for r in results if not r.passed]
        no_tool = [r.id for r in results if r.category in ("measurement", "health")
                   and not r.tools_called]
        not_stopped = [r.id for r in results if r.stopped not in ("stop", "answer", "final")]
        return {
            "backend": backend,
            "n": len(results),
            "passed": sum(r.passed for r in results),
            "accuracy": round(sum(r.passed for r in results) / len(results), 3),
            "by_category": by_cat,
            "latency_s": {"mean": round(sum(lat) / len(lat), 2),
                          "p50": round(lat[len(lat) // 2], 2),
                          "p90": round(lat[int(len(lat) * 0.9)], 2),
                          "max": round(lat[-1], 2)},
            "tool_failure_patterns": {
                "failed_ids": [r.id for r in fails],
                "measurement_health_no_tool": no_tool,
                "loop_not_cleanly_stopped": not_stopped,
                "mean_iterations": round(sum(r.iterations for r in results) / len(results), 2),
            },
            "cloudcompare_pending_ids": [r.id for r in results if r.cloudcompare_pending],
            "results": [r.__dict__ for r in results],
        }
