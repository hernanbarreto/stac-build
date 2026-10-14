# STAC-Builder — Semantic Service: SEVERAL requests in flight, answers in the sequential order.
#
# USER 2026-10-09: "Mandemos la mayor cantidad que podamos" — send the engine as many requests at
# once as it can take. Until that day the VLM stages asked ONE request at a time (docs/
# plan_determinismo.md point 80: --max-num-seqs 1, and every consumer looped): pccr's 295
# per-keyframe understanding calls at ~20-27 s each were 80-100 min on an engine whose KV cache
# holds ~28 of them at once.
#
# WHY THE ANSWERS DO NOT CHANGE: the engine runs with batch invariance (VLLM_BATCH_INVARIANT=1,
# semantic.serve: attention without split-KV, the batch-invariant matmul or the cuBLAS workspace
# form) — what a request computes does not depend on what else is in the batch —, without prefix
# caching, at temperature 0 with a fixed seed. And it is MEASURED, not trusted: every pass that
# sends several at once re-asks a sample of its requests ALONE afterwards and compares the answers
# byte for byte (:func:`self_check`; the sample is the user's rule of at least
# min_judge_closures(heldout_confidence) judges — 5 at 0.95). A difference is DECLARED (log +
# census), never a failure (USER 2026-10-09: "que no falle, que lo reporte sin fallar").
#
# HOW MANY AT ONCE — derived from the engine, never chosen (:func:`concurrency_plan`):
#   capacity  the engine's KV cache in blocks: ``vllm:cache_config_info`` num_gpu_blocks (GET
#             /metrics — semantic.backends.QwenLocalBackend.engine_kv_cache) minus the ONE null
#             block vLLM 0.19 never hands to a request (v1/core/block_pool.py:175);
#   cost      an UPPER BOUND on the blocks one request can hold: ceil((its prompt tokens + its
#             max_tokens) / block_size) — the scheduler never schedules a request past its prompt
#             plus its max_tokens (v1/core/sched/scheduler.py:387-401) and, without prefix
#             caching, no block is shared. The prompt tokens are MEASURED (the first call's
#             usage.prompt_tokens: exact for every call with the same text and the same image
#             size) or bounded from above (the UTF-8 bytes of a text-only prompt — a byte-level
#             BPE never has more tokens than bytes; the served context for a request nothing
#             measured: the engine accepts no request longer than it);
#   requests are admitted IN THE SEQUENTIAL ORDER while the costs in flight fit the capacity, so
#   no request is ever PREEMPTED (a preempted request is recomputed: the one engine path whose
#   answers nobody compared); never more threads than requests.
# No KV record (a client without /metrics — the tests' fakes —, a gauge that cannot be read): ONE
# request at a time, as before, declared in the log and the census.
#
# ORDER AND FAILURES (:func:`run_ordered`): results are handed to the caller in the order of the
# sequential loop, so every record, census entry, log line, aggregation and merge decision is
# built exactly as before. A call that fails fails the pass (point 88) with the error of the FIRST
# failing request in that order: once one failed nothing new is admitted, every request already in
# flight finishes (one HTTP call cannot be withdrawn from the engine), the results before the
# failure are delivered as the loop delivered them, and that error is raised — the same object,
# the same type.
#
# THREAD SAFETY: the threads only run the callers' request functions (semantic.client.
# SemanticClient.chat: one requests.post per call, nothing shared but the consumer name, which
# every pass hands in explicitly; semantic.call_log appends ONE JSON line per call under a lock).
# Everything else — the generator preparing the next request, the delivery, the logs — runs on the
# caller's thread.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# vLLM 0.19 keeps ONE block of its KV cache as the null block (v1/core/block_pool.py:175: popped
# from the free queue at start, never allocated): what requests can hold is num_gpu_blocks minus it
ENGINE_NULL_BLOCKS = 1
# the consumer suffix of a self-check call (logs/semantic_calls.jsonl tells a re-asked answer from
# the pass's own)
SELF_CHECK_SUFFIX = ".selfcheck"
# characters of each answer a declared mismatch shows on both sides of the first difference — a
# display width for the census, it decides nothing
DIFF_CONTEXT_CHARS = 60
# the section the self-check's sample size is read from (the user's rule of judges)
JUDGES_KEY = "correction_graph.graph.heldout_confidence"


class Cancelled(RuntimeError):
    """The pass was cancelled by its caller; ``delivered`` results had been handed over."""

    def __init__(self, delivered: int):
        super().__init__(f"cancelled after {delivered} result(s)")
        self.delivered = int(delivered)


@dataclass
class Job:
    """One item of an ordered pass: a request to run (``fn``, holding up to ``cost`` KV blocks
    while in flight) or, with ``fn`` None, an item that needs no engine (``value`` is its
    result — an object without a view, a caption kept by its stamp): it is delivered in its
    place in the order without taking a thread or any budget."""
    fn: Optional[Callable[[], Any]] = None
    cost: int = 0
    value: Any = None


class AnswerCapture:
    """A view of a semantic client that records the TEXT of every answer it returns, in call
    order (the self-check compares them), and — with ``consumer_suffix`` — names its calls apart
    in logs/semantic_calls.jsonl. One per request: nothing in it is shared between threads."""

    def __init__(self, client, consumer_suffix: Optional[str] = None):
        self.client = client
        self.consumer_suffix = consumer_suffix
        self.contents: List[Optional[str]] = []

    def chat(self, messages, *args, **kwargs):
        if self.consumer_suffix and kwargs.get("consumer"):
            kwargs["consumer"] = f"{kwargs['consumer']}{self.consumer_suffix}"
        resp = self.client.chat(messages, *args, **kwargs)
        self.contents.append(getattr(resp, "content", None))
        return resp

    @property
    def last(self) -> Optional[str]:
        return self.contents[-1] if self.contents else None


# ── how many at once ──────────────────────────────────────────────────────────────────────

def blocks(tokens: int, block_size: int) -> int:
    """KV blocks ``tokens`` occupy (ceil): what one request holds at most."""
    return -(-int(tokens) // int(block_size))


def engine_kv_cache(client, log: Callable[[str], Any] = print) -> Optional[Dict[str, Any]]:
    """The engine's KV cache as the client reads it (``client.engine_kv_cache()`` — the
    ``vllm:cache_config_info`` gauge of /metrics): ``{num_gpu_blocks, block_size, source}``, or
    None — DECLARED, and the pass then sends one request at a time — when the client exposes no
    such record (a test's fake) or it cannot be read."""
    getter = getattr(client, "engine_kv_cache", None)
    if not callable(getter):
        log("[concurrency] DECLARED: the client exposes no record of the engine's KV cache — "
            "one request at a time (nothing to derive a bound from)")
        return None
    try:
        kv = getter()
        n, bs = int(kv["num_gpu_blocks"]), int(kv["block_size"])
    except Exception as e:  # noqa: BLE001 — declared; one at a time is always safe
        log(f"[concurrency] DECLARED: the engine's KV cache could not be read "
            f"({type(e).__name__}: {e}) — one request at a time")
        return None
    if n <= ENGINE_NULL_BLOCKS or bs < 1:
        log(f"[concurrency] DECLARED: the engine reports {n} KV block(s) of {bs} token(s) — "
            f"nothing to derive a bound from; one request at a time")
        return None
    return {"num_gpu_blocks": n, "block_size": bs, "source": str(kv.get("source") or "")}


def concurrency_plan(kv: Optional[Mapping[str, Any]], footprints: Sequence[int], *,
                     n_requests: int, footprint_source: str) -> Dict[str, Any]:
    """How many requests of one pass go to the engine at once (see the module header): the
    threads = min(requests, capacity // the smallest cost), at least 1; the costs themselves are
    enforced per request by :func:`run_ordered` (a pass whose requests differ in size never
    holds more than the capacity). ``footprints``: an upper bound, in tokens, of every request's
    (or every distinct kind of request's) prompt + max_tokens; ``footprint_source`` says where
    they come from. Without ``kv`` (or with one request or none): one at a time, declared."""
    n = int(n_requests)
    rec: Dict[str, Any] = {
        "rule": ("admitted in the sequential order while the in-flight requests' upper-bound KV "
                 "blocks (ceil((prompt + max_tokens) / block_size)) fit the engine's KV cache "
                 "(num_gpu_blocks - the null block) — no request is ever preempted; never more "
                 "threads than requests (USER 2026-10-09: 'Mandemos la mayor cantidad que "
                 "podamos')"),
        "n_requests": n, "footprint_source": footprint_source,
    }
    fp = [int(t) for t in footprints if t is not None]
    if kv is None or n <= 1 or not fp:
        why = ("no record of the engine's KV cache" if kv is None else
               "one request or none" if n <= 1 else "no footprint bound for the requests")
        rec.update(threads=1, capacity_blocks=None, one_at_a_time=why)
        return rec
    cap = int(kv["num_gpu_blocks"]) - ENGINE_NULL_BLOCKS
    bs = int(kv["block_size"])
    costs = [blocks(t, bs) for t in fp]
    rec.update(kv_cache={"num_gpu_blocks": int(kv["num_gpu_blocks"]), "block_size": bs,
                         "source": kv.get("source")},
               capacity_blocks=cap,
               footprint_tokens={"min": min(fp), "max": max(fp)},
               cost_blocks={"min": min(costs), "max": max(costs)},
               threads=max(1, min(n, cap // min(costs))),
               bound_at_max_cost=max(1, min(n, cap // max(costs))),
               one_at_a_time=None)
    return rec


def cost_of(plan: Mapping[str, Any], footprint_tokens: Optional[int]) -> int:
    """The KV blocks one request may hold under ``plan``: its footprint in blocks, or the WHOLE
    capacity when nothing bounds it (it then runs alone); 0 when the pass runs one at a time."""
    cap = plan.get("capacity_blocks")
    if cap is None:
        return 0
    if footprint_tokens is None:
        return int(cap)
    return blocks(int(footprint_tokens), int(plan["kv_cache"]["block_size"]))


# ── the ordered pass ──────────────────────────────────────────────────────────────────────

def run_ordered(jobs: Iterable[Job], *, capacity: Optional[int], threads: int,
                on_result: Optional[Callable[[int, Any], None]] = None,
                cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Run ``jobs`` with up to ``threads`` requests in flight, admitted IN ORDER while the
    in-flight costs fit ``capacity`` (None: no budget, only the thread bound; a job costlier
    than the capacity runs alone), and hand every result to ``on_result(index, value)`` in the
    order of ``jobs`` — on the caller's thread. ``jobs`` is consumed lazily, on the caller's
    thread, one item ahead of what is admitted (its preparation — files read, crops encoded —
    stays out of the worker threads and its memory bounded by what is in flight); an exception
    raised while preparing an item is that item's failure, in its place in the order.

    Failure: nothing new is admitted, every request in flight finishes, the results BEFORE the
    first failing item (in order) are delivered and that item's exception is raised (the same
    object). ``cancelled`` (polled whenever a request finishes): the same stop, then
    :class:`Cancelled`. An exception of ``on_result`` stops the pass the same way and propagates.
    Returns ``{threads, n_items, n_calls, max_in_flight, max_blocks_in_flight}``."""
    width = max(1, int(threads))
    cap = None if capacity is None else int(capacity)
    slots: List[list] = []                 # [kind, payload, cost, released] per item, in order
    it = iter(jobs)
    head: Optional[Job] = None
    exhausted = False
    in_flight = used = delivered = 0
    stopped: Optional[str] = None          # "failed" | "cancelled"
    stats = {"threads": width, "n_items": 0, "n_calls": 0, "max_in_flight": 0,
             "max_blocks_in_flight": 0}
    ex = ThreadPoolExecutor(max_workers=width, thread_name_prefix="semantic-ordered")
    try:
        while True:
            # 1. every finished request gives its budget back; a failed one stops the admission
            for s in slots:
                if s[0] == "call" and not s[3] and s[1].done():
                    s[3] = True
                    in_flight -= 1
                    used -= s[2]
                    if s[1].exception() is not None:
                        stopped = stopped or "failed"
            # 2. admission, in order, while the threads and the budget allow
            while stopped is None and not exhausted:
                if head is None:
                    try:
                        head = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    except Exception as e:  # noqa: BLE001 — that item's failure, in its place
                        slots.append(["error", e, 0, True])
                        stopped = "failed"
                        break
                if head.fn is None:
                    slots.append(["value", head.value, 0, True])
                    head = None
                    continue
                cost = max(0, int(head.cost))
                if in_flight >= width or (in_flight and cap is not None and used + cost > cap):
                    break
                slots.append(["call", ex.submit(head.fn), cost, False])
                head = None
                in_flight += 1
                used += cost
                stats["n_calls"] += 1
                stats["max_in_flight"] = max(stats["max_in_flight"], in_flight)
                stats["max_blocks_in_flight"] = max(stats["max_blocks_in_flight"], used)
            # 3. delivery, in order
            while stopped is None and delivered < len(slots):
                kind, payload = slots[delivered][0], slots[delivered][1]
                if kind == "call":
                    if not payload.done():
                        break
                    if payload.exception() is not None:
                        stopped = "failed"
                        break
                    value = payload.result()
                else:
                    value = payload
                if on_result is not None:
                    on_result(delivered, value)
                delivered += 1
            if stopped is not None:
                break
            if exhausted and delivered == len(slots):
                break
            if cancelled is not None and cancelled():
                stopped = "cancelled"
                break
            running = [s[1] for s in slots if s[0] == "call" and not s[3]]
            if running:
                wait(running, return_when=FIRST_COMPLETED)
    except BaseException:
        # the caller's own code failed (on_result, cancelled): what is in flight finishes,
        # nothing new starts, the exception goes on
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)                 # every request in flight finishes
    stats["n_items"] = len(slots)
    if stopped == "cancelled":
        raise Cancelled(delivered)
    if stopped == "failed":
        # the FIRST failure in the sequential order: every request before it was admitted before
        # it and has finished; their results are delivered first, as the sequential loop did
        for i in range(delivered, len(slots)):
            kind, payload = slots[i][0], slots[i][1]
            err = payload if kind == "error" else (payload.exception() if kind == "call" else None)
            if err is not None:
                raise err
            if on_result is not None:
                on_result(i, payload.result() if kind == "call" else payload)
        raise RuntimeError("run_ordered: stopped on a failure that no item carries")  # unreachable
    return stats


# ── the self-check: a sample re-asked ALONE, compared byte for byte ───────────────────────

def _vendor_on_path() -> None:
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def self_check_judges(config: Optional[Mapping[str, Any]]) -> Tuple[Optional[int], Dict[str, Any]]:
    """How many requests the self-check re-asks: the user's rule of judges —
    ``min_judge_closures(correction_graph.graph.heldout_confidence)`` (loop_utils.loop_judge,
    the count every decide_change of the pipeline requires: 5 at the declared 0.95). Returns
    ``(n, record)``; ``(None, record with 'error')`` when the configuration does not carry the
    key — the self-check is then declared not run (it never fails a pass)."""
    gp = (((config or {}).get("correction_graph") or {}).get("graph")
          if isinstance(config, Mapping) else None)
    conf = gp.get("heldout_confidence") if isinstance(gp, Mapping) else None
    rule = (f"min_judge_closures({JUDGES_KEY}) — the user's rule of at least that many judges "
            f"(metric_lock.decide_change)")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 < float(conf) < 1.0:
        return None, {"rule": rule, "error": f"'{JUDGES_KEY}' is {conf!r} in the configuration — "
                                             f"the sample size is derived from it"}
    _vendor_on_path()
    from loop_utils.loop_judge import min_judge_closures
    n = int(min_judge_closures(float(conf)))
    return n, {"rule": rule, "confidence": float(conf), "n": n}


def sample_positions(candidates: Sequence[int], n: int) -> List[int]:
    """``n`` of ``candidates`` spread evenly over their order, both ends in (integer positions:
    deterministic); all of them when there are not more than ``n``."""
    c = list(candidates)
    if n <= 0 or not c:
        return []
    if len(c) <= n:
        return c
    if n == 1:
        return [c[0]]
    return [c[(j * (len(c) - 1)) // (n - 1)] for j in range(n)]


def _sha256(text: Optional[str]) -> Optional[str]:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_diff(concurrent: Optional[str], alone: Optional[str]) -> Dict[str, Any]:
    """A short account of two answers that differ: where the first character differs, both
    lengths in UTF-8 bytes, their sha256 and :data:`DIFF_CONTEXT_CHARS` characters of each
    around the difference."""
    a, b = concurrent or "", alone or ""
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    lo, hi = max(0, k - DIFF_CONTEXT_CHARS), k + DIFF_CONTEXT_CHARS
    return {"first_difference_at": k,
            "concurrent_bytes": len(a.encode("utf-8")), "alone_bytes": len(b.encode("utf-8")),
            "concurrent_sha256": _sha256(concurrent), "alone_sha256": _sha256(alone),
            "concurrent_excerpt": a[lo:hi], "alone_excerpt": b[lo:hi]}


def self_check(sample: Sequence[int], concurrent: Mapping[int, Optional[str]],
               ask_alone: Callable[[int], Optional[str]], *,
               describe: Callable[[int], Dict[str, Any]], judges: Mapping[str, Any],
               what: str, log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Re-ask every request of ``sample`` ALONE (``ask_alone(i)`` -> the answer's text; called
    one after another, with nothing else in flight) and compare it BYTE FOR BYTE with the answer
    the concurrent pass got (``concurrent[i]``). Never raises: a mismatch, and a re-asked call
    that fails, are DECLARED (log line + the returned record) — USER 2026-10-09: "que no falle,
    que lo reporte sin fallar". ``describe(i)`` names the request in the record."""
    rec: Dict[str, Any] = {"what": what, "judges": dict(judges), "n_sample": len(sample),
                           "sample": [describe(i) for i in sample], "n_identical": 0,
                           "mismatches": [], "errors": []}
    for i in sample:
        try:
            alone = ask_alone(i)
        except Exception as e:  # noqa: BLE001 — declared, the pass goes on
            rec["errors"].append(dict(describe(i), error=f"{type(e).__name__}: {e}"))
            continue
        if alone == concurrent.get(i):
            rec["n_identical"] += 1
        else:
            rec["mismatches"].append(dict(describe(i), **answer_diff(concurrent.get(i), alone)))
    rec["verdict"] = ("MISMATCH" if rec["mismatches"] else
                      "incomplete" if rec["errors"] else "identical")
    if rec["verdict"] == "identical":
        log(f"[concurrency] self-check ({what}): {rec['n_identical']} of {len(sample)} request(s) "
            f"re-asked ALONE answered byte for byte as in the concurrent pass")
    else:
        names = [{k: v for k, v in m.items() if k in ("call", "frame", "tile", "phrase",
                                                       "instance_id", "first_difference_at")}
                 for m in rec["mismatches"]]
        log(f"[concurrency] ⚠ DECLARED self-check ({what}): {rec['n_identical']} of {len(sample)} "
            f"re-asked alone identical, {len(rec['mismatches'])} MISMATCH(ES)"
            + (f" {names}" if names else "")
            + (f", {len(rec['errors'])} re-ask(s) failed: {rec['errors']}" if rec["errors"]
               else "")
            + " — the pass goes on (USER 2026-10-09: reported, never a failure)")
    return rec


def not_checked(reason: str, judges: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The self-check record of a pass that did not run it (one request at a time — nothing to
    compare —, no sample size in the configuration, no request)."""
    return {"verdict": "not run", "reason": reason, "judges": (dict(judges) if judges else None),
            "n_sample": 0, "n_identical": 0, "mismatches": [], "errors": []}


def plan_line(plan: Mapping[str, Any], stats: Optional[Mapping[str, Any]] = None) -> str:
    """One log line naming the bound of a pass and how it was derived."""
    if plan.get("capacity_blocks") is None:
        return (f"[concurrency] {plan.get('n_requests')} request(s), ONE at a time "
                f"({plan.get('one_at_a_time')})")
    kv = plan["kv_cache"]
    line = (f"[concurrency] {plan['n_requests']} request(s), up to {plan['threads']} in flight: "
            f"engine KV cache {kv['num_gpu_blocks']} block(s) of {kv['block_size']} tokens "
            f"({kv.get('source')}) − {ENGINE_NULL_BLOCKS} null = {plan['capacity_blocks']} block(s); "
            f"one request ≤ {plan['footprint_tokens']['max']} tokens = "
            f"{plan['cost_blocks']['max']} block(s) ({plan['footprint_source']})")
    if stats:
        line += (f"; at most {stats.get('max_in_flight')} were in flight "
                 f"({stats.get('max_blocks_in_flight')} block(s))")
    return line
