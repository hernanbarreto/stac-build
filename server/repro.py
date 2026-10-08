"""Reproducibility primitives shared by every stage (docs/plan_determinismo.md, 2026-10-07).

The plan's goal: two runs Omega → F6 over the same session give products identical BYTE FOR
BYTE. Every stage builds on the same few mechanisms, defined ONCE here:

- :func:`deterministic_torch` / :func:`enable_deterministic_torch` — torch with deterministic
  algorithms STRICT (an op without a deterministic kernel raises), cuDNN deterministic and not
  benchmarking, TF32 off for cuDNN and matmul, seeds fixed, the cuBLAS workspace pinned.
  (``precision.tracks.deterministic_torch`` was the first version of this; it is meant to become a
  re-export of this one.)
- :func:`require_exclusive_gpu` — no other process on the card before a GPU step: FAIL listing
  what is there, never degrade (point 4: another process on the card made Omega lower its
  resolution and DA3 split its windows).
- :func:`card_identity` — the card read through torch (plus its board memory from nvidia-smi,
  matched by uuid), never an 'unknown' sentinel (point 13); :func:`card_key` — its MODEL key
  (name | board MiB | compute capability), what per-card tables and reuse stamps compare.
- :func:`environment_record` — card, driver, torch / CUDA / cuDNN, BLAS core, CPU model, library
  versions, git state of the repo and of every vendored fork (points 12, 27, 37, 38, 54).
- :func:`stamp` / :func:`check_stamp` — sha256 of a step's inputs, code and config, and the list
  of what differs (points 8, 9, 10, 21, 22, 23, 31, 34, 44, 51).
- :func:`stable_id` — identifiers derived from their inputs, replacing random ids (36, 56).
- :func:`exact_float` / :func:`write_poses_exact` / :func:`write_intrinsics_exact` — float64 text
  that round-trips exactly (point 45: '{:.8g}' poses re-read by every stage rounded them; F5 run
  twice on the 'same' F2 poses gave a different initial cost, 3.2625 vs 3.2395 px).

Nothing here reads a wall clock or a random source. Light at import (numpy only): torch and
threadpoolctl are imported inside the functions that need them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_ROOT = REPO_ROOT / "vendor" / "VGGT-Long"

PathLike = Union[str, os.PathLike]


class ReproError(RuntimeError):
    """A reproducibility precondition does not hold (the card is shared, the card cannot be
    identified, a deterministic setting could not be applied ...). Never caught to degrade."""


# ── deterministic torch ─────────────────────────────────────────────────────────────────────

CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
# cuBLAS is deterministic only with a fixed workspace (NVIDIA cuBLAS docs, "Results
# reproducibility"; torch refuses cuBLAS calls under use_deterministic_algorithms without it).
# ONE value for every run — the one every launcher of this repo already sets (extract_da3_depth,
# intake/vram, intake/focal, intake/walk, precision/runner): another valid value could pick other
# kernels, so a different one is refused, not accepted.
CUBLAS_WORKSPACE_VALUE = ":4096:8"
# environment every GPU/CPU step subprocess is launched with: the cuBLAS workspace (above) and
# Python's string hashing fixed (set iteration order of str is part of the result otherwise; it
# can only be set BEFORE the interpreter starts — precision/runner already sets it to 0)
DETERMINISTIC_ENV: Dict[str, str] = {CUBLAS_WORKSPACE_ENV: CUBLAS_WORKSPACE_VALUE,
                                     "PYTHONHASHSEED": "0"}


def deterministic_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """A copy of ``base`` (default: os.environ) with :data:`DETERMINISTIC_ENV` set — the env of a
    step subprocess. A value already set to something else is REFUSED (two launchers disagreeing
    on it is exactly the variance this removes)."""
    env = dict(os.environ if base is None else base)
    for k, v in DETERMINISTIC_ENV.items():
        if k in env and env[k] != v:
            raise ReproError(f"{k}={env[k]!r} in the launch environment — every run uses {v!r}")
        env[k] = v
    return env


def ensure_cublas_workspace() -> None:
    """Pin CUBLAS_WORKSPACE_CONFIG in this process. It must be in the environment BEFORE torch
    initialises CUDA: once CUDA is up it cannot change any more and a wrong / missing value FAILS."""
    cur = os.environ.get(CUBLAS_WORKSPACE_ENV)
    if cur == CUBLAS_WORKSPACE_VALUE:
        return
    if cur is not None:
        raise ReproError(f"{CUBLAS_WORKSPACE_ENV}={cur!r} — every run uses "
                         f"{CUBLAS_WORKSPACE_VALUE!r}; launch with repro.deterministic_env()")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise ReproError(f"{CUBLAS_WORKSPACE_ENV} is unset and CUDA is already initialised — cuBLAS "
                         f"cannot be made deterministic now; set it before torch touches the GPU "
                         f"(launch with repro.deterministic_env())")
    os.environ[CUBLAS_WORKSPACE_ENV] = CUBLAS_WORKSPACE_VALUE


def torch_numerics_record() -> Dict[str, Any]:
    """The numerics torch runs with NOW (the legacy TF32 flags are the ones this module sets; a
    library that set the new fp32_precision API on cuDNN makes the legacy read raise — that
    mixed state is not certifiable and the error propagates)."""
    import torch
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "matmul_allow_bf16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "matmul_allow_fp16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "sdp_flash": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdp_mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdp_math": bool(torch.backends.cuda.math_sdp_enabled()),
        CUBLAS_WORKSPACE_ENV: os.environ.get(CUBLAS_WORKSPACE_ENV),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def _torch_state(torch) -> tuple:
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)


def _restore_torch_state(torch, prev: tuple) -> None:
    torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
    (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
     torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = prev[2:]


def _apply_deterministic_torch(seed: int) -> Dict[str, Any]:
    import random

    import torch
    if isinstance(seed, bool) or int(seed) != seed:
        raise ReproError(f"deterministic_torch: seed {seed!r} must be an integer")
    ensure_cublas_workspace()
    torch.use_deterministic_algorithms(True, warn_only=False)        # STRICT: raise, never warn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(int(seed))                                      # CPU and every CUDA device
    np.random.seed(int(seed) % (2 ** 32))
    random.seed(int(seed))
    rec = torch_numerics_record()
    want = {"deterministic_algorithms": True, "deterministic_warn_only": False,
            "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_allow_tf32": False,
            "matmul_allow_tf32": False}
    bad = {k: rec[k] for k, v in want.items() if rec[k] != v}
    if bad:
        raise ReproError(f"deterministic_torch: torch did not take the settings {bad}")
    rec["seed"] = int(seed)
    return rec


@contextlib.contextmanager
def deterministic_torch(seed: int):
    """torch deterministic STRICT inside the block (an op with no deterministic kernel RAISES),
    cuDNN deterministic + no benchmark, TF32 off (cuDNN and matmul), torch / numpy / random seeded,
    cuBLAS workspace pinned. Yields the numerics record (:func:`torch_numerics_record` + seed). The
    torch flags are restored on exit; the seeds are not (re-seed to repeat)."""
    import torch
    prev = _torch_state(torch)
    try:
        rec = _apply_deterministic_torch(seed)
        yield rec
    finally:
        _restore_torch_state(torch, prev)


def enable_deterministic_torch(seed: int) -> Dict[str, Any]:
    """The same settings as :func:`deterministic_torch` for the WHOLE process (a GPU script calls
    it first thing, before loading a model); returns the numerics record for its report."""
    return _apply_deterministic_torch(seed)


# ── the card ────────────────────────────────────────────────────────────────────────────────

# hang guard of one nvidia-smi / torch probe subprocess: past it the probe FAILS (a BOUND, not a
# decision — an answer that does not come is never replaced by a guess; point 13 was a 10 s
# timeout turning the card into 'unknown')
PROBE_TIMEOUT_S = 300

_CARD_PROBE = r"""
import json, sys
import torch
if not torch.cuda.is_available():
    sys.stderr.write("torch sees no CUDA device\n"); sys.exit(3)
i = int(sys.argv[1])
p = torch.cuda.get_device_properties(i)
print(json.dumps({"name": p.name, "total_memory_bytes": int(p.total_memory),
                  "capability": f"{p.major}.{p.minor}", "uuid": str(p.uuid),
                  "multi_processor_count": int(p.multi_processor_count),
                  "l2_cache_bytes": int(p.L2_cache_size)}))
"""


def _card_probe_inprocess(device: int) -> Dict[str, Any]:
    import torch
    p = torch.cuda.get_device_properties(int(device))
    return {"name": p.name, "total_memory_bytes": int(p.total_memory),
            "capability": f"{p.major}.{p.minor}", "uuid": str(p.uuid),
            "multi_processor_count": int(p.multi_processor_count),
            "l2_cache_bytes": int(p.L2_cache_size)}


def _card_probe_subprocess(device: int) -> Dict[str, Any]:
    try:
        r = subprocess.run([sys.executable, "-c", _CARD_PROBE, str(int(device))],
                           capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise ReproError(f"card_identity: the torch probe did not answer in {PROBE_TIMEOUT_S} s") from e
    if r.returncode != 0:
        raise ReproError(f"card_identity: the torch probe failed (exit {r.returncode}): "
                         f"{(r.stderr or r.stdout).strip()[-800:]}")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        raise ReproError(f"card_identity: unreadable probe answer {r.stdout!r}") from e


_CARD_KEY_SEP = " | "                    # card_key's field separator (parse_card_key splits on it)


def card_key(identity: Mapping[str, Any]) -> str:
    """The card MODEL as one string — what a per-card table (Omega footprint, DA3 window size) is
    keyed by, what 'the card changed' compares and what a reuse stamp carries:
    ``'<name> | <board MiB> MiB | sm_<capability>'``, the board memory being the card's
    ``memory_total_mib`` (nvidia-smi's memory.total, read by :func:`card_identity` for the same
    card, matched by uuid).

    NOT torch's ``total_memory``: that is the memory USABLE after the driver's reservation, not a
    property of the model. Measured 2026-10-07 on the A100 80GB PCIe (nvidia-smi memory.total
    81920 MiB, driver 595.91.07): torch read 85097971712 B when server/card_table.json was written
    (~06:21) and 85094825984 B — 3 MiB less, in all four conda envs — after the pod restart of
    ~07:17; keyed on it, the committed table no longer knew its own card and the first validation
    run (pccr, logs/server_20261007_071903.log) died in the focal probe. NOT the name alone: one
    name covers boards of different memory (the GeForce RTX 3060 ships with 8 GB and 12 GB). NOT
    the instance (uuid): two cards of one model measure alike. The uuid and torch's reading stay
    in the identity and in the environment records as recorded facts — never in this key."""
    try:
        name = str(identity["name"]).strip()
        mib = int(identity["memory_total_mib"])
        cap = str(identity["capability"]).strip()
    except (KeyError, TypeError, ValueError) as e:
        raise ReproError(f"card_key: the identity carries no name / memory_total_mib / capability "
                         f"({e!r}) — read the card with repro.card_identity") from e
    if not name or mib <= 0 or not cap or _CARD_KEY_SEP in name or name.lower() == "unknown":
        raise ReproError(f"card_key: incomplete card identity (name {name!r}, board memory {mib} "
                         f"MiB, capability {cap!r})")
    return f"{name}{_CARD_KEY_SEP}{mib} MiB{_CARD_KEY_SEP}sm_{cap}"


def parse_card_key(key: str) -> Dict[str, Any]:
    """The inverse of :func:`card_key`: ``{name, memory_total_mib, capability}``. A string not in
    card_key's format RAISES — a key written by older code (torch's bytes: '<name> | <bytes> B |
    sm_<cap>') included: it names no card model this code can size."""
    parts = str(key).rsplit(_CARD_KEY_SEP, 2)
    if len(parts) == 3 and parts[1].endswith(" MiB") and parts[2].startswith("sm_"):
        name, mib, cap = parts[0].strip(), parts[1][:-len(" MiB")], parts[2][len("sm_"):]
        if name and mib.isdigit() and int(mib) > 0 and cap:
            rec = {"name": name, "memory_total_mib": int(mib), "capability": cap}
            if card_key(rec) == str(key):
                return rec
    raise ReproError(f"{key!r} is not a card key ('<name> | <board MiB> MiB | sm_<capability>', "
                     f"repro.card_key)")


def card_identity(device: int = 0) -> Dict[str, Any]:
    """The card ``device`` (torch's index under this process's CUDA_VISIBLE_DEVICES) as torch
    reads it — name, usable memory (``total_memory_bytes``), compute capability, uuid ('GPU-…'),
    SM count, L2 size — plus its BOARD memory ``memory_total_mib`` (nvidia-smi's memory.total of
    the SAME card, matched by uuid) and :func:`card_key`. torch is read IN this process when it
    already initialised CUDA, otherwise in a short torch subprocess — so a launcher never keeps a
    CUDA context (and its memory) on the card it is about to hand to a step (the probe exits
    before this returns; a launcher calls :func:`require_exclusive_gpu` first, then this). Any
    failure RAISES — torch, nvidia-smi, or nvidia-smi not listing that uuid: there is no 'unknown'
    card (point 13)."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        ident = _card_probe_inprocess(device)
    else:
        ident = _card_probe_subprocess(device)
    name = str(ident.get("name") or "").strip()
    total = int(ident.get("total_memory_bytes") or 0)
    uuid = str(ident.get("uuid") or "").strip()
    cap = str(ident.get("capability") or "").strip()
    if not name or total <= 0 or not uuid or not cap or name.lower() == "unknown":
        raise ReproError(f"card_identity: torch returned an incomplete identity {ident!r}")
    out = {"device": int(device), "name": name, "total_memory_bytes": total, "capability": cap,
           "uuid": uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}",
           "multi_processor_count": int(ident.get("multi_processor_count") or 0),
           "l2_cache_bytes": int(ident.get("l2_cache_bytes") or 0)}
    # the board memory of THIS card (torch's uuid lacks the 'GPU-' prefix nvidia-smi lists — added
    # above); nvidia-smi failing raises in gpu_cards, a uuid it does not list raises here
    smi = {c["uuid"]: c for c in gpu_cards()}
    row = smi.get(out["uuid"])
    if row is None:
        raise ReproError(f"card_identity: nvidia-smi does not list the card {out['uuid']} (it lists "
                         f"{', '.join(sorted(smi))}) — its board memory, part of the card key, "
                         f"cannot be read")
    out["memory_total_mib"] = int(row["total_mib"])
    out["key"] = card_key(out)
    return out


def _nvidia_smi(args: Sequence[str]) -> str:
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT_S)
    except FileNotFoundError as e:
        raise ReproError("nvidia-smi is not installed — the card cannot be checked") from e
    except subprocess.TimeoutExpired as e:
        raise ReproError(f"nvidia-smi {' '.join(args)} did not answer in {PROBE_TIMEOUT_S} s") from e
    if r.returncode != 0:
        raise ReproError(f"nvidia-smi {' '.join(args)} failed (exit {r.returncode}): "
                         f"{(r.stderr or r.stdout).strip()[-800:]}")
    return r.stdout


def gpu_cards() -> List[Dict[str, Any]]:
    """Every card nvidia-smi sees: index, uuid, name, memory used / total (MiB), driver."""
    out = []
    for ln in _nvidia_smi(["--query-gpu=index,uuid,name,memory.used,memory.total,driver_version",
                           "--format=csv,noheader,nounits"]).splitlines():
        if not ln.strip():
            continue
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 6:
            raise ReproError(f"nvidia-smi: unreadable card row {ln!r}")
        idx, uuid, used, total, drv = f[0], f[1], f[-3], f[-2], f[-1]
        name = ",".join(f[2:-3]).strip()
        try:
            out.append({"index": int(idx), "uuid": uuid, "name": name, "used_mib": int(float(used)),
                        "total_mib": int(float(total)), "driver_version": drv})
        except ValueError as e:
            raise ReproError(f"nvidia-smi: unreadable card row {ln!r}") from e
    if not out:
        raise ReproError("nvidia-smi lists no card")
    return out


def gpu_compute_processes() -> List[Dict[str, Any]]:
    """Every compute process nvidia-smi lists: pid, name, used memory (MiB, None when the driver
    does not say), card uuid."""
    out = []
    for ln in _nvidia_smi(["--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                           "--format=csv,noheader,nounits"]).splitlines():
        s = ln.strip()
        if not s or s.lower().startswith("no running"):
            continue
        f = [x.strip() for x in s.split(",")]
        if len(f) < 4:
            raise ReproError(f"nvidia-smi: unreadable process row {ln!r}")
        try:
            pid = int(f[0])
        except ValueError as e:
            raise ReproError(f"nvidia-smi: unreadable process row {ln!r}") from e
        try:
            used: Optional[int] = int(float(f[-2]))
        except ValueError:
            used = None                                   # '[N/A]': listed all the same
        out.append({"pid": pid, "name": ",".join(f[1:-2]).strip(), "used_mib": used,
                    "gpu_uuid": f[-1]})
    return out


def _visible_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The cards this process may run on. CUDA_VISIBLE_DEVICES naming cards by uuid selects those;
    unset, or by index (CUDA's enumeration order need not be nvidia-smi's), or with MIG names:
    EVERY card — checking more never lets a shared card through."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return list(cards)
    toks = [t.strip() for t in cvd.split(",") if t.strip()]
    if not toks:
        raise ReproError("CUDA_VISIBLE_DEVICES is empty — this process can see no card")
    if all(t.startswith("GPU-") for t in toks):
        sel = [c for c in cards if any(c["uuid"].startswith(t) for t in toks)]
        if not sel:
            raise ReproError(f"CUDA_VISIBLE_DEVICES={cvd!r} names no card nvidia-smi lists")
        return sel
    if len(cards) == 1 and all(t.isdigit() for t in toks):
        if toks[0] != "0":
            raise ReproError(f"CUDA_VISIBLE_DEVICES={cvd!r} but the host has one card (index 0)")
        return list(cards)
    return list(cards)


def require_exclusive_gpu(log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """FAIL unless the card(s) this process may use are FREE: no compute process listed on them
    and no memory in use (point 4 — never degrade to a smaller resolution or window). Called by a
    launcher BEFORE it starts a GPU step, or by the step itself before torch touches CUDA: this
    process holding a CUDA context is refused, because inside a container nvidia-smi may list it
    under a host pid that cannot be told apart from a foreign one. The memory test also catches a
    process nvidia-smi cannot list (another container): measured 2026-10-07, A100 80GB PCIe,
    driver 595.91.07 — an idle card reads 0 MiB used.

    Returns the record of what was checked (cards with their used / total MiB, driver) — no pid,
    no time: it is the same record on every run that passes."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise ReproError("require_exclusive_gpu: this process already initialised CUDA — check the "
                         "card before torch touches it (or from the launcher, before the step)")
    cards = _visible_cards(gpu_cards())
    procs = gpu_compute_processes()
    me = os.getpid()
    uuids = {c["uuid"] for c in cards}
    foreign = [p for p in procs if p["gpu_uuid"] in uuids and p["pid"] != me]
    busy = [c for c in cards if c["used_mib"] > 0]
    if busy and not foreign:
        # USER 2026-10-07: memory in use with NO process on the card never stops a run (pccr: 4 MiB
        # of 81920 the driver had not freed yet right after vLLM was stopped) — logged only
        log("[gpu] DECLARED: " + "; ".join(f"{c['name']} {c['uuid']}: {c['used_mib']} MiB in use "
                                           f"with no process listed — the step goes on" for c in busy))
    if foreign:
        lines = [f"pid {p['pid']} {p['name'] or '?'} {p['used_mib'] if p['used_mib'] is not None else 'N/A'}"
                 f" MiB on {p['gpu_uuid']}" for p in foreign]
        lines += [f"{c['name']} {c['uuid']}: {c['used_mib']} MiB of {c['total_mib']} MiB in use"
                  for c in busy]
        msg = ("the GPU is NOT free — a GPU step needs the whole card (docs/plan_determinismo.md "
               "point 4; nothing is lowered to fit): " + "; ".join(lines))
        log(f"[gpu] {msg}")
        raise ReproError(msg)
    rec = {"cards": [{"uuid": c["uuid"], "name": c["name"], "used_mib": c["used_mib"],
                      "total_mib": c["total_mib"], "driver_version": c["driver_version"]}
                     for c in cards],
           "compute_processes": 0}
    log("[gpu] exclusive: " + "; ".join(f"{c['name']} {c['uuid']} free ({c['total_mib']} MiB, "
                                         f"driver {c['driver_version']})" for c in cards))
    return rec


# ── the environment ─────────────────────────────────────────────────────────────────────────

# distributions whose versions enter every record (their numerics reach the products); one not
# installed in this env is recorded as null — a fact, not an unknown
DEFAULT_LIBS: Tuple[str, ...] = (
    "numpy", "scipy", "torch", "torchvision", "opencv-python", "opencv-python-headless",
    "open3d", "pycolmap", "scikit-learn", "threadpoolctl", "transformers", "huggingface-hub",
    "safetensors", "onnxruntime", "onnxruntime-gpu", "xformers", "einops", "timm", "pyyaml",
    "plyfile", "trimesh", "kornia")

# environment variables that change numerics, threading or which files a run reads
ENV_KEYS: Tuple[str, ...] = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "OPENBLAS_CORETYPE", "MKL_CBWR", CUBLAS_WORKSPACE_ENV, "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
    "HF_HUB_OFFLINE")


def cpu_model() -> str:
    """The CPU model string (/proc/cpuinfo); RAISES when it cannot be read."""
    try:
        txt = Path("/proc/cpuinfo").read_text(errors="replace")
    except OSError as e:
        raise ReproError(f"cpu_model: /proc/cpuinfo unreadable ({e})") from e
    for key in ("model name", "Model", "Hardware", "cpu model"):
        for ln in txt.splitlines():
            k, sep, v = ln.partition(":")
            if sep and k.strip() == key and v.strip():
                return v.strip()
    raise ReproError("cpu_model: /proc/cpuinfo names no CPU model")


def blas_record() -> List[Dict[str, Any]]:
    """The BLAS / OpenMP libraries loaded in this process (threadpoolctl): API, version, the core
    type OpenBLAS dispatched to ('architecture' — point 38), threading layer, thread count."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError as e:
        raise ReproError("threadpoolctl is not installed — the BLAS core cannot be recorded") from e
    out = []
    for d in threadpool_info():
        out.append({k: d.get(k) for k in ("user_api", "internal_api", "version", "architecture",
                                          "threading_layer", "num_threads")}
                   | {"library": os.path.basename(str(d.get("filepath", "")))})
    return sorted(out, key=lambda r: (str(r["user_api"]), str(r["library"])))


def _git(root: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                       timeout=PROBE_TIMEOUT_S)
    if r.returncode != 0:
        raise ReproError(f"git -C {root} {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[-400:]}")
    return r.stdout if binary else r.stdout.decode(errors="replace")


def git_state(root: PathLike) -> Dict[str, Any]:
    """commit + whether the tree differs from it: sha256 of ``git diff HEAD`` (tracked files,
    submodules excluded — each is recorded on its own) and of the untracked (not ignored) files'
    names and contents — a new module is untracked until it is committed. Two equal records = the
    same code, dirty or not."""
    root = Path(root)
    commit = _git(root, "rev-parse", "HEAD").strip()
    diff = _git(root, "diff", "HEAD", "--no-color", "--no-ext-diff", "--ignore-submodules",
                binary=True)
    names = sorted(n for n in _git(root, "ls-files", "--others", "--exclude-standard", "-z")
                   .split("\0") if n)
    untracked = [[n, sha256_file(root / n) if (root / n).is_file() else None] for n in names]
    return {"commit": commit, "dirty": bool(diff) or bool(names),
            "diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_sha256": sha256_json(untracked), "n_untracked": len(names)}


def submodule_paths(root: PathLike = REPO_ROOT) -> List[str]:
    """The submodule paths declared in ``root``/.gitmodules (the vendored forks)."""
    p = Path(root) / ".gitmodules"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        k, sep, v = ln.partition("=")
        if sep and k.strip() == "path":
            out.append(v.strip())
    return sorted(out)


def _dist_version(name: str) -> Optional[str]:
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_record(*, gpu: bool, libs: Sequence[str] = DEFAULT_LIBS) -> Dict[str, Any]:
    """What a product depends on beyond its inputs, for its report (chunk_plan.json, windows.json,
    the precision step records ...): python, CPU model and the cores this process may use, the
    BLAS core, library versions, the deterministic environment variables, the git state of the
    repo and of every vendored fork, and — ``gpu=True`` — torch / CUDA / cuDNN and the card
    (:func:`card_identity`) with its driver. No time, host name or pid: two runs on the same
    machine and code give the same record. Any part that cannot be read RAISES."""
    rec: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "cpu": {"model": cpu_model(), "logical_cpus": int(os.cpu_count() or 0),
                "affinity_cpus": len(os.sched_getaffinity(0))},
        "blas": blas_record(),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
        "libs": {name: _dist_version(name) for name in libs},
        "git": {"repo": git_state(REPO_ROOT),
                "submodules": {p: git_state(REPO_ROOT / p) for p in submodule_paths(REPO_ROOT)
                               if (REPO_ROOT / p / ".git").exists()}},
    }
    if gpu:
        import torch
        card = card_identity(0)
        drivers = {c["uuid"]: c["driver_version"] for c in gpu_cards()}
        if card["uuid"] not in drivers:
            raise ReproError(f"environment_record: nvidia-smi does not list the card {card['uuid']}")
        rec["torch"] = {"version": str(torch.__version__), "cuda": str(torch.version.cuda),
                        "cudnn": int(torch.backends.cudnn.version() or 0),
                        "git_version": str(getattr(torch.version, "git_version", ""))}
        if not rec["torch"]["cudnn"]:
            raise ReproError("environment_record: torch reports no cuDNN")
        rec["gpu"] = dict(card, driver_version=drivers[card["uuid"]])
    return rec


# ── hashing, stamps, stable ids ─────────────────────────────────────────────────────────────

_HASH_BLOCK = 1 << 20                                     # read size only: never changes a digest


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(_HASH_BLOCK), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_path(path: PathLike) -> str:
    """A file's sha256, or a directory's: sha256 over the sorted (relative posix path, file sha256)
    of every file under it (bytecode caches excluded — never an input). A missing path RAISES."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if not p.is_dir():
        raise FileNotFoundError(f"stamp input {p} does not exist")
    rows = []
    for dirpath, dirnames, filenames in os.walk(p):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn.endswith(".pyc"):
                continue
            f = Path(dirpath) / fn
            rows.append(f"{f.relative_to(p).as_posix()}\0{sha256_file(f)}\n")
    h = hashlib.sha256()
    for r in sorted(rows):
        h.update(r.encode("utf-8"))
    return h.hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        if o.dtype.hasobject:
            raise TypeError("canonical_json: an object array has no stable bytes — convert it")
        a = np.ascontiguousarray(o)
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        return {"__ndarray__": a.dtype.str, "shape": list(a.shape),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(o, os.PathLike):
        return Path(o).as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted((canonical_json(x) for x in o))
    if isinstance(o, (bytes, bytearray)):
        return {"__bytes__": hashlib.sha256(bytes(o)).hexdigest()}
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    raise TypeError(f"canonical_json: {type(o).__name__} is not stampable")


def canonical_json(obj: Any) -> str:
    """One text per value: keys sorted, no whitespace, floats as Python's exact repr, numpy values
    and arrays (by dtype, shape and the sha256 of their bytes), paths, sets, dataclasses."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=True, default=_json_default)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


STAMP_VERSION = 1


def _rel_to_repo(p: Path) -> Optional[str]:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def _code_file(c: Any) -> Tuple[str, Path]:
    """(key, file) of a code entry: a module object, a dotted module name or a file path, inside
    this repo (the forks included). Library code is versioned by environment_record, not here."""
    if hasattr(c, "__file__") and not isinstance(c, (str, os.PathLike)):
        path = Path(c.__file__)
    elif isinstance(c, os.PathLike) or (isinstance(c, str) and (os.sep in c or c.endswith(".py"))):
        path = Path(c)
    elif isinstance(c, str):
        import importlib.util
        spec = importlib.util.find_spec(c)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            raise ReproError(f"stamp: module {c!r} cannot be found on sys.path")
        path = Path(spec.origin)
    else:
        raise ReproError(f"stamp: code entry {c!r} is not a module, module name or file")
    if not path.is_file():
        raise ReproError(f"stamp: code file {path} does not exist")
    key = _rel_to_repo(path)
    if key is None:
        raise ReproError(f"stamp: {path} is outside {REPO_ROOT} — library code is recorded by "
                         f"environment_record, not stamped as code")
    return key, path


def _named_inputs(inputs: Union[Mapping[str, PathLike], Iterable[PathLike]],
                  root: Optional[PathLike]) -> Dict[str, Path]:
    if isinstance(inputs, Mapping):
        return {str(k): Path(v) for k, v in inputs.items()}
    out: Dict[str, Path] = {}
    for x in inputs:
        p = Path(x)
        key = os.path.relpath(p, root).replace(os.sep, "/") if root is not None else p.name
        if key in out:
            raise ReproError(f"stamp: two inputs share the key {key!r} — pass a mapping "
                             f"{{name: path}} or a root")
        out[key] = p
    return out


def stamp(inputs: Union[Mapping[str, PathLike], Iterable[PathLike]] = (),
          code: Iterable[Any] = (), config: Optional[Mapping[str, Any]] = None, *,
          root: Optional[PathLike] = None) -> Dict[str, Any]:
    """The identity of a step's product: sha256 of every INPUT (files or directories, keyed by the
    mapping's names, else by the path relative to ``root``, else by the file name — never by an
    absolute path, so a copied session keeps its stamps), of every CODE file (module objects,
    dotted names or paths inside this repo, keyed by the repo-relative path) and of each top-level
    entry of ``config`` (canonical JSON — pass the sections the step reads, under their names).
    Returns {'stamp_version', 'inputs', 'code', 'config', 'sha256'}: JSON-able, deterministic.
    A missing input RAISES — a stamp of something absent says nothing."""
    ins = _named_inputs(inputs, root)
    st_in = {k: sha256_path(p) for k, p in sorted(ins.items())}
    st_code: Dict[str, str] = {}
    for c in code:
        k, p = _code_file(c)
        st_code[k] = sha256_file(p)
    if config is not None and not isinstance(config, Mapping):
        raise ReproError(f"stamp: config must be a mapping of named sections, got "
                         f"{type(config).__name__}")
    st_cfg = {str(k): sha256_json(v) for k, v in (config or {}).items()}
    body: Dict[str, Any] = {"stamp_version": STAMP_VERSION, "inputs": st_in,
                            "code": dict(sorted(st_code.items())),
                            "config": dict(sorted(st_cfg.items()))}
    body["sha256"] = sha256_json(body)
    return body


def check_stamp(saved: Any, now: Mapping[str, Any]) -> List[str]:
    """What differs between a SAVED stamp and the one computed NOW — [] when the product is
    reusable. Every difference is named (input / code file / config section: changed, new, gone);
    no saved stamp, an unreadable one or another stamp version is a difference too."""
    if not isinstance(now, Mapping) or "sha256" not in now:
        raise ReproError("check_stamp: 'now' is not a stamp (call repro.stamp)")
    if not isinstance(saved, Mapping) or "sha256" not in saved:
        return ["no stamp saved with the product (or it is unreadable)"]
    diffs: List[str] = []
    if saved.get("stamp_version") != now.get("stamp_version"):
        diffs.append(f"stamp version {saved.get('stamp_version')!r} != {now.get('stamp_version')!r}")
    for sec, label in (("inputs", "input"), ("code", "code"), ("config", "config")):
        a, b = saved.get(sec), now.get(sec)
        if not isinstance(a, Mapping):
            diffs.append(f"the saved stamp has no readable '{sec}'")
            continue
        for k in sorted(set(a) | set(b)):
            if k not in b:
                diffs.append(f"{label} '{k}' was stamped and is not one now")
            elif k not in a:
                diffs.append(f"{label} '{k}' is new (not in the saved stamp)")
            elif a[k] != b[k]:
                diffs.append(f"{label} '{k}' changed ({str(a[k])[:12]} -> {str(b[k])[:12]})")
    if not diffs and saved.get("sha256") != now.get("sha256"):
        diffs.append("the saved stamp's digest does not match its parts (it was edited)")
    return diffs


def stable_id(*parts: Any, n_hex: Optional[int] = None) -> str:
    """A deterministic identifier derived from ``parts`` (any canonical-JSON-able values: the
    inputs that make the thing what it is) — replaces uuid4 ids so a re-run writes the same
    files. Full sha256 hex by default; ``n_hex`` keeps an existing shorter format (the correction
    ledger's 8 characters)."""
    if not parts:
        raise ReproError("stable_id: an id must be derived from something")
    h = sha256_json(list(parts))
    if n_hex is None:
        return h
    if isinstance(n_hex, bool) or not 1 <= int(n_hex) <= len(h):
        raise ReproError(f"stable_id: n_hex {n_hex!r} must lie in [1, {len(h)}]")
    return h[:int(n_hex)]


# ── lossless float text (point 45) ──────────────────────────────────────────────────────────

def exact_float(x: Any) -> str:
    """The shortest text that reads back as EXACTLY the same float64 (Python's repr: round-trip
    exact by construction; '-0.0', 'inf', 'nan' included). float32 values widen exactly."""
    return repr(float(x))


def exact_row(values: Any) -> str:
    """One text row of floats, each :func:`exact_float`, separated by one space."""
    return " ".join(exact_float(v) for v in np.asarray(values, dtype=np.float64).ravel())


def write_rows_exact(path: PathLike, rows: Any) -> Path:
    """A 2-D table of floats as text, one :func:`exact_row` per line (what np.loadtxt and every
    reader of camera_poses.txt / intrinsic.txt already parse), written atomically (temporary file
    in the same directory, then rename)."""
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"write_rows_exact: rows of shape {arr.shape} — expected (N, M), N, M > 0")
    text = "".join(exact_row(r) + "\n" for r in arr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # temporary name of this writer (pid + thread): created with open() so the file keeps the
    # umask permissions every other writer of these files gives them (mkstemp would force 0600)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def write_poses_exact(path: PathLike, poses: Any) -> Path:
    """camera_poses.txt: one 4x4 pose per line, 16 values row-major, float64 round-trip exact.
    Accepts (N, 4, 4), (N, 16) or one (4, 4)."""
    arr = np.asarray(poses, dtype=np.float64)
    if arr.shape == (4, 4):
        arr = arr[None]
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        arr = arr.reshape(len(arr), 16)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"write_poses_exact: poses of shape {np.shape(poses)} — expected "
                         f"(N, 4, 4) or (N, 16)")
    return write_rows_exact(path, arr)


def write_intrinsics_exact(path: PathLike, rows: Any) -> Path:
    """intrinsic.txt: one 'fx fy cx cy' row per keyframe, float64 round-trip exact. Accepts (N, 4)
    or one (4,)."""
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape == (4,):
        arr = arr[None]
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"write_intrinsics_exact: rows of shape {np.shape(rows)} — expected (N, 4)"
                         f" 'fx fy cx cy'")
    return write_rows_exact(path, arr)
