# STAC-Builder — Phase 7 A6000 coexistence probe.
#
# Reports whether the vLLM semantic service and a reconstruction + Phase R can
# share the single 48 GB A6000: current GPU memory, the vLLM footprint, and the
# headroom left for DA3/VGGT + Phase R. It reads nvidia-smi and the semantic
# config (gpu_memory_utilization); it does not itself launch a reconstruction —
# the caller decides based on the reported headroom, and the report states the
# final memory config (or that temporal partitioning is required).
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import subprocess
from pathlib import Path


def _nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return {"error": out.stderr.strip() or "nvidia-smi failed"}
        line = out.stdout.strip().splitlines()[0]
        total, used, free = (int(x.strip()) for x in line.split(",")[:3])
        return {"total_mib": total, "used_mib": used, "free_mib": free}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def _semantic_util() -> float | None:
    try:
        import yaml
        cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
        sem = cfg.get("semantic", {})
        backends = sem.get("backends", {})
        b = backends.get(sem.get("default_backend", "qwen_local"), {})
        return float(b.get("gpu_memory_utilization")) if b.get("gpu_memory_utilization") else None
    except Exception:
        return None


# empirical peak GPU for the reconstruction stack on this repo (A6000 runs)
RECON_PEAK_MIB = {"da3_streaming": 9000, "vggt_omega": 14000, "phase_r": 1500}


def probe(recon_headroom_needed_mib: int = 15000) -> dict:
    gpu = _nvidia_smi()
    util = _semantic_util()
    rep = {"gpu": gpu, "semantic_gpu_memory_utilization": util,
           "recon_peak_estimate_mib": RECON_PEAK_MIB,
           "recon_headroom_needed_mib": recon_headroom_needed_mib}
    if "free_mib" in gpu:
        free = gpu["free_mib"]
        rep["headroom_free_mib"] = free
        rep["coexists"] = free >= recon_headroom_needed_mib
        if rep["coexists"]:
            rep["recommendation"] = (
                f"Coexists: {free} MiB free ≥ {recon_headroom_needed_mib} MiB needed. "
                f"Keep semantic at gpu_memory_utilization={util}; run reconstruction "
                f"+ Phase R concurrently.")
        else:
            rep["recommendation"] = (
                f"Insufficient headroom ({free} MiB free < {recon_headroom_needed_mib} "
                f"MiB). Lower semantic gpu_memory_utilization, or temporally partition: "
                f"stop the semantic service during VGGT-Ω, restart it for Phases 1-6.")
    return rep
