"""DA3's weights: ONE Hugging Face cache, a PINNED revision, the files' sha256 verified
(docs/plan_determinismo.md points 32 and 43).

Before: ``DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE-1.1')`` resolved
``refs/main`` on every run, in whatever HF_HOME the launcher had: the backend (scripts/start.sh)
read /workspace/hf_cache (the complete snapshot b2359bdf, 2026-05-30), a by-hand
``python -m precision.runner`` inherited the pod's /workspace/.cache/huggingface, whose DA3
snapshot was an empty directory beside a 0-byte ``.incomplete`` blob — a DA3 load from there
would have downloaded whatever ``main`` was that day. An upstream update would have changed every
window, the walk, the chunk plan and the gauge silently.

Now every DA3 load (extract_da3_depth.py — I3 windows, the focal probe, the anchors, the fork's
bridge anchors through it) reads the snapshot of the pinned revision inside :data:`HF_HOME`, the
one cache every launcher shares, offline, after checking each file against its pinned sha256.
A missing snapshot or a file that hashes otherwise FAILS — nothing is downloaded on the fly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

# THE one HF cache of this pod: the one scripts/start.sh exports for the backend (and
# scripts/serve_semantic.sh for vLLM), which holds the complete DA3 snapshot. A test keeps
# start.sh and this constant equal.
HF_HOME = "/workspace/hf_cache"

# The revision every validated run loaded (refs/main of /workspace/hf_cache since 2026-05-30)
# and the sha256 of each file of that snapshot: model.safetensors is the LFS object whose oid IS
# its sha256 (the blob's name in the cache); config.json measured with sha256sum 2026-10-07.
PINNED: Dict[str, Dict[str, object]] = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": {
        "revision": "b2359bdf726fb44ef62acca04d629dcf158053e7",
        "sha256": {
            "config.json": "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
            "model.safetensors": "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c",
        },
    },
}


class DA3WeightsError(RuntimeError):
    """The pinned DA3 weights are not where they must be, or are not the pinned bytes."""


def pinned(model_id: str) -> Mapping[str, object]:
    p = PINNED.get(str(model_id))
    if p is None:
        raise DA3WeightsError(f"DA3 model '{model_id}' has no pinned revision (da3_weights.PINNED: "
                              f"{sorted(PINNED)}) — pin its revision and file sha256s first")
    return p


def snapshot_dir(model_id: str, hf_home: Optional[os.PathLike] = None) -> Path:
    """The snapshot directory of the pinned revision inside the one cache (not verified)."""
    rev = str(pinned(model_id)["revision"])
    org_name = str(model_id).replace("/", "--")
    return Path(hf_home or HF_HOME) / "hub" / f"models--{org_name}" / "snapshots" / rev


def verified_snapshot(model_id: str, hf_home: Optional[os.PathLike] = None,
                      log=print) -> Path:
    """The pinned snapshot directory, every pinned file present and hashing to its sha256.
    Raises DA3WeightsError naming the file otherwise."""
    import repro
    d = snapshot_dir(model_id, hf_home)
    want = dict(pinned(model_id)["sha256"])
    if not d.is_dir():
        raise DA3WeightsError(f"the pinned DA3 snapshot {d} does not exist — the weights of "
                              f"{model_id} at revision {pinned(model_id)['revision']} must be in "
                              f"the one HF cache {Path(hf_home or HF_HOME)} (nothing is downloaded "
                              f"on the fly)")
    for name in sorted(want):                      # presence first: a missing file is the first fact
        if not (d / name).is_file():
            raise DA3WeightsError(f"{d / name} is missing from the pinned DA3 snapshot")
    for name, sha in sorted(want.items()):
        f = d / name
        got = repro.sha256_file(f)
        if got != sha:
            raise DA3WeightsError(f"{f} hashes to sha256 {got}, the pinned {model_id} file is "
                                  f"{sha} — another file than the validated one")
    log(f"[DA3 weights] {model_id} @ {str(pinned(model_id)['revision'])[:12]} verified "
        f"({len(want)} file(s) by sha256) in {Path(hf_home or HF_HOME)}")
    return d


def identity(model_id: str) -> Dict[str, object]:
    """What a product made with these weights records: model, revision and the files' sha256."""
    p = pinned(model_id)
    return {"model_id": str(model_id), "revision": str(p["revision"]),
            "sha256": dict(sorted(dict(p["sha256"]).items()))}


def hf_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """``base`` (default os.environ) with the one cache and Hugging Face offline — the environment
    of every DA3 subprocess, whoever launched it (backend, runner, by hand)."""
    env = dict(os.environ if base is None else base)
    env["HF_HOME"] = HF_HOME
    env.pop("HF_HUB_CACHE", None)
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env["HF_HUB_OFFLINE"] = "1"
    return env
