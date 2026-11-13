"""GenRecon's SAMPLED LATENTS on disk — the point a run continues from when its DECODE failed (USER 2026-11-12: "la
regla es que se continúa desde el punto de falla"; 2026-11-13: "Debemos hacer entrar pccr2408 hay que ver la forma
la técnica para que sea escalable a nubes masivas").

reconstruct_scene.py samples one shape latent and one texture latent per chunk (the long part: pccr 2026-08-24,
~40 min of the card) and then decodes them all together. Nothing was kept between the two: a decode that failed
began again at the sampling, and — the generation not being identical run to run — a decode could never be
repeated on the same latents to compare two ways of decoding. This file is what the decode needs and nothing else:

  shape, tex            one (feats, coords) pair per kept chunk, as the samplers left them (already de-normalised)
  coords_list           every kept chunk's active voxels (the coords_*.ply of the output)
  relative_translations the kept chunks' translations on the chunk lattice (the decode merges the chunks with them)
  m_c2o                 EVERY chunk's cube -> world matrix, [chunks, 4, 4] (the kept ones are indexed by chunk_indices;
                        chunk 0's lifts the decoded scene to the world)
  chunk_indices, coords_resolution, pipeline_type

Pure torch, no vendor import: the caller rebuilds its own sparse tensors. Written whole or not at all (a temporary
name, then os.replace), read with weights_only=True (tensors and plain containers alone).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

VERSION = 1
_KEYS = ("version", "shape", "tex", "coords_list", "relative_translations", "m_c2o", "chunk_indices",
         "coords_resolution", "pipeline_type")


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous()


def _pairs(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(_cpu(f), _cpu(c)) for f, c in pairs]


def payload(*, shape, tex, coords_list, relative_translations, m_c2o, chunk_indices, coords_resolution: int,
            pipeline_type: str) -> Dict[str, Any]:
    """The record as it is written: every tensor on the CPU, the scalars as plain python values."""
    return {"version": VERSION, "shape": _pairs(shape), "tex": _pairs(tex),
            "coords_list": [_cpu(c) for c in coords_list],
            "relative_translations": [_cpu(t) for t in relative_translations],
            "m_c2o": torch.stack([_cpu(m) for m in m_c2o]),
            "chunk_indices": [int(i) for i in chunk_indices],
            "coords_resolution": int(coords_resolution), "pipeline_type": str(pipeline_type)}


def validate(rec: Any, where: str = "the latents") -> Dict[str, Any]:
    """The record, or a ValueError naming what is wrong with it (a cut file, another version, lists that do not
    describe the same chunks)."""
    if not isinstance(rec, dict):
        raise ValueError(f"{where}: not a latents record ({type(rec).__name__})")
    missing = [k for k in _KEYS if k not in rec]
    if missing:
        raise ValueError(f"{where}: no {', '.join(missing)}")
    if rec["version"] != VERSION:
        raise ValueError(f"{where}: version {rec['version']!r}, this code reads version {VERSION}")
    n = len(rec["chunk_indices"])
    if n == 0:
        raise ValueError(f"{where}: no chunk")
    for k in ("shape", "tex", "coords_list", "relative_translations"):
        if len(rec[k]) != n:
            raise ValueError(f"{where}: {len(rec[k])} entries of '{k}' for {n} chunks")
    m = rec["m_c2o"]
    if not isinstance(m, torch.Tensor) or m.dim() != 3 or tuple(m.shape[1:]) != (4, 4):
        raise ValueError(f"{where}: m_c2o is not [chunks, 4, 4]")
    if min(rec["chunk_indices"]) < 0 or max(rec["chunk_indices"]) >= m.shape[0]:
        raise ValueError(f"{where}: chunk indices {min(rec['chunk_indices'])}..{max(rec['chunk_indices'])} "
                         f"outside the {m.shape[0]} chunks of m_c2o")
    for k in ("shape", "tex"):
        for i, (feats, coords) in enumerate(rec[k]):
            if feats.dim() != 2 or coords.dim() != 2 or coords.shape[1] != 4 or feats.shape[0] != coords.shape[0]:
                raise ValueError(f"{where}: '{k}' of chunk {rec['chunk_indices'][i]} holds feats "
                                 f"{tuple(feats.shape)} for coords {tuple(coords.shape)}")
    for i, ((_, cs), (_, ct)) in enumerate(zip(rec["shape"], rec["tex"])):
        if not torch.equal(cs, ct):
            raise ValueError(f"{where}: chunk {rec['chunk_indices'][i]}'s texture latent is not on its shape "
                             f"latent's voxels")
    return rec


def save(path, **fields) -> Path:
    """Write the record (see :func:`payload`) whole or not at all; returns the path."""
    path = Path(path)
    rec = validate(payload(**fields), "the latents to save")
    tmp = path.with_name(path.name + ".tmp")
    torch.save(rec, tmp)
    os.replace(tmp, path)
    return path


def load(path) -> Dict[str, Any]:
    """The record on the CPU, validated."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no sampled latents at {path}")
    return validate(torch.load(path, map_location="cpu", weights_only=True), str(path))


def summary(rec: Dict[str, Any]) -> str:
    rows = sum(int(f.shape[0]) for f, _ in rec["shape"])
    return (f"{len(rec['chunk_indices'])} chunks, {rows:,} latent voxels over the chunks, shape {rec['shape'][0][0].shape[1]} "
            f"and texture {rec['tex'][0][0].shape[1]} channels, pipeline {rec['pipeline_type']}")
