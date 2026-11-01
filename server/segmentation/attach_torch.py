"""The attach step's neighbour vote on the device (USER 2026-11-01: "todo debe correr en GPU").

`segmentation.pipeline._attach_unsegmented` grows every instance into the unsegmented cloud: per round,
every unlabelled point within ``attach_dist_m`` of labelled points adopts the instance most of its
``votes`` nearest labelled neighbours belong to (ties to the nearest). The numpy recipe asked a cKDTree
over all labelled points (pccr audit5: 4 rounds over ~20 M queries, 38 s of tree queries plus the index
rebuild). Here the bounded k nearest come from `segmentation.knn_torch.knn_within_radius` (exact, on the
grid at the attach distance) and the vote is the recipe's arithmetic on tensors: the candidate owner of
each neighbour, the agreement count of each column with its row (misses count nothing), the column with
the most agreement — the first maximum, i.e. the nearest among equals. The one declared difference:
among neighbours at EXACTLY equal distance the tree's order and the grid's order can differ, so a vote
split between two instances by such a tie can fall the other way (tests/test_dedupe_torch.py holds the
owners equal on synthetic clouds, where exact ties do not occur). The per-instance index rebuild stays
the recipe's single argsort.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from segmentation.knn_torch import knn_within_radius


def attach_rounds_t(xyz: np.ndarray, owner: np.ndarray, attach_dist_m: float, rounds: int, votes: int,
                    device: str, log: Callable = print) -> np.ndarray:
    """The recipe's rounds on ``device``: ``owner`` (N,) int64, -1 unlabelled, returned updated."""
    dev = str(device)
    X = torch.as_tensor(np.ascontiguousarray(np.asarray(xyz, np.float64)), device=dev)
    own = torch.as_tensor(np.ascontiguousarray(np.asarray(owner, np.int64)), device=dev)
    votes = max(1, int(votes))
    rounds = max(1, int(rounds))
    for r in range(rounds):
        lab_idx = torch.nonzero(own >= 0).reshape(-1)
        un_idx = torch.nonzero(own < 0).reshape(-1)
        if int(un_idx.shape[0]) == 0 or int(lab_idx.shape[0]) == 0:
            break
        _, I = knn_within_radius(X[lab_idx], X[un_idx], votes, float(attach_dist_m))     # (n_un, votes)
        hit = I >= 0
        lab_owner = own[lab_idx]
        cand = torch.where(hit, lab_owner[I.clamp(min=0)], torch.full_like(I, -1))
        if votes < 3:
            win = cand[:, 0]
        else:
            agree = torch.zeros(cand.shape, dtype=torch.int16, device=dev)
            for c in range(votes):
                agree[:, c] = (((cand == cand[:, c][:, None]) & hit).sum(1) * hit[:, c].to(torch.int16)).to(torch.int16)
            win = cand[torch.arange(int(cand.shape[0]), device=dev), torch.argmax(agree, dim=1)]
        got = win >= 0
        if not bool(got.any()):
            break
        own[un_idx[got]] = win[got]
        log(f"[SegPipeline]     📎 round {r + 1}/{rounds}: +{int(got.sum()):,} point(s) within "
            f"{attach_dist_m * 100:.0f} cm")
    return own.cpu().numpy()
