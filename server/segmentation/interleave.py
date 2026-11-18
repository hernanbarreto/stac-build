"""ONE OWNER PER SURFACE — two objects that are the same surface are ONE object.

USER 2026-11-18, on the segmentation of pccr 3108 ("cuando apago los elementos segmentados de la
nube de puntos no desaparecen los puntos" / "hay que arreglarlo de raíz"): MEASURED — of a cloud
point's 8 nearest neighbours (within 7 mm) 39 % belonged to ANOTHER OBJECT. Each point takes the
mask of its own keyframe (point 117), so a surface two keyframes see under two names or two ids
(wall and drywall, one tile tracked twice, a rack in two halves) is shared by two objects POINT BY
POINT. Since 2026-11-11 nothing but the floor merges after SAM3 ("no quiero hacer una pasada para
fusionar objetos, solo el piso … porque a veces une dos objetos que están muy alejados" — the
5 cm same-space test did), so the duplicates stayed.

THE TEST CANNOT JOIN DISTANT OBJECTS: object A is INTERLEAVED with B when at least ``share`` of
the nearest neighbours of A's points are B's points — its points sit, millimetres apart, among
B's. A pair merges when EITHER is interleaved with the other (a tile tracked in a few keyframes
lies wholly among the floor's points while the floor barely notices it); merges chain. The
survivor is the member with the most points: it keeps its id and its label, the others are
recorded as absorbed (``same_surface``, with the share that decided).

His answer to "do I change your rule of 11-11 to allow this merge, at 10 % or 20 %?": "1. Si"
(2026-11-18) — ``segmentation.interleave_merge.share`` = 0.10, the value recommended to him:
on pccr 3108 1213 objects → 280 and the neighbours of another object 39.3 % → 5.1 %; the tiles
stay separate tiles (at 0.20: 583 objects, 16.2 %).

The neighbours are an exact k-nearest search over the cloud (scipy's tree: 18.8 M points, 8
neighbours, ~20 s on the CPU — DECLARED, the numpy path of record; a device twin is to come).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

BLOCK = 2_000_000


def config(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """``segmentation.interleave_merge`` — a missing key fails naming it."""
    if cfg is None:
        from pathlib import Path
        import yaml
        cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text()) or {}
    sec = (cfg.get("segmentation") or {}).get("interleave_merge")
    if not isinstance(sec, dict):
        raise KeyError("config.yaml is missing 'segmentation.interleave_merge' (USER 2026-11-18: objects that "
                       "share a surface point by point are one object)")
    for key in ("enabled", "share", "neighbours"):
        if key not in sec:
            raise KeyError(f"config.yaml is missing 'segmentation.interleave_merge.{key}'")
    share, k = sec["share"], sec["neighbours"]
    if not isinstance(sec["enabled"], bool):
        raise ValueError("segmentation.interleave_merge.enabled must be true / false")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not (0.0 < float(share) <= 1.0):
        raise ValueError(f"segmentation.interleave_merge.share must be in (0, 1], got {share!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"segmentation.interleave_merge.neighbours must be a positive integer, got {k!r}")
    return {"enabled": sec["enabled"], "share": float(share), "neighbours": int(k)}


def neighbour_shares(xyz: np.ndarray, owner: np.ndarray, neighbours: int) -> Dict[Tuple[int, int], float]:
    """{(a, b): the share of the nearest neighbours of a's points that are b's points}, for every
    pair of objects with one at all. ``owner``: each point's object, ≤ 0 = none (such points are
    neighbours of nobody's and own no share)."""
    from scipy.spatial import cKDTree
    owner = np.asarray(owner, np.int64)
    n = int(owner.max()) + 1 if len(owner) else 0
    if n <= 1:
        return {}
    tree = cKDTree(xyz)
    counts: Dict[int, int] = {}
    for a in range(0, len(xyz), BLOCK):
        _d, idx = tree.query(xyz[a:a + BLOCK], k=neighbours + 1, workers=-1)
        own = np.repeat(owner[a:a + BLOCK], neighbours)
        other = owner[idx[:, 1:]].ravel()                      # (the first neighbour is the point itself)
        m = (own > 0) & (other > 0) & (own != other)
        key, c = np.unique(own[m] * n + other[m], return_counts=True)
        for k_, v in zip(key.tolist(), c.tolist()):
            counts[k_] = counts.get(k_, 0) + v
    points = np.bincount(owner[owner > 0], minlength=n)
    return {(k_ // n, k_ % n): v / float(neighbours * points[k_ // n]) for k_, v in counts.items()}


def groups(shares: Dict[Tuple[int, int], float], points: Dict[int, int], share: float) -> Dict[int, Dict[str, Any]]:
    """{absorbed id: {"into": survivor id, "share": the largest share that tied it to its group}}.
    A pair is tied when either's share in the other is at least ``share``; ties chain; a group's
    survivor is its member with the most points (a tie: the smaller id). Deterministic: the pairs
    are taken in id order."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tied: Dict[int, float] = {}
    for (a, b) in sorted(shares):
        s = shares[(a, b)]
        if s >= share:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
            for x in (a, b):
                tied[x] = max(tied.get(x, 0.0), s)
    members: Dict[int, List[int]] = {}
    for x in sorted(parent):
        members.setdefault(find(x), []).append(x)
    out: Dict[int, Dict[str, Any]] = {}
    for group in members.values():
        if len(group) < 2:
            continue
        survivor = min(group, key=lambda i: (-int(points.get(i, 0)), i))
        for x in group:
            if x != survivor:
                out[x] = {"into": survivor, "share": round(float(tied.get(x, 0.0)), 4)}
    return out


def merge_instances(instances: List[dict], xyz: np.ndarray, n_points: int, *, share: float, neighbours: int,
                    record: Optional[dict] = None, log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Merge, IN PLACE, the instances that share a surface point by point. ``instances``: the
    projection's (``id`` / ``instance_id``, ``label``, ``globalIndices``). The survivors take the
    absorbed ones' points (``total_points`` updated; a box is the caller's to refit); ``record``
    — the projection's absorbed map — gets one entry per absorbed instance. Returns the report."""
    def iid(inst: dict) -> int:
        return int(inst.get("instance_id", inst.get("id")))

    owner = np.zeros(int(n_points), np.int64)
    by_id: Dict[int, dict] = {}
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        owner[gi] = iid(inst)
        by_id[iid(inst)] = inst
    if any(i <= 0 for i in by_id):
        raise ValueError("an instance id ≤ 0 cannot be told from 'no object'")
    shares = neighbour_shares(xyz, owner, neighbours)
    points = {i: len(inst.get("globalIndices") or []) for i, inst in by_id.items()}
    before = _mixed(shares, points)
    absorbed = groups(shares, points, share)
    for loser, how in sorted(absorbed.items()):
        keeper, gone = by_id[how["into"]], by_id[loser]
        keeper["globalIndices"] = sorted(set(keeper["globalIndices"]) | set(gone["globalIndices"]))
        keeper["total_points"] = len(keeper["globalIndices"])
        if record is not None:
            record[int(loser)] = {"into": int(how["into"]), "into_label": keeper.get("label"),
                                  "label": gone.get("label"), "reason": "same_surface",
                                  "interleaved_share": how["share"], "points": len(gone["globalIndices"])}
    if absorbed:
        instances[:] = [inst for inst in instances if iid(inst) not in absorbed]
        owner_after = owner.copy()
        remap = np.arange(int(owner.max()) + 1)
        for loser, how in absorbed.items():
            remap[loser] = how["into"]
        owner_after = remap[owner]
        after = _mixed(neighbour_shares(xyz, owner_after, neighbours),
                       {iid(i): len(i["globalIndices"]) for i in instances})
    else:
        after = before
    report = {"rule": "USER 2026-11-18: objects that share a surface point by point are one object "
                      "(a pair merges when either has >= share of its points' nearest neighbours in the other)",
              "share": float(share), "neighbours": int(neighbours), "objects_before": len(by_id),
              "objects_after": len(instances), "absorbed": len(absorbed),
              "neighbours_of_another_object_before": round(before, 4),
              "neighbours_of_another_object_after": round(after, 4)}
    log(f"[SegPipeline]   same-surface merge: {report['objects_before']} → {report['objects_after']} objects "
        f"({len(absorbed)} absorbed; neighbours of another object {100 * before:.1f} % → {100 * after:.1f} %)")
    return report


def _mixed(shares: Dict[Tuple[int, int], float], points: Dict[int, int]) -> float:
    """Of the segmented points' nearest neighbours, the share that belongs to another object."""
    total = float(sum(points.values()))
    if not total:
        return 0.0
    return float(sum(s * points.get(a, 0) for (a, _b), s in shares.items()) / total)
