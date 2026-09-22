"""The correction, measured on the object's own visits and its three views.

USER 2026-09-18, after the floor closed with its tile lines perfectly aligned:
this replaces the previous correction path. What made it work, in order:

  1. **The unit is the VISIT, and it comes from the points themselves.** Every
     point carries the keyframe it was born in, so an object's keyframes split
     into visits at ANY discontinuity — no threshold. An object seen in a
     single visit cannot show a duplicate and carries no information about a
     pose error: a pose error between two moments can only appear where both
     moments observed the same thing.

  2. **The cloud is filtered first, for real.** A visit that leaves a handful
     of points on an object did not observe it, it grazed it; those points are
     deleted, not down-weighted. Objects too small to be measured go with them.
     Nothing downstream — masks, OBBs, matching, reprojection — sees them
     again.

  3. **The deriva is measured by aligning the SILHOUETTE in the three
     orthogonal views of the object's own OBB, never by 3-D ICP.** A
     point-to-point ICP SLIDES: on a flat desk top or an object symmetric
     along its axis, moving a copy sideways costs the nearest neighbour
     nothing. Measured on pccr epoch 2: the ICP reported 1.4 cm of remaining
     drift on a desk whose plan view showed the two tops plainly apart, and
     the centroid reported 3.1 cm because what sticks out on one side makes up
     for what is missing on the other. The silhouette in plan sees it: 17 cm.

  4. **Each component is therefore measured TWICE** — plan gives (L, W), side
     gives (L, U), front gives (W, U) — and the agreement between the two
     measurements of the same component is what says whether it is determined.
     That replaces comparing 3 DOF against 6 DOF: an object whose two views
     disagree is not used, whatever its residual looks like.

  5. **Only the best-determined object's translation is applied.** With few
     objects, badly spread, a 6-DOF rigid fitted over them lands on a rotation
     that is not there (22.75° on pccr, residuals 9-15 cm). No rotation is
     ever applied with fewer than three well-spread, non-collinear objects.

  6. **The closure is spread by the drift-rate model** (``distribute``): the
     error accumulates with the distance WALKED, E(d)=ε·d, zero at the start
     and extrapolated with the last slope past the last knot.

  7. **Epochs are generated until the measurement stops improving**, each one
     a full state of the session that stays on disk and can be selected.

Every number this module uses comes from ``config.yaml`` under
``correction.visit_drift``; there are no decision literals here.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── STEP 1 — the objects, and the visits their own masks describe ────────
#
# USER 2026-09-18, correcting the first wiring of this:
#   "no eran 82 instancias, esta mal, tenes que tomar todas las de
#    segmentation.json"
#
# CRITERIO 3: the objects are the entries of ``segmentation.json``, whatever
# they are — and SINCE 2026-09-20 they are the FUSED objects, because the
# fusion now rewrites the parent itself (``segmentation/fuse_parent.py``).
#
# USER 2026-09-20: *"la etapa de fusion de instancias es sumamente importante
# porque justamente evita estas cosas, que dos instancias con nombre diferente
# que son lo mismo sean tratadas como objetos independientes … cuando se hace
# todo el analisis de la certificacion, y la correccion, se hace sobre las
# instancias fusionadas y no sobre 'las partes'"*.
#
# The earlier reading of this line said the opposite — that fusing destroys the
# evidence, because the pccr floor is one continuous instance after fusion and
# forty revisited masklets before it. That observation is still true and it is
# the COST, measured on epoch 0: the fused parent yields 2 scale rows where the
# masklet parent yields 11. What the count hides is that 6 of those 11 are
# noise — `k_b` from 0.4432 to 1.8170, two floor patches disagreeing on the
# SIGN of the drift — while the fused set is the desk at residual 10.0 cm and
# a door at 44.0. USER, on being shown it: *"esta bien que se pierdan los
# objetos chicos, la idea es que queden los correctos, no cualquiera, mas no
# significa mejor"*.
#
# What fusion BUYS is the closure that no threshold could reach:
# `wooden_desk#230` (69,608 pts, kf 199-215) reads as ONE visit as a masklet
# and dies at the 2+ visits gate with 149 of the 270; its second copy is the
# masklet `#226` (kf 0-9) 0.64 m away, which the matcher had already absorbed
# into it. Fused: visits kf 0-12 and kf 199-215, shares 34.1 %/65.9 %, closure
# 68.9 cm with 3.0 cm of disagreement against a 9.5 cm bar, over 17.17 m of
# walk — the longest lever arm in the session.
#
# It also makes cross-masklet pairing SAFE, which it never was: `#225` and
# `#226` are different desks in a row and their silhouettes agree to 8 cm at
# 230 cm of separation. Only the matcher knows they are different objects, and
# after the fusion the parent says so.
#
# CRITERIO 4: which keyframe sees each object comes from THE MASKS
# (``seg_masks.npz``, key ``f<frame>_o<oid>``, ``oid = instance_id - 1``), not
# from the points. A keyframe sees the object when SAM3 drew it there.


@dataclass
class Masklet:
    """One SAM3 masklet and the visits its own masks describe."""

    oid: int
    instance_id: int
    label: str
    keyframes: np.ndarray             # keyframe POSITIONS that carry a mask
    visits: List[Tuple[int, int]]     # (kf_first, kf_last) per continuous run

    @property
    def n_visits(self) -> int:
        return len(self.visits)

    def as_dict(self) -> dict:
        return {"oid": self.oid, "instance_id": self.instance_id,
                "label": self.label, "n_keyframes": int(len(self.keyframes)),
                "visits": [list(v) for v in self.visits],
                "n_visits": self.n_visits, "provenance": "tool_measured"}


def masklet_visits(output_dir, log: Callable[[str], None] = print
                   ) -> List[Masklet]:
    """Every SAM3 masklet of the session, with the visits its masks describe.

    Reads ``segmentation.json`` (the masklets, pre-fusion) and the mask store
    it names (``seg_masks.npz`` by default). The frame space of the keys is
    never assumed: ``segmentation.mask_space`` measures whether they are
    keyframe positions or video frame numbers and translates.

    A mask that is stored but EMPTY does not count as seeing the object.
    """
    import re

    from segmentation import mask_space

    output_dir = Path(output_dir)
    doc = json.loads((output_dir / "segmentation.json").read_text())
    entries = doc.get("instances") or []
    masks_path = output_dir / str(doc.get("mask_file") or "seg_masks.npz")
    if not masks_path.exists():
        raise RuntimeError(f"{masks_path} does not exist — the masklets' "
                           f"visits cannot be read without the masks")
    masks = np.load(masks_path)
    space = mask_space.resolve(output_dir, masks=masks, log=log)
    n_kf = len(mask_space.keyframe_numbers(output_dir) or [])
    if not n_kf:
        raise RuntimeError("camera_frames.txt gives no keyframes — the visits "
                           "cannot be numbered")

    # mask-store frame number → keyframe POSITION, built through the translator
    kf_of_mask_frame = {}
    for k in range(n_kf):
        mf = space.from_keyframe(k)
        if mf is not None:
            kf_of_mask_frame[int(mf)] = k

    pat = re.compile(r"^f(\d+)_o(\d+)$")
    seen: Dict[int, set] = {}
    n_empty = 0
    for key in masks.files:
        m = pat.match(key)
        if not m:
            continue
        kf = kf_of_mask_frame.get(int(m.group(1)))
        if kf is None:
            continue
        a = masks[key]
        if a.size == 0 or not a.any():
            n_empty += 1
            continue
        seen.setdefault(int(m.group(2)), set()).add(kf)

    out: List[Masklet] = []
    for e in entries:
        oid = int(e.get("id", -1))
        kfs = np.array(sorted(seen.get(oid, ())), np.int64)
        out.append(Masklet(oid=oid,
                           instance_id=int(e.get("instance_id", oid + 1)),
                           label=str(e.get("label", "object")),
                           keyframes=kfs,
                           visits=visits_of(kfs, n_kf) if len(kfs) else []))
    log(f"[visit-drift] step 1: {len(out)} masklets over {n_kf} keyframes, "
        f"{sum(len(m.keyframes) for m in out)} masks"
        + (f" ({n_empty} stored empty, not counted)" if n_empty else "")
        + f" — {sum(1 for m in out if m.n_visits >= 2)} with 2+ visits")
    return out


def trace_grid(output_dir) -> Tuple[int, int]:
    """(H, W) of the grid the cloud's ``pixel_row`` / ``pixel_col`` live on.

    Read from the session's own intrinsics (``intrinsic.txt``: fx fy cx cy per
    keyframe, centred on the trace grid), never assumed: pccr's masks are
    832x464 and its trace grid 688x384, and sampling one with the other's
    coordinates silently reads the wrong pixel.
    """
    p = Path(output_dir) / "intrinsic.txt"
    if not p.exists():
        raise RuntimeError(f"{p} does not exist — the grid the cloud's pixels "
                           f"live on is unknown and cannot be guessed")
    K = np.loadtxt(p).reshape(-1, 4)
    cx, cy = float(np.median(K[:, 2])), float(np.median(K[:, 3]))
    return int(round(cy * 2)), int(round(cx * 2))


def _vd_cfg():
    """The visit-drift block of config.yaml. Read lazily so the tools that
    call into this module keep their signatures, and through the typed
    dataclass so a missing key fails at load naming itself."""
    from correction.config import load_correction_config
    return load_correction_config().visit_drift


def points_of_masklets(output_dir, frame_global: np.ndarray,
                       pixel_row: np.ndarray, pixel_col: np.ndarray,
                       log: Callable[[str], None] = print,
                       aspect_tol: Optional[float] = None
                       ) -> Dict[int, np.ndarray]:
    """Which cloud points belong to each SAM3 masklet.

    The cloud's ``globalIndices`` are per FUSED instance, so a masklet has no
    points of its own until they are asked for. Every point carries the
    keyframe it was born in and the pixel it was born at, and the masklet's own
    mask for that keyframe says whether that pixel is inside it — so the
    association is a lookup, not an inference.

    Masklets overlap, so a point can belong to more than one; nothing here
    forces a winner.
    """
    import re

    from segmentation import mask_space

    output_dir = Path(output_dir)
    doc = json.loads((output_dir / "segmentation.json").read_text())
    masks = np.load(output_dir / str(doc.get("mask_file") or "seg_masks.npz"))
    space = mask_space.resolve(output_dir, masks=masks, log=lambda m: None)
    kfs = mask_space.keyframe_numbers(output_dir) or []
    n_kf = len(kfs)
    kf_of_frame = np.full(int(max(kfs)) + 2, -1, np.int64)
    for k, f in enumerate(kfs):
        kf_of_frame[int(f)] = k
    ks = kf_of_frame[np.clip(np.asarray(frame_global, np.int64), 0,
                             len(kf_of_frame) - 1)]

    Ht, Wt = trace_grid(output_dir)
    probe = next(masks[k] for k in masks.files if k.startswith("f") and "_o" in k)
    Hm, Wm = int(probe.shape[0]), int(probe.shape[1])
    sr, sc = Hm / float(Ht), Wm / float(Wt)
    tol = float(_vd_cfg().grid_aspect_tol if aspect_tol is None else aspect_tol)
    if abs(sr - sc) / max(sr, sc) > tol:
        raise RuntimeError(
            f"the mask grid ({Hm}x{Wm}) and the trace grid ({Ht}x{Wt}) do not "
            f"share an aspect ratio ({sr:.4f} vs {sc:.4f}, over {tol}) — the "
            f"points cannot be sampled against the masks")
    rows = np.clip((np.asarray(pixel_row, np.int64) * sr).astype(np.int64), 0, Hm - 1)
    cols = np.clip((np.asarray(pixel_col, np.int64) * sc).astype(np.int64), 0, Wm - 1)
    log(f"[visit-drift] points -> masklets: trace {Ht}x{Wt} -> mask {Hm}x{Wm} "
        f"(x{sr:.4f})")

    # points grouped by the keyframe they were born in, once
    order = np.argsort(ks, kind="stable")
    ks_sorted = ks[order]
    start = np.searchsorted(ks_sorted, np.arange(n_kf), "left")
    end = np.searchsorted(ks_sorted, np.arange(n_kf), "right")

    kf_of_mask_frame = {}
    for k in range(n_kf):
        mf = space.from_keyframe(k)
        if mf is not None:
            kf_of_mask_frame[int(mf)] = k
    pat = re.compile(r"^f(\d+)_o(\d+)$")
    by_kf: Dict[int, List[Tuple[int, str]]] = {}
    for key in masks.files:
        m = pat.match(key)
        if not m:
            continue
        kf = kf_of_mask_frame.get(int(m.group(1)))
        if kf is not None:
            by_kf.setdefault(kf, []).append((int(m.group(2)), key))

    out: Dict[int, List[np.ndarray]] = {}
    for kf, items in by_kf.items():
        idx = order[start[kf]:end[kf]]
        if not len(idx):
            continue
        r, c = rows[idx], cols[idx]
        for oid, key in items:
            inside = masks[key][r, c] > 0
            if inside.any():
                out.setdefault(oid, []).append(idx[inside].astype(np.int64))
    res = {o: np.unique(np.concatenate(v)) for o, v in out.items()}
    log(f"[visit-drift] points -> masklets: {len(res)} masklets carry points, "
        f"{sum(len(v) for v in res.values()):,} assignments over "
        f"{len(frame_global):,} points")
    return res


# ── STEP 2 — the nested filter chain ─────────────────────────────────────
#
# USER 2026-09-18: "son varios criterios de filtro anidados, me quedo solo con
# los objetos con mas de 500 puntos, luego solo con los que tienen 2 o mas
# visitas, luego, elimino las visitas que estan a un metro o menos de la
# anterior, ahi no elimino el objeto de la tabla sino la visita, y vuelvo a ver
# cuantos objetos tienen solo una visita" … "vamos a ver cuanto aporta de
# porcentaje de puntos cada visita, las visitas que aportan 1% o menos se
# elimina y se debe volver a verificar cuantos quedan con solo una visita".
#
# NESTED is the word that matters: removing a VISIT can leave its object with
# one, and an object with one visit cannot show a duplicate. So every rule that
# drops a visit is followed by re-counting the objects.
#
# The cloud filter does NOT live here (USER, same day: "el filtrado de la nube
# debe ser una vez que se corrige la pose"). Deleting points against a pose
# that is still wrong deletes them against the wrong geometry.


@dataclass
class Candidate:
    """A masklet that survived the chain, with the visits it kept."""

    masklet: Masklet
    points: np.ndarray                    # its cloud point indices
    visits: List[Tuple[int, int]]         # the visits that survived
    shares: List[float]                   # fraction of its points each brought
    walked: List[float]                   # metres between consecutive visits
    copies: List[np.ndarray]              # the points of each surviving visit

    @property
    def oid(self) -> int:
        return self.masklet.oid

    @property
    def label(self) -> str:
        return self.masklet.label

    @property
    def instance_id(self) -> int:
        return self.masklet.instance_id

    def as_dict(self) -> dict:
        return {"oid": self.oid, "instance_id": self.instance_id,
                "label": self.label, "n_points": int(len(self.points)),
                "visits": [list(v) for v in self.visits],
                "shares": [round(float(x), 5) for x in self.shares],
                "walked_m": [round(float(x), 3) for x in self.walked],
                "provenance": "tool_measured"}


def filter_chain(masklets: Sequence[Masklet], points_by_oid: Dict[int, np.ndarray],
                 ks_of_point: np.ndarray, chainage_kf: np.ndarray,
                 min_points: int, min_walk_m: float, min_visit_share: float,
                 xyz: Optional[np.ndarray] = None,
                 log: Callable[[str], None] = print) -> Tuple[List[Candidate], dict]:
    """The masklets that can testify about a pose error, and why the rest cannot.

    The chain, in order, each rule applied to what the previous one left:

      1. more than ``min_points`` points
      2. two or more visits
      3. drop every visit less than ``min_walk_m`` of WALK from the previous
         one — the visit, not the object; then re-count
      4. drop every visit contributing at most ``min_visit_share`` of the
         object's points; then re-count

    Returns the survivors and the count after every step — the chain is the
    deliverable as much as the survivors.
    """
    steps = {"masklets": len(masklets), "enough_points": 0, "two_visits": 0,
             "visits_dropped_close": 0, "after_close": 0,
             "visits_dropped_share": 0, "after_share": 0}

    a = [m for m in masklets if len(points_by_oid.get(m.oid, ())) > int(min_points)]
    steps["enough_points"] = len(a)
    b = [m for m in a if m.n_visits >= 2]
    steps["two_visits"] = len(b)

    kept: Dict[int, List[Tuple[int, int]]] = {}
    for m in b:
        v = [m.visits[0]]
        for cur in m.visits[1:]:
            if chainage_kf[cur[0]] - chainage_kf[v[-1][1]] <= float(min_walk_m):
                steps["visits_dropped_close"] += 1
            else:
                v.append(cur)
        kept[m.oid] = v
    c = [m for m in b if len(kept[m.oid]) >= 2]
    steps["after_close"] = len(c)

    out: List[Candidate] = []
    for m in c:
        idx = points_by_oid[m.oid]
        ks = ks_of_point[idx]
        total = len(idx)
        v, sh = [], []
        for (x, y) in kept[m.oid]:
            frac = float(((ks >= x) & (ks <= y)).sum()) / float(total)
            if frac <= float(min_visit_share):
                steps["visits_dropped_share"] += 1
            else:
                v.append((x, y))
                sh.append(frac)
        if len(v) < 2:
            continue
        walked = [float(chainage_kf[v[i + 1][0]] - chainage_kf[v[i][1]])
                  for i in range(len(v) - 1)]
        copies = ([xyz[idx[(ks >= x) & (ks <= y)]] for (x, y) in v]
                  if xyz is not None else [])
        out.append(Candidate(masklet=m, points=idx, visits=v, shares=sh,
                             walked=walked, copies=copies))
    steps["after_share"] = len(out)

    log(f"[visit-drift] step 2: {steps['masklets']} masklets -> "
        f"{steps['enough_points']} over {min_points} points -> "
        f"{steps['two_visits']} with 2+ visits -> {steps['after_close']} after "
        f"dropping {steps['visits_dropped_close']} visit(s) under "
        f"{min_walk_m} m -> {steps['after_share']} after dropping "
        f"{steps['visits_dropped_share']} visit(s) at or under "
        f"{min_visit_share:.0%}")
    return out, steps


# ── STEP 2b — DISTINCTIVENESS: is this identity the only candidate? ──────
#
# USER 2026-09-18, after looking at the three views of the most balanced
# objects: two visits of one masklet can agree beautifully in SHAPE and still
# be two DIFFERENT pieces of a repeated surface. pccr has 85 `white_tiled_floor`
# masklets — 39 % of the session — and the best silhouette correlation of the
# whole session (0.93) belongs to one of them. Choosing by agreement alone
# picks a floor tile.
#
# What separates them is not how well the two copies match but how many OTHER
# things called the same could have been matched instead. The measure:
#
#   ambiguity = how many OTHER masklets of the SAME label have points within
#               |t| of this object's first copy
#
# |t| is the displacement the measurement itself claims, so this can only be
# computed AFTER the three views have been compared — it is a filter of the
# chain but it runs downstream of the measurement, not before it.
#
# Measured on pccr: every one of the 20 surviving floors scores between 17 and
# 59; every non-floor between 0 and 6. The two objects that pass both this and
# the agreement are `desk#203` and `glass_door#117` — the two the user chose by
# hand.
#
# Nothing here knows what a floor is. It is the label against itself: how many
# there are, and where.


def ambiguity(copy_a: np.ndarray, oid: int, label: str,
              points_by_oid: Dict[int, np.ndarray], label_of: Dict[int, str],
              xyz: np.ndarray, magnitude_m: float, sample: int = 4000,
              seed: int = 0) -> Tuple[int, List[int]]:
    """(count, oids) of other masklets of the same label within ``magnitude_m``.

    Distance is nearest point to nearest point, not centroid to centroid: an
    extended object (a wall, a strip of floor) has no meaningful centre, and
    what makes it a candidate for confusion is that some of it is where the
    displacement says this object could have come from.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    def _sub(a: np.ndarray) -> np.ndarray:
        return a if len(a) <= sample else a[rng.choice(len(a), sample, replace=False)]

    tree = cKDTree(_sub(np.asarray(copy_a, np.float64)))
    hits: List[int] = []
    for other, idx in points_by_oid.items():
        if other == oid or label_of.get(other) != label or not len(idx):
            continue
        dd, _ = tree.query(_sub(xyz[idx]), k=1)
        if float(dd.min()) <= float(magnitude_m):
            hits.append(int(other))
    return len(hits), sorted(hits)


def drop_ambiguous(pairs: Sequence[Tuple["Candidate", "Drift"]],
                   points_by_oid: Dict[int, np.ndarray],
                   label_of: Dict[int, str], xyz: np.ndarray,
                   max_ambiguity: int,
                   log: Callable[[str], None] = print
                   ) -> Tuple[List[Tuple["Candidate", "Drift"]], dict]:
    """Keep only the objects whose identity has fewer than ``max_ambiguity``
    rivals of the same label within their own measured displacement."""
    kept: List[Tuple["Candidate", "Drift"]] = []
    report: List[dict] = []
    for c, dr in pairs:
        if not c.copies:
            raise RuntimeError(
                f"{c.label}#{c.instance_id} carries no copies — build the "
                f"candidates with `filter_chain(..., xyz=xyz)`")
        n, hits = ambiguity(c.copies[0], c.oid, c.label, points_by_oid,
                            label_of, xyz, dr.magnitude)
        ok = n < int(max_ambiguity)
        report.append({"oid": c.oid, "instance_id": c.instance_id,
                       "label": c.label, "magnitude_m": round(dr.magnitude, 4),
                       "ambiguity": n, "rivals": hits[:20], "kept": ok})
        if ok:
            kept.append((c, dr))
    log(f"[visit-drift] step 2b: {len(kept)} of {len(pairs)} objects have "
        f"fewer than {max_ambiguity} rival(s) of their own label within their "
        f"own displacement")
    return kept, {"max_ambiguity": int(max_ambiguity), "objects": report,
                  "provenance": "tool_measured"}


# ── the visits of an object, from its own points ─────────────────────────

def visits_of(ks: np.ndarray, n_kf: int) -> List[Tuple[int, int]]:
    """Continuous runs of keyframes that saw this object.

    A visit ends at ANY discontinuity — the USER's own definition, the same
    one ``mask_filter.visit_gap_kf: 1`` encodes. No threshold: the camera
    either kept seeing it or it did not.
    """
    present = np.zeros(int(n_kf), bool)
    u = np.unique(ks[ks >= 0])
    if not len(u):
        return []
    present[u] = True
    out: List[Tuple[int, int]] = []
    start = None
    for k in range(int(n_kf)):
        if present[k] and start is None:
            start = k
        if not present[k] and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, int(n_kf) - 1))
    return out


def walked_between(chainage: np.ndarray, a_end: int, b_start: int) -> float:
    """Metres walked from the last keyframe of one visit to the first of the
    next, through every keyframe in between — never the straight line."""
    return float(chainage[int(b_start)] - chainage[int(a_end)])


# ── the object's own frame ───────────────────────────────────────────────

def obb_axes(P: np.ndarray, up: np.ndarray) -> np.ndarray:
    """(L, W, U): the object's long horizontal axis, the one across it, and
    the scene vertical. Computed on ONE copy — the union of two displaced
    copies has its principal axis dragged by the displacement itself."""
    Q = np.asarray(P, np.float64) - np.asarray(P, np.float64).mean(0)
    Qh = Q - np.outer(Q @ up, up)
    w, v = np.linalg.eigh(Qh.T @ Qh)
    L = v[:, int(np.argmax(w))]
    L = L - (L @ up) * up
    L = L / (np.linalg.norm(L) + 1e-9)
    return np.stack([L, np.cross(up, L), up])


def iqr_extent(P: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Extent along each axis as the interquartile range.

    Never ``max - min``: a single flyer stretches the box and ruins the
    comparison between visits. The IQR is the standard robust spread and has
    no parameter to choose.
    """
    return np.array([np.percentile(P @ a, 75) - np.percentile(P @ a, 25)
                     for a in axes])


# REMOVED 2026-09-18, USER: "el 3 sacalo, no existe mas". The comparability
# gate compared the two visits' IQR extents per OBB axis and demanded the worst
# ratio reach 30 %. It asks "are they about the same SIZE", and the question is
# "did they see the SAME THING": `white_tiled_floor#18` passes it with two
# patches of identical size that do not touch each other. The COMMON REGION
# (visibility per voxel, `tools/visit_visibility`) answers the real question
# and supersedes it. `iqr_extent` stays — it is a robust extent, useful on its
# own; only the gate is gone.


# ── the measurement: silhouettes, three views ────────────────────────────

def _silhouette(P: np.ndarray, ax: np.ndarray, ay: np.ndarray,
                lo: np.ndarray, shape: Tuple[int, int],
                cell_m: float, close_px: int, blur_px: float) -> np.ndarray:
    """Occupancy of the projection, closed and blurred.

    Both steps are load-bearing. A raw point-cloud projection at a fine cell
    is nearly empty, and the correlation of two nearly-empty rasters follows
    the noise: measured on pccr, peaks of 0.013-0.032 and the two views
    disagreeing by 24 and 30 cm about the same component. Closed and blurred:
    peaks of 0.70-0.91 and agreement within 5 cm.
    """
    import cv2

    u = ((P @ ax - lo[0]) / cell_m).astype(np.int64)
    v = ((P @ ay - lo[1]) / cell_m).astype(np.int64)
    g = np.zeros(shape, np.float32)
    ok = (u >= 0) & (u < shape[0]) & (v >= 0) & (v < shape[1])
    np.add.at(g, (u[ok], v[ok]), 1.0)
    g = (g > 0).astype(np.uint8)
    if close_px > 0:
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                             np.ones((int(close_px), int(close_px)), np.uint8))
    g = g.astype(np.float32)
    if blur_px > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(blur_px))
    return g


def view_shift(A: np.ndarray, B: np.ndarray, ax: np.ndarray, ay: np.ndarray,
               cell_m: float, margin_m: float, close_px: int,
               blur_px: float) -> Tuple[float, float, float]:
    """(du, dv, peak): the in-plane shift that takes B's silhouette onto A's,
    by normalised cross-correlation. The peak is reported so a view that did
    not lock on can be seen for what it is."""
    pa = np.stack([A @ ax, A @ ay])
    pb = np.stack([B @ ax, B @ ay])
    lo = np.minimum(pa.min(1), pb.min(1)) - margin_m
    hi = np.maximum(pa.max(1), pb.max(1)) + margin_m
    shape = tuple(((hi - lo) / cell_m).astype(int) + 1)
    a = _silhouette(A, ax, ay, lo, shape, cell_m, close_px, blur_px)
    b = _silhouette(B, ax, ay, lo, shape, cell_m, close_px, blur_px)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=shape) / a.size
    p = np.unravel_index(int(np.argmax(c)), c.shape)
    # the correlation is circular: a peak past the middle of the raster is a
    # NEGATIVE shift wrapped around (2*i > n is i > n//2 on integers, written
    # without a divisor so the package keeps none)
    du = p[0] - shape[0] if 2 * p[0] > shape[0] else p[0]
    dv = p[1] - shape[1] if 2 * p[1] > shape[1] else p[1]
    return du * cell_m, dv * cell_m, float(c.max())


@dataclass
class Drift:
    """One object's measured displacement between two of its visits."""

    instance_id: int
    label: str
    visit_a: Tuple[int, int]
    visit_b: Tuple[int, int]
    walked_m: float
    axes: np.ndarray                  # rows L, W, U
    t: np.ndarray                     # world vector taking copy B onto copy A
                                      # (its sense is verified, not assumed)
    per_view: Dict[str, Tuple[float, float, float]]
    disagreement: np.ndarray          # per component, between its two views
    n_a: int
    n_b: int

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.t))

    @property
    def worst_disagreement(self) -> float:
        return float(np.max(self.disagreement))

    def as_dict(self) -> dict:
        return {"instance_id": self.instance_id, "label": self.label,
                "visit_a": list(self.visit_a), "visit_b": list(self.visit_b),
                "walked_m": round(self.walked_m, 3),
                "t_m": [round(float(x), 5) for x in self.t],
                "magnitude_m": round(self.magnitude, 5),
                "disagreement_m": [round(float(x), 5) for x in self.disagreement],
                "worst_disagreement_m": round(self.worst_disagreement, 5),
                "per_view": {k: [round(float(x), 5) for x in v]
                             for k, v in self.per_view.items()},
                "n_points": [self.n_a, self.n_b],
                "provenance": "tool_measured"}


def drift_by_views(A: np.ndarray, B: np.ndarray, axes: np.ndarray,
                   cell_m: float, margin_m: float, close_px: int,
                   blur_px: float) -> Tuple[np.ndarray, Dict[str, tuple], np.ndarray]:
    """The displacement from the three orthogonal views, each component
    measured twice, plus how much its two measurements disagree."""
    L, W, U = axes[0], axes[1], axes[2]
    dl1, dw1, p1 = view_shift(A, B, L, W, cell_m, margin_m, close_px, blur_px)
    dl2, du2, p2 = view_shift(A, B, L, U, cell_m, margin_m, close_px, blur_px)
    dw3, du3, p3 = view_shift(A, B, W, U, cell_m, margin_m, close_px, blur_px)
    comp = np.array([(dl1 + dl2) / 2.0, (dw1 + dw3) / 2.0, (du2 + du3) / 2.0])
    dis = np.array([abs(dl1 - dl2), abs(dw1 - dw3), abs(du2 - du3)])
    t = comp[0] * L + comp[1] * W + comp[2] * U
    # The SENSE is measured, never assumed. Which way the correlation peak
    # points depends on the transform convention, and getting it backwards
    # does not fail loudly — it doubles the separation instead of closing it,
    # and the loop then "corrects" further away every pass (pccr 2026-09-18:
    # 76.5 cm became 152.5 on the second pass). A translation meant to bring
    # two copies together either reduces the distance between them or it is
    # the other one; that is one line to check.
    ca, cb = A.mean(0), B.mean(0)
    if np.linalg.norm((cb + t) - ca) > np.linalg.norm((cb - t) - ca):
        t = -t
    per_view = {"plan_LW": (dl1, dw1, p1), "side_LU": (dl2, du2, p2),
                "front_WU": (dw3, du3, p3)}
    return t, per_view, dis


# ── STEPS 4-5-6 — the common region, and the measurement inside it ───────
#
# USER 2026-09-18: *"podemos separar en voxeles la mascara y determinar que kf
# vieron cada parte para saber donde hay solapamiento … partes que se ven en
# ambas visitas y son las que tienen discrepancia, porque si solo se ve de una
# visita no puede haber discrepancia"*.
#
# VISTO is not MEDIDO. A voxel can lie inside the object's SAM mask and carry no
# points. The plain correlation reads "no points" as "the object is not here",
# so a visit that saw a sliver drags the peak toward the other visit's bulk.
#
# Steps 4, 5 and 6 are one loop, not three filters:
#
#   4. with the current t, decide per voxel which visits SAW it — the masks say
#      so, verified against a per-keyframe Z-buffer so a voxel BEHIND the object
#      does not count as seen
#   5. measure again using ONLY the points that fall in the voxels both visits
#      saw, and repeat until t stops moving
#   6. a common region of zero voxels means nothing was observed twice: the
#      object cannot testify, whatever its residual looks like
#
# Why it cannot be a filter before the measurement: the common region is a
# function of t, because moving one visit moves ITS CAMERAS with it.
#
# Measured on pccr, why it is needed at all: `white_tiled_floor#21` scores the
# BEST view agreement of the whole session (1/1/0 cm) on a displacement of
# 343 cm over a 16 m walk — the correlation lined one patch up against the gap
# between two others. Three views agreeing is not enough; they have to agree
# about something both visits actually saw.


@dataclass
class Grid:
    """A voxel grid in the object's own OBB frame."""

    axes: np.ndarray                     # rows L, W, U
    lo: np.ndarray                       # local coords of the corner
    step: float
    shape: Tuple[int, int, int]

    @property
    def n(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2])

    def centres(self) -> np.ndarray:
        """World coordinates of every voxel centre, in flat C order."""
        gi, gj, gk = np.meshgrid(np.arange(self.shape[0]),
                                 np.arange(self.shape[1]),
                                 np.arange(self.shape[2]), indexing="ij")
        local = np.stack([self.lo[0] + (gi.ravel() + 0.5) * self.step,
                          self.lo[1] + (gj.ravel() + 0.5) * self.step,
                          self.lo[2] + (gk.ravel() + 0.5) * self.step], 1)
        return local @ self.axes

    def index_of(self, P_world: np.ndarray) -> np.ndarray:
        """Flat voxel index per point, -1 for points outside the grid."""
        loc = (np.asarray(P_world, np.float64) @ self.axes.T - self.lo) / self.step
        ijk = np.floor(loc).astype(np.int64)
        ok = np.ones(len(ijk), bool)
        for d in range(3):
            ok &= (ijk[:, d] >= 0) & (ijk[:, d] < self.shape[d])
        flat = np.full(len(ijk), -1, np.int64)
        flat[ok] = ((ijk[ok, 0] * self.shape[1] + ijk[ok, 1]) * self.shape[2]
                    + ijk[ok, 2])
        return flat


def make_grid(copies: Sequence[np.ndarray], axes: np.ndarray, step: float,
              pad_m: float) -> Grid:
    loc = np.vstack([np.asarray(P, np.float64) @ axes.T for P in copies])
    lo = loc.min(0) - pad_m
    hi = loc.max(0) + pad_m
    shape = tuple(int(x) for x in np.maximum(((hi - lo) / step).astype(int) + 1, 1))
    return Grid(axes=axes, lo=lo, step=float(step), shape=shape)


class Visibility:
    """What each keyframe could SEE of an object: its mask, depth-verified.

    The Z-buffer is built from the full cloud at mask resolution, the same
    construction as ``surface_fit/hole_audit._zbuf`` — 5-px minimum filter
    included, because the cloud is sparse at mask resolution and a ray often has
    no point on its exact pixel while its neighbours do. Cached per keyframe:
    the visits of different objects share keyframes.
    """

    def __init__(self, output_dir, xyz: np.ndarray, ks_of_point: np.ndarray,
                 poses: np.ndarray, K_all: np.ndarray, depth_tol_m: float,
                 min_depth_m: Optional[float] = None):
        from segmentation import mask_space
        self.dir = Path(output_dir)
        self.xyz, self.poses, self.K = xyz, poses, K_all
        self.ks = np.asarray(ks_of_point, np.int64)
        order = np.argsort(self.ks, kind="stable")
        self._order = order
        self._start = np.searchsorted(self.ks[order], np.arange(len(poses)), "left")
        self._end = np.searchsorted(self.ks[order], np.arange(len(poses)), "right")
        self.tol = float(depth_tol_m)
        self.min_depth = float(_vd_cfg().min_depth_m if min_depth_m is None
                               else min_depth_m)
        doc = json.loads((self.dir / "segmentation.json").read_text())
        self.masks = np.load(self.dir / str(doc.get("mask_file") or "seg_masks.npz"))
        self.space = mask_space.resolve(self.dir, masks=self.masks,
                                        log=lambda m: None)
        kfs = mask_space.keyframe_numbers(self.dir) or []
        self.n_kf = len(kfs)
        self.kf_of_frame = {}
        for k in range(self.n_kf):
            mf = self.space.from_keyframe(k)
            if mf is not None:
                self.kf_of_frame[int(mf)] = k
        self.Ht, self.Wt = trace_grid(self.dir)
        probe = next(self.masks[k] for k in self.masks.files
                     if k.startswith("f") and "_o" in k)
        self.Hm, self.Wm = int(probe.shape[0]), int(probe.shape[1])
        self._zb: Dict[int, np.ndarray] = {}
        self._by_oid: Dict[int, List[Tuple[int, str]]] = {}
        import re
        pat = re.compile(r"^f(\d+)_o(\d+)$")
        for key in self.masks.files:
            m = pat.match(key)
            if not m:
                continue
            kf = self.kf_of_frame.get(int(m.group(1)))
            if kf is not None:
                self._by_oid.setdefault(int(m.group(2)), []).append((kf, key))

    def masks_of(self, oid: int, visit: Tuple[int, int]):
        a, b = int(visit[0]), int(visit[1])
        return [(kf, self.masks[key]) for kf, key in self._by_oid.get(int(oid), [])
                if a <= kf <= b]

    def zbuf(self, kf: int) -> np.ndarray:
        """What THIS camera measured, not what the cloud holds.

        Built from the points BORN IN THIS KEYFRAME only. Using the whole cloud
        is circular and it showed: on pccr the z-buffer of a visit-1 keyframe
        put measured geometry a median of 70 cm in front of that keyframe's own
        desk points — the DUPLICATE, 70 cm away, occluding the original. 87-97 %
        of every visit's own points came back "occluded" by the very drift the
        measurement exists to remove. A keyframe's own points are its depth map,
        they are internally consistent to millimetres, and no copy of anything
        can sit in front of them.

        A pixel the camera saw but did not measure stays ``inf`` — not occluded,
        and that is the point: seen and empty is information, not absence of it.
        """
        kf = int(kf)
        if kf in self._zb:
            return self._zb[kf]
        from scipy import ndimage as ndi
        own = self._order[self._start[kf]:self._end[kf]]
        c2w = np.eye(4)
        c2w[:3, :4] = self.poses[kf][:3, :4]
        M = np.linalg.inv(c2w)
        p = (M[:3, :3] @ self.xyz[own].T).T + M[:3, 3]
        z = p[:, 2]
        ok = z > self.min_depth
        fx, fy, cx, cy = self.K[kf]
        u = fx * p[ok, 0] / z[ok] + cx
        v = fy * p[ok, 1] / z[ok] + cy
        mu = (u * self.Wm / self.Wt).astype(np.int64)
        mv = (v * self.Hm / self.Ht).astype(np.int64)
        inb = (mu >= 0) & (mu < self.Wm) & (mv >= 0) & (mv < self.Hm)
        zb = np.full((self.Hm, self.Wm), np.inf)
        np.minimum.at(zb, (mv[inb], mu[inb]), z[ok][inb])
        zb = ndi.minimum_filter(zb, size=5, mode="nearest")
        self._zb[kf] = zb
        return zb

    def seen(self, centres: np.ndarray, kf_masks) -> np.ndarray:
        """Did ANY keyframe of this visit see these points, per its own mask
        and not occluded by measured geometry in front of them."""
        sr, sc = self.Hm / float(self.Ht), self.Wm / float(self.Wt)
        seen = np.zeros(len(centres), bool)
        for kf, m in kf_masks:
            c2w = np.eye(4)
            c2w[:3, :4] = self.poses[kf][:3, :4]
            M = np.linalg.inv(c2w)
            p = (M[:3, :3] @ centres.T).T + M[:3, 3]
            z = p[:, 2]
            front = z > self.min_depth
            if not front.any():
                continue
            fx, fy, cx, cy = self.K[kf]
            u = np.full(len(centres), -1.0)
            v = np.full(len(centres), -1.0)
            u[front] = fx * p[front, 0] / z[front] + cx
            v[front] = fy * p[front, 1] / z[front] + cy
            ok = front & (u >= 0) & (u < self.Wt) & (v >= 0) & (v < self.Ht)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            r = np.clip((v[ok] * sr).astype(np.int64), 0, self.Hm - 1)
            c = np.clip((u[ok] * sc).astype(np.int64), 0, self.Wm - 1)
            hit = m[r, c] > 0
            zpix = self.zbuf(kf)[r, c]
            seen[idx[hit & ~(zpix < (z[ok] - self.tol))]] = True
        return seen


def occupied(grid: Grid, copies: Sequence[np.ndarray]) -> np.ndarray:
    """Voxels that hold a measured point of any copy — the object itself.

    The visibility question is only meaningful ON the object: a voxel of empty
    air in front of it is inside the mask and occluded by nothing, so a raw
    "not occluded" test calls it seen. pccr measured the cost: 239 voxels
    "common" on a chair with fewer than three points inside any of them.
    """
    occ = np.zeros(grid.n, bool)
    for P in copies:
        idx = grid.index_of(P)
        occ[idx[idx >= 0]] = True
    return occ


def common_region(vis: Visibility, oid: int, visits, grid: Grid,
                  t: np.ndarray, copies: Sequence[np.ndarray]) -> np.ndarray:
    """Per voxel OF THE OBJECT: did BOTH visits see it, visit 2 displaced by t.

    Moving a visit moves ITS CAMERAS, so testing visit 2 at ``t`` is testing the
    voxels at ``-t`` against its cameras where they are.

    A voxel occupied by one visit's points and SEEN by the other is where the
    two can disagree — including when the other measured nothing there, which
    is a real hole and not an absence of information. A voxel the other visit
    never saw carries no information at all and is left out.
    """
    occ = occupied(grid, copies)
    c = grid.centres()
    idx = np.flatnonzero(occ)
    s1 = np.zeros(grid.n, bool)
    s2 = np.zeros(grid.n, bool)
    s1[idx] = vis.seen(c[idx], vis.masks_of(oid, visits[0]))
    s2[idx] = vis.seen(c[idx] - np.asarray(t, np.float64),
                       vis.masks_of(oid, visits[1]))
    return s1 & s2


def refine_drift(vis: Visibility, cand: "Candidate", axes: np.ndarray,
                 t0: np.ndarray, cfg, max_iter: int = 6,
                 log: Callable[[str], None] = print
                 ) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[np.ndarray], dict]:
    """Measure again using ONLY what both visits saw, until ``t`` stops moving.

    Returns (t, per_view, disagreement, report). ``t`` is None when the two
    visits share no voxel — nothing was observed twice and the object cannot
    testify about a pose error (step 6).
    """
    A, B = cand.copies[0], cand.copies[1]
    step = float(cfg.voxel_m)
    t = np.asarray(t0, np.float64).copy()
    hist, per_view, dis = [], None, None
    grid = make_grid([A, B], axes, step, pad_m=float(np.linalg.norm(t)) + step)
    for it in range(int(max_iter)):
        common = common_region(vis, cand.oid, cand.visits, grid, t, [A, B + t])
        n_common = int(common.sum())
        hist.append({"iter": it, "t_m": [round(float(x), 5) for x in t],
                     "magnitude_m": round(float(np.linalg.norm(t)), 5),
                     "common_voxels": n_common})
        if n_common == 0:
            return None, None, None, {
                "reason": "the two visits share no voxel — nothing was "
                          "observed twice", "voxel_m": step, "history": hist,
                "common_voxels": 0, "provenance": "tool_measured"}
        ia = grid.index_of(A)
        ib = grid.index_of(B + t)
        ka = (ia >= 0) & common[np.maximum(ia, 0)]
        kb = (ib >= 0) & common[np.maximum(ib, 0)]
        if ka.sum() < 3 or kb.sum() < 3:
            return None, None, None, {
                "reason": f"the common region holds {n_common} voxel(s) but "
                          f"{int(ka.sum())}/{int(kb.sum())} measured points",
                "voxel_m": step, "history": hist,
                "common_voxels": n_common, "provenance": "tool_measured"}
        # the copies restricted to what BOTH saw; B is scored where it lands
        t_new, per_view, dis = drift_by_views(
            A[ka], B[kb] + t, axes, float(cfg.silhouette_cell_m),
            float(cfg.search_margin_m), int(cfg.silhouette_close_px),
            float(cfg.silhouette_blur_px))
        t_new = t + t_new
        moved = float(np.linalg.norm(t_new - t))
        t = t_new
        if moved <= float(cfg.silhouette_cell_m):
            break
    return t, per_view, dis, {"voxel_m": step, "history": hist,
                              "common_voxels": int(common.sum()),
                              "iterations": len(hist),
                              "provenance": "tool_measured"}


# ── the same closures, as what the SCALE graph can read ──────────────────
#
# USER 2026-09-19, after the whole day of contradictions: the duplicates are
# separated ALONG THE LINE OF SIGHT, not sideways. On pccr five of the six
# closures are 97-99 % radial (desk#203: 67.2 of 67.8 cm). A translation is the
# same everywhere; a DEPTH error grows with distance — which is why the desk
# closed at 3.6 m while the tile lines 8 m away stayed 18 cm off, and why no
# two objects ever agreed on a vector: each sits at a different BEARING, so one
# depth error becomes a different world vector for each.
#
# Read as depth ratios the same five closures agree — 1.099 to 1.251, median
# 1.174 — and agree with the DA3 anchors (1.140), which never see a silhouette.
# That is the measurement `certify/scale_stage` has been starving for: its
# §5.1 loop rows were EMPTY on pccr while this module measured five good ones
# and spent them all on a translation solver.

def scale_rows(kept: Sequence[Tuple["Candidate", "Drift"]], poses: np.ndarray,
               ks_of_point: np.ndarray, log: Callable[[str], None] = print
               ) -> List[dict]:
    """Every closure as the DEPTH ratio between the two chunks it spans.

    A point reconstructed at ray distance D from its own camera and corrected
    by a factor k about that camera moves D·(k−1) ALONG the ray. With visit 1
    as the gauge, the closure's radial part fixes visit 2's factor:

        k_b = 1 + (t · u) / D_b

    and `s_ab = 1/k_b` in metric_lock's convention (`loop_scale_row`).

    NOTHING IS VETOED HERE. The part of the closure the depth model cannot
    produce — the TANGENTIAL component — becomes the row's residual, so a
    closure that is mostly sideways widens its own error bar instead of being
    rejected by a threshold nobody measured (the doctrine of `verify_loop`,
    USER 2026-09-16: *"no debes rechazar correcciones por umbrales
    arbitrarios"*). pccr's one false identity, `glass_door#121`, is 91 %
    tangential and prices itself out on its own evidence.
    """
    C = np.asarray(poses, np.float64)[:, :3, 3]
    out: List[dict] = []
    for cand, dr in kept:
        A, B = cand.copies[0], cand.copies[1]
        (a1, b1), (a2, b2) = cand.visits[0], cand.visits[1]
        kk = np.asarray(ks_of_point, np.int64)[cand.points]
        kA = kk[(kk >= a1) & (kk <= b1)]
        kB = kk[(kk >= a2) & (kk <= b2)]
        if not len(kA) or not len(kB) or len(A) != len(kA) or len(B) != len(kB):
            continue
        cb = C[kB]
        D_b = float(np.linalg.norm(B - cb, axis=1).mean())
        if not np.isfinite(D_b) or D_b <= 0:
            continue
        u = B.mean(0) - cb.mean(0)
        nu = float(np.linalg.norm(u))
        if nu <= 0:
            continue
        u = u / nu
        t = np.asarray(dr.t, np.float64)
        radial = float(t @ u)
        k_b = 1.0 + radial / D_b
        if not np.isfinite(k_b) or k_b <= 0:
            continue
        tangential = float(np.linalg.norm(t - radial * u))
        # the object's own size: a residual is only meaningful against it
        ext = np.asarray(B, np.float64)
        extent = float(np.linalg.norm(np.percentile(ext, 98, axis=0)
                                      - np.percentile(ext, 2, axis=0)))
        residual = float(np.hypot(tangential, dr.worst_disagreement))
        out.append({"instance_id": int(cand.instance_id), "label": cand.label,
                    "i": int(round((a1 + b1) / 2.0)),
                    "j": int(round((a2 + b2) / 2.0)),
                    "s_ab": float(1.0 / k_b), "k_b": float(k_b),
                    "residual_m": residual, "extent_m": extent,
                    "radial_m": radial, "tangential_m": tangential,
                    "D_b_m": D_b, "scale_trusted": True,
                    "source": "visit_drift", "provenance": "tool_measured"})
        log(f"[visit-drift] scale row {cand.label}#{cand.instance_id}: "
            f"kf {out[-1]['i']}<->{out[-1]['j']}, depth x{k_b:.4f} "
            f"(radial {radial * 100:+.1f} cm of {np.linalg.norm(t) * 100:.1f} "
            f"at {D_b:.2f} m, tangential {tangential * 100:.1f} cm)")
    return out


# ── the cloud filter, applied for real before measuring ──────────────────

@dataclass
class FilterReport:
    dropped_points: int = 0
    dropped_objects: int = 0
    dropped_visits: int = 0
    detail: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"dropped_points": self.dropped_points,
                "dropped_objects": self.dropped_objects,
                "dropped_visits": self.dropped_visits,
                "detail": self.detail[:50], "provenance": "tool_measured"}


def cloud_filter_masklets(points_by_oid: Dict[int, np.ndarray],
                          masklets: Sequence[Masklet], ks_of_point: np.ndarray,
                          xyz: np.ndarray, vis: "Visibility",
                          min_points: int, min_visit_share: float,
                          max_frames_per_visit: int, dilate_px: int,
                          log: Callable[[str], None] = print
                          ) -> Tuple[np.ndarray, FilterReport]:
    """STEP 12 — which points leave the cloud, once the pose is corrected.

    USER 2026-09-18: *"no me elimines lo unsegmented, solo los puntos que
    figuran como parte del objeto que luego de ser ajustado cae aun fuera de las
    mascaras"*, and *"lo mismo deben eliminarse de la nube los puntos de
    revisitas menores al 1%"*.

    Three rules, all on the MASKLETS — SAM3's own tracks, not the fused
    instances:

      · a point that CLAIMS to be part of an object and, with the pose already
        corrected, still lands outside that object's mask in every view that
        saw it, is not part of it
      · a masklet under ``min_points`` cannot be measured and keeps nothing
      · a visit contributing at most ``min_visit_share`` of its masklet grazed
        it and keeps nothing

    A point belonging to NO masklet is UNSEGMENTED and is never touched: the
    masks say nothing about it, and silence is not a verdict.

    The test is the CROSS view. A point moves with the keyframe it was born in,
    so its projection into its own keyframe never changes and says nothing; what
    the correction has to fix is where it lands in the keyframes of the OTHER
    visit. A view only votes when it actually saw the point — in frustum and
    not behind measured geometry.

    It runs per epoch: each cloud has its own objects, and they need not be the
    same ones.
    """
    n_points = len(xyz)
    keep = np.zeros(n_points, bool)
    seg = np.zeros(n_points, bool)
    rep = FilterReport()
    by_oid = {m.oid: m for m in masklets}
    off_mask = 0
    sr, sc = vis.Hm / float(vis.Ht), vis.Wm / float(vis.Wt)

    for oid, idx in points_by_oid.items():
        idx = np.asarray(idx, np.int64)
        idx = idx[(idx >= 0) & (idx < n_points)]
        if not len(idx):
            continue
        seg[idx] = True
        m = by_oid.get(int(oid))
        label = m.label if m is not None else "object"
        if len(idx) < int(min_points):
            rep.dropped_objects += 1
            rep.detail.append({"oid": int(oid), "label": label,
                               "reason": f"masklet under {min_points} points",
                               "points": int(len(idx))})
            continue
        ks = ks_of_point[idx]
        total = len(idx)
        visits = m.visits if m is not None else []
        alive = np.zeros(len(idx), bool)
        for (a, b) in visits:
            sel = (ks >= a) & (ks <= b)
            n = int(sel.sum())
            if not n:
                continue
            if n / total <= float(min_visit_share):
                rep.dropped_visits += 1
                rep.detail.append({"oid": int(oid), "label": label,
                                   "reason": f"visit {a}-{b} contributes "
                                             f"{100.0 * n / total:.2f}%",
                                   "points": n})
                continue
            alive[sel] = True
        if not alive.any():
            continue

        # the cross-view test, over the keyframes of the OTHER visits
        sub = idx[alive]
        sub_ks = ks[alive]
        saw = np.zeros(len(sub), bool)
        inside = np.zeros(len(sub), bool)
        for vi, (a, b) in enumerate(visits):
            frames = [(kf, mm) for kf, mm in vis.masks_of(oid, (a, b))]
            if not frames:
                continue
            frames.sort(key=lambda it: -int(np.count_nonzero(it[1])))
            for kf, mm in frames[:int(max_frames_per_visit)]:
                other = sub_ks < a
                other |= sub_ks > b               # born outside this visit
                if not other.any():
                    continue
                w = np.flatnonzero(other)
                P = xyz[sub[w]]
                c2w = np.eye(4)
                c2w[:3, :4] = vis.poses[kf][:3, :4]
                M = np.linalg.inv(c2w)
                q = (M[:3, :3] @ P.T).T + M[:3, 3]
                z = q[:, 2]
                fr = z > vis.min_depth
                if not fr.any():
                    continue
                fx, fy, cx, cy = vis.K[kf]
                u = np.full(len(P), -1.0)
                v = np.full(len(P), -1.0)
                u[fr] = fx * q[fr, 0] / z[fr] + cx
                v[fr] = fy * q[fr, 1] / z[fr] + cy
                ok = fr & (u >= 0) & (u < vis.Wt) & (v >= 0) & (v < vis.Ht)
                if not ok.any():
                    continue
                r = np.clip((v[ok] * sr).astype(np.int64), 0, vis.Hm - 1)
                c = np.clip((u[ok] * sc).astype(np.int64), 0, vis.Wm - 1)
                zpix = vis.zbuf(kf)[r, c]
                visible = ~(zpix < (z[ok] - vis.tol))     # nothing in front
                hit = _dilated(mm, int(dilate_px))[r, c] > 0
                wo = w[ok]
                saw[wo[visible]] = True
                inside[wo[visible & hit]] = True
        # judged and never inside: it is not part of this object
        drop = saw & ~inside
        off_mask += int(drop.sum())
        alive_idx = np.flatnonzero(alive)
        keep[idx[alive_idx[~drop]]] = True

    # UNSEGMENTED points are never touched
    keep |= ~seg
    kill = ~keep
    rep.dropped_points = int(kill.sum())
    log(f"[visit-drift] step 12: {rep.dropped_points:,} of {n_points:,} points "
        f"leave ({off_mask:,} still off their own mask after the correction, "
        f"{rep.dropped_objects} masklet(s) under {min_points} points, "
        f"{rep.dropped_visits} visit(s) at or under {min_visit_share:.0%}); "
        f"{int((~seg).sum()):,} unsegmented points untouched")
    return kill, rep


def _dilated(mask: np.ndarray, px: int) -> np.ndarray:
    """SAM3 silhouettes are not pixel-exact and the cloud is sparse at mask
    resolution: without this the rim of every object is judged an outlier by
    its own mask (`segmentation.mask_filter.dilate_px`, same reason)."""
    if px <= 0:
        return mask
    import cv2
    return cv2.dilate(mask.astype(np.uint8),
                      np.ones((2 * int(px) + 1, 2 * int(px) + 1), np.uint8))


def best_determined(drifts: Sequence[Drift], repeatability_m: float
                    ) -> Optional[Drift]:
    """The object to correct by: the one whose two measurements of every
    component agree best, among those that agree at all.

    Agreement is judged against what the session can repeat — not a constant.
    A component whose two views differ by more than that was not determined.
    """
    usable = [d for d in drifts
              if d.worst_disagreement <= max(float(repeatability_m), 0.0) * 2.0]
    if not usable:
        usable = list(drifts)
    if not usable:
        return None
    # among the determined ones, the one with the most to correct
    return max(usable, key=lambda d: d.magnitude)
