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
                       aspect_tol: Optional[float] = None,
                       mask_pixels: Optional[Tuple[np.ndarray, np.ndarray]] = None
                       ) -> Dict[int, np.ndarray]:
    """Which cloud points belong to each SAM3 masklet.

    The cloud's ``globalIndices`` are per FUSED instance, so a masklet has no
    points of its own until they are asked for. Every point carries the
    keyframe it was born in and the pixel it was born at, and the masklet's own
    mask for that keyframe says whether that pixel is inside it — so the
    association is a lookup, not an inference.

    Masklets overlap, so a point can belong to more than one; nothing here
    forces a winner.

    ``mask_pixels`` = (rows, cols): the birth pixels ALREADY on the mask grid, for a
    caller whose ``pixel_row``/``pixel_col`` do not live on the trace grid and that
    maps them exactly itself (precision/silhouette_filter.py: F6's undistorted native
    pixel through the session camera); the trace-grid scaling is then skipped and a
    negative entry is a pixel the mask grid does not cover (it belongs to nothing).
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

    probe = next(masks[k] for k in masks.files if k.startswith("f") and "_o" in k)
    Hm, Wm = int(probe.shape[0]), int(probe.shape[1])
    if mask_pixels is None:
        Ht, Wt = trace_grid(output_dir)
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
    else:
        mr = np.asarray(mask_pixels[0], np.int64)
        mc = np.asarray(mask_pixels[1], np.int64)
        covered = (mr >= 0) & (mr < Hm) & (mc >= 0) & (mc < Wm)
        ks = np.where(covered, ks, -1)                 # uncovered: no keyframe, no masklet
        rows, cols = np.clip(mr, 0, Hm - 1), np.clip(mc, 0, Wm - 1)
        log(f"[visit-drift] points -> masklets: birth pixels given on the mask grid "
            f"{Hm}x{Wm} ({int((~covered).sum()):,} not covered by it)")

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
                 log: Callable[[str], None] = print,
                 group_points: Optional[Dict[int, int]] = None
                 ) -> Tuple[List[Candidate], dict]:
    """The masklets that can testify about a pose error, and why the rest cannot.

    The chain, in order, each rule applied to what the previous one left:

      1. more than ``min_points`` points — of the FUSED OBJECT when
         ``group_points`` says which one the masklet ended up in (USER
         2026-09-22: a small mask that fuses into a big object is a piece of a
         big object, and the size test is about the object). Whether the
         masklet's own silhouette can then be measured is decided later, by
         the determination test, which is a measurement and not a cap
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

    def _size(m) -> int:
        own = len(points_by_oid.get(m.oid, ()))
        return int((group_points or {}).get(int(m.oid), own))

    a = [m for m in masklets if _size(m) > int(min_points)]
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


# ── THE CAMERA THE CLOUD WAS BUILT WITH ─────────────────────────────────
#
# AUDIT 2026-10-01 (object-edge definition, item #2): the masks were looked up —
# and the z-buffer built — with Omega's per-keyframe K from ``intrinsic.txt``
# (pccr: fx 354.7-400.3, cx 232, cy 416) while the cloud had been unprojected
# with F5's session camera (fx 391.87, cx 234.80, cy 414.06): 2.8 px off at the
# principal point and 13-23 px at the borders, against a 2-px rim tolerance.
# Every rim of every object was judged at the wrong pixel.
#
# Every projection of this module now goes through the camera that BUILT the
# cloud — ``output/camera.json`` (precision.camera: F0, refined by F5), the
# source precision/silhouette_filter.py projects with — through its lens and the
# exact grid maps of precision.camera. And that claim is MEASURED before anything
# is judged (``birth_systematic``): a point projected into its own birth keyframe
# must land on the birth pixel it carries. A camera that did not build the cloud
# leaves a SYSTEMATIC field there (affine in the pixel: a focal ratio and a
# principal-point offset); the consolidation's motion along the normal and the
# rounding of the stored pixel leave zero-mean scatter. When the systematic part
# exceeds the precision the birth pixel is stored at, the filter refuses to run
# instead of mixing two cameras.

# The birth pixel is stored as an INTEGER of the record grid: rounding alone
# cannot put the true projection farther than half a pixel per axis. A camera
# whose systematic misprojection exceeds it is not the camera of this cloud.
# (Not a decision: the resolution the provenance is written at.)
BIRTH_PIXEL_PRECISION_PX = 0.5


class CameraMismatchError(RuntimeError):
    """The cloud, its poses and the session camera do not describe ONE camera —
    the mask filter refuses to judge rather than mix them."""


def _c2w(pose: np.ndarray) -> np.ndarray:
    c2w = np.eye(4)
    c2w[:3, :4] = np.asarray(pose, np.float64)[:3, :4]
    return c2w


class SessionProjection:
    """The camera the cloud was built with, bound to the two grids this module
    reads: the RECORD grid the cloud's ``pixel_row``/``pixel_col`` live on (the
    trace grid ``intrinsic.txt`` describes; Omega's crop through
    ``precision.camera.grid_like``) and the SAM3 MASK grid (a full-frame resize of
    the native frame, ``precision.camera.mask_grid_for``).

    World → camera (c2w per keyframe) → K of the UNDISTORTED native frame →
    the lens (``precision.camera.distort_points``) → the native frame →
    ``native_to_grid`` → rounded to the pixel whose centre is nearest (pixel
    centres sit on integer coordinates everywhere in precision.camera).
    """

    def __init__(self, cam, record_grid, mask_grid, source: str):
        self.cam = cam
        self.K = cam.K()
        self.record_grid = record_grid
        self.mask_grid = mask_grid
        self.source = str(source)
        self._lens = bool(np.any(cam.dist()))

    @property
    def mask_hw(self) -> Tuple[int, int]:
        return int(self.mask_grid.h), int(self.mask_grid.w)

    def continuous(self, P_world: np.ndarray, pose: np.ndarray, min_depth: float,
                   grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(front, x, y, z): ``front`` per point (in front of the camera by more
        than ``min_depth``); x (column), y (row) on ``grid`` and the camera depth
        z, for the points in front only."""
        from precision.camera import distort_points, native_to_grid
        P = np.asarray(P_world, np.float64).reshape(-1, 3)
        M = np.linalg.inv(_c2w(pose))
        q = P @ M[:3, :3].T + M[:3, 3]
        z = q[:, 2]
        front = z > float(min_depth)
        if not front.any():
            e = np.zeros(0)
            return front, e, e, e
        zf = z[front]
        uv = np.stack([self.K[0, 0] * q[front, 0] / zf + self.K[0, 2],
                       self.K[1, 1] * q[front, 1] / zf + self.K[1, 2]], 1)
        if self._lens:
            uv = distort_points(uv, self.cam)
        g = native_to_grid(uv, grid)
        return front, g[:, 0], g[:, 1], zf

    def to_grid(self, P_world: np.ndarray, pose: np.ndarray, min_depth: float,
                grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(ok, rows, cols, z): ``ok`` per point (in front and inside ``grid``);
        the pixel and the camera depth for the ``ok`` points only."""
        front, x, y, z = self.continuous(P_world, pose, min_depth, grid)
        ok = np.zeros(len(front), bool)
        if not len(x):
            e = np.zeros(0, np.int64)
            return ok, e, e, np.zeros(0)
        c, r = np.rint(x), np.rint(y)
        inb = (np.isfinite(c) & np.isfinite(r) & (c >= 0) & (c < int(grid.w))
               & (r >= 0) & (r < int(grid.h)))
        ok[np.flatnonzero(front)[inb]] = True
        return ok, r[inb].astype(np.int64), c[inb].astype(np.int64), z[inb]

    def to_mask(self, P_world: np.ndarray, pose: np.ndarray, min_depth: float):
        """``to_grid`` on the SAM3 mask grid."""
        return self.to_grid(P_world, pose, min_depth, self.mask_grid)

    def record_to_mask(self, pixel_row: np.ndarray, pixel_col: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """A birth pixel of the record grid → the mask-grid pixel that shows it,
        through the exact grid maps (both grids are crops/resizes of the same
        native frame, so no lens is involved); -1 where the mask grid does not
        cover it. For ``points_of_masklets(mask_pixels=...)``: the scale-only
        mapping it does by itself ignores Omega's crop and truncates."""
        from precision.camera import grid_to_native, native_to_grid
        uv = np.stack([np.asarray(pixel_col, np.float64),
                       np.asarray(pixel_row, np.float64)], -1)
        g = native_to_grid(grid_to_native(uv, self.record_grid), self.mask_grid)
        c, r = np.rint(g[:, 0]), np.rint(g[:, 1])
        ok = ((c >= 0) & (c < int(self.mask_grid.w)) & (r >= 0)
              & (r < int(self.mask_grid.h)))
        return (np.where(ok, r, -1).astype(np.int64),
                np.where(ok, c, -1).astype(np.int64))


def birth_systematic(project: Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]],
                     xyz: np.ndarray, ks_of_point: np.ndarray, n_kf: int,
                     pixel_row: np.ndarray, pixel_col: np.ndarray,
                     grid_wh: Tuple[int, int]) -> dict:
    """The SYSTEMATIC misprojection of a camera over the cloud it is claimed to
    have built, measured on the points' own birth pixels.

    ``project(P, k)`` → (front, x, y) on the record grid for keyframe k. Per
    keyframe and per axis the birth residual is fitted as ``a + b·(p − centre)``
    — exactly the field a camera of another focal / principal point leaves (an
    unprojection with K' re-projected with K is affine in the pixel) — and the
    keyframe's systematic misprojection is the largest that fit predicts inside
    the frame, ``|a| + |b|·half-size``. Zero-mean scatter (the rounding of the
    stored pixel, the consolidation's motion along the normal) does not enter a
    fit over the keyframe's points. The session's value is the POINT-WEIGHTED
    median over keyframes: what the camera does to most of the cloud.
    """
    ks = np.asarray(ks_of_point, np.int64)
    pr = np.asarray(pixel_row, np.float64)
    pc = np.asarray(pixel_col, np.float64)
    W, H = int(grid_wh[0]), int(grid_wh[1])
    hx, hy = (W - 1) / 2.0, (H - 1) / 2.0
    valid = (ks >= 0) & (ks < int(n_kf))
    idx_all = np.flatnonzero(valid)
    order = idx_all[np.argsort(ks[idx_all], kind="stable")]
    kss = ks[order]
    cuts = np.flatnonzero(np.diff(kss)) + 1
    per_kf, weights, n_front, n_total = [], [], 0, 0
    abs_res = []
    for seg in (np.split(np.arange(len(order)), cuts) if len(order) else []):
        sel = order[seg]
        k = int(ks[sel[0]])
        front, x, y = project(xyz[sel], k)
        n_total += len(sel)
        if not len(x):
            continue
        s = sel[front]
        n_front += len(s)
        dx, dy = x - pc[s], y - pr[s]
        abs_res.append(np.maximum(np.abs(dx), np.abs(dy)))
        sx = sy = 0.0
        for d, p, half, axis in ((dx, pc[s] - hx, hx, "x"), (dy, pr[s] - hy, hy, "y")):
            pm = p - p.mean()
            var = float((pm * pm).sum())
            b = float((pm * (d - d.mean())).sum() / var) if var > 0 else 0.0
            a = float(d.mean() - b * p.mean())
            v = abs(a) + abs(b) * half
            if axis == "x":
                sx = v
            else:
                sy = v
        per_kf.append((k, max(sx, sy)))
        weights.append(len(s))
    if not per_kf:
        return {"n_points": int(n_total), "n_in_front": 0, "n_keyframes": 0,
                "systematic_px": float("inf"), "median_abs_px": float("inf"),
                "worst_keyframes": []}
    vals = np.array([v for _k, v in per_kf])
    w = np.asarray(weights, np.float64)
    o = np.argsort(vals, kind="stable")
    cum = np.cumsum(w[o])
    med = float(vals[o][np.searchsorted(cum, 0.5 * cum[-1])])
    worst = sorted(per_kf, key=lambda kv: -kv[1])[:5]
    return {"n_points": int(n_total), "n_in_front": int(n_front),
            "n_keyframes": len(per_kf), "systematic_px": med,
            "median_abs_px": float(np.median(np.concatenate(abs_res))),
            "worst_keyframes": [{"keyframe": int(k), "systematic_px": round(float(v), 3)}
                                for k, v in worst]}


def mask_store_hw(output_dir) -> Tuple[int, int]:
    """(H, W) of the session's SAM3 mask grid, read off the store itself."""
    output_dir = Path(output_dir)
    doc = json.loads((output_dir / "segmentation.json").read_text())
    p = output_dir / str(doc.get("mask_file") or "seg_masks.npz")
    with np.load(p) as masks:
        key = next((k for k in masks.files if k.startswith("f") and "_o" in k), None)
        if key is None:
            raise RuntimeError(f"{p} holds no mask — the mask grid is unknown")
        a = masks[key]
    return int(a.shape[0]), int(a.shape[1])


def projection_camera(output_dir, xyz: np.ndarray, ks_of_point: np.ndarray,
                      poses: np.ndarray, pixel_row: np.ndarray, pixel_col: np.ndarray,
                      min_depth_m: Optional[float] = None,
                      log: Callable[[str], None] = print) -> SessionProjection:
    """The camera the cloud was built with — ``camera.json`` — VERIFIED against
    the cloud and ``poses`` (the poses the cloud stands in) before anything is
    projected with it. Fails (``CameraMismatchError``) naming the measurement
    when the session camera does not reproduce the cloud's own birth pixels: a
    per-keyframe ``intrinsic.txt`` is never used in its place, because it is
    Omega's record of ANOTHER camera once F5 refined the session's (it says so
    in the message when it is the one that fits)."""
    from precision.camera import CAMERA_JSON_NAME, grid_like, load_camera_json, mask_grid_for

    output_dir = Path(output_dir)
    poses = np.asarray(poses, np.float64)
    ks = np.asarray(ks_of_point, np.int64)
    if len(ks) and int(ks.max()) >= len(poses):
        raise CameraMismatchError(
            f"the cloud has points born in keyframe {int(ks.max())} and the poses "
            f"hold {len(poses)} keyframes — the poses are not the cloud's")
    p = output_dir / CAMERA_JSON_NAME
    if not p.exists():
        raise CameraMismatchError(
            f"{p} does not exist — the camera the cloud was built with is unknown, "
            f"and the mask filter does not project with intrinsic.txt in its place "
            f"(Omega's per-keyframe record, another camera once F5 refines the "
            f"session's); run precision.camera (F0)")
    cam = load_camera_json(p)
    Ht, Wt = trace_grid(output_dir)
    g = cam.omega_grid
    rec = g if (int(g.w), int(g.h)) == (Wt, Ht) else grid_like(g, Wt, Ht, "record")
    proj = SessionProjection(cam, rec, mask_grid_for(cam.width, cam.height,
                                                     mask_store_hw(output_dir)),
                             source=f"{p.name} ({cam.source}, camera epoch "
                                    f"{cam.camera_epoch})")
    md = float(_vd_cfg().min_depth_m if min_depth_m is None else min_depth_m)

    def _session(P, k):
        front, x, y, _z = proj.continuous(P, poses[k], md, rec)
        return front, x, y

    chk = birth_systematic(_session, xyz, ks, len(poses), pixel_row, pixel_col, (Wt, Ht))
    if not chk["n_in_front"]:
        raise CameraMismatchError(
            f"no point of the cloud lies in front of its own birth keyframe with "
            f"{proj.source} and these poses — they are not the cloud's camera")
    if not chk["systematic_px"] <= BIRTH_PIXEL_PRECISION_PX:
        alt = ""
        ip = output_dir / "intrinsic.txt"
        if ip.exists():
            Ki = np.loadtxt(ip).reshape(-1, 4)
            if len(Ki) == len(poses):
                def _omega(P, k):
                    M = np.linalg.inv(_c2w(poses[k]))
                    q = np.asarray(P, np.float64) @ M[:3, :3].T + M[:3, 3]
                    front = q[:, 2] > md
                    fx, fy, cx, cy = Ki[k]
                    return (front, fx * q[front, 0] / q[front, 2] + cx,
                            fy * q[front, 1] / q[front, 2] + cy)
                a = birth_systematic(_omega, xyz, ks, len(poses), pixel_row, pixel_col,
                                     (Wt, Ht))
                alt = (f"; intrinsic.txt's per-keyframe camera misprojects it by "
                       f"{a['systematic_px']:.2f} px"
                       + (" — the cloud was built with Omega's camera, not the "
                          "session's" if a["systematic_px"] <= BIRTH_PIXEL_PRECISION_PX
                          else ""))
        raise CameraMismatchError(
            f"{proj.source} does not reproduce the cloud's birth pixels with these "
            f"poses: systematic misprojection {chk['systematic_px']:.2f} px "
            f"(point-weighted median over {chk['n_keyframes']} keyframes; worst "
            f"{chk['worst_keyframes'][:3]}) against the {BIRTH_PIXEL_PRECISION_PX} px "
            f"the birth pixel is stored at{alt} — the mask filter does not mix cameras")
    log(f"[visit-drift] camera: {proj.source} K=({cam.fx:.2f}, {cam.fy:.2f}, "
        f"{cam.cx:.2f}, {cam.cy:.2f}) reproduces the cloud's birth pixels — "
        f"systematic {chk['systematic_px']:.3f} px, median |residual| "
        f"{chk['median_abs_px']:.3f} px over {chk['n_in_front']:,} points / "
        f"{chk['n_keyframes']} keyframes; record grid {Wt}x{Ht}, mask grid "
        f"{proj.mask_grid.w}x{proj.mask_grid.h}")
    return proj


class Visibility:
    """What each keyframe could SEE of an object: its mask, depth-verified.

    The Z-buffer is built at mask resolution, the same construction as
    ``surface_fit/hole_audit._zbuf`` — 5-px minimum filter included, because the
    cloud is sparse at mask resolution and a ray often has no point on its exact
    pixel while its neighbours do. Cached per keyframe: the visits of different
    objects share keyframes.

    Every projection — the mask lookup AND the z-buffer — goes through ONE camera,
    ``camera``: the camera the cloud was built with (``projection_camera``,
    verified against the cloud's birth pixels before it gets here).
    """

    def __init__(self, output_dir, xyz: np.ndarray, ks_of_point: np.ndarray,
                 poses: np.ndarray, camera: "SessionProjection", depth_tol_m: float,
                 min_depth_m: Optional[float] = None):
        from segmentation import mask_space
        if not isinstance(camera, SessionProjection):
            raise TypeError(
                "Visibility projects with the camera the cloud was built with — pass "
                "visit_drift.projection_camera(...), not Omega's per-keyframe K "
                "(audit 2026-10-01: the two disagree by up to 23 px on pccr)")
        self.dir = Path(output_dir)
        self.xyz, self.poses, self.camera = xyz, poses, camera
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
        probe = next(self.masks[k] for k in self.masks.files
                     if k.startswith("f") and "_o" in k)
        self.Hm, self.Wm = int(probe.shape[0]), int(probe.shape[1])
        if (self.Hm, self.Wm) != camera.mask_hw:
            raise CameraMismatchError(
                f"the mask store is {self.Hm}x{self.Wm} and the camera was bound to a "
                f"{camera.mask_hw[0]}x{camera.mask_hw[1]} mask grid")
        self._zb: Dict[int, np.ndarray] = {}
        self._by_oid: Dict[int, List[Tuple[int, str]]] = {}
        self._by_kf: Dict[int, List[int]] = {}
        import re
        pat = re.compile(r"^f(\d+)_o(\d+)$")
        for key in self.masks.files:
            m = pat.match(key)
            if not m:
                continue
            kf = self.kf_of_frame.get(int(m.group(1)))
            if kf is not None:
                self._by_oid.setdefault(int(m.group(2)), []).append((kf, key))
                self._by_kf.setdefault(kf, []).append(int(m.group(2)))

    def masks_of(self, oid: int, visit: Tuple[int, int]):
        a, b = int(visit[0]), int(visit[1])
        return [(kf, self.masks[key]) for kf, key in self._by_oid.get(int(oid), [])
                if a <= kf <= b]

    def oids_at(self, kf: int) -> List[int]:
        """The masklets with a mask stored in keyframe ``kf`` (nothing is loaded)."""
        return sorted(set(self._by_kf.get(int(kf), [])))

    def project(self, P: np.ndarray, kf: int):
        """(ok, rows, cols, z) of world points in keyframe ``kf`` on the MASK grid,
        through the session camera: ``ok`` per point (in front and in frame), the
        pixel and the camera depth for the ``ok`` points only."""
        return self.camera.to_mask(P, self.poses[int(kf)], self.min_depth)

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
        ok, r, c, z = self.project(self.xyz[own], kf)
        zb = np.full((self.Hm, self.Wm), np.inf)
        np.minimum.at(zb, (r, c), z)
        zb = ndi.minimum_filter(zb, size=5, mode="nearest")
        self._zb[kf] = zb
        return zb

    def seen(self, centres: np.ndarray, kf_masks) -> np.ndarray:
        """Did ANY keyframe of this visit see these points, per its own mask
        and not occluded by measured geometry in front of them."""
        seen = np.zeros(len(centres), bool)
        for kf, m in kf_masks:
            ok, r, c, z = self.project(centres, kf)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            hit = m[r, c] > 0
            zpix = self.zbuf(kf)[r, c]
            seen[idx[hit & ~(zpix < (z - self.tol))]] = True
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

def rival_sigma_factor(n_rivals: int) -> float:
    """How much a closure's error bar widens when its identity is not unique.

    USER 2026-09-22, after test2's epoch 1: *"debe aplicar la correccion
    correcta"* — a repeated scene (rails, columns, identical warning labels)
    produces closures that LOOK perfect and are two different objects.
    `metal_track_rails#107` measured 99 % radial over 1.12 m and is one rail
    against another; the tangential residual cannot catch that, because
    nothing is wrong with the SHAPE match.

    What catches it is already measured and was being thrown away: how many
    OTHER masklets of the same label sit within this object's own measured
    displacement (`ambiguity`). With `n` rivals the pairing is one of `n + 1`
    equally plausible identities, so the measurement's variance grows by
    `n + 1` and its sigma by `sqrt(n + 1)`.

    It PRICES, it does not veto — the USER's standing rule since 2026-09-09.
    `max_ambiguity` as a veto was switched off on 2026-09-19 because on pccr
    it hid ten good floor closures; those closures keep speaking here, just
    with the error bar their ambiguity earns, and since they all AGREE their
    weighted sum still points the same way.
    """
    return float(np.sqrt(max(0, int(n_rivals)) + 1))


def scale_rows(kept: Sequence[Tuple["Candidate", "Drift"]], poses: np.ndarray,
               ks_of_point: np.ndarray, log: Callable[[str], None] = print,
               rivals_of: Optional[Dict[int, int]] = None
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
        _riv = int((rivals_of or {}).get(int(cand.oid), 0))
        _fac = rival_sigma_factor(_riv)
        residual *= _fac
        out.append({"instance_id": int(cand.instance_id), "label": cand.label,
                    "rivals": _riv, "sigma_factor": round(_fac, 3),
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
            f"at {D_b:.2f} m, tangential {tangential * 100:.1f} cm"
            + (f", {_riv} rival(s) → σ x{_fac:.2f}" if _riv else "") + ")")
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


def fused_object_roots(output_dir) -> Dict[int, int]:
    """oid → the instance id of the FUSED object the masklet ends up in (itself when
    nothing absorbed it). Masklets of one object never conflict with each other."""
    import json as _json
    p = Path(output_dir) / "segmentation_result.json"
    if not p.exists():
        return {}
    try:
        d = _json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        return {}
    absorbed: Dict[int, int] = {}
    for k, v in (d.get("absorbed") or {}).items():
        into = (v or {}).get("into")
        if into is None:
            continue
        try:
            absorbed[int(k)] = int(into)
        except (TypeError, ValueError):
            continue
    out: Dict[int, int] = {}
    ids = {int(inst.get("instance_id", inst.get("id", -1))) for inst in d.get("instances") or []}
    for iid in ids | set(absorbed):
        r, seen = int(iid), set()
        while r in absorbed and absorbed[r] >= 0 and r not in seen:
            seen.add(r)
            r = absorbed[r]
        out[int(iid) - 1] = r                # mask store oid = instance_id - 1
    return out


def fused_object_points(output_dir) -> Dict[int, int]:
    """oid → how many points the FUSED object it ends up in has.

    The matcher groups masklets into the session's objects
    (`segmentation_result.json`: `instances` + `absorbed`, which names the
    instance each absorbed one went `into`). A masklet is small; the object it
    belongs to need not be, and the "too small to be worth anything" test is
    about the OBJECT (USER 2026-09-22). Returns {} when the result is missing —
    the caller then judges the masklet on its own, as before.
    """
    import json as _json
    p = Path(output_dir) / "segmentation_result.json"
    if not p.exists():
        return {}
    try:
        d = _json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        return {}
    size: Dict[int, int] = {}
    for inst in d.get("instances") or []:
        iid = int(inst.get("instance_id", inst.get("id", -1)))
        n = inst.get("total_points")
        if n is None:
            n = len(inst.get("globalIndices") or ())
        size[iid] = int(n)
    absorbed = {}
    for k, v in (d.get("absorbed") or {}).items():
        into = (v or {}).get("into")
        if into is None:
            continue
        try:
            absorbed[int(k)] = int(into)
        except (TypeError, ValueError):
            continue

    def _root(iid: int, _seen=None) -> int:
        _seen = _seen or set()
        while iid in absorbed and absorbed[iid] >= 0 and iid not in _seen:
            _seen.add(iid)
            iid = absorbed[iid]
        return iid

    out: Dict[int, int] = {}
    for iid in set(list(size) + list(absorbed)):
        r = _root(int(iid))
        n = size.get(r)
        if n is None:
            continue
        out[int(iid) - 1] = int(n)          # mask store oid = instance_id - 1
    return out


def _top_frames(vis, oid: int, visit: Tuple[int, int], n: int) -> List[Tuple[int, np.ndarray]]:
    """The mask keyframes of one visit that show MOST of the object first, at most
    ``n`` of them (``segmentation.mask_filter.max_frames_per_visit``: a COST cap —
    a frame with three pixels of mask measures nothing)."""
    frames = list(vis.masks_of(oid, visit))
    frames.sort(key=lambda it: (-int(np.count_nonzero(it[1])), int(it[0])))
    return frames[:int(n)]


def other_mask_votes(vis, P: np.ndarray, birth: np.ndarray,
                     own_views: Dict[int, np.ndarray], rival_masks: Callable[[int], Optional[np.ndarray]],
                     dilate_px: int, occlusion_tol_rel: float, min_tri_deg: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RULE 4's ballot for the points ``P`` of one object X (birth keyframe per point
    in ``birth``): (n_votes, n_own, n_other) per point.

    A VOTE is a keyframe where X has a mask (``own_views``: {keyframe: X's mask},
    its own visit included — parallax inside one visit counts, USER 2026-09-29),
    other than the point's birth keyframe (its birth pixel is inside X's mask there
    by construction), that SAW the point: in frame, seen at a parallax of at least
    ``min_tri_deg`` from its birth ray (a view along that line sees every depth on
    it at one pixel and cannot tell where the point is — precision.refine's bound,
    the silhouette filter's rule), and not occluded by that keyframe's own measured
    depth (deeper than it by more than ``occlusion_tol_rel`` of it — the witness
    rule, ``loops.witness.occlusion_tol_rel``).

    The vote says OWN when the point lands inside X's mask dilated by ``dilate_px``
    (the rim tolerance of the point's OWN silhouette); OTHER when it does not and it
    lies ON another object's surface there: inside that object's UNDILATED mask
    (``rival_masks(kf)``: the pixels where another object — another fused root —
    drew a mask in keyframe kf) AND at the depth that keyframe measured at that
    pixel, within ``occlusion_tol_rel`` of it. A point merely behind another
    object's rim within an absolute tolerance (the first version's 15 cm) is not on
    its surface — that was the halo the old rule cut at every contact crease (a desk
    under a monitor foot, a wall behind a switch).
    """
    n = len(P)
    n_votes = np.zeros(n, np.int32)
    n_own = np.zeros(n, np.int32)
    n_other = np.zeros(n, np.int32)
    birth = np.asarray(birth, np.int64)
    cos_max = float(np.cos(np.radians(float(min_tri_deg))))
    tol = float(occlusion_tol_rel)
    centres = np.asarray(vis.poses, np.float64)[:, :3, 3]   # camera centre per keyframe
    for kf in sorted(own_views):
        w = np.flatnonzero(birth != int(kf))           # the birth keyframe never votes
        if not len(w):
            continue
        ok, r, c, z = vis.project(P[w], kf)
        if not ok.any():
            continue
        wo = w[ok]
        # parallax on the birth ray: min(θ, 180° − θ) ≥ min_tri_deg ⇔ |cos θ| ≤ cos(min)
        a = centres[birth[wo]] - P[wo]
        b = centres[int(kf)][None, :] - P[wo]
        cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1)
                                          * np.linalg.norm(b, axis=1), 1e-9)
        zb = vis.zbuf(kf)[r, c]
        measured = np.isfinite(zb)
        occluded = measured & (z > zb * (1.0 + tol))
        on_surface = measured & (np.abs(z - zb) <= tol * zb)
        vote = (np.abs(cos) <= cos_max) & ~occluded
        own = _dilated(own_views[kf], int(dilate_px))[r, c] > 0
        fm = rival_masks(int(kf))
        other = (~own & on_surface & (fm[r, c] > 0)) if fm is not None \
            else np.zeros(len(wo), bool)
        n_votes[wo] += vote.astype(np.int32)
        n_own[wo] += (vote & own).astype(np.int32)
        n_other[wo] += (vote & other).astype(np.int32)
    return n_votes, n_own, n_other


def cloud_filter_masklets(points_by_oid: Dict[int, np.ndarray],
                          masklets: Sequence[Masklet], ks_of_point: np.ndarray,
                          xyz: np.ndarray, vis: "Visibility",
                          min_points: int, min_visit_share: float,
                          max_frames_per_visit: int, dilate_px: int, *,
                          occlusion_tol_rel: float, min_votes: int,
                          min_inside_frac: float, min_tri_deg: float,
                          log: Callable[[str], None] = print,
                          group_points: Optional[Dict[int, int]] = None,
                          group_roots: Optional[Dict[int, int]] = None
                          ) -> Tuple[np.ndarray, FilterReport]:
    """STEP 12 — which points leave the cloud, once the pose is corrected.

    USER 2026-09-29: *"no debes probar los puntos contra su propia máscara, son
    contra el resto de las máscaras con las vistas de ellos"* — the fourth rule.
    REWRITTEN 2026-10-01 (audit of object-edge definition, USER: *"lo más
    importante es que los objetos deben tener mucha definición, corte en los
    filos, las aristas"* — and *"olvidate de las reglas que pusimos en su
    momento"*): as first wired, ONE view decided, the 2-px tolerance widened the
    FOREIGN mask over the point's own rim, "never inside its own" was vacuous for
    every single-visit object, and a point merely within 15 cm behind another
    object counted as being on it — it cut a halo at every contact crease. Now a
    point of X leaves by this rule when, among the views that SAW it
    (``other_mask_votes``: X's own mask keyframes, its own visit included, the
    birth keyframe excluded, at a parallax of at least ``min_tri_deg``,
    unoccluded), at least ``min_votes`` voted and it lies ON another object's
    surface inside that object's mask in at least ``min_inside_frac`` of them
    while inside its own (dilated) mask in less than ``min_inside_frac`` of them —
    the majority semantics of ``precision.cloud.silhouette_min_votes`` /
    ``silhouette_min_inside_frac``, read from there. Masklets fused into one
    object never conflict (``group_roots``).

    USER 2026-09-18: *"no me elimines lo unsegmented, solo los puntos que
    figuran como parte del objeto que luego de ser ajustado cae aun fuera de las
    mascaras"*, and *"lo mismo deben eliminarse de la nube los puntos de
    revisitas menores al 1%"*.

    The other three rules, all on the MASKLETS — SAM3's own tracks, not the fused
    instances:

      · a point that CLAIMS to be part of an object and, with the pose already
        corrected, still lands outside that object's mask in every view of its
        OTHER visits that saw it, is not part of it
      · an object under ``min_points`` keeps nothing — judged on the FUSED
        object when ``group_points`` says which masklets ended up in the same
        one (USER 2026-09-22: *"podriamos objetarlo luego de la union de
        segmentation ... porque pueden ser mascaras chicas que despues se
        fusionan ... pero si queda un objeto de menos de 1000 puntos no sirve
        de nada"*). A 300-point masklet that is one fragment of a 4.6 M-point
        floor is not a small object; it is a piece of a large one, and
        deleting it erodes the floor
      · a visit contributing at most ``min_visit_share`` of its masklet grazed
        it and keeps nothing

    A point belonging to NO masklet is UNSEGMENTED and is never touched: the
    masks say nothing about it, and silence is not a verdict.

    Every projection goes through ``vis.project`` — the camera the cloud was
    built with (``projection_camera``).

    It runs per epoch: each cloud has its own objects, and they need not be the
    same ones.
    """
    n_points = len(xyz)
    keep = np.zeros(n_points, bool)
    seg = np.zeros(n_points, bool)
    rep = FilterReport()
    by_oid = {m.oid: m for m in masklets}
    off_mask = 0
    off_other = 0
    roots = group_roots or {}

    def _root_of(o: int) -> int:
        return int(roots.get(int(o), int(o) + 1))

    def _union(oids, kf: int) -> Optional[np.ndarray]:
        u = None
        for o in oids:
            for _kf, mm in vis.masks_of(int(o), (kf, kf)):
                u = (mm > 0) if u is None else (u | (mm > 0))
        return u

    # How many OBJECTS (fused roots) drew a mask over each pixel of a keyframe —
    # loaded once per keyframe. Rule 4 asks it per view instead of pre-selecting
    # rivals by 3-D extent: the verdict is a VIEW's (its masks and its measured
    # depth), and a point within the relative tolerance of a flat surface lies
    # outside that surface's box — the box pre-filter of the first version
    # silently withheld exactly the wall a skirt hovers on.
    _cover: Dict[int, Optional[np.ndarray]] = {}

    def _objects_over(kf: int) -> Optional[np.ndarray]:
        if kf not in _cover:
            by_root: Dict[int, List[int]] = {}
            for o in vis.oids_at(kf):
                by_root.setdefault(_root_of(o), []).append(int(o))
            cnt = None
            for _r, os_ in sorted(by_root.items()):
                u = _union(os_, kf)
                if u is None:
                    continue
                # saturating uint8 (a cache of one byte per mask pixel per keyframe):
                # only "more objects than the point's own" is ever asked of it
                cnt = (u.astype(np.uint8) if cnt is None else
                       np.minimum(cnt.astype(np.uint16) + u, np.iinfo(np.uint8).max)
                       .astype(np.uint8))
            _cover[kf] = cnt
        return _cover[kf]

    for oid, idx in points_by_oid.items():
        idx = np.asarray(idx, np.int64)
        idx = idx[(idx >= 0) & (idx < n_points)]
        if not len(idx):
            continue
        seg[idx] = True
        m = by_oid.get(int(oid))
        label = m.label if m is not None else "object"
        _own = len(idx)
        _grp = int((group_points or {}).get(int(oid), _own))
        if _grp < int(min_points):
            rep.dropped_objects += 1
            rep.detail.append({"oid": int(oid), "label": label,
                               "reason": (f"object under {min_points} points"
                                          + (f" (masklet {_own}, fused object "
                                             f"{_grp})" if _grp != _own else "")),
                               "points": int(_own)})
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

        # RULE 1 — the cross-view test, over the keyframes of the OTHER visits
        sub = idx[alive]
        sub_ks = ks[alive]
        saw = np.zeros(len(sub), bool)
        inside = np.zeros(len(sub), bool)
        own_views: Dict[int, np.ndarray] = {}
        for vi, (a, b) in enumerate(visits):
            for kf, mm in _top_frames(vis, oid, (a, b), max_frames_per_visit):
                own_views[int(kf)] = mm
                other = sub_ks < a
                other |= sub_ks > b               # born outside this visit
                if not other.any():
                    continue
                w = np.flatnonzero(other)
                ok, r, c, z = vis.project(xyz[sub[w]], kf)
                if not ok.any():
                    continue
                zpix = vis.zbuf(kf)[r, c]
                visible = ~(zpix < (z - vis.tol))     # nothing in front
                hit = _dilated(mm, int(dilate_px))[r, c] > 0
                wo = w[ok]
                saw[wo[visible]] = True
                inside[wo[visible & hit]] = True

        # RULE 4 — on ANOTHER object's surface, by the majority of the views that saw it
        conflict = np.zeros(len(sub), bool)
        my_root = _root_of(int(oid))
        if own_views:
            def _rival_mask(kf: int, _me=my_root) -> Optional[np.ndarray]:
                """Pixels where ANOTHER object (another fused root) drew a mask."""
                cnt = _objects_over(kf)
                if cnt is None:
                    return None
                mine = _union([o for o in vis.oids_at(kf) if _root_of(o) == _me], kf)
                return cnt > (mine.astype(np.uint8) if mine is not None else 0)
            nv, no, nt = other_mask_votes(vis, xyz[sub], sub_ks, own_views, _rival_mask,
                                          dilate_px, occlusion_tol_rel, min_tri_deg)
            frac = float(min_inside_frac)
            conflict = ((nv >= int(min_votes)) & (nt >= frac * nv) & (no < frac * nv))
        # judged and never inside: it is not part of this object
        drop_own = saw & ~inside
        drop_other = conflict & ~drop_own
        off_mask += int(drop_own.sum())
        off_other += int(drop_other.sum())
        drop = drop_own | drop_other
        alive_idx = np.flatnonzero(alive)
        keep[idx[alive_idx[~drop]]] = True

    # UNSEGMENTED points are never touched
    keep |= ~seg
    kill = ~keep
    rep.dropped_points = int(kill.sum())
    rep.detail.append({"reason": "on another object's surface, inside its mask, in "
                                 "the majority of the views that saw it",
                       "points": int(off_other)})
    log(f"[visit-drift] step 12: {rep.dropped_points:,} of {n_points:,} points "
        f"leave ({off_mask:,} still off their own mask after the correction, "
        f"{off_other:,} on another object's surface in ≥ {min_inside_frac:.0%} of "
        f"≥ {min_votes} views that saw them, "
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
