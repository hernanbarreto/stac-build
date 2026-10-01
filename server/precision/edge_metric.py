"""Edge-definition metric — the CREASE PROFILE (edge audit 2026-10-01, §3).

USER 2026-10-01: *"lo más importante es que los objetos deben tener mucha
definición, corte en los filos, las aristas"*. Nothing in the repo measured
how sharp an edge is in a cloud; this module does, for every pair of adjacent
PLANE faces of every segmented object, from geometry alone (no GPU, no VLM —
provenance ``tool_measured``), on a frozen epoch's PLY plus its
``segmentation_result.json``.

Steps (the audit's S1-S9):

S1  points      the instance's ``globalIndices`` rows of the PLY, no confidence
                trim (``perfect_object._load_instance_cloud``, trim 0), raw frame.
S2  resolution  Δ = median 1-NN spacing (the construction of the detector's
                graph); for an A/B pair Δ = max(Δ_A, Δ_B).
S3  faces       the face regions of ``_detect_and_snap_cloud`` — its "plane"
                labels (never its snapped models), plus any cylinder/sphere
                region a TLS plane explains at least as well as the detector's
                curved model — refined by membership bands: pass 1 at
                ``p2c_inlier_dist_m``, later passes at z·σ_face (z from the
                declared confidence), a point joining the face (seed, current
                or in contact on the k-NN graph) whose band it sits deepest
                in, until the labels stop changing (≤ 3 passes). Adjacency =
                ≥ 1 contact edge of the detector's own k-NN graph.
S4  planes      total-least-squares plane per face (on its core points), the
                ideal edge line L from the two planes; a pair is skipped when
                its normals are parallel within their own bootstrap interval
                (seeded), or when a majority of one face lies inside the
                other's band (coplanar fragments); normals oriented so convex
                and concave creases are both an "L"; a face that continues past
                L (T-junction) is split into two creases.
S5  section     per point of A, B or unlabeled: s along L, the 2-D section
                coordinates, the interior angle φ, the crease extent S (overlap
                of the two faces' s-ranges).
S6  r̂           equivalent fillet radius = argmin over r ∈ {0, Δ, 2Δ, …, r_max}
                of the mean distance to the filleted L, ties within the
                sample's own noise (z standard errors of the paired excess)
                going to the smaller radius; r_max keeps the tangent point
                T(r) = r / tan(φ/2) on measured face; planes refitted on the
                core points (outside every crease's T(r̂)+Δ zone) until r̂ is a
                fixed point (≤ 3 refits).
S7  noise       σ = 1.4826·MAD of each face's core residuals;
                r_res = max(σ) / (1/sin(φ/2) − 1) — the smallest rounding the
                crease's own noise can reveal. r̂ < r_res reads "sharp within
                the instrument", not "proven sharp".
S8  band τ      1.4826·MAD of the signed distance to the r̂ profile of the band
                points (nearest profile point on the arc or within T(r̂)+Δ of
                the corner along an arm); reported as τ/σ (σ = max(σ_A, σ_B)).
S9  c, o        coverage = share of Δ-slices along S holding a band point that
                is not overshoot; overshoot = share of band points OUTSIDE the
                L — beyond at least one plane and farther than z·σ from both
                (skirt and flyers). A fillet lies inside the L and never counts.

Per crease, per object (length-weighted medians) and per epoch (length-weighted
medians over every crease). A-vs-B: faces detected ONCE on the reference, the
detector labels carried to the other epoch by (frame_global, pixel_row,
pixel_col) — or, when either PLY lacks those keys, objects paired by
instance_id (detected on each epoch, paired per object) — each epoch refitted
and measured on its own points, and judged with
``metric_lock.heldout_change`` at the declared confidence on r̂, τ/σ, (1 − c)
and o: improves / worsens / neither, no invented bar.

Declared limits (v1, audit §3): plane-plane creases only (a rounding the
detector labels as a separate cylinder is a plane-cylinder-plane junction and
gets no number); free borders thinner than ~2Δ give no crease; objects with
fewer than two plane faces get no number. Open3D's RANSAC is seeded; its
result is reproducible for a fixed OpenMP thread count.

CLI::

    python -m precision.edge_metric --session-output <dir> --ply <cloud.ply> \\
        --seg <segmentation_result.json> [--compare <other.ply> <other_seg.json>] \\
        --out <json>
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROVENANCE = "tool_measured"
LOG_TAG = "[edge]"

# Consistency constant of the MAD for a normal law (1 / Φ⁻¹(0.75)) — a
# mathematical constant, not a choice; the same one the gauge uses.
MAD_TO_SIGMA = 1.0 / 0.6744897501960817
# BOUNDS from the audit's own specification (§3 S3 / S6), not decisions: the
# membership passes stop when the labels stop changing, the plane refits when
# r̂ is a fixed point; these only cap a non-converging case (declared in the
# report as converged = false).
MAX_BAND_PASSES = 3
MAX_REFITS = 3
# Every random step is seeded (audit: Open3D's unseeded RANSAC returned six
# different planes in six runs).
SEED = 0

_KEY_FIELDS = ("frame_global", "pixel_row", "pixel_col")
VERDICT_METRICS = ("r_hat_m", "tau_over_sigma", "one_minus_coverage", "overshoot")


# ── configuration (strict: a missing key fails the load, naming it) ──────

class EdgeMetricConfigError(RuntimeError):
    """``config.yaml`` lacks a key the edge metric reads, or holds a bad value."""


@dataclass(frozen=True)
class EdgeMetricConfig:
    detect: Dict[str, Any]   # surface_fit — the section the cloud detector reads
    inlier_dist_m: float     # surface_fit.p2c_inlier_dist_m: S3 pass-1 band (the
                             # detector's own RANSAC membership distance)
    min_face_pts: int        # the cloud detector's smallest region, exactly as it
                             # applies it (perfect_object.detector_min_region_pts:
                             # surface_fit.perfect_min_region_faces or its in-code
                             # value) — a face (or a T-junction side) smaller than
                             # the detector's own minimum is not a face
    detect_max_pts: int      # surface_fit.p2c_detect_max_pts: detection subsample
    confidence: float        # correction_graph.graph.heldout_confidence: the
                             # declared confidence of every held-out verdict; z
                             # (1.96 at 0.95) and the bootstrap quantiles follow


def _need(section: Any, key: str, path: str) -> Any:
    if not isinstance(section, dict) or key not in section:
        raise EdgeMetricConfigError(
            f"config.yaml is missing mandatory key '{path}.{key}' (read by "
            f"precision/edge_metric.py) — there is no hidden default in code")
    return section[key]


def _need_num(section: Any, key: str, path: str, lo: float, hi: Optional[float] = None,
              integer: bool = False, lo_excl: bool = True) -> float:
    v = _need(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EdgeMetricConfigError(f"'{path}.{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise EdgeMetricConfigError(f"'{path}.{key}' must be an integer, got {v!r}")
    if (v <= lo if lo_excl else v < lo) or (hi is not None and v >= hi):
        raise EdgeMetricConfigError(f"'{path}.{key}' = {v} is outside its valid range")
    return int(v) if integer else float(v)


def load_edge_config(raw: Dict[str, Any]) -> EdgeMetricConfig:
    """Typed, strict read of the keys the metric uses from the parsed config.yaml."""
    from segmentation.perfect_object import detector_min_region_pts
    sf = _need(raw, "surface_fit", "<root>")
    cg = _need(raw, "correction_graph", "<root>")
    graph = _need(cg, "graph", "correction_graph")
    min_face = detector_min_region_pts(sf)
    if min_face < 3:
        raise EdgeMetricConfigError(f"'surface_fit.perfect_min_region_faces' = {min_face}: a "
                                    "plane needs at least 3 points")
    return EdgeMetricConfig(
        detect=dict(sf),
        inlier_dist_m=_need_num(sf, "p2c_inlier_dist_m", "surface_fit", 0.0),
        min_face_pts=min_face,
        detect_max_pts=_need_num(sf, "p2c_detect_max_pts", "surface_fit", 0, integer=True),
        confidence=_need_num(graph, "heldout_confidence", "correction_graph.graph", 0.0, 1.0),
    )


def _read_config_yaml() -> Dict[str, Any]:
    import yaml
    with open(Path(__file__).resolve().parents[1] / "config.yaml") as f:
        return yaml.safe_load(f) or {}


# ── vendor verdict (imported the way precision/gauge.py does) ─────────────

def _vendor_path() -> None:
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)


def _heldout_change():
    _vendor_path()
    from loop_utils.metric_lock import heldout_change      # vendor/VGGT-Long
    return heldout_change


def bootstrap_count() -> int:
    """The resample count of ``heldout_change`` itself (its declared default):
    the normals' bootstrap uses the same sample size as the verdict."""
    return int(inspect.signature(_heldout_change()).parameters["n_boot"].default)


def z_of(confidence: float) -> float:
    """Two-sided normal quantile of the declared confidence (1.96 at 0.95)."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + 0.5 * float(confidence)))


# ── small numerics ────────────────────────────────────────────────────────

def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64).ravel()
    if x.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def resolution(P: np.ndarray) -> float:
    """S2: median 1-NN spacing — the construction of the detector's contact
    graph (``perfect_object._knn_contact_edges``)."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, np.float64)
    dnn = cKDTree(P).query(P[:: max(1, len(P) // 5000)], k=2, workers=2)[0][:, 1]
    return float(np.median(dnn))


def tls_plane(P: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Total-least-squares plane: (unit normal n, d, centroid) with n·x + d = 0.
    The normal's sign is canonical (largest component positive)."""
    P = np.asarray(P, np.float64)
    c = P.mean(axis=0)
    Q = P - c
    _w, V = np.linalg.eigh(Q.T @ Q)
    n = V[:, 0]
    if n[int(np.argmax(np.abs(n)))] < 0:
        n = -n
    return n, -float(n @ c), c


def _bootstrap_normals(P: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """TLS normals of ``n_boot`` point-resamples (moment form: one weighted
    sum per resample, then a batched 3×3 eigen-solve)."""
    P = np.asarray(P, np.float64)
    n = len(P)
    Q = P - P.mean(axis=0)
    M = np.column_stack([Q, Q[:, 0] * Q[:, 0], Q[:, 0] * Q[:, 1], Q[:, 0] * Q[:, 2],
                         Q[:, 1] * Q[:, 1], Q[:, 1] * Q[:, 2], Q[:, 2] * Q[:, 2]])
    rng = np.random.default_rng(int(seed))
    mom = np.empty((int(n_boot), 9))
    for b in range(int(n_boot)):
        mom[b] = np.bincount(rng.integers(0, n, n), minlength=n) @ M
    mom /= n
    mu = mom[:, :3]
    S = np.empty((int(n_boot), 3, 3))
    S[:, 0, 0], S[:, 0, 1], S[:, 0, 2] = mom[:, 3], mom[:, 4], mom[:, 5]
    S[:, 1, 1], S[:, 1, 2], S[:, 2, 2] = mom[:, 6], mom[:, 7], mom[:, 8]
    S[:, 1, 0], S[:, 2, 0], S[:, 2, 1] = S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]
    S -= mu[:, :, None] * mu[:, None, :]
    return np.linalg.eigh(S)[1][:, :, 0]


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    v = np.asarray(values, np.float64)
    w = np.asarray(weights, np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return None
    v, w = v[m], w[m]
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(v) - 1)])


# ── S6/S8 geometry: the filleted L in the crease's 2-D section ────────────

@dataclass
class Profile:
    dist: np.ndarray      # unsigned distance to the profile
    signed: np.ndarray    # − inside the filleted L, + outside
    in_band: np.ndarray   # nearest profile point on the arc or within T(r)+Δ of the corner


def profile(q: np.ndarray, r: float, phi: float, delta: float) -> Profile:
    """Distance of 2-D section points ``q`` (N, 2) to the L of interior angle
    ``phi`` (arm A along +x, arm B at angle phi, both from the origin) rounded by
    a fillet of radius ``r`` tangent to both arms.

    Arm A's wedge-side normal is w_A = (0, 1), arm B's w_B = (sin φ, −cos φ);
    the region "inside the L" is the wedge between the arms minus the corner
    the fillet cuts off. Both convex and concave creases are this L (only the
    side the material sits on differs, and the metric does not use it)."""
    q = np.asarray(q, np.float64).reshape(-1, 2)
    cp, sp = float(np.cos(phi)), float(np.sin(phi))
    T = float(r) / float(np.tan(0.5 * phi)) if r > 0 else 0.0
    xa, ya = q[:, 0], q[:, 1]                          # along A, offset from A
    xb = q[:, 0] * cp + q[:, 1] * sp                   # along B
    yb = q[:, 0] * sp - q[:, 1] * cp                   # offset from B
    ta = np.maximum(xa, T)
    tb = np.maximum(xb, T)
    da = np.hypot(xa - ta, ya)
    db = np.hypot(xb - tb, yb)
    dist = np.minimum(da, db)
    foot = np.where(da <= db, ta, tb)
    on_arc = np.zeros(len(q), dtype=bool)
    inside = (ya >= 0.0) & (yb >= 0.0)
    if r > 0:
        v = q - np.array([T, float(r)])                # from the fillet centre
        # sector spanned by −w_A = (0, −1) and −w_B = (−sin φ, cos φ)
        bb = -v[:, 0] / sp
        aa = bb * cp - v[:, 1]
        in_sec = (aa >= 0.0) & (bb >= 0.0)
        rho = np.hypot(v[:, 0], v[:, 1])
        darc = np.where(in_sec, np.abs(rho - float(r)), np.inf)
        on_arc = darc < dist
        dist = np.where(on_arc, darc, dist)
        inside &= ~(in_sec & (rho > float(r)))
    signed = np.where(inside, -dist, dist)
    in_band = on_arc | (foot <= T + float(delta))
    return Profile(dist=dist, signed=signed, in_band=in_band)


def fillet_radius(Q: np.ndarray, phi: float, delta: float, r_max: float, z: float) -> float:
    """S6: r̂ = argmin over r ∈ {0, Δ, 2Δ, …, r_max} of the mean distance of the
    section points to the filleted L — with ties AT THE SAMPLE'S OWN NOISE
    resolved to the smaller radius: the smallest r whose paired excess over the
    minimum, d_r − d_min, has a mean within z standard errors of zero over the
    points the two profiles actually tell apart (a point both profiles place at
    the same distance is a tie and carries no evidence — the paired-test
    convention of dropping zero differences). A corner with no points (an
    eroded edge) leaves the cost flat from 0 up to the gap and a bare argmin
    would pick whichever radius the noise favours, reading erosion as
    rounding; within the noise the data cannot tell, and the smaller radius
    is what they support."""
    grid = delta * np.arange(int(np.floor(r_max / delta)) + 1)
    cost = np.array([float(np.mean(profile(Q, r, phi, delta).dist)) for r in grid])
    k_min = int(np.argmin(cost))
    d_min = profile(Q, float(grid[k_min]), phi, delta).dist
    for k in range(k_min):
        diff = profile(Q, float(grid[k]), phi, delta).dist - d_min
        diff = diff[diff != 0.0]
        if len(diff) < 2:
            return float(grid[k])
        se = float(np.std(diff, ddof=1)) / np.sqrt(len(diff))
        if float(np.mean(diff)) <= z * se:
            return float(grid[k])
    return float(grid[k_min])


# ── S3: faces ─────────────────────────────────────────────────────────────

def detect_seeds(P_det: np.ndarray, cfg: EdgeMetricConfig, safe: str,
                 log: Callable = print) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Face labels from the cloud detector (face ids 0..F-1, -1 for every other
    region and the residue) on every point, and the face pairs in contact on
    the detector's own k-NN graph.

    A face is a "plane" region — or a cylinder/sphere region that a plane
    explains at least as well as the detector's own curved model does (TLS
    plane residual σ ≤ model residual σ, both 1.4826·MAD). The detector picks
    plane vs cylinder by a normal-agreement score, and on flat faces that
    score lets a WORSE-fitting cylinder win (a synthetic box's last faces come
    out as R 0.2-0.9 m cylinders with residuals 1.4-9 mm against the plane's
    1.0 mm); the region is the detector's, its flatness is measured. A real
    rounding (a 25 mm corner strip: plane 11 mm vs cylinder 3 mm) stays out.

    Detection runs on at most ``p2c_detect_max_pts`` points (seeded
    subsample); the other points take the label of their nearest detected
    point (the S3 bands then re-test it)."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    from segmentation.perfect_object import _detect_and_snap_cloud, region_labels
    P_det = np.asarray(P_det, np.float64)
    n = len(P_det)
    if n <= cfg.detect_max_pts:
        sub = np.arange(n)
    else:
        sub = np.sort(np.random.default_rng(SEED).choice(n, cfg.detect_max_pts, replace=False))
    Ps = P_det[sub]
    # Open3D's RANSAC draws from its global generator inside an OpenMP loop: the
    # seed fixes the draws, ONE thread fixes which iteration gets which draw (a
    # tie in fitness then resolves the same way every run — observed flipping
    # the order of two near-equal fragments between identical runs)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        o3d.utility.random.seed(SEED)
        regions, _min, adj = _detect_and_snap_cloud(Ps, dict(cfg.detect), safe, log,
                                                    return_adjacency=True)
    plane_ids = []
    for i, r in enumerate(regions):
        if r["kind"] == "plane":
            plane_ids.append(i)
        elif r["kind"] in ("cylinder", "sphere") and r.get("model") is not None:
            Pr = Ps[np.asarray(r["v_idx"], np.int64)]
            if len(Pr) < 3:
                continue
            nr, dr, _c = tls_plane(Pr)
            if mad_sigma(Pr @ nr + dr) <= mad_sigma(np.asarray(r["model"].signed_distance(Pr))):
                plane_ids.append(i)
    face_of = np.full(len(regions) + 1, -1, dtype=np.int64)   # last slot: unlabeled
    for fi, ri in enumerate(plane_ids):
        face_of[ri] = fi
    lab_sub = face_of[region_labels(regions, len(sub))]       # -1 indexes the last slot
    if len(sub) == n:
        seeds = lab_sub
    else:
        seeds = lab_sub[cKDTree(Ps).query(P_det, k=1, workers=2)[1]]
    pairs = sorted({(min(int(face_of[i]), int(face_of[j])), max(int(face_of[i]), int(face_of[j])))
                    for (i, j) in adj if face_of[i] >= 0 and face_of[j] >= 0})
    return seeds.astype(np.int64), pairs


def refine_faces(P: np.ndarray, seeds: np.ndarray, inlier_dist_m: float,
                 z: float) -> Tuple[np.ndarray, int]:
    """S3 membership bands. Each pass fits every face's TLS plane on its
    current members and gives a point to the face whose band it sits deepest
    in (smallest |residual| / band), among its candidate faces: its seed face,
    its current face, and the faces of its neighbours on the detector's contact
    graph (``perfect_object._knn_contact_edges``) — membership is the plane
    band AND contact with the face, so the strip of a side face that a top
    face's RANSAC claimed returns to the side face instead of lingering as
    "unlabeled" inside another crease's section. Band = ``inlier_dist_m`` on
    pass 1, z·σ_face after (σ = 1.4826·MAD of the members' residuals). Stops
    when the labels stop changing (≤ 3 passes). Returns (labels, passes run)."""
    from segmentation.perfect_object import _knn_contact_edges
    P = np.asarray(P, np.float64)
    seeds = np.asarray(seeds, np.int64)
    faces = [int(f) for f in np.unique(seeds) if f >= 0]
    E = _knn_contact_edges(P) if len(P) > 1 else np.zeros((0, 2), np.int64)
    labels = seeds.copy()
    passes = 0
    for p in range(MAX_BAND_PASSES):
        passes = p + 1
        new = np.full(len(P), -1, dtype=np.int64)
        best = np.full(len(P), np.inf)
        for f in faces:
            mem = labels == f
            if int(mem.sum()) < 3:
                continue
            n, d, _c = tls_plane(P[mem])
            band = inlier_dist_m if p == 0 else z * mad_sigma(P[mem] @ n + d)
            if not band > 0:
                continue
            touch = np.zeros(len(P), dtype=bool)
            touch[E[:, 1][mem[E[:, 0]]]] = True
            touch[E[:, 0][mem[E[:, 1]]]] = True
            cand = np.nonzero((seeds == f) | mem | touch)[0]
            score = np.abs(P[cand] @ n + d) / band
            take = (score <= 1.0) & (score < best[cand])
            new[cand[take]] = f
            best[cand[take]] = score[take]
        changed = not np.array_equal(new, labels)
        labels = new
        if not changed:
            break
    return labels, passes


# ── S4: crease definitions (on the reference) ─────────────────────────────

@dataclass
class Crease:
    fa: int
    fb: int
    ref_ua: List[float]       # arm directions measured on the reference: they fix
    ref_ub: List[float]       # which side of each face this crease uses, in any epoch
    split_a: bool             # the face continues past L (T-junction side)
    split_b: bool
    theta_deg: float          # angle between the normal lines on the reference
    theta_null_deg: float     # their own bootstrap bar (parallel within it = skipped)

    def as_dict(self) -> Dict[str, Any]:
        return {"faces": [self.fa, self.fb], "split": [self.split_a, self.split_b],
                "theta_deg": self.theta_deg, "theta_null_deg": self.theta_null_deg}


def _line_frame(na, da, ca, nb, db, cb):
    """Unit direction e of the planes' intersection and the point o of it
    nearest the midpoint of the two centroids (None when parallel)."""
    e = np.cross(na, nb)
    le = float(np.linalg.norm(e))
    if le <= np.finfo(float).eps:
        return None
    e = e / le
    p0 = 0.5 * (ca + cb)
    o = np.linalg.solve(np.vstack([na, nb, e]), np.array([-da, -db, float(e @ p0)]))
    return e, o


def define_creases(P: np.ndarray, labels: np.ndarray, pairs: Sequence[Tuple[int, int]],
                   cfg: EdgeMetricConfig, z: float, n_boot: int,
                   log: Callable = print) -> Tuple[List[Crease], List[Dict[str, Any]]]:
    """S4 on the reference: the adjacent pairs that make a crease, oriented as
    an L, T-junctions split. Returns (creases, skipped pairs with the reason)."""
    P = np.asarray(P, np.float64)
    planes, sig, boot = {}, {}, {}
    for f in sorted({f for pr in pairs for f in pr}):
        m = labels == f
        if int(m.sum()) >= cfg.min_face_pts:
            n, d, c = tls_plane(P[m])
            planes[f] = (n, d, c)
            sig[f] = mad_sigma(P[m] @ n + d)
    creases: List[Crease] = []
    skipped: List[Dict[str, Any]] = []
    for fa, fb in pairs:
        if fa not in planes or fb not in planes:
            skipped.append({"faces": [fa, fb], "reason": "face under the detector's "
                            "minimum region size after the membership bands"})
            continue
        na, da, ca = planes[fa]
        nb, db, cb = planes[fb]
        for f in (fa, fb):
            if f not in boot:
                Pf = P[labels == f]
                bn = _bootstrap_normals(Pf, n_boot, SEED + f)
                nf = planes[f][0]
                bn *= np.where(bn @ nf < 0, -1.0, 1.0)[:, None]
                boot[f] = bn - (bn @ nf)[:, None] * nf[None, :]     # tangent deviations
        sgn = 1.0 if float(na @ nb) >= 0 else -1.0
        theta = float(np.arccos(np.clip(abs(float(na @ nb)), 0.0, 1.0)))
        # the angle two normals of ONE direction would show, given each face's own
        # resampling noise (deviations of A and of B, B aligned with A)
        null = np.arcsin(np.clip(np.linalg.norm(boot[fa] - sgn * boot[fb], axis=1), 0.0, 1.0))
        bar = float(np.quantile(null, cfg.confidence))
        if theta <= bar:
            skipped.append({"faces": [fa, fb], "reason": "parallel within the normals' "
                            "own bootstrap interval", "theta_deg": float(np.degrees(theta)),
                            "theta_null_deg": float(np.degrees(bar))})
            continue
        fr = _line_frame(na, da, ca, nb, db, cb)
        if fr is None:
            skipped.append({"faces": [fa, fb], "reason": "planes do not intersect"})
            continue
        e, o = fr
        sides, reach = {}, {}
        for f, nf, nother, dother, sother in ((fa, na, nb, db, sig[fb]),
                                             (fb, nb, na, da, sig[fa])):
            u = np.cross(e, nf)
            u /= np.linalg.norm(u)
            Pf = P[labels == f]
            al = (Pf - o) @ u
            if float(np.median(al)) < 0:
                u, al = -u, -al
            beyond = np.abs(Pf @ nother + dother) > z * sother   # off the other plane's band
            n_pos = int(np.sum(beyond & (al > 0)))
            n_neg = int(np.sum(beyond & (al < 0)))
            split = min(n_pos, n_neg) >= cfg.min_face_pts
            sides[f] = ([u, -u] if split else [u], split)
            # share of this face's points inside the OTHER plane's membership band
            reach[f] = float(np.mean(~beyond))
        if max(reach[fa], reach[fb]) > 0.5:
            # a MAJORITY of one face's points sits inside the other's membership
            # band (the user's majority rule for such votes, as
            # silhouette_min_inside_frac): at the band's resolution that face is
            # the other surface (two coplanar fragments of the detector), and a
            # crease between them is not observable
            skipped.append({"faces": [fa, fb], "reason": "a majority of one face lies in "
                            "the other's membership band", "theta_deg": float(np.degrees(theta)),
                            "share_in_other_band": [reach[fa], reach[fb]]})
            continue
        for ua in sides[fa][0]:
            for ub in sides[fb][0]:
                creases.append(Crease(fa=int(fa), fb=int(fb),
                                      ref_ua=[float(x) for x in ua],
                                      ref_ub=[float(x) for x in ub],
                                      split_a=bool(sides[fa][1]), split_b=bool(sides[fb][1]),
                                      theta_deg=float(np.degrees(theta)),
                                      theta_null_deg=float(np.degrees(bar))))
    log(f"{LOG_TAG} {len(creases)} crease(s) from {len(pairs)} adjacent face pair(s), "
        f"{len(skipped)} pair(s) skipped")
    return creases, skipped


# ── S5-S9: one crease, one measurement round ──────────────────────────────

def _measure_crease(P: np.ndarray, labels: np.ndarray, cr: Crease, planes: Dict[int, tuple],
                    sig: Dict[int, float], delta: float, z: float,
                    min_pts: int) -> Optional[Dict[str, Any]]:
    if cr.fa not in planes or cr.fb not in planes or not delta > 0:
        return None
    na, da, ca = planes[cr.fa]
    nb, db, cb = planes[cr.fb]
    fr = _line_frame(na, da, ca, nb, db, cb)
    if fr is None:
        return None
    e, o = fr
    ua = np.cross(e, na)
    ua /= np.linalg.norm(ua)
    if float(ua @ np.asarray(cr.ref_ua)) < 0:
        ua = -ua
    ub = np.cross(e, nb)
    ub /= np.linalg.norm(ub)
    if float(ub @ np.asarray(cr.ref_ub)) < 0:
        ub = -ub
    cp = float(np.clip(ua @ ub, -1.0, 1.0))
    phi = float(np.arccos(cp))
    sp = float(np.sin(phi))
    if sp <= np.finfo(float).eps:
        return None
    yhat = (ub - cp * ua) / sp
    R = P - o
    s = R @ e
    qx = R @ ua
    qy = R @ yhat
    xb = qx * cp + qy * sp                             # along arm B
    a_pts = labels == cr.fa
    if cr.split_a:
        a_pts &= qx >= 0
    b_pts = labels == cr.fb
    if cr.split_b:
        b_pts &= xb >= 0
    if int(a_pts.sum()) < min_pts or int(b_pts.sum()) < min_pts:
        return None
    s_lo = max(float(s[a_pts].min()), float(s[b_pts].min()))
    s_hi = min(float(s[a_pts].max()), float(s[b_pts].max()))
    if not s_hi > s_lo:
        return None
    in_S = (s >= s_lo) & (s <= s_hi)
    if not (a_pts & in_S).any() or not (b_pts & in_S).any():
        return None
    L_a = float(qx[a_pts & in_S].max())
    L_b = float(xb[b_pts & in_S].max())
    T_max = min(L_a, L_b)                              # the tangent point stays on measured face
    if not T_max > 0:
        return None
    r_max = T_max * float(np.tan(0.5 * phi))
    sect = (a_pts | b_pts | (labels < 0)) & in_S & (np.hypot(qx, qy) <= T_max + delta)
    Q = np.column_stack([qx[sect], qy[sect]])
    r_hat = fillet_radius(Q, phi, delta, r_max, z)
    pr = profile(Q, r_hat, phi, delta)
    sa, sb = float(sig[cr.fa]), float(sig[cr.fb])
    sig_max = max(sa, sb)
    r_res = sig_max / (1.0 / float(np.sin(0.5 * phi)) - 1.0)
    band = pr.in_band
    ya = Q[:, 1]
    yb = Q[:, 0] * sp - Q[:, 1] * cp
    over = band & (np.abs(ya) > z * sa) & (np.abs(yb) > z * sb) & ((ya < 0) | (yb < 0))
    n_band = int(band.sum())
    tau = mad_sigma(pr.signed[band]) if n_band else float("nan")
    length = s_hi - s_lo
    n_sl = max(1, int(np.ceil(length / delta)))
    s_sect = s[sect]
    good = band & ~over
    sl = np.clip(np.floor((s_sect[good] - s_lo) / delta).astype(np.int64), 0, n_sl - 1)
    coverage = float(len(np.unique(sl))) / n_sl
    T_hat = r_hat / float(np.tan(0.5 * phi))
    zone_a = (labels == cr.fa) & in_S & (np.abs(qx) <= T_hat + delta)
    zone_b = (labels == cr.fb) & in_S & (np.abs(xb) <= T_hat + delta)
    return {
        "phi_deg": float(np.degrees(phi)), "length_m": float(length), "delta_m": float(delta),
        "sigma_a_m": sa, "sigma_b_m": sb, "r_res_m": float(r_res), "r_hat_m": r_hat,
        "r_max_m": float(r_max), "sharp_within_instrument": bool(r_hat < r_res),
        "tau_m": float(tau),
        "tau_over_sigma": float(tau / sig_max) if np.isfinite(tau) and sig_max > 0 else float("nan"),
        "coverage": coverage,
        "overshoot": float(over.sum()) / n_band if n_band else float("nan"),
        "n_points": int(sect.sum()), "n_band": n_band,
        "_zone_a": zone_a, "_zone_b": zone_b,
    }


def measure_object(P: np.ndarray, labels: np.ndarray, creases: Sequence[Crease], delta: float,
                   cfg: EdgeMetricConfig, z: float) -> Dict[str, Any]:
    """S4-S9 for every crease of one object, the planes refitted on the core
    points (each face minus every crease's T(r̂)+Δ zone) until r̂ is a fixed
    point (≤ 3 refits). Returns per-crease records plus the object summary."""
    P = np.asarray(P, np.float64)
    faces = sorted({f for c in creases for f in (c.fa, c.fb)})
    cores = {f: labels == f for f in faces}
    prev, results, refits, converged = None, [], 0, False
    for it in range(1 + MAX_REFITS):
        planes, sig = {}, {}
        for f in faces:
            m = cores[f]
            if int(m.sum()) >= cfg.min_face_pts:
                n, d, c = tls_plane(P[m])
                planes[f] = (n, d, c)
                sig[f] = mad_sigma(P[m] @ n + d)
        results = [_measure_crease(P, labels, c, planes, sig, delta, z, cfg.min_face_pts)
                   for c in creases]
        key = tuple(None if r is None else r["r_hat_m"] for r in results)
        if key == prev:
            converged = True
            break
        prev = key
        if it == MAX_REFITS:
            break
        refits += 1
        cores = {f: labels == f for f in faces}
        for c, r in zip(creases, results):
            if r is not None:
                cores[c.fa] &= ~r["_zone_a"]
                cores[c.fb] &= ~r["_zone_b"]
    out = []
    for c, r in zip(creases, results):
        rec = c.as_dict()
        if r is None:
            rec["measured"] = False
        else:
            rec["measured"] = True
            rec.update({k: v for k, v in r.items() if not k.startswith("_")})
            rec["one_minus_coverage"] = 1.0 - rec["coverage"]
        out.append(rec)
    return {"creases": out, "refits": refits, "converged": converged,
            "summary": summarize(out)}


_SUMMARY_KEYS = ("r_hat_m", "r_res_m", "tau_m", "tau_over_sigma", "coverage",
                 "one_minus_coverage", "overshoot", "sigma_a_m", "sigma_b_m")


def summarize(creases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Length-weighted medians over measured creases."""
    m = [c for c in creases if c.get("measured")]
    w = [c["length_m"] for c in m]
    s: Dict[str, Any] = {"n_creases": len(m), "length_m": float(np.sum(w)) if m else 0.0}
    for k in _SUMMARY_KEYS:
        s[k] = weighted_median([c[k] for c in m], w) if m else None
    return s


# ── epochs: loading, label transfer, the run ──────────────────────────────

def _keys(fields: Dict[str, np.ndarray], rows: np.ndarray) -> Optional[np.ndarray]:
    """(frame_global, pixel_row, pixel_col) packed in one int64 per row."""
    if not all(k in fields for k in _KEY_FIELDS):
        return None
    rows = np.asarray(rows, np.int64)
    fg = np.asarray(fields["frame_global"])[rows].astype(np.int64)
    pr = np.asarray(fields["pixel_row"])[rows].astype(np.int64) & 0xFFFF
    pc = np.asarray(fields["pixel_col"])[rows].astype(np.int64) & 0xFFFF
    return (fg << 32) | (pr << 16) | pc


def carry_labels(keys_ref: np.ndarray, labels_ref: np.ndarray,
                 keys_other: np.ndarray) -> np.ndarray:
    """Each point of the other epoch takes the label of the reference point
    with its key; no match (or a key the reference labels two ways) → -1."""
    keys_other = np.asarray(keys_other, np.int64)
    if len(keys_ref) == 0:
        return np.full(len(keys_other), -1, dtype=np.int64)
    o = np.argsort(keys_ref, kind="stable")
    k, lab = np.asarray(keys_ref, np.int64)[o], np.asarray(labels_ref, np.int64)[o]
    uk, first, cnt = np.unique(k, return_index=True, return_counts=True)
    ul = lab[first]
    if (cnt > 1).any():
        lo, hi = np.minimum.reduceat(lab, first), np.maximum.reduceat(lab, first)
        ul = np.where(lo == hi, ul, -1)
    pos = np.clip(np.searchsorted(uk, keys_other), 0, len(uk) - 1)
    return np.where(uk[pos] == keys_other, ul[pos], -1).astype(np.int64)


def _instances(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [i for i in seg.get("instances", []) if i.get("globalIndices")]


def _iid(inst: Dict[str, Any]) -> int:
    return int(inst.get("instance_id", inst.get("id")))


def _instance_of_rows(seg: Dict[str, Any], n_rows: int) -> np.ndarray:
    io = np.full(n_rows, -1, dtype=np.int64)
    for inst in _instances(seg):
        gi = np.asarray(inst["globalIndices"], np.int64)
        gi = gi[(gi >= 0) & (gi < n_rows)]
        io[gi] = _iid(inst)
    return io


def _load(output_dir: Path, inst: Dict[str, Any], ply: Path, fields: Dict[str, np.ndarray],
          safe: str, log: Callable) -> Tuple[np.ndarray, np.ndarray]:
    from segmentation.perfect_object import _load_instance_cloud
    return _load_instance_cloud(output_dir, inst, {"p2c_conf_trim_pct": 0.0}, safe, log,
                                ply_path=ply, fields=fields, return_indices=True)


def _analyse(P: np.ndarray, output_dir: Path, cfg: EdgeMetricConfig, z: float, n_boot: int,
             safe: str, log: Callable) -> Dict[str, Any]:
    from segmentation.perfect_object import _to_display
    seeds, pairs = detect_seeds(_to_display(output_dir, P), cfg, safe, log)
    labels, passes = refine_faces(P, seeds, cfg.inlier_dist_m, z)
    creases, skipped = define_creases(P, labels, pairs, cfg, z, n_boot, log)
    return {"seeds": seeds, "labels": labels, "band_passes": passes, "pairs": pairs,
            "creases": creases, "skipped": skipped,
            "n_faces": int(len([f for f in np.unique(labels) if f >= 0]))}


def _epoch_block(ply: Path, seg_path: Path, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    allc = [c for ob in objects for c in ob.get("creases", [])]
    return {"ply": str(ply), "seg": str(seg_path), "summary": summarize(allc),
            "objects": objects}


def run(output_dir: Path, ply: Path, seg_path: Path,
        compare: Optional[Tuple[Path, Path]] = None, out_path: Optional[Path] = None,
        instance_ids: Optional[Sequence[int]] = None, cfg: Optional[EdgeMetricConfig] = None,
        log: Callable = print) -> Dict[str, Any]:
    """Measure one epoch (``ply`` + ``seg_path``), or the reference and the
    ``compare`` epoch with the A-vs-B verdict. Writes ``out_path`` when given.
    Runs with OpenMP/BLAS at one thread (as intake/parallax.py does): no
    reading may depend on how many threads split a reduction."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _run(output_dir, ply, seg_path, compare, out_path, instance_ids, cfg, log)


def _run(output_dir, ply, seg_path, compare, out_path, instance_ids, cfg, log) -> Dict[str, Any]:
    from segmentation.perfect_object import _read_ply_fields
    t0 = time.time()
    cfg = cfg or load_edge_config(_read_config_yaml())
    output_dir, ply, seg_path = Path(output_dir), Path(ply), Path(seg_path)
    z = z_of(cfg.confidence)
    n_boot = bootstrap_count()
    fa = _read_ply_fields(ply)
    seg_a = json.loads(seg_path.read_text())
    fb = seg_b = None
    keyed = False
    if compare is not None:
        ply_b, seg_b_path = Path(compare[0]), Path(compare[1])
        fb = _read_ply_fields(ply_b)
        seg_b = json.loads(seg_b_path.read_text())
        keyed = all(k in fa for k in _KEY_FIELDS) and all(k in fb for k in _KEY_FIELDS)
        n_b = len(fb["x"])
        inst_b = {_iid(i): i for i in _instances(seg_b)}
        if keyed:
            kb_all = _keys(fb, np.arange(n_b))
            ob = np.argsort(kb_all, kind="stable")
            kb_sorted = kb_all[ob]
            inst_of_b = _instance_of_rows(seg_b, n_b)
        log(f"{LOG_TAG} A-vs-B: labels carried by "
            f"{'(frame_global, pixel_row, pixel_col)' if keyed else 'nothing — objects paired by instance_id'}")
    wanted = None if instance_ids is None else {int(i) for i in instance_ids}
    obj_a: List[Dict[str, Any]] = []
    obj_b: List[Dict[str, Any]] = []
    pairs_crease: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    pairs_object: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for inst in _instances(seg_a):
        iid = _iid(inst)
        if wanted is not None and iid not in wanted:
            continue
        label = str(inst.get("label", "segment"))
        safe = f"{label}_{iid}"
        P_a, gi_a = _load(output_dir, inst, ply, fa, safe, log)
        rec_a: Dict[str, Any] = {"instance_id": iid, "label": label, "n_points": int(len(P_a))}
        if len(P_a) < 2 * cfg.min_face_pts:
            rec_a.update(creases=[], note="fewer points than two minimum faces")
            obj_a.append(rec_a)
            continue
        ref = _analyse(P_a, output_dir, cfg, z, n_boot, safe, log)
        rec_a.update(n_faces=ref["n_faces"], band_passes=ref["band_passes"],
                     skipped_pairs=ref["skipped"])
        if not ref["creases"]:
            rec_a.update(creases=[], note="fewer than two adjacent, non-parallel plane faces")
            obj_a.append(rec_a)
            continue
        delta = resolution(P_a)
        rec_a["delta_own_m"] = delta
        rec_b = None
        if compare is not None:
            if keyed:
                ka = _keys(fa, gi_a)
                pos = np.clip(np.searchsorted(kb_sorted, ka), 0, len(kb_sorted) - 1)
                hit = kb_sorted[pos] == ka
                cand = inst_of_b[ob[pos[hit]]]
                cand = cand[cand >= 0]
                iid_b = int(np.bincount(cand).argmax()) if len(cand) else None
            else:
                iid_b = iid if iid in inst_b else None
            if iid_b is None or iid_b not in inst_b:
                rec_b = {"instance_id": None, "paired_with": iid, "creases": [],
                         "note": "no paired instance in the other epoch"}
            else:
                P_b, gi_b = _load(output_dir, inst_b[iid_b], Path(compare[0]), fb, safe, log)
                rec_b = {"instance_id": iid_b, "label": str(inst_b[iid_b].get("label", "segment")),
                         "paired_with": iid, "n_points": int(len(P_b))}
                if keyed:
                    seeds_b = carry_labels(_keys(fa, gi_a), ref["seeds"], _keys(fb, gi_b))
                    labels_b, passes_b = refine_faces(P_b, seeds_b, cfg.inlier_dist_m, z)
                    creases_b = ref["creases"]
                    rec_b.update(band_passes=passes_b,
                                 carried_frac=float(np.mean(seeds_b >= 0)) if len(seeds_b) else 0.0)
                else:
                    ana_b = _analyse(P_b, output_dir, cfg, z, n_boot, safe, log)
                    labels_b, creases_b = ana_b["labels"], ana_b["creases"]
                    rec_b.update(band_passes=ana_b["band_passes"], n_faces=ana_b["n_faces"],
                                 skipped_pairs=ana_b["skipped"])
                delta_b = resolution(P_b)
                rec_b["delta_own_m"] = delta_b
                delta = max(delta, delta_b)
                if creases_b:
                    rec_b.update(measure_object(P_b, labels_b, creases_b, delta, cfg, z))
                else:
                    rec_b["creases"] = []
        rec_a.update(measure_object(P_a, ref["labels"], ref["creases"], delta, cfg, z))
        obj_a.append(rec_a)
        sa = rec_a["summary"]
        log(f"{LOG_TAG} {safe}: {sa['n_creases']} crease(s), Δ {delta * 1000:.1f} mm, "
            f"r̂ {_mm(sa['r_hat_m'])} (r_res {_mm(sa['r_res_m'])}), τ/σ {_fmt(sa['tau_over_sigma'])}, "
            f"c {_fmt(sa['coverage'])}, o {_fmt(sa['overshoot'])}")
        if rec_b is not None:
            obj_b.append(rec_b)
            if rec_b.get("summary"):
                pairs_object.append((sa, rec_b["summary"]))
            if keyed and rec_b.get("creases"):
                pairs_crease.extend(zip(rec_a["creases"], rec_b["creases"]))
    rep: Dict[str, Any] = {
        "metric": "crease_profile", "version": 1, "provenance": PROVENANCE,
        "params": {"confidence": cfg.confidence, "z": z, "bootstrap": n_boot, "seed": SEED,
                   "inlier_dist_m": cfg.inlier_dist_m, "min_face_pts": cfg.min_face_pts,
                   "detect_max_pts": cfg.detect_max_pts, "max_band_passes": MAX_BAND_PASSES,
                   "max_refits": MAX_REFITS},
        "reference": _epoch_block(ply, seg_path, obj_a)}
    if compare is not None:
        rep["compare"] = _epoch_block(Path(compare[0]), Path(compare[1]), obj_b)
        rep["verdict"] = verdict(pairs_crease if keyed else pairs_object,
                                 "crease" if keyed else "object", cfg.confidence)
        v = rep["verdict"]["metrics"]
        log(f"{LOG_TAG} verdict ({rep['verdict']['pairing']}, n {rep['verdict']['n_pairs']}): "
            + ", ".join(f"{k} {v[k]['verdict']}" for k in VERDICT_METRICS))
    rep["seconds"] = round(time.time() - t0, 1)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonable(rep), indent=1, allow_nan=False))
        log(f"{LOG_TAG} → {out_path} ({rep['seconds']} s)")
    return rep


def verdict(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]], pairing: str,
            confidence: float) -> Dict[str, Any]:
    """``heldout_change(before = reference, after = other)`` per metric over the
    paired units with both sides measured. Every metric is smaller-is-better
    (r̂, τ/σ, 1 − c, o), so "improves" means the other epoch's edges are crisper."""
    hc = _heldout_change()
    res: Dict[str, Any] = {"pairing": pairing, "confidence": confidence,
                           "n_pairs": len(pairs), "metrics": {}}
    for k in VERDICT_METRICS:
        b, a = [], []
        for ra, rb in pairs:
            if pairing == "crease" and not (ra.get("measured") and rb.get("measured")):
                continue
            va, vb = ra.get(k), rb.get(k)
            if va is None or vb is None or not (np.isfinite(va) and np.isfinite(vb)):
                continue
            b.append(float(va))
            a.append(float(vb))
        ch = hc(b, a, confidence=confidence)
        ch["verdict"] = "improves" if ch["improves"] else "worsens" if ch["worsens"] else "neither"
        ch["median_before"] = float(np.median(b)) if b else None
        ch["median_after"] = float(np.median(a)) if a else None
        res["metrics"][k] = ch
    return res


def _jsonable(o: Any) -> Any:
    """Strict JSON: NaN/inf (a crease with no band point has no τ, no o) → null."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _mm(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 1000:.1f} mm"


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.2f}"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session-output", required=True,
                    help="the session's output/ (floor_transform.npz for the detection frame)")
    ap.add_argument("--ply", required=True, help="reference cloud (cleaned_cloud.ply)")
    ap.add_argument("--seg", required=True, help="reference segmentation_result.json")
    ap.add_argument("--compare", nargs=2, metavar=("PLY", "SEG"),
                    help="the other epoch's cloud and segmentation (A-vs-B verdict)")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--instance-ids", type=int, nargs="*", default=None,
                    help="measure only these reference instances")
    a = ap.parse_args(argv)
    run(Path(a.session_output), Path(a.ply), Path(a.seg),
        compare=tuple(a.compare) if a.compare else None, out_path=Path(a.out),
        instance_ids=a.instance_ids)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
