"""Composites: the parts the VLM listed, each a template, decomposed GUIDED by that list.

Blind RANSAC gave the desk a spline and the stick three planes; with the VLM's part list (a top,
four equal legs under it) the decomposition is well posed: the first part is fitted on all the
points and takes what it explains within the fitter's own band (Huber's k × σ), the residue
splits into connected pieces (3-D occupancy at three point spacings — the cloud's own scale),
the pieces go to the remaining specs in size order (a spec of count k takes the k largest), each
part is initialised on its piece, and everything is optimised TOGETHER: the union's distance
(the nearest part) against the points, the union's samples against the masks. "equal" parts
share one section (the VLM's word; declared in the record); a piece the cloud does not have is
declared missing, never invented.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .catalog import FormAnswer, Part
from .fit import fit_cloud, subsample
from .templates import Frame, Section, Template, _t, build_template

TAIL_SIGMAS = 3.0   # a part takes its points up to the Gaussian tail (99.7 % at 3 σ): the 18 % a
                    # Huber-band removal leaves behind re-extracted the same leg as a second one
_SHARE_WORDS = ("equal", "same", "identical", "igual", "iguales", "mismo", "misma", "uniform")
_DISTINCT_WORDS = ("different", "distinct", "vary", "varying", "unequal", "distinto", "diferente")


def _part_answer(ans: FormAnswer, part: Part) -> FormAnswer:
    """The part as a single-form answer (its own form and orientation, the object's alignment)."""
    return FormAnswer(form=part.form, orientation=part.orientation, axis_aligned=ans.axis_aligned,
                      hollow=None, outline="unknown", profile="unknown", parts=[], proportions="",
                      alternatives=[], confidence=ans.confidence, what_would_help="", notes=part.relation)


def shares_section(part: Part) -> bool:
    rel = (part.relation or "").lower()
    if any(w in rel for w in _DISTINCT_WORDS):
        return False
    return part.count > 1


class Composite(Template):
    """The union of parts: distance = the nearest part's, samples and mesh concatenated."""
    name = "composite"
    form = "composite"

    def __init__(self, parts: List[Tuple[str, Template]], dev, notes: Optional[List[str]] = None):
        self.parts = list(parts)
        self.dev = dev
        self.notes = list(notes or [])

    def parameters(self) -> List[torch.Tensor]:
        out, seen = [], set()
        for _n, p in self.parts:
            for q in p.parameters():
                if id(q) not in seen:
                    seen.add(id(q))
                    out.append(q)
        return out

    def _per_part(self, P: torch.Tensor) -> torch.Tensor:
        return torch.stack([p.distance(P) for _n, p in self.parts], 0)

    def distance(self, P: torch.Tensor) -> torch.Tensor:
        return self._per_part(P).min(dim=0).values

    def assign(self, P: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._per_part(P).argmin(dim=0)

    def samples(self, n: int) -> torch.Tensor:
        k = max(1, len(self.parts))
        return torch.cat([p.samples(max(256, int(n) // k)) for _n, p in self.parts], 0)

    def measure_extents(self, P: torch.Tensor) -> None:
        a = self.assign(P)
        for i, (_n, p) in enumerate(self.parts):
            sel = a == i
            if int(sel.sum()) >= 10:
                p.measure_extents(P[sel])

    def bound_extents(self, P: torch.Tensor, sigma: float) -> None:
        """Each part is bounded by the points IT ALONE explains (within the noise tail of its
        surface and of no other part's): the ring where a leg meets the table top lies within
        both bands and would carry the leg's end up to the top's faces."""
        with torch.no_grad():
            D = self._per_part(P)
            within = D <= TAIL_SIGMAS * max(float(sigma), 1e-6)
            n_within = within.to(torch.int64).sum(dim=0)
            a = D.argmin(dim=0)
        for i, (_n, p) in enumerate(self.parts):
            unique = within[i] & (n_within == 1)
            sel = unique if int(unique.sum()) >= 10 else (a == i)
            if int(sel.sum()) >= 10:
                p.bound_extents(P[sel], sigma)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        Vs, Fs, off = [], [], 0
        for _n, p in self.parts:
            V, F = p.mesh()
            Vs.append(V)
            Fs.append(np.asarray(F, np.int64) + off)
            off += len(V)
        return np.vstack(Vs), np.vstack(Fs)

    def dims(self) -> Dict[str, object]:
        return {n: p.dims() for n, p in self.parts}

    def record(self) -> dict:
        return {"template": self.name, "form": self.form,
                "parts": [{"name": n, **p.record()} for n, p in self.parts], "notes": self.notes}


# ── the guided decomposition ───────────────────────────────────────────────────

def point_spacing(P: np.ndarray, seed: int = 0) -> float:
    """The cloud's own scale: the median nearest-neighbour distance (on a sample)."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, np.float64)
    if len(P) < 3:
        return 0.01
    Q = subsample(P, 20000, seed)
    d = cKDTree(Q).query(Q, k=2, workers=4)[0][:, 1]
    return max(float(np.median(d)), 1e-4)


def _components(P: np.ndarray, mask: np.ndarray, cell: float) -> List[np.ndarray]:
    from reconstruction.surface_fit.scene import _residue_components
    return _residue_components(P, mask, float(cell))


def _extract_plane(P: np.ndarray, band: float, iters: int, seed: int) -> Optional[np.ndarray]:
    """The best-supported plane (3-point RANSAC, PCA refinement) → its inlier mask."""
    n = len(P)
    if n < 3:
        return None
    rng = np.random.default_rng(int(seed))
    best, best_n = None, 0
    for _ in range(int(iters)):
        a, b, c = P[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm /= ln
        d = np.abs((P - a) @ nrm)
        k = int((d <= band).sum())
        if k > best_n:
            best, best_n = (nrm, a), k
    if best is None:
        return None
    nrm, a = best
    inl = np.abs((P - a) @ nrm) <= band
    for _ in range(2):
        q = P[inl]
        if len(q) < 3:
            break
        ctr = q.mean(0)
        _, _, vt = np.linalg.svd(q - ctr, full_matrices=False)
        nrm = vt[2]
        inl = np.abs((P - ctr) @ nrm) <= band
    return inl


def _circle_from_3(p: np.ndarray):
    (x1, y1), (x2, y2), (x3, y3) = p
    den = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(den) < 1e-12:
        return None, None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / den
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / den
    return np.array([cx, cy]), float(np.hypot(x1 - cx, y1 - cy))


def _circle_ls(p: np.ndarray):
    A = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))])
    b = (p ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:2]
    r2 = sol[2] + c @ c
    return c, (float(np.sqrt(r2)) if r2 > 0 else None)


def _extract_circle(P2: np.ndarray, axial: np.ndarray, band: float, iters: int, seed: int
                    ) -> Optional[np.ndarray]:
    """A circle in the section plane (3-point RANSAC, least-squares refinement) → its inlier
    mask. Scored by SUPPORT DENSITY — inliers per square metre of the cylinder's lateral surface
    (inliers × axial extent / circumference): a true bar's surface is populated all over, a big
    circle grazing a plate or a tiny one through a dense speck is not."""
    n = len(P2)
    if n < 3:
        return None
    rng = np.random.default_rng(int(seed))
    best, best_score = None, 0.0

    def score(inl: np.ndarray, r: float) -> float:
        k = int(inl.sum())
        if k < 10 or r <= 1e-4:
            return 0.0
        ext = float(np.percentile(axial[inl], 99.5) - np.percentile(axial[inl], 0.5))
        return k * max(ext, 1e-6) / (2 * math.pi * r)

    for _ in range(int(iters)):
        c, r = _circle_from_3(P2[rng.choice(n, 3, replace=False)])
        if c is None or r is None:
            continue
        inl = np.abs(np.linalg.norm(P2 - c, axis=1) - r) <= band
        sc = score(inl, r)
        if sc > best_score:
            best, best_score = (c, r, inl), sc
    if best is None:
        return None
    c, r, inl = best
    for _ in range(2):
        if int(inl.sum()) < 3:
            break
        c2, r2 = _circle_ls(P2[inl])
        if r2 is None:
            break
        inl2 = np.abs(np.linalg.norm(P2 - c2, axis=1) - r2) <= band
        if score(inl2, r2) >= best_score * 0.5:
            c, r, inl = c2, r2, inl2
    return inl


def _plane_density(P: np.ndarray, inl: np.ndarray) -> float:
    """Inliers per square metre of the plane piece (its in-plane bounding box)."""
    q = P[inl]
    if len(q) < 3:
        return 0.0
    ctr = q.mean(0)
    _, _, vt = np.linalg.svd(q - ctr, full_matrices=False)
    a = (q - ctr) @ vt[0]
    b = (q - ctr) @ vt[1]
    area = max(float(np.percentile(a, 99.5) - np.percentile(a, 0.5)), 1e-6) * \
        max(float(np.percentile(b, 99.5) - np.percentile(b, 0.5)), 1e-6)
    return float(inl.sum()) / area


def _circle_density(P2: np.ndarray, axial: np.ndarray, inl: np.ndarray) -> float:
    q = P2[inl]
    if len(q) < 3:
        return 0.0
    c, r = _circle_ls(q)
    if r is None or r <= 1e-4:
        return 0.0
    ext = max(float(np.percentile(axial[inl], 99.5) - np.percentile(axial[inl], 0.5)), 1e-6)
    return float(inl.sum()) / (2 * math.pi * r * ext)


def cloud_density(P: np.ndarray, seed: int = 0) -> float:
    """Points per square metre of measured surface, from the median nearest-neighbour distance d
    (a planar Poisson sampling: mean nearest-neighbour distance = 1 / (2 √ρ) → ρ ≈ 1 / (4 d²))."""
    d = point_spacing(P, seed)
    return 1.0 / (4.0 * d * d)


def extract_by_form(form: str, orientation: str, frame: Frame, P: np.ndarray, band: float,
                    iters: int, seed: int) -> Tuple[Optional[np.ndarray], float]:
    """(inlier mask, support density in points/m²) of ONE instance of ``form`` in ``P`` by the
    RANSAC of that form; (None, 0) when nothing can be extracted."""
    P = np.asarray(P, np.float64)
    if len(P) < 10:
        return None, 0.0
    if form in ("panel", "box", "profile"):
        inl = _extract_plane(P, band, iters, seed)
        return (None, 0.0) if inl is None else (inl, _plane_density(P, inl))
    if form == "cylinder":
        axis = None
        if orientation == "vertical":
            axis = frame.up
        elif orientation == "horizontal":
            # the horizontal direction of most variance
            q = P - np.median(P, 0)
            q = q - np.outer(q @ frame.up, frame.up)
            _, _, vt = np.linalg.svd(q[:: max(1, len(q) // 20000)], full_matrices=False)
            axis = vt[0] / max(np.linalg.norm(vt[0]), 1e-12)
        if axis is not None:
            u = np.cross(axis, frame.X if abs(axis @ frame.X) < 0.9 else frame.Z)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            P2, ax = np.column_stack([P @ u, P @ v]), P @ axis
            inl = _extract_circle(P2, ax, band, iters, seed)
            return (None, 0.0) if inl is None else (inl, _circle_density(P2, ax, inl))
        from reconstruction.geometry.primitives import fit_cylinder_ransac
        fit = fit_cylinder_ransac(P, dist_thresh=band, iters=int(iters), min_inlier_frac=0.0)
        if fit is None:
            return None, 0.0
        inl = np.asarray(fit.inliers, bool)
        ext = max(float(fit.s_extent[1] - fit.s_extent[0]), 1e-6)
        return inl, float(inl.sum()) / (2 * math.pi * max(float(fit.radius), 1e-4) * ext)
    if form == "sphere":
        rng = np.random.default_rng(int(seed))
        best, best_n = None, 0
        n = len(P)
        for _ in range(int(iters)):
            q = P[rng.choice(n, 4, replace=False)]
            A = np.column_stack([2 * q, np.ones(4)])
            b = (q ** 2).sum(1)
            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            c = sol[:3]
            r2 = sol[3] + c @ c
            if r2 <= 0:
                continue
            inl = np.abs(np.linalg.norm(P - c, axis=1) - math.sqrt(r2)) <= band
            if int(inl.sum()) > best_n:
                best, best_n, best_r = inl, int(inl.sum()), math.sqrt(r2)
        if best is None:
            return None, 0.0
        return best, best_n / (4 * math.pi * best_r * best_r)
    # cone, revolution, freeform: the largest connected piece (always a real piece of the cloud)
    comps = _components(P, np.ones(len(P), bool), 3.0 * point_spacing(P, seed))
    return (comps[0], float("inf")) if comps else (None, 0.0)


def build_composite(ans: FormAnswer, P: np.ndarray, frame: Frame, sigma: float, dev, *,
                    lr: float, block_iterations: int, max_blocks: int, huber_k: float,
                    max_points: int, seed: int, ransac_iters: int = 500, error_factor: float = 2.0,
                    log: Callable[[str], None] = lambda m: None) -> Tuple[Composite, dict]:
    """The composite initialised part by part on the object's points: every instance of every
    spec, in the VLM's order, is extracted from what is still unassigned by the RANSAC of ITS
    form (scored by support density), its template built on those points; equal parts share
    one section. What no extraction took is assigned to the nearest part by the joint fit.
    Returns (composite, record): what each part took, what is shared, what the cloud did not
    have (declared missing, never invented)."""
    if not ans.parts:
        raise ValueError("a composite needs the VLM's part list")
    P = np.asarray(P, np.float64)
    n = len(P)
    taken = np.zeros(n, dtype=bool)
    parts: List[Tuple[str, Template]] = []
    rec: Dict[str, object] = {"parts": [], "missing": [], "shared": [], "band_m": None}
    band = float(huber_k) * max(float(sigma), 1e-6)
    tail = TAIL_SIGMAS * max(float(sigma), 1e-6)
    rho = cloud_density(P, seed)
    rec["band_m"], rec["tail_m"], rec["cloud_density_per_m2"] = band, tail, rho
    rec["acceptance"] = (f"every instance the VLM named is extracted (≥ 10 inliers); a support density "
                         f"under the cloud's own ({rho:.0f} pts/m²) or the spec's first instance's, over "
                         f"the user's error factor {error_factor:g}, is declared weak_support")
    for si, spec in enumerate(ans.parts):
        pa = _part_answer(ans, spec)
        shared = shares_section(spec)
        shared_section: Optional[Section] = None
        got = 0
        first_density: Optional[float] = None
        for k in range(int(spec.count)):
            idx = np.nonzero(~taken)[0]
            if len(idx) < 10:
                break
            inl, density = extract_by_form(spec.form, spec.orientation, frame, P[idx], band,
                                           ransac_iters, seed + 97 * si + k)
            if inl is None or int(inl.sum()) < 10:
                break
            # the VLM's count is the form (USER: "si vlm dice una forma es esa forma"): every
            # instance it named is extracted; a weak support (the density under the cloud's own,
            # or the spec's first instance's, over the user's error factor) is DECLARED, never a
            # veto — a chair's backrest seen from the front alone has a tenth of the seat's
            # density and is no less real (pccr chair_609, 2026-10-16); the review round and the
            # consensus judge the whole
            ref = rho if first_density is None else min(rho, first_density)
            weak = bool(density < ref / float(error_factor))
            if weak:
                log(f"[forms] composite part {spec.name} #{k + 1}: the best {spec.form} has a support "
                    f"density of {density:.0f} pts/m² against {ref:.0f} — kept, declared weak")
            sel = idx[inl]
            try:
                t = build_template(pa, P[sel], frame, dev)
            except ValueError as e:
                log(f"[forms] composite part {spec.name}: {len(sel)} points unusable ({e})")
                break
            if first_density is None:
                first_density = density
            # the part takes its points up to the noise tail, not only the Huber band
            with torch.no_grad():
                dist = t.distance(_t(P[idx], dev)).detach().cpu().numpy()
            sel = idx[dist <= tail] if int((dist <= tail).sum()) >= int(inl.sum()) else sel
            if shared and hasattr(t, "section"):
                if shared_section is None:
                    shared_section = t.section
                else:
                    t.section = shared_section          # ONE section for the equal parts
            taken[sel] = True
            got += 1
            name = f"{spec.name}_{got}" if spec.count > 1 else spec.name
            parts.append((name, t))
            rec["parts"].append({"name": name, "form": spec.form, "points": int(len(sel)),
                                 "support_density_per_m2": float(density), "weak_support": weak,
                                 "init": f"the RANSAC of a {spec.form} on the unassigned points "
                                         f"(inliers within {band * 100:.1f} cm = Huber k × σ; "
                                         f"takes its points to {tail * 100:.1f} cm = {TAIL_SIGMAS:g} σ)"})
        if shared and got > 1:
            rec["shared"].append({"spec": spec.name, "section": "one section for all the equal parts",
                                  "relation": spec.relation})
        if got < spec.count:
            rec["missing"].append({"spec": spec.name, "wanted": int(spec.count), "found": int(got),
                                   "reason": "the cloud has no separate piece for it (declared, not invented)"})
            log(f"[forms] composite part {spec.name}: {got} of {spec.count} found in the cloud — "
                f"{spec.count - got} declared missing")
    if not parts:
        raise ValueError("no part of the composite could be initialised")
    comp_t = Composite(parts, dev, notes=[p.relation for p in ans.parts if p.relation])
    rec["unassigned_points"] = int((~taken).sum())
    rec["unassigned_go_to"] = "the nearest part, in the joint fit"
    return comp_t, rec
