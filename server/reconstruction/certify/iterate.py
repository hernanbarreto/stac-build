# STAC-Builder — the greedy correction loop (USER's mechanic, 2026-09-14).
#
#   "empezamos época 1 mejora, sigo con época 2, no mejora, pruebo con otra no
#    mejora, pruebo con otra mejora, época 3, pruebo con las que van quedando,
#    ninguna mejora, ok ... la ganadora es la combinación de mejoras que generó
#    la menor desviación"
#
# One duplicate at a time, and the MEASUREMENT decides:
#
#   · try ONE candidate on the current state;
#   · it improves  → it stays, that is the next epoch of the chain, and the
#     loop moves on with the candidates that remain;
#   · it does not  → discard the trial, try the next candidate;
#   · a whole pass over the pool with no improvement → converged.
#
# A candidate that failed goes BACK into the pool: the state moved under it, and
# a correction that did not help before can help now (USER, same day).
#
# Why this shape and not a batch fit. pccr 2026-09-14 has sixteen closures that
# contradict each other by a factor of five (69 to 376 cm for what is the same
# quantity). A least-squares fit over contradictory observations lands on a
# compromise that satisfies none of them — measured: 43 cm applied where the
# desks asked for 112-215, and the closure came out WORSE than before
# (0.946 → 1.228 m). No smooth model conciliates evidence that disagrees; the
# only thing that separates the closures that are right from the ones that are
# wrong is APPLYING one and looking. So the consistency of direction and rate
# is demoted to the ORDER of the trials — get to the answer sooner — and never
# decides: a mis-ordered pool costs time, not correctness.
#
# Why the trials never touch the disk. Fifteen candidates over several levels is
# a hundred-odd trials; an epoch costs minutes (cloud rewrite, octree, witnesses)
# so trials must be free. They are: the correction is per keyframe and every
# point knows its own, so a trial warps the INSTANCES' points and the cameras —
# tens of thousands of points against the cloud's 28 million — and measures
# there. Only the accepted chain is materialised, once, at the end.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── deviation: what "better" means ───────────────────────────────────────

def _offset_m(a: np.ndarray, b: np.ndarray, max_points: int = 4000,
              seed: int = 0) -> float:
    """Median nearest-neighbour distance from copy A to copy B — reported
    alongside the agreement so the curve is readable in metres, never the
    thing that decides."""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]
    d, _ = cKDTree(b).query(a, k=1)
    return float(np.median(d))


class Agreement:
    """THE measurement (USER 2026-09-14): "mejorar es que más puntos de nube
    caigan en las máscaras desde donde se vio, es eso. Y empeorar es justamente
    eso, menos puntos en su máscara."

    How many of an instance's points land inside its mask, over every keyframe
    where the instance was segmented. No separation threshold, no weighted
    objective, no arbitrary percentage: one count of points, and a trial
    improves when the count goes UP.

    Two things make that count mean what it says:

      · BOTH VISITS. A point projected into its OWN keyframe is blind to the
        correction — the point and its camera move together, so it lands
        exactly where it landed before. What moves is the CROSS projection, a
        point of one copy seen from the other visit's frames, and that is
        precisely where a duplicate shows itself. So every point is projected
        into every keyframe of its instance, not only into its own. (The
        opposite rule governs the geometric FILTER in segmentation/mask_filter:
        there, judging a copy by the other visit's frames would delete it. Same
        projection, opposite purpose.)
      · A FIXED SAMPLE. The same points and the same frames are scored on every
        trial, so the count is deterministic and any increase is real — which
        is why no minimum gain has to be invented to protect against noise.
    """

    __slots__ = ("inside", "seen", "per_instance", "separations")

    def __init__(self, inside: int, seen: int, per_instance: Dict[int, tuple],
                 separations: Dict[int, float]):
        self.inside, self.seen = int(inside), int(seen)
        self.per_instance = per_instance
        self.separations = separations

    @property
    def frac(self) -> float:
        return self.inside / self.seen if self.seen else float("nan")

    @property
    def median_sep_m(self) -> float:
        v = [x for x in self.separations.values() if np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    def better_than(self, other: "Agreement") -> bool:
        """More points in their masks. That is the whole rule."""
        return self.inside > other.inside

    def as_dict(self) -> dict:
        return {"points_in_mask": self.inside, "points_seen": self.seen,
                "agreement": self.frac, "median_separation_m": self.median_sep_m,
                "per_instance": {str(k): {"inside": v[0], "seen": v[1]}
                                 for k, v in self.per_instance.items()}}


# ── the candidates ───────────────────────────────────────────────────────

class Candidate:
    """One duplicate: the closure its two copies demand, and the points that
    let a trial be measured without reading anything."""

    __slots__ = ("key", "edge", "idx_a", "idx_b", "i", "j", "instance_id",
                 "label", "oid", "frames_a", "frames_b", "failures",
                 "sample_a", "sample_b", "_masks")

    def __init__(self, key: int, edge: dict, idx_a: np.ndarray, idx_b: np.ndarray,
                 frames_a: Sequence[int], frames_b: Sequence[int],
                 oid: Optional[int], samples: int = 4000, seed: int = 0):
        self.key = int(key)
        self.edge = edge
        self.idx_a, self.idx_b = idx_a, idx_b
        self.frames_a, self.frames_b = list(frames_a), list(frames_b)
        self.oid = oid
        self.i, self.j = int(edge["i"]), int(edge["j"])
        self.instance_id = int(edge.get("instance_id", -1))
        self.label = str(edge.get("label", "?"))
        self.failures = 0
        self._masks: Optional[List[tuple]] = None
        # the SAME points on every trial: a deterministic count is what lets
        # "more points in their mask" be the whole acceptance rule, with no
        # minimum gain invented to protect against sampling noise
        rng = np.random.default_rng(seed + self.key)
        self.sample_a = (rng.choice(len(idx_a), samples, replace=False)
                         if len(idx_a) > samples else np.arange(len(idx_a)))
        self.sample_b = (rng.choice(len(idx_b), samples, replace=False)
                         if len(idx_b) > samples else np.arange(len(idx_b)))

    def masks(self, loop) -> List[tuple]:
        """[(mask frame index, mask)] of this instance — BOTH visits."""
        if self._masks is None:
            ev = loop._evidence()
            self._masks = []
            if ev is not None and getattr(ev, "ok", False) and self.oid is not None:
                for mf, key in ev.frames_for(int(self.oid)):
                    try:
                        self._masks.append((int(mf), np.asarray(ev.masks[key]) > 0))
                    except Exception:  # noqa: BLE001 — a missing mask is one fewer vote
                        continue
        return self._masks

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"<{self.label}#{self.instance_id} kf {self.i}<->{self.j}>"


def _rate_order(pool: List[Candidate], d_kf: np.ndarray) -> List[Candidate]:
    """Trial order, never a filter. Under E(d)=ε·d a closure is only comparable
    to another once divided by the distance WALKED between its two ends — a
    duplicate at 18 m legitimately demands more than one at 5 m — so the
    candidates closest to the consensus RATE are tried first. Getting this
    wrong costs trials, never correctness: the measurement decides.
    Candidates that have already failed more often go last.
    """
    rates = []
    for c in pool:
        walk = abs(float(d_kf[c.i] - d_kf[c.j]))
        t = float(np.linalg.norm(np.asarray(c.edge["X"], np.float64)[:3, 3]))
        rates.append(t / walk if walk > 1e-6 else np.inf)
    finite = [r for r in rates if np.isfinite(r)]
    med = float(np.median(finite)) if finite else 0.0
    return [c for _k, c in sorted(
        zip([(p.failures, abs(r - med) if np.isfinite(r) else np.inf)
             for p, r in zip(pool, rates)], pool),
        key=lambda kv: kv[0])]


# ── the loop ─────────────────────────────────────────────────────────────

class GreedyLoop:
    """Carries the state, the pool and the chain. Built once per session."""

    def __init__(self, session, edges: List[dict], ccfg, cfg,
                 output_dir, session_dir, instances, gcfg,
                 log: Callable[[str], None] = print):
        from correction.distribute import chainage
        self.cfg, self.ccfg, self.gcfg = cfg, ccfg, gcfg
        self.output_dir, self.session_dir = output_dir, session_dir
        self.log = log
        self.base = session                      # never mutated: the on-disk state
        self.state = session                     # the accumulated state
        # certify carries the instances as the LIST segmentation_result.json
        # holds; the correction side indexes them by id. Accept either.
        self.instances: Dict[int, dict] = (
            instances if isinstance(instances, dict) else
            {int(i.get("instance_id", i.get("id", -1))): i for i in (instances or [])})
        self.d_kf = chainage(session.poses)
        self.n_kf = session.n_kf
        self.pool: List[Candidate] = []
        self.chain: List[dict] = []
        self.history: List[dict] = []
        self._ev = None
        self._c2m: Dict[int, int] = {}
        self._kf_of_mask: Dict[int, int] = {}
        self._build_pool(edges)

    # -- setup ------------------------------------------------------------

    def _build_pool(self, edges: List[dict]) -> None:
        from reconstruction.certify.loops_posthoc import _copy_indices
        from segmentation.erase import _mask_obj_by_iid
        try:
            obj_of = _mask_obj_by_iid(self.output_dir)
        except Exception as e:  # noqa: BLE001 — declared; reprojection then abstains
            self.log(f"[greedy] instance→oid map unavailable ({e}) — "
                     f"the reprojection judge will abstain")
            obj_of = {}
        w = int(self.ccfg_window())
        for k, e in enumerate(edges):
            if "X" not in e or not e.get("trusted"):
                continue
            inst = self.instances.get(int(e.get("instance_id", -1)))
            if inst is None:
                continue
            ia, ib, _ks = _copy_indices(self.base, inst, int(e["i"]), int(e["j"]), w)
            if len(ia) == 0 or len(ib) == 0:
                continue
            fa = [self.base.frames[x] for x in range(max(0, e["i"] - w),
                                                     min(self.n_kf, e["i"] + w + 1))]
            fb = [self.base.frames[x] for x in range(max(0, e["j"] - w),
                                                     min(self.n_kf, e["j"] + w + 1))]
            self.pool.append(Candidate(k, e, ia, ib, fa, fb,
                                       obj_of.get(int(e.get("instance_id", -1))),
                                       samples=int(self.gcfg.offset_samples)))
        self.log(f"[greedy] {len(self.pool)} candidate(s) in the pool")

    def ccfg_window(self) -> int:
        return int(self.gcfg.window_kf)

    def _evidence(self):
        if self._ev is None:
            from reconstruction.surface_fit.hole_audit import _Evidence
            from segmentation.pipeline import _mask_frame_lookup
            from pathlib import Path
            self._ev = _Evidence(Path(self.output_dir), Path(self.session_dir))
            try:
                z = np.load(Path(self.output_dir) / "seg_masks.npz", allow_pickle=True)
                self._c2m = _mask_frame_lookup(self.output_dir, z["frames"].tolist(),
                                               sorted({int(f) for f in self.base.fg}))
            except Exception as e:  # noqa: BLE001 — identity is the documented default
                self.log(f"[greedy] mask frame lookup unavailable ({e}) — identity assumed")
                self._c2m = {}
            # mask keyframe POSITION → keyframe index, the translation the
            # projection needs: the mask store is keyed by position, the poses
            # by real frame number
            m2c = {int(m): int(c) for c, m in self._c2m.items()}
            self._kf_of_mask = {}
            for mf in set(m2c) | {int(self.base.frames[k]) for k in range(self.n_kf)}:
                cf = m2c.get(int(mf), int(mf))
                kf = self.base.kf_index.get(int(cf))
                if kf is not None:
                    self._kf_of_mask[int(mf)] = int(kf)
        return self._ev

    # -- one trial --------------------------------------------------------

    def _solution(self, c: Candidate) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The per-keyframe correction ONE candidate implies, under the user's
        drift-rate model: its closure pins E(d)=ε·d and every keyframe gets
        −E(d_k), the reference copy included."""
        from correction.distribute import distribute
        X = np.asarray(c.edge["X"], np.float64)
        span = c.edge.get("later_kfs") or [c.i, c.i]
        sol = [{"anchor_kf": c.i, "kf_span": [int(span[0]), int(span[-1])],
                "R": X[:3, :3], "t": X[:3, 3], "k": 1.0}]
        try:
            R_kf, t_kf, k_kf, _rep = distribute(self.n_kf, self.d_kf, c.j, sol)
        except RuntimeError:
            # the anchor sits at or before the reference: this closure observes
            # nothing the model can spread
            return None
        return R_kf, t_kf, k_kf

    def _warp_points(self, idx: np.ndarray, R_kf, t_kf, k_kf,
                     state) -> np.ndarray:
        from correction.distribute import warp_subset
        return warp_subset(state.xyz, state.fg, state.ks, state.cam_center,
                           idx, R_kf, t_kf, k_kf)

    def _agreement(self, state, R_kf=None, t_kf=None, k_kf=None) -> Agreement:
        """Count the points that land inside their instance's mask, over every
        keyframe the instance was segmented in — on this state, optionally with
        a trial correction applied on top of it."""
        from correction.apply import transform_poses
        from segmentation.shape_proposer import _project_frame
        ev = self._evidence()
        poses = (transform_poses(state.poses, R_kf, t_kf)
                 if R_kf is not None else state.poses)
        inside_tot = seen_tot = 0
        per_inst, seps = {}, {}
        for c in self.pool:
            if R_kf is None:
                a, b = state.xyz[c.idx_a], state.xyz[c.idx_b]
            else:
                a = self._warp_points(c.idx_a, R_kf, t_kf, k_kf, state)
                b = self._warp_points(c.idx_b, R_kf, t_kf, k_kf, state)
            seps[c.key] = _offset_m(a, b, self.gcfg.offset_samples)
            if c.oid is None or ev is None or not getattr(ev, "ok", False):
                continue
            pts = np.vstack([a[c.sample_a], b[c.sample_b]])
            ins = seen = 0
            for mf, m in c.masks(self):
                kf = self._kf_of_mask.get(int(mf))
                if kf is None:
                    continue
                # project with the TRIAL poses: _project_frame reads the pose
                # off the evidence, so the evidence's own map is patched for
                # the duration of this frame only
                saved = ev.cam.pose_map.get(int(self.base.frames[kf]))
                ev.cam.pose_map[int(self.base.frames[kf])] = poses[kf]
                pr = _project_frame(ev, int(self.base.frames[kf]), pts)
                if saved is not None:
                    ev.cam.pose_map[int(self.base.frames[kf])] = saved
                if pr is None:
                    continue
                u, v, _z, front = pr
                mh, mw = m.shape[:2]
                mu = (u * mw / ev.kw).astype(np.int64)
                mv = (v * mh / ev.kh).astype(np.int64)
                inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
                if not inb.any():
                    continue
                seen += int(inb.sum())
                ins += int(m[mv[inb], mu[inb]].sum())
            per_inst[c.key] = (ins, seen)
            inside_tot += ins
            seen_tot += seen
        return Agreement(inside_tot, seen_tot, per_inst, seps)

    # -- the mechanic -----------------------------------------------------

    def run(self, max_epochs: Optional[int] = None) -> dict:
        t0 = time.time()
        max_epochs = int(max_epochs if max_epochs is not None else self.gcfg.max_epochs)
        ag = self._agreement(self.state)
        self.history.append({"epoch": 0, "accepted": None, **ag.as_dict()})
        self.log(f"[greedy] start: {ag.inside:,}/{ag.seen:,} points in their masks "
                 f"({ag.frac * 100:.2f}%), copies {ag.median_sep_m * 100:.1f} cm apart")
        n_trials = 0
        while len(self.chain) < max_epochs and self.pool:
            accepted = None
            for c in _rate_order(self.pool, self.d_kf):
                sol = self._solution(c)
                n_trials += 1
                if sol is None:
                    c.failures += 1
                    self.log(f"[greedy]   trial {c}: observes nothing the model can spread")
                    continue
                R_kf, t_kf, k_kf = sol
                ag_t = self._agreement(self.state, R_kf, t_kf, k_kf)
                # THE rule (USER 2026-09-14): more points in their masks is
                # better, fewer is worse. Nothing else, and no threshold — the
                # sample is fixed, so the count is exact and any change is real.
                if not ag_t.better_than(ag):
                    c.failures += 1
                    self.log(f"[greedy]   trial {c}: {ag_t.inside:,} points in mask "
                             f"vs {ag.inside:,} — worse, discarded")
                    continue
                self.state = self._advance(R_kf, t_kf, k_kf)
                self.pool.remove(c)
                ag = self._agreement(self.state)
                accepted = c
                self.chain.append({"key": c.key, "instance_id": c.instance_id,
                                   "label": c.label, "i": c.i, "j": c.j,
                                   "R_kf": R_kf, "t_kf": t_kf, "k_kf": k_kf})
                self.history.append({"epoch": len(self.chain), "accepted": repr(c),
                                     **ag.as_dict()})
                self.log(f"[greedy] epoch {len(self.chain)}: {c} accepted — "
                         f"{ag.inside:,}/{ag.seen:,} points in their masks "
                         f"({ag.frac * 100:.2f}%), copies {ag.median_sep_m * 100:.1f} cm apart, "
                         f"{len(self.pool)} candidate(s) left")
                break
            if accepted is None:
                self.log(f"[greedy] a full pass over {len(self.pool)} remaining "
                         f"candidate(s) improved nothing — converged")
                break
            # a candidate that failed earlier goes back in: the state moved
            # under it (USER 2026-09-14)
            for c in self.pool:
                c.failures = 0
        return self._report(n_trials, time.time() - t0)

    def _advance(self, R_kf, t_kf, k_kf):
        from reconstruction.certify.run import transformed_session
        return transformed_session(self.state, R_kf, t_kf, k_kf)

    def composed(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The chain as ONE per-keyframe transform, to be materialised once.

        Composition is left-multiplication, the convention every stage here
        uses: R = Rₙ·…·R₁ and t = Rₙ·t_prev + tₙ, per keyframe. Depth factors
        multiply.
        """
        if not self.chain:
            return None
        n = self.n_kf
        R = np.repeat(np.eye(3)[None], n, axis=0)
        t = np.zeros((n, 3))
        k = np.ones(n)
        for step in self.chain:
            Rs, ts, ks = step["R_kf"], step["t_kf"], step["k_kf"]
            t = np.einsum("nij,nj->ni", Rs, t) + ts
            R = np.einsum("nij,njk->nik", Rs, R)
            k = k * ks
        return R, t, k

    def _report(self, n_trials: int, elapsed: float) -> dict:
        first, last = self.history[0], self.history[-1]
        return {"version": 1, "provenance": "tool_measured",
                "measure": "points of an instance landing inside its mask, over every "
                           "keyframe it was segmented in, BOTH visits (USER 2026-09-14: "
                           "\"mejorar es que mas puntos de nube caigan en las mascaras "
                           "desde donde se vio\")",
                "mechanic": "greedy: one candidate at a time, the measurement decides; "
                            "a failed candidate returns to the pool; a full pass with "
                            "no improvement is convergence",
                "epochs": len(self.chain), "trials": n_trials,
                "candidates_left": len(self.pool),
                "points_in_mask_before": first["points_in_mask"],
                "points_in_mask_after": last["points_in_mask"],
                "agreement_before": first["agreement"],
                "agreement_after": last["agreement"],
                "separation_before_m": first["median_separation_m"],
                "separation_after_m": last["median_separation_m"],
                "curve": [{k: v for k, v in h.items() if k != "per_instance"}
                          for h in self.history],
                "chain": [{k: v for k, v in s.items()
                           if k not in ("R_kf", "t_kf", "k_kf")} for s in self.chain],
                "elapsed_s": round(elapsed, 1)}
