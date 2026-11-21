"""THE NEW ORDER, step 5: THE TIE OF THE CUBES, AFTER F5 (USER 2026-11-20: "Atada final de todos los cubos juntos, con
el primero fijo").

Each cube comes out of step 4 adjusted on its own: its versions of the keyframes and its landmarks, in its own frame.
Here every cube but the first gets ONE similarity (a scale, a turn, a shift), all solved TOGETHER, the first cube fixed,
from what the cubes share: a landmark one cube holds must fall, seen through the OTHER cube's versions of the keyframes
that observed it, on the pixels where F4 tracked it. Both things the cubes share are in that one cost — the two
versions of a keyframe (the camera) and the tracks both cubes hold (the landmark) — and it is in PIXELS, F5's own
measure with F5's own robust loss (refine.huber_px): no weight between metres and pixels is invented.

The solve is Levenberg-Marquardt on the exact cost, and it ENDS BY ITSELF: a step is taken only when the cost goes
down; when no step of any size lowers it and the damped step no longer changes any parameter, nothing moves and it stops.

Then ONE pose per keyframe: its version in the cube that OWNS it (the pipeline's own owner rule — the keyframe's Omega
record, whose depth F6 uses, is that cube's).

    python tie_cubes.py <scan dir> <f5_per_cube.npz> <out npz>"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)
CHUNK = 400_000                           # BOUND (memory): observations projected at a time; the result does not depend on it
DAMP = 3.0                                # BOUND (cost): the damping's step up / down; the optimum does not depend on it


def skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1), np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def rot(w):
    import cv2
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


class Tie:
    """The cost over every (landmark of cube a, observation in cube b) with a != b."""

    def __init__(self, groups, params, huber, n_cubes):
        self.g, self.params, self.huber, self.C = groups, np.asarray(params, np.float64), float(huber), n_cubes

    def _project(self, Z, W):
        from precision.refine import project_obs
        return project_obs(Z, W, self.params)

    def _points(self, st, g):
        s, Rm, tv = st
        Ya = s[g["a"]] * g["X"] @ Rm[g["a"]].T
        Y = Ya + tv[g["a"]]
        Z = (Y - tv[g["b"]]) @ Rm[g["b"]] / s[g["b"]]
        return Ya, Y, Z

    def residuals(self, st, g, sl=slice(None)):
        _Ya, _Y, Z = self._points(st, {**g, "X": g["X"][sl]})
        return self._project(Z, g["W"][sl]) - g["uv"][sl]

    def cost(self, st):
        """The Huber cost, summed exactly (math.fsum): a comparison of two costs is never decided by the order of a sum."""
        parts = []
        for g in self.g:
            for i in range(0, len(g["X"]), CHUNK):
                e = np.linalg.norm(self.residuals(st, g, slice(i, i + CHUNK)), axis=1)
                e = e[np.isfinite(e)]
                parts.append(np.where(e <= self.huber, 0.5 * e * e, self.huber * (e - 0.5 * self.huber)))
        return math.fsum(np.concatenate(parts).tolist())

    def normal(self, st):
        """J'WJ and J'Wr over the free cubes (1 .. C-1), 7 unknowns each: turn (3), shift (3), log scale (1)."""
        s, Rm, tv = st
        n = 7 * (self.C - 1)
        H, b = np.zeros((n, n)), np.zeros(n)
        for g in self.g:
            a, c = g["a"], g["b"]
            for i in range(0, len(g["X"]), CHUNK):
                sl = slice(i, i + CHUNK)
                gg = {**g, "X": g["X"][sl]}
                Ya, Y, Z = self._points(st, gg)
                W, uv = g["W"][sl], g["uv"][sl]
                r = self._project(Z, W) - uv
                ok = np.isfinite(r).all(1)
                e = np.linalg.norm(np.where(ok[:, None], r, 0.0), axis=1)
                w = np.where(ok, np.where(e <= self.huber, 1.0, self.huber / np.maximum(e, 1e-300)), 0.0)
                r = np.where(ok[:, None], r, 0.0)
                # d pixel / d Z by central differences on the point (the lens and the camera exactly as F5 projects)
                zc = np.abs(np.einsum("nj,nj->n", W[:, 2, :3], Z) + W[:, 2, 3])
                h = 1e-6 * np.maximum(zc, 1e-9)
                dP = np.empty((len(Z), 2, 3))
                for j in range(3):
                    d = np.zeros_like(Z); d[:, j] = h
                    dP[:, :, j] = (self._project(Z + d, W) - self._project(Z - d, W)) / (2.0 * h)[:, None]
                dP = np.where(np.isfinite(dP), dP, 0.0)
                RbT = Rm[c].T / s[c]
                Ja = Jb = None
                if a > 0:                     # the landmark's cube: Y = s e^l exp(w) R X + t + dt
                    dZ = np.concatenate([np.einsum("ij,njk->nik", RbT, -skew(Ya)),
                                         np.broadcast_to(RbT, (len(Z), 3, 3)),
                                         np.einsum("ij,nj->ni", RbT, Ya)[:, :, None]], axis=2)
                    Ja = np.einsum("nij,njk->nik", dP, dZ)
                if c > 0:                     # the observing cube: Z = R' exp(-w) (Y - t - dt) e^-l / s
                    dZ = np.concatenate([np.einsum("ij,njk->nik", RbT, skew(Y - tv[c])),
                                         np.broadcast_to(-RbT, (len(Z), 3, 3)), -Z[:, :, None]], axis=2)
                    Jb = np.einsum("nij,njk->nik", dP, dZ)
                for (p, Jp) in ((a, Ja), (c, Jb)):
                    if Jp is None:
                        continue
                    ip = slice(7 * (p - 1), 7 * p)
                    b[ip] += np.einsum("n,nij,ni->j", w, Jp, r)
                    for (q, Jq) in ((a, Ja), (c, Jb)):
                        if Jq is not None:
                            H[ip, slice(7 * (q - 1), 7 * q)] += np.einsum("n,nij,nik->jk", w, Jp, Jq)
        return H, b

    def moved(self, st, d):
        s, Rm, tv = (x.copy() for x in st)
        for c in range(1, self.C):
            w, dt, dl = d[7 * (c - 1):7 * (c - 1) + 3], d[7 * (c - 1) + 3:7 * (c - 1) + 6], d[7 * c - 1]
            Rm[c] = rot(w) @ Rm[c]; tv[c] = tv[c] + dt; s[c] = s[c] * math.exp(dl)
        return s, Rm, tv

    def solve(self, st, log):
        cost, mu, it = self.cost(st), 1e-3, 0
        log(f"[tie] start: cost {cost:.6f}")
        while True:
            H, b = self.normal(st)
            taken = False
            while True:
                d = np.linalg.solve(H + mu * np.diag(np.diag(H)), -b)
                new = self.moved(st, d)
                if all(np.array_equal(x, y) for x, y in zip(new, st)):
                    break                                    # the damped step changes no parameter: nothing moves
                c_new = self.cost(new)
                if c_new < cost:
                    st, mu, taken, it = new, mu / DAMP, True, it + 1
                    log(f"[tie] step {it}: cost {c_new:.6f} (down {cost - c_new:.3e}); the largest move of this step: "
                        f"turn {np.degrees(max(np.linalg.norm(d[7 * k:7 * k + 3]) for k in range(self.C - 1))):.2e} deg, "
                        f"shift {100 * max(np.linalg.norm(d[7 * k + 3:7 * k + 6]) for k in range(self.C - 1)):.2e} cm, "
                        f"scale {max(abs(d[7 * k + 6]) for k in range(self.C - 1)):.2e}")
                    cost = c_new
                    break
                mu *= DAMP
            if not taken:
                log(f"[tie] ended by itself after {it} step(s): no step lowers the cost and the damped step moves nothing")
                return st, cost


def main() -> int:
    scan, f5_npz, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out = scan / "output"
    from precision.config import load_precision_config
    from precision.tracks import load_tracks_v2
    pcfg = load_precision_config()
    log = lambda m: print(m, flush=True)   # noqa: E731
    t0 = time.time()
    z = np.load(f5_npz, allow_pickle=False)
    params, c2w, cube_of, kidx, kf = z["params"], z["c2w"], z["cube_of"], z["kf_index"], [int(x) for x in z["kf_frames"]]
    C = int(cube_of.max()) + 1
    w2c = np.linalg.inv(c2w)
    version = {(int(c), int(k)): v for v, (c, k) in enumerate(zip(cube_of, kidx))}
    lm_cube, lm_track, lm_X = z["landmark_cube"], z["landmark_track"], z["landmark_xyz"]
    ok = np.isfinite(lm_X).all(1)
    n_tracks = int(lm_track.max()) + 1
    index = np.full((C, n_tracks), -1, np.int64)
    index[lm_cube[ok], lm_track[ok]] = np.flatnonzero(ok)

    tr = load_tracks_v2(scan, members=("obs_track", "obs_frame", "obs_uv_native"))
    kf_of_frame = {f: i for i, f in enumerate(kf)}
    on_kf = np.isin(tr["obs_frame"], np.asarray(kf))
    ot = tr["obs_track"][on_kf].astype(np.int64)
    ok_ = np.asarray([kf_of_frame[int(f)] for f in tr["obs_frame"][on_kf]], np.int64)
    ouv = np.asarray(tr["obs_uv_native"][on_kf], np.float64)
    in_range = ot < n_tracks
    ot, ok_, ouv = ot[in_range], ok_[in_range], ouv[in_range]
    cubes_of_kf = {}
    for (c, k) in version:
        cubes_of_kf.setdefault(k, []).append(c)

    rng = float(pcfg.view_range_m)
    groups, own = [], []
    for b in range(C):
        in_b = np.asarray([b in cubes_of_kf.get(int(k), ()) for k in range(len(kf))])
        sel = in_b[ok_]
        t_b, k_b, uv_b = ot[sel], ok_[sel], ouv[sel]
        v_b = np.asarray([version[(b, int(k))] for k in range(len(kf)) if in_b[k]])
        v_of_k = np.full(len(kf), -1, np.int64); v_of_k[np.flatnonzero(in_b)] = v_b
        W_b = w2c[v_of_k[k_b]]
        for a in range(C):
            li = index[a, t_b]
            m = li >= 0
            X, W, uv = lm_X[li[m]], W_b[m], uv_b[m]
            # in front of the camera and within THE RANGE (reconstruction.view_range_m), as F5 itself takes them
            Xc = np.einsum("nij,nj->ni", W[:, :3, :3], X) + W[:, :3, 3]
            good = (Xc[:, 2] > 0) & (np.linalg.norm(Xc, axis=1) <= rng)
            g = {"a": a, "b": b, "X": X[good], "W": W[good], "uv": uv[good], "n_out": int((~good).sum())}
            (own if a == b else groups).append(g)
    tie = Tie(groups, params, pcfg.refine.huber_px, C)
    st0 = (np.ones(C), np.tile(np.eye(3), (C, 1, 1)), np.zeros((C, 3)))

    def med(st, g):
        if not len(g["X"]):
            return float("nan")
        e = np.linalg.norm(np.concatenate([tie.residuals(st, g, slice(i, i + CHUNK)) for i in range(0, len(g["X"]), CHUNK)]), axis=1)
        return float(np.median(e[np.isfinite(e)]))

    n_all = sum(len(g["X"]) for g in groups)
    log(f"[tie] {C} cubes; {n_all:,} observations of one cube's landmark through ANOTHER cube's version of the keyframe "
        f"({sum(g['n_out'] for g in groups):,} behind the camera or beyond the {rng:g} m range left out); "
        f"Huber {pcfg.refine.huber_px:g} px (F5's)")
    floor = {g["a"]: med(st0, g) for g in own}
    log("[tie] the floor — a cube's landmarks through its OWN versions (median px): "
        + ", ".join(f"cube {c}: {v:.3f}" for c, v in sorted(floor.items())))
    before = {(g["a"], g["b"]): med(st0, g) for g in groups}
    st, cost = tie.solve(st0, log)
    after = {(g["a"], g["b"]): med(st, g) for g in groups}
    for g in groups:
        if len(g["X"]):
            log(f"[tie]   cube {g['a']}'s landmarks through cube {g['b']}'s versions: {len(g['X']):>9,} observations, "
                f"median {before[(g['a'], g['b'])]:.3f} px -> {after[(g['a'], g['b'])]:.3f} px")
    s, Rm, tv = st
    for c in range(1, C):
        ang = np.degrees(np.arccos(np.clip((np.trace(Rm[c]) - 1) / 2, -1, 1)))
        log(f"[tie]   cube {c} moved as one piece: scale x{s[c]:.5f}, turn {ang:.3f} deg, shift {100 * np.linalg.norm(tv[c]):.2f} cm")

    # every version in the common frame; the two versions of the same keyframe, before / after
    Cc = c2w[:, :3, 3]
    C_t = s[cube_of][:, None] * np.einsum("nij,nj->ni", Rm[cube_of], Cc) + tv[cube_of]
    R_t = np.einsum("nij,njk->nik", Rm[cube_of], c2w[:, :3, :3])
    pairs = [(version[(a, k)], version[(b, k)]) for k, cs in cubes_of_kf.items() for i, a in enumerate(cs) for b in cs[i + 1:]]
    ia, ib = np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])

    def ang_between(Ra, Rb):
        return np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", Ra, Rb) - 1) / 2, -1, 1)))
    d0, d1 = np.linalg.norm(Cc[ia] - Cc[ib], axis=1), np.linalg.norm(C_t[ia] - C_t[ib], axis=1)
    a0, a1 = ang_between(c2w[ia, :3, :3], c2w[ib, :3, :3]), ang_between(R_t[ia], R_t[ib])
    log(f"[tie] the two versions of the same keyframe ({len(pairs)} keyframes in two cubes): apart {np.median(d0) * 100:.2f} cm "
        f"median / {np.percentile(d0, 90) * 100:.2f} worst tenth / {d0.max() * 100:.2f} most BEFORE -> {np.median(d1) * 100:.2f} / "
        f"{np.percentile(d1, 90) * 100:.2f} / {d1.max() * 100:.2f} cm AFTER; turned {np.median(a0):.3f} deg median / "
        f"{a0.max():.3f} most -> {np.median(a1):.3f} / {a1.max():.3f} deg")

    # ONE pose per keyframe: its version in the cube that owns it (its Omega record's chunk)
    owner = np.empty(len(kf), np.int64)
    for i, f in enumerate(kf):
        with np.load(out / "omega_run" / "results_output" / f"frame_{f}.npz") as rec:
            owner[i] = int(rec["chunk"])
    v_own = np.asarray([version[(int(owner[i]), i)] for i in range(len(kf))])
    c2w_kf = np.tile(np.eye(4), (len(kf), 1, 1))
    c2w_kf[:, :3, :3], c2w_kf[:, :3, 3] = R_t[v_own], C_t[v_own]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, params=params, wh=z["wh"], rung=str(z["rung"]), kf_frames=np.asarray(kf), owner=owner, c2w=c2w_kf,
             cube_scale=s, cube_R=Rm, cube_t=tv, versions_c2w=np.concatenate([R_t, C_t[:, :, None]], axis=2),
             cube_of=cube_of, kf_index=kidx,
             report=json.dumps({"observations": n_all, "cost": cost, "floor_px": {str(k): v for k, v in floor.items()},
                                "before_px": {f"{a}->{b}": v for (a, b), v in before.items()},
                                "after_px": {f"{a}->{b}": v for (a, b), v in after.items()},
                                "versions_apart_cm": {"before": [float(np.median(d0) * 100), float(d0.max() * 100)],
                                                      "after": [float(np.median(d1) * 100), float(d1.max() * 100)]}}))
    log(f"[tie] one pose per keyframe (its owner cube's version) written: {dst} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
