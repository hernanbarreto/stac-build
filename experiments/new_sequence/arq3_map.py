"""Step 4 of the architecture (ARQUITECTURA_mapa.md): THE MAP — AMB3R's cloud in metres (step 3's metre), up to the
range (15 m), in places; for every frame, the places it sees: what it measured ITSELF there, from the same side.

THE CLOUD: every frame AMB3R reconstructed (its record: depth on the pixel-stride grid, its camera) back-projected —
the pipeline's own map of what AMB3R saw (reconstruction.coverage_fill.CoverageMap, inside the range), in metres.
THE SESSION'S TOLERANCE: what AMB3R agrees with itself between consecutive keyframes — CoverageMap.tolerance (3σ of
the relative depth disagreement on one surface) on the records of AMB3R's own consecutive keyframes (the only
keyframes there are before the selection).
THE PLACES: the cloud in cubes of side tolerance × the typical distance (the median distance of the cloud's points
from the camera that measured them), and each cube in its FACES — the side each point was measured from (USER
2026-11-22: "imaginate una puerta, que la ves desde adentro y la ves desde afuera, apuntan al mismo lado pero no ven
el mismo punto"): the cube's surface normal is the least spread direction of its points, every point goes to the side
of the camera that measured it, and the points of one side are one face — a place, its centre their mean, its normal
pointing to that side (a cube of fewer than three points: the mean direction to its cameras).
WHAT A FRAME SEES (USER 2026-11-22: each frame counts only what it measured itself): a place is SEEN by a frame when it
projects into the frame's view, the frame's OWN depth agrees with it within the tolerance (|z − d| ≤ tol·d,
CoverageMap's rule) at one pixel at least of the 3 × 3 around where it falls, and the frame looks at it from its side;
HIDDEN when the frame looks at it from its side, does not see it, and its own depth there is nearer by more than the
tolerance (something in front). A frame AMB3R did not reconstruct is its
nearest reconstructed frame (reconstruction.amb3r_probe.source_records, the pipeline's rule).

On the card, deterministic: elementwise tests, sums of the cubes' points by sorted keys, integer counts."""
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

RANGE_M = 15.0                 # his rule: what a view reaches


def video_in_metres(video: dict, metre: float) -> dict:
    """AMB3R's pass with step 3's metre: back to AMB3R's units (make_metric's rule), then × the metre."""
    import arq3_probe as P
    base = P.amb3r_units(video)
    for key in ("pred", "rec_c2w"):
        m = np.asarray(base[key], np.float64).copy()
        m[:, :3, 3] *= float(metre)
        base[key] = m
    base["rec_depth"] = (np.asarray(base["rec_depth"], np.float64) * float(metre)).astype(np.float32)
    return base


def _segment_sums(v, starts, ends):
    """Sums of v (float64, sorted by segment) over [starts, ends): differences of one cumulative sum."""
    import torch
    cs = torch.cat([torch.zeros(1, dtype=torch.float64, device=v.device), torch.cumsum(v, 0)])
    return cs[ends] - cs[starts]


class Map:
    def __init__(self, video_m: dict, range_m: float = RANGE_M, device: str = "cuda", log=print):
        import torch
        from reconstruction.amb3r_probe import source_records
        from reconstruction.coverage_fill import CoverageMap
        t = self.t = torch
        cm = self.cm = CoverageMap(video_m, device, state=False, range_m=float(range_m))
        dev = self.dev = cm.dev
        self.range_m = float(range_m)
        self.Hs, self.Ws = cm.Hs, cm.Ws
        fn = np.asarray(video_m["frame_no"], np.int64)
        self.frame_no = fn
        self.F = len(fn)
        self.rec_index = np.asarray(video_m["rec_index"], np.int64)
        self.rec_of_frame = np.asarray(source_records(video_m, fn.tolist()), np.int64)
        # the session's tolerance, on AMB3R's own consecutive keyframes
        kf = [int(fn[i]) for i in np.asarray(video_m["kf"], np.int64)]
        recs = source_records(video_m, kf)
        pairs = [(int(a), int(b)) for a, b in zip(recs[:-1], recs[1:]) if a != b]
        self.tol, self.noise = cm.tolerance(pairs, log)
        self.tol_pairs = len(pairs)
        owner = torch.repeat_interleave(torch.arange(cm.F, device=dev), cm.n_per)
        self.typical = float((cm.X - cm.C[owner]).norm(dim=1).median())
        self.size = self.tol * self.typical
        self.points = int(cm.N)
        # the cubes: the points sorted by their cube
        X = cm.X
        lo = X.min(0).values.double()
        q = torch.floor((X.double() - lo) / self.size).long()
        ext = q.max(0).values + 1
        key = (q[:, 0] * ext[1] + q[:, 1]) * ext[2] + q[:, 2]
        del q
        order = torch.argsort(key, stable=True)
        _u, counts = torch.unique_consecutive(key[order], return_counts=True)
        del _u
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        seg = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts)
        Xs = X[order].double()
        cen = torch.stack([_segment_sums(Xs[:, a], starts, ends) for a in range(3)], 1) / counts[:, None].double()
        d = Xs - cen[seg]
        cov = torch.empty((len(counts), 3, 3), dtype=torch.float64, device=dev)
        for a in range(3):
            for b in range(a, 3):
                s = _segment_sums(d[:, a] * d[:, b], starts, ends) / counts.double()
                cov[:, a, b] = s
                cov[:, b, a] = s
        del d
        normal = torch.linalg.eigh(cov)[1][:, :, 0]                         # the least spread direction
        obs = cm.C[owner[order]].double() - Xs                               # each point towards its camera
        few = counts < 3
        if bool(few.any()):
            ob = obs / obs.norm(dim=1, keepdim=True).clamp_min(1e-12)
            mean_dir = torch.stack([_segment_sums(ob[:, a], starts, ends) for a in range(3)], 1)
            mean_dir = mean_dir / mean_dir.norm(dim=1, keepdim=True).clamp_min(1e-12)
            normal[few] = mean_dir[few]
            del ob
        back = (obs * normal[seg]).sum(1) < 0                                # the point was measured from behind
        del obs
        # the faces: (cube, side), the points sorted by it
        fkey = seg * 2 + back.long()
        forder = torch.argsort(fkey, stable=True)
        fk, fcount = torch.unique_consecutive(fkey[forder], return_counts=True)
        fends = torch.cumsum(fcount, 0)
        fstarts = fends - fcount
        Xf = Xs[forder]
        self.S = (torch.stack([_segment_sums(Xf[:, a], fstarts, fends) for a in range(3)], 1)
                  / fcount[:, None].double()).float()
        cube = fk // 2
        self.N = (normal[cube] * torch.where(fk % 2 == 1, -1.0, 1.0)[:, None].double()).float()
        self.count = fcount
        self.cube_of = cube
        self.cubes = len(counts)
        self.M = len(fcount)
        # each point's face (in the cloud's own order) — for the pictures
        face_sorted = torch.repeat_interleave(torch.arange(self.M, device=dev), fcount)
        pf = torch.empty(cm.N, dtype=torch.long, device=dev)
        pf[order[forder]] = face_sorted
        self.face_of_point = pf
        del Xs, Xf, order, forder, key, seg, back, fkey, owner, face_sorted
        # every frame's pose (the plan's camera path)
        self.c2w = torch.as_tensor(np.asarray(video_m["pred"], np.float64), device=dev)

    def judge(self, g: int, X, n):
        """(seen, hidden) masks of the points X (with their normals n) at record g: facing it, within the range, in
        its view, and its OWN depth agreeing within the tolerance at one pixel at least of the 3 × 3 around where the
        point falls (USER 2026-11-22, "si, mejor": a point falls 1–2 pixels from where a frame measured it — measured
        on bufferStop — and at an edge AMB3R's depth mixes the front and the back); hidden: not seen, and the frame's
        own depth there nearer by more than the tolerance (something in front)."""
        t, cm = self.t, self.cm
        Xc = X @ cm.R_w2c[g].T + cm.t_w2c[g]
        z = Xc[:, 2]
        zs = t.where(z > 1e-6, z, t.ones_like(z))
        u = t.round(cm.K[g, 0, 0] * Xc[:, 0] / zs + cm.K[g, 0, 2]).long()
        v = t.round(cm.K[g, 1, 1] * Xc[:, 1] / zs + cm.K[g, 1, 2]).long()
        dist = (X - cm.C[g]).norm(dim=1)
        ok = (z > 1e-3) & (u >= 0) & (u < cm.Ws) & (v >= 0) & (v < cm.Hs) & (dist <= self.range_m)
        ok &= ((cm.C[g] - X) * n).sum(1) > 0                               # from the point's side
        off = t.arange(-1, 2, device=self.dev)
        uu = (u[:, None, None] + off[None, None, :]).expand(-1, 3, 3)
        vv = (v[:, None, None] + off[None, :, None]).expand(-1, 3, 3)
        inside = (uu >= 0) & (uu < cm.Ws) & (vv >= 0) & (vv < cm.Hs)
        d = cm.D[g][vv.clamp(0, cm.Hs - 1), uu.clamp(0, cm.Ws - 1)]
        valid = inside & (d > 0)
        zz = z[:, None, None]
        agree = (valid & ((zz - d).abs() <= self.tol * d)).flatten(1).any(1)
        front = (valid & (d < zz - self.tol * d)).flatten(1).any(1)
        seen = ok & agree
        return seen, ok & ~agree & front

    def look_record(self, g: int):
        """(the places record g sees, the places it has hidden) — ids, ascending."""
        cm = self.cm
        dd = self.S - cm.C[g]
        idx = ((dd * dd).sum(1) <= self.range_m ** 2).nonzero()[:, 0]
        seen, hidden = self.judge(g, self.S[idx], self.N[idx])
        return idx[seen], idx[hidden]

    def look(self, i: int):
        """What video position i sees and has hidden: its nearest reconstructed frame's."""
        return self.look_record(int(self.rec_of_frame[i]))

    def up_and_plan(self):
        """The vertical (the camera axis whose world direction stays the same along the walk: the image's vertical,
        whatever the video's rotation) and the plan's two axes (the walk's main direction, its normal)."""
        R = self.c2w[:, :3, :3].double().cpu().numpy()
        means = [R[:, :, a].mean(0) for a in (0, 1)]
        up = means[int(np.argmax([np.linalg.norm(m) for m in means]))]
        up = up / np.linalg.norm(up)
        C = self.c2w[:, :3, 3].double().cpu().numpy()
        Ch = C - np.outer(C @ up, up)
        Ch = Ch - Ch.mean(0)
        _s, _v, vt = np.linalg.svd(Ch, full_matrices=False)
        e1 = vt[0] - (vt[0] @ up) * up
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        return up, e1, e2
