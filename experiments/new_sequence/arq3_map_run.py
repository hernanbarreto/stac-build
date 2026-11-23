"""Line 3 of CHECKLIST_implementacion.md — step 4, THE MAP — on the three test sessions: the map built with step 3's
metre, every frame's look (what it sees, what it has in view and is hidden), the tolerance and the sector measured,
and a picture per session for his eye: in plan, which frames see six places along the walk and, for one place that
something passes in front of, which frames see it and which have it hidden, with one image of each. Read only on the
sessions; written in /workspace/tmp/new_sequence/arq/map/ for the check (scripts/checklist/check_arq_3_map.sh).

    python arq3_map_run.py"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
OUT = Path("/workspace/tmp/new_sequence/arq/map")
PROBE = Path("/workspace/tmp/new_sequence/arq/probe")             # step 3's metre (arq3_probe_run.py)
SESSIONS = (("bufferStop", "bufferStop/scans/2026-08-11/src_default"),
            ("pccr_2026-08-31", "pccr/scans/2026-08-31/src_default"),
            ("pccr_2026-08-24", "pccr/scans/2026-08-24/src_default"))
PLACES = 6                     # places along the walk in the picture
CANDIDATES = 200               # the most hidden places looked at for the picture's example
REPEAT = 8                     # frames whose look is computed twice (the same answer)
SLICE_M = 0.5                  # the picture's plan: the map within this of the cameras' height (display only)


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runs_score(st: np.ndarray):
    """The frames, in video order, of the longest stretch in which a place is hidden between two frames that see it
    within one continuous stretch of having it in view: (hidden frames, the run's first, last) — status 2 seen,
    1 hidden, 0 not in view."""
    best = (0, -1, -1)
    i, n = 0, len(st)
    while i < n:
        if st[i] == 0:
            i += 1
            continue
        j = i
        while j < n and st[j] > 0:
            j += 1
        s = st[i:j]
        seen = np.nonzero(s == 2)[0]
        if len(seen) >= 2:
            mid = s[seen[0]:seen[-1] + 1]
            k = int((mid == 1).sum())
            if k > best[0]:
                best = (k, i + int(seen[0]), i + int(seen[-1]))
        i = j
    return best


def picture(name, scan, M, samples, example, st_ex, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from intake.quality import frame_file
    up, e1, e2 = M.up_and_plan()
    S = M.S.double().cpu().numpy()
    C = M.c2w[:, :3, 3].double().cpu().numpy()
    hc = float(np.median(C @ up))
    band = np.abs(S @ up - hc) <= SLICE_M
    Sx, Sy = S[band] @ e1, S[band] @ e2
    Cx, Cy = C @ e1, C @ e2
    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(2, PLACES)
    for k, smp in enumerate(samples):
        ax = fig.add_subplot(gs[0, k])
        ax.scatter(Sx, Sy, s=0.05, c="0.75", rasterized=True)
        ax.plot(Cx, Cy, lw=0.4, c="0.45")
        st = smp["status"]
        for code, col in ((1, "red"), (2, "limegreen")):
            m = st == code
            ax.scatter(Cx[m], Cy[m], s=3, c=col)
        X = S[smp["sector"]]
        ax.scatter([X @ e1], [X @ e2], marker="*", s=220, c="blue", edgecolors="k", zorder=5)
        ax.set_title(f"place {k + 1}: seen by {int((st == 2).sum())} frames,\nhidden for {int((st == 1).sum())}",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    # the example: a place something passes in front of
    ax = fig.add_subplot(gs[1, :2])
    X = S[example["sector"]]
    sel = st_ex > 0
    pts = np.vstack([C[sel], X[None]])
    lo, hi = (pts @ np.stack([e1, e2], 1)).min(0) - 1.0, (pts @ np.stack([e1, e2], 1)).max(0) + 1.0
    inb = (Sx >= lo[0]) & (Sx <= hi[0]) & (Sy >= lo[1]) & (Sy <= hi[1])
    ax.scatter(Sx[inb], Sy[inb], s=0.4, c="0.55", rasterized=True)
    ax.plot(Cx, Cy, lw=0.5, c="0.45")
    for code, col in ((1, "red"), (2, "limegreen")):
        m = st_ex == code
        ax.scatter(Cx[m], Cy[m], s=8, c=col)
    a, b = example["run_first"], example["run_last"]
    for f, col in ((example["hidden_frame"], "red"), (example["seen_frame"], "limegreen")):
        ax.plot([C[f] @ e1, X @ e1], [C[f] @ e2, X @ e2], c=col, lw=1.2)
    ax.scatter([X @ e1], [X @ e2], marker="*", s=300, c="blue", edgecolors="k", zorder=5)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"a place hidden while the camera passes: frames {int(M.frame_no[a])}–{int(M.frame_no[b])} — "
                 f"seen by {int((st_ex[a:b + 1] == 2).sum())}, hidden for {example['score']}", fontsize=10)
    for col_i, (key, label, color) in enumerate((("hidden_frame", "hidden", "red"), ("seen_frame", "seen", "limegreen"))):
        ax = fig.add_subplot(gs[1, 2 + 2 * col_i:4 + 2 * col_i])
        f = example[key]
        img = Image.open(frame_file(scan / "frames", int(M.frame_no[f]))).convert("RGB")
        sc = img.size[0] / M.Ws
        ax.imshow(img)
        ax.add_patch(matplotlib.patches.Circle(((example[key + "_uv"][0] + 0.5) * sc - 0.5,
                                                (example[key + "_uv"][1] + 0.5) * sc - 0.5),
                                               radius=max(8.0, 0.04 * img.size[0]), fill=False, ec=color, lw=3))
        ax.set_title(f"frame {int(M.frame_no[f])}: the place {label}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{name} — the map in plan (sectors within {SLICE_M} m of the cameras' height; green: frames that "
                 f"see the place ★, red: frames that have it in view and something in front)", fontsize=13)
    fig.tight_layout()
    fig.savefig(dst, dpi=70)
    plt.close(fig)


def project(M, i, X):
    """(u, v) of the world point X on video position i's pixel-stride grid."""
    Xc = M.Rw2c[i].double().cpu().numpy() @ X + M.tw2c[i].double().cpu().numpy()
    K = M.Kf[i].double().cpu().numpy()
    return [float(K[0, 0] * Xc[0] / Xc[2] + K[0, 2]), float(K[1, 1] * Xc[1] / Xc[2] + K[1, 2])]


def main() -> int:
    import torch
    import arq3_map as A
    from card import card
    code = {n: sha256(HERE / n) for n in ("arq3_map.py", "arq3_map_run.py", "arq3_probe.py")}
    OUT.mkdir(parents=True, exist_ok=True)
    with card("step 4 of the architecture: the map (line 3)", session="pccr"):
        for name, rel in SESSIONS:
            t0 = time.time()
            scan = P / rel
            metre = float(json.loads((PROBE / f"{name}.json").read_text())["metre"]["new"]["scale"])
            import arq3_probe as PR
            video = PR.load_video(scan / "intake" / "amb3r_video.npz")
            M = A.Map(A.video_in_metres(video, metre), log=print)
            t_build = time.time() - t0
            print(f"[map] {name}: tolerance {M.tol * 100:.2f} % ({M.tol_pairs} pairs of AMB3R's consecutive "
                  f"keyframes), typical distance {M.typical:.3f} m → sector {M.size * 100:.2f} cm; {M.points:,} "
                  f"points → {M.M:,} sectors; {M.F} frames ({t_build:.0f} s)", flush=True)
            dev = M.dev
            seen_n = torch.zeros(M.M, dtype=torch.int32, device=dev)
            hid_n = torch.zeros(M.M, dtype=torch.int32, device=dev)
            per_seen = np.zeros(M.F, np.int64)
            per_view = np.zeros(M.F, np.int64)
            one = torch.ones(1, dtype=torch.int32, device=dev)
            t1 = time.time()
            for i in range(M.F):                                       # pass 1: every frame's look
                s, v = M.look(i)
                hidden = v[~torch.isin(v, s)]
                seen_n.index_add_(0, s, one.expand(len(s)))
                hid_n.index_add_(0, hidden, one.expand(len(hidden)))
                per_seen[i], per_view[i] = len(s), len(v)
            t_look = time.time() - t1
            rep_frames = np.unique(np.round(np.linspace(0, M.F - 1, REPEAT)).astype(int)).tolist()
            same = all(torch.equal(M.look(i)[0], M.look(i)[0]) and torch.equal(M.look(i)[1], M.look(i)[1])
                       for i in rep_frames)
            # the places along the walk: at frames k/(PLACES+1) of the video, the seen sector nearest the image centre
            samples = []
            for k in range(1, PLACES + 1):
                i = int(round(k * M.F / (PLACES + 1)))
                s, _v = M.look(i)
                s = s[seen_n[s] >= 2]
                Pc = M.S[s] @ M.Rw2c[i].T + M.tw2c[i]
                K = M.Kf[i]
                du = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2] - (M.Ws - 1) / 2.0
                dv = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2] - (M.Hs - 1) / 2.0
                samples.append({"frame": i, "sector": int(s[torch.argmin(du * du + dv * dv)])})
            # the example's candidates: the most hidden of the places seen by many frames
            cand_ok = (seen_n >= 30).nonzero()[:, 0]
            top = cand_ok[torch.argsort(hid_n[cand_ok], descending=True, stable=True)[:CANDIDATES]]
            cands = torch.cat([torch.as_tensor([x["sector"] for x in samples], device=dev), top])
            status = torch.zeros((len(cands), M.F), dtype=torch.int8, device=dev)
            for i in range(M.F):                                       # pass 2: their status at every frame
                s, v = M.look(i)
                status[torch.isin(cands, v), i] = 1
                status[torch.isin(cands, s), i] = 2
            status = status.cpu().numpy()
            for k, smp in enumerate(samples):
                smp["status"] = status[k]
            scores = [runs_score(status[len(samples) + j]) for j in range(len(top))]
            j = int(np.argmax([sc[0] for sc in scores]))
            k_h, a, b = scores[j]
            st_ex = status[len(samples) + j]
            run = np.arange(a, b + 1)
            hid = run[st_ex[a:b + 1] == 1]
            hf = int(hid[len(hid) // 2])
            sf = int(run[st_ex[a:b + 1] == 2][np.argmin(np.abs(run[st_ex[a:b + 1] == 2] - hf))])
            sec = int(top[j])
            Xs = M.S[sec].double().cpu().numpy()
            example = {"sector": sec, "score": int(k_h), "run_first": int(a), "run_last": int(b),
                       "hidden_frame": hf, "seen_frame": sf,
                       "hidden_frame_uv": project(M, hf, Xs), "seen_frame_uv": project(M, sf, Xs)}
            png = OUT / f"{name}.png"
            picture(name, scan, M, samples, example, st_ex, png)
            np.savez(OUT / f"{name}.npz", metre=metre, range_m=M.range_m, tol=M.tol, noise=M.noise,
                     typical=M.typical, size=M.size, sector_centre=M.S.cpu().numpy(),
                     sector_points=M.count.cpu().numpy(), sector_seen_by=seen_n.cpu().numpy(),
                     sector_hidden_for=hid_n.cpu().numpy(), frame_no=M.frame_no,
                     c2w=M.c2w.cpu().numpy(), K=M.Kf.cpu().numpy(), grid_hw=np.array([M.Hs, M.Ws]),
                     frame_sees=per_seen, frame_in_view=per_view)
            doc = {"session": name, "scan": str(scan), "code": code, "metre": metre, "range_m": M.range_m,
                   "tolerance": M.tol, "tolerance_noise": M.noise, "tolerance_pairs": M.tol_pairs,
                   "typical_distance_m": M.typical, "sector_m": M.size, "points": M.points, "sectors": M.M,
                   "frames": M.F, "look_repeated_same": bool(same), "look_repeated_frames": rep_frames,
                   "frame_sees": {"p10": float(np.percentile(per_seen, 10)), "median": float(np.median(per_seen)),
                                  "p90": float(np.percentile(per_seen, 90))},
                   "frame_hidden_share_median": float(np.median(1.0 - per_seen / np.maximum(per_view, 1))),
                   "places": [{"frame": int(M.frame_no[x["frame"]]), "sector": x["sector"],
                               "seen_by": int((x["status"] == 2).sum()), "hidden_for": int((x["status"] == 1).sum())}
                              for x in samples],
                   "example": {**example, "hidden_frame": int(M.frame_no[hf]), "seen_frame": int(M.frame_no[sf]),
                               "run_first": int(M.frame_no[a]), "run_last": int(M.frame_no[b])},
                   "picture": str(png),
                   "seconds": {"build": round(t_build, 1), "look_all_frames": round(t_look, 1),
                               "total": round(time.time() - t0, 1)}}
            (OUT / f"{name}.json").write_text(json.dumps(doc))
            print(f"[map] {name}: a frame sees {doc['frame_sees']['median']:.0f} sectors (p10 "
                  f"{doc['frame_sees']['p10']:.0f}, p90 {doc['frame_sees']['p90']:.0f}), hides "
                  f"{doc['frame_hidden_share_median'] * 100:.0f} % of what it has in view; every frame's look in "
                  f"{t_look:.0f} s; the same look twice: {same}; example: frames {doc['example']['run_first']}–"
                  f"{doc['example']['run_last']}, hidden for {k_h} → {png}", flush=True)
            del M, seen_n, hid_n, status
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
