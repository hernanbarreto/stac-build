"""Figure 1: F5 on the synthetic rectangle walk — truth, Omega's drifted trajectory, F5 with
absolute priors, F5 with odometry priors (phase 3b)."""
import os, sys; _S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"); sys.path.insert(0, _S); sys.path.insert(0, os.path.join(_S, "tests"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
import test_loop_gap as T
R, LG, SL = T.R, T.LG, T.SL
c2w = T.rectangle_walk(); X = T.room_points()
track, frame, uv, wid, isl = T.observe(X, c2w, T.GT); split = T.split_of(track)
w2c0 = np.linalg.inv(T.drifted(c2w))
res = {}
for model in ("absolute", "relative"):
    cfg = replace(T.CFG, permutations=200, prior_model=model, prior_sigma_from_loops=False, rolling_shutter="never")
    core = R.refine_core(w2c0, T.GT, T.WH, track, frame, uv, split, list(range(T.N_KF)), 0.01, cfg, T.SOLVER,
                         error_factor=T.FAC, log=lambda *a: None, window_id=wid, is_loop=isl)
    res[model] = core
def centres(w2c): return SL.centres_of(w2c)
def aligned(C, Ct):
    Rk, t = LG.kabsch(C, Ct); return C @ Rk.T + t
Ct = c2w[:, :3, 3]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (title, w2c, key) in zip(axes, [("Omega + gauge (entrada de F5): deriva de rumbo 10°", w2c0, None),
                                        ("F5, priors absolutos (modelo anterior)", res["absolute"]["best"].w2c, "absolute"),
                                        ("F5, priors de odometría (fase 3b)", res["relative"]["best"].w2c, "relative")]):
    C = aligned(centres(w2c), Ct)
    err = np.linalg.norm(C - Ct, axis=1)
    ax.plot(Ct[:, 0], Ct[:, 2], "k-", lw=1.2, label="verdad (rectángulo 6×4 m)")
    ax.plot(C[:, 0], C[:, 2], "o-", ms=3, lw=1, color="tab:red" if err.max() > 0.05 else "tab:green", label="trayectoria")
    for a, b in zip(C, Ct):
        ax.plot([a[0], b[0]], [a[2], b[2]], color="0.6", lw=0.5)
    sm = (res[key]["loops"]["summary"] if key else None)
    gap = (f"gap de loop: {sm['gap_median_after_m']*100:.0f} cm (control {sm['control_median_after_m']*100:.0f} cm)" if key
           else f"gap de loop: {res['relative']['loops']['summary']['gap_median_before_m']*100:.0f} cm")
    ax.set_title(f"{title}\nerror RMS de posición {err.mean()*100:.1f} cm, máx {err.max()*100:.1f} cm · {gap}", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("F5 — escena sintética: 44 keyframes, tracks a 0.3 px, el recorrido vuelve sobre su primer lado (loop)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "01_f5_poses_odometria_vs_absolutos.png")
fig.savefig(out, dpi=130); print("saved", out)
