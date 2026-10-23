"""Figure 3: the bend — Omega's record (monocular scale-and-shift error in inverse depth) bent to
F5's landmarks under the two families; the judged family gives the plane back."""
import os, sys, tempfile
_S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"); sys.path.insert(0, _S); sys.path.insert(0, os.path.join(_S, "tests"))
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import test_depth_on_f5_compute as T
from precision import bend_model as BM
from precision.depth_on_f5 import compute

K = np.array([[T.FK[0], 0, T.FK[2]], [0, T.FK[1], T.FK[3]], [0, 0, 1]])
tmp = Path(tempfile.mkdtemp())
A_, B_ = 1.15, -0.08
s, cam, c2w = T._session(tmp / "a", a=A_, b0=B_)
res = {m: compute(s, T._pcfg(m), log=lambda *a: None) for m in (BM.INVERSE_DEPTH, BM.AFFINE_PIXEL)}
judged = compute(s, T._pcfg(BM.JUDGED), log=lambda *a: None)
mid = T.FRAMES[2]
raw = np.load(s / "output" / "omega_run" / "results_output" / f"frame_{mid}.npz")["depth"].astype(np.float64)
truth = T._plane_depth(c2w[2])


def world_pts(dep, T_c2w, mask):
    uu, vv = np.meshgrid(np.arange(T.FW), np.arange(T.FH))
    zz = dep[mask]
    X = np.stack([(uu[mask] - T.FK[2]) / T.FK[0] * zz, (vv[mask] - T.FK[3]) / T.FK[1] * zz, zz], 1)
    return X @ T_c2w[:3, :3].T + T_c2w[:3, 3]


n = np.array([0.1, 0.0, -1.0]); n /= np.linalg.norm(n); d0 = -3.0 / np.linalg.norm([0.1, 0, -1])
def dist_to_plane(P):
    return P @ n - d0

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
row = T.FH // 2
u = np.arange(T.FW)
axes[0].axhline(0, c="k", lw=2, label="plano verdadero (x·0.1 − z = −3)")
axes[0].plot(u, (raw[row] - truth[row]) * 100, "r--", lw=1.5,
             label=f"registro Omega, 1/z' = (1/z − b)/a con a={A_}, b={B_}: RMS al plano {T._plane_rms(raw, K)*100:.2f} cm")
for m, c, lab in ((BM.AFFINE_PIXEL, "tab:orange", "familia afín-en-píxel"), (BM.INVERSE_DEPTH, "tab:green", "familia en profundidad inversa")):
    z, agree, contra = res[m].voted[mid]
    zz = np.where(z > 0, z, np.nan)
    axes[0].plot(u, (zz[row] - truth[row]) * 100, "-", c=c, lw=1.5, label=f"{lab}: RMS al plano {T._plane_rms(z, K)*100:.3f} cm")
axes[0].set_xlabel("columna de píxel (fila central)"); axes[0].set_ylabel("z estimado − z verdadero [cm]")
axes[0].set_title("Keyframe central: el registro y las dos familias de bend, contra el plano", fontsize=9); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

# held-out score of the judged run, per window
sc = judged.score
ws = sorted(sc[BM.INVERSE_DEPTH])
x = np.arange(len(ws)); wdt = 0.38
axes[1].bar(x - wdt / 2, [sc[BM.AFFINE_PIXEL][w] * 100 for w in ws], wdt, color="tab:orange", label="afín-en-píxel")
axes[1].bar(x + wdt / 2, [sc[BM.INVERSE_DEPTH][w] * 100 for w in ws], wdt, color="tab:green", label="profundidad inversa")
axes[1].set_xticks(x); axes[1].set_xticklabels([f"ventana {w}" for w in ws]); axes[1].set_ylabel("error held-out [cm] (filas que el ajuste no vio) — escala log")
axes[1].set_yscale("log")
for k, w in enumerate(ws):
    for off, mm in ((-wdt / 2, BM.AFFINE_PIXEL), (wdt / 2, BM.INVERSE_DEPTH)):
        v = sc[mm][w] * 100
        axes[1].annotate(f"{v:.2e}" if v < 1e-3 else f"{v:.3f}", (k + off, max(v, 1e-7)), ha="center", va="bottom", fontsize=7)
axes[1].set_title(f"bend.model: judged → elige '{judged.window_rule['best_model']}' (score_B {judged.score_B*100:.2f} cm)", fontsize=9)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")

# the fused owner points of all five keyframes: distance to the true plane
for m, c, lab in ((BM.AFFINE_PIXEL, "tab:orange", "afín-en-píxel"), (BM.INVERSE_DEPTH, "tab:green", "profundidad inversa")):
    F = res[m].fusion
    P = []
    for i, f in enumerate(res[m].frames):
        msk = F.owner[f] & (F.depth[f] > 0)
        P.append(world_pts(F.depth[f].astype(np.float64), c2w[i], msk))
    P = np.concatenate(P); dd = dist_to_plane(P) * 100
    axes[2].hist(dd, bins=80, range=(-1.5, 1.5), color=c, alpha=0.6, label=f"{lab}: {len(P):,} propietarios, RMS {np.sqrt(np.mean(dd**2)):.2f} cm, "
                 f"espacio libre rechazado {F.report['free_space']['rejected']}")
axes[2].set_xlabel("distancia firmada al plano verdadero [cm]"); axes[2].set_ylabel("puntos publicados")
axes[2].set_title("Nube publicada (5 keyframes, fusión por propietario): distancia al plano", fontsize=9); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
fig.suptitle("F6 bend sobre F5 — sesión sintética de tests/test_depth_on_f5_compute.py (plano oblicuo, 5 keyframes, 300 landmarks)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "03_bend_familias.png")
fig.savefig(out, dpi=120); print("saved", out)
