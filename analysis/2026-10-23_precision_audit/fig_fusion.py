"""Figure 2: the owner fusion (one point per surface element) and the free-space carving."""
import os, sys; _S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"); sys.path.insert(0, _S); sys.path.insert(0, os.path.join(_S, "tests"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import test_owner_fusion as T
OF, K, H, W = T.OF, T.K, T.H, T.W
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# A/B: seven keyframes of one wall, each with its own depth bias (±0.7 %) → layers → one surface
biases = [1.007, 0.994, 1.003, 0.997, 1.006, 0.995, 1.001]
frames, c2w, depth, valid = T._scene(biases)
before = T._points(frames, c2w, depth, valid)
res = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=True, min_evidence=2, **T.CFG)
after = T._points(frames, c2w, res.depth, res.owner)
def side(ax, P, title, color):
    sel = np.abs(P[:, 1]) < 0.05               # a horizontal slice of the wall
    ax.scatter(P[sel, 0], P[sel, 2], s=2, c=color, alpha=0.6)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z (profundidad) [m]"); ax.grid(alpha=0.3); ax.set_title(title, fontsize=9)
side(axes[0, 0], before, f"ANTES (todas las observaciones publicadas): {len(before):,} puntos — pared a 4 m vista por 7 keyframes,\n"
     f"cada uno con su sesgo de profundidad: espesor RMS al plano {T._plane_rms(before)*100:.1f} cm (capas)", "tab:red")
side(axes[0, 1], after, f"DESPUÉS (fusión por propietario + carving): {len(after):,} puntos — un punto por elemento de superficie,\n"
     f"espesor RMS al plano {T._plane_rms(after)*100:.2f} cm; {res.report['consumed']:,} observaciones consumidas, {res.report['unverified']['rejected']:,} no verificadas", "tab:green")
for ax in axes[0]:
    ax.set_ylim(3.85, 4.35)
# C/D: a floater three cameras agree on, every other camera sees through it
frames, c2w, depth, valid = T._scene([1.0] * 7, xs=[0.3 * (i - 3) for i in range(7)], zs=[0.0] * 7)
blob = np.zeros((H, W), bool); blob[18:30, 26:38] = True
for f in (2, 3, 4):
    depth[f] = np.where(blob, T._wall(c2w[f], z0=2.8).astype(np.float32), depth[f])
r0 = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=False, min_evidence=1, **T.CFG)
r1 = OF.fuse(frames, depth, valid, K, c2w, 0.02, log=lambda *a: None, free_space=True, min_evidence=2, **T.CFG)
P0 = T._points(frames, c2w, r0.depth, r0.owner); P1 = T._points(frames, c2w, r1.depth, r1.owner)
def top(ax, P, title, color):
    ax.scatter(P[:, 0], P[:, 2], s=2, c=color, alpha=0.5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z (profundidad) [m]"); ax.grid(alpha=0.3); ax.set_title(title, fontsize=9)
    ax.set_ylim(2.4, 4.4)
n_fl0 = int(((P0[:, 2] > 2.5) & (P0[:, 2] < 3.1)).sum()); n_fl1 = int(((P1[:, 2] > 2.5) & (P1[:, 2] < 3.1)).sum())
top(axes[1, 0], P0, f"SIN carving: un volador a 2.8 m sobre el que 3 keyframes coinciden sobrevive\n({n_fl0} puntos del volador publicados)", "tab:red")
top(axes[1, 1], P1, f"CON carving de espacio libre (fase 3c): los otros 4 keyframes ven a través de él → rechazado\n"
    f"({n_fl1} puntos del volador; {r1.report['free_space']['rejected']} observaciones con razón 'contradicted')", "tab:green")
fig.suptitle("Fusión por propietario y carving — escenas sintéticas de tests/test_owner_fusion.py (vista superior / corte)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "02_fusion_capas_y_voladores.png")
fig.savefig(out, dpi=120); print("saved", out)
