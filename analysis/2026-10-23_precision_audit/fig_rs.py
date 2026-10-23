"""Figure 4: the rolling shutter — the global-shutter residual grows with the row; the search finds
the readout; the rectified solve is flat in the row."""
import os, sys
_S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"); sys.path.insert(0, _S); sys.path.insert(0, os.path.join(_S, "tests"))
from dataclasses import replace
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import test_precision_refine as T
import test_rolling_shutter as TR
from precision import refine as R
from precision import rolling_shutter as RS

X, c2w = T._scene(seed=4, n_kf=16)
gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
kf = [TR.STEP * k for k in range(len(c2w))]
track, frame, uv = TR._observe_rs(X, c2w, gt, 0.3, 9, kf, TR.TAU_F)
split = T._split(track)
init = np.linalg.inv(T._perturb(c2w, 0.1, 0.01, 3))
cfg_gs = replace(T.CFG, permutations=300, rolling_shutter="never")
cfg_rs = replace(T.CFG, permutations=300, rolling_shutter="judged", rs_passes=2)
gs = R.refine_core(init.copy(), gt, T.WH, track, frame, uv, split, kf, 0.01, cfg_gs, T.SOLVER, error_factor=T.FAC, log=lambda *a: None)
rs = R.refine_core(init.copy(), gt, T.WH, track, frame, uv, split, kf, 0.01, cfg_rs, T.SOLVER, error_factor=T.FAC, log=lambda *a: None)
rep = rs["rolling_shutter"]
H = T.WH[1]


def signed_residuals(core, tau):
    """Held-out observations: the residual's component along the rolling-shutter displacement
    direction d(p) (what a readout would move the pixel by), against the OBSERVED row — measured
    exactly as the stage's held-out is: with a readout tau the observations are rectified from the
    landmarks of the raw observations, the landmarks re-triangulated from the rectified ones, and
    the residual read against those. A degenerate displacement (|d| beyond the diagonal) is skipped."""
    best = core["best"]; params = best.params_by_block[0]; w2c = best.w2c
    held_g = core["held_groups"]
    motion = RS.keyframe_motion(w2c, np.asarray(kf, np.float64))
    Xh = R.triangulate_tracks(held_g, w2c, params, T.SOLVER, cfg_gs.min_tri_deg)
    if tau:
        groups_r = RS.rectify(held_g, Xh, w2c, params, H, motion, tau)
        X2 = R.triangulate_tracks(groups_r, w2c, params, T.SOLVER, cfg_gs.min_tri_deg)
    else:
        groups_r, X2 = held_g, Xh
    omega, dC, df = motion
    cy = float(params[3]); diag = float(np.hypot(*T.WH))
    rows, comp, norm = [], [], []
    for t, obs in held_g.items():
        P = X2.get(t)
        if P is None:
            continue
        for (i, p), (_i2, p_r) in zip(obs, groups_r[t]):
            p = np.asarray(p, np.float64); p_r = np.asarray(p_r, np.float64)
            q0 = R.project(P[None], w2c[i], params)[0]
            s = (float(p[1]) - cy) / H / float(df[i])
            q1 = R.project(P[None], RS.pose_at(w2c[i], omega[i], dC[i], s), params)[0]
            d = q1 - q0
            if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-6 or np.linalg.norm(d) > diag:
                continue
            r = p_r - q0
            if not np.all(np.isfinite(r)) or np.linalg.norm(r) > diag:
                continue
            rows.append(float(p[1])); comp.append(float(r @ d) / float(np.linalg.norm(d))); norm.append(float(np.linalg.norm(r)))
    return np.array(rows), np.array(comp), np.array(norm)


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
bins = np.linspace(0, H, 13)
for ax, (core, tau, name, c) in zip(axes[:2], ((gs, 0.0, f"obturador global (mejor peldaño {gs['best'].name})", "tab:red"),
                                               (rs, rep["tau_frames"], f"rolling shutter corregido: τ = {rep['tau_frames']:.3f} periodos de cuadro (inyectado {TR.TAU_F})", "tab:green"))):
    rows, comp, norm = signed_residuals(core, tau)
    ax.scatter(rows, comp, s=3, c=c, alpha=0.25)
    idx = np.digitize(rows, bins) - 1
    med = [np.median(comp[idx == k]) if (idx == k).any() else np.nan for k in range(len(bins) - 1)]
    ax.plot(0.5 * (bins[1:] + bins[:-1]), med, "k-o", ms=4, label="mediana por banda de filas")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("fila del píxel v"); ax.set_ylabel("residuo held-out a lo largo de la dirección de arrastre [px]")
    ax.set_title(f"{name}\nmediana |r| held-out {np.median(norm):.3f} px", fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_ylim(-2.5, 2.5)
# the search
S = sorted(rep["search"], key=lambda e: e["tau_frames"])
taus = [0.0] + [e["tau_frames"] for e in S]; meds = [rep["heldout_median_px_global_shutter"]] + [e["heldout_median_px"] for e in S]
axes[2].plot(taus, meds, "o-", c="tab:blue")
axes[2].axvline(TR.TAU_F, c="k", ls="--", lw=1, label=f"τ inyectado = {TR.TAU_F}")
axes[2].axvline(rep["tau_frames"], c="tab:green", ls=":", lw=2, label=f"τ elegido = {rep['tau_frames']:.3f}")
m = rep["measured_on_global_shutter_residuals"]
axes[2].axvline(m["tau_frames"], c="tab:red", ls=":", lw=1.5, label=f"lectura directa sobre residuos GS = {m['tau_frames']:.2f}\n(el BA absorbe la mayor parte de la firma)")
axes[2].set_xlabel("τ candidato [periodos de cuadro]"); axes[2].set_ylabel("mediana held-out [px] tras re-resolver")
axes[2].set_title(f"Búsqueda juzgada en rejilla (3f): tomado={rep['taken']}, veredicto {rep['verdict_vs_best'].get('improves')}", fontsize=9)
axes[2].grid(alpha=0.3); axes[2].legend(fontsize=7)
fig.suptitle("Rolling shutter — escena de tests/test_rolling_shutter.py (16 keyframes a 10 cuadros, τ = 0.9 periodos, ruido 0.3 px)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "04_rolling_shutter.png")
fig.savefig(out, dpi=120); print("saved", out)
print("GS held med", gs["rungs"][gs["best"].name]["heldout_median_px"], "RS", rep.get("heldout_median_px_rolling_shutter"))
