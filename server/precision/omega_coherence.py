"""Omega's COHERENCE probe — how many keyframes can ONE Omega pass chain before its
trajectory stops agreeing with the walk? (claude_stac.txt §4-F2 I4, USER 2026-09-29
decision 2B: the chunk size is a measurement of THIS scene, not a constant.)

WHY. The chunk size used to be declared (`chunk_walk_m`, `max_walk_single_pass_m`).
Measured on pccr 2026-08-31: with the same keyframe counts one scene broke and two did
not; the 12 m constant had been calibrated on the INFLATED Omega walk and, fed the
real DA3 walk, made 198-keyframe chunks — Omega's poses then read 31.75 m for a
17.5 m walk (+80 %, chunk 1 stretched 2.5×). VGGT-Long's own FAQ names the mechanism:
drift grows with the number of feed-forward hops at small baseline. Omega knows no
metres; what limits it is how many keyframes it chains in THIS scene.

WHAT IS MEASURED. Nested windows of the keyframes (the same start, lengths from the
config grid up to what the card holds and the session has) are each inferred by
Omega in one pass. For each window the camera centres Omega returns are compared
with the walk the DA3 windows measured (I3, ``intake/walk.json``: chainage per
keyframe, local error only — 1.3 cm median at the seams on pccr). Omega is up to
scale, so the comparison is RELATIVE: the ratio of Omega's arc length to the DA3
chainage over the first half of the window against the same ratio over the second
half. A coherent pass keeps that ratio; a pass that drifted stretches one half. The
bootstrap over the steps of each half (fixed seed) gives the noise of the difference;
the window is coherent when the confidence interval of ``log(ratio_2 / ratio_1)``
holds zero — the same criterion every held-out verdict in this repo uses.

WHAT IT DECIDES. ``chunk_keyframes`` = the largest coherent window; ``single_pass``
when that window is the whole keyframe set (and it fits the card). map_worker sizes
Omega's chunks from it (overlap half). A session where no tested window is coherent
gets the smallest tested length and says so.

CLI: ``python -m precision.omega_coherence --session <dir> --capacity <frames>``.
Report: ``output/omega_coherence.json`` (every window's numbers, tool_measured).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

PROBE_NAME = "omega_coherence.json"
PROBE_VERSION = 1
PROVENANCE = "tool_measured"
LOG_TAG = "[omega-coherence]"
MIN_WINDOW = 4                      # two steps per half: the smallest window with a verdict

Infer = Callable[[List[str], int, str], Dict[str, np.ndarray]]


class CoherenceError(RuntimeError):
    """The probe cannot measure — with the exact reason."""


def probe_lengths(n_kf: int, capacity: int, grid: Sequence[int]) -> List[int]:
    """The window lengths to try: every grid value the session and the card allow, plus
    the whole set when it fits the card (so 'single pass' is a tested window, never an
    extrapolation). Ascending, unique, at least MIN_WINDOW."""
    cap = int(min(int(n_kf), int(capacity)))
    if cap < MIN_WINDOW:
        raise CoherenceError(f"{n_kf} keyframe(s) / capacity {capacity}: fewer than {MIN_WINDOW}, "
                             f"no window to probe")
    ls = {int(v) for v in grid if MIN_WINDOW <= int(v) <= cap}
    ls.add(cap)
    return sorted(ls)


def window_of(files: Sequence[str], length: int, longest: int) -> List[str]:
    """The nested window of ``length`` keyframes: all windows share the START of the
    longest one, which is centred on the session."""
    files = list(files)
    if len(files) <= longest:
        start = 0
    else:
        start = (len(files) - int(longest)) // 2
    return files[start:start + int(length)]


def coherence_of(centres: np.ndarray, chainage: np.ndarray, *, confidence: float,
                 n_boot: int, seed: int) -> Dict[str, Any]:
    """Does Omega keep ONE scale along the window? ``centres`` (L,3) Omega's camera
    centres in the window's order; ``chainage`` (L,) the DA3 walk's chainage of the
    same keyframes (metres). Returns the half-to-half drift in log scale, its
    bootstrap confidence interval and the verdict."""
    c = np.asarray(centres, np.float64)
    ch = np.asarray(chainage, np.float64)
    if c.ndim != 2 or c.shape[1] != 3 or len(ch) != len(c):
        raise CoherenceError(f"centres {c.shape} and chainage {ch.shape} do not describe the "
                             f"same keyframes")
    if len(c) < MIN_WINDOW:
        raise CoherenceError(f"{len(c)} keyframe(s) in the window — {MIN_WINDOW} is the minimum")
    o = np.linalg.norm(np.diff(c, axis=0), axis=1)          # Omega's steps
    d = np.diff(ch)                                         # the walk's steps
    ok = np.isfinite(o) & np.isfinite(d) & (d > 0)
    if ok.sum() < MIN_WINDOW - 1:
        raise CoherenceError(f"{int(ok.sum())} usable step(s) in the window (chainage must be "
                             f"increasing and finite)")
    o, d = o[ok], d[ok]
    n = len(o)
    half = n // 2
    if half < 1 or n - half < 1:
        raise CoherenceError(f"{n} step(s): both halves need at least one step")
    a, b = np.arange(0, half), np.arange(half, n)

    def drift(ia, ib):
        r1 = o[ia].sum() / d[ia].sum()
        r2 = o[ib].sum() / d[ib].sum()
        if r1 <= 0 or r2 <= 0:
            return np.nan
        return float(np.log(r2 / r1))

    obs = drift(a, b)
    rng = np.random.default_rng(int(seed))
    boots = np.array([drift(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True))
                      for _ in range(int(n_boot))], np.float64)
    boots = boots[np.isfinite(boots)]
    if not np.isfinite(obs) or boots.size == 0:
        raise CoherenceError("the halves' scale ratio is not finite — Omega returned "
                             "degenerate camera centres")
    alpha = (1.0 - float(confidence)) / 2.0
    lo, hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1.0 - alpha))
    # the same ratio over quarters: where along the window the scale moves
    q = np.array_split(np.arange(n), 4)
    r_first = o[q[0]].sum() / d[q[0]].sum() if d[q[0]].sum() > 0 else np.nan
    quarters = [float(o[i].sum() / d[i].sum() / r_first) if (d[i].sum() > 0 and r_first > 0)
                else None for i in q]
    return {"n_steps": int(n), "omega_walk_m": float(o.sum()), "da3_walk_m": float(d.sum()),
            "omega_over_da3": float(o.sum() / d.sum()),
            "drift_log": obs, "drift_pct": float((np.exp(obs) - 1.0) * 100.0),
            "ci_log": [lo, hi], "confidence": float(confidence), "n_boot": int(boots.size),
            "quarters_rel_scale": quarters,
            "coherent": bool(lo <= 0.0 <= hi)}


def decide(results: Sequence[Dict[str, Any]], n_kf: int, capacity: int) -> Dict[str, Any]:
    """The largest coherent window sizes the chunks; the whole set coherent = one pass."""
    coherent = [r for r in results if r.get("coherent")]
    tested = [int(r["length"]) for r in results]
    if not tested:
        raise CoherenceError("no window was probed")
    if coherent:
        best = max(int(r["length"]) for r in coherent)
        why = f"largest coherent window of {sorted(tested)}"
    else:
        best = min(tested)
        why = "NO tested window is coherent — the smallest one is used and declared"
    single = bool(best >= int(n_kf) and int(n_kf) <= int(capacity))
    return {"chunk_keyframes": int(best), "overlap_keyframes": 0 if single else int(best) // 2,
            "single_pass": single, "why": why, "none_coherent": not coherent}


def omega_infer(session_dir: Path):
    """The production Omega on the session's own vendor config (resolution and mode as
    the reconstruction runs them). Returns (infer, resolution, mode)."""
    from precision.omega_probe import _vendor_path
    _vendor_path()
    import torch
    from config import cfg as raw_cfg
    from workers.map_worker import _build_vggtomega_config
    from base_models.vggtomega_adapter import VGGTOmegaAdapter
    vcfg = _build_vggtomega_config(raw_cfg, Path(session_dir) / "frames")
    res, mode = int(vcfg["Model"]["omega_resolution"]), str(vcfg["Model"]["omega_mode"])
    ad = VGGTOmegaAdapter(vcfg, device="cuda" if torch.cuda.is_available() else "cpu")
    ad.load()

    def infer(paths: List[str], resolution: int, pmode: str) -> Dict[str, np.ndarray]:
        ad.image_resolution, ad.preproc_mode = int(resolution), str(pmode)
        out = ad.infer_chunk(paths)
        f = lambda t: t.detach().float().cpu().numpy()[0]           # noqa: E731
        return {"c2w": f(out["extrinsic"])}
    return infer, res, mode


def run_coherence(session_dir: Path, ccfg, *, capacity: int, infer: Optional[Infer] = None,
                  resolution: Optional[int] = None, mode: Optional[str] = None,
                  log: Callable = print) -> Dict[str, Any]:
    """Probe the nested windows, decide, write the report."""
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    sel = frames_dir / "selected_frames.json"
    if not sel.exists():
        raise CoherenceError(f"{sel} does not exist — the probe windows the KEYFRAMES")
    files = sorted(json.loads(sel.read_text()).get("selected_files") or [],
                   key=lambda f: int(Path(f).stem))
    frames = [int(Path(f).stem) for f in files]
    from intake.walk import load_walk
    walk = load_walk(session_dir)
    if not walk or not walk.get("chainage"):
        raise CoherenceError(f"{session_dir / 'intake' / 'walk.json'} has no chainage — the DA3 "
                             f"windows (I3) must measure the walk before Omega is probed")
    chain = {int(r["frame"]): float(r["chainage_m"]) for r in walk["chainage"]}
    missing = [f for f in frames if f not in chain]
    if missing:
        raise CoherenceError(f"{len(missing)} keyframe(s) have no chainage in walk.json (e.g. "
                             f"{missing[:5]}) — the walk and the keyframes differ")
    lengths = probe_lengths(len(files), capacity, ccfg.lengths)
    if infer is None:
        infer, res_p, mode_p = omega_infer(session_dir)
        resolution = resolution or res_p
        mode = mode or mode_p
    if resolution is None or mode is None:
        raise CoherenceError("resolution and mode are required with an injected infer")
    longest = max(lengths)
    results = []
    for L in lengths:
        win = window_of(files, L, longest)
        t0 = time.time()
        pred = infer([str(frames_dir / f) for f in win], int(resolution), str(mode))
        c2w = np.asarray(pred["c2w"], np.float64)
        if c2w.shape[0] != len(win):
            raise CoherenceError(f"Omega returned {c2w.shape[0]} pose(s) for {len(win)} frame(s)")
        centres = c2w[:, :3, 3]
        ch = np.array([chain[int(Path(f).stem)] for f in win], np.float64)
        rec = {"length": int(L), "first_frame": int(Path(win[0]).stem),
               "last_frame": int(Path(win[-1]).stem), "seconds": round(time.time() - t0, 2),
               **coherence_of(centres, ch, confidence=ccfg.heldout_confidence,
                              n_boot=ccfg.bootstrap, seed=ccfg.seed)}
        results.append(rec)
        log(f"{LOG_TAG} L={L}: Omega {rec['omega_walk_m']:.2f} m vs DA3 {rec['da3_walk_m']:.2f} m, "
            f"half-to-half scale drift {rec['drift_pct']:+.1f} % "
            f"(CI {np.exp(rec['ci_log'][0]) * 100 - 100:+.1f}..{np.exp(rec['ci_log'][1]) * 100 - 100:+.1f} %) "
            f"→ {'coherent' if rec['coherent'] else 'DRIFTED'} ({rec['seconds']:.1f} s)")
    verdict = decide(results, len(files), capacity)
    from intake.quality import read_session_epochs
    report = {"version": PROBE_VERSION, "provenance": PROVENANCE, **read_session_epochs(session_dir),
              "params": {"lengths": list(ccfg.lengths), "heldout_confidence": ccfg.heldout_confidence,
                         "bootstrap": ccfg.bootstrap, "seed": ccfg.seed, "resolution": int(resolution),
                         "mode": str(mode), "capacity": int(capacity), "n_keyframes": len(files)},
              "windows": results, **verdict,
              "decides": "reconstruction.simple chunk size: chunk_keyframes / overlap_keyframes, "
                         "single_pass — read by workers.map_worker before the Omega pass"}
    out = session_dir / "output" / PROBE_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    log(f"{LOG_TAG} {verdict['why']} → chunk {verdict['chunk_keyframes']} keyframes"
        + (" (ONE pass)" if verdict["single_pass"] else f", overlap {verdict['overlap_keyframes']}")
        + f" → {out}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.omega_coherence",
                                 description="Omega coherence probe: the chunk size, measured.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--capacity", type=int, required=True,
                    help="frames the card holds in one Omega pass (map_worker's measured capacity)")
    args = ap.parse_args(argv)
    cc = load_precision_config().omega.coherence_probe
    if not cc.enabled:
        print(f"{LOG_TAG} reconstruction.precision.omega.coherence_probe.enabled is false")
        return 0
    run_coherence(Path(args.session), cc, capacity=args.capacity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
