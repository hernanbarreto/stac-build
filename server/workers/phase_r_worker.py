# STAC-Builder: Phase R Worker (Subprocess)
# Runs the semantic-anchoring refinement (server/phase_r) as a pipeline stage
# between cloud cleaning and TSDF, so refined poses/depth land in
# camera_poses.txt / results_output BEFORE the fusion reads them.
#
# The stage is an ENHANCEMENT layer with its own R.9 fail-safe: it never fails
# the reconstruction. No segmentation yet → skipped with a note; any internal
# error → logged, baseline artifacts untouched (writeback is backup-gated).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import json
import sys
from multiprocessing.connection import Connection
from pathlib import Path

from workers.base import WorkerPipe, run_worker_safe


def _window_map(session_dir: Path, output_dir: Path, config: dict) -> dict[int, str]:
    """frame number -> window id, mirroring the reconstruction chunking
    (chunk_size/chunk_overlap). Overlap frames are OWNED by the chunk whose
    centre is nearest — re-identification across windows itself happens purely
    by tracking continuity (R.1), this map only feeds the Sim(3) pose graph.

    SINGLE-WINDOW sessions (short captures fitting one chunk) get the SAME
    correction machinery via a pseudo-split: the one window is divided into
    contiguous frame segments (s000, s001, …) of ~single_window_split_frames
    each, and the inter-window Sim(3) graph runs BETWEEN the segments —
    intra-chunk drift is real (test2 left pairs at 10 cm inside one chunk).
    Instances tracked across a segment boundary provide the shared
    observations, exactly as chunk overlap does for multi-window sessions.
    The cloud writeback handles the non-chunk-aligned windows per point via
    the chunk origins sidecar."""
    frames_txt = None
    for base in (output_dir / "omega_run", output_dir / "maplong_run",
                 output_dir / "da3_run", output_dir):
        p = base / "camera_frames.txt"
        if p.exists():
            frames_txt = p
            break
    if frames_txt is None:
        return {}
    nums = [int(float(x)) for x in frames_txt.read_text().split()]
    recon = config.get("reconstruction", {})
    backend = recon.get("backend", "vggtomega")
    bcfg = recon.get(backend, {})
    chunk = int(bcfg.get("chunk_size", 120))
    overlap = int(bcfg.get("chunk_overlap", bcfg.get("overlap", 60)))
    stride = max(1, chunk - overlap)
    n_chunks = max(1, (max(len(nums) - overlap, 1) + stride - 1) // stride)
    out: dict[int, str] = {}
    for i, fn in enumerate(nums):
        k = int(round((i - chunk / 2) / stride))
        k = min(max(k, 0), n_chunks - 1)
        out[fn] = f"w{k:03d}"

    # single window → pseudo-split so the SAME Sim(3) anchoring applies
    split = int(config.get("phase_r", {}).get("single_window_split_frames", 60))
    if split > 0 and len(set(out.values())) <= 1 and len(nums) >= 2 * split:
        n_seg = max(2, round(len(nums) / split))
        seg_len = -(-len(nums) // n_seg)          # ceil
        out = {fn: f"s{min(i // seg_len, n_seg - 1):03d}"
               for i, fn in enumerate(nums)}
    return out


def _phase_r_work(pipe: WorkerPipe, session_dir: str, config: dict):
    session_path = Path(session_dir)
    output_dir = (session_path / "output").resolve()

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    if not (output_dir / "segmentation.json").exists():
        pipe.send_log("Phase R skipped: no segmentation.json yet (run the "
                      "auto-prompter / Segmentation Manager, then re-run TSDF "
                      "to fuse with anchoring)", level="warning")
        pipe.send_progress(100, "Phase R skipped (no segmentation)", stage="phase_r")
        return

    if pipe.check_cancel():
        return

    # Forward EVERY phase_r module log line ("PhaseR" logger: pipeline,
    # writeback, scale_check) to the pipe → server.log + UI, so a full run is
    # debuggable end to end from the log alone.
    import logging as _logging
    _pr_logger = _logging.getLogger("PhaseR")

    class _PipeLogHandler(_logging.Handler):
        def emit(self, record):
            try:
                pipe.send_log(f"[PhaseR] {record.getMessage()}",
                              level="warning" if record.levelno >= _logging.WARNING else "info")
            except Exception:
                pass

    if not any(isinstance(h, _PipeLogHandler) for h in _pr_logger.handlers):
        _ph = _PipeLogHandler()
        _ph.setLevel(_logging.INFO)
        _pr_logger.addHandler(_ph)
    _pr_logger.setLevel(_logging.INFO)

    try:
        from phase_r.pipeline import PhaseRPipeline

        pipe.send_progress(5, "Building instance store (R.1/R.8)...", stage="phase_r")
        wmap = _window_map(session_path, output_dir, config)
        n_windows = len(set(wmap.values())) if wmap else 1
        _pseudo = bool(wmap) and next(iter(set(wmap.values()))).startswith("s")
        pipe.send_log(f"Phase R: {len(wmap)} frames across {n_windows} window(s)"
                      + (" — SINGLE-WINDOW pseudo-split (same Sim(3) anchoring "
                         "between frame segments)" if _pseudo else ""))

        # R.6 — per-window DA3 scale priors (raw DA3/omega ratios, conf-gated).
        # Best-effort: {} → the pipeline falls back to neutral 1.0 priors.
        priors: dict[str, float] = {}
        try:
            from phase_r.scale_check import per_window_scale_priors
            priors = per_window_scale_priors(output_dir, wmap, config)
        except Exception as e:  # noqa: BLE001
            pipe.send_log(f"per-window S priors skipped: {e}", level="warning")

        pr = PhaseRPipeline(session_path, output_dir,
                            output_dir / "scene_r.db", config=config,
                            window_map=wmap,
                            window_scale_priors=priors or None)
        pipe.send_progress(20, "Vote + onion + Sim(3) refinement (R.2–R.6)...",
                           stage="phase_r")
        report = pr.run()

        (output_dir / "phase_r_report.json").write_text(json.dumps({
            "summary": report.summary(),
            "n_instances": report.n_instances,
            "n_windows": report.n_windows,
            "iterations": report.iterations,
            "converged": report.converged,
            "onion_bimodal": report.onion_bimodal,
            "onion_separation_median_m": report.onion_separation_median_m,
            "seams": report.seams,
            "scale_report": report.scale_report,
            "scale_conflicts": report.scale_conflicts,
            "failsafe": report.failsafe,
            "used_anchoring": report.used_anchoring,
            "writeback": report.writeback,
        }, indent=2, default=str))
        pipe.send_log(f"Phase R: {report.summary()}")

        # ── Post-anchoring S re-check (R.6) + gated APPLY ──
        # Re-estimates the DA3↔omega metric scale over eroded STRUCTURAL masks
        # with RAW depths. When the delta passes the R.6 gates
        # (scale_recheck_apply + min frames + min/max delta), the refined S is
        # APPLIED as a global similarity (chunks + aligned depth + poses +
        # marker + store); otherwise the reason is logged and reported.
        if (config.get("phase_r", {}) or {}).get("scale_recheck", True):
            try:
                from phase_r.scale_check import apply_refined_scale, refined_scale_report
                sr = refined_scale_report(output_dir, session_path, config)
                if sr:
                    if sr.get("s_refined"):
                        pipe.send_log(
                            f"S re-check (structural masks, raw depths): marker "
                            f"s={sr['s_marker']:.4f} → refined s={sr['s_refined']:.4f} "
                            f"(Δ {sr['delta_pct']:+.2f}%, {sr['n_frames']} frames)")
                        apply_res = apply_refined_scale(
                            output_dir, output_dir / "scene_r.db", sr, config)
                        sr["apply"] = apply_res
                        if apply_res.get("applied"):
                            pipe.send_log(f"S re-check APPLIED: ×{apply_res['delta']:.5f} "
                                          f"(s {apply_res['s_from']:.4f} → "
                                          f"{apply_res['s_to']:.4f})")
                        else:
                            pipe.send_log(f"S re-check NOT applied: "
                                          f"{apply_res.get('reason', 'gated')}")
                    else:
                        pipe.send_log(f"S re-check: {sr.get('verdict', 'no result')}")
                    rp = output_dir / "phase_r_report.json"
                    d = json.loads(rp.read_text())
                    d["scale_recheck"] = sr
                    rp.write_text(json.dumps(d, indent=2, default=str))
                else:
                    pipe.send_log("S re-check skipped: no basis (needs vggtomega run, "
                                  "da3_run kept, segmentation, structural instances)")
            except Exception as e:  # noqa: BLE001 — diagnostic, never fatal
                pipe.send_log(f"S re-check failed (non-fatal): {e}", level="warning")

        pipe.send_progress(100, f"Phase R done — anchoring "
                                f"{'kept' if report.used_anchoring else 'fallback'}",
                           stage="phase_r")
    except Exception as e:  # noqa: BLE001 — enhancement layer, never fatal
        pipe.send_log(f"Phase R failed non-fatally: {e}; reconstruction keeps "
                      f"the no-anchor baseline", level="warning")
        pipe.send_progress(100, "Phase R skipped (error — baseline kept)",
                           stage="phase_r")


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_phase_r_work, conn, session_dir, config)
