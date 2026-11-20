"""The precision chain AGAIN on a scan whose cloud is published — keeping F4's tracks (USER 2026-11-20, shown that
F2's scale along the walk re-opened what the cube graph had closed: "La manera es una única escala y que no rompa
nada" / "relanzas pccr sobreescribiendo. Relanza como dijiste te aprovechas lo que se pueda y lo que no se rehace").

What is reused: the cubes (Omega's records), the keyframes, F4's tracks (its inputs are the frames and the camera
F0 writes — neither changes). What is made again: F0 (seconds: the camera F4 saw, F5 starts from it as in a clean
chain), F2, F5, F6 and the floor check. THE PUBLISHED CLOUD IS REPLACED.

The pipeline's own job would compute F4 again (the chain runs every step from the first whose stamp changed), so
this runs ``precision.runner.run_chain`` with the step list minus F4 — the chain's own machinery: no finished
record matches that list, the chain goes back to Omega's epoch 0 from ``output/_restart_epochs`` and runs.

OUTSIDE THE BACKEND, declared: the card's lease (semantic.service — a file the backend, its stages and a CLI share)
is held by this process for the whole run, so the chat's engine and the voice answer "busy" and a job launched
meanwhile is refused; the voice is ended through the backend's own route (``POST /api/voice/stop``: "for GPU work
launched OUTSIDE the backend") and the chat's engine through ``workers.base.stop_semantic_service_verified``.

The scan's FROZEN configuration is what the steps read: it is frozen again with the keys named in SYNC taken from
the server's config.yaml of today (each difference declared) — the rest stays what the scan was made with.

    python -m tools.rerun_precision <scan dir>            what it would do, and whether it can (nothing is touched)
    python -m tools.rerun_precision <scan dir> --go       do it
"""
from __future__ import annotations

import argparse
import copy
import json
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

#: the keys of the precision section a re-run takes from today's config.yaml (the rule and the switches of
#: 2026-11-20); everything else is the scan's own frozen configuration
SYNC = ("reconstruction.precision.gauge.profile",
        "reconstruction.precision.cloud.vote_batch",
        "reconstruction.precision.tracks.npz_threads")
BACKEND = "https://127.0.0.1:8765"
KEEP = "f4_tracks"


def say(msg: str) -> None:
    print(f"{time.strftime('%m-%dT%H:%M:%S', time.gmtime())} [rerun] {msg}", flush=True)


def _get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _set(d, dotted, value):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = value


def _http(path: str, method: str = "GET", timeout: float = 30.0):
    req = urllib.request.Request(BACKEND + path, method=method, data=(b"{}" if method == "POST" else None),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as r:
        return json.loads(r.read() or b"{}")


def _card() -> list:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.rerun_precision")
    ap.add_argument("scan")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    scan = Path(a.scan).resolve()
    out = scan / "output"
    session = scan.parents[2].name
    from config import cfg as today
    from intake.run_config import STAGES_KEY, freeze_run_config, load_run_config
    from precision.config import load_precision_config
    from precision.product import product_is_live
    from precision.runner import chain_steps, run_chain
    from semantic.service import (JOB_PERMIT_OWNER, acquire_engine_lease, engine_lease_holder,
                                  release_engine_lease)

    frozen, sha = load_run_config(scan)
    new = copy.deepcopy(frozen)
    stages = new.pop(STAGES_KEY, None)
    changed = []
    for key in SYNC:
        was, now = _get(frozen, key), _get(today, key)
        if now is None:
            raise SystemExit(f"config.yaml has no {key}")
        if was != now:
            _set(new, key, now)
            changed.append(f"{key}: {was!r} -> {now!r}")
    say(f"{session}: the frozen configuration {sha[:12]}; taken from today's config.yaml: "
        + ("; ".join(changed) if changed else "nothing (already the same)"))
    pcfg_new = load_precision_config(new, log=lambda *x: None)
    steps = [s for s in chain_steps(pcfg_new) if s.key != KEEP]
    say("steps: " + " -> ".join(s.key for s in steps) + f"   (kept as it is: {KEEP})")

    problems = []
    live, why = product_is_live(out)
    say(f"published cloud now: {why}")
    for rel in ("precision/tracks.npz", "_restart_epochs/_epoch_0/camera_poses.txt", "camera_frames.txt",
                "omega_run/results_output", "da3_windows/windows.json"):
        if not (out / rel).exists():
            problems.append(f"{rel} is not on disk")
    n_kf = len((out / "camera_frames.txt").read_text().split()) if (out / "camera_frames.txt").exists() else 0
    n_rec = len(list((out / "omega_run/results_output").glob("frame_*.npz"))) if (out / "omega_run/results_output").exists() else 0
    if n_rec != n_kf:
        problems.append(f"{n_rec} Omega records for {n_kf} keyframes")
    try:
        active = _http("/api/pipelines/active").get("pipelines") or {}
        if active:
            problems.append(f"a pipeline job is active: {sorted(active)}")
    except Exception as e:  # noqa: BLE001 — declared: the backend may be down, the lease still decides
        say(f"the backend did not answer ({e}) — the lease alone decides")
    holder = engine_lease_holder()
    if holder is not None:
        problems.append(f"the card's lease is held: {holder.get('owner')} / {holder.get('stage')} (pid {holder.get('pid')})")
    say(f"{n_kf} keyframes, {n_rec} records; on the card now: {_card() or 'nothing'}")
    if problems:
        for p in problems:
            say(f"CANNOT RUN: {p}")
        return 2
    if not a.go:
        say("dry run — nothing was touched (--go runs it)")
        return 0

    acquire_engine_lease(JOB_PERMIT_OWNER, stage="precision chain again (F0, F2, F5, F6)", session=session,
                         token="rerun-precision")
    say("the card's lease is this process's until it ends")
    code = 1
    try:
        try:
            r = _http("/api/voice/stop", "POST", timeout=60.0)
            say(f"the voice: stopped {r.get('stopped')}")
        except Exception as e:  # noqa: BLE001 — the chain's own check of the card names what is left on it
            say(f"the voice could not be stopped through the backend ({e})")
        from workers.base import stop_semantic_service_verified
        stop_semantic_service_verified(None, stage="the precision chain", log=say)
        for _ in range(30):
            if not _card():
                break
            time.sleep(2)
        say(f"on the card before the chain: {_card() or 'nothing'}")
        fr = freeze_run_config(scan, new, stages=stages, log=say)
        pcfg = load_precision_config(session_dir=scan, log=say)
        assert [s.key for s in chain_steps(pcfg) if s.key != KEEP] == [s.key for s in steps]
        t0 = time.time()
        rep = run_chain(scan, pcfg, steps=steps, log=say,
                        before_gpu=lambda label: stop_semantic_service_verified(None, stage=f"precision {label}", log=say))
        say(f"core done in {(time.time() - t0) / 60:.1f} min — the published cloud (cloud.source {pcfg.cloud.source}) "
            f"is epoch {rep['epoch']} (configuration {fr['sha256'][:12]})")
        live, why = product_is_live(out)
        say(f"product: {why}")
        if not live:
            raise RuntimeError(f"the chain ended and the product is not live: {why}")
        from workers.map_worker import _discard_previous_epochs
        freed = _discard_previous_epochs(out)
        say(f"the previous cloud and the epochs of the other branch discarded: {freed / 1e6:.0f} MB freed "
            f"(the way back to epoch 0 stays in _restart_epochs)")
        try:
            _http(f"/api/sessions/{session}/cloud_changed", "POST")
            say("the open viewers were told the cloud changed")
        except Exception as e:  # noqa: BLE001
            say(f"the viewers could not be told ({e}) — a page reload shows the new cloud")
        code = 0
    except BaseException as e:  # noqa: BLE001 — said, then raised: the log holds the reason
        say(f"FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        release_engine_lease(JOB_PERMIT_OWNER, "rerun-precision")
        say(f"the card's lease released; exit {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
