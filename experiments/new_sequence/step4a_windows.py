"""THE NEW ORDER, step 4 (first part): the DA3 window depth F2 measures against.

The pipeline's F2 brings its I3 window depth files back itself when they are gone (precision.gauge.regenerate_windows:
intake.walk's own extractor, the plan checked against walk.json). The clean test session has none yet, so the same
function is called here once, on the session; the five cubes then read the same files.

    python step4a_windows.py <session scan dir>"""
import sys
from pathlib import Path

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)


def main() -> int:
    scan = Path(sys.argv[1])
    from intake.walk import load_walk
    from precision import gauge as G
    from precision.config import load_precision_config
    pcfg = load_precision_config(session_dir=scan)
    log = lambda m: print(m, flush=True)   # noqa: E731
    if not G.needs_window_regeneration(scan):
        log("[step 4a] the I3 window depth files are on disk — nothing to regenerate")
        return 0
    G.regenerate_windows(scan, pcfg.gauge, load_walk(scan), log=log)
    n = len(list((scan / "output" / "da3_windows").glob("window_*.npz")))
    log(f"[step 4a] {n} I3 window depth file(s) in {scan / 'output' / 'da3_windows'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
