"""Step 5 (arq3_select.py) on bufferStop and pccr 2026-08-31, under the card's lease — the keyframes DVLT is given
(USER 2026-11-22: "le pasemos los kf que salen de nuestro nuevo pipeline que estamos construyendo ahora"). Read only on
the sessions; written in /workspace/tmp/new_sequence/arq/select/.

    python arq3_select_run.py"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
OUT = Path("/workspace/tmp/new_sequence/arq/select")
SESSIONS = (("bufferStop", "bufferStop/scans/2026-08-11/src_default"),
            ("pccr_2026-08-31", "pccr/scans/2026-08-31/src_default"))


def main() -> int:
    from card import card
    with card("step 5 of the architecture: the keyframe selection", session="pccr"):
        for name, rel in SESSIONS:
            r = subprocess.run([sys.executable, str(HERE / "arq3_select.py"), name, str(P / rel), str(OUT / f"{name}.json")],
                               cwd=SERVER, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in r.stdout.splitlines():
                if line.startswith("[select]") or "Error" in line or "Traceback" in line or line.startswith("  File"):
                    print(line, flush=True)
            if r.returncode != 0:
                print(r.stdout[-3000:], flush=True)
                print(f"{name}: the selection failed (exit {r.returncode})", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
