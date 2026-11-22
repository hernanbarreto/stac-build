"""arq_vote_census.py on bufferStop_prueba and pccr3108_prueba, under the card's lease (USER 2026-11-22: "hace sobre
bufferstop y sobre 3108").

    python arq_vote_census_run.py"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
OUT = Path("/workspace/tmp/new_sequence/arq/vote_census")
SESSIONS = (("bufferStop", "bufferStop_prueba/scans/2026-08-11/src_default"),
            ("pccr3108", "pccr3108_prueba/scans/2026-08-31/src_default"))


def main() -> int:
    from card import card
    with card("F6's vote counted pixel by pixel (no publication)", session="bufferStop_prueba"):
        for name, rel in SESSIONS:
            print(f"\n===== {name}", flush=True)
            r = subprocess.run([sys.executable, str(HERE / "arq_vote_census.py"), str(P / rel), str(OUT / f"{name}.json")],
                               cwd=SERVER, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in r.stdout.splitlines():
                if line.startswith("[census]") or "Error" in line or "Traceback" in line or line.startswith("  File"):
                    print(line, flush=True)
            if r.returncode != 0:
                print(r.stdout[-3000:], flush=True)
                print(f"{name}: the census failed (exit {r.returncode})", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
