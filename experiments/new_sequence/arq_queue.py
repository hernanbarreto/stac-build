"""The night's queue of the architecture test (USER 2026-11-21: "creas una nueva sesion bufferstop_prueba y me dejas
esta nube ahi publicada, al terminar, haces lo mismo con pccr3104 y con pccr2408, en todos los casos cuando la nube se
publico, borra a mano todo el tendal de archivos que hayan quedado antes de empezar la siguiente reconstruccion").

One session after another, with no one attending: wait for the session's run (or start it), and when its cloud is
published remove what the run left behind (arq_clean.py) before the next one starts. A run or a cleaning that fails
STOPS the queue — nothing is retried blindly, the next session does not start on top of a failure.

    python arq_queue.py <logs dir> <name>=<scan dir>[:running] …

``:running`` — that session's run was launched already; its log (<logs dir>/<name>.log) is waited for.
Launched with the backend's interpreter and environment (v2_launch.py); each run is this process's child."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
LOCAL_TEST_DATA = Path("/root/dev_data")                    # the assistant's own test data on the pod's local disk
VOLUME_TEST_DATA = Path("/workspace/tmp/dev_data_from_local_disk")
CLEANER_BYTES_PER_POINT = 40                                # config.yaml postprocessing.clean_scratch_dir's note


def say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC  {msg}", flush=True)


def exit_code_of(log: Path):
    try:
        tail = log.read_bytes()[-400:].decode("utf-8", "replace").strip().splitlines()
    except OSError:
        return None
    if tail and tail[-1].startswith("exit "):
        return int(tail[-1].split()[1])
    return None


def run(name: str, scan: Path, log: Path) -> int:
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, "-u", str(HERE / "arq_front.py"), str(scan)], cwd=SERVER, stdout=f,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        code = p.wait()
    with open(log, "a") as f:
        f.write(f"exit {code}\n")
    return code


def local_disk_for_the_cleaner() -> None:
    """The cleaner's scratch is the pod's local disk (40 B a voted point): the assistant's own test data there moves
    to the volume before a long session starts."""
    if LOCAL_TEST_DATA.is_dir() and not VOLUME_TEST_DATA.exists():
        VOLUME_TEST_DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(LOCAL_TEST_DATA), str(VOLUME_TEST_DATA))
        say(f"{LOCAL_TEST_DATA} moved to {VOLUME_TEST_DATA}")
    free = shutil.disk_usage("/root").free
    say(f"the local disk has {free / 1e9:.1f} GB free: the cleaner's scratch for up to "
        f"{free / CLEANER_BYTES_PER_POINT / 1e6:.0f} M voted points")


def main() -> int:
    logs = Path(sys.argv[1])
    logs.mkdir(parents=True, exist_ok=True)
    items = []
    for a in sys.argv[2:]:
        name, rest = a.split("=", 1)
        running = rest.endswith(":running")
        items.append((name, Path(rest[:-len(":running")] if running else rest), running))
    for i, (name, scan, running) in enumerate(items):
        log = logs / f"{name}.log"
        if running:
            say(f"{name}: waiting for the run already launched ({log})")
            while exit_code_of(log) is None:
                time.sleep(20)
            code = exit_code_of(log)
        else:
            if len(list((scan / "frames").glob("*.jpg"))) > 5000:
                local_disk_for_the_cleaner()
            say(f"{name}: the run starts ({log})")
            code = run(name, scan, log)
        if code != 0:
            say(f"QUEUE STOPPED: {name}'s run ended with exit {code} — the next session does not start")
            return 1
        say(f"{name}: the cloud is published; removing what the run left behind")
        r = subprocess.run([sys.executable, str(HERE / "arq_clean.py"), str(scan), "--go"], cwd=SERVER, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in r.stdout.splitlines():
            if line.strip() and not line.startswith("[Config]"):
                say(f"{name}: {line}")
        if r.returncode != 0:
            say(f"QUEUE STOPPED: {name}'s cleaning failed (exit {r.returncode}) — the next session does not start")
            return 2
    say("the queue is done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
