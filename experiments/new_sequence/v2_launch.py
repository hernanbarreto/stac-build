"""Launch a version-2 driver as the backend launches its workers: the backend's own interpreter and environment, the
server directory as the working directory — detached from this shell, its output in a log that ends with
"exit <code>".

    python v2_launch.py <log file> <driver.py> [args …]"""
import os
import subprocess
import sys
from pathlib import Path

SERVER = "/workspace/stac-build/server"


def backend_pid() -> int:
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                cmd = (d / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if b"uvicorn" in cmd and b"main:app" in cmd:
                return int(d.name)
    raise SystemExit("the backend is not running — its environment cannot be taken")


def main() -> int:
    log, driver, args = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    pid = backend_pid()
    env = {}
    for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
        if b"=" in item:
            k, v = item.split(b"=", 1)
            env[k.decode()] = v.decode("utf-8", "replace")
    py = os.readlink(f"/proc/{pid}/exe")
    log.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in [py, "-u", driver, *args])
    cmd = f"{quoted} > '{log}' 2>&1; echo \"exit $?\" >> '{log}'"
    p = subprocess.Popen(["/bin/bash", "-c", cmd], cwd=SERVER, env=env, start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"launched pid {p.pid} with the backend's interpreter ({py}) and environment ({len(env)} variables) → {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
