"""Watch a run launched by v2_launch.py until it ends or stalls.

    python arq_watch.py <log file> <launcher pid> [stall minutes, default 20]

Ends with the run: prints the log's last lines and exits with the run's own code (the log's final "exit <code>").
A stall — for the given minutes the log did not grow, the card stayed under 5 % and the run's processes used less than
a fifth of one core — prints STALL with what was read and exits 97."""
import os
import subprocess
import sys
import time
from pathlib import Path

HZ = os.sysconf("SC_CLK_TCK")


def tree(root: int):
    kids = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                kids.setdefault(int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]), []).append(int(d.name))
            except (OSError, IndexError, ValueError):
                continue
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out


def cpu_seconds(pids) -> float:
    tot = 0
    for p in pids:
        try:
            f = (Path("/proc") / str(p) / "stat").read_text().rsplit(")", 1)[1].split()
            tot += int(f[11]) + int(f[12])
        except (OSError, IndexError, ValueError):
            continue
    return tot / HZ


def gpu_util() -> float:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 — a failed reading counts as busy: never a false stall
        return 100.0


def last_lines(log: Path, n: int = 12) -> str:
    try:
        data = log.read_bytes()[-6000:].decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-n:])


def main() -> int:
    log, root = Path(sys.argv[1]), int(sys.argv[2])
    stall_s = 60.0 * (float(sys.argv[3]) if len(sys.argv) > 3 else 20.0)
    quiet_since, size0, cpu0, gpu_max = time.time(), -1, None, 0.0
    while True:
        text = last_lines(log, 3)
        if text and text.splitlines()[-1].startswith("exit "):
            code = int(text.splitlines()[-1].split()[1])
            print(last_lines(log, 14))
            return 0 if code == 0 else (code if 0 < code < 90 else 1)
        size = log.stat().st_size if log.exists() else 0
        pids = tree(root)
        cpu = cpu_seconds(pids)
        g = gpu_util()
        if size != size0 or cpu0 is None:
            quiet_since, size0, cpu0, gpu_max = time.time(), size, cpu, g
        else:
            gpu_max = max(gpu_max, g)
            quiet = time.time() - quiet_since
            if quiet >= stall_s:
                if gpu_max < 5.0 and (cpu - cpu0) < 0.2 * quiet:
                    print(f"STALL: {quiet / 60:.0f} min without a log line, the card at most {gpu_max:.0f} %, the run's "
                          f"{len(pids)} process(es) used {cpu - cpu0:.0f} s of CPU")
                    print(last_lines(log, 10))
                    return 97
                quiet_since, cpu0, gpu_max = time.time(), cpu, g      # busy without logging: a new window
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
