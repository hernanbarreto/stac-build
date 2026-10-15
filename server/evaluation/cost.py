"""Phase 4 — the cost of each stage: wall time, GPU time, peak GPU memory (the card's and the
process tree's through nvidia-smi sampling; torch peaks the chain itself reports), peak RAM, and
money = GPU hours × the configured price per hour of the card (``evaluation.cost``). Totals per
variant and per minute of input video. The monitor samples from a thread and never touches the
chain's inputs or outputs — a test proves the results are the same with and without it.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evaluation.config import CostConfig

Sample = Tuple[float, float, float, float]      # gpu util %, card memory used MiB, our tree's GPU MiB, our RSS MiB


def nvidia_sample(pids: Optional[set] = None) -> Tuple[float, float, float]:
    """(utilization %, memory used MiB of the first card, GPU MiB of ``pids``) via nvidia-smi."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        util, mem = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")[:2]]
    except Exception:  # noqa: BLE001 — no card / no nvidia-smi: recorded as nan, never a crash
        return float("nan"), float("nan"), float("nan")
    proc = 0.0
    if pids:
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            for ln in r.stdout.strip().splitlines():
                f = [x.strip() for x in ln.split(",")]
                if len(f) >= 2 and f[0].isdigit() and int(f[0]) in pids:
                    try:
                        proc += float(f[1])
                    except ValueError:
                        pass
        except Exception:  # noqa: BLE001
            proc = float("nan")
    return util, mem, proc


def process_tree(root_pid: int) -> Tuple[set, float]:
    """(pids of the tree, RSS MiB summed) with psutil."""
    try:
        import psutil
        p = psutil.Process(root_pid)
        procs = [p] + p.children(recursive=True)
        rss = 0.0
        pids = set()
        for q in procs:
            try:
                rss += q.memory_info().rss / (1024 * 1024)
                pids.add(q.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return pids, rss
    except Exception:  # noqa: BLE001
        return {root_pid}, float("nan")


def default_sampler(root_pid: int) -> Sample:
    pids, rss = process_tree(root_pid)
    util, mem, proc = nvidia_sample(pids)
    return util, mem, proc, rss


def video_minutes(session_dir: Path) -> Optional[float]:
    """Duration of the scene's source video in minutes (ffprobe, else OpenCV), None when unknown."""
    vids = sorted(Path(session_dir).glob("source_video.*"))
    if not vids:
        return None
    v = vids[0]
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                           capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip()) / 60.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(v))
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return (n / fps) / 60.0 if fps else None
    except Exception:  # noqa: BLE001
        return None


def price_of(card_key: Optional[str], prices: Dict[str, float]) -> Tuple[float, str]:
    if card_key and card_key in prices:
        return float(prices[card_key]), card_key
    return float(prices["default"]), "default"


class StageMonitor:
    """Samples the card and the process tree every ``cfg.sample_interval_s`` while a variant runs;
    stage boundaries come from :meth:`stage`. ``sampler`` / ``clock`` are injectable (tests)."""

    def __init__(self, out_dir: Path, name: str, cfg: CostConfig, *, card_key: Optional[str], prices: Dict[str, float],
                 video_min: Optional[float] = None, sampler: Optional[Callable[[int], Sample]] = None,
                 clock: Callable[[], float] = time.monotonic, root_pid: Optional[int] = None,
                 interval_s: Optional[float] = None) -> None:
        self.out_dir = Path(out_dir)
        self.name = name
        self.cfg = cfg
        self.card_key = card_key
        self.price, self.price_source = price_of(card_key, prices)
        self.video_min = video_min
        self.sampler = sampler or default_sampler
        self.clock = clock
        self.root_pid = root_pid or os.getpid()
        self.interval = float(interval_s if interval_s is not None else cfg.sample_interval_s)
        self.samples: List[Tuple[float, str, Sample]] = []
        self.marks: List[Tuple[float, str, str]] = []
        self._stage = "setup"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.ru0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    # ── control ──
    def start(self) -> "StageMonitor":
        self.t_start = self.clock()
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name=f"cost-monitor-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stage(self, name: str, event: str = "start") -> None:
        with self._lock:
            self.marks.append((self.clock(), name, event))
            if event == "start":
                self._stage = name
        self._sample_once()

    def _sample_once(self) -> None:
        try:
            s = self.sampler(self.root_pid)
        except Exception:  # noqa: BLE001
            s = (float("nan"),) * 4
        with self._lock:
            self.samples.append((self.clock(), self._stage, s))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_once()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 3 * self.interval))
        self._sample_once()
        self.t_end = self.clock()
        rep = self.report()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "cost.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=float))
        return rep

    # ── the report ──
    def report(self) -> Dict[str, Any]:
        import math
        with self._lock:
            samples = list(self.samples)
        stages: Dict[str, Dict[str, Any]] = {}
        for i, (t, stage, s) in enumerate(samples):
            st = stages.setdefault(stage, {"wall_s": 0.0, "gpu_time_s": 0.0, "gpu_mem_peak_mib": float("nan"),
                                           "gpu_mem_proc_peak_mib": float("nan"), "rss_peak_mib": float("nan"),
                                           "n_samples": 0, "t_first": t, "t_last": t})
            st["n_samples"] += 1
            st["t_last"] = t
            util, mem, proc, rss = s
            for key, v in (("gpu_mem_peak_mib", mem), ("gpu_mem_proc_peak_mib", proc), ("rss_peak_mib", rss)):
                if v == v and (st[key] != st[key] or v > st[key]):
                    st[key] = v
            if i + 1 < len(samples):
                dt = samples[i + 1][0] - t
                if dt > 0:
                    st["wall_s"] += dt
                    if util == util:
                        st["gpu_time_s"] += dt * util / 100.0
        total_wall = (self.t_end or self.clock()) - (self.t_start or 0.0)
        gpu_time = sum(s["gpu_time_s"] for s in stages.values())
        gpu_hours = gpu_time / 3600.0
        wall_hours = total_wall / 3600.0
        peak = {k: max((s[k] for s in stages.values() if s[k] == s[k]), default=float("nan"))
                for k in ("gpu_mem_peak_mib", "gpu_mem_proc_peak_mib", "rss_peak_mib")}
        ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rep = {
            "variant": self.name, "card_key": self.card_key, "price_per_gpu_hour_usd": self.price,
            "price_source": self.price_source, "sample_interval_s": self.interval, "n_samples": len(samples),
            "stages": {k: {kk: (None if isinstance(vv, float) and math.isnan(vv) else vv) for kk, vv in v.items()}
                       for k, v in stages.items()},
            "totals": {"wall_s": total_wall, "gpu_time_s": gpu_time, "gpu_hours": gpu_hours,
                       "wall_hours": wall_hours, "cost_usd": gpu_hours * self.price,
                       "cost_wall_usd": wall_hours * self.price,
                       "gpu_mem_peak_mib": None if peak["gpu_mem_peak_mib"] != peak["gpu_mem_peak_mib"] else peak["gpu_mem_peak_mib"],
                       "gpu_mem_proc_peak_mib": None if peak["gpu_mem_proc_peak_mib"] != peak["gpu_mem_proc_peak_mib"] else peak["gpu_mem_proc_peak_mib"],
                       "rss_peak_mib": None if peak["rss_peak_mib"] != peak["rss_peak_mib"] else peak["rss_peak_mib"],
                       "children_maxrss_mib": (ru / 1024.0) if ru else None},
            "gpu_time_definition": "integral of nvidia-smi utilization.gpu over the wall time (seconds of a busy card)",
            "cost_definition": "gpu_hours x price_per_gpu_hour_usd; cost_wall_usd = wall_hours x price (a rented card is paid while idle)",
            "video_minutes": self.video_min,
            "marks": [{"t": t - (self.t_start or 0.0), "stage": n, "event": e} for t, n, e in self.marks],
        }
        if self.video_min:
            rep["per_video_minute"] = {"wall_s": total_wall / self.video_min, "gpu_time_s": gpu_time / self.video_min,
                                       "cost_usd": rep["totals"]["cost_usd"] / self.video_min,
                                       "cost_wall_usd": rep["totals"]["cost_wall_usd"] / self.video_min}
        rep["gpu_hours"] = gpu_hours
        rep["cost_usd"] = rep["totals"]["cost_usd"]
        return rep


def torch_peaks_reported(session_dir: Path) -> Dict[str, Any]:
    """The torch peak memories the chain's own timing files report (declared, in GB)."""
    out: Dict[str, Any] = {}
    o = Path(session_dir) / "output"
    for rel, keys in (("precision/depth_on_f5.timing.json", ("mono_detail", "vram_peak_gb")),
                      ("omega_probe.timing.json", ("resolutions",))):
        p = o / rel
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        node: Any = doc
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if node is not None:
            out[rel] = node
    return out


def attach_torch_peaks(rec: Dict[str, Any]) -> None:
    sd = rec.get("session_dir")
    if sd and isinstance(rec.get("cost"), dict):
        rec["cost"]["torch_peaks_reported"] = torch_peaks_reported(Path(sd))
