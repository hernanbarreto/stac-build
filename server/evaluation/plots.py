"""PNG charts of the battery (matplotlib, Agg): labelled axes and n per bin, nothing else."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _bins(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("bin") != "outside"]


def plot_curve(path: Path, rows: List[Dict[str, Any]], xlabel: str, ylabel: str, title: str,
               tolerances: Sequence[float] = ()) -> Path:
    rows = _bins(rows)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    x = list(range(len(rows)))
    med = [r["abs_median_m"] if r["status"] == "ok" else float("nan") for r in rows]
    p95 = [r["abs_p95_m"] if r["status"] == "ok" else float("nan") for r in rows]
    ax.plot(x, med, "o-", label="median")
    ax.plot(x, p95, "s--", label="P95")
    for t in tolerances:
        ax.axhline(t, color="grey", lw=0.8, ls=":", label=f"tolerance {t:g} m")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['bin']}\nn={r['n']}" + ("" if r["status"] == "ok" else "\n(no data)") for r in rows])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, lw=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_noise(path: Path, rows: List[Dict[str, Any]], xlabel: str, ylabel: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    planes = sorted({r["plane"] for r in rows}, key=lambda p: (p != "all", p))
    labels = None
    for p in planes:
        pr = [r for r in rows if r["plane"] == p]
        labels = labels or [r["bin"] for r in pr]
        y = [r["rms_m"] if r["status"] == "ok" else float("nan") for r in pr]
        ax.plot(range(len(pr)), y, "o-", label=f"{p} ({pr[0].get('label', '')})" if p != "all" else "all planes")
    if labels:
        ax.set_xticks(range(len(labels)))
        ns = {}
        for r in rows:
            ns.setdefault(r["bin"], 0)
            ns[r["bin"]] = max(ns[r["bin"]], r["n"])
        ax.set_xticklabels([f"{b}\nn={ns.get(b, 0)}" for b in labels])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, lw=0.3)
    if planes:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_scatter(path: Path, x: Sequence[float], y: Sequence[float], labels: Sequence[str], xlabel: str,
                 ylabel: str, title: str, highlight: Sequence[bool] = ()) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
    for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
        on = bool(highlight[i]) if i < len(highlight) else False
        ax.scatter([xi], [yi], s=46 if on else 28, marker="D" if on else "o", color="tab:red" if on else "tab:blue")
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, lw=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path
