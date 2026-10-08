"""The COMMITTED per-card table (docs/plan_determinismo.md points 3, 5, 6, 24, 25, 41).

Two decisions used to be re-derived on every run from what the card did at that moment:

- the I3 DA3 window size, floor() of a 2-frame calibration measured per session and cached in
  output/intake/da3_vram.json (pccr sat 0.24 % above the 26/25 boundary; a 20 MiB higher peak
  after a torch update would have changed the windows, the walk, the chunk plan and the anchors);
- Omega's per-frame footprint factor, LEARNED from the OOMs of any session and written to
  weights/omega_footprint.json (a crash of zaragoza moved every later session's resolution).

Both now come from ``server/card_table.json``, keyed by the card MODEL
(:func:`repro.card_key` of :func:`repro.card_identity`: ``'<name> | <board MiB> MiB |
sm_<capability>'`` — the name and capability read through torch, the board memory nvidia-smi's
memory.total of that same card; never an 'unknown' sentinel, never torch's usable-memory bytes,
which moved by 3 MiB across a pod restart on the same model). NO RUN WRITES IT: an entry is
measured by a calibration CLI run by hand on that card and committed with its provenance. A card
or resolution with no entry FAILS, naming the CLI.

The card's total memory used for sizing is the table's ``nvidia_smi_memory_total_mib`` (what
nvidia-smi reports as memory.total — the value every validated run was sized on, 80.0 GiB on
the A100), never re-read at run time; it must equal the MiB its key names (checked when the table
is loaded and when a CLI writes it).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

TABLE_PATH = Path(__file__).resolve().parent / "card_table.json"
TABLE_VERSION = 1

CALIBRATE_DA3_CLI = ("python -m intake.vram --calibrate --session <scan> "
                     "[--process-res native|<px>] [--model <model_id>]")
CALIBRATE_OMEGA_CLI = ("python -m reconstruction.chunk_plan --omega-footprint "
                       "(--from-oom-log <log> --predicted-peak-gib <GiB> | --linear) "
                       "--provenance '<what was measured>'")


class CardTableError(RuntimeError):
    """The table has no entry for this card / model / resolution, or it is malformed — always
    naming what is missing and the CLI that measures it."""


def key_memory_mib(card_key: str) -> int:
    """The board memory (MiB) a card key names (repro.parse_card_key). A key not in
    repro.card_key's format — one written by older code, keyed on torch's bytes — is a
    CardTableError."""
    import repro
    try:
        return int(repro.parse_card_key(card_key)["memory_total_mib"])
    except repro.ReproError as e:
        raise CardTableError(str(e)) from e


def load_table(path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """The table, every card key in repro.card_key's format and every entry's
    ``nvidia_smi_memory_total_mib`` equal to the MiB its key names — else CardTableError."""
    p = Path(path) if path is not None else TABLE_PATH
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise CardTableError(f"the card table {p} is unreadable ({e})") from e
    if not isinstance(doc, dict) or doc.get("version") != TABLE_VERSION \
            or not isinstance(doc.get("cards"), dict):
        raise CardTableError(f"{p} is not a version-{TABLE_VERSION} card table")
    for key, ent in doc["cards"].items():
        try:
            mib = key_memory_mib(key)
        except CardTableError as e:
            raise CardTableError(f"{p}: {e}") from e
        rec = ent.get("nvidia_smi_memory_total_mib") if isinstance(ent, dict) else None
        if isinstance(rec, bool) or not isinstance(rec, int) or rec != mib:
            raise CardTableError(f"{p}: the entry '{key}' records nvidia_smi_memory_total_mib "
                                 f"{rec!r}, its key names {mib} MiB — one card model, one size")
    return doc


def card_entry(card_key: str, path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """The table entry of ``card_key`` (repro.card_key). No entry → CardTableError naming both
    calibration CLIs."""
    doc = load_table(path)
    ent = doc["cards"].get(str(card_key))
    if not isinstance(ent, dict):
        known = ", ".join(sorted(doc["cards"])) or "none"
        raise CardTableError(
            f"the card '{card_key}' has no entry in {Path(path) if path else TABLE_PATH} (known: "
            f"{known}) — its DA3 window footprint and Omega footprint are measured on it with "
            f"`{CALIBRATE_DA3_CLI}` and `{CALIBRATE_OMEGA_CLI}`, then committed")
    return ent


def sizing_total_gib(entry: Mapping[str, Any]) -> float:
    """The card's total memory every size is computed on (GiB): nvidia-smi's memory.total."""
    try:
        mib = int(entry["nvidia_smi_memory_total_mib"])
    except (KeyError, TypeError, ValueError) as e:
        raise CardTableError("a card entry has no nvidia_smi_memory_total_mib") from e
    if mib <= 0:
        raise CardTableError(f"nvidia_smi_memory_total_mib {mib} is not a memory size")
    return mib / 1024.0


def omega_footprint_factor(entry: Mapping[str, Any], card_key: str = "") -> float:
    """Omega's measured per-frame footprint factor on this card (≥ 1 multiplies the linear
    0.086 GB/frame model of reconstruction.chunk_plan). Missing → CardTableError naming the CLI."""
    om = entry.get("omega") if isinstance(entry, Mapping) else None
    if not isinstance(om, Mapping) or "footprint_factor" not in om:
        raise CardTableError(f"the card '{card_key}' has no Omega footprint in the card table — "
                             f"measure it with `{CALIBRATE_OMEGA_CLI}`")
    f = float(om["footprint_factor"])
    if not f >= 1.0:
        raise CardTableError(f"the card '{card_key}' Omega footprint factor {f} is below the "
                             f"linear model (1.0) — the table is malformed")
    return f


def da3_footprint(entry: Mapping[str, Any], model_id: str, process_res: int,
                  card_key: str = "") -> Dict[str, Any]:
    """{weights_gib, per_token_gib, peak_gib, frames, tokens_per_frame, provenance} of DA3
    ``model_id`` at ``process_res`` on this card. per_token = (peak − weights) / tokens of the
    calibration window — the formula the run-time calibration used, now over committed numbers.
    Missing → CardTableError naming the calibration CLI."""
    by_model = (entry.get("da3") or {}).get(str(model_id)) if isinstance(entry, Mapping) else None
    rec = by_model.get(str(int(process_res))) if isinstance(by_model, Mapping) else None
    if not isinstance(rec, Mapping):
        have = sorted((by_model or {}).keys()) if isinstance(by_model, Mapping) else []
        raise CardTableError(
            f"the card '{card_key}' has no DA3 footprint for {model_id} at process_res "
            f"{int(process_res)} in the card table (measured: {have or 'none'}) — measure it on "
            f"this card with `{CALIBRATE_DA3_CLI}` and commit the entry")
    try:
        w, p = float(rec["weights_gib"]), float(rec["peak_gib"])
        n, tpf = int(rec["frames"]), int(rec["tokens_per_frame"])
    except (KeyError, TypeError, ValueError) as e:
        raise CardTableError(f"the DA3 entry {model_id}@{process_res} of '{card_key}' is "
                             f"malformed ({e})") from e
    if n < 1 or tpf < 1 or p < w:
        raise CardTableError(f"the DA3 entry {model_id}@{process_res} of '{card_key}' is "
                             f"inconsistent (frames {n}, tokens/frame {tpf}, peak {p} < weights {w}?)")
    return {"weights_gib": w, "peak_gib": p, "frames": n, "tokens_per_frame": tpf,
            "per_token_gib": (p - w) / float(n * tpf), "provenance": str(rec.get("provenance", ""))}


def _write_table(doc: Dict[str, Any], path: Optional[os.PathLike]) -> Path:
    p = Path(path) if path is not None else TABLE_PATH
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return p


def _entry_for_write(doc: Dict[str, Any], card_key: str, nvidia_smi_total_mib: int) -> Dict[str, Any]:
    """The entry a calibration CLI writes into: ``card_key`` in repro.card_key's format and naming
    the MiB nvidia-smi reads now (the CLIs pass card_identity's own key and memory_total_mib)."""
    key_mib = key_memory_mib(str(card_key))
    if key_mib != int(nvidia_smi_total_mib):
        raise CardTableError(f"'{card_key}' names {key_mib} MiB, nvidia-smi reads "
                             f"{nvidia_smi_total_mib} MiB now — one card model, one size")
    ent = doc["cards"].setdefault(str(card_key), {})
    old = ent.get("nvidia_smi_memory_total_mib")
    if old is not None and int(old) != int(nvidia_smi_total_mib):
        raise CardTableError(f"'{card_key}' is recorded with {old} MiB, nvidia-smi reads "
                             f"{nvidia_smi_total_mib} MiB now — one card model, one size")
    ent["nvidia_smi_memory_total_mib"] = int(nvidia_smi_total_mib)
    return ent


def write_da3_entry(card_key: str, nvidia_smi_total_mib: int, model_id: str, process_res: int,
                    measurement: Mapping[str, Any], path: Optional[os.PathLike] = None) -> Path:
    """CALIBRATION CLI ONLY (intake.vram --calibrate): record a measured DA3 footprint."""
    doc = load_table(path)
    ent = _entry_for_write(doc, card_key, nvidia_smi_total_mib)
    rec = {k: measurement[k] for k in ("weights_gib", "peak_gib", "frames", "tokens_per_frame",
                                       "provenance")}
    ent.setdefault("da3", {}).setdefault(str(model_id), {})[str(int(process_res))] = rec
    return _write_table(doc, path)


def write_omega_entry(card_key: str, nvidia_smi_total_mib: int, factor: float, provenance: str,
                      path: Optional[os.PathLike] = None) -> Path:
    """CALIBRATION CLI ONLY (reconstruction.chunk_plan --omega-footprint): record Omega's
    footprint factor for a card."""
    if not float(factor) >= 1.0:
        raise CardTableError(f"an Omega footprint factor below 1.0 ({factor}) under-reads the "
                             f"measured linear model")
    if not str(provenance).strip():
        raise CardTableError("an Omega footprint entry needs its provenance (what was measured)")
    doc = load_table(path)
    ent = _entry_for_write(doc, card_key, nvidia_smi_total_mib)
    ent["omega"] = {"footprint_factor": round(float(factor), 4), "provenance": str(provenance)}
    return _write_table(doc, path)


def probe_window_frames(model_id: str, process_res: int,
                        path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """The FOCAL PROBE's window layout per (model, process_res) — docs/plan_determinismo.md
    point 64: the probe's frames go into windows of this many frames on EVERY card (the layout
    used to follow the card's memory, so the session K — and with it every keyframe — depended on
    the card and on co-tenants). {window_frames, provenance}; no entry → CardTableError naming
    what to declare (the layout the validated run of that resolution used)."""
    doc = load_table(path)
    by_model = (doc.get("da3_probe_layout") or {}).get(str(model_id))
    rec = by_model.get(str(int(process_res))) if isinstance(by_model, Mapping) else None
    if not isinstance(rec, Mapping) or "window_frames" not in rec:
        have = sorted(k for k in (by_model or {}) if k != "about") if isinstance(by_model, Mapping) else []
        raise CardTableError(
            f"no committed focal-probe window layout for {model_id} at process_res "
            f"{int(process_res)} (declared: {have or 'none'}) — add da3_probe_layout['{model_id}']"
            f"['{int(process_res)}'] = {{window_frames, provenance}} to {Path(path) if path else TABLE_PATH}"
            f" (the layout of the validated run at that resolution) and commit it")
    n = int(rec["window_frames"])
    if n < 2:
        raise CardTableError(f"the probe layout {model_id}@{process_res} ({n} frames) measures no "
                             f"multi-view K — the table is malformed")
    return {"window_frames": n, "provenance": str(rec.get("provenance", ""))}


def visible_card_count(env: Optional[Mapping[str, str]] = None) -> int:
    """How many cards the process may run on (plan point 78): CUDA_VISIBLE_DEVICES's entries
    when set (an empty value: none), else every card nvidia-smi lists."""
    import repro
    e = os.environ if env is None else env
    cvd = e.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([t for t in cvd.split(",") if t.strip()])
    return len(repro.gpu_cards())


def require_one_visible_card(log=print) -> None:
    """FAIL unless exactly one card is visible (point 78): the identity every size is keyed by
    is that of THE device the job runs on — never nvidia-smi's first GPU, never one of several."""
    n = visible_card_count()
    if n != 1:
        raise CardTableError(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} leaves "
                             f"{n} card(s) visible — a run uses exactly one card (its identity, memory "
                             f"and windows are that card's); set CUDA_VISIBLE_DEVICES to the one card")


def current_card() -> Dict[str, Any]:
    """{identity, key, entry, total_gib} of the card this process would run on: the identity
    (repro.card_identity: torch + the board memory nvidia-smi lists for that uuid — fails, never
    'unknown'), its MODEL key, its table entry (fails naming the CLIs) and the sizing total."""
    import repro
    require_one_visible_card()
    ident = repro.card_identity(0)
    key = repro.card_key(ident)
    ent = card_entry(key)
    return {"identity": ident, "key": key, "entry": ent, "total_gib": sizing_total_gib(ent)}
