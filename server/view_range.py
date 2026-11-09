"""THE RANGE of every view (USER 2026-11-09): "ningún KF debe aportar puntos más allá de los 10 m respecto de su
posición" / "esos puntos tampoco deben ser considerados en etapas anteriores, de cierres ni de paridades, ni de nada" /
"los 10 m es una variable, la misma para todas estas modificaciones y para F6 para todo … si mañana descubrimos que el
error es aceptable a 12 m … lo cambiamos en el config y se deben ajustar todos".

ONE key, ``reconstruction.view_range_m``: a point farther than it from the camera that measured it is born by no keyframe
(F6) and weighs in no fit, pairing, scale, closure, co-visibility or statistic of any stage before it. Every stage reads
it through :func:`view_range_m` — never a number of its own."""
from __future__ import annotations

from typing import Any, Mapping, Optional

KEY = "reconstruction.view_range_m"


def view_range_m(raw: Optional[Mapping[str, Any]] = None) -> float:
    """The range in metres from ``raw`` (a whole configuration dict: the job's frozen one on the pipeline path), else
    from server/config.yaml read now. A missing or non-positive value FAILS naming the key."""
    if raw is None:
        from reconstruction.grid_knn import config_section
        return _check(config_section(("reconstruction",), ("view_range_m",))["view_range_m"])
    rec = raw.get("reconstruction") if isinstance(raw, Mapping) else None
    if not isinstance(rec, Mapping) or "view_range_m" not in rec:
        raise KeyError(f"config.yaml is missing mandatory key '{KEY}' — the one range of every view (USER 2026-11-09)")
    return _check(rec["view_range_m"])


def _check(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v) > 0.0:
        raise ValueError(f"'{KEY}' must be a positive number of metres, got {v!r}")
    return float(v)
