# STAC-Builder — Phase 5: the chat's answers, written by the tool layer.
#
# USER 2026-11-13, on the chat's time and its engine: "Determinista pero en 1 o 2 segundos
# ninguna es negociable". The model's part of a question is now ONE call — choosing the tool
# and its arguments. The sentence the user reads is composed HERE from the tool's result:
# deterministic, in the question's language, with the tool's own figures (nothing is rounded
# by a model, nothing is paraphrased), and it costs no second call of the engine.
#
# A tool with no entry here (or a result it cannot word) returns None: the orchestrator then
# falls back to the model for the wording, with the provenance check.

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

LANGS = ("es", "en")

_ES_WORDS = {"el", "la", "los", "las", "que", "qué", "cual", "cuál", "cuanto", "cuánto", "cuantos",
             "cuántos", "cuantas", "cuántas", "hay", "entre", "donde", "dónde", "esta", "está",
             "mostrame", "mostrá", "muestra", "ubica", "ubicá", "del", "de", "y", "es", "un", "una",
             "piso", "pared", "techo", "puerta", "puertas", "distancia", "holgura", "altura",
             "hola", "gracias", "solo", "todo", "todos", "apaga", "apagá", "encendé", "medí"}
_EN_WORDS = {"the", "what", "which", "how", "many", "much", "is", "are", "there", "between", "where",
             "show", "hide", "only", "all", "find", "locate", "of", "and", "a", "an", "to", "from",
             "floor", "wall", "ceiling", "door", "doors", "distance", "clearance", "height",
             "hello", "hi", "thanks", "measure"}


def language_of(question: str, default: str = "es") -> str:
    """The language of a question, by counting its function words (deterministic). ``default``
    when neither language has more."""
    words = [w.strip("¿?¡!.,;:()\"'").lower() for w in (question or "").split()]
    es = sum(1 for w in words if w in _ES_WORDS)
    en = sum(1 for w in words if w in _EN_WORDS)
    return "es" if es > en else "en" if en > es else default


def _n(v: Any, digits: int = 2) -> str:
    """A number as the tool gave it, at most ``digits`` decimals, no trailing zeros."""
    s = f"{float(v):.{digits}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _m(v: Any) -> str:
    """A length in metres; under 10 cm its millimetres too."""
    v = float(v)
    if abs(v) < 0.1:
        return f"{_n(v, 4)} m ({_n(v * 1000.0, 1)} mm)"
    return f"{_n(v, 2)} m"


def _vec(v: Any) -> str:
    return "(" + ", ".join(_n(x, 2) for x in v) + ")"


def _obj(res: Dict[str, Any], key: str, args: Dict[str, Any]) -> str:
    """label#id of the object a tool was called on, as the tool layer resolved it."""
    o = (res.get("about") or {}).get(key)
    if o:
        return f"{o['label']}#{o['id']}"
    return f"#{args.get(key)}"


def _t(lang: str, es: str, en: str) -> str:
    return es if lang == "es" else en


def _insufficient(res: Dict[str, Any], lang: str) -> Optional[str]:
    if res.get("insufficient_data"):
        return _t(lang, f"No hay datos suficientes: {res.get('reason', '')}.",
                  f"Insufficient data: {res.get('reason', '')}.")
    if res.get("error"):
        return _t(lang, f"La herramienta falló: {res['error']}.", f"The tool failed: {res['error']}.")
    return None


# ── one function per tool: (args, result, lang) -> sentence ─────────────────────────────────
def _focus_object(a, r, lang):
    ids = (r.get("viewer") or {}).get("ids") or [a.get("id")]
    name = f"{r.get('label', '')}#{ids[0]}"
    return _t(lang, f"{name} quedó seleccionado y enfocado en la vista.",
              f"{name} is selected and in view.")


def _get_object(a, r, lang):
    name = f"{r['label']}#{r['id']}"
    if "position_m" not in r:
        return _t(lang, f"{name} existe; no tiene caja medida.", f"{name} exists; it has no measured box.")
    s = r["size_m"]
    size = f"{_n(s['width'])} × {_n(s['height'])} × {_n(s['depth'])} m"
    return _t(lang, f"{name}: posición {_vec(r['position_m'])} m, tamaño {size} (ancho × alto × profundidad).",
              f"{name}: position {_vec(r['position_m'])} m, size {size} (width × height × depth).")


def _get_position(a, r, lang):
    return _t(lang, f"{_obj(r, 'id', a)} está en {_vec(r['position_m'])} m.",
              f"{_obj(r, 'id', a)} is at {_vec(r['position_m'])} m.")


def _get_object_size(a, r, lang):
    size = f"{_n(r['width_m'])} × {_n(r['height_m'])} × {_n(r['depth_m'])} m"
    return _t(lang, f"{_obj(r, 'id', a)} mide {size} (ancho × alto × profundidad).",
              f"{_obj(r, 'id', a)} measures {size} (width × height × depth).")


def _get_object_volume(a, r, lang):
    return _t(lang, f"El volumen de la caja de {_obj(r, 'id', a)} es {_n(r['bbox_volume_m3'], 3)} m³.",
              f"The box of {_obj(r, 'id', a)} has a volume of {_n(r['bbox_volume_m3'], 3)} m³.")


def _get_distance(a, r, lang):
    return _t(lang, f"La distancia entre {_obj(r, 'id1', a)} y {_obj(r, 'id2', a)} es {_m(r['distance_m'])}.",
              f"The distance between {_obj(r, 'id1', a)} and {_obj(r, 'id2', a)} is {_m(r['distance_m'])}.")


def _get_clearance(a, r, lang):
    return _t(lang, f"La holgura entre {_obj(r, 'id1', a)} y {_obj(r, 'id2', a)} es {_m(r['clearance_m'])}.",
              f"The clearance between {_obj(r, 'id1', a)} and {_obj(r, 'id2', a)} is {_m(r['clearance_m'])}.")


_PART = {"es": {"centroid": "el centro", "top": "la parte superior", "upper": "la parte superior",
                "bottom": "la base", "base": "la base", "lower": "la parte inferior",
                "highest": "el punto más alto", "lowest": "el punto más bajo",
                "closest": "el punto más cercano"},
         "en": {"centroid": "the centre", "top": "the top", "upper": "the upper part",
                "bottom": "the base", "base": "the base", "lower": "the lower part",
                "highest": "the highest point", "lowest": "the lowest point",
                "closest": "the closest point"}}


def _measure_between(a, r, lang):
    f1, f2 = _PART[lang].get(r.get("feature1"), ""), _PART[lang].get(r.get("feature2"), "")
    kind = _t(lang, "vertical" if r.get("axis") == "vertical" else "en línea recta",
              "vertical" if r.get("axis") == "vertical" else "straight")
    return _t(lang, f"La distancia {kind} entre {f1} de {_obj(r, 'id1', a)} y {f2} de "
                    f"{_obj(r, 'id2', a)} es {_m(r['distance_m'])}.",
              f"The {kind} distance between {f1} of {_obj(r, 'id1', a)} and {f2} of "
              f"{_obj(r, 'id2', a)} is {_m(r['distance_m'])}.")


def _get_extent(a, r, lang):
    return _t(lang, f"{_obj(r, 'id', a)}: alto {_m(r['height_m'])}; va de {_vec(r['min_m'])} a {_vec(r['max_m'])} m.",
              f"{_obj(r, 'id', a)}: height {_m(r['height_m'])}; from {_vec(r['min_m'])} to {_vec(r['max_m'])} m.")


def _get_height_profile(a, r, lang):
    st = [s for s in r.get("stations", []) if "y_min_m" in s]
    cl = [s["clearance_m"] for s in st if "clearance_m" in s]
    if cl:
        return _t(lang, f"Altura libre bajo {_obj(r, 'id', a)} sobre {_obj(r, 'floor_id', a)}: mínima "
                        f"{_m(min(cl))}, máxima {_m(max(cl))} en {len(cl)} tramos.",
                  f"Free height under {_obj(r, 'id', a)} over {_obj(r, 'floor_id', a)}: minimum "
                  f"{_m(min(cl))}, maximum {_m(max(cl))} over {len(cl)} stations.")
    lo, hi = min(s["y_min_m"] for s in st), max(s["y_max_m"] for s in st)
    return _t(lang, f"{_obj(r, 'id', a)}: su altura va de {_n(lo)} a {_n(hi)} m en {len(st)} tramos.",
              f"{_obj(r, 'id', a)}: its height runs from {_n(lo)} to {_n(hi)} m over {len(st)} stations.")


def _on(r, lang):
    """What the tilt was measured on: the element's face or its long axis."""
    return _t(lang, "su cara" if r.get("measured_on") == "face" else "su eje",
              "its face" if r.get("measured_on") == "face" else "its axis")


def _get_plumb(a, r, lang):
    if r.get("vertical_element") is False:
        return _t(lang, f"{_obj(r, 'id', a)} no es un elemento vertical: {_on(r, lang)} está a "
                        f"{_n(r['plumb_deviation_deg'], 1)}° de la vertical.",
                  f"{_obj(r, 'id', a)} is not a vertical element: {_on(r, lang)} is "
                  f"{_n(r['plumb_deviation_deg'], 1)}° from the vertical.")
    return _t(lang, f"{_obj(r, 'id', a)}: desplome de {_on(r, lang)} {_n(r['plumb_deviation_deg'], 3)}° "
                    f"({_n(r['deviation_mm_per_m'], 1)} mm/m).",
              f"{_obj(r, 'id', a)}: {_on(r, lang)} is out of plumb by {_n(r['plumb_deviation_deg'], 3)}° "
              f"({_n(r['deviation_mm_per_m'], 1)} mm/m).")


def _get_level(a, r, lang):
    if r.get("horizontal_element") is False:
        return _t(lang, f"{_obj(r, 'id', a)} no es un elemento horizontal: {_on(r, lang)} está a "
                        f"{_n(r['level_deviation_deg'], 1)}° de la horizontal.",
                  f"{_obj(r, 'id', a)} is not a horizontal element: {_on(r, lang)} is "
                  f"{_n(r['level_deviation_deg'], 1)}° from the horizontal.")
    return _t(lang, f"{_obj(r, 'id', a)}: desnivel de {_on(r, lang)} {_n(r['level_deviation_deg'], 3)}° "
                    f"({_n(r['deviation_mm_per_m'], 1)} mm/m).",
              f"{_obj(r, 'id', a)}: {_on(r, lang)} is out of level by {_n(r['level_deviation_deg'], 3)}° "
              f"({_n(r['deviation_mm_per_m'], 1)} mm/m).")


def _get_span(a, r, lang):
    return _t(lang, f"La luz de {_obj(r, 'id', a)} es {_m(r['span_m'])}.",
              f"The span of {_obj(r, 'id', a)} is {_m(r['span_m'])}.")


def _get_flatness_report(a, r, lang):
    if r.get("rms_mm") is None:
        return None
    ok = r.get("flatness_pass")
    verdict = "" if ok is None else _t(lang, "; cumple" if ok else "; no cumple", "; passes" if ok else "; fails")
    tol = "" if r.get("tolerance_mm") is None else _t(lang, f" (tolerancia {_n(r['tolerance_mm'], 1)} mm{verdict})",
                                                         f" (tolerance {_n(r['tolerance_mm'], 1)} mm{verdict})")
    return _t(lang, f"Planitud de {_obj(r, 'id', a)}: rms {_n(r['rms_mm'], 1)} mm, p95 {_n(r['p95_mm'], 1)} mm, "
                    f"máx {_n(r['max_mm'], 1)} mm{tol}.",
              f"Flatness of {_obj(r, 'id', a)}: rms {_n(r['rms_mm'], 1)} mm, p95 {_n(r['p95_mm'], 1)} mm, "
              f"max {_n(r['max_mm'], 1)} mm{tol}.")


def _count_objects(a, r, lang):
    if "by_label" in r:
        top = ", ".join(f"{k} {v}" for k, v in list(r["by_label"].items())[:6])
        return _t(lang, f"Hay {r['count']} objetos. Los más numerosos: {top}.",
                  f"There are {r['count']} objects. The most numerous: {top}.")
    names = ", ".join(r.get("matched_labels") or []) or str(r.get("label"))
    return _t(lang, f"Hay {r['count']} objetos «{names}».", f"There are {r['count']} “{names}” objects.")


def _list_objects(a, r, lang):
    if r.get("listed") == "by_label":
        top = ", ".join(f"{x['label']} ({x['count']})" for x in r["labels"][:8])
        return _t(lang, f"{r['count']} objetos, por etiqueta: {top}.",
                  f"{r['count']} objects, by label: {top}.")
    objs = r.get("objects") or []
    if not objs:
        return _t(lang, "No hay objetos con esa etiqueta.", "There is no object with that label.")
    ids = ", ".join(f"{o['label']}#{o['id']}" for o in objs)
    return _t(lang, f"{len(objs)} objetos: {ids}.", f"{len(objs)} objects: {ids}.")


def _labels(r) -> str:
    return ", ".join(r.get("labels") or [])


def _show_only(a, r, lang):
    return _t(lang, f"Se muestran solo {r.get('count', len((r.get('viewer') or {}).get('ids', [])))} objetos ({_labels(r)}).",
              f"Only {r.get('count', len((r.get('viewer') or {}).get('ids', [])))} objects are shown ({_labels(r)}).")


def _show_objects(a, r, lang):
    n = r.get("count", len((r.get("viewer") or {}).get("ids", [])))
    on = bool((r.get("viewer") or {}).get("visible", True))
    return _t(lang, f"{'Se muestran' if on else 'Se ocultaron'} {n} objetos ({_labels(r)}).",
              f"{n} objects {'shown' if on else 'hidden'} ({_labels(r)}).")


def _show_all(a, r, lang):
    return _t(lang, "Se muestran todos los objetos.", "Every object is shown.")


def _set_obb(a, r, lang):
    on = bool((r.get("viewer") or {}).get("on", True))
    n = r.get("count", len((r.get("viewer") or {}).get("ids", [])))
    return _t(lang, f"Cajas {'encendidas' if on else 'apagadas'} en {n} objetos.",
              f"Boxes turned {'on' if on else 'off'} on {n} objects.")


def _viewer_done(es: str, en: str) -> Callable:
    return lambda a, r, lang: _t(lang, es, en)


def _travel(a, r, lang):
    return _t(lang, f"Recorrido de {_obj(r, 'from_id', a)} a {_obj(r, 'to_id', a)}.",
              f"Travelling from {_obj(r, 'from_id', a)} to {_obj(r, 'to_id', a)}.")


def _evaluate_volume(a, r, lang):
    inside = ", ".join(f"{o['label']}#{o['id']}" for o in (r.get("objects_inside") or [])[:4])
    es = (f"El volumen de {_n(r['box_volume_m3'], 2)} m³ tiene {_n(100 * r['free_fraction'], 1)} % libre "
          f"({_n(r['free_volume_m3'], 2)} m³)")
    en = (f"The {_n(r['box_volume_m3'], 2)} m³ volume is {_n(100 * r['free_fraction'], 1)} % free "
          f"({_n(r['free_volume_m3'], 2)} m³)")
    if inside:
        es += f"; adentro: {inside}"
        en += f"; inside: {inside}"
    return _t(lang, es + ".", en + ".")


def _objects_in_volume(a, r, lang):
    objs = r.get("objects") or []
    if not objs:
        return _t(lang, "Ningún objeto toca ese volumen.", "No object touches that volume.")
    top = ", ".join(f"{o['label']}#{o['id']} ({_n(100 * o['fraction_inside'], 0)} %)" for o in objs[:6])
    return _t(lang, f"{len(objs)} objetos tocan el volumen: {top}.", f"{len(objs)} objects touch the volume: {top}.")


def _fits(a, r, lang):
    if r.get("fits"):
        return _t(lang, "Sí, entra.", "Yes, it fits.")
    why = r.get("reason")
    return _t(lang, "No entra" + (f": {why}" if why else "") + ".", "It does not fit" + (f": {why}" if why else "") + ".")


def _fits_through(a, r, lang):
    if "gap_m" in r:
        return _t(lang, f"{'Sí, pasa' if r['fits'] else 'No pasa'}: el paso entre los dos objetos es {_m(r['gap_m'])} "
                        f"y la pieza necesita {_m(r['item_min_dim_m'])} más {_m(r['margin_m'])} de margen.",
                  f"{'Yes, it passes' if r['fits'] else 'It does not pass'}: the gap between the two objects is "
                  f"{_m(r['gap_m'])} and the item needs {_m(r['item_min_dim_m'])} plus a {_m(r['margin_m'])} margin.")
    p = " × ".join(_n(v) for v in r.get("passage_m", []))
    return _t(lang, f"{'Sí, pasa' if r['fits'] else 'No pasa'}: la abertura mide {p} m.",
              f"{'Yes, it passes' if r['fits'] else 'It does not pass'}: the opening measures {p} m.")


def _measure_volume(a, r, lang):
    es = f"Del volumen a {_obj(r, 'id', a)}: {_m(r['clearance_m'])}"
    en = f"From the volume to {_obj(r, 'id', a)}: {_m(r['clearance_m'])}"
    if r.get("intersects"):
        es += f"; se cruzan ({r['points_inside']} puntos adentro)"
        en += f"; they intersect ({r['points_inside']} points inside)"
    return _t(lang, es + ".", en + ".")


def _define_volume(a, r, lang):
    size = " × ".join(_n(v) for v in r["size"])
    return _t(lang, f"Volumen «{r.get('name')}» creado: {size} m, centro {_vec(r['center'])} m.",
              f"Volume “{r.get('name')}” created: {size} m, centre {_vec(r['center'])} m.")


def _session_info(a, r, lang):
    return _t(lang, f"Proyecto {r.get('project')}: {r.get('n_objects')} objetos, {r.get('scan_frames')} frames, "
                    f"{r.get('cloud_points')} puntos en la nube.",
              f"Project {r.get('project')}: {r.get('n_objects')} objects, {r.get('scan_frames')} frames, "
              f"{r.get('cloud_points')} cloud points.")


def _describe(a, r, lang):
    text = r.get("description") or r.get("scene_description") or r.get("que_es")
    return str(text) if text else None


def _small_talk(a, r, lang):
    return _t(lang, "Decime qué querés medir, ubicar o ver de esta escena.",
              "Tell me what to measure, locate or show in this scene.")


COMPOSERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any], str], Optional[str]]] = {
    "focus_object": _focus_object, "get_object": _get_object, "get_position": _get_position,
    "get_object_size": _get_object_size, "get_object_volume": _get_object_volume,
    "get_distance": _get_distance, "get_clearance": _get_clearance,
    "measure_between": _measure_between, "get_extent": _get_extent,
    "get_height_profile": _get_height_profile, "get_plumb": _get_plumb, "get_level": _get_level,
    "get_span": _get_span, "get_flatness_report": _get_flatness_report,
    "count_objects": _count_objects, "list_objects": _list_objects,
    "show_only": _show_only, "show_objects": _show_objects, "show_all_objects": _show_all,
    "set_obb": _set_obb, "travel": _travel,
    "set_view": _viewer_done("Vista cambiada.", "View changed."),
    "rotate_view": _viewer_done("Vista girada.", "View rotated."),
    "color_cloud": _viewer_done("Color de la nube cambiado.", "Cloud colouring changed."),
    "clear_view": _viewer_done("Vista limpia.", "View cleared."),
    "evaluate_volume": _evaluate_volume, "objects_in_volume": _objects_in_volume,
    "fits_in_volume": _fits, "fits_through": _fits_through, "measure_volume": _measure_volume,
    "define_volume": _define_volume, "get_session_info": _session_info,
    "describe_scene": _describe, "describe_object": _describe, "small_talk": _small_talk,
}


def compose(tool: str, args: Dict[str, Any], result: Any, lang: str) -> Optional[str]:
    """The sentence for one tool call, or None when this tool (or this result) has no wording
    here."""
    if not isinstance(result, dict):
        return None
    lang = lang if lang in LANGS else "es"
    bad = _insufficient(result, lang)
    if bad:
        return bad
    fn = COMPOSERS.get(tool)
    if fn is None:
        return None
    try:
        return fn(args or {}, result, lang)
    except (KeyError, TypeError, ValueError, IndexError):
        return None                       # a result this wording does not know: the model words it


def compose_turn(trace: List[Dict[str, Any]], lang: str) -> Optional[str]:
    """The answer of a turn from every tool call it made, in order — None as soon as one of them
    has no wording (the whole turn then goes to the model)."""
    parts: List[str] = []
    for e in trace:
        s = compose(str(e.get("tool")), e.get("arguments") or {}, e.get("result"), lang)
        if s is None:
            return None
        if s not in parts:
            parts.append(s)
    return " ".join(parts) if parts else None
