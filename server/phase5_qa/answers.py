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

LANGS = ("es", "en", "fr")               # the system's three languages

_ES_WORDS = {"el", "la", "los", "las", "que", "qué", "cual", "cuál", "cuanto", "cuánto", "cuantos",
             "cuántos", "cuantas", "cuántas", "hay", "entre", "donde", "dónde", "esta", "está",
             "mostrame", "mostrá", "muestra", "ubica", "ubicá", "del", "de", "y", "es", "un", "una",
             "piso", "pared", "techo", "puerta", "puertas", "distancia", "holgura", "altura",
             "hola", "gracias", "solo", "todo", "todos", "apaga", "apagá", "encendé", "medí",
             "abrí", "abre", "abrir", "cerrá", "cierra", "conectá", "conecta", "con", "servidor",
             "sesión", "pantalla", "ocultá", "oculta", "quiero", "me", "llamo", "soy", "mi",
             "nombre", "puedes", "podés", "ayuda", "por", "favor", "malla", "nube", "no", "sí",
             "te", "lo", "se", "al", "en", "para", "si", "como", "cómo", "más", "pero", "le", "su",
             "este", "eso", "esto", "ese", "yo", "tú", "vos", "voy", "a", "ya", "muy", "bien",
             "también", "ahora", "aquí", "acá", "ahí", "cuando", "porque", "tengo", "puedo"}
_EN_WORDS = {"the", "what", "which", "how", "many", "much", "is", "are", "there", "between", "where",
             "show", "hide", "only", "all", "find", "locate", "of", "and", "a", "an", "to", "from",
             "floor", "wall", "ceiling", "door", "doors", "distance", "clearance", "height",
             "hello", "hi", "thanks", "measure", "open", "close", "connect", "server", "session",
             "screen", "full", "my", "name", "call", "me", "can", "you", "help", "please", "mesh",
             "cloud", "turn", "on", "off", "i", "want", "no", "yes", "it", "this", "that", "in",
             "for", "with", "not", "do", "does", "will", "would", "now", "here", "please", "but",
             "your", "am", "be", "at", "by"}
_FR_WORDS = {"le", "la", "les", "un", "une", "des", "du", "de", "est", "sont", "quel", "quelle",
             "quels", "quelles", "combien", "où", "entre", "et", "il", "y", "montre", "affiche",
             "masque", "cache", "mesure", "sol", "mur", "plafond", "porte", "portes", "distance",
             "hauteur", "jeu", "bonjour", "merci", "seulement", "tout", "tous", "ouvre", "ferme",
             "avec", "dans", "sur", "sous", "plein", "écran", "je", "m'appelle", "appelle", "mon",
             "nom", "peux", "tu", "aide", "s'il", "plaît", "connecte", "serveur", "session",
             "maillage", "nuage", "veux", "au", "non", "oui", "ne", "pas", "ce", "cette", "ça",
             "pour", "que", "qui", "mais", "vous", "nous", "votre", "en", "a", "à", "suis", "ai",
             "maintenant", "ici", "très", "bien", "aussi"}


def language_of(question: str, default: str = "es") -> str:
    """The language of a WRITTEN question, by counting its function words (deterministic).
    ``default`` when no language has more than the others (a spoken question's language is the
    recogniser's, not this)."""
    words = [w.strip("¿?¡!.,;:()\"").lower() for w in (question or "").split()]
    n = {"es": sum(1 for w in words if w in _ES_WORDS), "en": sum(1 for w in words if w in _EN_WORDS),
         "fr": sum(1 for w in words if w in _FR_WORDS)}
    best = max(n.values())
    winners = [l for l in LANGS if n[l] == best]
    if best == 0 or len(winners) != 1:
        return default if default in LANGS else "es"
    return winners[0]


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


def _t(lang: str, es: str, en: str, fr: str) -> str:
    return es if lang == "es" else fr if lang == "fr" else en


def _insufficient(res: Dict[str, Any], lang: str) -> Optional[str]:
    if res.get("insufficient_data"):
        why = res.get("reason", "")
        return _t(lang, f"No hay datos suficientes: {why}.", f"Insufficient data: {why}.",
                  f"Données insuffisantes : {why}.")
    if res.get("error"):
        return _t(lang, f"La herramienta falló: {res['error']}.", f"The tool failed: {res['error']}.",
                  f"L'outil a échoué : {res['error']}.")
    return None


# ── one function per tool: (args, result, lang) -> sentence ─────────────────────────────────
def _focus_object(a, r, lang):
    ids = (r.get("viewer") or {}).get("ids") or [a.get("id")]
    name = f"{r.get('label', '')}#{ids[0]}"
    return _t(lang, f"{name} quedó seleccionado y enfocado en la vista.",
              f"{name} is selected and in view.",
              f"{name} est sélectionné et cadré dans la vue.")


def _get_object(a, r, lang):
    name = f"{r['label']}#{r['id']}"
    if "position_m" not in r:
        return _t(lang, f"{name} existe; no tiene caja medida.", f"{name} exists; it has no measured box.",
                  f"{name} existe ; il n'a pas de boîte mesurée.")
    s = r["size_m"]
    size = f"{_n(s['width'])} × {_n(s['height'])} × {_n(s['depth'])} m"
    return _t(lang, f"{name}: posición {_vec(r['position_m'])} m, tamaño {size} (ancho × alto × profundidad).",
              f"{name}: position {_vec(r['position_m'])} m, size {size} (width × height × depth).",
              f"{name} : position {_vec(r['position_m'])} m, taille {size} (largeur × hauteur × profondeur).")


def _get_position(a, r, lang):
    return _t(lang, f"{_obj(r, 'id', a)} está en {_vec(r['position_m'])} m.",
              f"{_obj(r, 'id', a)} is at {_vec(r['position_m'])} m.",
              f"{_obj(r, 'id', a)} se trouve en {_vec(r['position_m'])} m.")


def _get_object_size(a, r, lang):
    size = f"{_n(r['width_m'])} × {_n(r['height_m'])} × {_n(r['depth_m'])} m"
    return _t(lang, f"{_obj(r, 'id', a)} mide {size} (ancho × alto × profundidad).",
              f"{_obj(r, 'id', a)} measures {size} (width × height × depth).",
              f"{_obj(r, 'id', a)} mesure {size} (largeur × hauteur × profondeur).")


def _get_object_volume(a, r, lang):
    v = _n(r['bbox_volume_m3'], 3)
    return _t(lang, f"El volumen de la caja de {_obj(r, 'id', a)} es {v} m³.",
              f"The box of {_obj(r, 'id', a)} has a volume of {v} m³.",
              f"Le volume de la boîte de {_obj(r, 'id', a)} est de {v} m³.")


def _get_distance(a, r, lang):
    o1, o2, d = _obj(r, 'id1', a), _obj(r, 'id2', a), _m(r['distance_m'])
    return _t(lang, f"La distancia entre {o1} y {o2} es {d}.", f"The distance between {o1} and {o2} is {d}.",
              f"La distance entre {o1} et {o2} est de {d}.")


def _get_clearance(a, r, lang):
    o1, o2, d = _obj(r, 'id1', a), _obj(r, 'id2', a), _m(r['clearance_m'])
    return _t(lang, f"La holgura entre {o1} y {o2} es {d}.", f"The clearance between {o1} and {o2} is {d}.",
              f"Le jeu entre {o1} et {o2} est de {d}.")


_PART = {"es": {"centroid": "el centro", "top": "la parte superior", "upper": "la parte superior",
                "bottom": "la base", "base": "la base", "lower": "la parte inferior",
                "highest": "el punto más alto", "lowest": "el punto más bajo",
                "closest": "el punto más cercano"},
         "en": {"centroid": "the centre", "top": "the top", "upper": "the upper part",
                "bottom": "the base", "base": "the base", "lower": "the lower part",
                "highest": "the highest point", "lowest": "the lowest point",
                "closest": "the closest point"},
         "fr": {"centroid": "le centre", "top": "le haut", "upper": "la partie haute",
                "bottom": "la base", "base": "la base", "lower": "la partie basse",
                "highest": "le point le plus haut", "lowest": "le point le plus bas",
                "closest": "le point le plus proche"}}


def _measure_between(a, r, lang):
    f1, f2 = _PART[lang].get(r.get("feature1"), ""), _PART[lang].get(r.get("feature2"), "")
    vertical = r.get("axis") == "vertical"
    o1, o2, d = _obj(r, 'id1', a), _obj(r, 'id2', a), _m(r['distance_m'])
    return _t(lang, f"La distancia {'vertical' if vertical else 'en línea recta'} entre {f1} de {o1} y {f2} de {o2} es {d}.",
              f"The {'vertical' if vertical else 'straight'} distance between {f1} of {o1} and {f2} of {o2} is {d}.",
              f"La distance {'verticale' if vertical else 'en ligne droite'} entre {f1} de {o1} et {f2} de {o2} est de {d}.")


def _get_extent(a, r, lang):
    o, h, lo, hi = _obj(r, 'id', a), _m(r['height_m']), _vec(r['min_m']), _vec(r['max_m'])
    return _t(lang, f"{o}: alto {h}; va de {lo} a {hi} m.", f"{o}: height {h}; from {lo} to {hi} m.",
              f"{o} : hauteur {h} ; de {lo} à {hi} m.")


def _get_height_profile(a, r, lang):
    st = [s for s in r.get("stations", []) if "y_min_m" in s]
    cl = [s["clearance_m"] for s in st if "clearance_m" in s]
    o = _obj(r, 'id', a)
    if cl:
        f, lo, hi, n = _obj(r, 'floor_id', a), _m(min(cl)), _m(max(cl)), len(cl)
        return _t(lang, f"Altura libre bajo {o} sobre {f}: mínima {lo}, máxima {hi} en {n} tramos.",
                  f"Free height under {o} over {f}: minimum {lo}, maximum {hi} over {n} stations.",
                  f"Hauteur libre sous {o} au-dessus de {f} : minimum {lo}, maximum {hi} sur {n} stations.")
    lo, hi = _n(min(s["y_min_m"] for s in st)), _n(max(s["y_max_m"] for s in st))
    return _t(lang, f"{o}: su altura va de {lo} a {hi} m en {len(st)} tramos.",
              f"{o}: its height runs from {lo} to {hi} m over {len(st)} stations.",
              f"{o} : sa hauteur va de {lo} à {hi} m sur {len(st)} stations.")


def _on(r, lang):
    """What the tilt was measured on: the element's face or its long axis."""
    face = r.get("measured_on") == "face"
    return _t(lang, "su cara" if face else "su eje", "its face" if face else "its axis",
              "sa face" if face else "son axe")


def _get_plumb(a, r, lang):
    o, on = _obj(r, 'id', a), _on(r, lang)
    if r.get("vertical_element") is False:
        d = _n(r['plumb_deviation_deg'], 1)
        return _t(lang, f"{o} no es un elemento vertical: {on} está a {d}° de la vertical.",
                  f"{o} is not a vertical element: {on} is {d}° from the vertical.",
                  f"{o} n'est pas un élément vertical : {on} est à {d}° de la verticale.")
    d, mm = _n(r['plumb_deviation_deg'], 3), _n(r['deviation_mm_per_m'], 1)
    return _t(lang, f"{o}: desplome de {on} {d}° ({mm} mm/m).",
              f"{o}: {on} is out of plumb by {d}° ({mm} mm/m).",
              f"{o} : faux aplomb de {on} {d}° ({mm} mm/m).")


def _get_level(a, r, lang):
    o, on = _obj(r, 'id', a), _on(r, lang)
    if r.get("horizontal_element") is False:
        d = _n(r['level_deviation_deg'], 1)
        return _t(lang, f"{o} no es un elemento horizontal: {on} está a {d}° de la horizontal.",
                  f"{o} is not a horizontal element: {on} is {d}° from the horizontal.",
                  f"{o} n'est pas un élément horizontal : {on} est à {d}° de l'horizontale.")
    d, mm = _n(r['level_deviation_deg'], 3), _n(r['deviation_mm_per_m'], 1)
    return _t(lang, f"{o}: desnivel de {on} {d}° ({mm} mm/m).",
              f"{o}: {on} is out of level by {d}° ({mm} mm/m).",
              f"{o} : défaut de niveau de {on} {d}° ({mm} mm/m).")


def _get_span(a, r, lang):
    o, d = _obj(r, 'id', a), _m(r['span_m'])
    return _t(lang, f"La luz de {o} es {d}.", f"The span of {o} is {d}.", f"La portée de {o} est de {d}.")


def _get_flatness_report(a, r, lang):
    if r.get("rms_mm") is None:
        return None
    ok = r.get("flatness_pass")
    verdict = "" if ok is None else _t(lang, "; cumple" if ok else "; no cumple", "; passes" if ok else "; fails",
                                       " ; conforme" if ok else " ; non conforme")
    tol = "" if r.get("tolerance_mm") is None else _t(
        lang, f" (tolerancia {_n(r['tolerance_mm'], 1)} mm{verdict})", f" (tolerance {_n(r['tolerance_mm'], 1)} mm{verdict})",
        f" (tolérance {_n(r['tolerance_mm'], 1)} mm{verdict})")
    o, rms, p95, mx = _obj(r, 'id', a), _n(r['rms_mm'], 1), _n(r['p95_mm'], 1), _n(r['max_mm'], 1)
    return _t(lang, f"Planitud de {o}: rms {rms} mm, p95 {p95} mm, máx {mx} mm{tol}.",
              f"Flatness of {o}: rms {rms} mm, p95 {p95} mm, max {mx} mm{tol}.",
              f"Planéité de {o} : rms {rms} mm, p95 {p95} mm, max {mx} mm{tol}.")


def _count_objects(a, r, lang):
    if "by_label" in r:
        top = ", ".join(f"{k} {v}" for k, v in list(r["by_label"].items())[:6])
        return _t(lang, f"Hay {r['count']} objetos. Los más numerosos: {top}.",
                  f"There are {r['count']} objects. The most numerous: {top}.",
                  f"Il y a {r['count']} objets. Les plus nombreux : {top}.")
    names = ", ".join(r.get("matched_labels") or []) or str(r.get("label"))
    return _t(lang, f"Hay {r['count']} objetos «{names}».", f"There are {r['count']} “{names}” objects.",
              f"Il y a {r['count']} objets « {names} ».")


def _list_objects(a, r, lang):
    if r.get("listed") == "by_label":
        top = ", ".join(f"{x['label']} ({x['count']})" for x in r["labels"][:8])
        return _t(lang, f"{r['count']} objetos, por etiqueta: {top}.", f"{r['count']} objects, by label: {top}.",
                  f"{r['count']} objets, par étiquette : {top}.")
    objs = r.get("objects") or []
    if not objs:
        return _t(lang, "No hay objetos con esa etiqueta.", "There is no object with that label.",
                  "Aucun objet avec cette étiquette.")
    ids = ", ".join(f"{o['label']}#{o['id']}" for o in objs)
    return _t(lang, f"{len(objs)} objetos: {ids}.", f"{len(objs)} objects: {ids}.", f"{len(objs)} objets : {ids}.")


def _labels(r) -> str:
    return ", ".join(r.get("labels") or [])


def _show_only(a, r, lang):
    n = r.get('count', len((r.get('viewer') or {}).get('ids', [])))
    return _t(lang, f"Se muestran solo {n} objetos ({_labels(r)}).", f"Only {n} objects are shown ({_labels(r)}).",
              f"Seuls {n} objets sont affichés ({_labels(r)}).")


def _show_objects(a, r, lang):
    n = r.get("count", len((r.get("viewer") or {}).get("ids", [])))
    on = bool((r.get("viewer") or {}).get("visible", True))
    return _t(lang, f"{'Se muestran' if on else 'Se ocultaron'} {n} objetos ({_labels(r)}).",
              f"{n} objects {'shown' if on else 'hidden'} ({_labels(r)}).",
              f"{n} objets {'affichés' if on else 'masqués'} ({_labels(r)}).")


def _show_all(a, r, lang):
    return _t(lang, "Se muestran todos los objetos.", "Every object is shown.", "Tous les objets sont affichés.")


def _set_obb(a, r, lang):
    on = bool((r.get("viewer") or {}).get("on", True))
    n = r.get("count", len((r.get("viewer") or {}).get("ids", [])))
    return _t(lang, f"Cajas {'encendidas' if on else 'apagadas'} en {n} objetos.",
              f"Boxes turned {'on' if on else 'off'} on {n} objects.",
              f"Boîtes {'activées' if on else 'désactivées'} sur {n} objets.")


def _viewer_done(es: str, en: str, fr: str) -> Callable:
    return lambda a, r, lang: _t(lang, es, en, fr)


def _travel(a, r, lang):
    o1, o2 = _obj(r, 'from_id', a), _obj(r, 'to_id', a)
    return _t(lang, f"Recorrido de {o1} a {o2}.", f"Travelling from {o1} to {o2}.", f"Trajet de {o1} à {o2}.")


def _evaluate_volume(a, r, lang):
    inside = ", ".join(f"{o['label']}#{o['id']}" for o in (r.get("objects_inside") or [])[:4])
    box, pct, free = _n(r['box_volume_m3'], 2), _n(100 * r['free_fraction'], 1), _n(r['free_volume_m3'], 2)
    es = f"El volumen de {box} m³ tiene {pct} % libre ({free} m³)"
    en = f"The {box} m³ volume is {pct} % free ({free} m³)"
    fr = f"Le volume de {box} m³ est libre à {pct} % ({free} m³)"
    if inside:
        es += f"; adentro: {inside}"
        en += f"; inside: {inside}"
        fr += f" ; à l'intérieur : {inside}"
    return _t(lang, es + ".", en + ".", fr + ".")


def _objects_in_volume(a, r, lang):
    objs = r.get("objects") or []
    if not objs:
        return _t(lang, "Ningún objeto toca ese volumen.", "No object touches that volume.",
                  "Aucun objet ne touche ce volume.")
    top = ", ".join(f"{o['label']}#{o['id']} ({_n(100 * o['fraction_inside'], 0)} %)" for o in objs[:6])
    return _t(lang, f"{len(objs)} objetos tocan el volumen: {top}.", f"{len(objs)} objects touch the volume: {top}.",
              f"{len(objs)} objets touchent le volume : {top}.")


def _fits(a, r, lang):
    if r.get("fits"):
        return _t(lang, "Sí, entra.", "Yes, it fits.", "Oui, ça rentre.")
    why = r.get("reason")
    tail = (f": {why}" if why else "") + "."
    return _t(lang, "No entra" + tail, "It does not fit" + tail, "Ça ne rentre pas" + (f" : {why}" if why else "") + ".")


def _fits_through(a, r, lang):
    yes = bool(r['fits'])
    if "gap_m" in r:
        gap, need, margin = _m(r['gap_m']), _m(r['item_min_dim_m']), _m(r['margin_m'])
        return _t(lang, f"{'Sí, pasa' if yes else 'No pasa'}: el paso entre los dos objetos es {gap} y la pieza "
                        f"necesita {need} más {margin} de margen.",
                  f"{'Yes, it passes' if yes else 'It does not pass'}: the gap between the two objects is {gap} "
                  f"and the item needs {need} plus a {margin} margin.",
                  f"{'Oui, ça passe' if yes else 'Ça ne passe pas'} : le passage entre les deux objets est de {gap} "
                  f"et la pièce a besoin de {need} plus {margin} de marge.")
    p = " × ".join(_n(v) for v in r.get("passage_m", []))
    return _t(lang, f"{'Sí, pasa' if yes else 'No pasa'}: la abertura mide {p} m.",
              f"{'Yes, it passes' if yes else 'It does not pass'}: the opening measures {p} m.",
              f"{'Oui, ça passe' if yes else 'Ça ne passe pas'} : l'ouverture mesure {p} m.")


def _measure_volume(a, r, lang):
    o, d = _obj(r, 'id', a), _m(r['clearance_m'])
    es, en, fr = f"Del volumen a {o}: {d}", f"From the volume to {o}: {d}", f"Du volume à {o} : {d}"
    if r.get("intersects"):
        es += f"; se cruzan ({r['points_inside']} puntos adentro)"
        en += f"; they intersect ({r['points_inside']} points inside)"
        fr += f" ; ils se croisent ({r['points_inside']} points à l'intérieur)"
    return _t(lang, es + ".", en + ".", fr + ".")


def _define_volume(a, r, lang):
    size = " × ".join(_n(v) for v in r["size"])
    return _t(lang, f"Volumen «{r.get('name')}» creado: {size} m, centro {_vec(r['center'])} m.",
              f"Volume “{r.get('name')}” created: {size} m, centre {_vec(r['center'])} m.",
              f"Volume « {r.get('name')} » créé : {size} m, centre {_vec(r['center'])} m.")


def _session_info(a, r, lang):
    p, n, f, c = r.get('project'), r.get('n_objects'), r.get('scan_frames'), r.get('cloud_points')
    return _t(lang, f"Proyecto {p}: {n} objetos, {f} frames, {c} puntos en la nube.",
              f"Project {p}: {n} objects, {f} frames, {c} cloud points.",
              f"Projet {p} : {n} objets, {f} images, {c} points dans le nuage.")


def _describe(a, r, lang):
    text = r.get("description") or r.get("scene_description") or r.get("que_es")
    return str(text) if text else None


def small_talk_default(lang: str) -> str:
    return _t(lang, "Dime qué quieres medir, ubicar o ver de esta escena.",
              "Tell me what to measure, locate or show in this scene.",
              "Dites-moi ce que vous voulez mesurer, localiser ou voir dans cette scène.")


def _small_talk(a, r, lang):
    """The conversation: what the model wrote in ``reply`` — unless it carries a figure with a
    unit (a conversation never states a measurement: the tools do), then the fixed sentence."""
    reply = " ".join(str(r.get("reply") or "").split())
    if reply:
        from . import provenance
        if not provenance.figures_in(reply):
            return reply
    return small_talk_default(lang)


def _help_commands(a, r, lang):
    if r.get("commands") is not None:
        titles = ", ".join(r["commands"])
        text = _t(lang, f"{r['title']}: {titles}.", f"{r['title']}: {titles}.", f"{r['title']} : {titles}.")
        if r.get("manual"):
            by_hand = ", ".join(r["manual"])
            text += " " + _t(lang, f"A mano en la interfaz: {by_hand}.", f"By hand in the interface: {by_hand}.",
                             f"À la main dans l'interface : {by_hand}.")
        return text
    areas = ", ".join(x["title"] for x in r.get("areas") or [])
    return _t(lang, f"Puedo ayudarte con: {areas}. Dime un área y te cuento sus comandos.",
              f"I can help with: {areas}. Name an area and I will list its commands.",
              f"Je peux vous aider avec : {areas}. Nommez un domaine et je liste ses commandes.")


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
    "set_view": _viewer_done("Vista cambiada.", "View changed.", "Vue changée."),
    "rotate_view": _viewer_done("Vista girada.", "View rotated.", "Vue pivotée."),
    "color_cloud": _viewer_done("Color de la nube cambiado.", "Cloud colouring changed.", "Couleur du nuage changée."),
    "clear_view": _viewer_done("Vista limpia.", "View cleared.", "Vue nettoyée."),
    "evaluate_volume": _evaluate_volume, "objects_in_volume": _objects_in_volume,
    "fits_in_volume": _fits, "fits_through": _fits_through, "measure_volume": _measure_volume,
    "define_volume": _define_volume, "get_session_info": _session_info,
    "describe_scene": _describe, "describe_object": _describe, "small_talk": _small_talk,
    "help__commands": _help_commands,
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
    if result.get("manual"):
        # a command the interface only does by hand (commands.db status: manual)
        title, area = result.get("title", ""), result.get("area", "")
        return _t(lang, f"«{title}» se hace a mano por ahora, en {area}.",
                  f"“{title}” is done by hand for now, under {area}.",
                  f"« {title} » se fait à la main pour l'instant, dans {area}.")
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


# ── why a command of the interface did not run ─────────────────────────────────────────────
# The interface runs its commands and reports a REASON code when one does not; the sentence is
# worded here, in the conversation's language ({detail}: what the executor adds).
_UI_REASONS = {
    "no_session": ("Primero hay que abrir una sesión.", "A session has to be open first.",
                   "Il faut d'abord ouvrir une session."),
    "not_connected": ("Primero hay que conectar con el servidor.", "Connect to the server first.",
                      "Il faut d'abord se connecter au serveur."),
    "server_down": ("El servidor no respondió.", "The server did not answer.", "Le serveur n'a pas répondu."),
    "not_found": ("No encontré «{detail}».", "I could not find “{detail}”.", "Je n'ai pas trouvé « {detail} »."),
    "ambiguous": ("Hay varias opciones: {detail}. ¿Cuál?", "There are several: {detail}. Which one?",
                  "Il y en a plusieurs : {detail}. Lequel ?"),
    "nothing_to_do": ("No hay nada para eso en esta sesión.", "There is nothing for that in this session.",
                      "Il n'y a rien pour cela dans cette session."),
    "needs_gesture": ("El navegador solo lo permite después de un clic o una tecla: usa F11.",
                      "The browser only allows it right after a click or a key: use F11.",
                      "Le navigateur ne l'autorise qu'après un clic ou une touche : utilisez F11."),
    "unavailable": ("Eso no está disponible ahora.", "That is not available right now.",
                    "Ce n'est pas disponible pour le moment."),
    "not_wired": ("Ese comando todavía no está conectado a la interfaz.",
                  "That command is not wired to the interface yet.",
                  "Cette commande n'est pas encore reliée à l'interface."),
    "failed": ("No se pudo: {detail}.", "It failed: {detail}.", "Échec : {detail}."),
}


def ui_reasons(lang: str) -> Dict[str, str]:
    """reason code → sentence, in ``lang``."""
    lang = lang if lang in LANGS else "es"
    return {k: _t(lang, *v) for k, v in _UI_REASONS.items()}


# ── the user's name ─────────────────────────────────────────────────────────────────────────
# USER 2026-11-14: "debe preguntarnos primero el nombre, y hablarnos usando nuestro nombre".
# Deterministic: a turn in which STAC only ACTED is confirmed by name ("Listo, Ana."), the first
# answer of a conversation starts with the name, and a sentence that already carries it is left
# as it is (the conversation's replies are written by the model, which is told the name).
def ask_name(lang: str) -> str:
    """STAC's first question to someone whose name it does not know."""
    return _t(lang if lang in LANGS else "es", "Hola, soy STAC. ¿Cómo te llamas?",
              "Hello, I am STAC. What is your name?", "Bonjour, je suis STAC. Comment vous appelez-vous ?")


def ask_name_again(lang: str) -> str:
    """… after doing what was asked by someone who has not said their name yet."""
    return _t(lang if lang in LANGS else "es", "¿Y cómo te llamas?", "And what is your name?",
              "Et comment vous appelez-vous ?")


def name_in_sentence(name: str, sentence: str) -> bool:
    """Is ``name`` something the user SAID? Every word of it must be a word of the sentence
    (case and accents apart) — the model once "extracted" the prompt's own example name from a
    sentence that gave none (2026-11-14)."""
    from commands.db import fold
    said = set(fold(sentence).split())
    words = fold(name).split()
    return bool(words) and all(w in said for w in words)


def done_prefix(name: Optional[str], lang: str) -> str:
    name = " ".join(str(name or "").split())
    if not name:
        return ""
    return _t(lang if lang in LANGS else "es", f"Listo, {name}.", f"Done, {name}.", f"C'est fait, {name}.")


def _lower_first(text: str) -> str:
    """The sentence's first letter in lower case when it starts with an ordinary capitalised
    word ("La holgura" → "la holgura"); an object reference, an acronym or a digit stays."""
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def address(text: str, name: Optional[str], lang: str, *, acted: bool, first: bool) -> str:
    """``text`` addressed to the user by ``name`` (see above). ``acted``: every call of the turn
    was an action and none failed. ``first``: the conversation's first answer."""
    name = " ".join(str(name or "").split())
    if not name or not text or name.lower() in text.lower():
        return text
    if acted:
        return f"{done_prefix(name, lang)} {text}"
    if first:
        return f"{name}, {_lower_first(text)}"
    return text
