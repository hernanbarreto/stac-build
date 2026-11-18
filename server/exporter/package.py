"""One export of a scan: the chosen items, a README that says what each is, and the ZIP.

    python -m exporter.package --session <id> --scan <date/source> --items e57,las,ply,glb,ifc,csv
                               [--confidence 0.25] [--lang es|en|fr]

The worker's entry point (exporter_api launches it in its own process): it prints ``PROGRESS <n>
<message>`` lines, then ``RESULT <json>`` — or ``ERROR <why>``.

  e57 · las · ply   the published cloud (exporter/cloud.py), filtered by the viewer's confidence
  glb               every published mesh of the scan (its textured binary glTF, as it is on disk)
  ifc               the GenRecon mesh cut by the segmentation: one element per object, with its
                    real shape (bim/ifc_from_mesh.py; config.yaml export.ifc)
  csv               every segmented object: id, label, points, where it is and how large

EXPORTED AS IT IS ON DISK (USER 2026-11-17: "correcto, tal como lo indicas"): the geometry of the
scan's current epoch. A calibration that was measured and not applied is NOT in the files — the
README says the epoch, the state of the calibration and its factor.

The ZIP is kept in the project's ``exports/`` directory, ONE per scan: a new export of a scan
replaces its previous one ("que no almacene al pedo").
"""
from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import cloud as C

ITEMS = ("e57", "las", "ply", "glb", "ifc", "csv")
CLOUD_ITEMS = ("e57", "las", "ply")
LANGS = ("es", "en", "fr")
# already packed: stored as they are; the rest is deflated fast (level 1)
_STORED = (".glb", ".e57")


class ExportError(ValueError):
    pass


def _server_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def context(session: str, scan_key: Optional[str]):
    from project_paths import resolve_session
    ctx = resolve_session(str(_server_dir()), session, scan_key) if scan_key else resolve_session(str(_server_dir()), session)
    if not Path(ctx.output_dir).is_dir():
        raise ExportError(f"{session} {scan_key or ''}: the scan has no output")
    return ctx


def slug(session: str, scan_key: Optional[str]) -> str:
    part = (scan_key or "scan").replace("/default", "").replace("/", "_")
    return f"{session}_{part}"


def zip_path(ctx, session: str, scan_key: Optional[str]) -> Path:
    return Path(ctx.project_dir) / "exports" / f"{slug(session, scan_key)}_stac_export.zip"


def meshes(output_dir: Path) -> List[Dict[str, Any]]:
    """The scan's published meshes: each folder of ``tsdf/`` with its binary glTF."""
    out = []
    root = Path(output_dir) / "tsdf"
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        meta_file = next(iter(sorted(folder.glob("*.meta.json"))), None)
        meta = json.loads(meta_file.read_text()) if meta_file else {}
        glb = folder / str(meta.get("glb_file") or f"{folder.name}.glb")
        if glb.exists():
            out.append({"folder": folder.name, "path": glb, "bytes": glb.stat().st_size, "faces": meta.get("n_faces"),
                        "textured": bool(meta.get("textured")), "provenance": str(meta.get("provenance") or "")})
    return out


def plan(session: str, scan_key: Optional[str]) -> Dict[str, Any]:
    """What this scan can export and how large each item is at most (every point of the cloud:
    a confidence filter only makes the cloud's files smaller)."""
    ctx = context(session, scan_key)
    out = Path(ctx.output_dir)
    items: Dict[str, Dict[str, Any]] = {}
    cloud_file = out / "cleaned_cloud.ply"
    n = 0
    if cloud_file.exists():
        data = C.open_ply(cloud_file)
        n = int(len(data))
        extra = sum(data.dtype[k].itemsize for k in data.dtype.names if k not in ("x", "y", "z", "red", "green", "blue"))
        items["ply"] = {"available": True, "bytes": n * data.dtype.itemsize}
        items["las"] = {"available": True, "bytes": n * (36 + extra)}       # LAS 1.4 format 7: 36 bytes + the extra dimensions
        items["e57"] = {"available": True, "bytes": n * 27}                 # 3 × float64 + 3 × uint8, before E57's packing
        del data
    else:
        for k in CLOUD_ITEMS:
            items[k] = {"available": False, "why": "no published cloud"}
    found = meshes(out)
    items["glb"] = ({"available": True, "bytes": sum(m["bytes"] for m in found), "meshes": [m["folder"] for m in found]}
                    if found else {"available": False, "why": "no published mesh"})
    store = out / "scene_r.db"
    if store.exists():
        con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
        try:
            objects = int(con.execute("SELECT COUNT(*) FROM instance_obb WHERE window_id = 'global'").fetchone()[0])
        finally:
            con.close()
        items["csv"] = {"available": objects > 0, "objects": objects, **({} if objects else {"why": "no segmented object"})}
        # THE IFC IS THE GENRECON MESH CUT BY THE SEGMENTATION (USER 2026-11-17): it needs both
        why = ("no segmented object" if not objects else "no published GenRecon mesh" if not found
               else "no per-point instance ids" if not (out / "instance_ids.npy").exists() else None)
        items["ifc"] = {"available": why is None, "elements": objects, **({} if why is None else {"why": why})}
    else:
        items["csv"] = {"available": False, "why": "the scan is not segmented"}
        items["ifc"] = {"available": False, "why": "the scan is not segmented"}
    previous = zip_path(ctx, session, scan_key)
    return {"session": session, "scan": scan_key, "points": n, "items": items, "state": state_of(out),
            "previous": ({"file": previous.name, "bytes": previous.stat().st_size,
                          "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(previous.stat().st_mtime))}
                         if previous.exists() else None)}


def state_of(output_dir: Path) -> Dict[str, Any]:
    """The geometry epoch on disk and what its calibration says (measured — applied or not)."""
    import calibration3d as cal
    from correction.epoch import current_epoch
    epoch = int(current_epoch(output_dir))
    try:
        s = cal.summary(cal.load(output_dir), epoch)
        calib = {"state": s.get("state"), "scale_factor": s.get("scale_factor"), "mode": s.get("mode_used"),
                 "n_gauge": s.get("n_gauge"), "n_test": s.get("n_test")}
    except cal.CalibrationError as e:
        calib = {"state": "unknown", "error": str(e)}
    return {"geometry_epoch": epoch, "calibration": calib}


def write_objects_csv(store: Path, out: Path) -> int:
    """Every segmented object's stored box, in the export frame (Z up, metres)."""
    con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT i.instance_id, i.label, o.obb_transform, o.obb_aabb, o.n_points FROM instances i "
            "JOIN instance_obb o ON o.instance_id = i.instance_id AND o.window_id = 'global' ORDER BY i.instance_id").fetchall()
    finally:
        con.close()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "label", "points", "center_x_m", "center_y_m", "center_z_m",
                    "size_a_m", "size_b_m", "height_m", "yaw_deg"])
        for iid, label, t_blob, a_blob, n_points in rows:
            T = np.frombuffer(t_blob, np.float32).reshape(4, 4).astype(np.float64)
            a = np.frombuffer(a_blob, np.float32).reshape(6).astype(np.float64)
            lo, hi = a[0::2], a[1::2]
            centre = C.Z_UP @ (T[:3, :3] @ ((lo + hi) / 2) + T[:3, 3])
            axis = C.Z_UP @ T[:3, 0]                       # the box's first horizontal axis, in the export frame
            size = hi - lo                                 # (a, height, b) in the box's own axes
            w.writerow([int(iid), label, int(n_points), *(f"{v:.4f}" for v in centre),
                        f"{size[0]:.4f}", f"{size[2]:.4f}", f"{size[1]:.4f}",
                        f"{np.degrees(np.arctan2(axis[1], axis[0])):.2f}"])
    return len(rows)


# ── the README ──────────────────────────────────────────────────────────────────────────────
_README_NAME = {"es": "LEEME.txt", "en": "README.txt", "fr": "LISEZMOI.txt"}
_T = {
    "title": {"es": "STAC Build — exportación del escaneo {scan}", "en": "STAC Build — export of the scan {scan}",
              "fr": "STAC Build — export du scan {scan}"},
    "made": {"es": "Generado el {when}.", "en": "Generated on {when}.", "fr": "Généré le {when}."},
    "frame": {
        "es": "MARCO. Todos los archivos están en el mismo marco: metros, eje Z hacia arriba, el piso en z = 0. "
              "El GLB usa el eje Y hacia arriba, como pide glTF: los programas que lo leen lo giran con la misma regla "
              "y la malla queda sobre la nube.",
        "en": "FRAME. Every file is in the same frame: metres, the Z axis up, the floor at z = 0. The GLB has its "
              "Y axis up, as glTF asks: the programs that read it turn it by the same rule and the mesh lands on the cloud.",
        "fr": "REPÈRE. Tous les fichiers sont dans le même repère : mètres, axe Z vers le haut, le sol à z = 0. Le GLB a "
              "son axe Y vers le haut, comme le demande glTF : les logiciels qui le lisent le tournent selon la même "
              "règle et le maillage se pose sur le nuage."},
    "epoch": {"es": "GEOMETRÍA. La de la época {epoch} del escaneo, tal como está en disco: nada se recalcula al exportar.",
              "en": "GEOMETRY. That of the scan's epoch {epoch}, as it is on disk: nothing is recomputed by the export.",
              "fr": "GÉOMÉTRIE. Celle de l'époque {epoch} du scan, telle qu'elle est sur le disque : rien n'est recalculé à l'export."},
    "uncalibrated": {
        "es": "ESCALA: SIN CALIBRAR. El escaneo no tiene mediciones de calibración sobre esta geometría: las longitudes son las de la reconstrucción.",
        "en": "SCALE: UNCALIBRATED. The scan has no calibration measurement on this geometry: lengths are the reconstruction's own.",
        "fr": "ÉCHELLE : NON CALIBRÉE. Le scan n'a aucune mesure de calibration sur cette géométrie : les longueurs sont celles de la reconstruction."},
    "pending": {
        "es": "ESCALA: CALIBRACIÓN MEDIDA Y NO APLICADA. El estado es «{state}» y el factor medido es {factor}: NO está en estos archivos. "
              "Una longitud real = la longitud de estos archivos × {factor}.",
        "en": "SCALE: CALIBRATION MEASURED, NOT APPLIED. Its state is “{state}” and the measured factor is {factor}: it is NOT in these files. "
              "A real length = the length in these files × {factor}.",
        "fr": "ÉCHELLE : CALIBRATION MESURÉE, NON APPLIQUÉE. Son état est « {state} » et le facteur mesuré est {factor} : il N'EST PAS dans ces fichiers. "
              "Une longueur réelle = la longueur dans ces fichiers × {factor}."},
    "applied": {
        "es": "ESCALA: CALIBRADA. El estado es «{state}» y el factor sobre esta geometría es 1: la calibración ya está aplicada.",
        "en": "SCALE: CALIBRATED. Its state is “{state}” and the factor on this geometry is 1: the calibration is applied.",
        "fr": "ÉCHELLE : CALIBRÉE. Son état est « {state} » et le facteur sur cette géométrie est 1 : la calibration est appliquée."},
    "axes": {
        "es": "ESCALA: calibración por ejes (un diagnóstico; no se aplica a la geometría). Estado «{state}».",
        "en": "SCALE: a per-axis calibration (a diagnosis; it is not applied to the geometry). State “{state}”.",
        "fr": "ÉCHELLE : calibration par axes (un diagnostic ; elle n'est pas appliquée à la géométrie). État « {state} »."},
    "cloud": {
        "es": "NUBE. {points} puntos de los {total} publicados{filter}. Cada punto es el del disco, medido por la reconstrucción.",
        "en": "CLOUD. {points} points of the {total} published{filter}. Each point is the one on disk, measured by the reconstruction.",
        "fr": "NUAGE. {points} points sur les {total} publiés{filter}. Chaque point est celui du disque, mesuré par la reconstruction."},
    "filter": {
        "es": " (filtro de confianza {conf}, el del visor: la confianza normalizada por [{lo}, {hi}] de {src})",
        "en": " (confidence filter {conf}, the viewer's: the confidence normalised by [{lo}, {hi}] of {src})",
        "fr": " (filtre de confiance {conf}, celui de la vue : la confiance normalisée par [{lo}, {hi}] de {src})"},
    "e57": {"es": "{file} — E57 (ASTM E2807): posición y color. ReCap / Revit, CloudCompare, Navisworks, Cyclone.",
            "en": "{file} — E57 (ASTM E2807): position and colour. ReCap / Revit, CloudCompare, Navisworks, Cyclone.",
            "fr": "{file} — E57 (ASTM E2807) : position et couleur. ReCap / Revit, CloudCompare, Navisworks, Cyclone."},
    "las": {
        "es": "{file} — LAS 1.4, formato de punto 7 (color de 16 bits), coordenadas con paso de 0,01 mm. Dimensiones extra: "
              "confidence, frame_global, pixel_row, pixel_col (el cuadro y el píxel de origen de cada punto). CloudCompare, QGIS, Global Mapper.",
        "en": "{file} — LAS 1.4, point format 7 (16-bit colour), coordinates in steps of 0.01 mm. Extra dimensions: confidence, "
              "frame_global, pixel_row, pixel_col (the frame and the pixel each point comes from). CloudCompare, QGIS, Global Mapper.",
        "fr": "{file} — LAS 1.4, format de point 7 (couleur 16 bits), coordonnées au pas de 0,01 mm. Dimensions supplémentaires : "
              "confidence, frame_global, pixel_row, pixel_col (l'image et le pixel d'origine de chaque point). CloudCompare, QGIS, Global Mapper."},
    "ply": {"es": "{file} — PLY binario con todas las propiedades del punto (color, confianza, cuadro y píxel de origen). MeshLab, Blender, CloudCompare.",
            "en": "{file} — binary PLY with every property of the point (colour, confidence, the frame and pixel it comes from). MeshLab, Blender, CloudCompare.",
            "fr": "{file} — PLY binaire avec toutes les propriétés du point (couleur, confiance, image et pixel d'origine). MeshLab, Blender, CloudCompare."},
    "glb": {
        "es": "{file} — malla con textura (glTF binario), {faces} caras. Procedencia: {prov}. Blender, visores web, Unreal / Unity.",
        "en": "{file} — textured mesh (binary glTF), {faces} faces. Provenance: {prov}. Blender, web viewers, Unreal / Unity.",
        "fr": "{file} — maillage texturé (glTF binaire), {faces} faces. Provenance : {prov}. Blender, visionneuses web, Unreal / Unity."},
    "ifc": {
        "es": "{file} — IFC4: la malla GenRecon cortada por la segmentación, {n} elementos, cada uno la parte de la malla de su objeto, "
              "con su clase (muro, losa, puerta, viga, ducto…; el resto como IfcBuildingElementProxy) y su id. Se conserva el {kept} % de los "
              "triángulos de la malla; el 95 % de la superficie queda a menos de {p95} mm de la malla completa. Revit, Navisworks, BIMcollab, Solibri.",
        "en": "{file} — IFC4: the GenRecon mesh cut by the segmentation, {n} elements, each one its object's part of the mesh, with its "
              "class (wall, slab, door, beam, duct…; the rest as IfcBuildingElementProxy) and its id. {kept} % of the mesh's triangles are "
              "kept; 95 % of the surface lies within {p95} mm of the full mesh. Revit, Navisworks, BIMcollab, Solibri.",
        "fr": "{file} — IFC4 : le maillage GenRecon découpé par la segmentation, {n} éléments, chacun la partie du maillage de son objet, "
              "avec sa classe (mur, dalle, porte, poutre, gaine… ; le reste en IfcBuildingElementProxy) et son id. {kept} % des triangles du "
              "maillage sont conservés ; 95 % de la surface est à moins de {p95} mm du maillage complet. Revit, Navisworks, BIMcollab, Solibri."},
    "csv": {
        "es": "{file} — los {n} objetos segmentados: id, etiqueta (propuesta por el VLM), puntos, centro, tamaño de su caja (a, b, altura) y giro. Excel.",
        "en": "{file} — the {n} segmented objects: id, label (proposed by the VLM), points, centre, the size of its box (a, b, height) and yaw. Excel.",
        "fr": "{file} — les {n} objets segmentés : id, étiquette (proposée par le VLM), points, centre, taille de sa boîte (a, b, hauteur) et lacet. Excel."},
}


def _t(key: str, lang: str, **values: Any) -> str:
    return _T[key][lang].format(**values)


def readme(lang: str, *, session: str, scan_key: Optional[str], state: Dict[str, Any], lines: List[str],
           when: str) -> str:
    cal = state["calibration"]
    factor = cal.get("scale_factor")
    if cal.get("mode") == "axes":
        scale = _t("axes", lang, state=cal.get("state"))
    elif cal.get("state") in (None, "uncalibrated", "unknown") or factor is None:
        scale = _t("uncalibrated", lang)
    elif float(factor) == 1.0:
        scale = _t("applied", lang, state=cal.get("state"))
    else:
        scale = _t("pending", lang, state=cal.get("state"), factor=f"{float(factor):.5f}")
    head = [_t("title", lang, scan=f"{session} {scan_key or ''}".strip()), _t("made", lang, when=when), "",
            _t("frame", lang), "", _t("epoch", lang, epoch=state["geometry_epoch"]), "", scale, ""]
    return "\n".join(head + lines) + "\n"


# ── the export ──────────────────────────────────────────────────────────────────────────────
def run(session: str, scan_key: Optional[str], items: List[str], *, confidence: float = 0.0, lang: str = "es",
        progress: Optional[Callable[[int, str], Any]] = None, now: Optional[float] = None) -> Dict[str, Any]:
    """Write the ZIP of ``items`` for the scan and return what it holds."""
    items = [k for k in ITEMS if k in items]                          # one order, whatever was asked
    if not items:
        raise ExportError(f"nothing to export: choose among {list(ITEMS)}")
    if lang not in LANGS:
        raise ExportError(f"language '{lang}' (one of {list(LANGS)})")
    say = progress or (lambda _p, _m: None)
    ctx = context(session, scan_key)
    out = Path(ctx.output_dir)
    available = plan(session, scan_key)["items"]
    missing = [f"{k}: {available[k].get('why')}" for k in items if not available[k]["available"]]
    if missing:
        raise ExportError("cannot export " + "; ".join(missing))
    target = zip_path(ctx, session, scan_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    # the files are written on the local disk (export.scratch_dir) and enter the ZIP from there
    from reconstruction.grid_knn import config_section
    work = Path(str(config_section(("export",), ("scratch_dir",))["scratch_dir"])) / target.stem
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    need = sum(int(available[k].get("bytes") or 0) for k in items if k in CLOUD_ITEMS)
    free = shutil.disk_usage(work).free
    if need > free:
        raise ExportError(f"the export's cloud files need up to {need / 2**30:.1f} GB in {work.parent} "
                          f"(export.scratch_dir) and {free / 2**30:.1f} GB are free")
    name = slug(session, scan_key)
    files: List[Dict[str, Any]] = []                                  # {path, arc, key}
    lines: List[str] = []
    report: Dict[str, Any] = {"session": session, "scan": scan_key, "items": items, "confidence": float(confidence)}
    # the steps' share of the bar: the cloud's writers and the ZIP weigh the most
    steps = [k for k in items] + ["zip"]
    span = 100.0 / len(steps)

    def at(step: str, frac: float, message: str) -> None:
        say(int(min(99, (steps.index(step) + frac) * span)), message)

    try:
        cloud_items = [k for k in items if k in CLOUD_ITEMS]
        if cloud_items:
            at(cloud_items[0], 0.0, "reading the cloud")
            cloud = C.Cloud(out, confidence)
            st = cloud.stats()
            report.update(points=st["points"], points_published=st["total"], bounds_min=st["min"], bounds_max=st["max"])
            flt = ""
            if cloud.range is not None:
                flt = _t("filter", lang, conf=f"{confidence:.2f}", lo=f"{cloud.range[0]:g}", hi=f"{cloud.range[1]:g}", src=cloud.range[2])
            lines += [_t("cloud", lang, points=f"{st['points']:,}", total=f"{st['total']:,}", filter=flt), ""]
            writers = {"ply": C.write_ply, "las": C.write_las,
                       "e57": lambda c, p, progress=None: C.write_e57(c, p, name=name, progress=progress)}
            for k in cloud_items:
                path = work / f"{name}_cloud.{k}"
                writers[k](cloud, path, progress=lambda f, _k=k: at(_k, f, f"writing the cloud as {_k.upper()}"))
                files.append({"path": path, "arc": path.name, "key": k})
                lines.append(_t(k, lang, file=path.name))
            del cloud
        if "glb" in items:
            at("glb", 0.0, "the mesh")
            for m in meshes(out):
                arc = f"{name}_mesh_{m['folder']}.glb"
                files.append({"path": m["path"], "arc": arc, "key": "glb", "source": True})
                lines.append(_t("glb", lang, file=arc, faces=f"{int(m['faces']):,}" if m["faces"] else "?",
                                prov=m["provenance"] or "?"))
        store = out / "scene_r.db"
        if "ifc" in items:
            at("ifc", 0.0, "writing the IFC")
            from bim import ifc_from_mesh
            from reconstruction.grid_knn import config_section
            how = config_section(("export", "ifc"), ("keep", "neighbours", "max_edge_voxels"))
            voxel = ifc_from_mesh.voxel_of()
            elements, cut = ifc_from_mesh.elements_of(
                out, ifc_from_mesh.mesh_of(out), keep=float(how["keep"]), max_distance=voxel,
                neighbours=int(how["neighbours"]), max_edge=float(how["max_edge_voxels"]) * voxel or None,
                measure=True, log=lambda m: at("ifc", 0.5, m))
            path = work / f"{name}_elements.ifc"
            ifc_from_mesh.build(elements, session=session, scan_key=scan_key or "",
                                geometry_epoch=str(state_of(out)["geometry_epoch"]),
                                described=f"STAC Build — the GenRecon mesh of {session} {scan_key or ''} cut by its segmentation"
                                ).write(str(path))
            files.append({"path": path, "arc": path.name, "key": "ifc"})
            dev = cut.get("deviation_m")
            lines.append(_t("ifc", lang, file=path.name, n=len(elements), kept=f"{100.0 * cut['faces'] / cut['mesh_faces']:.0f}",
                            p95=f"{dev['p95'] * 1000:.2f}" if dev else "0"))
            report["ifc_elements"] = len(elements)
            report["ifc"] = {k: cut[k] for k in ("mesh_faces", "faces", "faces_written", "faces_no_object", "keep") if k in cut}
            if dev:
                report["ifc"]["deviation_m"] = dev
        if "csv" in items:
            at("csv", 0.0, "writing the objects' table")
            path = work / f"{name}_objects.csv"
            n = write_objects_csv(store, path)
            files.append({"path": path, "arc": path.name, "key": "csv"})
            lines.append(_t("csv", lang, file=path.name, n=n))
            report["objects"] = n
        state = state_of(out)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(now if now is not None else time.time()))
        text = readme(lang, session=session, scan_key=scan_key, state=state, lines=lines, when=when)
        tmp = target.with_name(f".{target.name}.tmp")
        total = sum(f["path"].stat().st_size for f in files) or 1
        done = 0
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
            z.writestr(f"{name}/{_README_NAME[lang]}", text, compress_type=zipfile.ZIP_DEFLATED)
            for f in files:
                at("zip", done / total, f"packing {f['arc']}")
                stored = f["path"].suffix.lower() in _STORED
                z.write(f["path"], f"{name}/{f['arc']}", compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                        compresslevel=None if stored else 1)
                done += f["path"].stat().st_size
                if not f.get("source"):
                    f["path"].unlink()                                # (a written file is in the ZIP now)
        tmp.replace(target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report.update(file=target.name, path=str(target), bytes=target.stat().st_size, state=state,
                  files=[{"name": f["arc"], "item": f["key"]} for f in files], readme=_README_NAME[lang])
    say(100, "done")
    return report


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="exporter.package")
    ap.add_argument("--session", required=True)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--items", required=True)
    ap.add_argument("--confidence", type=float, default=0.0)
    ap.add_argument("--lang", default="es")
    a = ap.parse_args(argv)
    try:
        report = run(a.session, a.scan, [k.strip() for k in a.items.split(",") if k.strip()],
                     confidence=a.confidence, lang=a.lang,
                     progress=lambda p, m: print(f"PROGRESS {p} {m}", flush=True))
    except (ExportError, C.CloudError, FileNotFoundError, ValueError) as e:
        print(f"ERROR {e}", flush=True)
        return 1
    print("RESULT " + json.dumps(report, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
