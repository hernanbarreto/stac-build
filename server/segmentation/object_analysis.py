"""
Per-object DEEP ANALYSIS — always available (USER 2026-09-04: "se segmentó
algo → el chat debe tener el detalle SIEMPRE; no puede estar sujeto a que
corra el conversor a CAD").

One reusable entry, three callers:
  * end of every segmentation pipeline (auto, background, after the chat
    vLLM reloads) — fills the dossier of every instance;
  * the chat's ``describe_object`` tool — generates on the spot if missing;
  * the shape proposer (CAD pass) — reuses/refreshes with richer evidence.

The dossier (Spanish: qué es, descripción, características, materiales,
estado, función, interacción con el entorno) is ``vlm_proposed`` and cached
in scene_r.db meta ``object_analysis_<iid>``; evidence crops are kept under
``output/object_analysis/<safe>/`` for the user's eyes. Measurements in the
prompt are tool_measured context — the VLM never measures.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


def _store(output_dir: Path):
    from phase_r.instance_store import InstanceStore
    db = Path(output_dir) / "scene_r.db"
    return InstanceStore(str(db)) if db.exists() else None


def get_cached_analysis(output_dir: Path, instance_id: int) -> Optional[dict]:
    st = _store(output_dir)
    if st is None:
        return None
    raw = st.get_meta(f"object_analysis_{int(instance_id)}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def analyze_object(output_dir: Path, instance_id: int,
                   evidence_images: Optional[List[str]] = None,
                   extent_m: Optional[np.ndarray] = None,
                   identity_hint: Optional[str] = None,
                   sym_line: str = "",
                   refresh: bool = False,
                   log=print) -> Optional[dict]:
    """Detailed Spanish dossier of one segmented object. Cached; set
    ``refresh`` to regenerate. Builds its own evidence (context frame with a
    red box + isolated mask crops) unless ``evidence_images`` is given."""
    out = Path(output_dir)
    if not refresh:
        cached = get_cached_analysis(out, instance_id)
        if cached:
            return cached

    from PIL import Image, ImageDraw
    from segmentation.tsdf_export import _safe_label
    from segmentation.shape_proposer import (_ANALYSIS_SCHEMA, _chat_json,
                                             _calibrate_oid_lenient,
                                             _isolated_crop, _load_frame_rgb)
    from semantic.client import get_semantic_client
    from semantic.types import system as sys_msg, user as user_msg

    t0 = time.time()
    result = json.loads((out / "segmentation_result.json").read_text())
    inst = next((i for i in result.get("instances", [])
                 if int(i.get("instance_id", i.get("id"))) == int(instance_id)),
                None)
    if inst is None:
        raise ValueError(f"instance {instance_id} not found")
    label = str(inst.get("label", "segment"))
    safe = _safe_label(label, int(instance_id))
    session_dir = out.parent

    images: List[str] = list(evidence_images or [])
    if not images:
        # evidence from the SAM masks: context frame (red box) + isolated crops
        from reconstruction.surface_fit.hole_audit import _evidence
        ev = _evidence(out, session_dir)
        if not ev.ok:
            raise RuntimeError(f"{safe}: no mask/camera evidence")
        if extent_m is None:
            st = _store(out)
            P = st.get_points(int(instance_id)) if st else None
            if P is not None and len(P):
                extent_m = np.asarray(P, np.float64).ptp(axis=0)
        # oid via own points (raw cloud) — lenient mask-hit calibration
        import open3d as o3d
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        xyz = np.asarray(o3d.io.read_point_cloud(
            str(out / "cleaned_cloud.ply")).points)
        gi = gi[(gi >= 0) & (gi < len(xyz))]
        oid = _calibrate_oid_lenient(ev, xyz[gi], int(instance_id), log=log)
        if oid is None:
            raise RuntimeError(f"{safe}: could not match a SAM mask")
        frames = sorted(ev.frames_for(oid),
                        key=lambda fk: int((ev.masks[fk[1]] > 0).sum()),
                        reverse=True)
        dst = out / "object_analysis" / safe
        dst.mkdir(parents=True, exist_ok=True)
        for n, (fidx, key) in enumerate(frames[:3]):
            img = _load_frame_rgb(session_dir, fidx)
            if img is None:
                continue
            m = ev.masks[key] > 0
            mh, mw = m.shape
            W, H = img.size
            if n == 0:   # context frame: red box, full surroundings
                ys, xs = np.where(m)
                ctx = img.copy()
                dr = ImageDraw.Draw(ctx)
                dr.rectangle([int(xs.min() * W / mw), int(ys.min() * H / mh),
                              int(xs.max() * W / mw), int(ys.max() * H / mh)],
                             outline=(255, 40, 40), width=6)
                s = min(1.0, 1280 / max(ctx.size))
                if s < 1.0:
                    ctx = ctx.resize((int(ctx.width * s), int(ctx.height * s)))
                p = dst / "context.jpg"
                ctx.save(p, quality=88)
                images.append(str(p))
            mrgb = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                              .resize(img.size, Image.NEAREST)) > 0
            crop = _isolated_crop(img, mrgb)
            p = dst / f"iso_f{fidx}.jpg"
            crop.save(p, quality=88)
            images.append(str(p))
    if not images:
        raise RuntimeError(f"{safe}: no evidence images")

    ext_line = ""
    if extent_m is not None:
        e = np.asarray(extent_m, float)
        ext_line = (f"Tool-measured extent: {e[0]:.2f} × {e[1]:.2f} × "
                    f"{e[2]:.2f} m (width × height × depth). ")
    client = get_semantic_client(consumer="object_analysis")
    msgs = [
        sys_msg("You are an expert surveyor/engineer analyzing one segmented "
                "object of a 3D-scanned scene in depth. You describe and "
                "classify; you never estimate dimensions — the measured ones "
                "are given to you."),
        user_msg(
            f"Object (segmentation label '{label}'"
            + (f", identified as '{identity_hint}'" if identity_hint else "")
            + "). The FIRST image shows it in context (red box); the rest "
            "show it isolated (background darkened) or as measured data. "
            f"{ext_line}{sym_line}\n"
            "Produce a DETAILED dossier in SPANISH. Reply ONLY with JSON: "
            "{\"que_es\": qué es exactamente, "
            "\"descripcion_detallada\": 4-6 frases (forma, componentes "
            "visibles, acabados, singularidades), "
            "\"caracteristicas\": lista de rasgos concretos observables, "
            "\"materiales\": lista de materiales visibles, "
            "\"estado_aparente\": conservación/daños visibles, "
            "\"funcion\": para qué sirve, "
            "\"interaccion_con_entorno\": cómo se relaciona/apoya/conecta "
            "con lo que lo rodea en la escena, "
            "\"confidence\": 0-1}",
            images=images[:6]),
    ]
    parsed, _raw = _chat_json(client, msgs, _ANALYSIS_SCHEMA, 900, log=log)
    if not isinstance(parsed, dict):
        return None
    parsed["provenance"] = "vlm_proposed"
    parsed["generated"] = time.strftime("%Y-%m-%d %H:%M")
    st = _store(out)
    if st is not None:
        st.set_meta(f"object_analysis_{int(instance_id)}",
                    json.dumps(parsed, ensure_ascii=False))
    log(f"[analysis:{safe}] {str(parsed.get('que_es'))[:60]} "
        f"({round(time.time() - t0, 1)}s)")
    return parsed


def ensure_session_analyses(output_dir: Path, objects: bool = True,
                            log=print) -> dict:
    """Fill what the chat must ALWAYS know (user 2026-09-04), respecting the
    session's real state: the scene description always; the per-instance
    dossiers only when ``objects`` — i.e. AFTER segmentation has produced
    instances (at pipeline end there are none: only scene + cloud). Skips
    everything already cached; call only while the semantic service is up."""
    out = Path(output_dir)
    done = skipped = failed = 0
    res_path = out / "segmentation_result.json"
    instances = []
    if objects and res_path.exists():
        try:
            instances = json.loads(res_path.read_text()).get("instances", [])
        except Exception:  # noqa: BLE001
            instances = []
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id", 0)) or 0)
        if not iid:
            continue
        if get_cached_analysis(out, iid):
            skipped += 1
            continue
        try:
            if analyze_object(out, iid, log=log):
                done += 1
            else:
                failed += 1
        except Exception as e:  # noqa: BLE001 — one object never sinks the rest
            failed += 1
            log(f"[analysis] instance {iid} failed: {e}")
    # scene-level fallback/context: always make sure a description exists
    try:
        st = _store(out)
        if st is not None and not st.get_meta("scene_description"):
            from phase5_qa.tools import SpatialTools
            SpatialTools(st).describe_scene()
            log("[analysis] scene description generated")
    except Exception as e:  # noqa: BLE001
        log(f"[analysis] scene description failed: {e}")
    log(f"[analysis] session dossiers: {done} generated, {skipped} cached, "
        f"{failed} failed")
    return {"generated": done, "cached": skipped, "failed": failed}
