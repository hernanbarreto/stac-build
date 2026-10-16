"""The per-session runner of the forms stage: query → fit → correct → review → mesh → publish.

One object at a time (the largest first): its form record (asked when missing), its template on
its own cloud points at the session's σ, the corrector against every view that masks it, the
alternatives and the review rounds only in doubt, the mesh (GLB) and its full record into
``output/forms/<safe>/``, the published copy into ``output/tsdf/<safe>/`` (the viewer's list,
``method: "form"``), textured from the scan frames. A freeform object, or one without a usable
form, is meshed AS MEASURED (the live-cloud TSDF of its points) and says so. Every loader is
injectable (the tests run the orchestration with synthetic views and a faked VLM).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .catalog import FormAnswer, FormsConfig
from .composite import build_composite
from .fit import fit_cloud, residual_report
from .query import form_of_record, load_record, query_session_forms
from .review import apply_review, ask_review, overlays_for, review_prompt, verdict_record
from .hull import hull_mesh
from .silhouette import Silhouettes, compare_forms, correct, per_view_chamfer, views_from_evidence, worst_views
from .templates import Frame, Template, build_template

DIR = "forms"
METHOD = "form"
METHOD_TSDF = "form_tsdf"
METHOD_HULL = "hull"


# ── the scene frame ────────────────────────────────────────────────────────────

def session_up(output_dir: Path) -> np.ndarray:
    """The session's up in the cloud's frame: the floor levelling's (floor_transform.npz:
    display = s·R·raw + t with the floor at y = 0 → raw up = Rᵀ·ŷ); Y-up without it."""
    p = Path(output_dir) / "floor_transform.npz"
    if not p.exists():
        return np.array([0.0, 1.0, 0.0])
    d = np.load(p)
    R = np.asarray(d["R"], np.float64)
    up = R.T @ np.array([0.0, 1.0, 0.0])
    return up / max(float(np.linalg.norm(up)), 1e-12)


def scene_frame(output_dir: Path, points_of: Dict[int, np.ndarray], forms: Dict[int, FormAnswer],
                sigma: float, huber_k: float, ransac_iters: int, seed: int,
                log: Callable[[str], None] = lambda m: None) -> Frame:
    """up from the floor levelling; the horizontal X = the dominant direction of the VERTICAL
    PANELS the VLM named (the circular mean of their normals' azimuths folded modulo 90°,
    weighted by points) — the room's walls; without any, the display frame's X."""
    from .composite import _extract_plane
    up = session_up(output_dir)
    acc = 0.0 + 0.0j
    n_walls = 0
    e1 = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = e1 - up * float(e1 @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    for iid, ans in forms.items():
        if ans.form != "panel" or ans.orientation != "vertical" or iid not in points_of:
            continue
        P = points_of[iid]
        if len(P) < 100:
            continue
        sub = P[:: max(1, len(P) // 50000)]
        inl = _extract_plane(sub, huber_k * sigma, ransac_iters, seed)
        if inl is None or int(inl.sum()) < 50:
            continue
        q = sub[inl] - sub[inl].mean(0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        nrm = vt[2]
        nrm = nrm - up * float(nrm @ up)
        if np.linalg.norm(nrm) < 1e-6:
            continue
        nrm /= np.linalg.norm(nrm)
        az = np.arctan2(float(nrm @ e2), float(nrm @ e1))
        acc += int(inl.sum()) * np.exp(1j * 4.0 * az)        # folded modulo 90°
        n_walls += 1
    if n_walls and abs(acc) > 0:
        az = np.angle(acc) / 4.0
        X = np.cos(az) * e1 + np.sin(az) * e2
        log(f"[forms] scene frame: up from the floor levelling, X from {n_walls} vertical panel(s)")
        return Frame(up, X)
    p = Path(output_dir) / "floor_transform.npz"
    if p.exists():
        R = np.asarray(np.load(p)["R"], np.float64)
        X = R.T @ np.array([1.0, 0.0, 0.0])
        log("[forms] scene frame: up from the floor levelling, X from the display frame (no vertical panel named)")
        return Frame(up, X)
    log("[forms] scene frame: no floor levelling — Y-up, X along x (declared)")
    return Frame(up, None)


# ── one object ─────────────────────────────────────────────────────────────────

def _fit_params(fcfg: FormsConfig, huber_k: float, dev) -> dict:
    return dict(lr=fcfg.lr, block_iterations=fcfg.block_iterations, max_blocks=fcfg.max_blocks,
                huber_k=huber_k, max_points=fcfg.max_points, seed=fcfg.seed, dev=dev)


def build_for(ans: FormAnswer, P: np.ndarray, frame: Frame, sigma: float, fcfg: FormsConfig,
              huber_k: float, error_factor: float, dev, log) -> tuple:
    """(template, build record) of ``ans`` on ``P`` — a composite or a single template."""
    if ans.form == "composite":
        comp, rec = build_composite(ans, P, frame, sigma, dev, lr=fcfg.lr,
                                    block_iterations=fcfg.block_iterations, max_blocks=fcfg.max_blocks,
                                    huber_k=huber_k, max_points=fcfg.max_points, seed=fcfg.seed,
                                    ransac_iters=fcfg.ransac_iters, error_factor=error_factor, log=log)
        return comp, rec
    return build_template(ans, P, frame, dev), {"init": "the object's points (PCA, extents)"}


def fit_candidate(ans: FormAnswer, P: np.ndarray, frame: Frame, sigma: float, fcfg: FormsConfig,
                  huber_k: float, error_factor: float, confidence: float, sil: Optional[Silhouettes],
                  dev, log) -> dict:
    """One candidate form, fitted to the points and corrected on the views: the template and
    its records."""
    t0 = time.time()
    tpl, build_rec = build_for(ans, P, frame, sigma, fcfg, huber_k, error_factor, dev, log)
    cloud_rec = fit_cloud(tpl, P, sigma, **_fit_params(fcfg, huber_k, dev))
    out = {"answer": ans.to_dict(), "template": None, "build": build_rec, "cloud_fit": cloud_rec,
           "corrector": None, "residual": None, "seconds": None}
    if sil is not None and sil.views:
        out["corrector"] = correct(tpl, P, sigma, sil, lr=fcfg.lr_corrector, block_iterations=fcfg.block_iterations,
                                   max_blocks=fcfg.max_blocks, huber_k=huber_k, max_points=fcfg.max_points,
                                   surface_samples=fcfg.surface_samples, seed=fcfg.seed, dev=dev,
                                   confidence=confidence, log=log)
    out["residual"] = residual_report(tpl, P, sigma, dev)
    out["template"] = tpl.record()
    out["dims_m"] = tpl.dims()
    out["seconds"] = round(time.time() - t0, 2)
    out["_tpl"] = tpl
    return out


def _verdict(cand: dict) -> str:
    c = cand.get("corrector")
    return "no_views" if not c else str(c.get("verdict"))


def _pick(cands: List[dict], confidence: float, seed: int, log) -> int:
    """The candidate the views prefer — the first one unless another is CLEARLY better."""
    best = 0
    for i in range(1, len(cands)):
        a, b = cands[i].get("corrector"), cands[best].get("corrector")
        if not a or not b:
            continue
        c = compare_forms(per_view_chamfer(a), per_view_chamfer(b), confidence, seed)
        if c["a_better"]:
            best = i
    return best


def run_object(*, iid: int, label: str, safe: str, P: np.ndarray, ans: Optional[FormAnswer],
               frame: Frame, sigma: float, fcfg: FormsConfig, huber_k: float, error_factor: float,
               confidence: float, dev, views: list, client=None, catalog: Optional[List[str]] = None,
               frames_dir: Optional[Path] = None, to_native=None, batch_params: Optional[dict] = None,
               sil: Optional[Silhouettes] = None,
               log: Callable[[str], None] = lambda m: None) -> dict:
    """The whole chain of one object: candidates (the VLM's form, its alternatives in doubt), the
    review rounds in doubt, the chosen template. Returns the record with ``_tpl`` (the template to
    mesh) or ``_tpl`` None (as measured)."""
    rec: Dict[str, object] = {"instance_id": int(iid), "label": label, "n_points": int(len(P)),
                              "sigma_m": float(sigma), "n_views": len(views), "rounds": [],
                              "candidates": [], "chosen": None, "status": None, "_tpl": None}
    if ans is None or ans.form == "freeform":
        rec["status"] = "as_measured"
        rec["why"] = ("the VLM called it freeform" if ans is not None
                      else "no usable form answer (declared in the form record)")
        return rec
    if sil is None:
        sil = Silhouettes(views, sigma, huber_k, dev, **(batch_params or {})) if views else None
    cands: List[dict] = []
    catalog = catalog or list(fcfg.catalog)

    def add(a: FormAnswer, why: str):
        try:
            c = fit_candidate(a, P, frame, sigma, fcfg, huber_k, error_factor, confidence, sil, dev, log)
        except Exception as e:  # noqa: BLE001 — a candidate that cannot be built is declared
            log(f"[forms] {safe}: candidate {a.form} failed ({type(e).__name__}: {e})")
            cands.append({"answer": a.to_dict(), "why": why, "error": f"{type(e).__name__}: {e}",
                          "corrector": None, "_tpl": None})
            return
        c["why"] = why
        cands.append(c)
        log(f"[forms] {safe}: {a.form} ({why}) → {_verdict(c)}; residual median "
            f"{c['residual'].get('median_m', float('nan')) * 100:.2f} cm, within 2σ "
            f"{c['residual'].get('within_2sigma', 0) * 100:.0f} %")

    add(ans, "the VLM's form")
    # the alternatives, when the VLM was not sure or the views are in doubt
    if (ans.confidence == "low" or _verdict(cands[0]) == "doubt") and ans.alternatives:
        for alt in ans.alternatives[:2]:
            a2 = FormAnswer(**{**ans.__dict__, "form": alt, "parts": [], "alternatives": []})
            add(a2, "an alternative the VLM listed")

    def built():
        return [c for c in cands if c.get("_tpl") is not None]

    best_list = built() or cands
    best = _pick(best_list, confidence, fcfg.seed, log)
    best = cands.index(best_list[best])
    rounds = 1
    current = ans
    while (client is not None and frames_dir is not None and sil is not None and rounds < fcfg.max_rounds
           and _verdict(cands[best]) == "doubt" and cands[best].get("_tpl") is not None):
        rounds += 1
        c = cands[best]
        tpl = c["_tpl"]
        worst = worst_views(c["corrector"], fcfg.review_views)
        ovs = overlays_for(sil, tpl, worst, frames_dir, fcfg.surface_samples, to_native)
        if not ovs:
            rec["rounds"].append({"round": rounds, "kind": "review", "skipped": "no frame to draw on"})
            break
        cons = c["corrector"]
        miss_px = 0.0
        if cons.get("consensus_final_m") is not None and views:
            foots = [float(v.z_boundary.median()) / v.f_px for v in views if v.z_boundary.numel()]
            if foots:
                miss_px = float(cons["consensus_final_m"]) / max(float(np.median(foots)), 1e-9)
        prompt = review_prompt(current, label, len(ovs), c["residual"].get("within_2sigma", 0.0), miss_px)
        got = ask_review(client, prompt, [o["image"] for o in ovs], catalog, fcfg.max_tokens, log)
        rnd = {"round": rounds, "kind": "review", "views": [o["fidx"] for o in ovs],
               "verdict": verdict_record(got["verdict"]), "raw": got["raw"][:4000], "error": got["error"],
               "stamp": got["stamp"]}
        rec["rounds"].append(rnd)
        new = apply_review(current, got["verdict"])
        if new is None:
            rnd["applied"] = False
            matched = bool((got["verdict"] or {}).get("matches"))
            log(f"[forms] {safe}: review round {rounds}: {'matches' if matched else 'no change to apply'}")
            if matched and cands[best].get("corrector"):
                # the VLM confirms the drawn form on the views it doubted: the verdict says so
                cands[best]["corrector"]["verdict"] = "coincides_by_review"
            break
        rnd["applied"] = True
        current = new
        add(new, f"the review's correction (round {rounds})")
        best_list = built() or cands
        best = cands.index(best_list[_pick(best_list, confidence, fcfg.seed, log)])
    chosen = cands[best]
    rec["candidates"] = [{k: v for k, v in c.items() if k != "_tpl"} for c in cands]
    rec["chosen"] = best
    rec["_tpl"] = chosen.get("_tpl")
    rec["status"] = "form" if rec["_tpl"] is not None else "as_measured"
    if rec["_tpl"] is None:
        rec["why"] = "no candidate could be built — meshed as measured"
    rec["verdict"] = _verdict(chosen)
    rec["form"] = chosen["answer"]["form"]
    return rec


# ── meshing and publishing ─────────────────────────────────────────────────────

def mesh_as_measured(output_dir: Path, P: np.ndarray, frames: Sequence[int], raw_cfg: dict):
    """The live-cloud TSDF of the object's own points (never Poisson)."""
    from segmentation.tsdf_export import tsdf_mesh_of_points
    oc = dict((raw_cfg.get("tsdf") or {}).get("object") or {})
    if not oc:
        raise RuntimeError("tsdf.object is missing from the configuration")
    return tsdf_mesh_of_points(Path(output_dir), np.asarray(P, np.float64), list(frames), oc)


def publish(output_dir: Path, session_dir: Path, safe: str, V: np.ndarray, F: np.ndarray, meta: dict,
            texture: bool, log: Callable[[str], None] = lambda m: None) -> Path:
    """form.glb + the record under output/forms/<safe>/, the copy the viewer lists under
    output/tsdf/<safe>/ (``method`` form | form_tsdf), textured from the scan frames."""
    import shutil
    import trimesh
    fdir = Path(output_dir) / DIR / safe
    fdir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=np.asarray(V, np.float64), faces=np.asarray(F, np.int64), process=False)
    src = fdir / "form.glb"
    mesh.export(str(src))
    obj_dir = Path(output_dir) / "tsdf" / safe
    obj_dir.mkdir(parents=True, exist_ok=True)
    dst = obj_dir / f"{safe}.glb"
    shutil.copy2(src, dst)
    if texture:
        try:
            from reconstruction.texture_bake import bake_object_glb
            if bake_object_glb(dst, Path(session_dir), Path(output_dir)):
                meta = {**meta, "textured": True}
        except Exception as e:  # noqa: BLE001 — the mesh stays, untextured, declared
            log(f"[forms] {safe}: texture bake failed (mesh kept untextured): {type(e).__name__}: {e}")
            meta = {**meta, "texture_error": f"{type(e).__name__}: {e}"}
    meta = {**meta, "glb_file": dst.name, "source_dir": str(fdir)}
    (obj_dir / f"{safe}.meta.json").write_text(json.dumps(meta, indent=2))
    return dst


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ── the session ────────────────────────────────────────────────────────────────

def _instance_points(fields: dict, inst: dict, trim_pct: float):
    xyz = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(np.float64)
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    P = xyz[gi]
    conf = fields.get("confidence")
    if conf is not None and trim_pct > 0 and len(gi):
        c = np.asarray(conf, np.float64)[gi]
        keep = c >= float(np.percentile(c, trim_pct))
        P = P[keep]
    return P


def run_session(session_dir, *, instance_ids: Optional[List[int]], fcfg: FormsConfig, raw_cfg: dict,
                log: Callable[[str], None] = print,
                progress: Optional[Callable[[int, str], None]] = None,
                client=None, evidence=None, views_fn=None, points_fn=None, sigma: Optional[float] = None,
                texture: Optional[bool] = None, query: bool = True) -> dict:
    """Every requested instance (all when None), the largest first. Returns
    ``{ok: [...], failed: [...], skipped: [...], seconds}`` and writes output/forms/forms_run.json."""
    t_all = time.time()
    session_dir = Path(session_dir)
    out = session_dir / "output"
    frames_dir = session_dir / "frames"
    dev = fcfg.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("forms.device is cuda and no card is available — the stage does not fall back")
    instances = list(json.loads((out / "segmentation_result.json").read_text()).get("instances") or [])
    wanted = None if instance_ids is None else {int(i) for i in instance_ids}
    todo = [i for i in instances if wanted is None or int(i.get("instance_id", i.get("id"))) in wanted]
    todo.sort(key=lambda i: -int(i.get("total_points") or len(i.get("globalIndices") or [])))
    from segmentation.tsdf_export import _safe_label
    huber_k = float(raw_cfg["reconstruction"]["precision"]["gauge"]["huber_k"])
    error_factor = float(raw_cfg["correction_graph"]["graph"]["improvement_error_factor"])
    confidence = float(raw_cfg["correction_graph"]["graph"]["heldout_confidence"])
    if texture is None:
        texture = bool((raw_cfg.get("surface_fit") or {}).get("texture_objects", True))
    if sigma is None:
        from reconstruction.certify.repeatability import session_repeatability
        rep = session_repeatability(out, fallback_m=float(raw_cfg["correction"]["visit_drift"]["default_repeatability_m"]),
                                    log=log)
        sigma = float(rep["sigma_floor_m"])
    if query and todo:
        query_session_forms(out, session_dir, fcfg, instance_ids=[int(i.get("instance_id", i.get("id"))) for i in todo],
                            client=client, log=log)
    forms: Dict[int, Optional[FormAnswer]] = {}
    for inst in todo:
        iid = int(inst.get("instance_id", inst.get("id")))
        safe = _safe_label(str(inst.get("label") or "segment"), iid)
        forms[iid] = form_of_record(load_record(out, safe))
    if points_fn is None:
        from segmentation.perfect_object import _read_ply_fields
        fields = _read_ply_fields(out / "cleaned_cloud.ply")
        trim = float((raw_cfg.get("surface_fit") or {}).get("p2c_conf_trim_pct", 20.0))
        points_fn = lambda inst: _instance_points(fields, inst, trim)   # noqa: E731
    points_of = {int(i.get("instance_id", i.get("id"))): points_fn(i) for i in todo}
    if evidence is None and views_fn is None:
        from reconstruction.surface_fit.hole_audit import _evidence
        evidence = _evidence(out, session_dir)
        if not evidence.ok:
            log(f"[forms] no mask evidence ({evidence.reason}) — the corrector cannot run: forms fitted "
                f"on the points only (declared)")
            evidence = None
    oids_of: Dict[int, List[int]] = {}
    if evidence is not None or views_fn is not None:
        from segmentation.object_captioner import _oids_by_instance
        parent = json.loads((out / "segmentation.json").read_text()) if (out / "segmentation.json").exists() else {}
        res = json.loads((out / "segmentation_result.json").read_text())
        oids_of = _oids_by_instance(parent, res.get("absorbed") or {})
    to_native = None
    batch_params = None
    if evidence is not None:
        from .review import _mask_to_native
        from .silhouette import batch_params_from_evidence
        to_native = _mask_to_native(evidence.mask_grid)
        batch_params = batch_params_from_evidence(evidence)
    frame = scene_frame(out, points_of, {k: v for k, v in forms.items() if v is not None}, sigma, huber_k,
                        fcfg.ransac_iters, fcfg.seed, log)
    ok, failed, skipped = [], [], []
    for n, inst in enumerate(todo, 1):
        iid = int(inst.get("instance_id", inst.get("id")))
        label = str(inst.get("label") or "segment")
        safe = _safe_label(label, iid)
        t0 = time.time()
        if progress:
            progress(iid, "fit")
        try:
            P = points_of[iid]
            if len(P) < 10:
                skipped.append({"instance_id": iid, "label": label, "reason": f"{len(P)} points"})
                continue
            views = []
            if views_fn is not None:
                views = views_fn(iid, oids_of.get(iid, []))
            elif evidence is not None:
                views = views_from_evidence(evidence, oids_of.get(iid, []), sigma, fcfg.max_views,
                                            fcfg.boundary_px_per_view, fcfg.seed, log)
            sil = Silhouettes(views, sigma, huber_k, dev, **(batch_params or {})) if views else None
            hull_rec: Optional[dict] = None
            Vh = Fh = None
            if fcfg.mesh == "hull" and sil is not None:
                if progress:
                    progress(iid, "hull")
                oc = dict((raw_cfg.get("tsdf") or {}).get("object") or {})
                pc = raw_cfg["reconstruction"]["precision"]["cloud"]      # the silhouette filter's majority
                scene_t = evidence._cloud_tensor() if evidence is not None else None
                Vh, Fh, hull_rec = hull_mesh(P, sil.batches, voxel=float(oc["voxel_length"]),
                                             trunc=float(oc["sdf_trunc"]), fill_px=int(oc["fill_px"]),
                                             min_votes=int(pc["silhouette_min_votes"]),
                                             min_inside_frac=float(pc["silhouette_min_inside_frac"]),
                                             sigma=sigma, dev=dev, scene=scene_t, session_key=str(out), log=log)
                if len(Vh) == 0:
                    Vh = Fh = None
            rec = run_object(iid=iid, label=label, safe=safe, P=P, ans=forms.get(iid), frame=frame, sigma=sigma,
                             fcfg=fcfg, huber_k=huber_k, error_factor=error_factor, confidence=confidence,
                             dev=dev, views=views, client=client, catalog=list(fcfg.catalog),
                             frames_dir=frames_dir, to_native=to_native, batch_params=batch_params, sil=sil,
                             log=log)
            rec["hull"] = hull_rec
            tpl: Optional[Template] = rec.get("_tpl")
            if Vh is not None:
                # the measured envelope is what is published; the fitted form stays in the record
                V, F = Vh, Fh
                method = METHOD_HULL
            elif tpl is not None:
                if progress:
                    progress(iid, "mesh")
                V, F = tpl.mesh()
                method = METHOD
            else:
                if progress:
                    progress(iid, "tsdf")
                frames = sorted({v.cloud_frame for v in views}) if views else []
                if not frames and evidence is not None:
                    frames = sorted({evidence.cloud_frame(f) for oid in oids_of.get(iid, [])
                                     for f, _k in evidence.frames_for(oid) if evidence.cloud_frame(f) is not None})
                got = mesh_as_measured(out, P, frames, raw_cfg) if frames else None
                if got is None:
                    skipped.append({"instance_id": iid, "label": label,
                                    "reason": f"as measured but nothing integrated ({rec.get('why')})"})
                    log(f"[forms] {safe}: as measured — nothing integrated ({rec.get('why')})")
                    continue
                V, F = got
                method = METHOD_TSDF
            chosen = (rec["candidates"][rec["chosen"]] if rec.get("chosen") is not None else {})
            cons = chosen.get("corrector") or {}
            meta = {"method": method, "kind": rec.get("form") or "freeform", "instance_id": iid, "label": label,
                    "textured": False, "provenance": {"form": "vlm_proposed", "pose_and_dims": "tool_measured",
                                                      "mesh": ("the visual hull of the masks carved by the TSDF of the "
                                                               "object's points (tool_measured)" if method == METHOD_HULL
                                                               else "the fitted form")},
                    "hull": ({k: v for k, v in hull_rec.items() if k != "rounds"} if hull_rec else None),
                    "form": rec.get("form"), "verdict": rec.get("verdict"), "status": rec["status"],
                    "dims_m": chosen.get("dims_m"), "consensus_m": cons.get("consensus_final_m"),
                    "bar_m": cons.get("bar_m"), "iou": cons.get("iou_median"), "n_views": len(views),
                    "residual": chosen.get("residual"), "why": rec.get("why")}
            if progress:
                progress(iid, "publish")
            glb = publish(out, session_dir, safe, V, F, meta, texture, log)
            full = {**_json_safe(rec), "glb": str(glb), "method": method, "seconds": round(time.time() - t0, 1)}
            (out / DIR / safe / "form_fit.json").write_text(json.dumps(full, indent=1))
            ok.append({"instance_id": iid, "label": label, "dir": str(out / DIR / safe), "glb": str(glb),
                       "method": method, "form": rec.get("form"), "verdict": rec.get("verdict"),
                       "status": rec["status"]})
            log(f"[forms] {safe}: {method} ({rec.get('form') or 'as measured'}, {rec.get('verdict')}) in "
                f"{time.time() - t0:.1f} s ({n}/{len(todo)})")
            if progress:
                progress(iid, "done")
        except Exception as e:  # noqa: BLE001 — one object must not sink the session
            log(f"[forms] {safe}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
            failed.append({"instance_id": iid, "label": label, "reason": f"{type(e).__name__}: {e}"})
            if progress:
                progress(iid, "error")
    summary = {"version": 1, "n_instances": len(todo), "ok": ok, "failed": failed, "skipped": skipped,
               "sigma_m": sigma, "frame": frame.to_dict(), "seconds": round(time.time() - t_all, 1)}
    (out / DIR).mkdir(parents=True, exist_ok=True)
    (out / DIR / "forms_run.json").write_text(json.dumps(summary, indent=1))
    log(f"[forms] {len(ok)} published ({sum(1 for o in ok if o['method'] == METHOD)} forms, "
        f"{sum(1 for o in ok if o['method'] == METHOD_TSDF)} as measured, "
        f"{sum(1 for o in ok if o['method'] == METHOD_HULL)} hull), {len(failed)} failed, "
        f"{len(skipped)} skipped in {summary['seconds']} s")
    return summary
