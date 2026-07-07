# STAC-Builder — Phase 6: supervision report draft generator.
#
# Assembles a per-scene supervision report from the R.8 store and the upstream
# phases: element inventory (Phase 2), findings with captures + 3D coordinates
# (Phase 3), requested measurements (Phase 5 tools), capture coverage (Phase 4),
# and a "Reconstruction quality" section (onion / vote metrics, dynamic-pixel
# fraction, scale/marker conflicts — Phase R).
#
# Two invariants the spec is strict about:
#   * TRACEABILITY — every number carries its trace (tool + arguments + timestamp).
#   * PROVENANCE   — every VLM observation is marked "proposal pending validation"
#                    unless a human validated it; geometry is "tool_measured".
#
# Output is Markdown + image assets (docx/pdf is out of scope). Bilingual ES/FR.
# This generator MEASURES NOTHING itself — it reads tool_measured values from the
# store and the Phase 5 SpatialTools, and labels VLM text as proposed.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
from pathlib import Path

from phase5_qa.tools import SpatialTools
from phase_r.instance_store import InstanceStore
from phase6_report.i18n import strings


def _trace(tool: str, args: dict, ts: str) -> str:
    a = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"_({tool}({a}), {ts})_"


class ReportBuilder:
    def __init__(self, store_path: str, session_dir=None, config: dict | None = None,
                 lang: str = "es", coverage_json: str | None = None,
                 timestamp: str | None = None):
        self.store = InstanceStore(store_path)
        self.tools = SpatialTools(self.store)
        self.session_dir = Path(session_dir) if session_dir else None
        self.frames_dir = (self.session_dir / "frames") if self.session_dir else None
        cfg = (config or {}).get("phase6", {}) if config else {}
        self.lang = lang if lang in ("es", "fr") else cfg.get("lang", "es")
        self.coverage_json = coverage_json
        # timestamp is injected (deterministic reports / reproducible demos)
        self.ts = timestamp or "unset-timestamp"
        self.S = strings(self.lang)
        self.assets = []  # (rel_path, source_frame)

    # ── provenance helpers ──────────────────────────────────────────
    def _provenance_tag(self, origin: str, status: str) -> str:
        S = self.S
        if status in ("human_validated", "validated"):
            return f"✅ {S['validated']}"
        if origin == "vlm_proposed":
            return f"⚠️ {S['proposed']}"
        if origin == "tool_measured":
            return S["tool_measured"]
        return origin or "—"

    def _sev(self, sev: str) -> str:
        return self.S.get(f"severity_{sev}", sev)

    # ── sections ────────────────────────────────────────────────────
    def _inventory(self) -> list[str]:
        S = self.S
        out = [f"## {S['sec_inventory']}", ""]
        insts = self.store.list_instances()
        if not insts:
            return out + [S["no_instances"], ""]
        out.append(f"| {S['col_id']} | {S['col_class']} | {S['col_material']} | "
                   f"{S['col_state']} | {S['col_dims']} | {S['col_provenance']} |")
        out.append("|---|---|---|---|---|---|")
        for i in insts:
            iid = i["instance_id"]
            c = self.store.get_classification(iid)
            klass = c["class_final"] if c else i["label"]
            material = c["material"] if c else "—"
            state = c["state"] if c else "—"
            size = self.tools.get_object_size(iid)
            if "width_m" in size:
                dims = (f"{size['width_m']}×{size['height_m']}×{size['depth_m']} m "
                        + _trace("get_object_size", {"id": iid}, self.ts))
            else:
                dims = "—"
            prov = self._provenance_tag("vlm_proposed" if c else "tool_measured",
                                        i.get("status", ""))
            if c and c.get("conflict"):
                prov += f" · {S['conflict']}"
            out.append(f"| {iid} | {klass} | {material} | {state} | {dims} | {prov} |")
        out.append("")
        return out

    def _findings(self) -> list[str]:
        S = self.S
        out = [f"## {S['sec_findings']}", ""]
        finds = self.store.list_findings()
        if not finds:
            return out + [S["no_findings"], ""]
        finds.sort(key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(f["severity"], 3))
        for f in finds:
            fid = f["finding_id"]
            loc = (", ".join(f"{v:.2f}" for v in f["point3d"]) if f["point3d"]
                   else S["none"])
            prov = self._provenance_tag(f["origin"], f["status"])
            head = (f"### {S['col_type']}: {f['type']} — {S['col_severity']}: "
                    f"{self._sev(f['severity'])}")
            out.append(head)
            out.append("")
            out.append(f"- {S['col_location']}: {loc} · {S['col_instance']}: "
                       f"{f['instance_id']} · {S['col_confidence']}: {f['confidence']:.2f}")
            out.append(f"- {f['description']}")
            note = prov
            if f["correlated_residual"]:
                note += f" · {S['correlated']}"
            out.append(f"- {note}")
            fig = self._finding_figure(f)
            if fig:
                out.append("")
                out.append(f"![{S['figure']} {fid}]({fig[0]})")
                out.append(f"*{S['figure']} {fid} — {S['source_frame']}: "
                           f"{f['frame_id']:06d}*")
            out.append("")
        return out

    def _measurements(self) -> list[str]:
        S = self.S
        out = [f"## {S['sec_measurements']}", ""]
        out.append(f"| {S['col_instance']} | {S['col_metric']} | {S['col_value']} | "
                   f"{S['col_provenance']} |")
        out.append("|---|---|---|---|")
        rows = 0
        for i in self.store.list_instances():
            iid = i["instance_id"]
            c = self.store.get_classification(iid)
            klass = (c["class_final"] if c else i["label"])
            # structural elements -> plumb/level (spec: desplome / desnivel)
            for tool_name, key, unit in self._metric_plan(klass):
                res = getattr(self.tools, tool_name)(iid)
                if key in res:
                    val = f"{res[key]} {res.get('unit', unit)}"
                    tr = _trace(tool_name, {"id": iid}, self.ts)
                    out.append(f"| {iid} ({klass}) | {tool_name} | {val} {tr} | "
                               f"{S['tool_measured']} |")
                    rows += 1
        if rows == 0:
            out.append(f"| — | — | — | {S['none']} |")
        out.append("")
        return out

    @staticmethod
    def _metric_plan(klass: str):
        from phase2_classify.classify import is_structural
        k = (klass or "").lower()
        plan = [("get_object_volume", "bbox_volume_m3", "m^3")]
        if is_structural(k):
            if any(w in k for w in ("wall", "column", "beam")):
                plan.append(("get_plumb", "plumb_deviation_deg", "deg"))
            if any(w in k for w in ("floor", "slab", "platform")):
                plan.append(("get_level", "level_deviation_deg", "deg"))
        return plan

    def _coverage(self) -> list[str]:
        S = self.S
        out = [f"## {S['sec_coverage']}", ""]
        if not self.coverage_json or not Path(self.coverage_json).exists():
            out.append(S["coverage_none"]); out.append("")
            return out
        rep = json.load(open(self.coverage_json))
        n = rep.get("n_zones", 0)
        if n == 0:
            out.append(S["coverage_none"])
        else:
            out.append(f"**{S['coverage_zones']}: {n}**")
            out.append("")
            for c in rep.get("checklist", []):
                cen = ", ".join(f"{v:.1f}" for v in c["centroid"])
                out.append(f"- [ ] ({cen}) m — {c['instruction']}")
        out.append("")
        return out

    def _quality(self) -> list[str]:
        S = self.S
        out = [f"## {S['sec_quality']}", ""]
        out.append(f"| {S['col_instance']} | {S['onion']} | {S['vote']} |")
        out.append("|---|---|---|")
        for i in self.store.list_instances():
            iid = i["instance_id"]
            m = self.store.get_metrics(iid)
            onion = m.get("onion")
            if onion:
                ob = (f"{S['bimodal_yes']}, {S['separation']} "
                      f"{onion['separation_m']:.3f} m" if onion["bimodal"]
                      else S["bimodal_no"])
            else:
                ob = "—"
            vote = m.get("vote")
            vb = (f"{vote['mean_entropy']:.3f}" if vote and vote["mean_entropy"] is not None
                  else "—")
            out.append(f"| {iid} ({i['label']}) | {ob} | {vb} |")
        out.append("")
        dyn = self.store.get_meta("dynamic_pixel_fraction")
        if dyn is not None:
            out.append(f"- {S['dynamic_frac']}: {float(dyn):.4f} "
                       + _trace("scene_meta", {"key": "dynamic_pixel_fraction"}, self.ts))
        for key in ("scale_conflict", "marker_scale_conflict"):
            v = self.store.get_meta(key)
            if v is not None:
                out.append(f"- {S['scale_conflict']}: {v}")
        out.append("")
        return out

    # ── figures ─────────────────────────────────────────────────────
    def _finding_figure(self, f: dict):
        if not self.frames_dir or f["box_xywh"] is None or f["frame_id"] is None:
            return None
        p = self.frames_dir / f"{f['frame_id']:06d}.jpg"
        if not p.exists():
            return None
        from PIL import Image
        img = Image.open(p).convert("RGB")
        W, H = img.size
        x, y, w, h = f["box_xywh"]
        pad = 0.06
        x1 = max(0, (x - pad) * W); y1 = max(0, (y - pad) * H)
        x2 = min(W, (x + w + pad) * W); y2 = min(H, (y + h + pad) * H)
        if x2 - x1 < 8 or y2 - y1 < 8:
            crop = img
        else:
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        rel = f"assets/finding_{f['finding_id']}.jpg"
        self.assets.append((rel, crop, f["frame_id"]))
        return (rel, f["frame_id"])

    # ── assembly ────────────────────────────────────────────────────
    def build(self) -> str:
        S = self.S
        scene_type = self.store.get_meta("scene_type") or "—"
        scene_id = self.store.get_meta("scene_id") or Path(self.store.path).stem
        md = [f"# {S['title']}", "",
              f"- **{S['scene']}**: {scene_id}",
              f"- **{S['scene_type']}**: {scene_type}",
              f"- **{S['generated']}**: {self.ts}",
              f"- **{S['lang_name']}**",
              "",
              f"> {S['provenance_note']}", ""]
        md += self._inventory()
        md += self._findings()
        md += self._measurements()
        md += self._coverage()
        md += self._quality()
        return "\n".join(md)

    def write(self, out_dir) -> dict:
        out_dir = Path(out_dir)
        (out_dir / "assets").mkdir(parents=True, exist_ok=True)
        md = self.build()  # populates self.assets
        for rel, crop, _frame in self.assets:
            crop.save(out_dir / rel)
        name = f"report_{self.lang}.md"
        (out_dir / name).write_text(md)
        self.store.close()
        return {"report": str(out_dir / name), "n_assets": len(self.assets),
                "lang": self.lang}
