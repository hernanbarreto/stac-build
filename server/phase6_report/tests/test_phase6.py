# STAC-Builder — Phase 6 unit tests (synthetic store, no model, no frames).
#
# Verifies the two invariants: traceability (tool + args + timestamp on numbers)
# and provenance (VLM text marked proposed unless validated; geometry marked
# tool_measured), plus bilingual selection and section coverage.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.instance_store import InstanceStore  # noqa: E402
from phase6_report.report import ReportBuilder  # noqa: E402

TS = "2026-07-07T00:00:00Z"


def _store(path):
    st = InstanceStore(path)
    st.set_meta("scene_type", "test scene")
    st.upsert_instance(1, "wall", status="proposed")
    # a gravity-aligned-ish OBB so size/plumb tools return values
    T = np.eye(4)
    aabb = np.array([-0.5, 0.5, -1.0, 1.0, -0.05, 0.05])
    st.set_obb(1, T, aabb, np.zeros(3), n_points=200)
    pts = np.random.default_rng(0).normal(0, [0.5, 1.0, 0.02], size=(200, 3))
    st.set_points(1, pts)
    st.set_classification(1, class_final="tiled wall", material="tile", state="cracked",
                          notes="", confidence=0.9, conflict=True,
                          whitelist_eligible=True, best_frame=10)
    st.set_onion_metric(1, True, 0.12, 5.0)
    st.add_finding(instance_id=1, type="crack", severity="high", description="vertical crack",
                   confidence=0.85, frame_id=10, point3d=np.array([0.1, 0.2, 0.3]),
                   correlated_residual=True, status="proposed", origin="vlm_proposed")
    st.add_finding(instance_id=1, type="moisture", severity="low", description="damp patch",
                   confidence=0.6, frame_id=11, point3d=np.array([0.0, 0.0, 1.0]),
                   correlated_residual=False, status="human_validated", origin="human_validated")
    st.set_meta("dynamic_pixel_fraction", "0.0123")
    st.close()


def test_traceability_and_provenance_es():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.db"); _store(sp)
        md = ReportBuilder(sp, lang="es", timestamp=TS).build()
        # traceability: a tool call with args + the injected timestamp appears
        assert "get_object_size(id=1)" in md and TS in md
        # provenance: unvalidated VLM class flagged as proposal + conflict
        assert "propuesta (pendiente de validación)" in md
        assert "conflicto de etiqueta" in md
        # a human-validated finding is marked validated, not proposed
        assert "validado" in md
        # geometry labelled tool_measured
        assert "medido por herramienta" in md


def test_bilingual_titles_differ():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.db"); _store(sp)
        es = ReportBuilder(sp, lang="es", timestamp=TS).build()
        sp2 = os.path.join(d, "s2.db"); _store(sp2)
        fr = ReportBuilder(sp2, lang="fr", timestamp=TS).build()
        assert "Informe de supervisión" in es and "Rapport de supervision" in fr
        assert "Constats" in fr and "Hallazgos" in es


def test_all_sections_present():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.db"); _store(sp)
        md = ReportBuilder(sp, lang="es", timestamp=TS).build()
        for key in ("Inventario", "Hallazgos", "Mediciones", "Cobertura",
                    "Calidad de reconstrucción"):
            assert key in md, f"missing section {key}"
        # quality section surfaces the onion metric + dynamic fraction
        assert "bimodal" in md and "0.0123" in md


def test_findings_sorted_high_first():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.db"); _store(sp)
        md = ReportBuilder(sp, lang="es", timestamp=TS).build()
        assert md.index("crack") < md.index("moisture")  # high before low


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name); n += 1
    print(f"\n{n} tests passed")
