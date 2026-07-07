# STAC-Builder — Phase 6: bilingual report strings (ES / FR).
#
# Stakeholders are IN3 (ES/FR). Every user-facing label in the generated report
# comes from here; the report body carries no hard-coded natural language.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

STRINGS = {
    "es": {
        "lang_name": "Español",
        "title": "Informe de supervisión — borrador",
        "scene": "Escena",
        "scene_type": "Tipo de escena",
        "generated": "Generado",
        "provenance_note": ("Cada cifra lleva su traza (herramienta + argumentos "
                            "+ marca de tiempo). Toda observación del modelo visual "
                            "está marcada como **propuesta pendiente de validación** "
                            "salvo que haya sido validada por un humano."),
        "sec_inventory": "1. Inventario de elementos",
        "sec_findings": "2. Hallazgos",
        "sec_measurements": "3. Mediciones",
        "sec_coverage": "4. Cobertura de captura",
        "sec_quality": "5. Calidad de reconstrucción",
        "col_id": "ID", "col_class": "Clase", "col_material": "Material",
        "col_state": "Estado", "col_dims": "Dimensiones (An×Al×Pr)",
        "col_provenance": "Procedencia", "col_type": "Tipo", "col_severity": "Severidad",
        "col_location": "Ubicación 3D (m)", "col_confidence": "Confianza",
        "col_instance": "Instancia", "col_metric": "Métrica", "col_value": "Valor",
        "proposed": "propuesta (pendiente de validación)",
        "validated": "validado",
        "human_validated": "validado por humano",
        "tool_measured": "medido por herramienta",
        "vlm_proposed": "propuesto por modelo visual",
        "conflict": "conflicto de etiqueta",
        "correlated": "coincide con residuo geométrico (severidad elevada)",
        "no_findings": "No se registraron hallazgos.",
        "no_instances": "No hay elementos en el inventario.",
        "figure": "Figura", "source_frame": "fotograma de origen",
        "coverage_zones": "Zonas subescaneadas",
        "coverage_none": "No se detectaron zonas aisladas subescaneadas.",
        "recapture": "Recaptura sugerida",
        "onion": "Métrica de cebolla (doble superficie)",
        "vote": "Entropía de voto multi-vista",
        "dynamic_frac": "Fracción de píxeles dinámicos excluidos",
        "scale_conflict": "Conflicto de escala ancla/marcador",
        "bimodal_yes": "bimodal", "bimodal_no": "unimodal",
        "separation": "separación", "none": "ninguno", "measured_at": "medido",
        "severity_low": "baja", "severity_medium": "media", "severity_high": "alta",
    },
    "fr": {
        "lang_name": "Français",
        "title": "Rapport de supervision — brouillon",
        "scene": "Scène",
        "scene_type": "Type de scène",
        "generated": "Généré le",
        "provenance_note": ("Chaque chiffre porte sa traçabilité (outil + arguments "
                            "+ horodatage). Toute observation du modèle visuel est "
                            "marquée **proposition en attente de validation** sauf "
                            "validation humaine."),
        "sec_inventory": "1. Inventaire des éléments",
        "sec_findings": "2. Constats",
        "sec_measurements": "3. Mesures",
        "sec_coverage": "4. Couverture de capture",
        "sec_quality": "5. Qualité de reconstruction",
        "col_id": "ID", "col_class": "Classe", "col_material": "Matériau",
        "col_state": "État", "col_dims": "Dimensions (L×H×P)",
        "col_provenance": "Provenance", "col_type": "Type", "col_severity": "Gravité",
        "col_location": "Position 3D (m)", "col_confidence": "Confiance",
        "col_instance": "Instance", "col_metric": "Métrique", "col_value": "Valeur",
        "proposed": "proposition (en attente de validation)",
        "validated": "validé",
        "human_validated": "validé par un humain",
        "tool_measured": "mesuré par outil",
        "vlm_proposed": "proposé par le modèle visuel",
        "conflict": "conflit d'étiquette",
        "correlated": "coïncide avec un résidu géométrique (gravité relevée)",
        "no_findings": "Aucun constat enregistré.",
        "no_instances": "Aucun élément dans l'inventaire.",
        "figure": "Figure", "source_frame": "image source",
        "coverage_zones": "Zones sous-échantillonnées",
        "coverage_none": "Aucune zone isolée sous-échantillonnée détectée.",
        "recapture": "Recapture suggérée",
        "onion": "Métrique en pelure (double surface)",
        "vote": "Entropie de vote multi-vues",
        "dynamic_frac": "Fraction de pixels dynamiques exclus",
        "scale_conflict": "Conflit d'échelle ancrage/repère",
        "bimodal_yes": "bimodal", "bimodal_no": "unimodal",
        "separation": "séparation", "none": "aucun", "measured_at": "mesuré",
        "severity_low": "faible", "severity_medium": "moyenne", "severity_high": "élevée",
    },
}


def strings(lang: str) -> dict:
    return STRINGS.get(lang, STRINGS["es"])
