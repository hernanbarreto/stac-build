"""FORMS — the VLM decides the FORM, the cloud measures pose and dimensions (USER 2026-10-16).

"vlm dice cómo es y cómo debe ser; la nube orienta sobre dónde está y qué tamaño tiene; la nube
deforma mucho; entre los dos debemos obtener el resultado correcto … si vlm dice una forma es esa
forma". The per-object meshing of the Mesh button:

1. ``catalog`` / ``query``   the structured form query (JSON, a closed catalogue) per object,
                             from its isolated SAM3-mask views — ``vlm_proposed``;
2. ``templates``             parametric templates in torch (section extruded along an axis,
                             sphere, cone, revolution): signed distance, surface samples, mesh;
3. ``fit``                   pose + dimensions by a robust fit of THAT form to the object's
                             points, Huber at the session's own repeatability σ — ``tool_measured``;
4. ``silhouette``            the corrector: the form projected into EVERY keyframe that masks the
                             object, compared with the mask (occlusion-verified by the cloud's
                             z-buffer) and optimised with the points until the consensus stops
                             improving beyond its own noise; the bar is the cloud's own agreement
                             with the masks, never a chosen number;
5. ``composite``             parts guided by the VLM's list (count, orientation, relations);
6. ``review``                a second VLM round with the fitted form drawn over the worst views,
                             only when the converged consensus is under the measured bar;
7. ``run``                   the per-session runner: records, GLB + meta into output/tsdf/,
                             texture; a freeform object is meshed as measured (live-cloud TSDF).

The provenance rule, restated: the VLM decides the form and never measures; every number
(position, dimension, residual, consensus) is measured on the cloud and the masks.
"""
