"""THE EXPORTER — a scan's products as files other programs open, in one ZIP the user downloads.

USER 2026-11-17: "hagamos un exportador a IFC … podriamos exportar la nube de puntos compatible
con varios programas, la malla glb con la textura, y el IFC y no se si alguno mas, y que la
persona lo pueda descargar en su maquina" — and, to the proposal (the cloud as E57 + LAS + PLY,
the mesh as GLB, the elements as IFC, the objects as CSV; exported AS IT IS ON DISK with a README
that says the epoch and whether it is calibrated; chosen by ticks, the cloud filtered by the
viewer's confidence): "si los 3" / "correcto, tal como lo indicas" / "correcto, coincido con vos".

  cloud.py      the published cloud read in blocks and written as PLY, LAS 1.4 and E57
  package.py    the items of one export, its README and the ZIP; the worker's entry point
  (exporter_api.py: the routes — the plan, the launch in its own process, the state, the download)

ONE FRAME for everything that is exported: the viewer's display frame (metres, the floor at
height 0) written Z-UP — (x, y, z) → (x, −z, y) — as E57, LAS and IFC ask. The GLB keeps glTF's
own Y-up: the programs that read it turn it by that same rule, so the mesh lands on the cloud.
Nothing is recomputed: a point, a box and the mesh are the ones on disk.
"""
